"""The student engine: load Qwen2.5-1.5B, LoRA-fine-tune it (SFT then DPO), and
generate answers (base vs tuned) for the chat test.

Design choices that keep it small and transparent:
  * One LoRA "tuned" adapter. SFT trains it to copy the teacher; DPO continues
    training the same adapter with a preference loss.
  * DPO's reference policy is the *base* model, obtained for free by disabling
    the LoRA adapter (`peft`'s disable_adapter) — no second model in memory.
The model stays resident in the server process so chat is instant after loading.
"""

from __future__ import annotations

import os
import random
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "llm_tuned")
SYS = "You are a helpful, knowledgeable assistant. Answer accurately, concisely and practically."

# LoRA reaches the MLP layers (gate/up/down) as well as attention: factual
# knowledge in transformers lives substantially in the MLPs (key→value memories),
# so attention-only adapters cap how much a subject can actually be RETAINED.
LORA_KW = dict(r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
               target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"])


def set_base_model(name: str):
    """Switch the student (e.g. Qwen2.5-1.5B → 3B → 7B). Frees the current model;
    the next load() pulls the new one. Saved adapters are base-specific."""
    global BASE_MODEL
    if name and name != BASE_MODEL:
        BASE_MODEL = name
        free()

_M = {"tok": None, "base": None, "peft": None, "has_tuned": False}


def _adapter_path():
    """Where the saved 'tuned' adapter lives (peft writes named adapters into a
    subfolder), or None if there isn't one."""
    for cand in (os.path.join(ADAPTER_DIR, "tuned"), ADAPTER_DIR):
        if os.path.exists(os.path.join(cand, "adapter_config.json")):
            return cand
    return None


def status() -> dict:
    return {"loaded": _M["base"] is not None,
            "has_tuned": _M["has_tuned"] or (_adapter_path() is not None),
            "base_model": BASE_MODEL, "device": (torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU")}


def load():
    if _M["base"] is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _M["tok"] = AutoTokenizer.from_pretrained(BASE_MODEL)
        _M["base"] = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).to(DEVICE)
        _M["base"].eval()
        # reattach a previously-saved tuned adapter if present
        ap = _adapter_path()
        if ap:
            from peft import PeftModel
            _M["peft"] = PeftModel.from_pretrained(_M["base"], ap, adapter_name="tuned", is_trainable=True)
            _M["has_tuned"] = True
    return _M["tok"], _M["base"]


def _ensure_adapter():
    """Make sure a trainable 'tuned' LoRA adapter exists."""
    load()
    if _M["peft"] is None:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(**LORA_KW)
        _M["peft"] = get_peft_model(_M["base"], cfg, adapter_name="tuned")
    _M["has_tuned"] = True
    return _M["peft"]


def _adapter_ctx(model, use_tuned):
    return nullcontext() if use_tuned else model.disable_adapter()


# --- inference -------------------------------------------------------------

def generate(question: str, tuned: bool = True, max_new_tokens: int = 230,
             do_sample: bool = False, temperature: float = 0.8) -> str:
    tok, base = load()
    if _M["peft"] is None:
        model, use_ctx = base, nullcontext()
    else:
        model = _M["peft"]
        use_ctx = _adapter_ctx(model, tuned)
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": question}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                     return_tensors="pt", return_dict=True).to(DEVICE)
    model.eval()
    kw = {"do_sample": True, "temperature": temperature, "top_p": 0.95} if do_sample else {"do_sample": False}
    with torch.no_grad(), use_ctx:
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id, **kw)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# --- tokenisation helpers --------------------------------------------------

def _prompt_text(tok, question):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def _sft_loss(tok, model, question, answer):
    prompt = _prompt_text(tok, question)
    enc = tok(prompt + answer + tok.eos_token, return_tensors="pt").to(DEVICE)
    plen = tok(prompt, return_tensors="pt")["input_ids"].shape[1]
    labels = enc["input_ids"].clone()
    labels[0, :plen] = -100                       # only learn the answer tokens
    return model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels).loss


def _seq_logprob(tok, model, question, answer):
    """Sum log-prob of `answer` tokens given the prompt (for DPO)."""
    prompt = _prompt_text(tok, question)
    enc = tok(prompt + answer + tok.eos_token, return_tensors="pt").to(DEVICE)
    plen = tok(prompt, return_tensors="pt")["input_ids"].shape[1]
    logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
    logp = F.log_softmax(logits[0, :-1], dim=-1)
    tgt = enc["input_ids"][0, 1:]
    tok_lp = logp[torch.arange(tgt.shape[0]), tgt]
    return tok_lp[plen - 1:].sum()                # answer tokens only


# --- training --------------------------------------------------------------

def sft_train(examples, epochs=3, lr=1e-4, on_progress=None):
    tok, _ = load()
    model = _ensure_adapter()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    curve = []
    total = epochs * len(examples)
    step = 0
    for ep in range(epochs):
        random.shuffle(examples)
        for ex in examples:
            loss = _sft_loss(tok, model, ex["question"], ex["answer"])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            step += 1
            curve.append(round(loss.item(), 4))
            if on_progress and step % max(1, total // 40) == 0:
                on_progress(step, total, curve)
    model.eval()
    save_adapter()
    if on_progress:
        on_progress(total, total, curve)
    return curve


def lm_train(texts, epochs=1, lr=1e-4, on_progress=None):
    """'Reading period': continued pretraining on raw textbook text (plain
    causal-LM loss over every token). This is how new FACTS get into the weights;
    Q&A SFT afterwards teaches the model to retrieve and phrase them."""
    tok, _ = load()
    model = _ensure_adapter()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    curve = []
    total = epochs * max(1, len(texts))
    step = 0
    for _ in range(epochs):
        for txt in texts:
            if not (txt or "").strip():
                continue
            enc = tok(txt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
            loss = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                         labels=enc["input_ids"]).loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            curve.append(round(loss.item(), 4))
            if on_progress:
                on_progress(step, total, curve)
    model.eval()
    save_adapter()
    return curve


def dpo_train(pairs, epochs=1, lr=5e-5, beta=0.1, on_progress=None):
    tok, _ = load()
    model = _ensure_adapter()      # continues from the SFT-trained adapter
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    curve = []
    total = epochs * len(pairs)
    step = 0
    for ep in range(epochs):
        random.shuffle(pairs)
        for p in pairs:
            q, chosen, rejected = p["question"], p["chosen"], p["rejected"]
            lp_ch = _seq_logprob(tok, model, q, chosen)
            lp_rj = _seq_logprob(tok, model, q, rejected)
            with torch.no_grad(), model.disable_adapter():       # reference = base
                rf_ch = _seq_logprob(tok, model, q, chosen)
                rf_rj = _seq_logprob(tok, model, q, rejected)
            logits = beta * ((lp_ch - rf_ch) - (lp_rj - rf_rj))
            loss = -F.logsigmoid(logits)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            step += 1
            curve.append(round(loss.item(), 4))
            if on_progress and step % max(1, total // 40) == 0:
                on_progress(step, total, curve)
    model.eval()
    save_adapter()
    if on_progress:
        on_progress(total, total, curve)
    return curve


def chat(system: str, user: str, tuned: bool = True, max_new_tokens: int = 256,
         do_sample: bool = False, temperature: float = 0.7) -> str:
    """Run an arbitrary system+user prompt through the local model (any agent role)."""
    tok, base = load()
    if _M["peft"] is None:
        model, ctx = base, nullcontext()
    else:
        model, ctx = _M["peft"], _adapter_ctx(_M["peft"], tuned)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                     return_tensors="pt", return_dict=True).to(DEVICE)
    model.eval()
    kw = {"do_sample": True, "temperature": temperature, "top_p": 0.95} if do_sample else {"do_sample": False}
    with torch.no_grad(), ctx:
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id, **kw)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def eval_loss(question: str, answer: str, tuned: bool = True) -> float:
    """Cross-entropy of `answer` given `question` (held-out perplexity metric)."""
    tok, base = load()
    if _M["peft"] is None:
        model, ctx = base, nullcontext()
    else:
        model, ctx = _M["peft"], _adapter_ctx(_M["peft"], tuned)
    model.eval()
    with torch.no_grad(), ctx:
        return float(_sft_loss(tok, model, question, answer))


def fresh_train(examples, epochs=2, lr=1e-4, on_progress=None, texts=None):
    """Drop any existing adapter and train a brand-new one from scratch:
    first a reading pass over `texts` (textbook chapters, raw LM loss), then
    Q&A SFT on `examples`. Used to rebuild a model from its enabled skills."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    free()                       # clear base + peft + cache
    _M["tok"] = AutoTokenizer.from_pretrained(BASE_MODEL)
    _M["base"] = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).to(DEVICE)
    _M["base"].eval()
    if not examples and not texts:
        return []                # no enabled skills -> back to the base model
    if texts:
        lm_train(texts, epochs=1, lr=lr, on_progress=on_progress)
    if not examples:
        return []
    return sft_train(examples, epochs=epochs, lr=lr, on_progress=on_progress)


def save_adapter():
    if _M["peft"] is not None:
        os.makedirs(ADAPTER_DIR, exist_ok=True)
        _M["peft"].save_pretrained(ADAPTER_DIR)


# --- cohort: several student adapters on one resident base model -----------

def cohort_init(n):
    """Fresh base + `n` independent LoRA adapters ('stu0'..) so a cohort of
    students can be trained and answer separately, switched in-memory."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    free()
    _M["tok"] = AutoTokenizer.from_pretrained(BASE_MODEL)
    _M["base"] = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).to(DEVICE)
    cfg = LoraConfig(**LORA_KW)
    ids = [f"stu{i}" for i in range(n)]
    peft = get_peft_model(_M["base"], cfg, adapter_name=ids[0])
    for aid in ids[1:]:
        peft.add_adapter(aid, cfg)
    _M["peft"] = peft
    _M["has_tuned"] = False
    return ids


def _adapter_params(model, aid):
    return [p for nm, p in model.named_parameters() if f".{aid}." in nm and p.requires_grad]


def cohort_train(aid, examples, epochs=2, lr=1e-4):
    if not examples or _M["peft"] is None:
        return []
    tok, model = _M["tok"], _M["peft"]
    model.set_adapter(aid)
    model.train()
    params = _adapter_params(model, aid)
    opt = torch.optim.AdamW(params, lr=lr)
    curve = []
    for _ in range(epochs):
        random.shuffle(examples)
        for ex in examples:
            loss = _sft_loss(tok, model, ex["question"], ex["answer"])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            curve.append(round(loss.item(), 4))
    model.eval()
    return curve


def cohort_generate(aid, question, max_new_tokens=200, do_sample=False, temperature=0.8):
    tok, model = _M["tok"], _M["peft"]
    model.set_adapter(aid)
    model.eval()
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": question}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                     return_tensors="pt", return_dict=True).to(DEVICE)
    kw = {"do_sample": True, "temperature": temperature, "top_p": 0.95} if do_sample else {"do_sample": False}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id, **kw)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def free():
    """Drop the model from VRAM (e.g., to free the GPU for game training)."""
    _M["peft"] = None
    _M["base"] = None
    _M["tok"] = None
    _M["has_tuned"] = _adapter_path() is not None
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
