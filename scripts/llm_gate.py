"""Gate check: load Qwen2.5-1.5B, generate, and run one LoRA training step on GPU."""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
t = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
print(f"loaded {MODEL} in {time.time()-t:.0f}s | base vram {torch.cuda.memory_allocated()/1e9:.2f} GB")

# 1) generation works
msgs = [{"role": "user", "content": "In one sentence, what is the core idea of Q-learning?"}]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
print("GEN:", tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())

# 2) LoRA attaches + one training step runs
lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
model = get_peft_model(model, lora)
model.print_trainable_parameters()

text = "Question: What is SARSA?\nAnswer: SARSA is an on-policy temporal-difference control algorithm."
enc = tok(text, return_tensors="pt").to("cuda")
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
model.train()
loss = model(**enc, labels=enc["input_ids"]).loss
loss.backward(); opt.step()
print(f"LoRA training step OK | loss {float(loss):.3f} | peak vram {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print("GATE PASSED")
