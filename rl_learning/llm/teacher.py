"""The teacher: a large model behind an API that generates data and judges answers.

Provider-agnostic — uses Anthropic (Claude) if ANTHROPIC_API_KEY is set, else
OpenAI if OPENAI_API_KEY is set. (The user chose Claude, but only an OpenAI key
is present in this environment, so it falls back to OpenAI automatically; set the
Anthropic key to switch.)
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import List

from .topics import (JUDGE_SYSTEM, SCORE_SYSTEM, SYSTEM_TEACHER, judge_prompt,
                     qa_prompt, score_prompt)


# A local, OpenAI-compatible server: Ollama, LM Studio, vLLM, llama.cpp all speak
# this. Pointing the teacher here makes Stages 4-6 free and fully offline — no
# key, no tokens billed, nothing leaving the machine.
LOCAL_BASE_URL = os.environ.get("RL_LOCAL_LLM_URL", "http://localhost:11434/v1")
LOCAL_PROBE_TIMEOUT_S = 1.5


_PROBE_TTL_S = 10
_probe_cache: dict = {"at": 0.0, "result": None}


def discover_local_models(force: bool = False) -> dict:
    """Ask the local LLM server what it can serve. Never raises: not having one
    running is the normal case, not an error.

    Cached briefly — the UI polls status on a timer, and paying a network probe
    (worst case a full timeout) on every poll would stall it.
    """
    import urllib.request

    now = time.monotonic()
    if not force and _probe_cache["result"] and now - _probe_cache["at"] < _PROBE_TTL_S:
        return _probe_cache["result"]

    try:
        req = urllib.request.Request(f"{LOCAL_BASE_URL.rstrip('/')}/models")
        with urllib.request.urlopen(req, timeout=LOCAL_PROBE_TIMEOUT_S) as resp:
            payload = json.loads(resp.read())
        models = [m["id"] for m in payload.get("data", []) if m.get("id")]
        result = {"available": True, "base_url": LOCAL_BASE_URL, "models": models}
    except Exception:
        result = {"available": False, "base_url": LOCAL_BASE_URL, "models": []}

    _probe_cache.update(at=now, result=result)
    return result


def available_provider():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    # No key, but the user may be running their own model — that counts.
    if discover_local_models()["available"]:
        return "local"
    return None


# Approx USD per 1M tokens (input, output) — for the auto-improve budget cap.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.5, 10.0),
    "claude-3-5-sonnet-latest": (3.0, 15.0), "claude-3-5-haiku-latest": (0.80, 4.0),
}
_DEFAULT_PRICE = (1.0, 3.0)


TIMEOUT_S = 60          # per-request cap so one hung call can't stall a run
# A local model on a mid-range GPU is far slower per token than a hosted one, and
# the first call also pays for loading the weights into VRAM.
LOCAL_TIMEOUT_S = 300
RETRIES = 3             # transient failures (rate limit, 5xx, network) are retried
_BACKOFF_S = 2.0


def _mask_secrets(msg: str) -> str:
    """Never let an API key leak into logs/UI via an exception message."""
    return re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "sk-***", msg)


class Teacher:
    def __init__(self, provider: str = None, model: str = None, api_key: str = None,
                 base_url: str = None):
        # explicit api_key (e.g. pasted in the UI) wins; else use env keys
        self.provider = provider or ("anthropic" if (api_key and provider == "anthropic") else available_provider())
        if self.provider == "anthropic":
            import anthropic
            self.client = (anthropic.Anthropic(api_key=api_key, timeout=TIMEOUT_S) if api_key
                           else anthropic.Anthropic(timeout=TIMEOUT_S))
            self.model = model or "claude-3-5-sonnet-latest"
        elif self.provider == "openai":
            import openai
            self.client = (openai.OpenAI(api_key=api_key, timeout=TIMEOUT_S) if api_key
                           else openai.OpenAI(timeout=TIMEOUT_S))
            self.model = model or "gpt-4o-mini"
        elif self.provider == "local":
            # The user's own model, on their own machine. Same wire format as
            # OpenAI, so _chat_once needs no special case. Most local servers
            # ignore the key, but the client insists on a non-empty one.
            import openai
            self.base_url = (base_url or LOCAL_BASE_URL).rstrip("/")
            self.client = openai.OpenAI(
                api_key=api_key or "local", base_url=self.base_url,
                timeout=LOCAL_TIMEOUT_S,
            )
            self.model = model or self._first_local_model()
        else:
            raise RuntimeError(
                "No teacher available. Set ANTHROPIC_API_KEY or OPENAI_API_KEY, paste a "
                "key, or run a local model (Ollama / LM Studio / vLLM)."
            )
        self.in_tok = 0
        self.out_tok = 0

    def _first_local_model(self) -> str:
        found = discover_local_models()
        if not found["models"]:
            raise RuntimeError(
                f"No local model found at {self.base_url}. Start one "
                f"(e.g. `ollama pull qwen2.5:7b`) or pick a different teacher."
            )
        return found["models"][0]

    def cost_usd(self) -> float:
        # Running on your own hardware costs no tokens.
        if self.provider == "local":
            return 0.0
        pin, pout = PRICES.get(self.model, _DEFAULT_PRICE)
        return round(self.in_tok / 1e6 * pin + self.out_tok / 1e6 * pout, 4)

    # --- low-level chat ----------------------------------------------------

    def _chat_once(self, system, user, max_tokens, temperature):
        if self.provider == "anthropic":
            msg = self.client.messages.create(
                model=self.model, system=system, max_tokens=max_tokens, temperature=temperature,
                messages=[{"role": "user", "content": user}],
            )
            self.in_tok += msg.usage.input_tokens
            self.out_tok += msg.usage.output_tokens
            return msg.content[0].text
        resp = self.client.chat.completions.create(
            model=self.model, max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        if resp.usage:
            self.in_tok += resp.usage.prompt_tokens
            self.out_tok += resp.usage.completion_tokens
        return resp.choices[0].message.content

    def chat(self, system: str, user: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """One teacher call with retry on transient failures, so a single flaky
        request can't kill a 20-minute training run. Auth errors fail fast."""
        last = None
        for attempt in range(RETRIES + 1):
            try:
                return self._chat_once(system, user, max_tokens, temperature)
            except Exception as exc:
                name = type(exc).__name__
                if "Authentication" in name or "PermissionDenied" in name or "401" in str(exc):
                    raise RuntimeError(_mask_secrets(f"auth failed: {exc}")) from None
                last = exc
                if attempt < RETRIES:
                    time.sleep(_BACKOFF_S * (2 ** attempt))
        raise RuntimeError(_mask_secrets(f"teacher call failed after {RETRIES + 1} attempts: {last}")) from None

    # --- dataset generation ------------------------------------------------

    def generate_qa(self, topic_key: str, n: int) -> List[dict]:
        raw = self.chat(SYSTEM_TEACHER, qa_prompt(topic_key, n), max_tokens=2048)
        return _parse_qa(raw)

    # --- judging (for DPO preference pairs) --------------------------------

    def judge(self, question: str, answer_a: str, answer_b: str) -> str:
        """Return 'A' or 'B' — which answer the teacher prefers (temp 0 = consistent)."""
        out = self.chat(JUDGE_SYSTEM, judge_prompt(question, answer_a, answer_b),
                        max_tokens=4, temperature=0.0).strip().upper()
        return "A" if out.startswith("A") else "B"

    def generate_for_topic(self, topic: str, n: int, focus=None):
        """Free-text-topic Q&A generation (study material and exams)."""
        extra = ""
        if focus:
            extra = "Focus on aspects the student got wrong:\n- " + "\n- ".join(focus[:5]) + "\n"
        prompt = (f"Generate {n} diverse question-answer pairs about: {topic}.\n{extra}"
                  "MIX the formats: definitions, how/why questions, application problems, and "
                  "REVERSED phrasings of key relations (if one pair asks A→B, ask another B→A — "
                  "models only retain relations they have seen in both directions). "
                  "Answers 2-5 sentences, correct and specific. "
                  "Return ONLY a JSON array of objects with keys \"question\" and \"answer\".")
        from .topics import SYSTEM_TEACHER
        return _parse_qa(self.chat(SYSTEM_TEACHER, prompt, max_tokens=3000))

    def write_chapter(self, domain: str, subject: str, topics: str, words: int = 600) -> str:
        """A dense textbook chapter for the student's 'reading period' — raw text
        the student is pretrained on before the Q&A drills."""
        prompt = (f"Write a dense, factual textbook chapter on '{subject}' (part of a course on {domain}). "
                  f"Cover: {topics or subject}. State key relations in BOTH directions. "
                  f"About {words} words of plain prose. No preamble, no markdown headers.")
        return self.chat("You write expert textbook chapters, dense with correct, specific facts.",
                         prompt, max_tokens=1400, temperature=0.4).strip()

    def design_curriculum(self, domain: str, n=None):
        """The Dean: an ordered, coherent syllabus to make a student expert in `domain`.
        If `n` is falsy, the Dean decides how many subjects the domain actually needs."""
        count = (f"about {n} subjects" if n else
                 "as many subjects as are genuinely needed for real expertise (typically 8-16; do not pad or stop short)")
        prompt = (f"You are the dean of a school. Design a coherent, ordered curriculum to take a student from "
                  f"foundations to EXPERT in: {domain}. Include {count}, ordered prerequisite → advanced. "
                  "For each give keys \"subject\" (short), \"why\" (one line), \"depth_topics\" (3-5 sub-topics). "
                  "Return ONLY a JSON array of those objects, nothing else.")
        raw = self.chat("You are an expert curriculum designer.", prompt, max_tokens=4096, temperature=0.4)
        out = []
        for d in _parse_json_array(raw):
            if isinstance(d, dict) and d.get("subject"):
                out.append({"subject": str(d["subject"]).strip(), "why": str(d.get("why") or "").strip(),
                            "depth_topics": d.get("depth_topics") or []})
        return out

    def propose_prompt(self, current_prompt: str, weak_examples) -> str:
        """Architecture step: propose an improved system prompt for the student."""
        ex = "\n".join(f"- {w.get('question', '')}" for w in (weak_examples or [])[:5])
        p = (f"You tune system prompts for a small RL-assistant. Current prompt:\n\"{current_prompt}\"\n\n"
             f"It answered these poorly:\n{ex or '(none yet)'}\n\n"
             "Write ONE improved system prompt (a short paragraph) that yields more relevant, correct, concise "
             "answers. Return ONLY the new prompt text, no preamble.")
        return self.chat("You are a prompt optimizer.", p, max_tokens=300, temperature=0.5).strip().strip('"')

    def score(self, question: str, answer: str) -> int:
        """Rubric grade 1-5 for one answer (MT-Bench-style single grading)."""
        out = self.chat(SCORE_SYSTEM, score_prompt(question, answer), max_tokens=4, temperature=0.0)
        for ch in out:
            if ch in "12345":
                return int(ch)
        return 3


def _strip_fences(raw: str) -> str:
    return re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()


def _fix_escapes(s: str) -> str:
    """Double any backslash that isn't a valid JSON escape. The teacher's math/physics
    answers contain LaTeX (\\theta, \\epsilon, \\(, \\, …) which are invalid JSON
    escapes and otherwise raise 'Invalid \\escape'. Valid escape PAIRS are consumed
    (not just looked ahead) so the second char of '\\\\' can't be re-doubled."""
    return re.sub(r'\\([\\/"bfnrtu]|u[0-9a-fA-F]{4})|\\',
                  lambda m: m.group(0) if m.group(1) else "\\\\", s)


def _loads(s: str):
    """json.loads tolerant of literal control chars (raw newlines/tabs in strings)."""
    return json.loads(s, strict=False)


def _parse_json_array(raw: str):
    """Robustly load a JSON array: tolerate LaTeX backslashes, literal control chars,
    and salvage truncated output by cutting to the last complete object."""
    raw = _strip_fences(raw)
    start = raw.find("[")
    if start != -1:
        raw = raw[start:]
    for candidate in (raw, _fix_escapes(raw)):
        try:
            return _loads(candidate)
        except Exception:
            pass
    fixed = _fix_escapes(raw)
    end = fixed.rfind("}")
    if end != -1:
        candidate = fixed[:end + 1].rstrip().rstrip(",") + "]"
        try:
            return _loads(candidate)
        except Exception:
            pass
    return []


def _parse_qa(raw: str) -> List[dict]:
    """Pull a JSON array of {question, answer} out of the model's reply (robust to
    LaTeX escapes and truncation)."""
    out = []
    for d in _parse_json_array(raw):
        if not isinstance(d, dict):
            continue
        q, a = d.get("question"), d.get("answer")
        if q and a:
            out.append({"question": str(q).strip(), "answer": str(a).strip()})
    return out
