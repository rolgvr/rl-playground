"""Build the training data from the teacher, and persist it as JSONL.

  * SFT data   -- the teacher writes question/answer pairs across the RL topics.
  * DPO pairs  -- for each question the student samples two answers; the teacher
                  judges which is better, giving a (chosen, rejected) pair. This
                  is the RLAIF loop: the student learns from its own outputs,
                  ranked by the teacher.
"""

from __future__ import annotations

import json
import os
import random

from . import engine
from .seed_data import SEED_QA
from .teacher import Teacher
from .topics import TOPICS

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "llm_data")
os.makedirs(DATA_DIR, exist_ok=True)
SFT_PATH = os.path.join(DATA_DIR, "sft.jsonl")
PREF_PATH = os.path.join(DATA_DIR, "prefs.jsonl")


def _save(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _load(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sft_count():
    return len(_load(SFT_PATH))


def pref_count():
    return len(_load(PREF_PATH))


def load_sft(include_seed=True):
    rows = _load(SFT_PATH)
    if include_seed:
        rows = [dict(x) for x in SEED_QA] + rows
    return rows


def load_prefs():
    return _load(PREF_PATH)


def generate_sft(n_per_topic=8, provider=None, api_key=None, on_progress=None):
    """Teacher writes Q&A pairs for each topic; appended to the seed set."""
    teacher = Teacher(provider=provider, api_key=api_key)
    rows = []
    topics = list(TOPICS)
    for i, topic in enumerate(topics):
        try:
            rows += teacher.generate_qa(topic, n_per_topic)
        except Exception as exc:
            if on_progress:
                on_progress(i + 1, len(topics), len(rows), error=str(exc))
            raise
        if on_progress:
            on_progress(i + 1, len(topics), len(rows))
    _save(SFT_PATH, rows)
    return rows


def generate_prefs(n=24, provider=None, api_key=None, on_progress=None):
    """RLAIF pairs: student samples two answers, teacher judges the winner."""
    teacher = Teacher(provider=provider, api_key=api_key)
    pool = load_sft(include_seed=True)
    random.shuffle(pool)
    questions = [r["question"] for r in pool[:n]]
    prefs = []
    for i, q in enumerate(questions):
        a1 = engine.generate(q, tuned=True, do_sample=True, max_new_tokens=200)
        a2 = engine.generate(q, tuned=True, do_sample=True, max_new_tokens=200)
        if a1.strip() == a2.strip():
            continue
        winner = teacher.judge(q, a1, a2)
        chosen, rejected = (a1, a2) if winner == "A" else (a2, a1)
        prefs.append({"question": q, "chosen": chosen, "rejected": rejected})
        if on_progress:
            on_progress(i + 1, len(questions), len(prefs))
    _save(PREF_PATH, prefs)
    return prefs
