"""Autonomous self-improvement = distillation + agent-architecture search.

Phase 1 — distill: the teacher writes Q&A on the chosen topics and we SFT the
small student on it.
Phase 2 — search: the system builds several candidate *agent-loop architectures*
on its own (direct, self-refine on relevance, on correctness, double-critic,
self-consistency, verify+refine) and tests each on a fixed held-out topic set by
actually running the loop per question (student answers; critics judge). For every
configuration we save rich metadata — win-rate, avg LLM calls/question, refine
iterations, cost, time, and the delta vs the baseline — so you can see *why* one
architecture beats another and keep those learnings long-term.

Stops at a target win-rate OR a cap (rounds / minutes / USD). Works with an API
teacher (real) or a local stand-in (no key) so the whole machine is runnable.
Every event + per-config record is written to a JSONL the user can view/download.
"""

from __future__ import annotations

import json
import os
import time

RUN_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "autoimprove")
os.makedirs(RUN_DIR, exist_ok=True)

GEN_PROMPT = "You are an expert assistant. Answer the question accurately, concisely and practically."
REF_PROMPT = "Improve the previous answer using the critique. Keep what was right, fix what was flagged."


class LocalTeacher:
    """No-key stand-in: 'knows' the seed data, and can act as a (weak) local agent."""
    provider = "local"
    model = "local-qwen"

    def __init__(self):
        import random
        from .llm.seed_data import SEED_QA
        self.pool = [dict(x) for x in SEED_QA]
        random.shuffle(self.pool)
        self._i = 0

    def cost_usd(self):
        return 0.0

    def chat(self, system, user, max_tokens=300, temperature=0.2):
        from .llm import engine
        return engine.chat(system, user, max_new_tokens=min(320, max_tokens), do_sample=temperature > 0.3)

    def generate_for_topic(self, topic, n, focus=None):
        """Generate Q&A for the ACTUAL topic via the local model (domain-aware),
        not canned RL data. Lower quality than an API teacher, but on-topic."""
        from .llm import engine
        extra = (" Focus on: " + ", ".join(focus[:3])) if focus else ""
        qa = []
        for _ in range(n):
            q = engine.chat("You write one short, specific exam question. Output only the question.",
                            f"Topic: {topic}.{extra} Write one question.", max_new_tokens=60, do_sample=True)
            q = q.strip().split("\n")[0].strip()
            if not q:
                continue
            a = engine.chat("You are an expert. Answer correctly and concisely (2-4 sentences).",
                            f"Question: {q}", max_new_tokens=170)
            qa.append({"question": q, "answer": a.strip()})
        return qa or [{"question": f"Explain a key idea in {topic}.",
                       "answer": f"{topic} is an important area; key ideas should be studied in depth."}]

    def write_chapter(self, domain, subject, topics, words=300):
        """Local (weaker, free) textbook chapter for the reading period."""
        from .llm import engine
        return engine.chat("You write factual textbook prose.",
                           f"Write a dense ~{words}-word textbook passage on '{subject}' "
                           f"(part of {domain}). Cover: {topics or subject}. Plain prose only.",
                           max_new_tokens=420).strip()

    def design_curriculum(self, domain, n=None):
        """Build a curriculum FOR THE GIVEN DOMAIN using the local model; fall back
        to a generic domain-based outline (never the RL list)."""
        import re
        from .llm import engine
        n = n or 8
        txt = engine.chat(
            "You are a curriculum designer. Reply with a numbered list only, one short subject per line.",
            f"List {n} subjects, foundational to advanced, to become an expert in {domain}. No explanations.",
            max_new_tokens=240)
        subjects = []
        for line in txt.splitlines():
            s = re.sub(r"^\s*(\d+[\.\)]|[-*•])\s*", "", line).strip(" .")
            if 2 <= len(s) <= 80 and not s.lower().startswith(("here", "sure", "these", "subject")):
                if s not in subjects:
                    subjects.append(s)
        if len(subjects) < 2:   # fallback — still domain-specific, not RL
            subjects = [f"Foundations of {domain}", f"Core principles of {domain}",
                        f"Key methods in {domain}", f"Common pitfalls in {domain}",
                        f"Advanced topics in {domain}", f"Applications of {domain}"]
        return [{"subject": s, "why": f"part of {domain}", "depth_topics": []} for s in subjects[:n]]

    def judge(self, q, a, b):
        out = self.chat("Reply only 'A' or 'B' — which answer is better?",
                        f"Q: {q}\n\nA:\n{a}\n\nB:\n{b}\n\nBetter (A/B)?", max_tokens=3).strip().upper()
        return "A" if out.startswith("A") else "B"

    def score(self, q, a):
        out = self.chat("Grade the answer 1-5. Reply with only the digit.",
                        f"Q: {q}\n\nAnswer:\n{a}\n\nGrade (1-5):", max_tokens=3)
        for ch in out:
            if ch in "12345":
                return int(ch)
        return 3



def make_teacher(provider, api_key):
    from .llm.teacher import Teacher, available_provider
    if provider == "local":
        return LocalTeacher()
    if api_key or available_provider():
        try:
            t = Teacher(provider=(None if provider == "local" else provider), api_key=api_key)
            t.chat("ping", "Reply: ok", max_tokens=2)
            return t
        except Exception:
            pass
    return LocalTeacher()


# --- the architecture space: each builder returns (label, tactic, graph) -------
def _g(critic_model):
    """Candidate agent-loop architectures the search will try."""
    gen = {"id": "g", "type": "generator", "model": "local", "prompt": GEN_PROMPT}
    refn = {"id": "r", "type": "refiner", "model": "local", "prompt": REF_PROMPT}
    q = {"id": "q", "type": "input"}
    out = {"id": "o", "type": "output"}

    def crit(cid, ctype, th=4):
        return {"id": cid, "type": ctype, "model": critic_model, "threshold": th}

    configs = []
    # 1) direct — no critique
    configs.append(("Direct", "generator only, no critique",
                    {"nodes": [q, gen, out], "links": [
                        {"from": "q", "port": "out", "to": "g"}, {"from": "g", "port": "out", "to": "o"}]}))
    # 2) self-refine on relevance
    configs.append(("Self-refine · relevance", "critique relevance, refine if off-topic",
                    {"nodes": [q, gen, crit("c", "relevance_critic"), refn, out], "links": [
                        {"from": "q", "port": "out", "to": "g"}, {"from": "g", "port": "out", "to": "c"},
                        {"from": "c", "port": "pass", "to": "o"}, {"from": "c", "port": "fail", "to": "r"},
                        {"from": "r", "port": "out", "to": "c"}]}))
    # 3) self-refine on correctness
    configs.append(("Self-refine · correctness", "critique factual correctness, refine if wrong",
                    {"nodes": [q, gen, crit("c", "correctness_critic"), refn, out], "links": [
                        {"from": "q", "port": "out", "to": "g"}, {"from": "g", "port": "out", "to": "c"},
                        {"from": "c", "port": "pass", "to": "o"}, {"from": "c", "port": "fail", "to": "r"},
                        {"from": "r", "port": "out", "to": "c"}]}))
    # 4) double critic: relevance then correctness
    configs.append(("Double critic", "relevance AND correctness gates, refine on either",
                    {"nodes": [q, gen, crit("c", "relevance_critic"), crit("c2", "correctness_critic"), refn, out], "links": [
                        {"from": "q", "port": "out", "to": "g"}, {"from": "g", "port": "out", "to": "c"},
                        {"from": "c", "port": "pass", "to": "c2"}, {"from": "c", "port": "fail", "to": "r"},
                        {"from": "c2", "port": "pass", "to": "o"}, {"from": "c2", "port": "fail", "to": "r"},
                        {"from": "r", "port": "out", "to": "c"}]}))
    # 5) self-consistency
    configs.append(("Self-consistency", "sample several answers, keep the most complete",
                    {"nodes": [q, {"id": "g", "type": "self_consistency", "model": "local", "prompt": GEN_PROMPT}, out],
                     "links": [{"from": "q", "port": "out", "to": "g"}, {"from": "g", "port": "out", "to": "o"}]}))
    return configs


def run_autoimprove(topics, target=0.8, max_rounds=8, max_minutes=20, max_budget=0.50,
                    provider=None, api_key=None, on_event=None, on_node=None, stop_flag=None):
    from . import reasoning
    from .llm import engine

    teacher = make_teacher(provider, api_key)
    is_api = not isinstance(teacher, LocalTeacher)
    critic_model = "api" if is_api else "local"
    topic_str = ", ".join(topics) if isinstance(topics, list) else str(topics)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(RUN_DIR, f"run-{run_id}.jsonl")
    t0 = time.time()

    def log(ev):
        ev["t"] = round(time.time() - t0, 1)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev) + "\n")
        if on_event:
            on_event(ev)

    def capped():
        if stop_flag and stop_flag():
            return "user"
        if time.time() - t0 > max_minutes * 60:
            return "time"
        if teacher.cost_usd() >= max_budget and is_api:
            return "budget"
        return None

    log({"type": "start", "run_id": run_id, "topics": topic_str, "teacher": teacher.model,
         "is_api": is_api, "target": target, "caps": {"rounds": max_rounds, "minutes": max_minutes, "usd": max_budget}})

    eval_set = teacher.generate_for_topic(topic_str, 6)[:6]
    log({"type": "eval_set", "n": len(eval_set)})
    if not eval_set:
        log({"type": "error", "msg": "teacher produced no eval questions"})
        return {"run_id": run_id, "error": "no eval questions", "log_path": log_path}

    # Phase 1 — distil knowledge into the student
    train_qa = teacher.generate_for_topic(topic_str, 8)
    log({"type": "data", "added": len(train_qa), "total": len(train_qa), "cost": teacher.cost_usd()})
    curve = engine.sft_train([{"question": d["question"], "answer": d["answer"]} for d in train_qa], epochs=2, lr=1e-4)
    log({"type": "train", "examples": len(train_qa),
         "loss_start": curve[0] if curve else None, "loss_end": curve[-1] if curve else None})

    # Phase 2 — architecture search
    configs = _g(critic_model)[:max_rounds]
    results = []
    baseline_wr = None
    for ci, (label, tactic, graph) in enumerate(configs):
        reason = capped()
        if reason:
            log({"type": "stopped", "reason": reason}); break
        if on_node:
            on_node({"config": ci, "label": label, "graph": graph, "question": None, "node": None})
        log({"type": "config_start", "index": ci, "label": label, "tactic": tactic, "graph": graph})

        wins = ties = losses = 0
        calls = iters = 0
        cost_before = teacher.cost_usd()
        cstart = time.time()
        for qi, item in enumerate(eval_set):
            if capped():
                break
            q, gold = item["question"], item["answer"]
            if on_node:
                on_node({"config": ci, "label": label, "graph": graph, "question": q, "qi": qi, "node": None})
            res = reasoning.run_graph(graph, q, teacher=teacher, max_steps=16,
                                      on_step=lambda s: on_node and on_node({"config": ci, "node": s["id"], "question": q, "qi": qi}))
            student = res.get("final") or ""
            calls += len(res.get("steps", []))
            iters += res.get("iterations", 0)
            v1 = teacher.judge(q, student, gold)
            v2 = teacher.judge(q, gold, student)
            pref = (1 if v1 == "A" else 0) + (1 if v2 == "B" else 0)
            if pref == 2:
                wins += 1
            elif pref == 0:
                losses += 1
            else:
                ties += 1
        n = max(1, wins + ties + losses)
        winrate = round((wins + 0.5 * ties) / n, 3)
        if baseline_wr is None:
            baseline_wr = winrate
        rec = {"index": ci, "label": label, "tactic": tactic, "winrate": winrate,
               "wins": wins, "ties": ties, "losses": losses,
               "avg_calls": round(calls / n, 2), "avg_refine": round(iters / n, 2),
               "config_cost": round(teacher.cost_usd() - cost_before, 4),
               "time_s": round(time.time() - cstart, 1),
               "winrate_delta_vs_baseline": round(winrate - baseline_wr, 3),
               "nodes": [nd["type"] for nd in graph["nodes"]]}
        results.append(rec)
        log({"type": "config", **rec, "cost_total": teacher.cost_usd()})
        if winrate >= target:
            log({"type": "reached_target", "winrate": winrate, "label": label}); break

    best = max(results, key=lambda r: (r["winrate"], -r["avg_calls"])) if results else None
    summary = {"type": "final", "run_id": run_id, "configs_tested": len(results),
               "best": best, "results": results, "cost": teacher.cost_usd(),
               "log_path": log_path}
    log(summary)
    return summary
