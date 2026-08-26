"""Reasoning Lab — execute a graph of prompted agents (a reasoning loop).

Each node is an LLM call with a role + prompt, backed by either the local tuned
model or the teacher API. Critic nodes route pass/fail, so you can build
generate -> critique -> refine loops (Self-Refine / Reflexion).

The graph the browser sends is already cleaned to:
    nodes: [{id, type, prompt, model: "local"|"api", threshold}]
    links: [{from, port: "out"|"pass"|"fail", to}]
This runner walks it from the input node, keeping a running answer + feedback,
and follows critic pass/fail edges (bounded by max_steps to stop infinite loops).
"""

from __future__ import annotations

import re

# Palette: the behaviours the UI offers. `kind` drives execution + how many
# output ports a node has (critics have pass/fail).
NODE_TYPES = {
    "input":     {"label": "Question in", "kind": "io", "ports": ["out"],
                  "prompt": ""},
    "generator": {"label": "Generator", "kind": "gen", "ports": ["out"],
                  "prompt": "Answer the user's reinforcement-learning question accurately, concisely and practically."},
    "refiner":   {"label": "Refiner", "kind": "refine", "ports": ["out"],
                  "prompt": "Improve the previous answer using the critique. Keep what was right, fix what was flagged."},
    "relevance_critic": {"label": "Relevance critic", "kind": "critic", "ports": ["pass", "fail"],
                  "prompt": "Judge how directly the answer addresses the question. Reply 'SCORE: n' (1-5) then 'WHY: ...'."},
    "correctness_critic": {"label": "Correctness critic", "kind": "critic", "ports": ["pass", "fail"],
                  "prompt": "Judge the technical/factual correctness of the answer. Reply 'SCORE: n' (1-5) then 'WHY: ...'."},
    "verifier":  {"label": "Verifier (final gate)", "kind": "critic", "ports": ["pass", "fail"],
                  "prompt": "Is this answer good enough to return to the user? Reply 'SCORE: n' (1-5) then 'WHY: ...'."},
    "self_consistency": {"label": "Self-consistency (vote)", "kind": "consistency", "ports": ["out"],
                  "prompt": "Answer the question. (Sampled several times; the most consistent answer is kept.)"},
    "output":    {"label": "Answer out", "kind": "io", "ports": ["in"]},
}


def _agent(node, system, user, teacher, sample=False):
    """Run one node's LLM call on its chosen backend (falls back to local if no teacher)."""
    if node.get("model") == "api" and teacher is not None and hasattr(teacher, "chat"):
        return teacher.chat(system, user, max_tokens=400, temperature=0.7 if sample else 0.2)
    from .llm import engine
    return engine.chat(system, user, max_new_tokens=320, do_sample=sample)


def _parse_score(text):
    m = re.search(r"SCORE\s*[:=]?\s*([1-5])", text, re.I)
    if not m:
        m = re.search(r"\b([1-5])\b", text)
    score = int(m.group(1)) if m else 3
    why = re.split(r"WHY\s*[:=]", text, flags=re.I)
    feedback = why[1].strip() if len(why) > 1 else text.strip()
    return score, feedback[:600]


def run_graph(graph, question, provider=None, api_key=None, max_steps=24, on_step=None, teacher=None):
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    steps = []

    def emit(s):
        steps.append(s)
        if on_step:
            on_step(s)

    # adjacency: (from_id, port) -> to_id
    edge = {}
    for l in graph.get("links", []):
        edge[(l["from"], l.get("port", "out"))] = l["to"]

    # reuse a passed-in teacher (so cost is tracked across the run); else build one
    if teacher is None and any(n.get("model") == "api" for n in nodes.values()):
        try:
            from .llm.teacher import Teacher
            teacher = Teacher(provider=provider, api_key=api_key)
        except Exception:
            teacher = None

    # start at the input node (or the first generator if none)
    start = next((nid for nid, n in nodes.items() if n["type"] == "input"), None)
    if start is None:
        start = next((nid for nid, n in nodes.items() if n["type"] == "generator"), None)
    if start is None:
        return {"error": "add a Generator (and ideally a Question-in) node", "steps": []}

    ctx = {"question": question, "answer": "", "feedback": ""}
    cur = start
    iterations = 0

    for _ in range(max_steps):
        if cur is None or cur not in nodes:
            break
        node = nodes[cur]
        t = node["type"]
        spec = NODE_TYPES.get(t, {})
        kind = spec.get("kind")
        prompt = node.get("prompt") or spec.get("prompt", "")

        if kind == "io" and t == "input":
            emit({"id": cur, "type": t, "label": "Question", "text": question})
            cur = edge.get((cur, "out"))

        elif kind == "io" and t == "output":
            emit({"id": cur, "type": t, "label": "Final answer", "text": ctx["answer"]})
            break

        elif kind in ("gen", "refine"):
            user = f"Question: {ctx['question']}"
            if kind == "refine" and ctx["answer"]:
                user += f"\n\nPrevious answer:\n{ctx['answer']}\n\nCritique to address:\n{ctx['feedback'] or '(make it clearer)'}"
            elif ctx["feedback"]:
                user += f"\n\nNote from a reviewer:\n{ctx['feedback']}"
            out = _agent(node, prompt, user, teacher)
            ctx["answer"] = out
            emit({"id": cur, "type": t, "label": spec["label"], "text": out, "model": node.get("model")})
            cur = edge.get((cur, "out"))

        elif kind == "consistency":
            samples = [_agent(node, prompt, f"Question: {ctx['question']}", teacher, sample=True) for _ in range(3)]
            # keep the longest as a simple consensus proxy
            out = max(samples, key=len)
            ctx["answer"] = out
            emit({"id": cur, "type": t, "label": spec["label"], "text": out,
                          "model": node.get("model"), "note": f"{len(samples)} samples"})
            cur = edge.get((cur, "out"))

        elif kind == "critic":
            verdict_txt = _agent(node, prompt, f"Question: {ctx['question']}\n\nAnswer:\n{ctx['answer']}", teacher)
            score, feedback = _parse_score(verdict_txt)
            threshold = int(node.get("threshold", 4))
            passed = score >= threshold
            ctx["feedback"] = feedback
            emit({"id": cur, "type": t, "label": spec["label"], "score": score,
                          "threshold": threshold, "passed": passed, "text": feedback, "model": node.get("model")})
            if not passed:
                iterations += 1
            cur = edge.get((cur, "pass" if passed else "fail"))
        else:
            break

    return {"steps": steps, "final": ctx["answer"], "iterations": iterations}
