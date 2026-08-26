"""A 'university for LLMs': a Dean agent designs a curriculum, the student is
taught each subject to mastery, and every mastered subject is saved as a SKILL on
that model. Skills can be toggled on/off and retrained; the model's adapter is
rebuilt from the currently-enabled skills (manifest + retrain).

Reuses the teacher/student machinery. Distill-to-mastery per subject: generate
in-depth Q&A → SFT → a per-subject exam (teacher-judged) → repeat until the
mastery target or a per-subject attempt cap. After a full pass the Dean re-plans
weak subjects. Works with an API teacher or the local stand-in.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time

from . import reasoning
from .auto_improve import LocalTeacher, make_teacher

UNI_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "university")
os.makedirs(UNI_DIR, exist_ok=True)


def _safe(s):
    return "".join(c for c in s if c.isalnum() or c in "-_ ").strip() or "model"


def model_dir(name):
    d = os.path.join(UNI_DIR, _safe(name))
    os.makedirs(os.path.join(d, "skills"), exist_ok=True)
    return d


def manifest_path(name):
    return os.path.join(model_dir(name), "manifest.json")


def load_manifest(name):
    p = manifest_path(name)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {"model": name, "domain": "", "skills": [], "curriculum": [], "created": time.strftime("%Y-%m-%d %H:%M")}


def save_manifest(name, m):
    with open(manifest_path(name), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)


def list_models():
    out = []
    if not os.path.isdir(UNI_DIR):
        return out
    for d in sorted(os.listdir(UNI_DIR)):
        mp = os.path.join(UNI_DIR, d, "manifest.json")
        if os.path.exists(mp):
            try:
                with open(mp, encoding="utf-8") as f:
                    m = json.load(f)
                out.append({"model": m.get("model", d), "domain": m.get("domain", ""),
                            "skills": [{"name": s["name"], "mastery": s["mastery"], "enabled": s["enabled"]}
                                       for s in m.get("skills", [])]})
            except Exception:
                continue
    return out


def delete_model(name):
    """Remove a saved model and everything it learned (manifest, skills, lessons)."""
    import shutil
    d = os.path.join(UNI_DIR, _safe(name))
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
        return True
    return False


THESES_DIR = os.path.join(UNI_DIR, "_theses")


def _thesis_meta(rec):
    return {k: rec.get(k) for k in ("id", "model", "domain", "title", "date", "gpa",
                                    "mastered", "total", "specs", "skills", "words")}


def save_thesis(rec):
    os.makedirs(THESES_DIR, exist_ok=True)
    with open(os.path.join(THESES_DIR, rec["id"] + ".json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    # also drop a readable .md alongside for download/portability
    with open(os.path.join(THESES_DIR, rec["id"] + ".md"), "w", encoding="utf-8") as f:
        f.write(rec.get("thesis_md", ""))


def list_theses():
    out = []
    if not os.path.isdir(THESES_DIR):
        return out
    for fn in sorted(os.listdir(THESES_DIR), reverse=True):
        if fn.endswith(".json"):
            try:
                with open(os.path.join(THESES_DIR, fn), encoding="utf-8") as f:
                    out.append(_thesis_meta(json.load(f)))
            except Exception:
                continue
    return out


def get_thesis(tid):
    p = os.path.join(THESES_DIR, _safe(tid) + ".json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        rec = json.load(f)
    # attach the coursework ("papers") the student studied, grouped by subject
    rec["coursework"] = {s["name"]: _load_skill_data(rec["model"], s["name"])
                         for s in rec.get("skills", [])}
    return rec


def write_thesis(model_name, domain, manifest, teacher, t0, say=None, log=None):
    """The graduating student writes a capstone thesis on its domain — section by
    section over the subjects it mastered — saved with the agent's specs so it can
    always be traced back to this exact model."""
    from .llm import engine
    skills = [s for s in manifest.get("skills", []) if s.get("mastery", 0) > 0]
    if not skills:
        return None
    title = f"A Study of {domain}"
    if say:
        say("student", "thesis", f"Writing my graduation thesis on {domain}…")
    abstract = engine.chat("You are a graduating student. Write a concise thesis abstract.",
                           f"Thesis title: '{title}'. Write a 3-4 sentence abstract.", max_new_tokens=200)
    sections = []
    for s in skills:
        body = engine.chat(
            f"You are writing the '{s['name']}' chapter of your thesis on {domain}. Be accurate and in depth.",
            f"Write 1-2 well-structured paragraphs on {s['name']} as part of a thesis on {domain}.",
            max_new_tokens=340)
        sections.append((s["name"], (body or "").strip()))
    conclusion = engine.chat("You are concluding your thesis.",
                             f"Write a 3-4 sentence conclusion to a thesis on {domain}.", max_new_tokens=200)
    md = (f"# {title}\n\n*A graduation thesis by **{model_name}***  \n"
          f"*Domain: {domain} · {time.strftime('%Y-%m-%d')}*\n\n"
          f"## Abstract\n\n{(abstract or '').strip()}\n\n"
          + "\n\n".join(f"## {n}\n\n{b}" for n, b in sections)
          + f"\n\n## Conclusion\n\n{(conclusion or '').strip()}\n")
    try:
        info = engine.status()
    except Exception:
        info = {}
    specs = {
        "base_model": getattr(engine, "BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
        "adapter": "LoRA (SFT)", "teacher": getattr(teacher, "model", "?"),
        "device": info.get("device", "?"), "mastery_target_skills": len(skills),
    }
    gpa = round(sum(s["mastery"] for s in skills) / max(1, len(skills)), 3)
    rec = {
        "id": _safe(f"{model_name}-{time.strftime('%Y%m%d-%H%M%S')}"),
        "model": model_name, "domain": domain, "title": title,
        "date": time.strftime("%Y-%m-%d %H:%M"), "gpa": gpa,
        "mastered": sum(1 for s in skills if s["mastery"] >= 0.7), "total": len(skills),
        "skills": [{"name": s["name"], "mastery": s["mastery"]} for s in skills],
        "specs": specs, "thesis_md": md, "words": len(md.split()),
    }
    save_thesis(rec)
    if log:
        log({"type": "thesis", "model": model_name, "title": title, "id": rec["id"], "words": rec["words"]})
    return rec


def lessons_path(name):
    return os.path.join(model_dir(name), "lessons.jsonl")


def append_lesson(name, rec):
    with open(lessons_path(name), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def load_lessons(name):
    p = lessons_path(name)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _skill_file(name, skill):
    return os.path.join(model_dir(name), "skills", _safe(skill) + ".jsonl")


def _skill_text_file(name, skill):
    return os.path.join(model_dir(name), "skills", _safe(skill) + ".txt")


def _save_skill_text(name, skill, text):
    """The subject's textbook chapter (the student's reading-period material)."""
    with open(_skill_text_file(name, skill), "w", encoding="utf-8") as f:
        f.write(text or "")


def _load_skill_text(name, skill):
    p = _skill_text_file(name, skill)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        t = f.read().strip()
    return [t] if t else []


def enabled_chapters(name):
    m = load_manifest(name)
    out = []
    for s in m.get("skills", []):
        if s.get("enabled"):
            out += _load_skill_text(name, s["name"])
    return out


# --- the library card: retrieve coursework at inference time -----------------
_STOP = {"the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are",
         "what", "how", "why", "does", "do", "can", "with", "be", "it", "this", "that"}


def _overlap(a: str, b: str) -> float:
    """Jaccard overlap of content words — a tiny, dependency-free retriever."""
    ta = set(re.findall(r"[a-z0-9]+", a.lower())) - _STOP
    tb = set(re.findall(r"[a-z0-9]+", b.lower())) - _STOP
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def library_lookup(name, question, k=3):
    """Top-k coursework Q&A from the model's ENABLED skills, by similarity to the
    question. Weights hold competence; the library holds the facts."""
    scored = []
    for s in load_manifest(name).get("skills", []):
        if not s.get("enabled"):
            continue
        for d in _load_skill_data(name, s["name"]):
            sc = _overlap(question, d["question"] + " " + d["answer"])
            if sc > 0.06:
                scored.append((sc, d))
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored[:k]]


# --- general-ability gauge: did the cramming break the base model? ----------
GENERAL_SET = [
    ("What causes the seasons on Earth?",
     "The tilt of Earth's rotational axis relative to its orbital plane. As Earth orbits the Sun, each hemisphere is tilted toward the Sun for part of the year, receiving more direct sunlight and longer days."),
    ("Explain the difference between a virus and a bacterium.",
     "Bacteria are single-celled living organisms that can reproduce on their own; viruses are non-living packets of genetic material in a protein coat that must hijack a host cell to replicate. Antibiotics kill bacteria but do nothing against viruses."),
    ("Why does ice float on water?",
     "Water expands when it freezes: hydrogen bonds lock molecules into an open hexagonal lattice, making ice less dense than liquid water, so it floats."),
    ("What is compound interest?",
     "Interest earned on both the original principal and previously accumulated interest, so a balance grows exponentially over time rather than linearly."),
    ("Summarise the plot of Romeo and Juliet in two sentences.",
     "Two teenagers from feuding families in Verona fall in love and secretly marry. A failed messenger and a faked death lead each to believe the other dead, and both die by suicide, reconciling their families in grief."),
    ("How does a rainbow form?",
     "Sunlight enters raindrops, is refracted, reflected internally and refracted again on exit; different wavelengths bend by different amounts, separating white light into a spectrum seen at about 42 degrees from the antisolar point."),
    ("What is the capital of Japan and roughly how many people live in its metro area?",
     "Tokyo; its greater metropolitan area holds roughly 37 million people, the largest urban area in the world."),
    ("Why can't you divide by zero?",
     "Division is the inverse of multiplication: a/b asks which number times b gives a. No number times zero gives a non-zero a, and 0/0 would be satisfied by every number — so the operation has no consistent definition."),
]


def general_check():
    """Held-out GENERAL perplexity, base vs tuned. A big rise on the tuned side
    means domain cramming is eroding general ability (catastrophic forgetting)."""
    from .llm import engine
    base = sum(engine.eval_loss(q, a, tuned=False) for q, a in GENERAL_SET) / len(GENERAL_SET)
    tuned = sum(engine.eval_loss(q, a, tuned=True) for q, a in GENERAL_SET) / len(GENERAL_SET)
    pb, pt = math.exp(base), math.exp(tuned)
    return {"ppl_base": round(pb, 2), "ppl_tuned": round(pt, 2),
            "drift_pct": round((pt - pb) / pb * 100, 1)}


def _save_skill_data(name, skill, qa):
    with open(_skill_file(name, skill), "w", encoding="utf-8") as f:
        for d in qa:
            f.write(json.dumps({"question": d["question"], "answer": d["answer"]}) + "\n")


def _load_skill_data(name, skill):
    p = _skill_file(name, skill)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def enabled_examples(name):
    m = load_manifest(name)
    exs = []
    for s in m.get("skills", []):
        if s.get("enabled"):
            exs += _load_skill_data(name, s["name"])
    return exs


def rebuild_model(name, on_progress=None):
    """Retrain the model's adapter from scratch on its currently-enabled skills:
    a reading pass over the chapters, then SFT on the coursework."""
    from .llm import engine
    exs = enabled_examples(name)
    engine.fresh_train(exs, epochs=2, lr=1e-4, on_progress=on_progress,
                       texts=enabled_chapters(name))
    m = load_manifest(name)
    m["rebuilt"] = time.strftime("%Y-%m-%d %H:%M")
    save_manifest(name, m)
    return {"examples": len(exs), "enabled": sum(1 for s in m["skills"] if s["enabled"])}


def set_skill_enabled(name, skill, enabled, on_progress=None):
    m = load_manifest(name)
    for s in m["skills"]:
        if s["name"] == skill:
            s["enabled"] = bool(enabled)
    save_manifest(name, m)
    return rebuild_model(name, on_progress)


def retrain_skill(name, skill, provider=None, api_key=None, on_progress=None):
    m = load_manifest(name)
    teacher = make_teacher(provider, api_key)
    subj = next((s for s in m.get("curriculum", []) if s["subject"] == skill), {"subject": skill, "depth_topics": []})
    topics = ", ".join(subj.get("depth_topics", [])) or skill
    qa = teacher.generate_for_topic(f"{m.get('domain','')} — {skill} (in depth: {topics})", 8)
    if qa:
        _save_skill_data(name, skill, qa)
    return rebuild_model(name, on_progress)


class RunContext:
    """Shared run plumbing — transcript logging, live agent chat, and the
    stop/time/budget caps. Used by both the solo run and the cohort so the
    behaviour (and the transcript format) stays identical."""

    def __init__(self, log_path, teacher, is_api, max_minutes, max_budget,
                 on_event=None, on_chat=None, stop_flag=None):
        self.log_path = log_path
        self.teacher = teacher
        self.is_api = is_api
        self.max_minutes = max_minutes
        self.max_budget = max_budget
        self.on_event = on_event
        self.on_chat = on_chat
        self.stop_flag = stop_flag
        self.t0 = time.time()

    @property
    def elapsed(self):
        return round(time.time() - self.t0, 1)

    def log(self, ev):
        ev["t"] = self.elapsed
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev) + "\n")
        if self.on_event:
            self.on_event(ev)

    def say(self, who, role, text, **extra):
        if self.on_chat and text:
            m = {"who": who, "role": role, "text": str(text).strip(), "t": self.elapsed}
            m.update(extra)
            self.on_chat(m)

    def capped(self):
        if self.stop_flag and self.stop_flag():
            return "user"
        if time.time() - self.t0 > self.max_minutes * 60:
            return "time"
        if self.is_api and self.teacher.cost_usd() >= self.max_budget:
            return "budget"
        return None


def judge_pairwise(teacher, q, ans, gold):
    """Position-swapped pairwise judging vs the reference answer. Solo and cohort
    exams BOTH use this, so masteries/GPAs are comparable across modes."""
    v1 = teacher.judge(q, ans, gold)
    v2 = teacher.judge(q, gold, ans)
    pref = (1 if v1 == "A" else 0) + (1 if v2 == "B" else 0)
    return "pass" if pref == 2 else ("fail" if pref == 0 else "partial")


def mastery_from(verdicts):
    """Mastery = (wins + half-credit ties) / n — the one formula for all modes."""
    n = max(1, len(verdicts))
    return round((sum(1 for v in verdicts if v == "pass")
                  + 0.5 * sum(1 for v in verdicts if v == "partial")) / n, 3)


# --- the exam loop (student answers via a small reasoning loop, lit live) ------
def _exam_graph(critic_model):
    return {"nodes": [
        {"id": "q", "type": "input"},
        {"id": "g", "type": "generator", "model": "local",
         "prompt": "Answer the question accurately, concisely and in depth."},
        {"id": "c", "type": "relevance_critic", "model": critic_model, "threshold": 4},
        {"id": "r", "type": "refiner", "model": "local",
         "prompt": "Improve the answer using the critique; keep what was right."},
        {"id": "o", "type": "output"}],
        "links": [{"from": "q", "port": "out", "to": "g"}, {"from": "g", "port": "out", "to": "c"},
                  {"from": "c", "port": "pass", "to": "o"}, {"from": "c", "port": "fail", "to": "r"},
                  {"from": "r", "port": "out", "to": "c"}]}


def run_university(model_name, domain, target=0.7, max_subjects=6, max_minutes=20, max_budget=0.50,
                  depth=18, provider=None, api_key=None, on_event=None, on_node=None, on_chat=None,
                  control=None, stop_flag=None):
    from .llm import engine

    # `control` is a live dict the UI mutates to intervene mid-run.
    control = control if control is not None else {}
    for _k in ("guidance", "questions", "curriculum_ops"):
        control.setdefault(_k, [])
    control.setdefault("overrides", {})

    def pull(key):
        lst = control.get(key) or []
        out = list(lst)
        del lst[:]
        return out

    teacher = make_teacher(provider, api_key)
    is_api = not isinstance(teacher, LocalTeacher)
    critic_model = "api" if is_api else "local"
    graph = _exam_graph(critic_model)
    log_path = os.path.join(model_dir(model_name), f"transcript-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    ctx = RunContext(log_path, teacher, is_api, max_minutes, max_budget,
                     on_event=on_event, on_chat=on_chat, stop_flag=stop_flag)
    log, say, capped, t0 = ctx.log, ctx.say, ctx.capped, ctx.t0

    manifest = load_manifest(model_name)
    manifest["domain"] = domain
    log({"type": "start", "model": model_name, "domain": domain, "teacher": teacher.model,
         "is_api": is_api, "target": target})

    # Dean designs the curriculum (0/None → the Dean decides how many subjects)
    n_subjects = max_subjects if (max_subjects and max_subjects > 0) else None
    curriculum = teacher.design_curriculum(domain, n_subjects)
    if not curriculum:
        log({"type": "error", "msg": "Dean produced no curriculum"})
        return {"model": model_name, "error": "no curriculum", "log_path": log_path}
    manifest["curriculum"] = curriculum
    save_manifest(model_name, manifest)
    log({"type": "curriculum", "subjects": [{"subject": c["subject"], "why": c.get("why", ""),
         "depth_topics": c.get("depth_topics", [])} for c in curriculum]})

    def upsert_skill(name, mastery, n):
        for s in manifest["skills"]:
            if s["name"] == name:
                s.update({"mastery": mastery, "enabled": True, "n_examples": n,
                          "updated": time.strftime("%Y-%m-%d %H:%M")})
                return
        manifest["skills"].append({"name": name, "mastery": mastery, "enabled": True,
                                   "n_examples": n, "updated": time.strftime("%Y-%m-%d %H:%M")})

    GRADE_TXT = {"pass": "✓ Correct — as good as the reference.",
                 "partial": "≈ Partial — some merit, not fully there.",
                 "fail": "✗ Below the reference answer."}

    def _answer_in_room(q, qi, name):
        """Run the student through the reasoning loop, narrating each step live."""
        if on_node:
            on_node({"subject": name, "graph": graph, "question": q, "qi": qi, "node": None})

        def _step(s, _q=q, _qi=qi):
            if on_node:
                on_node({"subject": name, "node": s["id"], "question": _q, "qi": _qi})
            st = s.get("type")
            if st in ("generator", "self_consistency"):
                say("student", "answer", s.get("text", ""), subject=name, qi=_qi)
            elif st == "refiner":
                say("student", "refine", s.get("text", ""), subject=name, qi=_qi)
            elif st in ("relevance_critic", "correctness_critic", "verifier"):
                who = "teacher" if s.get("model") == "api" else "student"
                say(who, "critique", s.get("text", ""), score=s.get("score"),
                    passed=s.get("passed"), subject=name, qi=_qi)

        res = reasoning.run_graph(graph, q, teacher=teacher, max_steps=14, on_step=_step)
        return res.get("final") or "", res.get("iterations", 0)

    per_round = max(4, int(depth) // 3)        # `depth` ≈ examples per subject over ≤3 rounds

    def teach(subj):
        name = subj["subject"]
        topics = ", ".join(subj.get("depth_topics", [])) or name
        qa_all = _load_skill_data(model_name, name)
        mastery = 0.0
        say("teacher", "lesson", f"Today we study {name}.", subject=name)

        # --- reading period: the teacher writes a textbook chapter; the student
        # is PRETRAINED on the raw text before any Q&A drills (facts in first) ---
        if not _load_skill_text(model_name, name) and hasattr(teacher, "write_chapter") and not capped():
            chapter = teacher.write_chapter(domain, name, topics)
            if chapter:
                _save_skill_text(model_name, name, chapter)
                say("teacher", "coursework", f"📖 Wrote the {name} chapter ({len(chapter.split())} words).", subject=name)
                say("student", "study", "Reading the chapter closely…", subject=name)
                engine.lm_train([chapter], epochs=1, lr=1e-4)
                log({"type": "reading", "subject": name, "words": len(chapter.split()),
                     "cost": teacher.cost_usd()})

        for attempt in range(3):
            if capped():
                break
            # --- intervention: your guidance steers this study round ---
            guides = pull("guidance")
            for g in guides:
                say("you", "guide", g, subject=name)
            focus = (subj.get("depth_topics", []) or []) + guides
            extra = (" Emphasise: " + "; ".join(guides)) if guides else ""
            qa = teacher.generate_for_topic(f"{domain} — {name} (in depth: {topics}).{extra}",
                                            per_round, focus=focus or None)
            qa_all += qa
            _save_skill_data(model_name, name, qa_all)
            if qa:
                say("teacher", "coursework",
                    f"Prepared {len(qa)} study Q&A on {name}. e.g. “{qa[0]['question']}”", subject=name)
            log({"type": "study", "subject": name, "attempt": attempt, "examples": len(qa_all),
                 "cost": teacher.cost_usd(), "guidance": guides})
            # --- replay: mix in earlier subjects so they aren't forgotten ---
            earlier = [d for s in manifest["skills"] if s["name"] != name and s.get("enabled")
                       for d in _load_skill_data(model_name, s["name"])]
            replay = random.sample(earlier, min(len(earlier), max(2, len(qa_all) // 4))) if earlier else []
            say("student", "study",
                f"Training on {len(qa_all)} examples" + (f" (+{len(replay)} replayed from earlier subjects)" if replay else "") + "…",
                subject=name)
            engine.sft_train([{"question": d["question"], "answer": d["answer"]} for d in qa_all + replay],
                             epochs=2, lr=1e-4)
            # --- exam (teacher questions + any you injected live) ---
            exam = teacher.generate_for_topic(f"{domain} — {name}", 4)
            exam += [{"question": uq, "answer": None, "user": True} for uq in pull("questions")]
            exam_records = []
            for qi, item in enumerate(exam):
                if capped():
                    break
                # absorb questions you inject mid-exam, too
                for uq in pull("questions"):
                    exam.append({"question": uq, "answer": None, "user": True})
                q, gold, is_user = item["question"], item.get("answer"), item.get("user", False)
                say("you" if is_user else "teacher", "question", q, subject=name, qi=qi)
                ans, iters = _answer_in_room(q, qi, name)
                if is_user or not gold:
                    score = teacher.score(q, ans) if hasattr(teacher, "score") else 3
                    verdict = "pass" if score >= 4 else ("partial" if score == 3 else "fail")
                    say("teacher", "grade", f"On your question — rubric {score}/5. " + GRADE_TXT[verdict],
                        verdict=verdict, score=score, subject=name, qi=qi, question=q, user=True)
                else:
                    verdict = judge_pairwise(teacher, q, ans, gold)
                    say("teacher", "grade", GRADE_TXT[verdict], verdict=verdict, subject=name, qi=qi, question=q)
                exam_records.append({"question": q, "answer": ans, "gold": gold, "verdict": verdict,
                                     "refine_iters": iters, "user": is_user})
            # --- intervention: apply any grade overrides you made ---
            overrides = control.get("overrides") or {}
            for rec in exam_records:
                ov = overrides.get(rec["question"])
                if ov in ("pass", "fail", "partial"):
                    rec["verdict"], rec["overridden"] = ov, True
                    say("you", "override", f"Marked “{rec['question'][:60]}…” as {ov}.", subject=name)
            overrides.clear()
            # mastery from graded (non-user) questions
            graded = [r["verdict"] for r in exam_records if not r.get("user")]
            wins = graded.count("pass")
            ties = graded.count("partial")
            losses = graded.count("fail")
            mastery = mastery_from(graded)
            log({"type": "exam", "subject": name, "attempt": attempt, "mastery": mastery,
                 "wins": wins, "ties": ties, "losses": losses})
            append_lesson(model_name, {
                "ts": time.time(), "t": round(time.time() - t0, 1),
                "subject": name, "why": subj.get("why", ""), "attempt": attempt,
                "depth_topics": subj.get("depth_topics", []), "guidance": guides,
                "coursework": [{"question": d["question"], "answer": d["answer"]} for d in qa],
                "exam": exam_records, "mastery": mastery,
                "wins": wins, "ties": ties, "losses": losses,
                "examples_total": len(qa_all), "cost": teacher.cost_usd()})
            if mastery >= target:
                break
        return mastery, len(qa_all)

    def apply_curriculum_ops(done_names, idx):
        """Apply your live edits to the (future part of the) curriculum."""
        changed = False
        for op in pull("curriculum_ops"):
            kind = op.get("op")
            subj = (op.get("subject") or "").strip()
            if not subj:
                continue
            low = subj.lower()
            if kind == "add" and not any(c["subject"].lower() == low for c in curriculum):
                curriculum.append({"subject": subj, "why": op.get("why") or "added by you",
                                   "depth_topics": op.get("depth_topics") or []})
                say("you", "curriculum", f"Added subject: {subj}")
                changed = True
            elif kind == "remove" and low not in done_names:
                for ci in range(len(curriculum) - 1, idx, -1):     # only future subjects
                    if curriculum[ci]["subject"].lower() == low:
                        curriculum.pop(ci); changed = True
                if changed:
                    say("you", "curriculum", f"Removed subject: {subj}")
        if changed:
            manifest["curriculum"] = curriculum
            save_manifest(model_name, manifest)
            log({"type": "curriculum", "subjects": [{"subject": c["subject"], "why": c.get("why", ""),
                 "depth_topics": c.get("depth_topics", [])} for c in curriculum]})

    # teach every subject to mastery
    done_names = set()
    idx = 0
    while idx < len(curriculum):
        if capped():
            log({"type": "stopped", "reason": capped()}); break
        apply_curriculum_ops(done_names, idx)
        if idx >= len(curriculum):
            break
        subj = curriculum[idx]
        log({"type": "subject_start", "subject": subj["subject"], "why": subj.get("why", "")})
        mastery, n = teach(subj)
        upsert_skill(subj["subject"], mastery, n)
        save_manifest(model_name, manifest)
        log({"type": "subject_done", "subject": subj["subject"], "mastery": mastery,
             "mastered": mastery >= target})
        done_names.add(subj["subject"].lower())
        idx += 1

    # one re-plan pass over weak subjects (the Dean deepens them)
    if not capped():
        weak = [s for s in manifest["skills"] if s["mastery"] < target]
        if weak:
            log({"type": "replan", "weak": [s["name"] for s in weak]})
            for s in weak:
                if capped():
                    break
                subj = next((c for c in curriculum if c["subject"] == s["name"]), {"subject": s["name"], "depth_topics": []})
                log({"type": "subject_start", "subject": s["name"], "why": "re-teaching (was weak)"})
                mastery, n = teach(subj)
                upsert_skill(s["name"], mastery, n)
                save_manifest(model_name, manifest)
                log({"type": "subject_done", "subject": s["name"], "mastery": mastery, "mastered": mastery >= target})

    # --- graduation rebuild + FINALS WEEK ------------------------------------
    # Subjects were trained sequentially, so early ones may have faded. Rebuild
    # the adapter ONCE on everything jointly (chapters + all coursework), then
    # run a cumulative final exam to measure what was actually RETAINED.
    general = None
    if manifest["skills"] and not capped():
        say("teacher", "lesson", "📝 Finals week — first a full revision, then a cumulative exam over every subject.")
        log({"type": "finals_start", "subjects": [s["name"] for s in manifest["skills"]]})
        try:
            engine.fresh_train(enabled_examples(model_name), epochs=2, lr=1e-4,
                               texts=enabled_chapters(model_name))
        except Exception as exc:
            log({"type": "error", "msg": f"graduation rebuild failed: {exc!r}"})
        for s in manifest["skills"]:
            if capped():
                break
            exam = teacher.generate_for_topic(f"{domain} — {s['name']}", 3)
            verdicts = []
            for item in exam:
                ans = engine.generate(item["question"])
                verdicts.append(judge_pairwise(teacher, item["question"], ans, item["answer"]))
            s["retention"] = mastery_from(verdicts)
            log({"type": "final_exam", "subject": s["name"], "retention": s["retention"]})
            say("teacher", "grade",
                f"Final on {s['name']}: retained {int(s['retention'] * 100)}% "
                f"(was {int(s['mastery'] * 100)}% right after study).", subject=s["name"])
        save_manifest(model_name, manifest)
        # did all that cramming damage general ability?
        try:
            general = general_check()
            log({"type": "general_check", **general})
        except Exception as exc:
            log({"type": "error", "msg": f"general check failed: {exc!r}"})

    # graduation thesis — the student's capstone, archived with its specs
    thesis = None
    try:
        thesis = write_thesis(model_name, domain, manifest, teacher, t0, say=say, log=log)
    except Exception as exc:
        log({"type": "error", "msg": f"thesis failed: {exc!r}"})

    mastered = sum(1 for s in manifest["skills"] if s["mastery"] >= target)
    retained = [s["retention"] for s in manifest["skills"] if s.get("retention") is not None]
    summary = {"type": "final", "model": model_name, "skills": manifest["skills"],
               "mastered": mastered, "total": len(manifest["skills"]),
               "gpa": round(sum(s["mastery"] for s in manifest["skills"]) / max(1, len(manifest["skills"])), 3),
               "retention": round(sum(retained) / len(retained), 3) if retained else None,
               "general": general,
               "cost": teacher.cost_usd(), "log_path": log_path,
               "thesis_id": thesis["id"] if thesis else None,
               "thesis_title": thesis["title"] if thesis else None}
    save_manifest(model_name, manifest)
    log(summary)
    return summary


# ============================ COHORT (competing students) ====================
PERSONAS = {
    "diligent": {"attend": 1.0, "learn": 1.0, "emoji": "🤓", "blurb": "never misses a lecture, always learns from rivals"},
    "balanced": {"attend": 0.8, "learn": 0.6, "emoji": "🙂", "blurb": "usually studies, sometimes learns from rivals"},
    "social":   {"attend": 0.5, "learn": 0.25, "emoji": "😎", "blurb": "often skips to the common area, rarely copies rivals"},
}
_PERSONA_ORDER = ["diligent", "balanced", "social"]


def _social_break(ctx, engine, domain, in_common, subject, rounds=2):
    """A coffee break in the common area. The students who skipped the lecture
    chat among *themselves* — purely social, off the coursework.

    This is room-isolated on purpose: only common-area students take part, none
    of it enters any student's study data (a break is not a lecture), and the
    lecture room neither hears it nor answers into it.
    """
    if not in_common:
        return
    names = ", ".join(s["name"] for s in in_common)
    ctx.say("common area", "social",
            f"☕ {names} are hanging out in the common area.",
            room="common", subject=subject)

    history = []
    for _ in range(max(1, rounds)):
        for st in in_common:
            if ctx.capped():
                return
            heard = history[-1] if history else ""
            prompt = (
                f"You're on a coffee break in the common area, taking a breather from "
                f"studying {domain}. "
                + (f'A friend just said: "{heard}". Reply casually. '
                   if heard else "Start a bit of friendly small talk. ")
                + "ONE short, relaxed sentence — chit-chat, a joke, plans — NOT about the coursework."
            )
            line = engine.cohort_generate(st["aid"], prompt, max_new_tokens=48,
                                          do_sample=True, temperature=0.9).strip()
            if not line:
                continue
            history.append(line)
            ctx.say(st["name"], "social", line, room="common",
                    subject=subject, persona=st["persona"])


def run_cohort(model_name, domain, n_students=3, target=0.7, max_subjects=6, max_minutes=20,
               max_budget=0.50, depth=18, provider=None, api_key=None, on_event=None, on_node=None,
               on_chat=None, control=None, stop_flag=None):
    """A term with several competing students. Each subject, every student
    autonomously picks the lecture room or the common area (persona-driven).

    Rooms are isolated. The lecture room hears the lecture, sits the exam, sees
    each other's answers, and may learn from the best *peer in that room*. The
    common area is a coffee break: those students chat socially among themselves,
    don't see the exam, and score 0 on that subject — a real cost to skipping.
    Nothing crosses between the two. Only the best student is saved at the end."""
    import random as _r
    from .llm import engine

    n_students = max(2, min(4, int(n_students)))
    teacher = make_teacher(provider, api_key)
    is_api = not isinstance(teacher, LocalTeacher)
    run_ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(model_dir(model_name), f"cohort-{run_ts}.jsonl")
    ctx = RunContext(log_path, teacher, is_api, max_minutes, max_budget,
                     on_event=on_event, on_chat=on_chat, stop_flag=stop_flag)
    log, say, capped, t0 = ctx.log, ctx.say, ctx.capped, ctx.t0

    aids = engine.cohort_init(n_students)
    students = []
    for i, aid in enumerate(aids):
        persona = _PERSONA_ORDER[i % len(_PERSONA_ORDER)]
        students.append({"aid": aid, "name": f"{model_name}#{i + 1}", "persona": persona,
                         "emoji": PERSONAS[persona]["emoji"], "data": {}, "verdicts": {},
                         "exams": {}, "learned": 0, "skipped": [], "where": "lecture"})

    log({"type": "cohort_start", "model": model_name, "domain": domain, "teacher": teacher.model,
         "is_api": is_api, "n_students": n_students,
         "students": [{"name": s["name"], "persona": s["persona"], "emoji": s["emoji"],
                       "blurb": PERSONAS[s["persona"]]["blurb"]} for s in students]})

    curriculum = teacher.design_curriculum(domain, max_subjects if max_subjects and max_subjects > 0 else None)
    if not curriculum:
        log({"type": "error", "msg": "Dean produced no curriculum"})
        return {"model": model_name, "error": "no curriculum", "log_path": log_path}
    log({"type": "curriculum", "subjects": [{"subject": c["subject"], "why": c.get("why", ""),
         "depth_topics": c.get("depth_topics", [])} for c in curriculum]})

    def _all_examples(st):
        out = []
        for v in st["data"].values():
            out += v
        return out

    _POINTS = {"pass": 2, "partial": 1, "fail": 0}

    for subj in curriculum:
        if capped():
            log({"type": "stopped", "reason": capped()}); break
        name = subj["subject"]
        topics = ", ".join(subj.get("depth_topics", [])) or name
        log({"type": "subject_start", "subject": name, "why": subj.get("why", "")})

        # one shared lecture, open to all
        lecture = teacher.generate_for_topic(f"{domain} — {name} (in depth: {topics})",
                                             max(4, int(depth) // 3))
        say("teacher", "lecture", f"📣 The lecture on {name} is now open to all students.", subject=name)

        # each student decides: attend the lecture, or head to the common area
        for st in students:
            attend = _r.random() < PERSONAS[st["persona"]]["attend"]
            st["where"] = "lecture" if attend else "common"
            if attend:
                st["data"].setdefault(name, []).extend(lecture)
                say(st["name"], "attend", f"{st['emoji']} Attending the lecture on {name}.",
                    subject=name, persona=st["persona"], room="lecture")
                engine.cohort_train(st["aid"], _all_examples(st), epochs=2, lr=1e-4)
            else:
                st["skipped"].append(name)
                say(st["name"], "social", f"{st['emoji']} Heading to the common area ☕ instead.",
                    subject=name, persona=st["persona"], room="common")
        in_lecture = [s for s in students if s["where"] == "lecture"]
        in_common = [s for s in students if s["where"] == "common"]
        log({"type": "cohort_where", "subject": name,
             "where": {s["name"]: s["where"] for s in students}})

        # the common area has its own social conversation, walled off from the exam
        _social_break(ctx, engine, domain, in_common, name)

        # the exam happens in the lecture room. ONLY students who are in that room
        # sit it — common-area students are on a break: they never see the
        # questions and can't answer. A skipped subject means empty verdicts, which
        # scores 0 mastery below, so skipping has a real cost (as it should).
        exam = teacher.generate_for_topic(f"{domain} — {name}", 5)
        for qi, item in enumerate(exam):
            if capped():
                break
            q, gold = item["question"], item["answer"]
            say("teacher", "question", q, subject=name, qi=qi, room="lecture")
            if on_node:
                on_node({"subject": name, "question": q, "qi": qi, "node": "exam"})
            graded = []
            for st in in_lecture:
                ans = engine.cohort_generate(st["aid"], q, max_new_tokens=200)
                verdict = judge_pairwise(teacher, q, ans, gold)
                st["verdicts"].setdefault(name, []).append(verdict)
                st["exams"].setdefault(name, []).append(
                    {"question": q, "answer": ans, "gold": gold, "verdict": verdict})
                graded.append({"st": st, "ans": ans, "verdict": verdict})
                say(st["name"], "answer", ans, subject=name, qi=qi,
                    persona=st["persona"], verdict=verdict, room="lecture")
            if not graded:
                continue   # everyone's on a break — no exam to reveal this round
            # reveal: who did best (head-to-head judge breaks a tie at the top)
            graded.sort(key=lambda g: _POINTS[g["verdict"]], reverse=True)
            best = graded[0]
            if len(graded) > 1 and _POINTS[graded[1]["verdict"]] == _POINTS[best["verdict"]]:
                if teacher.judge(q, graded[1]["ans"], best["ans"]) == "A":
                    best = graded[1]
            say("teacher", "reveal",
                f"🏅 Best answer to Q{qi + 1}: {best['st']['name']} ({best['verdict']}).",
                subject=name, qi=qi, room="lecture")
            # a student may learn from the best peer answer — but only among peers
            # who were in the SAME room. The common area can't copy the exam room.
            for g in graded:
                st = g["st"]
                if st is not best["st"] and _r.random() < PERSONAS[st["persona"]]["learn"]:
                    st["data"].setdefault(name, []).append({"question": q, "answer": best["ans"]})
                    st["learned"] += 1
                    say(st["name"], "learn", f"{st['emoji']} Studying {best['st']['name']}'s answer to improve.",
                        subject=name, qi=qi, persona=st["persona"], room="lecture")

        # subject mastery per student — same formula as solo runs
        board = []
        for st in students:
            m = mastery_from(st["verdicts"].get(name, []))
            st.setdefault("mastery", {})[name] = m
            board.append({"name": st["name"], "mastery": m})
        log({"type": "cohort_subject", "subject": name, "board": board, "cost": teacher.cost_usd()})

    # ---- graduation: rank the cohort, save only the best ----
    def gpa(st):
        ms = list(st.get("mastery", {}).values())
        return round(sum(ms) / max(1, len(ms)), 3)

    ranked = sorted(students, key=gpa, reverse=True)
    leaderboard = [{"name": s["name"], "persona": s["persona"], "emoji": s["emoji"], "gpa": gpa(s),
                    "skipped": s["skipped"], "learned_from_peers": s["learned"],
                    "subjects": s.get("mastery", {})} for s in ranked]
    log({"type": "cohort_leaderboard", "board": leaderboard})

    winner = ranked[0]
    say("teacher", "grade", f"🎓 The valedictorian is {winner['name']} (GPA {int(gpa(winner) * 100)}%). "
        f"Only this student is saved.", subject="")

    # archive EVERY student's full record (answers, verdicts, choices) — the
    # losers' contrast with the winner is half the experiment's value
    with open(os.path.join(model_dir(model_name), f"cohort-{run_ts}-students.json"),
              "w", encoding="utf-8") as f:
        json.dump({"domain": domain, "leaderboard": leaderboard,
                   "students": [{"name": s["name"], "persona": s["persona"], "gpa": gpa(s),
                                 "skipped": s["skipped"], "learned_from_peers": s["learned"],
                                 "exams": s["exams"]} for s in ranked]}, f, indent=2)

    # the winner's lessons go into the model's learning record (timeline)
    for subj_name, recs in winner["exams"].items():
        append_lesson(model_name, {
            "ts": time.time(), "t": ctx.elapsed, "subject": subj_name, "attempt": 0,
            "why": f"cohort term — won by {winner['name']} ({winner['persona']})",
            "coursework": winner["data"].get(subj_name, []), "exam": recs,
            "mastery": winner.get("mastery", {}).get(subj_name, 0),
            "wins": sum(1 for r in recs if r["verdict"] == "pass"),
            "ties": sum(1 for r in recs if r["verdict"] == "partial"),
            "losses": sum(1 for r in recs if r["verdict"] == "fail"),
            "examples_total": len(winner["data"].get(subj_name, [])),
            "cost": teacher.cost_usd()})

    engine.free()   # release the cohort adapters before rebuilding the winner cleanly

    # persist the winner as the standard, loadable model (manifest + skills + tuned adapter)
    manifest = load_manifest(model_name)
    manifest["domain"] = domain
    manifest["skills"] = []
    for name, m in winner.get("mastery", {}).items():
        _save_skill_data(model_name, name, winner["data"].get(name, []))
        manifest["skills"].append({"name": name, "mastery": m, "enabled": True,
                                   "n_examples": len(winner["data"].get(name, [])),
                                   "updated": time.strftime("%Y-%m-%d %H:%M")})
    manifest["curriculum"] = curriculum
    manifest["cohort"] = {"won_by": winner["persona"], "students": leaderboard,
                          "graduated": time.strftime("%Y-%m-%d %H:%M")}
    save_manifest(model_name, manifest)
    general = None
    try:
        engine.fresh_train(enabled_examples(model_name), epochs=2, lr=1e-4,
                           texts=enabled_chapters(model_name))
        general = general_check()
        log({"type": "general_check", **general})
    except Exception as exc:
        log({"type": "error", "msg": f"winner rebuild failed: {exc!r}"})

    thesis = None
    try:
        thesis = write_thesis(model_name, domain, manifest, teacher, t0, say=say, log=log)
    except Exception as exc:
        log({"type": "error", "msg": f"thesis failed: {exc!r}"})

    summary = {"type": "final", "model": model_name, "skills": manifest["skills"], "cohort": True,
               "winner": winner["name"], "winner_persona": winner["persona"],
               "leaderboard": leaderboard, "gpa": gpa(winner), "general": general,
               "mastered": sum(1 for s in manifest["skills"] if s["mastery"] >= target),
               "total": len(manifest["skills"]), "cost": teacher.cost_usd(), "log_path": log_path,
               "thesis_id": thesis["id"] if thesis else None,
               "thesis_title": thesis["title"] if thesis else None}
    log(summary)
    return summary
