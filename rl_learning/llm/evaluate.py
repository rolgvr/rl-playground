"""Evaluate base vs tuned on the held-out set with industry-standard metrics.

  * win_rate   -- LLM-as-judge pairwise, A/B order swapped to cancel position bias
                  (Chatbot-Arena / AlpacaEval style). Needs a teacher key.
  * rubric     -- LLM-as-judge absolute 1-5 correctness+helpfulness (MT-Bench).
                  Needs a teacher key.
  * perplexity -- exp(mean cross-entropy) of the reference answer under each model
                  on the held-out set. Lower is better. Intrinsic, no key.
  * rouge_l    -- longest-common-subsequence F1 of the generated answer vs the
                  reference. Higher is better. Lexical, no key.

All on a fixed held-out set with a temperature-0 judge, so scores are consistent
and comparable across runs.
"""

from __future__ import annotations

import math

from . import engine
from .eval_set import EVAL_SET


# --- lexical metric (no API) ----------------------------------------------

def _lcs_len(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[len(a)][len(b)]


def rouge_l(pred: str, ref: str) -> float:
    p, r = pred.lower().split(), ref.lower().split()
    if not p or not r:
        return 0.0
    lcs = _lcs_len(p, r)
    prec, rec = lcs / len(p), lcs / len(r)
    return round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0


# --- the evaluation run ----------------------------------------------------

def run_eval(use_judge=False, provider=None, api_key=None, on_progress=None):
    teacher = None
    if use_judge:
        from .teacher import Teacher
        teacher = Teacher(provider=provider, api_key=api_key)

    n = len(EVAL_SET)
    samples = []
    base_loss, tuned_loss = [], []
    base_rouge, tuned_rouge = [], []
    base_rubric, tuned_rubric = [], []
    tuned_w = tie = base_w = 0

    for i, item in enumerate(EVAL_SET):
        q, ref = item["question"], item["reference"]
        base_ans = engine.generate(q, tuned=False, max_new_tokens=200)
        tuned_ans = engine.generate(q, tuned=True, max_new_tokens=200)

        # intrinsic metrics
        base_loss.append(engine.eval_loss(q, ref, tuned=False))
        tuned_loss.append(engine.eval_loss(q, ref, tuned=True))
        base_rouge.append(rouge_l(base_ans, ref))
        tuned_rouge.append(rouge_l(tuned_ans, ref))

        rec = {"question": q, "base": base_ans, "tuned": tuned_ans}

        if teacher is not None:
            # pairwise win rate with position swap
            v1 = teacher.judge(q, tuned_ans, base_ans)   # A=tuned
            v2 = teacher.judge(q, base_ans, tuned_ans)   # A=base
            votes_for_tuned = (1 if v1 == "A" else 0) + (1 if v2 == "B" else 0)
            if votes_for_tuned == 2:
                tuned_w += 1; rec["winner"] = "tuned"
            elif votes_for_tuned == 0:
                base_w += 1; rec["winner"] = "base"
            else:
                tie += 1; rec["winner"] = "tie"
            # rubric scores
            rb = teacher.score(q, base_ans); rt = teacher.score(q, tuned_ans)
            base_rubric.append(rb); tuned_rubric.append(rt)
            rec["rubric_base"], rec["rubric_tuned"] = rb, rt

        samples.append(rec)
        if on_progress:
            on_progress(i + 1, n, [])

    def avg(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    report = {
        "n": n,
        "judge_used": teacher is not None,
        "perplexity": {"base": round(math.exp(avg(base_loss)), 2),
                       "tuned": round(math.exp(avg(tuned_loss)), 2)},
        "eval_loss": {"base": avg(base_loss), "tuned": avg(tuned_loss)},
        "rouge_l": {"base": avg(base_rouge), "tuned": avg(tuned_rouge)},
        "samples": samples,          # all held-out Q&A (base + tuned answers)
    }
    if teacher is not None:
        report["win_rate"] = {"tuned": tuned_w, "tie": tie, "base": base_w,
                              "tuned_pct": round(100 * tuned_w / n, 1)}
        report["rubric"] = {"base": avg(base_rubric), "tuned": avg(tuned_rubric)}
    return report
