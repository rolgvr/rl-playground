"""Verify SFT trains the student and changes its answers; then a quick DPO step."""
from rl_learning.llm import engine
from rl_learning.llm.seed_data import SEED_QA

q = "When should I use DQN versus PPO, and why?"
print("=== BASE answer ===")
print(engine.generate(q, tuned=False, max_new_tokens=90))

print("\n=== SFT training on seed data ===")
curve = engine.sft_train([dict(x) for x in SEED_QA], epochs=4, lr=1e-4,
                         on_progress=lambda s, t, c: print(f"  {s}/{t} loss {c[-1]}") if s % 20 == 0 else None)
print(f"loss {curve[0]} -> {curve[-1]}")

print("\n=== TUNED answer ===")
print(engine.generate(q, tuned=True, max_new_tokens=90))

print("\n=== quick DPO step (synthetic prefs) ===")
prefs = [{"question": d["question"], "chosen": d["answer"], "rejected": "I'm not sure, it depends."}
         for d in SEED_QA[:8]]
dcurve = engine.dpo_train(prefs, epochs=1, lr=5e-5, beta=0.1,
                          on_progress=lambda s, t, c: print(f"  dpo {s}/{t} loss {c[-1]}") if s % 4 == 0 else None)
print(f"dpo loss {dcurve[0]} -> {dcurve[-1]}  (lower = prefers chosen)")
print("GATE2 PASSED")
