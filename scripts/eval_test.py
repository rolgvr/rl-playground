"""Verify the held-out evaluation produces sensible base-vs-tuned metrics (no judge)."""
import json
from rl_learning.llm import evaluate

rep = evaluate.run_eval(use_judge=False)
print(json.dumps({k: v for k, v in rep.items() if k != "samples"}, indent=2))
print("\nexample question:", rep["samples"][0]["question"])
print("base  :", rep["samples"][0]["base"][:110])
print("tuned :", rep["samples"][0]["tuned"][:110])
