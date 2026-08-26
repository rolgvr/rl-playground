"""Verify the auto-improve architecture search runs and records per-config metadata."""
from rl_learning import auto_improve

events = []
summary = auto_improve.run_autoimprove(
    topics=["reinforcement learning best practices"],
    target=0.99, max_rounds=2, max_minutes=12, max_budget=1.0,   # invalid env key → local stand-in
    on_event=lambda ev: events.append(ev),
    on_node=lambda d: None,
)
types = [e["type"] for e in events]
print("event types:", types)
print("configs tested:", summary.get("configs_tested"))
for r in summary.get("results", []):
    print(f"  {r['label']:26} winrate={r['winrate']} calls/q={r['avg_calls']} refine={r['avg_refine']} Δ={r['winrate_delta_vs_baseline']}")
print("best:", (summary.get("best") or {}).get("label"))
import os
print("log exists:", os.path.exists(summary.get("log_path", "")))
