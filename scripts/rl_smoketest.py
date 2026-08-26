"""Sanity check: do the RL agents converge to the pathfinder's optimum?"""
import random

from rl_learning.grid import GridWorld
from rl_learning.algorithms import ALGORITHMS
from rl_learning.rl.env import GridGame
from rl_learning.rl.agents import RL_AGENTS

random.seed(0)
g = GridWorld(
    12, 12, start=(0, 0), goal=(11, 11),
    walls={(r, 5) for r in range(0, 9)} | {(r, 8) for r in range(3, 12)},
    weights={(r, 3): 5 for r in range(0, 6)},
)
dij = ALGORITHMS["dijkstra"][0](g)
opt_cost = sum(g.cost(c) for c in dij.path[1:])
print(f"Dijkstra optimal: len={len(dij.path) - 1} cost={opt_cost}")

env = GridGame(g, slip=0.0)
print(f"env: {env.n_states} states, {env.n_actions} actions\n")

def report(r, label):
    cost = sum(g.cost(c) for c in r.path[1:]) if r.found else None
    if cost is None:
        match = "NO PATH"
    elif abs(cost - opt_cost) < 1e-6:
        match = "== optimal"
    else:
        match = f"cost={cost} (+{cost - opt_cost:.0f})"
    extra = f"solved@ep{r.solved_at}" if r.solved_at is not None else f"iters={r.episodes}"
    print(f"  {label:16} found={r.found} len={r.path_length:3} {match:20} {extra}  snaps={len(r.snapshots)} time={r.time_ms:.0f}ms")

print("--- deterministic ---")
for key in ["value_iteration", "policy_iteration", "q_learning", "sarsa", "expected_sarsa"]:
    fn, label, fam, _ = RL_AGENTS[key]
    r = fn(env) if fam == "model-based" else fn(env, episodes=600, seed=1)
    report(r, label)

print("\n--- DQN (neural net) ---")
from rl_learning.rl.agents import load_dqn
dqn_fn, dqn_label, _, _ = load_dqn()
report(dqn_fn(env, episodes=250, seed=1), dqn_label)

print("\n--- slippery (slip=0.2) — agents should still solve, paths may differ ---")
senv = GridGame(g, slip=0.2)
for key in ["value_iteration", "q_learning", "sarsa"]:
    fn, label, fam, _ = RL_AGENTS[key]
    r = fn(senv) if fam == "model-based" else fn(senv, episodes=800, seed=1)
    print(f"  {label:16} found={r.found} len={r.path_length:3} solved@{r.solved_at} snaps={len(r.snapshots)}")
