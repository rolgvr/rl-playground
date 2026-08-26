"""Value iteration -- the model-based optimum the learners are chasing.

Unlike Q-learning and SARSA, this method is *handed the rules of the game* (the
full transition + reward model, via `env.transitions`). It never plays an
episode. Instead it repeatedly sweeps every state and applies the Bellman
optimality update:

    V(s) <- max_a  sum_s'  P(s'|s,a) * [ r + gamma * V(s') ]

In words: the value of a state is the best action's expected immediate reward
plus the discounted value of where you land. Sweep until the values stop
changing; the greedy policy with respect to the converged values is optimal.

Why it's here: it is the *ground truth*. On the deterministic grid its policy
traces the same least-cost route as Dijkstra/A*. The model-free agents are
trying to discover, from trial and error, what value iteration computes directly.
Race them and watch the learners' value heatmap morph toward this one.

(It needs the model, so it can't be used when the rules are unknown -- which is
exactly the situation real RL is built for. That's the catch worth feeling.)
"""

from __future__ import annotations

import time

from ..result import RLResult
from ._common import finalize, snapshot_from_VP, snapshot_schedule


def _greedy_action(env, V, s, gamma):
    best_a, best_q = None, float("-inf")
    for a in range(env.n_actions):
        q = sum(p * (r + (0.0 if done else gamma * V[s2]))
                for p, s2, r, done in env.transitions(s, a))
        if q > best_q:
            best_q, best_a = q, a
    return best_a, best_q


def train(env, gamma=0.97, theta=1e-4, max_iters=400):
    t0 = time.perf_counter()
    V = {s: 0.0 for s in env.states}
    result = RLResult(algorithm="Value Iteration", category="tabular_rl",
                      note="model-based: given the rules, no episodes")

    sweeps = 0
    # We don't know the sweep count up front, so capture snapshots adaptively.
    for it in range(max_iters):
        delta = 0.0
        for s in env.states:
            if env.is_terminal(s):
                continue
            _, best_q = _greedy_action(env, V, s, gamma)
            delta = max(delta, abs(best_q - V[s]))
            V[s] = best_q
        sweeps = it + 1

        action_of = {s: _greedy_action(env, V, s, gamma)[0]
                     for s in env.states if not env.is_terminal(s)}
        result.snapshots.append(snapshot_from_VP(env, V, action_of, f"sweep {it}"))
        result.curve.append({"episode": it, "reward": round(V[env.start], 2), "steps": it})
        if delta < theta:
            break

    # Thin snapshots if it took many sweeps (keep it light for the UI).
    keep = snapshot_schedule(len(result.snapshots))
    result.snapshots = [s for i, s in enumerate(result.snapshots) if i in keep]

    policy = {s: _greedy_action(env, V, s, gamma)[0] for s in env.states}
    result.episodes = sweeps
    finalize(env, lambda s: policy.get(s), result)
    result.time_ms = (time.perf_counter() - t0) * 1000
    return result
