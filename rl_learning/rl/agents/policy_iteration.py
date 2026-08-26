"""Policy iteration -- optimise the policy directly, in two alternating steps.

Value iteration improves *values* until they're optimal, then reads off a policy.
Policy iteration works on the *policy* itself, bouncing between two phases until
the policy stops changing:

    1. Policy evaluation  -- fix the current policy and compute how good every
       state is *under that policy* (solve V for this policy).
    2. Policy improvement -- at every state, switch to the action that now looks
       best given those values.

Each round makes the policy provably no worse, and on a finite MDP it lands on
the optimal policy -- usually in very few rounds (often a handful), even though
each round does more work than a single value-iteration sweep. It's the same
destination as value iteration by a different road, and a clean illustration
that "evaluate, then act greedily, repeat" converges.
"""

from __future__ import annotations

import time

from ..result import RLResult
from ._common import finalize, snapshot_from_VP, snapshot_schedule


def _evaluate(env, policy, gamma, theta=1e-4, max_iters=1000):
    """Iterative policy evaluation: V for the given (deterministic) policy."""
    V = {s: 0.0 for s in env.states}
    for _ in range(max_iters):
        delta = 0.0
        for s in env.states:
            if env.is_terminal(s):
                continue
            a = policy[s]
            v = sum(p * (r + (0.0 if done else gamma * V[s2]))
                    for p, s2, r, done in env.transitions(s, a))
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V


def _greedy(env, V, s, gamma):
    best_a, best_q = None, float("-inf")
    for a in range(env.n_actions):
        q = sum(p * (r + (0.0 if done else gamma * V[s2]))
                for p, s2, r, done in env.transitions(s, a))
        if q > best_q:
            best_q, best_a = q, a
    return best_a


def train(env, gamma=0.97, max_rounds=50):
    t0 = time.perf_counter()
    policy = {s: 0 for s in env.states}      # start with "always go up"
    result = RLResult(algorithm="Policy Iteration", category="tabular_rl",
                      note="model-based: evaluate then improve")

    rounds = 0
    V = {s: 0.0 for s in env.states}
    for it in range(max_rounds):
        V = _evaluate(env, policy, gamma)
        stable = True
        for s in env.states:
            if env.is_terminal(s):
                continue
            best_a = _greedy(env, V, s, gamma)
            if best_a != policy[s]:
                policy[s] = best_a
                stable = False
        rounds = it + 1

        action_of = {s: policy[s] for s in env.states if not env.is_terminal(s)}
        result.snapshots.append(snapshot_from_VP(env, V, action_of, f"round {it}"))
        result.curve.append({"episode": it, "reward": round(V[env.start], 2), "steps": it})
        if stable:
            break

    keep = snapshot_schedule(len(result.snapshots))
    result.snapshots = [s for i, s in enumerate(result.snapshots) if i in keep]

    result.episodes = rounds
    finalize(env, lambda s: policy.get(s), result)
    result.time_ms = (time.perf_counter() - t0) * 1000
    return result
