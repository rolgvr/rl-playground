"""Helpers shared by the RL agents: snapshots, greedy policies, finalisation."""

from __future__ import annotations

import numpy as np

from ..result import RLResult, policy_grid, values_grid


def snapshot_from_Q(env, Q, label: str) -> dict:
    """Turn a Q-table into a value+policy snapshot for the UI."""
    value_of = {s: float(np.max(Q[env.index[s]])) for s in env.states}
    action_of = {s: int(np.argmax(Q[env.index[s]])) for s in env.states}
    return {"label": label, "values": values_grid(env, value_of),
            "policy": policy_grid(env, action_of)}


def snapshot_from_VP(env, V: dict, action_of: dict, label: str) -> dict:
    return {"label": label, "values": values_grid(env, V),
            "policy": policy_grid(env, action_of)}


def epsilon_greedy(env, Q, s, eps: float, rng) -> int:
    if rng.random() < eps:
        return int(rng.integers(env.n_actions))
    return int(np.argmax(Q[env.index[s]]))


def linear_epsilon(ep: int, episodes: int, start: float, end: float) -> float:
    if episodes <= 1:
        return end
    return end + (start - end) * (1.0 - ep / (episodes - 1))


def path_cost(env, path) -> float:
    return sum(env.grid.cost(c) for c in path[1:])


def finalize(env, policy_fn, result: RLResult) -> RLResult:
    """Roll out the learned greedy policy and fill in the comparison metrics."""
    path = env.greedy_rollout(policy_fn)
    reached = bool(path) and path[-1] == env.goal
    result.path = path
    result.found = reached
    result.path_length = (len(path) - 1) if reached else 0
    result.path_cost = path_cost(env, path) if reached else 0.0
    return result


def snapshot_schedule(total: int, want: int = 24):
    """Indices at which to capture a snapshot (always includes the last)."""
    if total <= want:
        return set(range(total))
    step = max(1, total // want)
    keep = set(range(0, total, step))
    keep.add(total - 1)
    return keep
