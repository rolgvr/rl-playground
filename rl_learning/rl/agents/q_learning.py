r"""Q-learning -- the headline model-free, off-policy learner.

The agent keeps a table Q(state, action): "how good is taking this action from
here, long-term?" It starts at all zeros (knows nothing) and improves the table
from raw experience using one update after every step:

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a')  -  Q(s,a) ]
                                 \_______________________/
                                  what we now think is true

The bracket is the "temporal-difference error": the gap between the old estimate
and a fresh, slightly-more-informed one. We nudge Q a fraction (`alpha`) toward
closing it. The `max_a'` is the *off-policy* part -- the agent bootstraps off the
best next action it knows of, even while it is still exploring with random moves.

Exploration: it acts epsilon-greedily (random move with probability epsilon,
best-known move otherwise) and `epsilon` decays from high to low, so it explores
wildly at first and exploits what it has learned later.

Watch the value heatmap: high value seeps outward from the goal as the agent
discovers which cells lead there, and the policy arrows swing to point along the
cheapest route -- converging on the same path A* computed.
"""

from __future__ import annotations

import time

import numpy as np

from ..result import RLResult
from ._common import (epsilon_greedy, finalize, linear_epsilon, snapshot_from_Q,
                      snapshot_schedule)


def train(env, episodes=400, alpha=0.2, gamma=0.97,
          eps_start=1.0, eps_end=0.05, seed=0):
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    result = RLResult(algorithm="Q-learning")
    snaps = snapshot_schedule(episodes)
    total_steps = 0

    for ep in range(episodes):
        eps = linear_epsilon(ep, episodes, eps_start, eps_end)
        s = env.reset()
        ep_reward = 0.0
        steps = 0
        for _ in range(env.max_steps):
            a = epsilon_greedy(env, Q, s, eps, rng)
            s2, r, done = env.step(s, a)
            i, i2 = env.index[s], env.index[s2]
            target = r + (0.0 if done else gamma * np.max(Q[i2]))
            Q[i, a] += alpha * (target - Q[i, a])
            s = s2
            ep_reward += r
            steps += 1
            total_steps += 1
            if done:
                if result.solved_at is None:
                    result.solved_at = ep
                break
        result.curve.append({"episode": ep, "reward": round(ep_reward, 2), "steps": steps})
        if ep in snaps:
            result.snapshots.append(snapshot_from_Q(env, Q, f"ep {ep}"))

    result.episodes = episodes
    result.train_steps = total_steps
    finalize(env, lambda s: int(np.argmax(Q[env.index[s]])), result)
    result.time_ms = (time.perf_counter() - t0) * 1000
    return result
