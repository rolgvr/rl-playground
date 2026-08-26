"""Expected SARSA -- SARSA with the luck taken out of the update.

SARSA bootstraps off `Q(s', a')` for the single next action it happened to pick,
so its updates are noisy: sometimes `a'` is the greedy move, sometimes a random
exploration. Expected SARSA replaces that one sample with its *expectation* over
the policy:

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * sum_a' pi(a'|s') Q(s',a')  -  Q(s,a) ]

Under an epsilon-greedy policy that average is cheap: the greedy action carries
weight (1 - eps + eps/n) and every action carries eps/n. Removing the sampling
noise usually means smoother, faster, more stable learning than plain SARSA --
often the best-behaved of the tabular trio. It still respects the exploring
policy (so it keeps SARSA's caution under slipperiness) while learning more
steadily.
"""

from __future__ import annotations

import time

import numpy as np

from ..result import RLResult
from ._common import (epsilon_greedy, finalize, linear_epsilon, snapshot_from_Q,
                      snapshot_schedule)


def _expected_value(env, Q, i2, eps):
    """E_{a' ~ epsilon-greedy} [ Q(s', a') ]."""
    n = env.n_actions
    best = int(np.argmax(Q[i2]))
    probs = np.full(n, eps / n)
    probs[best] += 1.0 - eps
    return float(np.dot(probs, Q[i2]))


def train(env, episodes=400, alpha=0.2, gamma=0.97,
          eps_start=1.0, eps_end=0.05, seed=0):
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    result = RLResult(algorithm="Expected SARSA")
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
            target = r + (0.0 if done else gamma * _expected_value(env, Q, i2, eps))
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
