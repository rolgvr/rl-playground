"""SARSA -- the on-policy cousin of Q-learning.

The name is the update's ingredients in order: State, Action, Reward, next
State, next Action -- (s, a, r, s', a'). The update is almost identical to
Q-learning, with one consequential change:

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * Q(s', a')  -  Q(s,a) ]

Q-learning bootstraps off `max_a' Q(s',a')` -- the *best* next action, whether or
not it will actually take it. SARSA bootstraps off `Q(s', a')` where `a'` is the
action it *really* takes next under its current exploring policy. So SARSA learns
the value of the policy it is genuinely following, exploration mistakes included.

Why it matters: on the slippery floor (or next to a cliff), SARSA learns a more
*cautious* policy -- it accounts for the random moves it might actually make and
steers clear of risky cells, where off-policy Q-learning assumes it will always
recover optimally. Flip slipperiness on and race the two to see SARSA hug the
safer route.
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
    result = RLResult(algorithm="SARSA")
    snaps = snapshot_schedule(episodes)
    total_steps = 0

    for ep in range(episodes):
        eps = linear_epsilon(ep, episodes, eps_start, eps_end)
        s = env.reset()
        a = epsilon_greedy(env, Q, s, eps, rng)
        ep_reward = 0.0
        steps = 0
        for _ in range(env.max_steps):
            s2, r, done = env.step(s, a)
            a2 = epsilon_greedy(env, Q, s2, eps, rng)
            i, i2 = env.index[s], env.index[s2]
            target = r + (0.0 if done else gamma * Q[i2, a2])
            Q[i, a] += alpha * (target - Q[i, a])
            s, a = s2, a2
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
