r"""DQN (Deep Q-Network) -- the same idea as Q-learning, but the table becomes a
neural network.

Tabular Q-learning stores one number per (state, action). That works on a 12x12
grid but cannot scale to games with millions of states, and it can't *generalise*
-- learning about one cell teaches it nothing about the cell next door. DQN
replaces the table with a small neural network that maps a state's features
(here: the agent's normalised (row, col)) to the four Q-values. Now nearby states
share structure, and the same machinery would extend to far bigger problems.

Three ingredients make training a network on RL data stable -- the tricks from
the 2015 Atari DQN paper, in miniature:

    * Experience replay -- store transitions in a buffer and learn from random
      minibatches, breaking the correlation between consecutive steps.
    * A target network -- a frozen copy of the net supplies the bootstrap target
      r + gamma * max_a' Q_target(s', a'), so we aren't chasing a moving goal.
    * Gradient descent -- minimise the squared TD error with Adam instead of the
      simple tabular nudge.

It is heavier and noisier than the tabular methods on a problem this small (a
table is perfect here), but it is the bridge to "real" deep RL. Watch its value
heatmap: smoother and blurrier than the tabular ones, because the network
interpolates between cells instead of memorising each.
"""

from __future__ import annotations

import random
import time

import numpy as np
import torch
import torch.nn as nn

from ..result import RLResult, policy_grid, values_grid
from ._common import finalize, linear_epsilon, snapshot_schedule


class QNet(nn.Module):
    def __init__(self, n_inputs, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def train(env, episodes=300, gamma=0.97, lr=1e-3, batch=64, train_every=4,
          buffer_size=8000, target_sync=300, eps_start=1.0, eps_end=0.1, seed=0):
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)

    # One-hot encode the cell. Raw (row,col) can't express where the walls are --
    # the net would only see a coordinate. One-hot over passable cells lets the
    # network represent each state's value properly: a table learned by gradient
    # descent. (Richer features would buy spatial generalisation; one-hot keeps
    # the lesson honest and the maze learnable.)
    n_in = env.n_states

    def feat(s):
        v = [0.0] * n_in
        v[env.index[s]] = 1.0
        return v

    all_states = env.states
    all_feats = torch.tensor([feat(s) for s in all_states], dtype=torch.float32)
    # Cap episode length to bound CPU cost, but keep it generous: too tight and
    # early random episodes never stumble onto a far goal, starving the net of
    # any learning signal on long mazes.
    max_ep_steps = min(env.max_steps, 8 * (env.grid.rows + env.grid.cols))

    net = QNet(n_in, env.n_actions)
    target = QNet(n_in, env.n_actions)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    buffer = []
    result = RLResult(algorithm="DQN", category="deep_rl",
                      note="neural-net Q from (row,col); experience replay + target net")
    snaps = snapshot_schedule(episodes)
    total_steps = 0

    def snapshot(label):
        with torch.no_grad():
            q = net(all_feats).numpy()
        value_of = {s: float(np.max(q[i])) for i, s in enumerate(all_states)}
        action_of = {s: int(np.argmax(q[i])) for i, s in enumerate(all_states)}
        return {"label": label, "values": values_grid(env, value_of),
                "policy": policy_grid(env, action_of)}

    def act(s, eps):
        if rng.random() < eps:
            return int(rng.integers(env.n_actions))
        with torch.no_grad():
            return int(torch.argmax(net(torch.tensor(feat(s), dtype=torch.float32))))

    for ep in range(episodes):
        eps = linear_epsilon(ep, episodes, eps_start, eps_end)
        s = env.reset()
        ep_reward = 0.0
        steps = 0
        for _ in range(max_ep_steps):
            a = act(s, eps)
            s2, r, done = env.step(s, a)
            buffer.append((feat(s), a, r, feat(s2), done))
            if len(buffer) > buffer_size:
                buffer.pop(0)
            s = s2
            ep_reward += r
            steps += 1
            total_steps += 1

            # learn from a random minibatch (not every step -- cheaper, stabler)
            if len(buffer) >= batch and total_steps % train_every == 0:
                idx = rng.choice(len(buffer), batch, replace=False)
                bs, ba, br, bs2, bd = zip(*(buffer[i] for i in idx))
                bs = torch.tensor(bs, dtype=torch.float32)
                bs2 = torch.tensor(bs2, dtype=torch.float32)
                ba = torch.tensor(ba, dtype=torch.int64).unsqueeze(1)
                br = torch.tensor(br, dtype=torch.float32)
                bd = torch.tensor(bd, dtype=torch.float32)
                q = net(bs).gather(1, ba).squeeze(1)
                with torch.no_grad():
                    q_next = target(bs2).max(1).values
                    tgt = br + gamma * q_next * (1.0 - bd)
                loss = nn.functional.mse_loss(q, tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()

            if total_steps % target_sync == 0:
                target.load_state_dict(net.state_dict())
            if done:
                if result.solved_at is None:
                    result.solved_at = ep
                break

        result.curve.append({"episode": ep, "reward": round(ep_reward, 2), "steps": steps})
        if ep in snaps:
            result.snapshots.append(snapshot(f"ep {ep}"))

    result.episodes = episodes
    result.train_steps = total_steps

    def greedy(s):
        with torch.no_grad():
            return int(torch.argmax(net(torch.tensor(feat(s), dtype=torch.float32))))

    finalize(env, greedy, result)
    result.time_ms = (time.perf_counter() - t0) * 1000
    return result
