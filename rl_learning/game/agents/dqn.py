"""Convolutional DQN (and its Double / Dueling variants) for Pac-Man, on GPU.

One network, one training loop; `variant` selects how the bootstrap target is
formed and whether the head is split into value/advantage streams. The agent
reads the board as a (channels, H, W) image and outputs a Q-value per move.
"""

from __future__ import annotations

import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvQNet(nn.Module):
    def __init__(self, in_ch, rows, cols, n_actions, dueling=False):
        super().__init__()
        self.dueling = dueling
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        )
        flat = 64 * rows * cols
        self.fc = nn.Sequential(nn.Linear(flat, 256), nn.ReLU())
        if dueling:
            self.value = nn.Linear(256, 1)
            self.adv = nn.Linear(256, n_actions)
        else:
            self.head = nn.Linear(256, n_actions)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x.flatten(1))
        if self.dueling:
            v = self.value(x)
            a = self.adv(x)
            return v + a - a.mean(dim=1, keepdim=True)
        return self.head(x)


def _play_episode(env, net, greedy=True, record=False):
    """Run one episode with the current policy; optionally record frames."""
    env.reset()
    frames = [env.frame()] if record else None
    done = False
    while not done:
        obs = torch.from_numpy(env.observation()).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            a = int(torch.argmax(net(obs)))
        _, _, done, _ = env.step(a)
        if record:
            frames.append(env.frame())
    return {"score": round(env.score, 2), "outcome": env.outcome,
            "steps": env.steps, "frames": frames}


def train(env, variant="dqn", params=None, on_checkpoint=None):
    params = params or {}
    episodes = int(params.get("episodes", 600))
    gamma = float(params.get("gamma", 0.97))
    lr = float(params.get("lr", 0.0005))
    batch = int(params.get("batch", 64))
    buffer_size = int(params.get("buffer", 20000))
    target_sync = int(params.get("target_sync", 800))
    eps_start = float(params.get("eps_start", 1.0))
    eps_end = float(params.get("eps_end", 0.05))
    decay_frac = float(params.get("eps_decay_frac", 0.6))
    dueling = variant in ("dueling", "dueling_double")
    double = variant in ("double", "dueling_double")

    t0 = time.perf_counter()
    torch.manual_seed(int(params.get("seed", 0)))
    random.seed(int(params.get("seed", 0)))

    net = ConvQNet(env.n_channels, env.rows, env.cols, env.n_actions, dueling).to(DEVICE)
    target = ConvQNet(env.n_channels, env.rows, env.cols, env.n_actions, dueling).to(DEVICE)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    buffer = deque(maxlen=buffer_size)
    curve = []
    checkpoints = []
    total_steps = 0
    decay_eps = max(1, int(episodes * decay_frac))

    # how often to record a watchable greedy game (≈8 checkpoints across training)
    ckpt_every = max(1, episodes // 8)

    for ep in range(episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * (ep / decay_eps))
        env.reset()
        obs = env.observation()
        done = False
        while not done:
            if random.random() < eps:
                a = random.randrange(env.n_actions)
            else:
                with torch.no_grad():
                    a = int(torch.argmax(net(torch.from_numpy(obs).unsqueeze(0).to(DEVICE))))
            nobs, r, done, _ = env.step(a)
            buffer.append((obs, a, r, nobs, done))
            obs = nobs
            total_steps += 1

            if len(buffer) >= batch:
                idx = np.random.randint(0, len(buffer), size=batch)
                bs, ba, br, bns, bd = zip(*(buffer[i] for i in idx))
                bs = torch.from_numpy(np.stack(bs)).to(DEVICE)
                bns = torch.from_numpy(np.stack(bns)).to(DEVICE)
                ba = torch.tensor(ba, dtype=torch.int64, device=DEVICE).unsqueeze(1)
                br = torch.tensor(br, dtype=torch.float32, device=DEVICE)
                bd = torch.tensor(bd, dtype=torch.float32, device=DEVICE)

                q = net(bs).gather(1, ba).squeeze(1)
                with torch.no_grad():
                    if double:
                        next_a = net(bns).argmax(1, keepdim=True)
                        q_next = target(bns).gather(1, next_a).squeeze(1)
                    else:
                        q_next = target(bns).max(1).values
                    tgt = br + gamma * q_next * (1.0 - bd)
                loss = F.smooth_l1_loss(q, tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()

            if total_steps % target_sync == 0:
                target.load_state_dict(net.state_dict())

        curve.append({"episode": ep, "score": round(env.score, 2),
                      "steps": env.steps, "outcome": env.outcome})

        if ep % ckpt_every == 0 or ep == episodes - 1:
            roll = _play_episode(env, net, greedy=True, record=True)
            roll["episode"] = ep
            checkpoints.append(roll)
            if on_checkpoint:
                on_checkpoint(ep, episodes, curve, roll)

    best = max(checkpoints, key=lambda c: c["score"]) if checkpoints else None
    config = {"net_type": "dqn", "rows": env.rows, "cols": env.cols,
              "n_channels": env.n_channels, "n_actions": env.n_actions, "dueling": dueling}
    return {
        "variant": variant,
        "curve": curve,
        "checkpoints": checkpoints,
        "best": best,
        "episodes": episodes,
        "device": (torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU"),
        "time_ms": round((time.perf_counter() - t0) * 1000, 1),
        "layout": env.static_layout(),
        "net": net,            # kept server-side for saving (stripped from JSON)
        "config": config,
    }
