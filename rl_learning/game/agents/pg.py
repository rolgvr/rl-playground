"""Policy-gradient agents: REINFORCE, A2C, and PPO.

Where the DQN family learns a *value* for every action and acts greedily, these
learn the *policy directly* — a network that outputs a probability for each move.
They are the other great branch of deep RL, and the one modern systems mostly
use. All three share one actor-critic network and differ only in how they turn an
episode of experience into a gradient:

    REINFORCE -- push up actions that led to high total return (Monte-Carlo),
                 using the critic only as a variance-reducing baseline.
    A2C       -- bootstrap the return with the critic (GAE) and take one
                 actor-critic step per episode, with an entropy bonus.
    PPO       -- like A2C, but reuse each episode for several epochs while
                 *clipping* how far the policy may move, for stable updates.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNet(nn.Module):
    """Shared conv trunk with an actor head (action logits) and a critic head V(s)."""

    def __init__(self, in_ch, rows, cols, n_actions):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        )
        flat = 64 * rows * cols
        self.fc = nn.Sequential(nn.Linear(flat, 256), nn.ReLU())
        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        h = self.fc(self.features(x).flatten(1))
        return self.actor(h), self.critic(h).squeeze(-1)


def _obs_t(env, obs):
    return torch.from_numpy(obs).unsqueeze(0).to(DEVICE)


def _greedy_play(env, net, record=True):
    """One deterministic game (argmax policy), recorded for the UI."""
    env.reset()
    frames = [env.frame()] if record else None
    done = False
    while not done:
        with torch.no_grad():
            logits, _ = net(_obs_t(env, env.observation()))
        _, _, done, _ = env.step(int(torch.argmax(logits)))
        if record:
            frames.append(env.frame())
    return {"score": round(env.score, 2), "outcome": env.outcome,
            "steps": env.steps, "frames": frames}


def _run_episode(env, net):
    """Play one episode sampling from the policy; return the trajectory tensors."""
    env.reset()
    obs_list, acts, logps, vals, rews = [], [], [], [], []
    done = False
    while not done:
        o = env.observation()
        logits, v = net(_obs_t(env, o))
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        obs_list.append(o)
        acts.append(int(a))
        logps.append(dist.log_prob(a))
        vals.append(v)
        _, r, done, _ = env.step(int(a))
        rews.append(r)
    return obs_list, acts, logps, vals, rews


def _gae(rews, vals, gamma, lam):
    """Generalised Advantage Estimation (terminal episode, so bootstrap = 0)."""
    adv, gae = [0.0] * len(rews), 0.0
    for t in reversed(range(len(rews))):
        next_v = vals[t + 1] if t + 1 < len(rews) else 0.0
        delta = rews[t] + gamma * next_v - vals[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    returns = [a + v for a, v in zip(adv, vals)]
    return adv, returns


def _config(env):
    return {"net_type": "policy", "rows": env.rows, "cols": env.cols,
            "n_channels": env.n_channels, "n_actions": env.n_actions}


def _result(variant, env, net, curve, checkpoints, episodes, t0):
    best = max(checkpoints, key=lambda c: c["score"]) if checkpoints else None
    return {"variant": variant, "curve": curve, "checkpoints": checkpoints, "best": best,
            "episodes": episodes,
            "device": (torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU"),
            "time_ms": round((time.perf_counter() - t0) * 1000, 1),
            "layout": env.static_layout(), "net": net, "config": _config(env)}


def _train_loop(env, net, params, update_fn, on_checkpoint):
    """Shared episode loop: run a game, update, record curve + checkpoints live."""
    episodes = int(params.get("episodes", 500))
    curve, checkpoints = [], []
    ckpt_every = max(1, episodes // 8)
    for ep in range(episodes):
        obs_list, acts, logps, vals, rews = _run_episode(env, net)
        update_fn(obs_list, acts, logps, vals, rews)
        curve.append({"episode": ep, "score": round(env.score, 2),
                      "steps": env.steps, "outcome": env.outcome})
        if ep % ckpt_every == 0 or ep == episodes - 1:
            roll = _greedy_play(env, net); roll["episode"] = ep
            checkpoints.append(roll)
            if on_checkpoint:
                on_checkpoint(ep, episodes, curve, roll)
    return curve, checkpoints, episodes


# --------------------------------------------------------------------------- REINFORCE
def train_reinforce(env, params=None, on_checkpoint=None):
    t0 = time.perf_counter()
    params = params or {}
    gamma = float(params.get("gamma", 0.99))
    ent_coef = float(params.get("entropy", 0.01))
    net = PolicyNet(env.n_channels, env.rows, env.cols, env.n_actions).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=float(params.get("lr", 0.001)))

    def update(obs_list, acts, logps, vals, rews):
        # Monte-Carlo discounted returns
        returns, g = [], 0.0
        for r in reversed(rews):
            g = r + gamma * g
            returns.insert(0, g)
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        values = torch.cat(vals)                             # [N] (cat, not stack)
        adv = returns - values.detach()                      # critic = baseline
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)        # normalise: steadier
        logp = torch.cat(logps)                              # [N]
        actor_loss = -(logp * adv).mean()
        critic_loss = F.mse_loss(values, returns)
        loss = actor_loss + 0.5 * critic_loss
        opt.zero_grad(); loss.backward(); opt.step()

    curve, ck, eps = _train_loop(env, net, params, update, on_checkpoint)
    return _result("reinforce", env, net, curve, ck, eps, t0)


# --------------------------------------------------------------------------- A2C
def train_a2c(env, params=None, on_checkpoint=None):
    t0 = time.perf_counter()
    params = params or {}
    gamma = float(params.get("gamma", 0.99))
    lam = float(params.get("gae_lambda", 0.95))
    ent_coef = float(params.get("entropy", 0.01))
    net = PolicyNet(env.n_channels, env.rows, env.cols, env.n_actions).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=float(params.get("lr", 0.001)))

    def update(obs_list, acts, logps, vals, rews):
        vals_d = [v.item() for v in vals]
        adv, returns = _gae(rews, vals_d, gamma, lam)
        adv = torch.tensor(adv, dtype=torch.float32, device=DEVICE)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        logits, values = net(torch.from_numpy(np.stack(obs_list)).to(DEVICE))
        dist = torch.distributions.Categorical(logits=logits)
        a = torch.tensor(acts, device=DEVICE)
        actor_loss = -(dist.log_prob(a) * adv).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy = dist.entropy().mean()
        loss = actor_loss + 0.5 * critic_loss - ent_coef * entropy
        opt.zero_grad(); loss.backward(); opt.step()

    curve, ck, eps = _train_loop(env, net, params, update, on_checkpoint)
    return _result("a2c", env, net, curve, ck, eps, t0)


# --------------------------------------------------------------------------- PPO
def train_ppo(env, params=None, on_checkpoint=None):
    t0 = time.perf_counter()
    params = params or {}
    gamma = float(params.get("gamma", 0.99))
    lam = float(params.get("gae_lambda", 0.95))
    clip = float(params.get("clip", 0.2))
    epochs = int(params.get("ppo_epochs", 4))
    ent_coef = float(params.get("entropy", 0.01))
    net = PolicyNet(env.n_channels, env.rows, env.cols, env.n_actions).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=float(params.get("lr", 0.0005)))

    def update(obs_list, acts, logps, vals, rews):
        vals_d = [v.item() for v in vals]
        adv, returns = _gae(rews, vals_d, gamma, lam)
        adv = torch.tensor(adv, dtype=torch.float32, device=DEVICE)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        obs = torch.from_numpy(np.stack(obs_list)).to(DEVICE)
        a = torch.tensor(acts, device=DEVICE)
        old_logp = torch.cat(logps).detach()              # [N] (cat, not stack)
        # several optimisation epochs over the same episode, with clipping
        for _ in range(epochs):
            logits, values = net(obs)
            dist = torch.distributions.Categorical(logits=logits)
            ratio = torch.exp(dist.log_prob(a) - old_logp)
            clipped = torch.clamp(ratio, 1 - clip, 1 + clip)
            actor_loss = -torch.min(ratio * adv, clipped * adv).mean()
            critic_loss = F.mse_loss(values, returns)
            entropy = dist.entropy().mean()
            loss = actor_loss + 0.5 * critic_loss - ent_coef * entropy
            opt.zero_grad(); loss.backward(); opt.step()

    curve, ck, eps = _train_loop(env, net, params, update, on_checkpoint)
    return _result("ppo", env, net, curve, ck, eps, t0)
