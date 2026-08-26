"""Save, list, load and test trained game models.

A saved model is two files in `models/`:
  * <name>.pt   -- the network weights + the config needed to rebuild it (torch)
  * <name>.json -- lightweight metadata, so listing models is fast (no torch load)

`build_net` reconstructs the right architecture from the config, so a model
saved months ago can be reloaded and watched playing again.
"""

from __future__ import annotations

import glob
import json
import os
import time

import torch

from .dqn import ConvQNet
from .pg import PolicyNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def _safe(name: str) -> str:
    keep = "-_.() "
    return "".join(c for c in name if c.isalnum() or c in keep).strip() or "model"


def build_net(config: dict):
    if config["net_type"] == "dqn":
        net = ConvQNet(config["n_channels"], config["rows"], config["cols"],
                       config["n_actions"], config.get("dueling", False))
    else:
        net = PolicyNet(config["n_channels"], config["rows"], config["cols"], config["n_actions"])
    return net.to(DEVICE)


def greedy_action(net, obs_t) -> int:
    out = net(obs_t)
    logits = out[0] if isinstance(out, tuple) else out   # policy: (logits, value); dqn: q
    return int(torch.argmax(logits))


def save_model(name: str, net, config: dict, meta: dict) -> str:
    name = _safe(name)
    base = os.path.join(MODELS_DIR, name)
    torch.save({"config": config, "meta": meta, "state_dict": net.state_dict()}, base + ".pt")
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return name


def list_models() -> list:
    out = []
    for jp in sorted(glob.glob(os.path.join(MODELS_DIR, "*.json"))):
        try:
            with open(jp, encoding="utf-8") as f:
                meta = json.load(f)
            out.append({"name": os.path.splitext(os.path.basename(jp))[0], **meta})
        except Exception:
            continue
    return out


def load_model(name: str):
    path = os.path.join(MODELS_DIR, _safe(name) + ".pt")
    if not os.path.exists(path):
        return None
    blob = torch.load(path, map_location=DEVICE, weights_only=False)
    net = build_net(blob["config"])
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return net, blob["config"], blob["meta"]


def delete_model(name: str):
    base = os.path.join(MODELS_DIR, _safe(name))
    for ext in (".pt", ".json"):
        if os.path.exists(base + ext):
            os.remove(base + ext)


def play_record(env, net) -> dict:
    """Play one greedy game with a loaded net; record frames for the UI."""
    env.reset()
    frames = [env.frame()]
    done = False
    while not done:
        with torch.no_grad():
            obs_t = torch.from_numpy(env.observation()).unsqueeze(0).to(DEVICE)
            a = greedy_action(net, obs_t)
        _, _, done, _ = env.step(a)
        frames.append(env.frame())
    return {"score": round(env.score, 2), "outcome": env.outcome,
            "steps": env.steps, "frames": frames}
