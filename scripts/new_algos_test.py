"""Re-verify PPO after the ratio-shape fix."""
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from rl_learning.game.pong import PongGame
from rl_learning.game.agents import GAME_AGENTS

for aid in ["ppo"]:
    entry = GAME_AGENTS[aid]
    res = entry["train"](PongGame(), {"episodes": 300, "seed": 1}, None)
    sc = [c["score"] for c in res["curve"]]
    n = len(sc)
    early, late = np.mean(sc[: n // 5]), np.mean(sc[-n // 5:])
    wins = [c["outcome"] for c in res["curve"][-n // 5:]].count("win")
    print(f"{entry['label']:12} early={early:+.1f} late={late:+.1f} ({late-early:+.1f})  "
          f"wins_last20%={wins}/{n//5}  best={res['best']['score']}  {res['time_ms']/1000:.0f}s")
