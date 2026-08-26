"""Does the Pac-Man env work, and does the GPU DQN actually learn to play?"""
import sys
import numpy as np

from rl_learning.game.pacman import PacManGame
from rl_learning.game.agents.dqn import train, DEVICE

print("device:", DEVICE)

# 1) env mechanics with a random policy
env = PacManGame()
print(f"maze {env.rows}x{env.cols}  pellets={len(env.pellets)}  ghosts={env.n_ghosts}")
obs = env.reset()
print("observation shape:", obs.shape)
import random
random.seed(0)
done = False
while not done:
    _, r, done, _ = env.step(random.randrange(4))
print(f"random episode: outcome={env.outcome} score={env.score:.1f} steps={env.steps}")

# 2) train DQN and check the score curve rises
variant = sys.argv[2] if len(sys.argv) > 2 else "double"
eps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
print(f"\ntraining variant={variant} episodes={eps} ...")
res = train(env, variant=variant, params={"episodes": eps, "seed": 1})
curve = [c["score"] for c in res["curve"]]
n = len(curve)
early = np.mean(curve[: n // 5])
late = np.mean(curve[-n // 5:])
wins = sum(1 for c in res["curve"][-n // 5:] if c["outcome"] == "win")
print(f"device={res['device']} time={res['time_ms']/1000:.1f}s")
print(f"avg score  first 20%: {early:.1f}   last 20%: {late:.1f}   (improvement: {late-early:+.1f})")
print(f"best checkpoint: score={res['best']['score']} outcome={res['best']['outcome']} @ep{res['best']['episode']}")
print(f"wins in last 20% of episodes: {wins}/{n//5}")
print(f"checkpoints recorded: {len(res['checkpoints'])}, frames in best: {len(res['best']['frames'])}")
