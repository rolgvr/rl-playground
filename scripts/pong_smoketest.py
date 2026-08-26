"""Does Pong train (net score rising), and does Pac-Man still work post-refactor?"""
import numpy as np
from rl_learning.game.pong import PongGame
from rl_learning.game.pacman import PacManGame
from rl_learning.game.agents.dqn import train, DEVICE

print("device:", DEVICE)

for name, env in [("Pong", PongGame()), ("Pac-Man", PacManGame())]:
    print(f"\n=== {name} ===  obs {env.observation().shape}  actions={env.n_actions}")
    res = train(env, variant="double", params={"episodes": 300, "seed": 1})
    sc = [c["score"] for c in res["curve"]]
    n = len(sc)
    early, late = np.mean(sc[: n // 5]), np.mean(sc[-n // 5:])
    outs = [c["outcome"] for c in res["curve"][-n // 5:]]
    wins = outs.count("win")
    print(f"  device={res['device']} time={res['time_ms']/1000:.1f}s")
    print(f"  score first 20%: {early:.1f}  last 20%: {late:.1f}  ({late-early:+.1f})")
    print(f"  wins in last 20%: {wins}/{len(outs)}   best checkpoint score={res['best']['score']}")
