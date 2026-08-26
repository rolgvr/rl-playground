import time
import numpy as np
from rl_learning.game.pong import PongGame
from rl_learning.game.agents.dqn import train

t = time.time()
res = train(PongGame(), variant="double", params={"episodes": 150, "seed": 1})
sc = [c["score"] for c in res["curve"]]
outs = [c["outcome"] for c in res["curve"][-30:]]
print(f"150 episodes in {time.time()-t:.0f}s")
print(f"  net score early={np.mean(sc[:30]):.1f}  late={np.mean(sc[-30:]):.1f}")
print(f"  wins in last 30: {outs.count('win')}/30   best checkpoint={res['best']['score']}")
