"""Iterate on DQN settings quickly: how far does it get, and how fast?"""
import sys
from rl_learning.grid import GridWorld
from rl_learning.rl.env import GridGame
from rl_learning.rl.agents.dqn import train as dqn

# Simple-ish maze first (open with one wall), then the hard one.
def make(hard):
    if hard:
        return GridWorld(12, 12, start=(0, 0), goal=(11, 11),
                         walls={(r, 5) for r in range(0, 9)} | {(r, 8) for r in range(3, 12)})
    return GridWorld(10, 10, start=(0, 0), goal=(9, 9),
                     walls={(r, 4) for r in range(0, 7)})

for hard in (False, True):
    g = make(hard)
    env = GridGame(g, slip=0.0)
    eps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    r = dqn(env, episodes=eps, seed=1)
    last = r.curve[-1]["reward"]
    avg_last = sum(c["reward"] for c in r.curve[-20:]) / 20
    print(f"{'HARD' if hard else 'easy'} {g.rows}x{g.cols}: found={r.found} len={r.path_length} "
          f"reached={r.path[-1]} goal={env.goal} solved@{r.solved_at} "
          f"lastR={last} avg20={avg_last:.1f} time={r.time_ms:.0f}ms")
