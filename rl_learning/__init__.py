"""rl_learning — a step-by-step reinforcement learning playground.

The package is organized as a ladder you climb one rung at a time:

    Stage 1  pathfinding   BFS, DFS, Dijkstra, A*        (rl_learning.algorithms)
    Stage 2  tabular RL     Q-learning, SARSA            (coming soon)
    Stage 3  deep RL        DQN, PPO                     (coming soon)

Every algorithm runs on the SAME object -- a `GridWorld` -- and returns the
SAME result shape -- a `Trace`.  That single contract is what lets the web app
animate any algorithm and race them against each other on identical data.
"""

from .grid import GridWorld
from .trace import Trace

__all__ = ["GridWorld", "Trace"]
__version__ = "0.1.0"
