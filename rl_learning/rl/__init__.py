"""Stage 2 -- reinforcement learning on the grid game.

Where Stage 1 algorithms were *given the whole map* and computed a path, the
agents here start knowing nothing. They can only try actions, see what reward
they get, and gradually figure out a good policy. The same grid the pathfinders
solved becomes a game:

    * the agent is a player that can move N/S/E/W
    * reaching the goal pays a reward (+)
    * every step costs a little, and crossing mud costs more (-)
    * (optional) the floor is slippery, so actions sometimes go sideways

The big payoff: train an agent from scratch, then race its *learned* policy
against A* (which cheated with full knowledge) on the very same map.
"""

from .env import GridGame, ACTIONS, ACTION_NAMES
from .result import RLResult

__all__ = ["GridGame", "ACTIONS", "ACTION_NAMES", "RLResult"]
