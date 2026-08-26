"""Stage 3 (deep RL) -- a real game: Pac-Man.

The maze stages taught planning and tabular learning on a problem small enough to
enumerate. Pac-Man is the opposite: pellets times ghost positions times power-up
timers is an astronomically large state space, so there is no table to fill. The
agent must learn straight from the *screen* -- a stack of grid layers (walls,
pellets, itself, the ghosts) fed to a convolutional network on the GPU. This is
the classic "deep RL plays an Atari game" setup, in miniature.
"""

from .pacman import PacManGame

__all__ = ["PacManGame"]
