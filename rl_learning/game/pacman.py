"""A compact Pac-Man environment for deep reinforcement learning.

Deliberately small (a ~9x9 maze, one or two ghosts) so a convolutional DQN can
show visible learning in minutes rather than the days a full 28x31 Atari board
would need. The mechanics are the real ones:

    * eat pellets (.) for points; clear them all to win the board
    * power pellets (o) turn the ghosts blue and edible for a while
    * a normal ghost catching you ends the game

The agent never sees "x,y coordinates". It sees the board as a stack of binary
image layers -- walls, pellets, power pellets, itself, normal ghosts, scared
ghosts -- exactly the kind of raw, high-dimensional observation deep RL exists
to handle. `observation()` returns that (C, H, W) tensor-ready array.

Ghosts use a simple, legible AI: breadth-first chase toward Pac-Man (or flee
when scared), with a dash of randomness so they are not perfectly predictable.
"""

from __future__ import annotations

import random
from collections import deque
from typing import List, Tuple

import numpy as np

# A small, connected maze. '#' wall, '.' pellet, 'o' power pellet,
# ' ' empty corridor, 'P' Pac-Man start, 'G' ghost start.
DEFAULT_MAZE = [
    "#########",
    "#o.....o#",
    "#.#.#.#.#",
    "#.......#",
    "#.#.#.#.#",
    "#...G...#",
    "#.#.#.#.#",
    "#o..P..o#",
    "#########",
]

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right
ACTION_NAMES = ["up", "down", "left", "right"]

# observation channels
CH_WALL, CH_PELLET, CH_POWER, CH_PAC, CH_GHOST, CH_SCARED = range(6)
N_CHANNELS = 6

# rewards
R_PELLET = 1.0
R_POWER = 3.0
R_EAT_GHOST = 8.0
R_STEP = -0.02      # mild time pressure
R_DEATH = -15.0
R_WIN = 30.0

SCARED_DURATION = 24


class PacManGame:
    def __init__(self, maze: List[str] = None, n_ghosts: int = 2, max_steps: int = 350):
        self.maze_src = maze or DEFAULT_MAZE
        self.rows = len(self.maze_src)
        self.cols = len(self.maze_src[0])
        self.n_actions = 4
        self.n_channels = N_CHANNELS
        self.n_ghosts = n_ghosts
        self.max_steps = max_steps

        self.walls = np.zeros((self.rows, self.cols), dtype=bool)
        self.pac_start = (1, 1)
        self.ghost_start = (self.rows // 2, self.cols // 2)
        self._pellet_cells: List[Tuple[int, int]] = []
        self._power_cells: List[Tuple[int, int]] = []

        for r, line in enumerate(self.maze_src):
            for c, ch in enumerate(line):
                if ch == "#":
                    self.walls[r, c] = True
                elif ch == ".":
                    self._pellet_cells.append((r, c))
                elif ch == "o":
                    self._power_cells.append((r, c))
                elif ch == "P":
                    self.pac_start = (r, c)
                elif ch == "G":
                    self.ghost_start = (r, c)
        self.reset()

    # --- lifecycle ---------------------------------------------------------

    def reset(self):
        self.pac = self.pac_start
        self.pellets = set(self._pellet_cells)
        # corridor cells without an explicit symbol also get a pellet, so the
        # board is satisfyingly full.
        for r in range(self.rows):
            for c in range(self.cols):
                if not self.walls[r, c] and (r, c) not in self._power_cells \
                        and (r, c) != self.pac_start and (r, c) != self.ghost_start:
                    self.pellets.add((r, c))
        self.power = set(self._power_cells)
        self.ghosts = [self._ghost_spawn(i) for i in range(self.n_ghosts)]
        self.scared = 0
        self.steps = 0
        self.score = 0.0
        self.done = False
        self.outcome = None        # "win" | "caught" | "timeout"
        return self.observation()

    def _ghost_spawn(self, i: int) -> Tuple[int, int]:
        gr, gc = self.ghost_start
        # spread ghosts out a little around the pen
        offsets = [(0, 0), (0, 1), (0, -1), (-1, 0)]
        r, c = gr + offsets[i % 4][0], gc + offsets[i % 4][1]
        if self.in_bounds((r, c)) and not self.walls[r, c]:
            return (r, c)
        return (gr, gc)

    # --- helpers -----------------------------------------------------------

    def in_bounds(self, cell) -> bool:
        return 0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols

    def passable(self, cell) -> bool:
        return self.in_bounds(cell) and not self.walls[cell[0], cell[1]]

    def _bfs_dist(self, source):
        """Distance from `source` to every passable cell (for ghost AI)."""
        dist = {source: 0}
        q = deque([source])
        while q:
            cur = q.popleft()
            for dr, dc in ACTIONS:
                nxt = (cur[0] + dr, cur[1] + dc)
                if self.passable(nxt) and nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    q.append(nxt)
        return dist

    def _move_ghost(self, pos):
        dist = self._bfs_dist(self.pac)
        options = [(pos[0] + dr, pos[1] + dc) for dr, dc in ACTIONS]
        options = [o for o in options if self.passable(o)]
        if not options:
            return pos
        if random.random() < 0.2:            # a little randomness
            return random.choice(options)
        scared = self.scared > 0
        # chase -> minimise distance to Pac-Man; flee -> maximise it
        key = (lambda o: -dist.get(o, 999)) if scared else (lambda o: dist.get(o, 999))
        return min(options, key=key)

    # --- step --------------------------------------------------------------

    def step(self, action: int):
        if self.done:
            return self.observation(), 0.0, True, {}

        reward = R_STEP
        self.steps += 1
        pac_old = self.pac

        # Pac-Man moves (stays put if it walks into a wall).
        target = (self.pac[0] + ACTIONS[action][0], self.pac[1] + ACTIONS[action][1])
        if self.passable(target):
            self.pac = target
        pac_new = self.pac

        # eat pellet / power pellet
        if self.pac in self.pellets:
            self.pellets.discard(self.pac)
            reward += R_PELLET
            self.score += R_PELLET
        if self.pac in self.power:
            self.power.discard(self.pac)
            self.scared = SCARED_DURATION
            reward += R_POWER
            self.score += R_POWER

        # ghosts move; a collision is landing on Pac-Man OR swapping cells with it
        caught = False
        updated = []
        for i, g in enumerate(self.ghosts):
            ng = self._move_ghost(g)
            collide = (ng == pac_new) or (g == pac_new and ng == pac_old)
            if collide:
                if self.scared > 0:
                    reward += R_EAT_GHOST
                    self.score += R_EAT_GHOST
                    ng = self._ghost_spawn(i)          # send the ghost home
                else:
                    caught = True
            updated.append(ng)
        self.ghosts = updated

        if self.scared > 0:
            self.scared -= 1

        info = {}
        if caught:
            reward += R_DEATH
            self.done = True
            self.outcome = "caught"
        elif not self.pellets:
            reward += R_WIN
            self.score += R_WIN
            self.done = True
            self.outcome = "win"
        elif self.steps >= self.max_steps:
            self.done = True
            self.outcome = "timeout"

        return self.observation(), reward, self.done, info

    # --- observation / rendering ------------------------------------------

    def observation(self) -> np.ndarray:
        obs = np.zeros((N_CHANNELS, self.rows, self.cols), dtype=np.float32)
        obs[CH_WALL] = self.walls.astype(np.float32)
        for (r, c) in self.pellets:
            obs[CH_PELLET, r, c] = 1.0
        for (r, c) in self.power:
            obs[CH_POWER, r, c] = 1.0
        obs[CH_PAC, self.pac[0], self.pac[1]] = 1.0
        for g in self.ghosts:
            obs[CH_SCARED if self.scared > 0 else CH_GHOST, g[0], g[1]] = 1.0
        return obs

    def frame(self) -> dict:
        """A JSON-able snapshot the browser can draw and animate."""
        return {
            "pac": list(self.pac),
            "ghosts": [{"pos": list(g), "scared": self.scared > 0} for g in self.ghosts],
            "pellets": [list(p) for p in self.pellets],
            "power": [list(p) for p in self.power],
            "score": round(self.score, 2),
            "step": self.steps,
            "scared": self.scared,
            "done": self.done,
            "outcome": self.outcome,
        }

    def static_layout(self) -> dict:
        return {
            "game": "pacman",
            "rows": self.rows,
            "cols": self.cols,
            "walls": [[r, c] for r in range(self.rows) for c in range(self.cols) if self.walls[r, c]],
        }
