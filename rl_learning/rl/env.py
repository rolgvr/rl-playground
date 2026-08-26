"""GridGame -- the grid maze reframed as a reinforcement-learning environment.

This wraps a `GridWorld` (the same walls / mud / start / goal you designed for
the pathfinders) and turns it into a Markov Decision Process (MDP): the formal
object every RL method consumes. An MDP is just four things:

    * states   -- every passable cell
    * actions  -- move up / down / left / right
    * a transition rule  P(next state | state, action)
    * a reward for each step

Two ways to interact, on purpose:

    * `step(s, a)`        -- TRY an action and see one sampled outcome. This is
                            all a model-free learner (Q-learning, SARSA) gets: it
                            must learn from experience, like a real agent.
    * `transitions(s, a)` -- ask for the FULL probability distribution of
                            outcomes. Only model-based methods (value / policy
                            iteration) use this -- they are "given the rules".

The reward design makes the RL-optimal policy line up with the pathfinder's
optimum: entering a cell costs its terrain weight (1 normal, 5 mud), and the
goal pays a bonus. Maximising total reward therefore means reaching the goal by
the cheapest route -- exactly what Dijkstra/A* compute. That is what lets a
trained agent be compared head-to-head with A*.

Slipperiness adds stochasticity: with probability `slip`, the floor sends the
agent sideways instead of where it aimed. Now the world is uncertain, planning a
single fixed path is fragile, and *learning a robust policy* starts to matter.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

from ..grid import GridWorld

Cell = Tuple[int, int]

# Action order is fixed so the frontend can draw policy arrows: 0=up, 1=down,
# 2=left, 3=right.
ACTIONS: List[Cell] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["up", "down", "left", "right"]
# The two sideways actions a slip can divert each action into.
_PERP = {0: [2, 3], 1: [2, 3], 2: [0, 1], 3: [0, 1]}

GOAL_REWARD = 10.0      # bonus for reaching the treasure


class GridGame:
    def __init__(self, grid: GridWorld, slip: float = 0.0, max_steps: int = None):
        self.grid = grid
        self.slip = float(slip)
        self.start = grid.start
        self.goal = grid.goal
        self.n_actions = len(ACTIONS)

        # Enumerate the passable cells once; agents index Q-tables by these.
        self.states: List[Cell] = [
            (r, c)
            for r in range(grid.rows)
            for c in range(grid.cols)
            if grid.passable((r, c))
        ]
        self.index: Dict[Cell, int] = {s: i for i, s in enumerate(self.states)}
        self.n_states = len(self.states)
        self.max_steps = max_steps or (grid.rows * grid.cols * 2)

    # --- core dynamics -----------------------------------------------------

    def is_terminal(self, s: Cell) -> bool:
        return s == self.goal

    def _move(self, s: Cell, a: int) -> Cell:
        """Where action `a` lands from `s` (staying put if blocked)."""
        nr, nc = s[0] + ACTIONS[a][0], s[1] + ACTIONS[a][1]
        nxt = (nr, nc)
        if self.grid.in_bounds(nxt) and self.grid.passable(nxt):
            return nxt
        return s

    def _reward(self, s_next: Cell) -> float:
        if s_next == self.goal:
            return GOAL_REWARD
        # Entering a cell costs its terrain weight -> mud hurts more.
        return -self.grid.cost(s_next)

    def transitions(self, s: Cell, a: int) -> List[Tuple[float, Cell, float, bool]]:
        """Full distribution: list of (probability, next_state, reward, done).

        Used by value/policy iteration, which are handed the model.
        """
        if self.is_terminal(s):
            return [(1.0, s, 0.0, True)]

        # Probability mass over the action actually executed.
        outcomes = {a: 1.0 - self.slip}
        for p in _PERP[a]:
            outcomes[p] = outcomes.get(p, 0.0) + self.slip / 2.0

        merged: Dict[Cell, float] = {}
        for act, prob in outcomes.items():
            if prob <= 0:
                continue
            s_next = self._move(s, act)
            merged[s_next] = merged.get(s_next, 0.0) + prob

        return [
            (prob, s_next, self._reward(s_next), s_next == self.goal)
            for s_next, prob in merged.items()
        ]

    def step(self, s: Cell, a: int) -> Tuple[Cell, float, bool]:
        """Sample ONE outcome of taking action `a` in state `s`.

        This is the only window a model-free learner gets onto the world.
        """
        if self.slip > 0 and random.random() < self.slip:
            a = random.choice(_PERP[a])
        s_next = self._move(s, a)
        return s_next, self._reward(s_next), (s_next == self.goal)

    # --- helpers shared by agents -----------------------------------------

    def reset(self) -> Cell:
        return self.start

    def greedy_rollout(self, policy, max_steps: int = None) -> List[Cell]:
        """Follow a greedy policy from the start and record the route taken.

        `policy` maps a state to an action index. Stops at the goal, or if it
        loops / stalls (a not-yet-trained policy can wander forever).
        """
        max_steps = max_steps or self.max_steps
        path = [self.start]
        s = self.start
        seen = set()
        for _ in range(max_steps):
            if s == self.goal:
                break
            a = policy(s)
            if a is None:
                break
            s_next = self._move(s, a)        # show the *intended* route
            if s_next == s or (s_next, a) in seen:
                break                        # stuck against a wall or looping
            seen.add((s_next, a))
            path.append(s_next)
            s = s_next
        return path
