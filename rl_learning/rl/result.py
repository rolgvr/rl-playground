"""RLResult -- the replayable record of one agent learning the game.

Stage 1's `Trace` recorded a search expanding cells. Reinforcement learning has
a different story to tell, so it needs a different record:

    * curve     -- reward (and steps) per training episode. This is the learning
                  curve: you watch it climb from "wandering randomly" to "going
                  straight to the goal".
    * snapshots -- the agent's value map and greedy policy, captured every so
                  often during training. Playing these back animates the policy
                  *forming* -- arrows snapping toward the goal, values bleeding
                  outward from it.
    * path      -- the route the FINAL learned policy takes. This is what gets
                  raced against A*.

`values` and `policy` are sent as full rows x cols grids (None on walls) so the
frontend can paint a heatmap and draw an arrow per cell with no extra logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


def values_grid(env, value_of: Dict) -> List[List[Optional[float]]]:
    """rows x cols grid of state values (None where there is a wall)."""
    g = env.grid
    out = [[None] * g.cols for _ in range(g.rows)]
    for s in env.states:
        out[s[0]][s[1]] = round(float(value_of.get(s, 0.0)), 3)
    return out


def policy_grid(env, action_of: Dict) -> List[List[Optional[int]]]:
    """rows x cols grid of greedy action indices (None where wall/terminal)."""
    g = env.grid
    out = [[None] * g.cols for _ in range(g.rows)]
    for s in env.states:
        if s == env.goal:
            continue
        a = action_of.get(s)
        if a is not None:
            out[s[0]][s[1]] = int(a)
    return out


@dataclass
class RLResult:
    algorithm: str
    category: str = "tabular_rl"          # "tabular_rl" | "deep_rl"

    curve: List[dict] = field(default_factory=list)        # [{episode, reward, steps}]
    snapshots: List[dict] = field(default_factory=list)    # [{label, values, policy}]
    path: List = field(default_factory=list)               # final greedy route (cells)

    found: bool = False
    path_length: int = 0
    path_cost: float = 0.0
    episodes: int = 0
    solved_at: Optional[int] = None       # first episode that reached the goal
    train_steps: int = 0                  # total environment steps taken
    time_ms: float = 0.0
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "category": self.category,
            "curve": self.curve,
            "snapshots": self.snapshots,
            "path": [list(c) for c in self.path],
            "found": self.found,
            "path_length": self.path_length,
            "path_cost": round(self.path_cost, 3),
            "episodes": self.episodes,
            "solved_at": self.solved_at,
            "train_steps": self.train_steps,
            "time_ms": round(self.time_ms, 3),
            "note": self.note,
        }
