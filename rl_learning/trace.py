"""The single result format every algorithm produces.

A `Trace` is deliberately dumb: it is just *what happened*, recorded step by
step, plus the final answer and some metrics.  Pathfinding searches, tabular RL,
and deep RL all describe their work using this same vocabulary, so the frontend
only has to know how to play back one thing.

Why this matters: to compare "how quickly" A* finds the goal versus Dijkstra, we
do not compare wall-clock time alone (that is noisy on a tiny grid).  We compare
*work done* -- how many cells each algorithm had to expand before it found the
path.  The `visited` list captures exactly that, in order, so the animation can
reveal one expansion per frame and you literally watch the faster algorithm
finish first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Cell = Tuple[int, int]  # (row, col)


@dataclass
class Trace:
    """A replayable record of one algorithm solving one grid."""

    algorithm: str                       # e.g. "A*"
    category: str = "pathfinding"        # "pathfinding" | "tabular_rl" | "deep_rl"

    # The story, in order. `visited[i]` is the i-th cell the algorithm committed
    # to / expanded. Replaying these one per frame is the core animation.
    visited: List[Cell] = field(default_factory=list)

    # The answer.
    path: List[Cell] = field(default_factory=list)   # start -> goal, empty if none
    found: bool = False

    # Headline numbers shown under each grid so you can compare at a glance.
    nodes_expanded: int = 0      # how much work it took (lower = smarter search)
    path_length: int = 0         # number of steps in the path
    path_cost: float = 0.0       # sum of cell weights along the path
    time_ms: float = 0.0         # wall-clock, for reference

    note: str = ""               # optional one-liner the algorithm wants to show

    def finalize(self) -> "Trace":
        """Fill in the derived metrics from `visited` / `path`."""
        self.nodes_expanded = len(self.visited)
        self.path_length = max(len(self.path) - 1, 0)
        self.found = len(self.path) > 0
        return self

    @staticmethod
    def _node(n):
        # Grid nodes are (row, col) tuples -> emit [row, col]. Road-network
        # nodes are plain ids (ints) -> emit them as-is. The frontend looks up
        # an id's coordinates in the graph payload when drawing the map view.
        if isinstance(n, (tuple, list)):
            return list(n)
        return n

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "category": self.category,
            "visited": [self._node(c) for c in self.visited],
            "path": [self._node(c) for c in self.path],
            "found": self.found,
            "nodes_expanded": self.nodes_expanded,
            "path_length": self.path_length,
            "path_cost": round(self.path_cost, 3),
            "time_ms": round(self.time_ms, 3),
            "note": self.note,
        }
