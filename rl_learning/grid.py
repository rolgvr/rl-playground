"""GridWorld -- the shared world every algorithm runs on.

This is "the same set of data" from the project goal.  You design one grid (the
maze, the start, the goal, optional heavy terrain) and hand the *exact same*
grid to BFS, Dijkstra, A*, and -- later -- a Q-learning agent.  Because they all
see identical data, any difference you observe is a difference in the algorithm,
not the problem.

Coordinates are (row, col), 0-indexed, origin top-left.

Weights let Dijkstra and A* shine: a cell's weight is the cost of *entering* it.
A plain maze has all weights 1, so Dijkstra behaves like BFS.  Paint some cells
with a higher weight ("mud") and the cost-aware searches will detour around them
while BFS/DFS plough straight through -- a vivid illustration of what "cost"
buys you.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

Cell = Tuple[int, int]

# 4-connected moves (up, down, left, right) and the 4 diagonals.
ORTHOGONAL = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAGONAL = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


class GridWorld:
    def __init__(
        self,
        rows: int,
        cols: int,
        start: Cell = (0, 0),
        goal: Optional[Cell] = None,
        walls: Optional[Iterable[Cell]] = None,
        weights: Optional[Dict[Cell, float]] = None,
        diagonal: bool = False,
    ):
        self.rows = rows
        self.cols = cols
        self.start = tuple(start)
        self.goal = tuple(goal) if goal is not None else (rows - 1, cols - 1)
        self.walls: Set[Cell] = {tuple(w) for w in (walls or [])}
        # weights maps a cell -> cost of entering it. Missing == 1.0 (normal).
        self.weights: Dict[Cell, float] = {tuple(k): float(v) for k, v in (weights or {}).items()}
        self.diagonal = diagonal

    # --- queries -----------------------------------------------------------

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def passable(self, cell: Cell) -> bool:
        return cell not in self.walls

    def cost(self, cell: Cell) -> float:
        """Cost of stepping *into* `cell`."""
        return self.weights.get(tuple(cell), 1.0)

    def neighbors(self, cell: Cell) -> List[Cell]:
        r, c = cell
        moves = ORTHOGONAL + DIAGONAL if self.diagonal else ORTHOGONAL
        result = []
        for dr, dc in moves:
            nxt = (r + dr, c + dc)
            if self.in_bounds(nxt) and self.passable(nxt):
                result.append(nxt)
        return result

    def move_cost(self, a: Cell, b: Cell) -> float:
        """Cost of moving between adjacent cells `a` and `b`.

        We use the *average* of the two cells' terrain weights so the cost is
        symmetric: going a->b costs the same as b->a. That symmetry is what lets
        bidirectional searches (which explore from the goal backwards) stay
        exact. Diagonal steps cost sqrt(2) times as much so diagonal movement is
        not artificially cheap. The minimum possible step cost stays at 1, which
        keeps the Manhattan/octile heuristic admissible.
        """
        base = (self.cost(a) + self.cost(b)) / 2.0
        if a[0] != b[0] and a[1] != b[1]:
            return base * 1.41421356
        return base

    # --- generic search interface ------------------------------------------
    # Every algorithm talks to a "problem" through just these two methods plus
    # `.start` / `.goal`. A road network (rl_learning/roads.py) implements the
    # same two methods, so the *identical* algorithm code runs on a grid or on
    # real streets -- the whole reason for this abstraction.

    def successors(self, node: Cell) -> List[Tuple[Cell, float]]:
        """Neighbours of `node` paired with the cost of reaching each."""
        return [(n, self.move_cost(node, n)) for n in self.neighbors(node)]

    def estimate(self, a: Cell, b: Cell) -> float:
        """Admissible distance estimate between any two cells.

        Manhattan distance for 4-connected grids, octile distance when diagonal
        moves are allowed. Both never overestimate as long as no step costs less
        than 1, so A* stays optimal. Bidirectional A* needs estimates toward
        *both* endpoints, which is why this takes two arguments.
        """
        dr = abs(a[0] - b[0])
        dc = abs(a[1] - b[1])
        if self.diagonal:
            return (dr + dc) + (1.41421356 - 2) * min(dr, dc)
        return dr + dc

    def heuristic(self, node: Cell) -> float:
        """Admissible estimate of the remaining cost from `node` to the goal."""
        return self.estimate(node, self.goal)

    # --- serialization (used by the web API) -------------------------------

    @classmethod
    def from_dict(cls, d: dict) -> "GridWorld":
        return cls(
            rows=int(d["rows"]),
            cols=int(d["cols"]),
            start=tuple(d.get("start", (0, 0))),
            goal=tuple(d["goal"]) if d.get("goal") is not None else None,
            walls=[tuple(w) for w in d.get("walls", [])],
            weights={tuple(k): v for k, v in d.get("weights", {}).items()} if isinstance(d.get("weights"), dict) else
                    {tuple(item[0]): item[1] for item in d.get("weights", [])},
            diagonal=bool(d.get("diagonal", False)),
        )

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "start": list(self.start),
            "goal": list(self.goal),
            "walls": [list(w) for w in self.walls],
            "weights": [[list(k), v] for k, v in self.weights.items()],
            "diagonal": self.diagonal,
        }
