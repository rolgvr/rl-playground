"""Depth-first search.

The idea: pick a direction and commit -- keep going deeper until you hit a dead
end, then back up and try the next branch. It uses a stack (here, an explicit
list) instead of BFS's queue. That one change -- LIFO instead of FIFO -- is the
entire difference between the two.

What to watch: DFS shoots off in long snaking tendrils rather than tidy rings.
On many mazes it stumbles onto *a* path very fast while expanding few cells, but
that path is usually far from the shortest. It is the cautionary tale of the
group: cheap, but no quality guarantee.
"""

from __future__ import annotations

import time

from ..trace import Trace
from ._common import path_cost, reconstruct_path


def solve(problem) -> Trace:
    t0 = time.perf_counter()
    trace = Trace(algorithm="DFS")

    start, goal = problem.start, problem.goal
    stack = [start]
    came_from = {start: start}

    while stack:
        current = stack.pop()
        trace.visited.append(current)
        if current == goal:
            break
        for nxt, _cost in problem.successors(current):
            if nxt not in came_from:
                came_from[nxt] = current
                stack.append(nxt)

    trace.path = reconstruct_path(came_from, start, goal)
    trace.path_cost = path_cost(problem, trace.path)
    trace.time_ms = (time.perf_counter() - t0) * 1000
    return trace.finalize()
