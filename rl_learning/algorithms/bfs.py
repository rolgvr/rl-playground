"""Breadth-first search.

The idea: explore the graph in expanding rings around the start. Because every
node one step away is visited before any node two steps away, the first time BFS
reaches the goal it has found a path with the *fewest steps*.

What to watch in the animation: BFS floods outward evenly in all directions. It
has no idea where the goal is, so on an open grid it expands a huge diamond of
cells -- great for spotting how "uninformed" search wastes work compared to A*.

Caveat: BFS counts *steps*, not *cost*. It walks straight through heavy terrain
because to BFS every move is one step. That is the difference Dijkstra fixes.
"""

from __future__ import annotations

import time
from collections import deque

from ..trace import Trace
from ._common import path_cost, reconstruct_path


def solve(problem) -> Trace:
    t0 = time.perf_counter()
    trace = Trace(algorithm="BFS")

    start, goal = problem.start, problem.goal
    frontier = deque([start])
    came_from = {start: start}

    while frontier:
        current = frontier.popleft()
        trace.visited.append(current)
        if current == goal:
            break
        for nxt, _cost in problem.successors(current):
            if nxt not in came_from:
                came_from[nxt] = current
                frontier.append(nxt)

    trace.path = reconstruct_path(came_from, start, goal)
    trace.path_cost = path_cost(problem, trace.path)
    trace.time_ms = (time.perf_counter() - t0) * 1000
    return trace.finalize()
