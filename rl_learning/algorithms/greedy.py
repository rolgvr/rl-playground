"""Greedy best-first search.

The idea: A* ranks nodes by `f = g + h`. Greedy throws away the `g` term and
ranks by the heuristic `h` alone -- it always charges toward whatever node
*looks* closest to the goal, ignoring how much it has already spent getting
there. That makes it the fastest of the informed searches to *reach* the goal,
but it is easily fooled: a wall between it and the goal sends it sniffing along
the barrier, and it offers no guarantee the path it finds is short.

What to watch: a very thin, fast beam -- usually even narrower than A* -- that
sometimes commits to a detour A* would have avoided. The contrast with A* is the
lesson: dropping `g` buys speed at the cost of optimality.
"""

from __future__ import annotations

import heapq
import time

from ..trace import Trace
from ._common import path_cost, reconstruct_path


def solve(problem) -> Trace:
    t0 = time.perf_counter()
    trace = Trace(algorithm="Greedy")

    start, goal = problem.start, problem.goal
    counter = 0
    frontier = [(problem.heuristic(start), counter, start)]
    came_from = {start: start}
    visited_set = {start}

    while frontier:
        _, _, current = heapq.heappop(frontier)
        trace.visited.append(current)
        if current == goal:
            break
        for nxt, _step in problem.successors(current):
            if nxt not in visited_set:
                visited_set.add(nxt)
                came_from[nxt] = current
                counter += 1
                heapq.heappush(frontier, (problem.heuristic(nxt), counter, nxt))

    trace.path = reconstruct_path(came_from, start, goal)
    trace.path_cost = path_cost(problem, trace.path)
    trace.time_ms = (time.perf_counter() - t0) * 1000
    return trace.finalize()
