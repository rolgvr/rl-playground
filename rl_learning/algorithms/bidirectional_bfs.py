"""Bidirectional BFS.

The idea: instead of one search ballooning out from the start until it happens
to hit the goal, run *two* BFS waves at once -- one growing from the start, one
growing backward from the goal -- and stop the instant they touch. Because a
search wave's size grows roughly like radius^2, two waves of half the radius
cover far less area than one full-radius wave. That is the whole trick: meeting
in the middle is cheaper than going all the way.

What to watch: two blobs creeping toward each other and clicking together
mid-map. On a big open grid it expands noticeably fewer cells than plain BFS for
the same (shortest, unweighted) path.
"""

from __future__ import annotations

import time
from collections import deque

from ..trace import Trace
from ._common import path_cost, reconstruct_bidirectional


def solve(problem) -> Trace:
    t0 = time.perf_counter()
    trace = Trace(algorithm="Bi-BFS")

    start, goal = problem.start, problem.goal
    if start == goal:
        trace.path = [start]
        return trace.finalize()

    frontier_f = deque([start]);  came_f = {start: start}
    frontier_b = deque([goal]);   came_b = {goal: goal}
    meet = None

    # Expand one node from each side per iteration so the two waves grow evenly.
    while frontier_f and frontier_b and meet is None:
        for frontier, came, other in (
            (frontier_f, came_f, came_b),
            (frontier_b, came_b, came_f),
        ):
            if not frontier:
                break
            current = frontier.popleft()
            trace.visited.append(current)
            for nxt, _step in problem.successors(current):
                if nxt not in came:
                    came[nxt] = current
                    if nxt in other:        # the two waves just touched
                        meet = nxt
                        break
                    frontier.append(nxt)
            if meet is not None:
                break

    trace.path = reconstruct_bidirectional(came_f, came_b, start, goal, meet)
    trace.path_cost = path_cost(problem, trace.path)
    trace.time_ms = (time.perf_counter() - t0) * 1000
    return trace.finalize()
