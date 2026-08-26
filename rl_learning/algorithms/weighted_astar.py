"""Weighted A*.

The idea: sit on the dial between A* and Greedy. A* uses `f = g + h`; Greedy
uses `f = h`. Weighted A* uses `f = g + W * h` with `W > 1`, deliberately
over-trusting the heuristic. The bigger `W`, the more the search behaves like
Greedy -- faster, expanding fewer nodes -- but the path it returns may be longer
than optimal by at most a factor of `W`. Here W = 1.5.

What to watch: a beam tighter than A*'s but looser than Greedy's, that reaches
the goal having expanded noticeably fewer nodes than A* -- while the reported
path length creeps slightly above the optimal one. It is the explicit
"speed vs quality" knob, and seeing the path cost tick up is the whole point.
"""

from __future__ import annotations

import heapq
import time

from ..trace import Trace
from ._common import reconstruct_path

WEIGHT = 1.5


def solve(problem) -> Trace:
    t0 = time.perf_counter()
    trace = Trace(algorithm="Weighted A*", note=f"W = {WEIGHT}")

    start, goal = problem.start, problem.goal
    counter = 0
    h0 = problem.heuristic(start)
    frontier = [(WEIGHT * h0, h0, counter, start)]
    cost_so_far = {start: 0.0}
    came_from = {start: start}
    settled = set()

    while frontier:
        _, _, _, current = heapq.heappop(frontier)
        if current in settled:
            continue
        settled.add(current)
        trace.visited.append(current)
        if current == goal:
            break
        for nxt, step in problem.successors(current):
            new_cost = cost_so_far[current] + step
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                came_from[nxt] = current
                h = problem.heuristic(nxt)
                counter += 1
                heapq.heappush(frontier, (new_cost + WEIGHT * h, h, counter, nxt))

    trace.path = reconstruct_path(came_from, start, goal)
    trace.path_cost = cost_so_far.get(goal, 0.0)
    trace.time_ms = (time.perf_counter() - t0) * 1000
    return trace.finalize()
