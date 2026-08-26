"""A* search.

The idea: Dijkstra ranks nodes by cost-so-far `g`. A* ranks them by
`f = g + h`, where `h` is a *heuristic* estimate of the remaining distance to
the goal. That extra term is a compass: of two nodes that cost the same to reach,
A* expands the one that is closer to the goal first. The result is the same
optimal path as Dijkstra, found while expanding dramatically fewer nodes.

The heuristic (supplied by the problem) must never *overestimate* the true
remaining cost ("admissible"), or A* can return a wrong path. On a grid it is
Manhattan/octile distance; on a road network it is straight-line distance.

What to watch: instead of a symmetric blob, A* carves a narrow beam straight at
the goal. Race it against Dijkstra on an open grid and the difference in nodes
expanded is the whole point of the lesson.
"""

from __future__ import annotations

import heapq
import time

from ..trace import Trace
from ._common import reconstruct_path


def solve(problem) -> Trace:
    t0 = time.perf_counter()
    trace = Trace(algorithm="A*")

    start, goal = problem.start, problem.goal
    # Heap key is (f, h, counter): primary sort by f = g + h, then break ties by
    # the smaller h (the node estimated closer to the goal). Without that tie-
    # breaker, an open grid with Manhattan distance gives *every* cell the same
    # f, so A* degenerates into expanding everything like Dijkstra. Preferring
    # smaller h focuses the search into a narrow beam toward the goal -- which is
    # exactly the behaviour the lesson is meant to show.
    counter = 0
    h0 = problem.heuristic(start)
    frontier = [(h0, h0, counter, start)]
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
                heapq.heappush(frontier, (new_cost + h, h, counter, nxt))

    trace.path = reconstruct_path(came_from, start, goal)
    trace.path_cost = cost_so_far.get(goal, 0.0)
    trace.time_ms = (time.perf_counter() - t0) * 1000
    return trace.finalize()
