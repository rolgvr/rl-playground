"""Dijkstra's algorithm (uniform-cost search).

The idea: always expand the unvisited node whose *total cost from the start* is
smallest. A priority queue keeps the cheapest-so-far node at the front. Because
we finalize nodes in order of increasing cost, the first time we settle the goal
we have its true cheapest cost -- guaranteed optimal, even with heavy terrain.

What to watch: like BFS it spreads in all directions, but the frontier bulges
*around* expensive cells instead of through them. Paint a stripe of "mud" and
you will see Dijkstra bend its path to avoid it while BFS marched straight in.

Relationship to the others: Dijkstra is BFS that understands cost. A* is
Dijkstra that also understands *direction*.
"""

from __future__ import annotations

import heapq
import time

from ..trace import Trace
from ._common import reconstruct_path


def solve(problem) -> Trace:
    t0 = time.perf_counter()
    trace = Trace(algorithm="Dijkstra")

    start, goal = problem.start, problem.goal
    counter = 0
    frontier = [(0.0, counter, start)]
    cost_so_far = {start: 0.0}
    came_from = {start: start}
    settled = set()

    while frontier:
        cost, _, current = heapq.heappop(frontier)
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
                counter += 1
                heapq.heappush(frontier, (new_cost, counter, nxt))

    trace.path = reconstruct_path(came_from, start, goal)
    trace.path_cost = cost_so_far.get(goal, 0.0)
    trace.time_ms = (time.perf_counter() - t0) * 1000
    return trace.finalize()
