"""Bidirectional Dijkstra.

The idea: the same "meet in the middle" trick as bidirectional BFS, but
cost-aware. Two Dijkstra searches advance from the start and from the goal; each
settles nodes in order of increasing cost from its own end. We keep track of the
best start->goal cost discovered through any node both sides have reached (call
it `mu`) and stop once neither frontier can possibly improve on it -- precisely
when the cheapest node still waiting on each side sum to at least `mu`.

That stopping rule is what keeps it *optimal*: it is not enough for the waves to
merely touch (the first contact may not be the cheapest crossing), so we keep
going just long enough to be sure no cheaper crossing exists.

What to watch: two cost-driven blobs bulging around expensive terrain toward each
other -- the optimal cost of Dijkstra, usually with fewer nodes expanded.
"""

from __future__ import annotations

import heapq
import time

from ..trace import Trace
from ._common import reconstruct_bidirectional


def solve(problem) -> Trace:
    t0 = time.perf_counter()
    trace = Trace(algorithm="Bi-Dijkstra")

    start, goal = problem.start, problem.goal
    if start == goal:
        trace.path = [start]
        return trace.finalize()

    dist_f = {start: 0.0}; came_f = {start: start}; settled_f = set()
    dist_b = {goal: 0.0};  came_b = {goal: goal};   settled_b = set()
    heap_f = [(0.0, start)]
    heap_b = [(0.0, goal)]
    mu = float("inf")
    meet = None

    def relax(u, heap, dist, came, dist_other):
        nonlocal mu, meet
        for nxt, step in problem.successors(u):
            nd = dist[u] + step
            if nxt not in dist or nd < dist[nxt]:
                dist[nxt] = nd
                came[nxt] = u
                heapq.heappush(heap, (nd, nxt))
                if nxt in dist_other:               # crossing found
                    total = nd + dist_other[nxt]
                    if total < mu:
                        mu, meet = total, nxt

    while heap_f and heap_b:
        # No remaining crossing can beat mu once the two cheapest waiting nodes
        # already sum to mu -> safe to stop.
        if heap_f[0][0] + heap_b[0][0] >= mu:
            break
        # Advance whichever side currently has the cheaper frontier.
        if heap_f[0][0] <= heap_b[0][0]:
            _, u = heapq.heappop(heap_f)
            if u in settled_f:
                continue
            settled_f.add(u); trace.visited.append(u)
            relax(u, heap_f, dist_f, came_f, dist_b)
        else:
            _, u = heapq.heappop(heap_b)
            if u in settled_b:
                continue
            settled_b.add(u); trace.visited.append(u)
            relax(u, heap_b, dist_b, came_b, dist_f)

    trace.path = reconstruct_bidirectional(came_f, came_b, start, goal, meet)
    trace.path_cost = mu if meet is not None else 0.0
    trace.time_ms = (time.perf_counter() - t0) * 1000
    return trace.finalize()
