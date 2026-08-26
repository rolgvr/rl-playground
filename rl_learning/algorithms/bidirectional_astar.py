"""Bidirectional A*.

The idea: combine the two big speed-ups at once. Like bidirectional Dijkstra it
grows two searches that meet in the middle; like A* each search is steered by a
heuristic -- the forward search aims at the goal, the backward search aims at the
start. Two informed beams converging on each other is, on large maps like road
networks, about the fastest classic shortest-path search there is. It is the
algorithm real routing engines lean on.

Optimality note: getting bidirectional A* *provably* optimal requires a fussy
stopping rule. We use the simple, fast one (stop when the two frontiers' f-values
can no longer beat the best crossing found), which makes it very quick but, like
Weighted A* and Greedy, it can occasionally return a path a hair longer than the
true shortest. If you need the guaranteed optimum, race it against Bi-Dijkstra.

What to watch: two tight beams, each pointed at the other's origin, snapping
together near the midpoint -- usually the fewest nodes expanded of any optimal
search here.
"""

from __future__ import annotations

import heapq
import time

from ..trace import Trace
from ._common import reconstruct_bidirectional


def solve(problem) -> Trace:
    t0 = time.perf_counter()
    trace = Trace(algorithm="Bi-A*")

    start, goal = problem.start, problem.goal
    if start == goal:
        trace.path = [start]
        return trace.finalize()

    # Heuristics: forward toward goal, backward toward start.
    hf = lambda n: problem.estimate(n, goal)
    hb = lambda n: problem.estimate(n, start)

    dist_f = {start: 0.0}; came_f = {start: start}; settled_f = set()
    dist_b = {goal: 0.0};  came_b = {goal: goal};   settled_b = set()
    heap_f = [(hf(start), start)]   # ordered by f = g + h
    heap_b = [(hb(goal), goal)]
    mu = float("inf")
    meet = None

    def relax(u, heap, dist, came, dist_other, h):
        nonlocal mu, meet
        for nxt, step in problem.successors(u):
            nd = dist[u] + step
            if nxt not in dist or nd < dist[nxt]:
                dist[nxt] = nd
                came[nxt] = u
                heapq.heappush(heap, (nd + h(nxt), nxt))
                if nxt in dist_other:
                    total = nd + dist_other[nxt]
                    if total < mu:
                        mu, meet = total, nxt

    while heap_f and heap_b:
        # Fast stopping rule: once the cheapest f-value waiting on each side
        # together reach the best crossing cost, further expansion is unlikely to
        # help. O(1) to check -- important on large road graphs.
        if heap_f[0][0] + heap_b[0][0] >= mu:
            break

        if heap_f[0][0] <= heap_b[0][0]:
            _, u = heapq.heappop(heap_f)
            if u in settled_f:
                continue
            settled_f.add(u); trace.visited.append(u)
            relax(u, heap_f, dist_f, came_f, dist_b, hf)
        else:
            _, u = heapq.heappop(heap_b)
            if u in settled_b:
                continue
            settled_b.add(u); trace.visited.append(u)
            relax(u, heap_b, dist_b, came_b, dist_f, hb)

    trace.path = reconstruct_bidirectional(came_f, came_b, start, goal, meet)
    trace.path_cost = mu if meet is not None else 0.0
    trace.time_ms = (time.perf_counter() - t0) * 1000
    return trace.finalize()
