"""Helpers shared by the search algorithms."""

from __future__ import annotations

from typing import Dict, List

Node = object  # a grid cell (row, col) or a road-network node id


def reconstruct_path(came_from: Dict, start, goal) -> List:
    """Walk the parent pointers back from goal to start, then reverse.

    Returns [] if the goal was never reached.
    """
    if goal not in came_from and goal != start:
        return []
    path = [goal]
    node = goal
    while node != start:
        node = came_from[node]
        path.append(node)
    path.reverse()
    return path


def reconstruct_bidirectional(came_from_fwd: Dict, came_from_bwd: Dict,
                              start, goal, meet) -> List:
    """Join the two halves of a bidirectional search at the meeting node.

    `came_from_fwd[n]` points one step back toward `start`.
    `came_from_bwd[n]` points one step back toward `goal`.
    So we walk `meet -> start`, reverse it, then walk `meet -> goal`.
    """
    if meet is None:
        return []
    # start ... meet
    front = [meet]
    node = meet
    while node != start:
        node = came_from_fwd[node]
        front.append(node)
    front.reverse()
    # meet ... goal
    node = meet
    while node != goal:
        node = came_from_bwd[node]
        front.append(node)
    return front


def path_cost(problem, path: List) -> float:
    """Sum the real edge costs along `path` by consulting the problem.

    Works for any problem (grid or road network) because it only uses
    `successors`. Used to report a fair cost even for algorithms like BFS/DFS
    that did not optimise cost themselves.
    """
    total = 0.0
    for a, b in zip(path, path[1:]):
        for nxt, cost in problem.successors(a):
            if nxt == b:
                total += cost
                break
    return total
