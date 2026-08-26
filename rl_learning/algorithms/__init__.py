"""Stage 1 algorithms: classic pathfinding searches.

Each function has the identical signature

    solve(problem) -> Trace

where `problem` exposes `.start`, `.goal`, `.successors(node)` and
`.heuristic(node)`. Both a `GridWorld` and a road network satisfy that
interface, so every algorithm here runs unchanged on a grid or on real streets.

`ALGORITHMS` is the registry the web server uses to look an algorithm up by
name. When we add Q-learning in Stage 2, it joins this same registry and
immediately becomes raceable against A*.

The algorithms span the major families:
  * uninformed search ............ BFS, DFS
  * uniform-cost (optimal) ....... Dijkstra
  * informed + optimal ........... A*
  * informed + fast (suboptimal) . Greedy, Weighted A*
  * meet-in-the-middle ........... Bi-BFS, Bi-Dijkstra, Bi-A*
"""

from .bfs import solve as bfs
from .dfs import solve as dfs
from .dijkstra import solve as dijkstra
from .astar import solve as astar
from .greedy import solve as greedy
from .weighted_astar import solve as weighted_astar
from .bidirectional_bfs import solve as bi_bfs
from .bidirectional_dijkstra import solve as bi_dijkstra
from .bidirectional_astar import solve as bi_astar

# id -> (callable, human label, "family" tag, short description shown in the UI)
ALGORITHMS = {
    "bfs": (bfs, "BFS", "uninformed",
            "Breadth-first search. Explores in rings; finds the fewest-steps path, but ignores cost and expands a lot."),
    "dfs": (dfs, "DFS", "uninformed",
            "Depth-first search. Dives deep down one corridor first. Fast to find *a* path, but rarely the shortest."),
    "dijkstra": (dijkstra, "Dijkstra", "optimal",
            "Uniform-cost search. Always shortest by total cost, respects heavy terrain, but expands in every direction."),
    "astar": (astar, "A*", "optimal",
            "Dijkstra plus a heuristic compass toward the goal. Same optimal path, far fewer nodes expanded."),
    "greedy": (greedy, "Greedy", "fast",
            "Greedy best-first. Charges at whatever looks closest to the goal. Very fast, easily fooled, not optimal."),
    "weighted_astar": (weighted_astar, "Weighted A*", "fast",
            "A* with the heuristic over-weighted (W=1.5). Faster than A*; path can be up to W times longer than optimal."),
    "bi_bfs": (bi_bfs, "Bi-BFS", "bidirectional",
            "Two BFS waves, from start and goal, meeting in the middle. Optimal (unweighted), expands less than plain BFS."),
    "bi_dijkstra": (bi_dijkstra, "Bi-Dijkstra", "bidirectional",
            "Two cost-aware searches meeting in the middle. Optimal cost like Dijkstra, usually fewer nodes expanded."),
    "bi_astar": (bi_astar, "Bi-A*", "bidirectional",
            "Two heuristic-guided beams converging from both ends. The fastest here on big maps; near-optimal."),
}

__all__ = ["ALGORITHMS"]
