"""RoadNetwork -- a real street graph that the same algorithms can search.

This is the map-view counterpart to `GridWorld`. It exposes the *identical*
search interface -- `.start`, `.goal`, `.successors(node)`, `.estimate(a, b)`,
`.heuristic(node)` -- so every algorithm in `rl_learning.algorithms` runs on it
with zero changes. The only differences from the grid are cosmetic to the
algorithms: nodes are OpenStreetMap node ids (ints) instead of (row, col)
tuples, and edge costs are real distances in metres instead of grid steps.

Where the data comes from
--------------------------
We query the public **Overpass API** for the drivable roads inside a bounding
box. Overpass returns OSM *ways* (roads) and the *nodes* (points) that make them
up. We turn consecutive nodes along each way into graph edges, weighting each by
the great-circle (haversine) distance between its endpoints.

Why haversine is a valid A* heuristic: the straight-line distance between two
points can never exceed the distance you'd actually drive along the roads, so it
never overestimates -- exactly the "admissible" property A* needs to stay
optimal.

No API key is required and nothing is billed; Overpass is a free OSM service.
Be a good citizen: keep bounding boxes small and don't hammer it.
"""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from typing import Dict, List, Tuple

# Several public Overpass mirrors. The main instance is the most reliable but
# can be slow (10-30s) or briefly overloaded (503/504), so we fall back to the
# others in order until one answers.
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

# Road classes we treat as drivable. Footways/cycleways are excluded so the
# network resembles what a car would use.
DRIVABLE = (
    "motorway|trunk|primary|secondary|tertiary|unclassified|"
    "residential|living_street|service|road|motorway_link|trunk_link|"
    "primary_link|secondary_link|tertiary_link"
)

LatLon = Tuple[float, float]


def haversine(a: LatLon, b: LatLon) -> float:
    """Great-circle distance between two (lat, lon) points, in metres."""
    R = 6371000.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


class RoadNetwork:
    def __init__(self, nodes: Dict[int, LatLon], adj: Dict[int, List[Tuple[int, float]]]):
        self.nodes = nodes                 # osm id -> (lat, lon)
        self.adj = adj                     # osm id -> [(neighbour id, distance_m), ...]
        self.start: int = None
        self.goal: int = None

    # --- the search interface (mirrors GridWorld) --------------------------

    def successors(self, node: int) -> List[Tuple[int, float]]:
        return self.adj.get(node, [])

    def estimate(self, a: int, b: int) -> float:
        return haversine(self.nodes[a], self.nodes[b])

    def heuristic(self, node: int) -> float:
        return self.estimate(node, self.goal)

    # --- helpers -----------------------------------------------------------

    def nearest(self, lat: float, lon: float) -> int:
        """The graph node closest to a clicked (lat, lon)."""
        target = (lat, lon)
        best, best_d = None, float("inf")
        for nid, coord in self.nodes.items():
            d = haversine(coord, target)
            if d < best_d:
                best, best_d = nid, d
        return best

    def edge_list(self) -> List[Tuple[int, int]]:
        """Unique undirected edges (a < b) for drawing the network once."""
        seen = set()
        for a, nbrs in self.adj.items():
            for b, _ in nbrs:
                key = (a, b) if a < b else (b, a)
                seen.add(key)
        return list(seen)

    # --- construction ------------------------------------------------------

    @classmethod
    def from_overpass(cls, south: float, west: float, north: float, east: float,
                      timeout: int = 60) -> "RoadNetwork":
        query = (
            "[out:json][timeout:55];"
            f'(way["highway"~"^({DRIVABLE})$"]({south},{west},{north},{east}););'
            "(._;>;);"
            "out body;"
        )
        body = ("data=" + urllib.parse.quote(query)).encode()

        last_err = None
        for url in OVERPASS_MIRRORS:
            try:
                data = urllib.request.urlopen(
                    urllib.request.Request(
                        url, data=body,
                        headers={"User-Agent": "rl-learning-playground/0.1 (educational)"},
                    ),
                    timeout=timeout,
                ).read()
                return cls._build(json.loads(data)["elements"])
            except Exception as exc:  # try the next mirror
                last_err = exc
        raise RuntimeError(f"all Overpass mirrors failed (last: {last_err})")

    @classmethod
    def _build(cls, elements: list) -> "RoadNetwork":
        coords: Dict[int, LatLon] = {}
        ways = []
        for el in elements:
            if el["type"] == "node":
                coords[el["id"]] = (el["lat"], el["lon"])
            elif el["type"] == "way" and el.get("nodes"):
                ways.append(el["nodes"])

        adj: Dict[int, List[Tuple[int, float]]] = {}
        used: Dict[int, LatLon] = {}

        def link(a: int, b: int):
            d = haversine(coords[a], coords[b])
            adj.setdefault(a, []).append((b, d))
            adj.setdefault(b, []).append((a, d))
            used[a] = coords[a]
            used[b] = coords[b]

        for node_ids in ways:
            for a, b in zip(node_ids, node_ids[1:]):
                if a in coords and b in coords:
                    link(a, b)

        return cls(used, adj)
