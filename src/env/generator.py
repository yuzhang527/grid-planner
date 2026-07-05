from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .astar import count_shortest_paths, shortest_path


@dataclass(frozen=True)
class GeneratedMap:
    # obstacle_map is indexed as obstacle_map[y][x].
    # Coordinates exposed to the agent are Cartesian [x, y].
    obstacle_map: list[list[int]]
    start: tuple[int, int]
    goal: tuple[int, int]
    shortest_path_len: int
    seed: int


def generate_map(
    grid_size: int,
    num_obstacles: int,
    seed: int,
    require_unique_shortest_path: bool = False,
    max_tries: int = 10_000,
) -> GeneratedMap:
    """Sample a reachable Cartesian grid map with fixed obstacle count K."""
    rng = random.Random(seed)

    start = (0, 0)
    goal = (grid_size - 1, grid_size - 1)

    candidate_cells = [
        (x, y)
        for y in range(grid_size)
        for x in range(grid_size)
        if (x, y) not in {start, goal}
    ]

    for _ in range(max_tries):
        obstacles = set(rng.sample(candidate_cells, num_obstacles))
        obstacle_map = np.zeros((grid_size, grid_size), dtype=int)

        for x, y in obstacles:
            obstacle_map[y, x] = 1

        map_list = obstacle_map.tolist()
        path = shortest_path(map_list, start, goal)
        if path is None:
            continue

        if require_unique_shortest_path and count_shortest_paths(map_list, start, goal) != 1:
            continue

        return GeneratedMap(
            obstacle_map=map_list,
            start=start,
            goal=goal,
            shortest_path_len=len(path) - 1,
            seed=seed,
        )

    raise RuntimeError(
        f"Could not sample a reachable map after {max_tries} tries. "
        f"Try smaller num_obstacles or disable require_unique_shortest_path."
    )
