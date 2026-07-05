from __future__ import annotations

from collections import deque
from typing import Iterable

# Cartesian coordinate convention:
# Coord = (x, y), origin (0, 0) is the bottom-left cell.
# x increases to the right; y increases upward.
Coord = tuple[int, int]

ACTIONS: dict[str, Coord] = {
    "UP": (0, 1),
    "DOWN": (0, -1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}


def in_bounds(pos: Coord, size: int) -> bool:
    x, y = pos
    return 0 <= x < size and 0 <= y < size


def is_obstacle(obstacle_map: list[list[int]], pos: Coord) -> bool:
    """obstacle_map is indexed as obstacle_map[y][x]."""
    x, y = pos
    return obstacle_map[y][x] == 1


def neighbors(pos: Coord, size: int) -> Iterable[tuple[str, Coord]]:
    x, y = pos
    for action, (dx, dy) in ACTIONS.items():
        nxt = (x + dx, y + dy)
        if in_bounds(nxt, size):
            yield action, nxt


def shortest_path(obstacle_map: list[list[int]], start: Coord, goal: Coord) -> list[Coord] | None:
    """Return one shortest path as Cartesian coordinates, including start and goal."""
    size = len(obstacle_map)
    q: deque[Coord] = deque([start])
    parent: dict[Coord, Coord | None] = {start: None}

    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for _, nxt in neighbors(cur, size):
            if is_obstacle(obstacle_map, nxt):
                continue
            if nxt not in parent:
                parent[nxt] = cur
                q.append(nxt)

    if goal not in parent:
        return None

    path: list[Coord] = []
    cur: Coord | None = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def shortest_action(obstacle_map: list[list[int]], start: Coord, goal: Coord) -> str | None:
    path = shortest_path(obstacle_map, start, goal)
    if path is None or len(path) < 2:
        return None

    nxt = path[1]
    dx = nxt[0] - start[0]
    dy = nxt[1] - start[1]

    for action, delta in ACTIONS.items():
        if delta == (dx, dy):
            return action
    return None


def count_shortest_paths(obstacle_map: list[list[int]], start: Coord, goal: Coord) -> int:
    """Count shortest paths under Cartesian coordinates."""
    size = len(obstacle_map)
    dist: dict[Coord, int] = {start: 0}
    ways: dict[Coord, int] = {start: 1}
    q: deque[Coord] = deque([start])

    while q:
        cur = q.popleft()
        for _, nxt in neighbors(cur, size):
            if is_obstacle(obstacle_map, nxt):
                continue

            nd = dist[cur] + 1
            if nxt not in dist:
                dist[nxt] = nd
                ways[nxt] = ways[cur]
                q.append(nxt)
            elif nd == dist[nxt]:
                ways[nxt] += ways[cur]

    return ways.get(goal, 0)
