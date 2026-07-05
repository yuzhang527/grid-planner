from __future__ import annotations

from collections import deque
from typing import Iterable

Coord = tuple[int, int]
ACTIONS: dict[str, Coord] = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}


def in_bounds(pos: Coord, size: int) -> bool:
    r, c = pos
    return 0 <= r < size and 0 <= c < size


def neighbors(pos: Coord, size: int) -> Iterable[tuple[str, Coord]]:
    r, c = pos
    for action, (dr, dc) in ACTIONS.items():
        nxt = (r + dr, c + dc)
        if in_bounds(nxt, size):
            yield action, nxt


def shortest_path(obstacle_map, start: Coord, goal: Coord) -> list[Coord] | None:
    """Return one shortest path as list of positions, including start and goal."""
    size = len(obstacle_map)
    q: deque[Coord] = deque([start])
    parent: dict[Coord, Coord | None] = {start: None}

    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for _, nxt in neighbors(cur, size):
            if obstacle_map[nxt[0]][nxt[1]] == 1:
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


def shortest_action(obstacle_map, start: Coord, goal: Coord) -> str | None:
    path = shortest_path(obstacle_map, start, goal)
    if path is None or len(path) < 2:
        return None
    nxt = path[1]
    dr = nxt[0] - start[0]
    dc = nxt[1] - start[1]
    for action, delta in ACTIONS.items():
        if delta == (dr, dc):
            return action
    return None


def count_shortest_paths(obstacle_map, start: Coord, goal: Coord) -> int:
    """Count shortest paths. Used only as an optional map filter."""
    size = len(obstacle_map)
    dist: dict[Coord, int] = {start: 0}
    ways: dict[Coord, int] = {start: 1}
    q: deque[Coord] = deque([start])

    while q:
        cur = q.popleft()
        for _, nxt in neighbors(cur, size):
            if obstacle_map[nxt[0]][nxt[1]] == 1:
                continue
            nd = dist[cur] + 1
            if nxt not in dist:
                dist[nxt] = nd
                ways[nxt] = ways[cur]
                q.append(nxt)
            elif nd == dist[nxt]:
                ways[nxt] += ways[cur]
    return ways.get(goal, 0)
