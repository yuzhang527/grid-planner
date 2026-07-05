from __future__ import annotations

from .astar import ACTIONS, in_bounds

Coord = tuple[int, int]


def adjacent_exact_feedback(obstacle_map: list[list[int]], pos: Coord) -> dict:
    """Strategy A: return exact state of four adjacent cells.

    We explicitly mark out-of-bound cells as wall, so the feedback protocol is stable.
    Coordinates are 0-indexed: [row, col].
    """
    size = len(obstacle_map)
    blocked: list[list[int]] = []
    free: list[list[int]] = []
    wall: list[list[int]] = []

    r, c = pos
    for _action, (dr, dc) in ACTIONS.items():
        nxt = (r + dr, c + dc)
        if not in_bounds(nxt, size):
            wall.append([nxt[0], nxt[1]])
        elif obstacle_map[nxt[0]][nxt[1]] == 1:
            blocked.append([nxt[0], nxt[1]])
        else:
            free.append([nxt[0], nxt[1]])

    return {
        "type": "adjacent_exact",
        "position": [r, c],
        "blocked": blocked,
        "free": free,
        "wall": wall,
    }
