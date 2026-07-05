from __future__ import annotations

from .astar import ACTIONS, in_bounds, is_obstacle

Coord = tuple[int, int]


def adjacent_exact_feedback(obstacle_map: list[list[int]], pos: Coord) -> dict:
    """Strategy A under Cartesian coordinates.

    Return exact states of four adjacent cells.
    Coordinates are [x, y], with (0, 0) at the bottom-left.
    """
    size = len(obstacle_map)
    blocked: list[list[int]] = []
    free: list[list[int]] = []
    wall: list[list[int]] = []

    x, y = pos
    for _action, (dx, dy) in ACTIONS.items():
        nxt = (x + dx, y + dy)

        if not in_bounds(nxt, size):
            wall.append([nxt[0], nxt[1]])
        elif is_obstacle(obstacle_map, nxt):
            blocked.append([nxt[0], nxt[1]])
        else:
            free.append([nxt[0], nxt[1]])

    return {
        "type": "adjacent_exact",
        "coordinate_system": "cartesian_bottom_left",
        "position": [x, y],
        "blocked": blocked,
        "free": free,
        "wall": wall,
    }
