from __future__ import annotations

import json
from typing import Any

from src.env.astar import shortest_action


def _grid_index_from_coord(size: int, coord: list[int]) -> tuple[int, int] | None:
    """Convert Cartesian [x, y] into belief_grid index [row, col].

    belief_grid is displayed top-to-bottom:
    row 0 corresponds to y=size-1; last row corresponds to y=0.
    """
    x, y = int(coord[0]), int(coord[1])
    if not (0 <= x < size and 0 <= y < size):
        return None
    return size - 1 - y, x


class MockOracleModel:
    """A deterministic oracle-like agent for smoke testing the environment pipeline."""

    def generate(
        self,
        prompt: str,
        *,
        env: Any,
        belief_grid: list[list[str]],
        last_feedback: dict,
        current_pos: list[int],
        goal: list[int],
    ) -> str:
        action = shortest_action(env.obstacle_map, tuple(current_pos), tuple(goal))
        if action is None:
            action = "UP"

        new_belief = [row[:] for row in belief_grid]
        size = len(new_belief)

        for cell in last_feedback.get("blocked", []):
            idx = _grid_index_from_coord(size, cell)
            if idx is not None:
                r, c = idx
                new_belief[r][c] = "O"

        for cell in last_feedback.get("free", []):
            idx = _grid_index_from_coord(size, cell)
            if idx is not None:
                r, c = idx
                new_belief[r][c] = "F"

        for cell in [current_pos, goal]:
            idx = _grid_index_from_coord(size, cell)
            if idx is not None:
                r, c = idx
                new_belief[r][c] = "F"

        return json.dumps(
            {
                "thought": "Using the oracle shortest path under Cartesian coordinates.",
                "nl_obstacles": "Updated adjacent exact feedback into the belief grid.",
                "belief_grid": new_belief,
                "action": action,
            },
            ensure_ascii=False,
        )
