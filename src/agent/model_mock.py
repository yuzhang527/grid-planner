from __future__ import annotations

import json
from typing import Any

from src.env.astar import shortest_action


def update_belief_from_feedback(belief_grid: list[list[str]], feedback: dict[str, Any], current_pos: list[int], goal: list[int]) -> list[list[str]]:
    new_belief = [row[:] for row in belief_grid]
    r, c = current_pos
    new_belief[r][c] = "F"
    new_belief[goal[0]][goal[1]] = "F"
    for rr, cc in feedback.get("blocked", []):
        if 0 <= rr < len(new_belief) and 0 <= cc < len(new_belief):
            new_belief[rr][cc] = "O"
    for rr, cc in feedback.get("free", []):
        if 0 <= rr < len(new_belief) and 0 <= cc < len(new_belief):
            new_belief[rr][cc] = "F"
    return new_belief


class MockOracleModel:
    """Debug backend.

    It uses the true map to choose the next shortest action, so it is NOT a research baseline.
    Its purpose is only to prove env -> prompt -> parser -> step -> log can run successfully.
    """

    def generate(self, prompt: str, *, env, belief_grid, last_feedback, current_pos, goal, **_: Any) -> str:
        new_belief = update_belief_from_feedback(belief_grid, last_feedback, current_pos, goal)
        action = shortest_action(env.obstacle_map, tuple(current_pos), tuple(goal)) or "UP"
        obj = {
            "thought": "I updated adjacent cells from feedback and will take a shortest safe step.",
            "nl_obstacles": f"Known blocked: {last_feedback.get('blocked', [])}; known free: {last_feedback.get('free', [])}.",
            "belief_grid": new_belief,
            "action": action,
        }
        return json.dumps(obj, ensure_ascii=False)
