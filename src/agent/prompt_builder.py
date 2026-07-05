from __future__ import annotations

import json
from typing import Any

# Cartesian coordinates:
# [x, y], origin at bottom-left; x rightward, y upward.
ACTIONS: dict[str, tuple[int, int]] = {
    "UP": (0, 1),
    "DOWN": (0, -1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}


def belief_to_text(belief_grid: list[list[str]]) -> str:
    """belief_grid rows are displayed top-to-bottom: y=size-1 down to y=0."""
    size = len(belief_grid)
    lines = []
    for row_index, row in enumerate(belief_grid):
        y = size - 1 - row_index
        lines.append(f"y={y}: " + " ".join(row))
    lines.append("x:   " + " ".join(str(x) for x in range(size)))
    return "\n".join(lines)


def _coord_to_action(current_pos: list[int], target: list[int]) -> str | None:
    dx = int(target[0]) - int(current_pos[0])
    dy = int(target[1]) - int(current_pos[1])

    for action, delta in ACTIONS.items():
        if delta == (dx, dy):
            return action
    return None


def legal_actions_from_feedback(current_pos: list[int], last_feedback: dict[str, Any]) -> list[str]:
    """Strategy A exposes exact adjacent free cells. Those are the only safe legal moves."""
    legal: list[str] = []

    for target in last_feedback.get("free", []):
        action = _coord_to_action(current_pos, target)
        if action is not None:
            legal.append(action)

    return [a for a in ["UP", "DOWN", "LEFT", "RIGHT"] if a in set(legal)]


def required_updates_from_feedback(
    *,
    grid_size: int,
    current_pos: list[int],
    goal: list[int],
    last_feedback: dict[str, Any],
) -> dict[str, Any]:
    """Facts that the model must incorporate into its next belief_grid.

    Coordinates are Cartesian [x, y].
    """
    obstacle_cells: list[list[int]] = []
    free_cells: list[list[int]] = []

    def in_bounds(cell: list[int]) -> bool:
        x, y = int(cell[0]), int(cell[1])
        return 0 <= x < grid_size and 0 <= y < grid_size

    for cell in last_feedback.get("blocked", []):
        if in_bounds(cell):
            obstacle_cells.append([int(cell[0]), int(cell[1])])

    for cell in last_feedback.get("free", []):
        if in_bounds(cell):
            free_cells.append([int(cell[0]), int(cell[1])])

    for cell in [current_pos, goal]:
        norm = [int(cell[0]), int(cell[1])]
        if in_bounds(norm) and norm not in free_cells:
            free_cells.append(norm)

    return {
        "coordinate_system": "cartesian_bottom_left",
        "set_O": obstacle_cells,
        "set_F": free_cells,
        "walls_not_grid_cells": last_feedback.get("wall", []),
    }


def _compact_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact_history = []

    for item in history[-8:]:
        compact_history.append(
            {
                "step": item["step_id"],
                "action": item["parsed_action"],
                "valid_move": item.get("valid_move"),
                "invalid_reason": item.get("invalid_reason"),
                "pos_before": item.get("agent_pos_before"),
                "pos_after": item["agent_pos_after"],
                "feedback": item["env_feedback"],
            }
        )

    return compact_history


def build_prompt(
    *,
    grid_size: int,
    start: list[int],
    goal: list[int],
    current_pos: list[int],
    last_feedback: dict[str, Any],
    belief_grid: list[list[str]],
    history: list[dict[str, Any]],
) -> str:
    legal_actions = legal_actions_from_feedback(current_pos, last_feedback)
    required_updates = required_updates_from_feedback(
        grid_size=grid_size,
        current_pos=current_pos,
        goal=goal,
        last_feedback=last_feedback,
    )
    compact_history = _compact_history(history)

    return f"""<task>
You are an agent in a {grid_size}x{grid_size} grid world. Reach the goal.
You do not know the full map. You only know feedback returned after each step.

Coordinate system:
- Coordinates are Cartesian [x, y].
- The origin [0, 0] is the bottom-left cell.
- x increases to the right.
- y increases upward.
- Therefore UP means y+1, DOWN means y-1, LEFT means x-1, RIGHT means x+1.

If you hit a wall or obstacle, you stay in place.
</task>

<start>{json.dumps(start)}</start>
<goal>{json.dumps(goal)}</goal>
<current_pos>{json.dumps(current_pos)}</current_pos>

<last_feedback>
{json.dumps(last_feedback, ensure_ascii=False)}
</last_feedback>

<required_belief_updates>
Apply these facts exactly in your output belief_grid:
{json.dumps(required_updates, ensure_ascii=False)}
Rules:
- Every coordinate in set_O must be marked "O".
- Every coordinate in set_F must be marked "F".
- Wall coordinates are outside the grid and must not be placed inside belief_grid.
- Preserve previously known O/F cells unless contradicted by the latest feedback.
</required_belief_updates>

<available_actions>
Choose action ONLY from this list of safe adjacent free moves:
{json.dumps(legal_actions, ensure_ascii=False)}
Do not choose actions that hit a wall or obstacle.
</available_actions>

<current_belief_grid>
Use O for obstacle, F for free, U for unknown.
Rows are shown top-to-bottom, i.e. y={grid_size - 1} down to y=0.
Columns are x=0 to x={grid_size - 1}.
{belief_to_text(belief_grid)}
</current_belief_grid>

<history>
{json.dumps(compact_history, ensure_ascii=False)}
</history>

<instruction>
First update the belief_grid using required_belief_updates.
Then choose exactly one next action from available_actions that moves toward the goal while avoiding known obstacles and walls.
Return JSON only. No markdown. No extra text.
</instruction>

<output_schema>
{{
  "thought": "brief reasoning, one sentence",
  "nl_obstacles": "brief natural-language obstacle/free-cell summary",
  "belief_grid": [["top row y={grid_size - 1}"], ["..."], ["bottom row y=0"]],
  "action": "one action from available_actions"
}}
</output_schema>
"""


def build_repair_prompt(
    *,
    original_prompt: str,
    bad_response: str,
    bad_action: str,
    legal_actions: list[str],
) -> str:
    return f"""{original_prompt}

<previous_invalid_response>
{bad_response}
</previous_invalid_response>

<repair_instruction>
Your previous action was "{bad_action}", which is not in available_actions.
You must return corrected JSON only.
The action must be exactly one of: {json.dumps(legal_actions, ensure_ascii=False)}.
Remember the coordinate system is Cartesian [x, y] with [0, 0] at the bottom-left.
No markdown. No extra text.
</repair_instruction>
"""
