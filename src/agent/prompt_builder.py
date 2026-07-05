from __future__ import annotations

import json
from typing import Any


def belief_to_text(belief_grid: list[list[str]]) -> str:
    return "\n".join(" ".join(row) for row in belief_grid)


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
    compact_history = []
    for item in history[-8:]:
        compact_history.append(
            {
                "step": item["step_id"],
                "action": item["parsed_action"],
                "pos_after": item["agent_pos_after"],
                "feedback": item["env_feedback"],
            }
        )

    return f"""<task>
You are an agent in a {grid_size}x{grid_size} grid world. Reach the goal.
You do not know the full map. You only know feedback returned after each step.
Coordinates are 0-indexed as [row, col].
Actions: UP, DOWN, LEFT, RIGHT.
If you hit a wall or obstacle, you stay in place.
</task>

<start>{json.dumps(start)}</start>
<goal>{json.dumps(goal)}</goal>
<current_pos>{json.dumps(current_pos)}</current_pos>

<last_feedback>
{json.dumps(last_feedback, ensure_ascii=False)}
</last_feedback>

<current_belief_grid>
Use O for obstacle, F for free, U for unknown.
{belief_to_text(belief_grid)}
</current_belief_grid>

<history>
{json.dumps(compact_history, ensure_ascii=False)}
</history>

<instruction>
Update your world model using last_feedback. Then choose exactly one next action that moves toward the goal while avoiding known obstacles and walls.
Return JSON only. No markdown. No extra text.
</instruction>

<output_schema>
{{
  "thought": "brief reasoning, one sentence",
  "nl_obstacles": "brief natural-language obstacle/free-cell summary",
  "belief_grid": [["F","U",...], ...],
  "action": "UP|DOWN|LEFT|RIGHT"
}}
</output_schema>
"""
