from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

VALID_ACTIONS = {"UP", "DOWN", "LEFT", "RIGHT"}
VALID_BELIEF = {"O", "F", "U"}


@dataclass
class ParsedResponse:
    thought: str
    nl_obstacles: str
    belief_grid: list[list[str]]
    action: str
    parse_error: bool = False
    error_message: str | None = None


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def _extract_first_json_object(text: str) -> str:
    text = _strip_code_fences(text)
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found.")
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    raise ValueError("JSON object braces are not balanced.")


def _normalize_belief_grid(value: Any, grid_size: int) -> list[list[str]]:
    if isinstance(value, str):
        rows = [row.strip().split() for row in value.strip().splitlines() if row.strip()]
    elif isinstance(value, list):
        rows = value
    else:
        raise ValueError("belief_grid must be a list of lists or a multiline string.")

    if len(rows) != grid_size:
        raise ValueError(f"belief_grid row count should be {grid_size}, got {len(rows)}.")

    out: list[list[str]] = []
    for row in rows:
        if isinstance(row, str):
            cells = row.strip().split()
        else:
            cells = [str(x).strip().upper() for x in row]
        if len(cells) != grid_size:
            raise ValueError(f"Each belief_grid row should have {grid_size} cells, got {len(cells)}.")
        for cell in cells:
            if cell not in VALID_BELIEF:
                raise ValueError(f"Invalid belief cell {cell!r}; allowed: O/F/U.")
        out.append(cells)
    return out


def parse_response(text: str, grid_size: int) -> ParsedResponse:
    try:
        obj_text = _extract_first_json_object(text)
        data = json.loads(obj_text)

        thought = str(data.get("thought", ""))
        nl_obstacles = str(data.get("nl_obstacles", ""))
        belief_grid = _normalize_belief_grid(data.get("belief_grid"), grid_size)
        action = str(data.get("action", "")).upper().strip()
        if action not in VALID_ACTIONS:
            raise ValueError(f"Invalid action {action!r}; allowed: {sorted(VALID_ACTIONS)}.")

        return ParsedResponse(
            thought=thought,
            nl_obstacles=nl_obstacles,
            belief_grid=belief_grid,
            action=action,
            parse_error=False,
        )
    except Exception as exc:
        return ParsedResponse(
            thought="",
            nl_obstacles="",
            belief_grid=[["U" for _ in range(grid_size)] for _ in range(grid_size)],
            action="UP",
            parse_error=True,
            error_message=str(exc),
        )
