from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def extract_json_block(text: str, tag: str) -> Any | None:
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    m = re.search(pattern, text, flags=re.S)
    if not m:
        return None
    raw = m.group(1).strip()
    try:
        return json.loads(raw)
    except Exception:
        return None


def extract_between(text: str, left: str, right: str) -> str | None:
    i = text.find(left)
    if i < 0:
        return None
    j = text.find(right, i + len(left))
    if j < 0:
        return None
    return text[i + len(left):j].strip()


def grid_get_cartesian_topdown(grid: list[list[str]], cell: list[int]) -> str | None:
    """belief_grid rows are y-descending. Coordinates are Cartesian [x,y]."""
    try:
        x, y = int(cell[0]), int(cell[1])
        size = len(grid)
        if x < 0 or y < 0 or x >= size or y >= size:
            return None
        row = size - 1 - y
        col = x
        return grid[row][col]
    except Exception:
        return None


def check_step(row: dict[str, Any]) -> dict[str, Any]:
    grid = row.get("parsed_belief_grid")
    prompt = row.get("prompt_text", "")

    required = extract_json_block(prompt, "required_belief_updates") or {}
    start = extract_between(prompt, "<start>", "</start>")
    goal = extract_between(prompt, "<goal>", "</goal>")

    start_cell = json.loads(start) if start else None
    goal_cell = json.loads(goal) if goal else None

    violations = []

    if not isinstance(grid, list):
        violations.append("missing_or_invalid_belief_grid")
        return {
            "belief_violation_count": len(violations),
            "belief_violations": violations,
        }

    for cell in required.get("set_O", []):
        val = grid_get_cartesian_topdown(grid, cell)
        if val != "O":
            violations.append(f"required_O_not_marked:{cell}:got_{val}")

    for cell in required.get("set_F", []):
        val = grid_get_cartesian_topdown(grid, cell)
        if val != "F":
            violations.append(f"required_F_not_marked:{cell}:got_{val}")

    if start_cell is not None and grid_get_cartesian_topdown(grid, start_cell) == "O":
        violations.append(f"start_marked_obstacle:{start_cell}")

    if goal_cell is not None and grid_get_cartesian_topdown(grid, goal_cell) == "O":
        violations.append(f"goal_marked_obstacle:{goal_cell}")

    return {
        "belief_violation_count": len(violations),
        "belief_violations": violations,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run dir, e.g. outputs/logs/strategy_A_qwen_smoke")
    args = ap.parse_args()

    run_dir = Path(args.run)
    steps_path = run_dir / "steps.jsonl"
    summary_path = run_dir / "summary.json"

    if not steps_path.exists():
        raise FileNotFoundError(steps_path)

    rows = []
    with steps_path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    summaries = []
    if summary_path.exists():
        with summary_path.open() as f:
            summaries = json.load(f)

    total_steps = len(rows)
    parse_errors = sum(bool(r.get("parse_error")) for r in rows)
    invalid_moves = sum(not bool(r.get("valid_move")) for r in rows)
    illegal_before_repair = sum(bool(r.get("illegal_action_before_repair")) for r in rows)
    repaired = sum(bool(r.get("repaired")) for r in rows)

    checked = []
    for r in rows:
        result = check_step(r)
        checked.append({**r, **result})

    belief_viol_steps = [r for r in checked if r["belief_violation_count"] > 0]
    belief_viol_total = sum(r["belief_violation_count"] for r in checked)

    print("=" * 80)
    print("RUN:", run_dir)
    print("Coordinate system: Cartesian [x, y], origin bottom-left.")
    print("belief_grid rows are top-to-bottom, y descending.")
    print("=" * 80)

    if summaries:
        print("\nEPISODE SUMMARY")
        for s in summaries:
            print(
                f"- {s['episode_id']}: success={s['success']} "
                f"steps={s['steps_used']} shortest={s['shortest_path_len']} "
                f"gap={s['optimality_gap']} final={s['final_pos']}"
            )

    print("\nSTEP METRICS")
    print("total_steps:", total_steps)
    print("parse_errors:", parse_errors)
    print("invalid_moves:", invalid_moves)
    print("illegal_action_before_repair:", illegal_before_repair)
    print("repair_used:", repaired)
    print("belief_violation_steps:", len(belief_viol_steps))
    print("belief_violation_total:", belief_viol_total)

    print("\nBELIEF VIOLATION DETAILS")
    for r in belief_viol_steps:
        print("-" * 80)
        print(
            f"episode={r['episode_id']} step={r['step_id']} "
            f"before={r['agent_pos_before']} action={r['parsed_action']} after={r['agent_pos_after']}"
        )
        for v in r["belief_violations"]:
            print("  ", v)

    out_path = run_dir / "belief_check.json"
    out = {
        "coordinate_system": "cartesian_bottom_left",
        "belief_grid_row_order": "top_to_bottom_y_descending",
        "total_steps": total_steps,
        "parse_errors": parse_errors,
        "invalid_moves": invalid_moves,
        "illegal_action_before_repair": illegal_before_repair,
        "repair_used": repaired,
        "belief_violation_steps": len(belief_viol_steps),
        "belief_violation_total": belief_viol_total,
        "checked_steps": [
            {
                "episode_id": r["episode_id"],
                "step_id": r["step_id"],
                "belief_violation_count": r["belief_violation_count"],
                "belief_violations": r["belief_violations"],
            }
            for r in checked
        ],
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
