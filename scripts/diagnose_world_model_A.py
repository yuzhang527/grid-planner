from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ACTIONS = {
    "UP": (0, 1),
    "DOWN": (0, -1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_between(text: str, left: str, right: str) -> str | None:
    i = text.find(left)
    if i < 0:
        return None
    j = text.find(right, i + len(left))
    if j < 0:
        return None
    return text[i + len(left):j].strip()


def extract_json_tag(text: str, tag: str):
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    m = re.search(pattern, text, flags=re.S)
    if not m:
        return None
    raw = m.group(1).strip()
    try:
        return json.loads(raw)
    except Exception:
        return None


def coord_to_topdown_index(size: int, coord: list[int] | tuple[int, int]) -> tuple[int, int] | None:
    """Cartesian [x,y] -> top-down display index [row,col].

    Top-down row order:
    row 0 = y=size-1
    row size-1 = y=0
    """
    x, y = int(coord[0]), int(coord[1])
    if not (0 <= x < size and 0 <= y < size):
        return None
    return size - 1 - y, x


def coord_to_bottomup_index(size: int, coord: list[int] | tuple[int, int]) -> tuple[int, int] | None:
    """Cartesian [x,y] -> bottom-up row index [row,col].

    This is the common wrong interpretation:
    row 0 = y=0
    row size-1 = y=size-1
    """
    x, y = int(coord[0]), int(coord[1])
    if not (0 <= x < size and 0 <= y < size):
        return None
    return y, x


def get_agent_cell(agent_grid: list[list[str]], coord: list[int], interpretation: str) -> str | None:
    size = len(agent_grid)
    if interpretation == "topdown_y_desc":
        idx = coord_to_topdown_index(size, coord)
    elif interpretation == "bottomup_y_asc":
        idx = coord_to_bottomup_index(size, coord)
    else:
        raise ValueError(f"Unknown interpretation: {interpretation}")

    if idx is None:
        return None
    r, c = idx

    try:
        return agent_grid[r][c]
    except Exception:
        return None


def get_topdown_cell(grid: list[list[str]], coord: list[int]) -> str | None:
    idx = coord_to_topdown_index(len(grid), coord)
    if idx is None:
        return None
    r, c = idx
    return grid[r][c]


def set_topdown_cell(grid: list[list[str]], coord: list[int] | tuple[int, int], value: str) -> None:
    idx = coord_to_topdown_index(len(grid), coord)
    if idx is None:
        return
    r, c = idx
    grid[r][c] = value


def init_gold(size: int, start: list[int], goal: list[int]) -> list[list[str]]:
    gold = [["U" for _ in range(size)] for _ in range(size)]
    set_topdown_cell(gold, start, "F")
    set_topdown_cell(gold, goal, "F")
    return gold


def apply_strategy_A_feedback(
    gold: list[list[str]],
    *,
    feedback: dict[str, Any],
    current_pos: list[int],
    goal: list[int],
) -> None:
    for cell in feedback.get("blocked", []):
        set_topdown_cell(gold, cell, "O")
    for cell in feedback.get("free", []):
        set_topdown_cell(gold, cell, "F")

    set_topdown_cell(gold, current_pos, "F")
    set_topdown_cell(gold, goal, "F")


def normalize_agent_grid(agent_grid: list[list[str]], interpretation: str) -> list[list[str]]:
    """Return agent grid normalized to top-down y-descending rows."""
    size = len(agent_grid)
    norm = [["U" for _ in range(size)] for _ in range(size)]

    for y in range(size):
        for x in range(size):
            coord = [x, y]
            val = get_agent_cell(agent_grid, coord, interpretation)
            set_topdown_cell(norm, coord, val if val in {"O", "F", "U"} else "?")

    return norm


def compare_belief(agent_grid: list[list[str]], gold_grid: list[list[str]], interpretation: str) -> dict[str, Any]:
    size = len(gold_grid)

    total = 0
    correct = 0

    known_total = 0
    known_correct = 0

    unknown_total = 0
    unknown_preserved = 0
    overclaim = 0

    obstacle_tp = obstacle_fp = obstacle_fn = 0
    free_tp = free_fp = free_fn = 0

    for y in range(size):
        for x in range(size):
            coord = [x, y]
            g = get_topdown_cell(gold_grid, coord)
            a = get_agent_cell(agent_grid, coord, interpretation)

            total += 1
            if a == g:
                correct += 1

            if g in {"O", "F"}:
                known_total += 1
                if a == g:
                    known_correct += 1
            elif g == "U":
                unknown_total += 1
                if a == "U":
                    unknown_preserved += 1
                if a in {"O", "F"}:
                    overclaim += 1

            if a == "O" and g == "O":
                obstacle_tp += 1
            elif a == "O" and g != "O":
                obstacle_fp += 1
            elif a != "O" and g == "O":
                obstacle_fn += 1

            if a == "F" and g == "F":
                free_tp += 1
            elif a == "F" and g != "F":
                free_fp += 1
            elif a != "F" and g == "F":
                free_fn += 1

    def safe_div(num: int, den: int):
        return None if den == 0 else num / den

    def f1(tp: int, fp: int, fn: int):
        p = safe_div(tp, tp + fp)
        r = safe_div(tp, tp + fn)
        if p is None or r is None or p + r == 0:
            return {"precision": p, "recall": r, "f1": None}
        return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r)}

    obstacle = f1(obstacle_tp, obstacle_fp, obstacle_fn)
    free = f1(free_tp, free_fp, free_fn)

    return {
        "all_cell_acc": safe_div(correct, total),
        "known_cell_acc": safe_div(known_correct, known_total),
        "unknown_preservation_rate": safe_div(unknown_preserved, unknown_total),
        "overclaim_rate": safe_div(overclaim, unknown_total),
        "known_cells": known_total,
        "unknown_cells": unknown_total,
        "obstacle_precision": obstacle["precision"],
        "obstacle_recall": obstacle["recall"],
        "obstacle_f1": obstacle["f1"],
        "free_precision": free["precision"],
        "free_recall": free["recall"],
        "free_f1": free["f1"],
    }


def target_after_action(pos: list[int], action: str) -> list[int] | None:
    if action not in ACTIONS:
        return None
    dx, dy = ACTIONS[action]
    return [int(pos[0]) + dx, int(pos[1]) + dy]


def action_consistency(agent_grid: list[list[str]], pos: list[int], action: str, interpretation: str) -> dict[str, Any]:
    target = target_after_action(pos, action)
    if target is None:
        return {
            "action_target": None,
            "action_target_belief": None,
            "action_hits_own_obstacle": None,
            "action_into_unknown": None,
        }

    val = get_agent_cell(agent_grid, target, interpretation)

    return {
        "action_target": target,
        "action_target_belief": val,
        "action_hits_own_obstacle": val == "O",
        "action_into_unknown": val == "U",
    }


def monotonicity_violations(prev_norm: list[list[str]] | None, cur_norm: list[list[str]]) -> int:
    if prev_norm is None:
        return 0

    count = 0
    for r in range(len(cur_norm)):
        for c in range(len(cur_norm[r])):
            prev = prev_norm[r][c]
            cur = cur_norm[r][c]
            if prev in {"O", "F"} and cur != prev:
                count += 1

    return count


def mean_ignore_none(values):
    vals = [v for v in values if v is not None]
    return None if not vals else sum(vals) / len(vals)


def render_topdown_grid(grid: list[list[str]], title: str) -> str:
    size = len(grid)
    lines = [title]
    for row_index, row in enumerate(grid):
        y = size - 1 - row_index
        lines.append(f"y={y}: " + " ".join(row))
    lines.append("x:   " + " ".join(str(x) for x in range(size)))
    return "\n".join(lines)


def make_step_case_md(
    *,
    row: dict[str, Any],
    gold: list[list[str]],
    agent_grid: list[list[str]],
    topdown_metrics: dict[str, Any],
    bottomup_metrics: dict[str, Any],
    topdown_action: dict[str, Any],
    bottomup_action: dict[str, Any],
) -> str:
    agent_as_topdown = normalize_agent_grid(agent_grid, "topdown_y_desc")
    agent_as_bottomup = normalize_agent_grid(agent_grid, "bottomup_y_asc")

    parts = []
    parts.append(f"## Step {row['step_id']}")
    parts.append("")
    parts.append(f"- before: `{row.get('agent_pos_before')}`")
    parts.append(f"- action: `{row.get('parsed_action')}`")
    parts.append(f"- after: `{row.get('agent_pos_after')}`")
    parts.append(f"- valid_move: `{row.get('valid_move')}`")
    parts.append(f"- parse_error: `{row.get('parse_error')}`")
    parts.append(f"- legal_actions: `{row.get('legal_actions')}`")
    parts.append("")
    parts.append("### Feedback after this action")
    parts.append("```json")
    parts.append(json.dumps(row.get("env_feedback"), ensure_ascii=False, indent=2))
    parts.append("```")
    parts.append("")
    parts.append("### Metrics under two row-order interpretations")
    parts.append("")
    parts.append("| Interpretation | known_acc | overclaim | unknown_preservation | free_f1 | obstacle_f1 | action_target_belief | into_U | hits_own_O |")
    parts.append("|---|---:|---:|---:|---:|---:|---|---:|---:|")
    parts.append(
        f"| topdown_y_desc | {topdown_metrics.get('known_cell_acc')} | {topdown_metrics.get('overclaim_rate')} | "
        f"{topdown_metrics.get('unknown_preservation_rate')} | {topdown_metrics.get('free_f1')} | {topdown_metrics.get('obstacle_f1')} | "
        f"{topdown_action.get('action_target_belief')} | {topdown_action.get('action_into_unknown')} | {topdown_action.get('action_hits_own_obstacle')} |"
    )
    parts.append(
        f"| bottomup_y_asc | {bottomup_metrics.get('known_cell_acc')} | {bottomup_metrics.get('overclaim_rate')} | "
        f"{bottomup_metrics.get('unknown_preservation_rate')} | {bottomup_metrics.get('free_f1')} | {bottomup_metrics.get('obstacle_f1')} | "
        f"{bottomup_action.get('action_target_belief')} | {bottomup_action.get('action_into_unknown')} | {bottomup_action.get('action_hits_own_obstacle')} |"
    )
    parts.append("")
    parts.append("### Gold belief")
    parts.append("```text")
    parts.append(render_topdown_grid(gold, "Gold belief, top-down y-desc"))
    parts.append("```")
    parts.append("")
    parts.append("### Agent raw belief_grid")
    parts.append("```text")
    for i, r in enumerate(agent_grid):
        parts.append(f"row={i}: " + " ".join(r))
    parts.append("```")
    parts.append("")
    parts.append("### Agent interpreted as topdown_y_desc")
    parts.append("```text")
    parts.append(render_topdown_grid(agent_as_topdown, "Agent normalized to top-down"))
    parts.append("```")
    parts.append("")
    parts.append("### Agent interpreted as bottomup_y_asc")
    parts.append("```text")
    parts.append(render_topdown_grid(agent_as_bottomup, "Agent normalized to top-down after bottom-up interpretation"))
    parts.append("```")
    parts.append("")
    parts.append("### Raw response")
    parts.append("```json")
    parts.append(str(row.get("raw_response_text")))
    parts.append("```")
    parts.append("")

    return "\n".join(parts)


def select_case_episodes(episode_metrics: list[dict[str, Any]]) -> list[str]:
    selected = []

    failures = [m for m in episode_metrics if not m.get("success")]
    failures = sorted(failures, key=lambda x: x.get("episode_id", ""))
    selected.extend([m["episode_id"] for m in failures[:2]])

    optimal_success = [
        m for m in episode_metrics
        if m.get("success") and m.get("optimality_gap") == 0
    ]
    optimal_success = sorted(
        optimal_success,
        key=lambda x: (
            -(x.get("topdown_mean_known_cell_acc") or 0),
            x.get("episode_id", ""),
        ),
    )
    selected.extend([m["episode_id"] for m in optimal_success[:2]])

    worst_gap = [
        m for m in episode_metrics
        if m.get("success") and m.get("optimality_gap") is not None
    ]
    worst_gap = sorted(
        worst_gap,
        key=lambda x: (
            -(x.get("optimality_gap") or 0),
            x.get("episode_id", ""),
        ),
    )
    selected.extend([m["episode_id"] for m in worst_gap[:2]])

    # Keep order and remove duplicates.
    seen = set()
    out = []
    for ep in selected:
        if ep not in seen:
            seen.add(ep)
            out.append(ep)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", nargs="*", default=None)
    ap.add_argument("--max-steps-per-case", type=int, default=9999)
    args = ap.parse_args()

    run_dir = Path(args.run)
    steps = load_jsonl(run_dir / "steps.jsonl")
    summaries = load_json(run_dir / "summary.json")

    by_ep = defaultdict(list)
    for row in steps:
        by_ep[row["episode_id"]].append(row)

    step_rows = []
    episode_rows = []
    case_payloads: dict[str, list[str]] = {}

    for summary in summaries:
        ep = summary["episode_id"]
        ep_steps = by_ep.get(ep, [])
        if not ep_steps:
            continue

        first_prompt = ep_steps[0]["prompt_text"]
        start = json.loads(extract_between(first_prompt, "<start>", "</start>"))
        goal = json.loads(extract_between(first_prompt, "<goal>", "</goal>"))
        size = int(ep_steps[0]["grid_size"])

        gold = init_gold(size, start, goal)
        prev_topdown_norm = None
        prev_bottomup_norm = None

        ep_topdown_known = []
        ep_bottomup_known = []
        ep_topdown_overclaim = []
        ep_bottomup_overclaim = []
        ep_topdown_action_U = 0
        ep_bottomup_action_U = 0
        ep_topdown_mono = 0
        ep_bottomup_mono = 0

        case_payloads[ep] = [f"# Diagnostic case: {ep}", ""]

        for row in ep_steps:
            prompt = row["prompt_text"]
            current_pos = json.loads(extract_between(prompt, "<current_pos>", "</current_pos>"))
            last_feedback = extract_json_tag(prompt, "last_feedback")
            if last_feedback is not None:
                apply_strategy_A_feedback(
                    gold,
                    feedback=last_feedback,
                    current_pos=current_pos,
                    goal=goal,
                )

            agent_grid = row.get("parsed_belief_grid")
            if not isinstance(agent_grid, list):
                continue

            topdown_metrics = compare_belief(agent_grid, gold, "topdown_y_desc")
            bottomup_metrics = compare_belief(agent_grid, gold, "bottomup_y_asc")

            topdown_action = action_consistency(
                agent_grid,
                row.get("agent_pos_before"),
                row.get("parsed_action"),
                "topdown_y_desc",
            )
            bottomup_action = action_consistency(
                agent_grid,
                row.get("agent_pos_before"),
                row.get("parsed_action"),
                "bottomup_y_asc",
            )

            topdown_norm = normalize_agent_grid(agent_grid, "topdown_y_desc")
            bottomup_norm = normalize_agent_grid(agent_grid, "bottomup_y_asc")

            topdown_mono = monotonicity_violations(prev_topdown_norm, topdown_norm)
            bottomup_mono = monotonicity_violations(prev_bottomup_norm, bottomup_norm)

            prev_topdown_norm = topdown_norm
            prev_bottomup_norm = bottomup_norm

            ep_topdown_known.append(topdown_metrics["known_cell_acc"])
            ep_bottomup_known.append(bottomup_metrics["known_cell_acc"])
            ep_topdown_overclaim.append(topdown_metrics["overclaim_rate"])
            ep_bottomup_overclaim.append(bottomup_metrics["overclaim_rate"])

            ep_topdown_action_U += int(bool(topdown_action["action_into_unknown"]))
            ep_bottomup_action_U += int(bool(bottomup_action["action_into_unknown"]))
            ep_topdown_mono += topdown_mono
            ep_bottomup_mono += bottomup_mono

            step_record = {
                "episode_id": ep,
                "step_id": row["step_id"],
                "seed": row.get("seed"),
                "success": summary.get("success"),
                "optimality_gap": summary.get("optimality_gap"),
                "pos_before": row.get("agent_pos_before"),
                "action": row.get("parsed_action"),
                "pos_after": row.get("agent_pos_after"),
                "topdown_known_cell_acc": topdown_metrics["known_cell_acc"],
                "bottomup_known_cell_acc": bottomup_metrics["known_cell_acc"],
                "known_acc_delta_bottomup_minus_topdown": (
                    None
                    if topdown_metrics["known_cell_acc"] is None or bottomup_metrics["known_cell_acc"] is None
                    else bottomup_metrics["known_cell_acc"] - topdown_metrics["known_cell_acc"]
                ),
                "topdown_overclaim_rate": topdown_metrics["overclaim_rate"],
                "bottomup_overclaim_rate": bottomup_metrics["overclaim_rate"],
                "topdown_action_into_unknown": topdown_action["action_into_unknown"],
                "bottomup_action_into_unknown": bottomup_action["action_into_unknown"],
                "topdown_action_hits_own_obstacle": topdown_action["action_hits_own_obstacle"],
                "bottomup_action_hits_own_obstacle": bottomup_action["action_hits_own_obstacle"],
                "topdown_monotonicity_violations": topdown_mono,
                "bottomup_monotonicity_violations": bottomup_mono,
            }
            step_rows.append(step_record)

            if len(case_payloads[ep]) < args.max_steps_per_case * 2 + 2:
                case_payloads[ep].append(
                    make_step_case_md(
                        row=row,
                        gold=[r[:] for r in gold],
                        agent_grid=agent_grid,
                        topdown_metrics=topdown_metrics,
                        bottomup_metrics=bottomup_metrics,
                        topdown_action=topdown_action,
                        bottomup_action=bottomup_action,
                    )
                )

        topdown_mean_known = mean_ignore_none(ep_topdown_known)
        bottomup_mean_known = mean_ignore_none(ep_bottomup_known)

        ep_row = {
            "episode_id": ep,
            "seed": summary.get("seed"),
            "success": summary.get("success"),
            "steps_used": summary.get("steps_used"),
            "shortest_path_len": summary.get("shortest_path_len"),
            "optimality_gap": summary.get("optimality_gap"),
            "topdown_mean_known_cell_acc": topdown_mean_known,
            "bottomup_mean_known_cell_acc": bottomup_mean_known,
            "known_acc_delta_bottomup_minus_topdown": (
                None
                if topdown_mean_known is None or bottomup_mean_known is None
                else bottomup_mean_known - topdown_mean_known
            ),
            "topdown_mean_overclaim_rate": mean_ignore_none(ep_topdown_overclaim),
            "bottomup_mean_overclaim_rate": mean_ignore_none(ep_bottomup_overclaim),
            "topdown_action_into_unknown_count": ep_topdown_action_U,
            "bottomup_action_into_unknown_count": ep_bottomup_action_U,
            "topdown_monotonicity_violation_total": ep_topdown_mono,
            "bottomup_monotonicity_violation_total": ep_bottomup_mono,
        }
        episode_rows.append(ep_row)

    mean_delta = mean_ignore_none([r["known_acc_delta_bottomup_minus_topdown"] for r in episode_rows])
    orientation_guess = (
        "bottomup_y_asc_may_explain_errors"
        if mean_delta is not None and mean_delta > 0.10
        else "row_order_confusion_not_sufficient"
    )

    summary_out = {
        "run_dir": str(run_dir),
        "orientation_guess": orientation_guess,
        "mean_known_acc_delta_bottomup_minus_topdown": mean_delta,
        "episodes": episode_rows,
    }

    out_dir = run_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "orientation_diagnosis.json").write_text(
        json.dumps(summary_out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = out_dir / "orientation_episode_metrics.csv"
    if episode_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
            writer.writeheader()
            writer.writerows(episode_rows)

    step_path = out_dir / "orientation_step_metrics.jsonl"
    with step_path.open("w", encoding="utf-8") as f:
        for row in step_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.episodes:
        selected_eps = args.episodes
    else:
        selected_eps = select_case_episodes(episode_rows)

    for ep in selected_eps:
        if ep in case_payloads:
            (out_dir / f"{ep}.md").write_text(
                "\n\n".join(case_payloads[ep]),
                encoding="utf-8",
            )

    print("=" * 100)
    print("DIAGNOSE WORLD MODEL A:", run_dir)
    print("=" * 100)
    print("orientation_guess:", orientation_guess)
    print("mean_known_acc_delta_bottomup_minus_topdown:", mean_delta)

    print("\nPER-EPISODE ORIENTATION CHECK")
    for r in episode_rows:
        print(
            f"{r['episode_id']:>10} | "
            f"success={str(r['success']):5s} | "
            f"gap={str(r['optimality_gap']):>4} | "
            f"top_known={r['topdown_mean_known_cell_acc']} | "
            f"bottom_known={r['bottomup_mean_known_cell_acc']} | "
            f"delta={r['known_acc_delta_bottomup_minus_topdown']} | "
            f"top_U_action={r['topdown_action_into_unknown_count']} | "
            f"bottom_U_action={r['bottomup_action_into_unknown_count']} | "
            f"top_mono={r['topdown_monotonicity_violation_total']} | "
            f"bottom_mono={r['bottomup_monotonicity_violation_total']}"
        )

    print("\nSaved:", out_dir / "orientation_diagnosis.json")
    print("Saved:", csv_path)
    print("Saved:", step_path)
    print("Saved case markdown files for:", ", ".join(selected_eps))


if __name__ == "__main__":
    main()
