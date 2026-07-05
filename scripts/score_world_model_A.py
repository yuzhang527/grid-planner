from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


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


def coord_to_grid_index(size: int, coord: list[int] | tuple[int, int]) -> tuple[int, int] | None:
    """Cartesian [x,y] -> top-down belief_grid index [row,col]."""
    x, y = int(coord[0]), int(coord[1])
    if not (0 <= x < size and 0 <= y < size):
        return None
    return size - 1 - y, x


def get_cell(grid: list[list[str]], coord: list[int] | tuple[int, int]) -> str | None:
    idx = coord_to_grid_index(len(grid), coord)
    if idx is None:
        return None
    r, c = idx
    try:
        return grid[r][c]
    except Exception:
        return None


def set_cell(grid: list[list[str]], coord: list[int] | tuple[int, int], value: str) -> None:
    idx = coord_to_grid_index(len(grid), coord)
    if idx is None:
        return
    r, c = idx
    grid[r][c] = value


def init_gold(size: int, start: list[int], goal: list[int]) -> list[list[str]]:
    gold = [["U" for _ in range(size)] for _ in range(size)]
    set_cell(gold, start, "F")
    set_cell(gold, goal, "F")
    return gold


def apply_visible_feedback(gold: list[list[str]], feedback: dict, current_pos: list[int], goal: list[int]) -> None:
    """For Strategy A, visible feedback gives exact O/F labels."""
    for cell in feedback.get("blocked", []):
        set_cell(gold, cell, "O")
    for cell in feedback.get("free", []):
        set_cell(gold, cell, "F")

    # Current position and goal are definitely free.
    set_cell(gold, current_pos, "F")
    set_cell(gold, goal, "F")


def flatten(grid: list[list[str]]) -> list[str]:
    return [x for row in grid for x in row]


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def compare_belief(agent_grid: list[list[str]], gold_grid: list[list[str]]) -> dict:
    a = flatten(agent_grid)
    g = flatten(gold_grid)

    n = len(g)
    all_acc = sum(x == y for x, y in zip(a, g)) / n if n else None

    known_idx = [i for i, y in enumerate(g) if y in {"O", "F"}]
    unknown_idx = [i for i, y in enumerate(g) if y == "U"]

    known_acc = (
        None
        if not known_idx
        else sum(a[i] == g[i] for i in known_idx) / len(known_idx)
    )

    unknown_preservation = (
        None
        if not unknown_idx
        else sum(a[i] == "U" for i in unknown_idx) / len(unknown_idx)
    )

    overclaim_rate = (
        None
        if not unknown_idx
        else sum(a[i] in {"O", "F"} for i in unknown_idx) / len(unknown_idx)
    )

    obstacle_tp = sum(x == "O" and y == "O" for x, y in zip(a, g))
    obstacle_fp = sum(x == "O" and y != "O" for x, y in zip(a, g))
    obstacle_fn = sum(x != "O" and y == "O" for x, y in zip(a, g))

    free_tp = sum(x == "F" and y == "F" for x, y in zip(a, g))
    free_fp = sum(x == "F" and y != "F" for x, y in zip(a, g))
    free_fn = sum(x != "F" and y == "F" for x, y in zip(a, g))

    obstacle = prf(obstacle_tp, obstacle_fp, obstacle_fn)
    free = prf(free_tp, free_fp, free_fn)

    return {
        "all_cell_acc": all_acc,
        "known_cell_acc": known_acc,
        "unknown_preservation_rate": unknown_preservation,
        "overclaim_rate": overclaim_rate,
        "obstacle_precision": obstacle["precision"],
        "obstacle_recall": obstacle["recall"],
        "obstacle_f1": obstacle["f1"],
        "free_precision": free["precision"],
        "free_recall": free["recall"],
        "free_f1": free["f1"],
        "known_cells": len(known_idx),
        "unknown_cells": len(unknown_idx),
    }


def target_after_action(pos: list[int], action: str) -> list[int] | None:
    if action not in ACTIONS:
        return None
    dx, dy = ACTIONS[action]
    return [int(pos[0]) + dx, int(pos[1]) + dy]


def action_consistency(agent_grid: list[list[str]], pos: list[int], action: str) -> dict:
    target = target_after_action(pos, action)
    if target is None:
        return {
            "action_target": None,
            "action_target_belief": None,
            "action_hits_own_obstacle": None,
            "action_into_unknown": None,
        }

    val = get_cell(agent_grid, target)
    return {
        "action_target": target,
        "action_target_belief": val,
        "action_hits_own_obstacle": val == "O",
        "action_into_unknown": val == "U",
    }


def monotonicity_violations(prev_agent: list[list[str]] | None, cur_agent: list[list[str]]) -> int:
    if prev_agent is None:
        return 0
    count = 0
    for r in range(len(cur_agent)):
        for c in range(len(cur_agent[r])):
            prev = prev_agent[r][c]
            cur = cur_agent[r][c]
            if prev in {"O", "F"} and cur != prev:
                count += 1
    return count


def mean_ignore_none(values):
    xs = [x for x in values if x is not None]
    return None if not xs else sum(xs) / len(xs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run)
    steps_path = run_dir / "steps.jsonl"
    summary_path = run_dir / "summary.json"

    steps = load_jsonl(steps_path)
    summaries = load_json(summary_path)

    by_ep = defaultdict(list)
    for row in steps:
        by_ep[row["episode_id"]].append(row)

    step_metrics = []
    episode_metrics = []

    for summary in summaries:
        ep = summary["episode_id"]
        ep_steps = by_ep[ep]
        if not ep_steps:
            continue

        first_prompt = ep_steps[0]["prompt_text"]
        start = json.loads(extract_between(first_prompt, "<start>", "</start>"))
        goal = json.loads(extract_between(first_prompt, "<goal>", "</goal>"))
        size = int(ep_steps[0]["grid_size"])

        gold = init_gold(size, start, goal)
        prev_agent_grid = None

        for row in ep_steps:
            prompt = row["prompt_text"]
            current_pos = json.loads(extract_between(prompt, "<current_pos>", "</current_pos>"))
            last_feedback = extract_json_tag(prompt, "last_feedback")

            if last_feedback is not None:
                apply_visible_feedback(gold, last_feedback, current_pos, goal)

            agent_grid = row.get("parsed_belief_grid")
            if not isinstance(agent_grid, list):
                metric = {
                    "episode_id": ep,
                    "step_id": row["step_id"],
                    "parse_error": True,
                    "all_cell_acc": None,
                    "known_cell_acc": None,
                    "overclaim_rate": None,
                    "unknown_preservation_rate": None,
                    "obstacle_f1": None,
                    "free_f1": None,
                    "action_hits_own_obstacle": None,
                    "action_into_unknown": None,
                    "monotonicity_violations": None,
                }
            else:
                cmp = compare_belief(agent_grid, gold)
                act = action_consistency(agent_grid, row["agent_pos_before"], row["parsed_action"])
                mono = monotonicity_violations(prev_agent_grid, agent_grid)

                metric = {
                    "episode_id": ep,
                    "step_id": row["step_id"],
                    "seed": row.get("seed"),
                    "success": summary.get("success"),
                    "pos_before": row.get("agent_pos_before"),
                    "action": row.get("parsed_action"),
                    "pos_after": row.get("agent_pos_after"),
                    "parse_error": bool(row.get("parse_error")),
                    "valid_move": bool(row.get("valid_move")),
                    "all_cell_acc": cmp["all_cell_acc"],
                    "known_cell_acc": cmp["known_cell_acc"],
                    "unknown_preservation_rate": cmp["unknown_preservation_rate"],
                    "overclaim_rate": cmp["overclaim_rate"],
                    "obstacle_precision": cmp["obstacle_precision"],
                    "obstacle_recall": cmp["obstacle_recall"],
                    "obstacle_f1": cmp["obstacle_f1"],
                    "free_precision": cmp["free_precision"],
                    "free_recall": cmp["free_recall"],
                    "free_f1": cmp["free_f1"],
                    "known_cells": cmp["known_cells"],
                    "unknown_cells": cmp["unknown_cells"],
                    "action_target": act["action_target"],
                    "action_target_belief": act["action_target_belief"],
                    "action_hits_own_obstacle": act["action_hits_own_obstacle"],
                    "action_into_unknown": act["action_into_unknown"],
                    "monotonicity_violations": mono,
                }

                prev_agent_grid = agent_grid

            step_metrics.append(metric)

        ep_m = {
            "episode_id": ep,
            "seed": summary.get("seed"),
            "success": summary.get("success"),
            "steps_used": summary.get("steps_used"),
            "shortest_path_len": summary.get("shortest_path_len"),
            "optimality_gap": summary.get("optimality_gap"),
            "mean_all_cell_acc": mean_ignore_none([m["all_cell_acc"] for m in step_metrics if m["episode_id"] == ep]),
            "mean_known_cell_acc": mean_ignore_none([m["known_cell_acc"] for m in step_metrics if m["episode_id"] == ep]),
            "mean_overclaim_rate": mean_ignore_none([m["overclaim_rate"] for m in step_metrics if m["episode_id"] == ep]),
            "mean_unknown_preservation_rate": mean_ignore_none([m["unknown_preservation_rate"] for m in step_metrics if m["episode_id"] == ep]),
            "mean_obstacle_f1": mean_ignore_none([m["obstacle_f1"] for m in step_metrics if m["episode_id"] == ep]),
            "mean_free_f1": mean_ignore_none([m["free_f1"] for m in step_metrics if m["episode_id"] == ep]),
            "action_hits_own_obstacle_count": sum(bool(m["action_hits_own_obstacle"]) for m in step_metrics if m["episode_id"] == ep),
            "action_into_unknown_count": sum(bool(m["action_into_unknown"]) for m in step_metrics if m["episode_id"] == ep),
            "monotonicity_violation_total": sum((m["monotonicity_violations"] or 0) for m in step_metrics if m["episode_id"] == ep),
        }
        episode_metrics.append(ep_m)

    aggregate = {
        "run_dir": str(run_dir),
        "num_episodes": len(episode_metrics),
        "success_rate": mean_ignore_none([1.0 if m["success"] else 0.0 for m in episode_metrics]),
        "mean_all_cell_acc": mean_ignore_none([m["mean_all_cell_acc"] for m in episode_metrics]),
        "mean_known_cell_acc": mean_ignore_none([m["mean_known_cell_acc"] for m in episode_metrics]),
        "mean_overclaim_rate": mean_ignore_none([m["mean_overclaim_rate"] for m in episode_metrics]),
        "mean_unknown_preservation_rate": mean_ignore_none([m["mean_unknown_preservation_rate"] for m in episode_metrics]),
        "mean_obstacle_f1": mean_ignore_none([m["mean_obstacle_f1"] for m in episode_metrics]),
        "mean_free_f1": mean_ignore_none([m["mean_free_f1"] for m in episode_metrics]),
        "total_action_hits_own_obstacle": sum(m["action_hits_own_obstacle_count"] for m in episode_metrics),
        "total_action_into_unknown": sum(m["action_into_unknown_count"] for m in episode_metrics),
        "total_monotonicity_violations": sum(m["monotonicity_violation_total"] for m in episode_metrics),
        "episodes": episode_metrics,
    }

    print("=" * 100)
    print("WORLD MODEL ANALYSIS:", run_dir)
    print("=" * 100)
    print("success_rate:", aggregate["success_rate"])
    print("mean_all_cell_acc:", aggregate["mean_all_cell_acc"])
    print("mean_known_cell_acc:", aggregate["mean_known_cell_acc"])
    print("mean_overclaim_rate:", aggregate["mean_overclaim_rate"])
    print("mean_unknown_preservation_rate:", aggregate["mean_unknown_preservation_rate"])
    print("mean_obstacle_f1:", aggregate["mean_obstacle_f1"])
    print("mean_free_f1:", aggregate["mean_free_f1"])
    print("total_action_hits_own_obstacle:", aggregate["total_action_hits_own_obstacle"])
    print("total_action_into_unknown:", aggregate["total_action_into_unknown"])
    print("total_monotonicity_violations:", aggregate["total_monotonicity_violations"])

    print("\nPER-EPISODE WORLD MODEL")
    for m in episode_metrics:
        print(
            f"{m['episode_id']:>10} | "
            f"success={str(m['success']):5s} | "
            f"gap={str(m['optimality_gap']):>4} | "
            f"known_acc={m['mean_known_cell_acc']} | "
            f"overclaim={m['mean_overclaim_rate']} | "
            f"obs_f1={m['mean_obstacle_f1']} | "
            f"free_f1={m['mean_free_f1']} | "
            f"own_O={m['action_hits_own_obstacle_count']} | "
            f"into_U={m['action_into_unknown_count']} | "
            f"mono={m['monotonicity_violation_total']}"
        )

    out_summary = run_dir / "world_model_summary.json"
    out_steps = run_dir / "world_model_step_metrics.jsonl"

    out_summary.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    with out_steps.open("w", encoding="utf-8") as f:
        for row in step_metrics:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nSaved:", out_summary)
    print("Saved:", out_steps)


if __name__ == "__main__":
    main()
