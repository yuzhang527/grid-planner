from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


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


def safe_mean(values):
    values = [v for v in values if v is not None]
    return None if not values else mean(values)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory, e.g. outputs/logs/strategy_A_qwen_10")
    args = ap.parse_args()

    run_dir = Path(args.run)
    summary_path = run_dir / "summary.json"
    steps_path = run_dir / "steps.jsonl"

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not steps_path.exists():
        raise FileNotFoundError(steps_path)

    summaries = load_json(summary_path)
    steps = load_jsonl(steps_path)

    by_ep = defaultdict(list)
    for row in steps:
        by_ep[row["episode_id"]].append(row)

    episode_rows = []
    for s in summaries:
        ep = s["episode_id"]
        ep_steps = by_ep.get(ep, [])

        parse_errors = sum(bool(x.get("parse_error")) for x in ep_steps)
        invalid_moves = sum(not bool(x.get("valid_move")) for x in ep_steps)
        illegal_before_repair = sum(bool(x.get("illegal_action_before_repair")) for x in ep_steps)
        repaired = sum(bool(x.get("repaired")) for x in ep_steps)

        repeated_positions = 0
        seen_transitions = set()
        for x in ep_steps:
            key = (tuple(x.get("agent_pos_before", [])), x.get("parsed_action"))
            if key in seen_transitions:
                repeated_positions += 1
            seen_transitions.add(key)

        row = {
            "episode_id": ep,
            "seed": s.get("seed"),
            "success": bool(s.get("success")),
            "steps_used": s.get("steps_used"),
            "shortest_path_len": s.get("shortest_path_len"),
            "optimality_gap": s.get("optimality_gap"),
            "final_pos": s.get("final_pos"),
            "goal_pos": s.get("goal_pos"),
            "parse_errors": parse_errors,
            "invalid_moves": invalid_moves,
            "illegal_action_before_repair": illegal_before_repair,
            "repair_used": repaired,
            "repeated_position_action_pairs": repeated_positions,
            "coordinate_system": s.get("coordinate_system", "unknown"),
        }
        episode_rows.append(row)

    n = len(episode_rows)
    success_n = sum(r["success"] for r in episode_rows)

    print("=" * 88)
    print("RUN:", run_dir)
    print("=" * 88)
    print(f"episodes: {n}")
    print(f"success: {success_n}/{n} = {success_n / n if n else 0:.3f}")
    print(f"mean_steps_success_only: {safe_mean([r['steps_used'] for r in episode_rows if r['success']])}")
    print(f"mean_optimality_gap_success_only: {safe_mean([r['optimality_gap'] for r in episode_rows if r['success']])}")
    print(f"total_parse_errors: {sum(r['parse_errors'] for r in episode_rows)}")
    print(f"total_invalid_moves: {sum(r['invalid_moves'] for r in episode_rows)}")
    print(f"total_illegal_action_before_repair: {sum(r['illegal_action_before_repair'] for r in episode_rows)}")
    print(f"total_repair_used: {sum(r['repair_used'] for r in episode_rows)}")
    print(f"total_repeated_position_action_pairs: {sum(r['repeated_position_action_pairs'] for r in episode_rows)}")

    print("\nPER-EPISODE")
    for r in episode_rows:
        print(
            f"{r['episode_id']:>10} | "
            f"success={str(r['success']):5s} | "
            f"steps={r['steps_used']:>2} | "
            f"shortest={r['shortest_path_len']:>2} | "
            f"gap={str(r['optimality_gap']):>4} | "
            f"parse={r['parse_errors']} | "
            f"invalid={r['invalid_moves']} | "
            f"illegal_pre_repair={r['illegal_action_before_repair']} | "
            f"repair={r['repair_used']} | "
            f"repeat={r['repeated_position_action_pairs']} | "
            f"final={r['final_pos']}"
        )

    out_json = run_dir / "analysis_summary.json"
    out_csv = run_dir / "episode_metrics.csv"

    aggregate = {
        "run_dir": str(run_dir),
        "episodes": n,
        "success_count": success_n,
        "success_rate": success_n / n if n else 0,
        "mean_steps_success_only": safe_mean([r["steps_used"] for r in episode_rows if r["success"]]),
        "mean_optimality_gap_success_only": safe_mean([r["optimality_gap"] for r in episode_rows if r["success"]]),
        "total_parse_errors": sum(r["parse_errors"] for r in episode_rows),
        "total_invalid_moves": sum(r["invalid_moves"] for r in episode_rows),
        "total_illegal_action_before_repair": sum(r["illegal_action_before_repair"] for r in episode_rows),
        "total_repair_used": sum(r["repair_used"] for r in episode_rows),
        "total_repeated_position_action_pairs": sum(r["repeated_position_action_pairs"] for r in episode_rows),
        "episodes_detail": episode_rows,
    }

    out_json.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()) if episode_rows else [])
        if episode_rows:
            writer.writeheader()
            writer.writerows(episode_rows)

    print("\nSaved:", out_json)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
