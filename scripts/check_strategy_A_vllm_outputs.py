from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run)
    steps_path = run_dir / "steps.jsonl"
    summary_path = run_dir / "summary.json"

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if not steps_path.exists():
        raise FileNotFoundError(f"Missing: {steps_path}")

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing: {summary_path}")

    steps = load_jsonl(steps_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    ep_counter = Counter(row["episode_id"] for row in steps)
    success_count = sum(1 for row in summary if row.get("success"))
    parse_errors = sum(1 for row in steps if row.get("parse_error"))
    invalid_moves = sum(1 for row in steps if not row.get("valid_move", True))
    repaired = sum(1 for row in steps if row.get("repaired"))

    print("=" * 70)
    print("Strategy A vLLM output check")
    print("=" * 70)
    print(f"run_dir:        {run_dir}")
    print(f"episodes:       {len(summary)}")
    print(f"step_rows:      {len(steps)}")
    print(f"success:        {success_count}/{len(summary)}")
    print(f"parse_errors:   {parse_errors}")
    print(f"invalid_moves:  {invalid_moves}")
    print(f"repaired_steps: {repaired}")
    print()
    print("First 10 episode step counts:")
    for ep, cnt in list(ep_counter.items())[:10]:
        print(f"  {ep}: {cnt}")

    required_step_keys = [
        "episode_id",
        "step_id",
        "prompt_text",
        "raw_response_text",
        "parsed_belief_grid",
        "parsed_action",
        "agent_pos_before",
        "agent_pos_after",
        "env_feedback",
    ]

    missing = []
    if steps:
        first = steps[0]
        for k in required_step_keys:
            if k not in first:
                missing.append(k)

    print()
    if missing:
        print("WARNING: first step row is missing keys:")
        for k in missing:
            print(f"  - {k}")
    else:
        print("Step schema check: OK")

    print("=" * 70)


if __name__ == "__main__":
    main()
