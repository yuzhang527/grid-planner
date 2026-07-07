from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root dir containing shard_0, shard_1, ...")
    ap.add_argument("--out", required=True, help="Merged output dir")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)

    shard_dirs = sorted(
        [p for p in root.glob("shard_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[-1]),
    )

    if not shard_dirs:
        raise FileNotFoundError(f"No shard dirs found under {root}")

    all_steps = []
    all_summaries = []

    for shard_dir in shard_dirs:
        steps_path = shard_dir / "steps.jsonl"
        summary_path = shard_dir / "summary.json"

        if not steps_path.exists():
            raise FileNotFoundError(f"Missing {steps_path}")
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing {summary_path}")

        steps = load_jsonl(steps_path)
        summaries = json.loads(summary_path.read_text(encoding="utf-8"))

        all_steps.extend(steps)
        all_summaries.extend(summaries)

        print(
            f"Loaded {shard_dir}: "
            f"steps={len(steps)}, episodes={len(summaries)}"
        )

    all_steps.sort(
        key=lambda r: (
            int(r.get("seed", 0)),
            int(r.get("step_id", 0)),
        )
    )
    all_summaries.sort(key=lambda r: int(r.get("seed", 0)))

    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "steps.jsonl", all_steps)
    (out / "summary.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    success = sum(1 for r in all_summaries if r.get("success"))

    print("=" * 70)
    print(f"Merged output: {out}")
    print(f"episodes:      {len(all_summaries)}")
    print(f"steps:         {len(all_steps)}")
    print(f"success:       {success}/{len(all_summaries)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
