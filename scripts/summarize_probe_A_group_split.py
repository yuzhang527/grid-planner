from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--metric", default="balanced_accuracy_mean")
    args = ap.parse_args()

    path = Path(args.csv)

    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)

    print("=" * 90)
    print(f"Best layer per task by {args.metric}")
    print("=" * 90)

    for task, rs in by_task.items():
        rs = sorted(rs, key=lambda r: to_float(r[args.metric]), reverse=True)
        best = rs[0]

        print(
            f"{task:28s} | "
            f"best_layer={int(best['layer']):>3d} | "
            f"bal_acc={float(best['balanced_accuracy_mean']):.3f}"
            f"±{float(best['balanced_accuracy_std']):.3f} | "
            f"macro_f1={float(best['macro_f1_mean']):.3f}"
            f"±{float(best['macro_f1_std']):.3f} | "
            f"valid_splits={best['valid_splits']}"
        )

    print("=" * 90)


if __name__ == "__main__":
    main()
