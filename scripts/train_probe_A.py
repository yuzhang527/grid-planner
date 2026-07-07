from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_meta(npz) -> list[dict[str, Any]]:
    return [json.loads(x) for x in npz["meta_json"].tolist()]


def safe_label_bool(v: Any) -> int | None:
    if v is None:
        return None
    return 1 if bool(v) else 0


def build_tasks(meta: list[dict[str, Any]], metric_map: dict[tuple[str, int], dict[str, Any]]):
    belief_map = {"O": 0, "F": 1, "U": 2}

    tasks: dict[str, list[int | None]] = {
        "action_target_belief_OFU": [],
        "action_into_unknown": [],
        "monotonicity_violation": [],
        "episode_success": [],
    }

    for m in meta:
        key = (str(m["episode_id"]), int(m["step_id"]))
        row = metric_map.get(key)

        if row is None:
            for k in tasks:
                tasks[k].append(None)
            continue

        b = row.get("action_target_belief")
        tasks["action_target_belief_OFU"].append(belief_map.get(b))

        tasks["action_into_unknown"].append(safe_label_bool(row.get("action_into_unknown")))

        mono = row.get("monotonicity_violations")
        tasks["monotonicity_violation"].append(None if mono is None else int(int(mono) > 0))

        tasks["episode_success"].append(safe_label_bool(row.get("success")))

    return tasks


def evaluate_layer_task(X: np.ndarray, y_raw: list[int | None], seed: int):
    idx = [i for i, y in enumerate(y_raw) if y is not None]
    if len(idx) < 20:
        return None

    y = np.array([y_raw[i] for i in idx], dtype=np.int64)
    X2 = X[idx]

    counts = Counter(y.tolist())
    if len(counts) < 2:
        return None
    if min(counts.values()) < 3:
        return None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X2,
            y,
            test_size=0.3,
            random_state=seed,
            stratify=y,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X2,
            y,
            test_size=0.3,
            random_state=seed,
            stratify=None,
        )

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        ),
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    return {
        "n": int(len(y)),
        "classes": {str(k): int(v) for k, v in counts.items()},
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "test_n": int(len(y_test)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run directory")
    ap.add_argument("--repr", required=True, help="NPZ from scripts/extract_repr_A.py")
    ap.add_argument("--metrics", default=None, help="world_model_step_metrics.jsonl")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run)
    repr_path = Path(args.repr)

    if args.metrics is None:
        metrics_path = run_dir / "world_model_step_metrics.jsonl"
    else:
        metrics_path = Path(args.metrics)

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing {metrics_path}. Run: PYTHONPATH=. python scripts/score_world_model_A.py --run {run_dir}"
        )

    npz = np.load(repr_path, allow_pickle=False)
    X_all = npz["X"]  # [N, L, H]
    layers = npz["layers"].tolist()
    meta = load_meta(npz)

    metrics = load_jsonl(metrics_path)
    metric_map = {
        (str(r["episode_id"]), int(r["step_id"])): r
        for r in metrics
    }

    tasks = build_tasks(meta, metric_map)

    rows_out: list[dict[str, Any]] = []
    for task_name, y_raw in tasks.items():
        for layer_pos, layer_id in enumerate(layers):
            X = X_all[:, layer_pos, :].astype(np.float32)
            result = evaluate_layer_task(X, y_raw, seed=args.seed)
            if result is None:
                continue

            item = {
                "task": task_name,
                "layer": int(layer_id),
                **result,
            }
            rows_out.append(item)
            print(
                f"{task_name:28s} | layer={int(layer_id):>3d} | "
                f"n={result['n']:>4d} | "
                f"acc={result['accuracy']:.3f} | "
                f"bal_acc={result['balanced_accuracy']:.3f} | "
                f"macro_f1={result['macro_f1']:.3f}"
            )

    if args.out is None:
        out_csv = run_dir / "probe_A_results.csv"
        out_json = run_dir / "probe_A_results.json"
    else:
        out_csv = Path(args.out)
        out_json = out_csv.with_suffix(".json")

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if rows_out:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "task",
                "layer",
                "n",
                "test_n",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "classes",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows_out:
                rr = dict(r)
                rr["classes"] = json.dumps(rr["classes"], ensure_ascii=False)
                writer.writerow(rr)

    out_json.write_text(
        json.dumps(rows_out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
