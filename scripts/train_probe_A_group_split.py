from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OFU_MAP = {"O": 0, "F": 1, "U": 2}
OFUW_MAP = {"O": 0, "F": 1, "U": 2, "WALL": 3}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_meta(npz) -> list[dict[str, Any]]:
    return [json.loads(str(x)) for x in npz["meta_json"].tolist()]


def bool_label(v: Any) -> int | None:
    if v is None:
        return None
    return 1 if bool(v) else 0


def cell_from_belief_grid(
    belief_grid: list[list[str]],
    x: int,
    y: int,
    size: int,
) -> str:
    """
    Project coordinate:
      coordinate_system = cartesian_bottom_left
      belief_grid row order = top_to_bottom_y_descending

    Therefore:
      grid_row = size - 1 - y
      grid_col = x
    """
    if x < 0 or y < 0 or x >= size or y >= size:
        return "WALL"

    row = size - 1 - y
    col = x

    try:
        val = str(belief_grid[row][col]).strip().upper()
    except Exception:
        return "U"

    if val not in {"O", "F", "U"}:
        return "U"

    return val


def local_neighbor_labels_from_step(row: dict[str, Any]) -> dict[str, int | None]:
    pos = row.get("agent_pos_before")
    grid = row.get("parsed_belief_grid")
    size = row.get("grid_size")

    if pos is None or grid is None or size is None:
        return {
            "local_UP_OFUW": None,
            "local_DOWN_OFUW": None,
            "local_LEFT_OFUW": None,
            "local_RIGHT_OFUW": None,
        }

    x, y = int(pos[0]), int(pos[1])
    size = int(size)

    candidates = {
        "local_UP_OFUW": (x, y + 1),
        "local_DOWN_OFUW": (x, y - 1),
        "local_LEFT_OFUW": (x - 1, y),
        "local_RIGHT_OFUW": (x + 1, y),
    }

    out: dict[str, int | None] = {}

    for name, (nx, ny) in candidates.items():
        label = cell_from_belief_grid(grid, nx, ny, size)
        out[name] = OFUW_MAP.get(label)

    return out


def build_tasks(
    meta: list[dict[str, Any]],
    metrics_map: dict[tuple[str, int], dict[str, Any]],
    steps_map: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, list[int | None]]:
    tasks: dict[str, list[int | None]] = {
        "action_target_belief_OFU": [],
        "action_into_unknown": [],
        "monotonicity_violation": [],
        "episode_success": [],
        "local_UP_OFUW": [],
        "local_DOWN_OFUW": [],
        "local_LEFT_OFUW": [],
        "local_RIGHT_OFUW": [],
    }

    for m in meta:
        key = (str(m["episode_id"]), int(m["step_id"]))

        metric_row = metrics_map.get(key)
        step_row = steps_map.get(key)

        if metric_row is None:
            tasks["action_target_belief_OFU"].append(None)
            tasks["action_into_unknown"].append(None)
            tasks["monotonicity_violation"].append(None)
            tasks["episode_success"].append(None)
        else:
            b = metric_row.get("action_target_belief")
            tasks["action_target_belief_OFU"].append(OFU_MAP.get(b))

            tasks["action_into_unknown"].append(
                bool_label(metric_row.get("action_into_unknown"))
            )

            mono = metric_row.get("monotonicity_violations")
            tasks["monotonicity_violation"].append(
                None if mono is None else int(int(mono) > 0)
            )

            tasks["episode_success"].append(
                bool_label(metric_row.get("success"))
            )

        if step_row is None:
            for k in [
                "local_UP_OFUW",
                "local_DOWN_OFUW",
                "local_LEFT_OFUW",
                "local_RIGHT_OFUW",
            ]:
                tasks[k].append(None)
        else:
            local = local_neighbor_labels_from_step(step_row)
            for k, v in local.items():
                tasks[k].append(v)

    return tasks


def evaluate_one_layer_group_split(
    X: np.ndarray,
    y_raw: list[int | None],
    groups_raw: list[str],
    seed: int,
    test_size: float,
    n_splits: int,
    min_class_count: int,
) -> dict[str, Any] | None:
    idx = [i for i, y in enumerate(y_raw) if y is not None]

    if len(idx) < 30:
        return None

    y = np.array([y_raw[i] for i in idx], dtype=np.int64)
    X2 = X[idx]
    groups = np.array([groups_raw[i] for i in idx])

    total_counts = Counter(y.tolist())

    if len(total_counts) < 2:
        return None

    if min(total_counts.values()) < min_class_count:
        return None

    splitter = GroupShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=seed,
    )

    split_results: list[dict[str, Any]] = []

    for split_id, (train_idx, test_idx) in enumerate(
        splitter.split(X2, y, groups=groups)
    ):
        y_train = y[train_idx]
        y_test = y[test_idx]

        train_counts = Counter(y_train.tolist())
        test_counts = Counter(y_test.tolist())

        # Skip bad group splits where train or test misses a class.
        if len(train_counts) < 2 or len(test_counts) < 2:
            continue

        # For multiclass tasks, require train has every class present in total label set.
        if set(train_counts.keys()) != set(total_counts.keys()):
            continue

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                solver="lbfgs",
            ),
        )

        clf.fit(X2[train_idx], y_train)
        pred = clf.predict(X2[test_idx])

        split_results.append(
            {
                "split_id": split_id,
                "train_n": int(len(train_idx)),
                "test_n": int(len(test_idx)),
                "train_groups": int(len(set(groups[train_idx].tolist()))),
                "test_groups": int(len(set(groups[test_idx].tolist()))),
                "accuracy": float(accuracy_score(y_test, pred)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y_test, pred)
                ),
                "macro_f1": float(
                    f1_score(y_test, pred, average="macro")
                ),
                "test_class_counts": {
                    str(k): int(v)
                    for k, v in test_counts.items()
                },
            }
        )

    if not split_results:
        return None

    def mean_std(key: str) -> tuple[float, float]:
        vals = np.array([r[key] for r in split_results], dtype=np.float64)
        return float(vals.mean()), float(vals.std(ddof=0))

    acc_m, acc_s = mean_std("accuracy")
    bal_m, bal_s = mean_std("balanced_accuracy")
    f1_m, f1_s = mean_std("macro_f1")

    return {
        "n": int(len(y)),
        "groups": int(len(set(groups.tolist()))),
        "classes": {str(k): int(v) for k, v in total_counts.items()},
        "valid_splits": int(len(split_results)),
        "accuracy_mean": acc_m,
        "accuracy_std": acc_s,
        "balanced_accuracy_mean": bal_m,
        "balanced_accuracy_std": bal_s,
        "macro_f1_mean": f1_m,
        "macro_f1_std": f1_s,
        "split_details": split_results,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--repr", required=True)
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--steps", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--test-size", type=float, default=0.30)
    ap.add_argument("--n-splits", type=int, default=20)
    ap.add_argument("--min-class-count", type=int, default=10)
    args = ap.parse_args()

    run_dir = Path(args.run)
    repr_path = Path(args.repr)

    metrics_path = (
        Path(args.metrics)
        if args.metrics is not None
        else run_dir / "world_model_step_metrics.jsonl"
    )

    steps_path = (
        Path(args.steps)
        if args.steps is not None
        else run_dir / "steps.jsonl"
    )

    if not repr_path.exists():
        raise FileNotFoundError(f"Missing repr npz: {repr_path}")

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing metrics file: {metrics_path}\n"
            f"Run first: PYTHONPATH=. python scripts/score_world_model_A.py --run {run_dir}"
        )

    if not steps_path.exists():
        raise FileNotFoundError(f"Missing steps file: {steps_path}")

    print(f"[group_probe] run={run_dir}")
    print(f"[group_probe] repr={repr_path}")
    print(f"[group_probe] metrics={metrics_path}")
    print(f"[group_probe] steps={steps_path}")

    npz = np.load(repr_path, allow_pickle=False)
    X_all = npz["X"]  # [N, L, H]
    layers = npz["layers"].tolist()
    meta = load_meta(npz)

    metrics_rows = load_jsonl(metrics_path)
    steps_rows = load_jsonl(steps_path)

    metrics_map = {
        (str(r["episode_id"]), int(r["step_id"])): r
        for r in metrics_rows
    }

    steps_map = {
        (str(r["episode_id"]), int(r["step_id"])): r
        for r in steps_rows
    }

    groups = [str(m["episode_id"]) for m in meta]
    tasks = build_tasks(meta, metrics_map, steps_map)

    print(f"[group_probe] X shape={X_all.shape}")
    print(f"[group_probe] layers={layers}")
    print(f"[group_probe] meta rows={len(meta)}")
    print(f"[group_probe] episodes={len(set(groups))}")

    rows_out: list[dict[str, Any]] = []
    json_out: list[dict[str, Any]] = []

    for task_name, y_raw in tasks.items():
        label_counts = Counter([y for y in y_raw if y is not None])
        print(f"\n[group_probe] task={task_name}, label_counts={dict(label_counts)}")

        for layer_pos, layer_id in enumerate(layers):
            X = X_all[:, layer_pos, :].astype(np.float32)

            result = evaluate_one_layer_group_split(
                X=X,
                y_raw=y_raw,
                groups_raw=groups,
                seed=args.seed,
                test_size=args.test_size,
                n_splits=args.n_splits,
                min_class_count=args.min_class_count,
            )

            if result is None:
                continue

            row = {
                "task": task_name,
                "layer": int(layer_id),
                "n": result["n"],
                "groups": result["groups"],
                "valid_splits": result["valid_splits"],
                "accuracy_mean": result["accuracy_mean"],
                "accuracy_std": result["accuracy_std"],
                "balanced_accuracy_mean": result["balanced_accuracy_mean"],
                "balanced_accuracy_std": result["balanced_accuracy_std"],
                "macro_f1_mean": result["macro_f1_mean"],
                "macro_f1_std": result["macro_f1_std"],
                "classes": result["classes"],
            }

            rows_out.append(row)

            json_item = dict(row)
            json_item["split_details"] = result["split_details"]
            json_out.append(json_item)

            print(
                f"{task_name:28s} | layer={int(layer_id):>3d} | "
                f"n={result['n']:>4d} | groups={result['groups']:>3d} | "
                f"splits={result['valid_splits']:>2d} | "
                f"bal_acc={result['balanced_accuracy_mean']:.3f}"
                f"±{result['balanced_accuracy_std']:.3f} | "
                f"macro_f1={result['macro_f1_mean']:.3f}"
                f"±{result['macro_f1_std']:.3f}"
            )

    if args.out is None:
        out_csv = run_dir / "probe_A_group_split_results.csv"
    else:
        out_csv = Path(args.out)

    out_json = out_csv.with_suffix(".json")

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "task",
        "layer",
        "n",
        "groups",
        "valid_splits",
        "accuracy_mean",
        "accuracy_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "classes",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_out:
            rr = dict(r)
            rr["classes"] = json.dumps(rr["classes"], ensure_ascii=False)
            writer.writerow(rr)

    out_json.write_text(
        json.dumps(json_out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:")
    print(f"  {out_csv}")
    print(f"  {out_json}")


if __name__ == "__main__":
    main()
