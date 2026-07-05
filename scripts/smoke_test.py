from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from src.agent.model_mock import MockOracleModel
from src.agent.runner import run_episode
from src.env.generator import generate_map
from src.env.grid_env import GridWorldEnv
from src.utils.io import write_json, write_jsonl


def _load_hf_model(model_name: str, temperature: float, max_new_tokens: int):
    from src.agent.model_hf import HFModel

    return HFModel(
        model_name=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )


def render_map(obstacle_map: list[list[int]], start: tuple[int, int], goal: tuple[int, int]) -> str:
    """Render with y descending so (0,0) is visibly bottom-left."""
    size = len(obstacle_map)
    lines: list[str] = []

    for y in range(size - 1, -1, -1):
        row = []
        for x in range(size):
            pos = (x, y)
            if pos == start:
                row.append("S")
            elif pos == goal:
                row.append("G")
            elif obstacle_map[y][x] == 1:
                row.append("#")
            else:
                row.append(".")
        lines.append(f"y={y}  " + " ".join(row))

    lines.append("     " + " ".join(str(x) for x in range(size)))
    lines.append("     x-axis, origin (0,0) is bottom-left")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    backend = cfg.get("backend", "mock_oracle")
    if backend == "mock_oracle":
        model = MockOracleModel()
    elif backend == "hf":
        model = _load_hf_model(
            model_name=cfg["model_name"],
            temperature=float(cfg.get("temperature", 0.0)),
            max_new_tokens=int(cfg.get("max_new_tokens", 256)),
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    all_steps = []
    summaries = []

    base_seed = int(cfg["seed"])
    num_episodes = int(cfg.get("num_episodes", 1))

    for ep_idx in range(num_episodes):
        seed = base_seed + ep_idx

        generated = generate_map(
            grid_size=int(cfg["grid_size"]),
            num_obstacles=int(cfg["num_obstacles"]),
            seed=seed,
            require_unique_shortest_path=bool(cfg.get("require_unique_shortest_path", False)),
        )

        env = GridWorldEnv(
            obstacle_map=generated.obstacle_map,
            start=generated.start,
            goal=generated.goal,
            max_steps=int(cfg["max_steps"]),
            feedback_type=str(cfg["feedback_type"]),
        )

        episode_id = f"A_seed{seed}"
        steps, summary = run_episode(env, model, episode_id=episode_id, seed=seed)
        all_steps.extend(steps)
        summaries.append(summary)

        print("=" * 60)
        print(f"Episode {episode_id}")
        print("Coordinate system: Cartesian [x, y], origin bottom-left")
        print(render_map(generated.obstacle_map, generated.start, generated.goal))
        print(
            f"success={summary['success']} "
            f"steps={summary['steps_used']} "
            f"shortest={summary['shortest_path_len']} "
            f"gap={summary['optimality_gap']}"
        )
        print("path:", [t["new_pos"] for t in summary["trajectory"]])

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "steps.jsonl", all_steps)
    write_json(output_dir / "summary.json", summaries)

    print("=" * 60)
    print("Saved step logs to:", output_dir / "steps.jsonl")
    print("Saved summaries to:", output_dir / "summary.json")


if __name__ == "__main__":
    sys.exit(main())
