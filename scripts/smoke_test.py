from __future__ import annotations

import argparse
from pathlib import Path

from src.agent.model_hf import HFChatModel
from src.agent.model_mock import MockOracleModel
from src.agent.runner import run_episode
from src.env.generator import generate_map
from src.env.grid_env import GridWorldEnv
from src.utils.io import ensure_dir, load_yaml, write_json, write_jsonl


def load_model(cfg):
    backend = cfg.get("backend", "mock_oracle")
    if backend == "mock_oracle":
        return MockOracleModel()
    if backend == "hf":
        return HFChatModel(
            cfg["model_name"],
            max_new_tokens=int(cfg.get("max_new_tokens", 256)),
            temperature=float(cfg.get("temperature", 0.0)),
        )
    raise ValueError(f"Unknown backend: {backend}")


def print_map(obstacle_map, start, goal):
    chars = []
    for r, row in enumerate(obstacle_map):
        out = []
        for c, val in enumerate(row):
            if (r, c) == start:
                out.append("S")
            elif (r, c) == goal:
                out.append("G")
            elif val == 1:
                out.append("#")
            else:
                out.append(".")
        chars.append(" ".join(out))
    return "\n".join(chars)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/exp_A_adjacent.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = ensure_dir(cfg.get("output_dir", "outputs/logs/strategy_A_smoke"))
    model = load_model(cfg)

    all_records = []
    summaries = []
    base_seed = int(cfg.get("seed", 123))
    for i in range(int(cfg.get("num_episodes", 1))):
        seed = base_seed + i
        gm = generate_map(
            grid_size=int(cfg["grid_size"]),
            num_obstacles=int(cfg["num_obstacles"]),
            seed=seed,
            require_unique_shortest_path=bool(cfg.get("require_unique_shortest_path", False)),
        )
        env = GridWorldEnv(
            obstacle_map=gm.obstacle_map,
            start=gm.start,
            goal=gm.goal,
            max_steps=int(cfg.get("max_steps", 20)),
            feedback_type=cfg.get("feedback_type", "adjacent_exact"),
        )
        episode_id = f"A_seed{seed}"
        records, summary = run_episode(env, model, episode_id=episode_id, seed=seed)
        all_records.extend(records)
        summaries.append(summary)

        print("=" * 60)
        print(f"Episode {episode_id}")
        print(print_map(gm.obstacle_map, gm.start, gm.goal))
        print(f"success={summary['success']} steps={summary['steps_used']} shortest={summary['shortest_path_len']} gap={summary['optimality_gap']}")
        print("path:", [t["new_pos"] for t in summary["trajectory"]])

    write_jsonl(out_dir / "steps.jsonl", all_records)
    write_json(out_dir / "summary.json", summaries)
    print("=" * 60)
    print(f"Saved step logs to: {out_dir / 'steps.jsonl'}")
    print(f"Saved summaries to: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
