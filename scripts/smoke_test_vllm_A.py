from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.agent.prompt_builder import (
    build_prompt,
    build_repair_prompt,
    legal_actions_from_feedback,
)
from src.agent.response_parser import parse_response
from src.agent.runner import initial_belief
from src.env.generator import generate_map
from src.env.grid_env import GridWorldEnv


SYSTEM_PROMPT = (
    "You are a grid-world planning agent.\n"
    "Return valid JSON only. No markdown.\n"
    "No extra text."
)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def action_is_legal(action: str, legal_actions: list[str]) -> bool:
    return action in set(legal_actions)


class VLLMChatModel:
    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_new_tokens: int,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 4096,
        max_num_seqs: int = 32,
        dtype: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)

        print(f"[vLLM] Loading tokenizer: {model_name}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        print(f"[vLLM] Loading model: {model_name}", flush=True)
        print(
            "[vLLM] "
            f"tensor_parallel_size={tensor_parallel_size}, "
            f"gpu_memory_utilization={gpu_memory_utilization}, "
            f"max_model_len={max_model_len}, "
            f"max_num_seqs={max_num_seqs}, "
            f"dtype={dtype}",
            flush=True,
        )

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=int(tensor_parallel_size),
            gpu_memory_utilization=float(gpu_memory_utilization),
            max_model_len=int(max_model_len),
            max_num_seqs=int(max_num_seqs),
            dtype=dtype,
            trust_remote_code=True,
        )

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )

        print("[vLLM] Model loaded.", flush=True)

    def to_chat_prompt(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate_batch(self, prompts: list[str], batch_size: int) -> list[str]:
        if not prompts:
            return []

        responses: list[str] = []

        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            chat_prompts = [self.to_chat_prompt(p) for p in chunk]

            outputs = self.llm.generate(
                chat_prompts,
                self.sampling_params,
                use_tqdm=False,
            )

            for out in outputs:
                if not out.outputs:
                    responses.append("")
                else:
                    responses.append(out.outputs[0].text.strip())

        return responses


def make_episode_state(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    generated = generate_map(
        grid_size=int(cfg["grid_size"]),
        num_obstacles=int(cfg["num_obstacles"]),
        seed=seed,
        require_unique_shortest_path=bool(
            cfg.get("require_unique_shortest_path", False)
        ),
    )

    env = GridWorldEnv(
        obstacle_map=generated.obstacle_map,
        start=generated.start,
        goal=generated.goal,
        max_steps=int(cfg["max_steps"]),
        feedback_type=str(cfg["feedback_type"]),
    )

    obs, info = env.reset()
    episode_id = f"A_seed{seed}"

    return {
        "episode_id": episode_id,
        "seed": seed,
        "env": env,
        "obs": obs,
        "belief": initial_belief(env.size, env.start, env.goal),
        "last_feedback": info["feedback"],
        "records": [],
        "parse_error_count": 0,
        "illegal_action_count": 0,
        "repair_used_count": 0,
        "done": False,
        "truncated": False,
    }


def finalize_episode(state: dict[str, Any]) -> dict[str, Any]:
    env = state["env"]
    success = env.agent_pos == env.goal

    return {
        "episode_id": state["episode_id"],
        "seed": state["seed"],
        "coordinate_system": "cartesian_bottom_left",
        "success": success,
        "steps_used": env.step_count,
        "shortest_path_len": env.shortest_path_len,
        "optimality_gap": None
        if not success
        else env.step_count - env.shortest_path_len,
        "parse_error_count": state["parse_error_count"],
        "illegal_action_count": state["illegal_action_count"],
        "repair_used_count": state["repair_used_count"],
        "final_pos": [env.agent_pos[0], env.agent_pos[1]],
        "goal_pos": [env.goal[0], env.goal[1]],
        "true_map": env.obstacle_map,
        "true_map_indexing": "obstacle_map[y][x], y=0 is bottom row",
        "trajectory": env.trajectory,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-episodes", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=None)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--tensor-parallel-size", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    if str(cfg.get("feedback_type")) != "adjacent_exact":
        raise ValueError("This script only supports Strategy A: adjacent_exact.")

    if args.num_episodes is not None:
        cfg["num_episodes"] = int(args.num_episodes)

    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir

    if args.max_num_seqs is not None:
        cfg["max_num_seqs"] = int(args.max_num_seqs)

    if args.gpu_memory_utilization is not None:
        cfg["gpu_memory_utilization"] = float(args.gpu_memory_utilization)

    if args.max_model_len is not None:
        cfg["max_model_len"] = int(args.max_model_len)

    if args.tensor_parallel_size is not None:
        cfg["tensor_parallel_size"] = int(args.tensor_parallel_size)

    batch_size = int(args.batch_size or cfg.get("batch_size", 32))

    model = VLLMChatModel(
        model_name=str(cfg["model_name"]),
        temperature=float(cfg.get("temperature", 0.0)),
        max_new_tokens=int(cfg.get("max_new_tokens", 256)),
        tensor_parallel_size=int(cfg.get("tensor_parallel_size", 1)),
        gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.90)),
        max_model_len=int(cfg.get("max_model_len", 4096)),
        max_num_seqs=int(cfg.get("max_num_seqs", batch_size)),
        dtype=str(cfg.get("dtype", "auto")),
    )

    base_seed = int(cfg["seed"])
    num_episodes = int(cfg.get("num_episodes", 1))

    states = [
        make_episode_state(cfg, seed=base_seed + ep_idx)
        for ep_idx in range(num_episodes)
    ]

    all_steps: list[dict[str, Any]] = []

    print(
        f"[vLLM Strategy A] episodes={num_episodes}, "
        f"batch_size={batch_size}, "
        f"max_steps={cfg['max_steps']}",
        flush=True,
    )

    round_idx = 0

    while True:
        active = [
            s
            for s in states
            if not (s["done"] or s["truncated"])
            and int(s["parse_error_count"]) < 3
        ]

        if not active:
            break

        round_idx += 1
        print(
            f"[vLLM Strategy A] round={round_idx}, active={len(active)}",
            flush=True,
        )

        prompts: list[str] = []
        legal_actions_list: list[list[str]] = []

        for s in active:
            env = s["env"]
            obs = s["obs"]

            prompt = build_prompt(
                grid_size=env.size,
                start=[env.start[0], env.start[1]],
                goal=[env.goal[0], env.goal[1]],
                current_pos=obs["current_pos"],
                last_feedback=s["last_feedback"],
                belief_grid=s["belief"],
                history=s["records"],
            )

            legal_actions = legal_actions_from_feedback(
                obs["current_pos"],
                s["last_feedback"],
            )

            prompts.append(prompt)
            legal_actions_list.append(legal_actions)

        raw_responses = model.generate_batch(
            prompts,
            batch_size=batch_size,
        )

        parsed_list = [
            parse_response(raw, active[i]["env"].size)
            for i, raw in enumerate(raw_responses)
        ]

        repair_indices: list[int] = []
        repair_prompts: list[str] = []

        raw_response_before_repair: list[str | None] = [
            None for _ in active
        ]
        repaired_flags = [False for _ in active]
        illegal_action_before_repair = [False for _ in active]
        parse_error_before_repair = [
            parsed.parse_error for parsed in parsed_list
        ]

        for i, parsed in enumerate(parsed_list):
            legal_actions = legal_actions_list[i]

            if (
                not parsed.parse_error
                and legal_actions
                and not action_is_legal(parsed.action, legal_actions)
            ):
                active[i]["illegal_action_count"] += 1
                illegal_action_before_repair[i] = True
                raw_response_before_repair[i] = raw_responses[i]

                repair_indices.append(i)
                repair_prompts.append(
                    build_repair_prompt(
                        original_prompt=prompts[i],
                        bad_response=raw_responses[i],
                        bad_action=parsed.action,
                        legal_actions=legal_actions,
                    )
                )

        if repair_prompts:
            repair_responses = model.generate_batch(
                repair_prompts,
                batch_size=batch_size,
            )

            for j, i in enumerate(repair_indices):
                repair_parsed = parse_response(
                    repair_responses[j],
                    active[i]["env"].size,
                )

                legal_actions = legal_actions_list[i]

                if (
                    not repair_parsed.parse_error
                    and (
                        not legal_actions
                        or action_is_legal(repair_parsed.action, legal_actions)
                    )
                ):
                    raw_responses[i] = repair_responses[j]
                    parsed_list[i] = repair_parsed
                    repaired_flags[i] = True
                    active[i]["repair_used_count"] += 1

        for i, s in enumerate(active):
            env = s["env"]
            obs_before = s["obs"]
            parsed = parsed_list[i]

            if parsed.parse_error:
                s["parse_error_count"] += 1
            else:
                s["belief"] = parsed.belief_grid

            obs, _reward, done, truncated, step_info = env.step(parsed.action)

            s["obs"] = obs
            s["last_feedback"] = step_info["feedback"]
            s["done"] = done
            s["truncated"] = truncated

            rec = {
                "episode_id": s["episode_id"],
                "step_id": len(s["records"]) + 1,
                "seed": s["seed"],
                "grid_size": env.size,
                "coordinate_system": "cartesian_bottom_left",
                "feedback_type": env.feedback_type,
                "prompt_text": prompts[i],
                "legal_actions": legal_actions_list[i],
                "raw_response_text": raw_responses[i],
                "raw_response_before_repair": raw_response_before_repair[i],
                "repaired": repaired_flags[i],
                "parse_error_before_repair": parse_error_before_repair[i],
                "illegal_action_before_repair": illegal_action_before_repair[i],
                "parse_error": parsed.parse_error,
                "parse_error_message": parsed.error_message,
                "parsed_thought": parsed.thought,
                "parsed_nl_obstacles": parsed.nl_obstacles,
                "parsed_belief_grid": parsed.belief_grid,
                "belief_grid_row_order": "top_to_bottom_y_descending",
                "parsed_action": parsed.action,
                "agent_pos_before": obs_before["current_pos"],
                "agent_pos_after": step_info["new_pos"],
                "valid_move": step_info["valid_move"],
                "invalid_reason": step_info["invalid_reason"],
                "env_feedback": s["last_feedback"],
                "done": done,
                "truncated": truncated,
            }

            s["records"].append(rec)
            all_steps.append(rec)

    summaries = [finalize_episode(s) for s in states]

    output_dir = Path(str(cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_dir / "steps.jsonl", all_steps)
    write_json(output_dir / "summary.json", summaries)

    success_count = sum(1 for x in summaries if x["success"])

    print("=" * 70)
    print(f"[vLLM Strategy A] Saved steps:   {output_dir / 'steps.jsonl'}")
    print(f"[vLLM Strategy A] Saved summary: {output_dir / 'summary.json'}")
    print(f"[vLLM Strategy A] success={success_count}/{len(summaries)}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
