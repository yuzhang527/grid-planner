from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def action_is_legal(action: str, legal_actions: list[str]) -> bool:
    return action in set(legal_actions)


class BatchedHFModel:
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
    ) -> None:
        self.model_name = model_name
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)

        print(f"[BatchedHFModel] Loading tokenizer: {model_name}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Important for decoder-only batched generation.
        self.tokenizer.padding_side = "left"

        print(f"[BatchedHFModel] Loading model: {model_name}", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print("[BatchedHFModel] Model loaded.", flush=True)

    def _to_chat_text(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate_batch(self, prompts: list[str], batch_size: int) -> list[str]:
        outputs_all: list[str] = []

        do_sample = self.temperature > 0.0

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = self.temperature

        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            texts = [self._to_chat_text(p) for p in chunk]

            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            ).to(self.model.device)

            input_len = inputs["input_ids"].shape[1]

            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    **gen_kwargs,
                )

            for i in range(generated.shape[0]):
                new_tokens = generated[i, input_len:]
                response = self.tokenizer.decode(
                    new_tokens,
                    skip_special_tokens=True,
                ).strip()
                outputs_all.append(response)

        return outputs_all


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
        "optimality_gap": None if not success else env.step_count - env.shortest_path_len,
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
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    if str(cfg.get("backend", "hf")) != "hf":
        raise ValueError("This batched runner is only for HF backend.")

    if str(cfg["feedback_type"]) != "adjacent_exact":
        raise ValueError("This batched runner only supports Strategy A: adjacent_exact.")

    batch_size = int(args.batch_size or cfg.get("batch_size", 8))

    model = BatchedHFModel(
        model_name=str(cfg["model_name"]),
        temperature=float(cfg.get("temperature", 0.0)),
        max_new_tokens=int(cfg.get("max_new_tokens", 256)),
    )

    base_seed = int(cfg["seed"])
    num_episodes = int(cfg.get("num_episodes", 1))

    states = [
        make_episode_state(cfg, seed=base_seed + ep_idx)
        for ep_idx in range(num_episodes)
    ]

    all_steps: list[dict[str, Any]] = []

    print(
        f"[batched_A] episodes={num_episodes}, batch_size={batch_size}, max_steps={cfg['max_steps']}",
        flush=True,
    )

    round_idx = 0

    while True:
        active = [
            s for s in states
            if not (s["done"] or s["truncated"]) and s["parse_error_count"] < 3
        ]

        if not active:
            break

        round_idx += 1
        print(
            f"[batched_A] round={round_idx}, active_episodes={len(active)}",
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

        raw_responses = model.generate_batch(prompts, batch_size=batch_size)
        parsed_list = [
            parse_response(raw, active[i]["env"].size)
            for i, raw in enumerate(raw_responses)
        ]

        # Batched repair for illegal but parseable actions.
        repair_indices: list[int] = []
        repair_prompts: list[str] = []

        for i, parsed in enumerate(parsed_list):
            legal_actions = legal_actions_list[i]
            if (
                not parsed.parse_error
                and legal_actions
                and not action_is_legal(parsed.action, legal_actions)
            ):
                active[i]["illegal_action_count"] += 1
                repair_indices.append(i)
                repair_prompts.append(
                    build_repair_prompt(
                        original_prompt=prompts[i],
                        bad_response=raw_responses[i],
                        bad_action=parsed.action,
                        legal_actions=legal_actions,
                    )
                )

        repaired_flags = [False for _ in active]
        raw_response_before_repair = [None for _ in active]
        illegal_action_before_repair = [False for _ in active]
        parse_error_before_repair = [p.parse_error for p in parsed_list]

        if repair_prompts:
            repair_responses = model.generate_batch(
                repair_prompts,
                batch_size=batch_size,
            )

            for j, i in enumerate(repair_indices):
                repaired_parsed = parse_response(
                    repair_responses[j],
                    active[i]["env"].size,
                )

                legal_actions = legal_actions_list[i]

                if (
                    not repaired_parsed.parse_error
                    and (
                        not legal_actions
                        or action_is_legal(repaired_parsed.action, legal_actions)
                    )
                ):
                    raw_response_before_repair[i] = raw_responses[i]
                    raw_responses[i] = repair_responses[j]
                    parsed_list[i] = repaired_parsed
                    repaired_flags[i] = True
                    illegal_action_before_repair[i] = True
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

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_dir / "steps.jsonl", all_steps)
    write_json(output_dir / "summary.json", summaries)

    success_count = sum(1 for x in summaries if x["success"])

    print("=" * 60)
    print(f"[batched_A] Saved step logs to: {output_dir / 'steps.jsonl'}")
    print(f"[batched_A] Saved summaries to: {output_dir / 'summary.json'}")
    print(f"[batched_A] success={success_count}/{len(summaries)}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
