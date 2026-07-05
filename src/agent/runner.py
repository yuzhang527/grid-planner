from __future__ import annotations

from typing import Any

from .prompt_builder import build_prompt
from .response_parser import parse_response


def initial_belief(grid_size: int, start: tuple[int, int], goal: tuple[int, int]) -> list[list[str]]:
    grid = [["U" for _ in range(grid_size)] for _ in range(grid_size)]
    grid[start[0]][start[1]] = "F"
    grid[goal[0]][goal[1]] = "F"
    return grid


def run_episode(env, model, *, episode_id: str, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    obs, info = env.reset()
    belief = initial_belief(env.size, env.start, env.goal)
    last_feedback = info["feedback"]
    records: list[dict[str, Any]] = []
    parse_error_count = 0

    done = False
    truncated = False
    while not (done or truncated):
        prompt = build_prompt(
            grid_size=env.size,
            start=[env.start[0], env.start[1]],
            goal=[env.goal[0], env.goal[1]],
            current_pos=obs["current_pos"],
            last_feedback=last_feedback,
            belief_grid=belief,
            history=records,
        )

        raw_response = model.generate(
            prompt,
            env=env,
            belief_grid=belief,
            last_feedback=last_feedback,
            current_pos=obs["current_pos"],
            goal=[env.goal[0], env.goal[1]],
        )
        parsed = parse_response(raw_response, env.size)
        if parsed.parse_error:
            parse_error_count += 1
        else:
            belief = parsed.belief_grid

        obs_before = obs
        obs, _reward, done, truncated, step_info = env.step(parsed.action)
        last_feedback = step_info["feedback"]

        rec = {
            "episode_id": episode_id,
            "step_id": len(records) + 1,
            "seed": seed,
            "grid_size": env.size,
            "feedback_type": env.feedback_type,
            "prompt_text": prompt,
            "raw_response_text": raw_response,
            "parse_error": parsed.parse_error,
            "parse_error_message": parsed.error_message,
            "parsed_thought": parsed.thought,
            "parsed_nl_obstacles": parsed.nl_obstacles,
            "parsed_belief_grid": parsed.belief_grid,
            "parsed_action": parsed.action,
            "agent_pos_before": obs_before["current_pos"],
            "agent_pos_after": step_info["new_pos"],
            "valid_move": step_info["valid_move"],
            "invalid_reason": step_info["invalid_reason"],
            "env_feedback": last_feedback,
            "done": done,
            "truncated": truncated,
        }
        records.append(rec)

        if parse_error_count >= 3:
            break

    success = env.agent_pos == env.goal
    summary = {
        "episode_id": episode_id,
        "seed": seed,
        "success": success,
        "steps_used": env.step_count,
        "shortest_path_len": env.shortest_path_len,
        "optimality_gap": None if not success else env.step_count - env.shortest_path_len,
        "parse_error_count": parse_error_count,
        "final_pos": [env.agent_pos[0], env.agent_pos[1]],
        "goal_pos": [env.goal[0], env.goal[1]],
        "true_map": env.obstacle_map,
        "trajectory": env.trajectory,
    }
    return records, summary
