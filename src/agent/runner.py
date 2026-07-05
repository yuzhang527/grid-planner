from __future__ import annotations

from typing import Any

from .prompt_builder import build_prompt, build_repair_prompt, legal_actions_from_feedback
from .response_parser import parse_response


def _grid_index_from_coord(size: int, coord: tuple[int, int] | list[int]) -> tuple[int, int]:
    """Convert Cartesian [x, y] to display-grid index.

    belief_grid rows are top-to-bottom:
    row 0 = y=size-1, last row = y=0.
    """
    x, y = int(coord[0]), int(coord[1])
    return size - 1 - y, x


def initial_belief(grid_size: int, start: tuple[int, int], goal: tuple[int, int]) -> list[list[str]]:
    grid = [["U" for _ in range(grid_size)] for _ in range(grid_size)]

    sr, sc = _grid_index_from_coord(grid_size, start)
    gr, gc = _grid_index_from_coord(grid_size, goal)

    grid[sr][sc] = "F"
    grid[gr][gc] = "F"

    return grid


def _action_is_legal(action: str, legal_actions: list[str]) -> bool:
    return action in set(legal_actions)


def run_episode(env, model, *, episode_id: str, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    obs, info = env.reset()
    belief = initial_belief(env.size, env.start, env.goal)
    last_feedback = info["feedback"]
    records: list[dict[str, Any]] = []

    parse_error_count = 0
    illegal_action_count = 0
    repair_used_count = 0

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

        legal_actions = legal_actions_from_feedback(obs["current_pos"], last_feedback)

        raw_response = model.generate(
            prompt,
            env=env,
            belief_grid=belief,
            last_feedback=last_feedback,
            current_pos=obs["current_pos"],
            goal=[env.goal[0], env.goal[1]],
        )

        parsed = parse_response(raw_response, env.size)

        repaired = False
        raw_response_before_repair = None
        parse_error_before_repair = parsed.parse_error
        illegal_action_before_repair = False

        if (not parsed.parse_error) and legal_actions and (not _action_is_legal(parsed.action, legal_actions)):
            illegal_action_count += 1
            illegal_action_before_repair = True
            raw_response_before_repair = raw_response

            repair_prompt = build_repair_prompt(
                original_prompt=prompt,
                bad_response=raw_response,
                bad_action=parsed.action,
                legal_actions=legal_actions,
            )

            repair_response = model.generate(
                repair_prompt,
                env=env,
                belief_grid=belief,
                last_feedback=last_feedback,
                current_pos=obs["current_pos"],
                goal=[env.goal[0], env.goal[1]],
            )

            repair_parsed = parse_response(repair_response, env.size)

            if (not repair_parsed.parse_error) and (
                not legal_actions or _action_is_legal(repair_parsed.action, legal_actions)
            ):
                raw_response = repair_response
                parsed = repair_parsed
                repaired = True
                repair_used_count += 1

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
            "coordinate_system": "cartesian_bottom_left",
            "feedback_type": env.feedback_type,
            "prompt_text": prompt,
            "legal_actions": legal_actions,
            "raw_response_text": raw_response,
            "raw_response_before_repair": raw_response_before_repair,
            "repaired": repaired,
            "parse_error_before_repair": parse_error_before_repair,
            "illegal_action_before_repair": illegal_action_before_repair,
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
        "coordinate_system": "cartesian_bottom_left",
        "success": success,
        "steps_used": env.step_count,
        "shortest_path_len": env.shortest_path_len,
        "optimality_gap": None if not success else env.step_count - env.shortest_path_len,
        "parse_error_count": parse_error_count,
        "illegal_action_count": illegal_action_count,
        "repair_used_count": repair_used_count,
        "final_pos": [env.agent_pos[0], env.agent_pos[1]],
        "goal_pos": [env.goal[0], env.goal[1]],
        "true_map": env.obstacle_map,
        "true_map_indexing": "obstacle_map[y][x], y=0 is bottom row",
        "trajectory": env.trajectory,
    }

    return records, summary
