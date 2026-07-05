from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .astar import ACTIONS, in_bounds, shortest_path
from .feedback import adjacent_exact_feedback

Coord = tuple[int, int]


@dataclass
class GridWorldEnv:
    obstacle_map: list[list[int]]
    start: Coord
    goal: Coord
    max_steps: int = 20
    feedback_type: str = "adjacent_exact"
    agent_pos: Coord = field(init=False)
    step_count: int = field(default=0, init=False)
    trajectory: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.size = len(self.obstacle_map)
        path = shortest_path(self.obstacle_map, self.start, self.goal)
        if path is None:
            raise ValueError("Map is not reachable from start to goal.")
        self.shortest_path_len = len(path) - 1
        self.reset()

    def reset(self) -> tuple[dict, dict]:
        self.agent_pos = self.start
        self.step_count = 0
        self.trajectory = []
        obs = self.get_public_obs()
        info = {"feedback": self._feedback(), "shortest_path_len": self.shortest_path_len}
        return obs, info

    def get_public_obs(self) -> dict:
        return {
            "grid_size": self.size,
            "current_pos": [self.agent_pos[0], self.agent_pos[1]],
            "goal_pos": [self.goal[0], self.goal[1]],
            "step_count": self.step_count,
            "max_steps": self.max_steps,
        }

    def _feedback(self) -> dict:
        if self.feedback_type != "adjacent_exact":
            raise NotImplementedError("This smoke project only implements Strategy A: adjacent_exact.")
        return adjacent_exact_feedback(self.obstacle_map, self.agent_pos)

    def step(self, action: str) -> tuple[dict, float, bool, bool, dict]:
        old_pos = self.agent_pos
        action = action.upper().strip()
        invalid_reason = None

        if action not in ACTIONS:
            new_pos = old_pos
            invalid_reason = "unknown_action"
        else:
            dr, dc = ACTIONS[action]
            cand = (old_pos[0] + dr, old_pos[1] + dc)
            if not in_bounds(cand, self.size):
                new_pos = old_pos
                invalid_reason = "wall"
            elif self.obstacle_map[cand[0]][cand[1]] == 1:
                new_pos = old_pos
                invalid_reason = "obstacle"
            else:
                new_pos = cand

        self.agent_pos = new_pos
        self.step_count += 1
        done = self.agent_pos == self.goal
        truncated = self.step_count >= self.max_steps and not done
        feedback = self._feedback()

        transition = {
            "step_count": self.step_count,
            "action": action,
            "old_pos": [old_pos[0], old_pos[1]],
            "new_pos": [new_pos[0], new_pos[1]],
            "valid_move": invalid_reason is None,
            "invalid_reason": invalid_reason,
            "feedback": feedback,
        }
        self.trajectory.append(transition)

        info = {
            "feedback": feedback,
            "old_pos": [old_pos[0], old_pos[1]],
            "new_pos": [new_pos[0], new_pos[1]],
            "valid_move": invalid_reason is None,
            "invalid_reason": invalid_reason,
            "shortest_path_len": self.shortest_path_len,
        }
        return self.get_public_obs(), 0.0, done, truncated, info
