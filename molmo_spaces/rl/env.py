"""Gymnasium macro environment for RBY1 high-level reinforcement learning."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from molmo_spaces.policy.learned_policy.rby1_rl_policy import RBY1RLPolicy
from molmo_spaces.rl.action_adapter import RBY1ActionAdapter
from molmo_spaces.rl.observation import RBY1ObservationBuilder
from molmo_spaces.rl.reward import RewardSnapshot, RewardTracker
from molmo_spaces.tasks.gym_env import GymEnv


class RBY1RLEnv(gym.Env):
    """Two-stage PICK/PLACE MDP backed by the ordinary MolmoSpaces task."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        exp_config=None,
        *,
        house_indices: list[int] | None = None,
        max_macro_steps: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        if exp_config is None:
            from molmo_spaces.evaluation.configs.evaluation_configs import RBY1RLEvalConfig

            exp_config = RBY1RLEvalConfig()
        exp_config.data_split = "train"
        exp_config.use_wandb = False
        exp_config.use_passive_viewer = False
        exp_config.num_envs = 1
        exp_config.task_sampler_config.task_batch_size = 1
        exp_config.task_sampler_config.house_inds = list(house_indices or [0, 1, 2, 3])
        if max_macro_steps is not None:
            exp_config.policy_config.rl_max_macro_steps = int(max_macro_steps)
        exp_config.policy_config.checkpoint_path = None

        self.exp_config = exp_config
        self.inner = GymEnv(exp_config=exp_config, render_mode=render_mode)
        self.render_mode = render_mode
        self.observation_builder = RBY1ObservationBuilder(
            max_candidates=exp_config.policy_config.rl_max_grasp_candidates,
            position_scale_m=exp_config.policy_config.rl_position_scale_m,
        )
        self.observation_space = self.observation_builder.observation_space
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, (RBY1ActionAdapter.action_dim,), dtype=np.float32
        )
        self.reward_tracker = RewardTracker()
        self.policy: RBY1RLPolicy | None = None
        self._raw_observation = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        raw_observation, raw_info = self.inner.reset(seed=seed, options=options)
        task = self.inner.task
        assert task is not None
        self.policy = RBY1RLPolicy(self.exp_config, task, load_checkpoint=False)
        task.register_policy(self.policy)
        self.policy.reset()
        self.reward_tracker.reset()
        self._raw_observation = raw_observation
        observation = self.policy.build_rl_observation(raw_observation)
        return observation, {"inner_info": raw_info[0], "macro_phase": "pick"}

    def step(self, action: np.ndarray):
        if self.policy is None or self.inner.task is None:
            raise RuntimeError("Call reset() before step().")
        if not self.policy.at_macro_boundary:
            raise RuntimeError("RBY1RLEnv.step called while the previous macro is still running")

        phase = self.policy.macro_phase
        before = self._snapshot()
        contacts_before = self._severe_robot_contacts()
        new_collision_contacts: set[tuple[int, int]] = set()
        self.policy.set_external_action(action)
        low_level_steps = 0
        success = False
        terminated_inner = False
        truncated_inner = False
        object_lost_during_macro = False
        info = self.inner.task.get_info()[0]

        while low_level_steps < self.exp_config.task_horizon:
            low_action = self.policy.get_action(self._raw_observation)
            self._raw_observation, _reward, terminal, truncated, infos = self.inner.step(
                low_action
            )
            low_level_steps += 1
            info = infos[0]
            new_collision_contacts.update(self._severe_robot_contacts() - contacts_before)
            success = bool(info.get("success", False))
            object_lost_during_macro = bool(
                phase == "place" and before.held and not success and not self._is_holding()
            )
            terminated_inner = bool(np.asarray(terminal).reshape(-1)[0])
            truncated_inner = bool(np.asarray(truncated).reshape(-1)[0])
            if (
                success
                or terminated_inner
                or truncated_inner
                or new_collision_contacts
                or object_lost_during_macro
                or self.policy.at_macro_boundary
            ):
                break

        after = self._snapshot()
        collision = bool(new_collision_contacts)
        decoded = self.policy.last_decoded_action
        invalid = self.policy.last_failure == "invalid_action"
        planning_failure = self.policy.last_failure == "planning_failure"
        supported_and_released = bool(
            info.get("supported_by_receptacle", False) and not info.get("robot_contact", True)
        )
        reward, reward_terms = self.reward_tracker.compute(
            phase,
            before,
            after,
            success=success,
            supported_and_released=supported_and_released,
            invalid_action=invalid,
            planning_failure=planning_failure,
            collision=collision,
            translation_fraction=(
                decoded.translation_m / self.policy_config.rl_max_base_translation_m
                if decoded is not None
                else 0.0
            ),
            yaw_fraction=(
                abs(decoded.yaw_delta_rad) / self.policy_config.rl_max_base_yaw_delta_rad
                if decoded is not None
                else 0.0
            ),
        )

        object_lost = object_lost_during_macro or (
            phase == "place" and before.held and not after.held and not success
        )
        macro_limit = self.policy.macro_steps >= self.policy_config.rl_max_macro_steps
        terminated = bool(success or terminated_inner or object_lost or collision or macro_limit)
        truncated = bool(truncated_inner or low_level_steps >= self.exp_config.task_horizon)
        observation = self.policy.build_rl_observation(self._raw_observation)
        result_info = {
            **info,
            "benchmark_success": success,
            "macro_phase": phase,
            "next_macro_phase": self.policy.macro_phase,
            "low_level_steps": low_level_steps,
            "reward_terms": reward_terms,
            "invalid_action": invalid,
            "planning_failure": planning_failure,
            "collision": collision,
            "selected_arm": self.policy.get_info()["rl_selected_arm"],
            "selected_candidate": self.policy.get_info()["rl_selected_candidate"],
        }
        return observation, reward, terminated, truncated, result_info

    @property
    def policy_config(self):
        return self.exp_config.policy_config

    def _snapshot(self) -> RewardSnapshot:
        assert self.inner.task is not None and self.policy is not None
        task = self.inner.task
        env = task.env
        om = env.object_managers[env.current_batch_index]
        pickup = om.get_object_by_name(task.config.task_config.pickup_obj_name)
        receptacle = om.get_object_by_name(task.config.task_config.place_receptacle_name)
        base_xy = env.current_robot.robot_view.base.pose[:2, 3]
        if self.policy.selected_arm is None:
            ee_positions = [
                env.current_robot.robot_view.get_move_group(name).leaf_frame_to_world[:3, 3]
                for name in ("left_gripper", "right_gripper")
            ]
            ee_distance = min(np.linalg.norm(pos - pickup.position) for pos in ee_positions)
        else:
            ee = env.current_robot.robot_view.get_move_group(
                f"{self.policy.selected_arm}_gripper"
            ).leaf_frame_to_world[:3, 3]
            ee_distance = np.linalg.norm(ee - pickup.position)
        held = self._is_holding()
        return RewardSnapshot(
            base_to_object=float(np.linalg.norm(base_xy - pickup.position[:2])),
            ee_to_object=float(ee_distance),
            object_to_receptacle=float(np.linalg.norm(pickup.position - receptacle.position)),
            object_z=float(pickup.position[2]),
            held=held,
        )

    def _is_holding(self) -> bool:
        if self.policy is None or self.policy.executor is None:
            return False
        try:
            return bool(self.policy.executor._grasping_something())
        except Exception:
            return False

    def _severe_robot_contacts(self) -> set[tuple[int, int]]:
        assert self.inner.task is not None
        env = self.inner.task.env
        data = env.current_data
        root = env.current_robot.robot_view.root_body_id
        object_manager = env.object_managers[env.current_batch_index]
        ignored_roots = {
            int(
                data.model.body_rootid[
                    object_manager.get_object_by_name(name).body_id
                ]
            )
            for name in (
                self.inner.task.config.task_config.pickup_obj_name,
                self.inner.task.config.task_config.place_receptacle_name,
            )
        }
        contacts: set[tuple[int, int]] = set()
        for contact in data.contact:
            if float(contact.dist) >= -0.001:
                continue
            body1 = int(data.model.body_rootid[data.model.geom_bodyid[contact.geom1]])
            body2 = int(data.model.body_rootid[data.model.geom_bodyid[contact.geom2]])
            other_body = body2 if body1 == root else body1
            if (body1 == root) ^ (body2 == root) and other_body not in ignored_roots:
                contacts.add(tuple(sorted((int(contact.geom1), int(contact.geom2)))))
        return contacts

    def render(self):
        return self.inner.render()

    def close(self) -> None:
        self.policy = None
        self.inner.close()
