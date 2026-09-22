"""SAC high-level policy with the existing RBY1 CuRobo executor underneath.

This module intentionally has no dependency on the LLM waypoint policy.  The
learned policy selects a base correction, grasp candidate, and arm; CuRobo owns
IK, motion generation, waypoint execution, and gripper phase transitions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from molmo_spaces.policy.base_policy import PlannerPolicy
from ..solvers.object_manipulation.base_object_manipulation_planner_policy import (
    BaseObjectManipulationPlannerPolicy,
    JointMoveSegment,
    JointMoveSequence,
)
from molmo_spaces.rl.action_adapter import DecodedRBY1Action, RBY1ActionAdapter
from molmo_spaces.rl.observation import (
    OBSERVATION_VERSION,
    GraspCandidateContext,
    RBY1ObservationBuilder,
    build_grasp_candidates,
)

log = logging.getLogger(__name__)


def validate_rl_checkpoint_metadata(
    checkpoint_path: str | Path,
    max_candidates: int,
    position_scale_m: float = 2.0,
    max_base_translation_m: float = 0.5,
    max_base_yaw_delta_rad: float = float(np.pi / 2),
) -> dict:
    checkpoint_path = Path(checkpoint_path)
    metadata_path = checkpoint_path.parent / "rl_metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"RL checkpoint metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "observation_version": OBSERVATION_VERSION,
        "action_dim": RBY1ActionAdapter.action_dim,
        "max_candidates": max_candidates,
        "position_scale_m": position_scale_m,
        "max_base_translation_m": max_base_translation_m,
        "max_base_yaw_delta_rad": max_base_yaw_delta_rad,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"RL checkpoint metadata is incompatible: {mismatches}")
    return metadata


class RBY1RLCuroboExecutor:
    """Factory for a forced-arm, single-candidate CuRobo pick/place policy.

    Kept as a factory so importing the RL observation/action utilities does not
    import CuRobo on CPU-only machines.
    """

    @staticmethod
    def create(config, task, arm: str, grasp_pose_world: np.ndarray):
        from ..solvers.object_manipulation.curobo_pick_and_place_planner_policy import (
            CuroboPickAndPlacePlannerPolicy,
        )

        class _ForcedExecutor(CuroboPickAndPlacePlannerPolicy):
            def __init__(self, cfg, bound_task):
                self._rl_arm = arm
                self._rl_grasp_pose_world = np.asarray(grasp_pose_world, dtype=float).copy()
                super().__init__(cfg, bound_task)

            def select_arm(self) -> None:
                selected_arm = self._rl_arm
                if selected_arm not in ("left", "right"):
                    raise ValueError(f"Unsupported RBY1 arm: {selected_arm}")
                self.arm_side = selected_arm
                self.arm_move_group_id = f"{selected_arm}_arm"
                self.gripper_move_group_id = f"{selected_arm}_gripper"
                if self._use_local_planner:
                    self._select_arm_local(selected_arm)
                else:
                    import random

                    from molmo_spaces.planner.curobo_planner_client import CuroboClient

                    self.client = CuroboClient(
                        base_url=random.choice(self.config.policy_config.server_urls),
                        arm=selected_arm,
                        timeout=self.config.policy_config.server_timeout,
                    )
                self.planner_joint_ranges = (
                    self.config.policy_config.left_planner_joint_ranges
                    if selected_arm == "left"
                    else self.config.policy_config.right_planner_joint_ranges
                )
                self.arm_start_idx, self.arm_end_idx = self.planner_joint_ranges[
                    self.arm_move_group_id
                ]

            def _get_pregrasp_poses(self) -> np.ndarray:
                pose = self._rl_grasp_pose_world.copy()
                pose[:3, 3] -= self.config.policy_config.pregrasp_z_offset * pose[:3, 2]
                return pose[None]

        return _ForcedExecutor(config, task)


class RBY1RLPolicy(PlannerPolicy):
    """BasePolicy-compatible bridge between a high-level SAC model and CuRobo."""

    def __init__(self, config, task=None, *, load_checkpoint: bool = True) -> None:
        super().__init__(config, task)
        self.policy_config = config.policy_config
        self.observation_builder = RBY1ObservationBuilder(
            max_candidates=self.policy_config.rl_max_grasp_candidates,
            position_scale_m=self.policy_config.rl_position_scale_m,
        )
        self.action_adapter = RBY1ActionAdapter(
            max_translation_m=self.policy_config.rl_max_base_translation_m,
            max_yaw_delta_rad=self.policy_config.rl_max_base_yaw_delta_rad,
        )
        self.model = None
        if load_checkpoint and self.policy_config.checkpoint_path:
            validate_rl_checkpoint_metadata(
                self.policy_config.checkpoint_path,
                self.policy_config.rl_max_grasp_candidates,
                self.policy_config.rl_position_scale_m,
                self.policy_config.rl_max_base_translation_m,
                self.policy_config.rl_max_base_yaw_delta_rad,
            )
            try:
                from stable_baselines3 import SAC
            except ImportError as exc:
                raise ImportError(
                    "RBY1RLPolicy requires the 'rl' extra: uv sync --extra rl"
                ) from exc
            self.model = SAC.load(
                self.policy_config.checkpoint_path, device=self.policy_config.rl_device
            )
        self._external_action: np.ndarray | None = None
        self.reset()

    @staticmethod
    def add_auxiliary_objects(config, spec) -> None:
        BaseObjectManipulationPlannerPolicy.add_auxiliary_objects(config, spec)

    @property
    def planners(self) -> dict:
        return {} if self.executor is None else self.executor.planners

    @property
    def retry_count(self) -> int:
        return self._retry_count

    @property
    def at_macro_boundary(self) -> bool:
        return self._at_macro_boundary

    @property
    def macro_phase(self) -> str:
        return self._macro_phase

    @property
    def last_failure(self) -> str:
        return self._last_failure

    @property
    def last_decoded_action(self) -> DecodedRBY1Action | None:
        return self._last_decoded_action

    @property
    def macro_steps(self) -> int:
        return self._macro_steps

    @property
    def selected_arm(self) -> str | None:
        return self._selected_arm

    @property
    def candidates(self) -> GraspCandidateContext:
        if self._candidates is None:
            self._candidates = build_grasp_candidates(
                self.task, self.policy_config.rl_max_grasp_candidates
            )
        return self._candidates

    def reset(self):
        self.executor = None
        self._base_move = None
        self._macro_phase = "pick"
        self._at_macro_boundary = True
        self._retry_count = 0
        self._macro_steps = 0
        self._last_failure = "none"
        self._last_decoded_action = None
        self._candidates: GraspCandidateContext | None = None
        self._selected_arm: str | None = None
        self._selected_candidate_index: int | None = None
        self._selected_grasp_pose: np.ndarray | None = None
        self._external_action = None
        self._irrecoverable_failure = False
        self._has_held_object = False
        self._initial_severe_contacts = self._severe_robot_contacts()

    def set_external_action(self, action: np.ndarray) -> None:
        if not self._at_macro_boundary:
            raise RuntimeError("External RL action may only be supplied at a macro boundary")
        self._external_action = np.asarray(action, dtype=np.float32).copy()

    def build_rl_observation(self, raw_observation=None) -> dict[str, np.ndarray]:
        if self._macro_phase == "place":
            candidates = GraspCandidateContext(
                poses_world=np.empty((0, 4, 4), dtype=float),
                collision_free=np.empty(0, dtype=bool),
                reachable_left=np.empty(0, dtype=bool),
                reachable_right=np.empty(0, dtype=bool),
            )
        else:
            candidates = self.candidates
        return self.observation_builder.build(
            self.task,
            candidates,
            self._macro_phase,
            self._retry_count,
            self._last_failure,
            raw_observation,
        )

    def get_action(self, observation):
        if self._irrecoverable_failure:
            return {"done": True, "success": False}
        if self._severe_robot_contacts() - self._initial_severe_contacts:
            return self._terminal_failure("collision")
        if self._macro_phase == "place" and self.executor is not None:
            try:
                holding = bool(self.executor._grasping_something())
            except (AttributeError, ValueError, RuntimeError):
                holding = False
            self._has_held_object = self._has_held_object or holding
            release_started = bool(getattr(self.executor, "opening_timesteps", 0))
            if self._has_held_object and not holding and not release_started:
                success = bool(self.task.get_info()[0].get("success", False))
                if not success:
                    return self._terminal_failure("grasp_failure")

        if self._macro_steps >= self.policy_config.rl_max_macro_steps:
            return {"done": True, "success": False}

        if self._at_macro_boundary:
            action = self._consume_high_level_action(observation)
            if action is None:
                return self.task.env.current_robot.robot_view.get_noop_ctrl_dict()
            self._start_macro(action)
            if self._at_macro_boundary:
                return self.task.env.current_robot.robot_view.get_noop_ctrl_dict()

        if self._base_move is not None:
            if not self._base_move.execute():
                return self._base_move.get_current_action()
            self._base_move = None
            try:
                self._start_or_resume_executor()
            except (ValueError, RuntimeError) as exc:
                return self._recoverable_failure("planning_failure", exc)
            return self.task.env.current_robot.robot_view.get_noop_ctrl_dict()

        if self.executor is None:
            try:
                self._start_or_resume_executor()
            except (ValueError, RuntimeError) as exc:
                return self._recoverable_failure("planning_failure", exc)

        # Stop after lift, before CuRobo begins PLACE, so the next env.step()
        # receives a fresh state and a new base correction.
        if self._macro_phase == "pick" and str(self.executor.get_phase()) == "place":
            self._macro_phase = "place"
            self._at_macro_boundary = True
            self._last_failure = "none"
            self._candidates = None
            return self.task.env.current_robot.robot_view.get_noop_ctrl_dict()

        try:
            action = self.executor.get_action(observation)
        except (ValueError, RuntimeError) as exc:
            return self._recoverable_failure("planning_failure", exc)

        if action.get("done", False):
            failure_reason = getattr(self.executor, "failure_reason", None)
            if failure_reason:
                action.pop("done", None)
                if self._macro_phase == "place":
                    # The object is still held, so a new base correction may
                    # make the same PLACE goal plannable on the next macro step.
                    phase_type = type(self.executor.current_phase)
                    self.executor.current_phase = phase_type.PLACE
                    self.executor.planned_trajectory = None
                return self._recoverable_failure("planning_failure", RuntimeError(failure_reason))
        return action

    def _consume_high_level_action(self, observation) -> np.ndarray | None:
        if self._external_action is not None:
            action = self._external_action
            self._external_action = None
            return action
        if self.model is None:
            raise RuntimeError("No SAC checkpoint loaded and no external action supplied")
        rl_observation = self.build_rl_observation(observation)
        action, _ = self.model.predict(
            rl_observation, deterministic=self.policy_config.rl_deterministic
        )
        return np.asarray(action, dtype=np.float32)

    def _start_macro(self, action: np.ndarray) -> None:
        robot_view = self.task.env.current_robot.robot_view
        decoded = self.action_adapter.decode(
            action,
            robot_view.base.pose.copy(),
            len(self.candidates) if self._macro_phase == "pick" else 0,
            self._macro_phase,
            self.task.env,
        )
        self._last_decoded_action = decoded
        self._macro_steps += 1
        self._at_macro_boundary = False
        if not decoded.valid:
            self._retry_count += 1
            self._last_failure = "invalid_action"
            self._at_macro_boundary = True
            return

        if self._macro_phase == "pick":
            assert decoded.candidate_index is not None and decoded.arm is not None
            self._selected_arm = decoded.arm
            self._selected_candidate_index = decoded.candidate_index
            self._selected_grasp_pose = self.candidates.poses_world[decoded.candidate_index].copy()

        base = robot_view.get_move_group("base")
        start = np.asarray(base.joint_pos, dtype=float).copy()
        goal = start.copy()
        goal[:3] = decoded.base_goal
        duration = max(
            0.25,
            decoded.translation_m / 0.20,
            abs(decoded.yaw_delta_rad) / 0.50,
        )
        self._base_move = JointMoveSequence(
            robot_view,
            settle_time=0.0,
            move_segments=[JointMoveSegment("rl_base_correction", None, {"base": goal}, duration)],
            is_holding_object=self._macro_phase == "place",
            gripper_move_group_id=(
                f"{self._selected_arm}_gripper" if self._selected_arm is not None else None
            ),
        )

    def _start_or_resume_executor(self) -> None:
        if self.executor is None:
            if self._selected_arm is None or self._selected_grasp_pose is None:
                raise RuntimeError("PICK action did not select an arm and grasp")
            self.executor = RBY1RLCuroboExecutor.create(
                self.config, self.task, self._selected_arm, self._selected_grasp_pose
            )

    def _recoverable_failure(self, kind: str, exc: BaseException) -> dict[str, Any]:
        log.warning("RBY1 RL macro failed (%s): %s", kind, exc)
        self._retry_count += 1
        self._last_failure = kind
        self._at_macro_boundary = True
        if self._macro_phase == "pick":
            self.executor = None
            self._candidates = None
        return self.task.env.current_robot.robot_view.get_noop_ctrl_dict()

    def _terminal_failure(self, kind: str) -> dict[str, Any]:
        self._last_failure = kind
        self._irrecoverable_failure = True
        return {"done": True, "success": False}

    def _severe_robot_contacts(self) -> set[tuple[int, int]]:
        if self.task is None:
            return set()
        env = self.task.env
        data = env.current_data
        robot_root = env.current_robot.robot_view.root_body_id
        object_manager = env.object_managers[env.current_batch_index]
        try:
            ignored_roots = {
                int(data.model.body_rootid[object_manager.get_object_by_name(name).body_id])
                for name in (
                    self.task.config.task_config.pickup_obj_name,
                    self.task.config.task_config.place_receptacle_name,
                )
            }
        except (AttributeError, KeyError, ValueError):
            return set()
        contacts = set()
        for contact in data.contact:
            if float(contact.dist) >= -0.001:
                continue
            body1 = int(data.model.body_rootid[data.model.geom_bodyid[contact.geom1]])
            body2 = int(data.model.body_rootid[data.model.geom_bodyid[contact.geom2]])
            if not ((body1 == robot_root) ^ (body2 == robot_root)):
                continue
            other_body = body2 if body1 == robot_root else body1
            if other_body not in ignored_roots:
                contacts.add(tuple(sorted((int(contact.geom1), int(contact.geom2)))))
        return contacts

    def get_phase(self) -> str:
        if self._at_macro_boundary:
            return f"rl_{self._macro_phase}_decision"
        if self._base_move is not None:
            return f"rl_{self._macro_phase}_base"
        if self.executor is not None:
            return str(self.executor.get_phase())
        return "unknown"

    def get_all_phases(self) -> dict[str, int]:
        names = [
            "unknown",
            "rl_pick_decision",
            "rl_pick_base",
            "pregrasp",
            "grasp",
            "lift",
            "rl_place_decision",
            "rl_place_base",
            "place",
            "postplace",
            "done",
        ]
        return {name: idx for idx, name in enumerate(names)}

    def get_info(self) -> dict:
        return {
            "rl_macro_phase": self._macro_phase,
            "rl_macro_steps": self._macro_steps,
            "rl_retry_count": self._retry_count,
            "rl_last_failure": self._last_failure,
            "rl_selected_arm": self._selected_arm,
            "rl_selected_candidate": self._selected_candidate_index,
        }
