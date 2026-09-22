"""Privileged, fixed-shape observations for the first RBY1 RL baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium.spaces as spaces
import numpy as np

from molmo_spaces.env.data_views import create_mlspaces_body
from molmo_spaces.utils.grasps import get_pickup_grasps
from molmo_spaces.utils.pose import pose_mat_to_7d

OBSERVATION_VERSION = 1
FAILURE_TYPES = ("none", "invalid_action", "planning_failure", "grasp_failure", "collision")


@dataclass(frozen=True)
class GraspCandidateContext:
    poses_world: np.ndarray
    collision_free: np.ndarray
    reachable_left: np.ndarray
    reachable_right: np.ndarray

    def __len__(self) -> int:
        return int(len(self.poses_world))


def _canonical_pose(pose: np.ndarray) -> np.ndarray:
    value = np.asarray(pose_mat_to_7d(pose), dtype=np.float32)
    if value[3] < 0:
        value[3:] *= -1
    return value


def build_grasp_candidates(task, max_candidates: int = 24) -> GraspCandidateContext:
    config = task.config
    om = task.env.object_managers[task.env.current_batch_index]
    pickup = om.get_object_by_name(config.task_config.pickup_obj_name)
    poses = np.asarray(
        get_pickup_grasps(task.env, pickup, grasp_libraries=config.policy_config.grasp_libraries),
        dtype=float,
    )
    if len(poses) == 0:
        return GraspCandidateContext(
            np.empty((0, 4, 4)), np.empty(0, bool), np.empty(0, bool), np.empty(0, bool)
        )

    robot_view = task.env.current_robot.robot_view
    left = robot_view.get_move_group("left_gripper").leaf_frame_to_world
    right = robot_view.get_move_group("right_gripper").leaf_frame_to_world
    left_dist = np.linalg.norm(poses[:, :3, 3] - left[:3, 3], axis=1)
    right_dist = np.linalg.norm(poses[:, :3, 3] - right[:3, 3], axis=1)
    order = np.argsort(np.minimum(left_dist, right_dist), kind="stable")[:max_candidates]
    poses = poses[order]
    left_dist = left_dist[order]
    right_dist = right_dist[order]

    collision_free = np.ones(len(poses), dtype=bool)
    try:
        from molmo_spaces.utils.grasp_sample import get_noncolliding_grasp_mask

        collision_free = np.asarray(
            get_noncolliding_grasp_mask(
                task.env.current_model,
                task.env.current_data,
                poses,
                config.policy_config.grasp_collision_batch_size,
            ),
            dtype=bool,
        )
    except Exception:
        # The collision bodies are optional outside the full task sampler.  Keep
        # the feature conservative but usable in unit/smoke environments.
        collision_free = np.ones(len(poses), dtype=bool)

    # Cheap, deterministic reach envelope. CuRobo remains the authority: these
    # are observations, not a gate that can reject a valid plan.
    reach_m = 1.25
    return GraspCandidateContext(
        poses_world=poses,
        collision_free=collision_free,
        reachable_left=left_dist <= reach_m,
        reachable_right=right_dist <= reach_m,
    )


class RBY1ObservationBuilder:
    proprio_dim = 41
    task_dim = 21
    candidate_dim = 13
    phase_context_dim = 2 + 1 + len(FAILURE_TYPES)

    def __init__(self, max_candidates: int = 24, position_scale_m: float = 2.0):
        self.max_candidates = int(max_candidates)
        self.position_scale_m = float(position_scale_m)
        self.observation_space = spaces.Dict(
            {
                "proprio": spaces.Box(-1.0, 1.0, (self.proprio_dim,), np.float32),
                "task": spaces.Box(-1.0, 1.0, (self.task_dim,), np.float32),
                "candidates": spaces.Box(
                    -1.0, 1.0, (self.max_candidates, self.candidate_dim), np.float32
                ),
                "candidate_mask": spaces.Box(
                    0.0, 1.0, (self.max_candidates,), np.float32
                ),
                "phase_context": spaces.Box(
                    -1.0, 1.0, (self.phase_context_dim,), np.float32
                ),
            }
        )

    def build(
        self,
        task,
        candidates: GraspCandidateContext,
        phase: str,
        retry_count: int,
        last_failure: str = "none",
        raw_observation: Any = None,
    ) -> dict[str, np.ndarray]:
        robot_view = task.env.current_robot.robot_view
        base_pose = robot_view.base.pose.copy()
        base_inv = np.linalg.inv(base_pose)

        base_7d = _canonical_pose(base_pose)
        base_7d[:3] = np.clip(base_7d[:3] / self.position_scale_m, -1, 1)
        qpos = robot_view.get_qpos_dict()
        joints = np.concatenate(
            [
                self._normalized_joint(robot_view, qpos, "torso", 6),
                self._normalized_joint(robot_view, qpos, "left_arm", 7),
                self._normalized_joint(robot_view, qpos, "right_arm", 7),
            ]
        )
        ee_values = []
        for gripper in ("left_gripper", "right_gripper"):
            rel = base_inv @ robot_view.get_move_group(gripper).leaf_frame_to_world
            pose = _canonical_pose(rel)
            pose[:3] = np.clip(pose[:3] / self.position_scale_m, -1, 1)
            ee_values.append(pose)
        proprio = np.concatenate([base_7d, joints, *ee_values]).astype(np.float32)

        pickup = create_mlspaces_body(
            task.env.current_data, task.config.task_config.pickup_obj_name
        )
        receptacle = create_mlspaces_body(
            task.env.current_data, task.config.task_config.place_receptacle_name
        )
        pickup_rel = _canonical_pose(base_inv @ pickup.pose)
        receptacle_rel = _canonical_pose(base_inv @ receptacle.pose)
        pickup_rel[:3] = np.clip(pickup_rel[:3] / self.position_scale_m, -1, 1)
        receptacle_rel[:3] = np.clip(receptacle_rel[:3] / self.position_scale_m, -1, 1)
        delta = np.clip(
            (receptacle.position - pickup.position) / self.position_scale_m, -1, 1
        ).astype(np.float32)
        grasp_flags = self._grasp_flags(raw_observation)
        task_state = np.concatenate([pickup_rel, receptacle_rel, delta, grasp_flags]).astype(
            np.float32
        )

        candidate_array = np.zeros(
            (self.max_candidates, self.candidate_dim), dtype=np.float32
        )
        mask = np.zeros(self.max_candidates, dtype=np.float32)
        left_ee = robot_view.get_move_group("left_gripper").leaf_frame_to_world
        right_ee = robot_view.get_move_group("right_gripper").leaf_frame_to_world
        for idx, pose_world in enumerate(candidates.poses_world[: self.max_candidates]):
            rel = base_inv @ pose_world
            pose = _canonical_pose(rel)
            pose[:3] = np.clip(pose[:3] / self.position_scale_m, -1, 1)
            d_left = min(np.linalg.norm(pose_world[:3, 3] - left_ee[:3, 3]) / 2.0, 1.0)
            d_right = min(np.linalg.norm(pose_world[:3, 3] - right_ee[:3, 3]) / 2.0, 1.0)
            tilt = np.arccos(np.clip(-pose_world[2, 2], -1.0, 1.0)) / np.pi
            candidate_array[idx] = np.concatenate(
                [
                    pose,
                    np.array(
                        [
                            d_left,
                            d_right,
                            float(candidates.reachable_left[idx]),
                            float(candidates.reachable_right[idx]),
                            float(candidates.collision_free[idx]),
                            tilt,
                        ],
                        dtype=np.float32,
                    ),
                ]
            )
            mask[idx] = 1.0

        phase_one_hot = np.array([phase == "pick", phase == "place"], dtype=np.float32)
        failure_one_hot = np.zeros(len(FAILURE_TYPES), dtype=np.float32)
        failure_name = last_failure if last_failure in FAILURE_TYPES else "none"
        failure_one_hot[FAILURE_TYPES.index(failure_name)] = 1
        phase_context = np.concatenate(
            [phase_one_hot, np.array([min(retry_count / 4.0, 1.0)], np.float32), failure_one_hot]
        ).astype(np.float32)
        result = {
            "proprio": proprio,
            "task": task_state,
            "candidates": candidate_array,
            "candidate_mask": mask,
            "phase_context": phase_context,
        }
        return result

    @staticmethod
    def _normalized_joint(robot_view, qpos: dict, name: str, size: int) -> np.ndarray:
        out = np.zeros(size, dtype=np.float32)
        if name not in qpos:
            return out
        value = np.asarray(qpos[name], dtype=float).reshape(-1)[:size]
        try:
            limits = np.asarray(
                robot_view.get_move_group(name).joint_pos_limits, dtype=float
            )[: len(value)]
            center = (limits[:, 0] + limits[:, 1]) / 2
            half = np.maximum((limits[:, 1] - limits[:, 0]) / 2, 1e-6)
            value = (value - center) / half
        except Exception:
            value = np.tanh(value)
        out[: len(value)] = np.clip(value, -1, 1)
        return out

    @staticmethod
    def _grasp_flags(raw_observation: Any) -> np.ndarray:
        if isinstance(raw_observation, list):
            raw_observation = raw_observation[0] if raw_observation else {}
        raw_observation = raw_observation or {}
        state = raw_observation.get("grasp_state_pickup_obj", {})
        values = []
        for gripper in ("left_gripper", "right_gripper"):
            item = state.get(gripper, {})
            values.extend(
                [
                    float(bool(item.get("touching", False))),
                    float(bool(item.get("held", False))),
                ]
            )
        return np.asarray(values, dtype=np.float32)
