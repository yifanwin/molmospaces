"""Translate normalized RL actions into safe RBY1 high-level commands."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class DecodedRBY1Action:
    base_goal: np.ndarray
    candidate_index: int | None
    arm: str | None
    translation_m: float
    yaw_delta_rad: float
    valid: bool = True
    invalid_reason: str | None = None


class RBY1ActionAdapter:
    """Decode the five-dimensional action used by SAC.

    Translation is expressed in the robot's current local XY frame.  Candidate
    and arm dimensions are ignored during PLACE, where the grasp selected in
    PICK remains authoritative.
    """

    action_dim = 5

    def __init__(self, max_translation_m: float = 0.5, max_yaw_delta_rad: float = np.pi / 2):
        self.max_translation_m = float(max_translation_m)
        self.max_yaw_delta_rad = float(max_yaw_delta_rad)

    def decode(
        self,
        action: np.ndarray,
        base_pose: np.ndarray,
        num_candidates: int,
        phase: str,
        env=None,
    ) -> DecodedRBY1Action:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (self.action_dim,) or not np.all(np.isfinite(action)):
            return self._invalid(base_pose, "action must be five finite values")
        action = np.clip(action, -1.0, 1.0)

        local_xy = action[:2].astype(float) * self.max_translation_m
        norm = float(np.linalg.norm(local_xy))
        if norm > self.max_translation_m:
            local_xy *= self.max_translation_m / norm
            norm = self.max_translation_m

        yaw = float(Rotation.from_matrix(base_pose[:3, :3]).as_euler("xyz")[2])
        world_xy = base_pose[:2, :2] @ local_xy
        yaw_delta = float(action[2]) * self.max_yaw_delta_rad
        goal = np.array(
            [base_pose[0, 3] + world_xy[0], base_pose[1, 3] + world_xy[1], yaw + yaw_delta],
            dtype=np.float64,
        )

        candidate_index = None
        arm = None
        if phase == "pick":
            if num_candidates <= 0:
                return self._invalid(base_pose, "no grasp candidates")
            unit = (float(action[3]) + 1.0) / 2.0
            candidate_index = min(int(unit * num_candidates), num_candidates - 1)
            arm = "left" if action[4] < 0 else "right"

        valid, reason = self._base_path_is_valid(env, base_pose, goal)
        return DecodedRBY1Action(
            base_goal=goal,
            candidate_index=candidate_index,
            arm=arm,
            translation_m=norm,
            yaw_delta_rad=yaw_delta,
            valid=valid,
            invalid_reason=reason,
        )

    @staticmethod
    def _invalid(base_pose: np.ndarray, reason: str) -> DecodedRBY1Action:
        yaw = float(Rotation.from_matrix(base_pose[:3, :3]).as_euler("xyz")[2])
        return DecodedRBY1Action(
            base_goal=np.array([base_pose[0, 3], base_pose[1, 3], yaw]),
            candidate_index=None,
            arm=None,
            translation_m=0.0,
            yaw_delta_rad=0.0,
            valid=False,
            invalid_reason=reason,
        )

    @staticmethod
    def _base_path_is_valid(
        env, base_pose: np.ndarray, goal: np.ndarray
    ) -> tuple[bool, str | None]:
        if env is None:
            return True, None
        try:
            base = env.current_robot.robot_view.get_move_group("base")
            if np.asarray(base.joint_pos).reshape(-1).size != 3:
                return False, "RBY1 base is not a three-DoF holonomic base"
            occupancy = env.get_occupancy_map(agent_radius=0.35)
            start = np.asarray(base_pose[:2, 3], dtype=float)
            samples = max(2, int(np.ceil(np.linalg.norm(goal[:2] - start) / 0.025)))
            xy = np.linspace(start, goal[:2], samples + 1)
            xyz = np.column_stack([xy, np.zeros(len(xy))])
            if hasattr(occupancy, "check_collision"):
                free = np.asarray(occupancy.check_collision(xyz), dtype=bool)
            elif hasattr(occupancy, "is_free"):
                free = np.asarray([occupancy.is_free(point) for point in xy], dtype=bool)
            else:
                return True, None
            if not bool(np.all(free)):
                return False, "base path crosses occupied space"
        except Exception as exc:
            return False, f"base occupancy validation failed: {type(exc).__name__}"
        return True, None
