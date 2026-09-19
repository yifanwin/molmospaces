"""CuRobo pick-and-place policy for the robosuite PandaOmron robot."""

from __future__ import annotations

import logging
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from curobo.geom.types import Cuboid, WorldConfig
from molmo_spaces.utils.planner_collision_geometry import body_collision_boxes
from molmo_spaces.utils.pose import pose_mat_to_7d

from molmo_spaces.planner.curobo_planner import CuroboPlanner
from molmo_spaces.policy.solvers.object_manipulation.curobo_pick_and_place_planner_policy import (
    CuroboPickAndPlacePlannerPolicy,
)

log = logging.getLogger(__name__)


class PandaOmronCuroboPickAndPlacePlannerPolicy(CuroboPickAndPlacePlannerPolicy):
    """Single-arm specialization with configurable base/torso participation."""

    _JOINT_NAMES = {
        "base": ["base_x", "base_y", "base_theta"],
        "torso": ["torso_height"],
        "arm": [f"panda_joint{i}" for i in range(1, 8)],
    }

    def select_arm(self) -> None:
        policy_config = self.config.policy_config
        if policy_config.server_urls:
            raise ValueError("PandaOmron CuRobo evaluation only supports local planning")
        if policy_config.curobo_planner_config is None:
            raise ValueError("PandaOmron CuRobo planner config is missing")

        self.arm_side = "panda_omron"
        self.arm_move_group_id = policy_config.arm_move_group_id
        self.gripper_move_group_id = policy_config.gripper_move_group_id
        self.planner_joint_ranges = dict(policy_config.planner_joint_ranges)
        self.arm_start_idx, self.arm_end_idx = self.planner_joint_ranges[
            self.arm_move_group_id
        ]

        planner_config = policy_config.curobo_planner_config.model_copy(deep=True)
        current = {
            group: self.task.env.current_robot.robot_view.get_move_group(group).joint_pos.tolist()
            for group in ("base", "torso", "arm")
        }
        # CuRobo's base coordinates are relative to the episode start pose.
        current["base"] = [0.0, 0.0, 0.0]

        active_groups = policy_config.planner_move_group_ids
        joint_names = [name for group in active_groups for name in self._JOINT_NAMES[group]]
        retract_config = [value for group in active_groups for value in current[group]]
        planner_config.kinematics_config = dict(planner_config.kinematics_config or {})
        planner_config.kinematics_config["cspace"] = {
            "joint_names": joint_names,
            "retract_config": retract_config,
            "null_space_weight": [1.0] * len(joint_names),
            "cspace_distance_weight": [1.0] * len(joint_names),
            "max_jerk": 500.0,
            "max_acceleration": 15.0,
        }
        planner_config.lock_joints = {
            joint_name: value
            for group in ("base", "torso", "arm")
            if group not in active_groups
            for joint_name, value in zip(self._JOINT_NAMES[group], current[group], strict=True)
        }

        log.info(
            "Instantiating local PandaOmron CuRobo planner with move groups: %s",
            active_groups,
        )
        self.planner = CuroboPlanner(config=planner_config)

    def _setup_collision_avoidance_config(self) -> None:
        # Preserve the existing obstacle selection, but do not fill furniture
        # cavities (e.g. U-shaped counters) with one body-wide bounding box.
        objects = self._get_collision_cuboids()
        env = self.task.env
        model, data = env.current_model, env.current_data
        om = env.object_managers[env.current_batch_index]
        world_to_base = np.linalg.inv(self._get_robot_base_pose())
        cuboids = []
        for obj_box in objects:
            obj = om.get_object_by_name(obj_box.name)
            for geom_id, pose, dims in body_collision_boxes(
                model, data, obj.body_id, world_to_base
            ):
                cuboids.append(Cuboid(
                    name=f"{obj_box.name}/geom_{geom_id}",
                    pose=pose_mat_to_7d(pose).tolist(),
                    dims=dims.tolist(),
                ))
        world_config = WorldConfig(cuboid=cuboids)
        self.planner.motion_gen.update_world(world_config)
        self.planner.world_config = world_config
        log.info("Loaded %d scene cuboids into local CuRobo", len(cuboids))
        # Report start-state overlap in the same geometry used by CuRobo.
        q = torch.as_tensor(self._get_planning_start_config(), device="cuda", dtype=torch.float32)[None]
        spheres = self.planner.motion_gen.kinematics.get_state(q).link_spheres_tensor.detach().cpu().numpy().reshape(-1, 4)
        overlaps = []
        for box in cuboids:
            local = (spheres[:, :3] - np.asarray(box.pose[:3])) @ Rotation.from_quat(box.pose[3:], scalar_first=True).as_matrix()
            delta = np.abs(local) - np.asarray(box.dims) / 2
            distance = np.linalg.norm(np.maximum(delta, 0), axis=1) + np.minimum(np.max(delta, axis=1), 0)
            penetration = spheres[:, 3] - distance
            valid = spheres[:, 3] > 0
            if np.any(valid & (penetration > 0)):
                overlaps.append((box.name, float(penetration[valid].max())))
        if overlaps:
            log.warning("CuRobo start-state overlaps (obstacle, penetration m): %s", overlaps)
