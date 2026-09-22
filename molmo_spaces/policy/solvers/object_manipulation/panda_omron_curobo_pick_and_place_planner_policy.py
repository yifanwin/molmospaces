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

# 与 panda_omron_spheres.yml 中 collision_spheres 的声明顺序一致，用于把 CuRobo 的
# 碰撞球索引（link_spheres_tensor 的行号）映射回机器人 link 名，便于定位起始状态
# 重叠来自哪个部位。顺序一旦与 yml 不符会在 _setup_collision_avoidance_config 的
# 自检里报错。
_SPHERE_LINK_NAMES: tuple[str, ...] = (
    ("omron_base",) * 6
    + ("panda_link0",)
    + ("panda_link1",) * 2
    + ("panda_link2",) * 2
    + ("panda_link3",) * 2
    + ("panda_link4",) * 2
    + ("panda_link5",) * 2
    + ("panda_link6",)
    + ("panda_link7",)
    + ("panda_hand",) * 2
)

# 起始状态重叠日志最多逐条打印的条目数，其余折叠为计数，避免长 episode 刷屏。
_MAX_LOGGED_OVERLAPS = 12


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
        kept, overlaps = self._split_start_overlaps(cuboids)
        world_config = WorldConfig(cuboid=kept)
        self.planner.motion_gen.update_world(world_config)
        self.planner.world_config = world_config
        log.info("Loaded %d scene cuboids into local CuRobo", len(kept))
        if overlaps:
            log.warning(
                "Dropped %d start-state overlapping cuboid(s) from the CuRobo world "
                "(robot link, obstacle, penetration m): %s",
                len(overlaps),
                overlaps[:_MAX_LOGGED_OVERLAPS],
            )

    def _split_start_overlaps(
        self, cuboids: list[Cuboid]
    ) -> tuple[list[Cuboid], list[tuple[str, str, float]]]:
        """把障碍物盒按"是否与起始状态重叠"分成保留与剔除两组。

        CuRobo 的碰撞球与障碍物盒都是保守近似：omron_base 用四个半径 0.17 m 的球
        覆盖底盘，向外超出真实几何约 0.1 m；panda_hand 的球也超出夹持点约 0.058 m
        （见 policy_configs 的 pregrasp_z_offset 注释）。抓取失败回退 PREGRASP 时
        手爪就停在目标物体旁边，这些外扩的球与目标物体、放置容器的盒子很容易重叠。

        若把这些障碍物留在规划世界里，trajopt 会从一个被判为碰撞的状态出发，整条
        episode 必然以 TRAJOPT_FAIL 结束（E4 实测 365 次起始重叠 100% 紧随规划
        失败，是 333 次 pregrasp 规划失败的主要来源，其中 268 次就发生在首次进入
        PREGRASP 时）。把它们暂时移出本次规划世界，机器人才能从既成位姿脱离；真实
        穿透仍由 MuJoCo 物理负责拦截，且下一次规划会重新构建世界，剔除范围不累积。

        Returns:
            (保留的盒子, [(robot link, 障碍物名, 穿透深度 m), ...])
        """
        q = torch.as_tensor(self._get_planning_start_config(), device="cuda", dtype=torch.float32)[None]
        spheres = self.planner.motion_gen.kinematics.get_state(q).link_spheres_tensor.detach().cpu().numpy().reshape(-1, 4)
        active = spheres[:, 3] > 0
        kept, overlaps = [], []
        for box in cuboids:
            local = (spheres[:, :3] - np.asarray(box.pose[:3])) @ Rotation.from_quat(box.pose[3:], scalar_first=True).as_matrix()
            delta = np.abs(local) - np.asarray(box.dims) / 2
            distance = np.linalg.norm(np.maximum(delta, 0), axis=1) + np.minimum(np.max(delta, axis=1), 0)
            penetration = spheres[:, 3] - distance
            hitting = active & (penetration > 0)
            if np.any(hitting):
                worst = int(np.argmax(np.where(hitting, penetration, -np.inf)))
                overlaps.append((self._sphere_label(worst), box.name, float(penetration[worst])))
            else:
                kept.append(box)
        return kept, overlaps

    @staticmethod
    def _sphere_label(sphere_index: int) -> str:
        """把 CuRobo 的碰撞球索引映射回可读标签。

        link_spheres_tensor 的顺序是 panda_omron_spheres.yml 中 collision_spheres 的
        声明顺序，其后追加 extra_collision_spheres 为 attached_object 生成的手部附加
        球（本模型 40 个），所以总球数（61）多于机身球数（21）。
        """
        if sphere_index < len(_SPHERE_LINK_NAMES):
            return _SPHERE_LINK_NAMES[sphere_index]
        return f"attached_object[{sphere_index - len(_SPHERE_LINK_NAMES)}]"
