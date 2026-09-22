import logging
from abc import abstractmethod
from typing import Any

import numpy as np
import torch
from curobo.geom.types import WorldConfig
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.planner.curobo_planner import CuroboPlanner
from molmo_spaces.policy.base_policy import PlannerPolicy
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.utils.pose import pose_mat_to_7d
from molmo_spaces.utils.profiler_utils import Profiler

log = logging.getLogger(__name__)


class CuroboPlannerPolicy(PlannerPolicy):
    """Base class for Curobo-based planner policies.

    This class provides common functionality for motion planning using Curobo,
    including trajectory execution, coordinate frame transformations, and
    gripper control. Subclasses should implement task-specific planning logic.
    """

    def __init__(self, config: MlSpacesExpConfig, task: BaseMujocoTask | None = None) -> None:
        super().__init__(config, task)
        self.config = config

        # Planner instances (to be set by subclasses)
        self.planner: CuroboPlanner | None = None
        self.planner_joint_ranges: dict[str, tuple[int, int]] = {}

        # Trajectory state
        self.planned_trajectory: list[list[float]] | None = None
        self.trajectory_index: int = 0
        self.steps_spent_in_waypoint: int = 0
        self._retry_count: int = 0

        # Arm selection
        self.arm_side: str | None = None
        self.arm_move_group_id: str | None = None
        self.gripper_move_group_id: str | None = None
        self.arm_start_idx: int = 0
        self.arm_end_idx: int = 0

        # Gripper state
        self.current_gripper_command: dict[str, float] = {}

        # 抓取瞬间"物体在夹爪坐标系下的位姿"，用于判断物体是否真的随夹爪运动。
        # None 表示尚未记录（LIFT 之前或本回合没做过抓取判定）。
        self._grasp_reference: np.ndarray | None = None

        # Profiler
        self.profiler = Profiler()

    @property
    def retry_count(self) -> int:
        return self._retry_count

    @property
    @abstractmethod
    def planners(self) -> dict[str, CuroboPlanner]:
        """Return dictionary of planner instances."""
        pass

    @property
    @abstractmethod
    def is_done(self) -> bool:
        """Property to expose completion state for task checking."""
        pass

    def reset(self) -> None:
        """Reset the policy state."""
        self.planned_trajectory = None
        self.trajectory_index = 0
        self.steps_spent_in_waypoint = 0
        self._retry_count = 0
        self.current_gripper_command = {}
        self._grasp_reference = None

    # ========== Joint Position Methods ==========

    def _get_current_joint_positions(self) -> list[float]:
        """Get current joint positions for all move groups.

        Returns:
            List of joint positions for all configured move groups.
        """
        joint_positions = []

        move_groups_for_planning = list(self.planner_joint_ranges.keys())
        for move_group in move_groups_for_planning:
            move_group_view = self.task.env.robots[0].robot_view.get_move_group(move_group)
            move_group_joint_pos = move_group_view.joint_pos.copy()
            # Clip to joint limits (mujoco uses soft constraints)
            move_group_joint_pos = np.clip(
                move_group_joint_pos,
                move_group_view.joint_pos_limits[:, 0],
                move_group_view.joint_pos_limits[:, 1],
            )
            joint_positions.extend(move_group_joint_pos.tolist())

        return joint_positions

    def _get_planning_start_config(self) -> np.ndarray:
        """Return the current planner state in its robot-relative convention.

        MolmoSpaces stores a mobile base pose in world coordinates, while the
        CuRobo models plan base motion relative to the episode's start pose.
        Non-base groups retain their simulator joint values.
        """
        start_config = np.asarray(self._get_current_joint_positions(), dtype=float)
        if "base" in self.planner_joint_ranges:
            start, end = self.planner_joint_ranges["base"]
            start_config[start:end] = 0.0
        return start_config

    # ========== Coordinate Frame Transformations ==========
    def _get_robot_base_pose(self) -> np.ndarray:
        """Return the common RobotView base pose for mobile robot adapters."""
        return self.task.env.robots[0].robot_view.base.pose

    def _transform_to_base_frame(self, target_ee_pose: np.ndarray) -> np.ndarray:
        """Transform target end-effector pose from world frame to robot base frame.

        Args:
            target_ee_pose: 7D pose [x, y, z, qw, qx, qy, qz] in world frame.

        Returns:
            4x4 transformation matrix in robot base frame.
        """
        robot_base_pose_tf = self._get_robot_base_pose()
        target_ee_pose_tf_world_f = np.eye(4)
        target_ee_pose_tf_world_f[:3, 3] = target_ee_pose[:3]
        target_ee_pose_tf_world_f[:3, :3] = R.from_quat(
            target_ee_pose[3:], scalar_first=True
        ).as_matrix()
        return np.linalg.inv(robot_base_pose_tf) @ target_ee_pose_tf_world_f

    def _transform_traj_to_world_frame(self, trajectory: list[list[float]]) -> None:
        """Transform planned trajectory from robot base frame to world frame.

        Modifies the trajectory in-place, converting base joint positions
        from robot-relative to world coordinates.

        Args:
            trajectory: List of waypoints to transform.
        """
        if "base" not in self.planner_joint_ranges:
            return
        robot_base_pose_tf = self._get_robot_base_pose()
        planner_joint_ranges = self.planner_joint_ranges
        for waypoint in trajectory:
            base_joint_x, base_joint_y, base_joint_yaw = waypoint[
                planner_joint_ranges["base"][0] : planner_joint_ranges["base"][1]
            ]
            base_joint_tf = np.eye(4)
            base_joint_tf[:3, 3] = np.array([base_joint_x, base_joint_y, 0.0])
            base_joint_tf[:3, :3] = R.from_euler("Z", base_joint_yaw, degrees=False).as_matrix()
            new_base_joint_tf = robot_base_pose_tf @ base_joint_tf
            waypoint[planner_joint_ranges["base"][0] : planner_joint_ranges["base"][1] - 1] = (
                new_base_joint_tf[:3, 3][:2]
            )
            waypoint[planner_joint_ranges["base"][1] - 1] = R.from_matrix(
                new_base_joint_tf[:3, :3]
            ).as_euler("XYZ", degrees=False)[2]

    def _target_pose_to_base_frame(self, target_pose: np.ndarray) -> np.ndarray:
        """Convert a 4x4 target pose matrix to 7D pose in robot base frame.

        Args:
            target_pose: 4x4 transformation matrix in world frame.

        Returns:
            7D pose [x, y, z, qw, qx, qy, qz] in robot base frame.
        """
        pose_7d = pose_mat_to_7d(target_pose)
        pose_base_frame_4x4 = self._transform_to_base_frame(pose_7d)
        pose_base_frame_7d = pose_mat_to_7d(pose_base_frame_4x4)
        return pose_base_frame_7d

    def _interpolate_joint_trajectory(
        self, start_config: np.ndarray, end_config: np.ndarray, num_steps: int = 10
    ) -> list[list[float]]:
        """Linearly interpolate between two joint configurations.

        Args:
            start_config: Starting joint configuration.
            end_config: Ending joint configuration.
            num_steps: Number of interpolation steps.

        Returns:
            List of waypoints interpolated between start and end configs.
        """
        trajectory = []
        for i in range(num_steps + 1):
            alpha = i / num_steps
            waypoint = (1 - alpha) * start_config + alpha * end_config
            trajectory.append(waypoint.tolist())
        return trajectory

    # ========== Waypoint and Action Methods ==========

    def _waypoint_to_action(self, waypoint: list[float]) -> dict[str, Any]:
        """Convert a waypoint (joint positions) to robot action dictionary.

        Args:
            waypoint: List of joint positions for all move groups.

        Returns:
            Dictionary mapping move group names to joint position arrays.
        """
        action = {}

        for move_group, (start_idx, end_idx) in self.planner_joint_ranges.items():
            if start_idx < len(waypoint) and end_idx <= len(waypoint):
                action[move_group] = np.array(waypoint[start_idx:end_idx])

        return action

    def _is_waypoint_reached(self, waypoint: list[float], tolerance: float = 0.0275) -> bool:
        """Check if the current robot state is close enough to the waypoint.

        Args:
            waypoint: Target joint positions.
            tolerance: Maximum allowed joint position error.

        Returns:
            True if all joints are within tolerance of the waypoint.
        """
        current_joint_pos = self._get_current_joint_positions()
        joint_diff = np.abs(np.array(current_joint_pos) - np.array(waypoint))
        return bool(np.all(joint_diff < tolerance))

    def _describe_waypoint_error(self, waypoint: list[float], top_n: int = 4) -> str:
        """列出与目标 waypoint 误差最大的几个关节，便于定位卡在哪一组上。

        只用 move group 名加组内序号标注（如 ``arm[3]``），因为基座关节存的是世界
        坐标而各机器人的组内关节名并不统一；超时诊断只需要指出是哪一组。
        """
        current = np.array(self._get_current_joint_positions())
        target = np.array(waypoint)
        if current.shape != target.shape:
            return f"关节向量长度不一致: current={current.shape}, waypoint={target.shape}"
        joint_diff = np.abs(current - target)
        labels = [
            f"{group}[{i - start}]"
            for group, (start, end) in self.planner_joint_ranges.items()
            for i in range(start, end)
        ]
        order = np.argsort(joint_diff)[::-1][:top_n]
        return ", ".join(f"{labels[i]}={joint_diff[i]:.4f}" for i in order)

    def _describe_stall(self) -> str:
        """超时诊断：区分"被物体挡住"与"伺服跟不上"。

        外部接触非空说明机器人在物理上抵住了场景物体；各组最大关节速度接近零
        说明已经停住（要么到位要么被挡），仍有明显速度则只是在缓慢逼近。
        """
        data = self.task.env.current_data
        model = data.model
        robot_namespace = self.config.robot_config.robot_namespace
        contacts: dict[str, int] = {}
        for contact in data.contact:
            root1 = model.body_rootid[model.geom_bodyid[contact.geom1]]
            root2 = model.body_rootid[model.geom_bodyid[contact.geom2]]
            name1 = model.body(root1).name
            name2 = model.body(root2).name
            robot_in_1 = name1.startswith(robot_namespace)
            robot_in_2 = name2.startswith(robot_namespace)
            if robot_in_1 == robot_in_2:
                continue
            robot_geom = contact.geom1 if robot_in_1 else contact.geom2
            other_body = name2 if robot_in_1 else name1
            robot_body = name1 if robot_in_1 else name2
            robot_geom_name = model.geom(robot_geom).name or robot_body
            key = f"{robot_geom_name} ↔ {other_body}"
            contacts[key] = contacts.get(key, 0) + 1
        view = self.task.env.current_robot.robot_view
        speeds = ", ".join(
            f"{group}={float(np.abs(view.get_move_group(group).joint_vel).max()):.4f}"
            for group in self.planner_joint_ranges
        )
        contact_summary = (
            "; ".join(f"{k}×{v}" for k, v in sorted(contacts.items(), key=lambda kv: -kv[1])[:3])
            or "无"
        )
        return f"外部接触({len(contacts)} 对): {contact_summary}; 各组最大关节速度: {speeds}"

    # ========== Trajectory Execution ==========

    def _execute_trajectory(self, gripper_command: dict[str, float]) -> dict[str, Any]:
        """Execute the planned trajectory with gripper control.

        Args:
            gripper_command: Dictionary with gripper control commands.

        Returns:
            Action dictionary for the robot.

        """
        if not self.planned_trajectory or self.trajectory_index >= len(self.planned_trajectory):
            self.failure_reason = "missing_or_exhausted_planned_trajectory"
            log.error(
                "Trajectory execution requested without an executable trajectory; "
                "ending episode as failure"
            )
            return {"done": True}

        # Skip waypoints that are already reached
        while self.trajectory_index < len(self.planned_trajectory):
            waypoint = self.planned_trajectory[self.trajectory_index]
            if self._is_waypoint_reached(waypoint):
                log.debug(f"[WAYPOINT] Skipping already-reached waypoint {self.trajectory_index}")
                self.trajectory_index += 1
                self.steps_spent_in_waypoint = 0
            else:
                break

        # If we've skipped all waypoints, return minimal action
        if self.trajectory_index >= len(self.planned_trajectory):
            action = {}
            action.update(gripper_command)
            self.current_gripper_command = gripper_command
            if hasattr(self, "get_look_at_action"):
                action.update(self.get_look_at_action())
            return action

        waypoint = self.planned_trajectory[self.trajectory_index]
        action = self._waypoint_to_action(waypoint)

        # Add gripper control
        action.update(gripper_command)
        self.current_gripper_command = gripper_command

        self.steps_spent_in_waypoint += 1

        # Check if waypoint reached
        if self._is_waypoint_reached(waypoint):
            log.debug(
                f"[WAYPOINT] Waypoint {self.trajectory_index} reached after "
                f"{self.steps_spent_in_waypoint} steps"
            )
            self.trajectory_index += 1
            self.steps_spent_in_waypoint = 0
        else:
            max_steps = getattr(self.config.policy_config, "max_steps_per_waypoint", 100)
            if self.steps_spent_in_waypoint >= max_steps:
                log.warning(
                    "[TIMEOUT] Timed out on reaching waypoint %d after %d steps "
                    "(largest joint errors: %s; %s). Will re-plan...",
                    self.trajectory_index,
                    self.steps_spent_in_waypoint,
                    self._describe_waypoint_error(waypoint),
                    self._describe_stall(),
                )
                max_reattempts = getattr(self.config.policy_config, "max_planning_reattempts", 3)
                if self.retry_count >= max_reattempts:
                    self.failure_reason = "max_planning_reattempts_reached"
                    log.error(
                        "Maximum planning reattempts (%d) reached; ending episode as failure",
                        max_reattempts,
                    )
                    return {"done": True}
                self.planned_trajectory = None
                self.trajectory_index = 0
                self._retry_count += 1
                self.steps_spent_in_waypoint = 0

        return action

    def _select_best_trajectory(
        self, planned_trajectories: list[list[list[float]] | None]
    ) -> list[list[float]] | None:
        """Select the trajectory with the least total joint movement.

        Args:
            planned_trajectories: List of candidate trajectories (may include None for failures).

        Returns:
            Best trajectory or None if all planning attempts failed.
        """
        # Filter out None trajectories (failed planning attempts)
        valid_trajectories = [traj for traj in planned_trajectories if traj is not None]

        if not valid_trajectories:
            log.warning("[TRAJECTORY SELECTION] All planning attempts failed")
            return None

        if len(valid_trajectories) == 1:
            return valid_trajectories[0]

        current_joint_pos = np.array(self._get_current_joint_positions())

        best_trajectory = None
        min_total_movement = float("inf")

        for trajectory in valid_trajectories:
            if not trajectory:
                continue

            total_movement = 0.0
            prev_joint_pos = current_joint_pos

            for waypoint in trajectory:
                waypoint_joint_pos = np.array(waypoint)
                joint_movement = np.linalg.norm(waypoint_joint_pos - prev_joint_pos)
                total_movement += joint_movement
                prev_joint_pos = waypoint_joint_pos

            if total_movement < min_total_movement:
                min_total_movement = total_movement
                best_trajectory = trajectory

        if best_trajectory is None:
            best_trajectory = valid_trajectories[0]

        log.info(
            f"Selected trajectory with total joint movement: {min_total_movement:.4f} "
            f"(out of {len(valid_trajectories)} candidates)"
        )
        return best_trajectory

    # ========== Velocity Constraints ==========

    def clip_to_velocity_constraint(self, action: dict[str, Any]) -> dict[str, Any]:
        """Clip action to respect velocity constraints.

        Args:
            action: Dictionary of commanded joint positions by move group.

        Returns:
            Clipped action dictionary.
        """
        velocity_constraints = getattr(self.config.policy_config, "velocity_constraints", {})
        clipped_action = action.copy()

        # velocity_constraints 的数值按 100 ms 的控制周期标定（见 policy_configs 的注释）。
        # 这里曾把 10 Hz 硬编码进换算，而 PandaOmron 的 policy_dt_ms 是 66 ms
        # （15.15 Hz），于是每步允许的变化量比设定值大 52%——LIFT 起步更猛，本来
        # 就"勉强夹住"的物体更容易被甩掉。改为按实际控制周期换算。
        step_s = self.config.policy_dt_ms / 1000.0
        constraint_ref_s = 0.1
        step_scale = step_s / constraint_ref_s

        for move_group, commanded_action in action.items():
            if move_group in velocity_constraints:
                max_delta_per_step = velocity_constraints[move_group] * step_scale
                move_group_view = self.task.env.robots[0].robot_view.get_move_group(move_group)
                move_group_joint_pos = move_group_view.joint_pos.copy()

                # Calculate difference, handling angular wraparound for base theta
                diff = commanded_action - move_group_joint_pos
                if move_group == "base":
                    # Normalize theta difference to [-π, π]
                    diff[2] = np.arctan2(np.sin(diff[2]), np.cos(diff[2]))

                clipped_delta = np.clip(diff, -max_delta_per_step, max_delta_per_step)
                clipped_action[move_group] = move_group_joint_pos + clipped_delta

        return clipped_action

    # ========== Gripper Methods ==========

    def _grasping_something(self, arm_side: str | None = None) -> bool:
        """Check if the gripper is grasping something.

        Determines grasp by checking if the gripper position deviates from
        the fully closed position by more than a threshold.

        Args:
            arm_side: Which arm to check ('left' or 'right'). Uses self.arm_side if None.

        Returns:
            True if gripper appears to be grasping an object.
        """
        if self.gripper_move_group_id is not None:
            gripper_move_group_id = self.gripper_move_group_id
        else:
            if arm_side is None:
                arm_side = self.arm_side
            gripper_move_group_id = f"{arm_side}_gripper"

        gripper_pos = self.task.env.robots[0].robot_view.get_move_group(
            gripper_move_group_id
        ).joint_pos
        gripper_closed_pos = getattr(self.config.policy_config, "gripper_closed_pos", 0.0)
        gripper_closed_tolerance = getattr(
            self.config.policy_config, "gripper_closed_tolerance", 0.01
        )

        gripper_deviation_from_closed = np.abs(gripper_pos - gripper_closed_pos).sum()
        is_grasping = gripper_deviation_from_closed > gripper_closed_tolerance

        log.debug(
            f"[GRIPPER] Gripper check - position: {gripper_pos}, "
            f"deviation: {gripper_deviation_from_closed:.4f}, grasping: {is_grasping}"
        )

        if not is_grasping:
            return False

        # 仅凭"夹爪没合到底"会产生两类假阳性：手指还在闭合途中，或只有一侧手指
        # 顶在桌面/柜台/其他几何上导致夹爪合不拢。E4 实测出现过
        # gripper0_right_finger1_pad_collision 顶住 countertop、以及目标物体只有
        # 单侧手指接触，两种情况下夹爪照样判为已抓取。是否要求双侧接触由配置决定：
        # 它更严格但会拒掉一部分"单侧凑巧够用"的抓取，默认关闭以保持既有行为。
        if not getattr(self.config.policy_config, "require_two_finger_contact", False):
            return True
        pickup_name = getattr(self.config.task_config, "pickup_obj_name", None)
        if pickup_name is None:
            return True
        finger_sides = self._finger_sides_contacting(pickup_name)
        if len(finger_sides) < 2:
            log.warning(
                "[GRIPPER] 夹爪未合拢，但与目标物体的接触只来自 %s；"
                "与目标物体的接触几何: %s，判定为未抓取",
                sorted(finger_sides) or "无手指",
                self._object_contact_geoms(pickup_name),
            )
            return False
        return True

    def _object_contact_geoms(self, object_name: str) -> list[str]:
        """列出所有与目标物体接触的对方几何名，用于诊断抓取判定。"""
        env = self.task.env
        data = env.current_data
        model = data.model
        pickup = env.object_managers[env.current_batch_index].get_object_by_name(object_name)

        geoms: list[str] = []
        for i in range(data.ncon):
            contact = data.contact[i]
            root1 = model.body_rootid[model.geom_bodyid[contact.geom1]]
            root2 = model.body_rootid[model.geom_bodyid[contact.geom2]]
            if (root1 == pickup.body_id) == (root2 == pickup.body_id):
                continue
            other = contact.geom2 if root1 == pickup.body_id else contact.geom1
            geoms.append(model.geom(other).name or f"geom{other}")
        return geoms[:4]

    # 手指标识子串：Panda 是 finger1/finger2（注意两个手指名里都含 "right"，
    # 因此不能按 left/right 区分），RBY1 系是 left_finger/right_finger。
    _FINGER_ID_PATTERNS = ("finger1", "finger2", "finger_l", "finger_r", "left_finger", "right_finger")

    def _finger_sides_contacting(self, object_name: str) -> set[str]:
        """返回与目标物体接触的手指标识集合（如 {"finger1", "finger2"}）。"""
        env = self.task.env
        data = env.current_data
        model = data.model
        pickup = env.object_managers[env.current_batch_index].get_object_by_name(object_name)

        sides: set[str] = set()
        for i in range(data.ncon):
            contact = data.contact[i]
            root1 = model.body_rootid[model.geom_bodyid[contact.geom1]]
            root2 = model.body_rootid[model.geom_bodyid[contact.geom2]]
            if (root1 == pickup.body_id) == (root2 == pickup.body_id):
                continue
            robot_geom = contact.geom2 if root1 == pickup.body_id else contact.geom1
            geom_name = (model.geom(robot_geom).name or "").lower()
            for pattern in self._FINGER_ID_PATTERNS:
                if pattern in geom_name:
                    sides.add(pattern)
                    break
        return sides

    # ========== 抓取保持判据 ==========

    def _grasp_still_valid(self) -> bool:
        """抓取是否仍然成立：夹爪判据 + 物体相对位姿未漂移。"""
        if not self._grasping_something():
            return False
        return self._object_pose_held()

    def _capture_grasp_reference(self) -> None:
        """记录抓取瞬间物体在夹爪坐标系下的位姿，供 LIFT 阶段比对。"""
        self._grasp_reference = self._object_relative_pose()

    def _object_relative_pose(self) -> np.ndarray | None:
        """目标物体在夹爪（TCP）坐标系下的位姿；拿不到目标物体时返回 None。"""
        pickup_name = getattr(self.config.task_config, "pickup_obj_name", None)
        if pickup_name is None:
            return None
        env = self.task.env
        pickup = env.object_managers[env.current_batch_index].get_object_by_name(pickup_name)
        tcp_pose = env.current_robot.robot_view.get_move_group(
            self.gripper_move_group_id
        ).leaf_frame_to_world
        return np.linalg.inv(tcp_pose) @ pickup.pose

    def _object_pose_held(self) -> bool:
        """物体是否仍相对夹爪保持在抓取瞬间的位姿。

        这比"两侧手指是否都接触物体"更本质：单侧接触但物体被稳定约束时同样算
        抓住；而手指顶住桌面/柜台这类假阳性会让物体相对夹爪明显漂移。是否启用由
        policy_config.require_object_pose_hold 控制，阈值默认 1 cm / 5.7°。
        """
        if not getattr(self.config.policy_config, "require_object_pose_hold", False):
            return True
        if self._grasp_reference is None:
            return True
        current = self._object_relative_pose()
        if current is None:
            return True

        pos_drift = float(np.linalg.norm(current[:3, 3] - self._grasp_reference[:3, 3]))
        rot_drift = float(
            R.from_matrix(current[:3, :3] @ self._grasp_reference[:3, :3].T).magnitude()
        )
        pos_tol = getattr(self.config.policy_config, "grasp_hold_pos_tolerance", 0.01)
        rot_tol = getattr(self.config.policy_config, "grasp_hold_rot_tolerance", 0.1)

        if pos_drift > pos_tol or rot_drift > rot_tol:
            log.warning(
                "[GRASP] 物体相对夹爪漂移 %.4f m / %.1f°（阈值 %.4f m / %.1f°），判定为未抓住",
                pos_drift,
                np.degrees(rot_drift),
                pos_tol,
                np.degrees(rot_tol),
            )
            return False
        return True

    # ========== Arm Selection ==========

    def select_arm(self) -> None:
        """Select which arm to use based on distance to pickup object.

        Also instantiates the motion planner for the selected arm.
        This lazy initialization saves ~11GB of GPU memory by only loading one arm's planner.
        """
        task_config = self.config.task_config
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        pickup_obj = om.get_object_by_name(task_config.pickup_obj_name)
        pickup_obj_pos = pickup_obj.position

        left_tcp_pose = self.task.env.current_robot.robot_view.get_move_group(
            "left_gripper"
        ).leaf_frame_to_world
        right_tcp_pose = self.task.env.current_robot.robot_view.get_move_group(
            "right_gripper"
        ).leaf_frame_to_world

        left_tcp_pos = left_tcp_pose[:3, 3]
        right_tcp_pos = right_tcp_pose[:3, 3]

        # Compute distances
        left_dist = np.linalg.norm(left_tcp_pos - pickup_obj_pos)
        right_dist = np.linalg.norm(right_tcp_pos - pickup_obj_pos)

        selected_arm = "left" if left_dist < right_dist else "right"
        log.info(
            f"Selected {selected_arm} arm (left dist: {left_dist:.3f}m, right dist: {right_dist:.3f}m)"
        )

        self.arm_side = selected_arm
        self.arm_move_group_id = f"{selected_arm}_arm"
        self.gripper_move_group_id = f"{selected_arm}_gripper"

        # Instantiate the planner for the selected arm only
        log.info(f"Instantiating motion planner for {selected_arm} arm")
        if selected_arm == "left":
            self.planner = CuroboPlanner(
                config=self.config.policy_config.left_curobo_planner_config
            )
            self.planner_joint_ranges = self.config.policy_config.left_planner_joint_ranges
        else:
            self.planner = CuroboPlanner(
                config=self.config.policy_config.right_curobo_planner_config
            )
            self.planner_joint_ranges = self.config.policy_config.right_planner_joint_ranges

        self.arm_start_idx = self.planner_joint_ranges[self.arm_move_group_id][0]
        self.arm_end_idx = self.planner_joint_ranges[self.arm_move_group_id][1]

    # ========== IK and Motion Planning ==========

    def solve_ik(self, target_pose: np.ndarray) -> None:
        """Solve inverse kinematics for a target pose and create interpolated trajectory.

        Args:
            target_pose: 4x4 transformation matrix for target end-effector pose in world frame.

        Raises:
            ValueError: If IK solution cannot be found.
        """
        init_config = self._get_planning_start_config()

        target_pose_7d = self._target_pose_to_base_frame(target_pose)
        joint_config, _ = self.planner.ik_solve(
            goal_pose=target_pose_7d.tolist(),
            seed_config=init_config.tolist(),
            disable_collision=True,
        )

        current_phase = getattr(self, "current_phase", "unknown")
        if joint_config is None:
            raise ValueError(f"Could not solve {current_phase} phase IK.")

        trajectory = self._interpolate_joint_trajectory(
            init_config,
            np.array(joint_config),
            num_steps=10,
        )
        self._transform_traj_to_world_frame(trajectory)
        self.planned_trajectory = trajectory

    def batch_plan_trajectory(self) -> bool:
        """Plan trajectory using batch motion planning.

        Uses the current phase to determine goal poses and plans trajectories
        in batches for efficiency. Sets self.planned_trajectory to the best
        trajectory found.
        """
        init_config = self._get_planning_start_config()

        # Setup collision avoidance if enabled
        if getattr(self.config.policy_config, "enable_collision_avoidance", False):
            if hasattr(self, "_setup_collision_avoidance_config"):
                self._setup_collision_avoidance_config()

        # Get goal poses based on current phase (to be provided by subclass)
        goal_poses = self._get_batch_goal_poses()
        if goal_poses is None or len(goal_poses) == 0:
            log.warning("[BATCH PLAN] No goal poses available")
            self.planned_trajectory = None
            return False

        total = goal_poses.shape[0]
        batch_size = getattr(self.config.policy_config, "batch_size", 8)
        max_batches = getattr(self.config.policy_config, "max_batch_plan_attempts", 4)
        num_poses = min(max_batches * batch_size, total)
        num_batches = (num_poses + batch_size - 1) // batch_size

        all_successful_trajectories = []
        failure_statuses: list[str] = []
        current_phase = getattr(self, "current_phase", "unknown")

        for batch_start in range(0, num_poses, batch_size):
            batch_end = min(batch_start + batch_size, num_poses)
            batch = goal_poses[batch_start:batch_end]

            if hasattr(self, "_show_poses"):
                self._show_poses(batch, style="tcp")
            if self.task.viewer:
                self.task.viewer.sync()

            log.info(
                f"Processing batch {batch_start // batch_size + 1}/{num_batches}: "
                f"poses {batch_start}-{batch_end - 1}"
            )

            # Transform poses to base frame and convert to 7D format
            batch_base_frame_7d = []
            for pose in batch:
                pose_base = self._target_pose_to_base_frame(pose)
                batch_base_frame_7d.append(pose_base.tolist())

            # Prepare batch inputs for planner
            batch_len = len(batch_base_frame_7d)
            start_states = [init_config.tolist()] * batch_len

            # Plan batch
            log.info(
                f"Planning batch of {batch_len} {current_phase} poses with {self.arm_side} arm"
            )
            result = self.planner.plan_batch(
                start_states,
                batch_base_frame_7d,
                verbose=False,
            )

            # Check for successes
            successes = result.success.cpu().numpy()
            log.info(
                f"{self.arm_side.capitalize()} arm: {np.sum(successes)}/{batch_len} "
                f"{current_phase} planning successes"
            )

            if not np.any(successes):
                status = getattr(result, "status", None)
                if status is not None:
                    if isinstance(status, (list, tuple)):
                        failure_statuses.extend(str(item) for item in status)
                    else:
                        failure_statuses.append(str(status))

            # Process successful trajectories
            if np.any(successes):
                optimized_plan = result.optimized_plan
                position = optimized_plan.position
                if position.ndim == 2:
                    position = position.unsqueeze(0)

                for i in range(batch_len):
                    if not successes[i]:
                        continue

                    trajectory = []
                    for t in range(position.shape[1]):
                        waypoint = position[i, t].cpu().tolist()
                        trajectory.append(waypoint)

                    self._transform_traj_to_world_frame(trajectory)
                    all_successful_trajectories.append(trajectory)

            if all_successful_trajectories:
                log.info(
                    f"Found {len(all_successful_trajectories)} successful trajectories, "
                    "skipping remaining batches"
                )
                break

        if all_successful_trajectories:
            self.planned_trajectory = self._select_best_trajectory(all_successful_trajectories)
            return True
        else:
            self.planned_trajectory = None
            status_summary = sorted(set(failure_statuses))
            log.warning(
                "[BATCH PLAN] No successful trajectories found across all batches; "
                "CuRobo statuses=%s",
                status_summary or ["unavailable"],
            )
            return False

    def _get_batch_goal_poses(self) -> np.ndarray | None:
        """Get goal poses for batch planning based on current phase.

        Subclasses should override this method to return appropriate goal poses.

        Returns:
            Array of 4x4 pose matrices or None if not applicable.
        """
        return None

    # ========== Visualization ==========

    def visualize_world_config_mesh(self, world_cfg: WorldConfig) -> None:
        """Visualize the world configuration as a mesh file.

        Args:
            world_cfg: Curobo WorldConfig to visualize.
        """
        from datetime import datetime

        import trimesh
        from curobo.geom.types import WorldConfig

        current_phase = getattr(self, "current_phase", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"world_config_mesh_{current_phase}_{timestamp}"
        obj_path = f"{file_name}.obj"
        initial_joint_configuration = np.array(self._get_current_joint_positions())

        try:
            # Create a combined scene with robot and world obstacles
            combined_scene = trimesh.scene.scene.Scene(base_frame="world_origin")

            # Add robot mesh if we have a joint configuration
            if initial_joint_configuration is not None and self.planner is not None:
                try:
                    config_for_mesh = np.array(initial_joint_configuration).copy()
                    # Zero out base joints to place robot at origin
                    config_for_mesh[:3] = 0.0
                    q_tensor = torch.tensor(config_for_mesh).unsqueeze(0).float().cuda()

                    robot_spheres_batch = self.planner.motion_gen.kinematics.get_robot_as_spheres(
                        q_tensor
                    )
                    if robot_spheres_batch and len(robot_spheres_batch) > 0:
                        robot_spheres = robot_spheres_batch[0]
                        for i, sphere in enumerate(robot_spheres):
                            sphere_mesh = trimesh.creation.icosphere(radius=sphere.radius)
                            sphere_transform = np.eye(4)
                            sphere_transform[:3, 3] = sphere.pose[:3]
                            combined_scene.add_geometry(
                                sphere_mesh,
                                geom_name=f"robot_sphere_{i}",
                                parent_node_name="world_origin",
                                transform=sphere_transform,
                            )
                        log.info(
                            f"Added {len(robot_spheres)} robot collision spheres to visualization"
                        )

                except Exception as e:
                    log.warning(f"Could not add robot mesh: {e}")

            # Add world obstacles to the scene
            try:
                world_scene = WorldConfig.get_scene_graph(world_cfg, process_color=True)
                for geom_name, geom in world_scene.geometry.items():
                    transform = world_scene.graph.get(geom_name)[0]
                    combined_scene.add_geometry(
                        geom,
                        geom_name=geom_name,
                        parent_node_name="world_origin",
                        transform=transform,
                    )
            except Exception as e:
                log.warning(f"Could not add world obstacles: {e}")

            # Export combined scene
            if len(combined_scene.geometry) > 0:
                combined_scene.export(obj_path)
                log.info(f"Successfully saved combined robot + world mesh to {obj_path}")
            else:
                log.debug("Skipping mesh export - scene is empty")

        except ValueError as e:
            if "empty scene" in str(e).lower():
                log.debug("Skipping mesh export - world config is empty")
            else:
                raise

    # ========== Look-at Action (can be overridden) ==========

    def get_look_at_action(self) -> dict[str, Any]:
        """Get action to look at a target.

        Subclasses can override this to implement head tracking.

        Returns:
            Dictionary with head control commands, or empty dict.
        """
        return {}
