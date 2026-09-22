"""P1 navigation-only adapter and audited execution loop."""

from contextlib import contextmanager
import hashlib
import json

import mujoco
import networkx as nx
import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.utils.linalg_utils import normalize_ang_error
from .snapshot import override_base_pose


TERMINAL_STATUSES = {
    "completed", "start_unavailable", "no_path", "timeout", "collision",
    "controller_error", "scene_changed",
}


def derive_seed(seed: int, label: str) -> int:
    payload = json.dumps([int(seed), label], separators=(",", ":")).encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


@contextmanager
def numpy_seed(seed: int):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def se2_pose(robot_view) -> np.ndarray:
    pose = robot_view.base.pose
    yaw = R.from_matrix(pose[:3, :3]).as_euler("xyz")[2]
    return np.array([pose[0, 3], pose[1, 3], yaw], dtype=float)


def object_pose(obj: MlSpacesObject) -> np.ndarray:
    return obj.pose.copy()


def pose_change(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    pos = float(np.linalg.norm(current[:3, 3] - reference[:3, 3]))
    rot = float(R.from_matrix(current[:3, :3] @ reference[:3, :3].T).magnitude())
    return pos, rot


class LastMileNavAdapter:
    """Expose exactly one frozen PnP target to A* without PnP termination semantics."""

    def __init__(self, task, target_name: str):
        self.task = task
        self.env = task.env
        self.target = MlSpacesObject(object_name=target_name, data=self.env.current_data)
        self.nav_objs = [[self.target]]
        self.step_count = 0

    def num_steps_taken(self) -> int:
        return self.step_count

    def get_nav_object_priority(self, batch_index: int):
        assert batch_index == 0
        return self.nav_objs[0][:]


def sample_remote_start(
    task,
    planner,
    goal_xy,
    target_xy,
    seed: int,
    radius_range=(1.5, 3.0),
    max_attempts=100,
):
    """Sample a collision-free pose in the A* component containing the navigation goal."""
    goal_node = planner.get_discrete_location(np.asarray(goal_xy))
    if goal_node is None or goal_node not in planner.graph:
        return None, {"attempts": 0, "detail": "non_plannable_goal"}
    component = nx.node_connected_component(planner.graph, tuple(goal_node))
    free = planner.map.get_free_points()
    distances = np.linalg.norm(free[:, :2] - np.asarray(target_xy)[None, :2], axis=1)
    eligible = free[(distances >= radius_range[0]) & (distances <= radius_range[1])]
    rng = np.random.default_rng(seed)
    if len(eligible):
        order = rng.choice(len(eligible), size=min(max_attempts, len(eligible)), replace=False)
    else:
        order = []
    robot_view = task.env.current_robot.robot_view
    z = float(robot_view.base.pose[2, 3])
    counts = {"eligible_points": int(len(eligible)), "component_rejected": 0,
              "collision_rejected": 0, "attempts": 0}
    for idx in order:
        counts["attempts"] += 1
        xy = eligible[idx, :2]
        node = planner.get_discrete_location(np.r_[xy, 0.0])
        if node is None or tuple(node) not in component:
            counts["component_rejected"] += 1
            continue
        yaw = float(rng.uniform(-np.pi, np.pi))
        pose = np.eye(4)
        pose[:3, 3] = [xy[0], xy[1], z]
        pose[:3, :3] = R.from_euler("z", yaw).as_matrix()
        if task.env.check_if_robot_collision_at_base_pose(
            robot_view, pose, task.config.robot_config.robot_namespace
        ):
            counts["collision_rejected"] += 1
            continue
        counts.update(distance_to_target=float(np.linalg.norm(xy - target_xy[:2])))
        return pose, counts
    counts["detail"] = "no_collision_free_pose_in_goal_component"
    return None, counts


def illegal_contacts(env, robot_namespace: str) -> list[dict]:
    model, data = env.current_model, env.current_data
    rows = []
    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.dist > 0:
            continue
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        root1 = int(model.body_rootid[body1])
        root2 = int(model.body_rootid[body2])
        root1_name = model.body(root1).name or ""
        root2_name = model.body(root2).name or ""
        robot1, robot2 = root1_name.startswith(robot_namespace), root2_name.startswith(robot_namespace)
        if not (robot1 or robot2):
            continue
        if root1 == root2:
            continue
        other = root2_name if robot1 else root1_name
        if "floor" in other.lower():
            continue
        rows.append({
            "geom1": model.geom(contact.geom1).name,
            "geom2": model.geom(contact.geom2).name,
            "body1": model.body(body1).name,
            "body2": model.body(body2).name,
            "root1": root1_name,
            "root2": root2_name,
            "distance": float(contact.dist),
        })
    return rows


def nonbase_positions(robot) -> dict[str, np.ndarray]:
    positions = {
        name: robot.robot_view.get_move_group(name).joint_pos.copy()
        for name in robot.controllers if name != "base"
    }
    # RBY1 的 head 没有控制器，但协议仍要求它保持初始姿态。
    try:
        positions["head"] = robot.robot_view.get_move_group("head").joint_pos.copy()
    except (KeyError, ValueError):
        pass
    return positions


def max_nonbase_drift(reference: dict[str, np.ndarray], robot) -> tuple[float, dict[str, float]]:
    per_group = {}
    for name, start in reference.items():
        current = robot.robot_view.get_move_group(name).joint_pos
        per_group[name] = float(np.max(np.abs(current - start))) if len(start) else 0.0
    return max(per_group.values(), default=0.0), per_group


def nonbase_targets(robot) -> dict[str, np.ndarray]:
    return {
        name: controller.target_pos.copy()
        for name, controller in robot.controllers.items()
        if name != "base" and hasattr(controller, "target_pos")
    }


def max_target_drift(reference: dict[str, np.ndarray], robot) -> tuple[float, dict[str, float]]:
    per_group = {}
    for name, start in reference.items():
        current = robot.controllers[name].target_pos
        per_group[name] = float(np.max(np.abs(current - start))) if len(start) else 0.0
    return max(per_group.values(), default=0.0), per_group


def synchronize_navigation_start(
    adapter: LastMileNavAdapter,
    control_steps=25,
    target_pos_tolerance=1e-3,
    target_rot_tolerance=np.deg2rad(1.0),
):
    """Explicitly settle position controllers at the teleported start before timing navigation."""
    task, env, robot = adapter.task, adapter.env, adapter.env.current_robot
    target_before = object_pose(adapter.target)
    base_before = se2_pose(robot.robot_view)
    # ``Robot.set_stationary`` intentionally skips controllers already marked
    # stationary.  Teleporting the base changes the coupled equilibrium, so force
    # every target to the current configuration before and after this explicit sync.
    for controller in robot.controllers.values():
        controller.set_to_stationary()
    events = []
    for ctrl_step in range(control_steps):
        robot.compute_control()
        for sim_step in range(int(task._n_sim_steps_per_ctrl)):
            mujoco.mj_step(env.current_model, env.current_data)
            contacts = illegal_contacts(env, task.config.robot_config.robot_namespace)
            pos_delta, rot_delta = pose_change(target_before, object_pose(adapter.target))
            if not np.isfinite(env.current_data.qpos).all() or not np.isfinite(env.current_data.qvel).all():
                events.append({"type": "controller_error", "detail": "nonfinite_start_sync",
                               "ctrl_step": ctrl_step, "sim_step": sim_step})
            if contacts:
                events.append({"type": "collision", "contacts": contacts,
                               "ctrl_step": ctrl_step, "sim_step": sim_step})
            if pos_delta > target_pos_tolerance or rot_delta > target_rot_tolerance:
                events.append({"type": "scene_changed", "target_position_delta": pos_delta,
                               "target_rotation_delta": rot_delta, "ctrl_step": ctrl_step,
                               "sim_step": sim_step})
            if events:
                break
        if events:
            break
    for manager in env.object_managers:
        manager.invalidate_data_cache()
    for controller in robot.controllers.values():
        controller.set_to_stationary()
    robot.compute_control()
    pos_delta, rot_delta = pose_change(target_before, object_pose(adapter.target))
    return {
        "control_steps": control_steps,
        "duration_sec": control_steps * task._ctrl_dt_ms / 1000.0,
        "base_change_se2": (se2_pose(robot.robot_view) - base_before).tolist(),
        "target_position_delta": pos_delta,
        "target_rotation_delta": rot_delta,
        "events": events,
    }


def settle_before_handover(
    adapter: LastMileNavAdapter,
    joints_start: dict[str, np.ndarray],
    control_steps=150,
    joint_transient_tolerance=2e-1,
    target_pos_tolerance=1e-3,
    target_rot_tolerance=np.deg2rad(1.0),
):
    """导航结束后停住底盘，等上半身弹性瞬态衰减，再读非 base 关节的稳态偏差。

    A* 的 waypoint 是阶跃指令，实测幅度为 Δxy 0.12-0.15 m、Δyaw 10-20°/100 ms
    （不是 0.8 m 截断半径暗示的 0.3 m/s）。这种阶跃会激励 torso 这类承载上半身
    惯量的关节产生弹性挠度：独立模型复现下峰值 ~0.055 rad，真实 P1-E02 里
    0.033-0.050 rad。它是有限刚度位置伺服（torso kp=4000）在底盘运动下的必然
    响应，不是指令被改写——`max_nonbase_command_drift` 恒为 0。

    所以「非 base 关节是否被导航改变」分两层判定：运动期间只受瞬态上限约束，
    稳态偏差在底盘停稳后再读。测得停住后约 1.0 s 才回到 1e-3 rad 以下、2 s 到
    1e-4，所以默认给 3.0 s 窗口（150 个控制步）。

    稳定期间不做任何新指令（底盘沿用 `set_to_stationary` 的当前位姿），其余检查
    （非法碰撞、目标位移、非有限状态、瞬态上限）与导航期间一致。
    返回 (事件列表, 稳态偏差, 逐组偏差, 逐控制步的偏差曲线)。
    """
    task, env = adapter.task, adapter.env
    robot = env.current_robot
    sim_steps_per_ctrl = int(task._n_sim_steps_per_ctrl)
    target_start = object_pose(adapter.target)
    joints_start = {name: value.copy() for name, value in joints_start.items()}
    events = []
    trace = []
    for ctrl_step in range(control_steps):
        robot.compute_control()
        for sim_step in range(sim_steps_per_ctrl):
            mujoco.mj_step(env.current_model, env.current_data)
            finite = np.isfinite(env.current_data.qpos).all() and np.isfinite(env.current_data.qvel).all()
            contacts = illegal_contacts(env, task.config.robot_config.robot_namespace)
            pos_delta, rot_delta = pose_change(target_start, object_pose(adapter.target))
            joint_drift, joint_per_group = max_nonbase_drift(joints_start, robot)
            if not finite:
                events.append({"type": "controller_error", "detail": "nonfinite_physics_state",
                               "ctrl_step": ctrl_step, "sim_step": sim_step})
            if joint_drift > joint_transient_tolerance:
                events.append({"type": "controller_error", "detail": "nonbase_joint_transient",
                               "max_drift": joint_drift, "per_group": joint_per_group,
                               "ctrl_step": ctrl_step, "sim_step": sim_step})
            if contacts:
                events.append({"type": "collision", "contacts": contacts,
                               "ctrl_step": ctrl_step, "sim_step": sim_step})
            if pos_delta > target_pos_tolerance or rot_delta > target_rot_tolerance:
                events.append({"type": "scene_changed", "target_position_delta": pos_delta,
                               "target_rotation_delta": rot_delta,
                               "ctrl_step": ctrl_step, "sim_step": sim_step})
            if events:
                break
        if events:
            break
        trace.append(max_nonbase_drift(joints_start, robot)[0])
    for manager in env.object_managers:
        manager.invalidate_data_cache()
    settled_drift, settled_per_group = max_nonbase_drift(joints_start, robot)
    return events, settled_drift, settled_per_group, trace


def execute_navigation(
    adapter: LastMileNavAdapter,
    policy,
    max_policy_steps=600,
    endpoint_tolerance=0.05,
    joint_drift_tolerance=1e-3,
    joint_transient_tolerance=2e-1,
    settle_control_steps=150,
    target_pos_tolerance=1e-3,
    target_rot_tolerance=np.deg2rad(1.0),
):
    """Execute base actions while monitoring every MuJoCo integration step.

    非 base 关节的固定性用两层判据：
    - 导航期间：任何非 base 控制器目标被改写即失败（0 容差，无噪声），
      物理关节位置的偏差另有 `joint_transient_tolerance` 的瞬态上限做护栏；
    - 底盘停稳后：`joint_drift_tolerance` 判稳态偏差，这才是交给评价器的姿态。
    """
    task, env = adapter.task, adapter.env
    robot = env.current_robot
    sim_steps_per_ctrl = int(task._n_sim_steps_per_ctrl)
    ctrl_steps_per_policy = int(task._n_ctrl_steps_per_policy)
    target_start = object_pose(adapter.target)
    joints_start = nonbase_positions(robot)
    targets_start = nonbase_targets(robot)
    max_observed_joint_drift = 0.0
    max_observed_joint_drift_by_group = {name: 0.0 for name in joints_start}
    trajectory = [se2_pose(robot.robot_view)]
    desired = [np.full(3, np.nan)]
    events = []
    status = None
    detail = None

    for policy_step in range(max_policy_steps + 1):
        try:
            action = policy.get_action(None)
        except Exception as exc:
            status, detail = "controller_error", f"policy:{type(exc).__name__}:{exc}"
            events.append({"type": status, "detail": detail, "policy_step": policy_step})
            break
        if action is None:
            status, detail = "controller_error", "policy_returned_none"
            break
        if action.get("done", False):
            status = policy.termination_reason or "controller_error"
            detail = policy.termination_detail or "missing_policy_termination_reason"
            break
        if policy_step >= max_policy_steps:
            status, detail = "timeout", "policy_step_budget_exhausted"
            break

        waypoint = np.asarray(action["base"], dtype=float)
        step_observations = []
        try:
            robot.update_control({"base": waypoint})
            # 「非 base 关节固定」的精确判据是控制器目标未被改写：这个量没有数值
            # 噪声，用 0 容差。任何一次重新 set_target（包括 update_control 的
            # stationary 分支把目标重设成当前关节角）都说明姿态已不是导航开始时那个。
            command_drift, command_per_group = max_target_drift(targets_start, robot)
            if command_drift > 0.0:
                step_observations.append(("controller_error", {
                    "detail": "nonbase_command_changed", "max_command_drift": command_drift,
                    "per_group": command_per_group, "ctrl_step": 0, "sim_step": 0,
                }))
            for ctrl_step in range(ctrl_steps_per_policy):
                if step_observations:
                    break
                robot.compute_control()
                for sim_step in range(sim_steps_per_ctrl):
                    mujoco.mj_step(env.current_model, env.current_data)
                    finite = np.isfinite(env.current_data.qpos).all() and np.isfinite(env.current_data.qvel).all()
                    contacts = illegal_contacts(env, task.config.robot_config.robot_namespace)
                    pos_delta, rot_delta = pose_change(target_start, object_pose(adapter.target))
                    joint_drift, joint_per_group = max_nonbase_drift(joints_start, robot)
                    max_observed_joint_drift = max(max_observed_joint_drift, joint_drift)
                    for name, value in joint_per_group.items():
                        max_observed_joint_drift_by_group[name] = max(
                            max_observed_joint_drift_by_group[name], value
                        )
                    if not finite:
                        step_observations.append(("controller_error", {
                            "detail": "nonfinite_physics_state", "ctrl_step": ctrl_step,
                            "sim_step": sim_step,
                        }))
                    if joint_drift > joint_transient_tolerance:
                        step_observations.append(("controller_error", {
                            "detail": "nonbase_joint_transient", "max_drift": joint_drift,
                            "per_group": joint_per_group, "ctrl_step": ctrl_step,
                            "sim_step": sim_step,
                        }))
                    if contacts:
                        step_observations.append(("collision", {
                            "contacts": contacts, "ctrl_step": ctrl_step, "sim_step": sim_step,
                        }))
                    if pos_delta > target_pos_tolerance or rot_delta > target_rot_tolerance:
                        step_observations.append(("scene_changed", {
                            "target_position_delta": pos_delta, "target_rotation_delta": rot_delta,
                            "ctrl_step": ctrl_step, "sim_step": sim_step,
                        }))
                    if step_observations:
                        break
                if step_observations:
                    break
        except Exception as exc:
            step_observations = [("controller_error", {
                "detail": f"execution:{type(exc).__name__}:{exc}",
            })]

        adapter.step_count += 1
        for manager in env.object_managers:
            manager.invalidate_data_cache()
        trajectory.append(se2_pose(robot.robot_view))
        desired.append(waypoint.copy())
        actual_drift, actual_per_group = max_nonbase_drift(joints_start, robot)
        max_observed_joint_drift = max(max_observed_joint_drift, actual_drift)
        for name, value in actual_per_group.items():
            max_observed_joint_drift_by_group[name] = max(
                max_observed_joint_drift_by_group[name], value
            )
        if step_observations:
            priority = {"controller_error": 0, "collision": 1, "scene_changed": 2}
            step_observations.sort(key=lambda item: priority[item[0]])
            status, event = step_observations[0]
            detail = event.get("detail", status)
            events.extend({"type": kind, "policy_step": policy_step, **payload}
                          for kind, payload in step_observations)
            break

    # 导航结束：把底盘就地停住，等上半身的弹性瞬态衰减掉，再判「非 base 关节
    # 是否被导航改变」。运动期间的峰值只进 artifact 供审计。
    settle = None
    settled_drift = None
    settled_per_group = None
    if status == "completed" and settle_control_steps > 0:
        robot.controllers["base"].set_to_stationary()
        settle_events, settled_drift, settled_per_group, settle_trace = settle_before_handover(
            adapter, joints_start,
            control_steps=settle_control_steps,
            joint_transient_tolerance=joint_transient_tolerance,
            target_pos_tolerance=target_pos_tolerance,
            target_rot_tolerance=target_rot_tolerance,
        )
        events.extend({"phase": "settle", **event} for event in settle_events)
        if settle_events:
            priority = {"controller_error": 0, "collision": 1, "scene_changed": 2}
            event = sorted(settle_events, key=lambda item: priority[item["type"]])[0]
            status, detail = event["type"], event.get("detail", event["type"])
        elif settled_drift > joint_drift_tolerance:
            status, detail = "controller_error", "nonbase_joint_drift"
            events.append({"phase": "settle", "type": status, "detail": detail,
                           "max_drift": settled_drift, "per_group": settled_per_group})
        settle = {
            "control_steps": settle_control_steps,
            "duration_sec": (
                settle_control_steps * sim_steps_per_ctrl * float(env.current_model.opt.timestep)
            ),
            "joint_drift_tolerance": joint_drift_tolerance,
            "max_settled_joint_drift": settled_drift,
            "settled_joint_drift": settled_per_group,
            # 每个控制步采一个点，供审计收敛速度；超限提前结束时会短于 control_steps
            "drift_trace": [float(value) for value in settle_trace],
            "events": settle_events,
        }

    actual = se2_pose(robot.robot_view)
    planned = policy.planned_endpoint
    endpoint_translation_error = None
    endpoint_yaw_error = None
    endpoint_se2_error = None
    if planned is not None:
        endpoint_translation_error = float(np.linalg.norm(actual[:2] - planned[:2]))
        endpoint_yaw_error = float(abs(normalize_ang_error(actual[2] - planned[2])))
        endpoint_se2_error = float(np.linalg.norm([
            actual[0] - planned[0], actual[1] - planned[1],
            normalize_ang_error(actual[2] - planned[2]),
        ]))
    actual_drift, actual_per_group = max_nonbase_drift(joints_start, robot)
    max_observed_joint_drift = max(max_observed_joint_drift, actual_drift)
    for name, value in actual_per_group.items():
        max_observed_joint_drift_by_group[name] = max(
            max_observed_joint_drift_by_group[name], value
        )
    command_drift, command_per_group = max_target_drift(targets_start, robot)
    pos_delta, rot_delta = pose_change(target_start, object_pose(adapter.target))
    if status == "completed" and (endpoint_se2_error is None or endpoint_se2_error >= endpoint_tolerance):
        status, detail = "controller_error", "endpoint_tolerance_violation"
    assert status in TERMINAL_STATUSES - {"start_unavailable"}
    return {
        "status": status,
        "termination_detail": detail,
        "policy_steps": adapter.step_count,
        "actual_endpoint": actual.tolist(),
        "planned_endpoint": None if planned is None else planned.tolist(),
        "endpoint_translation_error": endpoint_translation_error,
        "endpoint_yaw_error": endpoint_yaw_error,
        "endpoint_se2_error": endpoint_se2_error,
        "target_position_delta": pos_delta,
        "target_rotation_delta": rot_delta,
        "max_nonbase_command_drift": command_drift,
        "nonbase_command_drift": command_per_group,
        # 运动期间的峰值挠度（弹性瞬态，供审计与护栏核对）
        "max_nonbase_joint_drift": max_observed_joint_drift,
        "nonbase_joint_drift": max_observed_joint_drift_by_group,
        "nonbase_joint_transient_tolerance": joint_transient_tolerance,
        # 底盘停稳后的稳态偏差（这才是交给评价器的姿态）
        "max_settled_nonbase_joint_drift": settled_drift,
        "settled_nonbase_joint_drift": settled_per_group,
        "settle": settle,
        "events": events,
        "policy_info": policy.get_info(),
    }, np.asarray(trajectory), np.asarray(desired)
