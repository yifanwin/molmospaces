"""Asset-free regression tests for the P1 navigation protocol."""

from types import SimpleNamespace

import mujoco
import networkx as nx
import numpy as np

from molmo_spaces.evaluation.last_mile.nav_adapter import (
    LastMileNavAdapter, derive_seed, execute_navigation, illegal_contacts,
    pose_change, sample_remote_start,
)
from molmo_spaces.evaluation.last_mile.run_p1 import atomic_npz, validate_cached_result
from molmo_spaces.policy.solvers.navigation.astar_planner_policy import AStarPlannerPolicy


class Base:
    def __init__(self):
        self._pose = np.eye(4)

    @property
    def pose(self):
        return self._pose.copy()

    @pose.setter
    def pose(self, value):
        self._pose = np.asarray(value).copy()


class MoveGroup:
    def __init__(self, values):
        self.joint_pos = np.asarray(values, dtype=float)


class RobotView:
    def __init__(self):
        self.base = Base()
        self.groups = {"arm": MoveGroup([0.2, -0.1])}

    def get_move_group(self, name):
        return self.groups[name]


class BaseController:
    def set_to_stationary(self):
        pass


class Robot:
    def __init__(self):
        self.robot_view = RobotView()
        self.controllers = {"base": BaseController(), "arm": object()}

    def update_control(self, action):
        pose = np.eye(4)
        pose[:2, 3] = action["base"][:2]
        yaw = action["base"][2]
        pose[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
        self.robot_view.base.pose = pose

    def compute_control(self):
        pass


class Policy:
    def __init__(self, endpoint):
        self.planned_endpoint = np.asarray(endpoint)
        self.termination_reason = "completed"
        self.termination_detail = "all_waypoints_reached"
        self.calls = 0

    def get_action(self, _observation):
        self.calls += 1
        if self.calls == 1:
            return {"done": False, "base": self.planned_endpoint.copy()}
        return {"done": True, "base": self.planned_endpoint.copy()}

    def get_info(self):
        return {"termination_reason": self.termination_reason}


class NeverDonePolicy(Policy):
    def get_action(self, _observation):
        self.calls += 1
        return {"done": False, "base": self.planned_endpoint.copy()}


class NoPathPolicy(Policy):
    termination_reason = "no_path"
    termination_detail = "astar_no_path"

    def __init__(self, endpoint):
        super().__init__(endpoint)
        self.termination_reason = "no_path"
        self.termination_detail = "astar_no_path"

    def get_action(self, _observation):
        return {"done": True, "base": self.planned_endpoint.copy()}


def tiny_task():
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option gravity="0 0 0" timestep="0.004"/>
      <worldbody>
        <body name="target" pos="2 0 0.2"><geom type="sphere" size=".05" contype="0" conaffinity="0"/></body>
      </worldbody>
    </mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    robot = Robot()
    env = SimpleNamespace(current_model=model, current_data=data, current_robot=robot,
                          current_batch_index=0, object_managers=[])
    config = SimpleNamespace(robot_config=SimpleNamespace(robot_namespace="robot_0/"))
    return SimpleNamespace(env=env, config=config, _n_sim_steps_per_ctrl=1,
                           _n_ctrl_steps_per_policy=1)


def test_seed_derivation_is_stable_and_namespaced():
    assert derive_seed(123, "p1_goal") == derive_seed(123, "p1_goal")
    assert derive_seed(123, "p1_goal") != derive_seed(123, "p1_start")


def test_adapter_exposes_only_frozen_target():
    task = tiny_task()
    adapter = LastMileNavAdapter(task, "target")
    assert adapter.nav_objs == [[adapter.target]]
    assert adapter.get_nav_object_priority(0) == [adapter.target]
    assert adapter.num_steps_taken() == 0
    assert not hasattr(adapter, "judge_success")


def test_execute_navigation_uses_policy_done_not_pnp_success():
    task = tiny_task()
    adapter = LastMileNavAdapter(task, "target")
    result, trajectory, desired = execute_navigation(adapter, Policy([0.2, 0.0, 0.0]), 5)
    assert result["status"] == "completed"
    assert result["policy_steps"] == 1
    assert result["endpoint_se2_error"] == 0
    assert trajectory.shape == desired.shape == (2, 3)


def test_execute_navigation_classifies_settled_joint_drift():
    """物理偏差在运动期间只是弹性瞬态；底盘停稳后仍超 1e-3 rad 才判失败。"""
    task = tiny_task()
    adapter = LastMileNavAdapter(task, "target")
    original = task.env.current_robot.update_control

    def update_with_drift(action):
        original(action)
        task.env.current_robot.robot_view.groups["arm"].joint_pos[0] += 0.0011

    task.env.current_robot.update_control = update_with_drift
    result, _, _ = execute_navigation(adapter, Policy([0.2, 0.0, 0.0]), 5)
    assert result["status"] == "controller_error"
    assert result["termination_detail"] == "nonbase_joint_drift"
    assert result["max_settled_nonbase_joint_drift"] > 0.001
    assert result["events"][-1]["phase"] == "settle"


def test_execute_navigation_transient_ceiling_is_above_the_frozen_tolerance():
    """瞬态上限（0.2 rad）远高于冻结的 1e-3；只有量级上像真故障的偏差才在运动期间失败。"""
    task = tiny_task()
    original = task.env.current_robot.update_control

    def update_with_big_transient(action):
        original(action)
        task.env.current_robot.robot_view.groups["arm"].joint_pos[0] += 0.3

    task.env.current_robot.update_control = update_with_big_transient
    result, _, _ = execute_navigation(LastMileNavAdapter(task, "target"), Policy([0.2, 0.0, 0.0]), 5)
    assert result["status"] == "controller_error"
    assert result["termination_detail"] == "nonbase_joint_transient"
    assert result["max_nonbase_joint_drift"] > 0.2
    assert result["max_settled_nonbase_joint_drift"] is None


def test_execute_navigation_rejects_rewritten_nonbase_target():
    """「非 base 关节固定」的精确判据：控制器目标不得被改写，0 容差。"""
    task = tiny_task()

    class TargetPosController:
        def __init__(self):
            self.target = np.zeros(2)

        @property
        def target_pos(self):
            return self.target.copy()

        @property
        def stationary(self):
            return True

        def set_to_stationary(self):
            pass

    controller = TargetPosController()
    task.env.current_robot.controllers["arm"] = controller
    original = task.env.current_robot.update_control

    def update_and_rewrite(action):
        original(action)
        controller.target[0] += 0.01

    task.env.current_robot.update_control = update_and_rewrite
    result, _, _ = execute_navigation(LastMileNavAdapter(task, "target"), Policy([0.2, 0.0, 0.0]), 5)
    assert result["status"] == "controller_error"
    assert result["termination_detail"] == "nonbase_command_changed"
    assert result["max_nonbase_command_drift"] > 0.0
    # 物理关节位置没有被动过，说明失败来自指令而不是动力学
    assert result["max_nonbase_joint_drift"] == 0.0


def test_execute_navigation_timeout_and_no_path():
    task = tiny_task()
    result, _, _ = execute_navigation(
        LastMileNavAdapter(task, "target"), NeverDonePolicy([0.2, 0.0, 0.0]), 1
    )
    assert (result["status"], result["termination_detail"]) == (
        "timeout", "policy_step_budget_exhausted"
    )
    task = tiny_task()
    result, _, _ = execute_navigation(
        LastMileNavAdapter(task, "target"), NoPathPolicy([0.2, 0.0, 0.0]), 5
    )
    assert (result["status"], result["termination_detail"]) == (
        "no_path", "astar_no_path"
    )


def test_execute_navigation_collision_scene_change_and_priority(monkeypatch):
    import molmo_spaces.evaluation.last_mile.nav_adapter as module

    task = tiny_task()
    monkeypatch.setattr(module, "illegal_contacts", lambda *_: [{"body": "obstacle"}])
    result, _, _ = execute_navigation(
        LastMileNavAdapter(task, "target"), Policy([0.2, 0.0, 0.0]), 5
    )
    assert result["status"] == "collision"

    task = tiny_task()
    monkeypatch.setattr(module, "illegal_contacts", lambda *_: [])
    monkeypatch.setattr(module, "pose_change", lambda *_: (0.0011, 0.0))
    result, _, _ = execute_navigation(
        LastMileNavAdapter(task, "target"), Policy([0.2, 0.0, 0.0]), 5
    )
    assert result["status"] == "scene_changed"

    task = tiny_task()
    original = task.env.current_robot.update_control
    def update_with_drift(action):
        original(action)
        task.env.current_robot.robot_view.groups["arm"].joint_pos[0] += 0.3
    task.env.current_robot.update_control = update_with_drift
    monkeypatch.setattr(module, "illegal_contacts", lambda *_: [{"body": "obstacle"}])
    result, _, _ = execute_navigation(
        LastMileNavAdapter(task, "target"), Policy([0.2, 0.0, 0.0]), 5
    )
    assert result["status"] == "controller_error"
    assert {event["type"] for event in result["events"]} == {
        "controller_error", "collision", "scene_changed"
    }


def test_target_pose_threshold_math():
    a = np.eye(4)
    b = np.eye(4)
    b[0, 3] = 0.0011
    angle = np.deg2rad(1.1)
    b[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    pos, rot = pose_change(a, b)
    assert pos > 0.001 and rot > np.deg2rad(1)


def test_remote_start_is_seeded_connected_and_collision_checked():
    class Map:
        def get_free_points(self):
            return np.array([[0, 0, 0], [2, 0, 0], [2.5, 0, 0], [8, 0, 0]], dtype=float)

    class Planner:
        map = Map()
        graph = nx.path_graph([(0, 0), (1, 0), (2, 0), (3, 0)])

        def get_discrete_location(self, value):
            return (int(round(np.asarray(value)[0])), 0)

    robot = SimpleNamespace(robot_view=RobotView())
    checked = []
    env = SimpleNamespace(current_robot=robot)
    def collision(_view, pose, _namespace):
        checked.append(pose.copy())
        return False
    env.check_if_robot_collision_at_base_pose = collision
    task = SimpleNamespace(env=env, config=SimpleNamespace(
        robot_config=SimpleNamespace(robot_namespace="robot_0/")))
    first, info1 = sample_remote_start(task, Planner(), [0, 0], np.array([0, 0]), 9)
    second, info2 = sample_remote_start(task, Planner(), [0, 0], np.array([0, 0]), 9)
    np.testing.assert_array_equal(first, second)
    assert 1.5 <= info1["distance_to_target"] <= 3.0
    assert info1 == info2 and checked


def test_remote_start_stops_after_100_collision_rejections():
    class Map:
        def get_free_points(self):
            return np.c_[np.linspace(1.5, 3.0, 200), np.zeros(200), np.zeros(200)]
    class Planner:
        map = Map()
        graph = nx.complete_graph([(i, 0) for i in range(201)])
        def get_discrete_location(self, value):
            return (int(round((np.asarray(value)[0] - 1.5) / 1.5 * 199)), 0)
    task = SimpleNamespace(
        env=SimpleNamespace(
            current_robot=SimpleNamespace(robot_view=RobotView()),
            check_if_robot_collision_at_base_pose=lambda *_: True,
        ),
        config=SimpleNamespace(robot_config=SimpleNamespace(robot_namespace="robot_0/")),
    )
    pose, info = sample_remote_start(task, Planner(), [1.5, 0], np.array([0, 0]), 7)
    assert pose is None
    assert info["attempts"] == info["collision_rejected"] == 100


def test_atomic_artifact_and_hash_rejection(tmp_path):
    path = tmp_path / "trajectory.npz"
    atomic_npz(path, actual_se2=np.zeros((1, 3)))
    import hashlib
    good = hashlib.sha256(path.read_bytes()).hexdigest()
    validate_cached_result(
        {"status": "timeout", "artifact_sha256": {"trajectory.npz": good}}, tmp_path
    )
    path.write_bytes(b"changed")
    import pytest
    with pytest.raises(ValueError, match="哈希不匹配"):
        validate_cached_result(
            {"status": "timeout", "artifact_sha256": {"trajectory.npz": good}}, tmp_path
        )


def test_illegal_contact_allows_floor_but_rejects_obstacle():
    model = mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0"/>
      <worldbody>
        <geom name="floor_geom" type="plane" size="2 2 .1"/>
        <body name="robot_0/base" pos="0 0 .1"><freejoint/><geom name="robot_geom" type="sphere" size=".2" mass="1"/></body>
        <body name="cabinet" pos=".25 0 .1"><geom name="cabinet_geom" type="sphere" size=".2"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rows = illegal_contacts(SimpleNamespace(current_model=model, current_data=data), "robot_0/")
    assert any(row["root2"] == "cabinet" or row["root1"] == "cabinet" for row in rows)
    assert all("floor" not in (row["root1"] + row["root2"]).lower() for row in rows)


def test_astar_diagnostics_are_json_serializable():
    policy = AStarPlannerPolicy.__new__(AStarPlannerPolicy)
    policy._termination_reason = "completed"
    policy._termination_detail = "all_waypoints_reached"
    policy._target_pos_quat = (np.array([1.0, 2.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))
    policy._nav_plan = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.2]])
    policy._reached_waypoints = 2
    policy._retries_left = 3
    policy.config = SimpleNamespace(policy_config=SimpleNamespace(plan_max_retries=3))
    info = policy.get_info()
    assert info["planned_endpoint"] == [1.0, 2.0, 0.2]
    assert info["termination_reason"] == "completed"
