"""P2 evaluator 的无资产回归测试。"""

from contextlib import contextmanager
from types import SimpleNamespace
import hashlib
import sys

import mujoco
import numpy as np

from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.evaluation.last_mile.feasibility import (
    FeasibilityBudget, GraspPool, ManipulationFeasibilityEvaluator,
)


class Base:
    def __init__(self):
        self._pose = np.eye(4)

    @property
    def pose(self):
        return self._pose.copy()

    @pose.setter
    def pose(self, value):
        self._pose = np.asarray(value).copy()


class Group:
    def __init__(self, data, addresses, root_body_id, site_id=None, limits=None):
        self.data = data
        self.addresses = list(addresses)
        self.root_body_id = root_body_id
        self.site_id = site_id
        self._limits = np.asarray(limits if limits is not None else [], dtype=float).reshape(-1, 2)

    @property
    def joint_pos(self):
        return self.data.qpos[self.addresses].copy()

    @joint_pos.setter
    def joint_pos(self, value):
        if self.addresses:
            self.data.qpos[self.addresses] = value

    @property
    def joint_pos_limits(self):
        return self._limits.copy()

    @property
    def leaf_frame_to_world(self):
        pose = np.eye(4)
        if self.site_id is not None:
            pose[:3, 3] = self.data.site_xpos[self.site_id]
            pose[:3, :3] = self.data.site_xmat[self.site_id].reshape(3, 3)
        return pose


class View:
    def __init__(self, model, data):
        robot_body = int(model.body("robot_0/root").id)
        finger_body = int(model.body("robot_0/finger_tip").id)
        self.base = Base()
        self.root_body_id = robot_body
        self.groups = {
            "base": Group(data, [], robot_body),
            "arm": Group(data, [0, 1, 2], robot_body,
                         limits=[[-1, 1], [-1, 1], [0.05, 1.2]]),
            "gripper": Group(data, [], finger_body, int(model.site("tcp").id)),
        }

    def move_group_ids(self):
        return list(self.groups)

    def get_gripper_movegroup_ids(self):
        return ["gripper"]

    def get_move_group(self, name):
        return self.groups[name]

    def get_qpos_dict(self):
        return {name: group.joint_pos for name, group in self.groups.items()}

    def set_qpos_dict(self, values):
        for name, value in values.items():
            self.groups[name].joint_pos = value


class Kinematics:
    def __init__(self, view, model, data, unreachable=False):
        self.view, self.model, self.data = view, model, data
        self.unreachable = unreachable

    def ik(self, _gripper, pose, unlocked, q0, _base_pose, **_kwargs):
        assert unlocked == ["arm"]
        xyz = np.asarray(pose[:3, 3])
        if self.unreachable or np.any(np.abs(xyz[:2]) > 1) or not 0.05 <= xyz[2] <= 1.2:
            return None
        result = {key: np.asarray(value).copy() for key, value in q0.items()}
        result["arm"] = xyz.copy()
        self.view.set_qpos_dict(result)
        mujoco.mj_forward(self.model, self.data)
        return result


class BaseController:
    target_pos = np.zeros(3)

    def set_to_stationary(self):
        pass

    def compute_ctrl_inputs(self):
        return np.zeros(0)

    @property
    def robot_move_group(self):
        return SimpleNamespace(ctrl=np.zeros(0))


class Snapshot:
    def __init__(self, task):
        self.task = task
        self.qpos = task.env.current_data.qpos.copy()
        self.base = task.env.current_robot.robot_view.base.pose.copy()

    def restore(self, task):
        task.env.current_data.qpos[:] = self.qpos
        task.env.current_robot.robot_view.base.pose = self.base
        mujoco.mj_forward(task.env.current_model, task.env.current_data)

    @contextmanager
    def restored(self, task):
        self.restore(task)
        try:
            yield
        finally:
            self.restore(task)


def make_task(obstacle=None, unreachable=False):
    obstacle_xml = ""
    if obstacle is not None:
        x, y, z, radius = obstacle
        obstacle_xml = f'<body name="obstacle" pos="{x} {y} {z}"><geom name="obstacle_geom" type="sphere" size="{radius}"/></body>'
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option gravity="0 0 0"/>
      <worldbody>
        {obstacle_xml}
        <body name="robot_0/root">
          <joint name="x" type="slide" axis="1 0 0" range="-1 1"/>
          <joint name="y" type="slide" axis="0 1 0" range="-1 1"/>
          <joint name="z" type="slide" axis="0 0 1" range=".05 1.2"/>
          <body name="robot_0/finger_tip"><geom name="finger" type="sphere" size=".02"/><site name="tcp"/></body>
        </body>
        <body name="target" pos=".4 0 .4"><freejoint/><geom name="target_geom" type="sphere" size=".04" mass="1"/></body>
      </worldbody>
    </mujoco>''')
    data = mujoco.MjData(model)
    # arm qpos occupies [0:3], target free joint starts at 3
    data.qpos[:3] = [0.0, 0.0, 0.8]
    mujoco.mj_forward(model, data)
    view = View(model, data)
    robot = SimpleNamespace(robot_view=view, controllers={"base": BaseController()},
                            _last_unnoised_cmd_joint_pos=None)
    robot.kinematics = Kinematics(view, model, data, unreachable)
    env = SimpleNamespace(current_model=model, current_data=data, current_robot=robot)
    task = SimpleNamespace(env=env)
    evaluator = ManipulationFeasibilityEvaluator(task)
    # toy base 不在 MuJoCo qpos 中，绕过生产环境的控制器同步。
    evaluator._set_base_pose = lambda _pose: None
    target = MlSpacesObject("target", data)
    grasp = np.eye(4)
    # object-local grasp：tcp 与物体中心重合，接近轴 +Z。
    pool = GraspPool(np.asarray([grasp]), (0,), hashlib.sha256(grasp.tobytes()).hexdigest())
    budget = FeasibilityBudget(max_candidates=1, ik_seeds_per_arm=3, timeout_sec=5,
                               pregrasp_standoff_m=.10, lift_height_m=.05)
    return task, evaluator, target, pool, budget, Snapshot(task)


def run_case(obstacle=None, unreachable=False):
    task, evaluator, target, pool, budget, snapshot = make_task(obstacle, unreachable)
    before = task.env.current_data.qpos.copy()
    result = evaluator.evaluate(snapshot, task.env.current_robot, np.eye(4), target, pool, budget)
    np.testing.assert_array_equal(task.env.current_data.qpos, before)
    return result, (task, evaluator, target, pool, budget, snapshot)


def test_known_reachable_and_legal_finger_target_contact():
    result, _ = run_case()
    assert result["status"] == "feasible"
    assert result["layer_pass_counts"] == {
        "F_base": 1, "F_IK": 3, "F_approach": 3, "F_lift_proxy": 3,
    }
    assert len(result["candidate_diagnostics"]) == 3
    assert result["witness"]["candidate_id"] == 0


def test_known_unreachable_is_protocol_not_found_not_unknown():
    result, _ = run_case(unreachable=True)
    assert result["status"] == "not_found"
    assert result["first_failure_layer"] == "F_IK"
    assert len(result["candidate_diagnostics"]) == 3
    assert all(row["pregrasp_residual"] is None
               for row in result["candidate_diagnostics"])
    assert "finite protocol" in result["search_semantics"]


def test_initial_robot_collision_fails_at_base():
    result, _ = run_case(obstacle=(0.0, 0.0, 0.8, 0.04))
    assert result["status"] == "not_found"
    assert result["first_failure_layer"] == "F_base"
    assert result["base_collisions"]


def test_mid_path_collision_fails_at_approach():
    # current z=.8 -> pregrasp z=.3，障碍物位于中间；目标本身不与障碍接触。
    result, _ = run_case(obstacle=(0.2, 0.0, 0.55, 0.04))
    assert result["status"] == "not_found"
    assert result["first_failure_layer"] == "F_approach"
    assert any(row.get("failure_reason") == "to_pregrasp_collision"
               for row in result["candidate_diagnostics"])


def test_attached_target_sweep_collision_fails_only_at_lift():
    # 抬升 5 cm 时，半径 4 cm 的目标撞到上方障碍；半径 2 cm 的手指不撞。
    result, _ = run_case(obstacle=(0.4, 0.0, 0.485, 0.02))
    assert result["status"] == "not_found"
    assert result["layer_pass_counts"]["F_approach"] > 0
    assert result["first_failure_layer"] == "F_lift_proxy"
    assert any(row.get("failure_layer") == "F_lift_proxy"
               for row in result["candidate_diagnostics"])


def test_repeatability_state_pollution_and_no_curobo_import():
    _, case = run_case()
    task, evaluator, target, pool, budget, snapshot = case
    first = evaluator.evaluate(snapshot, task.env.current_robot, np.eye(4), target, pool, budget)
    second = evaluator.evaluate(snapshot, task.env.current_robot, np.eye(4), target, pool, budget)
    for result in (first, second):
        result.pop("elapsed_sec", None)
    assert first == second
    assert not any(name.startswith("curobo") for name in sys.modules)


def test_grasp_pool_is_object_relative_and_id_stable():
    pose = np.eye(4)
    pose[0, 3] = .1
    pool = GraspPool(np.asarray([pose]), (7,), "fixed")
    target_a = np.eye(4)
    target_b = np.eye(4)
    target_b[1, 3] = 2.0
    np.testing.assert_allclose(pool.world_poses(target_a)[0], pose)
    np.testing.assert_allclose(pool.world_poses(target_b)[0], target_b @ pose)
    assert pool.candidate_ids == (7,)
