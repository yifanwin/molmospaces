"""Integration tests for the optional robosuite PandaOmron adapter."""

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

pytest.importorskip("robosuite", minversion="1.5.2")

from molmo_spaces.configs.robot_configs import PandaOmronRobotConfig
from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.kinematics.parallel.warp_kinematics import SimpleWarpKinematics


def _compile_robot(pos=(0.0, 0.0), yaw=0.0, strip_meshes=False):
    config = PandaOmronRobotConfig()
    spec = mujoco.MjSpec()
    quat = R.from_euler("z", yaw).as_quat(scalar_first=True)
    config.robot_cls.add_robot_to_scene(
        config,
        spec,
        config.robot_namespace,
        list(pos),
        quat,
        strip_meshes=strip_meshes,
    )
    config.robot_cls.apply_control_overrides(spec, config)
    model = spec.compile()
    data = mujoco.MjData(model)
    view = config.robot_view_factory(data, config.robot_namespace)
    return config, model, data, view


def test_model_compiles_and_exposes_expected_move_groups():
    config, model, data, view = _compile_robot()
    assert view.move_group_ids() == ["base", "torso", "arm", "gripper"]
    assert [view.get_move_group(name).pos_dim for name in view.move_group_ids()] == [3, 1, 7, 2]
    assert model.nq == model.nv == model.nu == 13

    view.set_qpos_dict(config.init_qpos)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(view.get_move_group("torso").joint_pos, [0.2])
    assert view.get_gripper("gripper").inter_finger_dist == pytest.approx(0.041666)
    np.testing.assert_allclose(view.get_move_group("torso").ctrl_limits, [[0.0, 0.34]])
    assert model.camera("robot_0/robot0_robotview").id >= 0
    assert model.camera("robot_0/robot0_eye_in_hand").id >= 0
    assert model.camera("robot_0/camera_follower").id >= 0

    robot_collision_geom_ids = [
        i
        for i in range(model.ngeom)
        if model.geom(i).name.startswith("robot_0/") and model.geom(i).group[0] == 0
    ]
    assert robot_collision_geom_ids
    assert all(model.geom(i).rgba[3] == 0.0 for i in robot_collision_geom_ids)
    assert any(
        model.geom(i).name.startswith("robot_0/mobilebase0_")
        and model.geom(i).group[0] == 1
        and model.geom(i).type[0] == mujoco.mjtGeom.mjGEOM_MESH
        for i in range(model.ngeom)
    )


def test_base_qpos_is_world_pose_for_nonzero_insertion():
    position = np.array([1.2, -0.4])
    yaw = 0.7
    _, model, data, view = _compile_robot(position, yaw)
    mujoco.mj_forward(model, data)

    np.testing.assert_allclose(view.get_move_group("base").joint_pos, [*position, yaw])
    np.testing.assert_allclose(view.base.pose[:2, 3], position)
    np.testing.assert_allclose(
        view.base.pose[:3, :3], R.from_euler("z", yaw).as_matrix(), atol=1e-7
    )

    target = np.array([-0.3, 0.8, -2.9])
    target_pose = np.eye(4)
    target_pose[:2, 3] = target[:2]
    target_pose[:3, :3] = R.from_euler("z", target[2]).as_matrix()
    view.base.pose = target_pose
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(view.get_move_group("base").joint_pos, target, atol=1e-7)
    base_site = data.site("robot_0/base_site")
    np.testing.assert_allclose(base_site.xpos[:2], target[:2], atol=1e-7)
    np.testing.assert_allclose(
        base_site.xmat.reshape(3, 3), target_pose[:3, :3], atol=1e-7
    )


def test_base_controller_can_resync_after_pose_teleport():
    """Placement teleports must not leave the position servo targeting origin."""
    _, model, data, view = _compile_robot()
    mujoco.mj_forward(model, data)
    controller = JointPosController(view.get_move_group("base"))

    placed_pose = np.eye(4)
    placed_pose[:2, 3] = [1.1, -0.6]
    placed_pose[:3, :3] = R.from_euler("z", 0.8).as_matrix()
    view.base.pose = placed_pose
    mujoco.mj_forward(model, data)

    # This is the synchronization performed after place_robot_near().
    controller.reset()
    view.get_move_group("base").ctrl = controller.compute_ctrl_inputs()
    np.testing.assert_allclose(controller.target_pos, [1.1, -0.6, 0.8], atol=1e-7)
    np.testing.assert_allclose(view.get_move_group("base").ctrl, [1.1, -0.6, 0.8], atol=1e-7)


def test_position_servos_step_without_instability():
    config, model, data, view = _compile_robot()
    view.set_qpos_dict(config.init_qpos)
    mujoco.mj_forward(model, data)

    targets = {
        "base": np.array([0.15, -0.1, 0.2]),
        "torso": np.array([0.25]),
        "arm": np.asarray(config.init_qpos["arm"]) + [0.05, 0, 0, 0, 0, 0, 0],
        "gripper": np.array([0.04, -0.04]),
    }
    for name, target in targets.items():
        view.get_move_group(name).ctrl = target
    for _ in range(1500):
        mujoco.mj_step(model, data)

    assert np.all(np.isfinite(data.qpos))
    np.testing.assert_allclose(view.get_move_group("base").joint_pos, targets["base"], atol=0.04)
    np.testing.assert_allclose(
        view.get_move_group("torso").joint_pos, targets["torso"], atol=0.01
    )
    np.testing.assert_allclose(view.get_move_group("arm").joint_pos, targets["arm"], atol=0.01)
    assert view.get_gripper("gripper").is_open


@pytest.mark.parametrize("yaw", [0.15, -0.15])
def test_base_reaches_planner_tolerance_within_waypoint_budget(yaw):
    """底盘必须在 E4 的 1.98 秒预算内满足 0.0275 的规划容差。"""
    config, model, data, view = _compile_robot()
    model.opt.timestep = 0.002
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    view.set_qpos_dict(config.init_qpos)
    for name, values in config.init_qpos.items():
        view.get_move_group(name).ctrl = np.asarray(values)
    target = np.array([0.05, -0.05, yaw])
    view.base.ctrl = target
    mujoco.mj_forward(model, data)
    for _ in range(990):
        mujoco.mj_step(model, data)
    np.testing.assert_allclose(view.base.joint_pos, target, atol=0.005)


@pytest.mark.slow
def test_cpu_and_warp_kinematics_construct():
    config = PandaOmronRobotConfig()
    cpu = MlSpacesKinematics(config)
    warp = SimpleWarpKinematics(config, device="cpu")
    qpos = {name: np.asarray(value) for name, value in config.init_qpos.items()}
    base_pose = np.eye(4)
    cpu_fk = cpu.fk(qpos, base_pose)
    warp_fk = warp.fk(qpos, base_pose)
    assert set(cpu_fk) == {"base", "torso", "arm", "gripper"}
    np.testing.assert_allclose(warp_fk["arm"], cpu_fk["arm"], atol=1e-4)

    raised_torso = {name: value.copy() for name, value in qpos.items()}
    raised_torso["torso"] += 0.05
    raised_fk = cpu.fk(raised_torso, base_pose)
    assert raised_fk["arm"][2, 3] - cpu_fk["arm"][2, 3] == pytest.approx(0.05)

    target = cpu_fk["arm"].copy()
    target[:3, 3] += [0.01, -0.01, 0.01]
    solution = cpu.ik("arm", target, ["arm"], qpos, base_pose)
    assert solution is not None
    solved_fk = cpu.fk(solution, base_pose)
    np.testing.assert_allclose(solved_fk["arm"][:3, 3], target[:3, 3], atol=1e-3)


def test_mesh_stripped_model_compiles():
    _, model, _, _ = _compile_robot(strip_meshes=True)
    assert not np.any(model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
