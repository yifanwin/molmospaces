"""RBY1 夹爪位置伺服的回归测试。

MJCF 把两个夹爪 actuator 声明成 <motor>，而其余 25 个关节都是 <position>。
未改造时 XML 里残留的 biasprm（kp=4000、kv=400）因为 biastype=NONE 完全失效，
gainprm 也停在 <motor> 的默认值 1——ctrl 被当成力矩用：闭合指令 0.0 对应零力矩，
手指纹丝不动。实测 103:1 全场 205 步 inter_finger_dist 恒为 0.1（全开），
夹爪与目标物接触恒为 False，物体原地未动、任务必然判失败。
"""

import mujoco
import numpy as np
import pytest

from molmo_spaces.configs.robot_configs import RBY1Config
from molmo_spaces.robots.rby1 import GRIPPER_CTRL_RANGE, GRIPPER_KP, GRIPPER_KV

GRIPPER_ACTUATORS = ("robot_0/left_finger_act", "robot_0/right_finger_act")
LEFT_FINGER_JOINT = "robot_0/gripper_finger_l1"
FULLY_OPEN_DIST = 0.1  # inter_finger_dist_range 的上界
STEPS_TO_SETTLE = 500


def _compile_rby1() -> mujoco.MjModel:
    config = RBY1Config()
    spec = mujoco.MjSpec.from_file(str(config.get_robot_xml_path()))
    config.robot_cls.apply_control_overrides(spec, config)
    return spec.compile()


def _settle(model: mujoco.MjModel, data: mujoco.MjData, actuator_id: int, ctrl: float) -> None:
    data.ctrl[actuator_id] = ctrl
    for _ in range(STEPS_TO_SETTLE):
        mujoco.mj_step(model, data)


def _left_inter_finger_dist(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    qadr = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, LEFT_FINGER_JOINT)
    ]
    return float(np.abs(data.qpos[qadr : qadr + 2]).sum())


def test_gripper_actuators_are_position_servos():
    """gainprm[0] 与 biasprm[1] 必须同量反号，且 bias 真的生效。"""
    model = _compile_rby1()
    for name in GRIPPER_ACTUATORS:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert aid >= 0, f"actuator {name} 不存在"
        assert model.actuator_biastype[aid] == mujoco.mjtBias.mjBIAS_AFFINE
        assert model.actuator_gainprm[aid][0] == GRIPPER_KP
        assert model.actuator_biasprm[aid][1] == -GRIPPER_KP
        assert model.actuator_biasprm[aid][2] == -GRIPPER_KV
        np.testing.assert_allclose(model.actuator_ctrlrange[aid], GRIPPER_CTRL_RANGE)


def test_gripper_closes_and_opens_under_ctrl():
    """回归锚点：ctrl=0.0 必须让手指从全开走到闭合，ctrl=-0.05 再走回全开。"""
    model = _compile_rby1()
    data = mujoco.MjData(model)
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATORS[0])

    _settle(model, data, aid, GRIPPER_CTRL_RANGE[0])
    assert _left_inter_finger_dist(model, data) == pytest.approx(FULLY_OPEN_DIST, rel=0.01)

    _settle(model, data, aid, 0.0)
    assert _left_inter_finger_dist(model, data) < 0.01

    _settle(model, data, aid, GRIPPER_CTRL_RANGE[0])
    assert _left_inter_finger_dist(model, data) == pytest.approx(FULLY_OPEN_DIST, rel=0.01)


def test_only_gripper_actuators_differ_from_raw_model():
    """其余 25 个 actuator 的参数必须与原模型逐位一致。"""
    config = RBY1Config()
    raw = mujoco.MjModel.from_xml_path(str(config.get_robot_xml_path()))
    patched = _compile_rby1()
    assert raw.nu == patched.nu

    for i in range(raw.nu):
        name = mujoco.mj_id2name(raw, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        j = mujoco.mj_name2id(patched, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if name in GRIPPER_ACTUATORS:
            # 钉住改造前的状态，避免有人直接改 MJCF 让这些断言悄悄失效
            assert raw.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_NONE
            assert patched.actuator_biastype[j] == mujoco.mjtBias.mjBIAS_AFFINE
            continue
        assert patched.actuator_biastype[j] == raw.actuator_biastype[i]
        np.testing.assert_allclose(patched.actuator_gainprm[j], raw.actuator_gainprm[i])
        np.testing.assert_allclose(patched.actuator_biasprm[j], raw.actuator_biasprm[i])
        np.testing.assert_allclose(patched.actuator_ctrlrange[j], raw.actuator_ctrlrange[i])
