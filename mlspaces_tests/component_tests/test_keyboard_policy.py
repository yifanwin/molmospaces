from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from molmo_spaces.policy.learned_policy.keyboard_policy import Keyboard_Policy


class FakeKey:
    up = object()
    down = object()
    left = object()
    right = object()
    space = object()
    tab = object()


class FakeKeyCode:
    @staticmethod
    def from_char(char) -> tuple[str, str]:
        return ("char", char)


class FakeKeyboard:
    Key = FakeKey
    KeyCode = FakeKeyCode


class FakeMoveGroup:
    def __init__(self, *, joint_pos=None, pose=None):
        self.joint_pos = np.array(joint_pos if joint_pos is not None else [], dtype=float)
        self.leaf_frame_to_robot = np.array(pose if pose is not None else np.eye(4), dtype=float)


class FakeRobotView:
    def __init__(self):
        left_pose = np.eye(4)
        left_pose[:3, 3] = [0.4, 0.2, 0.8]
        right_pose = np.eye(4)
        right_pose[:3, 3] = [0.4, -0.2, 0.8]
        self.groups = {
            "base": FakeMoveGroup(joint_pos=[0.0, 0.0, 0.0]),
            "torso": FakeMoveGroup(joint_pos=np.zeros(6)),
            "left_arm": FakeMoveGroup(joint_pos=np.zeros(7)),
            "right_arm": FakeMoveGroup(joint_pos=np.zeros(7)),
            "left_gripper": FakeMoveGroup(joint_pos=[-0.05], pose=left_pose),
            "right_gripper": FakeMoveGroup(joint_pos=[-0.05], pose=right_pose),
            "head": FakeMoveGroup(joint_pos=np.zeros(2)),
        }
        self.base = SimpleNamespace(pose=np.eye(4))

    def get_move_group(self, name):
        return self.groups[name]

    def get_qpos_dict(self):
        return {name: group.joint_pos.copy() for name, group in self.groups.items()}

    def move_group_ids(self):
        return list(self.groups)

    def get_gripper_movegroup_ids(self):
        return ["left_gripper", "right_gripper"]


def make_policy(robot_type="rby1", ik_result=None):
    policy = Keyboard_Policy.__new__(Keyboard_Policy)
    policy.robot_type = robot_type
    policy.step_size = 0.005
    policy.rot_step = 0.02
    policy._keyboard = FakeKeyboard
    policy._pressed = set()
    policy._gripper_open = True
    policy.init_robot_pose = None
    policy.init_tcp_pose = None
    policy.current_position = None
    policy.current_rotation = None
    policy._reset_rby1_state()
    policy.render = Mock()

    robot_view = FakeRobotView()
    if ik_result is None:
        ik_result = robot_view.get_qpos_dict()
        ik_result["left_arm"] = np.arange(7, dtype=float)
        ik_result["right_arm"] = np.arange(7, dtype=float) + 10.0
    kinematics = SimpleNamespace(ik=Mock(return_value=ik_result))
    robot = SimpleNamespace(robot_view=robot_view, kinematics=kinematics)
    policy.task = SimpleNamespace(env=SimpleNamespace(current_robot=robot))
    return policy, robot_view, kinematics


def observation():
    return [{"robot_base_pose": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])}]


@pytest.mark.parametrize("robot_type", ["rby1", "rby1m"])
def test_rby1_names_enable_three_control_modes(robot_type):
    policy, _, _ = make_policy(robot_type)

    assert policy._is_rby1
    assert policy.rby1_control_mode == "left_arm"
    policy._on_press(FakeKey.tab)
    policy._on_press(FakeKey.tab)
    assert policy.rby1_control_mode == "right_arm"

    policy._on_release(FakeKey.tab)
    policy._on_press(FakeKey.tab)
    assert policy.rby1_control_mode == "base"
    policy._on_release(FakeKey.tab)
    policy._on_press(FakeKey.tab)
    assert policy.rby1_control_mode == "left_arm"


def test_rby1_grippers_toggle_independently_and_space_is_ignored_in_base_mode():
    policy, _, _ = make_policy()

    policy._on_press(FakeKey.space)
    policy._on_press(FakeKey.space)
    assert policy._rby1_gripper_open == {"left": False, "right": True}

    policy._on_release(FakeKey.space)
    policy._rby1_control_mode_index = 1
    policy._on_press(FakeKey.space)
    assert policy._rby1_gripper_open == {"left": False, "right": False}

    policy._on_release(FakeKey.space)
    policy._rby1_control_mode_index = 2
    policy._on_press(FakeKey.space)
    assert policy._rby1_gripper_open == {"left": False, "right": False}


def test_rby1_arm_mode_syncs_actual_pose_and_only_unlocks_selected_arm():
    policy, robot_view, kinematics = make_policy()
    model_input = policy.obs_to_model_input(observation())
    policy._pressed.add(FakeKey.up)

    action = policy.inference_model(model_input)

    args = kinematics.ik.call_args.args
    assert args[0] == "left_gripper"
    np.testing.assert_allclose(args[1][:3, 3], [0.405, 0.2, 0.8])
    assert args[2] == ["left_arm"]
    assert kinematics.ik.call_args.kwargs == {"rel_to_base": True}
    np.testing.assert_array_equal(action["left_arm"], np.arange(7, dtype=float))
    np.testing.assert_array_equal(action["left_gripper"], [-0.05])
    np.testing.assert_array_equal(action["right_gripper"], [-0.05])
    assert "right_arm" not in action
    assert "base" not in action
    np.testing.assert_allclose(
        policy._rby1_target_poses["left_arm"][:3, 3],
        robot_view.get_move_group("left_gripper").leaf_frame_to_robot[:3, 3]
        + np.array([0.005, 0.0, 0.0]),
    )


def test_rby1_right_arm_uses_right_gripper_target():
    policy, _, kinematics = make_policy()
    policy._rby1_control_mode_index = 1
    model_input = policy.obs_to_model_input(observation())

    action = policy.inference_model(model_input)

    args = kinematics.ik.call_args.args
    assert args[0] == "right_gripper"
    assert args[2] == ["right_arm"]
    np.testing.assert_array_equal(action["right_arm"], np.arange(7, dtype=float) + 10.0)
    assert "left_arm" not in action


def test_rby1_failed_ik_keeps_last_valid_target():
    policy, _, kinematics = make_policy()
    policy.obs_to_model_input(observation())
    old_target = policy._rby1_target_poses["left_arm"].copy()
    kinematics.ik.return_value = None
    policy._pressed.add(FakeKey.up)

    action = policy.inference_model({"control_mode": "left_arm"})

    np.testing.assert_array_equal(policy._rby1_target_poses["left_arm"], old_target)
    assert "left_arm" not in action
    assert set(action) == {"left_gripper", "right_gripper"}


def test_rby1_base_motion_uses_robot_local_coordinates_and_ignores_arm_height_keys():
    policy, robot_view, kinematics = make_policy()
    policy._rby1_control_mode_index = 2
    robot_view.get_move_group("base").joint_pos = np.array([1.0, 2.0, np.pi / 2])
    model_input = policy.obs_to_model_input(observation())
    policy._pressed.update({FakeKey.up, FakeKey.left, FakeKeyCode.from_char("a")})
    policy._pressed.add(FakeKeyCode.from_char("w"))

    action = policy.inference_model(model_input)

    np.testing.assert_allclose(
        action["base"],
        [1.0 - policy.step_size, 2.0 + policy.step_size, np.pi / 2 + policy.rot_step],
    )
    kinematics.ik.assert_not_called()
    assert set(action) == {"base", "left_gripper", "right_gripper"}


def test_rby1_q_ends_episode_and_reset_restores_left_arm_mode():
    policy, _, _ = make_policy()
    policy._rby1_control_mode_index = 2
    policy._rby1_gripper_open["left"] = False
    policy._pressed.add(FakeKeyCode.from_char("q"))

    assert policy.inference_model({"control_mode": "base"}) is None
    policy.reset()
    assert policy.rby1_control_mode == "left_arm"
    assert policy._rby1_gripper_open == {"left": True, "right": True}


def test_rby1_render_combines_three_camera_views(monkeypatch):
    policy, _, _ = make_policy()
    shown = Mock()
    policy._show_views = shown
    put_text = Mock()
    monkeypatch.setattr("molmo_spaces.policy.learned_policy.keyboard_policy.cv2.putText", put_text)
    obs = {
        "head_camera": np.zeros((4, 5, 3), dtype=np.uint8),
        "wrist_camera_l": np.ones((4, 5, 3), dtype=np.uint8),
        "wrist_camera_r": np.full((4, 5, 3), 2, dtype=np.uint8),
    }

    Keyboard_Policy.render(policy, obs)

    assert shown.call_args.args[0].shape == (4, 15, 3)
    assert put_text.call_args.args[1] == "Control: left_arm"


def test_franka_inference_behavior_is_unchanged():
    policy, robot_view, kinematics = make_policy("franka")
    policy.current_position = np.array([0.4, 0.0, 0.5])
    policy.current_rotation = np.eye(3)
    policy._gripper_open = False
    robot_view.groups = {
        "arm": FakeMoveGroup(joint_pos=np.zeros(7)),
        "gripper": FakeMoveGroup(joint_pos=np.zeros(2)),
    }
    robot_view.get_ctrl_dict = Mock(
        return_value={"arm": np.zeros(7), "gripper": np.zeros(1)}
    )
    robot_view.get_gripper_movegroup_ids = Mock(return_value=["gripper"])
    kinematics.ik.return_value = {"arm": np.ones(7), "gripper": np.zeros(2)}

    action = policy.inference_model({})

    assert kinematics.ik.call_args.args[0] == "arm"
    assert kinematics.ik.call_args.args[2] == ["arm"]
    np.testing.assert_array_equal(action["arm"], np.ones(7))
    np.testing.assert_array_equal(action["gripper"], [255.0])
