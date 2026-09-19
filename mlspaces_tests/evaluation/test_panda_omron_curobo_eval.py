from types import SimpleNamespace

import pytest

from molmo_spaces.configs.policy_configs import (
    panda_omron_planner_joint_ranges,
)
from molmo_spaces.evaluation.robot_eval_overrides import (
    panda_omron_robot_eval_override,
)


@pytest.mark.parametrize(
    ("groups", "ranges"),
    [
        (["arm"], {"arm": (0, 7)}),
        (["torso", "arm"], {"torso": (0, 1), "arm": (1, 8)}),
        (
            ["base", "torso", "arm"],
            {"base": (0, 3), "torso": (3, 4), "arm": (4, 11)},
        ),
    ],
)
def test_panda_omron_planner_move_group_modes(groups, ranges):
    assert panda_omron_planner_joint_ranges(groups) == ranges


def test_panda_omron_planner_rejects_unsupported_move_groups():
    with pytest.raises(ValueError, match="planner_move_group_ids"):
        panda_omron_planner_joint_ranges(["base", "arm"])


def test_panda_omron_eval_override_replaces_only_robot_specific_state():
    episode = SimpleNamespace(
        robot=SimpleNamespace(robot_name="rby1m", init_qpos={"head": [[0.0, 0.0]]}),
        task={"robot_base_pose": [1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0]},
    )
    initial_qpos = {
        "base": [0.0, 0.0, 0.0],
        "torso": [0.2],
        "arm": [0.0] * 7,
        "gripper": [0.02, -0.02],
    }
    runtime = SimpleNamespace(
        use_config_camera_system=False,
        repair_robot_base_pose_if_colliding=False,
    )
    exp_config = SimpleNamespace(
        robot_config=SimpleNamespace(name="panda_omron", init_qpos=initial_qpos),
        eval_runtime_params=runtime,
    )

    panda_omron_robot_eval_override(episode, exp_config)

    assert episode.robot.robot_name == "panda_omron"
    assert episode.robot.init_qpos == initial_qpos
    assert episode.robot.init_qpos is not initial_qpos
    assert episode.task["robot_base_pose"] == [1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert runtime.use_config_camera_system is True
    assert runtime.repair_robot_base_pose_if_colliding is True
