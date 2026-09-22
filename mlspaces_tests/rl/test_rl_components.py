import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from molmo_spaces.policy.learned_policy.rby1_rl_policy import (
    RBY1RLPolicy,
    validate_rl_checkpoint_metadata,
)
from molmo_spaces.rl.action_adapter import RBY1ActionAdapter
from molmo_spaces.rl.observation import OBSERVATION_VERSION, RBY1ObservationBuilder
from molmo_spaces.rl.reward import RewardSnapshot, RewardTracker


def _base_pose(x=1.0, y=2.0, yaw=np.pi / 2):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    pose[:2, 3] = [x, y]
    return pose


def test_action_adapter_uses_robot_local_frame_and_discretizes_candidate():
    adapter = RBY1ActionAdapter(max_translation_m=0.5, max_yaw_delta_rad=np.pi / 2)
    decoded = adapter.decode(
        np.array([1.0, 0.0, 0.5, 0.0, -0.1], np.float32),
        _base_pose(),
        num_candidates=5,
        phase="pick",
    )
    np.testing.assert_allclose(decoded.base_goal[:2], [1.0, 2.5], atol=1e-6)
    assert decoded.base_goal[2] == pytest.approx(3 * np.pi / 4)
    assert decoded.candidate_index == 2
    assert decoded.arm == "left"
    assert decoded.valid


def test_action_adapter_clips_translation_norm_and_ignores_pick_fields_during_place():
    adapter = RBY1ActionAdapter(max_translation_m=0.5)
    decoded = adapter.decode(
        np.array([1.0, 1.0, 0.0, -1.0, 1.0], np.float32),
        _base_pose(yaw=0),
        num_candidates=0,
        phase="place",
    )
    assert decoded.translation_m == pytest.approx(0.5)
    assert decoded.candidate_index is None
    assert decoded.arm is None


def test_observation_space_contract_is_fixed_float32():
    builder = RBY1ObservationBuilder(max_candidates=24)
    assert builder.observation_space["proprio"].shape == (41,)
    assert builder.observation_space["task"].shape == (21,)
    assert builder.observation_space["candidates"].shape == (24, 13)
    assert builder.observation_space["candidate_mask"].shape == (24,)
    assert builder.observation_space["phase_context"].shape == (8,)
    assert all(space.dtype == np.float32 for space in builder.observation_space.spaces.values())


def test_reward_events_are_awarded_only_once_and_success_remains_separate():
    tracker = RewardTracker()
    before = RewardSnapshot(1.0, 1.0, 1.0, 0.5, False)
    after = RewardSnapshot(0.5, 0.5, 1.0, 0.6, True)
    first, terms = tracker.compute(
        "pick",
        before,
        after,
        success=True,
        supported_and_released=False,
        invalid_action=False,
        planning_failure=False,
        collision=False,
        translation_fraction=0.0,
        yaw_fraction=0.0,
    )
    second, second_terms = tracker.compute(
        "pick",
        before,
        after,
        success=True,
        supported_and_released=False,
        invalid_action=False,
        planning_failure=False,
        collision=False,
        translation_fraction=0.0,
        yaw_fraction=0.0,
    )
    assert terms["grasp"] == 4.0 and terms["lift"] == 3.0
    assert terms["task_success"] == 10.0
    assert second_terms["grasp"] == second_terms["lift"] == 0.0
    assert second_terms["task_success"] == 0.0
    assert first > second


def test_checkpoint_metadata_validation(tmp_path: Path):
    checkpoint = tmp_path / "model.zip"
    checkpoint.touch()
    metadata = {
        "observation_version": OBSERVATION_VERSION,
        "action_dim": 5,
        "max_candidates": 24,
        "position_scale_m": 2.0,
        "max_base_translation_m": 0.5,
        "max_base_yaw_delta_rad": float(np.pi / 2),
    }
    (tmp_path / "rl_metadata.json").write_text(json.dumps(metadata))
    assert validate_rl_checkpoint_metadata(checkpoint, 24) == metadata
    with pytest.raises(ValueError, match="incompatible"):
        validate_rl_checkpoint_metadata(checkpoint, 12)


def test_rl_policy_does_not_reference_llm_waypoint_source():
    source = Path(
        "molmo_spaces/policy/learned_policy/rby1_rl_policy.py"
    ).read_text()
    assert "llm_waypoint_planner_policy" not in source
    assert "LLMDecision" not in source


class _FakeRobotView:
    @staticmethod
    def get_noop_ctrl_dict():
        return {"base": np.zeros(3)}


class _FakeExecutor:
    failure_reason = None

    def __init__(self, phase: str, action: dict | None = None):
        self.phase = phase
        self.action = action or {"base": np.ones(3)}

    def get_phase(self):
        return self.phase

    def get_action(self, _observation):
        return self.action.copy()


def _policy_at_executor_phase(macro_phase: str, executor: _FakeExecutor):
    policy = object.__new__(RBY1RLPolicy)
    policy.policy_config = SimpleNamespace(rl_max_macro_steps=4)
    policy.task = SimpleNamespace(
        env=SimpleNamespace(current_robot=SimpleNamespace(robot_view=_FakeRobotView()))
    )
    policy.executor = executor
    policy._base_move = None
    policy._macro_phase = macro_phase
    policy._at_macro_boundary = False
    policy._retry_count = 0
    policy._macro_steps = 1
    policy._last_failure = "none"
    policy._candidates = SimpleNamespace()
    policy._irrecoverable_failure = False
    policy._has_held_object = False
    policy._initial_severe_contacts = set()
    policy._severe_robot_contacts = lambda: set()
    return policy


def test_policy_pick_to_place_boundary_and_place_terminal_action():
    policy = _policy_at_executor_phase("pick", _FakeExecutor("place"))
    action = policy.get_action({})
    assert policy.macro_phase == "place"
    assert policy.at_macro_boundary
    np.testing.assert_array_equal(action["base"], np.zeros(3))

    terminal = _policy_at_executor_phase(
        "place", _FakeExecutor("done", {"done": True, "success": True})
    )
    assert terminal.get_action({}) == {"done": True, "success": True}


def test_recoverable_failure_returns_to_same_phase_and_counts_retry():
    executor = _FakeExecutor("pregrasp")

    def fail(_observation):
        raise RuntimeError("no plan")

    executor.get_action = fail
    policy = _policy_at_executor_phase("pick", executor)
    policy.get_action({})
    assert policy.macro_phase == "pick"
    assert policy.at_macro_boundary
    assert policy.retry_count == 1
    assert policy.last_failure == "planning_failure"


def test_policy_stops_at_four_macro_actions():
    policy = _policy_at_executor_phase("pick", _FakeExecutor("pregrasp"))
    policy._macro_steps = 4
    assert policy.get_action({}) == {"done": True, "success": False}
