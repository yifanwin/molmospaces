import json
import urllib.error
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from molmo_spaces.policy.solvers.object_manipulation import (
    llm_waypoint_planner_policy as llm_module,
)
from molmo_spaces.policy.solvers.object_manipulation.llm_waypoint_planner_policy import (
    LLMWaypointPlan,
    LLMWaypointPlannerPolicy,
    OpenAICompatibleClient,
    PlanValidationError,
    extract_json_object,
)
from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy import (
    PickAndPlacePlannerPolicy,
)


def _pose(x=0.0, y=0.0, z=0.5):
    return {"pose": [x, y, z, 1.0, 0.0, 0.0, 0.0], "speed": 0.1}


def _valid_plan():
    return {
        "schema_version": 1,
        "robot": "rby1",
        "arm": "left_arm",
        "grasp_candidate_id": 0,
        "segments": [
            {"phase": name, "motion": "ee", "waypoints": [_pose(z=0.4 + i * 0.02)]}
            for i, name in enumerate(("pregrasp", "grasp", "lift", "preplace", "place", "retreat"))
        ],
    }


def test_plan_schema_accepts_fixed_grammar():
    plan = LLMWaypointPlan.model_validate(_valid_plan())
    assert plan.waypoint_count == 6


def test_plan_schema_accepts_optional_separated_base_segments():
    data = _valid_plan()
    data["segments"].insert(
        0,
        {
            "phase": "base_approach",
            "motion": "base",
            "base_goal": [1.0, 2.0, 0.0],
            "duration_s": 3.0,
        },
    )
    data["segments"].insert(
        4,
        {"phase": "base_transfer", "motion": "base", "base_goal": [1.2, 2.0, 0.1]},
    )
    assert len(LLMWaypointPlan.model_validate(data).segments) == 8


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["segments"].reverse(),
        lambda p: p["segments"][0].update(phase="unknown"),
        lambda p: p["segments"][0]["waypoints"][0].update(pose=[float("nan")] * 7),
        lambda p: p["segments"][0].update(base_goal=[0, 0, 0]),
    ],
)
def test_plan_schema_rejects_malformed_or_coupled_motion(mutation):
    data = _valid_plan()
    mutation(data)
    with pytest.raises(ValidationError):
        LLMWaypointPlan.model_validate(data)


def test_json_extraction_accepts_plain_and_fenced_json():
    data = _valid_plan()
    text = json.dumps(data)
    assert extract_json_object(text) == data
    assert extract_json_object(f"```json\n{text}\n```") == data


def test_json_extraction_rejects_prose_and_nonfinite_constants():
    with pytest.raises(PlanValidationError):
        extract_json_object("Here is the plan: {}")
    with pytest.raises(PlanValidationError):
        extract_json_object('{"x": NaN}')


def test_client_requires_all_environment_variables(monkeypatch):
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="LLM_BASE_URL, LLM_API_KEY, LLM_MODEL"):
        OpenAICompatibleClient.from_env()


def test_client_never_exposes_api_key_on_http_error(monkeypatch):
    secret = "never-print-this-secret"
    client = OpenAICompatibleClient("https://example.invalid/v1", secret, "mock")

    def fail(*args, **kwargs):
        raise urllib.error.HTTPError("https://example.invalid", 401, "Unauthorized", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(RuntimeError) as error:
        client.complete("system", {"safe": True})
    assert secret not in str(error.value)


def test_llm_policy_skips_parallel_ik_warmup(monkeypatch):
    """RBY1 has sequential IK only; inherited reset must not request parallel IK."""

    def fake_parent_init(self, config, task):
        self.config = config
        self.task = task
        self.policy_config = config.policy_config
        self.ik_warmed_up = False

    monkeypatch.setattr(PickAndPlacePlannerPolicy, "__init__", fake_parent_init)
    monkeypatch.setattr(llm_module, "create_planner_client", lambda _timeout: object())
    config = SimpleNamespace(policy_config=SimpleNamespace(api_timeout_s=120.0))

    policy = LLMWaypointPlannerPolicy(config, task=object())

    assert policy.ik_warmed_up is True


def test_json_runner_filters_house_before_max_episodes(monkeypatch, tmp_path):
    from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner
    from molmo_spaces.evaluation import json_eval_runner as module

    episodes = [
        SimpleNamespace(house_index=1),
        SimpleNamespace(house_index=4),
        SimpleNamespace(house_index=4),
    ]
    monkeypatch.setattr(module, "load_all_episodes", lambda _: episodes)
    monkeypatch.setattr(ParallelRolloutRunner, "__init__", lambda self, config: None)
    config = SimpleNamespace(
        eval_runtime_params=SimpleNamespace(house_index=4, max_episodes=1, episode_idx=None),
        task_sampler_config=SimpleNamespace(house_inds=[], samples_per_house=0),
        benchmark_path=None,
    )
    runner = module.JsonEvalRunner(config, tmp_path)
    assert list(runner._episodes_by_house) == [4]
    assert len(runner._episodes_by_house[4]) == 1
    assert config.task_sampler_config.house_inds == [4]


@pytest.mark.parametrize("positional", [False, True])
@pytest.mark.parametrize("has_history", [False, True])
def test_planner_failure_preserves_diagnostic_history(monkeypatch, positional, has_history):
    from unittest.mock import Mock
    from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner
    from molmo_spaces.evaluation.json_eval_runner import JsonEvalRunner

    monkeypatch.setattr(
        ParallelRolloutRunner, "run_single_rollout",
        Mock(side_effect=ValueError("planning rejected")),
    )
    frame = [{"camera": "diagnostic"}]
    task = SimpleNamespace(
        observation_cache=[frame] if has_history else [],
        config=SimpleNamespace(freeze_task_config=Mock(return_value="frozen")),
        frozen_config=None,
    )

    def capture():
        task.observation_cache.append(frame)
        return frame, None, None, None, None

    task.get_and_cache_all_step_information = Mock(side_effect=capture)
    policy = SimpleNamespace(config=SimpleNamespace(policy_config=SimpleNamespace(policy_type="planner")))
    if positional:
        result = JsonEvalRunner.run_single_rollout(0, task, policy)
    else:
        result = JsonEvalRunner.run_single_rollout(episode_seed=0, task=task, policy=policy)
    assert result is False
    assert task.observation_cache == [frame]
    assert task.get_and_cache_all_step_information.call_count == (0 if has_history else 1)
    if not has_history:
        assert task.frozen_config == "frozen"


def test_nonplanner_failure_is_not_swallowed(monkeypatch):
    from unittest.mock import Mock
    from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner
    from molmo_spaces.evaluation.json_eval_runner import JsonEvalRunner

    monkeypatch.setattr(ParallelRolloutRunner, "run_single_rollout", Mock(side_effect=ValueError("bug")))
    policy = SimpleNamespace(config=SimpleNamespace(policy_config=SimpleNamespace(policy_type="learned")))
    with pytest.raises(ValueError, match="bug"):
        JsonEvalRunner.run_single_rollout(0, object(), policy)


def test_base_segment_contract_matches_validator():
    from molmo_spaces.policy.solvers.object_manipulation.llm_waypoint_planner_policy import PlanSegment

    segment = {"phase": "base_transfer", "motion": "base", "base_goal": [0.9, 8.95, 0.2]}
    assert PlanSegment.model_validate(segment).motion == "base"
    with pytest.raises(ValidationError):
        PlanSegment.model_validate({**segment, "base_goal": [0, 0, 0, 1, 0, 0, 0]})
    del segment["motion"]
    with pytest.raises(ValidationError):
        PlanSegment.model_validate(segment)


def test_grasp_sensor_handles_rejected_plan():
    import numpy as np
    from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
        GraspPoseSensor,
    )

    policy = object.__new__(LLMWaypointPlannerPolicy)
    policy.target_poses = {}
    task = SimpleNamespace(_registered_policy=policy)
    result = GraspPoseSensor().get_observation(None, task)
    np.testing.assert_array_equal(result, np.zeros(7, dtype=np.float32))


def test_single_failure_frame_can_be_saved_as_video(tmp_path):
    import numpy as np
    import imageio.v2 as imageio
    from molmo_spaces.env.sensors_cameras import CameraSensor
    from molmo_spaces.utils.save_utils import save_videos_from_raw_observations

    suite = SimpleNamespace(sensors={"camera": object.__new__(CameraSensor)})
    save_videos_from_raw_observations(
        [{"camera": np.zeros((32, 32, 3), dtype=np.uint8)}],
        tmp_path, fps=10, sensor_suite=suite,
    )
    path = tmp_path / "episode_00000000_camera.mp4"
    assert path.is_file() and path.stat().st_size > 0
    reader = imageio.get_reader(path)
    try:
        assert reader.get_data(0).shape == (32, 32, 3)
    finally:
        reader.close()


def test_failed_reset_is_safe_for_real_policy_sensors():
    """Exercise the actual phase/target sensors, not a mocked observation call."""
    import numpy as np
    from molmo_spaces.env.abstract_sensors import SensorSuite
    from molmo_spaces.env.sensors import PolicyPhaseSensor, PolicyNumRetriesSensor
    from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import GraspPoseSensor

    policy = object.__new__(LLMWaypointPlannerPolicy)
    policy.action_primitives = []
    policy.action_idx = 0
    policy.target_poses = {}
    policy._retry_count = 0
    task = SimpleNamespace(_registered_policy=policy)
    suite = SensorSuite([PolicyPhaseSensor(), PolicyNumRetriesSensor(), GraspPoseSensor()])
    observation = suite.get_observations(env=None, task=task)
    assert observation["policy_phase"] == policy.get_all_phases()["unknown"]
    assert observation["policy_num_retries"] == 0
    np.testing.assert_array_equal(observation["grasp_pose"], np.zeros(7))


def test_failed_reset_clears_old_plan_and_targets():
    from unittest.mock import Mock

    policy = object.__new__(LLMWaypointPlannerPolicy)
    policy.ik_warmed_up = True
    policy.action_primitives = [object()]
    policy.action_idx = 1
    policy.target_poses = {"grasp": object()}
    policy._compute_trajectory = Mock(side_effect=ValueError("rejected"))
    with pytest.raises(ValueError, match="rejected"):
        policy.reset()
    assert policy.action_primitives == []
    assert policy.action_idx == 0
    assert policy.target_poses == {}
    assert policy.get_phase() == "unknown"
