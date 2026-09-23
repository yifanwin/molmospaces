import json
import urllib.error
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from molmo_spaces.policy.solvers.object_manipulation import (
    llm_waypoint_planner_policy as llm_module,
)
from molmo_spaces.policy.solvers.object_manipulation import (
    pick_and_place_planner_policy as baseline_module,
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
    config = SimpleNamespace(
        policy_config=SimpleNamespace(api_timeout_s=120.0, llm_decision_source="llm")
    )

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


# ---------------------------------------------------------------------------
# 高层决策（LLMDecision）+ 局部确定性几何展开
# ---------------------------------------------------------------------------

# 接近轴 = [0, 0, -1]，倾角 0：纯顶抓。
_TOPDOWN_ROT = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
# 接近轴 = [1, 0, 0]，倾角 90：水平侧抓。
_LATERAL_ROT = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

# _policy_shell 里的固定场景：目标物底部 0.75、容器顶 0.92。
_PICKUP_CENTER = np.array([0.70, 0.10, 0.80])
_PICKUP_SIZE = np.array([0.06, 0.06, 0.10])
_RECEPTACLE_CENTER = np.array([0.50, 0.50, 0.90])
_RECEPTACLE_SIZE = np.array([0.20, 0.20, 0.04])


def _grasp(rotation, position=(0.70, 0.10, 0.85)):
    grasp = np.eye(4)
    grasp[:3, :3] = rotation
    grasp[:3, 3] = position
    return grasp


def _decision(**overrides):
    payload = {
        "schema_version": 1,
        "robot": "rby1",
        "arm": "left_arm",
        "grasp_candidate_id": 0,
        "approach_strategy": "lateral",
        "lift_height": 0.30,
        "preplace_height": 0.08,
    }
    payload.update(overrides)
    return llm_module.LLMDecision.model_validate(payload)


def _geometry(
    grasp,
    place_z=1.02,
    min_carry_z=1.09,
    preplace_xy=(0.50, 0.50),
    baseline_lift_height=None,
    baseline_preplace_height=0.07,
):
    return llm_module.SceneGeometry(
        grasp=grasp,
        place_z=place_z,
        min_carry_z=min_carry_z,
        preplace_xy=np.asarray(preplace_xy, dtype=float),
        baseline_lift_height=(
            min_carry_z - float(grasp[2, 3])
            if baseline_lift_height is None
            else baseline_lift_height
        ),
        baseline_preplace_height=baseline_preplace_height,
    )


def _build(decision, geometry, standoff=0.04, end_z_offset=0.05):
    return llm_module.build_plan_from_decision(
        decision=decision,
        geometry=geometry,
        current_base_xy=np.zeros(2),
        pregrasp_standoff=standoff,
        end_z_offset=end_z_offset,
        base_speed_mps=0.30,
        speed_fast=0.20,
        speed_slow=0.08,
    )


def _ee_poses(plan):
    return {
        segment.phase: segment.waypoints[0].matrix()
        for segment in plan.segments
        if segment.motion == "ee"
    }


def _policy_shell(monkeypatch, tmp_path, base_xy=(0.0, 0.0)):
    """造一个足够跑 _scene_payload / _validate_decision / _request_valid_plan 的 policy 壳。

    完全不碰 MuJoCo：body_aabb 被换成上面的固定场景，预检由调用方另行 monkeypatch。
    """

    def fake_body_aabb(model, data, object_id):
        if object_id == "pickup":
            return _PICKUP_CENTER, _PICKUP_SIZE
        return _RECEPTACLE_CENTER, _RECEPTACLE_SIZE

    monkeypatch.setattr(llm_module, "body_aabb", fake_body_aabb)
    monkeypatch.setattr(baseline_module, "body_aabb", fake_body_aabb)

    base_pose = np.eye(4)
    base_pose[0, 3], base_pose[1, 3] = base_xy
    ee_pose = np.eye(4)
    ee_pose[:3, 3] = [0.30, -0.20, 0.90]

    def get_move_group(name):
        if name == "base":
            return SimpleNamespace(
                joint_pos=np.array([base_xy[0], base_xy[1], 0.0]),
                leaf_frame_to_world=base_pose,
                root_body_id=0,
            )
        return SimpleNamespace(joint_pos=np.zeros(7), leaf_frame_to_world=ee_pose, root_body_id=1)

    policy = object.__new__(LLMWaypointPlannerPolicy)
    policy.config = SimpleNamespace(
        robot_config=SimpleNamespace(name="rby1", init_qpos={}), output_dir=str(tmp_path)
    )
    policy.policy_config = SimpleNamespace(
        llm_decision_source="llm",
        llm_max_api_calls=3,
        llm_max_waypoints=32,
        llm_pregrasp_standoff_m=None,
        llm_top_down_max_tilt_deg=20.0,
        llm_lift_height_bounds=(0.0, 0.60),
        llm_preplace_height_bounds=(0.0, 0.30),
        llm_base_xy_limit_m=2.0,
        llm_base_standoff_target_m=0.75,
        llm_base_step_limit_m=1.0,
        llm_base_yaw_tolerance_deg=15.0,
        llm_base_xy_tolerance_m=0.15,
        llm_base_max_moves=3,
        llm_geometric_base_approach=False,
        llm_base_speed_mps=0.30,
        pregrasp_z_offset=0.04,
        place_z_offset=0.07,
        end_z_offset=0.05,
        speed_fast=0.20,
        speed_slow=0.08,
        grasp_libraries=None,
    )
    policy.robot_view = SimpleNamespace(
        name="rby1",
        base=SimpleNamespace(pose=base_pose),
        get_qpos_dict=lambda: {"base": np.array([base_xy[0], base_xy[1], 0.0])},
        get_move_group=get_move_group,
    )
    policy.task = SimpleNamespace(
        env=SimpleNamespace(
            current_model=None,
            current_data=None,
            current_batch_index=0,
            object_managers=[SimpleNamespace(list_top_level_objects=lambda: [])],
        )
    )
    policy._available_arms = lambda: {"left_arm": "left_gripper"}
    policy._base_is_movable = lambda: True
    policy._selected_arm_id = None
    policy._selected_gripper_id = None
    policy._api_calls = 0
    policy._artifact_id = "test"

    pickup = SimpleNamespace(
        name="pickup", object_id="pickup", position=_PICKUP_CENTER.copy(), pose=np.eye(4)
    )
    receptacle = SimpleNamespace(
        name="receptacle", object_id="receptacle", position=_RECEPTACLE_CENTER.copy(),
        pose=np.eye(4),
    )
    return policy, pickup, receptacle


class _ScriptedClient:
    """按脚本逐次返回响应的假客户端，同时记录每次收到的 payload。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads = []

    def complete(self, system_prompt, payload):
        self.payloads.append(payload)
        content = self.responses[min(len(self.payloads) - 1, len(self.responses) - 1)]
        return llm_module.LLMResponse(
            content=content, latency_s=0.0, usage={}, request_id="scripted"
        )


def test_approach_tilt_separates_top_down_from_lateral():
    assert llm_module.approach_tilt_deg(_grasp(_TOPDOWN_ROT)) == pytest.approx(0.0)
    assert llm_module.approach_tilt_deg(_grasp(_LATERAL_ROT)) == pytest.approx(90.0)


def _yaw_pose(x=0.0, y=0.0, yaw_deg=0.0):
    """只有偏航的底盘位姿。绕 z 转 yaw 度。"""
    angle = np.radians(yaw_deg)
    pose = np.eye(4)
    pose[:3, :3] = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    pose[0, 3], pose[1, 3] = x, y
    return pose


def test_recommend_base_goal_faces_target_at_standoff():
    goal = llm_module.recommend_base_goal(
        np.array([0.0, 0.0]), np.array([1.0, 0.0]), standoff_m=0.75, step_limit_m=1.0
    )
    assert goal == pytest.approx([0.25, 0.0, 0.0])


def test_recommend_base_goal_backs_off_when_inside_standoff_but_aligned():
    """已在 standoff 之内、朝向已对时旧实现返回 None。

    这正是 100% 的 episode 拿不到 approach 阶段底盘推荐值的原因：底盘在原点、
    朝向 +X（yaw=0），目标在 (0.5, 0)，方位角已经等于底盘 yaw，唯一的问题是
    距离 0.5 比 standoff 0.75 近。新实现给出"沿 -X 退到 standoff"的具体位姿。
    """
    goal = llm_module.recommend_base_goal(
        np.array([0.0, 0.0]), np.array([0.5, 0.0]), standoff_m=0.75, step_limit_m=1.0
    )
    assert goal == pytest.approx([-0.25, 0.0, 0.0])


def test_recommend_base_goal_rotates_in_place_when_only_yaw_is_off():
    """位置已在 standoff 内、只差朝向时，首选"原地转向"：位置一动不动。

    底盘在原点但 yaw=+90 度（朝世界 +Y），目标在 (0.5, 0) 即世界 +X。
    旧实现只看距离（0.5 <= 0.75）就返回 None，底盘永远不会转过来。
    """
    goal = llm_module.recommend_base_goal(
        _yaw_pose(0.0, 0.0, 90.0), np.array([0.5, 0.0]), standoff_m=0.75, step_limit_m=1.0
    )
    assert goal == pytest.approx([0.0, 0.0, 0.0])


def test_recommend_base_goal_is_none_only_when_position_and_yaw_both_align():
    """位置与朝向都达标才返回 None。底盘在 (0.7, 0) 朝 -X，目标在原点。"""
    assert (
        llm_module.recommend_base_goal(
            _yaw_pose(0.7, 0.0, 180.0), np.array([0.0, 0.0]), standoff_m=0.75, step_limit_m=1.0
        )
        is None
    )


def test_base_alignment_reports_wrapped_yaw_error():
    """yaw_error_deg 归一化到 (-180, 180]，符号是"还差多少度"。"""
    alignment = llm_module.base_alignment(_yaw_pose(0.0, 0.0, 90.0), np.array([0.5, 0.0]))
    assert alignment["distance_m"] == pytest.approx(0.5)
    assert alignment["bearing_deg"] == pytest.approx(0.0)
    assert alignment["base_yaw_deg"] == pytest.approx(90.0)
    assert alignment["yaw_error_deg"] == pytest.approx(-90.0)


def test_base_goal_candidates_put_pure_rotation_first():
    """候选菜单把"原地转向"排在最前，且它的 translation_m 必须是 0。

    纯转向不移动位置、碰撞风险最低，是修复"底盘从不转向"最直接的杠杆。
    """
    candidates = llm_module.base_goal_candidates(
        _yaw_pose(0.0, 0.0, 90.0),
        np.array([0.5, 0.0]),
        standoff_m=0.75,
        yaw_tolerance_deg=15.0,
        xy_tolerance_m=0.15,
        step_limit_m=1.0,
        label="pickup",
    )
    assert [c["id"] for c in candidates][:1] == ["pickup_rotate_in_place"]
    rotation = candidates[0]
    assert rotation["translation_m"] == 0.0
    assert rotation["rotation_deg"] == pytest.approx(90.0)
    assert rotation["goal"] == pytest.approx([0.0, 0.0, 0.0])
    # 位置也偏（0.5 vs standoff 0.75）时会再给一条"平移到 standoff 并转向"。
    assert "pickup_standoff_face" in [c["id"] for c in candidates]


def test_base_goal_candidates_empty_when_already_aligned():
    """位置与朝向都对好时菜单为空，表示底盘不必动（模型可以填 null）。"""
    assert (
        llm_module.base_goal_candidates(
            _yaw_pose(0.7, 0.0, 180.0),
            np.array([0.0, 0.0]),
            standoff_m=0.75,
            yaw_tolerance_deg=15.0,
            xy_tolerance_m=0.15,
            step_limit_m=1.0,
            label="pickup",
        )
        == []
    )


def test_base_goal_candidates_respect_step_limit():
    """目标很远时单次平移不得超过 llm_base_step_limit_m。"""
    candidates = llm_module.base_goal_candidates(
        _yaw_pose(0.0, 0.0, 0.0),
        np.array([9.0, 0.0]),
        standoff_m=0.75,
        yaw_tolerance_deg=15.0,
        xy_tolerance_m=0.15,
        step_limit_m=1.0,
        label="pickup",
    )
    standoff_face = next(c for c in candidates if c["id"] == "pickup_standoff_face")
    assert standoff_face["translation_m"] == pytest.approx(1.0)


def test_build_plan_pregrasp_retreats_along_approach_axis():
    grasp = _grasp(_TOPDOWN_ROT)
    poses = _ee_poses(_build(_decision(), _geometry(grasp)))
    np.testing.assert_allclose(
        poses["pregrasp"][:3, 3], grasp[:3, 3] - 0.04 * grasp[:3, 2], atol=1e-9
    )
    # 桌面顶抓的 pregrasp 必须落在 grasp 上方：接近轴符号一旦签反，这条会立刻炸。
    assert poses["pregrasp"][2, 3] > grasp[2, 3]


def test_build_plan_lateral_pregrasp_offsets_sideways_not_upward():
    grasp = _grasp(_LATERAL_ROT)
    poses = _ee_poses(
        _build(_decision(approach_strategy="lateral"), _geometry(grasp))
    )
    np.testing.assert_allclose(
        poses["pregrasp"][:3, 3], grasp[:3, 3] - 0.04 * grasp[:3, 2], atol=1e-9
    )
    # 侧抓的接近轴是水平的，pregrasp 高度应与 grasp 相同而不是被抬高。
    assert poses["pregrasp"][2, 3] == pytest.approx(grasp[2, 3])


def test_build_plan_grasp_reproduces_candidate_exactly():
    grasp = _grasp(_TOPDOWN_ROT)
    poses = _ee_poses(_build(_decision(), _geometry(grasp)))
    np.testing.assert_allclose(poses["grasp"], grasp, atol=1e-9)


def test_build_plan_positions_follow_decision_heights():
    grasp = _grasp(_TOPDOWN_ROT)
    geometry = _geometry(grasp, place_z=0.90, min_carry_z=0.97)
    poses = _ee_poses(_build(_decision(lift_height=0.12, preplace_height=0.08), geometry))

    np.testing.assert_allclose(
        poses["lift"][:3, 3], [grasp[0, 3], grasp[1, 3], grasp[2, 3] + 0.12], atol=1e-9
    )
    np.testing.assert_allclose(poses["preplace"][:3, 3], [0.50, 0.50, 0.98], atol=1e-9)
    np.testing.assert_allclose(poses["place"][:3, 3], [0.50, 0.50, 0.90], atol=1e-9)
    np.testing.assert_allclose(
        poses["retreat"][:3, 3], poses["place"][:3, 3] - 0.05 * grasp[:3, 2], atol=1e-9
    )


def test_build_plan_keeps_grasp_orientation_on_every_segment():
    grasp = _grasp(_LATERAL_ROT)
    for phase, pose in _ee_poses(_build(_decision(), _geometry(grasp))).items():
        np.testing.assert_allclose(pose[:3, :3], grasp[:3, :3], atol=1e-9, err_msg=phase)


def test_build_plan_inserts_base_segments_in_grammar_order():
    grasp = _grasp(_TOPDOWN_ROT)
    decision = _decision(
        base_approach_goal=[0.1, 0.2, 0.3], base_transfer_goal=[0.4, 0.5, 0.6]
    )
    plan = _build(decision, _geometry(grasp))
    assert [segment.phase for segment in plan.segments] == [
        "base_approach",
        "pregrasp",
        "grasp",
        "lift",
        "base_transfer",
        "preplace",
        "place",
        "retreat",
    ]
    # 展开结果必须仍然满足内部 IR 自己的相位语法。
    assert len(llm_module.LLMWaypointPlan.model_validate(plan.model_dump()).segments) == 8


def test_build_plan_omits_base_segments_when_goals_are_null():
    plan = _build(_decision(), _geometry(_grasp(_TOPDOWN_ROT)))
    assert [segment.phase for segment in plan.segments] == [
        "pregrasp",
        "grasp",
        "lift",
        "preplace",
        "place",
        "retreat",
    ]


def test_build_plan_base_duration_scales_with_distance():
    geometry = _geometry(_grasp(_TOPDOWN_ROT))
    near = _build(_decision(base_approach_goal=[0.1, 0.0, 0.0]), geometry)
    far = _build(_decision(base_approach_goal=[0.9, 0.0, 0.0]), geometry)
    assert near.segments[0].duration_s == pytest.approx(2.0)  # clip 下界
    assert far.segments[0].duration_s > near.segments[0].duration_s


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.pop("lift_height"),
        lambda d: d.update(approach_strategy="from_above"),
        lambda d: d.update(lift_height=-0.1),
        lambda d: d.update(base_approach_goal=[0.0, 0.0]),
        lambda d: d.update(base_transfer_goal=[0.0, float("nan"), 0.0]),
        lambda d: d.update(segments=[]),
    ],
)
def test_decision_schema_rejects_invalid_payloads(mutation):
    payload = _decision().model_dump(mode="json")
    mutation(payload)
    with pytest.raises(ValidationError):
        llm_module.LLMDecision.model_validate(payload)


def test_decision_schema_accepts_minimal_payload():
    decision = llm_module.LLMDecision.model_validate(_decision())
    assert decision.base_approach_goal is None
    assert decision.base_transfer_goal is None
    assert decision.approach_strategy == "lateral"


def test_mock_client_accepts_decision_and_rejects_legacy_plan(tmp_path):
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps(_decision().model_dump(mode="json")))
    response = llm_module.FileBackedMockClient(decision_path).complete("system", {})
    assert json.loads(response.content)["grasp_candidate_id"] == 0

    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(_valid_plan()))
    with pytest.raises(ValueError, match="legacy waypoint plan"):
        llm_module.FileBackedMockClient(legacy_path)


def test_base_is_movable_requires_three_joint_holo_layout():
    policy = object.__new__(LLMWaypointPlannerPolicy)
    policy.robot_view = SimpleNamespace(
        get_move_group=lambda name: SimpleNamespace(joint_pos=np.zeros(3))
    )
    assert policy._base_is_movable() is True
    policy.robot_view = SimpleNamespace(
        get_move_group=lambda name: SimpleNamespace(joint_pos=np.zeros(7))
    )
    assert policy._base_is_movable() is False


def test_default_decision_reproduces_baseline_heights(monkeypatch, tmp_path):
    policy, pickup, receptacle = _policy_shell(monkeypatch, tmp_path)
    decision = policy._default_decision(pickup, receptacle, [_grasp(_TOPDOWN_ROT)], [["left_arm"]])

    # 容器顶 0.92，抓点比物体底部高 0.10 => place_z = 1.02，min_carry_z = 1.09。
    assert decision.lift_height == pytest.approx(1.09 - 0.85)
    # baseline 的 preplace 高度 = place_z_offset + grasp_z - pickup_z。
    assert decision.preplace_height == pytest.approx(0.07 + 0.85 - 0.80)
    assert decision.grasp_candidate_id == 0
    assert decision.approach_strategy == "top_down"  # 顶抓姿态，倾角 0
    assert decision.base_approach_goal is None


def test_geometric_expansion_matches_geometric_baseline(monkeypatch, tmp_path):
    """钉住"geometric 档 == 几何 baseline"这个前提：任何一方改公式这条都会红。"""
    from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy import (
        PickAndPlacePlannerPolicy,
    )

    policy, pickup, receptacle = _policy_shell(monkeypatch, tmp_path)
    grasp = _grasp(_TOPDOWN_ROT)

    baseline = object.__new__(PickAndPlacePlannerPolicy)
    baseline.policy_config = policy.policy_config
    baseline.check_feasible_ik = lambda pose: True
    # body_aabb 已被 _policy_shell 换成固定场景，model/data 只需要能被解引用。
    baseline.task = SimpleNamespace(
        env=SimpleNamespace(current_data=SimpleNamespace(model=None, data=None))
    )
    robot_view = SimpleNamespace(base=SimpleNamespace(pose=np.eye(4)))

    pregrasp, grasp_pose, lift = baseline._get_grasp_poses(
        grasp_pose_world=grasp,
        pickup_obj=pickup,
        place_receptacle=receptacle,
        robot_view=robot_view,
        task_config=SimpleNamespace(
            pickup_obj_start_pose=np.eye(4), pickup_obj_goal_pose=np.eye(4)
        ),
    )
    preplace, place, postplace = baseline._get_placement_poses(
        grasp_pose_world=grasp, pickup_obj=pickup, place_receptacle=receptacle
    )

    geometry = policy._geometry_for_candidate(0, pickup, receptacle, [grasp])
    assert geometry.place_z == pytest.approx(place[2, 3], abs=1e-9)
    assert geometry.min_carry_z == pytest.approx(lift[2, 3], abs=1e-9)
    assert geometry.baseline_lift_height == pytest.approx(lift[2, 3] - grasp[2, 3], abs=1e-9)
    assert geometry.baseline_preplace_height == pytest.approx(
        preplace[2, 3] - place[2, 3], abs=1e-9
    )
    np.testing.assert_allclose(geometry.preplace_xy, preplace[:2, 3], atol=1e-9)

    plan = _build(
        _decision(
            lift_height=geometry.baseline_lift_height,
            preplace_height=geometry.baseline_preplace_height,
        ),
        geometry,
    )
    poses = _ee_poses(plan)
    for phase, expected in (
        ("pregrasp", pregrasp),
        ("grasp", grasp_pose),
        ("lift", lift),
        ("preplace", preplace),
        ("place", place),
        ("retreat", postplace),
    ):
        np.testing.assert_allclose(poses[phase], expected, atol=1e-9, err_msg=phase)


def test_validate_decision_rejects_top_down_on_side_grasp(monkeypatch, tmp_path):
    policy, pickup, receptacle = _policy_shell(monkeypatch, tmp_path)
    decision = _decision(approach_strategy="top_down")
    with pytest.raises(llm_module.PlanValidationError, match="approach_tilt"):
        policy._validate_decision(
            decision, pickup, receptacle, [_grasp(_LATERAL_ROT)], [["left_arm"]]
        )


def test_validate_decision_rejects_lift_below_receptacle_clearance(monkeypatch, tmp_path):
    policy, pickup, receptacle = _policy_shell(monkeypatch, tmp_path)
    # 该场景的 lift 几何下限是 1.09 - 0.85 = 0.24；0.10 落在 bounds 内但物理上不够。
    with pytest.raises(llm_module.PlanValidationError, match="too low to clear the receptacle"):
        policy._validate_decision(
            _decision(lift_height=0.10), pickup, receptacle, [_grasp(_TOPDOWN_ROT)],
            [["left_arm"]],
        )


def test_validate_decision_rejects_arm_and_candidate_mismatches(monkeypatch, tmp_path):
    policy, pickup, receptacle = _policy_shell(monkeypatch, tmp_path)
    grasp = _grasp(_TOPDOWN_ROT)
    with pytest.raises(llm_module.PlanValidationError, match="invalid arm"):
        policy._validate_decision(
            _decision(arm="right_arm"), pickup, receptacle, [grasp], [["left_arm"]]
        )
    with pytest.raises(llm_module.PlanValidationError, match="out of range"):
        policy._validate_decision(
            _decision(grasp_candidate_id=3), pickup, receptacle, [grasp], [["left_arm"]]
        )
    with pytest.raises(llm_module.PlanValidationError, match="not reachable by"):
        policy._validate_decision(_decision(), pickup, receptacle, [grasp], [[]])


def test_validate_decision_rejects_base_goal_outside_radius(monkeypatch, tmp_path):
    policy, pickup, receptacle = _policy_shell(monkeypatch, tmp_path)
    decision = _decision(base_approach_goal=[5.0, 0.0, 0.0])
    with pytest.raises(llm_module.PlanValidationError, match="expected radius|allowed radius"):
        policy._validate_decision(
            decision, pickup, receptacle, [_grasp(_TOPDOWN_ROT)], [["left_arm"]]
        )


def test_validate_decision_rejects_base_goal_on_immobile_base(monkeypatch, tmp_path):
    policy, pickup, receptacle = _policy_shell(monkeypatch, tmp_path)
    policy._base_is_movable = lambda: False
    decision = _decision(base_approach_goal=[0.1, 0.0, 0.0])
    with pytest.raises(llm_module.PlanValidationError, match="holonomic base"):
        policy._validate_decision(
            decision, pickup, receptacle, [_grasp(_TOPDOWN_ROT)], [["left_arm"]]
        )


def test_rejected_decision_feeds_error_and_previous_decision_back(monkeypatch, tmp_path):
    """第二次请求必须带上错误历史、上一次的决策和可调整的高层旋钮。"""
    policy, pickup, receptacle = _policy_shell(monkeypatch, tmp_path)
    monkeypatch.setattr(policy, "_preflight_kinematics_and_contacts", lambda plan, obj: None)

    bad = _decision(approach_strategy="top_down")
    good = _decision(approach_strategy="lateral")
    client = _ScriptedClient(
        [json.dumps(bad.model_dump(mode="json")), json.dumps(good.model_dump(mode="json"))]
    )
    policy.client = client

    plan = policy._request_valid_plan(pickup, receptacle, [_grasp(_LATERAL_ROT)], [["left_arm"]])

    assert len(client.payloads) == 2
    second = client.payloads[1]
    assert "approach_tilt" in second["previous_validation_error"]
    assert len(second["validation_error_history"]) == 1
    assert second["previous_decision"]["approach_strategy"] == "top_down"
    assert second["validation_feedback"]["reason"] == "tilt"
    assert "approach_strategy" in second["validation_feedback"]["llm_adjustable"]
    # 第一次请求没有上一轮决策可参考。
    assert client.payloads[0]["previous_decision"] is None

    attempt_1 = json.loads((tmp_path / "llm_plans" / "plan_test_attempt_1.json").read_text())
    attempt_2 = json.loads((tmp_path / "llm_plans" / "plan_test_attempt_2.json").read_text())
    assert attempt_1["valid"] is False and "approach_tilt" in attempt_1["error"]
    assert attempt_2["valid"] is True
    # 展开后的 waypoint 与模型原始决策都要落盘，便于后续三档对比。
    assert attempt_2["parsed_plan"]["segments"] and attempt_2["decision"]["grasp_candidate_id"] == 0
    assert plan.grasp_candidate_id == 0


def test_plan_build_error_is_not_retried(monkeypatch, tmp_path):
    """本地几何展开崩了属于代码缺陷，不能伪装成模型错误去白烧 API 调用。"""
    policy, pickup, receptacle = _policy_shell(monkeypatch, tmp_path)

    def boom(**kwargs):
        raise llm_module.PlanBuildError("bad geometry")

    monkeypatch.setattr(llm_module, "build_plan_from_decision", boom)
    good = _decision()
    client = _ScriptedClient([json.dumps(good.model_dump(mode="json"))])
    policy.client = client

    with pytest.raises(llm_module.PlanBuildError):
        policy._request_valid_plan(pickup, receptacle, [_grasp(_TOPDOWN_ROT)], [["left_arm"]])
    assert len(client.payloads) == 1


def test_geometric_source_skips_the_api_entirely(monkeypatch, tmp_path):
    policy, pickup, receptacle = _policy_shell(monkeypatch, tmp_path)
    monkeypatch.setattr(policy, "_preflight_kinematics_and_contacts", lambda plan, obj: None)
    policy.policy_config.llm_decision_source = "geometric"
    policy.policy_config.llm_max_api_calls = 3

    def explode(system_prompt, payload):
        raise AssertionError("geometric 档不得调用 API")

    policy.client = SimpleNamespace(complete=explode)

    plan = policy._request_valid_plan(pickup, receptacle, [_grasp(_TOPDOWN_ROT)], [["left_arm"]])

    assert policy._api_calls == 0
    assert plan.grasp_candidate_id == 0
    record = json.loads((tmp_path / "llm_plans" / "plan_test_attempt_1.json").read_text())
    assert record["valid"] is True and record["plan_source"] == "geometric"


def test_get_all_phases_covers_base_segments():
    policy = object.__new__(LLMWaypointPlannerPolicy)
    phases = LLMWaypointPlannerPolicy.get_all_phases(policy)
    assert phases["base_approach"] > phases["unknown"]
    assert phases["base_transfer"] > phases["base_approach"]


# ---------------------------------------------------------------------------
# 接触容差：区分网格外壳擦碰与真实碰撞
# ---------------------------------------------------------------------------


def _fake_contact_world(monkeypatch, contacts, self_tol=0.002, env_tol=0.001):
    """造一个只够跑 _assert_no_new_robot_contact 的 policy 壳。

    geom 5/6 属于机器人（body 1 -> root 0），geom 7/8 属于环境（body 2 -> root 1）。
    """
    monkeypatch.setattr(llm_module, "descendant_bodies", lambda model, root: {1})
    policy = object.__new__(LLMWaypointPlannerPolicy)
    policy.policy_config = SimpleNamespace(
        llm_self_contact_tolerance_m=self_tol,
        llm_environment_contact_tolerance_m=env_tol,
    )
    policy.robot_view = SimpleNamespace(
        root_body_id=0,
        get_move_group=lambda name: SimpleNamespace(root_body_id=1),
    )
    policy._selected_gripper_id = "left_gripper"
    policy._contact_pairs = lambda: dict(contacts)
    policy.task = SimpleNamespace(
        env=SimpleNamespace(
            current_model=SimpleNamespace(
                geom_bodyid={5: 1, 6: 1, 7: 2, 8: 2},
                body_rootid={1: 0, 2: 1},
                geom=lambda gid: SimpleNamespace(name=f"geom_{gid}"),
                body=lambda bid: SimpleNamespace(name=f"body_{bid}"),
            )
        )
    )
    return policy


def test_select_new_contacts_reports_only_new_or_worsened():
    baseline = {(1, 2): -0.0001}
    current = {(1, 2): -0.02, (3, 4): -0.0002, (5, 6): 0.001}
    # (1,2) 恶化、(3,4) 新增；(5,6) 是正间隙（无接触），(1,2) 若未恶化也不算。
    assert set(llm_module.select_new_contacts(current, baseline)) == {(1, 2), (3, 4)}
    assert llm_module.select_new_contacts({(1, 2): -0.0001}, baseline) == {}


def test_grazing_self_contact_is_tolerated(monkeypatch):
    """torso 内链的 0.12 mm 网格外壳互穿不该判失败。"""
    policy = _fake_contact_world(monkeypatch, {(5, 6): -0.00012})
    ignored, depth = policy._assert_no_new_robot_contact({}, "pregrasp", pickup_root=9)
    assert ignored == 1
    assert depth == pytest.approx(0.00012)


def test_self_tolerance_is_wider_than_environment_tolerance(monkeypatch):
    """1.84 mm：机器人自身 link 互穿放行，同一深度撞到环境物体则失败。"""
    depth_mm = -0.00184
    self_contact = _fake_contact_world(monkeypatch, {(5, 6): depth_mm})
    ignored, _ = self_contact._assert_no_new_robot_contact({}, "pregrasp", pickup_root=9)
    assert ignored == 1

    env_contact = _fake_contact_world(monkeypatch, {(5, 7): depth_mm})
    with pytest.raises(llm_module.PlanValidationError, match="robot-environment contact"):
        env_contact._assert_no_new_robot_contact({}, "pregrasp", pickup_root=9)


def test_deep_self_contact_still_fails(monkeypatch):
    """臂/端效器真撞胸是 20 mm 级，必须继续拦住。"""
    policy = _fake_contact_world(monkeypatch, {(5, 6): -0.020})
    with pytest.raises(llm_module.PlanValidationError, match="self-contact"):
        policy._assert_no_new_robot_contact({}, "pregrasp", pickup_root=9)


def test_zero_tolerance_restores_old_behaviour(monkeypatch):
    policy = _fake_contact_world(
        monkeypatch, {(5, 6): -0.00012}, self_tol=0.0, env_tol=0.0
    )
    with pytest.raises(llm_module.PlanValidationError, match="penetration"):
        policy._assert_no_new_robot_contact({}, "pregrasp", pickup_root=9)


def test_baseline_contact_worsening_is_caught(monkeypatch):
    """基线里已有 0.1 mm 擦碰、后来恶化成 20 mm：旧的集合差会放过，现在必须报。"""
    policy = _fake_contact_world(monkeypatch, {(5, 6): -0.020})
    with pytest.raises(llm_module.PlanValidationError, match="penetration 20.00 mm"):
        policy._assert_no_new_robot_contact(
            {(5, 6): -0.0001}, "lift", pickup_root=9
        )


def test_environment_contact_is_labelled_separately(monkeypatch):
    policy = _fake_contact_world(monkeypatch, {(5, 7): -0.02})
    with pytest.raises(llm_module.PlanValidationError, match="robot-environment contact"):
        policy._assert_no_new_robot_contact({}, "lift", pickup_root=9)


def test_collision_substring_survives_in_error_message(monkeypatch):
    """summarize_llm_waypoint_eval.py 按 'collision' 子串分桶，措辞不能改掉它。"""
    policy = _fake_contact_world(monkeypatch, {(5, 6): -0.02})
    with pytest.raises(llm_module.PlanValidationError, match="new MuJoCo collision"):
        policy._assert_no_new_robot_contact({}, "grasp", pickup_root=9)


# --- 分步决策与多次底盘移动 --------------------------------------------------
#
# 旧语法把底盘移动焊死在两个固定槽位（base_approach 只能首位、base_transfer
# 只能夹在 lift 与 preplace 之间），"抓前转向 → 抬起后横移 → 放前微调"这类
# 多段调整无法表达；而实测 episode 起始底盘位置 100% 落在 standoff 内、朝向偏差
# 中位 21.1 度（26.7% 超过 30 度），恰恰需要"抓之前先把身子转过来"。下列用例锁定
# 放松后的语法边界。


def _base_seg(phase, goal):
    return {"phase": phase, "motion": "base", "base_goal": list(goal), "duration_s": 3.0}


def _ee_seg_list():
    return [
        {"phase": name, "motion": "ee", "waypoints": [_pose(z=0.4 + i * 0.02)]}
        for i, name in enumerate(
            ("pregrasp", "grasp", "lift", "preplace", "place", "retreat")
        )
    ]


def _plan_data(segments):
    return {
        "schema_version": 1,
        "robot": "rby1",
        "arm": "left_arm",
        "grasp_candidate_id": 0,
        "segments": segments,
    }


def test_plan_grammar_allows_multiple_base_moves_before_grasp():
    """抓取前可以有多段底盘移动（先把身子转过来，再挪到 standoff）。"""
    ee = _ee_seg_list()
    segments = (
        [_base_seg("base_approach", [0.10, 0.0, 0.50])]
        + [ee[0]]
        + [_base_seg("base_approach", [0.15, 0.0, 0.50])]
        + ee[1:]
    )
    plan = LLMWaypointPlan.model_validate(_plan_data(segments))
    assert [segment.phase for segment in plan.segments] == [
        "base_approach",
        "pregrasp",
        "base_approach",
        "grasp",
        "lift",
        "preplace",
        "place",
        "retreat",
    ]


def test_plan_grammar_rejects_adjacent_base_segments():
    """相邻底盘段等价于一次移动，拆开只会重复 duration 与预检采样。"""
    segments = [
        _base_seg("base_approach", [0.10, 0.0, 0.50]),
        _base_seg("base_approach", [0.20, 0.0, 0.50]),
    ] + _ee_seg_list()
    with pytest.raises(ValidationError, match="must not be adjacent"):
        LLMWaypointPlan.model_validate(_plan_data(segments))


def test_plan_grammar_rejects_base_approach_after_grasp():
    """底盘段命名按夹持状态划分：已夹持时必须用 base_transfer。"""
    ee = _ee_seg_list()
    segments = (
        ee[:2]
        + [_base_seg("base_approach", [0.20, 0.0, 0.50])]
        + ee[2:]
    )
    with pytest.raises(ValidationError, match="base_approach must occur before grasp"):
        LLMWaypointPlan.model_validate(_plan_data(segments))


def test_build_plan_from_steps_moves_base_before_chosen_phases():
    """steps 里挂在相位上的 base_goal 在该相位之前插入底盘段，段名按夹持状态取。"""
    plan = _build(
        _decision(
            steps=[
                {"phase": "pregrasp", "base_goal": [0.1, 0.0, 0.5]},
                {"phase": "grasp"},
                {"phase": "lift"},
                {"phase": "preplace", "base_goal": [0.2, 0.0, 0.5]},
                {"phase": "place"},
                {"phase": "retreat"},
            ]
        ),
        _geometry(_grasp(_LATERAL_ROT)),
    )
    assert [(segment.phase, segment.motion) for segment in plan.segments] == [
        ("base_approach", "base"),
        ("pregrasp", "ee"),
        ("grasp", "ee"),
        ("lift", "ee"),
        ("base_transfer", "base"),
        ("preplace", "ee"),
        ("place", "ee"),
        ("retreat", "ee"),
    ]


def test_steps_and_flat_expansion_agree_on_ee_poses():
    """两条展开路径的六个 EE 位姿必须逐位相同——差别只在底盘段的数量与位置。

    模型在 steps 里能决定的只有"哪一步之前把底盘摆到哪"，它依旧给不出任何位姿。
    """
    grasp = _grasp(_TOPDOWN_ROT)
    geometry = _geometry(grasp)
    flat = _build(_decision(), geometry)
    stepped = _build(
        _decision(
            steps=[
                {"phase": "pregrasp", "base_goal": [0.1, 0.0, 0.5]},
                {"phase": "grasp"},
                {"phase": "lift"},
                {"phase": "preplace", "base_goal": [0.2, 0.0, 0.5]},
                {"phase": "place"},
                {"phase": "retreat"},
            ]
        ),
        geometry,
    )
    flat_poses, stepped_poses = _ee_poses(flat), _ee_poses(stepped)
    assert set(flat_poses) == set(stepped_poses)
    for phase, pose in flat_poses.items():
        np.testing.assert_allclose(stepped_poses[phase], pose, atol=1e-12)


def test_validate_decision_rejects_steps_plus_flat_base_goals(monkeypatch, tmp_path):
    """steps 与扁平底盘字段是同一件事的两种写法，同时给会含糊，直接拒绝。"""
    policy, pickup, receptacle = _policy_shell(monkeypatch, tmp_path)
    decision = _decision(
        base_approach_goal=[0.1, 0.0, 0.5],
        steps=[{"phase": name} for name in
               ("pregrasp", "grasp", "lift", "preplace", "place", "retreat")],
    )
    with pytest.raises(PlanValidationError, match="two ways to say the same thing"):
        policy._validate_decision(
            decision, pickup, receptacle, [_grasp(_LATERAL_ROT)], [["left_arm"]]
        )


def test_validate_decision_rejects_incomplete_steps(monkeypatch, tmp_path):
    policy, pickup, receptacle = _policy_shell(monkeypatch, tmp_path)
    decision = _decision(steps=[{"phase": "pregrasp"}, {"phase": "grasp"}])
    with pytest.raises(PlanValidationError, match="missing phases"):
        policy._validate_decision(
            decision, pickup, receptacle, [_grasp(_LATERAL_ROT)], [["left_arm"]]
        )


def test_validate_decision_caps_base_moves(monkeypatch, tmp_path):
    """每段底盘至少占 2 s，回合预算有限，超过 llm_base_max_moves 直接拒绝。"""
    policy, pickup, receptacle = _policy_shell(monkeypatch, tmp_path)
    decision = _decision(
        steps=[
            {"phase": "pregrasp", "base_goal": [0.1, 0.0, 0.5]},
            {"phase": "grasp", "base_goal": [0.1, 0.0, 0.5]},
            {"phase": "lift", "base_goal": [0.1, 0.0, 0.5]},
            {"phase": "preplace", "base_goal": [0.1, 0.0, 0.5]},
            {"phase": "place"},
            {"phase": "retreat"},
        ]
    )
    with pytest.raises(PlanValidationError, match="llm_base_max_moves"):
        policy._validate_decision(
            decision, pickup, receptacle, [_grasp(_LATERAL_ROT)], [["left_arm"]]
        )


def test_validate_decision_rejects_base_goal_on_retreat(monkeypatch, tmp_path):
    """retreat 在放置之后、已松手，底盘段命名规则（按夹持状态划分）在那里两不靠。"""
    policy, pickup, receptacle = _policy_shell(monkeypatch, tmp_path)
    decision = _decision(
        steps=[
            {"phase": "pregrasp"},
            {"phase": "grasp"},
            {"phase": "lift"},
            {"phase": "preplace"},
            {"phase": "place"},
            {"phase": "retreat", "base_goal": [0.1, 0.0, 0.5]},
        ]
    )
    with pytest.raises(PlanValidationError, match="retreat cannot carry a base_goal"):
        policy._validate_decision(
            decision, pickup, receptacle, [_grasp(_LATERAL_ROT)], [["left_arm"]]
        )


def test_scene_payload_exposes_yaw_error_and_candidate_menu(monkeypatch, tmp_path):
    """prompt 必须暴露当前朝向偏差与可选底盘位姿。

    旧 payload 里 recommended_base_approach_goal 恒为 null——起始距离 100% 落在
    standoff 内——模型看不到"自己歪了 40 度"，也就从不转向。
    """
    policy, pickup, receptacle = _policy_shell(monkeypatch, tmp_path, base_xy=(0.0, 0.0))
    # 典型失败态：位置已在 standoff 内，朝向偏 90 度。
    policy.robot_view.base.pose = _yaw_pose(0.0, 0.0, 90.0)

    payload = policy._scene_payload(
        pickup, receptacle, [_grasp(_LATERAL_ROT)], [["left_arm"]], None, []
    )
    reference = payload["base_reference"]
    assert reference["base_yaw_deg"] == pytest.approx(90.0)
    # pickup 在 (0.70, 0.10)，从原点看的方位角约 8.13 度，减掉底盘 yaw 90 度。
    assert reference["alignment"]["pickup"]["yaw_error_deg"] == pytest.approx(-81.87, abs=0.05)

    candidates = reference["candidates"]["pickup"]
    assert candidates[0]["id"] == "pickup_rotate_in_place"
    assert candidates[0]["translation_m"] == 0.0
    # 向后兼容的单值推荐现在由菜单派生，因此不再是恒 null。
    assert reference["recommended_base_approach_goal"] == candidates[0]["goal"]


def test_resolve_base_goal_ref_translates_ids_and_rejects_unknown(monkeypatch, tmp_path):
    """候选 id 只能引用本次 prompt 真正给出的值。"""
    policy, pickup, receptacle = _policy_shell(monkeypatch, tmp_path, base_xy=(0.0, 0.0))
    policy.robot_view.base.pose = _yaw_pose(0.0, 0.0, 90.0)
    menu = policy._base_menu(pickup, receptacle)

    resolved = policy._resolve_base_goal_refs(
        _decision(base_approach_goal="pickup_rotate_in_place"), menu
    )
    assert resolved.base_approach_goal == menu["pickup"][0]["goal"]

    with pytest.raises(PlanValidationError, match="unknown base candidate"):
        policy._resolve_base_goal_refs(_decision(base_approach_goal="not_a_candidate"), menu)


def test_feedback_attaches_base_candidates_for_collision(monkeypatch, tmp_path):
    """碰撞类失败要把失败相位对应的底盘候选一并回灌。

    旧反馈只把 base_approach_goal 列进 llm_adjustable，模型仍得自己编世界坐标，
    实测编出来的值多半过不了预检。
    """
    policy, pickup, receptacle = _policy_shell(monkeypatch, tmp_path, base_xy=(0.0, 0.0))
    policy.robot_view.base.pose = _yaw_pose(0.0, 0.0, 90.0)
    menu = policy._base_menu(pickup, receptacle)

    feedback = policy._feedback_from_error(
        PlanValidationError("pregrasp: new MuJoCo collision between geoms ('a', 'b')"), menu
    )
    assert feedback["reason"] == "collision"
    assert feedback["failed_phase"] == "pregrasp"
    assert feedback["base_candidates"]["pickup"][0]["id"] == "pickup_rotate_in_place"

