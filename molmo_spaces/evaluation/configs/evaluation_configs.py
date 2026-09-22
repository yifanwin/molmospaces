"""

These configs are EXAMPLES of how to set up evaluation configs for use
with JSON benchmarks via molmo_spaces.evaluation.run_evaluation(). The anticipated
pattern is that users will create their own eval configs in their own repositories,
import run_evaluation from molmo_spaces.evaluation, and pass their config to it.

Example usage from an external repo:
    from molmo_spaces.evaluation import run_evaluation
    from my_repo.configs import MyPolicyEvalConfig

    results = run_evaluation(
        eval_config_cls=MyPolicyEvalConfig,
        benchmark_dir="/path/to/benchmark",
        checkpoint_path="/path/to/checkpoint",
    )

Eval configs provide:
- Robot config (factories for instantiation, gravcomp settings)
- Policy config (checkpoint path, camera names, action spec)
- Timing parameters (policy_dt_ms, ctrl_dt_ms, sim_dt_ms)

Episode-specific data (init_qpos, robot_base_pose, cameras, object_poses, task config)
comes from the JSON benchmark files, not from these configs. The benchmark JSON
is strictly authoritative for episode initialization.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import ClassVar

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.policy_configs import (
    BrownianMotionPolicyConfig,
    DummyPolicyConfig,
    LLMWaypointPlannerPolicyConfig,
    RBY1RLPolicyConfig,
    grasp_settle_steps,
)
from molmo_spaces.configs.policy_configs_baselines import (
    CAPPolicyConfig,
    DreamZeroPolicyConfig,
    PiPolicyConfig,
    TeleopPolicyConfig,
)
from molmo_spaces.configs.robot_configs import (
    ActionNoiseConfig,
    FrankaCAPRobotConfig,
    FrankaRobotConfig,
)
from molmo_spaces.configs.task_configs import (
    BaseMujocoTaskConfig,
    NavToObjTaskConfig,
    PickAndPlaceColorTaskConfig,
    PickAndPlaceTaskConfig,
)
from molmo_spaces.configs.task_sampler_configs import (
    BaseMujocoTaskSamplerConfig,
    NavToObjTaskSamplerConfig,
    PickAndPlaceColorTaskSamplerConfig,
    PickAndPlaceTaskSamplerConfig,
)
from molmo_spaces.data_generation.config.nav_to_obj_configs import NavToObjDataGenConfig
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
    FrankaPickAndPlaceDataGenConfig,
    PandaOmronCuroboPickAndPlaceDataGenConfig,
    PandaOmronPickAndPlaceDataGenConfig,
    RBY1PickAndPlaceDataGenConfig,
)
from molmo_spaces.policy.dummy_policy import BrownianMotionPolicy, DummyPolicy
from molmo_spaces.tasks.nav_task import NavToObjTask
from molmo_spaces.tasks.nav_task_sampler import NavToObjTaskSampler
from molmo_spaces.tasks.pick_and_place_color_task import PickAndPlaceColorTask
from molmo_spaces.tasks.pick_and_place_color_task_sampler import (
    PickAndPlaceColorTaskSampler,
)
from molmo_spaces.tasks.pick_and_place_task import PickAndPlaceTask
from molmo_spaces.tasks.pick_and_place_task_sampler import (
    PickAndPlaceTaskSampler,
)
from molmo_spaces.tasks.task_sampler import BaseMujocoTaskSampler
from molmo_spaces.utils.function_utils import make_lenient

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class RBY1CuroboPickPnPEvalConfig(RBY1PickAndPlaceDataGenConfig):
    """Oracle RBY1 pick-and-place evaluation with the existing CuRobo planner.

    The JSON benchmark remains authoritative for each episode's scene, robot,
    object, receptacle, and camera setup.  Unlike the learned MolmoBot policy,
    the inherited planner reads the simulator's ground-truth task state and
    uses CuRobo IK/TrajOpt; RGB observations are recorded for diagnostics but
    are not used to choose actions.
    """

    # Keep all episodes so the evaluator can report both successes and failures.
    requires_policy_auxiliary_objects: ClassVar[bool] = True
    filter_for_successful_trajectories: bool = False
    use_wandb: bool = False

    # 基类 MlSpacesExpConfig 的默认值为 False，若不覆盖，成功瞬间不会终止 rollout，
    # judge_success() 只在循环结束后调用一次，瞬时成功会被最终状态覆盖而记为失败。
    # 20260917_103152 那次运行即因此把成功率低估了 2.1 倍（2.67% vs 5.61%）。
    end_on_success: bool = True

    # Match the RBY1 benchmark/data-generation control rates.
    policy_dt_ms: float = 100.0
    ctrl_dt_ms: float = 20.0
    sim_dt_ms: float = 4.0
    # 与 20260917_103152 运行保持一致（该次运行的 pkl 中为 600）。
    task_horizon: int = 600

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.policy_config is None:
            raise RuntimeError(
                "CuRobo policy initialization failed. Run this config in a CUDA-enabled "
                "environment with the molmospaces curobo extra installed."
            )

        # RBY1PickAndPlaceDataGenConfig currently uses local CuRobo, but keep
        # this explicit so an upstream default change cannot silently switch
        # this oracle evaluation to a remote planner service.
        self.policy_config.server_urls = []
        self.robot_config.action_noise_config.enabled = False


class RBY1LLMWaypointPickPnPEvalConfig(RBY1PickAndPlaceDataGenConfig):
    """RBY1 pick-and-place evaluation with API-generated, locally checked waypoints."""

    requires_task_bound_policy: ClassVar[bool] = True
    filter_for_successful_trajectories: bool = False
    end_on_success: bool = True
    use_wandb: bool = False
    policy_dt_ms: float = 100.0
    ctrl_dt_ms: float = 20.0
    sim_dt_ms: float = 4.0
    task_horizon: int = 600
    policy_config: LLMWaypointPlannerPolicyConfig | None = None

    def _init_policy_config(self) -> LLMWaypointPlannerPolicyConfig:
        return LLMWaypointPlannerPolicyConfig(
            # The policy replaces this with the selected left/right arm before IK.
            ik_unlocked_move_group_ids=["left_arm"],
            go_home_move_group_ids=[],
            # 抓取候选姿态本身可能与环境相交（house 4 的 grasp 阶段碰撞即此类）；
            # 开启后由 add_auxiliary_objects 注入 grasp_collision_* 辅助体，
            # 供 _candidate_grasps 用 get_noncolliding_grasp_mask 过滤。
            filter_colliding_grasps=True,
            # RBY1 只暴露顺序 IK（约 1.3 s/次），原值 3 次调用在候选池大、
            # 校验反馈长时容易耗尽；提高到 8 次换取纠错机会。
            llm_max_api_calls=8,
            # 候选池过小会让 LLM 没有备选；配合 ik_checks 一起放大。
            llm_max_grasp_candidates=24,
            llm_max_grasp_ik_checks=96,
            # RBY1 的 torso 初始为全 0，只解锁单臂时桌面目标够不到
            # （实测 house 103 单臂残差 0.52 m / 172°，加 torso 后立即解出）。
            # base 不动：本 policy 的直线插值不是避障导航器。
            llm_ik_extra_groups=["torso"],
        )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        from molmo_spaces.policy.solvers.object_manipulation.llm_waypoint_planner_policy import (
            validate_llm_environment,
        )

        if self.policy_config is None:
            self.policy_config = self._init_policy_config()
        # geometric 档不调 API，因此不能在这里做凭据预检——否则没有 LLM 环境变量
        # 的机器上根本跑不了 A/B 对照。
        if self.policy_config.llm_decision_source == "llm":
            validate_llm_environment(self.policy_config.api_timeout_s)
        self.robot_config.action_noise_config.enabled = False


class RBY1LLMWaypointPickPnPGeometricEvalConfig(RBY1LLMWaypointPickPnPEvalConfig):
    """RBY1 对照档：同一套几何生成器，但决策来自本地几何默认值而非 LLM。

    与 RBY1CuroboPickPnPEvalConfig（上界）和 RBY1LLMWaypointPickPnPEvalConfig
    （LLM 决策）一起构成三档比较，用来回答"LLM 的高层决策相比几何默认值有没有
    增量价值"。不需要任何 LLM 环境变量。
    """

    requires_task_bound_policy: ClassVar[bool] = True
    policy_config: LLMWaypointPlannerPolicyConfig | None = None

    def _init_policy_config(self) -> LLMWaypointPlannerPolicyConfig:
        return super()._init_policy_config().model_copy(
            update={
                "llm_decision_source": "geometric",
                # 底盘先停到物体 standoff 处再求解。不动底盘时，RBY1 的顺序差分 IK
                # 必须靠 torso 去够桌面目标，实测会把 torso 解成自穿插构型
                # （link_torso_2 撞 link_torso_4），本地碰撞预检直接判失败。
                "llm_geometric_base_approach": True,
            }
        )


class RBY1RLEvalConfig(RBY1PickAndPlaceDataGenConfig):
    """RBY1 benchmark config for a staged SAC policy over CuRobo execution."""

    requires_task_bound_policy: ClassVar[bool] = True
    requires_policy_auxiliary_objects: ClassVar[bool] = True
    filter_for_successful_trajectories: bool = False
    end_on_success: bool = True
    use_wandb: bool = False
    policy_dt_ms: float = 100.0
    ctrl_dt_ms: float = 20.0
    sim_dt_ms: float = 4.0
    task_horizon: int = 600
    policy_config: RBY1RLPolicyConfig | None = None

    def _init_policy_config(self) -> RBY1RLPolicyConfig:
        base = super()._init_policy_config()
        return RBY1RLPolicyConfig.model_validate(base.model_dump())

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.policy_config is None:
            raise RuntimeError(
                "RBY1 RL evaluation requires the CUDA CuRobo runtime used by the existing "
                "RBY1 benchmark planner."
            )
        self.robot_config.action_noise_config.enabled = False


class PandaOmronCuroboPickPnPEvalConfig(
    PandaOmronCuroboPickAndPlaceDataGenConfig
):
    """PandaOmron CuRobo oracle for cross-robot JSON benchmark evaluation."""

    filter_for_successful_trajectories: bool = False
    end_on_success: bool = True
    use_wandb: bool = False
    policy_dt_ms: float = 66.0
    ctrl_dt_ms: float = 2.0
    sim_dt_ms: float = 2.0
    task_horizon: int = 606

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.policy_config is None:
            raise RuntimeError(
                "PandaOmron CuRobo initialization failed. Install both the robosuite "
                "and curobo extras in the active environment."
            )
        self.policy_config.server_urls = []
        self.robot_config.action_noise_config.enabled = False
        # 抓取判定必须等夹爪闭合完成才能做：判据是"夹爪位置偏离闭合位"，而闭合
        # 过程中仍在运动的手指同样偏离闭合位，窗口太短就会把"还在合拢"误判成
        # "已夹住物体"。原先 5 步 × 66 ms = 330 ms 短于 500 ms 的闭合时长，
        # 97.4% 的判定通过、70.7% 随后在 LIFT 阶段掉落。步数按 policy_dt_ms 反推。
        self.policy_config.max_grasping_timesteps = grasp_settle_steps(
            self.policy_dt_ms, self.policy_config.gripper_close_duration
        )


class PandaOmronLLMWaypointPickPnPEvalConfig(PandaOmronPickAndPlaceDataGenConfig):
    """House-filterable PandaOmron evaluation using API-generated waypoints."""

    requires_task_bound_policy: ClassVar[bool] = True
    policy_config: LLMWaypointPlannerPolicyConfig = LLMWaypointPlannerPolicyConfig(
        ik_unlocked_move_group_ids=["arm"],
        go_home_move_group_ids=[],
    )
    filter_for_successful_trajectories: bool = False
    end_on_success: bool = True
    use_wandb: bool = False
    policy_dt_ms: float = 66.0
    ctrl_dt_ms: float = 2.0
    sim_dt_ms: float = 2.0
    task_horizon: int = 606

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        from molmo_spaces.policy.solvers.object_manipulation.llm_waypoint_planner_policy import (
            validate_llm_environment,
        )

        if self.policy_config.llm_decision_source == "llm":
            validate_llm_environment(self.policy_config.api_timeout_s)
        self.robot_config.action_noise_config.enabled = False


class PandaOmronLLMWaypointPickPnPGeometricEvalConfig(PandaOmronLLMWaypointPickPnPEvalConfig):
    """PandaOmron 对照档：同一套几何生成器，决策来自本地几何默认值而非 LLM。"""

    requires_task_bound_policy: ClassVar[bool] = True
    policy_config: LLMWaypointPlannerPolicyConfig = LLMWaypointPlannerPolicyConfig(
        ik_unlocked_move_group_ids=["arm"],
        go_home_move_group_ids=[],
        llm_decision_source="geometric",
    )


class JsonBenchmarkEvalConfig(MlSpacesExpConfig):
    """
    Minimal base config for JSON benchmark evaluation.

    This config is designed for use ONLY with JSON benchmarks. It provides
    the minimal infrastructure needed to run a learned policy against a
    benchmark where all episode-specific data (task type, cameras, robot poses,
    object poses, etc.) comes from the benchmark JSON.

    Subclass this and provide:
    - robot_config: Robot configuration for instantiation
    - policy_config: Your learned policy configuration

    DO NOT provide task_sampler_config or task_config - those are placeholders
    that will be overridden by the benchmark. If you accidentally try to use
    this config for data generation (not evaluation), it will fail because
    the task sampler/config are minimal stubs.

    Example:
        class MyPolicyBenchmarkEvalConfig(JsonBenchmarkEvalConfig):
            robot_config = FrankaRobotConfig()
            policy_config = MyPolicyConfig(checkpoint_path="/path/to/ckpt")
    """

    # Required infrastructure - subclasses must provide robot_config and policy_config

    # Timing parameters - can be overridden per-policy as needed
    num_envs: int = 1
    num_workers: int = 1
    policy_dt_ms: float = 66.0
    ctrl_dt_ms: float = 2.0
    sim_dt_ms: float = 2.0
    task_horizon: int = 500

    # Viewer config (usually disabled for eval)
    use_passive_viewer: bool = False
    viewer_cam_dict: dict = {
        "distance": 5.0,
        "azimuth": 45.0,
        "elevation": -30.0,
        "lookat": [0.0, 0.0, 0.5],
    }

    # These are overridden by benchmark - provide placeholders to satisfy base class
    # DO NOT rely on these values; the benchmark JSON is authoritative.
    task_type: str = "pick"  # Overridden per-episode from benchmark
    scene_dataset: str = "procthor-10k"  # Overridden per-episode from benchmark
    data_split: str = "val"  # Overridden per-episode from benchmark
    camera_config: None = None  # Overridden per-episode from benchmark

    # Minimal stubs - these exist only to satisfy the base class.
    # JsonEvalTaskSampler replaces these entirely with benchmark data.
    # Note: task_sampler_class must be a valid class (not None) since pipeline.py
    # instantiates a worker-level task sampler. JsonEvalRunner overrides the per-episode
    # task sampler via get_episode_task_sampler, so this worker-level sampler is unused.
    task_sampler_config: BaseMujocoTaskSamplerConfig = BaseMujocoTaskSamplerConfig(
        task_sampler_class=BaseMujocoTaskSampler,
        house_inds=[0],  # Dummy value, overridden by JsonEvalRunner from benchmark
        samples_per_house=1,
        task_batch_size=1,
        max_tasks=10000,
    )
    task_config: BaseMujocoTaskConfig = BaseMujocoTaskConfig(task_cls=None)

    # Output config
    output_dir: Path = Path("eval_output")
    use_wandb: bool = False
    wandb_project: str = "mlspaces-benchmark-eval"
    filter_for_successful_trajectories: bool = False

    # Episode termination
    terminate_upon_success: bool = False

    @property
    def tag(self) -> str:
        return "json_benchmark_eval"


class DummyBenchmarkEvalConfig(JsonBenchmarkEvalConfig):
    """
    Test config that inherits from JsonBenchmarkEvalConfig.

    This tests the recommended pattern from evaluation/README.md:
    external repos should inherit from JsonBenchmarkEvalConfig and provide
    their robot_config and policy_config. The benchmark JSON provides all
    episode-specific data (cameras, poses, task params).

    Note: Prefixed with underscore to avoid pytest collection warning since
    this inherits from a class with __init__.
    """

    # Timing - short horizon for testing
    task_horizon: int = 10
    seed: int = 42
    policy_dt_ms: float = 200.0

    # Robot config - standard Franka
    robot_config: FrankaRobotConfig = FrankaRobotConfig()

    # Policy config - DummyPolicy returns empty dict (no-op)
    policy_config: DummyPolicyConfig = DummyPolicyConfig()

    @property
    def tag(self) -> str:
        return "dummy_json_benchmark"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        # Disable action noise for deterministic testing
        self.robot_config.action_noise_config = ActionNoiseConfig(enabled=False)


class PiPolicyEvalConfig(JsonBenchmarkEvalConfig):
    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_config: PiPolicyConfig = PiPolicyConfig()
    # policy_dt_ms: float = 200.0  # Match your model's expected control rate
    policy_dt_ms: float = 66.0  # ~15hz
    end_on_success: bool = True  # End episode immediately upon success, ignoring task_horizon

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False


class CAPPolicyEvalConfig(JsonBenchmarkEvalConfig):
    robot_config: FrankaCAPRobotConfig = FrankaCAPRobotConfig()
    policy_config: CAPPolicyConfig = CAPPolicyConfig()
    policy_dt_ms: float = 500.0  # Match your model's expected control rate

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False


class TeleopPolicyEvalConfig(JsonBenchmarkEvalConfig):
    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_config: TeleopPolicyConfig = TeleopPolicyConfig()
    policy_dt_ms: float = 40

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False


# @register_config("DummyPickPlaceEvalConfig")
class DummyPickPlaceEvalConfig(FrankaPickAndPlaceDataGenConfig):
    """Evaluation config for Dummy pick and place."""

    wandb_project: str = "dummy-eval"
    use_wandb: bool = False
    use_passive_viewer: bool = False
    wandb_name: str = f"dummy_pick_place_eval_{TIMESTAMP}"
    filter_for_successful_trajectories: bool = False
    task_type: str = "pick_and_place"
    task_horizon: int = 600
    output_dir: Path = Path("eval_output") / f"dummy_{TIMESTAMP}"

    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        house_inds=[5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145],
        samples_per_house=3,
    )
    task_config: PickAndPlaceTaskConfig = PickAndPlaceTaskConfig(task_cls=PickAndPlaceTask)

    policy_config: DummyPolicyConfig = DummyPolicyConfig()

    def _init_policy_config(self) -> DummyPolicyConfig:
        self.policy_config.policy_cls = DummyPolicy
        self.policy_config.policy_factory = make_lenient(DummyPolicy)
        return self.policy_config

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False


# @register_config("BrownianMotionPickPlaceEvalConfig")
class BrownianMotionPickPlaceEvalConfig(FrankaPickAndPlaceDataGenConfig):
    """Evaluation config for Dummy pick and place."""

    wandb_project: str = "brownian-motion-eval"
    use_wandb: bool = False
    use_passive_viewer: bool = False
    wandb_name: str = f"brownian_motion_pick_place_eval_{TIMESTAMP}"
    filter_for_successful_trajectories: bool = False
    task_type: str = "pick_and_place"
    task_horizon: int = 600
    output_dir: Path = Path("eval_output") / f"brownian_motion_{TIMESTAMP}"

    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        house_inds=[5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145],
        samples_per_house=3,
    )
    task_config: PickAndPlaceTaskConfig = PickAndPlaceTaskConfig(task_cls=PickAndPlaceTask)

    policy_config: BrownianMotionPolicyConfig = BrownianMotionPolicyConfig()

    def _init_policy_config(self) -> BrownianMotionPolicyConfig:
        self.policy_config.policy_cls = BrownianMotionPolicy
        self.policy_config.policy_factory = make_lenient(BrownianMotionPolicy)
        return self.policy_config

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False


# @register_config("BrownianMotionPickPlaceColorEvalConfig")
class BrownianMotionPickPlaceColorEvalConfig(BrownianMotionPickPlaceEvalConfig):
    wandb_name: str = f"brownian_motion_pick_place_color_eval_{TIMESTAMP}"
    task_type: str = "pick_and_place_color"

    task_sampler_config: PickAndPlaceColorTaskSamplerConfig = PickAndPlaceColorTaskSamplerConfig(
        task_sampler_class=PickAndPlaceColorTaskSampler,
        house_inds=[5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145],
        samples_per_house=3,
    )
    task_config: PickAndPlaceColorTaskConfig = PickAndPlaceColorTaskConfig(
        task_cls=PickAndPlaceColorTask
    )


class DreamZeroPolicyEvalConfig(JsonBenchmarkEvalConfig):
    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_config: DreamZeroPolicyConfig = DreamZeroPolicyConfig()
    policy_dt_ms: float = 66.0

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False


class DummyNavToObjEvalConfig(NavToObjDataGenConfig):
    """Evaluation config for Dummy pick and place."""

    wandb_project: str = "dummy-eval"
    use_wandb: bool = False
    use_passive_viewer: bool = False
    wandb_name: str = f"dummy_nav_to_obj_eval_{TIMESTAMP}"
    filter_for_successful_trajectories: bool = False
    task_type: str = "nav_to_obj"
    task_horizon: int = 600
    output_dir: Path = Path("eval_output") / f"dummy_{TIMESTAMP}"

    task_sampler_config: NavToObjTaskSamplerConfig = NavToObjTaskSamplerConfig(
        task_sampler_class=NavToObjTaskSampler,
        house_inds=[5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145],
        samples_per_house=3,
    )
    task_config: NavToObjTaskConfig = NavToObjTaskConfig(task_cls=NavToObjTask)

    policy_config: DummyPolicyConfig = DummyPolicyConfig()

    def _init_policy_config(self) -> DummyPolicyConfig:
        self.policy_config.policy_cls = DummyPolicy
        self.policy_config.policy_factory = make_lenient(DummyPolicy)
        return self.policy_config

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False


class BrownianNavToObjEvalConfig(DummyNavToObjEvalConfig):
    policy_config: BrownianMotionPolicyConfig = BrownianMotionPolicyConfig()

    def _init_policy_config(self) -> BrownianMotionPolicyConfig:
        self.policy_config.policy_cls = BrownianMotionPolicy
        self.policy_config.policy_factory = make_lenient(BrownianMotionPolicy)
        return self.policy_config
