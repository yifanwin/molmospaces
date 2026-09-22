"""Policy configuration classes for MolmoSpaces experiments."""

from __future__ import annotations

import os
import math
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import model_validator

from molmo_spaces.configs.abstract_config import Config
from molmo_spaces.planner.astar_planner import AStarPlannerConfig
from molmo_spaces.policy.base_policy import BasePolicy, PolicyFactory
from molmo_spaces.utils.function_utils import make_lenient

# Import CuroboPlannerConfig if available (requires GPU), otherwise create a stub
try:
    # P0 等纯 MuJoCo 工程验证可显式禁用可选 GPU 依赖；默认行为保持不变。
    if os.environ.get("MLSPACES_DISABLE_CUROBO") == "1":
        raise ImportError("CuRobo explicitly disabled for this process")
    from molmo_spaces.planner.curobo_planner import CuroboPlannerConfig
except (ImportError, RuntimeError):
    # Create a stub class when CuRobo isn't available (e.g., on non-GPU nodes)
    # This allows Pydantic to resolve forward references during config validation
    if TYPE_CHECKING:
        from molmo_spaces.planner.curobo_planner import CuroboPlannerConfig
    else:

        class CuroboPlannerConfig(Config):  # type: ignore
            """Stub for CuroboPlannerConfig when CuRobo is not available."""

            pass


class BasePolicyConfig(Config):
    """Base configuration for policies."""

    policy_cls: type[BasePolicy]
    policy_factory: PolicyFactory
    """
    Factory function to create the policy instance from a config and task, can be same as ``policy_cls``.
    """
    policy_type: str  # Type of the policy, e.g., "planner", "teleop", "learned", etc.
    force_enable_depth: bool = False
    """
    If true, require all cameras to record depth.
    In eval the cameras will be overridden, otherwise it will just require the camera system config to enable depth.
    """


class ObjectManipulationPlannerPolicyConfig(BasePolicyConfig):
    """Configuration for Franka pick planner policy."""

    policy_cls: type = None  # Will be set by importing module to avoid circular imports
    policy_factory: PolicyFactory | None = None
    policy_type: str = "planner"

    # Pick-and-place pose offsets
    pregrasp_z_offset: float = 0.04  # Height above object for pregrasp
    postgrasp_z_offset: float = 0.05  # Height above object for postgrasp
    grasp_z_offset: float = 0.03  # Lower distance from pregrasp to grasp
    place_z_offset: float = 0.07  # Lower distance from preplace to place
    end_z_offset: float = 0.05  # Height above place target for final pose

    # Speed settings
    speed_slow: float = 0.08  # m/s for precise movements
    speed_fast: float = 0.20  # m/s for transport movements
    move_settle_time: float = 0.1  # seconds

    # Gripper timing
    gripper_close_duration: float = 0.5  # Time to close gripper
    gripper_open_duration: float = 0.25  # Time to open gripper

    # Randomization parameters
    randomize_grasp: bool = False  # Enable grasp pose randomization
    grasp_xy_noise: float = 0.02  # Max XY offset from object center (meters)
    grasp_yaw_noise: float = 0.5  # Max rotation around Z-axis (radians)
    pregrasp_height_noise: float = 0.03  # Additional height variation for pregrasp
    postgrasp_height_noise: float = 0.02  # Height variation for lift phase

    # Retry behavior parameters
    max_retries: int = 3  # Maximum number of retry attempts
    gripper_empty_threshold: float = 0.002  # Gripper separation to detect empty gripper (meters)
    phase_timeout: float = 10.0  # Maximum time to spend in any phase (seconds)
    max_sequential_ik_failures: int = 8  # Maximum number of IK failures
    tcp_pos_err_threshold: float = 0.1  # Retry if position error is greater than this
    tcp_rot_err_threshold: float = np.radians(30.0)  # Retry if rotation error is greater than this

    # grasp sampling configuration (collision checking)
    filter_colliding_grasps: bool = True
    grasp_collision_batch_size: int = 128
    grasp_collision_max_grasps: int = 512
    grasp_width: float = 0.08
    grasp_length: float = 0.05
    grasp_height: float = 0.01
    grasp_base_pos: list[float] = [0.0, 0.0, -0.04]  # position of grasp base in tcp frame
    # grasp sampling configuration (cost weighting)
    grasp_pos_cost_weight: float = 1.0
    grasp_rot_cost_weight: float = 0.01
    grasp_vertical_cost_weight: float = 2.0
    grasp_com_dist_cost_weight: float = 8.0
    # grasp sampling configuration (feasibility checking)
    filter_feasible_grasps: bool = True
    grasp_feasibility_batch_size: int = 256
    grasp_feasibility_max_grasps: int = 256

    # Kinematics / return-home move groups. ``None`` preserves the legacy
    # behavior: IK may use every non-gripper move group and go-home commands
    # every group present in robot_config.init_qpos.  Mobile manipulators can
    # override these lists to keep the base fixed while manipulating objects.
    ik_unlocked_move_group_ids: list[str] | None = None
    go_home_move_group_ids: list[str] | None = None

    # which grasp libraries to use, in descending priority (will be filtered by availability for each asset)
    # if None, all available libraries for the object will be used
    grasp_libraries: list[str] | None = None

    # Debugging
    debug_poses: bool = False  # Enable debug printing for poses
    verbose: bool = True  # Enable verbose output for debugging


class OpenClosePlannerPolicyConfig(ObjectManipulationPlannerPolicyConfig):
    # For opening tasks: horizontal orientation is strongly preferred over position
    # grasp_horizontal_cost_weight is multiplied by 10x for opening tasks to strongly penalize vertical orientations
    # The cost uses squared term: (abs(z-axis z-component))^2, so vertical orientations get heavily penalized
    grasp_pos_cost_weight: float = 1.0
    grasp_rot_cost_weight: float = 0.05
    grasp_vertical_cost_weight: float = 0.0
    grasp_horizontal_cost_weight: float = (
        10.0  # Base weight, multiplied by 10x for opening tasks (effective: 20.0)
    )
    grasp_com_dist_cost_weight: float = 0.0
    pregrasp_z_offset: float = 0.04  # Height above object for postgrasp

    # Speed settings
    speed_slow: float = 0.04  # m/s for precise movements
    speed_fast: float = 0.08  # m/s for transport movements
    move_settle_time: float = 0.2  # seconds

    grasp_libraries: list[str] | None = ["droid"]  # only thor provides articulated grasps

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.solvers.object_manipulation.open_close_planner_policy import (
                OpenClosePlannerPolicy,
            )

            self.policy_cls = OpenClosePlannerPolicy
            self.policy_factory = OpenClosePlannerPolicy


class PickPlannerPolicyConfig(ObjectManipulationPlannerPolicyConfig):
    policy_cls: type = None  # Will be set in model_post_init to avoid circular imports
    postgrasp_z_offset: float = 0.08  # Height above object for postgrasp

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.solvers.object_manipulation.pick_planner_policy import (
                PickPlannerPolicy,
            )

            self.policy_cls = PickPlannerPolicy
            self.policy_factory = PickPlannerPolicy


class PickAndPlacePlannerPolicyConfig(ObjectManipulationPlannerPolicyConfig):
    policy_cls: type = None  # Will be set in model_post_init to avoid circular imports
    move_settle_time: float = 0.5

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy import (
                PickAndPlacePlannerPolicy,
            )

            self.policy_cls = PickAndPlacePlannerPolicy
            self.policy_factory = PickAndPlacePlannerPolicy


class LLMWaypointPlannerPolicyConfig(PickAndPlacePlannerPolicyConfig):
    """Configuration for OpenAI-compatible, locally validated waypoint plans."""

    llm_max_api_calls: int = 3
    api_timeout_s: float = 120.0
    llm_max_grasp_candidates: int = 12
    # RBY1 只暴露顺序 IK（约 1.3 s/次），因此不能对全部抓取做可达性检查。
    # 该上限决定最多用 IK 检查多少个候选：从最近的开始逐批扩大，
    # 直到集满 llm_max_grasp_candidates 个可达候选或达到本上限。
    # 取值过小会让候选池在 IK 阶段就耗尽（house 103 曾因此 0 候选）。
    llm_max_grasp_ik_checks: int = 96
    # 除选定的臂之外，IK 额外解锁的关节组（贯穿候选可达性检查、计划预检
    # 与执行时的 TCP 求解，三处口径必须一致）。
    # RBY1 的 torso 初始为全 0，只解锁单臂时桌面目标够不到——实测 house 103
    # 单臂残差 0.52 m / 172°，加上 torso 后同一候选立即解出。
    # 数据生成与 CuRobo 路径同样让 torso/base 参与
    # （见 object_manipulation_datagen_configs.py 的 ik_unlocked_move_group_ids
    # 与 rby1m_*_arm_holobase.yml）。本 policy 的直线插值不是避障导航器，
    # 因此只解锁 torso，不动 base。
    llm_ik_extra_groups: list[str] = []
    # 保留字段：waypoint 现在由本地几何确定性地展开（固定 6-8 段），不再由模型
    # 输出，因此这个上限不再参与校验。留着是为了让旧的 config 快照仍能反序列化。
    llm_max_waypoints: int = 32
    llm_base_xy_limit_m: float = 2.0
    llm_collision_sample_m: float = 0.03
    llm_collision_sample_deg: float = 5.0
    # 接触穿透容差：只有穿透超过它的新增接触才判失败。自接触与环境接触分开设，
    # 因为两类物理意义不同——link 之间的微量互穿是建模噪声，执行期会被 MuJoCo
    # 的接触求解器推开；撞到桌面或容器则是真干涉。
    #   自接触：RBY1 的 link_torso_2 与 link_torso_4 建模间隙只有 15 mm，而
    #   _preflight 的顺序差分 IK 会链式累积偏差。实测噪声 0.047 mm、边缘样本
    #   1.66-1.84 mm，取 2 mm 放行这类。臂/端效器真撞胸是 20-28 mm，仍被拦。
    #   环境接触：实测真干涉 4.78-12.54 mm，取 1 mm。
    # 两者设 0 都精确回到"任何新增接触都失败"的旧行为。
    llm_self_contact_tolerance_m: float = 0.002
    llm_environment_contact_tolerance_m: float = 0.001

    # 决策来源。llm = 调 API 要高层决策；geometric = 不调 API，直接用本地几何
    # 默认决策（最近的可行候选 + 复刻 baseline 的目标高度）。两档走完全相同的
    # 几何展开与 IK/碰撞预检，因此可用于控制变量的 A/B 对照；geometric 档在没有
    # LLM_BASE_URL/LLM_API_KEY/LLM_MODEL 的环境里也能跑。
    llm_decision_source: Literal["llm", "geometric"] = "llm"
    # pregrasp 沿抓取接近轴的退避距离。None = 复用 pregrasp_z_offset，与几何
    # baseline 逐位一致。模型自创的 pregrasp 曾达 0.29-0.39 m，顺序差分 IK 从
    # 那么远的种子出发会把 torso 解成自穿插构型。
    llm_pregrasp_standoff_m: float | None = None
    # approach_strategy="top_down" 时允许的接近轴倾角上限（与 world -Z 的夹角）。
    # 超限即判该候选不是顶抓，把"改用 lateral 或换候选"反馈给模型。
    llm_top_down_max_tilt_deg: float = 20.0
    # lift_height（相对 grasp 位姿的抬升量）的取值范围。下界通常由场景几何决定
    # （见 candidates 的 min_lift_height_m），这里只兜住荒谬值；上界防止抬到天花板。
    llm_lift_height_bounds: tuple[float, float] = (0.0, 0.60)
    # preplace_height（相对 place 位姿的高度）的取值范围。preplace 必须高于 place，
    # 否则水平移动会拖着物体扫过容器沿。
    llm_preplace_height_bounds: tuple[float, float] = (0.0, 0.30)
    # 由底盘目标距离推算 base 段的 duration_s（clip 到 2-8 s）。
    llm_base_speed_mps: float = 0.30
    # 推荐底盘位姿：停在目标 standoff 距离处、yaw 面向目标。作为几何先验写进
    # prompt，模型仍可自行决定是否采用。
    llm_base_standoff_target_m: float = 0.75
    # 单次底盘移动的最大平移量，应 <= llm_base_xy_limit_m。
    llm_base_step_limit_m: float = 1.0
    # geometric 档是否使用推荐底盘位姿。默认 False = 与几何 baseline 一样不动底盘，
    # 保证 geometric 档与 baseline 逐位可比；打开后得到"几何 + 移动底盘"的更强下界。
    llm_geometric_base_approach: bool = False

    # Execution failures end the episode. API replanning is reserved for the
    # pre-execution validation loop and can therefore never exceed three calls.
    max_retries: int = 0
    filter_colliding_grasps: bool = False
    filter_feasible_grasps: bool = False
    ik_unlocked_move_group_ids: list[str] | None = None
    go_home_move_group_ids: list[str] = []

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        from molmo_spaces.policy.solvers.object_manipulation.llm_waypoint_planner_policy import (
            LLMWaypointPlannerPolicy,
        )

        self.policy_cls = LLMWaypointPlannerPolicy
        self.policy_factory = LLMWaypointPlannerPolicy


class CuroboOpenClosePlannerPolicyConfig(OpenClosePlannerPolicyConfig):
    policy_cls: type = None  # Will be set in model_post_init to avoid circular imports
    left_curobo_planner_config: CuroboPlannerConfig | None = None  # will be set in model_post_init
    right_curobo_planner_config: CuroboPlannerConfig | None = None  # will be set in model_post_init
    left_planner_joint_ranges: dict[
        str, tuple
    ] = {  # Joint ranges for motion planning. Should match curobo config.
        # Move group : Joint indices in curobo config
        "base": (0, 3),
        "left_arm": (3, 10),
    }
    right_planner_joint_ranges: dict[
        str, tuple
    ] = {  # Joint ranges for motion planning. Should match curobo config.
        # Move group : Joint indices in curobo config
        "base": (0, 3),
        "right_arm": (3, 10),
    }
    enable_collision_avoidance: bool = True
    batch_size: int = 4
    max_grasping_timesteps: int = 5
    max_opening_timesteps: int = 5
    max_steps_per_waypoint: int = 10
    max_batch_plan_attempts: int = 4
    pregrasp_z_offset: float = 0.02
    max_planning_reattempts: int = 2
    gripper_closed_pos: float = 0.0
    gripper_closed_tolerance: float = 0.005
    velocity_constraints: dict[str, float] = {
        "base": 0.5,
        "head": 0.5,
        "right_arm": 0.5,
        "left_arm": 0.5,
    }
    grasp_vertical_cost_weight: float = 2.0
    attach_obj: bool = False
    max_settle_steps: int = 5
    max_height_adjustment_steps: int = 10
    server_timeout: float | None = (
        300.0  # gRPC deadline for motion planning calls (seconds), None = no deadline
    )
    server_urls: list[str] = [
        "jupiter-cs-aus-107.reviz.ai2.in:10002",
    ]


class CuroboPickAndPlacePlannerPolicyConfig(PickAndPlacePlannerPolicyConfig):
    policy_cls: type = None  # Will be set in model_post_init to avoid circular imports
    left_curobo_planner_config: CuroboPlannerConfig | None = None  # will be set in model_post_init
    right_curobo_planner_config: CuroboPlannerConfig | None = None  # will be set in model_post_init
    left_planner_joint_ranges: dict[
        str, tuple
    ] = {  # Joint ranges for motion planning. Should match curobo config.
        # Move group : Joint indices in curobo config
        "base": (0, 3),
        "left_arm": (3, 10),
    }
    right_planner_joint_ranges: dict[
        str, tuple
    ] = {  # Joint ranges for motion planning. Should match curobo config.
        # Move group : Joint indices in curobo config
        "base": (0, 3),
        "right_arm": (3, 10),
    }
    enable_collision_avoidance: bool = True
    batch_size: int = 4
    max_grasping_timesteps: int = 5
    max_opening_timesteps: int = 5
    max_steps_per_waypoint: int = 10
    max_batch_plan_attempts: int = 4
    pregrasp_z_offset: float = 0.02  # [m]
    # GRASP 阶段在 pregrasp 退距之上额外沿接近轴多进给的量 [m]。多给一点能让手指
    # 确实包住物体，但手爪中心会越过物体中心；对薄片物体，多进给会让手掌直接压到
    # 支撑面，MuJoCo 的接触力把关节顶住，waypoint 永远差一点到不了。
    grasp_approach_overshoot: float = 0.01
    max_planning_reattempts: int = 5
    gripper_closed_pos: float = 0.0  # [m]
    gripper_closed_tolerance: float = 0.005  # [m]
    velocity_constraints: dict[str, float] = {
        "base": 0.5,  # [m / policy_dt_ms]
        "head": 0.5,  # [rad / policy_dt_ms]
        "right_arm": 0.5,  # [rad / policy_dt_ms]
        "left_arm": 0.5,  # [rad / policy_dt_ms]
    }
    grasp_vertical_cost_weight: float = 0.5
    # --- grasp 批量 IK 可达性预筛（E2-IK-filter）---
    # 关闭时行为与改动前完全一致。开启后会在碰撞过滤之后、生成 pregrasp 之前，
    # 用 CuRobo 的批量 IK 筛掉运动学上不可达的候选。仅本地 planner（server_urls 为空）支持。
    enable_grasp_ik_prefilter: bool = False
    grasp_ik_prefilter_max_grasps: int = 128  # 预筛候选数上限，同时用作固定的 IK batch size
    grasp_ik_prefilter_num_seeds: int = 32  # 每个候选的 IK seed 数
    attach_obj: bool = False
    max_settle_steps: int = 5
    server_timeout: float | None = (
        300.0  # gRPC deadline for motion planning calls (seconds), None = no deadline
    )
    server_urls: list[str] = [
        "jupiter-cs-aus-107.reviz.ai2.in:10002",
    ]


def grasp_settle_steps(
    policy_dt_ms: float,
    gripper_close_duration: float,
    settle_margin_s: float = 0.15,
) -> int:
    """按控制周期换算"夹爪闭合 + 稳定"所需的最少判定步数。

    ``CuroboPlannerPolicy._grasping_something`` 靠夹爪位置偏离闭合位来判断是否夹住
    物体。若在闭合完成前就判定，仍在运动的手指同样偏离闭合位，会被误判成"已夹住"：
    E4 原先 5 步 × 66 ms = 330 ms 短于 500 ms 的闭合时长，97.4% 的判定都"通过"，
    随后 70.7% 在 LIFT 阶段掉落。按闭合时长反推步数可让不同 ``policy_dt_ms`` 的
    配置自动对齐，再加一小段稳定余量等夹爪真正停住。

    Args:
        policy_dt_ms: 策略控制周期（毫秒）。
        gripper_close_duration: 夹爪从张开到闭合所需时间（秒）。
        settle_margin_s: 闭合完成后的额外稳定时间（秒）。

    Returns:
        判定前需要等待的 policy 步数。
    """
    step_s = policy_dt_ms / 1000.0
    return math.ceil(gripper_close_duration / step_s) + math.ceil(settle_margin_s / step_s)


def panda_omron_planner_joint_ranges(
    planner_move_group_ids: list[str],
) -> dict[str, tuple[int, int]]:
    """Validate a PandaOmron planning mode and return contiguous action slices."""
    allowed = (
        ["arm"],
        ["torso", "arm"],
        ["base", "torso", "arm"],
    )
    if planner_move_group_ids not in allowed:
        raise ValueError(
            "planner_move_group_ids must be one of: ['arm'], "
            "['torso', 'arm'], or ['base', 'torso', 'arm']"
        )
    sizes = {"base": 3, "torso": 1, "arm": 7}
    ranges = {}
    offset = 0
    for move_group_id in planner_move_group_ids:
        ranges[move_group_id] = (offset, offset + sizes[move_group_id])
        offset += sizes[move_group_id]
    return ranges


class PandaOmronCuroboPickAndPlacePlannerPolicyConfig(
    CuroboPickAndPlacePlannerPolicyConfig
):
    """Single-arm CuRobo configuration for the robosuite PandaOmron robot."""

    curobo_planner_config: CuroboPlannerConfig | None = None
    planner_move_group_ids: list[str] = ["base", "torso", "arm"]
    planner_joint_ranges: dict[str, tuple[int, int]] = {
        "base": (0, 3),
        "torso": (3, 4),
        "arm": (4, 11),
    }
    arm_move_group_id: str = "arm"
    gripper_move_group_id: str = "gripper"
    attached_object_link_name: str = "attached_object"
    # The conservative hand sphere reaches 5.8 cm beyond the grip site.
    # A 2 cm approach offset puts even a valid grasp inside the target obstacle.
    pregrasp_z_offset: float = 0.10
    # 与 RBY1 不同，PandaOmron 的 pregrasp 退距已经给足 0.10 m，GRASP 阶段只需精确
    # 前进到抓取位姿。再额外进给会让手爪中心越过物体中心，在薄片物体（薄纸、纸巾）
    # 与低矮物体上直接压到支撑面，实测触发 6 个接触、关节差 0.04–0.14 rad 到不了，
    # 反复重试后 episode 失败（E4 日志 2849 次 waypoint 超时 / 449 次重试耗尽）。
    grasp_approach_overshoot: float = 0.0
    gripper_open_command: list[float] = [0.04, -0.04]
    gripper_close_command: list[float] = [0.0, 0.0]
    velocity_constraints: dict[str, float] = {
        "base": 0.5,
        "torso": 0.25,
        "arm": 0.5,
    }

    @model_validator(mode="after")
    def validate_planner_move_groups(self):
        self.planner_joint_ranges = panda_omron_planner_joint_ranges(
            self.planner_move_group_ids
        )
        return self

    def model_post_init(self, __context) -> None:
        from molmo_spaces.policy.solvers.object_manipulation.panda_omron_curobo_pick_and_place_planner_policy import (
            PandaOmronCuroboPickAndPlacePlannerPolicy,
        )

        self.policy_cls = PandaOmronCuroboPickAndPlacePlannerPolicy
        self.policy_factory = PandaOmronCuroboPickAndPlacePlannerPolicy


class PickAndPlaceNextToPlannerPolicyConfig(PickAndPlacePlannerPolicyConfig):
    policy_cls: type = None  # Will be set in model_post_init to avoid circular imports

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_next_to_planner_policy import (
            PickAndPlaceNextToPlannerPolicy,
        )

        self.policy_cls = PickAndPlaceNextToPlannerPolicy
        self.policy_factory = PickAndPlaceNextToPlannerPolicy


class PickAndPlaceColorPlannerPolicyConfig(PickAndPlacePlannerPolicyConfig):
    policy_cls: type = None  # Will be set in model_post_init to avoid circular imports

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_color_planner_policy import (
            PickAndPlaceColorPlannerPolicy,
        )

        self.policy_cls = PickAndPlaceColorPlannerPolicy
        self.policy_factory = PickAndPlaceColorPlannerPolicy


class DoorOpeningPolicyConfig(BasePolicyConfig):
    """Configuration for RBY1 door opening planner policy."""

    policy_cls: type = None  # Will be set by importing module to avoid circular imports
    policy_factory: PolicyFactory | None = None
    policy_type: str = "planner"

    # RBY1-specific policy parameters
    # Motion planning parameters
    left_curobo_planner_config: CuroboPlannerConfig | None = (
        None  # will be set in __init_policy_config
    )
    right_curobo_planner_config: CuroboPlannerConfig | None = (
        None  # will be set in __init_policy_config
    )

    left_planner_joint_ranges: dict[
        str, tuple
    ] = {  # Joint ranges for motion planning. Should match curobo config.
        # Move group : Joint indices in curobo config
        "base": (0, 3),
        "left_arm": (3, 10),
    }
    right_planner_joint_ranges: dict[
        str, tuple
    ] = {  # Joint ranges for motion planning. Should match curobo config.
        # Move group : Joint indices in curobo config
        "base": (0, 3),
        "right_arm": (3, 10),
    }
    velocity_constraints: dict[str, float] = {
        "base": 0.5,
        "head": 0.5,
        "right_arm": 0.5,
        "left_arm": 0.5,
    }
    enable_collision_avoidance: bool = True  # Whether to enable collision avoidance
    relevant_collision_objects_radius: float = (
        3.0  # Radius in meters from the door handle around which collision objects are considered
    )
    plan_in_robot_frame: bool = (
        True  # Whether to plan in robot frame or world frame (True keeps base stable)
    )
    max_planning_failures: int = 15

    # Trajectory execution parameters
    max_steps_per_waypoint: int = 10
    joint_position_tolerance: float = 0.0275

    # Gripper control parameters
    gripper_closed_pos: float = 0.0
    left_gripper_close_command: dict = {"left_gripper": 100.0}
    left_gripper_open_command: dict = {"left_gripper": -100.0}
    right_gripper_close_command: dict = {"right_gripper": 100.0}
    right_gripper_open_command: dict = {"right_gripper": -100.0}
    gripper_closed_tolerance: float = 0.005  # [m]
    max_grasping_timesteps: int = 5

    # Door opening parameters
    pre_grasp_distance: float = -0.18  # distance from door handle before grasping it
    articulation_deltas: list[float] = [
        (np.pi / 180.0) * 13.0
    ]  # delta radians to articulate door joint(s)
    first_pushing_articulation_deltas: list[float] = [
        (np.pi / 180.0) * 30.0
    ]  # special first delta articulation when pushing door

    # Recovery motion parameters
    recovery_motion_backward_distance: float = 0.02
    num_recovery_steps: int = 8

    # Debugging
    verbose: bool = False  # Enable verbose output for debugging


class NavToObjPlannerPolicyConfig(BasePolicyConfig):
    """Base configuration for navigation to object planner policies."""

    policy_cls: type = None  # Will be set by importing module to avoid circular imports
    policy_factory: PolicyFactory | None = None
    policy_type: str = "planner"

    # Recovery motion parameters
    recovery_motion_backward_distance: float = 0.02
    num_recovery_steps: int = 8

    # Debugging
    verbose: bool = True  # Enable verbose output for debugging


class AStarNavToObjPolicyConfig(NavToObjPlannerPolicyConfig):
    """Configuration for A* navigation policy (discrete grid-based planner)."""

    policy_cls: type = None

    # A* planner configuration
    planner_config: AStarPlannerConfig = AStarPlannerConfig()

    # A* planner parameters (for backward compatibility)
    map_path: str | None = None  # Path to occupancy map
    downscale: int = 5  # Downscaling factor for grid

    # Policy-related parameters
    path_interpolation_density: int = (
        1  # Num points to add between planner waypoint pairs (regardless of distance)
    )
    path_max_inter_waypoint_dist: float = 0.25  # Max distance between consecutive waypoints
    path_max_inter_waypoint_angle: float = float(
        np.deg2rad(10)
    )  # Max arc length between consecutive waypoints
    path_min_dist_to_target_center: float = (
        0.8  # Skip approaching target center below this distance
    )
    plan_max_retries: int = 3  # Allowed number of planning retries in episode

    # TODO the replanning criterion is weak, as it does not rely on actual collision,
    #  but a loose estimate based on rate decrease of spatial-angular distance to next waypoint.
    #  It needs further work to be usable, so you may want to keep a large value to prevent it for now.
    plan_fail_after_waypoint_steps: int = (
        10  # Number of steps within current waypoint to check for need to replan
    )

    plan_fail_max_dist_delta: float = 0.01  # Max difference between dists to waypoint to consider need to replan after plan_fail_after_waypoint_steps
    plan_stick_to_original_target: bool = (
        False  # Allows replanning to other possible valid targets when False
    )

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.solvers.navigation.astar_planner_policy import (
                AStarSmoothPlannerPolicy,
            )

            self.policy_cls = AStarSmoothPlannerPolicy
            self.policy_factory = AStarSmoothPlannerPolicy


class DummyPolicyConfig(BasePolicyConfig):
    """Policy config that uses DummyPolicy for testing."""

    policy_type: str = "dummy"
    policy_cls: type = None  # Set in model_post_init
    policy_factory: PolicyFactory | None = None

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.dummy_policy import DummyPolicy

            self.policy_cls = DummyPolicy
            self.policy_factory = make_lenient(DummyPolicy)


class BrownianMotionPolicyConfig(BasePolicyConfig):
    """Policy that applies Gaussian noise increments over noop control, resulting in Brownian motion."""

    policy_cls: type = None
    policy_factory: PolicyFactory | None = None
    policy_type: str = "dummy"
    std: float = 0.1

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.dummy_policy import BrownianMotionPolicy

            self.policy_cls = BrownianMotionPolicy
            self.policy_factory = make_lenient(BrownianMotionPolicy)
