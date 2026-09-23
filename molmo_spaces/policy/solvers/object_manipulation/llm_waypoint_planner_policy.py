"""OpenAI-compatible LLM waypoint planner for mobile pick-and-place.

The model proposes world-frame base/TCP waypoints.  This module deliberately
does not call CuRobo: local MuJoCo IK and contact checks are the execution gate.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from scipy.spatial.transform import Rotation

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.task_configs import PickAndPlaceTaskConfig
from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
    ActionPrimitive,
    GripperAction,
    JointMoveSegment,
    JointMoveSequence,
    NoopAction,
    TCPMoveSegment,
    TCPMoveSequence,
)
from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy import (
    PickAndPlacePlannerPolicy,
)
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.utils.grasp_sample import get_noncolliding_grasp_mask
from molmo_spaces.utils.grasps import get_pickup_grasps
from molmo_spaces.utils.linalg_utils import transform_to_twist, twist_to_transform
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb, descendant_bodies
from molmo_spaces.utils.pose import pos_quat_to_pose_mat, pose_mat_to_7d

log = logging.getLogger(__name__)

Phase = Literal[
    "base_approach",
    "pregrasp",
    "grasp",
    "lift",
    "base_transfer",
    "preplace",
    "place",
    "retreat",
]


class PlanValidationError(ValueError):
    """A model plan was parseable but unsafe or inconsistent."""


class PoseWaypoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pose: list[float] = Field(min_length=7, max_length=7)
    speed: float = Field(gt=0.0, le=0.5)

    @model_validator(mode="after")
    def finite_and_unit_quaternion(self):
        if not all(math.isfinite(value) for value in self.pose):
            raise ValueError("pose contains NaN or infinity")
        quat_norm = float(np.linalg.norm(self.pose[3:7]))
        if not 0.98 <= quat_norm <= 1.02:
            raise ValueError("quaternion must be unit length in [qw,qx,qy,qz] order")
        return self

    def matrix(self) -> np.ndarray:
        return pos_quat_to_pose_mat(self.pose[:3], self.pose[3:7])


class PlanSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    phase: Phase
    motion: Literal["base", "ee"]
    base_goal: list[float] | None = None
    waypoints: list[PoseWaypoint] | None = None
    duration_s: float | None = Field(default=None, gt=0.0, le=30.0)

    @model_validator(mode="after")
    def exactly_one_motion_payload(self):
        if self.motion == "base":
            if self.phase not in ("base_approach", "base_transfer"):
                raise ValueError("base motion is only allowed in base phases")
            if self.base_goal is None or len(self.base_goal) != 3 or self.waypoints is not None:
                raise ValueError("base segment requires only base_goal=[x,y,yaw]")
            if not all(math.isfinite(value) for value in self.base_goal):
                raise ValueError("base_goal contains NaN or infinity")
        else:
            if self.phase in ("base_approach", "base_transfer"):
                raise ValueError("base phase must use base motion")
            if not self.waypoints or self.base_goal is not None:
                raise ValueError("ee segment requires only non-empty waypoints")
        return self


class LLMWaypointPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    robot: str
    arm: str
    grasp_candidate_id: int = Field(ge=0)
    # 上限 14 = 6 个 EE 相位 + 最多 8 段底盘移动。旧上限 8 把底盘移动焊死在 2 次。
    segments: list[PlanSegment] = Field(min_length=6, max_length=14)

    @model_validator(mode="after")
    def fixed_phase_grammar(self):
        """EE 相位的相对顺序是不变式；底盘段可以插在任意相位之间。

        旧实现把底盘段也写死了槽位（base_approach 只能首位、base_transfer 只能
        夹在 lift 与 preplace 之间），于是"先转向再抓、抬起后再横移"这类多段调整
        无法表达。现在改成：只要两段底盘不相邻，就可以出现在任何位置，且允许同类
        底盘段出现多次。底盘段的命名按"当时是否已夹持物体"划分——
        base_approach 必须在 grasp 之前，base_transfer 必须在 grasp 之后、place
        之前；这与 _compute_trajectory 里 holding 标志的翻转点严格对应。
        """
        phases = [segment.phase for segment in self.segments]
        base_phases = ("base_approach", "base_transfer")
        expected = ["pregrasp", "grasp", "lift", "preplace", "place", "retreat"]
        ee_phases = [name for name in phases if name not in base_phases]
        if ee_phases != expected:
            raise ValueError(f"invalid phase order: {phases}")
        for prev, cur in zip(phases, phases[1:]):
            if prev in base_phases and cur in base_phases:
                # 相邻底盘段等价于一次移动，拆开只会重复 duration 与预检采样。
                raise ValueError(f"two base segments must not be adjacent: {phases}")
        grasp_idx = phases.index("grasp")
        place_idx = phases.index("place")
        for idx, name in enumerate(phases):
            if name == "base_approach" and idx >= grasp_idx:
                raise ValueError("base_approach must occur before grasp (not holding yet)")
            if name == "base_transfer" and not grasp_idx < idx < place_idx:
                raise ValueError("base_transfer must occur between grasp and place (holding)")
        return self

    @property
    def waypoint_count(self) -> int:
        return sum(len(segment.waypoints or []) for segment in self.segments)


ApproachStrategy = Literal["top_down", "lateral"]

# 底盘目标的两种写法：直接给 world 系 [x, y, yaw]，或引用 prompt 里
# base_reference.candidates 的 id（如 "pickup_rotate_in_place"）。
# 给出 id 写法是为了免去模型自编十五位小数的世界坐标：实测它抄推荐值时 94.5%
# 逐位一致，一旦要自己编（approach 阶段推荐值恒为 null 时）就多半过不了预检。
BaseGoalRef = list[float] | str


class PlanBuildError(RuntimeError):
    """本地几何展开失败。

    属于代码缺陷（几何公式或场景量取错），不是模型可修的问题，因此绝不能
    进入 LLM 重试循环——否则一个几何 bug 会被伪装成 N 次"模型错误"并白烧
    API 调用。
    """


# steps 的相位集合：六个 EE 相位都要列全（steps 是整条序列的替代表述）。
# retreat 出现在这里是为了保持清单完整，但它不允许挂 base_goal——它发生在放置
# 之后、已松手，底盘段的命名规则（按夹持状态划分）在那里两不靠。
StepPhase = Literal["pregrasp", "grasp", "lift", "preplace", "place", "retreat"]


class StepDecision(BaseModel):
    """模型对单个相位下的指令。steps 是"让模型决定每一步怎么做"的落点。

    旧契约一次给平铺的八个字段，本地几何按写死的顺序展开，底盘最多动两次且槽位
    固定。给 steps 之后，模型可以对每个相位单独表态：
      - 只列相位、不带 base_goal 的步骤表示"这一步按默认几何走"；
      - 带 base_goal 的相位表示"进入这一步之前，底盘先走到这个位姿"。
    未提供 steps 时完全退回旧的固定展开，因此老 artifact 与旧 prompt 仍可复现。
    """

    model_config = ConfigDict(extra="forbid")
    phase: StepPhase
    base_goal: BaseGoalRef | None = None


class LLMDecision(BaseModel):
    """LLM 的高层 last-mile 决策：只回答"操作哪里、用什么策略"，不含任何 waypoint。

    具体的 pregrasp/grasp/lift/preplace/place/retreat 位姿全部由本地确定性几何
    代码从本决策展开（见 build_plan_from_decision）。grasp pose 直接取所选候选，
    模型无法修改。

    lift_height / preplace_height 都是**相对量**：
      - lift_height 相对所选 grasp 位姿的抬升量
      - preplace_height 相对 place 位姿的高度
    低于本地算出的几何安全下限时会被校验拒绝，并把"至少需要多少"反馈回模型。
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    robot: str
    arm: str
    grasp_candidate_id: int = Field(ge=0)
    approach_strategy: ApproachStrategy = "lateral"
    lift_height: float = Field(gt=0.0, le=1.0)
    preplace_height: float = Field(gt=0.0, le=1.0)
    # world 系 [x, y, yaw] 或 base_reference.candidates 里的候选 id；None = 不移动底盘。
    base_approach_goal: BaseGoalRef | None = None
    base_transfer_goal: BaseGoalRef | None = None
    # 可选的分步决策。给了它就按它展开（允许任意位置插入底盘移动），
    # 没给就退回固定顺序的六段 + 两段底盘。
    steps: list[StepDecision] | None = None
    # 仅落盘供人工分析，不参与任何几何计算。
    rationale: str | None = Field(default=None, max_length=400)

    @model_validator(mode="after")
    def finite_and_shaped(self):
        if not math.isfinite(self.lift_height) or not math.isfinite(self.preplace_height):
            raise ValueError("lift_height and preplace_height must be finite")
        for name in ("base_approach_goal", "base_transfer_goal"):
            goal = getattr(self, name)
            if goal is None:
                continue
            if isinstance(goal, str):
                if not goal.strip():
                    raise ValueError(f"{name} candidate id must not be empty")
                continue
            if len(goal) != 3:
                raise ValueError(f"{name} must be world-frame [x, y, yaw] with exactly 3 values")
            if not all(math.isfinite(value) for value in goal):
                raise ValueError(f"{name} contains NaN or infinity")
        for step in self.steps or []:
            goal = step.base_goal
            if goal is None or isinstance(goal, str):
                continue
            if len(goal) != 3:
                raise ValueError(
                    f"steps[{step.phase}].base_goal must be [x, y, yaw] with exactly 3 values"
                )
            if not all(math.isfinite(value) for value in goal):
                raise ValueError(f"steps[{step.phase}].base_goal contains NaN or infinity")
        return self


@dataclass(frozen=True)
class SceneGeometry:
    """由当前场景 + 所选抓取候选推出的确定性几何量（世界系，米）。

    min_carry_z 是 lift 的安全下限：物体必须至少抬到容器顶面之上 place_z_offset，
    否则横移阶段会拖着物体扫过容器沿。baseline_lift_height /
    baseline_preplace_height 复刻几何 baseline（PickAndPlacePlannerPolicy）的
    目标高度，用于 llm_decision_source="geometric" 档与 prompt 里的几何先验。
    """

    grasp: np.ndarray
    place_z: float
    min_carry_z: float
    preplace_xy: np.ndarray
    baseline_lift_height: float
    baseline_preplace_height: float


def approach_tilt_deg(grasp: np.ndarray) -> float:
    """抓取接近轴与 world -Z 的夹角（度）。0 = 纯顶抓，90 = 水平侧抓。

    接近轴是抓取姿态的 z 轴，指向物体内部（指尖方向），因此顶抓时
    grasp[2, 2] ≈ -1。符号写反会让 top_down 判据整体失效。
    """
    return math.degrees(math.acos(float(np.clip(-grasp[2, 2], -1.0, 1.0))))


def wrap_deg(angle_deg: float) -> float:
    """把角度归一化到 (-180, 180]，供朝向偏差比较与反馈打印使用。"""
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def base_alignment(base_pose: np.ndarray, target_xy: np.ndarray) -> dict[str, float]:
    """底盘相对目标物的朝向与距离诊断（世界系，角度制）。

    yaw_error_deg 是本模块的核心诊断量。实测 RBY1 的 episode 起始底盘位置几乎总
    落在 standoff 半径之内（2774 次 attempt 里 100% 落在 0.51-0.70 m），但底盘
    朝向与"指向目标物方位角"仍有偏差：中位 21.1 度，26.7% 超过 30 度。旧实现只看
    距离，已经在 standoff 内就返回 None，模型因此从来看不到这个偏差，底盘也就
    从不转向——机械臂只能侧向或反向够物，扫过台面造成环境碰撞。

    yaw 约定与 recommend_base_goal 一致：yaw=0 表示底盘 +X 朝向世界 +X，
    "面向目标"即 yaw = atan2(dy, dx)。

    注意 pose_mat_to_7d 的顺序是 [x, y, z, qw, qx, qy, qz]，第 4 个分量是 qw
    不是 yaw；要拿 yaw 必须先还原成矩阵再取 atan2(R[1,0], R[0,0])。
    """
    base = np.asarray(base_pose, dtype=float)
    if base.ndim == 2:
        base_xy = base[:2, 3]
        base_yaw = math.atan2(float(base[1, 0]), float(base[0, 0]))
    else:
        flat = base.reshape(-1)
        base_xy = flat[:2]
        base_yaw = float(flat[2]) if flat.size > 2 else 0.0
    delta = np.asarray(target_xy, dtype=float).reshape(-1)[:2] - base_xy
    distance = float(np.linalg.norm(delta))
    bearing = math.atan2(float(delta[1]), float(delta[0]))
    return {
        "distance_m": distance,
        "bearing_deg": math.degrees(bearing),
        "base_yaw_deg": math.degrees(base_yaw),
        "yaw_error_deg": wrap_deg(math.degrees(bearing - base_yaw)),
    }


def base_goal_candidates(
    base_pose: np.ndarray,
    target_xy: np.ndarray,
    *,
    standoff_m: float,
    yaw_tolerance_deg: float,
    xy_tolerance_m: float,
    step_limit_m: float,
    label: str,
) -> list[dict[str, Any]]:
    """给模型用的底盘候选菜单：围绕一个目标物给出若干"对齐到可抓站姿"的方案。

    旧实现只给一个推荐值，且"已在 standoff 内就返回 None"，于是模型的全部选择
    退化成"抄推荐值或自己编世界坐标"（实测有推荐值时可抄的 580 次里 548 次逐位
    照抄，没有推荐值时 26% 自己编且多数过不了预检）。这里改成给一组带代价标注的
    候选，让模型按 translation_m / rotation_deg 自己权衡——纯转向不移动位置、
    碰撞风险最低，因此排在前面。

    返回的每项是
        {"id", "goal": [x, y, yaw], "translation_m", "rotation_deg", "note"}
    位置与朝向都已达标时返回空列表（= 底盘不必动，可以填 null）。

    底盘移动的避障仍完全靠 _preflight_kinematics_and_contacts 的离散采样接触
    检查，不是路径规划；候选只沿"当前指向目标"收缩，不做绕障。
    """
    base = np.asarray(base_pose, dtype=float)
    if base.ndim == 2:
        base_xy = base[:2, 3]
        base_yaw = math.atan2(float(base[1, 0]), float(base[0, 0]))
    else:
        flat = base.reshape(-1)
        base_xy = flat[:2]
        base_yaw = float(flat[2]) if flat.size > 2 else 0.0
    target = np.asarray(target_xy, dtype=float).reshape(-1)[:2]
    delta = target - base_xy
    distance = float(np.linalg.norm(delta))
    # 目标与底盘几乎重合时方位角无定义，此时不做任何缩放（避免除零）。
    if distance < 1e-6:
        return []
    bearing = math.atan2(float(delta[1]), float(delta[0]))
    yaw_error = abs(wrap_deg(math.degrees(bearing - base_yaw)))
    standoff_error = abs(distance - standoff_m)

    candidates: list[dict[str, Any]] = []

    def add(kind: str, position: np.ndarray, note: str) -> None:
        offset = position - base_xy
        candidates.append(
            {
                "id": f"{label}_{kind}",
                "goal": [
                    round(float(position[0]), 4),
                    round(float(position[1]), 4),
                    round(float(bearing), 4),
                ],
                "translation_m": round(float(np.linalg.norm(offset)), 4),
                "rotation_deg": round(yaw_error, 1),
                "note": note,
            }
        )

    # 1) 原地转向：只改 yaw、位置不动，碰撞风险最低，也最对症——实测回放 2774 次
    #    attempt，首选候选里 95% 是它，且平移量恒为 0；旧实现因为"距离达标即 None"
    #    让这条建议从未出现在 prompt 里。
    if yaw_error > yaw_tolerance_deg:
        add(
            "rotate_in_place",
            base_xy.copy(),
            f"keep position, rotate {yaw_error:.0f} deg to face the target",
        )
    # 2) 退/进到 standoff 并转向：位置与朝向一次修正。
    if standoff_error > xy_tolerance_m:
        direction = delta / distance
        position = target - standoff_m * direction
        # 单次平移上限：避免一次给出跨越房间的目标（预检大概率撞）。
        offset = position - base_xy
        offset_norm = float(np.linalg.norm(offset))
        if offset_norm > step_limit_m:
            position = base_xy + step_limit_m * offset / offset_norm
        add(
            "standoff_face",
            position,
            f"move to {standoff_m:.2f} m standoff and face the target",
        )
    # 3) 位置与朝向都达标时不推荐任何移动，返回空列表让模型自己决定要不要动。
    return candidates


def recommend_base_goal(
    base_xy: np.ndarray,
    target_xy: np.ndarray,
    standoff_m: float,
    step_limit_m: float,
    yaw_tolerance_deg: float = 15.0,
    xy_tolerance_m: float = 0.15,
) -> list[float] | None:
    """单一推荐底盘位姿（向后兼容入口，内部走 base_goal_candidates）。

    与旧实现的唯一行为差异：不再"已在 standoff 之内就返回 None"，而是先看朝向——
    位置达标但朝向偏了仍然推荐"原地转向"。只有位置与朝向都达标才返回 None。
    老实现正是这条"距离达标即 None"让 100% 的 episode 拿不到 approach 阶段的底盘
    推荐值，底盘因此从不转向。
    """
    candidates = base_goal_candidates(
        np.asarray(base_xy, dtype=float),
        np.asarray(target_xy, dtype=float),
        standoff_m=standoff_m,
        yaw_tolerance_deg=yaw_tolerance_deg,
        xy_tolerance_m=xy_tolerance_m,
        step_limit_m=step_limit_m,
        label="target",
    )
    return list(candidates[0]["goal"]) if candidates else None


def select_new_contacts(
    current: dict[tuple[int, int], float],
    baseline: dict[tuple[int, int], float],
) -> dict[tuple[int, int], float]:
    """挑出相对 baseline 新增或恶化的接触对，值为 signed distance（负 = 穿透）。

    原实现用集合差 `current - baseline`，语义是"出现过就永久豁免"：某一对在基线
    里已有 0.1 mm 擦碰、之后恶化成 20 mm 的真碰撞也会被放过。这里改成按距离比较；
    不在 baseline 里的接触对以 0 为基准，于是等价于原来的"新增接触"。

    是否构成失败由调用方按穿透深度与容差判定——RBY1 的 link_torso_2 与
    link_torso_4 建模间隙只有 15 mm，实测失败构型的穿透仅 0.02-0.24 mm（网格
    外壳擦碰），而真实碰撞（臂或端效器撞胸）深度 >= 5 mm，两者需要区分。
    """
    return {
        pair: depth for pair, depth in current.items() if depth < baseline.get(pair, 0.0)
    }


def _ee_segment(
    phase: Phase, position: np.ndarray, rotation: np.ndarray, speed: float
) -> PlanSegment:
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = position
    return PlanSegment(
        phase=phase,
        motion="ee",
        waypoints=[PoseWaypoint(pose=pose_mat_to_7d(pose).tolist(), speed=speed)],
    )


def _base_segment(
    phase: Phase, base_goal: list[float], reference_xy: np.ndarray, base_speed_mps: float
) -> PlanSegment:
    goal = [float(value) for value in base_goal]
    distance = float(
        np.linalg.norm(np.asarray(goal[:2]) - np.asarray(reference_xy, dtype=float)[:2])
    )
    return PlanSegment(
        phase=phase,
        motion="base",
        base_goal=goal,
        duration_s=float(np.clip(distance / max(base_speed_mps, 1e-6), 2.0, 8.0)),
    )


def build_plan_from_decision(
    *,
    decision: LLMDecision,
    geometry: SceneGeometry,
    current_base_xy: np.ndarray,
    pregrasp_standoff: float,
    end_z_offset: float,
    base_speed_mps: float,
    speed_fast: float,
    speed_slow: float,
) -> LLMWaypointPlan:
    """把高层决策确定性地展开成 LLMWaypointPlan（内部中间表示）。

    六段 EE 的姿态恒等于所选抓取候选的姿态（几何 baseline 的不变式，否则搬运
    过程中会转动物体）；只有位置随决策变化：

        pregrasp = G.pos - standoff * G.approach_axis   # 沿接近轴反向退避
        grasp    = G.pos                                # 直接取候选，不可修改
        lift     = [G.x, G.y, G.z + lift_height]
        preplace = [preplace.x, preplace.y, place_z + preplace_height]
        place    = [preplace.x, preplace.y, place_z]
        retreat  = place - end_z_offset * G.approach_axis

    approach_strategy 不改变这里的几何：pregrasp 一律沿候选自身的接近轴退避。
    它的作用是在 _validate_decision 里校验候选姿态与所声称策略是否一致。

    段序有两种来源：给了 decision.steps 就按模型的分步表态逐相位展开（底盘移动
    可以挂在任意多个相位之前），没给就退回上面写死的六段 + 两段底盘。两条路径
    产出的 EE 位姿完全相同，差别只在底盘段的数量与位置。
    """
    grasp = np.asarray(geometry.grasp, dtype=float)
    rotation = grasp[:3, :3]
    grasp_pos = grasp[:3, 3]
    approach_axis = rotation[:, 2]

    positions = {
        "pregrasp": grasp_pos - pregrasp_standoff * approach_axis,
        "grasp": grasp_pos,
        "lift": np.array(
            [grasp_pos[0], grasp_pos[1], grasp_pos[2] + decision.lift_height], dtype=float
        ),
        # preplace 的 XY 与 place 相同：容器中心 + (grasp - 物体原点) 的偏移，
        # 保证被持物体落在容器中央而不是手腕位置。
        "preplace": np.array(
            [
                geometry.preplace_xy[0],
                geometry.preplace_xy[1],
                geometry.place_z + decision.preplace_height,
            ],
            dtype=float,
        ),
        "place": np.array(
            [geometry.preplace_xy[0], geometry.preplace_xy[1], geometry.place_z], dtype=float
        ),
    }
    # 注意 approach_axis 是姿态的第三列（旋转矩阵），不是位置分量。
    positions["retreat"] = positions["place"] - end_z_offset * approach_axis

    speeds = {
        "pregrasp": speed_fast,
        "grasp": speed_slow,
        "lift": speed_slow,
        "preplace": speed_fast,
        "place": speed_slow,
        "retreat": speed_fast,
    }

    ee_phases: tuple[Phase, ...] = (
        "pregrasp",
        "grasp",
        "lift",
        "preplace",
        "place",
        "retreat",
    )
    # 底盘段的命名按"进入该相位时是否已夹持物体"划分，与 PlanSegment 的语法校验
    # 以及 _compute_trajectory 里 holding 标志的翻转点严格对应：
    #   grasp 及之前 = 尚未夹持 → base_approach
    #   grasp 之后、place 之前 = 已夹持 → base_transfer
    grasp_index = ee_phases.index("grasp")

    def base_phase_before(phase: Phase) -> Phase:
        return "base_approach" if ee_phases.index(phase) <= grasp_index else "base_transfer"

    segments: list[PlanSegment] = []
    if decision.steps is not None:
        # 分步展开：模型逐相位表态，底盘移动可以挂在任意一个（或多个）相位之前。
        # positions / rotation / speeds 仍全部来自本地几何——模型依旧给不出位姿，
        # 它决定的只是"哪一步之前先把底盘摆到哪"。
        for step in decision.steps:
            if step.base_goal is not None:
                segments.append(
                    _base_segment(
                        base_phase_before(step.phase),
                        step.base_goal,
                        current_base_xy,
                        base_speed_mps,
                    )
                )
            segments.append(
                _ee_segment(step.phase, positions[step.phase], rotation, speeds[step.phase])
            )
    else:
        # 固定展开（旧行为，未给 steps 时逐位保留）。
        if decision.base_approach_goal is not None:
            segments.append(
                _base_segment(
                    "base_approach", decision.base_approach_goal, current_base_xy, base_speed_mps
                )
            )
        for phase in ("pregrasp", "grasp", "lift"):
            segments.append(_ee_segment(phase, positions[phase], rotation, speeds[phase]))
        if decision.base_transfer_goal is not None:
            segments.append(
                _base_segment(
                    "base_transfer", decision.base_transfer_goal, current_base_xy, base_speed_mps
                )
            )
        for phase in ("preplace", "place", "retreat"):
            segments.append(_ee_segment(phase, positions[phase], rotation, speeds[phase]))

    plan = LLMWaypointPlan(
        robot=decision.robot,
        arm=decision.arm,
        grasp_candidate_id=decision.grasp_candidate_id,
        segments=segments,
    )

    # "LLM 不得修改 grasp pose" 的硬约束：展开后的 grasp 必须复刻所选候选。
    # 按位姿语义比较而不是矩阵逐元素相等——候选的旋转矩阵来自抓取库，并不是严格
    # 正交的（实测非正交度约 3e-4），而 pos_quat_to_pose_mat 会做正交化投影，
    # 逐元素相等会把这点投影误差误判成错误。1e-6 仍远小于任何真正的候选错配。
    grasp_segment = next(segment for segment in plan.segments if segment.phase == "grasp")
    generated = grasp_segment.waypoints[0].matrix()
    position_error = float(np.linalg.norm(generated[:3, 3] - grasp[:3, 3]))
    rotation_error = float(
        Rotation.from_matrix(grasp[:3, :3].T @ generated[:3, :3]).magnitude()
    )
    if position_error > 1e-6 or rotation_error > 1e-6:
        raise PlanBuildError(
            "generated grasp waypoint does not reproduce the selected candidate "
            f"(position error {position_error:.3e} m, rotation error {rotation_error:.3e} rad)"
        )
    return plan


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse plain JSON or one fenced JSON block without accepting trailing prose."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) < 3 or not lines[-1].strip().startswith("```"):
            raise PlanValidationError("unterminated JSON code fence")
        stripped = "\n".join(lines[1:-1]).strip()
        if stripped.lower().startswith("json\n"):
            stripped = stripped[5:].strip()
    try:
        result = json.loads(
            stripped,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {value}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise PlanValidationError(f"invalid JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise PlanValidationError("top-level response must be a JSON object")
    return result


@dataclass(frozen=True)
class LLMResponse:
    content: str
    latency_s: float
    usage: dict[str, Any]
    request_id: str | None


class OpenAICompatibleClient:
    """Minimal dependency-free client; authorization data never enters exceptions."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float = 120.0):
        if not base_url or not api_key or not model:
            raise ValueError("LLM_BASE_URL, LLM_API_KEY and LLM_MODEL must all be set")
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    @classmethod
    def from_env(cls, timeout_s: float = 120.0) -> OpenAICompatibleClient:
        missing = [
            name for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL") if not os.getenv(name)
        ]
        if missing:
            raise ValueError(f"Missing required LLM environment variables: {', '.join(missing)}")
        return cls(
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ["LLM_API_KEY"],
            model=os.environ["LLM_MODEL"],
            timeout_s=timeout_s,
        )

    def complete(self, system_prompt: str, payload: dict[str, Any]) -> LLMResponse:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            },
            allow_nan=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        start = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                response_body = response.read()
                request_id = response.headers.get("x-request-id")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"LLM API HTTP error {exc.code}") from None
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"LLM API transport error: {type(exc).__name__}") from None
        latency = time.monotonic() - start
        try:
            decoded = json.loads(response_body)
            content = decoded["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Malformed LLM API response: {type(exc).__name__}") from None
        return LLMResponse(
            content=content,
            latency_s=latency,
            usage=decoded.get("usage") or {},
            request_id=request_id,
        )


class FileBackedMockClient:
    """Offline client used by simulator integration tests and prompt debugging.

    注意 mock 每次都返回同一份内容，因此校验反馈循环在 mock 下永远无法自愈：
    它只能当 smoke test，不能验证纠错能力。
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise ValueError(f"LLM mock response file not found: {self.path}")
        # 在构造时就把格式错误暴露出来，而不是等走进重试循环后才失败。
        try:
            parsed = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        # LLMDecision 没有 segments 字段，因此出现它就是旧的 waypoint plan。
        if isinstance(parsed, dict) and "segments" in parsed:
            raise ValueError(
                f"LLM mock response {self.path} is a legacy waypoint plan (it has a `segments` "
                "array). The policy now consumes an LLMDecision and expects keys robot/arm/"
                "grasp_candidate_id/approach_strategy/lift_height/preplace_height."
            )

    def complete(self, system_prompt: str, payload: dict[str, Any]) -> LLMResponse:
        content = self.path.read_text(encoding="utf-8")
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                content = json.dumps(parsed)
        except json.JSONDecodeError:
            pass
        return LLMResponse(content=content, latency_s=0.0, usage={}, request_id="offline-mock")


def create_planner_client(timeout_s: float = 120.0):
    mock_path = os.getenv("LLM_MOCK_RESPONSE_FILE")
    if mock_path:
        return FileBackedMockClient(mock_path)
    return OpenAICompatibleClient.from_env(timeout_s)


def validate_llm_environment(timeout_s: float = 120.0) -> None:
    """Fail before simulator startup if neither mock nor live credentials work."""
    create_planner_client(timeout_s)


SYSTEM_PROMPT = """You are a robot last-mile decision maker. Return JSON only, with no prose.
You do NOT output poses or waypoints. A deterministic geometric module expands your
decision into the concrete end-effector and base trajectory, and a local MuJoCo IK and
collision preflight validates it before execution. You choose WHERE to operate and
WHICH strategy to use; the geometry chooses HOW to move there.

Return exactly these keys: schema_version, robot, arm, grasp_candidate_id,
approach_strategy, lift_height, preplace_height, and optionally base_approach_goal,
base_transfer_goal, rationale.
- grasp_candidate_id: index of one supplied candidate. Its pose becomes the grasp pose
  verbatim and cannot be modified by you. Choose an arm listed in that candidate's
  reachable_arms.
- approach_strategy: "top_down" or "lateral". The generated geometry is identical for
  both: the pre-grasp always retreats along the candidate's own approach axis. The
  strategy is a declaration that is checked against the candidate. "top_down" requires
  the candidate's approach axis to be within top_down_max_tilt_deg of straight down, so
  declaring it for a side grasp is rejected. "lateral" is always safe.
- lift_height: meters to raise the end effector ABOVE the grasp pose.
- preplace_height: meters the pre-place pose sits ABOVE the place pose.
  Both are checked against geometry_reference; a value below the geometric minimum is
  rejected with the minimum that is required.
- base_approach_goal / base_transfer_goal: world-frame [x, y, yaw] with exactly 3 values
  (yaw in radians, NOT the 7-value pose format), or null to keep the base fixed. A base
  move happens before pregrasp (base_approach) or right after lift (base_transfer).
  Moving the base does NOT move the end-effector targets: they stay world-anchored to
  the object and the receptacle, while the arm rides along with the base. Moving the
  base closer to a target is often what makes an otherwise unreachable grasp solvable.
  geometry_reference.recommended_base_approach_goal is a conservative, valid choice.

Keep the torso and unused arm fixed. Prefer high-clearance transport. Obey the supplied
schema, bounds and phase semantics. You cannot edit the trajectory: when
previous_validation_error is present, adjust the high-level knobs named in
validation_feedback.llm_adjustable (base goals, grab candidate, arm, approach_strategy,
lift_height, preplace_height). Do not repeat a decision listed in
validation_error_history, and build on previous_decision rather than starting over."""


class LLMWaypointPlannerPolicy(PickAndPlacePlannerPolicy):
    """LLM 做高层 last-mile 决策，传统几何模块负责具体运动。

    流水线：LLM high-level decision -> deterministic geometric waypoint generation
    -> MuJoCo IK / collision validation -> execution。

    模型只回答"抓哪个候选、用哪条臂、什么接近策略、底盘停哪、抬多高"；六个
    EE 目标与可选的底盘目标全部由 build_plan_from_decision 确定性展开，grasp
    pose 直接取所选候选。模型因此永远无法绕过本地校验器去改底层轨迹点：
    校验失败时它只能调整高层旋钮。

    已知的、本设计不解决的缺陷：
      1. mujoco_kinematics.ik 是纯阻尼最小二乘，无碰撞、无零空间正则，解完全由
         种子决定：_preflight 拿上一步的解作下一步种子，偏差会链式累积。实测
         RBY1 的 torso 内链（link_torso_2 / link_torso_4）建模间隙只有 15 mm，
         累积偏差会让 link 之间出现亚毫米级的网格外壳互穿——这不是物理碰撞，
         已由 llm_self_contact_tolerance_m 放行。真正的干涉（臂或端效器撞胸
         20-28 mm、手指撞桌或容器 4.8-12.5 mm）仍会被拦下，那来自直线插值路径
         没有避障，需要路径规划而不是放宽判据。
      2. _assert_no_new_robot_contact 只检查机器人参与的接触对，被搬运物体与
         桌/容器的新接触不会被拦下。
      3. reachable_arms 是在当前底盘位姿下标注的，模型移动底盘后该标注可能过期。
    """

    def __init__(self, config: MlSpacesExpConfig, task: BaseMujocoTask) -> None:
        super().__init__(config, task)
        # BaseObjectManipulationPlannerPolicy.reset() warms up the robot's
        # batched/parallel IK solver before calling _compute_trajectory().
        # RBY1 intentionally exposes only the sequential MuJoCo IK solver, and
        # this policy validates and executes every waypoint with that solver.
        # Mark the unrelated parallel solver as already warmed up so RBY1 can
        # reach candidate construction and the LLM request.
        self.ik_warmed_up = True
        # geometric 档不调 API，因此不能实例化客户端——否则在没有
        # LLM_BASE_URL/LLM_API_KEY/LLM_MODEL 的环境里，A/B 对照根本跑不起来。
        self.client = (
            None
            if self.policy_config.llm_decision_source == "geometric"
            else create_planner_client(self.policy_config.api_timeout_s)
        )
        self._api_calls = 0
        self._selected_gripper_id: str | None = None
        self._selected_arm_id: str | None = None
        self._artifact_id = f"{os.getpid()}_{time.time_ns()}"
        # 最近一次预检被容差放行的擦碰，成功与失败路径都要能取到。
        self._ignored_graze_count = 0
        self._max_ignored_penetration = 0.0

    def get_all_phases(self) -> dict[str, int]:
        # 基类没有两个 base phase，缺了会让 PolicyPhaseSensor 落到 -1 并刷 warning。
        return {**super().get_all_phases(), "base_approach": 10, "base_transfer": 11}

    def _base_is_movable(self) -> bool:
        """底盘是否是可写 (x, y, theta) 的三自由度全向底盘。

        只有这种布局才能把 base_goal=[x,y,yaw] 直接写进 qpos[:3]。free joint +
        轮子的底盘里 qpos[:3] 是 (x, y, z)，写 yaw 会落到 z 槽上。
        """
        base = self.robot_view.get_move_group("base")
        return len(np.asarray(base.joint_pos).reshape(-1)) == 3

    def _extra_ik_groups(self) -> list[str]:
        """配置中额外解锁、且当前机器人确实拥有的关节组（如 RBY1 的 torso）。"""
        available = set(self.robot_view.move_group_ids())
        return [
            group for group in self.policy_config.llm_ik_extra_groups if group in available
        ]

    def _get_ik_unlocked_move_group_ids(self) -> list[str]:
        if self._selected_arm_id is not None:
            return [self._selected_arm_id] + [
                group for group in self._extra_ik_groups() if group != self._selected_arm_id
            ]
        return super()._get_ik_unlocked_move_group_ids()

    def _available_arms(self) -> dict[str, str]:
        grippers = self.robot_view.get_gripper_movegroup_ids()
        if "left_gripper" in grippers and "right_gripper" in grippers:
            return {"left_arm": "left_gripper", "right_arm": "right_gripper"}
        if "gripper" in grippers and "arm" in self.robot_view.move_group_ids():
            return {"arm": "gripper"}
        raise ValueError(f"Unsupported robot gripper layout: {grippers}")

    def _candidate_grasps(
        self, pickup_obj: MlSpacesObject
    ) -> tuple[list[np.ndarray], list[list[str]]]:
        poses = get_pickup_grasps(
            self.task.env, pickup_obj, grasp_libraries=self.policy_config.grasp_libraries
        )
        arm_map = self._available_arms()

        # 碰撞过滤：排除抓取姿态本身就与环境（台面、容器、目标物）相交的候选。
        # 复用 CuRobo 路径同一套 grasp_collision_* 辅助体——它们由
        # add_auxiliary_objects 在 filter_colliding_grasps=True 时注入场景。
        if self.policy_config.filter_colliding_grasps:
            noncolliding_mask = get_noncolliding_grasp_mask(
                self.task.env.current_model,
                self.task.env.current_data,
                poses,
                self.policy_config.grasp_collision_batch_size,
            )
            if noncolliding_mask.any():
                log.info(
                    "Collision filter kept %d of %d grasp candidates",
                    int(noncolliding_mask.sum()),
                    len(poses),
                )
                poses = poses[noncolliding_mask]
            else:
                # 全部被判碰撞时保留原池：让后续 IK 与接触预检继续把关，
                # 而不是因为过滤本身把候选池清空。
                log.warning(
                    "Collision filter rejected all %d grasps; keeping unfiltered pool",
                    len(poses),
                )

        # Build a balanced nearest-pose pool instead of allowing one arm to
        # dominate the shared candidate list. Then use the same sequential
        # MuJoCo IK used by validation/execution to label arm compatibility.
        # 顺序长度取 IK 检查上限而非最终候选数：house 103 idx 1 曾因只检查
        # 12 个最近的候选且全部 IK 失败，得到空候选池而从未走到 LLM 请求。
        rankings = {
            arm: np.argsort(
                np.linalg.norm(
                    poses[:, :3, 3]
                    - self.robot_view.get_move_group(gripper).leaf_frame_to_world[:3, 3],
                    axis=1,
                ),
                kind="stable",
            )
            for arm, gripper in arm_map.items()
        }
        check_budget = min(
            len(poses),
            max(
                self.policy_config.llm_max_grasp_ik_checks,
                self.policy_config.llm_max_grasp_candidates,
            ),
        )
        ordered_ids: list[int] = []
        seen_ids: set[int] = set()
        rank = 0
        while len(ordered_ids) < check_budget:
            added = False
            for ids in rankings.values():
                if rank >= len(ids):
                    continue
                idx = int(ids[rank])
                if idx not in seen_ids:
                    seen_ids.add(idx)
                    ordered_ids.append(idx)
                    added = True
                    if len(ordered_ids) >= check_budget:
                        break
            if not added and all(rank >= len(ids) for ids in rankings.values()):
                break
            rank += 1

        robot = self.task.env.current_robot
        qpos = self.robot_view.get_qpos_dict()
        base_pose = self.robot_view.base.pose
        feasible_poses: list[np.ndarray] = []
        reachable_arms: list[list[str]] = []
        checked = 0
        for idx in ordered_ids:
            if len(feasible_poses) >= self.policy_config.llm_max_grasp_candidates:
                break
            checked += 1
            pose = poses[idx].copy()
            arms = []
            for arm, gripper in arm_map.items():
                unlocked = [arm] + [
                    group for group in self._extra_ik_groups() if group != arm
                ]
                solution = robot.kinematics.ik(gripper, pose, unlocked, qpos, base_pose)
                if solution is not None:
                    arms.append(arm)
                elif checked == 1:
                    # 只对首个候选记录失败残差，避免刷屏：残差量级可区分
                    # “目标在当前解锁口径下不可达”与“迭代收敛不佳”。
                    reached = self.robot_view.get_move_group(gripper).leaf_frame_to_world
                    log.info(
                        "IK miss [%s] unlock=%s candidate %d: pos_err=%.3fm rot_err=%.1fdeg",
                        arm,
                        unlocked,
                        idx,
                        float(np.linalg.norm(reached[:3, 3] - pose[:3, 3])),
                        math.degrees(
                            float(
                                Rotation.from_matrix(
                                    reached[:3, :3].T @ pose[:3, :3]
                                ).magnitude()
                            )
                        ),
                    )
            if arms:
                feasible_poses.append(pose)
                reachable_arms.append(arms)

        log.info(
            "Found %d arm-compatible grasp candidates after checking %d of %d "
            "ordered candidates (%d available after filtering)",
            len(feasible_poses),
            checked,
            len(ordered_ids),
            len(poses),
        )
        return feasible_poses, reachable_arms

    @staticmethod
    def _jsonable_qpos(qpos: dict[str, np.ndarray]) -> dict[str, list[float]]:
        return {name: np.asarray(value, dtype=float).tolist() for name, value in qpos.items()}

    def _scene_payload(
        self,
        pickup_obj: MlSpacesObject,
        receptacle: MlSpacesObject,
        candidates: list[np.ndarray],
        candidate_arms: list[list[str]],
        previous_error: str | None,
        error_history: list[str],
        previous_decision: LLMDecision | None = None,
        feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        model, data = self.task.env.current_model, self.task.env.current_data
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        anchors = np.stack(
            [self.robot_view.base.pose[:3, 3], pickup_obj.position, receptacle.position]
        )
        obstacles = []
        for obj in om.list_top_level_objects():
            if obj.name in (pickup_obj.name, receptacle.name):
                continue
            try:
                center, size = body_aabb(model, data, obj.object_id)
            except (ValueError, RuntimeError):
                continue
            distance = float(np.min(np.linalg.norm(anchors - center, axis=1)))
            obstacles.append(
                (distance, {"name": obj.name, "center": center.tolist(), "size": size.tolist()})
            )
        obstacles.sort(key=lambda item: item[0])
        arm_map = self._available_arms()
        max_tilt = self.policy_config.llm_top_down_max_tilt_deg
        # 每个候选都带上确定性几何量。模型必须能看出"这个候选是不是顶抓""选它时
        # lift 至少要抬多少"，否则它只能猜，校验反馈也就无从收敛。
        geometries = [
            self._geometry_for_candidate(idx, pickup_obj, receptacle, candidates)
            for idx in range(len(candidates))
        ]

        def candidate_payload(idx: int, pose: np.ndarray) -> dict[str, Any]:
            tilt = approach_tilt_deg(np.asarray(pose, dtype=float))
            return {
                "id": idx,
                "pose": pose_mat_to_7d(pose).tolist(),
                "reachable_arms": candidate_arms[idx],
                "approach_tilt_deg": round(tilt, 1),
                "top_down_ok": tilt <= max_tilt,
                "min_lift_height_m": round(
                    geometries[idx].min_carry_z - float(np.asarray(pose, dtype=float)[2, 3]), 4
                ),
            }

        lo_lift, hi_lift = self.policy_config.llm_lift_height_bounds
        lo_pre, hi_pre = self.policy_config.llm_preplace_height_bounds
        base_xy = self.robot_view.base.pose[:3, 3][:2]
        # 与 _resolve_base_goal_refs 共用同一份候选菜单，保证模型引用的 id 一定能解析。
        # 底盘不可动时整体置空：推荐值与候选一起变 null，模型不会去引用一个本机器人
        # 根本执行不了的位姿。
        menu = self._base_menu(pickup_obj, receptacle)
        if not self._base_is_movable():
            menu = {"pickup": [], "receptacle": []}
        return {
            "output_contract": {
                "json_schema": LLMDecision.model_json_schema(),
                "required_top_level_keys": [
                    "schema_version",
                    "robot",
                    "arm",
                    "grasp_candidate_id",
                    "approach_strategy",
                    "lift_height",
                    "preplace_height",
                ],
                "exact_shape_example": {
                    "schema_version": 1,
                    "robot": self.robot_view.name,
                    "arm": candidate_arms[0][0],
                    "grasp_candidate_id": 0,
                    "approach_strategy": "lateral",
                    "lift_height": round(geometries[0].baseline_lift_height, 3),
                    "preplace_height": round(geometries[0].baseline_preplace_height, 3),
                    "base_approach_goal": None,
                    "base_transfer_goal": None,
                    "rationale": "one short sentence",
                },
                "base_goal_format": (
                    "either a world-frame [x_m,y_m,yaw_rad] triple, or one of the ids under "
                    "base_reference.candidates (e.g. \"pickup_rotate_in_place\"); null keeps "
                    "the base fixed for that step"
                ),
                "forbidden": [
                    "markdown or prose",
                    "unknown fields",
                    "any pose, waypoint or quaternion value",
                    "torso commands",
                ],
            },
            "waypoint_generation_policy": {
                "summary": "You choose where to operate and which strategy to use; deterministic geometry expands that into the motion. You never output poses.",
                "pregrasp": "grasp pose retreated along the candidate's own approach axis by pregrasp_standoff_m",
                "grasp": "the selected candidate pose, used verbatim and not adjustable",
                "lift": "directly above grasp at grasp_z + lift_height, keeping the grasp orientation",
                "preplace": "above the receptacle at place_z + preplace_height, keeping the grasp orientation",
                "place": "object bottom resting on the receptacle top",
                "retreat": "place pose retreated along the approach axis",
                "orientation": "all six end-effector targets keep the selected grasp orientation; the grasped object is never rotated in transit",
                "base_approach": "base segment inserted before pregrasp or grasp; null omits it",
                "base_transfer": "base segment inserted between grasp and place (while holding); null omits it",
                "base_and_ee": "moving the base does NOT move the end-effector targets, which stay world-anchored to the object and receptacle",
                "base_decisions": (
                    "you decide whether and where the base moves, and before which step; "
                    "base_reference.alignment shows your current facing error and "
                    "base_reference.candidates lists concrete goals with their translation and "
                    "rotation cost. The most common useful move is a pure rotation into the "
                    "candidate's standoff facing - it does not translate and barely risks collision"
                ),
                "steps": (
                    "optional. To decide step by step, list all six ee phases in order "
                    "(pregrasp, grasp, lift, preplace, place, retreat) and attach base_goal to "
                    "any of them to move the base before that step, e.g. "
                    "[{\"phase\":\"pregrasp\",\"base_goal\":\"pickup_rotate_in_place\"},"
                    "{\"phase\":\"grasp\"},{\"phase\":\"lift\"},"
                    "{\"phase\":\"preplace\",\"base_goal\":\"receptacle_rotate_in_place\"},"
                    "{\"phase\":\"place\"},{\"phase\":\"retreat\"}]. "
                    "When steps is present, leave base_approach_goal and base_transfer_goal null"
                ),
            },
            "constraints": {
                "approach_strategy_rule": "top_down and lateral generate identical geometry; top_down additionally requires the candidate approach axis to be within top_down_max_tilt_deg of straight down, so it is rejected for a side grasp",
                "top_down_max_tilt_deg": max_tilt,
                "lift_height_bounds_m": [lo_lift, hi_lift],
                "preplace_height_bounds_m": [lo_pre, hi_pre],
                "lift_height_is_relative_to": "the grasp pose of the selected candidate",
                "preplace_height_is_relative_to": "the place pose",
                "pregrasp_standoff_m": self._pregrasp_standoff(),
                "base_bounds_relative_to_start_m": self.policy_config.llm_base_xy_limit_m,
                "torso_fixed": True,
                "unused_arm_fixed": True,
                "clearance_rule": "lift must clear the receptacle top before transport; a lift_height below the candidate's min_lift_height_m is rejected",
            },
            "base_reference": {
                "base_xy": np.asarray(base_xy, dtype=float).round(4).tolist(),
                "base_yaw_deg": round(
                    math.degrees(
                        math.atan2(
                            float(self.robot_view.base.pose[1, 0]),
                            float(self.robot_view.base.pose[0, 0]),
                        )
                    ),
                    2,
                ),
                "base_is_movable": self._base_is_movable(),
                # 当前朝向偏差。这是本轮新增的核心诊断量：实测底盘朝向与"指向目标物
                # 方位角"偏差中位 21.1 度、26.7% 超过 30 度，而旧 prompt 完全没暴露
                # 这个量——模型只能看见一个恒为 null 的推荐值。
                "alignment": {
                    "pickup": {
                        key: round(value, 3)
                        for key, value in base_alignment(
                            self.robot_view.base.pose, pickup_obj.position[:2]
                        ).items()
                    },
                    "receptacle": {
                        key: round(value, 3)
                        for key, value in base_alignment(
                            self.robot_view.base.pose, receptacle.position[:2]
                        ).items()
                    },
                    "meaning": (
                        "yaw_error_deg = how far the base is from facing the target "
                        "(wrapped to (-180, 180]); a large value means the arm has to reach "
                        "sideways or backwards, which is what drives the counter collisions"
                    ),
                },
                # 候选菜单：位置与朝向都已对好时该组为空列表（底盘不必动）。
                # 每项带 translation_m / rotation_deg，让模型按代价自行取舍——
                # rotate_in_place 不移动位置、只转向，碰撞风险最低。
                "candidates": menu,
                "candidate_usage": (
                    "base_approach_goal / base_transfer_goal / steps[].base_goal accept either a "
                    "world-frame [x, y, yaw] triple or one of these candidate ids"
                ),
                # 向后兼容的单一推荐值，现由候选菜单派生（旧实现传 2 维 base_xy 会让
                # "当前 yaw"被当成 0，转向判断随之失效）。
                "recommended_base_approach_goal": (
                    list(menu["pickup"][0]["goal"]) if menu["pickup"] else None
                ),
                "recommended_base_transfer_goal": (
                    list(menu["receptacle"][0]["goal"]) if menu["receptacle"] else None
                ),
                "note": (
                    "recommendations are conservative standoff poses facing the target; "
                    "any collision during the base move is caught by the local preflight "
                    "and reported back to you"
                ),
            },
            "robot": self.robot_view.name,
            "base_pose": pose_mat_to_7d(self.robot_view.base.pose).tolist(),
            "joint_positions": self._jsonable_qpos(self.robot_view.get_qpos_dict()),
            "end_effectors": {
                arm: pose_mat_to_7d(
                    self.robot_view.get_move_group(gripper).leaf_frame_to_world
                ).tolist()
                for arm, gripper in arm_map.items()
            },
            "pickup": self._object_payload(pickup_obj),
            "receptacle": self._object_payload(receptacle),
            "grasp_candidates": [
                candidate_payload(idx, pose) for idx, pose in enumerate(candidates)
            ],
            "nearby_obstacles": [value for _, value in obstacles[:64]],
            "previous_decision": (
                previous_decision.model_dump(mode="json")
                if previous_decision is not None
                else None
            ),
            "previous_validation_error": previous_error,
            "validation_error_history": list(error_history),
            "validation_feedback": feedback,
        }

    def _object_payload(self, obj: MlSpacesObject) -> dict[str, Any]:
        center, size = body_aabb(
            self.task.env.current_model, self.task.env.current_data, obj.object_id
        )
        return {
            "name": obj.name,
            "pose": pose_mat_to_7d(obj.pose).tolist(),
            "aabb_center": center.tolist(),
            "aabb_size": size.tolist(),
        }

    def _write_artifact(self, attempt: int, record: dict[str, Any]) -> None:
        artifact_dir = Path(self.config.output_dir) / "llm_plans"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / f"plan_{self._artifact_id}_attempt_{attempt}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False))

    def _pregrasp_standoff(self) -> float:
        """pregrasp 沿接近轴的退避距离。None 表示复用几何 baseline 的 pregrasp_z_offset。"""
        standoff = self.policy_config.llm_pregrasp_standoff_m
        return self.policy_config.pregrasp_z_offset if standoff is None else standoff

    def _base_menu(
        self, pickup_obj: MlSpacesObject, receptacle: MlSpacesObject
    ) -> dict[str, list[dict[str, Any]]]:
        """底盘候选菜单（pickup / receptacle 两组），围绕抓取站姿与放置站姿。

        _scene_payload 把它写进 base_reference.candidates，_resolve_base_goal_refs
        用同一份把候选 id 翻回 world 位姿——两处必须一致，所以只在这里构造一次。
        """
        config = self.policy_config
        kwargs = {
            "standoff_m": config.llm_base_standoff_target_m,
            "yaw_tolerance_deg": config.llm_base_yaw_tolerance_deg,
            "xy_tolerance_m": config.llm_base_xy_tolerance_m,
            "step_limit_m": config.llm_base_step_limit_m,
        }
        return {
            "pickup": base_goal_candidates(
                self.robot_view.base.pose, pickup_obj.position[:2], label="pickup", **kwargs
            ),
            "receptacle": base_goal_candidates(
                self.robot_view.base.pose, receptacle.position[:2], label="receptacle", **kwargs
            ),
        }

    def _resolve_base_goal_ref(
        self,
        goal: BaseGoalRef | None,
        menu: dict[str, list[dict[str, Any]]],
        field: str,
    ) -> list[float] | None:
        """把候选 id 或 [x, y, yaw] 统一翻成 world 位姿。

        候选 id 必须出现在本次 prompt 给出的菜单里，否则拒绝——这样模型的"引用"
        永远指向它真正看过的值，而不是随手编的自由字符串。
        """
        if goal is None:
            return None
        if isinstance(goal, str):
            for candidates in menu.values():
                for candidate in candidates:
                    if candidate["id"] == goal:
                        return [float(value) for value in candidate["goal"]]
            known = [candidate["id"] for candidates in menu.values() for candidate in candidates]
            raise PlanValidationError(
                f"{field} refers to unknown base candidate {goal!r}; use one of {known} "
                "or a world-frame [x, y, yaw] triple"
            )
        return [float(value) for value in goal]

    def _resolve_base_goal_refs(
        self, decision: LLMDecision, menu: dict[str, list[dict[str, Any]]]
    ) -> LLMDecision:
        """返回把候选 id 解析成 world 位姿后的决策副本，原对象不动。

        原始 decision 仍原样落盘（record["decision"]），解析结果另存
        record["resolved_decision"]——分析"模型到底抄没抄推荐值"要靠原始那份。
        """
        resolved = decision.model_copy(deep=True)
        resolved.base_approach_goal = self._resolve_base_goal_ref(
            decision.base_approach_goal, menu, "base_approach_goal"
        )
        resolved.base_transfer_goal = self._resolve_base_goal_ref(
            decision.base_transfer_goal, menu, "base_transfer_goal"
        )
        if decision.steps is not None and resolved.steps is not None:
            for source, target in zip(decision.steps, resolved.steps):
                target.base_goal = self._resolve_base_goal_ref(
                    source.base_goal, menu, f"steps[{source.phase}].base_goal"
                )
        return resolved

    def _geometry_for_candidate(
        self,
        candidate_id: int,
        pickup_obj: MlSpacesObject,
        receptacle: MlSpacesObject,
        candidates: list[np.ndarray],
    ) -> SceneGeometry:
        """取容器/目标物的 AABB，算出该候选下的确定性几何量（公式与几何 baseline 一致）。"""
        model, data = self.task.env.current_model, self.task.env.current_data
        receptacle_center, receptacle_size = body_aabb(model, data, receptacle.object_id)
        pickup_center, pickup_size = body_aabb(model, data, pickup_obj.object_id)

        grasp = np.asarray(candidates[candidate_id], dtype=float)
        grasp_z = float(grasp[2, 3])
        pickup_bottom_z = float(pickup_center[2] - pickup_size[2] / 2)
        pickup_z = float(pickup_obj.position[2])
        receptacle_top_z = float(receptacle_center[2] + receptacle_size[2] / 2)

        # pickup_obj_clearance_offset：抓点在物体底部之上多少，物体底部就能离 EE 多远。
        place_z = receptacle_top_z + max(grasp_z - pickup_bottom_z, 0.0)
        min_carry_z = place_z + self.policy_config.place_z_offset
        # 容器 XY 中心 + (抓取点 - 物体原点)：让被持物体落在容器中央，而不是手腕位置。
        preplace_xy = (
            np.asarray(receptacle.position, dtype=float)[:2]
            + grasp[:2, 3]
            - np.asarray(pickup_obj.position, dtype=float)[:2]
        )
        return SceneGeometry(
            grasp=grasp,
            place_z=place_z,
            min_carry_z=min_carry_z,
            preplace_xy=preplace_xy,
            baseline_lift_height=min_carry_z - grasp_z,
            # baseline 的 preplace 高度是 place_z + (place_z_offset + grasp_z - pickup_z)：
            # 其中 grasp_z - pickup_z 是 "_get_placement_poses 里 [:3,3] += grasp - pickup_obj.position"
            # 顺带加上的 z 分量（表现为从 lift 到 preplace 还会再上升几厘米）。这里刻意
            # 复刻该 artifact，好让 geometric 档与 baseline 逐位可比。
            baseline_preplace_height=self.policy_config.place_z_offset + grasp_z - pickup_z,
        )

    def _default_decision(
        self,
        pickup_obj: MlSpacesObject,
        receptacle: MlSpacesObject,
        candidates: list[np.ndarray],
        candidate_arms: list[list[str]],
    ) -> LLMDecision:
        """纯几何默认决策：最近的可行候选 + 复刻 baseline 的目标高度。

        这是 llm_decision_source="geometric" 档的决策来源，也是 prompt 里给模型的
        几何先验。底盘默认不动（与几何 baseline 一致）；只有显式打开
        llm_geometric_base_approach 时才采用推荐底盘位姿。
        """
        geometry = self._geometry_for_candidate(0, pickup_obj, receptacle, candidates)
        lo_lift, hi_lift = self.policy_config.llm_lift_height_bounds
        lo_pre, hi_pre = self.policy_config.llm_preplace_height_bounds

        base_goal = None
        if self.policy_config.llm_geometric_base_approach and self._base_is_movable():
            # 传完整 4x4 位姿而不是 [:2] 的 x,y：只给平面坐标会让"当前 yaw"退化成 0，
            # 朝向偏差算错，转向建议随之失效。
            base_goal = recommend_base_goal(
                self.robot_view.base.pose,
                pickup_obj.position[:2],
                self.policy_config.llm_base_standoff_target_m,
                self.policy_config.llm_base_step_limit_m,
                yaw_tolerance_deg=self.policy_config.llm_base_yaw_tolerance_deg,
                xy_tolerance_m=self.policy_config.llm_base_xy_tolerance_m,
            )

        tilt = approach_tilt_deg(geometry.grasp)
        return LLMDecision(
            robot=self.robot_view.name,
            arm=candidate_arms[0][0],
            grasp_candidate_id=0,
            approach_strategy=(
                "top_down" if tilt <= self.policy_config.llm_top_down_max_tilt_deg else "lateral"
            ),
            # 容器的绝对高度可能让复刻出来的抬升量为负（物体本来就悬在容器顶之上），
            # 此时用 bounds 兜底，保证默认决策永远能通过校验。
            lift_height=float(np.clip(geometry.baseline_lift_height, lo_lift, hi_lift)),
            preplace_height=float(np.clip(geometry.baseline_preplace_height, lo_pre, hi_pre)),
            base_approach_goal=base_goal,
            rationale="deterministic geometric default",
        )

    # 失败原因 -> 模型真正能拧的高层旋钮。模型改不了轨迹，所以每类失败都必须映射到
    # 一组可执行的替代决策，否则重试只是把同一个答案再写一遍。
    _ADJUSTABLE_BY_REASON = {
        "collision": [
            "base_approach_goal",
            "base_transfer_goal",
            "steps",
            "grasp_candidate_id",
            "approach_strategy",
        ],
        "ik": [
            "base_approach_goal",
            "base_transfer_goal",
            "steps",
            "grasp_candidate_id",
            "arm",
            "approach_strategy",
        ],
        "base_radius": ["base_approach_goal", "base_transfer_goal"],
        # 引用了菜单里不存在的候选 id。
        "base_goal": ["base_approach_goal", "base_transfer_goal"],
        "tilt": ["approach_strategy", "grasp_candidate_id"],
        "height": ["lift_height", "preplace_height"],
        "decision": [
            "grasp_candidate_id",
            "arm",
            "approach_strategy",
            "lift_height",
            "preplace_height",
        ],
        "format": [],
        "api": [],
    }

    def _feedback_from_error(
        self,
        exc: BaseException,
        menu: dict[str, list[dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        """把一次校验失败翻译成模型可执行的调整建议。

        错误串里的 "IK failed" / "collision" 等子串必须保留：
        scripts/evaluation/summarize_llm_waypoint_eval.py 按子串给失败分桶，
        改了措辞会让漏斗统计把失败全丢进 other_validation，跨版本对比失真。
        """
        message = str(exc)
        if isinstance(exc, ValidationError) or "invalid JSON" in message:
            reason = "format"
        elif isinstance(exc, RuntimeError):
            reason = "api"
        elif "collision" in message:
            reason = "collision"
        elif "IK failed" in message:
            reason = "ik"
        elif "base candidate" in message:
            reason = "base_goal"
        elif "base goal" in message:
            reason = "base_radius"
        elif "approach_tilt" in message:
            reason = "tilt"
        elif "lift_height" in message or "preplace_height" in message:
            reason = "height"
        else:
            reason = "decision"
        phase = next(
            (
                name
                for name in (
                    "base_approach",
                    "base_transfer",
                    "pregrasp",
                    "grasp",
                    "lift",
                    "preplace",
                    "place",
                    "retreat",
                )
                if f"{name}:" in message
            ),
            None,
        )
        feedback: dict[str, Any] = {
            "reason": reason,
            "llm_adjustable": self._ADJUSTABLE_BY_REASON[reason],
            "message": message,
        }
        if phase is not None:
            feedback["failed_phase"] = phase
        # 碰撞 / IK / 底盘半径这三类失败，模型最需要的是"换成哪个具体底盘位姿"。
        # 只告诉它 base_goal 可调（旧行为）等于让它继续自己编世界坐标——实测那样
        # 编出来的值多半过不了预检。这里把失败相位对应的候选菜单直接附上。
        if menu is not None and reason in ("collision", "ik", "base_radius", "base_goal"):
            group = "pickup" if phase in ("base_approach", "pregrasp", "grasp", None) else "receptacle"
            other = "receptacle" if group == "pickup" else "pickup"
            feedback["base_candidates"] = {
                group: menu.get(group, []),
                "hint": (
                    f"the failure is on phase {phase!r}; these goals are pre-checked standoff "
                    f"poses for the {group}. Prefer a candidate whose translation_m is small "
                    "(pure rotations barely risk collision). "
                    f"The {other} group is also still available."
                ),
            }
        return feedback

    def _request_valid_plan(
        self,
        pickup_obj: MlSpacesObject,
        receptacle: MlSpacesObject,
        candidates: list[np.ndarray],
        candidate_arms: list[list[str]],
    ) -> LLMWaypointPlan:
        """拿到一个通过本地校验的 waypoint 计划。

        llm 档向模型要高层决策并按 llm_max_api_calls 重试；geometric 档直接用本地
        确定性默认决策，只跑一次、不消耗 API。两档共用完全相同的展开与预检链路，
        因此 artifact 可以直接横向比较。
        """
        source = self.policy_config.llm_decision_source
        max_attempts = 1 if source == "geometric" else self.policy_config.llm_max_api_calls
        previous_error: str | None = None
        error_history: list[str] = []
        previous_decision: LLMDecision | None = None
        feedback: dict[str, Any] | None = None

        for attempt in range(1, max_attempts + 1):
            # 每次都从"尚未选臂"开始：上一次失败尝试留下的 arm 会污染
            # _get_ik_unlocked_move_group_ids()，让本次预检解锁错的关节组。
            self._selected_arm_id = None
            self._selected_gripper_id = None
            # 与选臂一起重置：否则上一次尝试的擦碰统计会被当成这一次的。
            self._ignored_graze_count = 0
            self._max_ignored_penetration = 0.0
            record: dict[str, Any] = {
                "plan_id": self._artifact_id,
                "attempt": attempt,
                "pickup_object": pickup_obj.name,
                "receptacle": receptacle.name,
                "plan_source": source,
            }
            try:
                if source == "geometric":
                    decision = self._default_decision(
                        pickup_obj, receptacle, candidates, candidate_arms
                    )
                    record.update(
                        raw_response=None,
                        latency_s=0.0,
                        usage={},
                        request_id="offline-geometric",
                    )
                else:
                    self._api_calls += 1
                    payload = self._scene_payload(
                        pickup_obj,
                        receptacle,
                        candidates,
                        candidate_arms,
                        previous_error,
                        error_history,
                        previous_decision,
                        feedback,
                    )
                    record["input"] = payload
                    response = self.client.complete(SYSTEM_PROMPT, payload)
                    record.update(
                        raw_response=response.content,
                        latency_s=response.latency_s,
                        usage=response.usage,
                        request_id=response.request_id,
                    )
                    decision = LLMDecision.model_validate(extract_json_object(response.content))

                previous_decision = decision
                record["decision"] = decision.model_dump(mode="json")
                # 候选 id -> world 位姿。原始 decision 已经落盘，这里另存解析结果，
                # 因为分析"模型到底抄没抄推荐值"必须看原始那份。
                decision = self._resolve_base_goal_refs(decision, self._base_menu(pickup_obj, receptacle))
                record["resolved_decision"] = decision.model_dump(mode="json")
                geometry = self._validate_decision(
                    decision, pickup_obj, receptacle, candidates, candidate_arms
                )
                # 选定臂必须在展开与预检之前生效：预检的 IK 解锁口径依赖它。
                self._selected_arm_id = decision.arm
                self._selected_gripper_id = self._available_arms()[decision.arm]
                plan = build_plan_from_decision(
                    decision=decision,
                    geometry=geometry,
                    current_base_xy=self.robot_view.base.pose[:3, 3][:2],
                    pregrasp_standoff=self._pregrasp_standoff(),
                    end_z_offset=self.policy_config.end_z_offset,
                    base_speed_mps=self.policy_config.llm_base_speed_mps,
                    speed_fast=self.policy_config.speed_fast,
                    speed_slow=self.policy_config.speed_slow,
                )
                # 被容差放行的擦碰也落盘：调 llm_self_contact_tolerance_m 时这是
                # 唯一可比的数字，也是判断有没有误放过真碰撞的依据。
                record["contact_stats"] = (
                    self._preflight_kinematics_and_contacts(plan, pickup_obj) or {}
                )
                record["parsed_plan"] = plan.model_dump(mode="json")
                record["valid"] = True
                self._write_artifact(attempt, record)
                log.info("%s decision accepted on attempt %d", source, attempt)
                return plan
            except PlanBuildError:
                # 本地几何展开崩了：代码缺陷，重试只会白烧 API 调用。
                raise
            except (RuntimeError, ValidationError, PlanValidationError, ValueError) as exc:
                previous_error = f"{type(exc).__name__}: {exc}"
                error_history.append(previous_error)
                feedback = self._feedback_from_error(
                    exc, self._base_menu(pickup_obj, receptacle)
                )
                record.update(
                    valid=False,
                    error=previous_error,
                    validation_feedback=feedback,
                    contact_stats=self._contact_stats(),
                )
                self._write_artifact(attempt, record)
                log.warning("%s decision attempt %d rejected: %s", source, attempt, previous_error)
        raise ValueError(f"LLM planning failed after {max_attempts} calls: {previous_error}")

    def _validate_decision(
        self,
        decision: LLMDecision,
        pickup_obj: MlSpacesObject,
        receptacle: MlSpacesObject,
        candidates: list[np.ndarray],
        candidate_arms: list[list[str]],
    ) -> SceneGeometry:
        """校验高层决策自洽、安全、可行，并返回展开所需的几何量。

        grasp pose 直接取所选候选，因此这里不再需要"grasp waypoint 与候选一致"
        这类检查——模型根本无从提出 grasp 位姿。
        """
        arm_map = self._available_arms()
        if decision.robot not in (self.robot_view.name, self.config.robot_config.name):
            raise PlanValidationError(f"robot mismatch: {decision.robot}")
        if decision.arm not in arm_map:
            raise PlanValidationError(
                f"invalid arm {decision.arm}; expected one of {list(arm_map)}"
            )
        if decision.grasp_candidate_id >= len(candidates):
            raise PlanValidationError(
                f"grasp_candidate_id {decision.grasp_candidate_id} is out of range "
                f"(0..{len(candidates) - 1})"
            )
        if decision.arm not in candidate_arms[decision.grasp_candidate_id]:
            raise PlanValidationError(
                f"grasp candidate {decision.grasp_candidate_id} is not reachable by "
                f"{decision.arm}; use one of {candidate_arms[decision.grasp_candidate_id]}"
            )

        geometry = self._geometry_for_candidate(
            decision.grasp_candidate_id, pickup_obj, receptacle, candidates
        )

        tilt = approach_tilt_deg(geometry.grasp)
        max_tilt = self.policy_config.llm_top_down_max_tilt_deg
        if decision.approach_strategy == "top_down" and tilt > max_tilt:
            raise PlanValidationError(
                f"candidate {decision.grasp_candidate_id} has approach_tilt={tilt:.1f}deg > "
                f"llm_top_down_max_tilt_deg={max_tilt:.1f}deg, so it is not a top-down grasp; "
                "use approach_strategy=lateral or a different grasp_candidate_id"
            )

        lo_lift, hi_lift = self.policy_config.llm_lift_height_bounds
        if not lo_lift <= decision.lift_height <= hi_lift:
            raise PlanValidationError(
                f"lift_height {decision.lift_height:.3f} is outside llm_lift_height_bounds "
                f"[{lo_lift:.3f}, {hi_lift:.3f}] m"
            )
        min_lift = geometry.min_carry_z - float(geometry.grasp[2, 3])
        if decision.lift_height < min_lift:
            raise PlanValidationError(
                f"lift_height {decision.lift_height:.3f} m is too low to clear the receptacle; "
                f"need >= {min_lift:.3f} m for candidate {decision.grasp_candidate_id} "
                f"(min_carry_z={geometry.min_carry_z:.3f} m)"
            )

        lo_pre, hi_pre = self.policy_config.llm_preplace_height_bounds
        if not lo_pre <= decision.preplace_height <= hi_pre:
            raise PlanValidationError(
                f"preplace_height {decision.preplace_height:.3f} is outside "
                f"llm_preplace_height_bounds [{lo_pre:.3f}, {hi_pre:.3f}] m"
            )

        # steps 与扁平底盘字段是同一件事的两种写法，同时给会让"以哪个为准"变得
        # 含糊；直接拒绝并说清怎么改，比静默取其一可预期。
        ee_phases = ("pregrasp", "grasp", "lift", "preplace", "place", "retreat")
        if decision.steps is not None:
            step_phases = [step.phase for step in decision.steps]
            repeated = sorted({name for name in step_phases if step_phases.count(name) > 1})
            if repeated:
                raise PlanValidationError(f"steps repeats phases {repeated}; list each exactly once")
            missing = [name for name in ee_phases if name not in step_phases]
            if missing:
                raise PlanValidationError(
                    f"steps is missing phases {missing}; list all six in order: {list(ee_phases)}"
                )
            if decision.base_approach_goal is not None or decision.base_transfer_goal is not None:
                raise PlanValidationError(
                    "base_approach_goal/base_transfer_goal and steps[].base_goal are two ways to "
                    "say the same thing; set the base moves inside steps and leave the flat "
                    "fields null, or drop steps entirely"
                )
            base_moves = sum(1 for step in decision.steps if step.base_goal is not None)
            if base_moves > self.policy_config.llm_base_max_moves:
                raise PlanValidationError(
                    f"steps requests {base_moves} base moves, more than "
                    f"llm_base_max_moves={self.policy_config.llm_base_max_moves}; "
                    "each move costs at least 2 s of the episode budget"
                )
            if any(step.phase == "retreat" and step.base_goal is not None for step in decision.steps):
                raise PlanValidationError(
                    "retreat cannot carry a base_goal: it happens after the object is "
                    "released, so the move is neither a base_approach nor a base_transfer; "
                    "attach the base_goal to preplace or place instead"
                )

        base_goals = [
            goal
            for goal in (decision.base_approach_goal, decision.base_transfer_goal)
            if goal is not None
        ]
        base_goals.extend(
            step.base_goal for step in (decision.steps or []) if step.base_goal is not None
        )
        if base_goals and not self._base_is_movable():
            raise PlanValidationError(
                "this robot's base is not a 3-DoF (x, y, yaw) holonomic base; "
                "base_approach_goal and base_transfer_goal must be null"
            )
        start_xy = np.asarray(self.robot_view.get_move_group("base").joint_pos[:2], dtype=float)
        for goal in base_goals:
            offset = float(np.linalg.norm(np.asarray(goal[:2], dtype=float) - start_xy))
            if offset > self.policy_config.llm_base_xy_limit_m:
                raise PlanValidationError(
                    f"base goal {[round(float(value), 3) for value in goal[:2]]} is {offset:.2f} m "
                    f"from the episode start base pose, beyond the allowed radius "
                    f"llm_base_xy_limit_m={self.policy_config.llm_base_xy_limit_m:.2f} m"
                )
        return geometry

    def _contact_pairs(self) -> dict[tuple[int, int], float]:
        """当前 MuJoCo 接触对及其 signed distance（负值表示穿透，越负越深）。

        同一对 geom 可能被多点接触记录多次，取最深的那条。
        """
        data = self.task.env.current_data
        pairs: dict[tuple[int, int], float] = {}
        for idx in range(data.ncon):
            contact = data.contact[idx]
            geoms = sorted((int(contact.geom1), int(contact.geom2)))
            key = (geoms[0], geoms[1])
            depth = float(contact.dist)
            if key not in pairs or depth < pairs[key]:
                pairs[key] = depth
        return pairs

    def _assert_no_new_robot_contact(
        self,
        baseline: dict[tuple[int, int], float],
        phase: str,
        pickup_root: int,
    ) -> tuple[int, float]:
        """检查当前状态里是否出现了会判失败的机器人接触。

        返回 (被容差放行的擦碰次数, 其中最深的穿透深度 m)，供 artifact 记录：
        调容差时这是唯一可比的数字，也是判断有没有误放过真碰撞的依据。
        """
        robot_root = self.robot_view.root_body_id
        model = self.task.env.current_model
        gripper_root = self.robot_view.get_move_group(
            self._selected_gripper_id
        ).root_body_id
        gripper_body_ids = descendant_bodies(model, gripper_root)
        holding_phases = {"grasp", "lift", "base_transfer", "preplace", "place"}
        ignored_count = 0
        ignored_depth = 0.0
        for geom_pair, depth in select_new_contacts(self._contact_pairs(), baseline).items():
            body_ids = tuple(int(model.geom_bodyid[geom_id]) for geom_id in geom_pair)
            roots = tuple(
                int(model.body_rootid[body_id]) for body_id in body_ids
            )
            if robot_root not in roots:
                continue
            if phase in holding_phases and pickup_root in roots:
                # Contact between the selected gripper fingers and the target
                # is required after grasp closure and persists while lifting
                # and transporting. Do not exempt target contact with the arm,
                # base, torso, or unused gripper.
                robot_body_id = next(
                    body_id
                    for body_id, root_id in zip(body_ids, roots)
                    if root_id == robot_root
                )
                if robot_body_id in gripper_body_ids:
                    continue
            # 到这里是本次新增（或相对基线恶化）、且不豁免的机器人接触。
            # 自接触与环境接触用不同容差：link 之间的微量互穿是建模噪声，执行期
            # 会被 MuJoCo 的接触求解器推开；撞到桌面或容器则是真干涉。
            is_self_contact = len(set(roots)) == 1
            tolerance = (
                self.policy_config.llm_self_contact_tolerance_m
                if is_self_contact
                else self.policy_config.llm_environment_contact_tolerance_m
            )
            if -depth <= tolerance:
                ignored_count += 1
                ignored_depth = max(ignored_depth, -depth)
                continue
            names = tuple(model.geom(geom_id).name or f"geom_{geom_id}" for geom_id in geom_pair)
            # 同时报出 body 名：机器人 MJCF 里的 geom 多为无名，只有 geom id
            # 无法判断撞到的是夹爪、小臂还是 torso。
            bodies = tuple(
                model.body(int(model.geom_bodyid[geom_id])).name or f"body_{int(model.geom_bodyid[geom_id])}"
                for geom_id in geom_pair
            )
            # 保留 "collision" 子串：summarize_llm_waypoint_eval.py 按子串分桶，
            # 改了措辞会让失败全落进 other_validation。
            contact_kind = "self-contact" if is_self_contact else "robot-environment contact"
            raise PlanValidationError(
                f"{phase}: new MuJoCo collision between geoms {names} (bodies {bodies}); "
                f"{contact_kind}, penetration {abs(depth) * 1000:.2f} mm "
                f"(tolerance {tolerance * 1000:.2f} mm)"
            )
        return ignored_count, ignored_depth

    def _preflight_kinematics_and_contacts(
        self, plan: LLMWaypointPlan, pickup_obj: MlSpacesObject
    ) -> dict[str, Any]:
        """逐采样点做 IK 与接触预检，返回被容差放行的擦碰统计供 artifact 记录。"""
        model, data = self.task.env.current_model, self.task.env.current_data
        snapshot = {name: value.copy() for name, value in self.robot_view.get_qpos_dict().items()}
        baseline = self._contact_pairs()
        ignored_graze_count = 0
        max_ignored_penetration = 0.0
        qstate = {name: value.copy() for name, value in snapshot.items()}
        current_pose = self.robot_view.get_move_group(
            self._selected_gripper_id
        ).leaf_frame_to_world.copy()
        pickup_root = pickup_obj.object_root_id
        try:
            for segment in plan.segments:
                if segment.motion == "base":
                    q0 = np.asarray(qstate["base"], dtype=float)
                    q1 = q0.copy()
                    q1[:3] = segment.base_goal
                    distance = np.linalg.norm(q1[:2] - q0[:2])
                    yaw_distance = abs(float(q1[2] - q0[2]))
                    samples = max(
                        2, math.ceil(distance / 0.03), math.ceil(yaw_distance / math.radians(5))
                    )
                    old_base_pose = self.robot_view.base.pose.copy()
                    for alpha in np.linspace(0.0, 1.0, samples + 1)[1:]:
                        qstate["base"] = q0 + (q1 - q0) * alpha
                        self.robot_view.set_qpos_dict(qstate)
                        mujoco.mj_forward(model, data)
                        graze_count, graze_depth = self._assert_no_new_robot_contact(
                            baseline, segment.phase, pickup_root
                        )
                        ignored_graze_count += graze_count
                        max_ignored_penetration = max(max_ignored_penetration, graze_depth)
                    new_base_pose = self.robot_view.base.pose.copy()
                    current_pose = new_base_pose @ np.linalg.inv(old_base_pose) @ current_pose
                    continue

                for waypoint_idx, waypoint in enumerate(segment.waypoints):
                    goal = waypoint.matrix()
                    lin, ang = transform_to_twist(np.linalg.inv(current_pose) @ goal)
                    samples = max(
                        1,
                        math.ceil(
                            float(np.linalg.norm(lin)) / self.policy_config.llm_collision_sample_m
                        ),
                        math.ceil(
                            float(np.linalg.norm(ang))
                            / math.radians(self.policy_config.llm_collision_sample_deg)
                        ),
                    )
                    for sample_idx, alpha in enumerate(
                        np.linspace(0.0, 1.0, samples + 1)[1:], start=1
                    ):
                        target = current_pose @ twist_to_transform(lin * alpha, ang * alpha)
                        solution = self.task.env.current_robot.kinematics.ik(
                            self._selected_gripper_id,
                            target,
                            self._get_ik_unlocked_move_group_ids(),
                            qstate,
                            self.robot_view.base.pose,
                        )
                        if solution is None:
                            target_7d = pose_mat_to_7d(target).round(6).tolist()
                            raise PlanValidationError(
                                f"{segment.phase}: local IK failed for arm "
                                f"{self._selected_arm_id}, waypoint {waypoint_idx}, "
                                f"interpolation sample {sample_idx}/{samples}, "
                                f"target={target_7d}"
                            )
                        qstate = {
                            name: np.asarray(value).copy() for name, value in solution.items()
                        }
                        self.robot_view.set_qpos_dict(qstate)
                        mujoco.mj_forward(model, data)
                        graze_count, graze_depth = self._assert_no_new_robot_contact(
                            baseline, segment.phase, pickup_root
                        )
                        ignored_graze_count += graze_count
                        max_ignored_penetration = max(max_ignored_penetration, graze_depth)
                    current_pose = goal
        finally:
            self.robot_view.set_qpos_dict(snapshot)
            mujoco.mj_forward(model, data)
            # 写实例变量而非只靠返回值：预检抛异常时也要能拿到已放行的擦碰，
            # 否则调容差时只看得见成功那一次的数字。
            self._ignored_graze_count = ignored_graze_count
            self._max_ignored_penetration = max_ignored_penetration
        return self._contact_stats()

    def _contact_stats(self) -> dict[str, Any]:
        """最近一次预检里被容差放行的擦碰统计。"""
        return {
            "ignored_graze_count": self._ignored_graze_count,
            "max_ignored_penetration_mm": round(self._max_ignored_penetration * 1000, 3),
        }

    def _tcp_sequence(
        self,
        segment: PlanSegment,
        start_pose: np.ndarray,
        holding: bool,
    ) -> tuple[TCPMoveSequence, np.ndarray]:
        moves = []
        previous = start_pose
        for waypoint in segment.waypoints:
            goal = waypoint.matrix()
            moves.append(
                TCPMoveSegment(
                    name=segment.phase,
                    start_pose=previous,
                    end_pose=goal,
                    speed=waypoint.speed,
                )
            )
            previous = goal
        return (
            TCPMoveSequence(
                self.robot_view,
                self._selected_tcp_to_jp,
                self.policy_config.move_settle_time,
                moves,
                is_holding_object=holding,
                gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
                tcp_pos_err_threshold=self.policy_config.tcp_pos_err_threshold,
                tcp_rot_err_threshold=self.policy_config.tcp_rot_err_threshold,
                gripper_move_group_id=self._selected_gripper_id,
            ),
            previous,
        )

    def _selected_tcp_to_jp(self, _mg_id: str, target_pose: np.ndarray) -> dict[str, Any]:
        return self._tcp_to_jp_fn(self._selected_gripper_id, target_pose)

    def _compute_trajectory(self) -> list[ActionPrimitive]:
        task_config = self.config.task_config
        if not isinstance(task_config, PickAndPlaceTaskConfig):
            raise ValueError("LLMWaypointPlannerPolicy only supports pick-and-place tasks")
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        pickup_obj = om.get_object_by_name(task_config.pickup_obj_name)
        receptacle = om.get_object_by_name(task_config.place_receptacle_name)
        candidates, candidate_arms = self._candidate_grasps(pickup_obj)
        if not candidates:
            raise ValueError("No locally arm-compatible grasp candidates available")
        plan = self._request_valid_plan(pickup_obj, receptacle, candidates, candidate_arms)

        actions: list[ActionPrimitive] = [
            GripperAction(
                self.robot_view, True, 0.0, gripper_move_group_id=self._selected_gripper_id
            )
        ]
        current_pose = self.robot_view.get_move_group(
            self._selected_gripper_id
        ).leaf_frame_to_world.copy()
        planned_base_pose = self.robot_view.base.pose.copy()
        holding = False
        for segment in plan.segments:
            if segment.motion == "base":
                current = self.robot_view.get_move_group("base").joint_pos.copy()
                goal = current.copy()
                goal[:3] = segment.base_goal
                actions.append(
                    JointMoveSequence(
                        self.robot_view,
                        self.policy_config.move_settle_time,
                        [
                            JointMoveSegment(
                                segment.phase, None, {"base": goal}, segment.duration_s or 3.0
                            )
                        ],
                        is_holding_object=holding,
                        gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
                        gripper_move_group_id=self._selected_gripper_id,
                    )
                )
                next_base_pose = planned_base_pose.copy()
                next_base_pose[:3, :3] = Rotation.from_euler("z", segment.base_goal[2]).as_matrix()
                next_base_pose[0, 3] = segment.base_goal[0]
                next_base_pose[1, 3] = segment.base_goal[1]
                current_pose = next_base_pose @ np.linalg.inv(planned_base_pose) @ current_pose
                planned_base_pose = next_base_pose
                continue
            sequence, current_pose = self._tcp_sequence(segment, current_pose, holding)
            actions.append(sequence)
            if segment.phase == "grasp":
                actions.append(
                    GripperAction(
                        self.robot_view,
                        False,
                        self.policy_config.gripper_close_duration,
                        gripper_move_group_id=self._selected_gripper_id,
                    )
                )
                holding = True
            elif segment.phase == "place":
                actions.append(
                    GripperAction(
                        self.robot_view,
                        True,
                        self.policy_config.gripper_open_duration,
                        gripper_move_group_id=self._selected_gripper_id,
                    )
                )
                holding = False
        actions.append(NoopAction(self.robot_view, 2.0))
        return actions
