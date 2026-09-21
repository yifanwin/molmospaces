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
    segments: list[PlanSegment] = Field(min_length=6, max_length=8)

    @model_validator(mode="after")
    def fixed_phase_grammar(self):
        phases = [segment.phase for segment in self.segments]
        expected = ["pregrasp", "grasp", "lift", "preplace", "place", "retreat"]
        if phases and phases[0] == "base_approach":
            phases = phases[1:]
        if "base_transfer" in phases:
            idx = phases.index("base_transfer")
            if idx != 3:
                raise ValueError("base_transfer must occur between lift and preplace")
            phases = phases[:idx] + phases[idx + 1 :]
        if phases != expected:
            raise ValueError(f"invalid phase order: {phases}")
        return self

    @property
    def waypoint_count(self) -> int:
        return sum(len(segment.waypoints or []) for segment in self.segments)


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
    """Offline client used by simulator integration tests and prompt debugging."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise ValueError(f"LLM mock response file not found: {self.path}")

    def complete(self, system_prompt: str, payload: dict[str, Any]) -> LLMResponse:
        content = self.path.read_text(encoding="utf-8")
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and set(parsed) >= {"robot", "arm", "segments"}:
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


SYSTEM_PROMPT = """You are a robot waypoint planner. Return JSON only, with no prose.
The top-level JSON must contain exactly: schema_version, robot, arm,
grasp_candidate_id, and segments. `segments` must be one ordered JSON array that
contains every required phase; `base_segment` and `ee_segment` are not valid
top-level keys. All poses are world-frame [x,y,z,qw,qx,qy,qz], quaternion scalar
first. Choose exactly one supplied grasp_candidate_id and an arm listed in that
candidate's reachable_arms. Keep the torso and unused arm fixed. Base and
end-effector motion must be separate segments. Every segment requires `motion`.
Base segments use motion="base" and base_goal=[x,y,yaw] (3 values, yaw in radians),
NOT the 7-value base_pose format, and must omit waypoints. EE segments use
motion="ee" and non-empty waypoints and must omit base_goal. Preserve the selected grasp
quaternion through pregrasp/grasp/lift and normally through transport/place;
the receptacle quaternion is not an end-effector target. Obey the supplied
schema, phase grammar, bounds, and collision boxes. Prefer short,
high-clearance paths. Correct every item in validation_error_history; do not
regress fixes from earlier attempts."""


class LLMWaypointPlannerPolicy(PickAndPlacePlannerPolicy):
    """Generate a complete plan, validate it locally, then execute primitives."""

    def __init__(self, config: MlSpacesExpConfig, task: BaseMujocoTask) -> None:
        super().__init__(config, task)
        # BaseObjectManipulationPlannerPolicy.reset() warms up the robot's
        # batched/parallel IK solver before calling _compute_trajectory().
        # RBY1 intentionally exposes only the sequential MuJoCo IK solver, and
        # this policy validates and executes every waypoint with that solver.
        # Mark the unrelated parallel solver as already warmed up so RBY1 can
        # reach candidate construction and the LLM request.
        self.ik_warmed_up = True
        self.client = create_planner_client(self.policy_config.api_timeout_s)
        self._api_calls = 0
        self._selected_gripper_id: str | None = None
        self._selected_arm_id: str | None = None
        self._artifact_id = f"{os.getpid()}_{time.time_ns()}"

    def _get_ik_unlocked_move_group_ids(self) -> list[str]:
        if self._selected_arm_id is not None:
            return [self._selected_arm_id]
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

        # Build a balanced nearest-pose pool instead of allowing one arm to
        # dominate the shared candidate list. Then use the same sequential
        # MuJoCo IK used by validation/execution to label arm compatibility.
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
        selected_ids: list[int] = []
        rank = 0
        while len(selected_ids) < self.policy_config.llm_max_grasp_candidates:
            added = False
            for ids in rankings.values():
                if rank >= len(ids):
                    continue
                idx = int(ids[rank])
                if idx not in selected_ids:
                    selected_ids.append(idx)
                    added = True
                    if len(selected_ids) >= self.policy_config.llm_max_grasp_candidates:
                        break
            if not added and all(rank >= len(ids) - 1 for ids in rankings.values()):
                break
            rank += 1

        robot = self.task.env.current_robot
        qpos = self.robot_view.get_qpos_dict()
        base_pose = self.robot_view.base.pose
        feasible_poses: list[np.ndarray] = []
        reachable_arms: list[list[str]] = []
        for idx in selected_ids:
            pose = poses[idx].copy()
            arms = []
            for arm, gripper in arm_map.items():
                solution = robot.kinematics.ik(gripper, pose, [arm], qpos, base_pose)
                if solution is not None:
                    arms.append(arm)
            if arms:
                feasible_poses.append(pose)
                reachable_arms.append(arms)

        log.info(
            "Found %d arm-compatible grasp candidates from %d nearest candidates",
            len(feasible_poses),
            len(selected_ids),
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
        return {
            "output_contract": {
                "json_schema": LLMWaypointPlan.model_json_schema(),
                "base_segment_example": {
                    "phase": "base_transfer",
                    "motion": "base",
                    "base_goal": [0.0, 0.0, 0.0],
                    "duration_s": 3.0,
                },
                "base_goal_format": "world-frame [x_m,y_m,yaw_rad]; exactly 3 values, not base_pose; example coordinates are illustrative only",
                "required_top_level_keys": [
                    "schema_version",
                    "robot",
                    "arm",
                    "grasp_candidate_id",
                    "segments",
                ],
                "exact_shape_example": {
                    "schema_version": 1,
                    "robot": self.robot_view.name,
                    "arm": candidate_arms[0][0],
                    "grasp_candidate_id": 0,
                    "segments": [
                        {
                            "phase": phase,
                            "motion": "ee",
                            "waypoints": [
                                {
                                    "pose": ["x", "y", "z", "qw", "qx", "qy", "qz"],
                                    "speed": 0.1,
                                }
                            ],
                        }
                        for phase in (
                            "pregrasp",
                            "grasp",
                            "lift",
                            "preplace",
                            "place",
                            "retreat",
                        )
                    ],
                },
                "forbidden": [
                    "markdown or prose",
                    "unknown fields",
                    "torso commands",
                    "base_segment or ee_segment as top-level keys",
                    "base_goal and waypoints in the same segment",
                ],
            },
            "constraints": {
                "phase_grammar": "[base_approach?],pregrasp,grasp,lift,[base_transfer?],preplace,place,retreat",
                "base_bounds_relative_to_start_m": self.policy_config.llm_base_xy_limit_m,
                "max_total_waypoints": self.policy_config.llm_max_waypoints,
                "torso_fixed": True,
                "unused_arm_fixed": True,
                "grasp_rule": "the final grasp waypoint must equal the selected candidate pose (3 cm / 20 deg tolerance)",
                "clearance_rule": "lift and preplace must clear all supplied AABBs before transport/descent",
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
                {
                    "id": idx,
                    "pose": pose_mat_to_7d(pose).tolist(),
                    "reachable_arms": candidate_arms[idx],
                }
                for idx, pose in enumerate(candidates)
            ],
            "nearby_obstacles": [value for _, value in obstacles[:64]],
            "previous_validation_error": previous_error,
            "validation_error_history": list(error_history),
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

    def _request_valid_plan(
        self,
        pickup_obj: MlSpacesObject,
        receptacle: MlSpacesObject,
        candidates: list[np.ndarray],
        candidate_arms: list[list[str]],
    ) -> LLMWaypointPlan:
        previous_error = None
        error_history: list[str] = []
        for attempt in range(1, self.policy_config.llm_max_api_calls + 1):
            self._api_calls += 1
            payload = self._scene_payload(
                pickup_obj,
                receptacle,
                candidates,
                candidate_arms,
                previous_error,
                error_history,
            )
            record: dict[str, Any] = {
                "plan_id": self._artifact_id,
                "attempt": attempt,
                "pickup_object": pickup_obj.name,
                "receptacle": receptacle.name,
                "input": payload,
            }
            try:
                response = self.client.complete(SYSTEM_PROMPT, payload)
                record.update(
                    raw_response=response.content,
                    latency_s=response.latency_s,
                    usage=response.usage,
                    request_id=response.request_id,
                )
                plan = LLMWaypointPlan.model_validate(extract_json_object(response.content))
                self._validate_plan(plan, candidates, candidate_arms, pickup_obj)
                record["parsed_plan"] = plan.model_dump(mode="json")
                record["valid"] = True
                self._write_artifact(attempt, record)
                log.info("LLM waypoint plan accepted on attempt %d", attempt)
                return plan
            except (RuntimeError, ValidationError, PlanValidationError, ValueError) as exc:
                previous_error = f"{type(exc).__name__}: {exc}"
                error_history.append(previous_error)
                record.update(valid=False, error=previous_error)
                self._write_artifact(attempt, record)
                log.warning("LLM waypoint plan attempt %d rejected: %s", attempt, previous_error)
        raise ValueError(
            f"LLM planning failed after {self.policy_config.llm_max_api_calls} calls: {previous_error}"
        )

    def _validate_plan(
        self,
        plan: LLMWaypointPlan,
        candidates: list[np.ndarray],
        candidate_arms: list[list[str]],
        pickup_obj: MlSpacesObject,
    ) -> None:
        arm_map = self._available_arms()
        if plan.robot not in (self.robot_view.name, self.config.robot_config.name):
            raise PlanValidationError(f"robot mismatch: {plan.robot}")
        if plan.arm not in arm_map:
            raise PlanValidationError(f"invalid arm {plan.arm}; expected one of {list(arm_map)}")
        if plan.grasp_candidate_id >= len(candidates):
            raise PlanValidationError("grasp_candidate_id is out of range")
        if plan.arm not in candidate_arms[plan.grasp_candidate_id]:
            raise PlanValidationError(
                f"grasp candidate {plan.grasp_candidate_id} is not reachable by {plan.arm}; "
                f"use one of {candidate_arms[plan.grasp_candidate_id]}"
            )
        if plan.waypoint_count > self.policy_config.llm_max_waypoints:
            raise PlanValidationError("too many waypoints")
        start_xy = np.asarray(self.robot_view.get_move_group("base").joint_pos[:2])
        for segment in plan.segments:
            if segment.base_goal is not None:
                if (
                    np.linalg.norm(np.asarray(segment.base_goal[:2]) - start_xy)
                    > self.policy_config.llm_base_xy_limit_m
                ):
                    raise PlanValidationError(f"{segment.phase}: base goal exceeds allowed radius")
        grasp_segment = next(segment for segment in plan.segments if segment.phase == "grasp")
        proposed_grasp = grasp_segment.waypoints[-1].matrix()
        candidate = candidates[plan.grasp_candidate_id]
        pos_error = float(np.linalg.norm(proposed_grasp[:3, 3] - candidate[:3, 3]))
        rot_error = float(
            Rotation.from_matrix(candidate[:3, :3].T @ proposed_grasp[:3, :3]).magnitude()
        )
        if pos_error > 0.03 or rot_error > math.radians(20.0):
            raise PlanValidationError(
                f"grasp waypoint differs from candidate (position={pos_error:.3f}m, rotation={math.degrees(rot_error):.1f}deg)"
            )
        self._selected_arm_id = plan.arm
        self._selected_gripper_id = arm_map[plan.arm]
        self._preflight_kinematics_and_contacts(plan, pickup_obj)

    def _contact_pairs(self) -> set[tuple[int, int]]:
        data = self.task.env.current_data
        pairs = set()
        for idx in range(data.ncon):
            contact = data.contact[idx]
            geoms = sorted((int(contact.geom1), int(contact.geom2)))
            pairs.add((geoms[0], geoms[1]))
        return pairs

    def _assert_no_new_robot_contact(
        self,
        baseline: set[tuple[int, int]],
        phase: str,
        pickup_root: int,
    ) -> None:
        robot_root = self.robot_view.root_body_id
        model = self.task.env.current_model
        gripper_root = self.robot_view.get_move_group(
            self._selected_gripper_id
        ).root_body_id
        gripper_body_ids = descendant_bodies(model, gripper_root)
        holding_phases = {"grasp", "lift", "base_transfer", "preplace", "place"}
        for geom_pair in self._contact_pairs() - baseline:
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
            names = tuple(model.geom(geom_id).name or f"geom_{geom_id}" for geom_id in geom_pair)
            raise PlanValidationError(f"{phase}: new MuJoCo collision between geoms {names}")

    def _preflight_kinematics_and_contacts(
        self, plan: LLMWaypointPlan, pickup_obj: MlSpacesObject
    ) -> None:
        model, data = self.task.env.current_model, self.task.env.current_data
        snapshot = {name: value.copy() for name, value in self.robot_view.get_qpos_dict().items()}
        baseline = self._contact_pairs()
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
                        self._assert_no_new_robot_contact(baseline, segment.phase, pickup_root)
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
                            [self._selected_arm_id],
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
                        self._assert_no_new_robot_contact(baseline, segment.phase, pickup_root)
                    current_pose = goal
        finally:
            self.robot_view.set_qpos_dict(snapshot)
            mujoco.mj_forward(model, data)

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
