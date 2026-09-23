"""P2 独立几何可行性评价器。

该模块只使用 MuJoCo 顺序 IK 和接触信息，不创建 policy，也不导入 CuRobo/LLM。
``not_found`` 的含义严格限定为：在给定候选、初值和时间预算内未找到完整链；
它不是不可达性的数学证明。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from molmo_spaces.env.data_views import MlSpacesObject, create_mlspaces_body
from molmo_spaces.utils.grasps import get_pickup_grasps
from molmo_spaces.utils.linalg_utils import transform_to_twist, twist_to_transform
from molmo_spaces.utils.mj_model_and_data_utils import descendant_bodies

from .snapshot import override_base_pose


LAYERS = ("F_base", "F_IK", "F_approach", "F_lift_proxy")


class FeasibilityUnknown(RuntimeError):
    """评价无法给出协议内标签，而不是协议搜索未找到解。"""


@dataclass(frozen=True)
class FeasibilityBudget:
    max_candidates: int = 32
    ik_seeds_per_arm: int = 3
    timeout_sec: float = 300.0
    pregrasp_standoff_m: float = 0.04
    lift_height_m: float = 0.05
    tcp_translation_step_m: float = 0.01
    tcp_rotation_step_deg: float = 3.0
    boundary_translation_step_m: float = 0.005
    boundary_rotation_step_deg: float = 1.5
    joint_step_deg: float = 3.0
    self_contact_tolerance_m: float = 0.002
    environment_contact_tolerance_m: float = 0.001
    ik_position_tolerance_m: float = 1e-3
    ik_rotation_tolerance_deg: float = 1.0
    ik_max_iter: int = 300

    def __post_init__(self):
        if self.max_candidates < 1 or self.ik_seeds_per_arm < 1:
            raise ValueError("候选数和 IK 初值数必须为正数")
        if self.timeout_sec <= 0:
            raise ValueError("timeout_sec 必须为正数")
        if self.ik_max_iter < 1:
            raise ValueError("ik_max_iter 必须为正数")
        if self.lift_height_m <= 0 or self.pregrasp_standoff_m <= 0:
            raise ValueError("pregrasp/lift 距离必须为正数")
        if self.tcp_translation_step_m > 0.01 or self.tcp_rotation_step_deg > 3.0:
            raise ValueError("主检查分辨率不得粗于 1 cm / 3 deg")


@dataclass(frozen=True)
class GraspPool:
    """固定在目标物坐标系中的候选；IDs 永不因站位变化而重排。"""

    object_poses: np.ndarray
    candidate_ids: tuple[int, ...]
    sha256: str

    def __post_init__(self):
        poses = np.asarray(self.object_poses)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError("grasp pool 必须为 [N,4,4]")
        if len(poses) != len(self.candidate_ids) or len(set(self.candidate_ids)) != len(poses):
            raise ValueError("grasp IDs 必须与候选一一对应且唯一")
        if not np.isfinite(poses).all():
            raise ValueError("grasp pool 含 NaN/Inf")

    def world_poses(self, target_pose: np.ndarray) -> np.ndarray:
        return np.asarray(target_pose)[None, :, :] @ self.object_poses


def make_grasp_pool(task, target: MlSpacesObject, max_candidates: int = 32,
                    grasp_libraries: list[str] | None = None) -> GraspPool:
    """只调用一次抓取库，保留源顺序并转为物体坐标。"""
    world = np.asarray(get_pickup_grasps(
        task.env, target, grasp_libraries=grasp_libraries
    ), dtype=float)
    if len(world) == 0:
        raise FeasibilityUnknown("missing_grasp_assets:no_candidates")
    count = min(len(world), int(max_candidates))
    local = np.linalg.inv(target.pose) @ world[:count]
    ids = tuple(range(count))
    payload = local.astype("<f8", copy=False).tobytes() + json.dumps(ids).encode()
    return GraspPool(local.copy(), ids, hashlib.sha256(payload).hexdigest())


def _copy_qpos(qpos: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(value, dtype=float).copy() for name, value in qpos.items()}


def _pose_error(actual: np.ndarray, goal: np.ndarray) -> tuple[float, float]:
    pos = float(np.linalg.norm(actual[:3, 3] - goal[:3, 3]))
    rot = float(Rotation.from_matrix(actual[:3, :3].T @ goal[:3, :3]).magnitude())
    return pos, math.degrees(rot)


def _json_qpath(path: list[dict[str, np.ndarray]], arm: str) -> list[list[float]]:
    return [np.asarray(q[arm], dtype=float).round(9).tolist() for q in path]


class ManipulationFeasibilityEvaluator:
    """在给定快照上依次执行 base、IK、approach、lift-proxy 检查。"""

    def __init__(self, task):
        self.task = task
        self.env = task.env
        self.robot = task.env.current_robot
        self.view = self.robot.robot_view
        self.model = task.env.current_model
        self.data = task.env.current_data
        self._deadline = math.inf
        self._budget = FeasibilityBudget()
        self._progress: dict[str, Any] | None = None

    def _check_time(self):
        if time.monotonic() > self._deadline:
            raise TimeoutError("feasibility budget exhausted")

    def _arms(self) -> dict[str, str]:
        groups = set(self.view.move_group_ids())
        grippers = set(self.view.get_gripper_movegroup_ids())
        if {"left_arm", "right_arm", "left_gripper", "right_gripper"} <= groups:
            return {"left_arm": "left_gripper", "right_arm": "right_gripper"}
        if {"arm", "gripper"} <= groups and "gripper" in grippers:
            return {"arm": "gripper"}
        raise FeasibilityUnknown(f"unsupported_robot_layout:{sorted(groups)}")

    def _target(self, target: str | MlSpacesObject) -> MlSpacesObject:
        return target if isinstance(target, MlSpacesObject) else MlSpacesObject(target, self.data)

    def _set_target_pose(self, target: MlSpacesObject, pose: np.ndarray):
        """MlSpacesObject 是只读观测视图；通过其实际 free-joint body 写虚拟位姿。"""
        try:
            mutable = create_mlspaces_body(self.data, target.name)
            mutable.pose = np.asarray(pose, dtype=float)
        except (AssertionError, ValueError, AttributeError) as exc:
            raise FeasibilityUnknown(f"target_not_movable:{target.name}") from exc

    def _set_base_pose(self, base_pose):
        pose = np.asarray(base_pose, dtype=float)
        if pose.shape == (3,):
            matrix = self.view.base.pose.copy()
            matrix[:3, :3] = Rotation.from_euler("z", pose[2]).as_matrix()
            matrix[:2, 3] = pose[:2]
            pose = matrix
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("base_pose 必须为 [x,y,yaw] 或 4x4")
        override_base_pose(self.task, pose)
        mujoco.mj_forward(self.model, self.data)

    def _finger_bodies(self, gripper: str) -> set[int]:
        group = self.view.get_move_group(gripper)
        descendants = set(descendant_bodies(self.model, group.root_body_id))
        named = {
            body_id for body_id in descendants
            if any(token in (self.model.body(body_id).name or "").lower()
                   for token in ("finger", "pad", "tip"))
        }
        # 某些机器人把整个夹爪做成一个 body；此时只能退回夹爪子树。
        return named or descendants

    def _contacts(self) -> dict[tuple[int, int], float]:
        pairs: dict[tuple[int, int], float] = {}
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if not np.isfinite(contact.dist):
                raise FeasibilityUnknown("nonfinite_contact_distance")
            pair = tuple(sorted((int(contact.geom1), int(contact.geom2))))
            pairs[pair] = min(pairs.get(pair, math.inf), float(contact.dist))
        return pairs

    def _contact_rows(self, target_root: int, gripper: str | None,
                      allow_finger_target: bool,
                      target_baseline: dict[tuple[int, int], float] | None = None) -> list[dict]:
        robot_root = int(self.view.root_body_id)
        fingers = self._finger_bodies(gripper) if gripper else set()
        rows = []
        for pair, depth in self._contacts().items():
            if depth > 0:
                continue
            bodies = tuple(int(self.model.geom_bodyid[g]) for g in pair)
            roots = tuple(int(self.model.body_rootid[b]) for b in bodies)
            names = tuple(self.model.body(r).name or "" for r in roots)
            robot_involved = robot_root in roots
            target_involved = target_root in roots
            if not (robot_involved or target_involved):
                continue
            if robot_involved and len(set(roots)) > 1:
                other = names[1] if roots[0] == robot_root else names[0]
                if "floor" in other.lower():
                    continue
            if allow_finger_target and robot_involved and target_involved:
                robot_body = bodies[0] if roots[0] == robot_root else bodies[1]
                if robot_body in fingers:
                    continue
            # 目标物原有的支撑接触在虚拟附着后可以消失，但不能恶化或产生新接触。
            if target_involved and not robot_involved and target_baseline is not None:
                old = target_baseline.get(pair)
                if old is not None and depth >= old - self._budget.environment_contact_tolerance_m:
                    continue
            self_contact = robot_involved and roots[0] == roots[1]
            tolerance = (self._budget.self_contact_tolerance_m if self_contact
                         else self._budget.environment_contact_tolerance_m)
            if -depth <= tolerance:
                continue
            rows.append({
                "geom_ids": list(pair),
                "geom_names": [self.model.geom(g).name or f"geom_{g}" for g in pair],
                "body_names": [self.model.body(b).name or f"body_{b}" for b in bodies],
                "root_names": list(names),
                "penetration_m": float(-depth),
                "kind": "self" if self_contact else "target" if target_involved else "environment",
            })
        return rows

    def _ik_seeds(self, initial: dict[str, np.ndarray], arm: str, count: int):
        seeds = [_copy_qpos(initial)]
        group = self.view.get_move_group(arm)
        limits = np.asarray(group.joint_pos_limits, dtype=float)
        finite = np.isfinite(limits).all(axis=1)
        fractions = (0.25, 0.75)
        for index in range(1, count):
            seed = _copy_qpos(initial)
            value = seed[arm].copy()
            fraction = fractions[(index - 1) % len(fractions)]
            value[finite] = limits[finite, 0] + fraction * (
                limits[finite, 1] - limits[finite, 0]
            )
            # 若需要超过 3 个初值，用确定性的镜像小扰动继续扩展。
            if index > len(fractions):
                delta = (index - len(fractions)) * math.radians(2.0)
                value[finite] = np.clip(value[finite] + (-1) ** index * delta,
                                        limits[finite, 0], limits[finite, 1])
            seed[arm] = value
            seeds.append(seed)
        return seeds

    def _solve(self, gripper: str, arm: str, goal: np.ndarray,
               q0: dict[str, np.ndarray], fixed: dict[str, np.ndarray]):
        self._check_time()
        base_before = self.view.base.pose.copy()
        try:
            solution = self.robot.kinematics.ik(
                gripper, goal, [arm], _copy_qpos(q0), base_before,
                max_iter=self._budget.ik_max_iter,
            )
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            raise FeasibilityUnknown(f"numeric_ik:{type(exc).__name__}") from exc
        if solution is None:
            # MlSpacesKinematics 在一份独立的 stripped robot model 上迭代；失败时
            # 不暴露最终迭代状态。这里不能把 task view 的当前误差冒充 IK 残差。
            return None, None
        solution = _copy_qpos(solution)
        if not all(np.isfinite(value).all() for value in solution.values()):
            raise FeasibilityUnknown("nonfinite_ik_solution")
        # IK 只允许改变所选臂。base/torso/另一臂/夹爪均须逐位固定。
        for name, value in fixed.items():
            if name != arm and not np.array_equal(solution[name], value):
                raise FeasibilityUnknown(f"ik_changed_locked_group:{name}")
        if not np.allclose(self.view.base.pose, base_before, rtol=0, atol=1e-12):
            raise FeasibilityUnknown("ik_changed_base_pose")
        limits = np.asarray(self.view.get_move_group(arm).joint_pos_limits, dtype=float)
        if np.any(solution[arm] < limits[:, 0] - 1e-9) or np.any(solution[arm] > limits[:, 1] + 1e-9):
            return None, None
        self.view.set_qpos_dict(solution)
        mujoco.mj_forward(self.model, self.data)
        error = _pose_error(self.view.get_move_group(gripper).leaf_frame_to_world, goal)
        if (error[0] > self._budget.ik_position_tolerance_m or
                error[1] > self._budget.ik_rotation_tolerance_deg):
            return None, error
        return solution, error

    def _joint_path(self, start, end, arm, gripper, target_root, allow_target=False):
        delta = np.max(np.abs(np.asarray(end[arm]) - np.asarray(start[arm])))
        start_pose = self.view.get_move_group(gripper).leaf_frame_to_world.copy()
        self.view.set_qpos_dict(end)
        mujoco.mj_forward(self.model, self.data)
        end_pose = self.view.get_move_group(gripper).leaf_frame_to_world.copy()
        self.view.set_qpos_dict(start)
        mujoco.mj_forward(self.model, self.data)
        lin, ang = transform_to_twist(np.linalg.inv(start_pose) @ end_pose)
        samples = max(1, math.ceil(delta / math.radians(self._budget.joint_step_deg)),
                      math.ceil(np.linalg.norm(lin) / self._budget.tcp_translation_step_m),
                      math.ceil(np.linalg.norm(ang) / math.radians(self._budget.tcp_rotation_step_deg)))
        path = []
        for alpha in np.linspace(0.0, 1.0, samples + 1)[1:]:
            self._check_time()
            q = _copy_qpos(start)
            q[arm] = np.asarray(start[arm]) + alpha * (np.asarray(end[arm]) - np.asarray(start[arm]))
            self.view.set_qpos_dict(q)
            mujoco.mj_forward(self.model, self.data)
            collisions = self._contact_rows(target_root, gripper, allow_target)
            if collisions:
                return None, collisions
            path.append(q)
        return path, []

    def _cartesian_path(self, start_pose, goal_pose, qstate, arm, gripper,
                        fixed, target_root, allow_target, attach_target=None,
                        target_baseline=None, fine=False):
        lin, ang = transform_to_twist(np.linalg.inv(start_pose) @ goal_pose)
        step_m = (self._budget.boundary_translation_step_m if fine
                  else self._budget.tcp_translation_step_m)
        step_deg = (self._budget.boundary_rotation_step_deg if fine
                    else self._budget.tcp_rotation_step_deg)
        samples = max(1, math.ceil(np.linalg.norm(lin) / step_m),
                      math.ceil(np.linalg.norm(ang) / math.radians(step_deg)))
        path = []
        for alpha in np.linspace(0.0, 1.0, samples + 1)[1:]:
            self._check_time()
            pose = start_pose @ twist_to_transform(lin * alpha, ang * alpha)
            solution, error = self._solve(gripper, arm, pose, qstate, fixed)
            if solution is None:
                return None, {"reason": "path_ik", "sample": len(path) + 1,
                              "samples": samples, "residual": error}
            qstate = solution
            self.view.set_qpos_dict(qstate)
            mujoco.mj_forward(self.model, self.data)
            if attach_target is not None:
                target, ee_to_target = attach_target
                self._set_target_pose(
                    target, self.view.get_move_group(gripper).leaf_frame_to_world @ ee_to_target
                )
                mujoco.mj_forward(self.model, self.data)
            collisions = self._contact_rows(
                target_root, gripper, allow_target, target_baseline=target_baseline
            )
            if collisions:
                return None, {"reason": "collision", "sample": len(path) + 1,
                              "samples": samples, "collisions": collisions}
            path.append(qstate)
        return path, None

    def _evaluate_impl(self, base_pose, target, pool: GraspPool,
                       budget: FeasibilityBudget) -> dict[str, Any]:
        self._budget = budget
        self._deadline = time.monotonic() + budget.timeout_sec
        started = time.monotonic()
        target = self._target(target)
        self._set_base_pose(base_pose)
        initial = _copy_qpos(self.view.get_qpos_dict())
        target_pose = target.pose.copy()
        target_root = int(target.object_root_id)
        base_collisions = self._contact_rows(target_root, None, False)
        layer_counts = {layer: 0 for layer in LAYERS}
        if base_collisions:
            return {
                "status": "not_found", "first_failure_layer": "F_base",
                "layer_pass_counts": layer_counts, "candidate_diagnostics": [],
                "base_collisions": base_collisions, "witness": None,
                "elapsed_sec": time.monotonic() - started,
            }
        layer_counts["F_base"] = 1
        arms = self._arms()
        world_poses = pool.world_poses(target_pose)
        diagnostics = []
        approach_witness = None
        witness = None
        self._progress = {
            "layer_pass_counts": layer_counts,
            "candidate_diagnostics": diagnostics,
            "base_collisions": [],
            "approach_only_witness": None,
        }
        for pool_index, (candidate_id, grasp) in enumerate(zip(pool.candidate_ids, world_poses)):
            if pool_index >= budget.max_candidates:
                break
            approach_axis = grasp[:3, 2]
            pregrasp = grasp.copy()
            pregrasp[:3, 3] -= budget.pregrasp_standoff_m * approach_axis
            lift = grasp.copy()
            lift[2, 3] += budget.lift_height_m
            for arm, gripper in arms.items():
                for seed_id, seed in enumerate(self._ik_seeds(initial, arm, budget.ik_seeds_per_arm)):
                    self._check_time()
                    self.view.set_qpos_dict(initial)
                    self._set_target_pose(target, target_pose)
                    mujoco.mj_forward(self.model, self.data)
                    row: dict[str, Any] = {
                        "candidate_id": int(candidate_id), "arm": arm,
                        "seed_id": seed_id, "failure_layer": None,
                    }
                    pre_q, pre_error = self._solve(gripper, arm, pregrasp, seed, initial)
                    if pre_q is None:
                        row.update(failure_layer="F_IK", failure_reason="pregrasp_ik",
                                   pregrasp_residual=pre_error)
                        diagnostics.append(row)
                        continue
                    grasp_q, grasp_error = self._solve(gripper, arm, grasp, pre_q, initial)
                    if grasp_q is None:
                        row.update(failure_layer="F_IK", failure_reason="grasp_ik",
                                   grasp_residual=grasp_error)
                        diagnostics.append(row)
                        continue
                    layer_counts["F_IK"] += 1
                    self.view.set_qpos_dict(initial)
                    mujoco.mj_forward(self.model, self.data)
                    to_pre, collisions = self._joint_path(
                        initial, pre_q, arm, gripper, target_root, False
                    )
                    if to_pre is None:
                        row.update(failure_layer="F_approach", failure_reason="to_pregrasp_collision",
                                   collisions=collisions)
                        diagnostics.append(row)
                        continue
                    approach, path_error = self._cartesian_path(
                        pregrasp, grasp, pre_q, arm, gripper, initial,
                        target_root, True, fine=False
                    )
                    if approach is None:
                        # 碰撞边界用 0.5 cm / 1.5 deg 复核，避免主网格恰好跨过窄接触。
                        if path_error.get("reason") == "collision":
                            self.view.set_qpos_dict(pre_q)
                            self._set_target_pose(target, target_pose)
                            mujoco.mj_forward(self.model, self.data)
                            refined, refined_error = self._cartesian_path(
                                pregrasp, grasp, pre_q, arm, gripper, initial,
                                target_root, True, fine=True
                            )
                            if refined is not None:
                                approach, path_error = refined, None
                            else:
                                path_error = {**refined_error, "boundary_rechecked": True}
                    if approach is None:
                        row.update(failure_layer="F_approach", **path_error)
                        diagnostics.append(row)
                        continue
                    layer_counts["F_approach"] += 1
                    full_approach = to_pre + approach
                    if approach_witness is None:
                        approach_witness = {
                            "candidate_id": int(candidate_id), "arm": arm,
                            "seed_id": seed_id, "joint_path": _json_qpath(full_approach, arm),
                        }
                        self._progress["approach_only_witness"] = approach_witness
                    self.view.set_qpos_dict(grasp_q)
                    self._set_target_pose(target, target_pose)
                    mujoco.mj_forward(self.model, self.data)
                    target_baseline = self._contacts()
                    ee = self.view.get_move_group(gripper).leaf_frame_to_world.copy()
                    ee_to_target = np.linalg.inv(ee) @ target_pose
                    lift_path, lift_error = self._cartesian_path(
                        grasp, lift, grasp_q, arm, gripper, initial,
                        target_root, True, attach_target=(target, ee_to_target),
                        target_baseline=target_baseline, fine=False
                    )
                    if lift_path is None:
                        if lift_error.get("reason") == "collision":
                            self.view.set_qpos_dict(grasp_q)
                            self._set_target_pose(target, target_pose)
                            mujoco.mj_forward(self.model, self.data)
                            target_baseline = self._contacts()
                            ee = self.view.get_move_group(gripper).leaf_frame_to_world.copy()
                            refined, refined_error = self._cartesian_path(
                                grasp, lift, grasp_q, arm, gripper, initial,
                                target_root, True,
                                attach_target=(target, np.linalg.inv(ee) @ target_pose),
                                target_baseline=target_baseline, fine=True
                            )
                            if refined is not None:
                                lift_path, lift_error = refined, None
                            else:
                                lift_error = {**refined_error, "boundary_rechecked": True}
                    if lift_path is None:
                        row.update(failure_layer="F_lift_proxy", **lift_error)
                        diagnostics.append(row)
                        continue
                    layer_counts["F_lift_proxy"] += 1
                    row["passed"] = True
                    diagnostics.append(row)
                    if witness is None:
                        witness = {
                            "candidate_id": int(candidate_id), "arm": arm,
                            "seed_id": seed_id,
                            "joint_path": _json_qpath(full_approach + lift_path, arm),
                            "segments": {"to_pregrasp": len(to_pre),
                                         "approach": len(approach), "lift": len(lift_path)},
                        }
        first_failure = next((layer for layer in LAYERS if layer_counts[layer] == 0), None)
        return {
            "status": "feasible" if witness is not None else "not_found",
            "first_failure_layer": None if witness is not None else first_failure,
            "layer_pass_counts": layer_counts, "candidate_diagnostics": diagnostics,
            "base_collisions": [], "approach_only_witness": approach_witness,
            "witness": witness, "elapsed_sec": time.monotonic() - started,
        }

    def evaluate(self, snapshot, robot, base_pose, target, grasp_pool: GraspPool,
                 budget: FeasibilityBudget | None = None) -> dict[str, Any]:
        """评价并无条件恢复输入快照；异常/超时统一返回 ``unknown``。"""
        if robot is not self.robot:
            raise ValueError("robot 必须是 task.env.current_robot")
        budget = budget or FeasibilityBudget()
        started = time.monotonic()
        self._progress = None
        try:
            with snapshot.restored(self.task):
                result = self._evaluate_impl(base_pose, target, grasp_pool, budget)
        except TimeoutError as exc:
            progress = self._progress or {}
            result = {"status": "unknown", "unknown_reason": "timeout",
                      "error": str(exc),
                      "layer_pass_counts": progress.get(
                          "layer_pass_counts", {layer: 0 for layer in LAYERS}
                      ),
                      "first_failure_layer": None,
                      "candidate_diagnostics": progress.get("candidate_diagnostics", []),
                      "base_collisions": progress.get("base_collisions", []),
                      "approach_only_witness": progress.get("approach_only_witness"),
                      "witness": None}
        except (FeasibilityUnknown, ValueError) as exc:
            result = {"status": "unknown", "unknown_reason": str(exc).split(":", 1)[0],
                      "error": str(exc), "layer_pass_counts": {layer: 0 for layer in LAYERS},
                      "first_failure_layer": None, "candidate_diagnostics": [], "witness": None}
        result.setdefault("elapsed_sec", time.monotonic() - started)
        result.update({
            "schema_version": 1,
            "protocol": "last-mile-p2-v1",
            "grasp_pool_sha256": grasp_pool.sha256,
            "grasp_pool_size": len(grasp_pool.candidate_ids),
            "budget": asdict(budget),
            "search_semantics": "not_found means no chain found under this finite protocol",
            "curobo_used": False,
            "llm_used": False,
        })
        return result

    def check_base_poses(self, snapshot, robot, base_poses, target) -> dict[str, Any]:
        """按 P2 ``F_base`` 语义检查一串完整机器人底盘位姿。

        整串位姿只恢复一次快照，供 P3 的廉价静态预检和走廊 ``C(A,B)``
        使用。正常或碰撞退出后均恢复原快照。
        """
        if robot is not self.robot:
            raise ValueError("robot 必须是 task.env.current_robot")
        poses = [np.asarray(pose, dtype=float).copy() for pose in base_poses]
        target = self._target(target)
        with snapshot.restored(self.task):
            target_root = int(target.object_root_id)
            for index, pose in enumerate(poses):
                self._set_base_pose(pose)
                collisions = self._contact_rows(target_root, None, False)
                if collisions:
                    return {
                        "collision_free": False,
                        "checked_poses": index + 1,
                        "first_collision_index": index,
                        "collisions": collisions,
                    }
        return {
            "collision_free": True,
            "checked_poses": len(poses),
            "first_collision_index": None,
            "collisions": [],
        }
