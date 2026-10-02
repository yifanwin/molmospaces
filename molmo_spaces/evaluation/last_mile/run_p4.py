"""P4：冻结代理配对上的真实 Pick 与实际 A→B 转移。"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import mujoco
import numpy as np

from .build import atomic_json, digest, jsonl
from .decision import formal_decision
from .feasibility import FeasibilityBudget, ManipulationFeasibilityEvaluator
from .local_search import local_to_world, wrap_angle
from .nav_adapter import nonbase_positions, nonbase_targets, max_nonbase_drift, max_target_drift, se2_pose
from .pick_validator import close_wait_steps, contact_state, selected_finger_bodies, strict_pick_verdict
from .run_p3 import _build_task, _load_pool
from .snapshot import EpisodeSnapshot, override_base_pose


PROTOCOL = {
    "schema_version": 1, "policy_dt_s": 0.1,
    "close_duration_s": 0.5, "close_extra_s": 0.2, "open_steps": 5,
    "hold_steps": 10, "joint_waypoint_steps": 1,
    "base_lock_translation_m": 0.02, "base_lock_yaw_deg": 3.0,
    "transfer_arrival_translation_m": 0.05, "transfer_arrival_yaw_deg": 5.0,
    "transfer_target_translation_m": 0.001, "transfer_target_yaw_deg": 1.0,
    "transfer_nonbase_transient_rad": 0.2, "transfer_nonbase_settled_rad": 0.001,
    "transfer_settle_control_steps": 150,
    "llm": "disabled", "curobo": "disabled", "place": "not_run",
}


def experiment_name(subset: str) -> str:
    """实验名随子集变化：正式批不得复用试点的 E07 名，否则会覆盖试点报告与图。"""
    return "P4-E07" if subset == "pilot" else "P4-formal-E01"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _verify_complete(root: Path, artifacts: dict[str, str]):
    marker = json.loads((root / "COMPLETE.json").read_text())
    for key, relative in artifacts.items():
        if marker.get(key) != digest(root / relative):
            raise ValueError(f"完成标记哈希不匹配：{root / relative}")


def _angle_error(a, b):
    return abs(wrap_angle(float(a) - float(b)))


def _pose_change(a: np.ndarray, b: np.ndarray):
    delta = np.linalg.inv(a) @ b
    return float(np.linalg.norm(delta[:3, 3])), float(math.acos(np.clip((np.trace(delta[:3, :3])-1)/2, -1, 1)))


def _sample(task, target, gripper, robot_root, target_root, finger_bodies, base_start, phase):
    env, view = task.env, task.env.current_robot.robot_view
    state = contact_state(env.current_model, env.current_data, robot_root, target_root, finger_bodies)
    base = se2_pose(view)
    state.update({
        "time_s": float(env.current_data.time), "phase": phase,
        "height_m": float(target.pose[2, 3]), "target_id_ok": int(target.object_root_id) == target_root,
        "finger_distance_m": float(view.get_gripper(gripper).inter_finger_dist),
        "base_xy_drift_m": float(np.linalg.norm(base[:2] - base_start[:2])),
        "base_yaw_drift_deg": math.degrees(_angle_error(base[2], base_start[2])),
    })
    return state


class ExecutionViolation(RuntimeError):
    """逐 4 ms 监控发现的协议失败。"""


def _tick(task, arm_action: dict | None = None, monitor=None):
    """100 ms 策略步 = 5×20 ms 控制步 = 25×4 ms MuJoCo 步。"""
    env, robot = task.env, task.env.current_robot
    if arm_action is not None:
        robot.update_control(arm_action)
    for _ in range(int(task._n_ctrl_steps_per_policy)):
        robot.compute_control()
        for _ in range(int(task._n_sim_steps_per_ctrl)):
            mujoco.mj_step(env.current_model, env.current_data)
            if monitor is not None:
                monitor()
            if not np.isfinite(env.current_data.qpos).all() or not np.isfinite(env.current_data.qvel).all():
                raise FloatingPointError("nonfinite_physics_state")
    for manager in env.object_managers:
        manager.invalidate_data_cache()


def _run_witness(task, target, evaluator, witness, label: str, trace: list[dict]):
    started = time.monotonic()
    if witness is None:
        return {"label": label, "status": "planner_no_witness", "Y_pick": 0,
                "motor_steps": 0, "elapsed_sec": time.monotonic() - started}
    arm = witness["arm"]
    gripper = evaluator._arms()[arm]
    robot, view = task.env.current_robot, task.env.current_robot.robot_view
    initial_z = float(target.pose[2, 3])
    robot_root = int(view.root_body_id)
    target_root = int(target.object_root_id)
    finger_bodies = selected_finger_bodies(task.env.current_model, gripper,
                                            evaluator._finger_bodies(gripper))
    base_start = se2_pose(view).copy()
    path = witness["joint_path"]
    segments = witness["segments"]
    if sum(segments.values()) != len(path) or any(len(q) != len(view.get_move_group(arm).joint_pos) for q in path):
        raise ValueError("P2/P3 witness 路径与分段不一致")
    phase_lengths = [("to_pregrasp", segments["to_pregrasp"]),
                     ("approach", segments["approach"]), ("lift", segments["lift"])]
    motor_steps, hold = 0, []
    status = "incomplete"

    def step(phase, action=None):
        nonlocal motor_steps, status
        def monitor():
            item = _sample(task, target, gripper, robot_root, target_root,
                           finger_bodies, base_start, phase)
            if item["base_xy_drift_m"] > PROTOCOL["base_lock_translation_m"] or \
                    item["base_yaw_drift_deg"] > PROTOCOL["base_lock_yaw_deg"]:
                trace.append(dict(item, label=label, phase="violation"))
                raise ExecutionViolation("base_lock_failed")
            if item["illegal"]:
                trace.append(dict(item, label=label, phase="violation"))
                raise ExecutionViolation("illegal_collision")
        _tick(task, action, monitor)
        motor_steps += 1
        sample = _sample(task, target, gripper, robot_root, target_root,
                         finger_bodies, base_start, phase)
        sample["label"] = label
        trace.append(sample)
        if sample["base_xy_drift_m"] > PROTOCOL["base_lock_translation_m"] or \
                sample["base_yaw_drift_deg"] > PROTOCOL["base_lock_yaw_deg"]:
            status = "base_lock_failed"
        elif sample["illegal"]:
            status = "illegal_collision"
        return sample

    try:
        robot.controllers["base"].set_to_stationary()
        # RBY1 gripper 有位置控制器；直接写 ctrl 会在 compute_control() 被
        # 控制器原先的张开 target 覆盖。必须经 update_control 设置目标。
        robot.update_control({gripper: np.asarray([-0.05])})
        for _ in range(PROTOCOL["open_steps"]):
            step("open")
            if status != "incomplete":
                break
        offset = 0
        for phase, count in phase_lengths:
            if phase == "lift" and status == "incomplete":
                robot.update_control({gripper: np.asarray([0.0])})
                for _ in range(close_wait_steps(PROTOCOL["close_duration_s"], PROTOCOL["policy_dt_s"])):
                    step("close")
                    if status != "incomplete":
                        break
            if status != "incomplete":
                break
            for q in path[offset:offset + count]:
                step(phase, {arm: np.asarray(q, dtype=float),
                             gripper: np.asarray([0.0 if phase == "lift" else -0.05])})
                if status != "incomplete":
                    break
            offset += count
            if status != "incomplete":
                break
        if status == "incomplete":
            for _ in range(PROTOCOL["hold_steps"]):
                hold.append(step("hold"))
                if status != "incomplete":
                    break
        if status == "incomplete":
            ok, status = strict_pick_verdict(hold, initial_z, PROTOCOL["hold_steps"])
        else:
            ok = False
    except ExecutionViolation as exc:
        ok, status = False, str(exc)
    except (FloatingPointError, ValueError, RuntimeError) as exc:
        ok, status = False, f"controller_error:{type(exc).__name__}:{exc}"
    y_pick = None if status == "base_lock_failed" or status.startswith("controller_error:") else int(ok)
    return {"label": label, "status": status, "Y_pick": y_pick,
            "motor_steps": motor_steps, "arm": arm, "gripper": gripper,
            "candidate_id": witness["candidate_id"], "seed_id": witness["seed_id"],
            "initial_target_z_m": initial_z, "elapsed_sec": time.monotonic() - started}


def _transfer(task, snapshot, target, evaluator, point, trace):
    """真实控制到 B；只用 P3 路径指令，不把底盘直接设到 B。"""
    snapshot.restore(task)
    env, robot = task.env, task.env.current_robot
    view = robot.robot_view
    target_start = target.pose.copy()
    arm_start = nonbase_positions(robot)
    target_commands = nonbase_targets(robot)
    a = se2_pose(view).copy()
    path = point["corridor"]["path_local"]
    if not path or point["corridor"].get("reachable") is not True:
        return {"status": "corridor_unavailable", "arrived": False}
    steps, max_nonbase = 0, 0.0
    status = "arrived"
    for local in path[1:]:
        waypoint = local_to_world(a, local)
        try:
            def monitor():
                change_m, change_rad = _pose_change(target_start, target.pose)
                contacts = contact_state(env.current_model, env.current_data,
                                         int(view.root_body_id), int(target.object_root_id), set())
                drift, _ = max_nonbase_drift(arm_start, robot)
                reason = None
                if contacts["illegal"]:
                    reason = "collision"
                elif drift > PROTOCOL["transfer_nonbase_transient_rad"]:
                    reason = "nonbase_transient"
                elif change_m > PROTOCOL["transfer_target_translation_m"] or \
                        math.degrees(change_rad) > PROTOCOL["transfer_target_yaw_deg"]:
                    reason = "scene_changed"
                if reason:
                    trace.append({"label": "transfer", "phase": "violation",
                                  "time_s": float(env.current_data.time), "reason": reason})
                    raise ExecutionViolation(reason)
            _tick(task, {"base": waypoint}, monitor)
            steps += 1
            drift, _ = max_nonbase_drift(arm_start, robot)
            max_nonbase = max(max_nonbase, drift)
            change_m, change_rad = _pose_change(target_start, target.pose)
            contacts = contact_state(env.current_model, env.current_data,
                                     int(view.root_body_id), int(target.object_root_id), set())
            trace.append({"label": "transfer", "phase": "base", "step": steps,
                          "base": se2_pose(view).tolist(), "target_change_m": change_m,
                          "target_change_deg": math.degrees(change_rad),
                          "nonbase_drift_rad": drift, "illegal": contacts["illegal"]})
            command_drift, _ = max_target_drift(target_commands, robot)
            if command_drift > 0:
                status = "nonbase_command_changed"
            elif drift > PROTOCOL["transfer_nonbase_transient_rad"]:
                status = "nonbase_transient"
            elif contacts["illegal"]:
                status = "collision"
            elif change_m > PROTOCOL["transfer_target_translation_m"] or \
                    math.degrees(change_rad) > PROTOCOL["transfer_target_yaw_deg"]:
                status = "scene_changed"
        except ExecutionViolation as exc:
            status = str(exc)
        except (FloatingPointError, ValueError, RuntimeError) as exc:
            status = f"controller_error:{type(exc).__name__}:{exc}"
        if status != "arrived":
            break
    if status == "arrived":
        robot.controllers["base"].set_to_stationary()
        for _ in range(PROTOCOL["transfer_settle_control_steps"]):
            robot.compute_control()
            for _ in range(int(task._n_sim_steps_per_ctrl)):
                mujoco.mj_step(env.current_model, env.current_data)
        for manager in env.object_managers:
            manager.invalidate_data_cache()
        drift, _ = max_nonbase_drift(arm_start, robot)
        change_m, change_rad = _pose_change(target_start, target.pose)
        if drift > PROTOCOL["transfer_nonbase_settled_rad"]:
            status = "nonbase_not_settled"
        elif change_m > PROTOCOL["transfer_target_translation_m"] or \
                math.degrees(change_rad) > PROTOCOL["transfer_target_yaw_deg"]:
            status = "scene_changed"
    actual = se2_pose(view)
    goal = np.asarray(point["world_pose"])
    pos_error = float(np.linalg.norm(actual[:2] - goal[:2]))
    yaw_error = math.degrees(_angle_error(actual[2], goal[2]))
    if status == "arrived" and (pos_error > PROTOCOL["transfer_arrival_translation_m"] or
                                yaw_error > PROTOCOL["transfer_arrival_yaw_deg"]):
        status = "arrival_error"
    return {"status": status, "arrived": status == "arrived", "path_steps": steps,
            "actual_B": actual.tolist(), "goal_B": goal.tolist(),
            "position_error_m": pos_error, "yaw_error_deg": yaw_error,
            "max_nonbase_drift_rad": max_nonbase}


def _episode(index, episode, row, p1, p2, p3, roots, output, inputs_sha):
    episode_dir = output / "episodes" / f"{index:03d}"
    result_path = episode_dir / "result.json"
    trace_path = episode_dir / "trace.jsonl"
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old.get("inputs_sha256") != inputs_sha or \
                old.get("trace_sha256") != digest(trace_path):
            raise ValueError(f"P4 缓存输入/执行日志哈希不匹配：{index}")
        return old
    result = {"subset_index": index, "episode_id": row["episode_id"],
              "house": row["house"], "target": row["target"],
              "p1_status": p1["status"], "p2_status": p2["status"],
              "p3_status": p3["status"], "inputs_sha256": inputs_sha}
    trace, task, sampler = [], None, None
    started = time.monotonic()
    try:
        if p1["status"] != "completed":
            result.update(status="skipped", skip_reason="invalid_A")
        else:
            if p3["status"] != "completed":
                raise ValueError("有效 A 缺少完整 P3 地图")
            task, sampler, snapshot = _build_task(index, episode, row, roots["p1"])
            evaluator = ManipulationFeasibilityEvaluator(task)
            target = evaluator._target(row["target"])
            pool_path = roots["p2"] / "episodes" / f"{index:03d}" / "grasp_pool.npz"
            if digest(pool_path) != p2["artifact_sha256"]["grasp_pool.npz"]:
                raise ValueError("P2 抓取池文件哈希不匹配")
            pool = _load_pool(pool_path, p2["grasp_pool_sha256"])
            if p3["grasp_pool_sha256"] != pool.sha256 or p3["budget"] != p2["budget"]:
                raise ValueError("A/B 抓取池或预算不一致")
            b_id = p3.get("nearest_reachable_feasible") if p2["status"] == "not_found" else None
            point = None
            if b_id:
                candidates = _rows(roots["p3"] / "episodes" / f"{index:03d}" / "candidates.jsonl")
                point = next((value for value in candidates if value["point_id"] == b_id), None)
                if point is None or point["status"] != "feasible" or not point["reachable"] or \
                        not point["in_main_disk"] or point["grasp_pool_sha256"] != pool.sha256:
                    raise ValueError("B* 与冻结 P3 代理结果不一致")
            order = ["A", "B"] if int(hashlib.sha256(f"{row['seed']}:{row['episode_id']}:P4".encode()).hexdigest(), 16) % 2 == 0 else ["B", "A"]
            result.update(status="completed", B_point_id=b_id, execution_order=[name for name in order if name == "A" or point],
                          grasp_pool_sha256=pool.sha256, budget=p2["budget"])
            for name in order:
                if name == "B" and point is None:
                    continue
                snapshot.restore(task)
                if name == "B":
                    evaluator._set_base_pose(point["world_pose"])
                witness = p2.get("witness")
                if name == "B":
                    # P3 的 E05 模型指纹当前不可重建；在可严格恢复的 E04 模型上
                    # 以完全相同池/预算复核预选 B，绝不直接执行跨模型见证路径。
                    validation = evaluator.evaluate(snapshot, task.env.current_robot,
                                                    point["world_pose"], target, pool,
                                                    FeasibilityBudget(**p2["budget"]))
                    result["B_revalidation_status"] = validation["status"]
                    result["B_revalidation_failure_layer"] = validation.get("first_failure_layer")
                    witness = validation.get("witness")
                    evaluator._set_base_pose(point["world_pose"])
                result[name] = _run_witness(task, target, evaluator, witness, name, trace)
                if name == "A" and p2["status"] == "unknown":
                    result[name].update(status="planning_unknown", Y_pick=None)
            if point is not None:
                transfer = _transfer(task, snapshot, target, evaluator, point, trace)
                result["transfer"] = transfer
                if transfer["arrived"]:
                    provenance = {"p4_inputs_sha256": inputs_sha, "episode_id": row["episode_id"],
                                  "kind": "post_transfer"}
                    arrived_snapshot = EpisodeSnapshot.capture(task, provenance)
                    actual = transfer["actual_B"]
                    replan = evaluator.evaluate(arrived_snapshot, task.env.current_robot, actual,
                                                target, pool, FeasibilityBudget(**p2["budget"]))
                    transfer["replan_status"] = replan["status"]
                    transfer["replan_failure_layer"] = replan.get("first_failure_layer")
                    result["B_after_transfer"] = _run_witness(task, target, evaluator,
                                                              replan.get("witness"), "B_after_transfer", trace)
                    transfer["pick_from_actual_arrival"] = True
                else:
                    result["B_after_transfer"] = {"label": "B_after_transfer", "status": "transfer_failed",
                                                  "Y_pick": 0, "motor_steps": 0}
    except Exception as exc:
        result.update(status="runner_error", error=f"{type(exc).__name__}:{exc}",
                      traceback=traceback.format_exc())
    finally:
        result["elapsed_sec"] = time.monotonic() - started
        episode_dir.mkdir(parents=True, exist_ok=True)
        jsonl(trace_path, trace)
        result["trace_sha256"] = digest(trace_path)
        atomic_json(result_path, result)
        if task is not None:
            task.close()
        if sampler is not None:
            sampler.close()
        gc.collect()
    print(json.dumps({"index": index, "status": result["status"],
                      "A": result.get("A", {}).get("status"),
                      "B": result.get("B", {}).get("status"),
                      "transfer": result.get("transfer", {}).get("status")}, ensure_ascii=False), flush=True)
    return result


def _rate(num, den):
    return num / den if den else None


def aggregate(rows: list[dict], p3_rows: list[dict], subset: str):
    valid = [r for r in rows if r["p1_status"] == "completed"]
    evaluated = [r for r in valid if r["p3_status"] == "completed" and r["p2_status"] != "unknown"]
    determinate = [r for r in evaluated if not p3_rows[r["subset_index"]].get("reachable_unknown")]
    m = sum(r["p2_status"] == "not_found" for r in determinate)
    l_geo = sum(r["p2_status"] == "not_found" and p3_rows[r["subset_index"]]["geo_rescue"] for r in determinate)
    l_reach = sum(r["p2_status"] == "not_found" and p3_rows[r["subset_index"]]["reachable_rescue"] for r in determinate)
    unknown = len(valid) - len(determinate)
    pair = [r for r in valid if r.get("B_point_id")]
    known_pair = [r for r in pair if r.get("A", {}).get("Y_pick") in (0, 1) and
                  r.get("B", {}).get("Y_pick") in (0, 1)]
    cells = {f"{a}{b}": sum(r["A"]["Y_pick"] == a and r["B"]["Y_pick"] == b
                          for r in known_pair) for a in (0, 1) for b in (0, 1)}
    transfers = [r for r in pair if "transfer" in r]
    arrived = [r for r in transfers if r["transfer"].get("arrived")]
    a_known = [r for r in valid if r.get("A", {}).get("Y_pick") in (0, 1)]
    a_success = sum(r["A"]["Y_pick"] for r in a_known)
    rescue_den = cells["00"] + cells["01"]
    transfer_known = [r for r in known_pair if r.get("B_after_transfer", {}).get("Y_pick") in (0, 1)]
    transfer_rescue = sum(r["A"]["Y_pick"] == 0 and r["B_after_transfer"]["Y_pick"] == 1
                          for r in transfer_known)
    return {
        "schema_version": 1, "experiment": experiment_name(subset), "subset": subset,
        "N_attempt": len(rows), "N_nav": len(valid), "N_eval": len(determinate),
        "M": m, "L_geo": l_geo, "L_reach": l_reach,
        "I_geo": _rate(l_geo, len(determinate)), "I_reach": _rate(l_reach, len(determinate)),
        "R_oracle": _rate(l_reach, m),
        "N_unknown_A_or_neighborhood": unknown,
        "I_reach_all_valid_lower": _rate(l_reach, len(valid)),
        "I_reach_all_valid_upper": _rate(l_reach + unknown, len(valid)),
        "navigation_yield": _rate(len(valid), len(rows)),
        "A_pick_known": len(a_known), "A_pick_success": a_success,
        "A_pick_rate_known": _rate(a_success, len(a_known)),
        "A_pick_all_valid_lower": _rate(a_success, len(valid)),
        "A_pick_all_valid_upper": _rate(a_success + len(valid) - len(a_known), len(valid)),
        "A_failure_stages": {s: sum(r.get("A", {}).get("status") == s for r in valid)
                             for s in sorted({r.get("A", {}).get("status", "missing") for r in valid})},
        "N_pair_preselected": len(pair), "N_pair_known": len(known_pair), "pair_cells": cells,
        "paired_difference": _rate(cells["01"] - cells["10"], len(known_pair)),
        "real_pick_rescue_given_A_failure": _rate(cells["01"], rescue_den),
        "N_transfer_attempted": len(transfers), "N_transfer_arrived": len(arrived),
        "transfer_arrival_rate": _rate(len(arrived), len(transfers)),
        "transfer_pick_success": sum(r.get("B_after_transfer", {}).get("Y_pick") == 1 for r in transfers),
        "transfer_rescue": transfer_rescue,
        "transfer_rescue_per_attempt": _rate(transfer_rescue, len(rows)),
        "transfer_minus_static_B": _rate(sum(r["B_after_transfer"]["Y_pick"] - r["B"]["Y_pick"]
                                             for r in transfer_known), len(transfer_known)),
        "seconds_per_successful_transfer_rescue": _rate(sum(r["elapsed_sec"] for r in transfer_known
                                                            if r["A"]["Y_pick"] == 0 and
                                                            r["B_after_transfer"]["Y_pick"] == 1), transfer_rescue),
        "direction_decision": "证据不足：10 条试点不是正式 100 条；不计算聚类区间或启动 Agent" if subset == "pilot"
                              else "待正式 house 聚类区间与 unknown 敏感性审计",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("p0-root", "p0-validation", "p1-root", "p2-root", "p3-root", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--subset", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--calibration-json", type=Path, default=None,
                        help="正式批严格 Pick 的物理正负控制验收记录")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    _verify_complete(args.p0_validation, {"inputs_sha256": "inputs.json", "episodes_sha256": "episodes.jsonl"})
    _verify_complete(args.p1_root, {"inputs_sha256": "inputs.json", "episodes_sha256": "episodes.jsonl", "summary_sha256": "summary.json"})
    _verify_complete(args.p2_root, {"inputs_sha256": "inputs.json", "episodes_sha256": "episodes.jsonl", "summary_sha256": "summary.json"})
    _verify_complete(args.p3_root, {"inputs_sha256": "inputs.json", "episodes_sha256": "episodes.jsonl", "candidates_sha256": "candidates.jsonl", "metrics_sha256": "metrics.json"})
    p0 = args.p0_root / args.subset
    episodes = json.loads((p0 / "benchmark.json").read_text())
    manifest = _rows(p0 / "manifest.jsonl")
    p1, p2, p3 = (_rows(root / "episodes.jsonl") for root in (args.p1_root, args.p2_root, args.p3_root))
    if not (len(episodes) == len(manifest) == len(p1) == len(p2) == len(p3)):
        raise ValueError("P0/P1/P2/P3 行数不一致")
    for i, row in enumerate(manifest):
        if p1[i].get("A") is not None and p3[i].get("A") is not None and \
                not np.array_equal(np.asarray(p1[i]["A"]), np.asarray(p3[i]["A"])):
            raise ValueError(f"P1 快照与 P3 冻结 A 不一致：{i}")
        if any(source[i].get("episode_id") != row["episode_id"] or source[i].get("subset_index") != i
               for source in (p1, p2, p3)):
            raise ValueError(f"P0/P1/P2/P3 episode 身份不一致：{i}")
    if args.subset == "pilot" and len(episodes) != 10:
        raise ValueError("试点必须是冻结的 10 条")
    impl = ["molmo_spaces/kinematics/mujoco_kinematics.py",
            "molmo_spaces/evaluation/last_mile/run_p4.py",
            "molmo_spaces/evaluation/last_mile/decision.py",
            "molmo_spaces/evaluation/last_mile/pick_validator.py",
            "scripts/evaluation/run_last_mile_p4.sh",
            "scripts/evaluation/plot_last_mile_p4.py",
            "mlspaces_tests/evaluation/test_last_mile_p4.py"]
    inputs = {"schema_version": 1, "protocol": dict(PROTOCOL, subset=args.subset),
              "p0_validation_complete_sha256": digest(args.p0_validation / "COMPLETE.json"),
              "p0_manifest_sha256": digest(p0 / "manifest.jsonl"),
              "p0_benchmark_sha256": digest(p0 / "benchmark.json"),
              "p1_complete_sha256": digest(args.p1_root / "COMPLETE.json"),
              "p2_complete_sha256": digest(args.p2_root / "COMPLETE.json"),
              "p3_complete_sha256": digest(args.p3_root / "COMPLETE.json"),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
              "calibration_sha256": digest(args.calibration_json) if args.calibration_json else None,
              "implementation_sha256": {rel: digest(repo / rel) for rel in impl},
              "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()}
    args.output.mkdir(parents=True, exist_ok=True)
    inputs_path = args.output / "inputs.json"
    if inputs_path.exists() and json.loads(inputs_path.read_text()) != inputs:
        raise ValueError("P4 输入/实现改变；请用新 E 编号目录")
    atomic_json(inputs_path, inputs)
    inputs_sha = digest(inputs_path)
    roots = {"p1": args.p1_root, "p2": args.p2_root, "p3": args.p3_root}
    results = [_episode(i, episode, row, p1[i], p2[i], p3[i], roots, args.output, inputs_sha)
               for i, (episode, row) in enumerate(zip(episodes, manifest))]
    jsonl(args.output / "episodes.jsonl", results)
    metrics = aggregate(results, p3, args.subset)
    if args.subset == "formal":
        calibration = json.loads(args.calibration_json.read_text()) if args.calibration_json else {}
        calibrated = (calibration.get("passed") is True and
                      calibration.get("physical_positive_controls", 0) > 0 and
                      calibration.get("physical_negative_controls", 0) > 0)
        metrics["formal_decision"] = formal_decision(results, p3, metrics, calibrated)
        metrics["direction_decision"] = metrics["formal_decision"]["direction_decision"]
    atomic_json(args.output / "metrics.json", metrics)
    complete = (all(r["status"] in ("completed", "skipped") for r in results) and
                sum(r["status"] == "completed" for r in results) == sum(r["status"] == "completed" for r in p1) and
                all("A" in r and r["A"].get("status") != "base_lock_failed"
                    for r in results if r["status"] == "completed") and
                all("transfer" in r and "B" in r and "B_after_transfer" in r and
                    r.get("B_revalidation_status") == "feasible" and
                    all(r.get(key, {}).get("status") != "base_lock_failed"
                        for key in ("B", "B_after_transfer"))
                    for r in results if r.get("B_point_id")))
    summary = {"experiment": experiment_name(args.subset), "subset": args.subset, "attempted": len(results),
               "valid_A": metrics["N_nav"], "pairs": metrics["N_pair_preselected"],
               "transfer_attempted": metrics["N_transfer_attempted"], "complete": complete,
               "metrics": metrics}
    atomic_json(args.output / "summary.json", summary)
    if complete:
        atomic_json(args.output / "COMPLETE.json", {"inputs_sha256": digest(inputs_path),
                    "episodes_sha256": digest(args.output / "episodes.jsonl"),
                    "metrics_sha256": digest(args.output / "metrics.json"),
                    "summary_sha256": digest(args.output / "summary.json")})
    print(json.dumps({"complete": complete, "attempted": len(results),
                      "valid_A": metrics["N_nav"], "pairs": metrics["N_pair_preselected"]},
                     ensure_ascii=False), flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
