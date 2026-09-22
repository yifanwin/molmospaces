"""Run P1: deterministic real A* navigation on a frozen P0 subset."""

import argparse
from copy import deepcopy
import functools
import gc
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import mujoco
import numpy as np

from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec
from molmo_spaces.evaluation.eval_main import EvalRuntimeParams
from .build import atomic_json, audit, digest, jsonl
from .config import P1Config
from .nav_adapter import (
    LastMileNavAdapter, TERMINAL_STATUSES, derive_seed, execute_navigation,
    illegal_contacts, numpy_seed, sample_remote_start, se2_pose,
    synchronize_navigation_start,
)
from .sampler import P0JsonEvalTaskSampler
from .snapshot import EpisodeSnapshot, integration_state, override_base_pose
from .validate import equal


PROTOCOL = {
    "name": "P1-E03",
    "pilot_count": 10,
    "formal_simulation": "not_run",
    "start_radius_m": [1.5, 3.0],
    "start_attempts": 100,
    "start_yaw_rad": [-float(np.pi), float(np.pi)],
    "nav_agent_radius_m": 0.30,
    "path_min_dist_to_target_center_m": 0.8,
    "plan_max_retries": 3,
    "plan_fail_after_waypoint_steps": 10,
    "policy_step_budget": 600,
    "policy_dt_ms": 100.0,
    "ctrl_dt_ms": 20.0,
    "sim_dt_ms": 4.0,
    "start_controller_sync_steps": 25,
    "endpoint_se2_tolerance": 0.05,
    # 非 base 关节固定性：分两层判据（详见 nav_adapter.settle_before_handover）
    # ① 导航期间任何非 base 控制器目标被改写即失败（0 容差，无噪声）；
    # ② 物理关节位置在运动期间只受瞬态上限约束——A* 的 waypoint 是 Δxy 0.12-0.15 m、
    #    Δyaw 10-20°/100 ms 的阶跃，会让 torso 产生 0.033-0.055 rad 的弹性挠度
    #    （torso kp=4000 扛着上半身惯量，实测值，独立模型复现一致）。逐 4 ms 的
    #    1e-3 rad 峰值判据对任何会移动底盘的导航都不可能满足；
    # ③ 底盘停稳 3.0 s 后按原 1e-3 判稳态偏差，这才是交给评价器的姿态。
    "nonbase_joint_drift_tolerance": 1e-3,
    "nonbase_joint_transient_tolerance": 2e-1,
    "end_controller_settle_steps": 150,
    "nonbase_fixed_invariant": (
        "导航期间无任何非 base 控制器目标被改写；底盘停稳 3.0 s 后物理关节位置"
        "相对导航开始状态 <= 1e-3 rad"
    ),
    "target_position_tolerance_m": 1e-3,
    "target_rotation_tolerance_rad": float(np.deg2rad(1.0)),
    "repair_robot_base_pose_if_colliding": False,
}


def config_value(value):
    if isinstance(value, functools.partial):
        return {"partial": config_value(value.func), "args": value.args, "kwargs": value.keywords}
    if callable(value):
        return f"{getattr(value, '__module__', type(value).__module__)}.{getattr(value, '__qualname__', type(value).__qualname__)}"
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"未适配配置类型：{type(value)}")


def atomic_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temp, path)


def implementation_hashes(repo: Path) -> dict[str, str]:
    paths = [
        *Path(__file__).parent.glob("*.py"),
        repo / "molmo_spaces/policy/solvers/navigation/astar_planner_policy.py",
        repo / "scripts/evaluation/run_last_mile_p1.sh",
        repo / "scripts/evaluation/plot_last_mile_p1.py",
    ]
    return {str(path.relative_to(repo)): digest(path) for path in sorted(paths) if path.exists()}


def build_inputs(p0_root: Path, repo: Path, subset: str) -> dict:
    p0_provenance = json.loads((p0_root / "provenance.json").read_text())
    p0_config = json.loads((p0_root / "config.json").read_text())
    data_audit = audit(p0_root)
    return {
        "schema_version": 1,
        "protocol": dict(
            PROTOCOL,
            subset=subset,
            episode_count=data_audit[f"{subset}_count"],
            formal_simulation="scheduled" if subset == "formal" else "not_run",
        ),
        "subset": subset,
        "p0_audit": data_audit,
        "p0_source_sha256": p0_config["source_sha256"],
        "p0_manifest_sha256": digest(p0_root / "manifest.jsonl"),
        "pilot_benchmark_sha256": digest(p0_root / "pilot/benchmark.json"),
        "pilot_manifest_sha256": digest(p0_root / "pilot/manifest.jsonl"),
        "formal_benchmark_sha256": digest(p0_root / "formal/benchmark.json"),
        "formal_manifest_sha256": digest(p0_root / "formal/manifest.jsonl"),
        "asset_metadata_sha256": p0_provenance["annotation_sha256"],
        "scene_metadata_sha256": p0_provenance["scene_metadata_sha256"],
        "asset_versions": __import__(
            "molmo_spaces.molmo_spaces_constants", fromlist=["DATA_TYPE_TO_SOURCE_TO_VERSION"]
        ).DATA_TYPE_TO_SOURCE_TO_VERSION,
        "mujoco_version": mujoco.__version__,
        "base_commit": os.popen(f"git -C {repo} rev-parse HEAD").read().strip(),
        "implementation_sha256": implementation_hashes(repo),
    }


def validate_cached_result(result: dict, episode_dir: Path):
    assert result["status"] in TERMINAL_STATUSES
    for name, expected in result.get("artifact_sha256", {}).items():
        path = episode_dir / name
        if not path.exists() or digest(path) != expected:
            raise ValueError(f"缓存产物哈希不匹配：{path}")
    if result["status"] == "completed" and "A_snapshot.npz" not in result.get("artifact_sha256", {}):
        raise ValueError("completed 结果缺少 A 快照")


def run_episode(index, episode, row, inputs, output: Path):
    episode_dir = output / "episodes" / f"{index:03d}"
    result_path = episode_dir / "result.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        validate_cached_result(previous, episode_dir)
        return previous

    episode_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    task = sampler = policy = None
    trajectory = np.empty((0, 3))
    desired = np.empty((0, 3))
    free_xy = np.empty((0, 2))
    result = {
        "subset_index": index,
        "source_index": row["source_index"],
        "episode_id": row["episode_id"],
        "house": row["house"],
        "target": row["target"],
        "episode_seed": row["seed"],
        "goal_seed": derive_seed(row["seed"], "p1_goal"),
        "start_seed": derive_seed(row["seed"], "p1_start"),
    }
    try:
        seed = row["seed"]
        random.seed(seed)
        np.random.seed(seed)
        if "torch" in sys.modules:
            sys.modules["torch"].manual_seed(seed)
        exp = P1Config(seed=seed, output_dir=output)
        exp.policy_config.path_min_dist_to_target_center = PROTOCOL["path_min_dist_to_target_center_m"]
        exp.policy_config.plan_max_retries = PROTOCOL["plan_max_retries"]
        exp.policy_config.plan_fail_after_waypoint_steps = PROTOCOL["plan_fail_after_waypoint_steps"]
        exp.policy_config.plan_stick_to_original_target = True
        exp.policy_config.planner_config.agent_radius = PROTOCOL["nav_agent_radius_m"]
        exp.eval_runtime_params = EvalRuntimeParams()
        exp.eval_runtime_params.repair_robot_base_pose_if_colliding = False
        spec = EpisodeSpec.model_validate(deepcopy(episode))
        spec.seed = seed
        sampler = P0JsonEvalTaskSampler(exp, spec, seed)
        task = sampler.sample_task(house_index=spec.house_index)
        if task is None:
            raise RuntimeError("初始化未返回任务")
        if task._registered_policy is not None:
            raise RuntimeError("P1 导航器不得注册进 PnP task")

        adapter = LastMileNavAdapter(task, row["target"])
        policy = exp.policy_config.policy_factory(exp, adapter)
        with numpy_seed(result["goal_seed"]):
            goal = policy.target_pos_quat
        if goal is None:
            result.update(status="no_path", termination_detail="goal_sampling_failed")
            trajectory = np.asarray([se2_pose(task.env.current_robot.robot_view)])
            desired = np.full_like(trajectory, np.nan)
        else:
            result["planned_goal_position"] = np.asarray(goal[0]).tolist()
            result["planned_goal_quaternion"] = np.asarray(goal[1]).tolist()
            target_xy = adapter.target.position[:2]
            start_pose, start_info = sample_remote_start(
                task, policy.nav_planner, goal[0], target_xy, result["start_seed"],
                radius_range=tuple(PROTOCOL["start_radius_m"]),
                max_attempts=PROTOCOL["start_attempts"],
            )
            result["start_sampling"] = start_info
            free = policy.nav_planner.map.get_free_points()[:, :2]
            stride = max(1, len(free) // 30000)
            free_xy = free[::stride]
            if start_pose is None:
                result.update(status="start_unavailable", termination_detail=start_info["detail"])
                trajectory = np.asarray([se2_pose(task.env.current_robot.robot_view)])
                desired = np.full_like(trajectory, np.nan)
            else:
                override_base_pose(task, start_pose)
                mujoco.mj_forward(task.env.current_model, task.env.current_data)
                sync = synchronize_navigation_start(
                    adapter,
                    control_steps=PROTOCOL["start_controller_sync_steps"],
                    target_pos_tolerance=PROTOCOL["target_position_tolerance_m"],
                    target_rot_tolerance=PROTOCOL["target_rotation_tolerance_rad"],
                )
                result["start_controller_sync"] = sync
                result["start_pose"] = se2_pose(task.env.current_robot.robot_view).tolist()
                result["target_position"] = adapter.target.position.tolist()
                if sync["events"]:
                    priority = {"controller_error": 0, "collision": 1, "scene_changed": 2}
                    event = sorted(sync["events"], key=lambda x: priority[x["type"]])[0]
                    result.update(status=event["type"], termination_detail="start_controller_sync",
                                  events=sync["events"])
                    trajectory = np.asarray([result["start_pose"]])
                    desired = np.full_like(trajectory, np.nan)
                else:
                    nav_result, trajectory, desired = execute_navigation(
                        adapter, policy,
                        max_policy_steps=PROTOCOL["policy_step_budget"],
                        endpoint_tolerance=PROTOCOL["endpoint_se2_tolerance"],
                        joint_drift_tolerance=PROTOCOL["nonbase_joint_drift_tolerance"],
                        joint_transient_tolerance=PROTOCOL["nonbase_joint_transient_tolerance"],
                        settle_control_steps=PROTOCOL["end_controller_settle_steps"],
                        target_pos_tolerance=PROTOCOL["target_position_tolerance_m"],
                        target_rot_tolerance=PROTOCOL["target_rotation_tolerance_rad"],
                    )
                    result.update(nav_result)
                    if result["status"] == "completed":
                        task.env.current_robot.controllers["base"].set_to_stationary()
                        task.env.current_robot.compute_control()
                        provenance = {
                            "p1_inputs_sha256": digest(output / "inputs.json"),
                            "episode_id": row["episode_id"],
                            "trajectory_points": len(trajectory),
                        }
                        snapshot_path = episode_dir / "A_snapshot.npz"
                        snapshot = EpisodeSnapshot.capture(task, provenance)
                        snapshot.save(snapshot_path)
                        loaded = EpisodeSnapshot.load(snapshot_path, provenance)
                        loaded.restore(task, provenance)
                        actual = EpisodeSnapshot.capture(task, provenance)
                        equal(loaded.state, actual.state)
                        equal(loaded.runtime, actual.runtime)
                        equal(loaded.state, integration_state(task.env.current_model, task.env.current_data))
                        result["A"] = se2_pose(task.env.current_robot.robot_view).tolist()
                        result["snapshot_model_sha256"] = loaded.model_sha256

        trajectory_path = episode_dir / "trajectory.npz"
        atomic_npz(
            trajectory_path,
            actual_se2=trajectory,
            desired_se2=desired,
            free_xy=free_xy,
            target_xy=np.asarray(result.get("target_position", [np.nan, np.nan]))[:2],
            planned_goal_xy=np.asarray(result.get("planned_goal_position", [np.nan, np.nan]))[:2],
            planned_endpoint=np.asarray(result.get("planned_endpoint") or [np.nan] * 3),
        )
        artifacts = {"trajectory.npz": digest(trajectory_path)}
        snapshot_path = episode_dir / "A_snapshot.npz"
        if snapshot_path.exists():
            artifacts["A_snapshot.npz"] = digest(snapshot_path)
        result["artifact_sha256"] = artifacts
    except Exception as exc:
        result.update(
            status="controller_error",
            termination_detail=f"{type(exc).__name__}:{exc}",
            traceback=traceback.format_exc(),
        )
    finally:
        result["elapsed_sec"] = time.monotonic() - start
        atomic_json(result_path, result)
        print(json.dumps({k: result.get(k) for k in (
            "subset_index", "house", "status", "termination_detail", "elapsed_sec"
        )}, ensure_ascii=False), flush=True)
        if task is not None:
            task.close()
        if sampler is not None:
            sampler.close()
        del task, sampler, policy
        gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subset", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--limit", type=int, default=None,
                        help="仅调试；默认运行所选子集的全部 episode")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = build_inputs(args.p0_root, repo, args.subset)
    inputs_path = args.output / "inputs.json"
    if inputs_path.exists() and json.loads(inputs_path.read_text()) != inputs:
        raise ValueError("P1 输入或实现已改变；请保留旧目录并使用新的运行目录")
    atomic_json(inputs_path, inputs)

    episodes = json.loads((args.p0_root / args.subset / "benchmark.json").read_text())
    rows = [json.loads(line) for line in
            (args.p0_root / args.subset / "manifest.jsonl").read_text().splitlines()]
    expected_count = len(rows)
    limit = expected_count if args.limit is None else min(args.limit, expected_count)
    results = [run_episode(i, episode, row, inputs, args.output)
               for i, (episode, row) in enumerate(zip(episodes, rows)) if i < limit]
    jsonl(args.output / "episodes.jsonl", results)
    counts = {status: sum(r["status"] == status for r in results)
              for status in sorted(TERMINAL_STATUSES)}
    valid = counts["completed"]
    complete = len(results) == expected_count and all(r["status"] in TERMINAL_STATUSES for r in results)
    summary = {
        "experiment": PROTOCOL["name"],
        "subset": args.subset,
        "expected": expected_count,
        "attempted": len(results),
        "classified": len(results),
        "valid_A": valid,
        "status_counts": counts,
        "complete": complete,
        "p2_gate": "open" if complete and (args.subset == "formal" or valid >= 5) else "closed",
        "p2_started": False,
        "formal_simulation": "completed" if args.subset == "formal" and complete else "not_run",
    }
    atomic_json(args.output / "summary.json", summary)
    if complete:
        atomic_json(args.output / "COMPLETE.json", {
            "inputs_sha256": digest(inputs_path),
            "episodes_sha256": digest(args.output / "episodes.jsonl"),
            "summary_sha256": digest(args.output / "summary.json"),
        })
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
