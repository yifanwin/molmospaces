"""在 P1 的有效 A 快照上运行 P2 分层可行性评价器。"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np

from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec
from molmo_spaces.evaluation.eval_main import EvalRuntimeParams

from .build import atomic_json, digest, jsonl
from .config import P1Config
from .feasibility import FeasibilityBudget, ManipulationFeasibilityEvaluator, make_grasp_pool
from .sampler import P0JsonEvalTaskSampler
from .snapshot import EpisodeSnapshot
from .run_p1 import PROTOCOL as P1_PROTOCOL


PROTOCOL = {
    "name": "P2-E01",
    "source_p1": "P1-E03",
    "candidate_pool_limit": 32,
    "ik_seeds_per_arm": 3,
    "ik_max_iter": 300,
    "pregrasp_standoff_m": 0.04,
    "lift_height_m": 0.05,
    "tcp_translation_step_m": 0.01,
    "tcp_rotation_step_deg": 3.0,
    "boundary_translation_step_m": 0.005,
    "boundary_rotation_step_deg": 1.5,
    "torso_fixed": True,
    "base_fixed": True,
    "repeat_evaluations": 2,
    "curobo": "disabled",
    "llm": "disabled",
}


def implementation_hashes(repo: Path):
    paths = [
        repo / "molmo_spaces/evaluation/last_mile/feasibility.py",
        repo / "molmo_spaces/evaluation/last_mile/run_p2.py",
        repo / "molmo_spaces/evaluation/last_mile/snapshot.py",
        repo / "scripts/evaluation/run_last_mile_p2.sh",
        repo / "mlspaces_tests/evaluation/test_last_mile_p2.py",
    ]
    return {str(path.relative_to(repo)): digest(path) for path in paths if path.exists()}


def canonical(result: dict):
    value = deepcopy(result)
    value.pop("elapsed_sec", None)
    return value


def run_episode(index, episode, row, p1_result, p1_root: Path, output: Path,
                timeout_sec: float, repeat_evaluations: int):
    episode_dir = output / "episodes" / f"{index:03d}"
    result_path = episode_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        for name, expected in result.get("artifact_sha256", {}).items():
            if digest(episode_dir / name) != expected:
                raise ValueError(f"P2 缓存产物哈希不匹配：{episode_dir / name}")
        return result
    episode_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    task = sampler = None
    result = {
        "subset_index": index, "source_index": row["source_index"],
        "episode_id": row["episode_id"], "house": row["house"],
        "target": row["target"], "p1_status": p1_result["status"],
    }
    try:
        if p1_result["status"] != "completed":
            result.update(status="skipped", skip_reason="invalid_A")
            return result
        seed = row["seed"]
        random.seed(seed)
        np.random.seed(seed)
        if "torch" in sys.modules:
            sys.modules["torch"].manual_seed(seed)
        # P1 快照的 task-config 指纹包含 P1 输出目录，必须原样重建。
        exp = P1Config(seed=seed, output_dir=p1_root)
        exp.policy_config.path_min_dist_to_target_center = P1_PROTOCOL["path_min_dist_to_target_center_m"]
        exp.policy_config.plan_max_retries = P1_PROTOCOL["plan_max_retries"]
        exp.policy_config.plan_fail_after_waypoint_steps = P1_PROTOCOL["plan_fail_after_waypoint_steps"]
        exp.policy_config.plan_stick_to_original_target = True
        exp.policy_config.planner_config.agent_radius = P1_PROTOCOL["nav_agent_radius_m"]
        exp.eval_runtime_params = EvalRuntimeParams()
        exp.eval_runtime_params.repair_robot_base_pose_if_colliding = False
        spec = EpisodeSpec.model_validate(deepcopy(episode))
        spec.seed = seed
        sampler = P0JsonEvalTaskSampler(exp, spec, seed)
        task = sampler.sample_task(house_index=spec.house_index)
        if task is None:
            raise RuntimeError("初始化未返回任务")
        provenance = {
            "p1_inputs_sha256": digest(p1_root / "inputs.json"),
            "episode_id": row["episode_id"],
            "trajectory_points": int(np.load(
                p1_root / "episodes" / f"{index:03d}" / "trajectory.npz"
            )["actual_se2"].shape[0]),
        }
        snapshot_path = p1_root / "episodes" / f"{index:03d}" / "A_snapshot.npz"
        snapshot = EpisodeSnapshot.load(snapshot_path, provenance)
        snapshot.restore(task, provenance)
        evaluator = ManipulationFeasibilityEvaluator(task)
        target = evaluator._target(row["target"])
        pool = make_grasp_pool(task, target, max_candidates=PROTOCOL["candidate_pool_limit"])
        with open(episode_dir / "grasp_pool.npz.tmp", "wb") as stream:
            np.savez_compressed(stream, object_poses=pool.object_poses,
                                candidate_ids=np.asarray(pool.candidate_ids),
                                sha256=np.frombuffer(pool.sha256.encode(), dtype=np.uint8))
        os.replace(episode_dir / "grasp_pool.npz.tmp", episode_dir / "grasp_pool.npz")
        budget = FeasibilityBudget(
            max_candidates=PROTOCOL["candidate_pool_limit"],
            ik_seeds_per_arm=PROTOCOL["ik_seeds_per_arm"],
            timeout_sec=timeout_sec,
            pregrasp_standoff_m=PROTOCOL["pregrasp_standoff_m"],
            lift_height_m=PROTOCOL["lift_height_m"],
            tcp_translation_step_m=PROTOCOL["tcp_translation_step_m"],
            tcp_rotation_step_deg=PROTOCOL["tcp_rotation_step_deg"],
            boundary_translation_step_m=PROTOCOL["boundary_translation_step_m"],
            boundary_rotation_step_deg=PROTOCOL["boundary_rotation_step_deg"],
            ik_max_iter=PROTOCOL["ik_max_iter"],
        )
        evaluations = [
            evaluator.evaluate(snapshot, task.env.current_robot, p1_result["A"],
                               target, pool, budget)
            for _ in range(repeat_evaluations)
        ]
        repeatable = all(canonical(value) == canonical(evaluations[0])
                         for value in evaluations[1:])
        if not repeatable:
            raise RuntimeError("同状态同预算的 P2 结果不一致")
        result.update(evaluations[0])
        result["repeat_evaluations"] = repeat_evaluations
        result["repeatable"] = repeatable
        result["p1_A"] = p1_result["A"]
        result["artifact_sha256"] = {"grasp_pool.npz": digest(episode_dir / "grasp_pool.npz")}
        if any(name.startswith("curobo") for name in sys.modules):
            raise RuntimeError("P2 意外初始化/导入 CuRobo")
    except Exception as exc:
        result.update(status="unknown", unknown_reason="runner_error",
                      error=f"{type(exc).__name__}:{exc}", traceback=traceback.format_exc())
    finally:
        result["runner_elapsed_sec"] = time.monotonic() - started
        atomic_json(result_path, result)
        if task is not None:
            task.close()
        if sampler is not None:
            sampler.close()
        del task, sampler
        gc.collect()
    print(json.dumps({key: result.get(key) for key in (
        "subset_index", "house", "status", "first_failure_layer", "runner_elapsed_sec"
    )}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0-root", type=Path, required=True)
    parser.add_argument("--p1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--timeout-sec", type=float, default=300.0)
    parser.add_argument("--repeat-evaluations", type=int, default=2)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = {
        "schema_version": 1, "protocol": PROTOCOL,
        "p0_manifest_sha256": digest(args.p0_root / "manifest.jsonl"),
        "p1_inputs_sha256": digest(args.p1_root / "inputs.json"),
        "p1_complete_sha256": digest(args.p1_root / "COMPLETE.json"),
        "implementation_sha256": implementation_hashes(repo),
        "base_commit": os.popen(f"git -C {repo} rev-parse HEAD").read().strip(),
    }
    inputs_path = args.output / "inputs.json"
    if inputs_path.exists() and json.loads(inputs_path.read_text()) != inputs:
        raise ValueError("P2 输入或实现已改变；请保留旧目录并使用新的运行目录")
    atomic_json(inputs_path, inputs)
    episodes = json.loads((args.p0_root / "pilot/benchmark.json").read_text())
    rows = [json.loads(line) for line in
            (args.p0_root / "pilot/manifest.jsonl").read_text().splitlines()]
    p1_results = [json.loads(line) for line in
                  (args.p1_root / "episodes.jsonl").read_text().splitlines()]
    results = [
        run_episode(index, episode, row, p1_result, args.p1_root, args.output,
                    args.timeout_sec, args.repeat_evaluations)
        for index, (episode, row, p1_result) in enumerate(zip(episodes, rows, p1_results))
        if index < args.limit
    ]
    jsonl(args.output / "episodes.jsonl", results)
    valid_a = [result for result in results if result["p1_status"] == "completed"]
    evaluated = [result for result in valid_a if result["status"] in {"feasible", "not_found", "unknown"}]
    summary = {
        "experiment": PROTOCOL["name"], "attempted": len(results),
        "valid_A": len(valid_a), "evaluated_A": len(evaluated),
        "status_counts": {status: sum(r["status"] == status for r in evaluated)
                          for status in ("feasible", "not_found", "unknown")},
        "repeatable": bool(evaluated) and all(r.get("repeatable") for r in evaluated),
        "unit_acceptance_tests": "run_separately",
        "complete": len(results) == 10 and len(evaluated) == len(valid_a),
    }
    atomic_json(args.output / "summary.json", summary)
    if summary["complete"]:
        atomic_json(args.output / "COMPLETE.json", {
            "inputs_sha256": digest(inputs_path),
            "episodes_sha256": digest(args.output / "episodes.jsonl"),
            "summary_sha256": digest(args.output / "summary.json"),
        })
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
