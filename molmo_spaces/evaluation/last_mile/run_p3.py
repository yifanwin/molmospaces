"""运行 P3：冻结试点的局部站位地图、走廊和简单预算基线。"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import gc
import hashlib
import json
import multiprocessing
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
from .feasibility import FeasibilityBudget, GraspPool, ManipulationFeasibilityEvaluator
from .local_search import (
    CorridorConfig, corridor_reachability, generate_grid, heuristic_pose,
    local_to_world, nearest, random_budget_results,
)
from .run_p1 import PROTOCOL as P1_PROTOCOL
from .sampler import P0JsonEvalTaskSampler
from .snapshot import EpisodeSnapshot


PROTOCOL = {
    "name": "P3-E01",
    "subset": "pilot",
    "source_p1_dir": "P1-E04",
    "snapshot_compatibility_rerun": "P1-E05_compat",
    "source_p2_dir": "P2-E02",
    "square_points": 245,
    "main_disk_points": 145,
    "additional_grid_queries_per_valid_A": 244,
    "random_budgets": [1, 4, 8, 16],
    "random_seeds_per_episode": 10,
    "heuristic_distances_m": [0.55, 0.65, 0.75],
    "heuristic_tie_order_m": [0.65, 0.55, 0.75],
    "pilot_expected_reachable_rescues": [1, 3],
    "pilot_go_signal_min_rescues": 1,
    "real_pick": "not_run",
    "formal_100": "not_run",
    "curobo": "disabled",
    "llm": "disabled",
}
CORRIDOR = CorridorConfig()


def implementation_hashes(repo: Path) -> dict[str, str]:
    paths = [
        repo / "molmo_spaces/evaluation/last_mile/feasibility.py",
        repo / "molmo_spaces/evaluation/last_mile/local_search.py",
        repo / "molmo_spaces/evaluation/last_mile/run_p3.py",
        repo / "scripts/evaluation/run_last_mile_p3.sh",
        repo / "scripts/evaluation/plot_last_mile_p3.py",
        repo / "mlspaces_tests/evaluation/test_last_mile_p3.py",
    ]
    return {str(path.relative_to(repo)): digest(path) for path in paths if path.exists()}


def _cache_key(inputs_sha: str, index: int, definition: dict) -> str:
    raw = json.dumps([inputs_sha, index, definition], sort_keys=True,
                     separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _load_pool(path: Path, expected_sha: str) -> GraspPool:
    with np.load(path, allow_pickle=False) as values:
        pool = GraspPool(values["object_poses"].copy(),
                         tuple(int(value) for value in values["candidate_ids"]),
                         values["sha256"].tobytes().decode())
    if pool.sha256 != expected_sha:
        raise ValueError("P2 grasp pool 内容哈希与结果不一致")
    return pool


def _slim_evaluation(value: dict) -> dict:
    keep = ("status", "first_failure_layer", "layer_pass_counts", "base_collisions",
            "approach_only_witness", "witness", "elapsed_sec", "unknown_reason", "error",
            "grasp_pool_sha256", "grasp_pool_size", "budget", "search_semantics")
    return {key: value[key] for key in keep if key in value}


def _build_task(index, episode, row, p1_root: Path):
    seed = row["seed"]
    random.seed(seed)
    np.random.seed(seed)
    if "torch" in sys.modules:
        sys.modules["torch"].manual_seed(seed)
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
        sampler.close()
        raise RuntimeError("初始化未返回任务")
    provenance = {
        "p1_inputs_sha256": digest(p1_root / "inputs.json"),
        "episode_id": row["episode_id"],
        "trajectory_points": int(np.load(
            p1_root / "episodes" / f"{index:03d}" / "trajectory.npz"
        )["actual_se2"].shape[0]),
    }
    snapshot = EpisodeSnapshot.load(
        p1_root / "episodes" / f"{index:03d}" / "A_snapshot.npz", provenance
    )
    snapshot.restore(task, provenance)
    return task, sampler, snapshot


def _evaluate_pose(evaluator, snapshot, target, pool, budget, world_pose):
    robot = evaluator.task.env.current_robot
    base = evaluator.check_base_poses(snapshot, robot, [world_pose], target)
    if not base["collision_free"]:
        return {
            "status": "not_found", "first_failure_layer": "F_base",
            "layer_pass_counts": {"F_base": 0, "F_IK": 0, "F_approach": 0,
                                  "F_lift_proxy": 0},
            "base_collisions": base["collisions"], "witness": None,
            "elapsed_sec": 0.0, "grasp_pool_sha256": pool.sha256,
            "grasp_pool_size": len(pool.candidate_ids), "budget": budget.__dict__,
            "search_semantics": "not_found means no chain found under this finite protocol",
        }
    return evaluator.evaluate(snapshot, robot, world_pose, target, pool, budget)


def _add_corridor(row, evaluator, snapshot, target, a, cache):
    if row["status"] not in {"feasible", "unknown"} or \
            row.get("layer_pass_counts", {}).get("F_base", 0) == 0:
        row.update(reachable=False, corridor=None)
        return
    checked = 0

    def collision_free(local_poses):
        nonlocal checked
        for offset, local in enumerate(local_poses):
            key = tuple(np.asarray(local).round(9))
            if cache.get(key) is False:
                return False
            if cache.get(key) is True:
                continue
            suffix = local_poses[offset:]
            worlds = [local_to_world(a, pose) for pose in suffix]
            result = evaluator.check_base_poses(
                snapshot, evaluator.task.env.current_robot, worlds, target
            )
            checked += result["checked_poses"]
            stop = result["first_collision_index"]
            for pos, sample in enumerate(suffix[:stop if stop is not None else len(suffix)]):
                cache[tuple(np.asarray(sample).round(9))] = True
            if stop is not None:
                cache[tuple(np.asarray(suffix[stop]).round(9))] = False
                return False
            return True
        return True

    corridor = corridor_reachability(row["local_pose"], collision_free, CORRIDOR)
    corridor["collision_checks"] = checked
    row["corridor"] = corridor
    row["reachable"] = bool(corridor["reachable"])


def _run_valid_episode(index, episode, row, p1_result, p2_result, roots, output,
                       inputs_sha, max_b_points=None):
    episode_dir = output / "episodes" / f"{index:03d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    result_path = episode_dir / "result.json"
    if result_path.exists() and max_b_points is None:
        result = json.loads(result_path.read_text())
        if result.get("inputs_sha256") != inputs_sha:
            raise ValueError(f"P3 episode {index} 缓存输入哈希不匹配")
        if digest(episode_dir / "candidates.jsonl") != result["artifact_sha256"]["candidates.jsonl"]:
            raise ValueError(f"P3 episode {index} candidates 缓存哈希不匹配")
        return result
    task = sampler = None
    started = time.monotonic()
    try:
        task, sampler, snapshot = _build_task(index, episode, row, roots["p1"])
        evaluator = ManipulationFeasibilityEvaluator(task)
        target = evaluator._target(row["target"])
        pool_path = roots["p2"] / "episodes" / f"{index:03d}" / "grasp_pool.npz"
        if digest(pool_path) != p2_result["artifact_sha256"]["grasp_pool.npz"]:
            raise ValueError("P2 grasp_pool 文件哈希不匹配")
        pool = _load_pool(pool_path, p2_result["grasp_pool_sha256"])
        budget = FeasibilityBudget(**p2_result["budget"])
        a = np.asarray(p1_result["A"], dtype=float)
        grid = generate_grid(a)
        point_dir = episode_dir / "points"
        point_dir.mkdir(exist_ok=True)
        candidates, b_done = [], 0
        corridor_cache = {(0.0, 0.0, 0.0): True}
        for definition in grid:
            definition = deepcopy(definition)
            key = _cache_key(inputs_sha, index, definition)
            point_path = point_dir / f"{definition['point_id']}.json"
            if definition["is_A"]:
                value = _slim_evaluation(p2_result)
                value.update(definition, cached_A=True, query_charged=0,
                             reachable=value["status"] == "feasible",
                             corridor={"reachable": True, "method": "identity",
                                       "expansions": 0, "path_local": [[0, 0, 0]]},
                             cache_key=key)
                candidates.append(value)
                continue
            if max_b_points is not None and b_done >= max_b_points:
                continue
            b_done += 1
            if point_path.exists():
                value = json.loads(point_path.read_text())
                if value.get("cache_key") != key:
                    raise ValueError(f"P3 点缓存定义改变：{point_path}")
            else:
                value = _slim_evaluation(_evaluate_pose(
                    evaluator, snapshot, target, pool, budget, definition["world_pose"]
                ))
                value.update(definition, cached_A=False, query_charged=1, cache_key=key)
                _add_corridor(value, evaluator, snapshot, target, a, corridor_cache)
                atomic_json(point_path, value)
            candidates.append(value)
            print(json.dumps({"episode": index, "point": definition["point_id"],
                              "status": value["status"], "reachable": value.get("reachable"),
                              "elapsed_sec": value.get("elapsed_sec")}, ensure_ascii=False), flush=True)
        if max_b_points is not None:
            jsonl(episode_dir / "candidates.jsonl", candidates)
            return {"subset_index": index, "status": "smoke_complete",
                    "points": len(candidates), "runner_elapsed_sec": time.monotonic() - started}

        # 三个预登记距离均是试点校准查询，不暗中使用全图选点。
        with snapshot.restored(task):
            target_xy = target.pose[:2, 3].copy()
        heuristic_rows = []
        for h_index, distance in enumerate(PROTOCOL["heuristic_distances_m"]):
            definition = heuristic_pose(a, target_xy, distance)
            value = {"heuristic_distance_m": distance, "point_id": f"H{h_index}",
                     "fixed_id": 1000 + h_index, "in_main_disk": True, "is_A": False,
                     **definition}
            if not definition["valid"]:
                value.update(status="not_found", first_failure_layer="constraint",
                             reachable=False, query_charged=1)
            else:
                value.update(_slim_evaluation(_evaluate_pose(
                    evaluator, snapshot, target, pool, budget, definition["world_pose"]
                )), query_charged=1)
                _add_corridor(value, evaluator, snapshot, target, a, corridor_cache)
            heuristic_rows.append(value)
        jsonl(episode_dir / "candidates.jsonl", candidates)
        jsonl(episode_dir / "heuristics.jsonl", heuristic_rows)
        main = [value for value in candidates if value["in_main_disk"]]
        nearest_geo, nearest_reach = nearest(main), nearest(main, True)
        random_rows = random_budget_results(main, row["seed"],
                                            PROTOCOL["random_budgets"],
                                            PROTOCOL["random_seeds_per_episode"])
        result = {
            "schema_version": 1, "experiment": PROTOCOL["name"],
            "subset_index": index, "source_index": row["source_index"],
            "episode_id": row["episode_id"], "house": row["house"],
            "target": row["target"], "seed": row["seed"],
            "p1_status": p1_result["status"], "A": p1_result["A"],
            "A_status": p2_result["status"], "status": "completed",
            "grid_points": len(candidates), "main_disk_points": len(main),
            "additional_grid_queries": sum(value["query_charged"] for value in candidates),
            "status_counts_square": {s: sum(value["status"] == s for value in candidates)
                                      for s in ("feasible", "not_found", "unknown")},
            "status_counts_disk": {s: sum(value["status"] == s for value in main)
                                    for s in ("feasible", "not_found", "unknown")},
            "nearest_feasible": nearest_geo and nearest_geo["point_id"],
            "nearest_reachable_feasible": nearest_reach and nearest_reach["point_id"],
            "geo_rescue": nearest_geo is not None,
            "reachable_rescue": nearest_reach is not None,
            "reachable_unknown": any(value["status"] == "unknown" and value.get("reachable")
                                     for value in main),
            "random_local": random_rows, "heuristics": heuristic_rows,
            "grasp_pool_sha256": pool.sha256, "budget": p2_result["budget"],
            "inputs_sha256": inputs_sha, "runner_elapsed_sec": time.monotonic() - started,
        }
        result["artifact_sha256"] = {
            "candidates.jsonl": digest(episode_dir / "candidates.jsonl"),
            "heuristics.jsonl": digest(episode_dir / "heuristics.jsonl"),
        }
        atomic_json(result_path, result)
        return result
    except Exception as exc:
        error = {"subset_index": index, "source_index": row["source_index"],
                 "episode_id": row["episode_id"], "house": row["house"],
                 "target": row["target"], "p1_status": p1_result["status"],
                 "status": "unknown", "unknown_reason": "runner_error",
                 "error": f"{type(exc).__name__}:{exc}", "traceback": traceback.format_exc(),
                 "inputs_sha256": inputs_sha, "runner_elapsed_sec": time.monotonic() - started}
        atomic_json(result_path, error)
        return error
    finally:
        if task is not None:
            task.close()
        if sampler is not None:
            sampler.close()
        del task, sampler
        gc.collect()


def run_episode_job(payload):
    index, episode, row, p1, p2, roots, output, inputs_sha, max_b = payload
    roots = {key: Path(value) for key, value in roots.items()}
    return _run_valid_episode(index, episode, row, p1, p2, roots, Path(output),
                              inputs_sha, max_b)


def run_shard_job(payload):
    """一个进程复用一个 episode/task，计算互不重叠的一组 B 点。"""
    (index, episode, row, p1_result, p2_result, roots, output, inputs_sha,
     shard_id, shard_count) = payload
    roots = {key: Path(value) for key, value in roots.items()}
    output = Path(output)
    episode_dir = output / "episodes" / f"{index:03d}"
    point_dir = episode_dir / "points"
    point_dir.mkdir(parents=True, exist_ok=True)
    task = sampler = None
    started = time.monotonic()
    try:
        task, sampler, snapshot = _build_task(index, episode, row, roots["p1"])
        evaluator = ManipulationFeasibilityEvaluator(task)
        target = evaluator._target(row["target"])
        pool_path = roots["p2"] / "episodes" / f"{index:03d}" / "grasp_pool.npz"
        if digest(pool_path) != p2_result["artifact_sha256"]["grasp_pool.npz"]:
            raise ValueError("P2 grasp_pool 文件哈希不匹配")
        pool = _load_pool(pool_path, p2_result["grasp_pool_sha256"])
        budget = FeasibilityBudget(**p2_result["budget"])
        a = np.asarray(p1_result["A"], dtype=float)
        selected = [definition for definition in generate_grid(a)
                    if not definition["is_A"] and definition["fixed_id"] % shard_count == shard_id]
        corridor_cache = {(0.0, 0.0, 0.0): True}
        evaluated = reused = 0
        for definition in selected:
            definition = deepcopy(definition)
            key = _cache_key(inputs_sha, index, definition)
            point_path = point_dir / f"{definition['point_id']}.json"
            if point_path.exists():
                value = json.loads(point_path.read_text())
                if value.get("cache_key") != key:
                    raise ValueError(f"P3 点缓存定义改变：{point_path}")
                reused += 1
                continue
            value = _slim_evaluation(_evaluate_pose(
                evaluator, snapshot, target, pool, budget, definition["world_pose"]
            ))
            value.update(definition, cached_A=False, query_charged=1, cache_key=key)
            _add_corridor(value, evaluator, snapshot, target, a, corridor_cache)
            atomic_json(point_path, value)
            evaluated += 1
            print(json.dumps({"episode": index, "shard": shard_id,
                              "point": definition["point_id"], "status": value["status"],
                              "elapsed_sec": value.get("elapsed_sec")}, ensure_ascii=False), flush=True)
        return {"subset_index": index, "shard": shard_id, "status": "completed",
                "assigned": len(selected), "evaluated": evaluated, "reused": reused,
                "elapsed_sec": time.monotonic() - started}
    finally:
        if task is not None:
            task.close()
        if sampler is not None:
            sampler.close()
        del task, sampler
        gc.collect()


def _skip_result(index, row, p1_result, p2_result, inputs_sha):
    return {"schema_version": 1, "experiment": PROTOCOL["name"],
            "subset_index": index, "source_index": row["source_index"],
            "episode_id": row["episode_id"], "house": row["house"],
            "target": row["target"], "p1_status": p1_result["status"],
            "p2_status": p2_result["status"], "status": "skipped",
            "skip_reason": "invalid_A", "inputs_sha256": inputs_sha}


def aggregate(results, output: Path, inputs_path: Path):
    results = sorted(results, key=lambda value: value["subset_index"])
    valid = [value for value in results if value["p1_status"] == "completed"]
    completed = [value for value in valid if value["status"] == "completed"]
    all_candidates = []
    for value in completed:
        path = output / "episodes" / f"{value['subset_index']:03d}" / "candidates.jsonl"
        all_candidates.extend(json.loads(line) | {"subset_index": value["subset_index"]}
                              for line in path.read_text().splitlines())
    jsonl(output / "episodes.jsonl", results)
    jsonl(output / "candidates.jsonl", all_candidates)

    # 试点选择 fixed-radius，先按可达数、再按可行数，最后采用冻结偏好次序。
    heuristic_scores = {}
    for distance in PROTOCOL["heuristic_distances_m"]:
        rows = [row for value in completed for row in value["heuristics"]
                if row["heuristic_distance_m"] == distance]
        heuristic_scores[str(distance)] = {
            "episodes": len(rows),
            "reachable_feasible": sum(row.get("status") == "feasible" and row.get("reachable")
                                      for row in rows),
            "feasible": sum(row.get("status") == "feasible" for row in rows),
            "unknown_reachable": sum(row.get("status") == "unknown" and row.get("reachable")
                                     for row in rows),
        }
    tie_rank = {value: -index for index, value in enumerate(PROTOCOL["heuristic_tie_order_m"])}
    selected_distance = max(PROTOCOL["heuristic_distances_m"], key=lambda distance: (
        heuristic_scores[str(distance)]["reachable_feasible"],
        heuristic_scores[str(distance)]["feasible"], tie_rank[distance]
    )) if completed else None

    random_metrics = {}
    for budget in PROTOCOL["random_budgets"]:
        rows = [row for value in completed for row in value["random_local"]
                if row["budget"] == budget]
        random_metrics[str(budget)] = {
            "trials": len(rows), "rescued": sum(row["rescued"] for row in rows),
            "rescue_upper": sum(row["rescue_upper"] for row in rows),
            "rate": sum(row["rescued"] for row in rows) / len(rows) if rows else None,
            "rate_upper": sum(row["rescue_upper"] for row in rows) / len(rows) if rows else None,
        }
    m = sum(value["A_status"] == "not_found" for value in completed)
    l_geo = sum(value["A_status"] == "not_found" and value["geo_rescue"] for value in completed)
    l_reach = sum(value["A_status"] == "not_found" and value["reachable_rescue"] for value in completed)
    upper = l_reach + sum(value["A_status"] == "not_found" and
                          not value["reachable_rescue"] and value["reachable_unknown"]
                          for value in completed)
    metrics = {
        "schema_version": 1, "experiment": PROTOCOL["name"],
        "N_attempt": len(results), "N_nav": len(valid),
        "N_eval": sum(not value["reachable_unknown"] for value in completed),
        "N_scan_complete": len(completed), "M_A_not_found": m,
        "L_geo": l_geo, "L_reach": l_reach, "L_reach_upper": upper,
        "I_geo": l_geo / len(completed) if completed else None,
        "I_reach": l_reach / len(completed) if completed else None,
        "I_reach_upper": upper / len(completed) if completed else None,
        "R_oracle": l_reach / m if m else None,
        "R_oracle_upper": upper / m if m else None,
        "nav_only_rescues": 0,
        "random_local": random_metrics,
        "heuristic_calibration": heuristic_scores,
        "selected_heuristic_distance_m": selected_distance,
        "pilot_go_signal": l_reach >= PROTOCOL["pilot_go_signal_min_rescues"],
        "formal_100": "not_run", "real_pick": "not_run",
        "inference": "descriptive_only_five_valid_A",
    }
    atomic_json(output / "metrics.json", metrics)
    complete = (len(results) == 10 and len(valid) == 5 and len(completed) == 5 and
                all(value["grid_points"] == 245 and value["main_disk_points"] == 145 and
                    value["additional_grid_queries"] == 244 for value in completed))
    summary = {"experiment": PROTOCOL["name"], "attempted": len(results),
               "valid_A": len(valid), "scan_completed": len(completed),
               "skipped_invalid_A": sum(value["status"] == "skipped" for value in results),
               "complete": complete, "metrics": metrics}
    atomic_json(output / "summary.json", summary)
    if complete:
        atomic_json(output / "COMPLETE.json", {
            "inputs_sha256": digest(inputs_path),
            "episodes_sha256": digest(output / "episodes.jsonl"),
            "candidates_sha256": digest(output / "candidates.jsonl"),
            "metrics_sha256": digest(output / "metrics.json"),
        })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0-root", type=Path, required=True)
    parser.add_argument("--p0-validation", type=Path, required=True)
    parser.add_argument("--p1-root", type=Path, required=True)
    parser.add_argument("--p1-reference", type=Path, required=True,
                        help="冻结语义来源 P1-E04；p1-root 是模型兼容且 A 完全一致的快照重放")
    parser.add_argument("--p2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shards-per-episode", type=int, default=1,
                        help="正式扫描的 episode 内互斥分片数；每个分片只初始化一次场景")
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--max-b-points", type=int, default=None,
                        help="仅供独立 smoke 输出目录；正式运行必须省略")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.max_b_points is not None and args.episode_index is None:
        raise ValueError("--max-b-points 必须同时指定单个 --episode-index")
    inputs = {
        "schema_version": 1, "protocol": PROTOCOL,
        "corridor": CORRIDOR.__dict__,
        "p0_validation_complete_sha256": digest(args.p0_validation / "COMPLETE.json"),
        "p0_validation_inputs_sha256": digest(args.p0_validation / "inputs.json"),
        "p0_manifest_sha256": digest(args.p0_root / "pilot/manifest.jsonl"),
        "p0_benchmark_sha256": digest(args.p0_root / "pilot/benchmark.json"),
        "p1_inputs_sha256": digest(args.p1_root / "inputs.json"),
        "p1_complete_sha256": digest(args.p1_root / "COMPLETE.json"),
        "p1_reference_inputs_sha256": digest(args.p1_reference / "inputs.json"),
        "p1_reference_complete_sha256": digest(args.p1_reference / "COMPLETE.json"),
        "p2_inputs_sha256": digest(args.p2_root / "inputs.json"),
        "p2_complete_sha256": digest(args.p2_root / "COMPLETE.json"),
        "implementation_sha256": implementation_hashes(repo),
        "base_commit": os.popen(f"git -C {repo} rev-parse HEAD").read().strip(),
    }
    inputs_path = args.output / "inputs.json"
    if inputs_path.exists() and json.loads(inputs_path.read_text()) != inputs:
        raise ValueError("P3 输入或实现已改变；请保留旧目录并使用新运行目录")
    atomic_json(inputs_path, inputs)
    inputs_sha = digest(inputs_path)
    atomic_json(args.output / "provenance.json", {
        "worktree": str(repo), "base_commit": inputs["base_commit"],
        "input_hashes": {key: value for key, value in inputs.items() if key.endswith("sha256")},
        "note": ("历史 E04/E02 summary 内部实验名仍为 E03/E01；以目录与哈希追溯。"
                 "E04 快照模型指纹不能由当前已提交模型重建，因此用相同输入重放 E05_compat；"
                 "运行器逐位断言五个有效 A 与 E04 完全一致，不迁移或绕过快照模型指纹。"),
    })
    episodes = json.loads((args.p0_root / "pilot/benchmark.json").read_text())
    rows = [json.loads(line) for line in
            (args.p0_root / "pilot/manifest.jsonl").read_text().splitlines()]
    p1 = [json.loads(line) for line in (args.p1_root / "episodes.jsonl").read_text().splitlines()]
    p1_reference = [json.loads(line) for line in
                    (args.p1_reference / "episodes.jsonl").read_text().splitlines()]
    p2 = [json.loads(line) for line in (args.p2_root / "episodes.jsonl").read_text().splitlines()]
    if not (len(episodes) == len(rows) == len(p1) == len(p1_reference) == len(p2) == 10):
        raise ValueError("P0/P1/P2 试点数量不是同一组 10 条")
    valid_indexes = [index for index, value in enumerate(p1) if value["status"] == "completed"]
    if valid_indexes != [2, 3, 5, 6, 8]:
        raise ValueError(f"有效 A 索引改变：{valid_indexes}")
    reference_valid = [index for index, value in enumerate(p1_reference)
                       if value["status"] == "completed"]
    if reference_valid != valid_indexes:
        raise ValueError("P1 兼容重放与冻结 E04 的有效 A 集合不一致")
    for index in valid_indexes:
        if not np.array_equal(np.asarray(p1[index]["A"]),
                              np.asarray(p1_reference[index]["A"])):
            raise ValueError(f"P1 兼容重放的 A 与 E04 不完全一致：{index}")
    budgets = [p2[index]["budget"] for index in valid_indexes]
    if any(value != budgets[0] for value in budgets[1:]):
        raise ValueError("P2 有效 A 使用了不同预算，不能公平扫描")
    selected = valid_indexes if args.episode_index is None else [args.episode_index]
    if any(index not in valid_indexes for index in selected):
        raise ValueError("P3 只能扫描有效 A")
    roots = {"p1": str(args.p1_root), "p2": str(args.p2_root)}
    payloads = [(index, episodes[index], rows[index], p1[index], p2[index], roots,
                 str(args.output), inputs_sha, args.max_b_points) for index in selected]
    scanned = []
    if args.shards_per_episode > 1 and args.max_b_points is None and args.episode_index is None:
        shard_payloads = [
            (index, episodes[index], rows[index], p1[index], p2[index], roots,
             str(args.output), inputs_sha, shard_id, args.shards_per_episode)
            for index in selected for shard_id in range(args.shards_per_episode)
        ]
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(args.workers, len(shard_payloads)),
                                 mp_context=ctx) as pool:
            futures = {pool.submit(run_shard_job, payload): (payload[0], payload[-2])
                       for payload in shard_payloads}
            for future in as_completed(futures):
                result = future.result()
                if result["status"] != "completed":
                    raise RuntimeError(f"P3 分片失败：{result}")
                print(json.dumps(result, ensure_ascii=False), flush=True)
        # 所有互斥点文件完成后，再按 episode 聚合并运行三个 heuristic 查询。
        with ProcessPoolExecutor(max_workers=min(5, args.workers), mp_context=ctx) as pool:
            futures = [pool.submit(run_episode_job, payload) for payload in payloads]
            scanned = [future.result() for future in as_completed(futures)]
    elif args.workers == 1 or len(payloads) == 1:
        scanned = [run_episode_job(payload) for payload in payloads]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(args.workers, len(payloads)),
                                 mp_context=ctx) as pool:
            futures = {pool.submit(run_episode_job, payload): payload[0] for payload in payloads}
            for future in as_completed(futures):
                result = future.result()
                scanned.append(result)
                print(json.dumps({"episode": futures[future], "status": result["status"]},
                                 ensure_ascii=False), flush=True)
    if args.max_b_points is not None or args.episode_index is not None:
        print(json.dumps(scanned, ensure_ascii=False), flush=True)
        return 0 if all(value["status"] in {"completed", "smoke_complete"} for value in scanned) else 1
    by_index = {value["subset_index"]: value for value in scanned}
    results = [by_index[index] if index in by_index else
               _skip_result(index, rows[index], p1[index], p2[index], inputs_sha)
               for index in range(10)]
    summary = aggregate(results, args.output, inputs_path)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
