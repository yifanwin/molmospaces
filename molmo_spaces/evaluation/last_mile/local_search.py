"""P3 局部 SE(2) 网格、走廊检查与公平预算基线。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json
import math
from typing import Callable, Iterable

import numpy as np


TRANSLATIONS_M = tuple(round(-0.3 + 0.1 * index, 10) for index in range(7))
YAWS_DEG = (-30, -15, 0, 15, 30)
MAIN_RADIUS_M = 0.30


def wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2 * math.pi) - math.pi)


def local_to_world(a: Iterable[float], local: Iterable[float]) -> np.ndarray:
    ax, ay, ayaw = np.asarray(a, dtype=float)
    dx, dy, dyaw = np.asarray(local, dtype=float)
    c, s = math.cos(ayaw), math.sin(ayaw)
    return np.asarray([ax + c * dx - s * dy, ay + s * dx + c * dy,
                       wrap_angle(ayaw + dyaw)])


def world_to_local(a: Iterable[float], b: Iterable[float]) -> np.ndarray:
    ax, ay, ayaw = np.asarray(a, dtype=float)
    bx, by, byaw = np.asarray(b, dtype=float)
    c, s = math.cos(ayaw), math.sin(ayaw)
    x, y = bx - ax, by - ay
    return np.asarray([c * x + s * y, -s * x + c * y, wrap_angle(byaw - ayaw)])


def generate_grid(a: Iterable[float]) -> list[dict]:
    rows = []
    fixed_id = 0
    for dx in TRANSLATIONS_M:
        for dy in TRANSLATIONS_M:
            radius = math.hypot(dx, dy)
            for yaw_deg in YAWS_DEG:
                local = [dx, dy, math.radians(yaw_deg)]
                rows.append({
                    "point_id": f"G{fixed_id:03d}", "fixed_id": fixed_id,
                    "local_pose": local, "world_pose": local_to_world(a, local).tolist(),
                    "translation_m": radius, "abs_yaw_deg": abs(yaw_deg),
                    "in_main_disk": radius <= MAIN_RADIUS_M + 1e-12,
                    "is_A": dx == 0 and dy == 0 and yaw_deg == 0,
                })
                fixed_id += 1
    assert len(rows) == 245
    assert sum(row["in_main_disk"] for row in rows) == 145
    assert sum(row["is_A"] for row in rows) == 1
    return rows


def interpolate_se2(start: Iterable[float], goal: Iterable[float],
                    translation_step_m=0.01, rotation_step_deg=3.0) -> list[np.ndarray]:
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    delta_xy = goal[:2] - start[:2]
    delta_yaw = wrap_angle(goal[2] - start[2])
    count = max(1, math.ceil(np.linalg.norm(delta_xy) / translation_step_m),
                math.ceil(abs(delta_yaw) / math.radians(rotation_step_deg)))
    return [np.asarray([*(start[:2] + alpha * delta_xy),
                        wrap_angle(start[2] + alpha * delta_yaw)])
            for alpha in np.linspace(0.0, 1.0, count + 1)[1:]]


@dataclass(frozen=True)
class CorridorConfig:
    translation_step_m: float = 0.01
    rotation_step_deg: float = 3.0
    search_xy_step_m: float = 0.05
    search_yaw_step_deg: float = 15.0
    search_radius_m: float = 0.45
    search_yaw_limit_deg: float = 30.0
    search_path_length_limit_m: float = 0.75
    search_max_expansions: int = 250


def _node_pose(node, cfg):
    return np.asarray([node[0] * cfg.search_xy_step_m,
                       node[1] * cfg.search_xy_step_m,
                       math.radians(node[2] * cfg.search_yaw_step_deg)])


def corridor_reachability(goal_local: Iterable[float],
                          collision_free: Callable[[list[np.ndarray]], bool],
                          cfg: CorridorConfig = CorridorConfig()) -> dict:
    """先检查直接路径，再在冻结邻域内做确定性 A*。"""
    start, goal = np.zeros(3), np.asarray(goal_local, dtype=float)
    direct = interpolate_se2(start, goal, cfg.translation_step_m, cfg.rotation_step_deg)
    if collision_free(direct):
        return {"reachable": True, "method": "direct", "expansions": 0,
                "path_local": [start.tolist()] + [pose.tolist() for pose in direct]}
    sy, sa = cfg.search_xy_step_m, math.radians(cfg.search_yaw_step_deg)
    source, target = (0, 0, 0), (round(goal[0] / sy), round(goal[1] / sy),
                                  round(goal[2] / sa))
    moves = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
             (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
             (0, 0, 1), (0, 0, -1))

    def h(node):
        pose = _node_pose(node, cfg)
        return float(np.linalg.norm(pose[:2] - goal[:2]) + .02 * abs(node[2] - target[2]))

    queue, best, parent = [(h(source), 0.0, source)], {source: 0.0}, {}
    expansions = 0
    while queue and expansions < cfg.search_max_expansions:
        _, cost, node = heapq.heappop(queue)
        if cost != best.get(node):
            continue
        expansions += 1
        if node == target:
            tail = interpolate_se2(_node_pose(node, cfg), goal,
                                   cfg.translation_step_m, cfg.rotation_step_deg)
            if not collision_free(tail):
                # 精确目标可以不在搜索格点上；目标边失败时继续搜索其他邻点。
                pass
            else:
                chain = [node]
                while chain[-1] != source:
                    chain.append(parent[chain[-1]])
                chain.reverse()
                path = [_node_pose(item, cfg).tolist() for item in chain]
                path.extend(pose.tolist() for pose in tail)
                return {"reachable": True, "method": "local_astar", "expansions": expansions,
                        "path_local": path}
        # 对非格点目标，也允许相邻格点用一条精确采样边连接目标。
        if node != target and np.linalg.norm(_node_pose(node, cfg)[:2] - goal[:2]) <= sy * 1.5:
            tail = interpolate_se2(_node_pose(node, cfg), goal,
                                   cfg.translation_step_m, cfg.rotation_step_deg)
            if collision_free(tail):
                chain = [node]
                while chain[-1] != source:
                    chain.append(parent[chain[-1]])
                chain.reverse()
                path = [_node_pose(item, cfg).tolist() for item in chain]
                path.extend(pose.tolist() for pose in tail)
                return {"reachable": True, "method": "local_astar", "expansions": expansions,
                        "path_local": path}
        pose = _node_pose(node, cfg)
        for move in moves:
            nxt = tuple(node[i] + move[i] for i in range(3))
            nxt_pose = _node_pose(nxt, cfg)
            if (np.linalg.norm(nxt_pose[:2]) > cfg.search_radius_m + 1e-12 or
                    abs(nxt_pose[2]) > math.radians(cfg.search_yaw_limit_deg) + 1e-12):
                continue
            new_cost = cost + float(np.linalg.norm(nxt_pose[:2] - pose[:2]))
            if (new_cost > cfg.search_path_length_limit_m + 1e-12 or
                    new_cost >= best.get(nxt, math.inf) - 1e-12):
                continue
            edge = interpolate_se2(pose, nxt_pose, cfg.translation_step_m,
                                   cfg.rotation_step_deg)
            if not collision_free(edge):
                continue
            best[nxt], parent[nxt] = new_cost, node
            heapq.heappush(queue, (new_cost + h(nxt), new_cost, nxt))
    return {"reachable": False, "method": "local_astar_exhausted",
            "expansions": expansions, "path_local": None}


def nearest(rows: Iterable[dict], require_reachable=False) -> dict | None:
    eligible = [row for row in rows if row.get("status") == "feasible" and
                (not require_reachable or row.get("reachable") is True)]
    return min(eligible, key=lambda row: (row["translation_m"], row["abs_yaw_deg"],
                                          row["fixed_id"])) if eligible else None


def heuristic_pose(a: Iterable[float], target_xy: Iterable[float], distance_m: float) -> dict:
    a, target_xy = np.asarray(a, dtype=float), np.asarray(target_xy, dtype=float)[:2]
    radial = a[:2] - target_xy
    norm = float(np.linalg.norm(radial))
    if norm < 1e-12:
        return {"valid": False, "reason": "A_at_target_xy"}
    xy = target_xy + distance_m * radial / norm
    world = [xy[0], xy[1], math.atan2(target_xy[1] - xy[1], target_xy[0] - xy[0])]
    local = world_to_local(a, world)
    valid = (np.linalg.norm(local[:2]) <= MAIN_RADIUS_M + 1e-12 and
             abs(local[2]) <= math.radians(30) + 1e-12)
    return {"valid": bool(valid), "reason": None if valid else "outside_local_limits",
            "world_pose": list(map(float, world)), "local_pose": local.tolist(),
            "translation_m": float(np.linalg.norm(local[:2])),
            "abs_yaw_deg": float(abs(math.degrees(local[2])))}


def derived_seed(episode_seed: int, seed_id: int) -> int:
    raw = json.dumps([int(episode_seed), "p3-random-local", int(seed_id)],
                     separators=(",", ":")).encode()
    return int(hashlib.sha256(raw).hexdigest()[:16], 16)


def random_budget_results(rows: list[dict], episode_seed: int,
                          budgets=(1, 4, 8, 16), seeds=10) -> list[dict]:
    candidates = [row for row in rows if row["in_main_disk"] and not row["is_A"]]
    assert len(candidates) == 144
    output = []
    for seed_id in range(seeds):
        order = np.random.default_rng(derived_seed(episode_seed, seed_id)).permutation(144)
        for budget in budgets:
            queried = [candidates[index] for index in order[:budget]]
            hit = next((row for row in queried if row.get("status") == "feasible" and
                        row.get("reachable") is True), None)
            upper = hit is not None or any(row.get("status") == "unknown" and
                                           row.get("reachable") is True for row in queried)
            output.append({"seed_id": seed_id, "seed": derived_seed(episode_seed, seed_id),
                           "budget": budget, "queries_charged": queried.index(hit) + 1 if hit else budget,
                           "rescued": hit is not None, "rescue_upper": upper,
                           "first_hit_point_id": hit and hit["point_id"],
                           "query_order": [row["point_id"] for row in queried]})
    return output
