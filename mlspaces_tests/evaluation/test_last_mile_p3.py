"""P3 的无资产回归测试。"""
import math
import numpy as np
from molmo_spaces.evaluation.last_mile.local_search import (
    CorridorConfig, corridor_reachability, generate_grid, heuristic_pose,
    local_to_world, nearest, random_budget_results, world_to_local,
)


def test_grid_counts_and_transform():
    a = [2, -1, math.pi / 2]
    rows = generate_grid(a)
    assert len(rows) == 245 and sum(r["in_main_disk"] for r in rows) == 145
    assert sum(r["is_A"] for r in rows) == 1
    local = [.2, -.1, math.radians(15)]
    np.testing.assert_allclose(world_to_local(a, local_to_world(a, local)), local)


def test_eight_shards_partition_all_b_points_once():
    rows = [row for row in generate_grid([0, 0, 0]) if not row["is_A"]]
    shards = [[row["point_id"] for row in rows if row["fixed_id"] % 8 == shard]
              for shard in range(8)]
    flattened = [point for shard in shards for point in shard]
    assert len(flattened) == len(set(flattened)) == 244
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


def test_nearest_tie_break():
    rows = [{"status": "feasible", "reachable": True, "translation_m": .1,
             "abs_yaw_deg": 15, "fixed_id": 3},
            {"status": "feasible", "reachable": True, "translation_m": .1,
             "abs_yaw_deg": 0, "fixed_id": 8}]
    assert nearest(rows, True)["fixed_id"] == 8


def test_corridor_direct_and_astar_detour():
    assert corridor_reachability([.2, 0, 0], lambda _: True)["method"] == "direct"
    def detour(poses):
        return all(not (.075 < p[0] < .175 and abs(p[1]) < .02) for p in poses)
    result = corridor_reachability([.2, 0, 0], detour,
                                   CorridorConfig(search_max_expansions=250))
    assert result["reachable"] and result["method"] == "local_astar"


def test_heuristic_rejects_outside_without_repair():
    assert heuristic_pose([0, 0, 0], [.8, 0], .55)["valid"]
    result = heuristic_pose([0, 0, math.pi], [.8, 0], .55)
    assert not result["valid"] and result["reason"] == "outside_local_limits"


def test_random_budget_charges_failures_and_stops_at_hit():
    rows = generate_grid([0, 0, 0])
    for row in rows:
        row.update(status="not_found", reachable=False)
    initial = random_budget_results(rows, 17, budgets=(4,), seeds=1)[0]
    by_id = {row["point_id"]: row for row in rows}
    by_id[initial["query_order"][1]].update(status="feasible", reachable=True)
    replay = random_budget_results(rows, 17, budgets=(4,), seeds=1)[0]
    assert replay["rescued"] and replay["queries_charged"] == 2


def test_base_pose_sequence_uses_p2_contact_semantics_and_restores_state():
    from mlspaces_tests.evaluation.test_last_mile_p2 import make_task
    task, evaluator, target, _, _, snapshot = make_task()
    before = task.env.current_data.qpos.copy()
    result = evaluator.check_base_poses(snapshot, task.env.current_robot,
                                        [np.eye(4), np.eye(4)], target)
    assert result == {"collision_free": True, "checked_poses": 2,
                      "first_collision_index": None, "collisions": []}
    np.testing.assert_array_equal(task.env.current_data.qpos, before)
