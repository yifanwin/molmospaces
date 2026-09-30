"""P4 无资产协议和判定器验收。"""
import json

from molmo_spaces.evaluation.last_mile.pick_validator import close_wait_steps, strict_pick_verdict
from molmo_spaces.evaluation.last_mile.run_p4 import aggregate, _verify_complete
from molmo_spaces.evaluation.last_mile.build import atomic_json, digest


def _sample(height=0.06, fingers=(1, 2), support=(), illegal=(), distance=.02):
    return {"height_m": height, "finger_bodies": fingers, "support_bodies": support,
            "illegal": illegal, "finger_distance_m": distance, "target_id_ok": True}


def test_close_wait_and_strict_positive_negative():
    assert close_wait_steps() == 7
    assert strict_pick_verdict([_sample()] * 10, 0.0) == (True, "success")
    assert strict_pick_verdict([_sample()] * 9, 0.0)[0] is False
    assert strict_pick_verdict([_sample(height=.049)] * 10, 0.0)[0] is False
    assert strict_pick_verdict([_sample(fingers=(1,))] * 10, 0.0)[0] is False
    assert strict_pick_verdict([_sample(support=("table",))] * 10, 0.0)[0] is False
    assert strict_pick_verdict([_sample(illegal=("robot_environment",))] * 10, 0.0)[0] is False
    wrong = _sample(); wrong["target_id_ok"] = False
    assert strict_pick_verdict([wrong] * 10, 0.0)[0] is False


def test_denominators_and_pairs_do_not_invent_independent_trials():
    rows = [
        {"subset_index": 0, "p1_status": "completed", "p2_status": "not_found", "p3_status": "completed",
         "B_point_id": "G001", "A": {"Y_pick": 0, "status": "planner_no_witness"},
         "B": {"Y_pick": 1}, "transfer": {"arrived": True},
         "B_after_transfer": {"Y_pick": 0}, "elapsed_sec": 3},
        {"subset_index": 1, "p1_status": "completed", "p2_status": "not_found", "p3_status": "completed",
         "A": {"Y_pick": 0, "status": "planner_no_witness"}},
        {"subset_index": 2, "p1_status": "collision", "p2_status": "skipped", "p3_status": "skipped"},
    ]
    p3 = [{"geo_rescue": True, "reachable_rescue": True, "reachable_unknown": False},
          {"geo_rescue": False, "reachable_rescue": False, "reachable_unknown": True}, {}]
    result = aggregate(rows, p3, "pilot")
    assert (result["N_attempt"], result["N_nav"], result["N_eval"], result["M"]) == (3, 2, 1, 1)
    assert result["I_reach"] == 1 and result["I_reach_all_valid_lower"] == .5
    assert result["pair_cells"]["01"] == 1 and result["paired_difference"] == 1
    assert result["transfer_minus_static_B"] == -1


def test_complete_marker_rejects_mutated_artifact(tmp_path):
    (tmp_path / "inputs.json").write_text('1')
    atomic_json(tmp_path / "COMPLETE.json", {"inputs_sha256": digest(tmp_path / "inputs.json")})
    _verify_complete(tmp_path, {"inputs_sha256": "inputs.json"})
    (tmp_path / "inputs.json").write_text('2')
    try:
        _verify_complete(tmp_path, {"inputs_sha256": "inputs.json"})
    except ValueError:
        pass
    else:
        raise AssertionError("mutated artifact should be rejected")


def test_formal_decision_requires_calibration_and_house_evidence():
    from molmo_spaces.evaluation.last_mile.decision import house_bootstrap_ci, formal_decision
    assert house_bootstrap_ci([(1, 0), (2, 1)]) is not None
    rows = [{"subset_index": i, "house": i // 5, "p1_status": "completed",
             "p2_status": "not_found", "p3_status": "completed"} for i in range(100)]
    p3 = [{"reachable_unknown": False, "reachable_rescue": False} for _ in rows]
    metrics = {"I_reach": 0.0, "I_reach_all_valid_lower": 0.0,
               "I_reach_all_valid_upper": 0.0, "N_pair_known": 0,
               "N_pair_preselected": 0, "N_eval": 100, "N_nav": 100}
    assert formal_decision(rows, p3, metrics)["direction_decision"] == "无结论"
    assert formal_decision(rows, p3, metrics, True)["direction_decision"] == "停止本协议下 last-mile 主方向"


def test_rby1_physical_finger_whitelist_is_bilateral():
    from types import SimpleNamespace
    from molmo_spaces.evaluation.last_mile.pick_validator import selected_finger_bodies
    names = ["world", "robot_0/EE_BODY_R", "robot_0/ee_finger_r1", "robot_0/ee_finger_r2"]
    model = SimpleNamespace(nbody=len(names), body=lambda i: SimpleNamespace(name=names[i]))
    assert selected_finger_bodies(model, "right_gripper", {1}) == {2, 3}
