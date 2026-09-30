"""P4 正式批的 house 聚类区间与预登记决策；试点绝不调用。"""

from __future__ import annotations

from collections import defaultdict
import numpy as np


def house_bootstrap_ci(records: list[tuple[int, float]], seed: int = 20260921,
                       draws: int = 2000):
    """一个源目标实例一条记录，house 是重采样单位。"""
    groups = defaultdict(list)
    for house, value in records:
        groups[house].append(float(value))
    houses = sorted(groups)
    if len(houses) < 2:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        selected = rng.choice(houses, size=len(houses), replace=True)
        trial = [value for house in selected for value in groups[house]]
        if trial:
            values.append(float(np.mean(trial)))
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))] if values else None


def formal_decision(rows: list[dict], p3: list[dict], metrics: dict,
                    calibration_passed: bool = False):
    """阈值按 2026-09-21 计划；证据缺口只能返回无结论。"""
    determinate = [r for r in rows if r["p1_status"] == "completed" and
                   r["p2_status"] != "unknown" and r["p3_status"] == "completed" and
                   not p3[r["subset_index"]].get("reachable_unknown")]
    incidence = [(r["house"], float(r["p2_status"] == "not_found" and
                                  p3[r["subset_index"]].get("reachable_rescue", False)))
                 for r in determinate]
    pair = [r for r in rows if r.get("B_point_id") and
            r.get("A", {}).get("Y_pick") in (0, 1) and
            r.get("B", {}).get("Y_pick") in (0, 1)]
    paired = [(r["house"], float(r["B"]["Y_pick"] - r["A"]["Y_pick"])) for r in pair]
    rescue_houses = len({r["house"] for r in determinate if r["p2_status"] == "not_found" and
                         p3[r["subset_index"]].get("reachable_rescue")})
    i_ci = house_bootstrap_ci(incidence)
    pair_ci = house_bootstrap_ci(paired)
    unknown_sensitive = (metrics["I_reach_all_valid_lower"] is None or
                         metrics["I_reach_all_valid_lower"] < .20 or
                         metrics["N_pair_known"] != metrics["N_pair_preselected"])
    result = {"I_reach_house_ci95": i_ci, "paired_difference_house_ci95": pair_ci,
              "rescue_houses": rescue_houses, "bootstrap_draws": 2000,
              "calibration_passed": bool(calibration_passed),
              "unknown_changes_continue": unknown_sensitive,
              "direction_decision": "无结论"}
    if not calibration_passed or len(rows) < 100 or i_ci is None:
        result["reason"] = "缺正式百例、有效 house 区间或严格 Pick 物理正负校准"
    elif (metrics["I_reach"] is not None and metrics["I_reach"] >= .20 and
          rescue_houses >= 5 and pair_ci is not None and pair_ci[0] > 0 and
          not unknown_sensitive):
        result.update(direction_decision="可进入预算受限 Agent 研究", reason="满足预登记继续阈值")
    elif (i_ci[1] < .05 and metrics["I_reach_all_valid_upper"] is not None and
          metrics["I_reach_all_valid_upper"] < .05 and
          metrics["N_eval"] / max(1, metrics["N_nav"]) >= .95):
        result.update(direction_decision="停止本协议下 last-mile 主方向", reason="满足预登记停止阈值")
    else:
        result["reason"] = "区间、跨 house、配对改善或 unknown 敏感性未达阈值"
    return result
