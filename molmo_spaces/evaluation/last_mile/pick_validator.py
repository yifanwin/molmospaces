"""P4 严格 Pick 判定；不使用 PnP success 或阶段标签。"""

from __future__ import annotations

import math

import numpy as np


def close_wait_steps(close_duration_s: float = 0.5, policy_dt_s: float = 0.1) -> int:
    return math.ceil((close_duration_s + 0.2) / policy_dt_s)


def selected_finger_bodies(model, gripper: str, fallback: set[int] | None = None) -> set[int]:
    """RBY1 真实碰撞 body 不保证在 gripper root 的 descendant 集合中。"""
    side = "l" if gripper == "left_gripper" else "r" if gripper == "right_gripper" else None
    if side is None:
        return set(fallback or ())
    names = {f"robot_0/ee_finger_{side}1", f"robot_0/ee_finger_{side}2"}
    found = {i for i in range(model.nbody) if (model.body(i).name or "") in names}
    if len(found) != 2:
        raise ValueError(f"所选夹爪的两根物理手指不完整：{gripper}")
    return found


def contact_state(model, data, robot_root: int, target_root: int, finger_bodies: set[int]):
    """读物理接触；目标与任何非机器人实体接触都算尚未脱离支撑。"""
    fingers, support, illegal = set(), [], []
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.dist > 0:
            continue
        bodies = (int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2]))
        roots = tuple(int(model.body_rootid[body]) for body in bodies)
        if target_root in roots:
            other = 1 if roots[0] == target_root else 0
            if roots[other] == robot_root:
                if bodies[other] in finger_bodies:
                    fingers.add(bodies[other])
                elif contact.dist < -0.001:
                    illegal.append(f"nonfinger_target:{model.body(bodies[other]).name}")
            else:
                support.append(model.body(bodies[other]).name or str(bodies[other]))
        elif roots[0] == roots[1] == robot_root and bodies[0] != bodies[1]:
            if contact.dist < -0.002:
                illegal.append("robot_self")
        elif robot_root in roots and roots[0] != roots[1]:
            other = roots[1] if roots[0] == robot_root else roots[0]
            if "floor" not in (model.body(other).name or "").lower() and contact.dist < -0.001:
                illegal.append(f"robot_environment:{model.body(bodies[1] if roots[0] == robot_root else bodies[0]).name}")
    return {"finger_bodies": sorted(fingers), "support_bodies": sorted(set(support)),
            "illegal": sorted(set(illegal))}


def strict_pick_verdict(samples: list[dict], initial_z: float, hold_steps: int = 10):
    """必须连续稳定持有 1 s；每个样本是独立的 100 ms 策略步终点。"""
    if len(samples) < hold_steps:
        return False, "hold_too_short"
    final = samples[-hold_steps:]
    if any(row.get("target_id_ok") is not True for row in final):
        return False, "wrong_target"
    if any(row.get("illegal") for row in final):
        return False, "illegal_collision"
    if any(row.get("support_bodies") for row in final):
        return False, "support_contact"
    if any(row.get("height_m", -math.inf) - initial_z < 0.05 for row in final):
        return False, "insufficient_lift_or_drop"
    if any(len(row.get("finger_bodies", ())) < 2 or row.get("finger_distance_m", 0) <= 0.002
           for row in final):
        return False, "not_held_by_selected_gripper"
    if any(not np.isfinite(row.get("height_m", math.nan)) for row in final):
        return False, "nonfinite_state"
    return True, "success"
