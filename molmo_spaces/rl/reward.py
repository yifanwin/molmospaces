"""Auditable shaped reward for macro-level RBY1 learning."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_REWARD_CONFIG = {
    "pick_base_progress": 2.0,
    "pick_ee_progress": 1.0,
    "grasp": 4.0,
    "lift": 3.0,
    "place_receptacle_progress": 2.0,
    "supported_and_released": 6.0,
    "task_success": 10.0,
    "invalid_action": -5.0,
    "planning_failure": -5.0,
    "collision": -5.0,
    "max_base_translation_cost": -0.25,
    "max_base_rotation_cost": -0.25,
}


@dataclass
class RewardSnapshot:
    base_to_object: float
    ee_to_object: float
    object_to_receptacle: float
    object_z: float
    held: bool


@dataclass
class RewardTracker:
    awarded_events: set[str] = field(default_factory=set)

    def reset(self) -> None:
        self.awarded_events.clear()

    def compute(
        self,
        phase: str,
        before: RewardSnapshot,
        after: RewardSnapshot,
        *,
        success: bool,
        supported_and_released: bool,
        invalid_action: bool,
        planning_failure: bool,
        collision: bool,
        translation_fraction: float,
        yaw_fraction: float,
    ) -> tuple[float, dict[str, float]]:
        terms: dict[str, float] = {}
        if phase == "pick":
            terms["base_progress"] = DEFAULT_REWARD_CONFIG["pick_base_progress"] * float(
                np.clip((before.base_to_object - after.base_to_object) / 0.5, -1, 1)
            )
            terms["ee_progress"] = DEFAULT_REWARD_CONFIG["pick_ee_progress"] * float(
                np.clip((before.ee_to_object - after.ee_to_object) / 0.5, -1, 1)
            )
            self._event(terms, "grasp", DEFAULT_REWARD_CONFIG["grasp"], after.held)
            self._event(
                terms,
                "lift",
                DEFAULT_REWARD_CONFIG["lift"],
                after.held and after.object_z >= before.object_z + 0.05,
            )
        else:
            terms["receptacle_progress"] = DEFAULT_REWARD_CONFIG[
                "place_receptacle_progress"
            ] * float(
                np.clip(
                    (before.object_to_receptacle - after.object_to_receptacle) / 0.5,
                    -1,
                    1,
                )
            )
            self._event(
                terms,
                "placed",
                DEFAULT_REWARD_CONFIG["supported_and_released"],
                supported_and_released,
            )
        self._event(terms, "task_success", DEFAULT_REWARD_CONFIG["task_success"], success)
        terms["invalid_action"] = (
            DEFAULT_REWARD_CONFIG["invalid_action"] if invalid_action else 0.0
        )
        terms["planning_failure"] = (
            DEFAULT_REWARD_CONFIG["planning_failure"] if planning_failure else 0.0
        )
        terms["collision"] = DEFAULT_REWARD_CONFIG["collision"] if collision else 0.0
        terms["base_translation_cost"] = DEFAULT_REWARD_CONFIG[
            "max_base_translation_cost"
        ] * float(np.clip(translation_fraction, 0, 1))
        terms["base_rotation_cost"] = DEFAULT_REWARD_CONFIG[
            "max_base_rotation_cost"
        ] * float(np.clip(yaw_fraction, 0, 1))
        return float(sum(terms.values())), terms

    def _event(self, terms: dict[str, float], name: str, value: float, condition: bool) -> None:
        granted = bool(condition and name not in self.awarded_events)
        terms[name] = value if granted else 0.0
        if granted:
            self.awarded_events.add(name)
