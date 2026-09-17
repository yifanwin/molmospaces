"""Robot view for robosuite's Panda mounted on an Omron mobile base."""

import numpy as np
from mujoco import MjData

from molmo_spaces.robots.robot_views.abstract import (
    GripperGroup,
    HoloJointsRobotBaseGroup,
    MJCFFrameMixin,
    RobotBaseGroup,
    RobotView,
    SimplyActuatedMoveGroup,
)
from molmo_spaces.utils.mj_model_and_data_utils import body_pose


class PandaOmronBaseGroup(HoloJointsRobotBaseGroup):
    """World-frame ``x, y, yaw`` virtual joints of the Omron base."""

    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        model = mj_data.model
        joints = [
            model.joint(f"{namespace}mobilebase0_joint_mobile_{axis}").id
            for axis in ("forward", "side", "yaw")
        ]
        actuators = [
            model.actuator(f"{namespace}mobilebase0_actuator_mobile_{axis}").id
            for axis in ("forward", "side", "yaw")
        ]
        super().__init__(
            mj_data,
            world_site_id=model.site(f"{namespace}world").id,
            holo_base_site_id=model.site(f"{namespace}base_site").id,
            joint_ids=joints,
            actuator_ids=actuators,
            root_body_id=model.body(f"{namespace}mobilebase0_base").id,
        )


class PandaOmronTorsoGroup(MJCFFrameMixin, SimplyActuatedMoveGroup):
    def __init__(
        self, mj_data: MjData, base: RobotBaseGroup, namespace: str = ""
    ) -> None:
        model = mj_data.model
        self._root_body_id = model.body(f"{namespace}mobilebase0_fixed_support").id
        self._leaf_body_id = model.body(f"{namespace}mobilebase0_support").id
        super().__init__(
            mj_data,
            [model.joint(f"{namespace}mobilebase0_joint_torso_height").id],
            [model.actuator(f"{namespace}mobilebase0_actuator_torso_height").id],
            self._root_body_id,
            base,
        )

    @property
    def leaf_frame_id(self) -> int:
        return self._leaf_body_id

    @property
    def leaf_frame_type(self):
        return "body"

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._root_body_id)


class PandaArmGroup(MJCFFrameMixin, SimplyActuatedMoveGroup):
    def __init__(
        self, mj_data: MjData, base: RobotBaseGroup, namespace: str = ""
    ) -> None:
        model = mj_data.model
        self._root_body_id = model.body(f"{namespace}robot0_link0").id
        self._ee_site_id = model.site(f"{namespace}gripper0_right_grip_site").id
        super().__init__(
            mj_data,
            [model.joint(f"{namespace}robot0_joint{i}").id for i in range(1, 8)],
            [model.actuator(f"{namespace}robot0_torq_j{i}").id for i in range(1, 8)],
            self._root_body_id,
            base,
        )

    @property
    def leaf_frame_id(self) -> int:
        return self._ee_site_id

    @property
    def leaf_frame_type(self):
        return "site"

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._root_body_id)


class PandaGripperGroup(MJCFFrameMixin, GripperGroup):
    def __init__(
        self, mj_data: MjData, base: RobotBaseGroup, namespace: str = ""
    ) -> None:
        model = mj_data.model
        self._root_body_id = model.body(f"{namespace}gripper0_right_right_gripper").id
        self._ee_site_id = model.site(f"{namespace}gripper0_right_grip_site").id
        super().__init__(
            mj_data,
            [
                model.joint(f"{namespace}gripper0_right_finger_joint1").id,
                model.joint(f"{namespace}gripper0_right_finger_joint2").id,
            ],
            [
                model.actuator(f"{namespace}gripper0_right_gripper_finger_joint1").id,
                model.actuator(f"{namespace}gripper0_right_gripper_finger_joint2").id,
            ],
            self._root_body_id,
            base,
        )

    @property
    def leaf_frame_id(self) -> int:
        return self._ee_site_id

    @property
    def leaf_frame_type(self):
        return "site"

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._root_body_id)

    def set_gripper_ctrl_open(self, open: bool) -> None:
        self.ctrl = np.array([0.04, -0.04]) if open else np.array([0.0, 0.0])

    @property
    def inter_finger_dist_range(self) -> tuple[float, float]:
        return 0.0, 0.08

    @property
    def inter_finger_dist(self) -> float:
        return float(self.joint_pos[0] - self.joint_pos[1])


class PandaOmronRobotView(RobotView):
    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        base = PandaOmronBaseGroup(mj_data, namespace)
        super().__init__(
            mj_data,
            {
                "base": base,
                "torso": PandaOmronTorsoGroup(mj_data, base, namespace),
                "arm": PandaArmGroup(mj_data, base, namespace),
                "gripper": PandaGripperGroup(mj_data, base, namespace),
            },
        )

    @property
    def name(self) -> str:
        return f"{self._namespace}panda_omron"

    @property
    def base(self) -> PandaOmronBaseGroup:
        return self._move_groups["base"]
