"""MolmoSpaces adapter for robosuite's PandaOmron composite robot."""

import logging
from typing import TYPE_CHECKING, cast

import mujoco
import numpy as np
from mujoco import MjData, MjSpec
from scipy.spatial.transform import Rotation as R

from molmo_spaces.controllers.abstract import Controller
from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.controllers.joint_rel_pos import JointRelPosController
from molmo_spaces.env.sensors import TCPPoseSensor
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.kinematics.parallel.warp_kinematics import SimpleWarpKinematics
from molmo_spaces.robots.abstract import Robot

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from molmo_spaces.configs.robot_configs import PandaOmronRobotConfig


log = logging.getLogger(__name__)


class PandaOmronRobot(Robot):
    def __init__(self, mj_data: MjData, config: "MlSpacesExpConfig") -> None:
        super().__init__(mj_data, config)
        robot_config = cast("PandaOmronRobotConfig", config.robot_config)
        self._robot_view = robot_config.robot_view_factory(
            mj_data, robot_config.robot_namespace
        )
        self._kinematics = MlSpacesKinematics(robot_config)
        self._parallel_kinematics = SimpleWarpKinematics(robot_config)

        arm_controller = (
            JointRelPosController(self._robot_view.get_move_group("arm"))
            if robot_config.command_mode["arm"] == "joint_rel_position"
            else JointPosController(self._robot_view.get_move_group("arm"))
        )
        base_controller = {
            "holo_joint_planar_position": JointPosController,
            "holo_joint_rel_planar_position": JointRelPosController,
        }[robot_config.command_mode["base"]](self._robot_view.get_move_group("base"))
        self._controllers = {
            "base": base_controller,
            "torso": JointPosController(self._robot_view.get_move_group("torso")),
            "arm": arm_controller,
            "gripper": JointPosController(self._robot_view.get_move_group("gripper")),
        }

    @property
    def namespace(self) -> str:
        return self.exp_config.robot_config.robot_namespace

    @property
    def robot_view(self):
        return self._robot_view

    @property
    def kinematics(self):
        return self._kinematics

    @property
    def parallel_kinematics(self):
        return self._parallel_kinematics

    @property
    def controllers(self) -> dict[str, Controller]:
        return self._controllers

    def create_robot_sensors(self):
        return super().create_robot_sensors() + [TCPPoseSensor(uuid="tcp_pose")]

    def get_arm_move_group_ids(self) -> list[str]:
        return ["arm"]

    def reset(self) -> None:
        for move_group, qpos in self.exp_config.robot_config.init_qpos.items():
            self._robot_view.get_move_group(move_group).joint_pos = qpos

    @staticmethod
    def robot_model_root_name() -> str:
        return "robot0_base"

    @staticmethod
    def _compose_robosuite_spec() -> MjSpec:
        try:
            import robosuite
            from robosuite.models.bases import robot_base_factory
            from robosuite.models.grippers import gripper_factory
            from robosuite.models.robots import create_robot
        except ImportError as exc:
            raise ImportError(
                "PandaOmronRobot requires robosuite 1.5.x. Install it with "
                "`pip install -e '.[mujoco,robosuite]'` or install the local checkout "
                "with `pip install -e ../robosuite`."
            ) from exc

        version = tuple(int(part) for part in robosuite.__version__.split(".")[:2])
        if version != (1, 5):
            raise RuntimeError(
                f"PandaOmronRobot supports robosuite 1.5.x, found {robosuite.__version__}."
            )

        model = create_robot("PandaOmron", idn=0)
        model.add_base(robot_base_factory(model.default_base, idn=0))
        model.update_joints()
        model.update_actuators()
        gripper = gripper_factory(model.default_gripper["right"], idn="0_right")
        model.add_gripper(gripper, model.eef_name["right"])
        return MjSpec.from_string(model.get_xml())

    @staticmethod
    def _make_position_servo(
        spec: MjSpec,
        actuator_name: str,
        joint_name: str,
        kp: float,
        kv: float,
        ctrlrange: tuple[float, float] | list[float],
        preserve_ctrlrange_as_forcerange: bool = False,
    ) -> None:
        actuator = spec.actuator(actuator_name)
        joint = spec.joint(joint_name)
        if actuator is None:
            raise RuntimeError(f"robosuite PandaOmron actuator not found: {actuator_name}")
        if joint is None:
            raise RuntimeError(f"robosuite PandaOmron joint not found: {joint_name}")

        old_ctrlrange = actuator.ctrlrange.copy()
        actuator.gainprm[:] = 0
        actuator.gainprm[0] = kp
        actuator.biasprm[:] = 0
        actuator.biasprm[1] = -kp
        actuator.biasprm[2] = -kv
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
        actuator.ctrllimited = 1
        actuator.ctrlrange = np.asarray(ctrlrange, dtype=float)
        if preserve_ctrlrange_as_forcerange:
            actuator.forcelimited = 1
            actuator.forcerange = old_ctrlrange

    @classmethod
    def _load_robot_spec(
        cls, robot_config: "PandaOmronRobotConfig", strip_meshes: bool = False
    ) -> MjSpec:
        spec = cls._compose_robosuite_spec()

        for joint_suffix, kp, kv, ctrlrange in zip(
            ("forward", "side", "yaw"),
            robot_config.base_kp,
            robot_config.base_kv,
            robot_config.base_ctrlrange,
            strict=True,
        ):
            cls._make_position_servo(
                spec,
                f"mobilebase0_actuator_mobile_{joint_suffix}",
                f"mobilebase0_joint_mobile_{joint_suffix}",
                kp,
                kv,
                ctrlrange,
            )
            # 原速度控制模型的静摩擦会让位置伺服停在目标外：
            # 偏航轴 250 / 5000 = 0.05 rad，超过规划器的 0.0275 容差。
            # 与升降轴一样移除控制器遗留的关节摩擦，保留几何接触摩擦。
            spec.joint(f"mobilebase0_joint_mobile_{joint_suffix}").frictionloss = 0.0

        cls._make_position_servo(
            spec,
            "mobilebase0_actuator_torso_height",
            "mobilebase0_joint_torso_height",
            robot_config.torso_kp,
            robot_config.torso_kv,
            (0.0, 0.34),
        )
        # robosuite uses a very large static friction value because its controller
        # commands motor force directly. It would prevent small position-servo moves.
        spec.joint("mobilebase0_joint_torso_height").frictionloss = 0.0
        for i, (kp, kv) in enumerate(
            zip(robot_config.arm_kp, robot_config.arm_kv, strict=True), start=1
        ):
            joint = spec.joint(f"robot0_joint{i}")
            if joint is None:
                raise RuntimeError(f"robosuite PandaOmron joint not found: robot0_joint{i}")
            cls._make_position_servo(
                spec,
                f"robot0_torq_j{i}",
                f"robot0_joint{i}",
                kp,
                kv,
                tuple(joint.range),
                preserve_ctrlrange_as_forcerange=True,
            )

        # robosuite relies on its renderer hiding geom group 0. MolmoSpaces
        # renders every geom group, which would otherwise expose the yellow-green
        # collision meshes (most visibly the large Omron bounding box) on top of
        # the detailed group-1 visual meshes. Keep these geoms fully active for
        # physics, but make only their render material transparent.
        for body in spec.bodies:
            for geom in body.geoms:
                if geom.group == 0:
                    geom.material = ""
                    geom.rgba = [0.0, 0.0, 0.0, 0.0]

        if strip_meshes:
            for body in spec.bodies:
                for geom in list(body.geoms):
                    if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                        spec.delete(geom)
        return spec

    @classmethod
    def add_robot_to_scene(
        cls,
        robot_config: "PandaOmronRobotConfig",
        spec: MjSpec,
        prefix: str,
        pos: list[float],
        quat: list[float],
        randomize_textures: bool = False,
        strip_meshes: bool = False,
    ) -> None:
        if randomize_textures:
            log.warning(
                "PandaOmron uses robosuite's source materials; "
                "texture randomization is unsupported."
            )

        pos = pos + [0.0] if len(pos) == 2 else pos
        rotation = R.from_quat(quat, scalar_first=True)
        roll, pitch, yaw = rotation.as_euler("xyz")
        if not np.allclose([roll, pitch], [0.0, 0.0]):
            raise ValueError("PandaOmron base only supports planar initial roll/pitch")

        robot_spec = cls._load_robot_spec(robot_config, strip_meshes=strip_meshes)
        root = robot_spec.body(cls.robot_model_root_name())
        if root is None:
            raise RuntimeError("robosuite PandaOmron root body not found: robot0_base")
        mobile_base = robot_spec.body("mobilebase0_base")
        if mobile_base is None:
            raise RuntimeError("robosuite PandaOmron base body not found: mobilebase0_base")
        planar_pivot = robot_spec.joint("mobilebase0_joint_mobile_yaw").pos.copy()
        mobile_base.add_site(
            name="base_site", pos=planar_pivot, quat=[1, 0, 0, 0]
        )
        # A navigation-style chase camera. It is attached to the Omron body, so
        # it follows both translation and yaw while keeping the whole robot and
        # the workspace in front of it in view. The orientation matches the
        # existing RBY1 ``camera_follower`` with a lower height for PandaOmron.
        mobile_base.add_camera(
            name="camera_follower",
            pos=[-1.3, 0.0, 2.2],
            quat=[0.653288, 0.2705823, -0.2705823, -0.653288],
            fovy=60.0,
        )

        # The joint reference values make qpos itself the world-frame x/y/yaw.
        for idx, suffix in enumerate(("forward", "side")):
            joint = robot_spec.joint(f"mobilebase0_joint_mobile_{suffix}")
            joint.axis = rotation.inv().apply(np.eye(3)[idx])
            joint.ref = pos[idx]
        robot_spec.joint("mobilebase0_joint_mobile_yaw").ref = yaw

        # ``pos`` denotes the planar joint pivot (the physical base reference),
        # while robosuite's model root is offset from that pivot.
        attach_pos = np.asarray(pos) - rotation.apply(planar_pivot)
        attach_frame = spec.worldbody.add_frame(pos=attach_pos, quat=quat)
        attach_frame.attach_body(root, prefix, "")
        spec.worldbody.add_site(name=f"{prefix}world", pos=[0, 0, 0])

    @classmethod
    def apply_control_overrides(
        cls, spec: MjSpec, robot_config: "PandaOmronRobotConfig"
    ) -> None:
        if not robot_config.gravcomp:
            return
        arm_root = spec.body(f"{robot_config.robot_namespace}robot0_link0")
        if arm_root is None:
            raise RuntimeError("Panda arm root not found while applying gravity compensation")
        arm_root.gravcomp = 1.0
        for body in arm_root.find_all("body"):
            body.gravcomp = 1.0
