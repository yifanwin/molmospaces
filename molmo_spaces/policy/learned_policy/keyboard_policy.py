import logging
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Keyboard_Policy(InferencePolicy):
    RBY1_ROBOT_TYPES = {"rby1", "rby1m"}
    RBY1_CONTROL_MODES = ("left_arm", "right_arm", "base")

    def __init__(
        self,
        exp_config: MlSpacesExpConfig,
    ) -> None:
        super().__init__(exp_config)
        from pynput import keyboard

        self._keyboard = keyboard
        self.robot_type = exp_config.robot_config.name
        self.step_size = exp_config.policy_config.step_size
        self.rot_step = exp_config.policy_config.rot_step
        self._pressed = set()
        self._gripper_open = True
        self._reset_rby1_state()
        self._listener = self._keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()
        log.info(
            "Keyboard policy started. "
            "w/s: up/down (z), arrows: x/y. "
            "a/d: yaw, e/r: pitch, z/c: roll. "
            "Space: toggle gripper. q: end episode."
        )
        if self._is_rby1:
            log.info(
                "RBY-1 controls enabled. Tab cycles left arm, right arm, and base modes."
            )

    @property
    def _is_rby1(self) -> bool:
        return self.robot_type in self.RBY1_ROBOT_TYPES

    def _reset_rby1_state(self) -> None:
        self._rby1_control_mode_index = 0
        self._rby1_target_poses: dict[str, np.ndarray] = {}
        self._rby1_base_target: np.ndarray | None = None
        self._rby1_target_needs_sync = True
        self._rby1_gripper_open = {"left": True, "right": True}

    @property
    def rby1_control_mode(self) -> str:
        return self.RBY1_CONTROL_MODES[self._rby1_control_mode_index]

    def _on_press(self, key):
        if key in self._pressed:
            return
        self._pressed.add(key)
        if self._is_rby1 and key == self._keyboard.Key.tab:
            self._rby1_control_mode_index = (self._rby1_control_mode_index + 1) % len(
                self.RBY1_CONTROL_MODES
            )
            self._rby1_target_needs_sync = True
            log.info("RBY-1 control mode: %s", self.rby1_control_mode)
        elif key == self._keyboard.Key.space and self._is_rby1:
            if self.rby1_control_mode == "base":
                log.info("Space is ignored in RBY-1 base mode; switch to an arm to use a gripper.")
            else:
                side = self.rby1_control_mode.removesuffix("_arm")
                self._rby1_gripper_open[side] = not self._rby1_gripper_open[side]
        elif key == self._keyboard.Key.space:
            self._gripper_open = not self._gripper_open

    def _on_release(self, key):
        self._pressed.discard(key)

    def _key(self, char):
        return self._keyboard.KeyCode.from_char(char) in self._pressed

    def _get_delta_position(self):
        dx, dy, dz = 0.0, 0.0, 0.0
        if self._keyboard.Key.up in self._pressed:
            dx += self.step_size
        if self._keyboard.Key.down in self._pressed:
            dx -= self.step_size
        if self._keyboard.Key.left in self._pressed:
            dy += self.step_size
        if self._keyboard.Key.right in self._pressed:
            dy -= self.step_size
        if self._key("w"):
            dz -= self.step_size
        if self._key("s"):
            dz += self.step_size
        return np.array([dx, dy, dz])

    def _get_delta_rotation(self):
        roll, pitch, yaw = 0.0, 0.0, 0.0
        if self._key("a"):
            yaw += self.rot_step
        if self._key("d"):
            yaw -= self.rot_step
        if self._key("e"):
            pitch += self.rot_step
        if self._key("r"):
            pitch -= self.rot_step
        if self._key("z"):
            roll += self.rot_step
        if self._key("c"):
            roll -= self.rot_step
        return R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()

    def _is_paused(self):
        return self._key("q")

    def prepare_model(self):
        pass

    def reset(self):
        self.init_robot_pose = None
        self.init_tcp_pose = None
        self.current_position = None
        self.current_rotation = None
        self._gripper_open = True
        self._reset_rby1_state()

    def _show_views(self, views: np.ndarray) -> None:
        try:
            window_exists = cv2.getWindowProperty("views", cv2.WND_PROP_VISIBLE) >= 0
        except cv2.error:
            window_exists = False

        if not window_exists:
            screen_res = (1920, 1080)
            aspect_ratio = views.shape[0] / views.shape[1]
            new_width = int(screen_res[0] * 0.9)
            new_height = int(new_width * aspect_ratio)
            cv2.namedWindow("views", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("views", new_width, new_height)
            cv2.moveWindow(
                "views", (screen_res[0] - new_width) // 2, (screen_res[1] - new_height) // 2
            )

        cv2.imshow("views", cv2.cvtColor(views, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def render(self, obs):
        if self.robot_type == "floating_rum":
            self._show_views(obs["wrist_camera"])
        elif self._is_rby1:
            camera_names = ("head_camera", "wrist_camera_l", "wrist_camera_r")
            frames = [obs[name] for name in camera_names if name in obs]
            if not frames:
                log.warning("No RBY-1 camera observations are available for keyboard teleoperation")
                return
            views = np.concatenate(frames, axis=1)
            cv2.putText(
                views,
                f"Control: {self.rby1_control_mode}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            self._show_views(views)
        else:
            views = np.concatenate(
                [np.flip(np.flip(obs["wrist_camera"], axis=0), axis=1), obs["exo_camera_1"]], axis=1
            )
            self._show_views(views)

    def obs_to_model_input(self, obs):
        if isinstance(obs, list):
            obs = obs[0]
        self.render(obs)

        robot_pose = obs["robot_base_pose"]
        T_world_robot = np.eye(4)
        T_world_robot[:3, :3] = R.from_quat(robot_pose[3:], scalar_first=True).as_matrix()
        T_world_robot[:3, 3] = robot_pose[:3]

        if self.robot_type == "floating_rum":
            if self.init_robot_pose is None:
                self.init_robot_pose = T_world_robot
                self.current_position = T_world_robot[:3, 3].copy()
                self.current_rotation = T_world_robot[:3, :3].copy()
            return {"T_world_robot": T_world_robot}

        if self._is_rby1:
            robot_view = self.task.env.current_robot.robot_view
            control_mode = self.rby1_control_mode
            if self._rby1_target_needs_sync:
                if control_mode == "base":
                    self._rby1_base_target = robot_view.get_move_group("base").joint_pos.copy()
                else:
                    side = control_mode.removesuffix("_arm")
                    gripper_id = f"{side}_gripper"
                    self._rby1_target_poses[control_mode] = (
                        robot_view.get_move_group(gripper_id).leaf_frame_to_robot.copy()
                    )
                self._rby1_target_needs_sync = False
            return {"T_world_robot": T_world_robot, "control_mode": control_mode}

        tcp_pose = obs["tcp_pose"]
        qpos = obs["qpos"]["arm"]

        T_robot_tcp = np.eye(4)
        T_robot_tcp[:3, :3] = R.from_quat(tcp_pose[3:], scalar_first=True).as_matrix()
        T_robot_tcp[:3, 3] = tcp_pose[:3]

        if self.init_tcp_pose is None:
            self.init_tcp_pose = T_robot_tcp.copy()
            self.current_position = T_robot_tcp[:3, 3].copy()
            self.current_rotation = T_robot_tcp[:3, :3].copy()

        T_world_tcp = T_world_robot @ T_robot_tcp
        return {
            "qpos": qpos,
            "T_world_robot": T_world_robot,
            "T_world_tcp": T_world_tcp,
            "T_robot_tcp": T_robot_tcp,
        }

    def inference_model(self, model_input):
        if self._is_paused():
            return None

        if self._is_rby1:
            return self._rby1_inference(model_input)

        self.current_position += self.current_rotation @ self._get_delta_position()
        self.current_rotation = self._get_delta_rotation() @ self.current_rotation

        if self.robot_type == "floating_rum":
            goal_pose = self.init_robot_pose.copy()
            goal_pose[:3, 3] = self.current_position
            goal_pose[:3, :3] = self.current_rotation

            goal_pose_7d = np.array(
                list(goal_pose[:3, 3])
                + list(R.from_matrix(goal_pose[:3, :3]).as_quat(scalar_first=True))
            )
            return {
                "base": goal_pose_7d,
                "gripper": np.array([0.0]) if self._gripper_open else np.array([-255.0]),
            }
        else:
            kinematics = self.task.env.current_robot.kinematics
            robot_view = self.task.env.current_robot.robot_view

            gripper_mgs = set(robot_view.get_gripper_movegroup_ids())
            mgs_except_gripper = [x for x in robot_view.move_group_ids() if x not in gripper_mgs]

            new_pose = np.eye(4)
            new_pose[:3, 3] = self.current_position
            new_pose[:3, :3] = self.current_rotation

            jp = kinematics.ik(
                "arm",
                new_pose,
                mgs_except_gripper,
                robot_view.get_qpos_dict(),
                robot_view.base.pose,
                rel_to_base=True,
            )
            action = robot_view.get_ctrl_dict()
            if jp is not None:
                action.update({mg_id: jp[mg_id] for mg_id in mgs_except_gripper})

            action["gripper"] = np.array([0.0]) if self._gripper_open else np.array([255.0])
            return action

    def _rby1_gripper_actions(self) -> dict[str, np.ndarray]:
        return {
            f"{side}_gripper": np.array([-0.05 if is_open else 0.0])
            for side, is_open in self._rby1_gripper_open.items()
        }

    def _rby1_inference(self, model_input) -> dict[str, np.ndarray]:
        control_mode = model_input["control_mode"]
        action = self._rby1_gripper_actions()

        if control_mode == "base":
            assert self._rby1_base_target is not None
            target = self._rby1_base_target.copy()
            forward = 0.0
            lateral = 0.0
            if self._keyboard.Key.up in self._pressed:
                forward += self.step_size
            if self._keyboard.Key.down in self._pressed:
                forward -= self.step_size
            if self._keyboard.Key.left in self._pressed:
                lateral += self.step_size
            if self._keyboard.Key.right in self._pressed:
                lateral -= self.step_size

            heading = target[2]
            target[0] += np.cos(heading) * forward - np.sin(heading) * lateral
            target[1] += np.sin(heading) * forward + np.cos(heading) * lateral
            if self._key("a"):
                target[2] += self.rot_step
            if self._key("d"):
                target[2] -= self.rot_step
            target[2] = np.arctan2(np.sin(target[2]), np.cos(target[2]))
            self._rby1_base_target = target
            action["base"] = target
            return action

        robot_view = self.task.env.current_robot.robot_view
        kinematics = self.task.env.current_robot.kinematics
        side = control_mode.removesuffix("_arm")
        gripper_id = f"{side}_gripper"
        current_target = self._rby1_target_poses[control_mode]
        candidate_target = current_target.copy()
        candidate_target[:3, 3] += current_target[:3, :3] @ self._get_delta_position()
        candidate_target[:3, :3] = self._get_delta_rotation() @ current_target[:3, :3]

        joint_positions = kinematics.ik(
            gripper_id,
            candidate_target,
            [control_mode],
            robot_view.get_qpos_dict(),
            robot_view.base.pose,
            rel_to_base=True,
        )
        if joint_positions is not None:
            self._rby1_target_poses[control_mode] = candidate_target
            action[control_mode] = joint_positions[control_mode]
        return action

    def model_output_to_action(self, model_output):
        return model_output

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = "keyboard"
        info["timestamp"] = time.time()
        if self._is_rby1:
            info["control_mode"] = self.rby1_control_mode
        return info
