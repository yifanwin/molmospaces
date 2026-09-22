from types import SimpleNamespace

import pytest

from molmo_spaces.configs.policy_configs import (
    panda_omron_planner_joint_ranges,
)
from molmo_spaces.evaluation.robot_eval_overrides import (
    panda_omron_robot_eval_override,
)


def test_panda_omron_eval_preserves_first_success():
    """评测不能继承数据生成的成功后继续执行语义。"""
    from molmo_spaces.evaluation.configs.evaluation_configs import (
        PandaOmronCuroboPickPnPEvalConfig,
    )

    assert PandaOmronCuroboPickPnPEvalConfig.model_fields["end_on_success"].default is True


@pytest.mark.parametrize(
    ("groups", "ranges"),
    [
        (["arm"], {"arm": (0, 7)}),
        (["torso", "arm"], {"torso": (0, 1), "arm": (1, 8)}),
        (
            ["base", "torso", "arm"],
            {"base": (0, 3), "torso": (3, 4), "arm": (4, 11)},
        ),
    ],
)
def test_panda_omron_planner_move_group_modes(groups, ranges):
    assert panda_omron_planner_joint_ranges(groups) == ranges


def test_panda_omron_planner_rejects_unsupported_move_groups():
    with pytest.raises(ValueError, match="planner_move_group_ids"):
        panda_omron_planner_joint_ranges(["base", "arm"])


def test_panda_omron_eval_override_replaces_only_robot_specific_state():
    episode = SimpleNamespace(
        robot=SimpleNamespace(robot_name="rby1m", init_qpos={"head": [[0.0, 0.0]]}),
        task={"robot_base_pose": [1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0]},
    )
    initial_qpos = {
        "base": [0.0, 0.0, 0.0],
        "torso": [0.2],
        "arm": [0.0] * 7,
        "gripper": [0.02, -0.02],
    }
    runtime = SimpleNamespace(
        use_config_camera_system=False,
        repair_robot_base_pose_if_colliding=False,
    )
    exp_config = SimpleNamespace(
        robot_config=SimpleNamespace(name="panda_omron", init_qpos=initial_qpos),
        eval_runtime_params=runtime,
    )

    panda_omron_robot_eval_override(episode, exp_config)

    assert episode.robot.robot_name == "panda_omron"
    assert episode.robot.init_qpos == initial_qpos
    assert episode.robot.init_qpos is not initial_qpos
    assert episode.task["robot_base_pose"] == [1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert runtime.use_config_camera_system is True
    assert runtime.repair_robot_base_pose_if_colliding is True


def test_grasp_settle_steps_covers_gripper_close_duration():
    """判定窗口必须覆盖夹爪闭合时长，否则会把手指出在合拢误判成"已夹住物体"。"""
    from molmo_spaces.configs.policy_configs import grasp_settle_steps

    # E4：66 ms 周期、500 ms 闭合。旧值 5 步 = 330 ms 小于闭合时长。
    steps = grasp_settle_steps(policy_dt_ms=66.0, gripper_close_duration=0.5)
    assert steps * 66.0 / 1000.0 >= 0.5
    assert steps == 11

    # E2：100 ms 周期、500 ms 闭合，恰好整除时仍要留出稳定余量。
    steps_e2 = grasp_settle_steps(policy_dt_ms=100.0, gripper_close_duration=0.5)
    assert steps_e2 * 100.0 / 1000.0 > 0.5
    assert steps_e2 == 7


def test_panda_omron_sphere_labels_cover_body_and_attached_object():
    """碰撞球标签要能区分机身球与 CuRobo 为 attached_object 追加的手部附加球。"""
    pytest.importorskip("curobo")
    from molmo_spaces.policy.solvers.object_manipulation.panda_omron_curobo_pick_and_place_planner_policy import (
        _SPHERE_LINK_NAMES,
        PandaOmronCuroboPickAndPlacePlannerPolicy,
    )

    label = PandaOmronCuroboPickAndPlacePlannerPolicy._sphere_label
    assert label(0) == "omron_base"
    assert label(len(_SPHERE_LINK_NAMES) - 1) == "panda_hand"
    assert label(len(_SPHERE_LINK_NAMES)) == "attached_object[0]"
    assert label(60) == "attached_object[39]"


@pytest.mark.slow
def test_split_start_overlaps_drops_boxes_touching_start_state():
    """起始状态重叠的障碍物必须被剔除：留在世界里 trajopt 必然从碰撞状态出发而失败。"""
    torch = pytest.importorskip("torch")
    pytest.importorskip("curobo")
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA：重叠检测在多边形碰撞球上运行")

    import numpy as np
    from curobo.geom.types import Cuboid

    from molmo_spaces.policy.solvers.object_manipulation.panda_omron_curobo_pick_and_place_planner_policy import (
        PandaOmronCuroboPickAndPlacePlannerPolicy,
    )

    policy = object.__new__(PandaOmronCuroboPickAndPlacePlannerPolicy)
    policy.planner_joint_ranges = {"arm": (0, 7)}
    policy._get_planning_start_config = lambda: np.zeros(7)
    # 单个球心在原点、半径 0.1 m 的碰撞球
    policy.planner = SimpleNamespace(
        motion_gen=SimpleNamespace(
            kinematics=SimpleNamespace(
                get_state=lambda q: SimpleNamespace(
                    link_spheres_tensor=torch.tensor([[[0.0, 0.0, 0.0, 0.1]]])
                )
            )
        )
    )
    identity = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    overlapping = Cuboid(name="near", pose=identity, dims=[0.1, 0.1, 0.1])
    clear = Cuboid(name="far", pose=[5.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dims=[0.1, 0.1, 0.1])

    kept, overlaps = policy._split_start_overlaps([overlapping, clear])

    assert [box.name for box in kept] == ["far"]
    assert [(link, box) for link, box, _ in overlaps] == [("omron_base", "near")]


def test_panda_omron_grasp_does_not_overshoot_into_support_surface():
    """GRASP 阶段不该在 pregrasp 退距之外额外进给。

    多进给会让手爪中心越过物体中心，薄片物体上手掌直接压到支撑面，实测触发接触、
    关节差 0.04–0.14 rad 到不了，episode 在反复重试后失败。RBY1 的默认多进给
    0.01 m 保持不变，只收掉 PandaOmron 这一侧。
    """
    from molmo_spaces.configs.policy_configs import (
        CuroboPickAndPlacePlannerPolicyConfig,
        PandaOmronCuroboPickAndPlacePlannerPolicyConfig,
    )

    field = "grasp_approach_overshoot"
    assert CuroboPickAndPlacePlannerPolicyConfig.model_fields[field].default == 0.01
    assert PandaOmronCuroboPickAndPlacePlannerPolicyConfig.model_fields[field].default == 0.0


def test_clearance_grid_measures_distance_to_nearest_obstacle():
    """净空图 = 自由格到最近障碍的距离（米），用于判断底盘四周是否宽敞。"""
    import numpy as np

    from molmo_spaces.utils.scene_maps import ProcTHORMap

    occupancy = np.ones((21, 21), dtype=bool)
    occupancy[10, :] = False  # 第 10 行是障碍，其余自由

    thor_map = ProcTHORMap(
        occupancy=occupancy,
        world_to_map=np.eye(4),
        map_to_world=np.eye(4),
        px_per_m=100,
    )
    grid = thor_map._clearance_grid()

    assert grid[10, 0] == pytest.approx(0.0)  # 障碍本身
    assert grid[5, 0] == pytest.approx(0.05)  # 距障碍 5 像素 = 5 cm
    assert grid[0, 0] == pytest.approx(0.10)  # 距障碍 10 像素 = 10 cm


def test_panda_omron_eval_override_widens_base_placement_search():
    """底盘比默认粗筛半径更宽，评测覆盖必须放半径并启用净空择优。

    实测 mobilebase0_pedestal_feet_col 的水平半径是 0.438 m，默认的
    robot_base_pose_repair_map_radius=0.40 m 覆盖不住，筛出的"自由点"可能根本
    放不下底盘。
    """
    from molmo_spaces.evaluation.robot_eval_overrides import (
        panda_omron_robot_eval_override,
    )

    episode = SimpleNamespace(
        robot=SimpleNamespace(robot_name="rby1m", init_qpos={}),
        task={"robot_base_pose": [0.0] * 7},
    )
    initial_qpos = {"base": [0.0, 0.0, 0.0], "torso": [0.2], "arm": [0.0] * 7}
    runtime = SimpleNamespace(
        use_config_camera_system=False,
        repair_robot_base_pose_if_colliding=False,
    )
    exp_config = SimpleNamespace(
        robot_config=SimpleNamespace(name="panda_omron", init_qpos=initial_qpos),
        eval_runtime_params=runtime,
    )

    panda_omron_robot_eval_override(episode, exp_config)

    assert runtime.robot_base_pose_repair_map_radius >= 0.438
    assert runtime.robot_base_pose_repair_candidate_limit > 1


def test_datagen_and_eval_share_the_same_grasp_settle_window():
    """DataGen 与 Eval 必须用同一个抓取判定窗口。

    修复一度只落在 Eval config 上，DataGen 侧仍是默认的 5 步 = 330 ms，
    短于 500 ms 的夹爪闭合时长。
    """
    import pytest as _pytest

    from molmo_spaces.configs.policy_configs import grasp_settle_steps
    from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
        PandaOmronCuroboPickAndPlaceDataGenConfig,
    )

    _pytest.importorskip("robosuite")
    config = PandaOmronCuroboPickAndPlaceDataGenConfig()
    policy_config = config._init_policy_config()

    expected = grasp_settle_steps(
        config.policy_dt_ms, policy_config.gripper_close_duration
    )
    assert policy_config.max_grasping_timesteps == expected
    # 判定窗口必须覆盖夹爪闭合时长
    window_s = policy_config.max_grasping_timesteps * config.policy_dt_ms / 1000.0
    assert window_s >= policy_config.gripper_close_duration


def test_velocity_clip_scales_with_policy_dt_not_hardcoded_10hz():
    """速度限制按实际控制周期换算：66 ms 下每步位移应比 100 ms 标定值小 34%。"""
    import numpy as np

    from molmo_spaces.policy.solvers.curobo_planner_policy import CuroboPlannerPolicy

    class _StubPolicy(CuroboPlannerPolicy):
        """只用来测 clip_to_velocity_constraint，补齐抽象方法。"""

        @property
        def planners(self):
            return {}

        @property
        def is_done(self):
            return False

        def get_action(self, info):
            return {}

        def get_phase(self):
            return None

        def get_all_phases(self):
            return {}

    def build(policy_dt_ms):
        policy = object.__new__(_StubPolicy)
        policy.config = SimpleNamespace(
            policy_dt_ms=policy_dt_ms,
            policy_config=SimpleNamespace(velocity_constraints={"arm": 0.5}),
        )
        group = SimpleNamespace(joint_pos=np.zeros(7))
        policy.task = SimpleNamespace(
            env=SimpleNamespace(
                robots=[SimpleNamespace(robot_view=SimpleNamespace(get_move_group=lambda _g: group))]
            )
        )
        return policy

    # 远大于限制的目标位移 → 应被裁剪到上限
    clipped_66 = build(66.0).clip_to_velocity_constraint({"arm": np.full(7, 5.0)})
    clipped_100 = build(100.0).clip_to_velocity_constraint({"arm": np.full(7, 5.0)})

    assert clipped_100["arm"][0] == pytest.approx(0.5)  # 100 ms 标定值
    assert clipped_66["arm"][0] == pytest.approx(0.5 * 0.66)  # 66 ms 按比例缩小
    assert clipped_66["arm"][0] < clipped_100["arm"][0]


def test_object_pose_hold_flags_drift_relative_to_gripper():
    """抓取保持判据：物体相对夹爪的位姿漂移超阈值即判为未抓住。

    这替代"两侧手指都接触"的几何代理：单侧接触但物体被稳住时不应误伤，
    而手指顶住桌面（物体相对夹爪漂移）必须被识别。
    """
    import numpy as np

    from molmo_spaces.policy.solvers.curobo_planner_policy import CuroboPlannerPolicy

    class _StubPolicy(CuroboPlannerPolicy):
        @property
        def planners(self):
            return {}

        @property
        def is_done(self):
            return False

        def get_action(self, info):
            return {}

        def get_phase(self):
            return None

        def get_all_phases(self):
            return {}

    def build(require_hold=True):
        policy = object.__new__(_StubPolicy)
        policy.config = SimpleNamespace(
            policy_config=SimpleNamespace(
                require_object_pose_hold=require_hold,
                grasp_hold_pos_tolerance=0.01,
                grasp_hold_rot_tolerance=0.10,
            )
        )
        policy._grasp_reference = np.eye(4)
        return policy

    reference = np.eye(4)

    # 未漂移 → 仍算抓住
    policy = build()
    policy._object_relative_pose = lambda: reference.copy()
    assert policy._object_pose_held() is True

    # 位置漂移 2 cm（阈值 1 cm）→ 判定未抓住
    drifted = reference.copy()
    drifted[0, 3] = 0.02
    policy._object_relative_pose = lambda: drifted.copy()
    assert policy._object_pose_held() is False

    # 姿态漂移 30°（阈值 5.7°）→ 判定未抓住
    from scipy.spatial.transform import Rotation as _R

    tilted = reference.copy()
    tilted[:3, :3] = _R.from_euler("x", np.radians(30)).as_matrix()
    policy._object_relative_pose = lambda: tilted.copy()
    assert policy._object_pose_held() is False

    # 关闭该判据时一律放行
    policy_off = build(require_hold=False)
    policy_off._object_relative_pose = lambda: drifted.copy()
    assert policy_off._object_pose_held() is True
