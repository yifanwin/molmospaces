# MolmoSpaces 动作轨迹生成机制总结

> 调研范围：`scripts/datagen/` 及其直接调用的 `molmo_spaces/data_generation/`、`molmo_spaces/policy/`、`molmo_spaces/planner/`、`molmo_spaces/tasks/` 和 `molmo_spaces/utils/save_utils.py`。
>
> 本文基于当前工作区代码，重点回答两个问题：**动作轨迹如何产生**，以及**目前有哪些生成方法**。

## 1. 结论概览

MolmoSpaces 并不是先离线生成一条统一格式的轨迹，再把它送入模拟器；它的主流程是一个在线 rollout 闭环：

1. 配置指定场景、任务采样器、机器人、相机和策略；
2. 任务采样器在每个 house 中采样目标物体、初始状态、目标状态和机器人初始位姿；
3. 策略根据当前观测或模拟器真值生成一个动作，或者生成一段 action chunk；
4. `BaseMujocoTask` 将动作交给机器人控制器，并推进 MuJoCo 物理仿真；
5. 每次采样观测时缓存状态、动作、奖励、成功标记和图像；
6. episode 结束后按成功条件过滤，保存为 HDF5，图像另存为 MP4。

当前代码中的轨迹来源可归纳为五类：

| 类别 | 核心方法 | 典型任务 | 主要入口/配置 |
|---|---|---|---|
| 几何规划 + IK | 抓取姿态采样、任务空间直线插值、逐步逆运动学 | pick、pick-and-place、open、close | Franka/RUM 的 `*PlannerPolicyConfig` |
| CuRobo 运动规划 | GPU IK/TrajOpt、碰撞检测、关节轨迹插值、阶段状态机 | RBY1 pick、pick-and-place、开门/开合 | `RBY1*DataGenConfig`、`DoorOpening*Config` |
| A* 导航规划 | 占据栅格、A*、路径稠密化/样条平滑、失败重规划 | RBY1 navigate-to-object | `NavToObjDataGenConfig` |
| 学习策略推理 | 图像/机器人状态输入模型，模型输出关节与夹爪动作 | 模型评测或数据回放式采集 | `PiPolicyConfig`、双臂 YAM PI 等 |
| 人工遥操作 | 键盘、SpaceMouse、手机位姿映射到末端/底盘目标 | 人工示教轨迹 | `TeleopPolicyConfig` |

此外还有 `DummyPolicy` 和 `BrownianMotionPolicy`，主要用于测试或随机运动，不是 `scripts/datagen/run_pipeline.py` 暴露的正式选项。

---

## 2. “轨迹”在代码中的三个层次

阅读代码时需要区分三类容易混淆的概念：

1. **任务级动作序列**：例如“到预抓取位姿 → 到抓取位姿 → 闭合夹爪 → 抬起”。它由 planner policy 的阶段或 action primitive 描述。
2. **控制命令轨迹**：策略每个 policy step 返回的 `dict[str, array]`，键通常是 `arm`、`base`、`gripper`、`left_arm`、`right_arm` 等 move group。
3. **落盘数据轨迹**：HDF5 中按观测时刻排列的 `qpos`、`qvel`、`actions/*`、奖励和成功标记。它是 rollout 期间传感器采样结果，不一定等于规划器内部的全部 waypoint。

因此，规划器内部 waypoint 数、真实物理控制步数和 HDF5 时间维 `T` 不一定相等。

---

## 3. 端到端生成流程

### 3.1 配置与入口

推荐的通用入口是：

```bash
export PYTHONPATH="${PYTHONPATH}:."
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python -m molmo_spaces.data_generation.main FrankaPickDroidDataGenConfig
```

执行过程为：

```text
配置注册/加载
  → 创建 MlSpacesExpConfig
  → ParallelRolloutRunner
  → 按 house 和 batch 分配 worker
  → TaskSampler.sample_task()
  → policy_config.policy_factory(config, task)
  → task.reset()
  → policy.get_action_chunk(obs) 或 policy.get_action(obs)
  → task.step_chunk(...) / task.step(...)
  → robot.update_control() + MuJoCo physics step
  → task.get_history()
  → prepare_episode_for_saving()
  → save_trajectories() + MP4
```

主要实现位置：

- 通用入口：[`molmo_spaces/data_generation/main.py`](../../molmo_spaces/data_generation/main.py)
- 并行 rollout：[`molmo_spaces/data_generation/pipeline.py`](../../molmo_spaces/data_generation/pipeline.py)
- task step 与历史缓存：[`molmo_spaces/tasks/task.py`](../../molmo_spaces/tasks/task.py)
- HDF5/视频保存：[`molmo_spaces/utils/save_utils.py`](../../molmo_spaces/utils/save_utils.py)

### 3.2 任务采样先决定“要完成什么”

每个 worker 持有一个 task sampler，并逐 house 采样 episode。采样器负责：

- 选择场景和 house；
- 选择 pickup object、receptacle 或可动关节；
- 采样目标物体初始/目标姿态；
- 采样机器人初始位姿；
- 可选地随机化灯光、纹理和动力学；
- 构造具体 `BaseMujocoTask`。

`samples_per_house` 表示希望收集的条数。最大尝试数约为
`samples_per_house × max_total_attempts_multiplier`。如果只保留成功轨迹，失败 episode 不计入已收集数量，因此会继续采样，直到收够或耗尽尝试次数。

### 3.3 策略产生动作

通用 runner 优先调用：

```python
action_chunk = policy.get_action_chunk(observation) or [policy.get_action(observation)]
```

- 普通 planner/teleop 通常没有 chunk，每次返回一个动作；
- 支持 chunk 的学习策略可一次产生多步动作，chunk 内开环执行；
- `None` 动作或包含 `done=True` 的动作会结束 episode；
- `end_on_success=True` 时，可在达到成功条件后提前停止。

### 3.4 动作如何变成物理运动

`BaseMujocoTask._apply_action()` 的关键路径是：

```text
动作字典
  → robot.update_control(action)
  → 每个 policy step 内执行若干 control step
  → 每个 control step 调用 robot.compute_control()
  → 每个 control step 内推进若干 MuJoCo simulation step
```

三个时间尺度必须整除：

- `policy_dt_ms`：策略动作周期；
- `ctrl_dt_ms`：机器人控制器周期；
- `sim_dt_ms`：MuJoCo 仿真周期。

配置强制满足：

```text
policy_dt_ms % ctrl_dt_ms == 0
ctrl_dt_ms % sim_dt_ms == 0
```

机器人层还可通过 `robot_config.action_noise_config` 给动作加噪声。动作噪声是对已有轨迹的扰动，不是独立的规划方法。

---

## 4. 方法一：几何规划、任务空间插值与 IK

实现入口：

- [`base_object_manipulation_planner_policy.py`](../../molmo_spaces/policy/solvers/object_manipulation/base_object_manipulation_planner_policy.py)
- [`pick_planner_policy.py`](../../molmo_spaces/policy/solvers/object_manipulation/pick_planner_policy.py)
- [`pick_and_place_planner_policy.py`](../../molmo_spaces/policy/solvers/object_manipulation/pick_and_place_planner_policy.py)
- [`open_close_planner_policy.py`](../../molmo_spaces/policy/solvers/object_manipulation/open_close_planner_policy.py)

### 4.1 轨迹生成步骤

1. 从对象的 grasp library 读取候选抓取位姿；
2. 可按碰撞、IK 可达性和代价函数筛选候选；
3. 从最佳 grasp pose 派生 pregrasp、grasp、lift、preplace、place、retreat 等关键位姿；
4. 把关键位姿组织成一组 action primitive；
5. 在相邻 TCP 位姿之间按速度和时间做 SE(3) 直线/旋量插值；
6. 每个 policy step 对当前插值位姿求 IK，输出关节位置命令；
7. 在阶段间插入夹爪开合、关节空间回 home 和 noop settling 动作。

主要 primitive 包括：

- `TCPMoveSequence`：任务空间位姿插值，每步 IK；
- `JointMoveSequence`：关节空间线性插值；
- `GripperAction`：定时开合夹爪；
- `NoopAction`：保持并等待系统稳定。

### 4.2 不同任务的阶段

**Pick**：

```text
打开夹爪 → pregrasp → grasp → 闭合夹爪 → lift
```

**Pick-and-place**：

```text
打开夹爪 → pregrasp → grasp → 闭合夹爪 → lift
→ preplace → place → 打开夹爪 → retreat → go_home → settle
```

`pick_and_place_next_to` 和 `pick_and_place_color` 主要改变放置目标的选取/计算，底层仍使用同一套 primitive 与 IK 跟踪机制。

**Open/close**：

- 对 hinge joint 依据关节轴生成圆弧上的目标；
- 对 slide joint 沿滑动轴生成直线路径；
- 再用相同的 TCP 插值与 IK 机制跟踪。

### 4.3 失败与重试

以下情况会触发保持、重新规划或 episode 失败：

- 连续 IK 失败超过阈值；
- TCP 位置/旋转跟踪误差过大；
- 闭合夹爪后检测到未抓住物体；
- 单个阶段超时；
- 重试次数超过 `max_retries`。

### 4.4 特点

优点是实现简单、阶段含义清晰、容易生成结构化专家示教；局限是中间路径主要靠直线插值和逐点 IK，默认不等价于完整的全局避障运动规划。

---

## 5. 方法二：CuRobo 碰撞感知运动规划

实现入口：

- [`curobo_planner.py`](../../molmo_spaces/planner/curobo_planner.py)
- [`curobo_planner_policy.py`](../../molmo_spaces/policy/solvers/curobo_planner_policy.py)
- [`curobo_pick_and_place_planner_policy.py`](../../molmo_spaces/policy/solvers/object_manipulation/curobo_pick_and_place_planner_policy.py)
- [`curobo_open_close_planner_policy.py`](../../molmo_spaces/policy/solvers/object_manipulation/curobo_open_close_planner_policy.py)
- RBY1 开门状态机：[`opening_solver.py`](../../molmo_spaces/policy/solvers/opening_solver.py)

### 5.1 适用范围

当前注册配置中，RBY1/RBY1M 的以下任务主要使用 CuRobo：

- `RBY1PickDataGenConfig`；
- `RBY1PickAndPlaceDataGenConfig`；
- `RBY1OpenDataGenConfig`；
- `DoorOpeningDataGenConfig` / `DoorOpeningDebugConfig`。

### 5.2 生成过程

1. 根据目标位姿和可达性选择左臂或右臂；
2. 将附近场景物体近似为碰撞 mesh/cuboid，并加载机器人 collision spheres；
3. 批量采样 pregrasp、place 或 articulation 目标位姿；
4. 使用 CuRobo 的 IK + TrajOpt 搜索无碰关节轨迹；
5. 从成功候选中选取最佳轨迹；
6. 按 `interpolation_dt` 对关节轨迹插值并逐 waypoint 执行；
7. 状态机控制夹爪、抓取验证、抬升、放置、开合关节和失败恢复。

Pick-and-place 的核心阶段为：

```text
PREGRASP → GRASP → LIFT → PLACE → POSTPLACE → DONE
```

Open/close 的核心阶段为：

```text
HEIGHT_SELECTION → PREGRASP → GRASP → ARTICULATE
→ POSTARTICULATE → DONE
```

规划既可使用本地 GPU，也可通过 `CuroboClient` 调用 gRPC 服务。`server_urls=[]` 表示使用本地 CuRobo；本地模式要求 CUDA GPU 和可选 CuRobo 依赖。

### 5.3 与普通 IK planner 的区别

普通 planner 先规定一条较简单的任务空间曲线，再逐点求 IK；CuRobo 则在关节空间用优化器搜索满足末端目标、关节约束、自碰撞和场景碰撞约束的整段轨迹，更适合自由度高且需要全身/双臂避障的 RBY1。

---

## 6. 方法三：A* 导航轨迹

实现入口：

- [`astar_planner.py`](../../molmo_spaces/planner/astar_planner.py)
- [`astar_planner_policy.py`](../../molmo_spaces/policy/solvers/navigation/astar_planner_policy.py)
- [`nav_to_obj_configs.py`](../../molmo_spaces/data_generation/config/nav_to_obj_configs.py)

轨迹生成过程：

1. 从 iTHOR/ProcTHOR/Holodeck 场景生成二维 occupancy map；
2. 按机器人安全半径膨胀障碍，并下采样为规划栅格；
3. `NavGoalSampler` 在目标物体附近采样可导航终点；
4. 将机器人起点和目标点映射到图节点；
5. 在带 distance-transform 代价的栅格图上执行 A*；
6. 将离散路径映射回世界坐标；
7. 在距离目标中心约 `path_min_dist_to_target_center` 处截断；
8. 对路径加密，并构造 `(x, y, yaw)` waypoint；
9. 当前配置实际使用 `AStarSmoothPlannerPolicy`，通过样条平滑位置并用切线方向生成 yaw；
10. 若长时间未接近 waypoint，则把当前位置加入 blacklist 并尝试重规划。

最终动作通常是：

```python
{"base": np.array([x, y, yaw]), "done": False}
```

到达全部 waypoint 后返回 noop base command 和 `done=True`。

运行示例：

```bash
python -m molmo_spaces.data_generation.main NavToObjDataGenConfig
```

---

## 7. 方法四：学习策略生成轨迹

相关配置位于 [`policy_configs_baselines.py`](../../molmo_spaces/configs/policy_configs_baselines.py)。当前代码定义了 PI、DreamZero、CAP、Bimanual YAM PI 等学习策略，但 `scripts/datagen/run_pipeline.py` 的命令行主要暴露 `pi`。

### 7.1 PI Policy

实现：[`pi_policy.py`](../../molmo_spaces/policy/learned_policy/pi_policy.py)

1. 读取外部相机、腕部相机、机械臂关节、夹爪状态和语言 prompt；
2. resize 图像并组成模型输入；
3. 调用本地 OpenPI checkpoint，或 WebSocket 远端模型；
4. 模型一次预测 `chunk_size` 个动作；
5. 每个动作的前 7 维映射到 arm，索引 7 映射到连续或二值 gripper command。

动作格式示例：

```python
{
    "arm": action[:7],
    "gripper": np.array([0.0]),  # 或 np.array([255.0])
}
```

### 7.2 两个 runner 的 chunk 行为不同

- 通用 `ParallelRolloutRunner` 调用 `get_action_chunk()`，PI 的整段预测会在下一次取观测前开环执行；
- `scripts/datagen/run_pipeline.py` 自定义了 `MyRolloutRunner.run_single_rollout()`，只调用 `get_action()`，因此它会逐步执行 PI buffer，并在每个动作后重新采样观测，但只在 buffer 用完后重新推理。

分析数据时间对齐时必须确认数据是由哪个 runner 生成的。

### 7.3 其他学习策略

- `DreamZeroPolicyConfig`：远端 DreamZero 推理；
- `CAPPolicyConfig`：远端 CAP，可选 VLM；
- `BimanualYamPiPolicyConfig`：通过 LeRobot gRPC 服务获得双臂动作 chunk。

这些策略在全项目中已有配置，但没有全部接入 `run_pipeline.py --policy` 的 choices，因此不能把“类已存在”理解为“该脚本命令行已可直接运行”。

---

## 8. 方法五：人工遥操作轨迹

`TeleopPolicyConfig` 支持三类设备：

| 设备 | 实现 | 轨迹生成方式 |
|---|---|---|
| 键盘 | `keyboard_policy.py` | 按键产生 TCP 平移/旋转增量，IK 转关节命令 |
| SpaceMouse | `spacemouse_policy.py` | 6DoF 输入积分为 TCP/底盘目标，按钮控制夹爪 |
| 手机 | `phone_policy.py` | Teledex/Mujoco AR 手机相对位姿映射到机器人 TCP 或浮动底盘 |

对固定机械臂，遥操作目标通常先转换成末端位姿，再通过 IK 生成关节命令；对 `floating_rum`，可直接输出 7D base pose 和夹爪命令。

默认 `TeleopPolicyConfig.device` 是 `keyboard`。要使用手机或 SpaceMouse，必须显式构造：

```python
TeleopPolicyConfig(device="phone")
# 或
TeleopPolicyConfig(device="spacemouse")
```

[`phone_teleop.md`](phone_teleop.md) 描述了手机连接流程，但它给出的 `run_pipeline.py --policy teleop` 命令在当前代码中会创建默认键盘策略，并不会自动把 device 改成 `phone`，因此运行手机遥操作前还需要修改/注入配置。

---

## 9. 专用两阶段脚本：RBY1 导航后抓取

[`run_rby1_nav_pick.py`](run_rby1_nav_pick.py) 串联两种已有方法：

1. 用 `NavToObjDataGenConfig` 和 A* 生成导航 episode；
2. 从导航 HDF5 的 `traj_0/obs_scene` 读取 `object_name`；
3. 用同一 house 和同一对象名配置 `RBY1PickDataGenConfig`；
4. 用本地 GPU CuRobo 生成抓取 episode。

示例：

```bash
python scripts/datagen/run_rby1_nav_pick.py \
  --scene-dataset holodeck-objaverse \
  --house-index 0 \
  --target-types Cup
```

注意：这不是一条连续的“导航 + 抓取”物理轨迹。脚本明确将两阶段保存为独立 episode，导航结束时的模拟器状态和机器人状态不会传给抓取阶段；复用的只是场景索引和目标对象名。

---

## 10. 输出数据如何形成

### 10.1 历史缓存

`task.reset()` 会先缓存初始观测。每次 `task.step()` 后继续缓存：

- observations；
- rewards；
- terminals / truncateds；
- successes；
- last action；
- episode 级 `obs_scene` 元数据。

`obs_scene` 包含任务类型、语言描述、`policy_dt_ms`、指代表达、冻结配置，以及 planner 的 phase 字典或学习策略 checkpoint 信息。

### 10.2 HDF5 与视频

当前实际输出大致为：

```text
<output_dir>/
├── running_log.log
├── experiment_config_<timestamp>.pkl
├── house_0/
│   ├── trajectories_batch_1_of_1.h5
│   ├── episode_00000000_<camera>_batch_1_of_1.mp4
│   └── ...
└── house_N/
    └── ...
```

HDF5 的主要结构为：

```text
traj_N/
├── obs/
│   ├── agent/qpos, qvel
│   ├── extra/...                 # TCP、物体、抓取、阶段等传感器
│   ├── sensor_param/...          # 相机内外参
│   └── sensor_data/...           # 视频路径等
├── actions/
│   ├── commanded_action          # 定长 uint8 中的 JSON 字符串
│   ├── joint_pos                 # 定长 uint8 中的 JSON 字符串
│   ├── joint_pos_rel             # 定长 uint8 中的 JSON 字符串
│   ├── ee_pose                   # 定长 uint8 中的 JSON 字符串
│   └── ee_twist                  # 具体字段取决于任务传感器
├── env_states/
├── rewards
├── terminated
├── truncated
├── success
├── fail
└── obs_scene
```

字典型 action sensor 会先被 JSON 序列化，再填充到固定长度的 `uint8` 数组，因此读取
`actions/*` 后还需要去掉末尾 `\x00` 并按 UTF-8/JSON 解码。RGB/depth 帧在 batch tensor
化之前写成 MP4，并从 HDF5 的大张量中移除，以降低内存峰值。

### 10.3 动作对齐的重要细节

HDF5 中的 `actions/*` 主要来自 observation 内的 action sensors，而不是直接序列化 `task.action_cache`：

- reset 时还没有 action，因此第 0 帧动作是 padding/空字典；
- 普通 `step()` 后的 action sensor 对应刚执行的命令；
- terminal 帧的部分 commanded-joint 传感器会返回 sentinel 空值；
- `step_chunk()` 只在 chunk 最后采样一次观测，中间动作不会各自形成 HDF5 时间行，保存的 `LastActionSensor` 只反映 chunk 的最后一个动作。

所以对于 action-chunk 策略，落盘的动作序列并不完整表示 chunk 内每个低层命令，且 `fps = 1000 / policy_dt_ms` 也不能直接解释为 HDF5 行的真实采样频率。用于训练前应先确认时序和 padding 约定。

### 10.4 成功过滤

- `filter_for_successful_trajectories=True`：主目录只保留成功 episode；失败 episode 有约 1% 概率写到并列 `debug/` 目录；
- `False`：成功和失败都写主目录；
- `end_on_success=True`：成功后立即停止，而不是继续跑到 horizon；
- episode 最终成功值由 `task.judge_success()` 决定。

---

## 11. `scripts/datagen/` 各文件的角色

| 文件 | 作用 | 是否直接生成动作轨迹 |
|---|---|---|
| `README.md` | 通用数据生成说明 | 否 |
| `run_pipeline.py` | 参数化构建配置、切换 planner/PI/teleop、运行 rollout | 是 |
| `run_rby1_nav_pick.py` | 串联 A* 导航和 CuRobo 抓取 | 是，两个独立 episode |
| `eval_policy_example.py` | 加载冻结配置并示范自定义 runner/config patch | 可运行 rollout，但偏评测示例 |
| `phone_teleop.md` | 手机遥操作说明 | 否 |
| `robot_conversion_patches.py` | 将旧 Franka/DROID 冻结配置修补为 RUM 配置 | 否 |
| `combine_trajs_into_h5.py` | 合并已有轨迹文件 | 否 |
| `compare_configs.py` | 比较两个 pickled Pydantic 配置 | 否 |
| `print_configs.py` | 检查/打印 benchmark 配置信息 | 否 |
| `fetch_assets.py` | 安装场景、机器人、抓取库等资源 | 否 |
| `upload_videos_to_wandb.py` | 上传已有 episode 视频 | 否 |

---

## 12. 当前代码中值得注意的问题

### 12.1 `run_pipeline.py --policy rum` 当前不可用

`get_policy_config("rum")` 返回 `RumPolicyConfig()`，但当前仓库中没有该类的定义或导入；只有底层 `RUMClient`。因此该选项会触发 `NameError`，不能算作已接通的轨迹生成方法。

### 12.2 手机遥操作文档与默认配置不一致

如第 8 节所述，`--policy teleop` 默认选择 keyboard，不会根据 `phone_teleop.md` 自动选择 phone。

### 12.3 `scripts/datagen/README.md` 的部分输出描述已落后于实现

README 示例写有 `config.json`，但当前 `MlSpacesExpConfig.save_config()` 实际写入的是 `experiment_config_<timestamp>.pkl`。分析或自动扫描输出时应以当前实现为准。

### 12.4 `run_pipeline.py` 会强制单 worker

无论原始配置如何，脚本都会设置 `exp_config.num_workers = 1`。要进行正式多进程规模化生成，应优先使用注册配置入口，或修改该覆盖逻辑。

### 12.5 `terminated` 字段存在键名不一致

`prepare_episode_for_saving()` 写入的键名是 `terminals`，但 `save_trajectories()` 当前检查的是
`terminateds`。因此保存器通常走 fallback，生成“仅最后一帧为 True”的 `terminated` 数组，
而不是直接使用 task 缓存的 terminal 序列。若下游依赖精确终止原因，应先修复或显式校验该字段。

### 12.6 “场景随机化”不等于新的轨迹算法

灯光、纹理、动力学、相机和机器人初始位姿随机化会让同一策略产生多样轨迹，但真正决定动作的仍是 planner、学习策略或人类输入。动作噪声同理，是扰动机制而不是新的规划器。

---

## 13. 如何选择方法

| 目标 | 建议方法 |
|---|---|
| 快速生成 Franka/RUM 抓取专家数据 | 几何 planner + IK |
| 生成 Franka pick-and-place/open/close 分阶段示教 | 对应的 manipulation planner config |
| RBY1 高自由度避障抓取或开门 | CuRobo planner，优先使用注册的 RBY1 配置 |
| 从远处导航到目标物体 | A* smooth planner |
| 评估视觉语言动作模型 | PI/其他 learned policy，并检查 action chunk 时序 |
| 收集人类示教 | keyboard/SpaceMouse/phone teleop |
| 测试保存链路或控制稳定性 | Dummy/Brownian motion |

总体上，正式数据生成推荐采用：

```text
注册的 DataGenConfig + ParallelRolloutRunner
```

`scripts/datagen/run_pipeline.py` 更适合本地调试、策略切换和冻结 benchmark 评测；`run_rby1_nav_pick.py` 则是面向“同场景、同目标”的两阶段专项工具。
