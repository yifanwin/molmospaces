# MolmoSpaces 现有训练轨迹生成脚本

本文整理当前 MolmoSpaces 工作区中可以生成训练轨迹的现成入口，并区分正式数据生成、调试工具、专项脚本、示例和测试数据生成器。


| 任务 | 配置 |
|---|---|
| Franka 抓取 | `FrankaPickDroidDataGenConfig`、`FrankaPickRandomizedDataGenConfig`、`FrankaPickOmniCamConfig` |
| Franka 抓取放置 | `FrankaPickAndPlaceDataGenConfig`、`FrankaPickAndPlaceDroidDataGenConfig` |
| 按颜色/相邻放置 | `FrankaPickAndPlaceColorDataGenConfig`、`FrankaPickAndPlaceNextToDataGenConfig` |
| Franka 开关物体 | `FrankaOpenDataGenConfig`、`FrankaCloseDataGenConfig` |
| RUM 抓取 | `RUMPickDataGenConfig` |
| RBY1 抓取 | `RBY1PickDataGenConfig` |
| RBY1 抓取放置 | `RBY1PickAndPlaceDataGenConfig` |
| RBY1 开启物体 | `RBY1OpenDataGenConfig` |
| RBY1 开门 | `DoorOpeningDataGenConfig` |
| RBY1 导航到物体 | `NavToObjDataGenConfig` |



## 1. 快速结论

真正能够执行完整 rollout，并生成 HDF5 轨迹和相机 MP4 的入口主要有：

| 入口 | 适用场景 | 推荐程度 |
|---|---|---|
| `python -m molmo_spaces.data_generation.main <ConfigName>` | 正式、批量生成训练数据 | 推荐 |
| `scripts/datagen/run_pipeline.py` | 快速调试、切换任务/机器人/策略、打开 viewer | 推荐用于调试 |
| `scripts/datagen/run_rby1_nav_pick.py` | 分别生成 RBY1 导航和 RBY1M 抓取轨迹 | 专项使用 |
| `examples/custom_assets/datagen.py` | 自定义场景和资产中的 Franka 抓取 | 示例 |
| `examples/add_robot/xarm7_datagen.py` | 自定义 XArm7 机器人的抓取轨迹 | 示例 |
| `scripts/datagen/eval_policy_example.py` | 在 frozen benchmark/config 上运行策略 rollout | 偏评测 |

如果需要正式生成训练集，优先使用注册的 `DataGenConfig` 和通用主入口。

---

## 2. 通用正式生成入口

入口文件：[`molmo_spaces/data_generation/main.py`](../molmo_spaces/data_generation/main.py)

运行方式：

```bash
export PYTHONPATH="${PYTHONPATH}:."
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python -m molmo_spaces.data_generation.main <ConfigName>
```

也可以通过 `module:ClassName` 指定自定义配置：

```bash
python -m molmo_spaces.data_generation.main \
  molmo_spaces.data_generation.config.door_opening_configs:DoorOpeningDataGenConfig
```

该入口会：

1. 加载注册的实验配置；
2. 创建 `ParallelRolloutRunner`；
3. 按 house 和 batch 分配 rollout；
4. 采样任务并执行策略；
5. 根据配置过滤成功轨迹；
6. 保存 HDF5、MP4、运行日志和实验配置。

### 2.1 现成任务配置

#### Franka 抓取

| 配置 | 说明 |
|---|---|
| `FrankaPickDroidDataGenConfig` | Franka 抓取，使用 DROID 相机配置 |
| `FrankaPickGoProD405D455DataGenConfig` | Franka 抓取，使用 GoPro、D405 和 D455 相机 |
| `FrankaPickRandomizedDataGenConfig` | Franka 抓取，使用随机化外部相机 |
| `FrankaPickOmniCamConfig` | Franka 抓取，使用 OmniCam 配置 |

配置位置：[`object_manipulation_datagen_configs.py`](../molmo_spaces/data_generation/config/object_manipulation_datagen_configs.py)

示例：

```bash
python -m molmo_spaces.data_generation.main FrankaPickDroidDataGenConfig
```

#### Franka 抓取放置

| 配置 | 说明 |
|---|---|
| `FrankaPickAndPlaceDataGenConfig` | 常规抓取放置 |
| `FrankaPickAndPlaceEasyDataGenConfig` | 使用较简单相机配置的抓取放置 |
| `FrankaPickAndPlaceDroidDataGenConfig` | 使用 DROID 相机的抓取放置 |
| `FrankaPickAndPlaceGoProD405D455DataGenConfig` | 使用 GoPro、D405 和 D455 相机 |
| `FrankaPickAndPlaceNextToDataGenConfig` | 将物体放到另一个物体旁边 |
| `FrankaPickAndPlaceNextToDroidDataGenConfig` | DROID 相机版本的相邻放置 |
| `FrankaPickAndPlaceColorDataGenConfig` | 根据颜色选择放置目标 |
| `FrankaPickAndPlaceColorDroidDataGenConfig` | DROID 相机版本的颜色放置 |

#### Franka 开启和关闭

| 配置 | 说明 |
|---|---|
| `FrankaOpenDataGenConfig` | 开启可动关节物体 |
| `FrankaCloseDataGenConfig` | 关闭可动关节物体 |

#### RUM 抓取

| 配置 | 说明 |
|---|---|
| `RUMPickDataGenConfig` | Floating RUM 抓取，使用几何规划器 |

#### RBY1/RBY1M 操作

| 配置 | 说明 |
|---|---|
| `RBY1PickDataGenConfig` | RBY1M 抓取，使用 CuRobo |
| `RBY1PickAndPlaceDataGenConfig` | RBY1M 抓取放置，使用双臂 CuRobo |
| `RBY1OpenDataGenConfig` | RBY1M 开启可动关节物体 |
| `DoorOpeningDataGenConfig` | RBY1M 在 ProcTHOR 场景中生成开门轨迹 |
| `DoorOpeningDebugConfig` | 单场景、单 house 的开门调试配置 |

开门配置位置：[`door_opening_configs.py`](../molmo_spaces/data_generation/config/door_opening_configs.py)

示例：

```bash
python -m molmo_spaces.data_generation.main RBY1PickDataGenConfig
python -m molmo_spaces.data_generation.main DoorOpeningDataGenConfig
python -m molmo_spaces.data_generation.main DoorOpeningDebugConfig
```

RBY1 操作配置通常依赖 CUDA GPU 和 CuRobo。

#### RBY1 导航

| 配置 | 说明 |
|---|---|
| `NavToObjDataGenConfig` | RBY1 使用 A* 导航到目标物体 |

配置位置：[`nav_to_obj_configs.py`](../molmo_spaces/data_generation/config/nav_to_obj_configs.py)

示例：

```bash
python -m molmo_spaces.data_generation.main NavToObjDataGenConfig
```

### 2.2 标准输出

输出目录通常采用以下结构：

```text
<output_dir>/<ConfigName>/<timestamp>/
├── running_log.log
├── experiment_config_<timestamp>.pkl
└── house_<N>/
    ├── trajectories_batch_<i>_of_<n>.h5
    ├── episode_<id>_<camera>_batch_<i>_of_<n>.mp4
    └── ...
```

HDF5 中通常包括：

- 机器人 `qpos` 和 `qvel`；
- commanded action 和其他动作传感器；
- reward、success、fail、terminated 和 truncated；
- TCP、目标物体和任务阶段等额外观测；
- 场景、任务、语言指令和冻结配置元数据；
- 相机内外参及对应视频路径。

更多说明参见 [`scripts/datagen/README.md`](../scripts/datagen/README.md)。

---

## 3. 参数化调试入口：`run_pipeline.py`

脚本：[`scripts/datagen/run_pipeline.py`](../scripts/datagen/run_pipeline.py)

该脚本可以动态组合任务、机器人、策略、场景和随机化参数，适合本地试跑和 viewer 调试。

典型命令：

```bash
python scripts/datagen/run_pipeline.py \
  --task_type pick \
  --robot franka \
  --policy planner \
  --scene_dataset ithor \
  --house_inds 1 \
  --samples_per_house 4 \
  --seed 2
```

打开 viewer：

```bash
python scripts/datagen/run_pipeline.py --viewer --seed 1
```

### 3.1 支持的任务

- `pick`
- `open`
- `close`
- `pick_and_place`
- `pick_and_place_color`
- `pick_and_place_next_to`
- `packing`
- `nav_to_obj`

### 3.2 支持的机器人

- `franka`
- `droid`
- `rum`
- `rby1`
- `yam`
- `bimanual_yam`

### 3.3 支持的策略参数

- `planner`：基于规则或运动规划的专家策略；
- `pi`：PI 学习策略；
- `teleop`：遥操作策略；
- `rum`：命令行中存在该选项，但当前代码引用了未定义的 `RumPolicyConfig`，无法直接运行。

### 3.4 常用参数

| 参数 | 作用 |
|---|---|
| `--task_type` | 选择任务 |
| `--robot` | 选择机器人 |
| `--policy` | 选择策略 |
| `--scene_dataset` | 选择场景数据集 |
| `--data_split` | 选择数据 split |
| `--house_inds` | 指定 house 索引 |
| `--target_types` | 限制目标物体类别 |
| `--samples_per_house` | 每个 house 采样数量 |
| `--filter_for_successful_trajectories` | 仅保存成功轨迹 |
| `--randomize_lighting` | 随机化光照 |
| `--randomize_textures` | 随机化纹理 |
| `--randomize_dynamics` | 随机化动力学参数 |
| `--randomize_scene` | 随机场景参数 |
| `--seed` | 设置随机种子 |
| `--run_name_prefix` | 设置输出目录前缀 |

### 3.5 使用限制

- 脚本会把 `num_workers` 强制设置为 `1`，因此更适合调试而不是大规模并行生成；
- `--policy rum` 当前不可用；
- `--policy teleop` 默认使用 keyboard，不会根据手机遥操作文档自动切换到 phone；
- 正式批量生成应优先使用注册配置和通用主入口。

---

## 4. RBY1 导航与抓取专项脚本

脚本：[`scripts/datagen/run_rby1_nav_pick.py`](../scripts/datagen/run_rby1_nav_pick.py)

> 注意：该文件目前位于工作区中，但尚未被 Git 跟踪。

该脚本串联两个独立的数据生成阶段：

1. 使用 `NavToObjDataGenConfig` 和 A* 生成 RBY1 导航 episode；
2. 从导航 HDF5 中读取目标实例；
3. 使用相同 house 和目标对象配置 `RBY1PickDataGenConfig`；
4. 使用本地 GPU CuRobo 生成 RBY1M 抓取 episode。

基本用法：

```bash
python scripts/datagen/run_rby1_nav_pick.py \
  --scene-dataset holodeck-objaverse \
  --house-index 0 \
  --target-types Cup
```

主要参数：

| 参数 | 作用 |
|---|---|
| `--scene-dataset` | 场景数据集 |
| `--data-split` | 数据 split |
| `--house-index` | house 索引 |
| `--house-variant` | house 变体 |
| `--target-types` | 目标类别 |
| `--nav-target-instance` | 指定导航目标实例 |
| `--nav-radius-min` | 导航目标最小采样半径 |
| `--nav-radius-max` | 导航目标最大采样半径 |
| `--nav-success-distance` | 导航成功距离阈值 |
| `--nav-waypoint-distance` | 导航 waypoint 距离 |
| `--require-same-instance` | 要求两阶段使用同一目标实例 |
| `--navigation-h5` | 复用已有导航 HDF5 |
| `--output-root` | 指定输出根目录 |
| `--allow-action-noise` | 允许动作噪声 |

默认输出结构：

```text
<output-root>/
├── navigation/
│   └── house_<N>/
│       └── trajectories_batch_1_of_1.h5
└── pick/
    └── house_<N>/
        └── trajectories_batch_1_of_1.h5
```

### 4.1 重要限制

该脚本生成的是两个独立 episode，不是一条物理连续的“导航后抓取”轨迹：

- 导航结束时的机器人状态不会传递给抓取阶段；
- 两阶段分别 reset 模拟器和任务；
- 默认只保证使用相同场景和目标类别；
- 启用对应参数后可以要求使用相同目标实例。

---

## 5. 自定义场景 Franka 抓取示例

脚本：[`examples/custom_assets/datagen.py`](../examples/custom_assets/datagen.py)

用途：

- 加载用户自定义 `scene.xml`；
- 注册自定义 asset library 和 grasp library；
- 使用 Franka 和 DROID 相机执行抓取；
- 对机器人及目标物体初始位姿进行随机化；
- 使用标准 rollout 保存 HDF5 和 MP4。

运行方式：

```bash
cd examples/custom_assets
python -m molmo_spaces.data_generation.main datagen:CustomAssetsDataGenConfig
```

需要从示例目录运行，以保证 `asset_library` 和 `scene.xml` 的相对路径正确。

相关教程：[`docs/tutorials/custom_assets.md`](tutorials/custom_assets.md)

---

## 6. 自定义机器人 XArm7 抓取示例

脚本：[`examples/add_robot/xarm7_datagen.py`](../examples/add_robot/xarm7_datagen.py)

用途：

- 演示新机器人如何接入数据生成系统；
- 使用 XArm7 执行抓取；
- 使用腕部 ZED Mini 和外部 ZED 2 相机；
- 在 ProcTHOR house 中生成完整 rollout。

运行方式：

```bash
cd examples/add_robot
python -m molmo_spaces.data_generation.main \
  xarm7_datagen:XArm7PickDataGenConfig
```

相关教程：[`docs/tutorials/add_robot.md`](tutorials/add_robot.md)

---

## 7. Frozen benchmark 策略评测入口

脚本：[`scripts/datagen/eval_policy_example.py`](../scripts/datagen/eval_policy_example.py)

运行方式：

```bash
python scripts/datagen/eval_policy_example.py \
  --eval /path/to/frozen/config-or-benchmark
```

该脚本会：

- 加载已有 frozen benchmark 或实验配置；
- 修补部分旧机器人配置；
- 关闭动作噪声；
- 执行完整 `ParallelRolloutRunner`；
- 保存轨迹和视频。

它可以生成完整轨迹，但主要用途是 policy evaluation，而不是从场景中构建常规训练集。

---

## 8. 测试数据生成器

以下脚本位于测试目录，主要用于更新视觉、关节状态或随机化回归基准，不建议直接用于构建训练集。

| 脚本 | 内容 | 完整 HDF5 轨迹 |
|---|---|---|
| `mlspaces_tests/data_generation/generate_test_data_pick.py` | Franka pick 回归数据 | 是，但仅用于测试 |
| `mlspaces_tests/data_generation/generate_test_data_pick_and_place.py` | Franka pick-and-place 图像和 qpos | 否 |
| `mlspaces_tests/data_generation/generate_test_data_rum_pick.py` | RUM pick 图像和 qpos | 否 |
| `mlspaces_tests/data_generation/generate_test_data_rum_open_close.py` | RUM open/close 图像和 qpos | 否 |
| `mlspaces_tests/data_generation_curobo/generate_test_data_rby1_pnp.py` | RBY1 CuRobo pick-and-place 回归数据 | 否 |
| `mlspaces_tests/data_generation_curobo/generate_test_data_rby1_door_opening.py` | RBY1 开门回归数据 | 否 |

其中 [`generate_test_data_pick.py`](../mlspaces_tests/data_generation/generate_test_data_pick.py) 会额外调用完整 rollout runner，但其配置是固定 house、固定 seed 和测试规模，仍然不适合直接用于正式训练数据生成。

---

## 9. 不生成训练轨迹的辅助脚本

以下工具不会执行完整任务 rollout：

| 脚本 | 作用 |
|---|---|
| `scripts/datagen/combine_trajs_into_h5.py` | 合并已有轨迹文件 |
| `scripts/datagen/upload_videos_to_wandb.py` | 上传已有轨迹对应的视频 |
| `scripts/datagen/compare_configs.py` | 比较已有实验配置 |
| `scripts/datagen/print_configs.py` | 打印实验配置 |
| `scripts/datagen/fetch_assets.py` | 下载运行所需资产 |
| `scripts/datagen/robot_conversion_patches.py` | 修补旧 frozen config 的机器人配置 |
| `mlspaces_tests/data_generation/generate_test_data_thormap.py` | 生成导航 occupancy map 图片 |
| `mlspaces_tests/data_generation/generate_test_data_randomization.py` | 生成随机化回归基准 |
| `examples/view_scene.py` | 加载并显示场景 |

这些脚本可能处理轨迹、场景或测试数据，但不会生成可直接用于训练的完整动作 trajectory。

---

## 10. 选择建议

| 目标 | 推荐入口 |
|---|---|
| 正式批量生成训练集 | `python -m molmo_spaces.data_generation.main <ConfigName>` |
| 快速验证任务、机器人和场景 | `scripts/datagen/run_pipeline.py` |
| Franka/RUM 抓取、放置、开关专家轨迹 | 对应的 manipulation `DataGenConfig` |
| RBY1 抓取、放置、开门 | RBY1/CuRobo `DataGenConfig` |
| RBY1 导航 | `NavToObjDataGenConfig` |
| 同场景导航和抓取两个独立数据集 | `scripts/datagen/run_rby1_nav_pick.py` |
| 自定义场景和资产 | `examples/custom_assets/datagen.py` |
| 接入并测试自定义机器人 | `examples/add_robot/xarm7_datagen.py` |
| Frozen benchmark 策略评测 | `scripts/datagen/eval_policy_example.py` |

正式生产推荐采用：

```text
注册的 DataGenConfig + ParallelRolloutRunner
```

`run_pipeline.py` 更适合本地调试；测试目录中的生成器应仅用于维护回归基准。
