# `mlspaces_tests` 测试体系解析与 Holodeck-Objaverse 测试建议

> 分析对象：`/home/wenyifan/wenyifan/IndoorGen/molmospaces/mlspaces_tests`  
> 分析日期：2026-08-18  
> 说明：本文基于源码静态分析、当前资产目录检查，以及一次 `pytest --collect-only` 尝试。文中的“输出”既包括被测函数的返回值，也包括测试生成的文件和判定结果。

## 1. 结论摘要

`mlspaces_tests` 实际包含三类内容，不能把目录中所有脚本都当作普通 pytest：

1. **正式 pytest 测试**：14 个文件，共 **218 个逻辑测试函数/方法**（参数化后实际 case 数会更多）。
2. **场景批量质量检测 CLI**：`scenes/test_*.py` 中的穿透、稳定性、可拾取性、可开合性和性能测试；它们没有 pytest `test_*` 方法，主要通过命令行运行并写 JSON/TXT。
3. **基准数据生成器与探索脚本**：生成 `.npy`、PNG、HDF5 基准，或进行 iTHOR 抓取、开合、质量估计和可视化调试。

对当前目标场景 **`holodeck-objaverse`**：

- 最合适的现成测试是 `scenes/` 下五个支持 `--dataset holodeck-objaverse` 的批量脚本：
  **加载/性能、深穿透、静态稳定性、拾取物体提升、关节开合**。
- 正式 pytest 中，**唯一直接以 Holodeck-Objaverse 为输入的套件**是
  `test_json_eval_task_sampler.py` 和 `test_json_benchmark_integration.py` 使用的 10 条 val benchmark。
- `test_randomization.py` 固定为 iTHOR house 8，`test_thormap.py` 只支持 iTHOR/ProcTHOR，现状下不能直接覆盖 Holodeck。
- Franka/RUM/RBY1 的常规数据生成回归测试大多被测试配置覆盖成 iTHOR 或 ProcTHOR，并不等价于 Holodeck 回归。

## 2. 目录结构与测试数量

| 子目录 | 定位 | pytest 逻辑测试数 | 主要依赖 |
|---|---:|---:|---|
| `component_tests/` | 相机鱼眼变换、运动学单元/组件测试 | 71 | NumPy、PyTorch、Warp、MuJoCo、SciPy、scikit-image |
| `data_generation/` | Franka/RUM 数据生成、随机化、地图、JSON benchmark | 113 | MuJoCo、渲染器、场景/机器人/测试基准资产 |
| `data_generation_curobo/` | RBY1 + CuRobo 数据生成和配置 | 34 | CUDA、CuRobo、RBY1/场景/测试基准资产 |
| `scenes/` | 数据集级物理质量检测和历史探索脚本 | 0（不是 pytest 用例） | MuJoCo；部分脚本还需 JAX/MJX、OpenCV、pandas、matplotlib |

参数化会增加收集 case，例如：

- `test_thormap`：1 个逻辑方法展开为 4 个 case。
- RBY1M 双臂 IK：左右臂分别展开。
- Warp 运动学会按可用设备展开为 CPU，以及存在时的 CUDA。

## 3. 测试输入/输出的共同约定

### 3.1 外部回归基准

数据生成测试不是完全自包含的。测试从资源管理器软链接目录读取：

```text
<MLSPACES_ASSETS_DIR>/test_data/<source>/
```

当前版本固定在 `molmo_spaces/molmo_spaces_constants.py`：

| source | 当前版本 |
|---|---|
| `franka_pick` | `20260610` |
| `franka_pick_and_place` | `20260529` |
| `rby1_door_opening` | `20260812_2` |
| `rby1_pnp` | `20260610` |
| `rum_open_close` | `20260305` |
| `rum_pick` | `20260209` |
| `test_randomized_data` | `20251209` |
| `thormap` | `20251209` |

基准内容包括初始/动作后相机 `.npy`、最终关节位置 `.npy`、期望 HDF5、占据图 PNG，以及随机化 metadata/NPY/PNG。

### 3.2 常见输入

- 固定 seed、固定 house index、单环境、单 episode。
- MuJoCo XML 场景及其引用的 mesh/texture/object/robot/grasp 资产。
- 机器人初始 `qpos`、相机配置、任务采样器配置和 policy 配置。
- 保存的视觉/状态/HDF5 回归基准。

### 3.3 常见输出

- pytest 的 pass/fail/skip 和断言误差。
- `mlspaces_tests/data_generation/test_output/` 或
  `mlspaces_tests/data_generation_curobo/test_output/` 下的 HDF5/视频输出。
- `test_debug_images/` 下的视觉差异、深度误差或占据图差异图。
- 两个 data generation `conftest.py` 在会话结束时写入 `profiling_results.txt`。
- `scenes/` 批量脚本在仓库根目录或当前目录写结果 JSON、错误/警告文件和汇总 TXT。

## 4. `component_tests`：组件级测试

### 4.1 `test_fisheye_warping.py`（28 个逻辑测试）

| 测试组 | 测试内容 | 输入 | 期望输出/判定 |
|---|---|---|---|
| CameraIntrinsics | 相机内参形状、结构、主点、焦距公式 | GoPro FOV、高宽常量 | `3x3` 内参；主点居中；焦距符合投影公式 |
| DistortionParameters | 默认/随机畸变参数 | `k1..k4`、随机因子 | key 完整；随机参数至少一项改变 |
| DistortionGrid | 畸变采样网格 | 内参、GoPro 分辨率、CPU/CUDA device | `[1,H,W,2]`，无 NaN/Inf，设备正确 |
| ImageWarping | 单帧、batch、预计算 grid、裁剪/缩放、异常输入 | `[B,3,H,W]` float tensor | 输出尺寸/值域正确；缺参数或错误通道/尺寸时报错 |
| VideoWarping | 多帧鱼眼变换 | `[T,H,W,3]` uint8 | `[T,H2,W2,3]`、uint8、0–255 |
| PointWarping | 像素点映射 | 图像中心点、内参和畸变参数 | 返回整数坐标且落在合理中心区域 |
| Consistency | 在线生成 grid 与预计算 grid 一致 | 同一张程序生成的网格图 | `torch.allclose` |
| VisualRegression | 与保存 PNG 做视觉回归 | `fisheye_reference_grid.png`、`fisheye_expected_warped.png` | 尺寸正确，SSIM `> 0.99`；基准不存在时 skip |

参考图由 `generate_fisheye_reference_images.py` 生成到
`component_tests/test_data/`。当前仓库中该目录不存在，因此两项视觉回归会 skip。

### 4.2 `test_kinematics.py`（43 个逻辑测试）

测试两个求解器：

- `MlSpacesKinematics`：基于 MuJoCo。
- `SimpleWarpKinematics`：Warp 并行运动学，按 CPU/CUDA 参数化。

覆盖 Franka 与 RBY1M：

| 能力 | 输入 | 输出/判定 |
|---|---|---|
| FK | robot config 的初始 qpos、单位或偏移 base pose | 各末端 `4x4` 齐次变换；旋转正交、行列式为 1 |
| FK 灵敏度 | 扰动 arm/base qpos | 世界系末端姿态应改变；base-relative FK 应保持相应不变性 |
| IK 可达目标 | 从 FK 结果平移 2–4 cm 的目标 | 返回关节解；FK 回代位置误差约 `1e-3`～`2e-3` |
| IK 不可达目标 | `[10,10,10]` | 返回 `None` |
| base pose | 平移并 yaw 45° 的 base | 世界系/base-relative 变换一致 |
| batch | 3/4 个目标与 qpos batch | batch 长度正确，每项收敛 |
| 状态完整性 | 仅开放 arm move group | 非执行组（如 gripper）不改变 |
| 缓存污染 | 连续输入不同 qpos/target | 每次返回对应新结果，再输入原值能恢复原结果 |
| 求解器一致性 | 同一 Franka/RBY1M qpos/target | MuJoCo 与 Warp FK 接近，IK 均回代到目标 |

这部分不依赖具体室内场景，因此适合作为所有场景测试之前的快速基础门禁。

## 5. `data_generation`：数据生成与 benchmark 测试

### 5.1 配置结构：`test_datagen_configs.py`（22 个）

对象为 Franka pick（DROID/随机相机）、pick-and-place、open，以及注册表内 Franka/RUM configs。

输入是 Pydantic config 实例；测试：

- 可实例化且继承 `MlSpacesExpConfig`。
- 具备 experiment 层必需字段和五类 sub-config。
- 主 config 与 sub-config 不允许重复字段。
- 字段名模糊相似度达到 `0.76` 时判为容易混淆。
- `policy_dt_ms` 必须整除 `ctrl_dt_ms`，后者必须整除 `sim_dt_ms`。
- `output_dir`、`tag` 与任务类型匹配。
- 注册 config 的 `output_dir` 和 `tag` 全局唯一。

输出只有 pass/fail，不运行仿真。

### 5.2 Franka pick：`test_franka_pick.py`（11 个）

两套输入配置：DROID 固定相机和 randomized 相机。测试配置固定 seed、house 8、`TissueBox`、单任务、6-step horizon、关闭动作/纹理/关节噪声；基础数据集是 **ProcTHOR-10K**，不是 Holodeck。

测试链：

1. import 和 policy config 属性。
2. task sampler 内部计数、任务类型、相机分辨率 `(624,352)`、控制周期。
3. reset 时相机数组与 `.npy` 基准比较。
4. policy 执行 10 步，验证 9 维 qpos 确实改变并与最终 qpos 基准比较。
5. 动作后相机与基准比较，并验证画面发生变化。
6. `ParallelRolloutRunner` 跑 1 条轨迹，输出 HDF5/视频并检查 FPS。
7. Franka pick 的 HDF5 会与基准递归比较；目前临时忽略 `object_image_points`，因为新旧 HDF5 格式不同。

### 5.3 Franka pick-and-place：`test_franka_pick_and_place.py`（19 个）

输入为 DROID RGB 配置和 GoPro + D405/D455 配置，同样固定 house 8、`TissueBox`、单任务、关闭噪声。

除 sampler/视觉/qpos/runner 回归外，额外测试：

- DROID 配置只有 2 个 RGB camera，不应含 depth camera。
- GoPro 配置有 3 个 camera，只有 wrist D405 开启 depth。
- depth observation 是二维 `float32`，非负、无 NaN/Inf，距离范围合理。
- 已知浮点深度经 RGB 编解码最大误差 `< 1e-5`。
- 两帧深度经 MP4 round-trip 后，mean error `< 3 mm`，P95 `< 6 mm`。
- 动作后的深度应变化，但大幅变化像素比例不能异常地超过 80%。

输出为测试 HDF5/视频及可选深度调试图。这里的 integration 主要检查文件和 FPS，不像 Franka pick 那样完整比较 HDF5 内容。

### 5.4 RUM：`test_rum_pick.py`（7 个）与 `test_rum_open_close.py`（14 个）

| 套件 | 实际场景输入 | 相机 | 核心输出/判定 |
|---|---|---|---|
| RUM pick | ProcTHOR-10K house 8、`TissueBox` | 2 个 exocentric，`624x352` | 9 维 floating base/gripper qpos、初始/动作后视觉、单条 HDF5/FPS |
| RUM open | iTHOR house 8、drawer，初始 open%=0 | exocentric | policy 可实例化、关节运动、视觉回归、单条 HDF5/FPS |
| RUM close | iTHOR house 8、drawer，初始 open%=0.5 | exocentric | 同上，目标为 close |

虽然生产 `RUMPickDataGenConfig` 默认是 Holodeck-Objaverse，但测试类显式覆盖成
`procthor-10k`，因此现有 RUM pick pytest **不是 Holodeck 测试**。

### 5.5 随机化：`test_randomization.py`（3 个）

输入：iTHOR house 8、seed 3，以及 `test_randomized_data` 中保存的 baseline。

| 测试 | 输入状态 | 处理 | 输出/判定 |
|---|---|---|---|
| dynamics | mass/inertia/friction baseline | 随机化所有带 joint 的物体 | 至少一个物体改变，且不存在应改变却完全未改变的物体 |
| texture | material/RGBA 和 camera PNG baseline | category texture/material randomization | geom 有变化；相机图改变；SSIM `< 0.95` |
| lighting | light 属性 baseline | 同 seed 再随机化 | 所有 light 的 pos/dir/specular/ambient/diffuse/active 精确复现 |

失败时会在 `test_output/randomization/` 保存随机化图像。该测试当前不能通过参数切换到 Holodeck。

### 5.6 THOR map：`test_thormap.py`（1 个逻辑测试，4 个 case）

输入：ProcTHOR-10K house 8 和 iTHOR house 8；各测试原始与按 agent radius `0.25 m` 膨胀的 occupancy map，分辨率 `100 px/m`。

输出：现场生成 occupancy 与保存 PNG 的 IoU，必须 `> 0.98`；失败时输出当前、期望和 XOR difference 图。映射 class 只有 `ProcTHORMap` 与 `iTHORMap`，未覆盖 Holodeck。

### 5.7 JSON sampler：`test_json_eval_task_sampler.py`（21 个）

输入是仓库内 `data_generation/test_benchmark/benchmark.json` 的 10 条 Holodeck val episode。

测试内容：

- 从字符串动态 import Pick/PickAndPlace/Opening task，及三类错误路径。
- dict/Pydantic 的 robot-mounted 与 exocentric camera spec 转为 camera config。
- 解析 episode 必需字段、7 元 base pose、camera、task class、source、scene modifications。
- 所有 episode 的 task class 可导入、camera config 可构造。
- 绕过 sampler 的昂贵初始化，直接验证 camera config 和 task class 构建。
- 两项“print”测试只打印 episode/benchmark 摘要，断言价值较弱。

输出是内存中的 `EpisodeSpec`/camera config 和终端摘要，不启动完整仿真。

### 5.8 JSON benchmark 端到端：`test_json_benchmark_integration.py`（15 个）

输入 benchmark 的 house index 为：

```text
1194, 1603, 2853, 3491, 3681, 4558, 4587, 5294, 7142, 8226
```

均为 `scene_dataset=holodeck-objaverse`、`data_split=val`。为缩短测试，仿真 fixture 只取第一条 house 1194，使用 Franka + `DummyPolicy`、单 worker、5 个 policy step（HDF5 中预期 6 帧）。

输出/判定：

- `run_evaluation()` 返回 `EvaluationResults`，总数大于 0。
- 每 episode 有 house、bool success、非负 step 数。
- 输出目录包含 HDF5。
- DummyPolicy 不移动，期望 success count 为 0。
- 从 HDF5 解码首帧 qpos，逐 move group 对比 benchmark 初始值。
- `obs_scene.object_name` 等于 benchmark 的 `pickup_obj_name`。
- 另验证完整 benchmark 共 10 条、第一条字段/camera/task class/qpos 固定值。

模块只用 `holodeck-objaverse-val` **目录是否存在**决定 skip；它不验证目标 XML 是否已链接。因此只有 metadata 目录、没有所需 house XML 时，可能不会 skip 而是在执行阶段失败。

## 6. `data_generation_curobo`：RBY1/CuRobo

### 6.1 `test_datagen_configs.py`（22 个）

与普通 config 测试规则相同，但增加 `DoorOpeningDataGenConfig`，并对注册表内**所有** config 检查唯一 output directory/tag。输入只为 config object，输出为 pass/fail。

### 6.2 `test_rby1_pnp.py`（6 个）

- 输入：`RBY1PickAndPlaceDataGenConfig`、house 0、seed 0、单任务、关闭 action noise、CuRobo 本地 planner（`server_urls=[]`）。基础场景配置不是 Holodeck 专项。
- 相机：`head_camera`、左右 wrist RGB/深度，期望 RGB shape `(1024,576,3)`。
- policy 跑 10 步；输入/输出 qpos 均为 29 维，最终值与 `.npy` 基准误差 `< 0.05`。
- integration 输出 `house_0/trajectories_batch_1_of_1.h5` 并验证视频 FPS。

### 6.3 `test_rby1_door_opening.py`（6 个）

- 输入：`DoorOpeningDataGenConfig`、ProcTHOR-10K house 22、seed 4734、单任务；CuRobo mesh/OBB collision cache 被显式放大。
- 输出：三相机视觉回归、29 维 qpos（基准误差 `< 0.02`）、动作后视觉、单条 HDF5/FPS。
- 需要 CUDA/CuRobo，属于高成本测试。

## 7. `scenes`：数据集级场景质量测试

### 7.1 五个适合 Holodeck 的核心 CLI

这些脚本都接受单个 XML，或 `--dataset/--split/--houses-folder/--start/--end` 的批量输入；都支持 `holodeck-objaverse`。

| 文件 | 测试什么 | 主要输入 | 输出 |
|---|---|---|---|
| `test_runtime_performance.py` | XML 能否解析/编译/step；场景规模、contacts、constraints、MuJoCo timer、实时因子 | XML、timestep（默认 2 ms）、nstep（默认 5000） | `history_runtime_test_*.json`、errors；compare 模式输出 2 张 PNG。warning 虽被收集，但当前没有写入结果 |
| `test_penetration_objects.py` | settle 1 s 后接触距离 `< -0.01 m` 的深穿透 | XML、free bodies、contacts | `penetration_test_results_*.json`，记录 body/geom pair、penetration dist、category，以及 errors；定义了 warnings 文件名但当前未写入 |
| `test_stability_mp.py` | 无外力 settle 后的漂移、jitter、关节自发运动 | free body、hinge/slide joint；默认 settle 1 s、monitor 3 s | `stability_test_results_*.json`：body 位移、累计抖动、joint qpos/open%、相关容器内物体 |
| `test_lift_force_mp.py` | 标记为 pickupable 的 free body 是否能被合理向上力移动至少 5 cm | 质量、重力，力裁剪至 0.1–30 N，默认模拟 1 s | `history_lift_force_test_*.json`：成功/失败物体、mass、升高、移动距离、bad-qacc/warnings |
| `test_articulation_force_mp.py` | hinge/slide joint 在 20 N/Nm 下能否走过至少 66.7% range | 所有关节或上次失败关节，默认 monitor 2 s | `history_articulation_force_test_*.json`：可/不可开 joint、open%、阻挡物体、warnings |

重要语义：这些脚本将“场景成功处理”和“物理质量全部通过”分开。一个场景存在深穿透、漂移或不可开关节时，通常仍会成功写 JSON，进程也可能返回 0；真正的质量门禁需要读取 JSON 中的 `total`、失败列表或 `all_pass`，不能只看 shell exit code。

### 7.2 可视化配套

| 文件 | 输入 | 输出/用途 |
|---|---|---|
| `test_lift_force_vis.py` | 单场景、单 body 的交互式 MuJoCo viewer | 手动施力，打印 lift success 和高度差 |
| `test_articulation_force_vis.py` | 单场景、单 joint | 手动施力，打印 open success、起止 qpos/open% |
| `test_utils.py` | 上述 JSON | 排序/汇总、失败 category 统计，生成 `stats_fails_*.txt` |

### 7.3 历史/探索性脚本（主要是 iTHOR）

| 文件 | 作用 | 典型输入/输出 |
|---|---|---|
| `gripper_teleop.py` | gripper mocap teleoperation/录制控制器 | MuJoCo model/data；视频、metadata，mocap NPZ 保存目前被注释 |
| `ithor_artiuclate_test.py` | RUM gripper 沿路径批量测试 handle 开合 | iTHOR floorplan/handle；每 handle 轨迹/视频、success metric JSON、聚合 JSON/力直方图 |
| `ithor_artiuclate_test_mjx.py` | 上述流程的 MJX batch/GPU 版本 | floorplan、batch size、GPU/video 选项；MJX success JSON/视频 |
| `ithor_artiuclate_test_mjx_simple.py` | 简化 MJX batch 版本 | floorplan、batch size；`success_metric_mjx_simple.json` |
| `ithor_egg_test.py` | 对 egg/目标物枚举大量 grasp | scene、grasp 集合、进程数；碰撞/可拾取计数 JSON、可选视频 |
| `ithor_envs_test.py` | 多场景稳定性和帧率校准 | 硬编码场景集合/时序；CSV 与多张 histogram/分析 PNG |
| `ithor_force_test.py` / `_orig.py` | 逐关节增加力，统计开启所需 force 与随后 qvel | iTHOR XML；forces/qvel/failed joints JSON 和 PNG |
| `ithor_grasp_test.py` | grasp pose 放置、闭合、提升，测试 pickable | iTHOR scene/grasp assets；成功/失败列表、视频、results JSON |
| `ithor_grasp_articulate_test.py` | grasp articulable object 后开合 | iTHOR scene/articulation grasps；抓取/提升成功失败 JSON、视频 |
| `ithor_object_mass.py` | 仿真提升力与估计质量关系 | movable objects；质量散点图 |
| `ithor_object_mass_est.py` | 旧的质量估计实验 | 顶层实验代码，无标准化测试输出 |
| `model_tweaker.py` | 批量修正 dresser XML offset，并记录 tweak 状态 | houses folder；原地改 XML 和 status JSON |

这些脚本多有硬编码路径、顶层执行或交互 viewer，不宜直接作为 CI，也不适合原样套用到 Holodeck。

## 8. 基准生成脚本（不是测试）

| 脚本 | 生成输入 | 生成输出 |
|---|---|---|
| `component_tests/generate_fisheye_reference_images.py` | 程序生成 GoPro 网格图和固定畸变参数 | 两张视觉回归 PNG |
| `generate_test_data_pick.py` | DROID/randomized Franka pick 固定 episode | 初始/动作后 camera `.npy`、最终 qpos、基准 HDF5/视频目录 |
| `generate_test_data_pick_and_place.py` | DROID/GoPro pick-and-place | RGB/depth `.npy`（depth 存 float16）、最终 qpos |
| `generate_test_data_rum_pick.py` | 固定 RUM pick episode | 两个 exo camera 的初始/动作后 `.npy`、最终 qpos |
| `generate_test_data_rum_open_close.py` | 固定 RUM open/close | 初始/动作后 camera `.npy`、最终 qpos |
| `generate_test_data_randomization.py` | iTHOR house 8 原始及 seed 3 随机化状态 | dynamics/texture/lighting `.npy`、camera PNG、metadata JSON |
| `generate_test_data_thormap.py` | iTHOR/ProcTHOR house 8 | 原始/膨胀 occupancy PNG |
| CuRobo 两个 `generate_test_data_rby1_*.py` | RBY1 PnP house 0 / door house 22 | 三相机初始/动作后 `.npy`、29 维最终 qpos |

存在一个文档/代码路径差异：`mlspaces_tests/README.md` 描述输出到
`mlspaces_tests/test_data/<...>`，而当前生成脚本实际写到各自子目录下的
`data_generation/test_data/<...>` 或 `data_generation_curobo/test_data/<...>`；正式测试则读取资源管理器的 `<MLSPACES_ASSETS_DIR>/test_data/<...>`。更新基准时需要明确执行“本地生成 → 上传 → constants 固定版本 → 资源管理器安装”的流程。

## 9. 当前 Holodeck-Objaverse 资产状态

当前工作区 `data/assets/scenes` 中：

- `holodeck-objaverse-train` 有 4 个 XML：`train_0.xml`、`train_199.xml` 及各自 ceiling 版本，即 2 个可直接测的基础场景。
- `holodeck-objaverse-val` 目录存在 metadata/build settings，但当前目录扫描未发现 XML。
- Objaverse object 资产和通用 `test_data` 基准已存在。

因此：

- 立即可对 `train_0.xml`、`train_199.xml` 做五类 scene CLI smoke/quality test。
- JSON benchmark pytest 需要 val houses 1194 等被资源管理器正确链接/懒加载；仅目录存在不代表测试输入已经就绪。

## 10. 针对 Holodeck-Objaverse 的推荐测试方案

### 10.1 第一层：基础组件门禁

先跑不依赖 Holodeck 的组件测试，排除相机/运动学基础问题：

```bash
source setup_env.sh
.venv/bin/python -m pytest mlspaces_tests/component_tests -q
```

### 10.2 第二层：单场景加载与性能 smoke

```bash
source setup_env.sh
SCENE=data/assets/scenes/holodeck-objaverse-train/train_0.xml

.venv/bin/python mlspaces_tests/scenes/test_runtime_performance.py \
  --dataset holodeck-objaverse --split train \
  --model "$SCENE" --nstep 500 --identifier holodeck_smoke
```

该测试最适合先确认 XML/mesh/texture 能编译，MuJoCo 能稳定 step，并记录实时因子、contact/constraint 数。它当前只记录性能，没有 pass/fail 阈值；CI 中应补充例如 `real_time_factor` 下限和禁止 BADQACC/BADQPOS 的断言。

### 10.3 第三层：物理质量四件套

```bash
source setup_env.sh
SCENE=data/assets/scenes/holodeck-objaverse-train/train_0.xml

.venv/bin/python mlspaces_tests/scenes/test_penetration_objects.py \
  --dataset holodeck-objaverse --split train --house "$SCENE" \
  --identifier holodeck_smoke

.venv/bin/python mlspaces_tests/scenes/test_stability_mp.py \
  --dataset holodeck-objaverse --split train --house "$SCENE" \
  --identifier holodeck_smoke

.venv/bin/python mlspaces_tests/scenes/test_lift_force_mp.py \
  --dataset holodeck-objaverse --split train --house "$SCENE" \
  --identifier holodeck_smoke

.venv/bin/python mlspaces_tests/scenes/test_articulation_force_mp.py \
  --dataset holodeck-objaverse --split train --house "$SCENE" \
  --identifier holodeck_smoke
```

建议质量门禁：

1. **必须无处理错误和 MuJoCo bad-qpos/bad-qvel/bad-qacc。**
2. **深穿透 pair 数为 0**，或只允许经过人工维护的 allowlist。
3. **稳定性失败为 0**：free body 漂移 `< 5 cm`、累计 jitter `< 10 cm`、关节自发移动 `< 10% range`。
4. **所有被识别的 pickupable body 可提升 5 cm**；同时检查“识别数量不能为 0”，避免 Objaverse 命名未被 `ALL_PICKUP_TYPES_ITHOR` 命中而形成假通过。
5. **所有 articulable joint 达到 66.7% range**；容器内物体导致阻塞时单独分类，不应简单归因于关节资产错误。

### 10.4 第四层：小批量数据集回归

当前只有 house 0 和 199，可执行：

```bash
source setup_env.sh
HOUSES=data/assets/scenes/holodeck-objaverse-train

.venv/bin/python mlspaces_tests/scenes/test_penetration_objects.py \
  --dataset holodeck-objaverse --split train --houses-folder "$HOUSES" \
  --start 0 --end 200 --max-workers 2 --identifier holodeck_subset
```

对 stability/lift/articulation/runtime 采用同样的 `--houses-folder --start --end`。注意目录扫描会排除或跳过部分 ceiling/orig/non-settled 变体，各脚本的过滤细节略有不同。

### 10.5 第五层：Holodeck benchmark 端到端

资源安装并确认 val house 1194 可用后：

```bash
source setup_env.sh
.venv/bin/python -m pytest \
  mlspaces_tests/data_generation/test_json_eval_task_sampler.py \
  mlspaces_tests/data_generation/test_json_benchmark_integration.py \
  -v -s
```

这是当前最接近真实 Holodeck 任务路径的测试：benchmark JSON → camera/task/robot config → scene modification → task sampler → MuJoCo → HDF5。但 DummyPolicy 只验证“任务应失败”，没有验证真正的 pick 成功；建议后续增加一个固定 planner/checkpoint 的成功 episode，至少验证一条可完成轨迹。

### 10.6 建议新增的 Holodeck 专项 pytest

按优先级建议补充：

1. **`test_holodeck_scene_smoke.py`**：参数化 house 0/199，断言 XML 编译、`mj_forward`/100 step 无 warning/NaN，body/geom/camera 数大于 0。
2. **`test_holodeck_scene_quality.py`**：把 penetration/stability CLI 的核心函数包装成 pytest，并直接断言失败计数，而不是只写 JSON。
3. **`test_holodeck_metadata_consistency.py`**：每个 Objaverse body 的 metadata、mesh、material/texture、free joint、category/synset 可解析，引用文件存在。
4. **`test_holodeck_occupancy.py`**：明确 Holodeck map class/加载逻辑，检查 occupancy 非空、robot spawn 可达、关键房间连通。
5. **`test_holodeck_randomization.py`**：将当前 hard-coded iTHOR randomization fixture 参数化为 dataset/house，并保存 Holodeck baseline。
6. **`test_holodeck_pick_regression.py`**：固定 house/object/base pose/grasp，不依赖随机 sampler，验证 reset 图像、规划、抓取和 HDF5 schema。
7. **性能阈值测试**：按 geom/contact 数分桶设 realtime factor 或 ms/step 上限，避免只记录不报警。

## 11. 当前可执行性与已发现问题

本次执行：

```bash
.venv/bin/python -m pytest --collect-only -q \
  mlspaces_tests/component_tests \
  mlspaces_tests/data_generation \
  mlspaces_tests/data_generation_curobo
```

结果是在发生 10 个 collection error 前收集到 85 个 case。错误来自当前受限执行环境没有先应用正确的资源环境，并且资源管理器尝试在只读 cache 目录创建 `.lock`；随后多个模块在计算出的 cache 路径找不到 `objects/thor/material-database.json`。这说明：

- 运行测试前必须 `source setup_env.sh`，确保 `MLSPACES_CACHE_DIR`、`MLSPACES_ASSETS_DIR`、`PYTHONPATH`、EGL 配置一致。
- 测试“收集”本身就会初始化资源管理器并读取 material database，不是纯粹的无副作用 collection。
- data generation suite 会在 collection/session finish 阶段写 profiling 文件；只读 CI 环境需要提供可写工作目录。
- CuRobo 测试还需要可用 CUDA/CuRobo；普通 `slow` marker 已注册，但当前昂贵测试并未普遍标注 `@pytest.mark.slow`，不能可靠地用 `-m 'not slow'` 排除。
- scene CLI 有少量“参数存在但没有生效”的迹象：例如 lift 脚本解析了 `--distance-to-move`，实际判定仍直接使用全局常量 `DISTANCE_TO_LIFT=0.05`；性能脚本收集 warning 后也没有放入 `ResultsCache`。将它们接入 CI 前应先修正或显式规避。

## 12. 推荐执行顺序

1. `component_tests`。
2. Holodeck 单 XML runtime smoke。
3. 单 XML penetration + stability。
4. 单 XML lift + articulation，并检查测试对象计数非零。
5. house 0/199 小批量回归。
6. 安装 benchmark val houses 后跑 JSON sampler 与端到端 evaluation。
7. 最后才跑完整 Franka/RUM/RBY1 数据生成回归和 CuRobo 测试。

这个顺序能最快区分：资产缺失/路径问题、XML 编译问题、基础动力学问题、交互属性问题，以及高层任务/数据管线问题。
