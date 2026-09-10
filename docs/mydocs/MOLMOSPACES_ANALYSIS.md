# MolmoSpaces 仓库解析

> 分析对象：`/home/wenyifan/wenyifan/IndoorGen/molmospaces`
>
> 分析时间：2026-08-14
>
> 当前版本：`0.2.0`（以 `pyproject.toml` 为准）

## 1. 项目定位

MolmoSpaces 是一个基于 MuJoCo 的机器人仿真、操作与导航数据生态，目标是为视觉语言策略（VLM/VLA）提供：

- 室内场景、物体、机器人和抓取资源；
- 面向操作与导航任务的任务定义及随机采样；
- 基于规划器、遥操作和学习策略的数据生成；
- JSON benchmark 的策略评测与结果汇总；
- HDF5 轨迹、MP4 视频及后处理工具；
- 面向 Isaac Sim 和 ManiSkill 的资源转换/适配包。

项目本身更偏向“仿真数据生成与策略推理评测基础设施”，不是完整的策略训练框架。仓库内没有发现典型的 optimizer、backward、DataLoader 或 Trainer 训练主循环，学习策略主要通过 checkpoint、本地模型或远程服务进行推理。

依据：[`README.md`](README.md#L1-L13)、[`pyproject.toml`](pyproject.toml#L1-L20)。

## 2. 顶层结构

```text
molmospaces/
├── molmo_spaces/              # MuJoCo 主 Python 包
│   ├── configs/               # 实验、机器人、任务、相机、策略配置
│   ├── controllers/           # 关节位置、相对位置、速度等控制器
│   ├── data_generation/       # 配置注册与并行数据生成
│   ├── env/                   # MuJoCo 环境、相机、传感器、对象管理
│   ├── evaluation/            # JSON benchmark 与策略评测
│   ├── grasp_generation/      # 刚体/可动体抓取生成
│   ├── housegen/              # 从 JSON 生成 MJCF 房屋场景
│   ├── kinematics/            # FK/IK 及并行运动学
│   ├── planner/               # A*、cuRobo 与机器人专用规划器
│   ├── policy/                # 规划策略、随机策略、学习策略、遥操作
│   ├── renderer/              # OpenGL、Filament 和离线渲染
│   ├── robots/                # Franka、RBY1、RUM、YAM 等机器人
│   ├── tasks/                 # pick、open/close、packing、navigation 等任务
│   ├── utils/                 # 轨迹保存、资源、相机、线代及进程工具
│   └── molmo_spaces_constants.py # 资源路径和版本定义
├── molmo_spaces_isaac/        # Isaac Sim 资源转换适配包
├── molmo_spaces_maniskill/    # ManiSkill 资源适配包
├── scripts/                   # 下载、数据处理、评测、调试脚本
├── mlspaces_tests/            # 组件、数据生成、场景及 cuRobo 测试
├── docs/                      # MkDocs 文档和 API 文档
├── examples/                  # 添加机器人、自定义资产示例
├── bin/wheels/                # MuJoCo/Filament 等本地 wheel
├── pyproject.toml             # 打包、依赖、ruff 和入口配置
└── mkdocs.yml                 # 文档站点配置
```

完整模块职责可参考 [`docs/code_structure.md`](docs/code_structure.md#L7-L140)。

## 3. 核心抽象与依赖关系

### 3.1 配置层

顶层配置类是 `MlSpacesExpConfig`，聚合以下配置：

```text
MlSpacesExpConfig
├── camera_config
├── robot_config
├── task_sampler_config
├── task_config
├── policy_config
├── policy_dt_ms / ctrl_dt_ms / sim_dt_ms
├── task_horizon / seed
└── output_dir / profiler / wandb
```

配置类通过 `@register_config("名称")` 注册到全局 registry，数据生成入口会自动导入 `data_generation/config/*.py`，随后按命令行名称查找配置类。`MlSpacesExpConfig.model_post_init()` 会校验 `policy_dt_ms` 是 `ctrl_dt_ms` 的整数倍、`ctrl_dt_ms` 是 MuJoCo 仿真步长的整数倍。

关键位置：

- [`molmo_spaces/configs/abstract_exp_config.py`](molmo_spaces/configs/abstract_exp_config.py#L31-L109)
- [`molmo_spaces/data_generation/config_registry.py`](molmo_spaces/data_generation/config_registry.py#L14-L73)

### 3.2 机器人控制层

机器人采用三层抽象：

```text
Robot
└── RobotView
    └── MoveGroup（arm / gripper / base 等）
```

- `MoveGroup` 抽象 MuJoCo 关节、执行器、位姿、限制和 Jacobian；
- `RobotView` 按 move group ID 批量读写 qpos、ctrl 和 Jacobian；
- `Robot` 组合 `RobotView`、controllers 和 kinematics；
- `Controller` 把策略动作转换成 MuJoCo ctrl；
- 支持绝对关节位置、相对关节位置、关节速度等控制模式；
- 可对机械臂动作施加 TCP 空间噪声，再通过 Jacobian 伪逆映射回关节空间。

动作格式通常为：

```python
{"arm": arm_action, "gripper": gripper_action}
```

相关说明见 [`docs/concepts.md`](docs/concepts.md#L5-L147) 和 [`molmo_spaces/robots/abstract.py`](molmo_spaces/robots/abstract.py#L27-L307)。

### 3.3 环境、任务和采样器

三者职责明确：

```text
TaskSampler（拥有 Env 生命周期）
└── Env（MuJoCo 物理、渲染、机器人、相机、对象）
    └── Task（episode 交互、观测、奖励、成功判定）
```

- `CPUMujocoEnv` 编译场景，创建多个 `MjData`，初始化机器人、渲染器、相机和对象管理器；
- `BaseMujocoTask` 实现 Gymnasium 风格的 `reset()` / `step()`，维护观测、奖励、终止状态和轨迹缓存；
- `TaskSampler` 加载房屋、随机化物体/机器人/相机，并创建具体任务；
- 当前文档指出批量环境接口存在，但 batch 大于 1 尚未充分测试，实际应优先使用 `n_batch=1`。

关键位置：

- [`molmo_spaces/env/env.py`](molmo_spaces/env/env.py#L38-L228)
- [`molmo_spaces/tasks/task.py`](molmo_spaces/tasks/task.py#L37-L87)
- [`docs/concepts.md`](docs/concepts.md#L148-L248)

### 3.4 策略和规划

策略统一通过 `BasePolicy` 工作，主要类型包括：

- planner-based：抓取、放置、开关门/抽屉、导航等；
- learned policy：Pi、CAP、DreamZero 等 checkpoint 推理策略；
- teleoperation：TeleDex、键盘、SpaceMouse 等；
- dummy/random：测试和调试策略。

策略可以返回 action chunk。runner 会先查询策略，再通过 `task.step_chunk()` 开环执行多个动作，减少高频模型调用。Pi 类策略还支持本地 checkpoint 或 WebSocket 远程推理。

规划侧包含：

- `astar_planner.py`：基于室内占用图的二维导航路径规划；
- cuRobo planner/client/server：GPU 加速机械臂运动规划；
- object manipulation planner：通过 TCP 插值、IK、抓取阶段机和失败重试执行操作任务。

## 4. 数据生成流程

数据生成入口是 [`molmo_spaces/data_generation/main.py`](molmo_spaces/data_generation/main.py#L31-L149)。典型流程如下：

```text
命令行配置名
    ↓
导入并注册 Experiment Config
    ↓
实例化 MlSpacesExpConfig
    ↓
生成 output_dir，初始化可选 W&B
    ↓
ParallelRolloutRunner
    ↓
按 house / batch 创建 work items
    ↓
worker 创建或复用 TaskSampler
    ↓
采样 Task → 创建 Policy → 可选 Viewer
    ↓
Task.reset()
    ↓
循环：policy.get_action_chunk() → task.step_chunk()
    ↓
judge_success()
    ↓
收集 history，按配置过滤成功轨迹
    ↓
保存 HDF5 与 MP4
```

`ParallelRolloutRunner` 以 house 为主要并行单位：

- CUDA 可用时使用 `forkserver`，否则使用 `spawn`；
- 多 worker 进程通过共享计数器分配 work item；
- 记录成功数、总数、完成 house 数和跳过数；
- 支持连续采样失败、连续 rollout 失败和不可恢复失败阈值；
- 已存在对应 `trajectories_batch_*.h5` 时跳过该批次；
- runner 适合离线数据生成和在线评测，不适合作为 RL 的高效向量化训练 runner。

实现重点：[`molmo_spaces/data_generation/pipeline.py`](molmo_spaces/data_generation/pipeline.py#L335-L475)、[`molmo_spaces/data_generation/pipeline.py`](molmo_spaces/data_generation/pipeline.py#L478-L610)、[`molmo_spaces/data_generation/pipeline.py`](molmo_spaces/data_generation/pipeline.py#L701-L809)。

运行示例：

```bash
# 推荐先创建 Python 3.11 环境
pip install -e ".[mujoco]"

# 配置驱动的数据生成
python -m molmo_spaces.data_generation.main FrankaPickOmniCamConfig

# 调试/快速测试
python scripts/datagen/run_pipeline.py --viewer --seed 1
```

## 5. Task 时间步与 rollout 语义

系统使用三级时间步：

```text
sim_dt_ms    MuJoCo 物理积分步长
    ↓ 多个 sim step
ctrl_dt_ms   控制器更新频率
    ↓ 多个 control tick
policy_dt_ms 策略查询频率
```

一次 `task.step(action)` 会：

1. 将动作发送给各机器人 controller；
2. 在一个 policy step 内执行多个 control tick；
3. 每个 control tick 执行多个 MuJoCo simulation step；
4. 轮询传感器；
5. 计算 reward、terminated、truncated 和 success 并写入缓存。

`task.reset()` 只重置 episode 级状态、传感器、策略和缓存，不负责把环境物理状态恢复到初始场景；环境状态由 sampler 在采样阶段准备。详见 [`molmo_spaces/tasks/task.py`](molmo_spaces/tasks/task.py#L224-L416)。

## 6. 观测与轨迹数据格式

典型输出目录：

```text
train/
└── house_{house_idx}/
    ├── trajectories_batch_{batch_idx}_of_{n_batches}.h5
    ├── episode_{ep_idx:08d}_{camera_name}_batch_*.mp4
    └── ...
```

HDF5 中每条轨迹包含：

```text
traj_{i}/
├── actions/
│   ├── commanded_action
│   ├── joint_pos
│   ├── joint_pos_rel
│   ├── ee_pose
│   └── ee_twist
├── obs/
│   ├── agent/qpos
│   ├── agent/qvel
│   ├── extra/
│   ├── sensor_data/
│   └── sensor_param/
├── obs_scene
├── rewards
├── success
├── terminated
└── truncated
```

重要对齐规则：第 `i` 个 state 对应第 `i+1` 个 action；首 action 是 dummy，末 action 可能是 `done` sentinel。因此训练前通常要丢弃首 action，若只监督运动动作，还应丢弃末 action 和末两条 state。

数据后处理顺序通常是：

1. `repair_video_paths.py` 修复视频路径；
2. `validate_trajectories.py` 添加 `valid_traj_mask`；
3. `calculate_stats.py` 计算轨迹和聚合统计量；
4. 根据下游格式转换脚本导出训练数据。

参考：[`docs/data_format.md`](docs/data_format.md#L1-L66)、[`docs/data_processing.md`](docs/data_processing.md#L1-L82)、[`scripts/data/process_data.sh`](scripts/data/process_data.sh)。

## 7. JSON Benchmark 评测流程

评测入口是 [`molmo_spaces/evaluation/eval_main.py`](molmo_spaces/evaluation/eval_main.py#L450-L719)，同时提供 CLI 和 `run_evaluation()` Python API。

```text
加载 benchmark.json
    ↓
解析 EpisodeSpec
    ↓
确定 task horizon
（CLI 覆盖 > benchmark task_horizon_sec）
    ↓
创建评测配置
（关闭 action noise/profile，固定 seed=42，保存全部轨迹）
    ↓
JsonEvalRunner 按 house 分组并行
    ↓
每 episode 创建 JsonEvalTaskSampler
    ↓
恢复场景、机器人、物体位姿和相机
    ↓
运行策略 rollout
    ↓
输出 HDF5/视频及 EpisodeResult
    ↓
可选 W&B 日志与视频上传
```

评测支持：

- checkpoint 覆盖；
- `--task_horizon_steps` / `--task_horizon_sec`；
- 单 episode、最大 episode 数；
- 相机名称和评测相机随机化；
- Filament renderer；
- 自定义 XML 物体替换；
- 多 worker 评测。

示例：

```bash
python molmo_spaces/evaluation/eval_main.py \
  molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig \
  --benchmark_dir assets/bench/path-to-benchmark.json \
  --checkpoint_path <path/to/checkpoint> \
  --task_horizon_steps 500 \
  --no_wandb
```

评测会在模块导入时校验资源版本；若当前 robots/scenes/objects/grasps 版本与 benchmark 预期不一致，会在真正执行 CLI 前失败。生命周期说明见 [`docs/evaluation_lifecycle.md`](docs/evaluation_lifecycle.md#L12-L53)。

## 8. 资产与跨模拟器支持

资源由 `molmo_spaces_constants.py` 和 ResourceManager 管理，支持本地缓存、R2/Hugging Face 下载以及固定版本覆盖。常用环境变量包括：

| 变量 | 作用 |
|---|---|
| `MLSPACES_ASSETS_DIR` | 资产缓存/安装目录 |
| `MLSPACES_CACHE_DIR` | 通用缓存目录 |
| `MLSPACES_PINNED_ASSETS_FILE` | 覆盖默认资产版本 |
| `MLSPACES_OBJAVERSE_ASSETS_DIR` | Objaverse 资产目录 |
| `MUJOCO_EGL_DEVICE_ID` | MuJoCo EGL 渲染设备 |

项目提供：

- `molmo_spaces_isaac`：USD 资产/房屋转换和 Isaac Lab/Isaac Sim 支持；
- `molmo_spaces_maniskill`：MJCF 资产下载和 ManiSkill 使用；
- 主仓库的数据生成与 benchmark 评测目前仅支持 MuJoCo。

## 9. 安装、测试与开发

### 安装

项目要求 Python `>=3.11`，支持 Linux 和 macOS：

```bash
conda create -n mlspaces python=3.11
conda activate mlspaces
pip install -e ".[mujoco]"
```

可选 extras：

- `dev`：ruff、mypy、pre-commit、类型检查；
- `mujoco-filament`：Filament renderer；
- `grasp`：抓取生成依赖；
- `housegen`：房屋生成依赖；
- `curobo`：cuRobo GPU 规划；
- `docs`：MkDocs 文档构建。

### 测试

```bash
PYTHONPATH=. pytest mlspaces_tests/data_generation
PYTHONPATH=. pytest mlspaces_tests/data_generation_curobo
PYTHONPATH=. pytest mlspaces_tests/component_tests
```

格式化：

```bash
ruff format .
ruff check .
```

### 房屋生成

安装后可使用唯一根包 console entrypoint：

```bash
generate-houses --help
```

该命令位于 [`molmo_spaces/housegen/exporter.py`](molmo_spaces/housegen/exporter.py#L273-L411)，可处理 iTHOR、ProcTHOR 和 Holodeck JSON 场景。

## 10. 工程风险与注意事项

### 高优先级

1. **配置保存/加载文件名不一致**：`save_config()` 写入 `experiment_config_<timestamp>.pkl`，而 `load_config()` 固定读取 `experiment_config.pkl`。依赖自动恢复或 `--eval` pickle 的流程可能无法直接加载保存结果。见 [`abstract_exp_config.py`](molmo_spaces/configs/abstract_exp_config.py#L116-L135)。
2. **评测资源版本强耦合**：`eval_main.py` 在 import 阶段执行版本断言，资产版本未正确安装会导致任何评测命令提前失败。见 [`eval_main.py`](molmo_spaces/evaluation/eval_main.py#L62-L133)。
3. **批量环境支持有限**：文档明确指出 batch 大于 1 尚未充分测试，部分任务还断言 `n_batch == 1`。生产使用应先采用单环境模式。见 [`docs/concepts.md`](docs/concepts.md#L152-L173) 和 [`tasks/task.py`](molmo_spaces/tasks/task.py#L432-L450)。

### 中优先级

4. **数据生成后必须后处理**：视频路径、轨迹有效性和统计量不是生成阶段完全保证的，直接将原始 HDF5 用于训练可能导致视频缺失、轨迹错配或归一化错误。见 [`docs/data_processing.md`](docs/data_processing.md#L7-L56)。
5. **依赖体量大且包含平台/GPU差异**：MuJoCo、PyTorch、JAX、Warp、Filament、cuRobo 等同时存在，CUDA 版本和本地 wheel 选择会显著影响安装成功率。
6. **文档存在版本滞后可能**：`docs/development.md` 仍提及旧 Python/bpy 安装方式，而当前 `pyproject.toml` 要求 Python 3.11；执行安装时应以 `pyproject.toml` 和 README 为准。
7. **并行数据生成资源占用高**：每个 worker 可能持有 MuJoCo 模型、渲染器、资产和策略实例，GPU/CPU 内存应根据场景复杂度逐步增加 worker 数量。
8. **多进程与远程策略连接限制**：多 worker 下不能简单复用不可 pickle 的预加载 WebSocket/msgpack 策略连接，应让每个 worker 独立创建策略连接。
9. **文档/构建断链**：`mkdocs.yml` 引用了仓库中未发现的 `docs/evaluation_guide.md`，README 还引用了未发现的 `beaker_scripts/RUNNER_SETUP.md`，可能导致文档构建或新用户上手失败。
10. **不可信 pickle 输入风险**：配置恢复、房屋工具和部分策略客户端使用 `pickle.load/loads`；若输入文件或网络端点不可信，反序列化可能执行任意代码。应将其限制在可信数据边界，或逐步迁移为 JSON/msgpack 等安全格式并增加来源校验。
11. **依赖与平台复现风险**：主包依赖大量 GPU/仿真组件，`curobo` 使用 Git commit，Filament 使用本地 CPython 3.11 Linux wheel，且缺少统一锁文件；建议按 MuJoCo/Isaac/ManiSkill 拆分环境并锁定关键依赖版本。

9. **文档存在断链**：`mkdocs.yml` 引用了仓库中未发现的 `docs/evaluation_guide.md`，README 还引用了未发现的 `beaker_scripts/RUNNER_SETUP.md`，可能导致文档构建或新用户上手失败。
10. **不可信 pickle 输入风险**：配置恢复、房屋工具和部分策略客户端使用 `pickle.load/loads`；若输入文件或网络端点不可信，反序列化可能执行任意代码。应将其限制在可信数据边界，或逐步迁移为 JSON/msgpack 等安全格式并增加来源校验。
11. **依赖与平台复现风险**：主包依赖大量 GPU/仿真组件，`curobo` 使用 Git commit，Filament 使用本地 CPython 3.11 Linux wheel，且缺少统一锁文件；建议按 MuJoCo/Isaac/ManiSkill 拆分环境并锁定关键依赖版本。
12. **跨模拟器环境可能冲突**：Isaac 适配包的 Torch/CUDA 约束与主包默认依赖不同，建议分别创建虚拟环境，不要假设一个环境可以覆盖全部模拟器。
13. **测试偏集成且硬件依赖明显**：当前测试主要覆盖组件和数据生成，cuRobo 需要 GPU/特定环境；建议补充 CPU headless smoke test，并在 CI 中明确 MuJoCo、Isaac、ManiSkill、cuRobo 的支持矩阵。

## 11. 推荐阅读顺序

1. [`README.md`](README.md)：安装、快速测试、数据生成和评测入口；
2. [`docs/code_structure.md`](docs/code_structure.md)：目录职责和信息流；
3. [`docs/concepts.md`](docs/concepts.md)：Robot/Env/Task/TaskSampler 抽象；
4. [`molmo_spaces/configs/abstract_exp_config.py`](molmo_spaces/configs/abstract_exp_config.py)：配置层；
5. [`molmo_spaces/data_generation/main.py`](molmo_spaces/data_generation/main.py) 与 [`pipeline.py`](molmo_spaces/data_generation/pipeline.py)：数据生成主链路；
6. [`molmo_spaces/tasks/task.py`](molmo_spaces/tasks/task.py)：时间步、动作、缓存和 episode 语义；
7. [`docs/data_format.md`](docs/data_format.md) 与 [`docs/data_processing.md`](docs/data_processing.md)：下游数据消费；
8. [`molmo_spaces/evaluation/eval_main.py`](molmo_spaces/evaluation/eval_main.py) 与 [`docs/evaluation_lifecycle.md`](docs/evaluation_lifecycle.md)：benchmark 评测。

## 12. 总结

MolmoSpaces 的核心价值在于把“大规模室内资产 + MuJoCo 物理仿真 + 机器人控制/规划 + 视觉传感器 + 轨迹保存 + benchmark 评测”整合成可配置、可并行的研究基础设施。其主链路是：

```text
资产/场景 → TaskSampler → MuJoCo Env → Robot/Controller
         → Task/Sensor → Policy/Planner → Rollout
         → HDF5/MP4 → 后处理 → 训练或评测
```

如果后续目标是接入新机器人，优先阅读 `robots/`、`robot_views/`、`configs/robot_configs.py`、`controllers/` 以及 `docs/tutorials/add_robot.md`；如果目标是生成训练数据，优先阅读 `data_generation/config/`、`tasks/`、`policy/` 和 `utils/save_utils.py`；如果目标是复现 benchmark，则先确认资源版本，再阅读 `evaluation/benchmark_schema.py`、`json_eval_runner.py` 和 `docs/evaluation_lifecycle.md`。
