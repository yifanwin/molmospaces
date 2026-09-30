# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

本仓库是 [allenai/molmospaces](https://github.com/allenai/molmospaces) 的**本地 fork**（`main` 分支，基于 v0.2.9），用于机器人操作/导航的仿真数据生成与策略评测。相对 upstream 已叠加大量本地改动（2026-09-29 时 70 个文件、约 2 万行），核心是：

- **LLM waypoint 策略**：`molmo_spaces/policy/solvers/object_manipulation/llm_waypoint_planner_policy.py`（1617 行，upstream 没有）—— LLM 只做高层决策（选抓取候选、手臂、approach 策略、lift/preplace 高度、底盘目标），6 个 EE 目标由 `build_plan_from_decision` 确定性展开，再走 MuJoCo IK/碰撞校验。
- **PandaOmron 机器人**（robosuite Panda 臂 + Omron 移动底盘）：`molmo_spaces/robots/panda_omron.py`、`robot_views/panda_omron_view.py`、`robots/curobo/panda_omron/*.urdf|yml`。
- **评测脚本与探针**：`scripts/evaluation/`（LLM waypoint 场景评测、内存泄漏探针、结果汇总）。
- **本地文档**：`docs/mydocs/` 中文笔记（安装、测试体系、planner 评测入口、PandaOmron E4 修复记录）。

## 环境与依赖（务必遵守）

**这个 venv 由 uv 管理，不要用 pip 装东西。** `uv sync` 默认严格同步，会静默卸载 lockfile 之外的包。

```bash
source setup_env.sh          # 必须，先于任何运行
uv sync --extra mujoco --extra robosuite   # 每次都要带上当前生效的 extra，否则那个 extra 会被移除
```

- `setup_env.sh` 设置 `MLSPACES_CACHE_DIR`（NAS 只读真实文件）、`MLSPACES_ASSETS_DIR`（必须本地，内含指向 cache 的软链接和依赖 mmap 的 `.lmdb`）、`MUJOCO_GL=egl`/`PYOPENGL_PLATFORM=egl`、`MALLOC_ARENA_MAX=2` 等内存参数，以及 `CUDA_VISIBLE_DEVICES`/`MUJOCO_EGL_DEVICE_ID`（默认 GPU 3）。
- `mujoco`（3.5.0）与 `mujoco-filament`（3.7.1）两个 extra **互斥**，`uv sync --all-extras` 会失败。
- 万不得已用 pip 后，之后每次同步必须 `uv sync --inexact`。
- 只有 `mujoco` + `robosuite` 都装了才能跑 PandaOmron 任务；`uv sync` 只写 `--extra robosuite` 会把 mujoco 换掉。

## 常用命令

```bash
# 数据生成
python -m molmo_spaces.data_generation.main FrankaPickDroidDataGenConfig
python -m molmo_spaces.data_generation.main PandaOmronPickAndPlaceDataGenConfig
# 也可指定模块跳过全量自动导入：module.path:ClassName

# 评测（第 1 个位置参数必须是 module:ClassName，裸注册名必报 "Available configs: []"）
python -m molmo_spaces.evaluation.eval_main \
    molmo_spaces.evaluation.configs.evaluation_configs:RBY1LLMWaypointPickPnPEvalConfig \
    --benchmark_dir "$MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v1/procthor-10k/RBY1PickAndPlaceDataGenConfig/RBY1PickAndPlaceDataGenConfig_20260209_json_benchmark" \
    --idx 3 --num_workers 1 --no_wandb --task_horizon_steps 600 \
    --env_file ../.env

# 单场景调试
python scripts/datagen/run_pipeline.py --viewer --seed 1

# 本地 LLM waypoint 评测（推荐直接用现成脚本）
bash scripts/evaluation/run_llm_waypoint_scenes_eval.sh      # SCENES="103:1 97:3" 覆盖
bash scripts/evaluation/run_house4_llm_waypoint_eval.sh <baseline|gate|full|mock>   # 默认 gate
bash scripts/evaluation/run_probe_memory_leak.sh

# 结果汇总
python scripts/evaluation/summarize_llm_waypoint_scenes.py <run_dir> [<run_dir> ...] [--output x.json]
python scripts/benchmarks/eval_to_csv.py <run_dir> <policy_name> --success-condition both --output-csv out.csv

# 测试
PYTHONPATH=. pytest mlspaces_tests/data_generation
PYTHONPATH=. pytest mlspaces_tests/data_generation -m "not slow"
PYTHONPATH=. pytest mlspaces_tests/component_tests/test_panda_omron.py::test_model_compiles_and_exposes_expected_move_groups -v
PYTHONPATH=. pytest mlspaces_tests/data_generation_curobo   # 需要 GPU + cuRobo

# 格式化（提交前）
ruff format .
pre-commit run --all-files   # CI 的 ruff-checks 跑的是完整 pre-commit，不只是 ruff check
```

## 三个 entry point

| 入口 | 用途 |
|---|---|
| `molmo_spaces/data_generation/main.py` | 数据生成，位置参数是注册名或 `module:ClassName` |
| `molmo_spaces/evaluation/eval_main.py` | JSON benchmark 评测，位置参数同上 |
| `scripts/datagen/run_pipeline.py` | 临场构造实验、调试、被动 viewer |

`eval_main` 的注册表在导入时是**空的**（`evaluation_configs.py` 里 `@register_config` 全被注释掉），所以派发类必须写全路径 `molmo_spaces.evaluation.configs.evaluation_configs:XXX`。`data_generation/main.py` 不带冒号时会 `auto_import_configs()` 扫描 `data_generation/config/*.py` 填充注册表。

`eval_main` 常用 flag：`--benchmark_dir`（必填）、`--idx`、`--house_index`（本地新增）、`--max_episodes`、`--num_workers`、`--task_horizon_steps|--task_horizon_sec`（互斥）、`--no_wandb`、`--output_dir`、`--camera_names`、`--use_eval_cameras`、`--use-filament`、`--add_custom_object`、`--env_file`（本地新增，KEY=VALUE，`setdefault` 不覆盖已有环境变量）。

## 架构：三层栈

```
Task Sampler  ──拥有──►  Env(MuJoCo)  ◄──持有──  Task
（加载场景、随机化、          物理 + 渲染              episodic 交互、reward、
  生成任务实例）                                      judge_success、observation
```

- **Task Sampler**（`molmo_spaces/tasks/task_sampler.py`）拥有 env 生命周期：编译场景、放机器人/物体、配相机，`sample_task()` 返回一个可用的 `Task`。`JsonEvalTaskSampler`（评测用）不随机化，只确定性地按 JSON 落位。
- **Task**（`molmo_spaces/tasks/task.py`）包裹但不拥有 env。`reset()` **不会**调 `env.reset()`，它只清自己的 bookkeeping。
- **时序**：`task.step(action)` = 一个 policy step，内部循环 `n_ctrl_steps_per_policy` 个 control tick，每个 tick 跑 `n_sim_steps_per_ctrl` 个物理子步。三种 dt：`sim_dt` < `ctrl_dt_ms` < `policy_dt_ms`（`MlSpacesExpConfig.model_post_init` 校验整除关系）。
- **batch > 1 实际不可用**：env API 名义上支持 batched `MjData`，但全栈有 sharp edges，按 `n_batch=1` 处理。

## 架构：机器人抽象

`Robot` → `RobotView` → `MoveGroup`（`molmo_spaces/robots/abstract.py`、`robot_views/abstract.py`）。

- Move group 把底层 MuJoCo 关节/执行器**抽象掉**：关节数与执行器数可以不等（镜像夹爪 2 关节 1 执行器），自由关节 qpos 7 维而 qvel 6 维，甚至完全"伪造"（`FloatingRUMBaseGroup` 读写 mocap body 而无真实执行器）。
- 字符串 group id（`"arm"`、`"gripper"`、`"base"`、`"torso"`）是全代码库的动作字典键。
- **action 格式**：`dict[str, np.ndarray]`，如 `{"arm": np.array([...]), "gripper": np.array([...])}`；含义由 `command_mode`（`joint_position` / `joint_rel_position`）决定，映射到不同 `Controller` 子类。
- 相机系统 `molmo_spaces/env/sensors_cameras.py`；PandaOmron 用 robosuite 的 `eye_in_hand`/`robotview` + 挂在 Omron 上的第三人称 `camera_follower`。

## 架构：配置系统

`MlSpacesExpConfig`（`molmo_spaces/configs/abstract_exp_config.py`）聚合 camera / robot / task_sampler / task / policy 五个子配置，加 `policy_dt_ms`/`ctrl_dt_ms`/`sim_dt_ms`、`task_horizon`、`seed`、`output_dir`、`wandb` 等。`@register_config("Name")` 把类注册进 `data_generation/config_registry.py`。

**评测 config = datagen config + 挂上 policy**。指向 benchmark 后，episode 级字段（相机、`init_qpos`、物体位姿、instruction）被 JSON 覆盖；`policy_dt_ms` 等时序参数**不在** episode 里，由 config/CLI 提供（同一个 benchmark 可以用不同控制频率复跑）。

## 评测管线

`eval_main.run_evaluation()` → `JsonEvalRunner`（`evaluation/json_eval_runner.py`，继承 `ParallelRolloutRunner`，通过覆盖 hook 定制）→ 每 worker 跑 `process_single_house`。

关键行为：

- **成功判据来自 h5**，不是日志。`collect_episode_results()`（`molmo_spaces/utils/eval_utils.py:462`）扫 `house_*/trajectories*.h5`，对每个 `traj_*` 取 `success` 数组：`success[-1]` 是 at-end 成败，`any(success)` 是 oracle success。基准 episode **只跑一次、不重试**（`get_max_episode_attempts` 返回 `len(episode_specs)`）。
- `--idx N` 单独用时是**全局** episode 下标；同时给 `--house_index` 时，N 是**先按 house 过滤后**的下标（过滤顺序：house_index → max_episodes → idx，见 `json_eval_runner.py:196`）。
- 输出：`<output_dir>/<config>/<timestamp>/house_<i>/trajectories*.h5` + 每个相机的 `episode_*.mp4`，另有 `running_log.log`、`experiment_config_*.pkl`。LLM waypoint policy 还会写 `<output_dir>/llm_plans/*.json`（逐 attempt 的计划与失败分类）。**视频默认全存、成功失败都存，无 CLI 开关。**
- 批量评测多 worker，但 `preloaded_policy` 只支持单 worker。
- 数据集版本在导入时被 `_assert_data_versions_match()` 硬断言（`eval_main.py:102/133`），版本不符直接抛错而不是静默跑错资产。
- policy 失败（`ValueError`/`RuntimeError`）且 `policy_type == "planner"` 时会被记为"该 episode 不成功"而不中断整轮（`json_eval_runner.py:355`）。learned policy 没这个待遇。

## 目录约定与踩坑

- **`mlspaces_tests/scenes/` 下的 `test_*.py` 大多不是 pytest**（如 `test_penetration_objects.py`、`test_stability_mp.py`、`test_lift_force_mp.py`），而是带 `--dataset/--split/--houses-folder/--start/--end/--max-workers` 的批量质量检测 CLI（穿透、稳定性、可拾取性、开合力），由 `scripts/run_all_physics_tests.sh` 批量驱动并写 JSON/TXT。该目录整体被 pyproject 的 ruff `exclude` 排除，pytest 会收集它但只有少数文件真正含用例。
- `output_dir` 带时间戳：本地路径自动加 `/<config>/<timestamp>`，只有 `/mnt/shared` 前缀才用简单结构。
- `.gitignore` 忽略 `/*.json`、`*.h5`、`*.png`、`eval_output/`、`experiment_output`、`test_outputs`、`external_src/`、`typings`、`assets/*`。
- 数据分两层：`MLSPACES_CACHE_DIR`（NAS，真实文件）与 `MLSPACES_ASSETS_DIR`（本地，软链接 + `.lmdb`）。**不要**把 assets 挪到 NAS，`.lmdb` 依赖 mmap。

## 多仓库 / worktree 布局（易踩坑）

真正的 git 仓库是 `/data0/wenyifan/MoMaTrajGen/molmospaces/.git`。`/data0/wenyifan/MoMaTrajGen/.git` 是空目录（不是仓库，在那里跑 git 会 `fatal: not a git repository`）。

`/data0/wenyifan/MoMaTrajGen/.worktrees/` 下是**它的 worktree**（见 `git worktree list`），各有分支：

| worktree | 分支 | 主题 |
|---|---|---|
| `last-mile-p0` | `exp/last-mile-p0` | last-mile 预实验 P0–P3（导航终点→可行性评价器） |
| `llm-base-motion` (`llm-base`) | `llm-base-motion` | 让 LLM 底盘真正能动、决策权交回模型 |
| `panda-omron-e4` | `fr3omron` / `fix/panda-omron-e4` | PandaOmron E4 失败修复（torso、gripper、起始位姿重叠） |
| `rl-waypoint` | `rl-waypoint` | RL policy |

- `origin` = `https://github.com/yifanwin/molmospaces.git`（本人 fork），`upstream` = allenai/molmospaces。`.github/workflows/sync-upstream.yaml` 每 6 小时把 upstream/main 推到 `main_public` 分支。
- 工作树里跑测试时，`.venv` 通常软链到 `molmospaces/.venv`（`last-mile-p0` 没有）。**代码解析已实测（2026-09-29）**：editable finder 用 `sys.meta_path.append`，排在 `PathFinder` 之后，所以 `cd <worktree> && PYTHONPATH=$PWD <venv>/bin/python` 加载的就是 worktree 代码（`molmo_spaces.__file__` 会打印 worktree 路径）。反之如果有会话说 "worktree 里总加载主工作区代码"，那是 append/insert 行为变化前的旧结论。
- 顶层 `/data0/wenyifan/MoMaTrajGen/` 还有若干相关仓库：`robosuite`（editable 依赖来源）、`MolmoBot`（策略代码来源）、`openpi-molmospaces`、`IndoorGen`（场景生成）、`robocasa`、`RoboGen`；以及中文分析报告（`多机器人抓放评测综合分析总.md`、`Last-mile预实验分阶段实施计划_20260921.md`、`2026-09-19-任务驱动场景编辑实施方案.md`）。

## 本地 LLM 策略的环境变量

`--env_file` 指向 `/data0/wenyifan/MoMaTrajGen/.env`，需要 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL` 三者齐全（`OpenAICompatibleClient.from_env`），否则 A/B 对照跑不起来。`LLM_MOCK_RESPONSE_FILE` 可切到 `FileBackedMockClient`（见 `mlspaces_tests/evaluation/llm_decision_mock.json`），用于不调 API 的离线校验。

## 代码风格

- ruff：line-length 100、双引号、`target-version=py311`；`scripts/`、`tests/`、`mlspaces_tests/scenes/` 被排除。`known-third-party = ["wandb"]` 是为了让 isort 分组在 CI 与本地一致。
- 全局约定：对话与文档用简体中文，代码注释用中文；硬编码与超参数提到文件顶部作全局变量；不写 try-except 异常处理。
