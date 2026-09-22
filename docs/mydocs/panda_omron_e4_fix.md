# E4 PandaOmron 评测修复记录（第二轮）

> 分析对象：`eval_output/PandaOmronCuroboPickPnPEvalConfig/20260920_115052/running_log.log`
> —— E4 的完整评测运行（26 h 49 m、875 episodes、成功率 0.57%）。
> 分支 `fix/panda-omron-e4`，本轮修复均在 worktree `.worktrees/panda-omron-e4` 中进行。

## 1. 分析：三条独立证据链

### 1.1 起始状态近似重叠 → 规划必然失败（首要瓶颈）

| 观察 | 数据 |
|---|---|
| `CuRobo start-state overlaps` 次数 | **365** |
| 其中紧随规划失败（`TRAJOPT_FAIL`）的比例 | **365 / 365 = 100%** |
| 发生时机 | 268 次在首次进入 PREGRASP、97 次在抓取失败回退 PREGRASP 之后 |
| 与 MuJoCo 判定的一致性 | **100% 发生在 `place_robot_near` 判定"已修复为无碰撞"的位姿上** |
| 实测重叠构成 | `panda_hand` ↔ 目标物体 0.9–2.5 cm；`omron_base` ↔ 柜台 2.9 mm |

**根因**：CuRobo 侧的碰撞几何是保守近似——

- `omron_base`：4 个半径 **0.17 m** 的球覆盖底盘，向外超出真实几何约 0.1 m；
- `panda_hand`：碰撞球超出夹持点 **5.8 cm**（`policy_configs.py` 的 `pregrasp_z_offset` 注释已记录此事实）；
- 障碍物盒：取 MuJoCo geom 的局部 AABB，对空心容器与倾斜几何偏大。

抓取失败回退 PREGRASP 时手爪正停在目标物体旁，这些外扩的球与目标物体、放置容器的
盒子重叠；`check_start_validity=False` 虽不直接拦截，但 trajopt 从"碰撞状态"出发无法
收敛 → `TRAJOPT_FAIL` → **整条 episode 直接失败**。这是 333 次 pregrasp 规划失败的主要来源。

### 1.2 抓取判定窗口短于夹爪闭合时长

```
5 步 × 66 ms = 330 ms  <  gripper_close_duration = 500 ms
```

`_grasping_something()` 的判据是"夹爪位置偏离闭合位"，而**闭合过程中仍在运动的手指
同样偏离闭合位**。结果：**97.4%** 的判定"通过"，随后 **70.7%** 在 LIFT 阶段掉落
（300 次判定中 212 次掉落）。

### 1.3 GRASP 阶段的轨迹被物理阻挡（本轮新发现）

原日志 2849 次 `[TIMEOUT]` **全部用满 30 步**，重试 6 次后 episode 失败（449 次）。
本轮新增诊断后实测：

```
[TIMEOUT] Timed out on reaching waypoint 4 after 30 steps
  (largest joint errors: arm[1]=0.0390, arm[3]=0.0116, arm[5]=0.0106, base[2]=0.0069;
   外部接触数=6, 各组最大关节速度: base=0.0006, torso=0.0000, arm=0.0037)
```

**关节速度趋零 + 存在真实接触 → 被挡住**，而非伺服跟不上。

另外用 FK 对比脚本排除了"模型不一致"这一可能：CuRobo 的 URDF 与 MuJoCo（robosuite
生成）模型的**正运动学完全一致**（位置差 `0 m`、姿态差 `0.0005 rad`）。

对应代码是 `_execute_grasp_phase` 里写死的额外进给：

```python
grasp_pose_world[:3, 3] = tcp + z_direction * (pregrasp_z_offset + 0.01)   # 0.01 是硬编码
```

PandaOmron 的 `pregrasp_z_offset` 已是 0.10 m，再额外进给 0.01 m 会让手爪中心**越过**
物体中心；对薄片物体（纸毛巾、纸巾）与低矮物体，就是手掌直接压到支撑面，接触力把关节
顶住，waypoint 永远差 0.04–0.14 rad 到不了。

## 2. 修复内容

| # | 文件 | 内容 |
|---|---|---|
| 1 | `policy/solvers/object_manipulation/panda_omron_curobo_pick_and_place_planner_policy.py` | 新增 `_split_start_overlaps()`：把与起始状态重叠的障碍物盒**从本次规划世界剔除**并打日志；每次规划重新构建世界，剔除范围不累积。真实穿透仍由 MuJoCo 物理拦截 |
| 2 | `configs/policy_configs.py` + `evaluation/configs/evaluation_configs.py` | 新增 `grasp_settle_steps()`，E4 评测配置按 `policy_dt_ms` 反推 `max_grasping_timesteps`（**5 → 11 步 = 726 ms**，覆盖 500 ms 闭合 + 226 ms 稳定） |
| 3 | `configs/policy_configs.py` + `policy/solvers/object_manipulation/curobo_pick_and_place_planner_policy.py` | 把写死的 `+ 0.01` 进给提为 `grasp_approach_overshoot` 配置项；**PandaOmron 置 0.0**，RBY1 保持 0.01（不影响 E2） |
| 4 | `policy/solvers/curobo_planner_policy.py` | `[TIMEOUT]` 日志增加三类诊断：**关节级误差**、**外部接触对**、**各组关节速度**，用于区分"被挡住"与"伺服跟不上" |
| 5 | `utils/scene_maps.py` + `env.py` + `evaluation/eval_main.py` + `evaluation/robot_eval_overrides.py` | **基座放置策略**：扩大粗筛半径并新增底盘净空择优（详见 §2.1） |

修复 1 的日志形态（实测）：

```
Loaded 447 scene cuboids into local CuRobo
Dropped 10 start-state overlapping cuboid(s) from the CuRobo world
(robot link, obstacle, penetration m): [('omron_base', 'countertop_.../geom_487', 0.0029),
 ('panda_hand', 'papertowel_.../geom_685', 0.0214), ('panda_hand', 'place_receptacle/...', 0.0262), ...]
```

### 2.1 基座放置策略

**要解决的问题**：底盘卡在窄空间。实测形态是
`mobilebase0_pedestal_feet_col ↔ stand` 接触、base 关节差 0.03–0.04 rad 到不了、
episode 反复重试后失败（E4 的 idx 213 即为此类）。

**两处根因**（都在 `env.place_robot_near`）：

1. **粗筛半径小于底盘**。占用图按 `robot_base_pose_repair_map_radius = 0.40 m` 膨胀，
   而实测底盘 `mobilebase0_pedestal_feet_col` 的水平半径是 **0.438 m**（其余
   `mobilebase0_g0_col` / `g1_col` / `support` 只有 0.30 m 左右）。筛出来的"自由点"
   其实放不下底盘，缺口有 4 cm。
2. **取第一个可行点就返回**。`check_camera_visibility=False`（E4 的取值）时代码直接
   `break`，完全不评估落点周围的空间，容易挑到"当前无碰撞、一平移就被卡住"的窄缝。

**策略**：

| 环节 | 做法 |
|---|---|
| 考虑底盘尺寸 | 粗筛半径提到 **0.45 m**（覆盖实测 0.438 m 并留余量） |
| 考虑周围障碍 | 对膨胀后的占用图做**距离变换**，得到每个自由格的**净空**（到最近障碍的距离，米） |
| 搜索落点 | 沿用目标周围 0.4–1.25 m 的采样范围，收集 **8 个**无碰撞候选（而非 1 个） |
| 择优 | 按净空**降序**排序，选四周最宽敞的落点 |

**接入点**：

- `utils/scene_maps.py`：`ProcTHORMap.clearance_m(pos)` 与 `_clearance_grid()`（惰性计算并缓存）
- `env.py`：`place_robot_near` 新增 `candidate_limit` 参数；候选记录净空并按净空排序
- `evaluation/eval_main.py`：`robot_base_pose_repair_candidate_limit`（**默认 1 = 历史行为**）
- `evaluation/robot_eval_overrides.py`：PandaOmron 侧设半径 `0.45` / 候选 `8`

**零影响保证**：`candidate_limit=1` 时行为与改动前完全一致（仍是"取第一个可行点"），
只有 PandaOmron 的评测覆盖会启用择优，其余机器人、数据生成与导航路径不受影响。

## 3. 验证

### 3.1 单 episode 回归（house_2 / benchmark idx 116，papertowel）

| 环节 | 修复前（原日志） | 修复后 |
|---|---|---|
| 起始重叠 | 11 个 geom 重叠 → `TRAJOPT_FAIL` | 检测到 10 个盒重叠 → 剔除 → 继续 |
| PREGRASP 规划 | **0/4 全败** | **4/4 成功** |
| 抓取判定 | 5 步（330 ms，夹爪未合拢） | **11 步（726 ms）**，日志 `Object not grasped after 11 timesteps` |
| 后续 | episode 立即失败 | 进入 GRASP 阶段（暴露下一层瓶颈） |

### 3.2 批量对比（8 个曾发生起始重叠的 episode）

`idx = 116, 672, 14, 60, 93, 104, 149, 213`（覆盖 papertowel / mug / apple /
Irishpotato / tissuepaper / atomizer）。这些 episode 在原日志中**全部**因起始重叠而
在 PREGRASP 阶段规划失败。

| 轮次 | 代码状态 | 成功 | 说明 |
|---|---|---|---|
| 原日志（基线） | 修复前 | **0 / 8** | 全部终止于 `TRAJOPT_FAIL`（规划死锁，机器人无法从碰撞位姿出发） |
| 第一轮 | 修复 1 + 2 | **1 / 8** | IDX 672（mug）**完整成功**；其余失败点从规划层移到执行层 |
| 第二轮 | 修复 1 + 2 + 3 | 见 §3.3 | 复跑 IDX 116 / 213 观察修复 3 的作用 |

**失败性质的变化比数字更重要**：修复后不再出现"规划从碰撞状态出发必然失败"的死锁，
PREGRASP 规划实测 **0/4 → 4/4**。1/8 的成功率提升有限，是因为清除死锁后暴露出的是
**更下游的真实限制**，而不是又一层软件缺陷。

### 3.3 剩余失败的逐条归因（实测接触对）

修复后新增的诊断直接给出了失败性质：

| idx | 物体 | 实测接触 | 性质 |
|---|---|---|---|
| 116 | papertowel | `gripper0_right_finger2_collision ↔ papertowel`、`gripper0_right_hand_collision ↔ papertowel`，关节速度归零 | **薄片物体抓取短板**：手爪中心对准物体中心时，指根/手掌已经压在薄片上（纸毛巾厚约 1–2 cm，而手指长约 4–5 cm） |
| 213 | atomizer | `mobilebase0_pedestal_feet_col ↔ stand×2`，重叠深度 **7.8–8.0 cm** | **底盘卡在场景支架上**：这是真实碰撞，任何规划都无法从该位姿出发；属场景与机器人尺寸不匹配 |

两者都不是本轮修复能覆盖的软件缺陷：前者需要抓取姿态层面的改进（报告 §9 P3 #17），
后者需要基座放置策略或场景筛选。

### 3.4 基座放置策略的验证

测试对象是用户指定的、原 E4 日志中**全部失败且都带 `start-state overlaps`** 的场景。

| house | idx | 物体 | 候选净空范围 → 选中 | 结果 | 失败主导关节 |
|---|---|---|---|---|---|
| 103 | 6 | spoon | 0.011–0.237 m → **0.237 m** | 失败 | `arm` |
| 97 | 969 | candle | 0.020–0.393 m → **0.393 m** | 失败 | `arm` |
| 97 | 970 | **ladle** | 0.055–0.507 m → **0.507 m** | ✅ **成功** | — |
| 95 | 947 | papertowel | 0.015–**0.065** m → 0.065 m | 失败 | **`base` 卡住** |
| 132 | 42 | Irishpotato | 未触发修复（起始位姿无碰撞） | 失败 | **`base` 卡住** |
| 107 | 10 | egg | 0.090–0.433 m → **0.433 m** | 失败 | `arm` |
| 107 | 11 | bottle | 未触发修复 | 未完成 | — |
| 108 | 12 | winebottle | 0.065–0.150 m → **0.150 m** | 失败 | `arm` |

**1 / 8 成功**（原日志 0 / 8）。

**策略确实生效的地方**：

1. **6 次触发择优时，每次都选到了净空最大的候选**。最极端的是 house 103：8 个候选里
   净空最小的只有 **0.011 m**（1.1 cm，几乎贴着障碍），按改动前"取第一个可行点"很可能
   就落在这种位置；现在选的是 0.237 m。
2. **失败结构整体从"底盘"迁移到"机械臂"**。8 个 episode 里 4 个的主导失败关节是
   `arm`；`base` 卡住只剩 2 个（且都有各自明确的原因，见下）。

**策略边界（诚实记录）**：

- **场景整体就窄**：house 95 的候选净空**上限只有 0.065 m**——附近根本没有宽敞点，
  择优只能"矮子里拔将军"，底盘仍然被卡。这类场景需要扩大采样范围（去更远处找落点，
  代价是机器人离目标更远）。
- **运动中卡住**：house 132 起始位姿**无碰撞**（没触发修复），是在后续移动中撞上窄道。
  这超出了"选起始落点"的作用范围，需要底盘路径层面的规划。

**house 97 的 ladle 完整成功**是目前唯一的"细长物体"成功案例（E1 中 ladle 5.5%）。

## 4. 尚未解决

1. **GRASP 阶段的物理阻挡**：修复 3 去掉了确定性的过度进给，但复跑 IDX 116 / 213
   显示两者仍失败——**因为它们的接触来自更下游的原因**（薄片物体上指根压物、底盘卡
   支架，见 §3.3），不是过度进给。修复 3 的作用在这两个样本上未被独立验证，如需确认
   应挑"非薄片、底盘活动空间充足"的样本复跑。
2. **success 判定漏洞（跨实验共用，未改）**：`pick_and_place_task.py` 的 `success`
   表达式不含物体-容器距离约束，且 `carry_forward` 缓存可用历史支撑顶替接触证据。
   E4 的 5 个"成功"中 4 个属此类假阳性。**该判定为 E1–E4 共用，改动会影响全部历史结果**，
   故本轮未动，建议单独立项。
3. **底座位姿碰撞修复失败 13 次**：`Could not repair colliding cross-robot base pose`，
   这 13 条 episode 被直接跳过。
4. **waypoint 到达容差**：`_is_waypoint_reached` 对全部关节统一用 0.0275，对底盘（米）
   与机械臂（弧度）语义不同。放宽容差会牺牲抓取精度，本轮未改，仅补充诊断。
5. **薄片 / 细长物体抓取**：IDX 116 的实际失败点，与 E1/E3 的形态学结论一致
   （纸毛巾、纸巾、刀/勺/叉系统性失败）。需从抓取姿态采样与夹爪策略入手。

## 5. 结论

本轮把 E4 的失败从**"规划死锁"**推进到**"可解释的物理限制"**：

- 清除的死锁：起始状态近似重叠导致 `TRAJOPT_FAIL`（365 次 / 100% 相关），
  使 PREGRASP 规划从 **0/4 全败** 变为 **4/4 成功**；
- 修正的语义错误：抓取判定窗口 330 ms < 闭合 500 ms，导致 97.4% 的判定误通过；
- 补上的可观测性：`[TIMEOUT]` 现在直接给出**卡在哪个关节、被什么挡住、关节是否还在动**，
  剩余失败无需再靠推测归因。

**但成功率提升有限（验证集 0/8 → 1/8）**，因为死锁之下压着的是能力短板
（薄片抓取）与场景约束（底盘卡支架），这两类不是软件缺陷，本轮修复不覆盖。

## 6. 复现

### 6.1 先确认代码版本（重要）

本仓库的 venv 是 `pip install -e` 装的，它的 `__editable___molmo_spaces_*_finder` 把
`molmo_spaces` **硬编码到主工作区** `/data0/wenyifan/MoMaTrajGen/molmospaces`。该 finder
挂在 `sys.meta_path` 上，**优先级高于 `sys.path`**，因此在 worktree 里直接执行
`python -m molmo_spaces...`（或设 `PYTHONPATH`）**仍然会加载主工作区的代码**，
worktree 里的修改不会生效。

用仓库自带的 `scripts/run_from_source.py` 指定代码目录——它会摘掉 editable finder、
把指定目录插到 `sys.path` 首位，并**断言实际解析到的包就在该目录下**（否则直接报错退出）。
启动时会打印实际加载路径供核对：

```
[run_from_source] molmo_spaces = /data0/wenyifan/MoMaTrajGen/.worktrees/panda-omron-e4/molmo_spaces/__init__.py
```

### 6.2 跑评测

```bash
cd /data0/wenyifan/MoMaTrajGen/molmospaces          # 用这里的 venv 与 benchmark 数据
CUDA_VISIBLE_DEVICES=3 MUJOCO_EGL_DEVICE_ID=3 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python /data0/wenyifan/MoMaTrajGen/.worktrees/panda-omron-e4/scripts/run_from_source.py \
  --source /data0/wenyifan/MoMaTrajGen/.worktrees/panda-omron-e4 \
  molmo_spaces.evaluation.eval_main -- \
  --benchmark_dir /nas/wenyifan/molmospaces_data/cache/benchmarks/molmospaces-bench-v1/20260408/procthor-10k/FrankaPickandPlaceDroidMiniBench/FrankaPickandPlaceDroidMiniBench_20260111_json_benchmark \
  --idx 116 --task_horizon_steps 606 --num_workers 1 --no_wandb \
  --output_dir eval_output/e4_verify
```

### 6.3 跑单元测试

```bash
W=/data0/wenyifan/MoMaTrajGen/.worktrees/panda-omron-e4
cd /data0/wenyifan/MoMaTrajGen/molmospaces
.venv/bin/python $W/scripts/run_from_source.py --source $W \
  pytest -- -q $W/mlspaces_tests/evaluation/test_panda_omron_curobo_eval.py -m "not slow"
```

> `runpy` 会对 `molmo_spaces.evaluation.eval_main` 报一条
> `RuntimeWarning: ... found in sys.modules after import of package ...`，这是
> `molmo_spaces/evaluation/__init__.py` 会导入子模块导致的既有现象，不影响执行。
