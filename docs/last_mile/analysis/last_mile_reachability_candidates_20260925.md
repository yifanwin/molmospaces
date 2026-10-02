# Reachability 场景检索：pick 近端 / place 远端超限 / 桌面杂乱

**日期**：2026-09-25
**问题**：在 `formal_E01` 或 benchmark.json 中，能否找到「桌面 pick&place，pick 物体在近端、place 位置在远端几乎达到机械臂伸长极限，桌面杂乱障碍多，若底盘固定则高风险碰障碍或无法规划，底盘向远端适当移动有利于完成 place」的场景？

**结论**：
- **benchmark 源库（2000 条）里有，且不止一条**，最典型的是 `idx=442 / 555 / 1160`。
- **`formal_E01` 里没有**这类干净场景。原因见第 3 节：P1 协议强制 A\* 路径在距目标物中心 0.8 m 处截断，100 条实测终点处机械臂**对 pick 物体就已经全部够不到**（98 条中 0 条可达），不存在「pick 够得到而只有 place 够不到」的对比。

---

## 1. 判据

用**肩关节到目标的 3D 距离 ÷ 臂展**作为「是否在伸长极限」的度量：

- 臂展上限 **0.793 m**：肩关节 `link_right_arm_0` → TCP `ee_site_r` 的最大伸展，由 MJCF 连杆长度（上臂 0.276 + 前臂 0.258 + 腕 0.137 + 手）实测得出，与代码侧 `RBY1MConfig` 一致。
- 肩关节位置：`torso` 全 0 时相对 base 为 `(0, ∓0.22, 1.386) m`（左/右）。
- 比值 < 1.0 表示**躯干不动**即可够到；> 1.0 需要躯干前倾或移动底盘。躯干 6 关节可动时伸展上界远大于此（实测可达 1.9 m），但那属于弯腰姿态，带碰撞风险。

数据来源：`molmospaces_data/assets/benchmarks/molmospaces-bench-v1/procthor-10k/RBY1PickAndPlaceDataGenConfig/RBY1PickAndPlaceDataGenConfig_20260209_json_benchmark/benchmark.json`（sha256 `e8e08e49ba6798bb…`，**等于 P0 的 `p0_source_sha256`**，即 2000 条源库；`source_index` 就是它的下标）。

## 2. 最佳候选（源库 2000 条内）

筛选条件：`pick < 1.0` 且 `place > 1.3` 且「把 place 拉进臂展所需的前移量」在 0.15–0.9 m 之间，共 **67 条**（若去掉前移量条件则为 68 条）。按 (place比 − pick比) 排序的前几名：

| idx | house | pick 比 | place 比 | 底盘需前移 | 同桌面物体 | 走廊障碍 | 最近障碍距 place | 任务 |
|-----|-------|--------|---------|-----------|-----------|---------|----------------|------|
| **442** | 302 | 0.85 | **1.42** | 0.32 m | **16** | 1 | 0.252 m | pick up the orange spray bottle and place it on the dark box |
| 555 | 350 | 0.81 | 1.43 | 0.40 m | 11 | 1 | **0.171 m** | pick up the blue bottle and place it on the ashtray |
| 1329 | 693 | 0.95 | 1.51 | 0.56 m | 4 | 1 | 0.362 m | pick up the green bottle and place it on the black cooking pot |
| 1257 | 652 | 0.97 | 1.53 | 0.39 m | 5 | 1 | 0.225 m | pick up the blue soap bottle and place it on the … |
| 1160 | 604 | 0.85 | 1.32 | 0.43 m | 4 | 1 | 0.153 m | pick up the wooden cup and place it on the bowl |
| 9 | 100 | 0.76 | 1.33 | 0.29 m | 3 | 0 | 0.326 m | pick up the white ceramic mug and place it on the … |

**最典型的一条：`idx=442`（house 302, val）**

- 基座 `(9.779, 5.869)`，yaw 266.4°
- pick：`atomizer_f2c2791e…_1_0_6`，世界坐标 `(9.653, 5.290, 1.073)` → 肩距/臂展 = **0.85**（近端，躯干不动可及）
- place：`place_receptacle/0_0/DecorativeBox2`，`(9.814, 4.847, 1.014)` → 肩距/臂展 = **1.42**（远端，超限）
- 水平距离：base→pick 0.592 m，base→place **1.022 m**，pick→place 0.471 m
- **把 place 拉进臂展需底盘沿 base→place 方向前移 0.322 m**
- 桌面杂物（同桌面 z±0.12 且 1.2 m 内共 16 件）：sponge 距 place 0.252 m 且**正好压在同一条 pick→place 连线走廊上（距路径 0.06 m）**，另有 mug 0.361、boiler 0.395、papertowel 0.463、fork 0.472、butterknife 0.487 …
- 几何上 base→pick→place **几乎共线**，是「面向远端、底盘前移即可缓解」的教科书式布局

`idx=555`（house 350）同样典型：pick 0.81 倍臂展、place 1.43 倍、需前移 0.397 m；桌上 fork(0.171)/pen(0.175)/knife(0.180) 三件紧贴 place，bowl(0.300) 在走廊上。

## 3. 为什么 `formal_E01` 里找不到这类场景

两条协议约束叠加，使它不可能出现「pick 够得到、只有 place 够不到」：

1. **A\* 路径强制截断**：`path_min_dist_to_target_center_m = 0.8`（`molmo_spaces/robots/.../astar_planner_policy.py`），路径在距目标物中心 **0.8 m** 处就被截断，导航终点 A 因此永远落在物体 0.8 m 之外。
2. **P2 冻结躯干**：`torso_fixed=True`（`run_p2.py` 的 PROTOCOL），`base_fixed=True`。

实测统计（本分析计算，基于 `formal_E01/episodes.jsonl` 的 `actual_endpoint` 与 benchmark 的物体世界坐标）：

- 98 条有实测终点的 episode 中，**A→pick 的肩距/臂展比 < 1.0 的条数 = 0**（中位数 1.64）；A→place 中位数 1.72。
- 即 P1 结束时机械臂**对所有物体都够不到** —— 这正是 P2 得出 38/38 首失败层 `F_IK` 的直接原因，与 `docs/last_mile/p2/last_mile_p2_e02_20260922.md` 里「不是数学上不可达」的声明一致：它是**协议伪影**，不是场景本身的 reachability 性质。
- 反过来，同样这 100 条在 **benchmark 原生基座**下，`F_IK` 的对比就出现了：`idx=1378`(0.83/1.35)、`idx=389`(0.84/1.24)、`idx=1782`(0.85/1.19)、`idx=1590`(0.83/1.09)、`idx=223`(1.01/1.37)。

**在 P0 formal 那 100 条子集内部**（`source_index` 范围 82–1969），最接近本场景的是：

| idx | house | status | 原生 pick 比 | 原生 place 比 | 说明 |
|-----|-------|--------|-------------|--------------|------|
| 1378 | 715 | no_path | 0.83 | 1.35 | pick→place 跨度最大 |
| 389 | 273 | completed | 0.84 | 1.24 | 导航成功到达 A，但 A 处仍够不到 place |
| 290 | 221 | no_path | 1.09 | 1.42 | 绝对距离最远 |
| 223 | 195 | completed | 1.01 | 1.37 | |

而第 2 节的最佳候选（442 / 555 / 1160 / 1329）**都不在这 100 条里**，要用它们需要重新抽子集。

## 4. 重要限定

- 本文的 pick/place 比值是**纯运动学**判据（肩距 vs 连杆最大伸展），**不含碰撞**。用数值 IK（DLS，13 自由度：躯干 6 + 右臂 7）实测：这些目标在**允许躯干前倾**时残差可收敛到 2 cm 以内，即**运动学上仍可能够到**。真正卡住的是**碰撞与规划**（躯干前倾 26°–56° 才能勉强够到远端，会压到桌面/杂物）—— 这与「固定不动有极高风险碰到障碍物、或者无法规划路径」的描述一致，而不是「数学上不可达」。
- 因此要把「底盘移动有利」做成硬证据，应当用仓库现成工具做**固定基座 vs 移动基座**的对照评估：
  `molmo_spaces/evaluation/last_mile/feasibility.py` 的 `ManipulationFeasibilityEvaluator.evaluate(snapshot, robot, base_pose, target, grasp_pool, budget)`，其中 `base_pose` 接受 `[x, y, yaw]`；P3（`run_p3.py`）已实现「在 A 点 30 cm 圆盘内搜索可行的 B」这一流程，`I_reach` 即为救援率（试点 1/5 = 0.2）。
- P3 formal（`p3_20260923/formal_E01`）截至本文撰写时仍在运行中（已完成 15 条，无 `metrics.json`）。

## 5. 复现

本次分析脚本存放于作业临时目录（随作业清理）：
`$CLAUDE_JOB_DIR/tmp/` 下 `reach_scan.py`（全库筛选）、`show_cand.py`（布局展开）、`formal_reach.py`（formal 子集画像）、`analyze2.py`（IK 对照）。判据为纯几何计算，依赖 `benchmark.json` 与 `episodes.jsonl`，无需仿真环境；IK 部分依赖 `mujoco` 与 `molmospaces_data/assets/robots/rby1m/rby1_v1.2_site_control.xml`。
