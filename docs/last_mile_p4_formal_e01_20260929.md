# Last-mile P4-formal-E01 正式批执行报告

**结论：无结论**（缺正式百例、有效 house 区间或严格 Pick 物理正负校准）

## 口径与漏斗

冻结清单 100 条，导航产生有效 A 38 条，具有确定 A 标签与完整确定邻域 N_eval=38 条。
A 第一次几何评价 not_found M=38；其中存在可行 B 的 L_geo=18，存在可达可行 B 的 L_reach=18。
I_geo=0.47368421052631576，I_reach=0.47368421052631576，R_oracle=0.47368421052631576；unknown=0，I_reach 的 最保守/最乐观界限 0.47368421052631576–0.47368421052631576。

## 执行层

A 侧得到确定 Pick 标签 38/38，成功 0 次；A 侧全部有效 A 的成功率界限 0.0–0.0。
A 侧状态分布：{'planner_no_witness': 38}。
预选配对 18，配对双方均有确定标签 18（转移后另有确定标签 18）。
配对四格 (A,B)：{'00': 16, '01': 2, '10': 0, '11': 0}；配对差 0.1111111111111111；A 失败时 B 静态救援率 0.1111111111111111。
转移尝试 18，到达 18（到达率 1.0）；转移后成功 1，转移相对静态 B 差 -0.05555555555555555。

![配对真实 Pick](figures/last_mile_p4_formal_e01_paired_pick.png)

图 1：每行一个冻结源目标实例，三列分别是 A、静态 B 与转移后 B 的真实 Pick 结果；颜色含义见正文与图例文字。

| 索引 | house | A 状态 | B* | B 静态 | 转移状态 | B 转移后 |
|---:|---:|---|---|---|---|---|
| 0 | 433 | skipped | — | — | — | — |
| 1 | 749 | skipped | — | — | — | — |
| 2 | 207 | planner_no_witness | — | — | — | — |
| 3 | 288 | skipped | — | — | — | — |
| 4 | 676 | planner_no_witness | — | — | — | — |
| 5 | 765 | skipped | — | — | — | — |
| 6 | 831 | skipped | — | — | — | — |
| 7 | 155 | skipped | — | — | — | — |
| 8 | 335 | skipped | — | — | — | — |
| 9 | 15 | skipped | — | — | — | — |
| 10 | 437 | skipped | — | — | — | — |
| 11 | 247 | skipped | — | — | — | — |
| 12 | 533 | planner_no_witness | G121 | insufficient_lift_or_drop | arrived | planner_no_witness |
| 13 | 405 | planner_no_witness | G156 | success | arrived | planner_no_witness |
| 14 | 654 | planner_no_witness | G155 | support_contact | arrived | planner_no_witness |
| 15 | 807 | skipped | — | — | — | — |
| 16 | 467 | planner_no_witness | G164 | insufficient_lift_or_drop | arrived | insufficient_lift_or_drop |
| 17 | 171 | planner_no_witness | — | — | — | — |
| 18 | 706 | planner_no_witness | G124 | success | arrived | success |
| 19 | 420 | skipped | — | — | — | — |
| 20 | 258 | skipped | — | — | — | — |
| 21 | 153 | skipped | — | — | — | — |
| 22 | 271 | skipped | — | — | — | — |
| 23 | 626 | skipped | — | — | — | — |
| 24 | 720 | skipped | — | — | — | — |
| 25 | 669 | skipped | — | — | — | — |
| 26 | 786 | skipped | — | — | — | — |
| 27 | 803 | skipped | — | — | — | — |
| 28 | 724 | planner_no_witness | G185 | insufficient_lift_or_drop | arrived | planner_no_witness |
| 29 | 696 | skipped | — | — | — | — |
| 30 | 721 | planner_no_witness | — | — | — | — |
| 31 | 17 | skipped | — | — | — | — |
| 32 | 334 | skipped | — | — | — | — |
| 33 | 633 | skipped | — | — | — | — |
| 34 | 843 | skipped | — | — | — | — |
| 35 | 704 | planner_no_witness | G164 | insufficient_lift_or_drop | arrived | insufficient_lift_or_drop |
| 36 | 546 | skipped | — | — | — | — |
| 37 | 446 | skipped | — | — | — | — |
| 38 | 887 | skipped | — | — | — | — |
| 39 | 222 | planner_no_witness | G124 | insufficient_lift_or_drop | arrived | planner_no_witness |
| 40 | 34 | planner_no_witness | — | — | — | — |
| 41 | 832 | skipped | — | — | — | — |
| 42 | 490 | skipped | — | — | — | — |
| 43 | 518 | planner_no_witness | — | — | — | — |
| 44 | 344 | skipped | — | — | — | — |
| 45 | 679 | planner_no_witness | — | — | — | — |
| 46 | 83 | skipped | — | — | — | — |
| 47 | 429 | skipped | — | — | — | — |
| 48 | 643 | skipped | — | — | — | — |
| 49 | 497 | skipped | — | — | — | — |
| 50 | 828 | planner_no_witness | G156 | insufficient_lift_or_drop | arrived | insufficient_lift_or_drop |
| 51 | 783 | planner_no_witness | G158 | insufficient_lift_or_drop | arrived | planner_no_witness |
| 52 | 60 | planner_no_witness | — | — | — | — |
| 53 | 310 | planner_no_witness | — | — | — | — |
| 54 | 496 | skipped | — | — | — | — |
| 55 | 763 | skipped | — | — | — | — |
| 56 | 566 | planner_no_witness | G155 | insufficient_lift_or_drop | arrived | planner_no_witness |
| 57 | 848 | planner_no_witness | — | — | — | — |
| 58 | 501 | planner_no_witness | G199 | insufficient_lift_or_drop | arrived | planner_no_witness |
| 59 | 746 | skipped | — | — | — | — |
| 60 | 718 | skipped | — | — | — | — |
| 61 | 379 | skipped | — | — | — | — |
| 62 | 850 | planner_no_witness | — | — | — | — |
| 63 | 557 | planner_no_witness | — | — | — | — |
| 64 | 837 | planner_no_witness | G190 | insufficient_lift_or_drop | arrived | planner_no_witness |
| 65 | 26 | planner_no_witness | — | — | — | — |
| 66 | 408 | planner_no_witness | G155 | insufficient_lift_or_drop | arrived | planner_no_witness |
| 67 | 660 | skipped | — | — | — | — |
| 68 | 13 | skipped | — | — | — | — |
| 69 | 779 | skipped | — | — | — | — |
| 70 | 795 | planner_no_witness | — | — | — | — |
| 71 | 666 | skipped | — | — | — | — |
| 72 | 195 | planner_no_witness | — | — | — | — |
| 73 | 511 | planner_no_witness | G190 | insufficient_lift_or_drop | arrived | planner_no_witness |
| 74 | 699 | planner_no_witness | — | — | — | — |
| 75 | 488 | planner_no_witness | — | — | — | — |
| 76 | 631 | skipped | — | — | — | — |
| 77 | 29 | skipped | — | — | — | — |
| 78 | 628 | skipped | — | — | — | — |
| 79 | 264 | skipped | — | — | — | — |
| 80 | 694 | skipped | — | — | — | — |
| 81 | 542 | skipped | — | — | — | — |
| 82 | 500 | planner_no_witness | — | — | — | — |
| 83 | 98 | skipped | — | — | — | — |
| 84 | 534 | planner_no_witness | G158 | insufficient_lift_or_drop | arrived | planner_no_witness |
| 85 | 584 | skipped | — | — | — | — |
| 86 | 221 | skipped | — | — | — | — |
| 87 | 232 | skipped | — | — | — | — |
| 88 | 239 | skipped | — | — | — | — |
| 89 | 715 | skipped | — | — | — | — |
| 90 | 707 | skipped | — | — | — | — |
| 91 | 94 | skipped | — | — | — | — |
| 92 | 273 | planner_no_witness | G120 | not_held_by_selected_gripper | arrived | insufficient_lift_or_drop |
| 93 | 262 | skipped | — | — | — | — |
| 94 | 487 | planner_no_witness | — | — | — | — |
| 95 | 414 | skipped | — | — | — | — |
| 96 | 404 | skipped | — | — | — | — |
| 97 | 762 | planner_no_witness | — | — | — | — |
| 98 | 484 | skipped | — | — | — | — |
| 99 | 740 | planner_no_witness | G120 | insufficient_lift_or_drop | arrived | planner_no_witness |

## 预登记决策

house 聚类 bootstrap：I_reach 区间 [0.3157894736842105, 0.631578947368421]，配对差区间 [0.0, 0.2777777777777778]，出现救援的 house 数 18（重采样 2000 次）。
严格 Pick 物理正负校准：缺失；unknown 是否会改变方向判断：False。

## 证据边界

A 侧未出现真实夹持的原因必须在报告里与被评价器判定的几何可行性分开陈述：A 处动作链是否可执行，取决于 P1 的导航截断协议与冻结 torso，不能读成场景本身不可操作。
严格执行日志按 100 ms 记一行，非法接触与底盘漂移在每个 4 ms 步审计；无 witness 记作执行前规划失败，不等价于物理夹持失败。
原始日志：`eval_output/last_mile/p4_20260929/formal_E01/episodes/NNN/trace.jsonl`；源数据：`metrics.json`、`episodes.jsonl`。
图源：`scripts/evaluation/plot_last_mile_p4.py`；图为 SVG/PNG，报告引用 PNG。


## 配对机制诊断 #12

P3 预选 B*=G121；B 同池同预算重新评价为 `feasible`。
闭爪实际等待 7 个 100 ms 步；末步指间距 0.0445 m；双指目标接触 6 个策略步；最大抬高 0.0471 m。
B 静态 Pick=`insufficient_lift_or_drop`；转移到达误差 5.4115368058437406e-05 m / 0.4277577852403616°；实际到达位姿重评价 `unknown`，Pick=`planner_no_witness`。

## 配对机制诊断 #13

P3 预选 B*=G156；B 同池同预算重新评价为 `feasible`。
闭爪实际等待 7 个 100 ms 步；末步指间距 0.0349 m；双指目标接触 22 个策略步；最大抬高 0.0702 m。
B 静态 Pick=`success`；转移到达误差 0.0036118751694279297 m / 0.25478432892764247°；实际到达位姿重评价 `not_found`，Pick=`planner_no_witness`。

## 配对机制诊断 #14

P3 预选 B*=G155；B 同池同预算重新评价为 `feasible`。
闭爪实际等待 7 个 100 ms 步；末步指间距 0.0359 m；双指目标接触 11 个策略步；最大抬高 0.0558 m。
B 静态 Pick=`support_contact`；转移到达误差 0.00363988229360133 m / 0.5095455654140006°；实际到达位姿重评价 `unknown`，Pick=`planner_no_witness`。

## 配对机制诊断 #16

P3 预选 B*=G164；B 同池同预算重新评价为 `feasible`。
闭爪实际等待 7 个 100 ms 步；末步指间距 0.0109 m；双指目标接触 1 个策略步；最大抬高 0.0480 m。
B 静态 Pick=`insufficient_lift_or_drop`；转移到达误差 0.0033710695164361672 m / 0.340242198387118°；实际到达位姿重评价 `feasible`，Pick=`insufficient_lift_or_drop`。

## 配对机制诊断 #18

P3 预选 B*=G124；B 同池同预算重新评价为 `feasible`。
闭爪实际等待 7 个 100 ms 步；末步指间距 0.0467 m；双指目标接触 21 个策略步；最大抬高 0.0517 m。
B 静态 Pick=`success`；转移到达误差 5.2999117090735984e-05 m / 0.5101840439619664°；实际到达位姿重评价 `feasible`，Pick=`success`。

## 配对机制诊断 #28

P3 预选 B*=G185；B 同池同预算重新评价为 `feasible`。
闭爪实际等待 7 个 100 ms 步；末步指间距 0.0116 m；双指目标接触 1 个策略步；最大抬高 0.0461 m。
B 静态 Pick=`insufficient_lift_or_drop`；转移到达误差 0.003477843283443792 m / 0.22185039069264384°；实际到达位姿重评价 `unknown`，Pick=`planner_no_witness`。

## 配对机制诊断 #35

P3 预选 B*=G164；B 同池同预算重新评价为 `feasible`。
闭爪实际等待 7 个 100 ms 步；末步指间距 0.0188 m；双指目标接触 4 个策略步；最大抬高 0.0058 m。
B 静态 Pick=`insufficient_lift_or_drop`；转移到达误差 0.0033710699200566463 m / 0.3402426778196051°；实际到达位姿重评价 `feasible`，Pick=`insufficient_lift_or_drop`。

## 配对机制诊断 #39

P3 预选 B*=G124；B 同池同预算重新评价为 `feasible`。
闭爪实际等待 7 个 100 ms 步；末步指间距 0.0507 m；双指目标接触 23 个策略步；最大抬高 0.0513 m。
B 静态 Pick=`insufficient_lift_or_drop`；转移到达误差 5.290414259372109e-05 m / 0.5101840843960528°；实际到达位姿重评价 `unknown`，Pick=`planner_no_witness`。

## 配对机制诊断 #50

P3 预选 B*=G156；B 同池同预算重新评价为 `feasible`。
闭爪实际等待 7 个 100 ms 步；末步指间距 0.0007 m；双指目标接触 0 个策略步；最大抬高 0.0000 m。
B 静态 Pick=`insufficient_lift_or_drop`；转移到达误差 0.0036114421230980936 m / 0.254788605885271°；实际到达位姿重评价 `feasible`，Pick=`insufficient_lift_or_drop`。

## 配对机制诊断 #51

P3 预选 B*=G158；B 同池同预算重新评价为 `feasible`。
闭爪实际等待 7 个 100 ms 步；末步指间距 0.0007 m；双指目标接触 0 个策略步；最大抬高 0.0006 m。
B 静态 Pick=`insufficient_lift_or_drop`；转移到达误差 0.003610213185247683 m / 0.25467767835707833°；实际到达位姿重评价 `unknown`，Pick=`planner_no_witness`。

## 配对机制诊断 #56

P3 预选 B*=G155；B 同池同预算重新评价为 `feasible`。
闭爪实际等待 7 个 100 ms 步；末步指间距 0.0140 m；双指目标接触 0 个策略步；最大抬高 0.0189 m。
B 静态 Pick=`insufficient_lift_or_drop`；转移到达误差 0.003639843003758123 m / 0.509536258295432°；实际到达位姿重评价 `unknown`，Pick=`planner_no_witness`。

## 配对机制诊断 #58

P3 预选 B*=G199；B 同池同预算重新评价为 `feasible`。
闭爪实际等待 7 个 100 ms 步；末步指间距 0.0110 m；双指目标接触 1 个策略步；最大抬高 0.0476 m。
B 静态 Pick=`insufficient_lift_or_drop`；转移到达误差 0.0034750954592430864 m / 0.22151857291721683°；实际到达位姿重评价 `unknown`，Pick=`planner_no_witness`。

## 配对机制诊断 #64

P3 预选 B*=G190；B 同池同预算重新评价为 `feasible`。
闭爪实际等待 7 个 100 ms 步；末步指间距 0.0007 m；双指目标接触 0 个策略步；最大抬高 0.0181 m。
B 静态 Pick=`insufficient_lift_or_drop`；转移到达误差 0.0035760473320268814 m / 0.2544123910062225°；实际到达位姿重评价 `unknown`，Pick=`planner_no_witness`。

## 配对机制诊断 #66

P3 预选 B*=G155；B 同池同预算重新评价为 `feasible`。
闭爪实际等待 7 个 100 ms 步；末步指间距 0.0089 m；双指目标接触 19 个策略步；最大抬高 0.0404 m。
B 静态 Pick=`insufficient_lift_or_drop`；转移到达误差 0.003639964645370231 m / 0.5095409376310638°；实际到达位姿重评价 `not_found`，Pick=`planner_no_witness`。

## 配对机制诊断 #73

P3 预选 B*=G190；B 同池同预算重新评价为 `feasible`。
闭爪实际等待 7 个 100 ms 步；末步指间距 0.0007 m；双指目标接触 0 个策略步；最大抬高 0.0000 m。
B 静态 Pick=`insufficient_lift_or_drop`；转移到达误差 0.003576052786492689 m / 0.2544127569555517°；实际到达位姿重评价 `unknown`，Pick=`planner_no_witness`。

## 配对机制诊断 #84

P3 预选 B*=G158；B 同池同预算重新评价为 `feasible`。
闭爪实际等待 7 个 100 ms 步；末步指间距 0.0007 m；双指目标接触 0 个策略步；最大抬高 0.0008 m。
B 静态 Pick=`insufficient_lift_or_drop`；转移到达误差 0.0036105165186604044 m / 0.2546866711076629°；实际到达位姿重评价 `unknown`，Pick=`planner_no_witness`。

## 配对机制诊断 #92

P3 预选 B*=G120；B 同池同预算重新评价为 `feasible`。
闭爪实际等待 7 个 100 ms 步；末步指间距 0.0227 m；双指目标接触 6 个策略步；最大抬高 0.0653 m。
B 静态 Pick=`not_held_by_selected_gripper`；转移到达误差 4.8800224388199706e-05 m / 0.510184404048541°；实际到达位姿重评价 `feasible`，Pick=`insufficient_lift_or_drop`。

## 配对机制诊断 #99

P3 预选 B*=G120；B 同池同预算重新评价为 `feasible`。
闭爪实际等待 7 个 100 ms 步；末步指间距 0.0476 m；双指目标接触 21 个策略步；最大抬高 0.0501 m。
B 静态 Pick=`insufficient_lift_or_drop`；转移到达误差 4.899786067573395e-05 m / 0.5101838040062143°；实际到达位姿重评价 `unknown`，Pick=`planner_no_witness`。

## 执行审计与已知缺陷（人工补充）

执行顺序分布（A only / B→A / A→B）：{('A',): 20, ('B', 'A'): 8, ('A', 'B'): 10}；违反协议的 trace 行 0 条，非法接触行 0 条。
转移全部到达 18/18；到位误差平移最大 0.00364 m、yaw 最大 0.51018°。
转移后 replan 状态分布：{'unknown': 11, 'not_found': 2, 'feasible': 5}；B 转移后 Pick 状态分布：{'planner_no_witness': 13, 'insufficient_lift_or_drop': 4, 'success': 1}。

### 已知缺陷：夹爪 qpos 在限位外 1e-16 被夹回，伪造出 unknown

`feasibility._solve()` 用 `np.array_equal` 逐位检查「IK 只改变了所选臂」，而 `MlSpacesKinematics._constrain_state()` 每个迭代步都会把全部 move group clip 到 `joint_pos_limits`。快照恢复后的夹爪关节会落在一个「超上界约 3e-16」的浮点态上（例如 `left_gripper=[-0.04999999999754231, +0.050000000000000315]`，上界 0.05），clip 把它改成名义值 0.05 → 逐位比较失败 → `FeasibilityUnknown(ik_changed_locked_group)` → `evaluate()` 返回 `unknown`。

一个进程内可重复复现：index 12 转移后 replan、index 02 点 G140 都命中同一断言，日志里 `error` 分别为 `ik_changed_locked_group:left_gripper` 与 `:right_gripper`。

影响面（已核对，均不改变方向判断）：
- P4 formal：18 个配对里转移后 replan `unknown` 11 个，这些条目的 Pick 结果记作 `planner_no_witness` 而非物理失败；另有 5 个 replan `feasible`，真实 Pick 照样失败，说明该伪影不支配结论。
- P3 formal：全图 986 个点（圆盘 381 个）被同一断言记为 `unknown`；P3 的 `reachable_unknown` 为 0，`I_reach` 不受影响，但 `unknown` 点计入了状态分布。
- P2 formal：0 个 `unknown`（A 处在 F_IK 就返回 `not_found`，从未走到需要逐位比较的分支）。

正确修法是让 locked-group 比较带容差、或让 IK 的限位 clip 不作用于未解锁的 move group；两者都会改变 `feasibility.py` 的哈希，必须在新运行目录里重跑，不能续跑既有结果。本轮正式批**不做**该修复，只把现象如实写入报告。

### 预登记阈值逐条核对

- 继续条件：`I_reach`=0.474 ≥ 0.20 ✓；出现救援的 house 18 个 ≥ 5 ✓；配对执行改善的 house 聚类区间 [0.0, 0.2777777777777778] 下界 = 0 ✗；unknown 不改变方向 ✓；严格 Pick 物理正负校准缺失 ✗。
- 停止条件：`I_reach` 区间上界 0.632 ≥ 0.05，不满足停止阈值 ✗。
- 逐条核对的结果是「两边都不成立」，因此决策函数返回无结论；真正阻断继续的是**配对执行改善的区间下界为 0**与**缺失物理校准**，不是样本量或 house 覆盖。

### 本轮可下的事实性结论

几何层面：A 处 38/38 首失败层为 `F_IK`（P2 记录），18/38 的邻域存在可达可行 B，`I_reach`=0.474（house 聚类区间 [0.316, 0.632]）。这与 `p1-fik-is-protocol-artifact` 的结论一致：A 处够不到是 P1 截断 + 冻结 torso 的协议伪影，不能读成场景不可操作。
执行层面：18 个静态 B 只有 2 个真实 Pick 成功（`house 405` 与 `house 706`），转移全部到达、到位误差 < 4 mm / < 0.52°，但转移后只有 1 个成功、转移相对静态 B 的成功率差 -0.056。即「几何可行」与「真实抓得住」之间还有一层：多数失败是双指未能形成持续持有（`insufficient_lift_or_drop` 14/18，最大抬高常见 < 5 cm）。

结论：本轮**只支持几何代理层面的站位效应**；把它升级为真实执行层面的 last-mile，需要先修夹持/抬升链，并补物理正负校准的严格 Pick 记录。