# Last-mile 预实验文档索引

本目录集中存放 last-mile（P0–P4）预实验的验收报告、运行手册与专题分析。
每个阶段的报告与其图放在同一阶段子目录下；跨阶段的材料放在 `runbook/` 与 `analysis/`。

```
docs/last_mile/
├── p0/  清单冻结与快照恢复验收      └── figures/（如该阶段出图）
├── p1/  真实 A* 导航与状态交接
├── p2/  分层可行性评价器
├── p3/  局部站位扫描与查询预算
├── p4/  真实 Pick 执行（试点 + 正式批）
├── runbook/  运行手册（命令、判据、故障表）
└── analysis/ 专题分析（torso 漂移、reachability 场景检索）
```

## 按阶段

| 阶段 | 报告 | 内容 | 日期 |
|---|---|---|---|
| P0 | [`p0/last_mile_p0_20260921.md`](p0/last_mile_p0_20260921.md) | Minimal P0 验收：试点 10 条 + 正式 100 条清单冻结、`A→restore→A` 与顺序污染检查 10/10 通过 | 2026-09-21 |
| P1 | [`p1/last_mile_p1_e02_20260921.md`](p1/last_mile_p1_e02_20260921.md) | P1-E02 试点验收（E02） | 2026-09-21 |
| P1 | [`p1/last_mile_p1_e03_20260921.md`](p1/last_mile_p1_e03_20260921.md) | P1-E03 试点验收（E03，P2/P3 的语义基准） | 2026-09-21 |
| P1 | [`p1/last_mile_p1_202609221-100.md`](p1/last_mile_p1_202609221-100.md) | **P1 正式批（100 条）**：38 个有效 A，P2 gate open；标题写 E03 是沿用当时命名 | 2026-09-23 |
| P2 | [`p2/last_mile_p2_e02_20260922.md`](p2/last_mile_p2_e02_20260922.md) | P2-E02：分层可行性评价器（`F_base/F_IK/F_approach/F_lift_proxy`）试点验收 | 2026-09-22 |
| P3 | [`p3/last_mile_p3_e01_20260922.md`](p3/last_mile_p3_e01_20260922.md) | P3-E01 试点：5 个有效 A × 245 点局部扫描，`I_reach=0.2`，附查询预算与救援率图 | 2026-09-22 |
| P4 | [`p4/last_mile_p4_e07_20260923.md`](p4/last_mile_p4_e07_20260923.md) | P4-E07 试点：几何可行 B 已真实到达，静态与转移后 Pick 均未成功 | 2026-09-23 |
| P4 | [`p4/last_mile_p4_formal_e01_20260929.md`](p4/last_mile_p4_formal_e01_20260929.md) | **P4 正式批（100 条）**：38 个有效 A / 18 个配对，转移救援 1 次，决策「无结论」；末节记录当时的 `_constrain_state` 限位 clip 缺陷 | 2026-09-29 |
| P4 | [`p4/last_mile_p4_formal_e01_20260930.md`](p4/last_mile_p4_formal_e01_20260930.md) | **P4 正式批（修复后重跑）**：IK 不再 clip 锁定组；指标逐位不变，转移后 replan 的 11 个伪 `unknown` 转为真实标签 | 2026-09-30 |

## 运行手册

| 手册 | 覆盖范围 |
|---|---|
| [`runbook/last_mile_runbook_p0_p3_20260923.md`](runbook/last_mile_runbook_p0_p3_20260923.md) | P0–P3：各阶段命令、判据、轮次历史、故障表、产物与命名规范 |
| [`runbook/last_mile_runbook_p4_20260923.md`](runbook/last_mile_runbook_p4_20260923.md) | P4：试点与正式批命令、指标口径、决策规则 |

## 专题分析

| 分析 | 问题 |
|---|---|
| [`analysis/last_mile_torso_drift_analysis_20260922.md`](analysis/last_mile_torso_drift_analysis_20260922.md) | last-mile 的 torso 自漂移与 LLM planner 的 torso 自穿插有无共性根因（结论：不同根因，共享「阈值定在噪声底之下」的结构性缺陷） |
| [`analysis/last_mile_reachability_candidates_20260925.md`](analysis/last_mile_reachability_candidates_20260925.md) | 全库检索「pick 近端 / place 远端超限 / 桌面杂乱」场景，并解释为什么 `formal_E01` 里找不到这类干净场景 |

## 相关位置

- 实现说明：`molmo_spaces/evaluation/last_mile/README.md`
- 数据产物：`eval_output/last_mile/p{0,1,2,3,4}_<日期>/`（目录只读，换代码或协议须新建 `E0N`）
- 实施计划：主工作区根目录 `Last-mile预实验分阶段实施计划_20260921.md`（不在 worktree 内）

> 2026-09-30 整理前，报告平铺在 `docs/`、图集中在 `docs/figures/`。历史手册里出现的 `docs/last_mile_*.md`、`docs/figures/...` 路径按本目录布局对应理解；
> 默认输出路径已同步更新：`run_last_mile_p1.sh` 的 `REPORT`/`FIGURE_PREFIX` 默认值、`plot_last_mile_p3.py` 与 `plot_last_mile_p4.py` 的 `--repo` 派生路径。
> 注意报告重生成会覆盖人工补充段落（P4 正式批报告末尾的「执行审计与已知缺陷」就是此类），重跑前先备份。
