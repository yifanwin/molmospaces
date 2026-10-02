# Last-mile P4 运行手册（2026-09-23）

本轮在 `exp/last-mile-p0` worktree 实施 P4。P4-E07 是冻结 10 条试点的正式冒烟目录；E01–E06 是输入兼容、夹爪控制和接触判定的调试记录，不参与结果汇总。P0–P3 原目录只读，不重建、不覆盖。`COMPLETE.json` 表示全部条目已分类并且预选 B 已在当前模型重评价为 feasible，不表示真实 Pick 成功。

## 试点命令

```bash
cd /home/wenyifan/wenyifan/MoMaTrajGen/.worktrees/last-mile-p0
bash scripts/evaluation/run_last_mile_p4.sh test
OUTPUT="$PWD/eval_output/last_mile/p4_20260923/E07_cuda5" bash scripts/evaluation/run_last_mile_p4.sh run
OUTPUT="$PWD/eval_output/last_mile/p4_20260923/E07_cuda5" bash scripts/evaluation/run_last_mile_p4.sh report
```

脚本默认 `CUDA_VISIBLE_DEVICES=5`、`MUJOCO_EGL_DEVICE_ID=5`、`MUJOCO_GL=egl`，禁止 LLM/CuRobo。几何评价和 MuJoCo 执行主要用 CPU；记录 cuda:5 是设备可见性与 EGL 来源，不是 GPU 加速声明。`inputs.json` 记录设备变量、上游完成标记和代码指纹。同目录恢复时若输入或实现变化，runner 会拒绝；使用新的 E 编号目录。

当前 P1-E05_compat 快照的模型指纹无法在当前环境重建，**不能绕过恢复检查**。本轮改用可严格恢复的 P1-E04；运行器逐位核对其 A 与 P3 冻结 A，一旦有差异立即拒绝。P3 的 B* 仍在看真实 Pick 前固定，但其旧 witness 不直接跨模型执行：先在 E04 当前模型用 P2 同一抓取池/预算对 B* 重评价，再执行新 witness。若重评价不再 feasible，P4 不写 `COMPLETE.json`。

试点预期结构：10 条源记录、5 个有效 A、5 次固定预算 A Pick；只有 #8 的 B*=G169 进入静态 B 与实际 A→B→Pick 配对。无 witness 计作执行前规划失败，不表示物理夹爪接触失败。转移使用 P3 保存的走廊，通过真实底盘控制器执行；在实际到达位姿重新评价并 Pick，不瞬移补偿误差。A/B 顺序以 episode seed 和 ID 决定，双方独立恢复同一 A 快照。

## 结果检查

```bash
../../molmospaces/.venv/bin/python -m json.tool eval_output/last_mile/p4_20260923/E07_cuda5/summary.json
../../molmospaces/.venv/bin/python -m json.tool eval_output/last_mile/p4_20260923/E07_cuda5/metrics.json
../../molmospaces/.venv/bin/python -m json.tool eval_output/last_mile/p4_20260923/E07_cuda5/episodes/008/result.json
```

逐条证据在 `episodes/NNN/{result.json,trace.jsonl}`。执行日志按 100 ms 记一行，机器人/目标非法接触、底盘漂移和转移目标扰动在每个 4 ms MuJoCo 步审计，违例会额外记 `phase=violation`。严格 Pick：0.5 s 闭爪 + 0.2 s 余量（至少 7 个策略步）、选定两指持有、脱离支撑、目标抬高至少 5 cm，连续维持 1 s；不使用旧 PnP `success`。`planner_no_witness` 是固定有限协议失败，`unknown` 和审计失败不可偷偷按 Pick 失败处理。

`metrics.json` 的主 incidence 分母 `N_eval` 只含确定 A 标签和完整确定邻域的源目标实例；另报 `N_nav`、`N_attempt` 与 unknown 界限。执行层的 A 成功率以全部有效 A 为目标群体，预选配对差只适用于代理选中子集。试点不算 house 聚类区间，也不触发 Agent 决策。报告和经视觉检查的 SVG/PNG 位于 `docs/last_mile/p4/last_mile_p4_e07_20260923.md`、`docs/last_mile/p4/figures/`。

## 正式 100 条（2026-09-29 已完成）

P1 formal、P2 formal 必须先各有有效 `COMPLETE.json`。P3 已支持 `SUBSET=formal`，formal 的 P1 reference 可与 P1 root 为同一完成运行，fixed-radius 固定使用试点选择的 0.65 m，不在 formal 地图重新挑选；输出必须是新 `formal_E01` 目录。P3 formal 只支持扫描/机器可读汇总；旧 P3 绘图脚本硬编码试点，formal `report` 会明确拒绝，不能把试点图误标为正式结果。之后设置 `SUBSET=formal`、相应 `P1_ROOT/P2_ROOT/P3_ROOT/OUTPUT` 执行 P4 脚本。正式方向判定还需 `--calibration-json` 指向已审计的物理正负 Pick 控制记录；没有该记录，决策函数只能返回“无结论”。不得用试点逻辑单测冒充物理阳性校准。

运行记录：`eval_output/last_mile/p4_20260929/formal_E01/`（100 条 / 38 个有效 A / 18 个配对，约 1 小时 10 分钟，`COMPLETE.json` 已写）。报告 `docs/last_mile/p4/last_mile_p4_formal_e01_20260929.md`，图 `docs/last_mile/p4/figures/last_mile_p4_formal_e01_paired_pick.*`。决策输出 `无结论`。

```bash
# 复现本轮两步：先跑 run，再跑 report（report 会重建报告与图，随后追加审计段）
SUBSET=formal OUTPUT="$PWD/eval_output/last_mile/p4_20260929/formal_E01" \
  bash scripts/evaluation/run_last_mile_p4.sh run
SUBSET=formal OUTPUT="$PWD/eval_output/last_mile/p4_20260929/formal_E01" \
  bash scripts/evaluation/run_last_mile_p4.sh report --date 20260929
```

⚠️ **环境指纹**：P4 formal 依赖 P1 formal 的快照逐位恢复，必须同时满足 (1) 从 `/home/wenyifan/wenyifan/...` 软链路径调用（让 `P1Config.output_dir` 字符串与快照记录一致）、(2) `MLSPACES_ASSETS_DIR=$PWD/../molmospaces_data/assets`（`.worktrees/molmospaces_data/assets`，它才与快照的 `model_sha256` 一致）。任何一条不满足都会报 `任务配置与快照不匹配` / `快照模型指纹不匹配`。`/data0` 直连路径与仓库根的 `molmospaces_data/assets` 都会失败。

⚠️ **已知实现缺陷（2026-09-30 已修复，上一轮结果保留只读）**：`_solve()` 的 locked-group 逐位比较被 `_constrain_state()` 的限位 clip 打破（夹爪 qpos 超上界 1e-16 → 被夹到名义值）。表现是把可评价样本记成 `unknown`：P4 2026-09-29 那轮转移后 replan 11/18 unknown、P3 formal 986 点 unknown。修复为「`_constrain_state()` 只 clip IK 解锁的 move group」，见 `molmo_spaces/kinematics/mujoco_kinematics.py`；因该文件是 IK 行为的一部分，`run_p2/p3/p4` 的实现指纹已补上它，旧目录会明确拒绝续跑而不是静默混用。

## 修复后重跑（2026-09-30）

```bash
SUBSET=formal OUTPUT="$PWD/eval_output/last_mile/p4_20260930/E01" \
  bash scripts/evaluation/run_last_mile_p4.sh run
SUBSET=formal OUTPUT="$PWD/eval_output/last_mile/p4_20260930/E01" \
  bash scripts/evaluation/run_last_mile_p4.sh report --date 20260930
```

报告 `docs/last_mile/p4/last_mile_p4_formal_e01_20260930.md`，图 `docs/last_mile/p4/figures/..._20260930_paired_pick.*`。

对照结论：**所有 incidence 与配对指标逐位不变**（`I_reach`=0.474、配对四格 `{00:16, 01:2, 10:0, 11:0}`、转移后成功 1）；唯一变化是转移后 replan 的 11 个 `unknown` 消失，变成与同条目静态 B 相同的标签（`feasible` 16 / `not_found` 2），转移后 Pick 状态从 13 个伪 `planner_no_witness` 变成真实的 `insufficient_lift_or_drop` 等。即修复只改了「无法规划」的标签归属，没有改变任何一次真实抓取成败，决策仍为「无结论」。

⚠️ **同 run_tag 的图会被日期区分**：`plot_last_mile_p4.py` 的图名与报告名都带 `--date`，同一子集多次运行时不会互相覆盖；重生成报告仍会擦掉人工补充段，先备份。

**尚未在修复后实现上重跑的阶段**：P3 formal（`p3_20260923/formal_E01`）与 P2 formal（`p2_20260922/formal_E01`）仍是修复前产物。它们的结论不受影响（P3 的 unknown 点全部 `reachable=False`、P2 无 unknown），但要引用「同一实现下的完整证据链」需新目录重跑。
