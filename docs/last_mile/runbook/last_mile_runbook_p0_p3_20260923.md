# Last-mile P0→P3 运行手册

**2026-09-23｜worktree `exp/last-mile-p0`（`.worktrees/last-mile-p0`）｜命令与判据以当前工作区为准：HEAD `2e96bec` + 未提交的 P3 formal / P4 改动。本文只覆盖 P0–P3，P4 见 `docs/last_mile/runbook/last_mile_runbook_p4_20260923.md`。**

> 2026-09-30 目录整理后，报告与图从 `docs/last_mile_*.md` + `docs/figures/` 迁到 `docs/last_mile/{p0..p4,runbook,analysis}/`，
> 每阶段的图放在该阶段的 `figures/` 子目录。下文的命令与判据仍然有效，只有路径按新布局理解。

| 阶段 | 试点（10 条 / 5 个有效 A） | 正式（formal 100 条） |
|---|---|---|
| P0 冻结 + 状态恢复 | ✅ `validation_minimal/` 10/10 | ✅ `validation_formal_minimal/` 100/100 |
| P1 真实导航 A | ✅ `p1_20260921/E04`、`p1_20260922/E05_compat` 各 5 个有效 A | ✅ `p1_20260922/formal_E01` 100 条分类、38 个有效 A |
| P2 可行性评价器 | ✅ `p2_20260922/E02` 5/5 有确定结果 | ✅ `p2_20260922/formal_E01` 38 个有效 A 全有确定标签（38×`not_found`，首失败层全为 `F_IK`） |
| P3 局部扫描 + 基线 | ✅ `p3_20260922/E01` 5×245 点 | ⬜ 入口已就绪（`SUBSET=formal`），尚未运行 |

**当前待跑：P3 formal**（§5）。P0（试点 + 正式）、P1 试点 + 正式、P2 试点 + 正式均已完成。

## 0. 速览

| 阶段 | 入口 | 输出 | 成功判据（不是"任务成功"） | 实测 |
|---|---|---|---|---|
| P0 建清单 | `run_last_mile_p0.sh build` | `p0_20260921/{pilot,formal}/` | 100 条 formal 冻结完成 | 一次即可，勿重跑 |
| P0 验收 | `run_last_mile_p0.sh validate --subset formal` | `validation_formal_minimal/` | `passed==expected==100 && COMPLETE.json` | 35.4 s/条（已跑完） |
| P1 导航 | `run_last_mile_p1.sh run --subset formal` | `p1_20260922/formal_E01/` | `classified==expected==100 && COMPLETE.json` | ✅ ≈71 s/条 → 100 条约 2 h |
| P1 出图 | `run_last_mile_p1.sh report` | `docs/last_mile/p1/*.md` + `docs/last_mile/p1/figures/` | 图与 md 生成 | ✅ 需显式设 REPORT/FIGURE_PREFIX |
| P2 评价 | `run_last_mile_p2.sh run --subset formal` | `p2_20260922/formal_E01/` | 有效 A 全部有确定标签 + `COMPLETE.json` | ✅ ≈78 s/有效 A |
| P3 扫描（试点） | `run_last_mile_p3.sh all` | `p3_20260922/E01/` | 5 个有效 A ×245 点 `complete` | ✅ 20:40→23:28（含重试与聚合） |
| P3 扫描（正式） | `SUBSET=formal run_last_mile_p3.sh all` | `p3_20260923/formal_E01/` | 全部有效 A ×245 点 `complete` | ⬜ 38 个有效 A，粗估 8–20 h |

所有命令都在 worktree 根目录执行；脚本自动设置 venv、assets、`MUJOCO_GL=egl` 和 `MLSPACES_DISABLE_CUROBO=1`，不读 API 凭据。**长任务一律在 tmux 里跑**（`tmux new -s p3formal` 或 attach 已有 session），不要用 `nohup ... &` 丢到后台：日志要能随时看，中断后才好判断该续跑还是换目录。

## 1. 前置检查

```bash
cd /home/wenyifan/wenyifan/MoMaTrajGen/.worktrees/last-mile-p0   # 等价于 /data0/wenyifan/MoMaTrajGen/.worktrees/last-mile-p0
test -x ../../molmospaces/.venv/bin/python && test -d ../../molmospaces_data/assets
```

**必做自检：确认加载的是 worktree 代码，不是主工作区。**

```bash
PYTHONPATH=$PWD ../../molmospaces/.venv/bin/python -c \
  "import molmo_spaces, molmo_spaces.evaluation.last_mile as m; print(molmo_spaces.__file__); print(m.__file__)"
```

必须打印 `.worktrees/last-mile-p0/...`。若打印 `.../molmospaces/molmo_spaces/...`，说明走的是 venv 的 editable 映射（指向主工作区），此时 `last_mile` 会直接 `ModuleNotFoundError`——这本身是响亮失败，不会静默跑错代码；处理办法就是保持 cwd 在 worktree 根目录并按脚本方式运行（脚本已设 `PYTHONPATH=$REPO`）。

环境事实：venv `/data0/wenyifan/MoMaTrajGen/molmospaces/.venv`；资产 `molmospaces_data/assets`；P0/P1 缓存写 NAS `/nas/wenyifan/molmospaces_data/cache`，P2/P3 默认写 `/tmp`。`/data0` 已用 97%，长跑前留意剩余空间（快照与轨迹是主要占用）。机器 208 核 / 251 GB。

**纪律**

1. 不要重跑 `build`：清单已冻结，重建会刷新 provenance，与既有验收记录混在一起。确需新批次时用新 `OUTPUT` 目录。
2. 不要覆盖试点目录；formal 一律写 `formal_E01` 这类新目录。
3. 不要替换失败样本，不要删除失败记录；补样只按 `reserves.jsonl` + 单独替换记录。
4. 不要改代码/配置/资产后再续跑同一目录：`inputs.json` 记录输入、实现文件哈希和 `base_commit`，任一项变化 runner 都会拒绝（这是特性）。**"实现文件"包括 runner 自己**——2026-09-23 就是改了 `run_p3.py` 和 `run_last_mile_p3.sh` 之后再跑 `p3_20260922/E01`，直接 `ValueError: P3 输入或实现已改变`。正确做法是用新的 `E0N` 目录，旧目录保持只读。
5. 单 worker 串行是设计；P1/P2 不支持手工多进程写同一输出目录（P3 除外，见 §5）。

## 2. P0：冻结数据 + 状态恢复验收

```bash
bash scripts/evaluation/run_last_mile_p0.sh test
bash scripts/evaluation/run_last_mile_p0.sh validate                     # 试点 10 条
bash scripts/evaluation/run_last_mile_p0.sh validate --subset formal     # 正式 100 条（已完成）
```

- 输出：`eval_output/last_mile/p0_20260921/validation_minimal/`、`validation_formal_minimal/`，判据看 `summary.json` 的 `expected/attempted/passed/complete` 与 `COMPLETE.json`。
- 续跑：已有成功条目的 `result.json` 会先校验产物哈希再复用；已 `failed` 的条目不会静默重试——查明原因后先把整个验证目录改名留档，再新建一轮。
- **`validation_full_interrupted/` 只是被中断的旧 full 协议调试记录（只跑完 3/10 条，无 `summary.json`/`COMPLETE.json`），不是有效验收，也不参与续跑**。有效 P0 = `validation_minimal/`（10/10）+ `validation_formal_minimal/`（100/100）。旧 `validation_debug_01..04`、`validation_assets_missing` 同理只作留档。
- 只有需要重跑旧 10 轮完整协议时才用 `validate --mode full`，输出进独立的 `validation_full/`。

## 3. P1：真实导航产生 nominal A

```bash
bash scripts/evaluation/run_last_mile_p1.sh test
# 试点（已完成；换代码后重跑要用新的 E 编号目录）
OUTPUT="$PWD/eval_output/last_mile/p1_20260922/E06" bash scripts/evaluation/run_last_mile_p1.sh run
# 正式 100 条（已完成，约 2 h）：在 tmux 里跑，日志同时落盘
tmux new -s p1formal
P1_FORMAL="$PWD/eval_output/last_mile/p1_20260922/formal_E01"
OUTPUT="$P1_FORMAL" bash scripts/evaluation/run_last_mile_p1.sh run --subset formal \
  2>&1 | tee eval_output/last_mile/p1_20260922/formal_E01_console.log
# 出图：必须在 COMPLETE.json 出现之后——report 读 summary.json，
# 而 summary.json 只在全部 episode 跑完后写出，未跑完时必然 FileNotFoundError
# REPORT/FIGURE_PREFIX 必须显式给：默认值写死指向试点 E01 的报告和图，
# 不覆盖会直接改写 E01 的文档与图片（2026-09-23 实际发生过）
OUTPUT="$P1_FORMAL" \
REPORT="$PWD/docs/last_mile/p1/last_mile_p1_formal_20260923.md" \
FIGURE_PREFIX="$PWD/docs/last_mile/p1/figures/last_mile_p1_formal_navigation" \
  bash scripts/evaluation/run_last_mile_p1.sh report
```

现有正式报告：`docs/last_mile/p1/last_mile_p1_202609221-100.md`（文件名沿用当时的命名，图仍挂在 `docs/last_mile/p1/figures/last_mile_p1_e01_navigation.*` 前缀下，容易被误认成试点图）。

产物：`episodes.jsonl`、`summary.json`、`COMPLETE.json`、`inputs.json`、`episodes/NNN/{result.json,trajectory.npz,A_snapshot.npz(仅有效 A)}`。

判据：`attempted==classified==expected && complete==true`，**不是 100 条导航全成功**；必须保留 `collision / no_path / controller_error / scene_changed / start_unavailable / timeout` 各计数。`COMPLETE.json` 只表示每条都有唯一终止分类。

Gate：试点批 `valid_A>=5` 才开 P2；formal 批以 `complete` 为准。`A_snapshot.npz` 只在 `completed` 且通过终点误差、非法碰撞、目标位移、非底盘漂移和快照恢复检查时写出。

续跑：按 `episodes/NNN/result.json` 缓存逐条复用（并校验 `artifact_sha256`），中途中断直接重跑同一命令即可；`episodes.jsonl`/`summary.json` 在全部完成后才写。

轮次历史（追责/复现时按此对号）：

| 目录 | 结果 | 说明 |
|---|---|---|
| `p1_20260921/E01` | 0 有效 A | 逐 4 ms 用 1e-3 判据查非底盘关节，把导航挠度全误判为 `controller_error` |
| `p1_20260921/E02` | 0 有效 A | 引入瞬态/稳态两层判据后仍是 5 条 `controller_error`（4×`nonbase_joint_drift`、1×`nonbase_joint_transient`）、2 条 `collision`、3 条 `no_path` |
| `p1_20260921/E03` | **5 有效 A** | 协议定稿，索引 `[2,3,5,6,8]` 即后续所有阶段的有效 A 集合 |
| `p1_20260921/E04` | 5 有效 A | 夹爪伺服修复后重跑，**P3 的语义基准（p1-reference）** |
| `p1_20260922/E05_compat` | 5 有效 A | 同输入重放：E04 快照的模型指纹无法由当前已提交模型重建，E05 逐位断言 A 与 E04 一致 |

⚠️ 后两份不能互相替代：P3 需要 `--p1-root=E05_compat`（模型兼容）+ `--p1-reference=E04`（冻结语义）。

## 4. P2：独立分层可行性评价器

```bash
bash scripts/evaluation/run_last_mile_p2.sh test
# 试点（已完成）
P1_ROOT="$PWD/eval_output/last_mile/p1_20260922/E05_compat" OUTPUT="$PWD/eval_output/last_mile/p2_20260922/E02" \
  bash scripts/evaluation/run_last_mile_p2.sh run
# 正式 100 条（已完成）：在 tmux 里跑；必须在 P1 formal 出 COMPLETE.json 之后
tmux new -s p2formal
P1_FORMAL="$PWD/eval_output/last_mile/p1_20260922/formal_E01"
P2_FORMAL="$PWD/eval_output/last_mile/p2_20260922/formal_E01"
P1_ROOT="$P1_FORMAL" OUTPUT="$P2_FORMAL" bash scripts/evaluation/run_last_mile_p2.sh run --subset formal \
  2>&1 | tee eval_output/last_mile/p2_20260922/formal_E01_console.log
```

- 协议（继承 P2-E01/E02）：32 候选 × 2 臂 × 3 IK 初值 = 192 条链，IK ≤300 迭代，pregrasp 0.04 m，抬升代理 0.05 m，base/torso 固定，CuRobo 与 LLM 均禁用，每 A 完整评价 2 次做重复性检查。
- 输出仍应有 100 行：P1 无有效 A 的条目记 `skipped/invalid_A`（**不算 manipulation failure**），有效 A 才跑 `F_base → F_IK → F_approach → F_lift_proxy`。
- 判据：有效 A 全部有 `feasible/not_found/unknown` 确定标签、两次结果一致、`complete==true` 且存在 `COMPLETE.json`。
- `not_found` 的语义始终是"本固定有限协议未找到完整链"，不是数学上不可达。
- P2 formal 要求 P1 的 `episodes.jsonl` 行数等于 100，否则直接 `ValueError`。

## 5. P3：局部 SE(2) 扫描 + 公平预算基线

```bash
bash scripts/evaluation/run_last_mile_p3.sh test
# 试点（已完成；换代码后重跑要用新的 E 编号目录，p3_20260922/E01 保持只读）
bash scripts/evaluation/run_last_mile_p3.sh all        # = run --workers 20 --shards-per-episode 8 + report
# 正式 38 个有效 A（当前待跑，粗估 8–20 h）：在 tmux 里跑，SUBSET=formal 自动换整套默认路径
tmux new -s p3formal
SUBSET=formal bash scripts/evaluation/run_last_mile_p3.sh all
#   展开后等价于：
#   --subset formal --p0-validation <p0>/validation_formal_minimal
#   --p1-root=--p1-reference=$PWD/eval_output/last_mile/p1_20260922/formal_E01
#   --p2-root=$PWD/eval_output/last_mile/p2_20260922/formal_E01
#   --output=$PWD/eval_output/last_mile/p3_20260923/formal_E01   （全新目录，不会碰到 E01）
# 分步跑（同一套默认路径，只是显式写出来）
SUBSET=formal bash scripts/evaluation/run_last_mile_p3.sh run --workers 20 --shards-per-episode 8
```

- 扫描区域：A 的底盘局部系方形网格 245 点，主统计区为欧氏半径 ≤30 cm 的圆盘 145 点；每点都是同一 A 快照恢复后独立评价，A 的结果复用 P2 缓存，每个有效 A 新增 244 个 B 查询。
- 输出：`episodes/NNN/points/G*.json`（逐点互斥缓存，分片崩溃后重跑同一命令可续）、`candidates.jsonl`、`metrics.json`、`summary.json`、`COMPLETE.json`；`--workers` 与 `--shards-per-episode` 是 P3 唯一允许的并行入口（spawn 进程池，天然每分片一次场景初始化，内存可控）。试点 E01 的实测窗口为 20:40→23:28（含重试与末尾 5 分钟聚合阶段），比 P1/P2 重得多；提高 `P3_WORKERS` 前先看内存余量。
- formal 与试点的协议差异：formal 只用单一 heuristic 距离 `0.65`（试点扫 0.55/0.65/0.75 三档），`I_reach` 的分母改为 `N_eval`（可达性有确定结论的点），另给 `I_reach_all_valid_lower` / `I_reach_upper` 两个界。**formal 不产出图文报告**：`report` 会直接 `exit 2`，读 `metrics.json`/`summary.json` 即可，不要拿试点的 `plot_last_mile_p3.py` 去画正式数据。
- 单 episode 调试：必须用**独立 smoke 目录**，且 `--max-b-points` 只能与 `--episode-index` 同用：`... run --episode-index 8 --max-b-points 4 --output eval_output/last_mile/smoke_xxx`。正式输出目录禁止出现 smoke 参数。
- 两份 P1 不能混：试点批需要 `--p1-root=E05_compat` + `--p1-reference=E04`（E04 快照的模型指纹无法由当前提交重建，E05 是逐位一致的重放）；formal 批因为 P1 是一次跑完的，脚本把 `p1-reference` 直接指向 `formal_E01` 自己即可。
- 2026-09-23 的代码改动放开了 formal：`run_p3.py` 新增 `--subset`、试点专用断言（有效 A 索引 `[2,3,5,6,8]`）只在 `pilot` 下生效、`expected` 按 10/100 分支。这也意味着**改动之前产出的 `p3_20260922/E01` 不能被新版代码续跑**（实现哈希变了），详见 §1 纪律 4。

## 6. 状态自检

```bash
cd /home/wenyifan/wenyifan/MoMaTrajGen/.worktrees/last-mile-p0
../../molmospaces/.venv/bin/python - <<'PY'
import json
from pathlib import Path
runs = {
 "P0 pilot":   "eval_output/last_mile/p0_20260921/validation_minimal",
 "P0 formal":  "eval_output/last_mile/p0_20260921/validation_formal_minimal",
 "P1 E03":     "eval_output/last_mile/p1_20260921/E03",
 "P1 E04":     "eval_output/last_mile/p1_20260921/E04",
 "P1 E05c":    "eval_output/last_mile/p1_20260922/E05_compat",
 "P2 E02":     "eval_output/last_mile/p2_20260922/E02",
 "P3 E01":     "eval_output/last_mile/p3_20260922/E01",
 "P1 formal":  "eval_output/last_mile/p1_20260922/formal_E01",
 "P2 formal":  "eval_output/last_mile/p2_20260922/formal_E01",
 "P3 formal":  "eval_output/last_mile/p3_20260923/formal_E01",
 "P4 E07":     "eval_output/last_mile/p4_20260923/E07_cuda5",
}
for name, path in runs.items():
    d = Path(path)
    if not d.exists():
        print(f"{name:10s} 未运行")
        continue
    out = []
    for key in ("attempted", "classified", "valid_A", "passed", "evaluated_A", "scan_completed"):
        s = d / "summary.json"
        if s.exists() and key in json.loads(s.read_text()):
            out.append(f"{key}={json.loads(s.read_text())[key]}")
    done = (d / "COMPLETE.json").exists()
    print(f"{name:10s} COMPLETE={done} " + " ".join(out))
PY
```

长任务进行中看进度：控制台日志被 LMDB 进度条灌满（大量 `\r`、行数不可靠），用 JSON 行数才准：

```bash
LOG=eval_output/last_mile/p1_20260922/formal_E01_console.log
grep -c '^{' "$LOG"                      # 已完成条数（每条打印一行 {subset_index, house, status, ...}）
ls eval_output/last_mile/p1_20260922/formal_E01/episodes | wc -l    # 已开始条数
test -f eval_output/last_mile/p1_20260922/formal_E01/COMPLETE.json && echo "跑完，可以 report"
```

P1/P2 每条 episode 完成都会打印一行 `{subset_index, house, status, termination_detail, elapsed_sec}`；`episodes.jsonl`、`summary.json`、`COMPLETE.json` 只在全部完成后一次性写出，所以"没有 summary.json"不等于任务失败。

## 7. 常见故障

| 现象 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: molmo_spaces.evaluation.last_mile` | 加载了主工作区的 `molmo_spaces`（无 last_mile） | 在 worktree 根目录运行脚本，见 §1 自检 |
| `ValueError: … 输入或实现已改变`（P3 报 `P3 输入或实现已改变`） | 输入、配置或**实现文件本身**（`run_p3.py` / `run_last_mile_p3.sh` / `feasibility.py` / `local_search.py`）变了，或 `base_commit` 变了 | 旧目录保持只读，用新的 `E0N` 目录重跑；正式批直接 `SUBSET=formal`（落到全新的 `p3_20260923/formal_E01`），**不要**删改旧 `inputs.json` 绕过校验 |
| `ValueError: P1 结果数量 … 与 formal 子集 100 不一致` | P1 未完成或 `--subset` 与目录不匹配 | 先让 P1 formal 出 `COMPLETE.json` |
| `ValueError: P3 只能扫描有效 A` / `有效 A 索引改变` | 传错 `--episode-index`，或 pilot 批的 P1 输入不是 E03/E04/E05 那套（该断言现在只在 pilot 生效） | 试点用默认 `p1-root=E05_compat`、`p1-reference=E04`；formal 用 `SUBSET=formal` |
| `P3 formal 图文报告尚未实现` | formal 的 `report` 显式禁用 | 直接读 `metrics.json`/`summary.json` |
| 大量 `controller_error:nonbase_joint_drift` | E01 时代的旧判据（逐 4 ms 用 1e-3） | 已是历史；当前用瞬态/稳态两层判据，勿回退 |
| P1 跑得比预期慢 | 每条含场景初始化 + 真实 MuJoCo 步进 | 与试点的 s/条 线性外推，勿中途改协议"加速" |
| 长跑内存上涨 | glibc arena 膨胀（非对象泄漏） | 长任务可加 `MALLOC_ARENA_MAX=2`；P3 分片天然有界 |
| `summary.experiment` 与目录名不一致 | 协议名硬编码为 `P1-E03`/`P2-E01` | 以目录名 + `inputs.json` 哈希追溯轮次，不要看该字段 |

## 8. 产物与命名规范

- 阶段目录：`eval_output/last_mile/p{0,1,2,3,4}_<日期>/`；同一阶段换代码或协议 → 新 `E0N` 目录，formal 批用 `formal_E01`。旧目录一律只读（P4 手册同样规定"P0–P3 原目录只读，不重建、不覆盖"）。
- 每个运行必带 `inputs.json`（输入 + 实现 + base_commit 指纹）与 `COMPLETE.json`（P3 另写 `provenance.json`）；`COMPLETE.json` 只证明"该阶段该子集全部条目都有唯一分类"，不证明任务成功。
- 报告与图：`docs/last_mile/p<阶段>/last_mile_<阶段>_<实验号>_<日期>.md` + 同目录 `figures/` 下同名 png/svg；图表只画本次真实数据。
- 相关文档：计划 [`Last-mile预实验分阶段实施计划_20260921.md`](../../../../../Last-mile预实验分阶段实施计划_20260921.md)（在主工作区根目录，不在 worktree 内）、阶段验收 `docs/last_mile/p{0,1,2,3}/*.md`（正式批报告是 `docs/last_mile/p1/last_mile_p1_202609221-100.md`）、**P4 另有手册 `docs/last_mile/runbook/last_mile_runbook_p4_20260923.md`**、实现说明 `molmo_spaces/evaluation/last_mile/README.md`、P0 冻结数据说明 `eval_output/last_mile/p0_20260921/README.md`（该文写于 P3 formal 支持之前，"当前还没有 P3 runner"等句子已过时，以本手册为准）。
