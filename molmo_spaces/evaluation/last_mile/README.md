# Last-mile P0：数据冻结和恢复验证

本次按用户确认的 **Minimal P0** 收尾：只做 RBY1 val 内诊断准备，不运行导航、抓取或策略；通过后结束本次任务，不进入 P1/P2。新子集保留原 PnP 任务定义；不构成官方 PnP 新评测结果。

## 运行

在独立 worktree 中执行：

```bash
bash scripts/evaluation/run_last_mile_p0.sh build
bash scripts/evaluation/run_last_mile_p0.sh test
bash scripts/evaluation/run_last_mile_p0.sh validate
```

默认共享工作区中的 `molmospaces/.venv` 和 `molmospaces_data/assets`。可通过 `WORKSPACE_ROOT`、`PYTHON_BIN`、`MLSPACES_ASSETS_DIR`、`MLSPACES_CACHE_DIR`、`OUTPUT`、`SOURCE` 覆盖；不读取 API 凭据。

默认输出：`eval_output/last_mile/p0_20260921/`。

- `pilot/benchmark.json`：10 条，5 houses，每 house 2 个不同目标。
- `formal/benchmark.json`：100 条，按 house 轮询，至多 5 个目标/house，排除试点 houses。
- 每组独立的 `benchmark_metadata.json`、`manifest.jsonl`；根目录联合 manifest。
- `reserves.jsonl`：冻结候补顺序，按 `reserve_group` 区分试点/正式候补。
- `excluded.jsonl`：类别不符、元数据缺失和重复目标，保留源索引与原因。
- `config.json`、`provenance.json`：种子、阈值、源哈希、资产元数据指纹、代码版本。
- `validation_minimal/`：本次生效的简化协议、逐条配置、数值快照、失败详情、汇总及全部通过后的 `COMPLETE.json`。
- 旧 `validation_debug_*`、`validation_assets_missing`、`validation_full_interrupted` 仅保留工程调试记录，不与本次结果合并计数。

子集 JSON 不更改 episode 内容；P0 专用 sampler 在深拷贝上应用 manifest 的种子。直接使用旧评测入口可加载子集，但不会自动应用 P0 sidecar，也不等于执行了 P0 协议。

## 抽样规则

类别来自场景对象元数据，以其 `asset_id` 查验资产注释存在；场景类别负责区分酒瓶和调料瓶，另保存资产原始类别。缺失元数据不使用名称补猜。去重单位为相同 dataset/split 内的 `(house, pickup_obj_name)`。

主种子 `20260921`，所有选择顺序通过 SHA256 固定。同一目标多个源 episode 取固定哈希顺序第一条；不参考抓取结果或轨迹长度。先取能提供两个不同目标的前 5 个 houses 作试点，再对其余 houses 轮询选择正式集。

候补不自动替换：失败仍留在原分母。需要补样时另建替换记录，包含原索引、原因、候补索引和替换后的 house 分布；试点候补仅来自已冻结试点 houses，正式候补不得来自试点 houses。不复制条目或增加 seed 凑独立目标数。

## 状态接口和边界

- `EpisodeSnapshot.capture(task, provenance)` / `restore(task, provenance)`：MuJoCo `mjSTATE_INTEGRATION`、控制器、任务缓存、已审计传感器缓存、Python/NumPy/PyTorch RNG、传感器空间 RNG。
- `save` / `load`：单个原子写入 NPZ，JSON 元数据＋数组，禁用 pickle。
- `snapshot.restored(task)`：异常退出也恢复。
- `override_base_pose(task, pose)`：只改底盘位姿及对应控制目标/输入，不执行稳定步。

只支持单环境和未注册策略；未知控制器、传感器、策略或不匹配模型必须拒绝。模型指纹覆盖编译后的 MuJoCo 模型；**不支持在快照之间任意修改模型参数或更换模型**。P1/P2 新增策略、传感器或修改相机/模型的逻辑，须先扩展适配器并重新测试，不能默认本次验收已经覆盖。

P0 使用独立 sampler 强制关闭 repair、拒绝机器人替换；源数据中出现缺失 body 不能沿用通用 sampler 的静默跳过。原默认 sampler 保持不变。

脚本显式设置 `MLSPACES_DISABLE_CUROBO=1`，避免通用 policy config 在导入时加载可选 CuRobo 依赖；不设置该变量时原行为不变。P0 不实例化 Dummy/LLM/CuRobo 策略。

## Minimal P0 验收与续跑

默认 `validate` 执行 Minimal P0，10 条试点逐条验证：

1. 从固定 seed 初始化，显式关闭自动 repair，保存并重新加载 snapshot。
2. 记录直接 A；扰动关节速度、控制输入/target、任务与传感器缓存、RNG 后恢复，确认 A 与直接 A 一致。
3. 从同一快照将底盘平移 0.1 m 得到测试 B，执行状态检查，再恢复到 A，确认与直接 A 一致。

这里的 A/B 是工程测试位姿，不是 P1 导航终点或 P2 可行性标签。数值状态 `rtol=0, atol=1e-10`；对象位置/旋转矩阵/速度 `atol=1e-8`；离散状态与 RNG 精确比较。Minimal P0 不运行 10 轮重复恢复和 20 步动力学重放。

当前有效协议位于 `validation_minimal/inputs.json` 的 `validation_protocol`；根目录 `config.json` 保留原冻结记录中的完整验收参数，不作为本轮简化运行参数。仅在另行需要时才使用 `validate --mode full` 运行旧完整验收，输出到独立的 `validation_full/`。

续跑校验数据、配置、实现和快照哈希，不重复计数。已有失败不会自动当作未运行重试；修复后将对应验证目录改名保留，再新建一轮。`--limit 1` 只做调试，不能判定完整 Minimal P0 通过。正式 100 条本轮不运行仿真。
