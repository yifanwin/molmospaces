# Planner 评测入口

RBY1 与 PandaOmron 的 oracle CuRobo / LLM waypoint 均由 molmospaces 维护，
不再依赖 MolmoBot 的 `olmo` 配置或 `run_eval.py`。

| 机器人 | CuRobo oracle | LLM 决策 | 几何对照（不调 API） |
|---|---|---|---|
| RBY1 | `RBY1CuroboPickPnPEvalConfig` | `RBY1LLMWaypointPickPnPEvalConfig` | `RBY1LLMWaypointPickPnPGeometricEvalConfig` |
| PandaOmron | `PandaOmronCuroboPickPnPEvalConfig` | `PandaOmronLLMWaypointPickPnPEvalConfig` | `PandaOmronLLMWaypointPickPnPGeometricEvalConfig` |

配置模块统一为 `molmo_spaces.evaluation.configs.evaluation_configs`。

## LLM 的职责边界

LLM 是**高层 last-mile 决策器**，不输出任何 waypoint：

```
LLM high-level decision -> deterministic geometric waypoint generation
                        -> MuJoCo IK / collision validation -> execution
```

模型只回答 `grasp_candidate_id`、`arm`、`approach_strategy`、`lift_height`、
`preplace_height`，以及可选的 `base_approach_goal` / `base_transfer_goal`。
六个 EE 目标与底盘目标由 `build_plan_from_decision` 确定性展开；grasp pose 直接取
所选候选，模型无法修改。校验失败时模型只能调整上述高层旋钮，改不了轨迹点。
`approach_strategy` 不改变生成的几何，只用于校验候选姿态与所声称策略是否一致。

`geometric` 档（两个 `...GeometricEvalConfig`）走完全相同的展开与预检链路，但决策
来自本地几何默认值，因此不需要任何 LLM 环境变量，用于回答"LLM 的高层决策相比几何
默认值有没有增量价值"。

## mock fixture 契约

`mock` 模式需要 `LLM_MOCK_RESPONSE_FILE`，内容必须是 **`LLMDecision`** 而不是旧的
waypoint plan。最小形状见 `mlspaces_tests/evaluation/llm_decision_mock.json`：

```json
{"schema_version": 1, "robot": "rby1", "arm": "left_arm", "grasp_candidate_id": 0,
 "approach_strategy": "lateral", "lift_height": 0.35, "preplace_height": 0.08}
```

给旧格式（含 `segments` 数组）时客户端会在构造阶段直接报错。mock 每次返回同一份
内容，所以反馈循环在 mock 下无法自愈，只能验证管线连通，不验证纠错能力。

## 接触预检的穿透容差

`_assert_no_new_robot_contact` 原先只看"有没有新增接触对"，不看穿透深度。RBY1 的
`link_torso_2` 与 `link_torso_4` 建模间隙只有 15 mm，而轨迹预检拿上一步 IK 的解
作下一步种子，偏差会链式累积，让这对 link 出现亚毫米级的网格外壳互穿——实测噪声
仅 0.047 mm。旧判据把这类数值噪声当成致命碰撞，是 55 次预检失败里 41 次的真因。

现在按接触类型分别设容差（`LLMWaypointPlannerPolicyConfig`）：

| 类型 | 容差 | 依据 |
|---|---|---|
| 自接触（两侧都是机器人） | 2 mm | 实测噪声 0.047 mm、边缘样本 1.66–1.84 mm；执行期会被 MuJoCo 的接触求解器推开 |
| 机器人与环境接触 | 1 mm | 实测真干涉 4.78–12.54 mm（手指撞桌/容器） |

两者设 0 精确回到旧行为。每次预检被放行的擦碰次数与最深深度写进 plan artifact 的
`contact_stats`，成功与失败路径都有——调容差时这是唯一可比的数字。

容差解决的是**误判**，不是所有失败。环境碰撞那一类是直线插值路径无避障导致的真
干涉，需要避障路径规划。

## RBY1 夹爪的位置伺服修正

RBY1 的 MJCF 把两个夹爪 actuator 声明成 `<motor>`（其余 25 个关节都是
`<position>`），于是 XML 里残留的 `biasprm`（kp=4000、kv=400）因为
`biastype=NONE` 完全失效，`gainprm` 也停在 `<motor>` 的默认值 1——ctrl 被当成
**力矩**用：闭合指令 `0.0` 对应零力矩，手指纹丝不动。

后果是**规划全都白做**：夹爪从未碰到过物体。103:1 全场 205 步
`inter_finger_dist` 恒为 0.1（全开），`grasp_state_pickup_obj` 的
`touching`/`held` 恒为 `False`，物体停在起始位置——reward 反推的"物体到容器
AABB 中心 0.3328 m"与 plan 里 place 目标反推的 0.3308 m 吻合到 2 mm。

这条也解释了此前"预检全过、动作序列跑满 600 步、`judge_success` 仍为 False"
的执行期零成功：`GripperAction` 下发的闭合指令在 h5 的 `actions/joint_pos` 里
看得见（step 52 起为 `0.0`），但同期的 `obs/agent/qpos` 始终是 `-0.05`。

`RBY1.apply_control_overrides` 现在按其余 `<position>` 关节同样的 kp/kv 把两个
夹爪还原成位置伺服（参数见 `rby1.py` 顶部的 `GRIPPER_KP` / `GRIPPER_KV` /
`GRIPPER_CTRL_RANGE`），ctrl 即目标关节角：`-0.05` 全开、`0.0` 闭合，与
`RBY1GripperGroup.set_gripper_ctrl_open` 的取值一致。回归测试见
`mlspaces_tests/component_tests/test_rby1_gripper.py`。

从 molmospaces 目录执行：

```bash
source setup_env.sh
.venv/bin/python -m molmo_spaces.evaluation.eval_main \
  molmo_spaces.evaluation.configs.evaluation_configs:RBY1CuroboPickPnPEvalConfig \
  --benchmark_dir /path/to/RBY1/benchmark \
  --house_index 4 --num_workers 1 --no_wandb \
  --task_horizon_steps 600 --output_dir "$PWD/eval_output"

bash scripts/evaluation/run_house4_llm_waypoint_eval.sh baseline
# 其他模式：gate、full、mock
.venv/bin/python scripts/evaluation/summarize_llm_waypoint_eval.py /path/to/run
```

成对脚本默认使用本仓 `.venv`，结果写到本仓 `eval_output/`。PandaOmron 执行
Franka benchmark，属于跨机器人评测。`RBY_BENCH`、`PANDA_BENCH`、`OUT`、
`PYTHON_BIN` 可由环境变量覆盖。沿用旧脚本的两侧 600 步上限，避免迁移改变预算；
单独调用原生入口时须显式设置上限，否则按 benchmark 推导。

LLM 模式沿用工作区根目录 `.env`（可用 `LLM_ENV_FILE` 改路径），不覆盖已导出的
环境变量，不打印凭据。直接调用原生入口可传 `--env_file /path/to/.env`。
`mock` 另需导出 `LLM_MOCK_RESPONSE_FILE`；它仍执行仿真和本地校验。

旧的 `MolmoBotRBY1CuroboPickPnPEvalConfig`、
`MolmoBotRBY1LLMWaypointPickPnPEvalConfig` 和 MolmoBot 下的成对启动/汇总脚本
已移除，调用方应切换上述路径。MolmoBot 的 `run_eval.py` 只接受其 learned policy。
历史输出保留原位，不改写旧日志、配置快照或结果目录。

当前 `setup_env.sh` 的 assets 是工作区的 `molmospaces_data/assets`，与
`molmo_spaces_data/assets` 不同；不要将本地 assets 整体替换为 NAS cache，
其中可能包含依赖本地文件系统的 LMDB。benchmark 可以单独使用 NAS 绝对路径。
