# Planner 评测入口

RBY1 与 PandaOmron 的 oracle CuRobo / LLM waypoint 均由 molmospaces 维护，
不再依赖 MolmoBot 的 `olmo` 配置或 `run_eval.py`。

| 机器人 | CuRobo oracle | LLM waypoint |
|---|---|---|
| RBY1 | `RBY1CuroboPickPnPEvalConfig` | `RBY1LLMWaypointPickPnPEvalConfig` |
| PandaOmron | `PandaOmronCuroboPickPnPEvalConfig` | `PandaOmronLLMWaypointPickPnPEvalConfig` |

配置模块统一为 `molmo_spaces.evaluation.configs.evaluation_configs`。

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
