#!/usr/bin/env bash
set -euo pipefail

# 在指定的 (house, episode) 场景上跑 RBY1 LLM waypoint 闭环评测，用于复查
# “已知可解场景上 LLM waypoint 能否闭环”，并产出有时长的运动视频。
#
# 场景以 house:idx 传入，idx 是该 house 过滤后的 episode 下标
# （与《多机器人抓放评测_高成功率任务_Episode_House分析.md》里的
# “house N episode M”一一对应，已用 benchmark.json 核对）。
#
# 本脚本与 run_house4_llm_waypoint_eval.sh 相互独立：后者被
# mlspaces_tests/evaluation/test_planner_migration.py 断言了 4 个 mode 的行为，
# 不应改动。

# Run from any directory. This script never prints API credentials.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MOLMOSPACES_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
source "$MOLMOSPACES_ROOT/setup_env.sh"
cd "$MOLMOSPACES_ROOT"

# setup_env.sh 默认把渲染放到 GPU 2；本脚本默认改用空闲的 GPU 3，
# 可用 CUDA_DEVICE 覆盖。
CUDA_DEVICE="${CUDA_DEVICE:-3}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export MUJOCO_EGL_DEVICE_ID="$CUDA_DEVICE"

RBY_BENCH="${RBY_BENCH:-$MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v1/procthor-10k/RBY1PickAndPlaceDataGenConfig/RBY1PickAndPlaceDataGenConfig_20260209_json_benchmark}"
OUT="${OUT:-$MOLMOSPACES_ROOT/eval_output/llm_waypoint_scenes}"
PYTHON_BIN="${PYTHON_BIN:-$MOLMOSPACES_ROOT/.venv/bin/python}"
TASK_HORIZON_STEPS="${TASK_HORIZON_STEPS:-600}"

# 默认场景集（分析文档中的稳定正例 + planner 多次成功案例）：
#   103:1 / 97:3 — CuRobo 与 learned policy 都成功的跨方法正例
#   95:0 / 95:1 — wine bottle，planner 在该 house 多次成功
DEFAULT_SCENES="103:1 97:3 95:0 95:1"
read -r -a SCENE_LIST <<< "${SCENES:-$DEFAULT_SCENES}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi

echo "Benchmark : $RBY_BENCH"
echo "Output    : $OUT"
echo "GPU       : $CUDA_DEVICE"
echo "Scenes    : ${SCENE_LIST[*]}"

for scene in "${SCENE_LIST[@]}"; do
  house="${scene%%:*}"
  idx="${scene##*:}"
  echo "=== house $house / episode $idx ==="
  "$PYTHON_BIN" -m molmo_spaces.evaluation.eval_main \
    molmo_spaces.evaluation.configs.evaluation_configs:RBY1LLMWaypointPickPnPEvalConfig \
    --benchmark_dir "$RBY_BENCH" \
    --house_index "$house" \
    --idx "$idx" \
    --num_workers 1 \
    --no_wandb \
    --task_horizon_steps "$TASK_HORIZON_STEPS" \
    --output_dir "$OUT" \
    --env_file "${LLM_ENV_FILE:-$MOLMOSPACES_ROOT/../.env}"
done

echo "All scenes finished. Output root: $OUT"
