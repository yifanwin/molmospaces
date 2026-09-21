#!/usr/bin/env bash
set -euo pipefail

# Run from any directory. This script never prints API credentials.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MOLMOSPACES_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
source "$MOLMOSPACES_ROOT/setup_env.sh"
cd "$MOLMOSPACES_ROOT"

RBY_BENCH="${RBY_BENCH:-$MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v1/procthor-10k/RBY1PickAndPlaceDataGenConfig/RBY1PickAndPlaceDataGenConfig_20260209_json_benchmark}"
PANDA_BENCH="${PANDA_BENCH:-$MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickandPlaceDroidMiniBench/FrankaPickandPlaceDroidMiniBench_20260111_json_benchmark}"
OUT="${OUT:-$MOLMOSPACES_ROOT/eval_output}"
MODE="${1:-gate}"
PYTHON_BIN="${PYTHON_BIN:-$MOLMOSPACES_ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi

run_eval() {
  "$PYTHON_BIN" -m molmo_spaces.evaluation.eval_main "$@" \
    --house_index 4 --num_workers 1 --output_dir "$OUT" --no_wandb \
    --task_horizon_steps 600 --env_file "${LLM_ENV_FILE:-$MOLMOSPACES_ROOT/../.env}"
}

case "$MODE" in
  baseline)
    run_eval --benchmark_dir "$RBY_BENCH" \
      molmo_spaces.evaluation.configs.evaluation_configs:RBY1CuroboPickPnPEvalConfig
    run_eval --benchmark_dir "$PANDA_BENCH" \
      molmo_spaces.evaluation.configs.evaluation_configs:PandaOmronCuroboPickPnPEvalConfig
    ;;
  gate)
    run_eval --benchmark_dir "$RBY_BENCH" --idx 0 \
      molmo_spaces.evaluation.configs.evaluation_configs:RBY1LLMWaypointPickPnPEvalConfig
    run_eval --benchmark_dir "$PANDA_BENCH" --idx 0 \
      molmo_spaces.evaluation.configs.evaluation_configs:PandaOmronLLMWaypointPickPnPEvalConfig
    ;;
  full)
    run_eval --benchmark_dir "$RBY_BENCH" \
      molmo_spaces.evaluation.configs.evaluation_configs:RBY1LLMWaypointPickPnPEvalConfig
    run_eval --benchmark_dir "$PANDA_BENCH" \
      molmo_spaces.evaluation.configs.evaluation_configs:PandaOmronLLMWaypointPickPnPEvalConfig
    ;;
  mock)
    : "${LLM_MOCK_RESPONSE_FILE:?Set LLM_MOCK_RESPONSE_FILE to a response JSON file}"
    run_eval --benchmark_dir "$RBY_BENCH" --idx 0 \
      molmo_spaces.evaluation.configs.evaluation_configs:RBY1LLMWaypointPickPnPEvalConfig
    run_eval --benchmark_dir "$PANDA_BENCH" --idx 0 \
      molmo_spaces.evaluation.configs.evaluation_configs:PandaOmronLLMWaypointPickPnPEvalConfig
    ;;
  *)
    echo "Usage: $0 {baseline|mock|gate|full}" >&2
    exit 2
    ;;
esac
