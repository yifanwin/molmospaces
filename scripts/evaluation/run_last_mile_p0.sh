#!/usr/bin/env bash
# P0 专用入口：不加载 .env 凭据，不初始化操作策略。
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$REPO/../.." && pwd)}"
export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$WORKSPACE_ROOT/molmospaces_data/assets}"
export MLSPACES_CACHE_DIR="${MLSPACES_CACHE_DIR:-/nas/wenyifan/molmospaces_data/cache}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/last-mile-p0-matplotlib}"
export MLSPACES_DISABLE_CUROBO=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_ROOT/molmospaces/.venv/bin/python}"
OUTPUT="${OUTPUT:-$REPO/eval_output/last_mile/p0_20260921}"
SOURCE="${SOURCE:-$MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v1/procthor-10k/RBY1PickAndPlaceDataGenConfig/RBY1PickAndPlaceDataGenConfig_20260209_json_benchmark/benchmark.json}"
MODE="${1:-all}"
shift || true
case "$MODE" in
  build) "$PYTHON_BIN" "$REPO/molmo_spaces/evaluation/last_mile/build.py" --source "$SOURCE" --assets "$MLSPACES_ASSETS_DIR" --output "$OUTPUT" "$@" ;;
  validate) "$PYTHON_BIN" -m molmo_spaces.evaluation.last_mile.validate --output "$OUTPUT" "$@" ;;
  test) "$PYTHON_BIN" -m pytest "$REPO/mlspaces_tests/evaluation/test_last_mile_p0.py" -q "$@" ;;
  all) bash "$0" build; bash "$0" test; bash "$0" validate "$@" ;;
  *) echo '用法：run_last_mile_p0.sh {build|validate|test|all}' >&2; exit 2 ;;
esac
