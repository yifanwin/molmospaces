#!/usr/bin/env bash
# P1 专用入口：真实 A* 导航，不加载 API 凭据，不运行操作策略。
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$REPO/../.." && pwd)}"
export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$WORKSPACE_ROOT/molmospaces_data/assets}"
export MLSPACES_CACHE_DIR="${MLSPACES_CACHE_DIR:-/nas/wenyifan/molmospaces_data/cache}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/last-mile-p1-matplotlib}"
export MLSPACES_DISABLE_CUROBO=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_ROOT/molmospaces/.venv/bin/python}"
P0_ROOT="${P0_ROOT:-$REPO/eval_output/last_mile/p0_20260921}"
OUTPUT="${OUTPUT:-$REPO/eval_output/last_mile/p1_20260921/E01}"
REPORT="${REPORT:-$REPO/docs/last_mile_p1_e01_20260921.md}"
FIGURE_PREFIX="${FIGURE_PREFIX:-$REPO/docs/figures/last_mile_p1_e01_navigation}"
MODE="${1:-all}"
shift || true
case "$MODE" in
  test) "$PYTHON_BIN" -m pytest "$REPO/mlspaces_tests/evaluation/test_last_mile_p0.py" "$REPO/mlspaces_tests/evaluation/test_last_mile_p1.py" -q "$@" ;;
  run) "$PYTHON_BIN" -m molmo_spaces.evaluation.last_mile.run_p1 --p0-root "$P0_ROOT" --output "$OUTPUT" "$@" ;;
  report) "$PYTHON_BIN" "$REPO/scripts/evaluation/plot_last_mile_p1.py" --output "$OUTPUT" --report "$REPORT" --figure-prefix "$FIGURE_PREFIX" "$@" ;;
  all) bash "$0" test; bash "$0" run "$@"; bash "$0" report ;;
  *) echo '用法：run_last_mile_p1.sh {test|run|report|all}' >&2; exit 2 ;;
esac
