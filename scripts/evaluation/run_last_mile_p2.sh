#!/usr/bin/env bash
# P2 独立 evaluator：禁止 API 和 CuRobo，复用 P1-E03 的有效 A。
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$REPO/../.." && pwd)}"
export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$WORKSPACE_ROOT/molmospaces_data/assets}"
export MLSPACES_CACHE_DIR="${MLSPACES_CACHE_DIR:-/tmp/last-mile-p2-cache}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export MLSPACES_DISABLE_CUROBO=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
unset OPENAI_API_KEY LLM_API_KEY LLM_BASE_URL LLM_MODEL || true
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_ROOT/molmospaces/.venv/bin/python}"
P0_ROOT="${P0_ROOT:-$REPO/eval_output/last_mile/p0_20260921}"
P1_ROOT="${P1_ROOT:-$REPO/eval_output/last_mile/p1_20260921/E03}"
OUTPUT="${OUTPUT:-$REPO/eval_output/last_mile/p2_20260922/E01}"
MODE="${1:-all}"
shift || true
case "$MODE" in
  test) "$PYTHON_BIN" -m pytest "$REPO/mlspaces_tests/evaluation/test_last_mile_p0.py" "$REPO/mlspaces_tests/evaluation/test_last_mile_p1.py" "$REPO/mlspaces_tests/evaluation/test_last_mile_p2.py" -q "$@" ;;
  run) "$PYTHON_BIN" -m molmo_spaces.evaluation.last_mile.run_p2 --p0-root "$P0_ROOT" --p1-root "$P1_ROOT" --output "$OUTPUT" "$@" ;;
  all) bash "$0" test; bash "$0" run "$@" ;;
  *) echo '用法：run_last_mile_p2.sh {test|run|all}' >&2; exit 2 ;;
esac
