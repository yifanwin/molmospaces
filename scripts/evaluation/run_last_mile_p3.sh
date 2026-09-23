#!/usr/bin/env bash
# P3 局部扫描：禁止 API/CuRobo，默认复用冻结 pilot、P1-E04 和 P2-E02。
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$REPO/../.." && pwd)}"
export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$WORKSPACE_ROOT/molmospaces_data/assets}"
export MLSPACES_CACHE_DIR="${MLSPACES_CACHE_DIR:-/tmp/last-mile-p3-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/last-mile-p3-matplotlib}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export MLSPACES_DISABLE_CUROBO=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
unset OPENAI_API_KEY LLM_API_KEY LLM_BASE_URL LLM_MODEL || true
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_ROOT/molmospaces/.venv/bin/python}"
P0_ROOT="${P0_ROOT:-$REPO/eval_output/last_mile/p0_20260921}"
P0_VALIDATION="${P0_VALIDATION:-$P0_ROOT/validation_minimal}"
P1_ROOT="${P1_ROOT:-$REPO/eval_output/last_mile/p1_20260922/E05_compat}"
P1_REFERENCE="${P1_REFERENCE:-$REPO/eval_output/last_mile/p1_20260921/E04}"
P2_ROOT="${P2_ROOT:-$REPO/eval_output/last_mile/p2_20260922/E02}"
OUTPUT="${OUTPUT:-$REPO/eval_output/last_mile/p3_20260922/E01}"
MODE="${1:-all}"
shift || true
COMMON=(--p0-root "$P0_ROOT" --p0-validation "$P0_VALIDATION" --p1-root "$P1_ROOT" --p1-reference "$P1_REFERENCE" --p2-root "$P2_ROOT" --output "$OUTPUT")
case "$MODE" in
  test) "$PYTHON_BIN" -m pytest "$REPO/mlspaces_tests/evaluation/test_last_mile_p0.py" "$REPO/mlspaces_tests/evaluation/test_last_mile_p1.py" "$REPO/mlspaces_tests/evaluation/test_last_mile_p2.py" "$REPO/mlspaces_tests/evaluation/test_last_mile_p3.py" -q "$@" ;;
  run) "$PYTHON_BIN" -m molmo_spaces.evaluation.last_mile.run_p3 "${COMMON[@]}" "$@" ;;
  report) "$PYTHON_BIN" "$REPO/scripts/evaluation/plot_last_mile_p3.py" --input "$OUTPUT" --repo "$REPO" "$@" ;;
  all) bash "$0" test; bash "$0" run --workers "${P3_WORKERS:-20}" --shards-per-episode "${P3_SHARDS_PER_EPISODE:-8}"; bash "$0" report ;;
  *) echo '用法：run_last_mile_p3.sh {test|run|report|all}' >&2; exit 2 ;;
esac
