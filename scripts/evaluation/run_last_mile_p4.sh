#!/usr/bin/env bash
# P4：不读取 LLM 密钥、不启动 CuRobo；默认在冻结试点上运行。
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$REPO/../.." && pwd)}"
export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$WORKSPACE_ROOT/molmospaces_data/assets}"
export MLSPACES_CACHE_DIR="${MLSPACES_CACHE_DIR:-/tmp/last-mile-p4-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/last-mile-p4-matplotlib}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export MLSPACES_DISABLE_CUROBO=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-5}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
unset OPENAI_API_KEY LLM_API_KEY LLM_BASE_URL LLM_MODEL || true
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_ROOT/molmospaces/.venv/bin/python}"
SUBSET="${SUBSET:-pilot}"
P0_ROOT="${P0_ROOT:-$REPO/eval_output/last_mile/p0_20260921}"
if [[ "$SUBSET" == formal ]]; then
  P0_VALIDATION="${P0_VALIDATION:-$P0_ROOT/validation_formal_minimal}"
  P1_ROOT="${P1_ROOT:-$REPO/eval_output/last_mile/p1_20260922/formal_E01}"
  P2_ROOT="${P2_ROOT:-$REPO/eval_output/last_mile/p2_20260922/formal_E01}"
  P3_ROOT="${P3_ROOT:-$REPO/eval_output/last_mile/p3_20260923/formal_E01}"
  OUTPUT="${OUTPUT:-$REPO/eval_output/last_mile/p4_20260929/formal_E01}"
  REPORT_DATE="${REPORT_DATE:-20260929}"
else
  P0_VALIDATION="${P0_VALIDATION:-$P0_ROOT/validation_minimal}"
  P1_ROOT="${P1_ROOT:-$REPO/eval_output/last_mile/p1_20260921/E04}"
  P2_ROOT="${P2_ROOT:-$REPO/eval_output/last_mile/p2_20260922/E02}"
  P3_ROOT="${P3_ROOT:-$REPO/eval_output/last_mile/p3_20260922/E01}"
  OUTPUT="${OUTPUT:-$REPO/eval_output/last_mile/p4_20260923/E01}"
  REPORT_DATE="${REPORT_DATE:-20260923}"
fi
MODE="${1:-all}"
shift || true
case "$MODE" in
  test) "$PYTHON_BIN" -m pytest "$REPO/mlspaces_tests/evaluation/test_last_mile_p4.py" -q "$@" ;;
  run) "$PYTHON_BIN" -m molmo_spaces.evaluation.last_mile.run_p4 \
    --p0-root "$P0_ROOT" --p0-validation "$P0_VALIDATION" \
    --p1-root "$P1_ROOT" --p2-root "$P2_ROOT" --p3-root "$P3_ROOT" \
    --output "$OUTPUT" --subset "$SUBSET" "$@" ;;
  report) "$PYTHON_BIN" "$REPO/scripts/evaluation/plot_last_mile_p4.py" --input "$OUTPUT" --repo "$REPO" --date "$REPORT_DATE" "$@" ;;
  all) bash "$0" test; bash "$0" run; bash "$0" report ;;
  *) echo '用法：run_last_mile_p4.sh {test|run|report|all}' >&2; exit 2 ;;
esac
