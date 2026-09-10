#!/usr/bin/env bash
set -uo pipefail

# ==================== 可修改配置 ====================
DATASET="holodeck-objaverse"
SPLIT="train"
HOUSES=""
START=0
END=100000
MAX_WORKERS=4
BASE_IDENTIFIER="holodeck_subset"
OUTPUT_ROOT="test_outputs/physics_tests"
SHOW_PROGRESS=true
# ====================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
IDENTIFIER="${BASE_IDENTIFIER}_${START}_${END}"
RUN_OUTPUT="$ROOT_DIR/$OUTPUT_ROOT/$IDENTIFIER"

cd "$ROOT_DIR"
source "$ROOT_DIR/setup_env.sh"

if [[ -z "$HOUSES" ]]; then
  HOUSES="$MLSPACES_ASSETS_DIR/scenes/${DATASET}-${SPLIT}"
fi

if (( START < 0 || END <= START )); then
  printf '场景范围无效：要求 0 <= START < END，当前为 [%d, %d)\n' \
    "$START" "$END" >&2
  exit 1
fi

if (( MAX_WORKERS < 1 )); then
  printf 'MAX_WORKERS 必须大于等于 1，当前为 %d\n' "$MAX_WORKERS" >&2
  exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
  printf '找不到 Python 解释器：%s\n' "$PYTHON" >&2
  exit 1
fi

if [[ ! -d "$HOUSES" ]]; then
  printf '找不到场景目录：%s\n' "$HOUSES" >&2
  exit 1
fi

if [[ -e "$RUN_OUTPUT" ]]; then
  printf '输出目录已存在，请修改 BASE_IDENTIFIER、START 或 END：%s\n' "$RUN_OUTPUT" >&2
  exit 1
fi

mkdir -p "$RUN_OUTPUT"

collect_artifacts() {
  local test_name="$1"
  local output_dir="$RUN_OUTPUT/$test_name"
  shift

  mkdir -p "$output_dir"
  for file in "$@"; do
    if [[ -f "$ROOT_DIR/$file" ]]; then
      mv "$ROOT_DIR/$file" "$output_dir/"
    fi
  done
}

run_test() {
  local test_name="$1"
  local script_path="$2"
  local log_file="$RUN_OUTPUT/$test_name/run.log"
  local status
  shift 2

  mkdir -p "$RUN_OUTPUT/$test_name"
  printf '开始测试：%s\n' "$test_name"

  local command=(
    "$PYTHON" "$script_path"
    --dataset "$DATASET"
    --split "$SPLIT"
    --houses-folder "$HOUSES"
    --start "$START"
    --end "$END"
    --max-workers "$MAX_WORKERS"
    --identifier "$IDENTIFIER"
  )

  if [[ "$SHOW_PROGRESS" == true ]]; then
    set +o pipefail
    "${command[@]}" 2>&1 | tee "$log_file"
    status=${PIPESTATUS[0]}
    set -o pipefail
  else
    "${command[@]}" >"$log_file" 2>&1
    status=$?
  fi

  printf '%s\n' "$status" >"$RUN_OUTPUT/$test_name/exit_code.txt"
  collect_artifacts "$test_name" "$@"

  if (( status == 0 )); then
    printf '完成测试：%s\n' "$test_name"
  else
    printf '测试失败：%s（退出码 %d，日志：%s）\n' \
      "$test_name" "$status" "$log_file" >&2
  fi

  return "$status"
}

failed_tests=()

run_test penetration mlspaces_tests/scenes/test_penetration_objects.py \
  "penetration_test_results_${DATASET}_${SPLIT}_${IDENTIFIER}.json" \
  "penetration_test_warnings_${DATASET}_${SPLIT}_${IDENTIFIER}.json" \
  "penetration_test_errors_${DATASET}_${SPLIT}_${IDENTIFIER}.json" \
  "stats_fails_penetration_test_${DATASET}_${SPLIT}_${IDENTIFIER}.txt" \
  || failed_tests+=(penetration)

run_test stability mlspaces_tests/scenes/test_stability_mp.py \
  "stability_test_results_${DATASET}_${SPLIT}_${IDENTIFIER}.json" \
  "stability_test_warnings_${DATASET}_${SPLIT}_${IDENTIFIER}.json" \
  "stability_test_errors_${DATASET}_${SPLIT}_${IDENTIFIER}.json" \
  "stats_fails_stability_test_${DATASET}_${SPLIT}_${IDENTIFIER}.txt" \
  || failed_tests+=(stability)

run_test lift_force mlspaces_tests/scenes/test_lift_force_mp.py \
  "history_lift_force_test_${DATASET}_${SPLIT}_${IDENTIFIER}.json" \
  "lift_force_test_mujoco_warnings_${DATASET}_${SPLIT}_${IDENTIFIER}.json" \
  "lift_force_test_could_not_process_${DATASET}_${SPLIT}_${IDENTIFIER}.txt" \
  "stats_fails_lift_test_${DATASET}_${SPLIT}_${IDENTIFIER}.txt" \
  || failed_tests+=(lift_force)

run_test articulation_force mlspaces_tests/scenes/test_articulation_force_mp.py \
  "history_articulation_force_test_${DATASET}_${SPLIT}_${IDENTIFIER}.json" \
  "articulation_force_test_mujoco_warnings_${DATASET}_${SPLIT}_${IDENTIFIER}.json" \
  "articulation_force_test_could_not_process_${DATASET}_${SPLIT}_${IDENTIFIER}.txt" \
  "stats_fails_articulation_test_${DATASET}_${SPLIT}_${IDENTIFIER}.txt" \
  "info_bodies_within_sites_${DATASET}_${SPLIT}_${IDENTIFIER}.json" \
  || failed_tests+=(articulation_force)

printf '数据集：%s\n分片：%s\n范围：[%d, %d)\n并发数：%d\n标识符：%s\n' \
  "$DATASET" "$SPLIT" "$START" "$END" "$MAX_WORKERS" "$IDENTIFIER" \
  >"$RUN_OUTPUT/run_info.txt"

if (( ${#failed_tests[@]} > 0 )); then
  printf '以下测试失败：%s\n' "${failed_tests[*]}" >&2
  printf '所有输出已整理到：%s\n' "$RUN_OUTPUT"
  exit 1
fi

printf '全部测试完成，输出目录：%s\n' "$RUN_OUTPUT"
