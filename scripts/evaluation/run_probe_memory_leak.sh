#!/usr/bin/env bash
set -euo pipefail

# 逐 episode 内存泄漏探针启动器（只读测量，不修改被测代码）。
#
#   bash scripts/evaluation/run_probe_memory_leak.sh                     # 默认 house 82、9 个 episode
#   bash scripts/evaluation/run_probe_memory_leak.sh --self-check         # 只验证普查机制，不跑评测
#   bash scripts/evaluation/run_probe_memory_leak.sh --house_index 4 --max_episodes 3
#
# 说明：探针需要独占一个 GPU 做 EGL 渲染 + CuRobo 规划，默认沿用 setup_env.sh 里的
# CUDA_VISIBLE_DEVICES（本仓库为 3）。覆盖率只有 house 级：先按 house 过滤再截断 episode 数。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MOLMOSPACES_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
source "$MOLMOSPACES_ROOT/setup_env.sh"
cd "$MOLMOSPACES_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$MOLMOSPACES_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi

echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<未设置>} (setup_env.sh 默认 3)"
echo "普查日志: /tmp/probe_memleak/census.log"

exec "$PYTHON_BIN" scripts/evaluation/probe_memory_leak.py "$@"
