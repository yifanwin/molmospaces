#!/usr/bin/env bash

# 本地 Ubuntu 桌面版。必须使用 `source setup_env.sh`，否则无法修改当前 shell 的环境。
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf '请使用 source 加载此脚本：source setup_env.sh\n' >&2
  exit 1
fi

# 退出所有嵌套的 Conda 环境，避免 Conda 的 Python/动态库干扰本地环境。
if [[ "${CONDA_SHLVL:-0}" =~ ^[1-9][0-9]*$ ]]; then
  if type conda >/dev/null 2>&1; then
    while [[ "${CONDA_SHLVL:-0}" =~ ^[1-9][0-9]*$ ]]; do
      previous_conda_shlvl="$CONDA_SHLVL"
      conda deactivate || break
      [[ "${CONDA_SHLVL:-0}" == "$previous_conda_shlvl" ]] && break
    done
    unset previous_conda_shlvl
  else
    printf '警告：检测到 Conda 环境，但当前 shell 中找不到 conda 函数，请先手动执行 conda deactivate。\n' >&2
  fi
fi

MOLMOSPACES_ROOT="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &&
  pwd
)"
MOMATRAJGEN_ROOT="$(dirname "$MOLMOSPACES_ROOT")"

export MLSPACES_CACHE_DIR="$MOMATRAJGEN_ROOT/molmospaces_data/cache"
export MLSPACES_ASSETS_DIR="$MOMATRAJGEN_ROOT/molmospaces_data/assets"

# 已使用 editable install 时通常不需要 PYTHONPATH。
# 如果确实需要：
export PYTHONPATH="$MOLMOSPACES_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# 本地有显示器时不强制 EGL，让 MuJoCo/GLFW 使用桌面显示。
# 同时清理可能由服务器脚本留下的 EGL 设置。
unset MUJOCO_GL
unset PYOPENGL_PLATFORM
unset MUJOCO_EGL_DEVICE_ID
