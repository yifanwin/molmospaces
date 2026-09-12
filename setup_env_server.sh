#!/usr/bin/env bash

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

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
