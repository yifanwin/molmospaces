#!/usr/bin/env bash

MOLMOSPACES_ROOT="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &&
  pwd
)"
MOMATRAJGEN_ROOT="$(dirname "$MOLMOSPACES_ROOT")"

# cache 已迁移到 NAS（CIFS 挂载，只存真实文件）；
# assets 必须留在本地：内含指向 cache 的软链接和依赖 mmap 的 .lmdb。
export MLSPACES_CACHE_DIR="/nas/wenyifan/molmospaces_data/cache"
export MLSPACES_ASSETS_DIR="$MOMATRAJGEN_ROOT/molmospaces_data/assets"

# 已使用 editable install 时通常不需要 PYTHONPATH。
# 如果确实需要：
export PYTHONPATH="$MOLMOSPACES_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# 处理内存增长
export MALLOC_ARENA_MAX=2              # 限制 arena 数量（最关键，几乎无性能代价）
export MALLOC_TRIM_THRESHOLD_=131072   # 空闲超过 128 KB 就归还 OS
export MALLOC_MMAP_THRESHOLD_=131072   # 大块走 mmap，free 时直接归还

export MUJOCO_EGL_DEVICE_ID=3
export CUDA_VISIBLE_DEVICES=3
