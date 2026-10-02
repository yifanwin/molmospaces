#!/usr/bin/env python
"""按本地清单为 scene 包补建符号链接（不联网、不碰 NAS）。

为什么需要它：molmospaces 自带的 `_ensure_package_linked` 每个文件都要
`Path.resolve()` + `dst.exists()`，而链接目标是 CIFS 挂载（/nas），
实测一次 NAS stat 约 250 ms（不存在的文件更慢）。procthor-10k-train 有 1 万个包、
约 30 万个文件，按官方路径建链要几天。

本脚本的叶子清单直接来自 assets 目录里已有的 `mjthor_resources_combined_meta.json.gz`
（该文件是包内容的权威清单），建链过程是纯本地操作（os.symlink），
因此 30 万个链接只需几分钟。

前提：NAS cache 上对应的包已解压完成（`.{package}_complete_extract` 存在）。

用法：
    source setup_env.sh
    python scripts/assets/fast_relink_scene_split.py --source procthor-10k-train --dry-run
    python scripts/assets/fast_relink_scene_split.py --source procthor-10k-train
"""

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

COMBINED_TRIES_NAME = "mjthor_resources_combined_meta.json.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="如 procthor-10k-train")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only-existing",
        action="store_true",
        default=True,
        help="只给 NAS 上确实存在（有 extract flag）的包建链，默认开启",
    )
    return parser.parse_args()


def iter_leaves(node: dict, prefix: str = ""):
    """展开 meta 里的嵌套结构，产出相对路径。"""
    for name, child in node.items():
        rel = f"{prefix}/{name}" if prefix else name
        if isinstance(child, dict) and child:
            yield from iter_leaves(child, rel)
        else:
            yield rel


def main() -> int:
    args = parse_args()

    from molmo_spaces.molmo_spaces_constants import (
        ASSETS_DIR,
        DATA_CACHE_DIR,
        DATA_TYPE_TO_SOURCE_TO_VERSION,
    )

    source = args.source
    version = DATA_TYPE_TO_SOURCE_TO_VERSION["scenes"][source]
    prefix = source.rsplit("-", 1)[-1]

    link_dir = ASSETS_DIR / "scenes" / source
    cache_dir = DATA_CACHE_DIR / "scenes" / source / version
    meta_path = link_dir / COMBINED_TRIES_NAME

    if not meta_path.exists():
        # 有时只有 NAS cache 里有清单
        meta_path = cache_dir / COMBINED_TRIES_NAME
    print(f"链接树   = {link_dir}")
    print(f"NAP cache = {cache_dir}")
    print(f"包清单   = {meta_path}", flush=True)

    with gzip.open(meta_path, "rt") as f:
        archive_to_paths = json.load(f)

    # 只处理该 split 的 house 包（排除 housegen_build_settings 之类的杂项）
    head = f"{source}_{prefix}_"
    pkgs = {}
    for name, paths in archive_to_paths.items():
        if not name.startswith(head):
            continue
        tail = name[len(head) :].split(".tar")[0]
        if tail.isdigit():
            pkgs[name] = paths
    print(f"远端清单共 {len(pkgs)} 个 house 包", flush=True)

    local_names = set(os.listdir(link_dir))
    # 一次 listdir 拿到 NAS 上已解压完的包（逐包 stat 在 CIFS 上要 0.5~2s/次）
    import time

    t = time.time()
    cache_names = set(os.listdir(cache_dir))
    print(f"扫描 NAS cache：{time.time() - t:.1f}s，{len(cache_names)} 条目", flush=True)
    extracted = {
        n[1 : -len("_complete_extract")] for n in cache_names if n.endswith("_complete_extract")
    }

    done = 0
    skipped_no_extract = 0
    created = 0
    t0 = None

    for pkg, paths in pkgs.items():
        # 逐叶子检查本地是否已链接（lexists 是纯本地操作，很快）
        leaves = list(iter_leaves(paths))
        todo = [rel for rel in leaves if not os.path.lexists(link_dir / rel)]
        if not todo:
            done += 1
            continue

        # 该包在 NAS 上必须已解压完，否则先跳过
        if pkg not in extracted:
            skipped_no_extract += 1
            continue

        if args.dry_run:
            created += len(todo)
            continue

        if t0 is None:
            t0 = time.time()
        for rel in todo:
            dst = link_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(cache_dir / rel, dst)
            created += 1
        (link_dir / f".{pkg}_complete_links").touch(exist_ok=True)

    if args.dry_run:
        print(f"已完整 {done} 个包；待补 {len(pkgs) - done - skipped_no_extract} 个包，"
              f"约 {created} 条链接；NAS 未解压 {skipped_no_extract} 个")
        return 0

    dt = time.time() - t0 if t0 else 0
    print(f"建链完成：{created} 条链接，用时 {dt:.1f}s")
    print(f"（此前已完成 {done} 个包；本地尚无 house 主文件而跳过 {skipped_no_extract} 个）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
