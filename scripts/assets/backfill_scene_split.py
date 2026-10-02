#!/usr/bin/env python
"""补齐某个 scene split 的全部 archive（下载到 NAS cache + 在 assets 建链接）。

背景：molmospaces 的 scene 是**按需下载**的（install_scene_from_path），
所以 assets/scenes/<split> 里通常只有历史上用到过的那批 house。
本脚本补齐远端存在、本地还缺的 archive。

用法：
    source setup_env.sh
    python scripts/assets/backfill_scene_split.py --source procthor-10k-val --dry-run
    python scripts/assets/backfill_scene_split.py --source procthor-10k-val --repair-broken

数据落点（由 setup_env.sh 决定）：
    真实文件 -> $MLSPACES_CACHE_DIR/scenes/<source>/<version>/   （NAS）
    符号链接 -> $MLSPACES_ASSETS_DIR/scenes/<source>/            （本地）

两个性能要点：
1. 不要调 get_resource_manager()——它 setup 全部数据源，在 NAS 上 stat 几十万文件会卡死。
   这里只构造含单个 source 的 ResourceManager。
2. 不要用 os.path.exists 遍历链接——目标在 NAS（CIFS），每次 stat 要几十毫秒。
   改为一次 listdir 拿到 NAS 上的文件名集合 + os.readlink 读本地链接目标。
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 远端压缩包清单文件名（与 molmospaces_resources.constants 一致）
REMOTE_MANIFEST_NAME = "mjthor_resource_file_to_size_mb.json"

# 判断一个 house 在链接树里是否"可用"的顶层文件
HOUSE_SUFFIXES = [".xml", ".json", "_ceiling.xml", "_map.png", "_metadata.json"]

# 下载量安全上限（MB），超过就中止，防止误下整个数据集
MAX_DOWNLOAD_MB_DEFAULT = 500.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="如 procthor-10k-val")
    parser.add_argument("--dry-run", action="store_true", help="只报告，不下载不改动")
    parser.add_argument(
        "--repair-broken",
        action="store_true",
        help="同时修复目标已失效的链接（如 val 的 891~999 段）",
    )
    parser.add_argument("--max-download-mb", type=float, default=MAX_DOWNLOAD_MB_DEFAULT)
    parser.add_argument(
        "--skip-linking",
        action="store_true",
        default=True,
        help="只下载解压到 NAS，不逐文件建链接（改用 fast_relink_scene_split.py，快 100 倍）",
    )
    return parser.parse_args()


def scan_link_states(link_dir: Path, cache_dir: Path, prefix: str, all_ids: set[int]):
    """扫描链接树，返回 (可用, 断链, 从未下载)。

    判定依据：本地的链接名集合 + NAS cache 目录里的文件名集合（各一次 listdir）
    + os.readlink（纯本地读）。全程不 stat NAS。
    """
    local_names = set(os.listdir(link_dir))
    cache_names = set(os.listdir(cache_dir))

    ok_ids, broken_ids = set(), set()
    for i in all_ids:
        if f"{prefix}_{i}.xml" not in local_names:
            continue
        try:
            target = os.readlink(link_dir / f"{prefix}_{i}.xml")
        except OSError:
            broken_ids.add(i)
            continue
        # 目标必须以当前 cache 目录为前缀，且文件确实在 cache 里
        target_name = os.path.basename(target)
        if str(target).startswith(str(cache_dir)) and target_name in cache_names:
            ok_ids.add(i)
        else:
            broken_ids.add(i)

    never_ids = all_ids - ok_ids - broken_ids
    return ok_ids, broken_ids, never_ids


def main() -> int:
    args = parse_args()

    from molmo_spaces.molmo_spaces_constants import (
        ASSETS_DIR,
        DATA_CACHE_DIR,
        DATA_TYPE_TO_SOURCE_TO_VERSION,
    )
    from molmospaces_resources import ResourceManager, R2RemoteStorage

    source = args.source
    version = DATA_TYPE_TO_SOURCE_TO_VERSION["scenes"][source]
    prefix = source.rsplit("-", 1)[-1]  # procthor-10k-val -> val

    link_dir = ASSETS_DIR / "scenes" / source
    cache_dir = DATA_CACHE_DIR / "scenes" / source / version

    print(f"source  = {source}   version = {version}", flush=True)
    print(f"链接树  = {link_dir}", flush=True)
    print(f"真实文件 = {cache_dir}", flush=True)

    if not (cache_dir / REMOTE_MANIFEST_NAME).exists():
        print(f"缓存里没有远端清单：{cache_dir / REMOTE_MANIFEST_NAME}")
        print("请先跑一次任意 molmospaces 流程让该 split 完成 setup。")
        return 1

    with open(cache_dir / REMOTE_MANIFEST_NAME) as f:
        remote_manifest = json.load(f)

    # 远端该 split 的全部 house 编号
    head = f"{source}_{prefix}_"
    all_ids = set()
    for name in remote_manifest:
        if not name.startswith(head):
            continue
        tail = name[len(head) :].split(".tar")[0]
        if tail.isdigit():
            all_ids.add(int(tail))
    print(f"远端清单：{len(all_ids)} 个 house", flush=True)

    ok_ids, broken_ids, never_ids = scan_link_states(link_dir, cache_dir, prefix, all_ids)

    def fmt(ids: set[int]) -> str:
        s = sorted(ids)
        return f"{s[:8]}{' ...' if len(s) > 8 else ''}"

    print(f"  可用     {len(ok_ids):5d}", flush=True)
    print(f"  断链     {len(broken_ids):5d}  {fmt(broken_ids)}", flush=True)
    print(f"  从未下载 {len(never_ids):5d}  {fmt(never_ids)}", flush=True)

    todo = never_ids | (broken_ids if args.repair_broken else set())
    pkgs = [f"{source}_{prefix}_{i}.tar.zst" for i in sorted(todo)]
    download_mb = sum(remote_manifest.get(p, 0.0) for p in pkgs)
    print(f"待处理 {len(pkgs)} 个 archive，需从远端取 {download_mb:.1f} MB", flush=True)
    if broken_ids and not args.repair_broken:
        print(f"（{len(broken_ids)} 个断链项已跳过，加 --repair-broken 可一并修复）")

    if not pkgs:
        print("无需补齐。")
        return 0
    if download_mb > args.max_download_mb:
        print(f"下载量超过上限 {args.max_download_mb} MB，已中止。")
        return 1
    if args.dry_run:
        print("\n--dry-run，未做任何改动。")
        return 0

    manager = ResourceManager(
        remote_storage=R2RemoteStorage("mujoco-thor-resources"),
        data_type_to_source_to_version={"scenes": {source: version}},
        symlink_dir=ASSETS_DIR,
        cache_dir=DATA_CACHE_DIR,
        force_install=False,
        cache_lock=True,
        symlink_lock=True,
    )

    # 修复失效链接：删掉链接和对应的 link flag，让 installer 重建
    if args.repair_broken and broken_ids:
        removed = 0
        for i in sorted(broken_ids):
            for s in HOUSE_SUFFIXES:
                p = link_dir / f"{prefix}_{i}{s}"
                if p.is_symlink() and not p.exists():
                    p.unlink()
                    removed += 1
            assets_sub = link_dir / f"{prefix}_{i}_assets"
            if assets_sub.is_dir():
                for child in assets_sub.iterdir():
                    if child.is_symlink() and not child.exists():
                        child.unlink()
                        removed += 1
            flag = link_dir / f".{source}_{prefix}_{i}.tar.zst_complete_links"
            if flag.exists():
                flag.unlink()
        print(f"清理失效链接 {removed} 条", flush=True)

    # 官方 installer 的建链路径每个文件都要 stat NAS（实测 ~300ms/文件），
    # 万级 house 会跑成几小时。默认只下载解压，建链交给 fast_relink_scene_split.py。
    print(f"开始安装 {len(pkgs)} 个 archive ...", flush=True)
    manager.install_packages(
        "scenes", {source: pkgs}, skip_linking=args.skip_linking
    )
    if not args.skip_linking:
        print("（提示：逐文件建链很慢，建议改用 fast_relink_scene_split.py）")

    ok_after, broken_after, never_after = scan_link_states(link_dir, cache_dir, prefix, all_ids)
    print(f"\n完成：{source} 可用 {len(ok_after)} / {len(all_ids)}"
          f"（断链 {len(broken_after)}，未下载 {len(never_after)}）")
    if broken_after or never_after:
        print(f"  仍缺: {sorted(broken_after | never_after)[:20]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
