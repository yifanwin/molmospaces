#!/usr/bin/env python
"""并行把 scene split 的 archive 下载并解压到 NAS cache（不建链接）。

为什么不用官方的 install_packages：它有两个慢点
1. "Checking" 阶段是串行的，每检查一个包的 extract flag 都要 stat NAS，
   实测不存在文件的 negative lookup 要 0.5~2s，1 万个包光检查就要几小时；
   本脚本改成一次 listdir 拿到全部 flag。
2. 它的链接阶段每个文件都要 resolve+exists（NAS stat），改用
   fast_relink_scene_split.py 纯本地建链。

所以完整流程是：
    fast_backfill_scene_split.py --source procthor-10k-train   # 下载+解压到 NAS
    fast_relink_scene_split.py    --source procthor-10k-train   # 本地建链接

用法：
    source setup_env.sh
    python scripts/assets/fast_backfill_scene_split.py --source procthor-10k-train --dry-run
    python scripts/assets/fast_backfill_scene_split.py --source procthor-10k-train
"""

import argparse
import io
import json
import os
import stat
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import zstandard as zstd

# R2 公开桶（与 molmospaces_resources.remote_storage.R2RemoteStorage.KNOWN_URLS 一致）
R2_BASE_URL = "https://pub-3555e9bb2d304fab9c6c79819e48aa40.r2.dev"

REMOTE_MANIFEST_NAME = "mjthor_resource_file_to_size_mb.json"

# 并发下载解压的线程数：瓶颈在 NAS 写小文件（实测 ~12 文件/秒/线程组），
# 线程太多收益递减，24 是实测较稳的值。
DEFAULT_WORKERS = 24

REQUEST_TIMEOUT = (15, 600)  # (连接, 读取)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="如 procthor-10k-train")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-download-mb", type=float, default=5000.0, help="下载量安全上限（MB）"
    )
    return parser.parse_args()


def safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    """解压到 dest，做路径穿越检查。

    注意：官方 installer 解压时会把文件 chmod 成 0444（read_only），所以
    半途中断的包会留下只读文件；再解压到同一路径会 EACCES。这里按官方
    _safe_extract 的做法吞掉"文件已存在"的 PermissionError。
    """
    root = str(dest.resolve())
    for member in tar:
        target = os.path.realpath(os.path.join(dest, member.name))
        if not target.startswith(root):
            raise RuntimeError(f"路径穿越: {member.name}")
        try:
            tar.extract(member, path=dest)
        except PermissionError:
            if not os.path.exists(target):
                raise


def download_and_extract(pkg: str, source: str, version: str, cache_dest: Path) -> tuple[str, bool]:
    url = f"{R2_BASE_URL}/scenes/{source}/{version}/{pkg}"
    flag = cache_dest / f".{pkg}_complete_extract"
    if flag.exists():
        return pkg, True
    try:
        resp = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        with zstd.ZstdDecompressor().stream_reader(resp.raw) as reader:
            with tarfile.open(fileobj=reader, mode="r|*") as tar:
                safe_extract(tar, cache_dest)
        flag.touch(exist_ok=False)
        return pkg, True
    except Exception as exc:
        return pkg, f"{type(exc).__name__}: {str(exc)[:120]}"


def main() -> int:
    args = parse_args()

    from molmo_spaces.molmo_spaces_constants import (
        DATA_CACHE_DIR,
        DATA_TYPE_TO_SOURCE_TO_VERSION,
    )

    source = args.source
    version = DATA_TYPE_TO_SOURCE_TO_VERSION["scenes"][source]
    prefix = source.rsplit("-", 1)[-1]
    cache_dest = Path(DATA_CACHE_DIR) / "scenes" / source / version

    print(f"source = {source}  version = {version}")
    print(f"解压目标 = {cache_dest}", flush=True)

    if not (cache_dest / REMOTE_MANIFEST_NAME).exists():
        print("cache 里没有远端清单，请先跑一次官方流程。")
        return 1

    with open(cache_dest / REMOTE_MANIFEST_NAME) as f:
        remote_manifest = json.load(f)

    head = f"{source}_{prefix}_"
    all_pkgs = {}
    for name, size in remote_manifest.items():
        if not name.startswith(head):
            continue
        if name[len(head) :].split(".tar")[0].isdigit():
            all_pkgs[name] = size
    print(f"远端清单：{len(all_pkgs)} 个 house 包", flush=True)

    # 一次 listdir 拿到已解压的包（避免逐包 stat NAS）
    t = time.time()
    cache_names = set(os.listdir(cache_dest))
    print(f"扫描 cache 目录：{time.time() - t:.1f}s", flush=True)
    have = {n[1 : -len("_complete_extract")] for n in cache_names if n.endswith("_complete_extract")}

    todo = {p: s for p, s in all_pkgs.items() if p not in have}
    dl_mb = sum(todo.values())
    print(f"已解压 {len(have)}，待处理 {len(todo)}，需下载 {dl_mb:.1f} MB", flush=True)

    if not todo:
        print("无需下载。")
        return 0
    if dl_mb > args.max_download_mb:
        print(f"下载量超过上限 {args.max_download_mb} MB，已中止。")
        return 1
    if args.dry_run:
        print("--dry-run，未做任何改动。")
        return 0

    failed = {}
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {
            pool.submit(download_and_extract, pkg, source, version, cache_dest): pkg
            for pkg in todo
        }
        for fut in as_completed(futs):
            pkg, ok = fut.result()
            done += 1
            if ok is not True:
                failed[pkg] = ok
            if done % 20 == 0 or done == len(todo):
                rate = done / (time.time() - t0)
                eta = (len(todo) - done) / rate if rate else 0
                print(
                    f"  {done}/{len(todo)}  失败 {len(failed)}  "
                    f"{rate:.2f} 包/秒  ETA {eta / 60:.0f} 分钟",
                    flush=True,
                )
                if failed:
                    # 失败原因归类，便于判断是网络问题还是 NAS 写超时
                    from collections import Counter

                    kinds = Counter(v.split(":")[0] for v in failed.values())
                    print(f"    失败类型: {dict(kinds)}", flush=True)

    print(f"\n完成：成功 {len(todo) - len(failed)}，失败 {len(failed)}，用时 {(time.time() - t0) / 60:.1f} 分钟")
    if failed:
        print(f"失败清单（前 10）：{list(failed.items())[:10]}")
        print("重跑本脚本即可续传（已完成的包会被跳过）。")
        return 1
    print("下一步：python scripts/assets/fast_relink_scene_split.py --source " + source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
