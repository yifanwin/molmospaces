#!/usr/bin/env bash
# 等 procthor-10k-test / train 的下载解压跑完后，自动建链接并校验。
# 设计成可以无人值守长跑：失败会自动重试，全部完成后写一份结果到日志。
set -u

MOLMOSPACES=/data0/wenyifan/MoMaTrajGen/molmospaces
LOG=/tmp/finish_backfill.log
cd "$MOLMOSPACES" || exit 1
# shellcheck disable=SC1091
source setup_env.sh

PY=./.venv/bin/python
SPLITS=(procthor-10k-test procthor-10k-train)

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "等待现有下载进程结束..."
while pgrep -f "fast_backfill_scene_split" > /dev/null; do
    sleep 300
done
log "下载进程已退出，进入重试循环（把失败包补掉）"

for round in $(seq 1 30); do
    all_ok=1
    for s in "${SPLITS[@]}"; do
        if ! "$PY" -u scripts/assets/fast_backfill_scene_split.py --source "$s" --workers 24 >> "$LOG" 2>&1; then
            all_ok=0
            log "  $s 第 $round 轮仍有失败"
        fi
    done
    [ "$all_ok" -eq 1 ] && { log "所有包已解压完成"; break; }
    log "第 $round 轮未全成，60s 后重试"
    sleep 60
done

log "开始建链接"
for s in "${SPLITS[@]}"; do
    "$PY" -u scripts/assets/fast_relink_scene_split.py --source "$s" >> "$LOG" 2>&1
    log "  $s 建链结束"
done

log "=== 最终校验 ==="
"$PY" -u - <<'PYEOF' >> "$LOG" 2>&1
import os
VER = {'procthor-10k-train': '20251122_with_occupancy',
       'procthor-10k-test': '20251121_with_occupancy'}
SUF = ['.xml', '.json', '_ceiling.xml', '_map.png', '_metadata.json']
assets = '/data0/wenyifan/MoMaTrajGen/molmospaces_data/assets/scenes'
cache_base = '/nas/wenyifan/molmospaces_data/cache/scenes'
for src, ver in VER.items():
    pfx = src.rsplit('-', 1)[-1]
    link = f'{assets}/{src}'
    names = set(os.listdir(link))
    cache = f'{cache_base}/{src}/{ver}'
    cnames = set(os.listdir(cache))
    ok = 0
    for i in range(10000 if 'train' in src else 1000):
        b = f'{pfx}_{i}'
        if f'{b}.xml' not in names:
            continue
        if not all(f'{b}{s}' in names for s in SUF):
            continue
        if f'.{src}_{b}.tar.zst_complete_extract' not in cnames:
            continue
        ok += 1
    print(f'{src}: 完整可用 {ok}')
PYEOF
log "全部流程结束"
