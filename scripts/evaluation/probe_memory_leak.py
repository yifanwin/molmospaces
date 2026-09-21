#!/usr/bin/env python3
"""逐 episode 内存泄漏探针（只读测量，不修改被测代码）

结论（2026-09-21 实测，先看这里再决定要不要重跑）
--------------------------------------------------
1. **不是对象泄漏**：CPUMujocoEnv / MjModel / MjData / Renderer / CuroboPlanner / MotionGen
   每 episode 全部被正确释放（存活数恒定），gc 跟踪对象总字节恒定 0.13 GiB；
2. **真因是 glibc malloc 的 arena 膨胀与碎片**：每 episode 重建整栋房子的场景 + 新建线程池，
   释放后的空闲块留在 arena 里不还给 OS。实测 arena 3.51→4.28→4.76 GiB 持续涨，
   其中空闲块 1.29→2.16→2.36 GiB（近 50% 浪费）；
3. **调优有效**：加 `MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=131072
   MALLOC_MMAP_THRESHOLD_=131072` 后，ΔRSS 从 +0.73 GiB/episode 降到 +0.11 GiB/episode
   （3 轮累计 5.38 → 2.88 GiB），arena 空闲块不再膨胀。

背景
----
E2（CuRobo planner，`MolmoBotRBY1CuroboPickPnPEvalConfig/20260920_170037`）长跑 16.4 h，
wenyifan 进程 RSS 从 17 GiB 单调涨到 183 GiB（+10.1 GiB/h ≈ +0.38 GiB/episode，全程零回落），
系统可用内存跌到 6.6 GiB 后被内核 OOM killer 杀掉。E3（learned policy，无 CuRobo）
同样单调上涨（+0.59 GiB/episode）——两者策略不同却同速，指向共用路径：
**每个 episode 重建整栋房子的场景模型（MjSpec → compile）并新建 CPUMujocoEnv**
（E2 443 次 / E3 226 次 "Scene updated"，与 episode 数 1:1）。

本探针做什么
------------
复用真实 pipeline（run_evaluation），只在 `cleanup_episode_resources` 外面套一层普查钩子，
跑一个 house 的前 N 个 episode（默认 house 82：E2 里 9 个 episode、约 15 分钟），
每个 episode 结束后打印：

1. RSS 增量 + `mallinfo2`/`smaps_rollup` 堆拆分（malloc arena vs mmap）—— 定位层次；
2. 存活对象计数（env / MjModel / MjData / Renderer / planner / task / sampler）——
   哪一类逐 episode 线性增长，哪一类就是泄漏载体；
3. gc 跟踪对象的类型计数与**类型字节**增量，并对增长最多的类型取最大的几个实例
   看内容与引用链；
4. （可选，TRACEMALLOC_ENABLED=True）Python 侧分配点差分。

为什么不能只靠 gc
-----------------
mujoco.MjModel / MjData 不被 gc 跟踪（`gc.is_tracked=False`），`gc.get_objects()` 看不见它们，
所以在构造时登记 weakref（它们支持 weakref），之后统计"还剩几个活着"。

用法
----
    bash scripts/evaluation/run_probe_memory_leak.sh            # 推荐：自动 source setup_env.sh
    bash scripts/evaluation/run_probe_memory_leak.sh --self-check   # 只验证普查机制，不跑评测
    .venv/bin/python scripts/evaluation/probe_memory_leak.py --help

产物
----
    /tmp/probe_memleak/census.log      普查日志（同时打到 stdout）
    /tmp/probe_memleak/eval_output/    评测产物（独立目录，不污染正式 eval_output）
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import importlib
import importlib.util
import inspect
import os
import sys
import time
import tracemalloc
import types
import weakref
from collections import Counter
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ 配置区
DEFAULT_BENCHMARK = (
    "/nas/wenyifan/molmospaces_data/cache/benchmarks/molmospaces-bench-v1/20260408/procthor-10k/"
    "RBY1PickAndPlaceDataGenConfig/RBY1PickAndPlaceDataGenConfig_20260209_json_benchmark"
)
DEFAULT_CONFIG_CLS = (
    "molmo_spaces.evaluation.configs.evaluation_configs:RBY1CuroboPickPnPEvalConfig"
)
DEFAULT_HOUSE_INDEX = 82        # E2 里 9 个 episode、约 15 分钟跑完
DEFAULT_MAX_EPISODES = 9
DEFAULT_TASK_HORIZON_STEPS = 600

PROBE_ROOT = Path("/tmp/probe_memleak")
CENSUS_LOG = PROBE_ROOT / "census.log"
EVAL_OUTPUT_DIR = PROBE_ROOT / "eval_output"

RSS_ABORT_GB = 60.0             # 自我保护：探针不该把机器跑成事故
REFERRER_MAX_CALLS = 3          # 每个 episode 最多做几次引用链回溯（gc.get_referrers 要扫全部跟踪对象）
REFERRER_DEPTH = 6
REFERRER_MAX_LINES = 50
REFERRER_MAX_NODES = 20         # 每棵树最多对多少个节点跑 gc.get_referrers
CENSUS_TOP_TYPES = 15
TRACEMALLOC_ENABLED = False     # 置 True 可定位 Python 侧分配点（会拖慢约 1 倍）
TRACEMALLOC_DIFF_TOP = 10

# 逐 episode 存活数会被统计的对象类别
WATCHED = (
    "CPUMujocoEnv",
    "MjModel",
    "MjData",
    "Renderer",
    "TaskSampler",
    "Task",
    "Policy",
    "CuroboPlanner",
    "MotionGen",
)
# ---------------------------------------------------------------- 配置区结束


SURVIVORS: dict[str, list[weakref.ref]] = {name: [] for name in WATCHED}
CREATED: Counter = Counter()
PREV_ALIVE: dict[str, int] = {name: 0 for name in WATCHED}
EPISODE_NO = 0
SCENE_REBUILDS = 0
LAST_RSS_GB = 0.0
LAST_TYPE_CENSUS: Counter = Counter()
LAST_TYPE_SIZES: Counter = Counter()
LAST_TOTAL_TRACKED = 0
LAST_TRACEMALLOC = None
REFERRER_BUDGET = 0


# ------------------------------------------------------------------ 基础设施
def emit(text: str = "") -> None:
    """同时打到 stdout 与普查日志（worker 会重定向 stdout，落盘这份才是保底）"""
    print(text, flush=True)
    with open(CENSUS_LOG, "a", encoding="utf-8") as fh:
        fh.write(text + "\n")


def rss_gb() -> float:
    with open("/proc/self/status", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


def available_gb() -> float:
    with open("/proc/meminfo", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


class MallInfo2(ctypes.Structure):
    """glibc 的 mallinfo2 —— 把进程堆拆成"从 OS 拿的 arena"与"真正在用的字节"

    用途：跟踪字节不动、RSS 却在涨时，用它区分
      (a) malloc 碎片/arena 膨胀（arena 涨、uordblks 不涨）
      (b) 内存根本不经 glibc malloc —— 在 mmap 区里（GPU 驱动 / EGL / CUDA / mmap 文件）
    """

    _fields_ = [
        ("arena", ctypes.c_size_t),
        ("ordblks", ctypes.c_size_t),
        ("smblks", ctypes.c_size_t),
        ("hblks", ctypes.c_size_t),
        ("hblkhd", ctypes.c_size_t),
        ("usmblks", ctypes.c_size_t),
        ("fsmblks", ctypes.c_size_t),
        ("uordblks", ctypes.c_size_t),
        ("fordblks", ctypes.c_size_t),
        ("keepcost", ctypes.c_size_t),
    ]


LIBC = ctypes.CDLL("libc.so.6")
LIBC.mallinfo2.restype = MallInfo2


def heap_stats() -> dict:
    """malloc 堆与 smaps 的拆分（都换算成 GiB）"""
    info = LIBC.mallinfo2()
    stats = {
        "malloc_arena": info.arena / 1073741824,
        "malloc_inuse": info.uordblks / 1073741824,
        "malloc_free": info.fordblks / 1073741824,
        "mmap_blocks": info.hblkhd / 1073741824,
    }
    with open("/proc/self/smaps_rollup", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("Rss:"):
                stats["smaps_rss"] = int(line.split()[1]) / 1048576
            elif line.startswith("Private_Dirty:"):
                stats["private_dirty"] = int(line.split()[1]) / 1048576
            elif line.startswith("Anonymous:"):
                stats["anonymous"] = int(line.split()[1]) / 1048576
    return stats


def track(label: str, obj) -> None:
    """登记一个新构造的对象（weakref 指向它，不阻止其被回收）"""
    if obj is None:
        return
    CREATED[label] += 1
    SURVIVORS[label].append(weakref.ref(obj))


def track_if_absent(label: str, obj) -> None:
    """登记对象，但同一个对象不重复登记（普查时的补登记用，按身份判重）"""
    if obj is None:
        return
    if any(ref() is obj for ref in SURVIVORS[label]):
        return
    track(label, obj)


def alive(label: str) -> list:
    """当前仍活着的该类对象"""
    live = []
    for ref in SURVIVORS[label]:
        obj = ref()
        if obj is not None:
            live.append(obj)
    return live


def short(obj) -> str:
    """给引用链用的紧凑描述"""
    if isinstance(obj, dict):
        if "__name__" in obj and "__file__" in obj:
            return f"module({obj['__name__']})"
        return f"dict(len={len(obj)}, keys={list(obj.keys())[:6]})"
    if isinstance(obj, (list, tuple, set)):
        return f"{type(obj).__name__}(len={len(obj)})"
    if isinstance(obj, str):
        return f"str({obj[:60]!r})"
    return f"{type(obj).__module__}.{type(obj).__qualname__} @{id(obj):#x}"


def holding_slot(container, target) -> str:
    """容器里是哪个 key/index 指向目标的 —— 引用链里最有用的一行"""
    if isinstance(container, dict):
        for key, value in container.items():
            if value is target:
                return f"  [key={key!r}]"
    if isinstance(container, (list, tuple)):
        for index, value in enumerate(container):
            if value is target:
                return f"  [index={index}]"
    return ""


def is_transient(obj) -> bool:
    """迭代器等临时对象是死胡同，跳过（它们只会在链上刷屏）"""
    return type(obj).__name__.endswith("iterator")


def interest(obj) -> int:
    """引用链的探索优先级：本仓库对象 > 容器 > mujoco/numpy 等第三方内部结构"""
    if type(obj).__module__.startswith("molmo_spaces"):
        return 0
    if isinstance(obj, (dict, list, tuple)):
        return 1
    return 2


# ------------------------------------------------------------------ 挂钩
def patch_init(module_name: str, class_name: str, label: str, child_attrs: tuple = ()) -> bool:
    """把某个类的 __init__ 包一层：构造完成后登记 self 与指定子属性

    child_attrs 是 ((属性名, 统计标签), ...)，一个属性可以归到别的类别下
    （例如 env 的 `_mj_model` 归到 "MjModel"）。
    """
    if importlib.util.find_spec(module_name) is None:
        return False
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name, None)
    if cls is None:
        return False
    original = cls.__init__

    def patched(self, *args, **kwargs):
        original(self, *args, **kwargs)
        track_if_absent(label, self)
        for attr, attr_label in child_attrs:
            track_if_absent(attr_label, getattr(self, attr, None))

    cls.__init__ = patched
    return True


def patch_scene_rebuild_counter() -> bool:
    """统计 setup_robot_scene 调用次数，并登记它返回的 MjModel

    该函数返回的就是本 episode 新编译出来的整栋房子的模型——逐 episode 泄漏的首要嫌疑。
    """
    import molmo_spaces.tasks.task_sampler as sampler_module

    original = sampler_module.BaseMujocoTaskSampler.setup_robot_scene

    def patched(self, *args, **kwargs):
        global SCENE_REBUILDS
        SCENE_REBUILDS += 1
        model = original(self, *args, **kwargs)
        track_if_absent("MjModel", model)
        return model

    sampler_module.BaseMujocoTaskSampler.setup_robot_scene = patched
    return True


def patch_cleanup_hook() -> bool:
    """hook 点：pipeline 每完成一个 episode 就调用 cleanup_episode_resources"""
    import molmo_spaces.data_generation.pipeline as pipeline

    original = pipeline.cleanup_episode_resources
    signature = inspect.signature(original)

    def wrapper(*args, **kwargs):
        original(*args, **kwargs)          # 先让被测代码完成它自己的清理
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        run_census(
            task=bound.arguments["task"],
            task_sampler=bound.arguments["task_sampler"],
            policy=bound.arguments["policy"],
        )

    pipeline.cleanup_episode_resources = wrapper
    return True


def install_hooks() -> list:
    hooks = []
    if patch_init("molmo_spaces.env.env", "CPUMujocoEnv", "CPUMujocoEnv",
                  child_attrs=(("_mj_model", "MjModel"),)):
        hooks.append("CPUMujocoEnv.__init__")
    if patch_init("molmo_spaces.tasks.task", "BaseMujocoTask", "Task"):
        hooks.append("BaseMujocoTask.__init__")
    if patch_init("molmo_spaces.tasks.task_sampler", "BaseMujocoTaskSampler", "TaskSampler"):
        hooks.append("BaseMujocoTaskSampler.__init__")
    if patch_init("molmo_spaces.policy.base_policy", "BasePolicy", "Policy"):
        hooks.append("BasePolicy.__init__")
    if patch_init("molmo_spaces.planner.curobo_planner", "CuroboPlanner", "CuroboPlanner",
                  child_attrs=(("motion_gen", "MotionGen"),)):
        hooks.append("CuroboPlanner.__init__")
    if patch_init("molmo_spaces.renderer.opengl_rendering", "MjOpenGLRenderer", "Renderer"):
        hooks.append("MjOpenGLRenderer.__init__")
    if patch_init("molmo_spaces.renderer.filament_rendering", "MjFilamentRenderer", "Renderer"):
        hooks.append("MjFilamentRenderer.__init__")
    if patch_scene_rebuild_counter():
        hooks.append("setup_robot_scene")
    if patch_cleanup_hook():
        hooks.append("cleanup_episode_resources")
    return hooks


def sweep_renderers() -> None:
    """兜底补登记：renderer / MjData 是懒创建的，普查时从存活的 env 上再扫一遍"""
    for env in alive("CPUMujocoEnv"):
        track_if_absent("MjModel", getattr(env, "_mj_model", None))
        track_if_absent("Renderer", getattr(env, "_renderer", None))
        for data in getattr(env, "_mj_datas", None) or []:
            track_if_absent("MjData", data)
    for planner in alive("CuroboPlanner"):
        track_if_absent("MotionGen", getattr(planner, "motion_gen", None))


# ------------------------------------------------------------------ 普查
def episode_label(task) -> str:
    """抓一句"本 episode 在搬什么物体"，方便和评测日志对时间线"""
    for path in (("config", "task_config", "pickup_obj_name"),
                 ("config", "pickup_obj_name"),
                 ("task_config", "pickup_obj_name")):
        node = task
        for attr in path:
            node = getattr(node, attr, None)
        if node is not None:
            return str(node)
    return type(task).__name__ if task is not None else "unknown"


def print_referrer_tree(obj, label: str) -> None:
    """打印谁还拽着这个对象（gc.get_referrers 按引用扫描，非跟踪对象也能查到引用者）"""
    global REFERRER_BUDGET
    if REFERRER_BUDGET <= 0:
        return
    REFERRER_BUDGET -= 1
    emit(f"  引用链 [{label}] {short(obj)}")
    seen = {id(obj)}
    queue = [(obj, 1)]
    lines = 0
    nodes = 0
    while queue and lines < REFERRER_MAX_LINES and nodes < REFERRER_MAX_NODES:
        current, depth = queue.pop(0)
        if depth > REFERRER_DEPTH:
            continue
        nodes += 1
        # 容器优先（dict/list 才标得出"是谁的哪个字段"），迭代器这类临时对象跳过
        referrers = [
            ref
            for ref in gc.get_referrers(current)
            if not isinstance(ref, (types.FrameType, weakref.ReferenceType))
            and not is_transient(ref)
        ]
        referrers.sort(key=interest)
        for referrer in referrers:
            if id(referrer) in seen:
                continue
            seen.add(id(referrer))
            emit(f"{'    ' * depth}← {short(referrer)}{holding_slot(referrer, current)}")
            lines += 1
            if lines >= REFERRER_MAX_LINES:
                emit("    …（引用链输出到上限）")
                break
            queue.append((referrer, depth + 1))
    if lines == 0:
        emit("    （Python 侧无引用者：要么它挂在 C 层全局上，要么只被 weakref 指向）")


def type_census() -> tuple[Counter, Counter]:
    """返回 (类型计数, 类型字节数)

    字节数用 sys.getsizeof 浅算 + numpy 数组按其 nbytes 计（视图会重复计，仅作量级参考）。
    这一步能抓到"对象个数没涨但内存涨"的泄漏，例如少数几个大 numpy 数组。
    """
    counts: Counter = Counter()
    sizes: Counter = Counter()
    for obj in gc.get_objects():
        key = f"{type(obj).__module__}.{type(obj).__qualname__}"
        counts[key] += 1
        if isinstance(obj, np.ndarray):
            sizes[key] += obj.nbytes
        else:
            sizes[key] += sys.getsizeof(obj)
    return counts, sizes


def find_sample_of_type(key: str):
    """从 gc 里捞一个该类型的实例，用于看内容/引用链"""
    for obj in gc.get_objects():
        if f"{type(obj).__module__}.{type(obj).__qualname__}" == key:
            return obj
    return None


def object_weight(obj) -> int:
    """粗排"这个大不大"：numpy 按 nbytes，容器按 getsizeof（随长度增长）"""
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    return sys.getsizeof(obj)


def find_biggest_of_type(key: str, count: int = 3) -> list:
    """取该类型里最大的几个实例 —— 泄漏的载体通常是大对象，随机的老实例没信息量"""
    found = []
    for obj in gc.get_objects():
        if f"{type(obj).__module__}.{type(obj).__qualname__}" == key:
            found.append((object_weight(obj), obj))
    found.sort(key=lambda item: -item[0])
    return [obj for _, obj in found[:count]]


def element_histogram(obj, limit: int = 200) -> str:
    """容器里装的是什么 —— 一眼看出是帧、路径点还是网格"""
    if isinstance(obj, dict):
        items = list(obj.values())[:limit]
    elif isinstance(obj, (list, tuple, set, frozenset)):
        items = list(obj)[:limit]
    else:
        return ""
    histogram: Counter = Counter()
    for item in items:
        histogram[f"{type(item).__module__}.{type(item).__qualname__}"] += 1
    return ", ".join(f"{key}×{value}" for key, value in histogram.most_common(6))


def sample_preview(obj, limit: int = 5) -> str:
    """给增长类型看几眼内容，判断它是什么"""
    if isinstance(obj, np.ndarray):
        return f"ndarray shape={obj.shape} dtype={obj.dtype}"
    if isinstance(obj, (set, frozenset, list, tuple)):
        items = list(obj)[:limit]
        return f"len={len(obj)} 前 {len(items)} 个元素: {items}"
    if isinstance(obj, dict):
        items = list(obj.items())[:limit]
        return f"len={len(obj)} 前 {len(items)} 项: {items}"
    return short(obj)


def run_census(task, task_sampler, policy) -> None:
    global EPISODE_NO, LAST_RSS_GB, LAST_TYPE_CENSUS, LAST_TYPE_SIZES, LAST_TOTAL_TRACKED
    global LAST_TRACEMALLOC, REFERRER_BUDGET

    EPISODE_NO += 1
    REFERRER_BUDGET = REFERRER_MAX_CALLS

    gc.collect()
    gc.collect()
    sweep_renderers()
    track_if_absent("TaskSampler", task_sampler)
    track_if_absent("Policy", policy)

    rss = rss_gb()
    delta = rss - LAST_RSS_GB
    tracked_objects = len(gc.get_objects())

    emit("")
    emit("=" * 78)
    emit(f"episode #{EPISODE_NO} 结束 | object={episode_label(task)} | "
         f"{time.strftime('%H:%M:%S')}")
    emit(f"RSS {rss:.2f} GiB   Δ {delta:+.2f} GiB   | 系统可用 {available_gb():.1f} GiB   "
         f"| gc 跟踪对象 {tracked_objects:,}")

    heap = heap_stats()
    emit(f"  malloc: arena {heap['malloc_arena']:.2f} GiB（在用 {heap['malloc_inuse']:.2f} "
         f"/ 空闲 {heap['malloc_free']:.2f}）| mmap 块 {heap['mmap_blocks']:.2f} GiB")
    emit(f"  smaps: Rss {heap['smaps_rss']:.2f} | Anonymous {heap['anonymous']:.2f} "
         f"| Private_Dirty {heap['private_dirty']:.2f} GiB")
    emit("  读法：malloc arena 与 smaps 都不涨而 RSS 涨 → 真泄漏在别处（GPU 驱动/C 层全局）；")
    emit("        只有 arena 涨 → malloc 碎片/arena 膨胀，不是对象泄漏。")

    created_now = ", ".join(f"{name}×{CREATED[name]}" for name in WATCHED if CREATED[name])
    emit(f"累计新建: {created_now or '无'}")
    emit(f"场景重建(setup_robot_scene) {SCENE_REBUILDS} 次")

    emit("-" * 78)
    emit(f"{'类别':<16}{'新建':>6}{'仍存活':>8}{'上次存活':>10}{'本集增量':>10}")
    grown = []
    for name in WATCHED:
        live = alive(name)
        previous = PREV_ALIVE[name]
        emit(f"{name:<16}{CREATED[name]:>6}{len(live):>8}{previous:>10}{len(live) - previous:>+10}")
        if len(live) - previous > 0:
            grown.append((name, live))
        PREV_ALIVE[name] = len(live)

    if grown:
        emit("-" * 78)
        emit("↑ 上面这些类别的存活数在本 episode 增加了 —— 逐个看引用链:")
        for name, live in grown:
            print_referrer_tree(live[-1], name)
    else:
        emit("（本 episode 无类别存活数增长）")

    counts, sizes = type_census()
    growth = Counter()
    for key, value in counts.items():
        growth[key] = value - LAST_TYPE_CENSUS.get(key, value)
    size_growth = Counter()
    for key, value in sizes.items():
        size_growth[key] = value - LAST_TYPE_SIZES.get(key, 0)
    total_tracked = sum(sizes.values())

    positive = [
        (key, delta_count)
        for key, delta_count in growth.items()
        if delta_count > 0 and not key.startswith("weakref.")   # 探针自己的 weakref 登记表是噪声
    ]
    positive.sort(key=lambda item: -item[1])
    if positive:
        emit("-" * 78)
        emit(f"gc 跟踪对象增量 top{CENSUS_TOP_TYPES}（Python 侧，C 层对象不在此列）:")
        for key, delta_count in positive[:CENSUS_TOP_TYPES]:
            emit(f"    {delta_count:>+8}  {key}")

    heavy = [(key, delta) for key, delta in size_growth.items() if delta > 1048576]
    heavy.sort(key=lambda item: -item[1])
    emit("-" * 78)
    emit(f"跟踪对象总字节 {total_tracked / 1073741824:.2f} GiB  "
         f"(Δ {total_tracked - LAST_TOTAL_TRACKED:+.0f} MiB)   "
         f"| RSS Δ {delta * 1024:+.0f} MiB")
    emit("这两个数对照看：跟踪字节涨得和 RSS 一样多 → 泄漏是 Python 对象；")
    emit("跟踪字节不动而 RSS 涨 → 泄漏在 C 层（mujoco / GL / warp），Python 侧看不到。")
    if heavy:
        emit(f"类型字节增长 top{CENSUS_TOP_TYPES}（>1 MiB 才列）:")
        for key, delta_bytes in heavy[:CENSUS_TOP_TYPES]:
            emit(f"    {delta_bytes / 1048576:>+9.2f} MiB  {key}")

    # 对增长最多的类型直接看最大的几个实例 —— 泄漏载体通常就在这里
    suspect_key = heavy[0][0] if heavy else (positive[0][0] if positive else None)
    if suspect_key is not None:
        emit("-" * 78)
        emit(f"增长最多的类型 {suspect_key} —— 取其中最大的实例:")
        for sample in find_biggest_of_type(suspect_key):
            emit(f"    {object_weight(sample) / 1024:.1f} KiB  {sample_preview(sample)}")
            histogram = element_histogram(sample)
            if histogram:
                emit(f"        元素构成: {histogram}")
            print_referrer_tree(sample, suspect_key)

    LAST_TYPE_CENSUS = counts
    LAST_TYPE_SIZES = sizes
    LAST_TOTAL_TRACKED = total_tracked

    if TRACEMALLOC_ENABLED:
        snapshot = tracemalloc.take_snapshot()
        if LAST_TRACEMALLOC is not None:
            emit("-" * 78)
            emit("Python 侧分配增长 top（tracemalloc 差分）:")
            for stat in snapshot.compare_to(LAST_TRACEMALLOC, "lineno")[:TRACEMALLOC_DIFF_TOP]:
                frame = stat.traceback[0]
                emit(f"    {stat.size_diff / 1048576:+8.2f} MiB  {frame.filename}:{frame.lineno}")
        LAST_TRACEMALLOC = snapshot

    LAST_RSS_GB = rss
    if rss > RSS_ABORT_GB:
        emit(f"!! RSS {rss:.2f} GiB 已超过自我保护阈值 {RSS_ABORT_GB} GiB，探针主动退出")
        os._exit(3)


# ------------------------------------------------------------------ 收尾与入口
def print_summary() -> None:
    emit("")
    emit("=" * 78)
    emit(f"汇总：{EPISODE_NO} 个 episode | 场景重建 {SCENE_REBUILDS} 次 | "
         f"RSS 终点 {rss_gb():.2f} GiB")
    emit(f"{'类别':<16}{'新建':>6}{'仍存活':>8}{'每 episode 存活增量':>22}")
    leaks = []
    for name in WATCHED:
        live = len(alive(name))
        per_episode = live / EPISODE_NO if EPISODE_NO else 0.0
        emit(f"{name:<16}{CREATED[name]:>6}{live:>8}{per_episode:>22.2f}")
        if per_episode >= 0.5:
            leaks.append(name)
    if leaks:
        emit("")
        emit(f"泄漏载体候选（每 episode 至少留存 1 个）: {', '.join(leaks)}")
        emit("下一步：看上面这些类别的引用链，确认是哪个长生命周期对象/环拽住了它们。")
    else:
        emit("")
        emit("没有类别呈现'每 episode 留存一个'的模式：泄漏更可能在 C 层（GL/驱动）或")
        emit("Python 容器里（看 gc 跟踪对象增量那一节）。")


class _SelfCheckHolder:
    """自检用的假 holder：模拟"某个长生命周期对象（如 env）拽着模型" """

    def __init__(self, model) -> None:
        self.mj_model = model
        self.mj_datas = []


def self_check() -> None:
    """不跑评测，只验证普查机制：故意造 3 个不释放的模型，看存活数是否每次 +1"""
    emit("== 自检：验证普查机制（不跑评测）==")
    hooks = install_hooks()
    emit(f"钩子安装: {', '.join(hooks)}")
    emit(f"molmo_spaces: {importlib.import_module('molmo_spaces').__file__}")
    import mujoco

    keepalive = []          # 模块级持有 → 模拟"被长生命周期对象拽住"的泄漏
    for _ in range(3):
        holder = _SelfCheckHolder(mujoco.MjModel.from_xml_string("<mujoco/>"))
        keepalive.append(holder)
        track("MjModel", holder.mj_model)
        run_census(None, None, None)
    print_summary()
    emit(f"自检期望：MjModel 每次 +1，引用链能指到 _SelfCheckHolder 与 keepalive 列表")
    emit(f"（keepalive 持有 {len(keepalive)} 个 holder）")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="逐 episode 内存泄漏探针（只读测量）")
    parser.add_argument("--benchmark_dir", default=DEFAULT_BENCHMARK,
                        help="JSON benchmark 目录")
    parser.add_argument("--eval_config_cls", default=DEFAULT_CONFIG_CLS,
                        help="评测配置类，module:ClassName")
    parser.add_argument("--house_index", type=int, default=DEFAULT_HOUSE_INDEX,
                        help="只跑这个 house（先按 house 过滤，再截断 episode 数）；-1 表示不过滤")
    parser.add_argument("--max_episodes", type=int, default=DEFAULT_MAX_EPISODES,
                        help="最多跑几个 episode（不过滤 house 时按 benchmark 顺序跨多个 house）")
    parser.add_argument("--task_horizon_steps", type=int, default=DEFAULT_TASK_HORIZON_STEPS,
                        help="每个 episode 的最大步数")
    parser.add_argument("--output_dir", default=str(EVAL_OUTPUT_DIR),
                        help="评测产物目录（独立目录，勿指向正式 eval_output）")
    parser.add_argument("--self-check", action="store_true",
                        help="只验证普查机制，不跑评测")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    PROBE_ROOT.mkdir(parents=True, exist_ok=True)
    CENSUS_LOG.write_text("", encoding="utf-8")
    if TRACEMALLOC_ENABLED:
        tracemalloc.start(1)

    if args.self_check:
        self_check()
        return

    house_filter = None if args.house_index < 0 else args.house_index
    emit(f"探针启动 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    emit(f"  molmo_spaces: {importlib.import_module('molmo_spaces').__file__}")
    emit(f"  配置类: {args.eval_config_cls}")
    emit(f"  house {house_filter if house_filter is not None else '全部（按 benchmark 顺序）'}，"
         f"最多 {args.max_episodes} 个 episode，{args.task_horizon_steps} 步/episode")
    hooks = install_hooks()
    emit(f"  已挂钩: {', '.join(hooks)}")
    emit(f"  基线 RSS {rss_gb():.2f} GiB | 系统可用 {available_gb():.1f} GiB")

    from molmo_spaces.evaluation.eval_main import run_evaluation

    run_evaluation(
        eval_config_cls=args.eval_config_cls,
        benchmark_dir=Path(args.benchmark_dir),
        task_horizon_steps=args.task_horizon_steps,
        output_dir=Path(args.output_dir),
        num_workers=1,                      # 单 worker 才在本人进程内跑，钩子与 RSS 才有效
        use_wandb=False,
        max_episodes=args.max_episodes,
        house_index=house_filter,
    )

    print_summary()
    emit(f"普查日志: {CENSUS_LOG}")
    emit(f"评测产物: {args.output_dir}")


if __name__ == "__main__":
    main()
