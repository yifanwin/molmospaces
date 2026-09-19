"""从 ProcTHOR 场景生成带自定义 target 的可达性研究场景。

本示例以 procthor-10k-val 的 val_4 为源，使用 ceiling 变体（带天花板，视觉更真实）。

核心流程：
    定位源场景 -> 建立自包含工作目录 -> 加载 MjSpec -> 编辑（删/移/加）
    -> 保存（含已知序列化坑的修复）-> 同步 metadata -> 编译 + 落地仿真验证

运行：
    cd ~/wenyifan/MoMaTrajGen/molmospaces
    source setup_env.sh && source .venv/bin/activate
    python custom_scenes/reachability_procthor/build_scene.py
"""

import json
import re
from pathlib import Path

import mujoco
import numpy as np

from molmo_spaces.molmo_spaces_constants import get_scenes

# ============================================================================
# 配置区：所有可调参数集中在这里
# ============================================================================

DATASET = "procthor-10k"
SPLIT = "val"
HOUSE_INDEX = 4

# 变体选择：
#   "ceiling" —— 带天花板，视觉与光照更接近真实室内（推荐做数据生成）
#   "base"    —— 无天花板，俯视与光照更干净（推荐做调试可视化）
# 两者共享同一份 <房子>_assets/ 目录，物理行为一致。
VARIANT = "ceiling"

# 自包含工作目录：必须保持 scenes/<数据集名>/ 的相对深度
WORK_DIR = Path(__file__).resolve().parent / "work"
DATASET_DIR_NAME = f"{DATASET}-{SPLIT}"
OUT_STEM = f"house_{HOUSE_INDEX:03d}"

# 源数据集里 mesh 目录的固定前缀（base 与 ceiling 共用）
SRC_MESH_PREFIX = f"{SPLIT}_{HOUSE_INDEX}_assets/"
DST_MESH_PREFIX = f"{OUT_STEM}_assets/"

# 要删除的场景物体：按 category 小写子串匹配（可从 metadata 里查）
REMOVE_CATEGORY_KEYWORDS = ["pillow", "plunger", "garbagecan"]

# 要挪动的场景物体（category 子串 -> z 轴偏移量，单位米）
MOVE_CATEGORY_Z_OFFSET = {"apple": 0.02, "egg": 0.02}

# 自定义 target 物体（一个带 freejoint 的圆柱，替代杯子）
TARGET_NAME = "target_cup"
TARGET_XML = f"""
<mujoco>
  <worldbody>
    <body name="{TARGET_NAME}">
      <freejoint name="{TARGET_NAME}_free"/>
      <geom name="{TARGET_NAME}_geom" type="cylinder" size="0.04 0.08"
            mass="0.2" rgba="0.9 0.2 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""

# target 的世界坐标初始位置（对应 val_4 餐厅区的餐桌上方，略高以便靠重力落下）
TARGET_POS = [3.49, 9.43, 0.95]

# 落地仿真参数
SETTLE_MAX_STEPS = 3000          # 最多仿真步数（0.002s/步 -> 6s）
SETTLE_VEL_TOL = 1e-2            # 判定静止的速度阈值（m/s）
SETTLE_STILL_STREAK = 50         # 连续多少步都低于阈值才算真的静止


# ============================================================================
# 已知坑的修复函数
# ============================================================================


def fix_gridlayout(xml: str) -> str:
    """修复 MjSpec.to_xml() 的 skybox gridlayout 序列化 bug。

    MuJoCo 内部把 gridlayout 存成定长数组（mjMAXTEXLAYOUT=12），
    to_xml() 会把整个数组写出来，导致长度与 gridsize 乘积不匹配，
    重新解析时报 "gridlayout length must match gridsize"。
    """
    pattern = r'gridsize="([0-9 ]+)" gridlayout="([^"]*)"'

    def truncate(match: re.Match) -> str:
        grid_size = [int(v) for v in match.group(1).split()]
        n_expected = grid_size[0] * grid_size[1]
        return f'gridsize="{match.group(1)}" gridlayout="{match.group(2)[:n_expected]}"'

    return re.sub(pattern, truncate, xml)


def setup_workdir(src_scene: Path, assets_root: Path) -> Path:
    """建立自包含目录，返回目标场景 XML 的路径。

    场景 XML 内部用相对路径引用资源：
        <mesh>      <数据集目录>/val_4_assets/xxx.obj
        <texture>   ../../objects/thor/...
    因此输出目录必须复刻 scenes/<数据集名>/ 这一层深度。
    """
    dataset_dir = WORK_DIR / "scenes" / DATASET_DIR_NAME
    dataset_dir.mkdir(parents=True, exist_ok=True)

    objects_link = WORK_DIR / "objects"
    if not objects_link.exists():
        objects_link.symlink_to(assets_root / "objects")

    assets_link = dataset_dir / DST_MESH_PREFIX.rstrip("/")
    if not assets_link.exists():
        assets_link.symlink_to(src_scene.parent / SRC_MESH_PREFIX.rstrip("/"))

    return dataset_dir / f"{OUT_STEM}.xml"


# ============================================================================
# 场景编辑
# ============================================================================


def match_category(name: str, keywords: list[str]) -> bool:
    """body 名形如 pillow_<hash>_1_0_2，用子串匹配最省事。"""
    lowered = name.lower()
    return any(kw in lowered for kw in keywords)


def list_bodies(spec: mujoco.MjSpec) -> list[mujoco.MjsBody]:
    """按顺序收集所有顶层 body（先收集再操作，避免破坏迭代器）。"""
    bodies = []
    body = spec.worldbody.first_body()
    while body is not None:
        bodies.append(body)
        body = spec.worldbody.next_body(body)
    return bodies


def delete_objects(spec: mujoco.MjSpec, keywords: list[str]) -> list[str]:
    """删除匹配关键词的顶层 body，返回被删名字列表。"""
    targets = [b for b in list_bodies(spec) if match_category(b.name, keywords)]
    names = [b.name for b in targets]
    for b in targets:
        spec.delete(b)
    return names


def move_objects_up(spec: mujoco.MjSpec, offset_map: dict[str, float]) -> list[str]:
    """按 category 把物体沿 z 轴抬高一点，避免初始嵌进桌面。"""
    moved = []
    for body in list_bodies(spec):
        for keyword, dz in offset_map.items():
            if keyword in body.name.lower():
                body.pos[2] = body.pos[2] + dz
                moved.append(body.name)
                break
    return moved


def add_target(spec: mujoco.MjSpec, pos: list[float]) -> str:
    """把自定义 target 挂进场景，返回它在场景里的完整 body 名。

    attach_body 的源 body 必须保持引用，否则源 MjSpec 被回收后 body 悬空，
    会报 "child element is not a body or frame"。
    """
    template = mujoco.MjSpec.from_string(TARGET_XML)
    target_body = template.worldbody.bodies[0]

    frame = spec.worldbody.add_frame(pos=pos)
    frame.attach_body(target_body, "custom_", "")
    return f"custom_{TARGET_NAME}"


# ============================================================================
# 落地验证
# ============================================================================


def settle_check(model: mujoco.MjModel, body_name: str) -> tuple[np.ndarray, int]:
    """让 target 自由落体到静止，返回 (最终位置, 实际步数)。

    判据用「速度」而不是「单步位移」：刚释放时单步位移几乎为 0，
    用位移判据会立刻误判成已静止（本仓库实测踩过）。
    速度则要求连续多步都低于阈值，才能真正代表落稳。
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body_id = model.body(body_name).id
    dof_adr = model.jnt_dofadr[model.body_jntadr[body_id]]

    still_streak = 0
    steps_used = 0
    for step in range(SETTLE_MAX_STEPS):
        mujoco.mj_step(model, data)
        steps_used = step + 1
        speed = float(np.linalg.norm(data.qvel[dof_adr : dof_adr + 6]))
        if speed < SETTLE_VEL_TOL:
            still_streak += 1
            if still_streak >= SETTLE_STILL_STREAK:
                break
        else:
            still_streak = 0

    return data.xpos[body_id].copy(), steps_used


# ============================================================================
# metadata 同步
# ============================================================================


def sync_metadata(src_meta_path: Path, out_meta_path: Path, deleted: list[str],
                  added_name: str) -> None:
    """同步 metadata：去掉被删物体，登记新物体。

    MolmoSpaces 的 TaskSampler 靠这份 metadata 找可抓取物体，
    不同步的话新物体搜不到、被删物体还留着幽灵条目。
    """
    meta = json.loads(src_meta_path.read_text())
    objects = meta["objects"]

    for name in deleted:
        objects.pop(name, None)

    objects[added_name] = {
        "hash_name": added_name,
        "asset_id": TARGET_NAME,
        "object_id": TARGET_NAME,
        "category": "Cup",
        "object_enum": "custom",
        "is_static": False,
        "name_map": {
            "bodies": {added_name: TARGET_NAME},
            "joints": {f"{added_name}_free": f"{TARGET_NAME}_free"},
            "sites": {},
        },
        "mjcf_path": "",
        "parent": None,
        "room_id": None,
        "children": [],
    }

    out_meta_path.write_text(json.dumps(meta, indent=2))


# ============================================================================
# 主流程
# ============================================================================


def main() -> None:
    index_map = get_scenes(DATASET, SPLIT)[SPLIT]
    entry = index_map[HOUSE_INDEX]
    src_scene = Path(entry[VARIANT])
    src_meta = src_scene.parent / f"{src_scene.stem.replace('_' + VARIANT, '')}_metadata.json"
    print(f"[1] 源场景: {src_scene.name} (变体={VARIANT})")

    # assets 根目录 = .../assets/scenes/<数据集> 往上两级
    assets_root = src_scene.parent.parent.parent
    out_xml = setup_workdir(src_scene, assets_root)
    print(f"[2] 输出:   {out_xml}")

    spec = mujoco.MjSpec.from_file(str(src_scene))
    meta = json.loads(src_meta.read_text())
    static_names = [n for n, v in meta["objects"].items() if v["is_static"]]
    dynamic_names = [n for n, v in meta["objects"].items() if not v["is_static"]]
    print(f"[3] 加载成功: 顶层 body {len(spec.worldbody.bodies)} 个"
          f"（静态 {len(static_names)} / 动态 {len(dynamic_names)}）")

    ceilings = [b.name for b in list_bodies(spec) if b.name.startswith("ceiling_")]
    print(f"[3b] 天花板 body: {ceilings}（纯视觉，无碰撞）")

    deleted = delete_objects(spec, REMOVE_CATEGORY_KEYWORDS)
    print(f"[4] 删除 {len(deleted)} 个物体: {[n[:22] for n in deleted]}")

    moved = move_objects_up(spec, MOVE_CATEGORY_Z_OFFSET)
    print(f"[5] 抬高 {len(moved)} 个物体: {[n[:22] for n in moved]}")

    added = add_target(spec, TARGET_POS)
    print(f"[6] 加入 target: {added} @ {TARGET_POS}")

    model = spec.compile()
    print(f"[7] 编译通过: nbody={model.nbody}, ngeom={model.ngeom}")

    final_pos, used = settle_check(model, added)
    print(f"[7b] 落地仿真 {used} 步后位置: {np.round(final_pos, 3)}"
          f"（初始 z={TARGET_POS[2]}）")

    xml = spec.to_xml()
    xml = xml.replace(SRC_MESH_PREFIX, DST_MESH_PREFIX)
    xml = fix_gridlayout(xml)
    out_xml.write_text(xml)
    print(f"[8] 已保存 XML: {out_xml.stat().st_size / 1e6:.2f} MB")

    out_meta = out_xml.parent / f"{out_xml.stem}_metadata.json"
    sync_metadata(src_meta, out_meta, deleted, added)
    print(f"[9] 已同步 metadata: {out_meta.name} ({len(json.loads(out_meta.read_text())['objects'])} 个物体)")

    # 回读校验：确保产物真的能被 MuJoCo 重新解析
    reloaded = mujoco.MjSpec.from_file(str(out_xml)).compile()
    body_names = [reloaded.body(i).name for i in range(reloaded.nbody)]
    print(f"[10] 回读验证: nbody={reloaded.nbody}, ngeom={reloaded.ngeom}")
    print(f"     target 存在: {added in body_names}")
    print(f"     天花板保留: {sum(1 for n in body_names if n.startswith('ceiling_'))} 个")
    print(f"     删除生效:   {not any(match_category(n, REMOVE_CATEGORY_KEYWORDS) for n in body_names)}")
    print()
    print(f"查看场景: python -m mujoco.viewer --mjcf {out_xml}")


main()
