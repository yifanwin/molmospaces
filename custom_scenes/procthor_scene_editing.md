# ProcTHOR 场景引入与编辑指南

承接上一份「最小 MuJoCo 场景」教程。那一阶段你手搓了一个 `地面 + 桌子 + target` 的裸 MJCF；
本阶段把它替换成 **MolmoSpaces 官方发布的 ProcTHOR 场景**，并在其上做编辑。

本文以 `procthor-10k-val` 的 **`val_4_ceiling.xml`** 为例，所有 API 与输出均在当前仓库实测通过。
配套脚本：[build_scene.py](reachability_procthor/build_scene.py)

---

## 0. 先厘清三层结构

上一阶段容易误以为「场景 = 一个 XML」。在 MolmoSpaces 里实际是三层：

```
molmospaces_data/assets/            <- ASSETS_DIR（本地，含软链接 + .lmdb）
├── objects/                        <- 物体资产库（thor / objaverse）
└── scenes/                         <- 场景数据集根目录（SCENES_ROOT）
    ├── procthor-10k-val/           <- 一个「数据集/切分」
    │   ├── val_4.xml               <- 单个房子：base 变体（无天花板）
    │   ├── val_4_ceiling.xml       <- 同房子：ceiling 变体（带天花板）
    │   ├── val_4_assets/           <- 该房子的 mesh（墙、房间轮廓、天花板）
    │   ├── val_4_metadata.json     <- 物体清单（谁可抓、父子关系…）
    │   ├── val_4_map.png           <- 俯视占用图
    │   └── val_4.json              <- 原始 ProcTHOR house JSON
    └── procthor-10k-train/
```

三个必须记住的点：

1. **场景 XML 不是自包含的**。它靠相对路径引用资源：
   - mesh → `<数据集目录>/val_4_assets/xxx.obj`
   - 物体贴图 → `../../objects/thor/...`
2. **`metadata.json` 是给 TaskSampler 用的**，不是给 MuJoCo 用的。MuJoCo 只读 XML；
   但你想让官方任务采样器「看见」自己的 target，就必须同步 metadata。
3. **XML 里物体的名字是 hash 名**（`pillow_c2e5b74c90bcaa6..._1_0_2`），不是 `pillow`。
   按类别找物体要子串匹配。

---

## 1. 环境：必须用 venv

这一步是硬性的。系统 python 和项目 venv 里的 MuJoCo 版本不同：

| 解释器 | MuJoCo |
|---|---|
| 系统 `python` | 3.11.0 |
| `molmospaces/.venv` | **3.5.0** ← 项目实际使用的 |

```bash
cd ~/wenyifan/MoMaTrajGen/molmospaces
source setup_env.sh          # 设置 MLSPACES_ASSETS_DIR / MLSPACES_CACHE_DIR
source .venv/bin/activate
python -c "import mujoco; print(mujoco.__version__)"   # 应为 3.5.0
```

`setup_env.sh` 里 `MLSPACES_ASSETS_DIR` 指向仓库内的 `molmospaces_data/assets`，
`MLSPACES_CACHE_DIR` 指向 NAS。不 source 它，`get_scenes_root()` 会去 `~/.cache` 找，报错。

> 路径说明：`/home/wenyifan/wenyifan` 是 `/data0/wenyifan` 的软链接，两者是同一份数据，
> 写哪个都能用。

---

## 2. 找到场景：`get_scenes()` 与变体选择

官方入口在 [molmo_spaces_constants.py:546](../molmo_spaces/molmo_spaces_constants.py#L546)：

```python
from molmo_spaces.molmo_spaces_constants import get_scenes

index_map = get_scenes("procthor-10k", "val")["val"]
entry = index_map[4]
# {"base":    ".../procthor-10k-val/val_4.xml",
#  "ceiling": ".../procthor-10k-val/val_4_ceiling.xml",
#  "map":     ".../procthor-10k-val/val_4_map.png"}
```

`dataset_name` 可取：`ithor`、`procthor-10k`、`procthor-objaverse`、`holodeck-objaverse`。

### 2.1 base vs ceiling：实测差异

同一房子有两个变体，**共享同一个 `val_4_assets/` 目录**。实测 `val_4` 的差异：

| | `val_4.xml` (base) | `val_4_ceiling.xml` (ceiling) |
|---|---|---|
| 行数 | 3313 | 3329 |
| 顶层 body | 102 | 102（多 4 个 `ceiling_*`） |
| 新增内容 | — | `ceiling_4/5/6/7`，z≈3.423 |

多出来的天花板长这样：

```xml
<body name="ceiling_4">
  <geom name="ceiling_4_visual_0" class="__VISUAL_MJT__"
        pos="0 0 3.42304" quat="0.707107 0.707107 0 0"
        type="mesh" mass="1e-08" material="KitchenTilesCheckerboard"
        mesh="ceiling_4"/>
</body>
```

注意它属于 `__VISUAL_MJT__` class（`contype=0 conaffinity=0 mass=1e-08`）——
**纯视觉几何，不参与碰撞**。所以：

- 物理仿真行为：两变体**完全一致**
- 视觉/光照：ceiling 更接近真实室内，适合出数据
- 俯视调试：base 更干净（没有东西挡在上面）

**怎么选**：

| 用途 | 建议 |
|---|---|
| 数据生成、策略学习、需要真实感 | `ceiling` |
| 调试图、手撸坐标、看俯视布局 | `base` |
| **可达性地图构建** | **两者皆可，结果完全相同**（见 7.1） |

### 2.2 ⚠️ 陷阱：返回的是全集，不是本地实有

`get_scenes()` 读的是**远程归档清单**，会为未下载的房子返回**并不存在的路径**。
实测 `procthor-10k-train`：

```
索引总数: 10000
本地真实存在: 22
```

批量生成前必须自己过滤：

```python
from pathlib import Path

index_map = get_scenes("procthor-10k", "val")["val"]
available = {i: v for i, v in index_map.items() if Path(v["base"]).exists()}
print(f"本地可用: {len(available)} 个")
```

---

## 3. 加载场景并读懂它

```python
import json
import mujoco
from pathlib import Path

scene_path = Path(get_scenes("procthor-10k", "val")["val"][4]["ceiling"])
spec = mujoco.MjSpec.from_file(str(scene_path))

# metadata 文件名去掉变体后缀
meta_path = scene_path.parent / f"{scene_path.stem.replace('_ceiling', '')}_metadata.json"
meta = json.loads(meta_path.read_text())

print(f"顶层 body: {len(spec.worldbody.bodies)}")
print(f"静态物体: {sum(1 for v in meta['objects'].values() if v['is_static'])}")
print(f"动态物体: {sum(1 for v in meta['objects'].values() if not v['is_static'])}")
```

实测输出（`val_4_ceiling`）：

```
顶层 body: 102
静态物体: 26     # 墙、门框、台面、餐桌、冰箱
动态物体: 36     # 抱枕、笔记本、鸡蛋… 这些才有 freejoint
```

val_4 是个较大的房子（`nbody=202, ngeom=1436`），含 5 个房间
（metadata 里 `room_id ∈ {1,4,5,6,7}`）。

**注意 `MjSpec.from_file()` 只接受 `str`，不接受 `Path`**（MuJoCo 3.5.0 的类型签名如此）。

### 遍历所有顶层 body

编辑场景的核心是这两个游标 API（官方写法见
[arena_utils.py:87](../molmo_spaces/env/arena/arena_utils.py#L87)）：

```python
def list_bodies(spec):
    bodies = []
    body = spec.worldbody.first_body()
    while body is not None:
        bodies.append(body)
        body = spec.worldbody.next_body(body)
    return bodies

for b in list_bodies(spec):
    print(b.name, b.pos)
```

**官方编辑范例**就在 `molmo_spaces/env/arena/arena_utils.py`，值得通读：
`fix_remove_objects_within_inner_sites` / `fix_move_objects_within_sites_up_abit` /
`fix_remove_all_toasters`。它们展示了删、移两类操作的标准写法。

---

## 4. 三类编辑操作

### 4.1 删除物体

```python
def delete_objects(spec, keywords):
    # 必须先收集再删除 —— 边遍历边删会破坏迭代器
    targets = [b for b in list_bodies(spec)
               if any(k in b.name.lower() for k in keywords)]
    names = [b.name for b in targets]
    for b in targets:
        spec.delete(b)
    return names

deleted = delete_objects(spec, ["pillow", "plunger", "garbagecan"])
```

`spec.delete()` 可删 body，也可删 geom / joint（见
[abstract.py:381](../molmo_spaces/robots/abstract.py#L381) 删 mesh geom 的用法）。

### 4.2 移动物体

`body.pos` 是可写数组，但要**整体赋值**，`body.pos[2] += x` 在部分版本上不生效：

```python
body.pos[2] = body.pos[2] + 0.02
```

用途：ProcTHOR 场景里物体常「嵌」在桌面里，抬高 2cm 可避免初始穿透。

### 4.3 添加自定义物体

这是你最需要的能力。官方在
[pick_task_sampler.py:402](../molmo_spaces/tasks/pick_task_sampler.py#L402) 的做法是
`MjSpec.from_file` 读物体 XML → `add_frame` → `attach_body`。我们用 `from_string` 省掉文件：

```python
TARGET_XML = """
<mujoco>
  <worldbody>
    <body name="target_cup">
      <freejoint name="target_cup_free"/>
      <geom name="target_cup_geom" type="cylinder" size="0.04 0.08"
            mass="0.2" rgba="0.9 0.2 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""

# 保持引用！源 spec 被回收会导致 body 悬空
template = mujoco.MjSpec.from_string(TARGET_XML)
target_body = template.worldbody.bodies[0]

frame = spec.worldbody.add_frame(pos=[3.49, 9.43, 0.95])
frame.attach_body(target_body, "custom_", "")     # prefix 会加在 body 名前
```

实测结果：新 body 名叫 **`custom_target_cup`**（prefix + 原名）。

> **坑**：写成 `spec.worldbody.add_frame(pos=...).attach_body(mujoco.MjSpec.from_string(T).worldbody.bodies[0], ...)`
> 会报 `child element is not a body or frame` —— 临时 `MjSpec` 被 GC，body 成了悬空指针。
> 务必先把 template 赋给变量。

`attach_body(obj, prefix, suffix)` 的第二、三参数是命名空间，官方用它做去重
（`f"pickupable_{i}_{j}/"`），批量加同类物体时很有用。

### 4.4 怎么知道往哪放：先查支撑面

别猜坐标，直接从场景里查台面/桌子的位置。val_4 的支撑面实测：

```
CounterTop   countertop_9609aa68...   pos=[0.94, 8.18, 0.47]
DiningTable  diningtable_42d3dec8...  pos=[7.90, 11.38, 0.41]
DiningTable  diningtable_f113cf7f...  pos=[3.49, 9.43, 0.38]   <- 本例选它
Fridge       refrigerator_93040c07... pos=[2.51, 7.32, 0.98]
```

```python
spec = mujoco.MjSpec.from_file(str(scene_path))
model = spec.compile()
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

for name, v in meta["objects"].items():
    if v["is_static"] and v["category"] in ("CounterTop", "DiningTable", "Fridge"):
        print(v["category"], name, data.xpos[model.body(name).id])
```

`mj_forward` 是必须的——不跑一次前向，`data.xpos` 全是 0。

---

## 5. 保存：三个真实的坑

**这一节是本文最重要的部分**，全部为实测踩出来的。

### 5.1 `to_xml()` 的 skybox gridlayout 序列化 bug

```python
spec.to_file("out.xml")               # 能写出去
mujoco.MjSpec.from_file("out.xml")    # ValueError: gridlayout length must match gridsize
```

原因：MuJoCo 内部把 `gridlayout` 存成定长数组（`mjMAXTEXLAYOUT = 12`），
`to_xml()` 会把**整个数组**写出来，而 `gridsize="2 4"` 只要求 8 个字符。

`val_4_ceiling.xml` 同样踩这个坑（skybox 叫 `SkyGarden`）。对比同一行：

```xml
<!-- 原始 -->
<texture type="skybox" name="SkyGarden" ... gridsize="2 4" gridlayout="LFRB.D.."/>
<!-- to_xml() 之后 -->
<texture type="skybox" name="SkyGarden" ... gridsize="2 4" gridlayout="LFRB.D......"/>
```

修复（按 `gridsize` 乘积截断）：

```python
import re

def fix_gridlayout(xml: str) -> str:
    pattern = r'gridsize="([0-9 ]+)" gridlayout="([^"]*)"'

    def truncate(m):
        grid_size = [int(v) for v in m.group(1).split()]
        return f'gridsize="{m.group(1)}" gridlayout="{m.group(2)[:grid_size[0] * grid_size[1]]}"'

    return re.sub(pattern, truncate, xml)
```

### 5.2 相对路径必须有正确的目录深度

场景 XML 里写死了相对路径：

```xml
<mesh name="room_4" file="val_4_assets/room_4.obj" scale="1 1 -1"/>
<texture name="..." file="../../objects/thor/Textures/KitchenTilesCheckerboard.png"/>
```

所以编辑后的 XML **不能随手丢到任意目录**。若放到 `custom_scenes/xxx/house_004.xml`：

- `val_4_assets/` → 找不到（报 `Error opening file 'val_4_assets/room_4.obj'`）
- `../../objects/` → 解析成 `custom_scenes/objects/`，同样找不到

**结论：输出目录必须复刻 `scenes/<数据集名>/` 这一层深度**，即 XML 位于
`.../scenes/procthor-10k-val/house_004.xml`，其上方两级必须存在 `objects/`。

### 5.3 自包含工作目录的搭法

用软链接复用官方资产，不复制几十 GB：

```python
WORK_DIR = Path("custom_scenes/reachability_procthor/work")
DATASET_DIR = WORK_DIR / "scenes" / "procthor-10k-val"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

(WORK_DIR / "objects").symlink_to(ASSETS_ROOT / "objects")        # 供 ../../objects/
(DATASET_DIR / "house_004_assets").symlink_to(SRC / "val_4_assets")  # 供 mesh
```

并把 XML 里的 mesh 前缀从 `val_4_assets/` 重写成 `house_004_assets/`：

```python
xml = spec.to_xml().replace("val_4_assets/", "house_004_assets/")
```

最终结构：

```
custom_scenes/reachability_procthor/work/
├── objects -> .../molmospaces_data/assets/objects
└── scenes/procthor-10k-val/
    ├── house_004.xml
    ├── house_004_metadata.json
    └── house_004_assets -> .../procthor-10k-val/val_4_assets
```

### 5.4 同步 metadata

删了物体、加了 target，metadata 必须同步，否则 TaskSampler 里会出现「幽灵物体」，
而你的 target 搜不到。格式参考
[examples/custom_assets/scene_metadata.json](../examples/custom_assets/scene_metadata.json)：

```python
meta["objects"][added_name] = {
    "hash_name": added_name,
    "asset_id": "target_cup",
    "object_id": "target_cup",
    "category": "Cup",
    "object_enum": "custom",
    "is_static": False,
    "name_map": {"bodies": {added_name: "target_cup"},
                 "joints": {f"{added_name}_free": "target_cup_free"},
                 "sites": {}},
    "mjcf_path": "",
    "parent": None,
    "room_id": None,
    "children": [],
}
```

---

## 6. 落地验证：别用位移判据

加完 target 要确认它初始高度合理——既别悬空太久，也别嵌进桌面。直观写法是「仿真到不动为止」，
但**判据如果选「单步位移」，会立刻误判**：

```python
# 错误写法：刚释放时单步位移几乎为 0，第一步就 break
prev = data.xpos[body_id].copy()
for _ in range(400):
    mujoco.mj_step(model, data)
    if np.linalg.norm(data.xpos[body_id] - prev) < 1e-3:
        break                      # ← 第一次迭代就命中，物体"看起来"没动
    prev = data.xpos[body_id].copy()
```

实测现象：跑了 400 步后位置仍是 `0.95`，与初值完全相同 —— 看着像"没下落"，
实际是判据提前退出了。真实验证：物体在正常自由落体，5 步（0.01s）降 0.0006 m，
正好等于 `0.5·g·t²`。

**正确做法：用速度判据 + 连续多步确认。**

```python
SETTLE_MAX_STEPS = 3000      # 0.002s/步 -> 最多 6s
SETTLE_VEL_TOL = 1e-2        # m/s
SETTLE_STILL_STREAK = 50     # 连续 50 步都低于阈值才算落稳

def settle_check(model, body_name):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body_id = model.body(body_name).id
    dof_adr = model.jnt_dofadr[model.body_jntadr[body_id]]

    still_streak = 0
    for step in range(SETTLE_MAX_STEPS):
        mujoco.mj_step(model, data)
        speed = float(np.linalg.norm(data.qvel[dof_adr : dof_adr + 6]))
        if speed < SETTLE_VEL_TOL:
            still_streak += 1
            if still_streak >= SETTLE_STILL_STREAK:
                return data.xpos[body_id].copy(), step + 1
        else:
            still_streak = 0
    return data.xpos[body_id].copy(), SETTLE_MAX_STEPS
```

---

## 7. 跑通：实测输出

```bash
cd ~/wenyifan/MoMaTrajGen/molmospaces
source setup_env.sh && source .venv/bin/activate
python custom_scenes/reachability_procthor/build_scene.py
```

```
[1] 源场景: val_4_ceiling.xml (变体=ceiling)
[2] 输出:   .../custom_scenes/reachability_procthor/work/scenes/procthor-10k-val/house_004.xml
[3] 加载成功: 顶层 body 102 个（静态 26 / 动态 36）
[3b] 天花板 body: ['ceiling_4', 'ceiling_5', 'ceiling_6', 'ceiling_7']（纯视觉，无碰撞）
[4] 删除 3 个物体: ['pillow_c2e5b74c90bcaa6', 'pillow_c2e5b74c90bcaa6', 'pillow_6afe3635cad7781']
[5] 抬高 3 个物体: ['egg_ab1e6b2ddd5b7baaf1', 'egg_f137dcdedb58eda5ef', 'apple_038a0ea9b393da66']
[6] 加入 target: custom_target_cup @ [3.49, 9.43, 0.95]
[7] 编译通过: nbody=197, ngeom=1427
[7b] 落地仿真 202 步后位置: [3.49 9.43 0.83]（初始 z=0.95）
[8] 已保存 XML: 0.65 MB
[9] 已同步 metadata: house_004_metadata.json (60 个物体)
[10] 回读验证: nbody=197, ngeom=1427
     target 存在: True
     天花板保留: 4 个
     删除生效:   True
```

**怎么读 [7b]**：target 从 z=0.95 落到 **0.83**，降了 12cm 后稳定。
圆柱半高 0.08，所以底面停在 **0.75** —— 正是餐桌面的高度，说明放置正确（落在桌上，没穿模）。

第 10 步的**回读验证不能省**。`to_xml()` 的坑（5.1）只有在重新解析时才会暴露，
写出去不报错不等于文件可用。

查看场景：

```bash
python -m mujoco.viewer --mjcf custom_scenes/reachability_procthor/work/scenes/procthor-10k-val/house_004.xml
```

> 远程服务器无 X11 时 viewer 起不来。可用 `MUJOCO_GL=egl` 离屏渲染出图验证；
> 本机 `DISPLAY=localhost:10.0`，X11 转发可用。

---

## 8. 用 `ProcTHORMap` 找 base 位置（直达你的 A/B 需求）

这是本阶段对你研究最有价值的一步。`ProcTHORMap` 能从场景导出**可通行区域**，
直接给出机器人 base 的候选位置（见 [scene_maps.py:304](../molmo_spaces/utils/scene_maps.py#L304)）。

### 8.1 ceiling 场景能用吗？—— 能，且与 base 完全等价

这是用 ceiling 变体时最该先问的问题：天花板会不会把俯视渲染盖住、弄坏占用图？
**不会。** 实测对比同一房子的两个变体：

```
base     occupancy=(1329, 1356)  可通行=37%  点数=660537  rooms={2:'room_4', 3:'room_5', 4:'room_6', 5:'room_7'}
ceiling  occupancy=(1329, 1356)  可通行=37%  点数=660537  rooms={2:'room_4', 3:'room_5', 4:'room_6', 5:'room_7'}
```

**逐位相同**。原因是 `from_mj_model_path` 内部有一步自动剥离天花板——
`collect_ceiling_geoms_recursively`（[scene_maps.py:509](../molmo_spaces/utils/scene_maps.py#L509)）
按名字递归收集所有含 `ceiling` 的 geom 并排除：

```python
def collect_ceiling_geoms_recursively(body_spec):
    for geom in body_spec.geoms:
        if geom.name and "ceiling" in geom.name.lower():
            ceiling_geoms.append(geom)
    for child in body_spec.bodies:
        collect_ceiling_geoms_recursively(child)
```

**推论**：只要你自定义的天花板 body/geom **名字里带 `ceiling`**，就会被同样处理。
反过来说，如果你给天花板起了别的名字，它会污染地图。

### 8.2 导出可通行点

```python
import numpy as np
import mujoco
from molmo_spaces.utils.scene_maps import ProcTHORMap

xml = "custom_scenes/reachability_procthor/work/scenes/procthor-10k-val/house_004.xml"
scene_map = ProcTHORMap.from_mj_model_path(xml)   # 必须传 str

free_points = scene_map.get_free_points()          # 直接是 (N, 3) 世界坐标
print(f"可通行点: {len(free_points)} 个, shape={free_points.shape}")

# target 的世界位置
spec = mujoco.MjSpec.from_file(xml)
model = spec.compile()
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
target_pos = data.xpos[model.body("custom_target_cup").id]
print(f"target 世界位置: {np.round(target_pos, 3)}")

# 距 target 最近的若干候选 base 位置 —— 这就是你的 B 点候选集
dist = np.linalg.norm(free_points[:, :2] - target_pos[:2], axis=1)
for i in np.argsort(dist)[:5]:
    print(f"   {np.round(free_points[i], 3)}   距离 {dist[i]:.2f} m")
```

实测输出：

```
可通行点: 660537 个, shape=(660537, 3)
target 世界位置: [3.49 9.43 0.95]
距 target 最近的 5 个候选 base 位置:
   [3.221 8.975 -0. ]   距离 0.53 m
   [3.762 8.975 -0. ]   距离 0.53 m
   [3.211 8.975 -0. ]   距离 0.53 m
   [3.772 8.975 -0. ]   距离 0.54 m
   [3.221 8.965 -0. ]   距离 0.54 m
```

> 注意 `target 世界位置` 读到的是 **0.95（初始高度）**，不是第 7 节落地后的 0.83。
> 这是对的：XML 里保存的就是初始位姿，`mj_forward` 只做前向计算、不推进物理。
> 要拿落地后的位置，得先 `mj_step` 若干步（见第 6 节）。

**注意 `get_free_points()` 返回的已经是世界坐标 (N,3)**，不要再喂给 `pos_px_to_m()`
（那个函数期望 `(N,2)` 的像素坐标，会抛 `AssertionError`）。
地图自带 `px_per_m = 100`，需要像素坐标时才用 `pos_m_to_px()`（它期望 `(N,3)`）。

### 8.3 ⚠️ `check_collision()` 的返回值是反的

这是上游 API 的一个命名陷阱，**实测证据**：

```
occupancy dtype=bool, True 占比=36.7%
get_free_points 数量: 660537  == occupancy True 的数量: 660537   # 完全相等
餐桌旁的候选点 [3.221, 8.975]   check_collision -> [ True]      # 这是可通行点
房子边界外  [50, 50]            check_collision -> [False]
墙外/地图外 [-5, -5]            check_collision -> [False]
```

读源码（[scene_maps.py:353](../molmo_spaces/utils/scene_maps.py#L353)）就能看清：

```python
ret[in_range_mask] = self.occupancy[pos_px[in_range_mask, 0], pos_px[in_range_mask, 1]]
# ret[~in_range_mask] = True      # ← 这行被注释掉了
return ret
```

`occupancy` 里 **`True` 表示自由空间**（`get_free_points` 就是 `argwhere(occupancy)`）。
所以：

| 返回值 | 含义 |
|---|---|
| `True` | 该点**可通行**（不是碰撞！） |
| `False` | 被占据 **或** 出界（两者无法区分） |

**用法建议**：把它当成 `is_free(pos)` 来读，别被名字骗了。若确实需要「出界也算碰撞」的语义，
得自己补上被注释掉的那行。

### 8.4 其他有用方法

| 方法 | 用途 |
|---|---|
| `get_free_points_by_room("room_4")` | 按房间取可通行点（val_4 实测 room_4 有 164706 个） |
| `save(path)` / `load(path)` | 缓存地图，避免每次重建（val_4 这种大房子重建较慢） |
| `room_ids_to_name` | 房间 id ↔ 名字，用于限定 base 在哪个房间 |
| `check_collision(pos)` | 见 8.3，实为「是否可通行」，注意语义 |

有了它，你的 **A → B 搜索**就从「手填坐标试」变成「在 `free_points` 上枚举 + 用
CuRobo/IK 判可达性」。

---

## 9. 接机器人（下一步的入口）

官方接口在 [abstract.py:391](../molmo_spaces/robots/abstract.py#L391)：

```python
robot_config.robot_cls.add_robot_to_scene(
    robot_config,
    spec,
    prefix="robot_0/",
    pos=[0.0, 0.0],              # base 的 (x, y)
    quat=[1.0, 0.0, 0.0, 0.0],   # 朝向；yaw 90° 即绕 z 转 90°
)
```

但更省事的是直接复用 `setup_robot_scene` 的完整流程
（[task_sampler.py:530](../molmo_spaces/tasks/task_sampler.py#L530)），它一次性处理了
「加载场景 → 加机器人 → 加载任务物体 → 删黑名单 body → 编译」。

---

## 10. 建议的目录组织

```
custom_scenes/
├── procthor_scene_editing.md          <- 本文
├── reachability_minimal/              <- 上一阶段：裸 MJCF
│   ├── scene.xml
│   └── build_scene.py
└── reachability_procthor/
    ├── build_scene.py                 <- 本阶段：ProcTHOR 编辑
    └── work/                          <- 生成物（自包含）
        ├── objects -> ...
        └── scenes/procthor-10k-val/
            ├── house_004.xml
            ├── house_004_metadata.json
            └── house_004_assets -> ...
```

批量生成时，把 `HOUSE_INDEX` 提到循环外，`OUT_STEM` 改成 `f"house_{i:03d}"`，
每个房子独立建一次 `house_XXX_assets` 软链即可。注意先按 2.2 过滤出本地实有的索引。

---

## 11. 更新后的最小闭环

```
① 选一个本地实有的 ProcTHOR 场景（推荐 ceiling 变体）
        ↓
② MjSpec 加载 + 读 metadata（确认动/静物体、天花板）
        ↓
③ 查支撑面位置 -> 编辑：删杂物 / 抬高物体 / attach 自定义 target
        ↓
④ to_xml + fix_gridlayout + 重写 mesh 路径 → 自包含目录
        ↓
⑤ 落地仿真确认放置合理 + 回读验证（两者都必须！）
        ↓
⑥ ProcTHORMap 导出可通行点 → 枚举 base 候选
        ↓
⑦ 加 RBY-1，用 IK/CuRobo 验证
   A：不可达      B：可达
```

先不要用 `housegen`（那是 Blender 重生成整条数据集的重型工具链，见
[housegen/README.md](../molmo_spaces/housegen/README.md)），
也先不要碰 `procthor-objaverse`（物体是 Objaverse 网格，更重）。
用 `val_4_ceiling` 这张现成的房子把 ①–⑦ 跑通，再扩大规模。

---

## 附：本阶段踩坑速查

| 现象 | 原因 | 解决 |
|---|---|---|
| `No module named 'compress_json'` | 用了系统 python | `source .venv/bin/activate` |
| `from_file(): incompatible function arguments` | 传了 `Path` | 传 `str(path)` |
| `gridlayout length must match gridsize` | `to_xml()` 定长数组 bug | `fix_gridlayout()` 按 gridsize 截断 |
| `Error opening file 'val_4_assets/...'` | 输出目录深度不对 | 复刻 `scenes/<数据集名>/` 两层 |
| `Error opening file '../../objects/...'` | 同上 | 上层两级要有 `objects/`（软链即可） |
| `child element is not a body or frame` | template MjSpec 被 GC | 先把 `from_string` 结果赋给变量 |
| `body_mocap` 不存在 | 属性名记错 | 用 `model.body_mocapid[bid]`（-1 表示非 mocap） |
| 物体「不下落」 | 用单步位移做静止判据 | 改用速度判据 + 连续多步确认 |
| `AssertionError` in `pos_px_to_m` | 重复转换坐标 | `get_free_points()` 已是世界坐标 |
| `check_collision()` 返回 `True` 却不是碰撞 | 上游命名反了 | 它实为 `is_free`，见 8.3 |
| 天花板污染占用图 | geom 名不含 `ceiling` | 名字里必须带 `ceiling`，才会被自动剥离 |
| `get_scenes` 返回上万但多数不可用 | 读的是远程归档清单 | 用 `Path(...).exists()` 过滤 |
