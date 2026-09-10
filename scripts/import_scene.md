# MolmoSpaces Holodeck 数据下载与场景配置指南

## 1. 安装下载依赖

在当前仓库根目录执行：

```bash
cd /data0/wenyifan/IndoorGen/molmospaces

conda activate mlspaces

python -m pip install datasets zstandard huggingface_hub tqdm
```

如果还没有安装 MolmoSpaces：

```bash
python -m pip install -e ".[mujoco]"
```

如果使用 Hugging Face Token：

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

公开数据集通常不需要 Token，但大规模下载时建议配置，以避免限流。

---

## 2. 推荐的 Hugging Face 下载命令

下载带 occupancy map 的 Holodeck 多房间场景：

```bash
python scripts/assets/hf_download.py \
  /data0/wenyifan/IndoorGen/molmospaces_data \
  --data_source_dir \
  mujoco/scenes/holodeck-objaverse-train/20251217_with_occupancy \
  --versioned \
  --yes
```

该命令对应的数据源为：

```text
mujoco/scenes/holodeck-objaverse-train/20251217_with_occupancy
```

下载完成后，场景缓存结构预计类似：

```text
/data0/wenyifan/IndoorGen/molmospaces_data/
└── mujoco/
    ├── scenes/
    │   └── holodeck-objaverse-train/
    │       └── 20251217_with_occupancy/
    └── mjthor_data_type_to_source_to_versions.json
```

建议保留 `--versioned`，因为这种目录结构可以直接作为 MolmoSpaces 的资源缓存目录使用。

### 查看下载结果

```bash
find /data0/wenyifan/IndoorGen/molmospaces_data/mujoco/scenes \
  -type f -name "*.xml" | sort | head -20
```

### 查看场景数量

```bash
find /data0/wenyifan/IndoorGen/molmospaces_data/mujoco/scenes \
  -type f -name "*.xml" | wc -l
```

> **重要：** 这个命令会下载整个 Holodeck 版本。

`20251217_with_occupancy` 不是单个场景，而是一个完整数据源版本，可能包含大量房屋场景，所需磁盘空间和下载时间会比较大。

如果只是想先验证流程，不建议一开始下载完整的 Holodeck 数据。可以先用仓库自带的小场景：

```bash
python -m mujoco.viewer \
  --mjcf examples/custom_assets/scene.xml
```

确认 MuJoCo 和 viewer 正常后，再下载 Holodeck。

---

## 3. 如果需要机器人和可操作物体

只查看静态多房间场景时，首先可以只下载 scene source。

如果要在 MolmoSpaces 中插入机器人并运行 manipulation task，还需要对应依赖。

### Franka DROID

```bash
python scripts/assets/hf_download.py \
  /data0/wenyifan/IndoorGen/molmospaces_data \
  --data_source_dir mujoco/robots/franka_droid/20260127 \
  --versioned \
  --yes
```

### Objaverse 物体

Holodeck 使用 Objaverse 物体时，还建议下载：

```bash
python scripts/assets/hf_download.py \
  /data0/wenyifan/IndoorGen/molmospaces_data \
  --data_source_dir mujoco/objects/objaverse/20260131 \
  --versioned \
  --yes
```

### Objaverse / Objathor Metadata

如果需要物体 metadata：

```bash
python scripts/assets/hf_download.py \
  /data0/wenyifan/IndoorGen/molmospaces_data \
  --data_source_dir mujoco/objects/objathor_metadata/20260129 \
  --versioned \
  --yes
```

### 抓取任务资源

如果需要抓取任务：

```bash
python scripts/assets/hf_download.py \
  /data0/wenyifan/IndoorGen/molmospaces_data \
  --data_source_dir mujoco/grasps/droid_objaverse/20251218 \
  --versioned \
  --yes
```

### 功能和资源对应关系

| 目标              | 需要的资源                             |
| --------------- | --------------------------------- |
| 只查看多房间场景        | `scenes/holodeck-objaverse-train` |
| 场景 + Franka 机器人 | 再加 `robots/franka_droid`          |
| Objaverse 物体交互  | 再加 `objects/objaverse`            |
| 物体语义 metadata   | 再加 `objects/objathor_metadata`    |
| 抓取/拾取任务         | 再加 `grasps/droid_objaverse`       |

### 使用 RBY1

如果更想使用 RBY1，可以把机器人资源替换成：

```bash
python scripts/assets/hf_download.py \
  /data0/wenyifan/IndoorGen/molmospaces_data \
  --data_source_dir mujoco/robots/rby1/20251224 \
  --versioned \
  --yes
```

但 RBY1 的部分规划任务还依赖 cuRobo 和 GPU，初次导入建议先用 Franka 或只做静态查看。

---

## 4. 配置资源目录

创建或修改项目中的 `setup_env.sh`：

```bash
export MLSPACES_CACHE_DIR=/data0/wenyifan/IndoorGen/molmospaces_data/cache
export MLSPACES_ASSETS_DIR=/data0/wenyifan/IndoorGen/molmospaces_data/assets
export PYTHONPATH=/data0/wenyifan/IndoorGen/molmospaces:${PYTHONPATH:-}

# 有显示器时可以先不设置这两个变量
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

加载环境：

```bash
source setup_env.sh
```

确认变量：

```bash
printf 'MLSPACES_CACHE_DIR=%s\n' "$MLSPACES_CACHE_DIR"
printf 'MLSPACES_ASSETS_DIR=%s\n' "$MLSPACES_ASSETS_DIR"
```

---

## 5. 让 MolmoSpaces 建立资源链接

创建资源版本固定文件：

```bash
cat > /tmp/molmospaces_holodeck_assets.json <<'JSON'
{
  "robots": {
    "franka_droid": "20260127"
  },
  "scenes": {
    "holodeck-objaverse-train": "20251217_with_occupancy"
  },
  "objects": {
    "objaverse": "20260131",
    "objathor_metadata": "20260129"
  },
  "grasps": {
    "droid_objaverse": "20251218"
  }
}
JSON
```

配置并初始化：

```bash
export MLSPACES_PINNED_ASSETS_FILE=/tmp/molmospaces_holodeck_assets.json
export MLSPACES_FORCE_INSTALL=True

python -m molmo_spaces.molmo_spaces_constants
```

检查 MolmoSpaces 资源目录：

```bash
find "$MLSPACES_ASSETS_DIR" \
  -maxdepth 4 \
  -type f -o -type l | sort | head -50
```

通常：

* `MLSPACES_CACHE_DIR`：保存实际下载和解压的版本化资源。
* `MLSPACES_ASSETS_DIR`：保存 MolmoSpaces 使用的链接目录。

---

## 6. 获取一个具体场景

先列出 Holodeck 场景：

```bash
python - <<'PY'
from molmo_spaces.molmo_spaces_constants import get_scenes

scenes = get_scenes(
    "holodeck-objaverse",
    split="train",
)["train"]

available_scenes = {
    house_index: variants
    for house_index, variants in scenes.items()
    if variants is not None and variants.get("base") is not None
}

print("可用场景数量:", len(available_scenes))
for house_index, variants in list(available_scenes.items())[:10]:
    print(
        "房屋索引:", house_index,
        "XML:", variants["base"],
        "occupancy map:", variants["map"],
    )
PY
```

选择一个场景并安装它依赖的对象和抓取资源：

```bash
python - <<'PY'
from molmo_spaces.molmo_spaces_constants import get_scenes
from molmo_spaces.utils.lazy_loading_utils import (
    install_scene_with_objects_and_grasps_from_path,
)

HOUSE_INDEX = 0

scenes = get_scenes(
    "holodeck-objaverse",
    split="train",
)["train"]

scene = scenes[HOUSE_INDEX]
scene_path = scene["base"]

print("房屋索引:", HOUSE_INDEX)
print("场景 XML:", scene_path)
print("occupancy map:", scene["map"])

installed = install_scene_with_objects_and_grasps_from_path(scene_path)

print("安装结果:", installed)
print("已安装场景:", scene_path)
PY
```

这里的 `HOUSE_INDEX = 0` 可以换成任意存在的房屋索引，例如：

```python
HOUSE_INDEX = 123
```

注意，`scenes` 是以房屋索引为 key 的字典，并不是列表；索引对应的值包含：

* `base`：标准场景 XML。
* `ceiling`：带天花板的场景 XML（如果存在）。
* `map`：occupancy map 图片（如果存在）。

建议先用 `0` 或 `1` 验证流程。

---

## 7. 用 MuJoCo Viewer 查看场景

先打印最终可用的 XML 路径：

```bash
find "$MLSPACES_ASSETS_DIR/scenes" \
  -type f -o -type l | sort | head -30
```

然后使用实际找到的 XML：

```bash
python -m mujoco.viewer \
  --mjcf /absolute/path/to/selected_scene.xml
```

也可以使用已有的 `examples/view_scene.py`，将：

```python
SCENE_PATH = Path(__file__).parent / "custom_assets" / "scene.xml"
```

改成：

```python
SCENE_PATH = Path(
    "/absolute/path/to/selected_scene.xml"
)
```

再运行：

```bash
python examples/view_scene.py
```

如果是无显示器服务器，可以使用离屏方式，不依赖 viewer：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

---

## 8. 离屏渲染三个固定视角

仓库根目录下的 `render_three_views.py` 会对同一场景输出三张图片：

* `top.png`：垂直俯视图。
* `oblique_45.png`：方位角为 45°、向下倾斜 45°。
* `oblique_225.png`：方位角为 225°、向下倾斜 45°，与前一个斜视角相对。

脚本顶部集中配置了场景路径、输出目录、图像尺寸、相机距离和三个视角。默认场景为：

```text
/data0/wenyifan/IndoorGen/molmospaces_data/assets/scenes/holodeck-objaverse-train/train_0.xml
```

使用项目的 UV 环境运行。在有图形界面的机器上执行：

```bash
uv run --no-sync python render_three_views.py
```

在无显示器服务器上，使用 EGL 离屏渲染：

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  uv run --no-sync python render_three_views.py
```

这里使用 `--no-sync` 直接复用项目现有的 `.venv`，避免运行渲染脚本时重新解析或更新依赖。

图片默认保存在：

```text
rendered_views/
├── top.png
├── oblique_45.png
└── oblique_225.png
```

默认的 `CAMERA_DISTANCE_SCALE = 1.5` 会让三个视角更贴近场景。若还需拉近可继续减小该值；如果场景被裁切，则可增大到 `2.0`、`2.5` 或 `3.0`。如果需要渲染其他场景，修改 `MJCF_PATH` 即可。

`mujoco.viewer` 用于交互式查看，不能通过现有命令行参数一次导出这三个固定视角；该脚本使用 `mujoco.Renderer` 完成离屏渲染。

---

## 9. 接入 MolmoSpaces 的场景配置

官方场景不应设置为：

```python
scene_dataset = "user"
```

`user` 只适用于自定义 XML 场景。

官方 Holodeck 场景应使用类似配置：

```python
scene_dataset = "holodeck-objaverse"
```

并在 task sampler 中指定房屋索引：

```python
task_sampler_config = PickTaskSamplerConfig(
    dataset_name="holodeck-objaverse",
    house_inds=[0],
    house_variant="base",
    samples_per_house=1,
)
```

调试阶段建议：

```python
num_workers = 1
task_batch_size = 1
samples_per_house = 1
house_inds = [0]
```

先确保一个场景、一个 episode 能够成功创建，再增加房屋数量和 worker。

---

## 10. 推荐执行顺序

1. 用仓库的 `examples/custom_assets/scene.xml` 验证 MuJoCo 安装。
2. 下载 Holodeck scene source。
3. 检查是否生成 `.xml`、mesh、texture 和 occupancy 文件。
4. 用 `mujoco.viewer` 直接加载一个 XML。
5. 下载 Objaverse objects。
6. 使用 `get_scenes()` 获取场景列表。
7. 调用 `install_scene_with_objects_and_grasps_from_path()` 安装单个场景依赖。
8. 再加入 Franka DROID 或 RBY1。
9. 最后运行 MolmoSpaces 的 task sampler 和数据生成流程。

---

## 11. 最小第一步命令

首先执行：

```bash
cd /data0/wenyifan/IndoorGen/molmospaces

python scripts/assets/hf_download.py \
  /data0/wenyifan/IndoorGen/molmospaces_data \
  --data_source_dir \
  mujoco/scenes/holodeck-objaverse-train/20251217_with_occupancy \
  --versioned \
  --yes
```

该命令完成后，先不要急着运行机器人任务。

执行：

```bash
find /data0/wenyifan/IndoorGen/molmospaces_data/mujoco/scenes \
  -type f -name "*.xml" | sort | head
```

确认确实有 XML 场景文件之后，再继续配置资源链接和加载单个多房间场景。
