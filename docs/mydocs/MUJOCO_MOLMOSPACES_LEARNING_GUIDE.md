# MuJoCo 与 MolmoSpaces 场景导入及学习操作指南

> 目标：使用 MuJoCo 查看和运行一个场景，并逐步理解 MolmoSpaces 的环境、机器人、任务、策略和数据生成流程。
>
> 适用目录：
>
> - MolmoSpaces 源码：`/home/wenyifan/wenyifan/IndoorGen/molmospaces`
> - 当前已下载资源目录：`/home/wenyifan/wenyifan/IndoorGen/molmospaces_data`
>
> 本文按“先单独学习 MuJoCo，再接入 MolmoSpaces，最后运行任务和数据生成”的顺序组织。



## 1. 先确认当前资源状态

目前 `molmospaces_data` 中已经看到资源管理器的锁文件和版本索引：

```text
molmospaces_data/
├── assets/
│   ├── .lock
│   └── mjthor_data_type_to_source_to_versions.json
└── cache/
    ├── .lock
    └── mjthor_data_type_to_source_to_versions.json
```

这说明资源目录已经被初始化或访问过，但不等于完整场景、机器人和物体文件已经下载完成。继续操作前先检查：

```bash
find /home/wenyifan/wenyifan/IndoorGen/molmospaces_data \
  -type f \( -name '*.xml' -o -name '*.mjcf' -o -name '*.stl' -o -name '*.obj' -o -name '*.ply' \) \
  | sort | head -100

find /home/wenyifan/wenyifan/IndoorGen/molmospaces_data \
  -type d \( -name scenes -o -name robots -o -name objects -o -name grasps \) \
  | sort
```

如果没有输出 XML 或 MJCF 文件，说明还不能直接用该目录加载场景，需要先安装资源或使用仓库中的示例场景。

## 2. 创建 Python 环境并安装 MuJoCo

建议使用 Python 3.11。MuJoCo、PyTorch、JAX、Warp 和渲染依赖较多，不建议直接污染系统 Python。

```bash
cd /home/wenyifan/wenyifan/IndoorGen/molmospaces

conda create -n molmospaces python=3.11 -y
conda activate molmospaces

python -m pip install --upgrade pip
python -m pip install -e ".[mujoco]"
```

也可以使用 `uv`：

```bash
cd /home/wenyifan/wenyifan/IndoorGen/molmospaces
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[mujoco]"
```

检查安装：

```bash
python - <<'PY'
import mujoco
import numpy
print("MuJoCo:", mujoco.__version__)
print("NumPy:", numpy.__version__)
PY
```

如果只想先学习 MuJoCo，而暂时不安装 MolmoSpaces 的全部依赖，可以单独创建轻量环境：

```bash
conda create -n mujoco-basic python=3.11 -y
conda activate mujoco-basic
python -m pip install "mujoco~=3.5.0" numpy glfw
```

## 3. 配置 MolmoSpaces 资源目录

MolmoSpaces 使用 `MLSPACES_ASSETS_DIR` 作为可直接访问的资源目录，使用 `MLSPACES_CACHE_DIR` 作为缓存目录。当前建议先将已下载目录配置为缓存目录，并将 `assets` 子目录作为资源目录：

```bash
export MLSPACES_CACHE_DIR=/home/wenyifan/wenyifan/IndoorGen/molmospaces_data/cache
export MLSPACES_ASSETS_DIR=/home/wenyifan/wenyifan/IndoorGen/molmospaces_data/assets
export PYTHONPATH=/home/wenyifan/wenyifan/IndoorGen/molmospaces:$PYTHONPATH
```

检查环境变量：

```bash
printf 'MLSPACES_CACHE_DIR=%s\n' "$MLSPACES_CACHE_DIR"
printf 'MLSPACES_ASSETS_DIR=%s\n' "$MLSPACES_ASSETS_DIR"
```

建议将环境变量写入一个项目脚本，避免每次手动输入：

```bash
cat > /home/wenyifan/wenyifan/IndoorGen/molmospaces/setup_env.sh <<'SH'
#!/usr/bin/env bash
export MLSPACES_CACHE_DIR=/home/wenyifan/wenyifan/IndoorGen/molmospaces_data/cache
export MLSPACES_ASSETS_DIR=/home/wenyifan/wenyifan/IndoorGen/molmospaces_data/assets
export PYTHONPATH=/home/wenyifan/wenyifan/IndoorGen/molmospaces:${PYTHONPATH:-}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
SH

source /home/wenyifan/wenyifan/IndoorGen/molmospaces/setup_env.sh
```

说明：

- 有桌面显示器并使用交互 viewer 时，可以暂时不设置 `MUJOCO_GL=egl`；
- 无显示器的服务器建议使用 EGL；
- `MUJOCO_EGL_DEVICE_ID` 可以指定 GPU，例如 `export MUJOCO_EGL_DEVICE_ID=0`；
- 不要把 `molmospaces_data/assets` 和 `molmospaces_data/cache` 混为同一个逻辑目录，资源管理器通常使用缓存目录保存版本化资源，再在 assets 目录建立链接。

## 4. 第一阶段：单独学习 MuJoCo

### 4.1 使用仓库示例场景

仓库内提供了一个最小自定义 MJCF 场景：

```text
examples/custom_assets/scene.xml
examples/custom_assets/scene_metadata.json
examples/custom_assets/asset_library/red_block/red_block.xml
```

先检查引用文件是否存在：

```bash
cd /home/wenyifan/wenyifan/IndoorGen/molmospaces
find examples/custom_assets -maxdepth 4 -type f | sort
```

启动 MuJoCo viewer：

```bash
cd /home/wenyifan/wenyifan/IndoorGen/molmospaces/examples/custom_assets
python -m mujoco.viewer --mjcf scene.xml
```

如果 viewer 不支持当前命令形式，使用下面的 Python 脚本。

### 4.2 编写最小 MuJoCo 加载脚本

创建 `/home/wenyifan/wenyifan/IndoorGen/molmospaces/examples/view_scene.py`：

```python
from pathlib import Path

import mujoco
import mujoco.viewer


SCENE_PATH = Path(__file__).parent / "custom_assets" / "scene.xml"


model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
```

运行：

```bash
cd /home/wenyifan/wenyifan/IndoorGen/molmospaces
python examples/view_scene.py
```

这个阶段重点学习：

- `<option>`：仿真器积分器和 timestep；
- `<asset>`：mesh、texture、material、嵌套 model；
- `<worldbody>`：地面、灯光、body、geom；
- `<joint>` / `<freejoint>`：自由度和状态；
- `<actuator>`：控制输入；
- `MjModel`：静态模型；
- `MjData`：仿真运行时状态；
- `mj_step()`：推进一个或多个物理步；
- `qpos`、`qvel`、`ctrl`：位置、速度和控制量。

### 4.3 用 Python 验证模型信息

```bash
cd /home/wenyifan/wenyifan/IndoorGen/molmospaces
python - <<'PY'
from pathlib import Path
import mujoco

scene = Path("examples/custom_assets/scene.xml").resolve()
model = mujoco.MjModel.from_xml_path(str(scene))
data = mujoco.MjData(model)

print("scene:", scene)
print("nq:", model.nq)
print("nv:", model.nv)
print("nu:", model.nu)
print("nbody:", model.nbody)
print("ngeom:", model.ngeom)
print("ncam:", model.ncam)
print("initial qpos:", data.qpos)
PY
```

如果这里失败，优先解决 MJCF 路径、`meshdir`、`texturedir`、`<include>` 或 `<attach>` 的相对路径问题，不要立即进入 MolmoSpaces 调试。

## 5. 将某个场景导入 MuJoCo

### 5.1 场景必须是 MJCF/XML

MuJoCo 原生加载的是 MJCF/XML。常见文件结构如下：

```text
my_scene/
├── scene.xml
├── scene_metadata.json       # 可选，MolmoSpaces 更推荐提供
├── assets/
│   ├── meshes/
│   ├── textures/
│   └── materials/
└── objects/
```

先用 MuJoCo 原生 API验证场景：

```bash
python - <<'PY'
from pathlib import Path
import mujoco

scene = Path("/absolute/path/to/my_scene/scene.xml")
model = mujoco.MjModel.from_xml_path(str(scene))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
print("Loaded:", scene)
print("nq/nv/nu:", model.nq, model.nv, model.nu)
PY
```

然后查看：

```bash
python -m mujoco.viewer --mjcf /absolute/path/to/my_scene/scene.xml
```

### 5.2 常见 XML 路径问题

- `file="assets/table.obj"` 是相对于当前 XML 文件所在目录，而不是当前 shell 目录；
- `<include file="..."/>` 的相对路径同样需要从 XML 文件目录计算；
- Linux 大小写敏感，`Mesh.obj` 和 `mesh.obj` 不等价；
- mesh、texture 和 XML 必须一并保留；
- 如果场景 XML 中已经包含机器人，后续交给 MolmoSpaces 时可能造成重复插入；MolmoSpaces 自定义场景通常只放环境和物体，机器人由配置运行时插入；
- 一个场景可以先在 MuJoCo viewer 中加载成功，再接入 MolmoSpaces；不要跳过这个验证步骤。

## 6. 将用户场景接入 MolmoSpaces

MolmoSpaces 主要通过 `scene_dataset="user"` 和 `task_sampler_config.scene_xml_paths` 使用用户场景。最小配置逻辑如下：

```python
scene_dataset = "user"
task_sampler_config.scene_xml_paths = ["/absolute/path/to/my_scene/scene.xml"]
task_sampler_config.house_variant = "base"
```

对应的配置层级是：

```text
MlSpacesExpConfig
└── task_sampler_config
    ├── dataset_name = "user"
    ├── scene_xml_paths = [".../scene.xml"]
    ├── house_inds = None
    └── house_variant = "base"
```

### 6.1 先运行自定义场景，不做复杂任务

建议先复制示例脚本：

```bash
cd /home/wenyifan/wenyifan/IndoorGen/molmospaces
cp examples/custom_assets/datagen.py /tmp/molmospaces_custom_scene_datagen.py
```

然后修改脚本中的路径：

```python
scene_xml_paths=[
    "/absolute/path/to/my_scene/scene.xml",
],
```

如果场景中没有 `red_block`，不要直接使用示例里的 `BlockPickupTaskSampler`。示例 sampler 会查找：

```python
pickup_obj_name = "red_block"
```

此时应将任务配置中的目标对象名改成实际 MuJoCo body 名称，并为该物体提供适合机器人和任务的初始位姿、目标位姿及抓取信息。

运行配置：

```bash
cd /home/wenyifan/wenyifan/IndoorGen/molmospaces
python -m molmo_spaces.data_generation.main \
  /tmp/molmospaces_custom_scene_datagen.py:CustomAssetsDataGenConfig
```

### 6.2 自定义场景的关键限制

MolmoSpaces 自带的 `PickTaskSampler`、导航 sampler 和部分场景随机化逻辑是为官方室内场景设计的。自定义场景通常需要自己实现：

- 机器人初始位置；
- 目标物体选择；
- 目标物体和机器人位姿随机化；
- 任务目标位姿；
- 摄像机位置和可见性；
- 碰撞检查；
- 自定义物体 metadata；
- 抓取姿态（如果使用抓取规划器）。

官方教程中的核心做法是继承 `PickTaskSampler`，重写 `_sample_and_place_robot()`，然后注册 `scene_dataset="user"` 的实验配置。参考：

- `docs/tutorials/custom_assets.md`
- `examples/custom_assets/datagen.py`

## 7. 如果使用 MolmoSpaces 官方场景资源

如果目标是加载 iTHOR、ProcTHOR、Holodeck 等官方场景，推荐通过资源管理器安装，而不是手动复制单个 XML。

### 7.1 安装资源

```bash
cd /home/wenyifan/wenyifan/IndoorGen/molmospaces
source setup_env.sh
export MLSPACES_FORCE_INSTALL=True
python -m molmo_spaces.molmo_spaces_constants
```

该步骤会按照 `molmo_spaces/molmo_spaces_constants.py` 中的资源版本下载并组织 robots、scenes、objects、grasps 等资源。网络、磁盘和权限不足时，命令会失败或只留下不完整缓存。

### 7.2 安装某一个场景及其依赖

```bash
python - <<'PY'
from molmo_spaces.molmo_spaces_constants import get_scenes
from molmo_spaces.utils.lazy_loading_utils import install_scene_with_objects_and_grasps_from_path

scene_path = get_scenes("ithor", "train")["train"][1]
print("Installing:", scene_path)
install_scene_with_objects_and_grasps_from_path(scene_path)
PY
```

查看官方场景：

```bash
python -m mujoco.viewer --mjcf \
  "$MLSPACES_ASSETS_DIR/scenes/ithor/FloorPlan1_physics.xml"
```

实际场景文件名可能随资源版本变化。先执行：

```bash
find "$MLSPACES_ASSETS_DIR/scenes" -type f -name '*.xml' | sort | head -30
```

## 8. 从 MuJoCo 学习到 MolmoSpaces 的推荐顺序

### 第 1 步：MuJoCo XML

掌握：

1. worldbody、body、geom、joint；
2. freejoint 与 hinge/slide joint；
3. actuator 与 ctrl；
4. camera、light、texture、mesh；
5. timestep、solver、collision；
6. `MjModel` / `MjData` / `mj_step`；
7. viewer 和离线 rendering。

练习：修改 `examples/custom_assets/scene.xml` 的地面尺寸、方块位置、光源和 timestep，并观察结果。

### 第 2 步：MuJoCo Python API

练习读取和修改：

```python
model.nq
model.nv
model.nu
data.qpos
data.qvel
data.ctrl
```

练习调用：

```python
mujoco.mj_forward(model, data)
mujoco.mj_step(model, data)
mujoco.mj_resetData(model, data)
```

### 第 3 步：MolmoSpaces Robot

阅读：

- `molmo_spaces/robots/abstract.py`
- `molmo_spaces/robots/robot_views/abstract.py`
- `molmo_spaces/controllers/abstract.py`
- `molmo_spaces/controllers/joint_pos.py`
- `molmo_spaces/configs/robot_configs.py`

重点理解：

```text
Robot
└── RobotView
    └── MoveGroup
        ├── arm
        ├── gripper
        └── base
```

MolmoSpaces 的策略动作通常是：

```python
{
    "arm": arm_action,
    "gripper": gripper_action,
}
```

### 第 4 步：MolmoSpaces Env / Task

阅读：

- `molmo_spaces/env/env.py`
- `molmo_spaces/tasks/task.py`
- `molmo_spaces/tasks/task_sampler.py`
- `docs/concepts.md`

理解三层职责：

```text
TaskSampler：加载场景、创建环境、随机化任务
    ↓
Env：持有 MjModel、MjData、机器人、渲染器和对象
    ↓
Task：执行 reset/step、观测、奖励、成功判断和轨迹缓存
```

特别注意：`task.reset()` 不负责重新加载或恢复环境物理状态；场景状态通常由 sampler 在创建 task 之前准备。

### 第 5 步：MolmoSpaces Policy

阅读：

- `molmo_spaces/policy/base_policy.py`
- `molmo_spaces/policy/solvers/`
- `molmo_spaces/policy/learned_policy/`
- `molmo_spaces/planner/`

策略主要分为：

- planner-based policy：不需要训练 checkpoint，可用于理解任务执行链路；
- learned policy：加载 checkpoint 或连接远程推理服务；
- teleoperation policy：人工遥操作；
- dummy/random policy：调试用。

初学时建议先运行 planner-based pick/open/close 任务，再学习 Pi/CAP 等 learned policy。

### 第 6 步：数据生成

阅读：

- `molmo_spaces/data_generation/main.py`
- `molmo_spaces/data_generation/pipeline.py`
- `molmo_spaces/utils/save_utils.py`
- `docs/data_format.md`
- `docs/data_processing.md`

数据生成链路：

```text
Experiment Config
    → TaskSampler
    → Task.reset()
    → Policy.get_action_chunk()
    → Task.step_chunk()
    → judge_success()
    → HDF5 + MP4
    → 修复视频路径 / 有效性检查 / 统计量
```

## 9. 推荐的最小实验路线

### 实验 A：纯 MuJoCo 静态场景

```bash
cd /home/wenyifan/wenyifan/IndoorGen/molmospaces
conda activate molmospaces
python -m mujoco.viewer --mjcf examples/custom_assets/scene.xml
```

目标：确认 XML、mesh、texture 和 viewer 都正常。

### 实验 B：纯 MuJoCo Python 仿真

运行 `examples/view_scene.py`，在代码中增加：

```python
for _ in range(1000):
    mujoco.mj_step(model, data)
```

目标：理解模型和数据对象，以及仿真的时间推进。

### 实验 C：MolmoSpaces 自定义场景

使用 `examples/custom_assets/datagen.py`，先保留 `red_block` 和示例资产，确认官方自定义场景教程可运行。

目标：理解 user scene、robot runtime insertion、task sampler 和 policy。

### 实验 D：替换为自己的场景

按以下顺序替换：

1. 只替换地面和静态几何；
2. 确认 MuJoCo viewer 可加载；
3. 确认 MolmoSpaces 能创建 Env；
4. 添加机器人配置；
5. 添加一个可操作物体；
6. 实现自定义 sampler；
7. 最后再启用 planner、抓取和数据保存。

### 实验 E：官方室内场景

完成资源下载后，先用 `python -m mujoco.viewer` 查看官方 XML，再运行 MolmoSpaces 的数据生成配置，例如：

```bash
python -m molmo_spaces.data_generation.main FrankaPickOmniCamConfig
```

第一次运行建议设置单 worker、少量 episode，并关闭 viewer 以外的复杂功能。

## 10. 常见问题排查

### 10.1 找不到 XML 或 mesh

```bash
find /home/wenyifan/wenyifan/IndoorGen/molmospaces_data \
  -type f \( -name '*.xml' -o -name '*.obj' -o -name '*.stl' \) | head
```

检查 XML 中 `file=`、`meshdir`、`texturedir` 的相对路径。最稳妥的做法是保持原始目录结构，不要只复制 XML 文件。

### 10.2 `mjpython` 或 viewer 无法启动

Linux 先尝试：

```bash
python -m mujoco.viewer --mjcf /absolute/path/to/scene.xml
```

无显示器服务器使用 EGL：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python -m mujoco.viewer --mjcf /absolute/path/to/scene.xml
```

如果没有图形环境，改用离屏渲染脚本，不要依赖 passive viewer。

### 10.3 MolmoSpaces 找不到官方资源

检查：

```bash
printf '%s\n' "$MLSPACES_ASSETS_DIR"
find "$MLSPACES_ASSETS_DIR" -maxdepth 3 -type d | sort | head -50
```

并重新执行：

```bash
export MLSPACES_FORCE_INSTALL=True
python -m molmo_spaces.molmo_spaces_constants
```

### 10.4 自定义场景找不到目标物体

确认 MuJoCo body 名称：

```python
for body_id in range(model.nbody):
    print(body_id, model.body(body_id).name)
```

`pickup_obj_name`、metadata 中的 object id 和实际 body 名称必须保持一致或在 sampler 中明确映射。

### 10.5 自定义场景没有机器人

这是预期行为。MolmoSpaces 的自定义场景教程通常不把机器人写入场景 XML，而是在环境创建时通过 `robot_config` 插入机器人。不要为了让 viewer 看到机器人而盲目把机器人复制进 XML，否则可能造成重复插入。

### 10.6 多 batch 或多 worker 出错

初学和调试阶段使用：

```python
num_workers = 1
task_batch_size = 1
```

MolmoSpaces 的 batch API 存在，但文档明确提示 batch 大于 1 尚未充分测试，部分 task 仍要求 `n_batch == 1`。

### 10.7 资源版本不匹配

JSON benchmark 评测会在模块导入阶段检查资源版本。若资源版本和 benchmark 生成版本不一致，评测可能在真正运行前失败。需要使用匹配的资源版本，而不是随意替换 XML。

## 11. 建议的学习资料顺序

1. MuJoCo XML / MJCF 官方文档；
2. MuJoCo Python API：`MjModel`、`MjData`、viewer、rendering；
3. 仓库 `examples/custom_assets/scene.xml`；
4. `docs/tutorials/custom_assets.md`；
5. `docs/concepts.md`；
6. `molmo_spaces/env/env.py`；
7. `molmo_spaces/tasks/task.py`；
8. `molmo_spaces/robots/abstract.py` 和 `controllers/`；
9. `molmo_spaces/data_generation/main.py` / `pipeline.py`；
10. `docs/data_format.md` 和 `docs/data_processing.md`；
11. `molmo_spaces/evaluation/eval_main.py`。

## 12. 最终检查清单

在开始复杂任务前，确认以下项目：

- [ ] `python -c "import mujoco"` 成功；
- [ ] MuJoCo 能独立加载目标 XML；
- [ ] 所有 mesh、texture、include 文件路径正确；
- [ ] `MLSPACES_CACHE_DIR` 指向 `/home/wenyifan/wenyifan/IndoorGen/molmospaces_data/cache`；
- [ ] `MLSPACES_ASSETS_DIR` 指向 `/home/wenyifan/wenyifan/IndoorGen/molmospaces_data/assets`；
- [ ] `molmospaces_data/assets` 中确实存在目标 XML 或有效资源链接；
- [ ] `scene_dataset="user"` 时使用绝对 `scene_xml_paths`；
- [ ] 自定义场景的机器人插入策略已明确；
- [ ] 目标物体 body 名称与 task config 一致；
- [ ] `num_workers=1`、batch size 为 1 的最小流程已通过；
- [ ] 再逐步增加 planner、camera、抓取、并行和数据保存功能。

## 13. 关键文件索引

- MuJoCo 资源管理说明：`docs/assets.md`
- 自定义资产教程：`docs/tutorials/custom_assets.md`
- 自定义场景示例：`examples/custom_assets/scene.xml`
- 自定义数据生成示例：`examples/custom_assets/datagen.py`
- 环境抽象：`molmo_spaces/env/env.py`
- 任务抽象：`molmo_spaces/tasks/task.py`
- 机器人抽象：`molmo_spaces/robots/abstract.py`
- 数据生成入口：`molmo_spaces/data_generation/main.py`
- 并行 rollout：`molmo_spaces/data_generation/pipeline.py`
- 数据格式：`docs/data_format.md`
- 数据后处理：`docs/data_processing.md`

最重要的原则是：先让 MuJoCo 原生加载目标场景，再让 MolmoSpaces 创建环境，最后才接入机器人、任务、策略和数据生成。这样可以把 XML/资源问题、仿真器问题和 MolmoSpaces 业务逻辑问题分开定位。
