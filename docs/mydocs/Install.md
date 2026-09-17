# MolmoSpaces 使用 uv 安装 RBY-1 环境简要说明

适用场景：Ubuntu + uv + MuJoCo + RBY-1 / PandaOmron，主要用于仿真、遥操作和数据生成。

## ⚠️ 首要原则：不要用 `uv pip install` 装项目依赖

`uv sync` **默认是严格同步**（等价于没加 `--inexact`），会把 lockfile 之外的包**静默卸载**。
也就是说，`uv pip install` 装的东西，下一次 `uv sync` 就没了。

2026-09-17 实际踩坑：9/16 用 `pip install -e ../robosuite` 装好，9/17 跑了一次 `uv sync` 后被清理干净，
数据生成直接报 `ModuleNotFoundError: No module named 'robosuite'`。

/ 现象：`.venv/lib/python3.11/site-packages/` 里只剩下 `__editable__.molmo_spaces-*.pth`，其余全靠 uv 管理。

所以在本项目里：

| 想做的事 | 正确做法 | 错误做法 |
| --- | --- | --- |
| 装项目依赖 | `uv sync --extra xxx` | `uv pip install -e ".[xxx]"` |
| 加一个依赖 | 写进 pyproject.toml 再 `uv sync --extra xxx` | `uv pip install pkg` |
| 万不得已用 pip | 之后每次同步都要 `uv sync --inexact` | 直接 `uv sync` |

## 1. 基础环境

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv sync --extra mujoco
```

`uv sync` 会自动读取 pyproject.toml 的 `requires-python`、生成/复用 `.venv`，并且把项目本身以 editable 方式装好。

## 2. robosuite（Panda + Omron 机器人）

`robosuite` 是 **optional extra，默认不装**。凡是涉及 PandaOmron 的任务（如 `PandaOmronPickAndPlaceDataGenConfig` 数据生成）都必须装，
否则运行时报：

```
ImportError: PandaOmronRobot requires robosuite 1.5.x.
Install it with `pip install -e '.[mujoco,robosuite]'` or install the local checkout with `pip install -e ../robosuite`.
```

正确装法（一并带上 mujoco，避免 extra 被换掉）：

```bash
uv sync --extra mujoco --extra robosuite
```

会额外拉入 `robosuite==1.5.2`、`numba`、`llvmlite`、`mink`、`qpsolvers`、`quadprog`。

> 注意：必须带上当前生效的 `mujoco` extra。`uv sync` 不带某个 extra 会把它**移除**，
> 只写 `--extra robosuite` 会把 mujoco 相关依赖弄丢。本环境用的是 `mujoco`（mujoco 3.5.0），
> 不是 `mujoco-filament`（那会是 3.7.1），传错会把 mujoco 换掉。

> 启动时打印的 `[robosuite WARNING] No private macro file found` / `Could not import robosuite_models`
> 是无害的，不影响 PandaOmron 使用。

## 3. 运行前必须设置的环境变量

```bash
source setup_env.sh
```

该脚本设置：

* `MLSPACES_CACHE_DIR` / `MLSPACES_ASSETS_DIR`：资源与缓存路径（assets 必须留本地，内含指向 cache 的软链接和依赖 mmap 的 `.lmdb`）
* `MUJOCO_GL=egl` / `PYOPENGL_PLATFORM=egl`：无头渲染
* `MUJOCO_EGL_DEVICE_ID` / `CUDA_VISIBLE_DEVICES`：指定使用的 GPU

## 4. cuRobo

机器已有 CUDA 12.8 时，先装对应 wheel 的 torch：

```bash
uv pip install \
    "torch~=2.7.0" \
    "torchvision>=0.22.0,<0.23.0" \
    --index-url https://download.pytorch.org/whl/cu128
```

再装 cuRobo：

```bash
uv pip install ninja evdev setuptools wheel
uv pip install -e ".[mujoco,curobo]" --no-build-isolation
```

> ⚠️ 这两步是 `uv pip install`，属于「首要原则」里的例外情况。
> **之后再跑 `uv sync` 会把 torch 换回 PyPI 默认版并卸掉 curobo**。
> 如果确实需要再同步，加 `--inexact` 保留它们。

## 5. housegen 什么时候安装

`housegen` 用于室内场景生成和场景转换，不是普通仿真的必需依赖。以下情况需要安装：

* 从 iTHOR / ProcTHOR / Holodeck JSON 生成 MolmoSpaces 场景
* 自己进行程序化室内场景生成
* 修改、生成大量 house layout
* 研究场景生成 pipeline

```bash
uv sync --extra mujoco --extra housegen
```

## 6. Extra 速查

| Extra             | 使用场景                                       |
| ----------------- | ------------------------------------------ |
| `mujoco`          | 基础仿真、teleop、数据生成（mujoco 3.5.0）             |
| `robosuite`       | Panda + Omron 机器人，PandaOmron 任务**必需**      |
| `curobo`          | 自动运动规划                                     |
| `housegen`        | 室内场景生成 / JSON 场景转换                         |
| `grasp`           | grasp generation pipeline                  |
| `dev`             | 项目开发、lint、测试等开发工具                          |
| `mujoco-filament` | 替代渲染后端（mujoco 3.7.1），**与 `mujoco` 互斥，不可同时启用** |

> `uv sync --all-extras` 会失败：`mujoco` 与 `mujoco-filament` 已在 pyproject.toml 的
> `[tool.uv] conflicts` 中声明互斥。按需逐个 `--extra` 指定即可。
