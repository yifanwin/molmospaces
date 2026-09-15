# MolmoSpaces 使用 uv 安装 RBY-1 环境简要说明

适用场景：Ubuntu + uv + MuJoCo + RBY-1，主要用于 RBY-1 仿真、遥操作和数据生成。


安装基础 MuJoCo 环境：

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[mujoco]"
```


## 3. cuRobo


机器已有 CUDA 12.8 时，可：

```bash
uv pip install \
    "torch~=2.7.0" \
    "torchvision>=0.22.0,<0.23.0" \
    --index-url https://download.pytorch.org/whl/cu128
```

再安装：
```bash
uv pip install ninja evdev setuptools wheel
uv pip install -e ".[mujoco,curobo]" --no-build-isolation
```


## 4. housegen 什么时候安装

`housegen` 用于室内场景生成和场景转换，不是普通 RBY-1 仿真的必需依赖。

以下情况需要安装：

* 从 iTHOR / ProcTHOR / Holodeck JSON 生成 MolmoSpaces 场景
* 自己进行程序化室内场景生成
* 修改、生成大量 house layout
* 研究场景生成 pipeline


```bash
uv pip install -e ".[mujoco,housegen]"
```


后续按需求增加：

| Extra      | 使用场景                      |
| ---------- | ------------------------- |
| `mujoco`   | 基础仿真、RBY-1 teleop、数据生成    |
| `curobo`   | RBY-1 自动运动规划              |
| `housegen` | 室内场景生成 / JSON 场景转换        |
| `grasp`    | grasp generation pipeline |
| `dev`      | 项目开发、lint、测试等开发工具         |

