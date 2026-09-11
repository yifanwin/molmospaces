# 使用原生键盘遥操作

MolmoSpaces 自带键盘策略，无需手机或 SpaceMouse。运行 `teleop` 策略时，默认输入设备就是 `keyboard`。

## 1. 启动

先进入仓库并激活已安装 MuJoCo 依赖的环境：

```bash
cd molmospaces
source .venv/bin/activate
```

启动一个简单的 DROID/Franka 抓取任务：

```bash
python scripts/datagen/run_pipeline.py \
  --policy teleop \
  --viewer \
  --robot droid \
  --task_type pick \
  --scene_dataset ithor \
  --house_inds 1 \
  --samples_per_house 1
```

首次运行会自动准备所需资源。macOS 如需打开 MuJoCo 调试窗口，请按项目安装说明使用 `mjpython` 替代 `python`。

## 2. 按键

| 按键 | 动作 |
|---|---|
| `↑` / `↓` | 末端沿局部 **+X / -X** 移动 |
| `←` / `→` | 末端沿局部 **+Y / -Y** 移动 |
| `S` / `W` | 末端沿局部 **+Z / -Z** 移动 |
| `Z` / `C` | 绕 X 轴正向 / 反向旋转（roll） |
| `E` / `R` | 绕 Y 轴正向 / 反向旋转（pitch） |
| `A` / `D` | 绕 Z 轴正向 / 反向旋转（yaw） |
| `Space` | 切换夹爪开合 |
| `Q` | 结束当前 episode |

按住移动或旋转键可连续控制；可同时按平移键和旋转键。键盘监听是全局的，不要求终端窗口保持焦点。

## 3. 通过 X11 在远程服务器操作

X11 图像传输较慢，可先去掉 `--viewer`，只保留键盘策略自动打开的相机画面。若环境中曾强制使用无窗口渲染，运行前可执行：

```bash
unset MUJOCO_GL PYOPENGL_PLATFORM
```

```bash
python scripts/datagen/run_pipeline.py \
  --policy teleop \
  --robot droid \
  --task_type pick \
  --scene_dataset ithor \
  --house_inds 1 \
  --samples_per_house 1
```

> `ssh -Y` 给予远程程序更多本机 X Server 权限，只应对可信服务器使用。

## 4. 调整灵敏度

默认每个策略周期平移 **0.005 m**、旋转 **0.02 rad**。在
`molmo_spaces/configs/policy_configs_baselines.py` 的 `TeleopPolicyConfig` 中修改：

```python
step_size: float = 0.005
rot_step: float = 0.02
```

数值过大容易抖动或导致逆运动学失败，建议小幅调整。

## 常见问题

- **按键没有反应**：确认终端出现 `Keyboard policy started`。远程运行时还要确认 `DISPLAY` 非空，并将焦点放在转发出的相机窗口或终端上。
- **机械臂突然不动**：目标位姿可能暂时无逆运动学解，松键后向相反方向移动，再降低灵敏度。
- **`W` 同时改变调试窗口显示**：MuJoCo viewer 自带 `W` 线框快捷键；将焦点切到相机画面或终端后再操作。
- **想操作固定 benchmark**：使用 `TeleopPolicyEvalConfig` 运行 `molmo_spaces/evaluation/eval_main.py`，并通过 `--benchmark_dir` 指向 benchmark 目录。
