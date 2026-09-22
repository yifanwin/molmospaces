"""绘制 P2 无资产正/负路径验收夹具（诊断图，不是 P1 场景结果）。"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), sharex=True, sharey=True)
    cases = [("已知可达：四层通过", None), ("中途碰撞：停在 F_approach", (.2, .55, .04))]
    for ax, (title, obstacle) in zip(axes, cases):
        start, pre = np.array([0, .8]), np.array([.4, .3])
        grasp, lift = np.array([.4, .4]), np.array([.4, .45])
        if obstacle:
            hit = np.asarray(obstacle[:2])
            ax.plot([start[0], hit[0]], [start[1], hit[1]], "-o", color="#0072B2", label="执行至碰撞")
            ax.plot([hit[0], pre[0], grasp[0], lift[0]], [hit[1], pre[1], grasp[1], lift[1]],
                    "--", color="#999999", label="未执行路径")
        else:
            ax.plot([start[0], pre[0]], [start[1], pre[1]], "-o", color="#0072B2", label="当前 TCP→pregrasp")
            ax.plot([pre[0], grasp[0], lift[0]], [pre[1], grasp[1], lift[1]], "-o", color="#009E73", label="approach/lift")
        ax.add_patch(Circle(grasp, .04, fill=False, color="#CC79A7", lw=2, label="目标物"))
        if obstacle:
            ax.add_patch(Circle(obstacle[:2], obstacle[2], color="#D55E00", alpha=.75, label="障碍物"))
        ax.scatter(*start, marker="s", s=55, color="black", zorder=4)
        ax.set(title=title, xlabel="x（m）", ylabel="z（m）", xlim=(-.05, .5), ylim=(.2, .9))
        ax.grid(alpha=.2)
        ax.set_aspect("equal")
    handles, labels = [], []
    for ax in axes:
        h, label = ax.get_legend_handles_labels()
        handles += h; labels += label
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(bottom=.23, wspace=.22)
    fig.suptitle("P2 路径与附着物体验收夹具（无资产回归测试）", y=.99)
    for suffix in (".svg", ".png"):
        fig.savefig(str(args.output_prefix) + suffix, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
