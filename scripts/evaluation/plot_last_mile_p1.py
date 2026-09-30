#!/usr/bin/env python3
"""Render the P1 navigation evidence figure and acceptance report."""

import argparse
import json
import math
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "completed": "#238b45", "start_unavailable": "#969696", "no_path": "#d95f0e",
    "timeout": "#756bb1", "collision": "#cb181d", "controller_error": "#8c2d04",
    "scene_changed": "#dd3497",
}

# 面板版式：≤10 条沿用试点版式（5 列、字体完整），更多条数切换为 10 列密排网格
PILOT_PANELS = 10
WIDE_COLS = 10
PILOT_COLS = 5
PILOT_CELL = (3.0, 3.25)   # 试点版式单元格尺寸（英寸）
WIDE_CELL = 2.4            # 密排版式单元格边长（英寸）
PILOT_DPI = 180
WIDE_DPI = 160
FORMAL_BATCH = 100  # 正式批冻结规模（P0 formal 清单）


def render(output: Path, report: Path, figure_prefix: Path, experiment: str = "P1",
           report_date: str | None = None):
    summary = json.loads((output / "summary.json").read_text())
    episodes = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    n = len(episodes)
    wide = n > PILOT_PANELS
    cols = WIDE_COLS if wide else min(n, PILOT_COLS)
    rows = math.ceil(n / cols)
    if wide:
        cell_w = cell_h = WIDE_CELL
        dpi = WIDE_DPI
    else:
        cell_w, cell_h = PILOT_CELL
        dpi = PILOT_DPI
    title_fs, tick_fs = (5, 4) if wide else (9, 7)
    fig, axes = plt.subplots(rows, cols, figsize=(cell_w * cols, cell_h * rows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    for idx, (ax, row) in enumerate(zip(axes, episodes)):
        path = output / "episodes" / f"{row['subset_index']:03d}" / "trajectory.npz"
        if path.exists():
            with np.load(path) as data:
                free = data["free_xy"]
                actual = data["actual_se2"]
                target = data["target_xy"]
                goal = data["planned_goal_xy"]
                endpoint = data["planned_endpoint"]
            if len(free):
                ax.scatter(free[:, 0], free[:, 1], s=.15, c="#d9d9d9", rasterized=True)
            if len(actual):
                ax.plot(actual[:, 0], actual[:, 1], color=COLORS[row["status"]], lw=1.8)
                ax.scatter(*actual[0, :2], marker="s", s=35, c="#2171b5", label="start")
                ax.scatter(*actual[-1, :2], marker="o", s=30, c=COLORS[row["status"]], label="actual")
            if np.isfinite(target).all():
                ax.scatter(*target, marker="*", s=70, c="#000000", label="target")
            if np.isfinite(goal).all():
                ax.scatter(*goal, marker="x", s=40, c="#6a51a3", label="goal")
            if np.isfinite(endpoint).all():
                ax.scatter(*endpoint[:2], marker="+", s=55, c="#e6550d", label="planned A")
        ax.set_title(f"#{row['subset_index']} H{row['house']} · {row['status']}", fontsize=title_fs)
        ax.set_aspect("equal", adjustable="datalim")
        if wide:
            # 密排网格只在左缘/下缘标注轴名，避免 100 个面板重复文字
            if idx % cols == 0:
                ax.set_ylabel("world y (m)", fontsize=6)
            if idx >= n - cols:
                ax.set_xlabel("world x (m)", fontsize=6)
        else:
            ax.set_xlabel("world x (m)", fontsize=8)
            ax.set_ylabel("world y (m)", fontsize=8)
        ax.tick_params(labelsize=tick_fs)
    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=8)
    subset_label = {"formal": "formal", "pilot": "frozen pilot"}.get(
        summary.get("subset", ""), "frozen pilot")
    fig.suptitle(f"{experiment} real A* navigation trajectories "
                 f"(one panel per {subset_label} episode, n={n})")
    svg = figure_prefix.with_suffix(".svg")
    png = figure_prefix.with_suffix(".png")
    fig.savefig(svg, dpi=dpi)
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    assert svg.exists() and png.exists()
    plt.imread(png)  # Rendering/readability smoke check.

    table = []
    for row in episodes:
        err = row.get("endpoint_se2_error")
        table.append(
            f"| {row['subset_index']} | {row['house']} | `{row['status']}` | "
            f"{row.get('policy_steps', 0)} | {'—' if err is None else f'{err:.4f}'} | "
            f"{row.get('termination_detail', '—')} |"
        )
    no_valid_a = ("本轮没有 `completed`，所以按协议没有生成 `A_snapshot.npz`。"
                  if summary["valid_A"] == 0 else "")
    rel_figure = Path("figures") / png.name
    today = report_date or date.today().isoformat()
    formal_simulation = summary.get("formal_simulation", "not_run")
    expected = summary.get("expected", n)
    subset_zh = "正式" if summary.get("subset") == "formal" else "试点"
    boundary_runtime = ("本轮没有运行 IK、抓取或局部搜索。" if formal_simulation == "completed"
                        else f"本轮没有运行 IK、抓取、局部搜索或正式 {FORMAL_BATCH} 条仿真。")
    conclusion = (
        f"冻结的 **{summary['attempted']}** 条{subset_zh}样本均已分类，得到 **{summary['valid_A']}** 个有效 nominal A；"
        f"P2 gate 为 **{summary['p2_gate']}**。这只验收真实导航和状态交接，不评价抓取或 last-mile 效果。"
    )

    text = f"""# Last-mile {experiment} 验收说明

**{today}｜P1 协议完成；未启动 P2。**

{conclusion}

## 结果

| 项目 | 实际结果 |
|---|---|
| 尝试/分类 | {summary['attempted']} / {summary['classified']} |
| 有效 A | {summary['valid_A']} |
| 终止分布 | `{json.dumps(summary['status_counts'], ensure_ascii=False, sort_keys=True)}` |
| P2 gate | `{summary['p2_gate']}`；本任务未启动 P2 |
| 正式批仿真 | 哈希与格式已审计；`formal_simulation={formal_simulation}` |

![{experiment} 真实导航轨迹]({rel_figure.as_posix()})

图中每个面板对应一条{subset_zh}样本。灰点是 0.30 m 安全半径下的可通行区域；线为控制器实际执行轨迹；方块、星号、叉号、加号分别表示远端起点、目标物体、导航采样目标和截断后的规划 A。颜色表示最终终止分类。观测单位为单个源目标实例；没有统计不确定度。

## 逐条审计

| # | house | 状态 | 策略步 | 终点 SE(2) 误差 | 详情 |
|---:|---:|---|---:|---:|---|
{chr(10).join(table)}

每个 `completed` 都通过终点误差、非法碰撞、目标物体位移、非底盘关节漂移和 A 快照恢复检查。{no_valid_a}失败不补样，也不从分母中删除。完整输入、轨迹、事件、快照、逐条结果和完成标记位于 `{output}`。

## 边界

- {boundary_runtime}
- A* 的 0.8 m 是路径截断半径，不代表实际目标距离恰为 0.8 m。
- `COMPLETE.json` 只表示 {expected} 条均有终止分类，不表示 {expected}/{expected} 导航成功。
- 当前资产环境仍是 P0 记录的本地版本；不声称复现其他资产版本。
"""
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text)
    assert rel_figure.name in report.read_text() and png.exists()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--figure-prefix", type=Path, required=True)
    parser.add_argument("--date", default=None, help="报告抬头日期；缺省用今天")
    parser.add_argument("--experiment", default=None,
                        help="报告标题用的实验名；缺省读 output/summary.json")
    args = parser.parse_args()
    experiment = args.experiment
    if experiment is None:
        summary_path = args.output / "summary.json"
        experiment = json.loads(summary_path.read_text())["experiment"]
    render(args.output, args.report, args.figure_prefix, experiment, args.date)


if __name__ == "__main__":
    main()
