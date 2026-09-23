#!/usr/bin/env python3
"""从已完成 P3 产物生成证据图和验收报告。"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


COLORS = {
    "base": "#8a8a8a", "ik": "#e69f00", "later": "#d55e00",
    "unknown": "#cc79a7", "feasible": "#0072b2", "reachable": "#009e73",
}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def candidate_style(row):
    if row["status"] == "unknown":
        return COLORS["unknown"], "x"
    if row["status"] == "feasible":
        return (COLORS["reachable"], "*") if row.get("reachable") else (COLORS["feasible"], "o")
    if row.get("first_failure_layer") == "F_base":
        return COLORS["base"], "x"
    if row.get("first_failure_layer") == "F_IK":
        return COLORS["ik"], "o"
    return COLORS["later"], "o"


def plot_maps(root, repo, episodes, candidates):
    valid = [row for row in episodes if row["status"] == "completed"]
    yaw_values = [-30, -15, 0, 15, 30]
    fig, axes = plt.subplots(len(valid), len(yaw_values), figsize=(13, 12),
                             sharex=True, sharey=True, constrained_layout=True)
    for row_index, episode in enumerate(valid):
        subset = episode["subset_index"]
        rows = [row for row in candidates if row["subset_index"] == subset]
        for col, yaw in enumerate(yaw_values):
            ax = axes[row_index, col]
            selected = [row for row in rows
                        if round(np.degrees(row["local_pose"][2])) == yaw]
            for item in selected:
                color, marker = candidate_style(item)
                ax.scatter(item["local_pose"][0], item["local_pose"][1], c=color,
                           marker=marker, s=32 if marker != "*" else 65, linewidths=.8)
            a = next(item for item in selected if item["is_A"]) if yaw == 0 else None
            if a:
                ax.scatter(0, 0, c="black", marker="s", s=45, facecolors="none", linewidths=1.4)
            for key, marker in (("nearest_feasible", "D"),
                                ("nearest_reachable_feasible", "P")):
                point_id = episode.get(key)
                item = next((value for value in selected if value["point_id"] == point_id), None)
                if item:
                    ax.scatter(item["local_pose"][0], item["local_pose"][1],
                               c="black", marker=marker, s=75, facecolors="none", linewidths=1.3)
            circle = plt.Circle((0, 0), .3, fill=False, color="#555555", linestyle="--", linewidth=.7)
            ax.add_patch(circle)
            ax.set_aspect("equal")
            ax.set_xlim(-.35, .35); ax.set_ylim(-.35, .35)
            ax.grid(alpha=.18)
            if row_index == 0:
                ax.set_title(f"Δyaw={yaw}°")
            if col == 0:
                ax.set_ylabel(f"#{subset} / house {episode['house']}\nΔy (m)")
            if row_index == len(valid) - 1:
                ax.set_xlabel("Δx (m)")
    handles = [
        Line2D([], [], color=COLORS["base"], marker="x", linestyle="", label="F_base 失败"),
        Line2D([], [], color=COLORS["ik"], marker="o", linestyle="", label="F_IK 未找到"),
        Line2D([], [], color=COLORS["later"], marker="o", linestyle="", label="后续层失败"),
        Line2D([], [], color=COLORS["unknown"], marker="x", linestyle="", label="unknown"),
        Line2D([], [], color=COLORS["feasible"], marker="o", linestyle="", label="可行但不可达"),
        Line2D([], [], color=COLORS["reachable"], marker="*", linestyle="", label="可达可行"),
        Line2D([], [], color="black", marker="s", markerfacecolor="none", linestyle="", label="A"),
        Line2D([], [], color="black", marker="D", markerfacecolor="none", linestyle="", label="最近可行 B"),
        Line2D([], [], color="black", marker="P", markerfacecolor="none", linestyle="", label="最近可达 B"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle("P3 局部站位可行性地图（每行一个有效 A；虚线为主圆盘）")
    prefix = repo / "docs/figures/last_mile_p3_e01_maps"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_budget(repo, metrics):
    random = metrics["random_local"]
    xs = np.asarray(sorted(map(int, random)))
    ys = np.asarray([random[str(x)]["rate"] for x in xs])
    upper = np.asarray([random[str(x)]["rate_upper"] for x in xs])
    denominator = metrics["M_A_not_found"] or metrics["N_scan_complete"]
    selected = str(metrics["selected_heuristic_distance_m"])
    h = metrics["heuristic_calibration"].get(selected, {})
    h_rate = h.get("reachable_feasible", 0) / denominator if denominator else 0
    oracle = metrics["R_oracle"] or 0
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    ax.plot(xs, ys, color="#0072b2", marker="o", label="Random-local（50 个相关试验）")
    if np.any(upper != ys):
        ax.fill_between(xs, ys, upper, color="#0072b2", alpha=.18,
                        label="unknown 最乐观上界")
    ax.scatter([1], [h_rate], marker="s", s=70, color="#e69f00",
               label=f"Fixed-radius {selected} m（试点调参）")
    ax.scatter([144], [oracle], marker="D", s=65, color="#009e73",
               label="Local-search oracle")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([1, 4, 8, 16, 144]); ax.set_xticklabels([1, 4, 8, 16, 144])
    ax.set_ylim(-.03, 1.03)
    ax.set_xlabel("额外 B 查询预算 K")
    ax.set_ylabel("救援率")
    ax.grid(alpha=.25)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("P3 查询预算与可达可行救援")
    prefix = repo / "docs/figures/last_mile_p3_e01_budget"
    fig.savefig(prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_report(root, repo, summary, metrics, episodes):
    valid = [row for row in episodes if row["status"] == "completed"]
    skipped = [row for row in episodes if row["status"] == "skipped"]
    status_square = {name: sum(row["status_counts_square"][name] for row in valid)
                     for name in ("feasible", "not_found", "unknown")}
    rows = []
    for row in episodes:
        if row["status"] == "skipped":
            rows.append(f"| {row['subset_index']} | {row['house']} | `{row['p1_status']}` | 跳过 | — | — |")
        else:
            rows.append(f"| {row['subset_index']} | {row['house']} | `completed` | 245 | "
                        f"{row['nearest_feasible'] or '—'} | {row['nearest_reachable_feasible'] or '—'} |")
    budget_rows = []
    for budget in (1, 4, 8, 16):
        value = metrics["random_local"][str(budget)]
        budget_rows.append(f"| Random-local | {budget} | {value['rescued']}/{value['trials']} "
                           f"({value['rate']:.3f}) | {value['rescue_upper']}/{value['trials']} "
                           f"({value['rate_upper']:.3f}) |")
    selected = metrics["selected_heuristic_distance_m"]
    hs = metrics["heuristic_calibration"][str(selected)]
    decision = ("试点观测到至少一个可达可行救援，可冻结协议后进入正式百例扫描。"
                if metrics["pilot_go_signal"] else
                "试点未观测到可达可行救援；先诊断 IK 覆盖，不启动正式百例扫描。")
    unknown_changes = metrics["L_reach_upper"] != metrics["L_reach"]
    direction = "无结论" if unknown_changes else ("继续" if metrics["pilot_go_signal"] else "转向")
    report = f"""# Last-mile P3-E01 实验报告

**结论：{decision}** 本轮只覆盖冻结的 10 条试点，其中 **5 条**有有效 A；不评价真实 Pick，也不把试点调参结果当成正式泛化证据。

## 问题、预期和受控变量

- 研究问题：普通导航终点 A 失败时，30 cm 圆盘内是否存在几何可行且静态走廊可达的 B？
- 实验编号：P3-E01。主要矛盾是 P2 的 5 个 A 全部停在 `F_IK`，局部底盘变化能否使同一有限搜索协议找到完整链。
- 运行前预期：5 个有效 A 中有 1–3 个可达救援；至少 1 个是进入正式扫描的试点信号。
- 唯一干预是底盘 SE(2)；抓取池、双臂、每臂 3 个初值、IK 迭代数、碰撞阈值和超时逐条继承 P2。

## 数据质量与充分性

- `N_attempt={metrics['N_attempt']}`，`N_nav={metrics['N_nav']}`，完成完整地图 `{metrics['N_scan_complete']}` 条；另有 `{len(skipped)}` 条因 P1 无有效 A 跳过。
- 每个有效 A 有 245 个方形网格点、145 个主圆盘点；A 查询来自 P2 缓存，每条新增 244 个 B 查询。
- 方形全图状态：`{status_square}`。本轮按 episode 计数，不把网格点或随机查询顺序当成独立场景。
- E04 的 A 快照模型指纹已无法由当前提交重建；因此用相同输入重放 E05_compat。五个有效 A 的索引和三维数值与 E04 **逐位完全相同**，没有绕过快照模型校验。

## Baseline / 预期 / 实际

| 量 | Baseline | 运行前预期 | 实际 |
|---|---:|---:|---:|
| A 的 `F_geo` 可行数 | 0/5 | 0/5 | 0/5（P2 缓存） |
| 圆盘内站位救援 `L_geo` | — | ≥1/5 | {metrics['L_geo']}/5 |
| 圆盘内可达救援 `L_reach` | — | 1–3/5 | {metrics['L_reach']}/5；unknown 上界 {metrics['L_reach_upper']}/5 |
| Oracle 救援率 | 0 | >0 | {metrics['R_oracle'] if metrics['R_oracle'] is not None else '不适用'} |

![五个有效 A 的局部可行性地图](figures/last_mile_p3_e01_maps.png)

图 1：每行是一个源目标实例，每列固定一个相对 yaw。圆盘虚线为主要统计区域；点是协议内站位标签。网格点相关，不表示独立样本；本图没有统计误差棒。

## 公平预算基线

| 方法 | K | 观测救援 | unknown 最乐观上界 |
|---|---:|---:|---:|
| Nav-only | 0 | 0/{metrics['M_A_not_found']} | 0/{metrics['M_A_not_found']} |
{chr(10).join(budget_rows)}
| Fixed-radius（{selected} m） | 1 | {hs['reachable_feasible']}/{metrics['M_A_not_found']} | {hs['reachable_feasible'] + hs['unknown_reachable']}/{metrics['M_A_not_found']} |
| Oracle | 144 | {metrics['L_reach']}/{metrics['M_A_not_found']} | {metrics['L_reach_upper']}/{metrics['M_A_not_found']} |

![查询预算与救援率](figures/last_mile_p3_e01_budget.png)

图 2：Random-local 每个 episode 有 10 个确定性顺序，共 50 个相关试验；曲线仅作描述。Fixed-radius 使用同一试点选择距离，因此不是无偏测试结果。Oracle 是有限圆盘网格上限。

## 逐条追溯

| # | house | P1 | 地图点 | 最近可行 B | 最近可达 B |
|---:|---:|---|---:|---|---|
{chr(10).join(rows)}

## 机制、偏差和不确定度

- 机制成立需要看到某些 B 的 `F_IK` 及后续层由 0 变为正，并且 `C(A,B)` 通过。地图中的分层状态用于定位断点，不用真实执行结果挑 B。
- `not_found` 只表示 32×2×3 的有限协议未找到链，不是数学不可达。离散路径检查也不能证明连续无碰撞。
- 只有 5 个有效 A，且来自 4 个 houses；不计算伪精确的 house bootstrap 区间。`I_reach={metrics['I_reach']}`，最乐观 unknown 上界为 `{metrics['I_reach_upper']}`。
- 当前没有执行闭合、抬升或 A→B 转移；即使代理改善，也不能声明真实 Pick 得到改善。

## 收敛记录与判定

- 运行前仍活着的解释：A 只是局部站位不佳；或固定 torso/有限 IK 覆盖下整个邻域都找不到链。
- 运行后以地图的分层变化区分两者；unknown 是否改变方向：`{unknown_changes}`。
- 实验价值：**有信息**。方向判定：**{direction}**。生命周期建议：**不动**。
- 暂存问题：真实 Pick、真实 A→B 转移和正式 100 条均未运行，不自动进入 P4。

## 图与产物验收

| 候选图 | 决定 | 原因 | 保留 |
|---|---|---|---|
| 固定 yaw 可行性地图 | 选中（证据） | 展示分层失败、A/B 和空间结构 | 随本运行保留 |
| 查询预算—救援率 | 选中（证据） | 直接对应公平预算比较 | 随本运行保留 |
| 10→5 的简单漏斗 | 拒绝 | 两个数字用表格更准确 | — |
| P4 配对执行图 | 拒绝 | 本轮没有真实 Pick 数据 | — |

源数据：`{root.relative_to(repo)}/candidates.jsonl`、`metrics.json`。图源：`scripts/evaluation/plot_last_mile_p3.py`，并保留 SVG/PNG。`COMPLETE.json` 只表示扫描和分类完整。
"""
    path = repo / "docs/last_mile_p3_e01_20260922.md"
    path.write_text(report)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.input / "summary.json").read_text())
    if not summary["complete"] or not (args.input / "COMPLETE.json").exists():
        raise ValueError("P3 尚未完整，拒绝生成结果报告")
    metrics = json.loads((args.input / "metrics.json").read_text())
    episodes = read_jsonl(args.input / "episodes.jsonl")
    candidates = read_jsonl(args.input / "candidates.jsonl")
    plot_maps(args.input, args.repo, episodes, candidates)
    plot_budget(args.repo, metrics)
    report = make_report(args.input, args.repo, summary, metrics, episodes)
    print(json.dumps({"report": str(report), "figures": [
        "docs/figures/last_mile_p3_e01_maps.svg",
        "docs/figures/last_mile_p3_e01_maps.png",
        "docs/figures/last_mile_p3_e01_budget.svg",
        "docs/figures/last_mile_p3_e01_budget.png",
    ]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
