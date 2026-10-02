"""只基于 P4 真实执行记录生成试点/正式证据图与报告。"""
import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False

from molmo_spaces.evaluation.last_mile.build import digest

# 状态 → (颜色, 简称)；默认落到「执行失败」。
STATUS_STYLE = {
    "success": ("#2b8c72", "成功"),
    "planner_no_witness": ("#718096", "无路径"),
    "planning_unknown": ("#718096", "规划未知"),
    "insufficient_lift_or_drop": ("#c95d3a", "未抬升"),
    "hold_too_short": ("#c95d3a", "保持过短"),
    "not_held_by_selected_gripper": ("#c95d3a", "未夹持"),
    "support_contact": ("#c95d3a", "仍在支撑"),
    "illegal_collision": ("#8c2d04", "非法碰撞"),
    "base_lock_failed": ("#8c2d04", "底盘漂移"),
    "arrival_error": ("#8c2d04", "到位超差"),
    "transfer_failed": ("#c95d3a", "转移失败"),
    "corridor_unavailable": ("#c95d3a", "走廊不可达"),
    "scene_changed": ("#8c2d04", "场景扰动"),
}
FALLBACK_HELD = ("#718096", "未知")
FALLBACK_FAILED = ("#c95d3a", "执行失败")


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _display_path(root: Path, repo: Path) -> str:
    """相对 repo 显示；路径不在 repo 内（如干跑）时回退为绝对路径。"""
    try:
        return str(root.resolve().relative_to(repo.resolve()))
    except ValueError:
        return str(root.resolve())


def _style(trial):
    status = trial.get("status", "unknown")
    if status in STATUS_STYLE:
        return STATUS_STYLE[status]
    return FALLBACK_FAILED if trial.get("Y_pick") == 0 else FALLBACK_HELD


def _paired_figure(pairs, figures, figure_name):
    if not pairs:
        return
    fig, ax = plt.subplots(figsize=(8, max(2.6, 0.34 * len(pairs) + 1)))
    for yi, row in enumerate(pairs):
        trial_rows = [row.get("A", {}), row.get("B", {}), row.get("B_after_transfer", {})]
        for xi, trial in enumerate(trial_rows):
            color, label = _style(trial)
            ax.scatter(xi, yi, s=620, marker="s", color=color, edgecolor="#222222", linewidth=0.8)
            ax.text(xi, yi, label, ha="center", va="center", color="white", fontsize=7.5)
    ax.set_xticks(range(3), ["A", "B 静态", "B 转移后"])
    ax.set_yticks(range(len(pairs)), [f"#{r['subset_index']} / house {r['house']}" for r in pairs])
    ax.set_xlim(-.5, 2.5)
    ax.set_ylim(-.6, len(pairs)-.4)
    ax.invert_yaxis()
    ax.set_title("冻结代理配对：真实 Pick 与实际转移")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(figures / f"{figure_name}.{suffix}", dpi=170)
    plt.close(fig)


def _paired_markdown(pairs):
    lines = ["| 索引 | house | A 状态 | B* | B 静态 | 转移状态 | B 转移后 |",
             "|---:|---:|---|---|---|---|---|"]
    for r in pairs:
        lines.append(
            f"| {r['subset_index']} | {r['house']} | {r.get('A', {}).get('status', r['status'])} "
            f"| {r.get('B_point_id') or '—'} | {r.get('B', {}).get('status', '—')} "
            f"| {r.get('transfer', {}).get('status', '—')} "
            f"| {r.get('B_after_transfer', {}).get('status', '—')} |")
    return lines


def _mechanism_lines(root, pairs, title):
    lines = []
    for row in pairs:
        trace = rows(root / "episodes" / f"{row['subset_index']:03d}" / "trace.jsonl")
        b = [item for item in trace if item.get("label") == "B"]
        close = [item for item in b if item.get("phase") == "close"]
        heights = [item["height_m"] for item in b if item.get("height_m") is not None]
        contact_steps = sum(len(item.get("finger_bodies", [])) >= 2 for item in b)
        max_lift = max(heights) - row["B"]["initial_target_z_m"] if heights else None
        close_distance = close[-1]["finger_distance_m"] if close else None
        close_text = f"{close_distance:.4f}" if close_distance is not None else "未闭合"
        lift_text = f"{max_lift:.4f}" if max_lift is not None else "未知"
        transfer = row.get("transfer", {})
        lines += ["", f"## {title} #{row['subset_index']}", "",
                  f"P3 预选 B*={row['B_point_id']}；B 同池同预算重新评价为 `{row.get('B_revalidation_status')}`。",
                  f"闭爪实际等待 {len(close)} 个 100 ms 步；末步指间距 {close_text} m；双指目标接触 {contact_steps} 个策略步；最大抬高 {lift_text} m。",
                  f"B 静态 Pick=`{row['B']['status']}`；转移到达误差 {transfer.get('position_error_m')} m / {transfer.get('yaw_error_deg')}°；实际到达位姿重评价 `{transfer.get('replan_status')}`，Pick=`{row['B_after_transfer']['status']}`。"]
    return lines


def _pilot_report(metrics, root, repo, data, pairs, figure_name):
    lines = [
        f"# Last-mile {metrics['experiment']} 试点执行报告", "",
        "**结论：试点的几何可行 B 已真实到达，但静态与转移后 Pick 均未成功；正式方向判定仍为证据不足。**", "",
        "## 试点口径", "",
        f"冻结样本 {metrics['N_attempt']} 条，有效 A {metrics['N_nav']} 条，完整确定邻域 {metrics['N_eval']} 条。",
        f"代理站位救援 L_geo={metrics['L_geo']}，可达救援 L_reach={metrics['L_reach']}；I_reach={metrics['I_reach']}，R_oracle={metrics['R_oracle']}。",
        f"unknown={metrics['N_unknown_A_or_neighborhood']}；按全部有效 A 的最保守/最乐观 I_reach 界限为 {metrics['I_reach_all_valid_lower']}–{metrics['I_reach_all_valid_upper']}。",
        "P3 的 B* 在看真实 Pick 前选定；A/B 从同一快照独立恢复。无可执行见证路径按执行前规划失败计 0。", "",
        "## 真实执行", "",
        f"A 成功 {metrics['A_pick_success']}/{metrics['A_pick_known']}；预选配对 {metrics['N_pair_preselected']}，已知配对 {metrics['N_pair_known']}。",
        f"配对四格 (A,B)：{metrics['pair_cells']}；配对差 {metrics['paired_difference']}；A 失败时 B 静态救援率 {metrics['real_pick_rescue_given_A_failure']}。",
        f"转移尝试 {metrics['N_transfer_attempted']}，到达 {metrics['N_transfer_arrived']}；转移后成功 {metrics['transfer_pick_success']}；转移相对静态 B 差 {metrics['transfer_minus_static_B']}。", "",
    ]
    if pairs:
        lines += [f"![配对真实 Pick](figures/{figure_name}.png)", "",
                  "图 1：每行是一个冻结的源目标实例；灰色表示执行前无路径，橙色表示真实执行未通过严格 Pick，绿色表示成功。无误差棒，试点不做统计推断。", ""]
    lines += _paired_markdown(data) + ["", "## 证据边界与下一步", "",
              "真实执行成功必须满足双指接触、目标离开支撑并抬高至少 5 cm、连续稳定 1 s；闭爪至少 7 个 100 ms 步。",
              "当前样本不计算 house 聚类区间，不套用正式 100 条继续/停止阈值；完整 Place、跨机器人复验均未运行。",
              "若 B 静态成功但转移后失败，诊断转移误差、场景扰动和实际到达位姿重规划；若 B 静态失败，仅支持几何代理改善。", "",
              "## 视觉检查和产物", "",
              "配对状态图选中：一眼区分 A、静态 B 与转移后 B；简单漏斗拒绝：计数用正文更准确。",
              f"原始日志：`{_display_path(root, repo)}/episodes/NNN/trace.jsonl`；源数据：`metrics.json`、`episodes.jsonl`。",
              "图源：`scripts/evaluation/plot_last_mile_p4.py`；图为 SVG/PNG，报告引用 PNG。", ""]
    lines += _mechanism_lines(root, pairs, "配对")
    lines += ["", "因果链：P3 的 F_geo(B)=1、静态走廊可达；P4 实际转移到达且在实际到达位姿重新评价仍可行；但真实夹爪没有形成持续双指持有，目标最大抬升远低于 5 cm。",
              "本轮未预登记真实 Pick 成功率的数值预期，因此不把 0/1 写成事后阈值验证；它只把断点定位在几何代理之后的夹持/抬升阶段。", "",
              "P1-E05_compat 的模型指纹目前无法在当前环境重建；本轮严格恢复 P1-E04，验证 A 数值与 P3 一致，并在该模型上对 B* 重新评价；没有直接执行跨模型旧 witness。",
              "判定器的逻辑正负控制已通过；本轮无真实阳性，物理真阳性灵敏度仍未校准。", ""]
    return lines


def _formal_report(metrics, root, repo, data, pairs, figure_name):
    decision = metrics.get("formal_decision", {})
    cells = metrics.get("pair_cells", {})
    transfer_known = cells["00"] + cells["01"] + cells["10"] + cells["11"]
    lines = [
        f"# Last-mile {metrics['experiment']} 正式批执行报告", "",
        f"**结论：{decision.get('direction_decision', '无结论')}**（{decision.get('reason', '未运行决策')}）", "",
        "## 口径与漏斗", "",
        f"冻结清单 {metrics['N_attempt']} 条，导航产生有效 A {metrics['N_nav']} 条，"
        f"具有确定 A 标签与完整确定邻域 N_eval={metrics['N_eval']} 条。",
        f"A 第一次几何评价 not_found M={metrics['M']}；其中存在可行 B 的 L_geo={metrics['L_geo']}，"
        f"存在可达可行 B 的 L_reach={metrics['L_reach']}。",
        f"I_geo={metrics['I_geo']}，I_reach={metrics['I_reach']}，R_oracle={metrics['R_oracle']}；"
        f"unknown={metrics['N_unknown_A_or_neighborhood']}，I_reach 的 "
        f"最保守/最乐观界限 {metrics['I_reach_all_valid_lower']}–{metrics['I_reach_all_valid_upper']}。",
        "", "## 执行层", "",
        f"A 侧得到确定 Pick 标签 {metrics['A_pick_known']}/{metrics['N_nav']}，成功 {metrics['A_pick_success']} 次；"
        f"A 侧全部有效 A 的成功率界限 {metrics['A_pick_all_valid_lower']}–{metrics['A_pick_all_valid_upper']}。",
        f"A 侧状态分布：{metrics['A_failure_stages']}。",
        f"预选配对 {metrics['N_pair_preselected']}，配对双方均有确定标签 {metrics['N_pair_known']}（"
        f"转移后另有确定标签 {transfer_known}）。",
        f"配对四格 (A,B)：{cells}；配对差 {metrics['paired_difference']}；"
        f"A 失败时 B 静态救援率 {metrics['real_pick_rescue_given_A_failure']}。",
        f"转移尝试 {metrics['N_transfer_attempted']}，到达 {metrics['N_transfer_arrived']}"
        f"（到达率 {metrics['transfer_arrival_rate']}）；转移后成功 {metrics['transfer_pick_success']}，"
        f"转移相对静态 B 差 {metrics['transfer_minus_static_B']}。", "",
    ]
    if pairs:
        lines += [f"![配对真实 Pick](figures/{figure_name}.png)", "",
                  "图 1：每行一个冻结源目标实例，三列分别是 A、静态 B 与转移后 B 的真实 Pick 结果；颜色含义见正文与图例文字。", ""]
    lines += _paired_markdown(data) + ["", "## 预登记决策", "",
              f"house 聚类 bootstrap：I_reach 区间 {decision.get('I_reach_house_ci95')}，"
              f"配对差区间 {decision.get('paired_difference_house_ci95')}，"
              f"出现救援的 house 数 {decision.get('rescue_houses')}（重采样 {decision.get('bootstrap_draws')} 次）。",
              f"严格 Pick 物理正负校准：{'通过' if decision.get('calibration_passed') else '缺失'}；"
              f"unknown 是否会改变方向判断：{decision.get('unknown_changes_continue')}。", "",
              "## 证据边界", "",
              "A 侧未出现真实夹持的原因必须在报告里与被评价器判定的几何可行性分开陈述："
              "A 处动作链是否可执行，取决于 P1 的导航截断协议与冻结 torso，不能读成场景本身不可操作。",
              "严格执行日志按 100 ms 记一行，非法接触与底盘漂移在每个 4 ms 步审计；"
              "无 witness 记作执行前规划失败，不等价于物理夹持失败。",
              f"原始日志：`{_display_path(root, repo)}/episodes/NNN/trace.jsonl`；源数据：`metrics.json`、`episodes.jsonl`。",
              "图源：`scripts/evaluation/plot_last_mile_p4.py`；图为 SVG/PNG，报告引用 PNG。", ""]
    lines += _mechanism_lines(root, pairs, "配对机制诊断")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--date", default="20260923",
                        help="报告文件名日期后缀；试点默认 20260923")
    args = parser.parse_args()
    root = args.input
    marker = json.loads((root / "COMPLETE.json").read_text())
    for name in ("inputs", "episodes", "metrics", "summary"):
        rel = f"{name}.jsonl" if name == "episodes" else f"{name}.json"
        if marker[f"{name}_sha256"] != digest(root / rel):
            raise ValueError(f"P4 {name} 校验失败")
    data = rows(root / "episodes.jsonl")
    metrics = json.loads((root / "metrics.json").read_text())
    valid = [r for r in data if r["p1_status"] == "completed"]
    pairs = [r for r in valid if r.get("B_point_id")]
    docs = args.repo / "docs/last_mile/p4"
    figures = docs / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    run_tag = metrics["experiment"].lower().replace("-", "_")
    # 同一子集的多次运行（如 2026-09-30 的 IK 修复重跑）会共用 run_tag，
    # 图和报告都必须带日期，否则后一次会静默覆盖前一次的证据。
    figure_name = f"last_mile_{run_tag}_{args.date}_paired_pick"
    _paired_figure(pairs, figures, figure_name)
    if metrics.get("subset") == "formal":
        lines = _formal_report(metrics, root, args.repo, data, pairs, figure_name)
    else:
        lines = _pilot_report(metrics, root, args.repo, data, pairs, figure_name)
    report = docs / f"last_mile_{run_tag}_{args.date}.md"
    report.write_text("\n".join(lines))
    print(report)


if __name__ == "__main__":
    main()
