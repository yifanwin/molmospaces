"""汇总 LLM waypoint 运行产物的视频时长与规划结果。

用途：闭环测试是否真的产生了"有时长的运动视频"。规划阶段就退出的 episode
只会录到 1 帧（planner-failure diagnostic frame），时长约 0.1 s；真正执行过的
episode 会有与步数匹配的帧数（fps = 1000 / policy_dt_ms = 10）。

用法：
    .venv/bin/python scripts/evaluation/summarize_llm_waypoint_scenes.py <run_dir> [...]
    .venv/bin/python scripts/evaluation/summarize_llm_waypoint_scenes.py <run_dir> --output report.md
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

# 视频文件名形如 episode_00000000_head_camera_batch_1_of_1.mp4
VIDEO_RE = re.compile(r"episode_(\d+)_(.+?)(?:_batch_\d+_of_\d+)?\.mp4$")
# running_log.log 里每个 episode 的判定结果
RESULT_RE = re.compile(r"house (\d+) episode (\d+).*completed with success=(True|False)")


def probe_video(path: Path) -> tuple[int, float]:
    """返回 (帧数, 时长秒)；ffprobe 读不到时以 -1 表示。"""
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames,duration",
            "-of", "default=nw=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    frames, duration = -1, -1.0
    for line in completed.stdout.splitlines():
        if line.startswith("nb_frames="):
            frames = int(line.split("=", 1)[1])
        elif line.startswith("duration="):
            duration = float(line.split("=", 1)[1])
    return frames, duration


def plan_summary(run_dir: Path) -> list[dict]:
    """按 plan_id 汇总一次规划的关键信息：尝试次数、是否通过校验、最终错误。"""
    attempts: dict[str, list[dict]] = {}
    for path in sorted((run_dir / "llm_plans").glob("*.json")):
        record = json.loads(path.read_text())
        attempts.setdefault(record.get("plan_id", "legacy"), []).append(record)
    summaries = []
    for plan_id, records in attempts.items():
        records.sort(key=lambda item: item.get("attempt", 0))
        accepted = next((item for item in records if item.get("valid")), None)
        summaries.append(
            {
                "plan_id": plan_id,
                "attempts": len(records),
                "accepted_at": accepted.get("attempt") if accepted else None,
                "pickup": records[0].get("pickup_object"),
                "receptacle": records[0].get("receptacle"),
                "last_error": records[-1].get("error"),
                "latency_s": round(float(records[-1].get("latency_s") or 0.0), 2),
            }
        )
    return summaries


def summarize(run_dir: Path) -> dict:
    videos = []
    for house_dir in sorted(run_dir.glob("house_*")):
        for video_path in sorted(house_dir.glob("episode_*.mp4")):
            match = VIDEO_RE.match(video_path.name)
            if not match:
                continue
            frames, duration = probe_video(video_path)
            videos.append(
                {
                    "house": house_dir.name,
                    "episode": int(match[1]),
                    "camera": match[2],
                    "frames": frames,
                    "duration_s": duration,
                    "size_kb": round(video_path.stat().st_size / 1024, 1),
                }
            )
    outcomes = []
    log_path = run_dir / "running_log.log"
    if log_path.exists():
        for line in log_path.read_text(errors="replace").splitlines():
            if match := RESULT_RE.search(line):
                outcomes.append(
                    {
                        "house": int(match[1]),
                        "episode": int(match[2]),
                        "success": match[3] == "True",
                    }
                )
    return {
        "run": str(run_dir),
        "videos": videos,
        "outcomes": outcomes,
        "plans": plan_summary(run_dir),
    }


def render(rows: list[dict]) -> str:
    lines = ["# LLM waypoint 闭环测试汇总", ""]
    for row in rows:
        lines.append(f"## `{row['run']}`")
        lines.append("")
        # 同一 episode 的多路相机帧数应一致，取每集最大值作为代表
        per_episode: dict[tuple[str, int], list[dict]] = {}
        for video in row["videos"]:
            per_episode.setdefault((video["house"], video["episode"]), []).append(video)
        success_map = {(item["house"], item["episode"]): item["success"] for item in row["outcomes"]}
        lines.append("| house | episode | 相机数 | 帧数 | 时长(s) | 任务判定 |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for (house, episode), videos in sorted(per_episode.items()):
            frames = max(video["frames"] for video in videos)
            duration = max(video["duration_s"] for video in videos)
            verdict = success_map.get((house, episode))
            verdict_text = "成功" if verdict else ("失败" if verdict is False else "未记录")
            lines.append(
                f"| {house} | {episode} | {len(videos)} | {frames} | {duration:.1f} | {verdict_text} |"
            )
        lines.append("")
        lines.append("| plan_id | 尝试 | 通过于第几次 | 目标物 | 最终错误 |")
        lines.append("|---|---:|---:|---|---|")
        for plan in row["plans"]:
            accepted = plan["accepted_at"] if plan["accepted_at"] else "未通过"
            error = (plan["last_error"] or "—").replace("|", "\\|")[:110]
            lines.append(
                f"| `{plan['plan_id'][-12:]}` | {plan['attempts']} | {accepted} | "
                f"{(plan['pickup'] or '?')[:34]} | {error} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = render([summarize(path) for path in args.run_dirs])
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"written: {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
