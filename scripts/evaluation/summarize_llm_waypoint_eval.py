"""Create a compact Markdown failure funnel from one or more eval run dirs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


RESULT_RE = re.compile(
    r"house (\d+) episode (\d+).*completed with success=(True|False)"
)


def summarize(run_dir: Path) -> dict:
    attempts = [
        json.loads(path.read_text())
        for path in sorted(run_dir.glob("llm_plans/*.json"))
    ]
    by_plan = {}
    for record in attempts:
        by_plan.setdefault(record.get("plan_id", "legacy"), []).append(record)
    valid = [record for record in attempts if record.get("valid")]
    valid_at_1 = sum(
        any(item.get("valid") and item.get("attempt") == 1 for item in records)
        for records in by_plan.values()
    )
    valid_at_3 = sum(
        any(item.get("valid") for item in records) for records in by_plan.values()
    )
    failures = Counter()
    for record in attempts:
        if record.get("valid"):
            continue
        error = str(record.get("error", "unknown"))
        if "HTTP" in error or "transport" in error or "Malformed LLM API" in error:
            failures["api"] += 1
        elif "ValidationError" in error or "invalid JSON" in error:
            failures["format"] += 1
        elif "IK failed" in error:
            failures["ik"] += 1
        elif "collision" in error.lower():
            failures["collision"] += 1
        else:
            failures["other_validation"] += 1
    episodes = []
    log_path = run_dir / "running_log.log"
    if log_path.exists():
        for line in log_path.read_text(errors="replace").splitlines():
            if match := RESULT_RE.search(line):
                episodes.append(
                    {
                        "house": int(match[1]),
                        "episode": int(match[2]),
                        "success": match[3] == "True",
                    }
                )
            if any(
                marker in line
                for marker in (
                    "Planner/IK rollout failed",
                    "Too many sequential IK failures",
                    "Object is not in grasp",
                    "Max retries",
                )
            ):
                failures["execution"] += 1
    return {
        "run": str(run_dir),
        "attempts": len(attempts),
        "valid_plans": len(valid),
        "plans": len(by_plan),
        "valid_plan_at_1": valid_at_1,
        "valid_plan_at_3": valid_at_3,
        "failures": failures,
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [summarize(path) for path in args.run_dirs]
    lines = [
        "# House 4 LLM waypoint evaluation",
        "",
        "| Run | API attempts | Valid plans | valid@1 | valid@3 | Task success |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        successes = sum(ep["success"] for ep in row["episodes"])
        total = len(row["episodes"])
        lines.append(
            f"| `{row['run']}` | {row['attempts']} | {row['valid_plans']} | "
            f"{row['valid_plan_at_1']}/{row['plans']} | "
            f"{row['valid_plan_at_3']}/{row['plans']} | {successes}/{total} |"
        )
    lines.extend(["", "## Failure funnel", ""])
    for row in rows:
        failures = (
            ", ".join(
                f"{key}={value}" for key, value in sorted(row["failures"].items())
            )
            or "none"
        )
        lines.append(f"- `{row['run']}`: {failures}")
    output = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
