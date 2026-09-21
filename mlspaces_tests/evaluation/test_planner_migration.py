"""评测归属迁移的轻量回归测试（不启动仿真、不调用 API）。"""
import ast
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/evaluation/run_house4_llm_waypoint_eval.sh"


@pytest.mark.parametrize("mode", ["baseline", "gate", "full", "mock"])
def test_launcher_uses_native_entrypoint(tmp_path, mode):
    capture = tmp_path / "calls"
    python = tmp_path / "python"
    python.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CAPTURE"\n')
    python.chmod(0o755)
    env = dict(os.environ, PYTHON_BIN=str(python), CAPTURE=str(capture),
               LLM_MOCK_RESPONSE_FILE=str(tmp_path / "response.json"),
               RBY_BENCH="/test/rby", PANDA_BENCH="/test/franka", OUT="/test/output")
    subprocess.run(["bash", str(SCRIPT), mode], env=env, cwd=tmp_path, check=True)
    calls = capture.read_text().splitlines()
    assert len(calls) == 2
    for call in calls:
        assert call.startswith("-m molmo_spaces.evaluation.eval_main ")
        assert "olmo.eval" not in call
        assert "--output_dir /test/output" in call
        assert "--task_horizon_steps 600" in call
        assert ("--idx 0" in call) == (mode in {"gate", "mock"})
    suffix = "Curobo" if mode == "baseline" else "LLMWaypoint"
    assert f":RBY1{suffix}PickPnPEvalConfig" in calls[0]
    assert f":PandaOmron{suffix}PickPnPEvalConfig" in calls[1]


def test_configs_have_no_molmobot_dependency():
    path = ROOT / "molmo_spaces/evaluation/configs/evaluation_configs.py"
    tree = ast.parse(path.read_text())
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert {"RBY1CuroboPickPnPEvalConfig", "RBY1LLMWaypointPickPnPEvalConfig"} <= names
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("olmo")
