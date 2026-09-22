"""从指定的代码根目录运行 molmo_spaces 模块。

项目 venv 里的 ``__editable___molmo_spaces_*_finder`` 把 ``molmo_spaces`` 硬编码到
``pip install -e`` 时的源码目录。该 finder 挂在 ``sys.meta_path`` 上，**优先级高于
``sys.path``**，所以在 git worktree 里直接执行 ``python -m molmo_spaces...`` 仍然会
加载主工作区的代码——worktree 里的修改不生效，验证结果会指向错误的代码版本。

本脚本先摘掉这个 finder，再把 ``--source`` 指定的目录插到 ``sys.path`` 首位，并断言
实际加载的包确实来自该目录，然后运行目标模块。

用法::

    cd <装好 venv 的项目根目录>
    .venv/bin/python scripts/run_from_source.py \\
        --source /data0/wenyifan/MoMaTrajGen/.worktrees/panda-omron-e4 \\
        molmo_spaces.evaluation.eval_main -- --idx 116 --no_wandb

``--`` 之后的参数会原样传给目标模块。
"""

from __future__ import annotations

import argparse
import importlib.util
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="molmo_spaces 包所在的代码根目录（worktree 路径）",
    )
    parser.add_argument("module", help="要运行的模块，如 molmo_spaces.evaluation.eval_main")
    parser.add_argument(
        "module_args",
        nargs=argparse.REMAINDER,
        help="传给目标模块的参数（用 -- 分隔）",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    if not (source / "molmo_spaces" / "__init__.py").exists():
        raise SystemExit(f"{source} 下没有 molmo_spaces 包")

    sys.meta_path = [
        finder for finder in sys.meta_path if "EditableFinder" not in type(finder).__name__
    ]
    sys.path.insert(0, str(source))

    # 只查 spec 不导入包体，避免目标模块在 runpy 执行前先被 import 一次。
    spec = importlib.util.find_spec("molmo_spaces")
    origin = Path(spec.origin).resolve() if spec is not None and spec.origin else None
    if origin is None or not origin.is_relative_to(source):
        raise SystemExit(
            f"molmo_spaces 解析到 {origin}，不在 {source} 下；"
            "editable finder 可能换了实现，需要一并处理"
        )

    print(f"[run_from_source] molmo_spaces = {origin}", flush=True)

    module_args = args.module_args
    if module_args[:1] == ["--"]:
        module_args = module_args[1:]
    sys.argv = [args.module, *module_args]
    runpy.run_module(args.module, run_name="__main__")


if __name__ == "__main__":
    main()
