# -*- coding: utf-8 -*-
"""agent/core/paths.py —— 仓库根锚点。

只依赖 pathlib，**绝不能 import 本包内其他模块**（会被 core 下任何模块安全引用而不成环）。

约定：
  - 凡是从 ``__file__`` 推导的路径 → 一律锚到 ``REPO_ROOT``（模块搬进子包后，
    ``Path(__file__).parent`` 会静默指向新目录，不报错但数据落错地方）。
  - 凡是从 cwd 推导的路径（``Path("logs")`` / ``Path("./conversations")`` …）→ 保持原样，
    它们本来就要求「从仓库根运行」。
"""
from pathlib import Path

# agent/core/paths.py → agent/core → agent → 仓库根
REPO_ROOT = Path(__file__).resolve().parents[2]

assert (REPO_ROOT / "agent" / "__init__.py").exists(), f"REPO_ROOT 推断错误: {REPO_ROOT}"
