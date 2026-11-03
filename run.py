#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""启动地面站界面。

仓库里没有把包安装进环境时（``sys.flags.safe_path`` 生效、``python -m`` 找不到包），
用这个脚本启动最省事::

    python run.py                 # 正常启动
    python run.py --no-start      # 只开界面，不自动开始监测
    python run.py --shot a.png    # 无头截图，用于出图
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forestfire_gs.ui.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())