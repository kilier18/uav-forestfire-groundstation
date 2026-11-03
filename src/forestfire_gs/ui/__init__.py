# -*- coding: utf-8 -*-
"""PyQt5 地面站界面层（论文 3.6 节）。

拆分方式：

    theme.py      调色板 + 全局 QSS，界面风格集中在一处
    widgets.py    自绘控件（视频窗、地图、地平仪、状态机条、指标图）
    main_window.py  主窗口装配 + 引擎信号泵 + 交互逻辑
    app.py        入口：``python -m forestfire_gs.ui.app``
"""
from __future__ import annotations

__all__ = ["theme", "widgets", "main_window", "app"]