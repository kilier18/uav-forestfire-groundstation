# -*- coding: utf-8 -*-
"""界面配色与全局样式表。

风格取向：指挥中心暗色。底色接近夜间监视器，信息用高饱和但克制的强调色，
告警红/琥珀只在真正需要注意的地方出现，避免整屏都在喊。
"""
from __future__ import annotations

# ---- 基础色 ----
BG = "#070b10"          # 窗口底
PANEL = "#0c141d"       # 卡片面
PANEL_SOFT = "#0a1119"  # 卡片内嵌面
EDGE = "#1a2a39"        # 描边
EDGE_SOFT = "#132030"

TEXT = "#dce8f2"
TEXT_DIM = "#8ba3b8"
TEXT_FAINT = "#5b7285"

ACCENT = "#35e0a1"      # 主强调：运行中 / 正常
ACCENT_DIM = "#1d6f57"
INFO = "#4fc3f7"        # 遥测 / 链路
WARN = "#ffb648"        # 疑似
DANGER = "#ff5d5d"      # 火情告警
DANGER_DIM = "#7a2626"
IDLE = "#4a5c6b"

# ---- 字体 ----
FONT_UI = "Microsoft YaHei UI"
FONT_MONO = "Consolas"

QSS = f"""
* {{
    font-family: "{FONT_UI}", "Segoe UI", sans-serif;
    color: {TEXT};
}}
QMainWindow, QWidget#Root {{
    background: {BG};
}}
QFrame#Card {{
    background: {PANEL};
    border: 1px solid {EDGE};
    border-radius: 10px;
}}
QFrame#Inner {{
    background: {PANEL_SOFT};
    border: 1px solid {EDGE_SOFT};
    border-radius: 8px;
}}
QLabel {{
    background: transparent;
}}
QLabel#Brand {{
    font-size: 19px;
    font-weight: 700;
    letter-spacing: 1px;
}}
QLabel#BrandSub {{
    font-size: 10px;
    color: {TEXT_FAINT};
    letter-spacing: 1px;
}}
QLabel#CardTitle {{
    font-size: 12px;
    font-weight: 700;
    color: {TEXT_DIM};
    letter-spacing: 2px;
}}
QLabel#SectionTag {{
    font-size: 10px;
    color: {TEXT_FAINT};
    letter-spacing: 1px;
}}
QLabel#Mono {{
    font-family: "{FONT_MONO}", monospace;
}}
QLabel#StatBig {{
    font-family: "{FONT_MONO}", monospace;
    font-size: 21px;
    font-weight: 700;
}}
QLabel#StatKey {{
    font-size: 10px;
    color: {TEXT_FAINT};
    letter-spacing: 1px;
}}
QLabel#Pill {{
    background: {PANEL_SOFT};
    border: 1px solid {EDGE};
    border-radius: 11px;
    padding: 2px 10px;
    font-family: "{FONT_MONO}", monospace;
    font-size: 11px;
    color: {TEXT_DIM};
}}
QPushButton {{
    background: {PANEL_SOFT};
    border: 1px solid {EDGE};
    border-radius: 6px;
    padding: 6px 13px;
    font-size: 12px;
    color: {TEXT};
}}
QPushButton:hover {{
    border-color: {ACCENT_DIM};
    background: #10202b;
}}
QPushButton:pressed {{
    background: #0a1620;
}}
QPushButton:disabled {{
    color: {TEXT_FAINT};
    border-color: {EDGE_SOFT};
    background: #0a1017;
}}
QPushButton#Primary {{
    background: {ACCENT};
    border: 1px solid {ACCENT};
    color: #04140e;
    font-weight: 700;
}}
QPushButton#Primary:hover {{
    background: #4cf0b4;
}}
QPushButton#Danger {{
    background: {DANGER};
    border: 1px solid {DANGER};
    color: #1a0505;
    font-weight: 700;
}}
QPushButton#Danger:hover {{
    background: #ff7474;
}}
QPushButton#Ghost {{
    background: transparent;
}}
QScrollArea, QScrollArea > QWidget > QWidget {{
    background: transparent;
    border: none;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: #22394b;
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: #2e4d63;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    height: 0px;
    background: transparent;
}}
QToolTip {{
    background: {PANEL};
    color: {TEXT};
    border: 1px solid {EDGE};
    padding: 4px 8px;
}}
"""


def pill_style(color: str, filled: bool = False) -> str:
    """生成一枚状态胶囊的 QSS（告警卡片、状态点用）。"""
    if filled:
        return (f"background: {color}; color: #04140e; border: 1px solid {color};"
                f"border-radius: 9px; padding: 1px 9px; font-size: 10px;"
                f"font-weight: 700;")
    return (f"background: transparent; color: {color}; border: 1px solid {color};"
            f"border-radius: 9px; padding: 1px 9px; font-size: 10px;")


def card_style(accent: str = EDGE) -> str:
    """告警卡片：左侧色条靠 QSS 的 border-left 模拟。"""
    return (f"QFrame#AlarmCard {{ background: {PANEL_SOFT}; border: 1px solid {EDGE};"
            f" border-left: 3px solid {accent}; border-radius: 6px; }}")