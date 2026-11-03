"""出图脚本：为 README / 论文 / GitHub Pages 生成界面截图与演示 GIF。

无法目视审图的机器上用 ``QT_QPA_PLATFORM=offscreen`` 也能出图；Windows 上另外需要
``QT_QPA_FONTDIR=C:/Windows/Fonts``，否则 Qt 找不到字体目录、截图里没有文字。

用法::

    python tools/shots.py                    # 生成全部截图 + GIF 到 docs/img
    python tools/shots.py --only hero        # 只出一张
    python tools/shots.py --width 1920 --height 1080
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402
from PyQt5.QtCore import QEventLoop, QPoint, QTimer  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from forestfire_gs.config import StationConfig  # noqa: E402
from forestfire_gs.ui.main_window import MainWindow  # noqa: E402

# 每张图：等待秒数、截图前动作
PLAN = {
    "hero": (5.0, None),                 # 监测中 + 已确认告警，封面图
    "panel-monitor": (1.0, "idle"),        # 待命界面：未开始监测
    "panel-alarm": (5.0, "resolve"),     # 指挥处置：点「确认告警」后
    "panel-dismiss": (5.0, "dismiss"),   # 误报撤销
    "panel-stats": (6.0, "crop"),        # 底部遥测 + 指标面板（裁切）
}


def wait(app: QApplication, seconds: float, tick_ms: int = 20) -> None:
    loop = QEventLoop()
    timer = QTimer()
    timer.timeout.connect(app.processEvents)
    timer.start(tick_ms)
    QTimer.singleShot(int(seconds * 1000), loop.quit)
    loop.exec_()
    timer.stop()


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "docs" / "img"))
    p.add_argument("--only", default="")
    p.add_argument("--width", type=int, default=1560)
    p.add_argument("--height", type=int, default=980)
    p.add_argument("--gif", action="store_true", default=True)
    p.add_argument("--no-gif", dest="gif", action="store_false")
    p.add_argument("--gif-frames", type=int, default=48)
    p.add_argument("--gif-interval", type=int, default=90)
    a = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv[:1])

    def fresh(autostart: bool = True) -> MainWindow:
        win = MainWindow(StationConfig(), autostart=autostart)
        win.resize(a.width, a.height)
        win.show()
        return win

    wanted = [a.only] if a.only else list(PLAN)
    for name in wanted:
        seconds, action = PLAN[name]
        win = fresh(autostart=(action != "idle"))
        if action == "resolve":
            QTimer.singleShot(int(seconds * 700), win._resolve)
        elif action == "dismiss":
            QTimer.singleShot(int(seconds * 700), win._dismiss)
        wait(app, seconds)
        path = out / f"{name}.png"
        pix = win.grab()
        if action == "crop":                      # 只留底部遥测 + 指标面板
            top = win.chart_err.mapTo(win, QPoint(0, 0)).y() - 26
            left = win.horizon.mapTo(win, QPoint(0, 0)).x() - 26
            pix = pix.copy(left, max(0, top), pix.width() - left - 12,
                           pix.height() - max(0, top) - 8)
        pix.save(str(path), "PNG")
        print(f"[shot] {path.relative_to(ROOT)}  {pix.width()}x{pix.height()}  "
              f"fsm={win.engine.alarm.fsm.state}  告警卡={win.alarm_box.count() - 1}")
        win.engine.stop()
        win.close()
        wait(app, 0.2)

    if a.gif and not a.only:
        win = fresh()
        wait(app, 1.2)
        frames = []
        for i in range(a.gif_frames):
            pm = win.grab().scaled(a.width // 2, a.height // 2)
            path = out / "_frame.png"
            pm.save(str(path), "PNG")
            frames.append(Image.open(path).convert("RGB").copy())
            wait(app, a.gif_interval / 1000.0)
        gif = out / "demo.gif"
        frames[0].save(gif, save_all=True, append_images=frames[1:],
                       duration=a.gif_interval, loop=0, optimize=True)
        (out / "_frame.png").unlink(missing_ok=True)
        print(f"[gif] {gif.relative_to(ROOT)}  {len(frames)} 帧  "
              f"{gif.stat().st_size / 1024:.0f} KB")
        win.engine.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())