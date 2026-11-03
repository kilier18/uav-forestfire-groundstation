"""界面自检：几何、文字挤压、区域像素特征。

在无法肉眼审图的机器上（CI / 无桌面会话）用「几何 + 像素 + 排版」三重断言代替目视，
每条断言都对应界面上一个必须存在的视觉要素，任一失败即以非零码退出。

用法::

    python tools/qa_ui.py                     # 默认 1560x960，跑 4 秒后自检
    python tools/qa_ui.py --size 1280x800 --seconds 6 --resolve
    python tools/qa_ui.py --out artifacts/qa.png

Windows 上无桌面会话时需 ``QT_QPA_PLATFORM=offscreen QT_QPA_FONTDIR=C:/Windows/Fonts``，
否则 Qt 找不到字体目录，截图里所有文字都会消失。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PyQt5.QtCore import QPoint, QTimer  # noqa: E402
from PyQt5.QtWidgets import QApplication, QLabel  # noqa: E402

from forestfire_gs.config import StationConfig  # noqa: E402
from forestfire_gs.ui.main_window import MainWindow  # noqa: E402

CHECKS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


def crop(img: np.ndarray, geom) -> np.ndarray:
    """按控件在窗口内的绝对几何裁出区域。"""
    x, y, w, h = geom
    H, W = img.shape[:2]
    return img[max(0, y):min(H, y + h), max(0, x):min(W, x + w)]


def geom_of(win, widget):
    pt = widget.mapTo(win, QPoint(0, 0))
    return (pt.x(), pt.y(), widget.width(), widget.height())


def channels(sub: np.ndarray):
    return (sub[..., 0].astype(float), sub[..., 1].astype(float), sub[..., 2].astype(float))


def lum_mask(sub: np.ndarray, thr: float) -> np.ndarray:
    r, g, b = channels(sub)
    return 0.299 * r + 0.587 * g + 0.114 * b > thr


def red_mask(sub: np.ndarray) -> np.ndarray:
    r, g, b = channels(sub)
    return (r > 150) & (r - g > 60) & (r - b > 60)


def warm_mask(sub: np.ndarray) -> np.ndarray:
    r, g, b = channels(sub)
    return (r > 195) & (g > 110) & (b < 140)


def blue_mask(sub: np.ndarray) -> np.ndarray:
    r, g, b = channels(sub)
    return (b > 110) & (b - r > 40)


def stat_line(name: str, sub: np.ndarray) -> str:
    r, g, b = channels(sub)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return (f"  {name:<10} 亮度 {lum.mean():5.1f} max {lum.max():5.1f} "
            f"暖色 {warm_mask(sub).mean() * 100:5.2f}% "
            f"告警红 {red_mask(sub).mean() * 100:5.2f}% "
            f"亮字 {lum_mask(sub, 170).mean() * 100:5.2f}%")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "artifacts" / "qa.png"))
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--size", default="1560x960")
    p.add_argument("--resolve", action="store_true", help="截图前点一次确认告警")
    a = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    w, h = (int(v) for v in a.size.lower().split("x"))
    app = QApplication(sys.argv[:1])
    win = MainWindow(StationConfig(), autostart=True)
    win.resize(w, h)
    win.show()

    timer = QTimer()
    timer.timeout.connect(app.processEvents)
    timer.start(20)

    def finish():
        print("\n== 几何 ==")
        gs = {}
        for name, widget in (("video", win.video), ("map", win.map), ("fsm", win.fsm),
                             ("alarm", win.alarm_box.parentWidget()),
                             ("chart_err", win.chart_err), ("chart_lat", win.chart_lat),
                             ("horizon", win.horizon)):
            g = geom_of(win, widget)
            gs[name] = g
            print(f"  {name:<10} {g[2]:4d} x {g[3]:4d} @ {g[0]:4d},{g[1]:4d}  "
                  f"visible={widget.isVisible()}")
        print(f"  告警卡 {win.alarm_box.count() - 1} 张  帧数 {win.frames}  "
              f"状态机 {win.engine.alarm.fsm.state}  "
              f"按钮可用 确认={win.btn_resolve.isEnabled()} 误报={win.btn_dismiss.isEnabled()}")

        path = Path(a.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        win.grab().save(str(path), "PNG")
        img = np.asarray(Image.open(path).convert("RGB"))
        print(f"\n== 截图 {path} {img.shape[1]}x{img.shape[0]} ==")

        print("\n== 排版：文字是否被裁（sizeHint 宽 > 实际宽）==")
        squeezed = []
        for lab in win.findChildren(QLabel):
            if not lab.text() or not lab.isVisible():
                continue
            need = lab.sizeHint().width()
            if need > lab.width() + 6 and not lab.wordWrap():
                squeezed.append((need, lab.width(), lab.text()[:40]))
        for need, has, text in squeezed[:12]:
            print(f"  {need:4d} > {has:4d}  {text!r}")
        check("文字无水平裁切", not squeezed, f"{len(squeezed)} 处")

        print("\n== 像素要素 ==")
        vid = crop(img, gs["video"])
        mp = crop(img, gs["map"])
        al = crop(img, gs["alarm"])
        hd = crop(img, (0, 0, img.shape[1], gs["video"][1] - 4))
        ce = crop(img, gs["chart_err"])
        cl = crop(img, gs["chart_lat"])
        ho = crop(img, gs["horizon"])
        for name, sub in (("顶栏", hd), ("视频", vid), ("地图", mp), ("告警卡", al),
                          ("曲线图", ce), ("时延图", cl), ("地平仪", ho), ("整屏", img)):
            print(stat_line(name, sub))

        # 视频：火盆质心应落在画面中下部，且有暖色高光
        wm = warm_mask(vid)
        if wm.any():
            ys, xs = np.nonzero(wm)
            cx, cy = xs.mean() / vid.shape[1], ys.mean() / vid.shape[0]
        else:
            cx = cy = -1.0
        check("视频有火盆高光", int(wm.sum()) > 400, f"{int(wm.sum())} px")
        check("火盆位于画面中下部", 0.35 < cx < 0.68 and 0.42 < cy < 0.82,
              f"质心 ({cx:.3f}, {cy:.3f})")
        check("视频有检测框", int(red_mask(vid).sum()) > 250,
              f"{int(red_mask(vid).sum())} px")

        # 地图：火点标记 + 无人机 + 航线 + 文字
        check("地图有火点标记", int(red_mask(mp).sum()) > 80, f"{int(red_mask(mp).sum())} px")
        check("地图有航线", int(blue_mask(mp).sum()) > 120, f"{int(blue_mask(mp).sum())} px")
        check("地图有文字标注", int(lum_mask(mp, 190).sum()) > 60,
              f"{int(lum_mask(mp, 190).sum())} px")

        # 告警卡：红色状态条 + 正文文字
        check("告警卡有状态条", int(red_mask(al).sum()) > 40, f"{int(red_mask(al).sum())} px")
        check("告警卡有文字", int(lum_mask(al, 120).sum()) > 1200,
              f"{int(lum_mask(al, 120).sum())} px (亮度>120)")

        # 图表：要有亮色数据线，不能是空画布
        check("误差图有曲线", int(lum_mask(ce, 120).sum()) > 250,
              f"{int(lum_mask(ce, 120).sum())} px")
        check("时延图有条形", int(lum_mask(cl, 120).sum()) > 250,
              f"{int(lum_mask(cl, 120).sum())} px")
        peak = int(lum_mask(ho, 110).sum(axis=1).max())     # 地平线应横扫整个控件
        check("地平仪有地平线", peak > ho.shape[1] * 0.55, f"最长行 {peak}/{ho.shape[1]} px")

        # 顶栏：必须有亮色文字（字体缺失时整块发暗）
        check("顶栏文字可见", int(lum_mask(hd, 170).sum()) > 700,
              f"{int(lum_mask(hd, 170).sum())} px")

        bad = [n for n, ok, _ in CHECKS if not ok]
        print(f"\n== 结论：{len(CHECKS) - len(bad)}/{len(CHECKS)} 通过"
              + ("" if not bad else "；未通过：" + "、".join(bad)) + " ==")
        win.engine.stop()
        app.quit()
        sys.exit(1 if bad else 0)

    if a.resolve:
        QTimer.singleShot(int(a.seconds * 700), win._resolve)
    QTimer.singleShot(int(a.seconds * 1000), finish)
    app.exec_()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())