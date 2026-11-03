# -*- coding: utf-8 -*-
"""界面入口。

    python -m forestfire_gs.ui.app                 # 正常启动
    python -m forestfire_gs.ui.app --shot a.png    # 启动 3 s 后截图并退出（CI / 文档用）
    python -m forestfire_gs.ui.app --no-start      # 只开界面，不自动开始监测
"""
from __future__ import annotations

import argparse
import sys

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QApplication

from ..config import StationConfig
from .main_window import MainWindow


def build_app(argv=None) -> QApplication:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication.instance() or QApplication(argv or sys.argv[:1])
    app.setApplicationName("ForestFire GS")
    app.setApplicationDisplayName("林火之眼 · 无人机森林防火监测地面站")
    return app


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="森林防火无人机地面站界面")
    p.add_argument("--shot", default="", help="截图输出路径（给出即截图后退出）")
    p.add_argument("--delay", type=int, default=3200, help="截图前等待毫秒数")
    p.add_argument("--no-start", action="store_true", help="不自动开始监测")
    p.add_argument("--resolve-at", type=int, default=0,
                   help="截图前第多少毫秒自动点一次“确认告警”，用于演示已处置态")
    p.add_argument("--config", default="", help="配置文件路径")
    p.add_argument("--yolo", action="store_true",
                   help="用 Ultralytics YOLOv8 做真实推理（需 pip install -e .[vision] 与权重）")
    a = p.parse_args(argv)

    cfg = StationConfig.load(a.config or None)
    detector = None
    if a.yolo:
        from ..detector import YoloDetector
        detector = YoloDetector(cfg.detect.model_path, conf=cfg.detect.conf,
                                nms_iou=cfg.detect.nms_iou, imgsz=cfg.detect.imgsz)
        print(f"detector -> YOLO ({cfg.detect.model_path})")
    app = build_app()
    win = MainWindow(cfg, autostart=not a.no_start, detector=detector)
    win.show()

    if a.resolve_at > 0:
        QTimer.singleShot(a.resolve_at, win._resolve)

    if a.shot:
        def _grab():
            pix = win.grab()
            img = pix.toImage()
            ok = img.save(a.shot, "PNG")
            print(f"screenshot -> {a.shot} ok={ok} size={img.width()}x{img.height()}")
            app.quit()
        QTimer.singleShot(a.delay, _grab)

    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())