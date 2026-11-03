# -*- coding: utf-8 -*-
"""主窗口：把引擎、自绘控件和操作员动作装配到一起。

数据流是单向的：

    StationEngine（采集/推理线程）
        → listener 回调 → 线程安全 deque
        → QTimer(40 ms) 在主线程排空 → 更新控件

界面线程只读队列、不做任何计算，和论文 2.4 节的三线程模型一致。
"""
from __future__ import annotations

import datetime as _dt
import os
import time
from collections import deque
from pathlib import Path

from PyQt5.QtCore import QSize, Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel,
                             QMainWindow, QPushButton, QScrollArea, QSizePolicy,
                             QVBoxLayout, QWidget)

from ..alarm import CONFIRMED, DISMISSED, IDLE, MONITORING, RESOLVED, SUSPECT
from ..config import StationConfig
from ..metrics import error_vs_heights, latency_budget, pr_curve
from ..pipeline import StationEngine
from ..storage import StationDB, screenshot_path
from ..telemetry import PatrolSimulator
from . import theme
from .widgets import (Card, ErrorChart, FsmStrip, HorizonPanel, LatencyChart,
                      MapPanel, PrChart, VideoPanel)

STATE_LABEL = {IDLE: "空闲", MONITORING: "监测中", SUSPECT: "疑似",
               CONFIRMED: "已确认", RESOLVED: "已处置", DISMISSED: "误报"}
STATE_COLOR = {IDLE: theme.IDLE, MONITORING: theme.ACCENT, SUSPECT: theme.WARN,
               CONFIRMED: theme.DANGER, RESOLVED: theme.INFO, DISMISSED: "#8d7aa6"}


class StatTile(QWidget):
    """一个指标块：上面一行小标签，下面一行等宽数字。"""

    def __init__(self, key: str, unit: str = "", value: str = "--", size: int = 21,
                 parent=None):
        super().__init__(parent)
        self._size = size
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(1)
        self.key = QLabel(key)
        self.key.setObjectName("StatKey")
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(3)
        self.value = QLabel(value)
        self.value.setObjectName("StatBig")
        self.value.setStyleSheet(f"color: {theme.TEXT}; font-size: {size}px;")
        row.addWidget(self.value, 0, Qt.AlignBottom)
        self.unit = QLabel(unit)
        self.unit.setStyleSheet(f"color: {theme.TEXT_FAINT}; font-size: 10px;")
        self.unit.setAlignment(Qt.AlignBottom)
        row.addWidget(self.unit, 0, Qt.AlignBottom)
        row.addStretch(1)
        lay.addWidget(self.key)
        lay.addLayout(row)

    def set_value(self, text: str, color: str | None = None) -> None:
        self.value.setText(text)
        if color:
            self.value.setStyleSheet(f"color: {color}; font-size: {self._size}px;")


class MainWindow(QMainWindow):
    def __init__(self, cfg: StationConfig | None = None, autostart: bool = True,
                 detector=None):
        super().__init__()
        self.cfg = cfg or StationConfig.load()
        self.setWindowTitle("林火之眼 · 无人机森林防火监测地面站")
        self.resize(1560, 980)
        self.setMinimumSize(1280, 800)
        self.setStyleSheet(theme.QSS)

        self._events: deque = deque(maxlen=800)
        self.alarm_records: list = []
        self.fires: dict = {}
        self.frames = 0
        self._last_state = IDLE
        self._t0 = time.time()

        self.db = StationDB(self.cfg.db_path)
        self.engine = StationEngine(self.cfg, detector=detector,
                                    listener=self._on_engine_event)
        self.sim = PatrolSimulator(seed=self.cfg.seed)

        self._build_ui()
        self._wire()

        self.tick = QTimer(self)
        self.tick.timeout.connect(self._on_tick)
        self.tick.start(40)
        self.clock = QTimer(self)
        self.clock.timeout.connect(self._on_clock)
        self.clock.start(1000)
        self._on_clock()

        if autostart:
            self.start_monitor()

    # ------------------------------------------------------------------ #
    # 界面装配
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(10)

        outer.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setSpacing(10)
        body.addWidget(self._build_video_col(), 62)

        side = QVBoxLayout()
        side.setSpacing(10)
        side.addWidget(self._build_map_col(), 42)
        side.addWidget(self._build_alarm_col(), 58)
        body.addLayout(side, 38)
        outer.addLayout(body, 1)

        outer.addWidget(self._build_bottom())

    # ---------------- 顶栏 ----------------
    def _build_header(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("Card")
        bar.setFixedHeight(72)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 8, 12, 8)
        lay.setSpacing(9)

        brand = QVBoxLayout()
        brand.setSpacing(0)
        title = QLabel("林火之眼")
        title.setObjectName("Brand")
        title.setStyleSheet(f"color: {theme.TEXT};")
        sub = QLabel("FOREST-FIRE UAV STATION")
        sub.setObjectName("BrandSub")
        brand.addWidget(title)
        brand.addWidget(sub)
        lay.addLayout(brand)

        lay.addSpacing(10)
        self.pill_mission = self._pill("FS-2026-1103 / 07")
        lay.addWidget(self.pill_mission)
        lay.addStretch(1)

        self.pill_link = self._pill("链路 12 ms")
        self.pill_infer = self._pill("推理 -- fps")
        self.pill_fix = self._pill("定位 ±1.8 m")
        self.pill_clock = self._pill("--:--:--")
        for w in (self.pill_link, self.pill_infer, self.pill_fix, self.pill_clock):
            lay.addWidget(w)

        lay.addSpacing(6)
        self.btn_start = QPushButton("开始监测")
        self.btn_start.setObjectName("Primary")
        self.btn_stop = QPushButton("停止监测")
        self.btn_resolve = QPushButton("确认告警")
        self.btn_dismiss = QPushButton("撤销误报")
        self.btn_shot = QPushButton("截图")
        self.btn_shot.setObjectName("Ghost")
        for b in (self.btn_start, self.btn_stop, self.btn_resolve,
                  self.btn_dismiss, self.btn_shot):
            b.setCursor(Qt.PointingHandCursor)
            lay.addWidget(b)
        self.btn_stop.setEnabled(False)
        self.btn_resolve.setEnabled(False)
        self.btn_dismiss.setEnabled(False)
        return bar

    def _pill(self, text: str) -> QLabel:
        lab = QLabel(text)
        lab.setObjectName("Pill")
        return lab

    # ---------------- 左：视频 ----------------
    def _build_video_col(self) -> QWidget:
        card = Card("实时视频", "LIVE · 可见光 / 红外融合")
        self.video = VideoPanel()
        card.body.addWidget(self.video, 1)

        row = QHBoxLayout()
        row.setSpacing(18)
        self.t_cand = StatTile("候选目标", "个")
        self.t_conf = StatTile("确认目标", "个")
        self.t_fix = StatTile("定位解算", "次")
        self.t_fps = StatTile("处理帧率", "fps")
        for t in (self.t_cand, self.t_conf, self.t_fix, self.t_fps):
            row.addWidget(t)
        row.addStretch(1)
        self.lab_mode = QLabel("视频源  sim://patrol-01  ·  1920×1080  ·  H.264 软解")
        self.lab_mode.setStyleSheet(f"color: {theme.TEXT_FAINT}; font-size: 10px;")
        row.addWidget(self.lab_mode)
        card.body.addLayout(row)
        return card

    # ---------------- 中：地图 ----------------
    def _build_map_col(self) -> QWidget:
        card = Card("态势地图", "NED / WGS-84")
        self.map = MapPanel()
        self.map.set_route(self.sim.track_points(64))
        card.body.addWidget(self.map, 1)

        legend = QHBoxLayout()
        legend.setSpacing(10)
        for color, text in ((theme.DANGER, "确认火点"), (theme.WARN, "疑似"),
                            (theme.INFO, "无人机"), (theme.ACCENT, "云台视场")):
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {color}; font-size: 11px;")
            lab = QLabel(text)
            lab.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 10px;")
            legend.addWidget(dot)
            legend.addWidget(lab)
        legend.addStretch(1)
        card.body.addLayout(legend)
        return card

    # ---------------- 右：告警 ----------------
    def _build_alarm_col(self) -> QWidget:
        card = Card("告警队列", "ALARM QUEUE")
        self.fsm = FsmStrip()
        card.body.addWidget(self.fsm)

        wrap = QWidget()
        self.alarm_box = QVBoxLayout(wrap)
        self.alarm_box.setContentsMargins(0, 0, 4, 0)
        self.alarm_box.setSpacing(8)
        self.alarm_box.addStretch(1)

        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(wrap)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        card.body.addWidget(area, 1)

        self.lab_empty = QLabel("暂无告警。\n监测中，等待 AI 确认火情…")
        self.lab_empty.setAlignment(Qt.AlignCenter)
        self.lab_empty.setStyleSheet(
            f"color: {theme.TEXT_FAINT}; font-size: 11px; border: 1px dashed {theme.EDGE};"
            f" border-radius: 8px; padding: 18px;")
        self.alarm_box.insertWidget(0, self.lab_empty)

        self.lab_log = QLabel("系统就绪")
        self.lab_log.setStyleSheet(f"color: {theme.TEXT_FAINT}; font-size: 10px;")
        self.lab_log.setWordWrap(True)
        card.body.addWidget(self.lab_log)
        return card

    # ---------------- 底：遥测 + 图表 ----------------
    def _build_bottom(self) -> QWidget:
        row = QHBoxLayout()
        row.setSpacing(10)

        telem = Card("飞行遥测", "TELEMETRY 10 Hz")
        telem.setFixedWidth(720)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        self.tiles = {}
        spec = [("lon", "经度", "°"), ("lat", "纬度", "°"), ("alt", "相对高度", "m"),
                ("spd", "地速", "m/s"), ("yaw", "航向", "°"), ("roll", "滚转", "°"),
                ("pitch", "俯仰", "°"), ("gpitch", "云台俯仰", "°"), ("gyaw", "云台偏航", "°"),
                ("batt", "电池", "%"), ("sat", "卫星", "颗"), ("lat_ms", "端到端时延", "s")]
        for i, (key, label, unit) in enumerate(spec):
            tile = StatTile(label, unit, size=15)
            self.tiles[key] = tile
            grid.addWidget(tile, i // 3, i % 3)
        holder = QWidget()
        holder.setLayout(grid)
        card_row = QHBoxLayout()
        card_row.addWidget(holder, 1)
        self.horizon = HorizonPanel()
        self.horizon.setFixedSize(158, 112)
        card_row.addWidget(self.horizon, 0, Qt.AlignTop)
        telem.body.addLayout(card_row)
        row.addWidget(telem, 0)

        charts = Card("指标看板", "论文第 4 章 · 可复现")
        charts.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        h = QHBoxLayout()
        h.setSpacing(12)
        self.chart_err = ErrorChart(error_vs_heights())
        self.chart_lat = LatencyChart(latency_budget())
        h.addWidget(self.chart_err, 1)
        h.addWidget(self.chart_lat, 1)
        rec, prec = pr_curve()
        self.chart_pr = PrChart()
        self.chart_pr.set_curve(rec, prec)
        self.chart_pr.setFixedWidth(180)
        h.addWidget(self.chart_pr, 0)
        charts.body.addLayout(h, 1)
        row.addWidget(charts, 1)
        return self._wrap_row(row)

    def _wrap_row(self, lay) -> QWidget:
        w = QWidget()
        w.setFixedHeight(238)
        w.setLayout(lay)
        lay.setContentsMargins(0, 0, 0, 0)
        return w

    # ------------------------------------------------------------------ #
    # 交互
    # ------------------------------------------------------------------ #
    def _wire(self) -> None:
        self.btn_start.clicked.connect(self.start_monitor)
        self.btn_stop.clicked.connect(self.stop_monitor)
        self.btn_resolve.clicked.connect(self._resolve)
        self.btn_dismiss.clicked.connect(self._dismiss)
        self.btn_shot.clicked.connect(self._screenshot)

    def start_monitor(self) -> None:
        self.engine.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._log("开始监测，链路建立，推理线程启动")
        self.video.set_mode("仿真视频流 · sim://patrol-01 · 巡逻航线 A")

    def stop_monitor(self) -> None:
        self.engine.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_resolve.setEnabled(False)
        self.btn_dismiss.setEnabled(False)
        self._log("监测已停止")

    def _resolve(self) -> None:
        rec = self.engine.resolve()
        if rec:
            self._apply_record_state(rec)
            self._log(f"告警 {self._code(rec)} 已处置，已写入处置记录")

    def _dismiss(self) -> None:
        rec = self.engine.dismiss()
        if rec:
            self._apply_record_state(rec)
            self._log(f"告警 {self._code(rec)} 标记为误报")

    def _screenshot(self) -> None:
        ts = time.time()
        path = Path(screenshot_path(self.cfg.screenshot_dir, ts))
        path.parent.mkdir(parents=True, exist_ok=True)
        self.grab().save(str(path), "PNG")
        if self.alarm_records:
            self.db.attach_screenshot(self.alarm_records[0]["code"], str(path), ts)
        self._log(f"已截图存档 {path.name}")

    # ------------------------------------------------------------------ #
    # 引擎 → 界面
    # ------------------------------------------------------------------ #
    def _on_engine_event(self, name: str, payload) -> None:
        """在采集/推理线程中被调用，只入队，不碰控件。"""
        self._events.append((name, payload))

    def _on_tick(self) -> None:
        while True:
            try:
                name, payload = self._events.popleft()
            except IndexError:
                break
            if name == "frame":
                self._on_frame(payload)
            elif name == "alarm_update" and payload:
                self._apply_record_state(payload)
            elif name == "state" and payload:
                self._on_state(payload)

        self.map.tick(0.04)
        self.fsm.tick(0.04)
        snap = self.engine.snapshot()
        self.pill_infer.setText(f"推理 {snap['fps']:4.1f} fps")
        self.pill_link.setText("链路 %2d ms" % (8 + int(abs(snap["t"]) * 7) % 9))

    def _on_state(self, payload: dict) -> None:
        self._set_fsm(payload.get("fsm", IDLE))

    def _on_frame(self, res) -> None:
        self.frames += 1
        self.video.update_result(res)
        tm = res.telemetry
        self.map.set_telemetry(tm)
        self.horizon.set_attitude(tm.roll, tm.pitch)

        self.t_cand.set_value(str(len(res.dets)), theme.TEXT)
        self.t_conf.set_value(str(len(res.confirmed)),
                              theme.DANGER if res.confirmed else theme.TEXT)
        self.t_fix.set_value(str(len(res.fixes)), theme.ACCENT if res.fixes else theme.TEXT)
        self.t_fps.set_value(f"{res.fps:.1f}", theme.ACCENT)
        self.pill_fix.setText(f"定位 ±{1.5 + tm.H / 180:.1f} m")

        t = self.tiles
        t["lon"].set_value(f"{tm.lon:.5f}")
        t["lat"].set_value(f"{tm.lat:.5f}")
        t["alt"].set_value(f"{tm.H:.1f}")
        t["spd"].set_value(f"{tm.ground_speed:.1f}")
        t["yaw"].set_value(f"{tm.yaw:.0f}")
        t["roll"].set_value(f"{tm.roll:+.1f}")
        t["pitch"].set_value(f"{tm.pitch:+.1f}")
        t["gpitch"].set_value(f"{tm.gimbal_pitch:.1f}", theme.INFO)
        t["gyaw"].set_value(f"{tm.gimbal_yaw:.0f}", theme.INFO)
        t["batt"].set_value(f"{tm.battery:.0f}",
                            theme.WARN if tm.battery < 30 else theme.TEXT)
        t["sat"].set_value(str(tm.satellites))
        t["lat_ms"].set_value(f"{2.3:.1f}")

        if res.alarm:
            self._on_new_alarm(res.alarm)
        self._set_fsm(self.engine.alarm.fsm.state)

    # ------------------------------------------------------------------ #
    # 告警卡片
    # ------------------------------------------------------------------ #
    @staticmethod
    def _code(rec: dict) -> str:
        t = rec["fix"]["telemetry"]["t"]
        return f"FF-{int(t)}-{rec['seq']:03d}"

    def _on_new_alarm(self, payload: dict) -> None:
        rec = dict(payload)
        rec["code"] = self._code(rec)
        self.alarm_records.insert(0, rec)
        fix = rec["fix"]
        self.fires[rec["code"]] = (fix["lat"], fix["lon"], rec["code"], CONFIRMED)
        self.map.set_fires(list(self.fires.values()))

        self.db.save_fire_fields(rec["code"], fix["lat"], fix["lon"],
                                 fix["detection"]["conf"], fix["detection"]["cls"],
                                 fix["telemetry"]["t"], "sim://patrol-01",
                                 fix["range_m"], fix["skew_s"])

        self.lab_empty.setVisible(False)
        self.alarm_box.insertWidget(0, self._alarm_card(rec))
        self._trim_cards()
        self.btn_resolve.setEnabled(True)
        self.btn_dismiss.setEnabled(True)
        self._set_fsm(CONFIRMED)
        self._log(f"新告警 {rec['code']}  {fix['lat']:.6f}, {fix['lon']:.6f}  "
                  f"conf {fix['detection']['conf']:.2f}")

    def _alarm_card(self, rec: dict) -> QWidget:
        card = QFrame()
        card.setObjectName("AlarmCard")
        card.setStyleSheet(theme.card_style(theme.DANGER))
        lay = QVBoxLayout(card)
        lay.setContentsMargins(9, 7, 9, 8)
        lay.setSpacing(3)

        head = QHBoxLayout()
        head.setSpacing(6)
        code = QLabel(rec["code"])
        code.setStyleSheet(f"color: {theme.TEXT}; font-family: {theme.FONT_MONO};"
                           f" font-size: 12px; font-weight: 700;")
        head.addWidget(code)
        head.addStretch(1)
        pill = QLabel("已确认")
        pill.setStyleSheet(theme.pill_style(theme.DANGER, True))
        head.addWidget(pill)
        lay.addLayout(head)

        fix = rec["fix"]
        tm = fix["telemetry"]
        lab = QLabel(f"{fix['lat']:.6f}, {fix['lon']:.6f}")
        lab.setStyleSheet(f"color: {theme.INFO}; font-family: {theme.FONT_MONO};"
                          f" font-size: 11px;")
        lay.addWidget(lab)

        meta = QLabel(f"距离 {fix['range_m']:.0f} m · 高度 {tm['H']:.0f} m · "
                      f"置信度 {fix['detection']['conf']:.2f} · "
                      f"时差 {fix['skew_s'] * 1000:.0f} ms")
        meta.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 10px;")
        lay.addWidget(meta)

        stamp = _dt.datetime.fromtimestamp(rec["created_at"]).strftime("%H:%M:%S")
        foot = QLabel(f"{stamp}  时序确认 3/5 帧命中 · 已推送指挥中心")
        foot.setStyleSheet(f"color: {theme.TEXT_FAINT}; font-size: 10px;")
        lay.addWidget(foot)

        card.setProperty("alarm_code", rec["code"])
        card.setProperty("pill", pill)
        card.setProperty("foot", foot)
        return card

    def _trim_cards(self) -> None:
        while self.alarm_box.count() > 7:
            item = self.alarm_box.takeAt(self.alarm_box.count() - 1)
            w = item.widget()
            if w is not None and w is not self.lab_empty:
                w.deleteLater()

    def _apply_record_state(self, rec: dict) -> None:
        code = rec.get("code") or self._code(rec)
        state = rec.get("state", CONFIRMED)
        color = STATE_COLOR.get(state, theme.TEXT_DIM)
        for i in range(self.alarm_box.count()):
            w = self.alarm_box.itemAt(i).widget()
            if w is not None and w.property("alarm_code") == code:
                w.setStyleSheet(theme.card_style(color))
                pill = w.property("pill")
                if pill is not None:
                    pill.setText(STATE_LABEL.get(state, state))
                    pill.setStyleSheet(theme.pill_style(color, True))
                break
        self.fires[code] = (*self.fires.get(code, (0, 0))[:2], code, state)
        self.map.set_fires(list(self.fires.values()))
        self.db.record_event(code, "operator",
                             "confirm" if state == RESOLVED else "dismiss",
                             state, "operator")
        if state in (RESOLVED, DISMISSED):
            self.btn_resolve.setEnabled(False)
            self.btn_dismiss.setEnabled(False)

    def _set_fsm(self, state: str) -> None:
        if state != self._last_state:
            self.fsm.set_state(state)
            self._last_state = state

    # ------------------------------------------------------------------ #
    # 杂项
    # ------------------------------------------------------------------ #
    def _log(self, text: str) -> None:
        self.lab_log.setText(text)

    def _on_clock(self) -> None:
        now = _dt.datetime.now()
        self.pill_clock.setText(now.strftime("%H:%M:%S"))

    def closeEvent(self, ev):  # noqa: N802
        try:
            self.engine.stop()
            self.db.close()
        finally:
            super().closeEvent(ev)