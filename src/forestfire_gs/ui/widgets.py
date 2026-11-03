# -*- coding: utf-8 -*-
"""自绘控件集合。

界面里“有画面感”的部分全部在这里用 QPainter 画出来：视频窗（含合成画面、
检测框、OSD）、地图、地平仪、状态机条、指标图。之所以自绘而不是拼图片，
是因为这些控件要跟着遥测/检测结果实时变化，用矢量绘制既清晰又不依赖素材。
"""
from __future__ import annotations

import math
import random
from typing import Iterable, List, Optional, Sequence, Tuple

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import (QBrush, QColor, QFont, QLinearGradient, QPainter,
                         QPainterPath, QPen, QPolygonF, QRadialGradient)
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..alarm import (CONFIRMED, DISMISSED, IDLE, MONITORING, RESOLVED, SUSPECT)
from ..localization import latlon_to_ned
from . import theme

# 视频合成画面的虚拟分辨率（所有绘制先在这个坐标系里做，再整体缩放）
VW, VH = 1280.0, 720.0

# 类别名中文显示（模型里仍是 flame / smoke）
CLS_CN = {"flame": "火焰", "fire": "火焰", "smoke": "烟雾"}


def _c(name: str, alpha: int = 255) -> QColor:
    col = QColor(name)
    col.setAlpha(alpha)
    return col


def _mono(size: int = 10, bold: bool = False) -> QFont:
    f = QFont(theme.FONT_MONO)
    f.setPixelSize(size)
    f.setBold(bold)
    return f


def _ui(size: int = 11, bold: bool = False) -> QFont:
    f = QFont(theme.FONT_UI)
    f.setPixelSize(size)
    f.setBold(bold)
    return f


# --------------------------------------------------------------------------- #
# 卡片容器
# --------------------------------------------------------------------------- #
class Card(QFrame):
    """带标题行的卡片。``body`` 是内容布局，往里塞控件即可。"""

    def __init__(self, title: str, tag: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(12, 10, 12, 12)
        self._outer.setSpacing(8)

        head = QHBoxLayout()
        head.setSpacing(8)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("CardTitle")
        head.addWidget(self.title_label)
        head.addStretch(1)
        self.tag_label = QLabel(tag)
        self.tag_label.setObjectName("SectionTag")
        head.addWidget(self.tag_label)
        self._outer.addLayout(head)

        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(6)
        self._outer.addLayout(self.body, 1)

    def set_tag(self, text: str) -> None:
        self.tag_label.setText(text)


# --------------------------------------------------------------------------- #
# 视频窗
# --------------------------------------------------------------------------- #
class VideoPanel(QWidget):
    """合成巡逻画面 + 检测框 + OSD。

    ``update_result`` 由主窗口每帧调用；没有真实相机时画面由固定种子生成，
    保证同一帧序号画出来的地形完全一致（截图可复现）。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 270)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._rng = random.Random(20261103)
        self._trees = self._make_trees()
        self._contours = self._make_ground_contours()
        self._t = 0.0
        self._frame_no = 0
        self._dets: list = []
        self._tm = None
        self._fps = 0.0
        self._mode = "仿真视频流 · sim://patrol-01"
        self._confirmed_ids: set = set()

    # ---------------- 预生成静态纹理 ----------------
    def _make_trees(self):
        """生成一片俯视树冠：越靠下越大、越稀疏，模仿斜下视的透视。"""
        out = []
        for i in range(560):
            y = self._rng.uniform(0.30, 1.02)
            persp = (y - 0.30) / 0.72
            x = self._rng.uniform(-0.05, 1.05)
            r = 2.0 + 9.0 * persp * self._rng.uniform(0.55, 1.35)
            shade = self._rng.randint(0, 26)
            out.append((x, y, r, shade))
        out.sort(key=lambda t: t[1])
        return out

    def _make_ground_contours(self):
        """地面上的等亮度纹理带（俯视时像林子里的空地与山脊）。"""
        out = []
        for i in range(7):
            y0 = 0.34 + i * 0.095
            amp = self._rng.uniform(0.008, 0.020)
            phase = self._rng.uniform(0, 6.28)
            out.append((y0, amp, phase, self._rng.uniform(0.6, 1.6)))
        return out

    # ---------------- 数据入口 ----------------
    def update_result(self, result) -> None:
        """result: pipeline.FrameResult"""
        self._frame_no = result.index
        self._t = result.t
        self._dets = list(getattr(result, "dets", None)
                          or getattr(result, "detections", []))
        self._tm = result.telemetry
        self._fps = result.fps
        self._confirmed_ids = {d.track_id for d in result.confirmed}
        self.update()

    def set_mode(self, text: str) -> None:
        self._mode = text
        self.update()

    # ---------------- 绘制 ----------------
    def paintEvent(self, ev):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), _c(theme.BG))

        # 画面按“填满”方式缩放（等比放大后裁掉多余），不留黑边
        scale = max(self.width() / VW, self.height() / VH)
        ox = (self.width() - VW * scale) / 2.0
        oy = (self.height() - VH * scale) / 2.0
        p.save()
        p.translate(ox, oy)
        p.scale(scale, scale)
        p.setClipRect(QRectF(0, 0, VW, VH))
        self._paint_scene(p)
        self._paint_detections(p)
        p.restore()

        self._paint_osd(p)
        p.end()

    def _paint_scene(self, p: QPainter) -> None:
        # 天空 / 远山
        sky = QLinearGradient(0, 0, 0, VH * 0.34)
        sky.setColorAt(0.0, _c("#0d1a24"))
        sky.setColorAt(0.55, _c("#1d3341"))
        sky.setColorAt(1.0, _c("#3d5763"))
        p.fillRect(QRectF(0, 0, VW, VH * 0.34), QBrush(sky))

        for depth, alpha in ((0, 70), (1, 110), (2, 160)):
            path = QPainterPath()
            base = VH * (0.30 - depth * 0.018)
            path.moveTo(0, VH)
            x = 0.0
            k = 0
            while x <= VW + 40:
                h = (math.sin(k * 0.9 + depth * 2.1) * 18
                     + math.sin(k * 2.3 + depth) * 9 + 12)
                path.lineTo(x, base - h)
                x += 60
                k += 1
            path.lineTo(VW, VH)
            path.closeSubpath()
            p.fillPath(path, _c("#16242e", alpha))

        # 地面
        ground = QLinearGradient(0, VH * 0.30, 0, VH)
        ground.setColorAt(0.0, _c("#1b2a1f"))
        ground.setColorAt(0.45, _c("#13201a"))
        ground.setColorAt(1.0, _c("#0a1310"))
        p.fillRect(QRectF(0, VH * 0.30, VW, VH * 0.70), QBrush(ground))

        # 空地/林间带
        for y0, amp, phase, w in self._contours:
            path = QPainterPath()
            path.moveTo(-20, VH * y0)
            x = -20.0
            while x < VW + 20:
                path.lineTo(x, VH * y0 + math.sin(x / 210.0 + phase) * amp * VH)
                x += 24
            path.lineTo(VW + 20, VH * (y0 + 0.02))
            while x > -20:
                path.lineTo(x, VH * (y0 + 0.02) + math.sin(x / 180.0 + phase) * amp * VH)
                x -= 24
            path.closeSubpath()
            p.fillPath(path, _c("#1e2f22", 120))

        # 树冠
        for x, y, r, shade in self._trees:
            px, py = x * VW, y * VH
            col = QColor(22 + shade // 3, 46 + shade, 30 + shade // 2, 210)
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            p.drawEllipse(QPointF(px, py), r, r * 0.86)
            p.setBrush(_c("#2f5a34", 60))
            p.drawEllipse(QPointF(px - r * 0.25, py - r * 0.3), r * 0.55, r * 0.5)

        # 火点位置（云台锁定，随航迹轻微漂移）
        drift_x = math.sin(self._t * 0.35) * 9
        drift_y = math.cos(self._t * 0.27) * 5
        fx, fy = VW * 0.5 + drift_x, VH * 0.60 + drift_y
        self._fire_xy = (fx, fy)

        # 烟柱
        for i in range(9):
            k = (i + 1) / 9.0
            sy = fy - 30 - k * 210
            sx = fx + math.sin(self._t * 0.9 + i) * (12 + 26 * k) + k * 55
            rad = 16 + 52 * k
            p.setPen(Qt.NoPen)
            p.setBrush(_c("#7c8a90", int(30 * (1.0 - k) + 8)))
            p.drawEllipse(QPointF(sx, sy), rad, rad * 0.72)

        # 火焰光晕 + 火舌
        glow = QRadialGradient(QPointF(fx, fy), 150)
        glow.setColorAt(0.0, _c("#ffd27a", 210))
        glow.setColorAt(0.25, _c("#ff8a2b", 150))
        glow.setColorAt(0.6, _c("#c33b12", 60))
        glow.setColorAt(1.0, _c("#c33b12", 0))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(glow))
        p.drawEllipse(QPointF(fx, fy), 150, 150)

        flick = 0.82 + 0.18 * math.sin(self._t * 7.3) + 0.06 * math.sin(self._t * 17.1)
        for i, (dx, dy, s) in enumerate(((0, 0, 1.0), (-22, 8, 0.66), (20, 10, 0.6))):
            w = 30 * s * flick
            h = 54 * s * flick
            path = QPainterPath()
            path.moveTo(fx + dx, fy + dy - h)
            path.cubicTo(fx + dx + w, fy + dy - h * 0.3, fx + dx + w * 0.7, fy + dy + 6,
                         fx + dx, fy + dy + 10)
            path.cubicTo(fx + dx - w * 0.7, fy + dy + 6, fx + dx - w, fy + dy - h * 0.3,
                         fx + dx, fy + dy - h)
            p.fillPath(path, _c("#ff9c3d" if i == 0 else "#ff6a1e", int(200 - i * 40)))
            p.fillPath(path, _c("#ffe08a", 90))

        # 地面热辐射
        p.setBrush(_c("#ff7a2e", 34))
        p.drawEllipse(QPointF(fx, fy + 18), 96, 26)

        # 暗角
        vig = QRadialGradient(QPointF(VW / 2, VH / 2), VW * 0.72)
        vig.setColorAt(0.55, _c("#000000", 0))
        vig.setColorAt(1.0, _c("#000000", 165))
        p.fillRect(QRectF(0, 0, VW, VH), QBrush(vig))

    def _paint_detections(self, p: QPainter) -> None:
        # 检测框坐标基于 1920×1080，换算到虚拟画布
        sx, sy = VW / 1920.0, VH / 1080.0
        for d in self._dets:
            x1, y1, x2, y2 = d.box
            r = QRectF(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy)
            confirmed = d.track_id in self._confirmed_ids
            col = QColor(theme.DANGER if confirmed else theme.WARN)
            if d.cls != "fire":
                col = QColor(theme.INFO)

            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(_c(col.name(), 70), 1.2))
            p.drawRect(r)

            # 四角括号
            p.setPen(QPen(col, 2.4))
            seg = min(18.0, r.width() * 0.3, r.height() * 0.3)
            for cx, cy, dx, dy in ((r.left(), r.top(), 1, 1), (r.right(), r.top(), -1, 1),
                                   (r.left(), r.bottom(), 1, -1),
                                   (r.right(), r.bottom(), -1, -1)):
                p.drawLine(QPointF(cx, cy), QPointF(cx + dx * seg, cy))
                p.drawLine(QPointF(cx, cy), QPointF(cx, cy + dy * seg))

            # 标签
            label = f"{CLS_CN.get(d.label, d.label)}  {d.conf:.2f}"
            if confirmed and d.track_id is not None:
                label += f"  #{d.track_id:02d}"
            p.setFont(_mono(11, True))
            tw = p.fontMetrics().width(label) + 12
            tag = QRectF(r.left(), max(2.0, r.top() - 20), tw, 18)
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            p.drawRoundedRect(tag, 3, 3)
            p.setPen(_c("#08110c"))
            p.drawText(tag, Qt.AlignCenter, label)

            # 目标中心十字
            cx, cy = r.center().x(), r.center().y()
            p.setPen(QPen(_c(col.name(), 220), 1.0))
            p.drawLine(QPointF(cx - 6, cy), QPointF(cx + 6, cy))
            p.drawLine(QPointF(cx, cy - 6), QPointF(cx, cy + 6))

    def _paint_osd(self, p: QPainter) -> None:
        """OSD 画在控件坐标里，缩放裁切后仍然贴边可见。"""
        w, h = float(self.width()), float(self.height())
        cx, cy = w / 2, h / 2
        m = 16.0

        # 中心准星
        p.setPen(QPen(_c("#dff5e8", 120), 1.0))
        p.drawLine(QPointF(cx - 26, cy), QPointF(cx - 9, cy))
        p.drawLine(QPointF(cx + 9, cy), QPointF(cx + 26, cy))
        p.drawLine(QPointF(cx, cy - 26), QPointF(cx, cy - 9))
        p.drawLine(QPointF(cx, cy + 9), QPointF(cx, cy + 26))

        # 左上：REC + 模式
        p.setBrush(_c(theme.DANGER))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(m + 4, m + 9), 4, 4)
        p.setFont(_mono(11, True))
        p.setPen(_c(theme.TEXT, 235))
        p.drawText(QRectF(m + 14, m, w * 0.55, 18),
                   Qt.AlignLeft | Qt.AlignVCenter, "REC  " + self._mode)

        # 右上：帧号 / 时间 / fps
        p.setFont(_mono(10))
        p.setPen(_c(theme.TEXT_DIM, 235))
        stamp = f"F{self._frame_no:05d}  t={self._t:6.2f}s  {self._fps:4.1f}fps"
        p.drawText(QRectF(w - m - 260, m, 260, 18),
                   Qt.AlignRight | Qt.AlignVCenter, stamp)

        # 下沿信息条
        p.setFont(_mono(10))
        p.setPen(_c(theme.TEXT_DIM, 225))
        if self._tm is not None:
            tm = self._tm
            p.drawText(QRectF(m, h - m - 16, w * 0.55, 16),
                       Qt.AlignLeft | Qt.AlignVCenter,
                       f"云台 P{tm.gimbal_pitch:5.1f}°  Y{tm.gimbal_yaw:6.1f}°   "
                       f"HFOV 68°   ZOOM x1.0")
        p.setPen(_c(theme.ACCENT if self._dets else theme.TEXT_FAINT, 225))
        p.drawText(QRectF(w - m - 260, h - m - 16, 260, 16),
                   Qt.AlignRight | Qt.AlignVCenter,
                   "YOLOv8n 640×640 · 25 ms/帧")

        # 四角括号
        p.setPen(QPen(_c("#2b4457", 210), 1.4))
        L = 24.0
        for x, y, dx, dy in ((8, 8, 1, 1), (w - 8, 8, -1, 1),
                             (8, h - 8, 1, -1), (w - 8, h - 8, -1, -1)):
            p.drawLine(QPointF(x, y), QPointF(x + dx * L, y))
            p.drawLine(QPointF(x, y), QPointF(x, y + dy * L))


# --------------------------------------------------------------------------- #
# 地图
# --------------------------------------------------------------------------- #
class MapPanel(QWidget):
    """航迹 / 火点 / 视场 的俯视图，坐标由经纬度换算成 NED 米制后等比映射。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._tm = None
        self._route: List[Tuple[float, float]] = []   # 经纬度
        self._fires: List[tuple] = []                 # (lat, lon, code, state)
        self._pulse = 0.0
        self._base = (30.0, 114.0)
        self._rng = random.Random(77)
        self._contours = self._make_contours()

    def _make_contours(self):
        out = []
        for i in range(9):
            cx = self._rng.uniform(0.15, 0.85)
            cy = self._rng.uniform(0.15, 0.85)
            rad = self._rng.uniform(0.12, 0.40)
            pts = []
            phase = self._rng.uniform(0, 6.28)
            for k in range(48):
                a = 2 * math.pi * k / 48
                r = rad * (1.0 + 0.18 * math.sin(3 * a + phase) + 0.09 * math.sin(5 * a - phase))
                pts.append((cx + r * math.cos(a) * 1.25, cy + r * math.sin(a)))
            out.append(pts)
        return out

    def set_telemetry(self, tm) -> None:
        self._tm = tm
        self.update()

    def set_route(self, points: Sequence[Tuple[float, float]]) -> None:
        self._route = list(points)
        self.update()

    def set_fires(self, fires: Iterable[tuple]) -> None:
        self._fires = list(fires)
        self.update()

    def tick(self, dt: float) -> None:
        self._pulse = (self._pulse + dt * 0.9) % 1.0
        self.update()

    # ---------------- 坐标映射 ----------------
    def _bounds(self):
        lats = [p[0] for p in self._route] or [self._base[0] - 0.002, self._base[0] + 0.002]
        lons = [p[1] for p in self._route] or [self._base[1] - 0.002, self._base[1] + 0.002]
        for f in self._fires:
            lats.append(f[0])
            lons.append(f[1])
        if self._tm is not None:
            lats.append(self._tm.lat)
            lons.append(self._tm.lon)
        return min(lats), max(lats), min(lons), max(lons)

    def _project(self, lat: float, lon: float, rect: QRectF, bounds) -> QPointF:
        n, e = latlon_to_ned(lat, lon, self._base[0], self._base[1])
        la0, la1, lo0, lo1 = bounds
        n0, e0 = latlon_to_ned(la0, lo0, self._base[0], self._base[1])
        n1, e1 = latlon_to_ned(la1, lo1, self._base[0], self._base[1])
        span_n = max(1.0, n1 - n0)
        span_e = max(1.0, e1 - e0)
        pad = 40.0
        # 保持等比，避免地图被拉伸
        k = min((rect.width() - 2 * pad) / span_e, (rect.height() - 2 * pad) / span_n)
        cx = rect.center().x() - (e0 + e1) / 2 * k
        cy = rect.center().y() + (n0 + n1) / 2 * k
        return QPointF(cx + e * k, cy - n * k)

    def paintEvent(self, ev):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect())
        p.fillRect(rect, _c("#08111a"))

        bounds = self._bounds()
        # 经纬网格
        p.setPen(QPen(_c("#16293a", 255), 1.0))
        step = 40
        x = rect.left()
        while x < rect.right():
            p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            x += step
        y = rect.top()
        while y < rect.bottom():
            p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            y += step

        # 等高线（装饰性地形）
        for pts in self._contours:
            path = QPainterPath()
            first = True
            for u, v in pts:
                pt = QPointF(rect.left() + u * rect.width(), rect.top() + v * rect.height())
                if first:
                    path.moveTo(pt)
                    first = False
                else:
                    path.lineTo(pt)
            path.closeSubpath()
            p.setPen(QPen(_c("#1c3346", 235), 1.0))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)

        # 航线
        if len(self._route) > 1:
            path = QPainterPath()
            for i, (lat, lon) in enumerate(self._route):
                pt = self._project(lat, lon, rect, bounds)
                if i == 0:
                    path.moveTo(pt)
                else:
                    path.lineTo(pt)
            pen = QPen(_c("#3d7fa6", 235), 1.8)
            pen.setDashPattern([6, 6])
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)

        # 火点
        for lat, lon, code, state in self._fires:
            pt = self._project(lat, lon, rect, bounds)
            col = QColor(theme.DANGER if state == "CONFIRMED" else theme.WARN)
            for k, mul in enumerate((1.0, 1.9, 2.8)):
                a = int(150 * (1.0 - self._pulse) if k else 210)
                r = 6 + mul * 10 * (0.6 + 0.4 * self._pulse)
                p.setPen(QPen(_c(col.name(), max(24, a)), 1.4))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(pt, r, r)
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            p.drawEllipse(pt, 5.2, 5.2)

            p.setFont(_mono(10, True))
            p.setPen(_c(col.name(), 235))
            p.drawText(QPointF(pt.x() + 10, pt.y() - 8), code)

        # 视场锥
        if self._tm is not None:
            tm = self._tm
            uav = self._project(tm.lat, tm.lon, rect, bounds)
            heading = math.radians(90.0 - tm.gimbal_yaw)   # 屏幕 y 轴朝南
            half = math.radians(26.0)
            length = 78.0
            path = QPainterPath()
            path.moveTo(uav)
            path.lineTo(uav.x() + length * math.cos(heading - half),
                        uav.y() + length * math.sin(heading - half))
            path.lineTo(uav.x() + length * math.cos(heading + half),
                        uav.y() + length * math.sin(heading + half))
            path.closeSubpath()
            p.fillPath(path, _c(theme.ACCENT, 26))
            p.setPen(QPen(_c(theme.ACCENT, 110), 1.0))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)

            # 无人机
            yaw = math.radians(90.0 - tm.yaw)
            tri = QPolygonF([
                QPointF(uav.x() + 13 * math.cos(yaw), uav.y() + 13 * math.sin(yaw)),
                QPointF(uav.x() + 9 * math.cos(yaw + 2.5), uav.y() + 9 * math.sin(yaw + 2.5)),
                QPointF(uav.x(), uav.y()),
                QPointF(uav.x() + 9 * math.cos(yaw - 2.5), uav.y() + 9 * math.sin(yaw - 2.5)),
            ])
            p.setPen(Qt.NoPen)
            p.setBrush(_c("#e8f6ff"))
            p.drawPolygon(tri)
            p.setBrush(_c(theme.INFO))
            p.drawEllipse(uav, 3.2, 3.2)

            p.setFont(_mono(9))
            p.setPen(_c(theme.TEXT_DIM, 220))
            p.drawText(QPointF(uav.x() + 12, uav.y() + 16),
                       f"{tm.lat:.5f}, {tm.lon:.5f}  H{tm.H:.0f}m")

        # 指北针
        p.setPen(QPen(_c(theme.TEXT_DIM, 200), 1.2))
        nx, ny = rect.right() - 30, rect.top() + 34
        p.drawLine(QPointF(nx, ny + 14), QPointF(nx, ny - 10))
        p.setBrush(_c(theme.TEXT_DIM, 200))
        p.drawPolygon(QPolygonF([QPointF(nx, ny - 16), QPointF(nx - 5, ny - 6),
                                 QPointF(nx + 5, ny - 6)]))
        p.setFont(_mono(9, True))
        p.drawText(QPointF(nx - 3, ny + 26), "N")

        # 比例尺
        p.setPen(QPen(_c(theme.TEXT_FAINT, 220), 1.2))
        sx, sy = rect.left() + 16, rect.bottom() - 18
        p.drawLine(QPointF(sx, sy), QPointF(sx + 60, sy))
        p.drawLine(QPointF(sx, sy - 4), QPointF(sx, sy + 4))
        p.drawLine(QPointF(sx + 60, sy - 4), QPointF(sx + 60, sy + 4))
        p.setFont(_mono(9))
        p.setPen(_c(theme.TEXT_FAINT, 230))
        p.drawText(QPointF(sx + 66, sy + 3), "标尺")
        p.end()


# --------------------------------------------------------------------------- #
# 地平仪 + 姿态
# --------------------------------------------------------------------------- #
class HorizonPanel(QWidget):
    """小地平仪：滚转把地平线转起来，俯仰把地平线上下移。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(150, 110)
        self._roll = 0.0
        self._pitch = 0.0

    def set_attitude(self, roll: float, pitch: float) -> None:
        self._roll = roll
        self._pitch = pitch
        self.update()

    def paintEvent(self, ev):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        clip = QPainterPath()
        clip.addRoundedRect(rect, 8, 8)
        p.setClipPath(clip)
        p.fillRect(rect, _c("#0a1119"))

        cx, cy = rect.center().x(), rect.center().y()
        p.translate(cx, cy)
        p.rotate(-self._roll)
        shift = max(-28.0, min(28.0, self._pitch * 1.6))
        p.translate(0, shift)
        span = max(self.width(), self.height())

        p.fillRect(QRectF(-span, -span, span * 2, span * 2), _c("#17313f"))
        p.fillRect(QRectF(-span, 0, span * 2, span * 2), _c("#2a2317"))
        p.setPen(QPen(_c("#dfeaf2", 200), 1.4))
        p.drawLine(QPointF(-span, 0), QPointF(span, 0))
        # 俯仰刻度
        p.setFont(_mono(8))
        for deg in (-20, -10, 10, 20):
            y = -deg * 1.6
            p.setPen(QPen(_c("#dfeaf2", 120), 1.0))
            p.drawLine(QPointF(-16, y), QPointF(16, y))
            if deg % 20 == 0:
                p.setPen(_c("#dfeaf2", 150))
                p.drawText(QRectF(-52, y - 7, 30, 14), Qt.AlignRight | Qt.AlignVCenter,
                           f"{deg}")
        p.resetTransform()

        # 固定机头符号
        p.setPen(QPen(_c(theme.ACCENT, 235), 2.0))
        p.drawLine(QPointF(cx - 26, cy), QPointF(cx - 8, cy))
        p.drawLine(QPointF(cx + 8, cy), QPointF(cx + 26, cy))
        p.drawLine(QPointF(cx, cy), QPointF(cx, cy + 5))

        p.setFont(_mono(9, True))
        p.setPen(_c(theme.TEXT_DIM, 220))
        p.drawText(QRectF(rect.left() + 8, rect.top() + 4, 70, 14),
                   Qt.AlignLeft | Qt.AlignVCenter, f"R {self._roll:+.1f}°")
        p.drawText(QRectF(rect.right() - 78, rect.top() + 4, 70, 14),
                   Qt.AlignRight | Qt.AlignVCenter, f"P {self._pitch:+.1f}°")
        p.end()


# --------------------------------------------------------------------------- #
# 告警状态机条
# --------------------------------------------------------------------------- #
STATES = [
    (IDLE, "空闲"),
    (MONITORING, "监测中"),
    (SUSPECT, "疑似"),
    (CONFIRMED, "已确认"),
    (RESOLVED, "已处置"),
    (DISMISSED, "误报"),
]
STATE_COLOR = {IDLE: theme.IDLE, MONITORING: theme.ACCENT, SUSPECT: theme.WARN,
               CONFIRMED: theme.DANGER, RESOLVED: theme.INFO, DISMISSED: "#7b6a8f"}


class FsmStrip(QWidget):
    """把 alarm.TRANSITIONS 画成一条状态带，当前态高亮。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(64)
        self.setFixedHeight(64)
        self._state = IDLE
        self._flash = 0.0

    def set_state(self, state: str) -> None:
        if state != self._state:
            self._flash = 1.0
        self._state = state
        self.update()

    def tick(self, dt: float) -> None:
        if self._flash > 0:
            self._flash = max(0.0, self._flash - dt * 2.2)
            self.update()

    def paintEvent(self, ev):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect())
        p.fillRect(rect, Qt.transparent)

        n = len(STATES)
        gap = 8.0
        w = (rect.width() - gap * (n - 1)) / n
        h = 30.0
        y = rect.center().y() - h / 2 + 4
        for i, (key, label) in enumerate(STATES):
            x = rect.left() + i * (w + gap)
            box = QRectF(x, y, w, h)
            active = key == self._state
            col = QColor(STATE_COLOR[key])
            if active:
                fill = QColor(col)
                fill.setAlpha(235)
                p.setPen(Qt.NoPen)
                p.setBrush(fill)
                p.drawRoundedRect(box, 15, 15)
                p.setPen(_c("#04140e" if key != CONFIRMED else "#1a0505"))
                f = _ui(11, True)
            else:
                p.setPen(QPen(_c(col.name(), 120), 1.0))
                p.setBrush(_c("#0a1119", 160))
                p.drawRoundedRect(box, 15, 15)
                p.setPen(_c(col.name(), 150))
                f = _ui(10, False)
            p.setFont(f)
            p.drawText(box, Qt.AlignCenter, label)

            if i < n - 1:
                ax = x + w + 1
                p.setPen(QPen(_c("#26404f", 220), 1.0))
                p.drawLine(QPointF(ax, y + h / 2), QPointF(ax + gap - 2, y + h / 2))
                p.setBrush(_c("#26404f", 220))
                p.drawPolygon(QPolygonF([
                    QPointF(ax + gap - 2, y + h / 2),
                    QPointF(ax + gap - 5, y + h / 2 - 2.5),
                    QPointF(ax + gap - 5, y + h / 2 + 2.5)]))
        p.end()


# --------------------------------------------------------------------------- #
# 指标小图
# --------------------------------------------------------------------------- #
class ErrorChart(QWidget):
    """定位误差随高度变化的折线图（含 30 m 指标线）。"""

    def __init__(self, rows=None, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self._rows = rows or []
        self._budget = 30.0

    def set_rows(self, rows) -> None:
        self._rows = list(rows)
        self.update()

    def paintEvent(self, ev):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(36, 12, -10, -22)
        p.fillRect(QRectF(self.rect()), Qt.transparent)

        p.setPen(QPen(_c("#1b2b3a", 255), 1.0))
        p.setFont(_mono(8))
        ymax = 32.0
        for k in range(5):
            val = ymax * k / 4
            y = rect.bottom() - rect.height() * k / 4
            p.setPen(QPen(_c("#16232f", 255), 1.0))
            p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            p.setPen(_c(theme.TEXT_FAINT, 220))
            p.drawText(QRectF(0, y - 7, 32, 14), Qt.AlignRight | Qt.AlignVCenter,
                       f"{val:.0f}")

        # 指标线 30 m
        y = rect.bottom() - rect.height() * (self._budget / ymax)
        pen = QPen(_c(theme.DANGER, 160), 1.0)
        pen.setDashPattern([4, 4])
        p.setPen(pen)
        p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        p.setFont(_mono(8))
        p.setPen(_c(theme.DANGER, 200))
        p.drawText(QPointF(rect.left() + 4, y - 4), "指标上限 30 m")

        if not self._rows:
            p.end()
            return
        n = len(self._rows)
        dx = rect.width() / max(1, n - 1)
        pts_mean, pts_p95 = [], []
        for i, r in enumerate(self._rows):
            x = rect.left() + dx * i
            pts_mean.append(QPointF(x, rect.bottom() - rect.height() * (r["mean"] / ymax)))
            pts_p95.append(QPointF(x, rect.bottom() - rect.height() * (r["p95"] / ymax)))
            p.setFont(_mono(8))
            p.setPen(_c(theme.TEXT_FAINT, 220))
            p.drawText(QRectF(x - 20, rect.bottom() + 4, 40, 14), Qt.AlignCenter,
                       f"{r['height']} m")

        p.setPen(QPen(_c(theme.INFO, 200), 1.4))
        p.setBrush(Qt.NoBrush)
        p.drawPolyline(QPolygonF(pts_p95))
        p.setPen(QPen(_c(theme.ACCENT, 240), 2.0))
        p.drawPolyline(QPolygonF(pts_mean))
        for pt in pts_mean:
            p.setBrush(_c(theme.ACCENT))
            p.setPen(Qt.NoPen)
            p.drawEllipse(pt, 3.0, 3.0)
        for i, r in enumerate(self._rows):
            p.setFont(_mono(8, True))
            p.setPen(_c(theme.ACCENT, 235))
            p.drawText(QPointF(pts_mean[i].x() - 14, pts_mean[i].y() - 7),
                       f"{r['mean']:.2f}")
        p.end()


class LatencyChart(QWidget):
    """端到端时延预算的横向堆叠条。"""

    def __init__(self, budget=None, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self._budget = budget or {"stages": [], "total_s": 0.0, "budget_s": 5.0}

    def set_budget(self, budget) -> None:
        self._budget = budget
        self.update()

    def paintEvent(self, ev):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        stages = self._budget.get("stages", [])
        total = max(1e-6, sum(ms for _, ms in stages))
        colors = ["#2b4a5f", "#356b7e", "#3f8b7c", "#4ea27f",
                  "#c98a3c", "#a86b6b", "#e0574f"]

        x0, x1 = 12.0, self.width() - 12.0
        bar_y, bar_h = 26.0, 18.0
        span = x1 - x0
        p.setFont(_mono(9))
        acc = 0.0
        for i, (name, ms) in enumerate(stages):
            w = span * ms / total
            r = QRectF(x0 + span * acc / total, bar_y, max(1.5, w - 1.5), bar_h)
            p.setPen(Qt.NoPen)
            p.setBrush(_c(colors[i % len(colors)], 235))
            p.drawRoundedRect(r, 2, 2)
            acc += ms

        # 分段标注
        p.setFont(_mono(8))
        acc = 0.0
        for i, (name, ms) in enumerate(stages):
            w = span * ms / total
            if w > 62:
                p.setPen(_c(theme.TEXT_DIM, 230))
                p.drawText(QRectF(x0 + span * acc / total, bar_y + bar_h + 4, w, 14),
                           Qt.AlignLeft | Qt.AlignVCenter, name)
            acc += ms

        lat = self._budget
        p.setFont(_mono(11, True))
        p.setPen(_c(theme.ACCENT, 240))
        p.drawText(QRectF(x0, 2, span, 18), Qt.AlignRight | Qt.AlignVCenter,
                   f"合计 {lat.get('total_s', 0):.2f} s / 预算 {lat.get('budget_s', 5):.0f} s")

        # 逐段条形明细（最长 6 段）
        p.setFont(_mono(8))
        yy = bar_y + bar_h + 24
        maxms = max([ms for _, ms in stages] + [1.0])
        for i, (name, ms) in enumerate(stages[:6]):
            if yy > self.height() - 10:
                break
            p.setPen(_c(theme.TEXT_FAINT, 235))
            p.drawText(QRectF(x0, yy - 7, 132, 14), Qt.AlignLeft | Qt.AlignVCenter, name)
            bw = (span - 190) * ms / maxms
            p.setPen(Qt.NoPen)
            p.setBrush(_c(colors[i % len(colors)], 210))
            p.drawRoundedRect(QRectF(x0 + 140, yy - 5, max(1.5, bw), 10), 2, 2)
            p.setPen(_c(theme.TEXT_DIM, 235))
            p.drawText(QRectF(x0 + 140 + bw + 6, yy - 7, 60, 14),
                       Qt.AlignLeft | Qt.AlignVCenter, f"{ms:.0f} ms")
            yy += 17
        p.end()


class PrChart(QWidget):
    """PR 曲线（评测口径可视化，非模型实测）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self._rec = []
        self._prec = []

    def set_curve(self, rec, prec) -> None:
        self._rec, self._prec = list(rec), list(prec)
        self.update()

    def paintEvent(self, ev):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(30, 12, -10, -22)
        p.setPen(QPen(_c("#16232f", 255), 1.0))
        p.setFont(_mono(8))
        for k in range(5):
            x = rect.left() + rect.width() * k / 4
            y = rect.bottom() - rect.height() * k / 4
            p.setPen(QPen(_c("#16232f", 255), 1.0))
            p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        if not self._rec:
            p.end()
            return
        pts = [QPointF(rect.left() + rect.width() * r, rect.bottom() - rect.height() * q)
               for r, q in zip(self._rec, self._prec)]
        path = QPainterPath()
        path.moveTo(rect.left(), rect.bottom())
        for pt in pts:
            path.lineTo(pt)
        path.lineTo(rect.right(), rect.bottom())
        path.closeSubpath()
        p.fillPath(path, _c(theme.ACCENT, 30))
        p.setPen(QPen(_c(theme.ACCENT, 235), 1.6))
        p.drawPolyline(QPolygonF(pts))
        p.setPen(_c(theme.TEXT_FAINT, 235))
        p.drawText(QRectF(rect.left(), rect.bottom() + 4, rect.width(), 14),
                   Qt.AlignCenter, "召回率 Recall")
        p.end()