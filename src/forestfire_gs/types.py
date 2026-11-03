# -*- coding: utf-8 -*-
"""跨模块共享的数据结构。

设计原则：检测、定位、告警、界面四层之间只通过这几个结构交换信息，
任何一个模块都能被替换而不影响其他模块（论文 2.4 节）。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Detection:
    """一次检测结果。box 为像素坐标 (x1, y1, x2, y2)。"""

    box: tuple
    cls: int = 0                 # 0 火焰 / 1 烟雾
    conf: float = 0.0
    track_id: int = -1           # 由时序确认给出的轨迹编号，-1 表示未关联
    hits: int = 0                # 窗口内命中次数
    t: float = 0.0               # 帧时间戳（秒，单调时钟）

    # ---- 几何便捷属性 ----
    @property
    def center(self) -> tuple:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def label(self) -> str:
        return "火焰" if self.cls == 0 else "烟雾"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["box"] = list(self.box)
        return d


@dataclass
class Telemetry:
    """一帧遥测。角度单位为度，高度为相对起飞点的对地高度（米）。"""

    t: float = 0.0
    lat: float = 30.0000
    lon: float = 114.0000
    H: float = 100.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    gimbal_pitch: float = 30.0
    gimbal_yaw: float = 0.0
    ground_speed: float = 0.0
    battery: float = 100.0
    satellites: int = 18

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class FireFix:
    """火点解算结果：经纬度 + 相对无人机的北东位移 + 当时的遥测快照。"""

    lat: float
    lon: float
    dn: float
    de: float
    range_m: float
    telemetry: Telemetry
    detection: Detection
    skew_s: float = 0.0

    @property
    def coords(self) -> str:
        return f"{self.lat:.6f}, {self.lon:.6f}"

    def as_dict(self) -> dict:
        return {"lat": self.lat, "lon": self.lon, "dn": self.dn, "de": self.de,
                "range_m": self.range_m, "skew_s": self.skew_s,
                "telemetry": self.telemetry.as_dict(),
                "detection": self.detection.as_dict()}


@dataclass
class AlarmRecord:
    """一条告警记录，贯穿状态机与界面列表。"""

    seq: int
    created_at: float
    fix: FireFix
    state: str = "确认告警"
    pushed: bool = False
    resolved_at: Optional[float] = None
    dismissed: bool = False
    note: str = ""
    events: list = field(default_factory=list)

    @property
    def code(self) -> str:
        return f"FF-{int(self.fix.telemetry.t)}-{self.seq:03d}"

    def as_dict(self) -> dict:
        return {"seq": self.seq, "created_at": self.created_at, "state": self.state,
                "pushed": self.pushed, "dismissed": self.dismissed,
                "resolved_at": self.resolved_at, "note": self.note,
                "fix": self.fix.as_dict(), "events": list(self.events)}