# -*- coding: utf-8 -*-
"""集中配置模块（对应论文 3.1 节"配置集中、逻辑分层、界面独立"）。

所有可调参数都在这里，检测 / 定位 / 告警三个模块只读配置，不反向依赖界面。
默认值与论文第 2、3、4 章一致：

    相机内参      fx = fy = 1400 px，主点 (960, 540)，1920×1080
    检测          conf = 0.25，NMS IoU = 0.50，输入 640×640
    时序确认      窗口 5 帧、命中 3 次、IoU 关联阈值 0.30
    告警          冷却 30 s，空间去重 60 m
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
WEIGHTS_DIR = ROOT / "weights"


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw) if raw else default


@dataclass(frozen=True)
class CameraIntrinsics:
    """针孔相机内参，单位像素。fx / fy 由标定或镜头参数给出。"""

    width: int = 1920
    height: int = 1080
    fx: float = 1400.0
    fy: float = 1400.0
    cx: float = 960.0
    cy: float = 540.0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class DetectionConfig:
    """检测模块参数。conf 不是一个固定死值：清晨傍晚可下调，正午上调。"""

    model_path: str = str(WEIGHTS_DIR / "fire_yolov8n.onnx")
    conf: float = 0.25
    nms_iou: float = 0.50
    imgsz: int = 640
    class_names: tuple = ("flame", "smoke")   # 0 = 火焰, 1 = 烟雾


@dataclass
class TrackConfig:
    """滑动窗口连续帧确认参数，见论文 3.3 节。"""

    window: int = 5
    min_hits: int = 3
    iou_thr: float = 0.30


@dataclass
class AlarmConfig:
    """告警状态机参数。空间阈值用于"同一火点"判定，避免扩展前后重复推送。"""

    push_cooldown: float = 30.0
    spatial_gate_m: float = 60.0
    telemetry_max_skew_s: float = 0.5


@dataclass
class StationConfig:
    """地面站整体配置。可 json 落盘，界面"参数"面板改动后即时生效。"""

    camera: CameraIntrinsics = field(default_factory=CameraIntrinsics)
    detect: DetectionConfig = field(default_factory=DetectionConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    alarm: AlarmConfig = field(default_factory=AlarmConfig)

    video_source: str = "sim://patrol-01"        # 也支持 0 / rtsp://... / *.mp4
    db_path: str = str(DATA_DIR / "station.db")
    screenshot_dir: str = str(DATA_DIR / "shots")
    seed: int = 20261103
    target_fps: int = 25

    # ---------------- 便捷属性 ----------------
    @property
    def queue_size(self) -> int:
        return 4

    # ---------------- 持久化 ----------------
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path | str | None = None) -> Path:
        path = Path(path) if path else DATA_DIR / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str | None = None) -> "StationConfig":
        path = Path(path) if path else DATA_DIR / "config.json"
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        cam = CameraIntrinsics(**raw.get("camera", {}))
        det = DetectionConfig(**raw.get("detect", {}))
        if isinstance(det.class_names, list):
            det.class_names = tuple(det.class_names)
        return cls(camera=cam, detect=det,
                   track=TrackConfig(**raw.get("track", {})),
                   alarm=AlarmConfig(**raw.get("alarm", {})),
                   video_source=raw.get("video_source", "sim://patrol-01"),
                   db_path=raw.get("db_path", str(DATA_DIR / "station.db")),
                   screenshot_dir=raw.get("screenshot_dir", str(DATA_DIR / "shots")),
                   seed=raw.get("seed", 20261103),
                   target_fps=raw.get("target_fps", 25))


DEFAULT_CONFIG = StationConfig()