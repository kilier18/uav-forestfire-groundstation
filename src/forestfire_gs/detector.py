# -*- coding: utf-8 -*-
"""AI 火情视觉检测（对应论文 3.3 节、附录 A.1）。

- ``iou``              计算两框交并比
- ``YoloDetector``     YOLOv8 推理封装（延迟导入 ultralytics / onnxruntime，未装也能 import）
- ``MockDetector``     模拟检测器，与 YoloDetector 同接口，用于无模型离线联调与演示
- ``TemporalConfirmer`` 滑动窗口连续帧确认，压雾、山岚、逆光造成的单帧误报

依赖：numpy（只有走 YOLO 路径时才需要 ultralytics）
"""
from __future__ import annotations

import numpy as np

from .types import Detection


def _box(x):
    """允许传 Detection 或 (x1, y1, x2, y2) 元组，统一取出坐标。"""
    return x.box if isinstance(x, Detection) else tuple(x)


def iou(a, b) -> float:
    """a、b 均为 (x1, y1, x2, y2)，返回交并比；无交集返回 0。"""
    ax1, ay1, ax2, ay2 = _box(a)
    bx1, by1, bx2, by2 = _box(b)
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


class YoloDetector:
    """YOLOv8 推理封装。ultralytics 在构造时才导入，避免无 GPU 环境导入失败。

    用法::

        det = YoloDetector("weights/fire_yolov8n.pt")
        for d in det.predict(frame_bgr):
            print(d.box, d.cls, d.conf)
    """

    def __init__(self, model_path: str, conf: float = 0.25,
                 nms_iou: float = 0.50, imgsz: int = 640):
        from ultralytics import YOLO          # 延迟导入
        self.model = YOLO(model_path)
        self.conf, self.nms_iou, self.imgsz = conf, nms_iou, imgsz

    def predict(self, frame, t: float = 0.0):
        r = self.model.predict(frame, conf=self.conf, iou=self.nms_iou,
                               imgsz=self.imgsz, verbose=False)[0]
        if r.boxes is None:
            return []
        return [Detection(box=tuple(float(v) for v in b.xyxy[0].tolist()),
                          cls=int(b.cls[0]),               # 0=火焰，1=烟雾
                          conf=float(b.conf[0]), t=t) for b in r.boxes]


class MockDetector:
    """模拟检测器。

    按 ``fire_rate`` 的概率在预设火点附近生成一个带抖动的检测框（持续火情），
    按 ``false_alarm_rate`` 的概率随机生成一个低置信度框（雾、山岚引起的单帧误报）。
    对外接口与 YoloDetector 一致，可在未加载模型时替换使用。
    """

    def __init__(self, seed: int = 0, fire_center=(960, 480), radius: float = 70,
                 fire_rate: float = 0.85, false_alarm_rate: float = 0.08,
                 secondary_center=None, class_names=("flame", "smoke")):
        self.rng = np.random.default_rng(seed)
        self.fire_center, self.radius = tuple(fire_center), radius
        self.fire_rate, self.false_alarm_rate = fire_rate, false_alarm_rate
        self.secondary_center = secondary_center
        self.class_names = class_names

    def predict(self, frame=None, t: float = 0.0):
        out = []
        if self.secondary_center and self.rng.random() < 0.55:
            sx, sy = self.secondary_center
            dx, dy = self.rng.normal(0, 5, 2)
            out.append(Detection(box=(sx - 46 + dx, sy - 40 + dy, sx + 46 + dx, sy + 40 + dy),
                                 cls=1, conf=float(np.clip(self.rng.normal(0.74, 0.05), 0.3, 0.99)),
                                 t=t))
        if self.rng.random() < self.fire_rate:            # 持续火情：几乎每帧都有
            cx, cy = self.fire_center
            dx, dy = self.rng.normal(0, 4, 2)             # 检测框抖动
            r = self.radius
            conf = float(np.clip(self.rng.normal(0.88, 0.05), 0.3, 0.99))
            out.append(Detection(box=(cx - r + dx, cy - r * 0.8 + dy,
                                      cx + r + dx, cy + r * 0.9 + dy),
                                 cls=0, conf=conf, t=t))
        if self.rng.random() < self.false_alarm_rate:     # 偶发误检，低置信度
            x, y = self.rng.uniform(120, 800), self.rng.uniform(120, 520)
            out.append(Detection(box=(x, y, x + 54, y + 44), cls=1,
                                 conf=float(self.rng.uniform(0.25, 0.45)), t=t))
        return out


class TemporalConfirmer:
    """滑动窗口连续帧确认。

    用 IoU 把相邻帧中的同类目标关联为轨迹，同一轨迹在 ``window`` 帧内命中次数
    达到 ``min_hits`` 才判定为确认目标。偶发误检命中次数不足，会随窗口滑出被丢弃；
    持续火情则保留（论文图 3-6）。
    """

    def __init__(self, window: int = 5, min_hits: int = 3, iou_thr: float = 0.30):
        self.window, self.min_hits, self.iou_thr = window, min_hits, iou_thr
        self.tracks: list = []        # 每条轨迹：box / hits / age / cls / id
        self._next_id = 1

    def update(self, dets):
        """输入本帧检测结果，返回已确认目标（Detection 列表，带 track_id 与 hits）。"""
        used = [False] * len(dets)

        # ① 已有轨迹与当前检测做 IoU 匹配，取最大且过阈值者
        for tr in self.tracks:
            best, idx = 0.0, -1
            for i, d in enumerate(dets):
                if used[i] or d.cls != tr["cls"]:
                    continue
                v = iou(tr["box"], d.box)
                if v > best:
                    best, idx = v, i
            tr["age"] += 1
            if idx >= 0 and best >= self.iou_thr:
                d = dets[idx]
                tr["box"] = tuple(d.box)
                tr["hits"] += 1
                d.track_id, d.hits = tr["id"], tr["hits"]
                used[idx] = True

        # ② 未匹配上的检测新建轨迹
        for i, d in enumerate(dets):
            if used[i]:
                continue
            self.tracks.append({"id": self._next_id, "box": tuple(d.box), "hits": 1,
                                "age": 1, "cls": d.cls})
            d.track_id, d.hits = self._next_id, 1
            self._next_id += 1

        # ③ 连续 window 帧未命中的轨迹淘汰
        self.tracks = [t for t in self.tracks if t["age"] - t["hits"] < self.window]

        return [d for d in dets if d.hits >= self.min_hits]

    def reset(self):
        self.tracks.clear()
        self._next_id = 1

    # ---- 便于统计误报抑制效果 ----
    @property
    def statistics(self) -> dict:
        confirmed = [t for t in self.tracks if t["hits"] >= self.min_hits]
        return {"tracks": len(self.tracks), "confirmed": len(confirmed),
                "window": self.window, "min_hits": self.min_hits}


if __name__ == "__main__":
    # 自检：30 帧模拟数据，看单帧误报能否被滤掉
    det = MockDetector(seed=1)
    tc = TemporalConfirmer(window=5, min_hits=3)
    raw = kept = 0
    for i in range(30):
        dets = det.predict(t=i * 0.04)
        out = tc.update(dets)
        raw += len(dets)
        kept += len(out)
        if out:
            print(f"帧{i:02d} 检测{len(dets)}个 确认{len(out)}个")
    print(f"30 帧共检出 {raw} 个候选，经时序确认保留 {kept} 个")