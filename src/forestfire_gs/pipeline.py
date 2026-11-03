# -*- coding: utf-8 -*-
"""采集 / 推理 / 呈现 三线程流水线（对应论文 2.4 节线程模型、图 2-2）。

    采集线程  读视频帧与遥测，分别放进两个队列（队列满时丢最旧一帧）
    推理线程  从帧队列取帧 → 预处理 → 推理 → 后处理 → 时序确认 → 定位
    界面线程  只消费结果信号，不参与计算

队列长度故意设得小：延迟比完整性重要（论文 2.5 节）。
本模块不依赖 Qt，界面层通过 ``listener`` 回调接收事件，因此也能跑在无头环境。
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from .alarm import AlarmManager
from .config import StationConfig
from .detector import MockDetector, TemporalConfirmer, YoloDetector
from .localization import Locator
from .telemetry import PatrolSimulator
from .types import Detection, Telemetry


class LatestQueue:
    """定长队列，满时丢弃最旧元素，保证界面看到的总是最新数据。"""

    def __init__(self, maxsize: int = 4):
        self.q: queue.Queue = queue.Queue(maxsize=maxsize)

    def put(self, item):
        try:
            self.q.put_nowait(item)
        except queue.Full:
            try:
                self.q.get_nowait()          # 丢最旧
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(item)
            except queue.Full:
                pass

    def get(self, timeout: float = 0.5):
        return self.q.get(timeout=timeout)

    def clear(self):
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                return


@dataclass
class FramePacket:
    index: int
    t: float
    image: object = None            # BGR ndarray 或 None（仿真时界面自行绘制）


@dataclass
class FrameResult:
    """推理线程交给界面线程的一帧结果。"""

    index: int
    t: float
    telemetry: Telemetry
    dets: list = field(default_factory=list)          # 本帧全部候选
    confirmed: list = field(default_factory=list)     # 通过时序确认的目标
    fixes: list = field(default_factory=list)         # 定位结果
    alarm: dict | None = None                         # 新产生的告警
    infer_ms: float = 0.0
    fps: float = 0.0


class StationEngine:
    """地面站运行时。界面按钮 → 本类方法 → 线程 → 回调。"""

    def __init__(self, cfg: StationConfig | None = None, detector=None,
                 listener=None, simulate_infer_ms: float = 25.0):
        self.cfg = cfg or StationConfig()
        self.listener = listener
        self.simulate_infer_ms = simulate_infer_ms

        # 合成视频把火盆画在画面 60% 高度处，烟雾柱在其上方；检测框对齐这两个位置
        self.detector = detector or MockDetector(seed=self.cfg.seed,
                                                 fire_center=(960, 645),
                                                 radius=64,
                                                 secondary_center=(1010, 300))
        self.confirmer = TemporalConfirmer(**vars(self.cfg.track))
        self.locator = Locator(self.cfg.camera, self.cfg.alarm.telemetry_max_skew_s)
        self.alarm = AlarmManager(push_cooldown=self.cfg.alarm.push_cooldown,
                                  spatial_gate_m=self.cfg.alarm.spatial_gate_m)
        self.sim = PatrolSimulator(seed=self.cfg.seed)

        self.frame_q = LatestQueue(self.cfg.queue_size)
        self.tm_history = deque(maxlen=120)
        self._tm_lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list = []
        self.running = False
        self._last = 0.0
        self._fps = 0.0
        self._drop = 0

    # ---------------- 生命周期 ----------------
    def start(self):
        if self.running:
            return
        self._stop.clear()
        self.confirmer.reset()
        self.alarm.start()
        self.running = True
        self._threads = [threading.Thread(target=self._capture_loop, name="capture", daemon=True),
                         threading.Thread(target=self._infer_loop, name="infer", daemon=True)]
        for t in self._threads:
            t.start()
        self._emit("state", {"running": True, "fsm": self.alarm.fsm.state})

    def stop(self):
        if not self.running:
            return
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.0)
        self.running = False
        self.frame_q.clear()
        self._emit("state", {"running": False, "fsm": self.alarm.fsm.state})

    def shutdown(self):
        self.stop()

    # ---------------- 采集线程 ----------------
    def _capture_loop(self):
        period = 1.0 / max(1, self.cfg.target_fps)
        idx = 0
        while not self._stop.is_set():
            tm = self.sim.step(period)
            with self._tm_lock:
                self.tm_history.append(tm)
            self.frame_q.put(FramePacket(index=idx, t=tm.t))
            idx += 1
            time.sleep(period)

    # ---------------- 推理线程 ----------------
    def _infer_loop(self):
        last_t = time.time()
        while not self._stop.is_set():
            try:
                pkt = self.frame_q.get(timeout=0.3)
            except queue.Empty:
                continue
            t0 = time.perf_counter()

            dets = self.detector.predict(None, t=pkt.t)
            for d in dets:
                d.t = pkt.t
            confirmed = self.confirmer.update(dets)

            with self._tm_lock:
                history = list(self.tm_history)
            tm = self.locator.nearest_telemetry(pkt.t, history) or (history[-1] if history else None)

            fixes = []
            for d in confirmed:
                fx = self.locator.fix(d, tm)
                if fx:
                    fixes.append(fx)

            alarm = self.alarm.feed(confirmed, fixes, now=pkt.t)

            # 模拟推理耗时，使界面帧率与论文 25 fps 的口径一致
            cost = self.simulate_infer_ms
            time.sleep(min(0.06, cost / 1000.0))
            infer_ms = (time.perf_counter() - t0) * 1000.0

            now = time.time()
            dt = now - last_t
            last_t = now
            if dt > 0:
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt) if self._fps else 1.0 / dt

            res = FrameResult(index=pkt.index, t=pkt.t,
                              telemetry=tm or Telemetry(t=pkt.t),
                              dets=dets, confirmed=confirmed, fixes=fixes,
                              alarm=alarm, infer_ms=infer_ms, fps=self._fps)
            self._last = pkt.t
            self._emit("frame", res)

    # ---------------- 操作员动作 ----------------
    def resolve(self):
        r = self.alarm.resolve()
        self._emit("alarm_update", r or {})
        return r

    def dismiss(self):
        r = self.alarm.dismiss()
        self._emit("alarm_update", r or {})
        return r

    def snapshot(self) -> dict:
        return {"running": self.running, "fps": self._fps, "t": self._last,
                "fsm": self.alarm.fsm.state, "stats": self.alarm.stats()}

    # ---------------- 回调 ----------------
    def _emit(self, name: str, payload):
        if self.listener:
            try:
                self.listener(name, payload)
            except Exception:
                pass                     # 界面异常不应拖垮推理线程


class HeadlessReporter:
    """无界面演示用：把事件汇总成控制台输出（论文 4.4 节端到端演示）。"""

    def __init__(self):
        self.frames = 0
        self.alarms = 0
        self.first_confirm_frame = None
        self.rows = []

    def __call__(self, name: str, payload):
        if name == "frame":
            self.frames += 1
            if payload.confirmed and self.first_confirm_frame is None:
                self.first_confirm_frame = payload.index
            if payload.index < 16:
                self.rows.append((payload.index, len(payload.dets), len(payload.confirmed),
                                  len(payload.fixes), payload.telemetry.H, payload.telemetry.gimbal_pitch))
            if payload.alarm:
                self.alarms += 1
        elif name == "state":
            pass


if __name__ == "__main__":
    rep = HeadlessReporter()
    eng = StationEngine(listener=rep)
    eng.start()
    time.sleep(3.0)
    eng.stop()
    print(f"{'帧':>3} {'候选':>4} {'确认':>4} {'定位':>4} {'高度/m':>7} {'云台俯仰/°':>10}")
    for r in rep.rows:
        print(f"{r[0]:3d} {r[1]:4d} {r[2]:4d} {r[3]:4d} {r[4]:7.1f} {r[5]:10.1f}")
    print(f"共处理 {rep.frames} 帧，产生告警 {rep.alarms} 次，"
          f"首次确认在第 {rep.first_confirm_frame} 帧")