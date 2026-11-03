# -*- coding: utf-8 -*-
"""告警与处置状态机（对应论文 3.5 节、附录 A.3、图 3-3）。

把"什么时候该告警"从模糊判断变成明确的迁移条件：

    待机 → 监测中（开始监测）
    监测中 → 疑似（时序确认出目标）
    疑似 → 确认告警（判定通过，推送）
    确认告警 → 处置完成 / 撤销（操作员动作）

推送设冷却期，同一火点（空间距离小于 spatial_gate_m）在冷却期内不重复推；
撤销记录用于统计误报率。
"""
from __future__ import annotations

import math
import time

IDLE, MONITORING = "待机", "监测中"
SUSPECT, CONFIRMED = "疑似", "确认告警"
RESOLVED, DISMISSED = "处置完成", "撤销"

STATES = (IDLE, MONITORING, SUSPECT, CONFIRMED, RESOLVED, DISMISSED)

# 允许的迁移：当前状态 → {动作: 目标状态}
TRANSITIONS = {
    IDLE:       {"start": MONITORING},
    MONITORING: {"suspect": SUSPECT, "stop": IDLE},
    SUSPECT:    {"confirm": CONFIRMED, "reject": MONITORING},
    CONFIRMED:  {"resolve": RESOLVED, "dismiss": DISMISSED},
    RESOLVED:   {"reset": MONITORING},
    DISMISSED:  {"reset": MONITORING},
}


class AlarmFSM:
    """六状态告警状态机。非法迁移直接抛错，便于测试发现逻辑问题。"""

    def __init__(self, push_cooldown: float = 30.0):
        self.state = IDLE
        self.events: list = []            # 迁移日志，供回放与统计
        self.push_cooldown = push_cooldown
        self.last_push_ts = 0.0
        self.push_count = 0
        self.push_log: list = []

    def _fire(self, action: str, detail: str = "") -> str:
        nxt = TRANSITIONS.get(self.state, {}).get(action)
        if nxt is None:
            raise ValueError(f"非法迁移：{self.state} --{action}--> ?")
        self.events.append({"from": self.state, "action": action,
                            "to": nxt, "detail": detail})
        self.state = nxt
        return nxt

    # ---------- 对外接口 ----------
    def start(self):
        return self._fire("start")

    def stop(self):
        return self._fire("stop")

    def on_detection(self, confirmed):
        """检测线程回调：监测中且有确认目标 → 进疑似。"""
        if self.state == MONITORING and confirmed:
            return self._fire("suspect", f"确认目标 {len(confirmed)} 个")
        return self.state

    def on_suspect_pass(self, payload: dict | None = None):
        """疑似判定通过 → 确认告警；冷却期内不重复推送。"""
        if self.state != SUSPECT:
            return self.state
        st = self._fire("confirm")
        now = time.time()
        if now - self.last_push_ts >= self.push_cooldown:
            self.last_push_ts = now
            self.push_count += 1
            self._push(payload or {})
        return st

    def reject(self):
        return self._fire("reject")

    def resolve(self):
        return self._fire("resolve")

    def dismiss(self):
        """操作员标记误报，记入误报率统计。"""
        return self._fire("dismiss")

    def reset(self):
        return self._fire("reset")

    def _push(self, payload: dict):
        """推送：坐标 / 置信度 / 截图 / 时间 / 视频源，可接弹窗、声音或 Webhook。"""
        self.push_log.append(payload)
        self.events.append({"from": "PUSH", "action": "push", "to": "PUSH",
                            "detail": payload.get("coords", "告警推送")})

    # ---------- 统计 ----------
    def stats(self) -> dict:
        dismissed = sum(1 for e in self.events if e["action"] == "dismiss")
        confirmed = sum(1 for e in self.events if e["action"] == "confirm")
        rate = dismissed / confirmed if confirmed else 0.0
        return {"state": self.state, "pushes": self.push_count,
                "confirmed": confirmed, "dismissed": dismissed,
                "false_alarm_rate": round(rate, 4)}


class AlarmManager:
    """在状态机之上叠加"火点去重 + 记录归档"，供界面列表与数据库使用。"""

    def __init__(self, push_cooldown: float = 30.0, spatial_gate_m: float = 60.0):
        self.fsm = AlarmFSM(push_cooldown=push_cooldown)
        self.spatial_gate_m = spatial_gate_m
        self.records: list = []           # AlarmRecord
        self._active = None               # 当前处置中的告警
        self._seq = 0

    # ---------- 生命周期 ----------
    def start(self):
        self.fsm.start()
        return self.fsm.state

    def stop(self):
        try:
            self.fsm.stop()
        except ValueError:
            # 告警未处置完就停止监测，先回到监测中再停
            self.fsm.reset()
        return self.fsm.state

    # ---------- 检测输入 ----------
    def feed(self, confirmed, fixes, now: float | None = None) -> dict | None:
        """喂入一帧的确认目标与定位结果。

        返回新告警事件（dict）或 None。同一火点在空间阈值内、冷却期内不重复产生新告警。
        """
        if not confirmed or not fixes:
            return None
        fix = fixes[0]
        # 空间去重：与已有告警距离小于阈值视为同一火情，只更新状态
        for rec in self.records:
            d = _dist_m(rec.fix.lat, rec.fix.lon, fix.lat, fix.lon)
            if d < self.spatial_gate_m:
                rec.fix = fix                 # 刷新观测，不新建告警
                return None

        if self.fsm.state == IDLE:
            self.fsm.start()
        self.fsm.on_detection(confirmed)
        self.fsm.on_suspect_pass({"coords": fix.coords,
                                  "conf": round(fix.detection.conf, 3)})
        self._seq += 1
        from .types import AlarmRecord
        rec = AlarmRecord(seq=self._seq, created_at=now or time.time(),
                          fix=fix, state=self.fsm.state, pushed=True,
                          events=[dict(e) for e in self.fsm.events[-3:]])
        self.records.insert(0, rec)
        self._active = rec
        # 回到监测中，等待下一次触发（处置动作由操作员触发）
        self.fsm.on_detection([])
        return rec.as_dict()

    # ---------- 操作员动作 ----------
    def resolve(self) -> dict | None:
        rec = self._active
        if rec is None:
            return None
        try:
            self.fsm.resolve()
            self.fsm.reset()
        except ValueError:
            return None
        rec.state, rec.resolved_at = RESOLVED, time.time()
        self._active = None
        return rec.as_dict()

    def dismiss(self) -> dict | None:
        rec = self._active
        if rec is None:
            return None
        try:
            self.fsm.dismiss()
            self.fsm.reset()
        except ValueError:
            return None
        rec.state, rec.dismissed, rec.resolved_at = DISMISSED, True, time.time()
        self._active = None
        return rec.as_dict()

    def stats(self) -> dict:
        s = self.fsm.stats()
        s["records"] = len(self.records)
        s["active"] = self._active.code if self._active else None
        return s


def _dist_m(lat1, lon1, lat2, lon2) -> float:
    """两点水平距离（米），小范围下用等距圆柱近似即可。"""
    dn = (lat2 - lat1) * math.radians(1) * 6378137.0
    de = (lon2 - lon1) * math.radians(1) * 6378137.0 * math.cos(math.radians(lat1))
    return math.hypot(dn, de)


if __name__ == "__main__":
    fsm = AlarmFSM(push_cooldown=30.0)
    fsm.start()                     # 待机 → 监测中
    fsm.on_detection([1])           # 有确认目标 → 疑似
    fsm.on_suspect_pass()           # 判定通过 → 确认告警并推送
    fsm.resolve()                   # 操作员处置完成
    fsm.reset()                     # 回到监测中

    fsm.on_detection([1])
    fsm.on_suspect_pass()           # 冷却期内不再重复推送
    fsm.dismiss()                   # 操作员标记误报 → 撤销
    print(f"推送次数 {fsm.push_count}（冷却期内重复告警已抑制）")
    print(stats := fsm.stats())

    for e in fsm.events:            # 迁移日志可直接导出，供复盘与误报率统计
        print(f"{e['from']} --{e['action']}--> {e['to']}  {e['detail']}")