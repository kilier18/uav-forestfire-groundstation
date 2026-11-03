# -*- coding: utf-8 -*-
"""遥测仿真（对应论文 2.6 节"无硬件替代方案"）。

没有真机时，用一条可复现的巡护航线驱动定位模块：无人机沿圆角矩形航线飞行，
云台始终指向已知火点，因此"画面中的火点"与"地图上的火点"是同一套几何关系，
正向投影与反向解算能在同一条链路上闭环验证。真机到位后只要把数据源换成
飞控下行数据即可，字段不变。
"""
from __future__ import annotations

import math

from .types import Telemetry

# 基准点：湖北某林场（演示坐标，非真实火情地点）
BASE_LAT, BASE_LON = 30.0000, 114.0000


class PatrolSimulator:
    """巡护航线 + 云台指向仿真。

    参数
    ----
    base      起飞点经纬度
    fire      被观测火点经纬度（云台自动指向它）
    half_w    航线半宽（米，北向）
    half_h    航线半高（米，东向）
    speed     巡航地速（米/秒）
    height    巡航高度（米）
    """

    def __init__(self, base=(BASE_LAT, BASE_LON), fire=(30.00110, 114.00072),
                 half_w: float = 130.0, half_h: float = 105.0,
                 speed: float = 11.5, height: float = 100.0, seed: int = 20261103):
        self.base_lat, self.base_lon = base
        self.fire_lat, self.fire_lon = fire
        self.half_w, self.half_h = half_w, half_h
        self.speed, self.height = speed, height
        self._s = 0.0                 # 沿航线累计路程
        self._t = seed * 0.0
        self._rng_state = seed

    # ---------- 航线几何 ----------
    def _perimeter(self) -> float:
        return 4.0 * (self.half_w + self.half_h) * 0.92

    def _point_at(self, s: float) -> tuple:
        """按弧长 s 取航线上的机体系位置（相对基准点的北、东位移，米）。"""
        per = self._perimeter()
        s = s % per
        seg = [self.half_h * 2 * 0.92, self.half_w * 2 * 0.92,
               self.half_h * 2 * 0.92, self.half_w * 2 * 0.92]
        idx, acc = 0, 0.0
        while idx < 3 and s > acc + seg[idx]:
            acc += seg[idx]
            idx += 1
        f = (s - acc) / max(seg[idx], 1e-6)
        # 四个角点（北, 东）
        corners = [(-self.half_w, -self.half_h), (-self.half_w, self.half_h),
                   (self.half_w, self.half_h), (self.half_w, -self.half_h)]
        n0, e0 = corners[idx]
        n1, e1 = corners[(idx + 1) % 4]
        return n0 + (n1 - n0) * f, e0 + (e1 - e0) * f

    # ---------- 主循环 ----------
    def step(self, dt: float = 0.1) -> Telemetry:
        self._s += self.speed * dt
        self._t += dt
        n, e = self._point_at(self._s)
        n2, e2 = self._point_at(self._s + 2.0)
        heading = math.degrees(math.atan2(e2 - e, n2 - n)) % 360.0

        # 航迹微扰动：真实飞控的姿态不会完全平直
        wob = math.sin(self._t * 1.7)
        roll = 1.2 * wob
        pitch = 0.8 * math.sin(self._t * 1.1 + 0.4)

        # 目标相对无人机
        fn, fe = self._to_ned(self.fire_lat, self.fire_lon)
        dn, de = fn - n, fe - e
        dist = math.hypot(dn, de)
        hgt = self.height

        # 云台指向火点：先把目标方向转进机体系，再解 gimbal_yaw / gimbal_pitch
        yaw = math.radians(heading)
        bx = math.cos(yaw) * dn + math.sin(yaw) * de
        by = -math.sin(yaw) * dn + math.cos(yaw) * de
        gp = math.degrees(math.acos(max(-1.0, min(1.0, hgt / max(dist, 1e-6)))))
        gy = math.degrees(math.atan2(by, bx))
        # 云台伺服误差 + 轻微扫视
        gp += 0.35 * math.sin(self._t * 0.9)
        gy += 0.6 * math.sin(self._t * 0.55)

        lat, lon = self._to_latlon(n, e)
        return Telemetry(
            t=round(self._t, 3), lat=lat, lon=lon, H=hgt + 1.5 * math.sin(self._t * 0.3),
            roll=roll, pitch=pitch, yaw=heading,
            gimbal_pitch=gp, gimbal_yaw=gy,
            ground_speed=self.speed, battery=max(20.0, 92.0 - self._t * 0.012),
            satellites=17 + int(1.5 + 1.5 * math.sin(self._t * 0.07)))

    # ---------- 坐标工具 ----------
    def _to_ned(self, lat: float, lon: float) -> tuple:
        dn = (lat - self.base_lat) * math.radians(1) * 6378137.0
        de = (lon - self.base_lon) * math.radians(1) * 6378137.0 * math.cos(math.radians(self.base_lat))
        return dn, de

    def _to_latlon(self, n: float, e: float) -> tuple:
        lat = self.base_lat + math.degrees(n / 6378137.0)
        lon = self.base_lon + math.degrees(e / (6378137.0 * math.cos(math.radians(self.base_lat))))
        return lat, lon

    # ---------- 供界面标绘 ----------
    def track_points(self, n: int = 64) -> list:
        """整条航线的采样点（经纬度），界面用来画预置航线。"""
        out = []
        per = self._perimeter()
        for i in range(n):
            nn, ee = self._point_at(per * i / n)
            out.append(self._to_latlon(nn, ee))
        return out


if __name__ == "__main__":
    sim = PatrolSimulator()
    for _ in range(3):
        tm = sim.step(0.1)
        print(tm)
    print("航线采样点", len(sim.track_points()))