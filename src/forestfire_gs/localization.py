# -*- coding: utf-8 -*-
"""单目火点大地坐标解算（对应论文 3.4 节、附录 A.2）。

坐标系约定
----------
    世界系 NED（北-东-地），无人机为原点，地面点 z = +H（Down 为正）
    机体系 x 前 / y 右 / z 下；相机系 x 右 / y 下 / z 沿光轴
    旋转顺序固定为 ZYX 外旋：先绕 z 偏航、再绕 y 俯仰、最后绕 x 横滚

解算链（论文图 3-2）
--------------------
    像素 → 相机射线 d_c → 机体系 d_b → NED d_n → 与地面求交 → 经纬度

注意：与地面求交按"平地假设"处理（H 为该方向的地面高差）；山地林区若有 DEM，
应把 H 换成该射线方向上的地形高差。
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .types import Detection, FireFix, Telemetry

EARTH_R = 6378137.0          # WGS-84 长半轴，米


# ---------------- ① 基本旋转（右手系，角度为弧度） ----------------
def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def R_bn(roll, pitch, yaw):
    """机体系 → NED，ZYX 外旋：先偏航、再俯仰、后横滚。"""
    return Rz(yaw) @ Ry(pitch) @ Rx(roll)


def R_cb(gimbal_pitch, gimbal_yaw):
    """相机系 → 机体系（云台俯仰、方位）。"""
    return Rz(gimbal_yaw) @ Ry(gimbal_pitch)


# ---------------- ② 局部 NED 位移与经纬度互转 ----------------
def ned_to_latlon(n, e, lat0, lon0):
    """相对 (lat0, lon0) 的北东位移（米）→ 绝对经纬度。"""
    return (lat0 + math.degrees(n / EARTH_R),
            lon0 + math.degrees(e / (EARTH_R * math.cos(math.radians(lat0)))))


def latlon_to_ned(lat, lon, lat0, lon0):
    """绝对经纬度 → 相对 (lat0, lon0) 的北东位移（米）。"""
    return ((lat - lat0) * math.radians(1) * EARTH_R,
            (lon - lon0) * math.radians(1) * EARTH_R * math.cos(math.radians(lat0)))


# ---------------- ③ 反投影：像素 → 地面点 ----------------
def pixel_to_ground(u, v, cam, R_bn_mat, R_cb_mat, H):
    """四步坐标变换后与地面求交，返回相对无人机的 (n, e) 位移（米）。

    射线不朝下（DN 分量 <= 0）时无交点，返回 None。
    """
    # 第 1 步：像素归一化，消去焦距与主点的影响
    d_c = np.array([(u - cam["cx"]) / cam["fx"],
                    (v - cam["cy"]) / cam["fy"], 1.0])

    # 第 2、3 步：相机 → 机体 → NED
    d_n = R_bn_mat @ (R_cb_mat @ d_c)

    # 第 4 步：沿射线与地面 z = H 求交
    if d_n[2] <= 1e-9:
        return None
    t = H / d_n[2]
    return float(t * d_n[0]), float(t * d_n[1])


def solve(u, v, cam, uav):
    """检测框中心像素 + 一帧遥测 → 火点经纬度；无交点返回 None。

    uav 字段：lat、lon、H（米）、roll、pitch、yaw、gimbal_pitch、gimbal_yaw（度）
    """
    roll, pitch, yaw = np.radians([uav["roll"], uav["pitch"], uav["yaw"]])
    gp, gy = np.radians([uav["gimbal_pitch"], uav["gimbal_yaw"]])
    ne = pixel_to_ground(u, v, cam, R_bn(roll, pitch, yaw), R_cb(gp, gy), uav["H"])
    if ne is None:
        return None
    return ned_to_latlon(ne[0], ne[1], uav["lat"], uav["lon"])


# ---------------- ④ 正投影：已知地面点 → 像素（供闭环自检与仿真） ----------------
def project(uav, lat, lon, cam):
    """把已知经纬度的目标投到画面上，返回 (u, v)；目标在相机背后时返回 None。"""
    n, e = latlon_to_ned(lat, lon, uav["lat"], uav["lon"])
    d_n = np.array([n, e, uav["H"]])
    d_n = d_n / np.linalg.norm(d_n)
    roll, pitch, yaw = np.radians([uav["roll"], uav["pitch"], uav["yaw"]])
    gp, gy = np.radians([uav["gimbal_pitch"], uav["gimbal_yaw"]])
    d_c = R_cb(gp, gy).T @ (R_bn(roll, pitch, yaw).T @ d_n)   # 逆向回相机系
    if d_c[2] <= 1e-9:
        return None                       # 目标在相机背后
    # 注：这里不做画幅裁剪。误差分析需要"目标像素"这一真值，即使它落在
    # 画幅之外（例如 50 m 低空、云台固定 30° 时），也照样给出像素坐标。
    return (cam["fx"] * d_c[0] / d_c[2] + cam["cx"],
            cam["fy"] * d_c[1] / d_c[2] + cam["cy"])


# ---------------- ⑤ 误差分析：正向加噪 → 反向解算 → 统计 ----------------
def error_stats(true_lat, true_lon, uav, cam, sigma_px=2.0, sigma_deg=0.5,
                n=300, seed=0):
    """蒙特卡洛统计水平定位误差（米），返回 mean / median / p95。"""
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n):
        uv = project(uav, true_lat, true_lon, cam)
        if uv is None:
            continue
        noisy = dict(uav)
        for k in ("roll", "pitch", "yaw", "gimbal_pitch", "gimbal_yaw"):
            noisy[k] += rng.normal(0, sigma_deg)                # 角度噪声
        est = solve(uv[0] + rng.normal(0, sigma_px),            # 像素噪声
                    uv[1] + rng.normal(0, sigma_px), cam, noisy)
        if est is None:
            continue
        dn, de = latlon_to_ned(est[0], est[1], true_lat, true_lon)
        errs.append(float(math.hypot(dn, de)))
    a = np.array(errs) if errs else np.zeros(1)
    return {"n": int(a.size), "mean": float(a.mean()),
            "median": float(np.median(a)), "p95": float(np.percentile(a, 95))}


# ---------------- ⑥ 面向流水线的高层封装 ----------------
class Locator:
    """把"检测框 + 最近时间戳遥测"解算成 FireFix，并负责遥测时效性判断。"""

    def __init__(self, camera, max_skew_s: float = 0.5):
        self.cam = camera.as_dict() if hasattr(camera, "as_dict") else dict(camera)
        self.max_skew_s = max_skew_s

    def nearest_telemetry(self, t: float, telemetry: list) -> Optional[Telemetry]:
        """取与当前帧时间最接近的一帧遥测（时间戳最近邻对齐）。"""
        if not telemetry:
            return None
        return min(telemetry, key=lambda tm: abs(tm.t - t))

    def fix(self, det: Detection, tm: Telemetry) -> Optional[FireFix]:
        if tm is None:
            return None
        if abs(tm.t - det.t) > self.max_skew_s:
            return None                      # 遥测过期，本次定位放弃
        u, v = det.center
        est = solve(u, v, self.cam, tm.as_dict())
        if est is None:
            return None
        dn, de = latlon_to_ned(est[0], est[1], tm.lat, tm.lon)
        rng = math.hypot(dn, de)
        return FireFix(lat=est[0], lon=est[1], dn=dn, de=de, range_m=rng,
                       telemetry=tm, detection=det, skew_s=abs(tm.t - det.t))


if __name__ == "__main__":
    cam = {"width": 1920, "height": 1080, "fx": 1400.0, "fy": 1400.0,
           "cx": 960.0, "cy": 540.0}
    uav = {"lat": 30.0, "lon": 114.0, "H": 100.0, "roll": 0.0, "pitch": 0.0,
           "yaw": 0.0, "gimbal_pitch": 30.0, "gimbal_yaw": 0.0}
    t_lat, t_lon = 30.0005, 114.0003          # 约 55 m 北、29 m 东

    # 自检 1：正反投影应闭合
    uv = project(uav, t_lat, t_lon, cam)
    est = solve(uv[0], uv[1], cam, uav)
    dn, de = latlon_to_ned(est[0], est[1], t_lat, t_lon)
    print(f"投影像素 {uv[0]:.1f},{uv[1]:.1f}  闭合误差 {math.hypot(dn, de):.3f} m")

    # 自检 2：不同高度下的定位误差（对应论文 4.3 节）
    for h in (50, 100, 150, 200):
        uav["H"] = float(h)
        s = error_stats(t_lat, t_lon, uav, cam)
        print(f"高度{h:3d}m  均值{s['mean']:.2f}m  中位{s['median']:.2f}m  "
              f"P95 {s['p95']:.2f}m")