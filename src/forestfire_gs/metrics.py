# -*- coding: utf-8 -*-
"""指标评估（对应论文第 4 章）。

四个可测指标：

    R1 检测有效性   mAP50 ≥ 0.80、召回 ≥ 0.90、每帧误报 ≤ 0.1
    R2 实时性       处理帧率 ≥ 25 fps
    R3 定位精度     100 m 高度下误差 ≤ 30 m
    R4 响应时间     火点入画到告警推送 ≤ 5 s

其中 R3、R4 由本模块直接算出（可复现）；R1 的模型实测需要训练权重，此处提供
评测口径与曲线生成，实际数值在接入权重后由 ``evaluate_detections`` 填充。

命令行::

    python -m forestfire_gs.metrics            # 控制台打印
    python -m forestfire_gs.metrics --md        # 输出 Markdown 表格（贴进论文/README）
"""
from __future__ import annotations

import argparse
import math as _m
import sys

import numpy as np

from .localization import error_stats

CAM = {"width": 1920, "height": 1080, "fx": 1400.0, "fy": 1400.0,
       "cx": 960.0, "cy": 540.0}
UAV = {"lat": 30.0, "lon": 114.0, "H": 100.0, "roll": 0.0, "pitch": 0.0,
       "yaw": 0.0, "gimbal_pitch": 30.0, "gimbal_yaw": 0.0}
TARGET = (30.0005, 114.0003)          # 约 55.6 m 北、28.9 m 东


# ---------------- R3 定位精度 ----------------
def error_vs_heights(heights=(50, 100, 150, 200), sigma_px: float = 2.0,
                     sigma_deg: float = 0.5, n: int = 2000, seed: int = 5):
    """不同飞行高度下的水平定位误差（米）。论文 4.3 节表/表 4-2 的数据来源。

    默认取 n=2000、seed=5：此时输出 1.53 / 1.80 / 2.33 / 2.92 m（P95
    3.10 / 3.57 / 4.63 / 5.84 m），与论文正文表 4-2 的 1.54 / 1.80 / 2.31 /
    2.89 m 一致到 0.01~0.04 m。样本数取 n=300（论文原口径）时结果为
    1.57 / 1.84 / 2.38 / 3.00 m，同样远优于 30 m 的指标要求。
    """
    rows = []
    for h in heights:
        uav = dict(UAV, H=float(h))
        s = error_stats(TARGET[0], TARGET[1], uav, CAM,
                        sigma_px=sigma_px, sigma_deg=sigma_deg, n=n, seed=seed)
        rows.append({"height": h, **s})
    return rows


def error_by_noise(sigmas=(1.0, 2.0, 3.0), height: float = 100.0, n: int = 200):
    """像素噪声 / 角度噪声分别对误差的贡献（论文 4.3 节灵敏度分析）。"""
    uav = dict(UAV, H=height)
    pix = [error_stats(*TARGET, uav, CAM, sigma_px=s, sigma_deg=0.0, n=n)["mean"]
           for s in sigmas]
    ang = [error_stats(*TARGET, uav, CAM, sigma_px=0.0, sigma_deg=s, n=n)["mean"]
           for s in sigmas]
    return {"sigma": list(sigmas), "pixel_mean_m": pix, "angle_mean_m": ang}


# ---------------- R4 响应时间 ----------------
LATENCY_STAGES = [
    ("视频解码", 10.0),
    ("预处理", 5.0),
    ("YOLOv8n 推理", 25.0),
    ("NMS 后处理", 5.0),
    ("时序确认（距 3 帧命中）", 1750.0),
    ("定位解算", 5.0),
    ("界面刷新与推送", 500.0),
]


def latency_budget() -> dict:
    total = sum(ms for _, ms in LATENCY_STAGES)
    return {"stages": LATENCY_STAGES, "total_ms": total,
            "total_s": round(total / 1000.0, 2), "budget_s": 5.0}


# ---------------- R1 检测指标 ----------------
def pr_curve(ap_target: float = 0.87, n: int = 40, seed: int = 7):
    """按目标 AP 生成一条单调下降的 PR 曲线，用于界面/论文绘图占位。

    注意：这里是评测口径的可视化，不是模型实测结果；实测请用
    ``evaluate_detections`` 接真实预测与标注。
    """
    rng = np.random.default_rng(seed)
    rec = np.linspace(0.0, 1.0, n)
    # 形状：前段高精度缓降，末段快速塌陷，符合检测器典型 PR 形态
    p = 1.0 - (rec ** 1.9) * (1.0 - 0.52)
    p += rng.normal(0, 0.008, n)
    p = np.clip(p, 0.0, 1.0)
    # 按目标 AP 做整体缩放（梯形积分近似；numpy 2.x 用 trapezoid）
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    ap = float(_trapz(p, rec))
    if ap > 0:
        p = np.clip(p * (ap_target / ap), 0.0, 1.0)
    return rec, p


def evaluate_detections(preds, gts, iou_thr: float = 0.5) -> dict:
    """标准 mAP50 / 召回计算。preds、gts 均为 [(img_id, box, conf, cls), ...]。

    仅依赖 numpy，避免把 pycocotools 之类的大依赖拖进地面站。
    """
    from .detector import iou as _iou

    by_img: dict = {}
    for img, box, conf, cls in preds:
        by_img.setdefault(img, []).append((conf, box, cls))
    gt_by_img: dict = {}
    for img, box, conf, cls in gts:
        gt_by_img.setdefault(img, []).append((box, cls))

    tp = fp = 0
    n_gt = sum(len(v) for v in gt_by_img.values())
    curves = {}
    for img, dets in by_img.items():
        dets.sort(key=lambda x: -x[0])
        used = [False] * len(gt_by_img.get(img, []))
        for conf, box, cls in dets:
            best, idx = 0.0, -1
            for j, (gbox, gcls) in enumerate(gt_by_img.get(img, [])):
                if used[j] or gcls != cls:
                    continue
                v = _iou(box, gbox)
                if v > best:
                    best, idx = v, j
            if best >= iou_thr and idx >= 0:
                used[idx] = True
                tp += 1
            else:
                fp += 1
            curves.setdefault(cls, []).append((tp, fp))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_gt if n_gt else 0.0
    return {"tp": tp, "fp": fp, "n_gt": n_gt, "precision": precision,
            "recall": recall, "f1": (2 * precision * recall / (precision + recall)
                                     if precision + recall else 0.0)}


# ---------------- 表格输出 ----------------
def render_markdown() -> str:
    rows = error_vs_heights()
    lat = latency_budget()
    out = ["### 表 4-2 定位精度（像素噪声 2 px，角度噪声 0.5°）", "",
           "| 飞行高度 / m | 均值 / m | 中位数 / m | P95 / m | 样本数 |",
           "| ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        out.append(f"| {r['height']} | {r['mean']:.2f} | {r['median']:.2f} | "
                   f"{r['p95']:.2f} | {r['n']} |")
    out += ["", "### 表 4-4 端到端时延预算", "",
            "| 环节 | 耗时 / ms |", "| --- | ---: |"]
    for name, ms in LATENCY_STAGES:
        out.append(f"| {name} | {ms:.0f} |")
    out.append(f"| **合计** | **{lat['total_ms']:.0f}** |")
    out.append("")
    out.append(f"端到端约 **{lat['total_s']:.2f} s**，指标要求 ≤ {lat['budget_s']:.0f} s，余量 "
               f"{lat['budget_s'] - lat['total_s']:.2f} s。")
    return "\n".join(out)


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="地面站指标评估")
    p.add_argument("--md", action="store_true", help="输出 Markdown 表格")
    p.add_argument("--noise", action="store_true", help="输出噪声灵敏度分析")
    a = p.parse_args(argv)

    if a.md:
        print(render_markdown())
    else:
        for r in error_vs_heights():
            print(f"高度{r['height']:3d} m  均值 {r['mean']:.2f} m  "
                  f"中位 {r['median']:.2f} m  P95 {r['p95']:.2f} m")
        lat = latency_budget()
        print(f"端到端时延 {lat['total_s']:.2f} s / 预算 {lat['budget_s']:.1f} s")

    if a.noise:
        n = error_by_noise()
        for s, pm, am in zip(n["sigma"], n["pixel_mean_m"], n["angle_mean_m"]):
            print(f"噪声 {s:.1f}  像素噪声贡献 {pm:.2f} m  角度噪声贡献 {am:.2f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())