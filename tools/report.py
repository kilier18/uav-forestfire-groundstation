# -*- coding: utf-8 -*-
"""一键跑出论文第 4 章的指标表格（不需要界面、不开线程）。

    python tools/report.py

输出 Markdown 表格，可直接粘到论文/README 里。数值全部由
``forestfire_gs.metrics`` 与 ``forestfire_gs.localization`` 现算，不是写死的。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:                                  # Windows 控制台默认 GBK，中文会乱码
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                     # pragma: no cover
    pass

from forestfire_gs.metrics import (error_by_noise, error_vs_heights,  # noqa: E402
                                   latency_budget, render_markdown)


def main() -> int:
    print("# 无人机森林防火地面站 · 指标复算\n")
    print(render_markdown())

    rows = error_vs_heights()
    print("\n## 定位精度（R3：100 m 高度 ≤ 30 m）\n")
    print("| 飞行高度 | 样本数 | 均值 | 中位 | P95 | 指标 |")
    print("| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        ok = "达标" if r["mean"] <= 30.0 else "超标"
        print(f"| {r['height']} m | {r['n']} | {r['mean']:.2f} m | "
              f"{r['median']:.2f} m | {r['p95']:.2f} m | {ok} |")

    noise = error_by_noise()
    print("\n## 噪声灵敏度（100 m 高度，单因素）\n")
    print("| σ | 仅像素噪声 | 仅角度噪声 |")
    print("| --- | --- | --- |")
    for s, a, b in zip(noise["sigma"], noise["pixel_mean_m"],
                       noise["angle_mean_m"]):
        print(f"| {s} | {a:.2f} m | {b:.2f} m |")

    b = latency_budget()
    print(f"\n## 端到端时延（R4：≤ {b['budget_s']:.0f} s）\n")
    print(f"合计 **{b['total_s']:.2f} s**，其中时序确认（{b['stages'][4][1]:.0f} ms）"
          f"占 {b['stages'][4][1] / b['total_ms'] * 100:.0f}%，是主要开销。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())