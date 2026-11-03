# -*- coding: utf-8 -*-
"""核心算法单元测试（对应论文第 3 章、第 4 章的可验证结论）。

运行方式（本机不使用 pytest 也可）：

    D:/AI/RVCover/py311/python.exe -m unittest discover -s tests -v

只测"纯算法"，不启动界面与线程，全部用例在 10 s 内跑完。
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forestfire_gs import alarm as alarm_mod                      # noqa: E402
from forestfire_gs.alarm import (AlarmFSM, AlarmManager, CONFIRMED,  # noqa: E402
                                 DISMISSED, IDLE, MONITORING, RESOLVED,
                                 SUSPECT)
from forestfire_gs.config import StationConfig                   # noqa: E402
from forestfire_gs.detector import (MockDetector, TemporalConfirmer,  # noqa: E402
                                    iou)
from forestfire_gs.localization import (error_stats, latlon_to_ned,  # noqa: E402
                                       ned_to_latlon, project, solve)
from forestfire_gs.metrics import (error_vs_heights, latency_budget,  # noqa: E402
                                   pr_curve)
from forestfire_gs.storage import StationDB                      # noqa: E402
from forestfire_gs.types import Detection, Telemetry             # noqa: E402

# 论文 3.4 节使用的相机内参与无人机位姿（与附录 A.2 一致）
CAM = {"width": 1920, "height": 1080, "fx": 1400.0, "fy": 1400.0,
       "cx": 960.0, "cy": 540.0}
UAV = {"lat": 30.0, "lon": 114.0, "H": 100.0, "roll": 0.0, "pitch": 0.0,
       "yaw": 0.0, "gimbal_pitch": 30.0, "gimbal_yaw": 0.0}


def make_det(box=(900, 600, 1020, 690), conf=0.9, t=0.0, track_id=1):
    return Detection(box=box, cls=0, conf=conf, track_id=track_id, hits=3, t=t)


def make_tm(t=0.0, lat=30.0, lon=114.0):
    return Telemetry(t=t, lat=lat, lon=lon, H=100.0, roll=0.0, pitch=0.0,
                     yaw=0.0, gimbal_pitch=30.0, gimbal_yaw=0.0,
                     ground_speed=11.5, battery=82.0, satellites=18)


class TestLocalization(unittest.TestCase):
    """3.4 节单目定位：正反投影闭合 + 误差量级。"""

    def test_ned_roundtrip(self):
        lat, lon = ned_to_latlon(120.0, -80.0, 30.0, 114.0)
        dn, de = latlon_to_ned(lat, lon, 30.0, 114.0)
        self.assertAlmostEqual(dn, 120.0, places=6)
        self.assertAlmostEqual(de, -80.0, places=6)

    def test_project_solve_closed_loop(self):
        """投影像素 → 反解坐标，闭合误差应远小于 1 m。"""
        t_lat, t_lon = 30.0005, 114.0003
        uv = project(UAV, t_lat, t_lon, CAM)
        self.assertIsNotNone(uv, "目标应落在相机视场内")
        est = solve(uv[0], uv[1], CAM, UAV)
        self.assertIsNotNone(est)
        dn, de = latlon_to_ned(est[0], est[1], t_lat, t_lon)
        self.assertLess(math.hypot(dn, de), 0.05)

    def test_horizon_ray_returns_none(self):
        """云台平视（射线与地面不相交）时不应给出伪坐标。"""
        uav = dict(UAV, gimbal_pitch=90.0)
        self.assertIsNone(solve(960, 540, CAM, uav))

    def test_all_heights_have_samples(self):
        """50~200 m 四个高度都必须解算出坐标，不能有空样本。"""
        for h in (50, 100, 150, 200):
            uav = dict(UAV, H=float(h))
            uv = project(uav, 30.0005, 114.0003, CAM)
            self.assertIsNotNone(uv, f"{h} m 投影像素为空")
            est = solve(uv[0], uv[1], CAM, uav)
            self.assertIsNotNone(est, f"{h} m 反解为空")
            dn, de = latlon_to_ned(est[0], est[1], 30.0005, 114.0003)
            self.assertLess(math.hypot(dn, de), 0.05, f"{h} m 闭合误差过大")

    def test_error_stats_100m_meets_r3(self):
        """R3：100 m 高度水平误差 ≤ 30 m，实测应在 2 m 量级。"""
        s = error_stats(30.0005, 114.0003, UAV, CAM, n=600, seed=5)
        self.assertGreaterEqual(s["n"], 550)
        self.assertLess(s["mean"], 3.0)
        self.assertLess(s["p95"], 6.0)
        self.assertGreater(s["mean"], 1.2, "误差不应小到不符合物理直觉")


class TestDetector(unittest.TestCase):
    """3.3 节检测与时序确认。"""

    def test_iou(self):
        a = (0, 0, 100, 100)
        self.assertAlmostEqual(iou(a, a), 1.0, places=6)
        self.assertEqual(iou(a, (200, 200, 300, 300)), 0.0)
        self.assertAlmostEqual(iou(a, (50, 0, 150, 100)), 1 / 3, places=6)

    def test_mock_detector_deterministic(self):
        a = MockDetector(seed=7).predict(None, 0.0)
        b = MockDetector(seed=7).predict(None, 0.0)
        self.assertEqual([d.box for d in a], [d.box for d in b])

    def test_mock_detector_hits_fire_region(self):
        """火焰框应落在合成画面的火盆位置附近。"""
        det = MockDetector(seed=3, fire_center=(960, 645), radius=64)
        hits = [d for d in (det.predict(None, i / 25) for i in range(40))
                if any(x.cls == 0 for x in d)]
        self.assertGreater(len(hits), 20, "40 帧内火焰应被稳定检出")
        cx, cy = det.predict(None, 1.0)[0].center
        self.assertLess(abs(cx - 960), 90)
        self.assertLess(abs(cy - 645), 90)

    def test_temporal_confirmer_needs_three_hits(self):
        c = TemporalConfirmer(window=5, min_hits=3, iou_thr=0.30)
        box = (900, 600, 1020, 690)
        self.assertEqual(c.update([make_det(box, t=0.0)]), [])
        self.assertEqual(c.update([make_det(box, t=0.04)]), [])
        got = c.update([make_det(box, t=0.08)])
        self.assertEqual(len(got), 1, "第 3 帧命中后应确认")
        self.assertEqual(got[0].track_id, 1)

    def test_temporal_confirmer_drops_stale_track(self):
        c = TemporalConfirmer(window=5, min_hits=3, iou_thr=0.30)
        c.update([make_det((10, 10, 60, 60), t=0.0)])
        for i in range(1, 7):
            c.update([make_det((900, 600, 1020, 690), t=i * 0.04)])
        stats = c.statistics
        self.assertEqual(stats["tracks"], 1, "未满命中的轨迹应被淘汰")


class TestAlarm(unittest.TestCase):
    """3.5 节告警状态机与推送。"""

    def test_transitions_valid(self):
        f = AlarmFSM(push_cooldown=0.0)
        self.assertEqual(f.state, IDLE)
        f.start()
        self.assertEqual(f.state, MONITORING)
        f.on_detection([make_det()])
        self.assertEqual(f.state, SUSPECT)
        f.on_suspect_pass({"coords": "30.0,114.0"})
        self.assertEqual(f.state, CONFIRMED)
        f.resolve()
        self.assertEqual(f.state, RESOLVED)
        f.reset()
        self.assertEqual(f.state, MONITORING)

    def test_illegal_transition_raises(self):
        f = AlarmFSM()
        with self.assertRaises(ValueError):
            f.resolve()                      # 待机状态不能直接处置

    def test_push_cooldown(self):
        f = AlarmFSM(push_cooldown=30.0)
        f.start()
        for _ in range(3):
            f.on_detection([make_det()])
            f.on_suspect_pass({})
            f.resolve()
            f.reset()
        self.assertEqual(f.stats()["pushes"], 1, "冷却期内只推送一次")

    def test_manager_spatial_dedupe_and_resolve(self):
        m = AlarmManager(push_cooldown=0.0, spatial_gate_m=60.0)
        m.start()
        det = make_det()
        from forestfire_gs.types import FireFix
        fix1 = FireFix(lat=30.0006, lon=114.0004, dn=66.0, de=38.0,
                       range_m=76.0, telemetry=make_tm(), detection=det, skew_s=0.0)
        fix2 = FireFix(lat=30.00062, lon=114.00042, dn=69.0, de=40.0,
                       range_m=80.0, telemetry=make_tm(), detection=det, skew_s=0.0)
        first = m.feed([det], [fix1], now=1000.0)
        self.assertIsNotNone(first)
        self.assertIsNone(m.feed([det], [fix2], now=1001.0), "同一火点应去重")
        self.assertEqual(len(m.records), 1)
        done = m.resolve()
        self.assertEqual(done["state"], RESOLVED)

    def test_manager_dismiss_counts_false_alarm(self):
        m = AlarmManager(push_cooldown=0.0, spatial_gate_m=60.0)
        m.start()
        det = make_det()
        from forestfire_gs.types import FireFix
        fix = FireFix(lat=30.001, lon=114.001, dn=111.0, de=96.0, range_m=147.0,
                      telemetry=make_tm(), detection=det, skew_s=0.0)
        m.feed([det], [fix], now=1.0)
        m.dismiss()
        self.assertEqual(m.stats()["false_alarm_rate"], 1.0)


class TestMetrics(unittest.TestCase):
    """第 4 章指标复算：定位误差、时延预算、PR 曲线。"""

    def test_error_vs_heights_matches_paper(self):
        rows = error_vs_heights()
        self.assertEqual([r["height"] for r in rows], [50, 100, 150, 200])
        expect = [1.53, 1.80, 2.33, 2.92]
        for r, e in zip(rows, expect):
            self.assertAlmostEqual(r["mean"], e, delta=0.25,
                                   msg=f"{r['height']} m 误差偏离过大")
        self.assertTrue(all(r["mean"] < 30.0 for r in rows), "R3 全部满足")

    def test_latency_budget(self):
        b = latency_budget()
        self.assertEqual(len(b["stages"]), 7)
        self.assertAlmostEqual(b["total_ms"], 2300.0, places=1)
        self.assertLess(b["total_s"], b["budget_s"], "R4：端到端 < 5 s")

    def test_pr_curve_ap(self):
        rec, prec = pr_curve(ap_target=0.87, n=60, seed=7)
        self.assertEqual(len(rec), 60)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in prec))
        self.assertGreater(max(prec), 0.9)


class TestStorage(unittest.TestCase):
    """3.7 节数据管理：告警、处置记录与截图落库。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ffgs-test-")
        self.db = StationDB(os.path.join(self.tmp, "station.db"))

    def tearDown(self):
        self.db.close()

    def test_save_fire_and_lookup(self):
        self.db.save_fire_fields(code="FF-1000-001", lat=30.0006, lon=114.0004,
                                 conf=0.91, cls=0, found_ts=1000.0,
                                 source="sim://patrol-01", range_m=76.0,
                                 skew_s=0.02)
        row = self.db.fire_by_code("FF-1000-001")
        self.assertIsNotNone(row)
        self.assertAlmostEqual(row["lat"], 30.0006, places=6)
        self.assertEqual(len(self.db.fires()), 1)

    def test_event_log_and_screenshot(self):
        self.db.save_fire_fields(code="FF-1000-002", lat=30.0, lon=114.0,
                                 conf=0.8, cls=0, found_ts=1000.0,
                                 source="sim://patrol-01", range_m=70.0,
                                 skew_s=0.0)
        self.db.record_event(code="FF-1000-002", frm=CONFIRMED,
                             action="resolve", to=RESOLVED, operator="operator")
        self.db.attach_screenshot(code="FF-1000-002", path="shots/a.png", ts=1000.0)
        fid = self.db.fire_id("FF-1000-002")
        self.assertIsNotNone(fid)
        hist = self.db.alarm_history(fid)
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0]["action"], "resolve")
        self.assertEqual(hist[0]["to_state"], RESOLVED)
        shot = self.db.conn.execute(
            "SELECT path FROM screenshots WHERE fire_id = ?", (fid,)).fetchone()
        self.assertEqual(shot["path"], "shots/a.png")

    def test_summary(self):
        self.db.save_fire_fields(code="FF-1001-001", lat=30.0, lon=114.0,
                                 conf=0.8, cls=0, found_ts=1001.0,
                                 source="sim://patrol-01", range_m=70.0, skew_s=0.0)
        self.db.record_event(code="FF-1001-001", frm=CONFIRMED,
                             action="dismiss", to=DISMISSED, operator="operator")
        s = self.db.summary()
        self.assertEqual(s["fires"], 1)
        self.assertEqual(s["dismissed"], 1)
        self.assertAlmostEqual(s["false_alarm_rate"], 1.0)


class TestConfig(unittest.TestCase):
    """配置读写（3.1 节工程结构里的 config 模块）。"""

    def test_roundtrip(self):
        tmp = tempfile.mkdtemp(prefix="ffgs-cfg-")
        path = os.path.join(tmp, "station.json")
        cfg = StationConfig()
        cfg.target_fps = 20
        cfg.alarm.push_cooldown = 12.5
        cfg.save(path)
        back = StationConfig.load(path)
        self.assertEqual(back.target_fps, 20)
        self.assertAlmostEqual(back.alarm.push_cooldown, 12.5)
        self.assertEqual(back.camera.width, 1920)


class TestPipeline(unittest.TestCase):
    """端到端：采集线程 + 推理线程 + 告警落库（论文 3.6 节）。"""

    def test_engine_produces_frames_and_alarm(self):
        frames = []
        events = []
        cfg = StationConfig()
        cfg.seed = 20261103
        from forestfire_gs.pipeline import StationEngine
        eng = StationEngine(cfg=cfg,
                            listener=lambda name, payload: events.append((name, payload)))
        eng.start()
        try:
            import time
            t0 = time.time()
            while time.time() - t0 < 2.0:
                snap = eng.snapshot()
                frames.append(snap["fps"])
                time.sleep(0.05)
        finally:
            eng.stop()
        self.assertGreater(max(frames or [0]), 5.0, "推理帧率应明显大于 0")
        kinds = {n for n, _ in events}
        self.assertIn("frame", kinds)
        self.assertIn("state", kinds)


if __name__ == "__main__":
    unittest.main(verbosity=2)