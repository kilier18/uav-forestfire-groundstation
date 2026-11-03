# -*- coding: utf-8 -*-
"""数据管理（对应论文 3.7 节、图 3-7）。

三张表用火情编号串起来：

    fire_events(id, code, lat, lon, conf, cls, found_ts, source, range_m, skew_s)
    alarm_events(id, fire_id, from_state, action, to_state, ts, operator)
    screenshots(id, fire_id, path, ts)

截图只存路径、文件按日期分目录，避免数据库被二进制撑大；查询时先检索再按需加载。
"""
from __future__ import annotations

import datetime as _dt
import os
import sqlite3
from pathlib import Path

from .types import AlarmRecord, FireFix

SCHEMA = """
CREATE TABLE IF NOT EXISTS fire_events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    code      TEXT UNIQUE,
    lat       REAL NOT NULL,
    lon       REAL NOT NULL,
    conf      REAL,
    cls       INTEGER,
    found_ts  REAL,
    source    TEXT,
    range_m   REAL,
    skew_s    REAL
);
CREATE TABLE IF NOT EXISTS alarm_events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    fire_id   INTEGER REFERENCES fire_events(id),
    from_state TEXT, action TEXT, to_state TEXT,
    ts        REAL, operator TEXT
);
CREATE TABLE IF NOT EXISTS screenshots (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    fire_id   INTEGER REFERENCES fire_events(id),
    path      TEXT, ts REAL
);
CREATE INDEX IF NOT EXISTS idx_fire_found ON fire_events(found_ts);
CREATE INDEX IF NOT EXISTS idx_alarm_fire ON alarm_events(fire_id);
"""


class StationDB:
    """地面站本地库（SQLite 单文件，随程序分发）。"""

    def __init__(self, path: str | Path = "data/station.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---------- 写入 ----------
    def save_fire(self, rec: AlarmRecord, source: str = "sim://patrol-01") -> int:
        fix = rec.fix
        cur = self.conn.execute(
            "INSERT OR REPLACE INTO fire_events"
            "(code, lat, lon, conf, cls, found_ts, source, range_m, skew_s)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (rec.code, fix.lat, fix.lon, fix.detection.conf, fix.detection.cls,
             rec.created_at, source, fix.range_m, fix.skew_s))
        self.conn.commit()
        fire_id = cur.lastrowid
        for ev in rec.events:
            self.conn.execute(
                "INSERT INTO alarm_events(fire_id, from_state, action, to_state, ts, operator)"
                " VALUES (?,?,?,?,?,?)",
                (fire_id, ev.get("from"), ev.get("action"), ev.get("to"),
                 rec.created_at, "值班员"))
        self.conn.commit()
        return fire_id

    def add_screenshot(self, fire_id: int, path: str | Path, ts: float | None = None):
        self.conn.execute("INSERT INTO screenshots(fire_id, path, ts) VALUES (?,?,?)",
                          (fire_id, str(path), ts or _dt.datetime.now().timestamp()))
        self.conn.commit()

    # ---------- 界面层使用的便捷写入（按火情编号操作） ----------
    def save_fire_fields(self, code: str, lat: float, lon: float, conf: float,
                         cls: int, found_ts: float, source: str = "sim://patrol-01",
                         range_m: float = 0.0, skew_s: float = 0.0) -> int:
        """界面从告警 dict 落库；与 ``save_fire(AlarmRecord)`` 写同一张表。"""
        cur = self.conn.execute(
            "INSERT OR REPLACE INTO fire_events"
            "(code, lat, lon, conf, cls, found_ts, source, range_m, skew_s)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (code, lat, lon, conf, cls, found_ts, source, range_m, skew_s))
        self.conn.commit()
        return cur.lastrowid

    def fire_id(self, code: str) -> int | None:
        row = self.conn.execute("SELECT id FROM fire_events WHERE code = ?",
                                (code,)).fetchone()
        return row["id"] if row else None

    def record_event(self, code: str, frm: str, action: str, to: str,
                     operator: str = "值班员") -> None:
        fid = self.fire_id(code)
        if fid is None:
            return
        self.log_event(fid, frm, action, to, operator)

    def attach_screenshot(self, code: str, path: str | Path,
                          ts: float | None = None) -> None:
        fid = self.fire_id(code)
        if fid is None:
            return
        self.add_screenshot(fid, path, ts)

    def log_event(self, fire_id: int, frm: str, action: str, to: str, operator: str = "值班员"):
        self.conn.execute(
            "INSERT INTO alarm_events(fire_id, from_state, action, to_state, ts, operator)"
            " VALUES (?,?,?,?,?,?)",
            (fire_id, frm, action, to, _dt.datetime.now().timestamp(), operator))
        self.conn.commit()

    # ---------- 查询 ----------
    def fires(self, limit: int = 50) -> list:
        rows = self.conn.execute(
            "SELECT * FROM fire_events ORDER BY found_ts DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def fire_by_code(self, code: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM fire_events WHERE code = ?", (code,)).fetchone()
        return dict(r) if r else None

    def alarm_history(self, fire_id: int) -> list:
        rows = self.conn.execute(
            "SELECT * FROM alarm_events WHERE fire_id = ? ORDER BY ts", (fire_id,)).fetchall()
        return [dict(r) for r in rows]

    def daily_counts(self) -> list:
        rows = self.conn.execute(
            "SELECT date(found_ts,'unixepoch','localtime') d, count(*) n"
            " FROM fire_events GROUP BY d ORDER BY d").fetchall()
        return [dict(r) for r in rows]

    def summary(self) -> dict:
        n = self.conn.execute("SELECT count(*) c FROM fire_events").fetchone()["c"]
        dismissed = self.conn.execute(
            "SELECT count(*) c FROM alarm_events WHERE action='dismiss'").fetchone()["c"]
        return {"fires": n, "dismissed": dismissed,
                "false_alarm_rate": round(dismissed / n, 4) if n else 0.0}

    def close(self):
        self.conn.close()


def screenshot_path(root: str | Path, ts: float | None = None) -> Path:
    """按日期分目录生成截图路径（截图内容由界面负责写入）。"""
    ts = ts or _dt.datetime.now().timestamp()
    day = _dt.datetime.fromtimestamp(ts).strftime("%Y%m%d")
    d = Path(root) / day
    d.mkdir(parents=True, exist_ok=True)
    return d / f"fire_{int(ts * 1000) % 100000:05d}.jpg"


if __name__ == "__main__":
    import tempfile

    from .types import Detection, Telemetry

    with tempfile.TemporaryDirectory() as td:
        db = StationDB(Path(td) / "demo.db")
        tm = Telemetry(t=1.0, lat=30.0011, lon=114.0007)
        det = Detection(box=(900, 420, 1020, 520), cls=0, conf=0.91, t=1.0)
        fix = FireFix(lat=30.0011, lon=114.0007, dn=122.0, de=67.0, range_m=140.0,
                      telemetry=tm, detection=det)
        rec = AlarmRecord(seq=1, created_at=1.0, fix=fix,
                          events=[{"from": "MONITORING", "action": "confirm", "to": "CONFIRMED"}])
        fid = db.save_fire(rec)
        db.add_screenshot(fid, screenshot_path(td))
        print(db.fires(), db.summary(), db.alarm_history(fid))
        db.close()