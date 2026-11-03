# -*- coding: utf-8 -*-
"""无人机森林防火监测地面站 —— 核心包。

模块划分（与论文 2.4 节"系统功能模块划分"图一致）：

    config       集中配置：内参、阈值、确认窗口、告警冷却、存储路径
    types        跨模块数据结构：Detection / Telemetry / FireFix / AlarmRecord
    detector     YOLOv8 推理封装 + MockDetector + 滑动窗口连续帧确认
    localization 单目火点大地坐标解算（像素 → 相机 → 机体 → NED → 经纬度）
    telemetry    遥测仿真：巡护航线 + 云台扫视
    alarm        告警与处置状态机（六状态，含去重与误报统计）
    storage      SQLite 火情/告警/截图索引
    metrics      指标评估：定位误差、时延预算、PR 曲线
    pipeline     采集 / 推理 / 呈现 三线程流水线
"""

__version__ = "1.0.0"
__all__ = ["config", "types", "detector", "localization", "telemetry",
           "alarm", "storage", "metrics", "pipeline"]