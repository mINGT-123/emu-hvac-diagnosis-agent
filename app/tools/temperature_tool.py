from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class TempWindowStats:
    latest_temp: float
    mean_temp: float
    min_temp: float
    max_temp: float
    slope_per_min: float

    def as_text(self) -> str:
        direction = "升温" if self.slope_per_min > 0 else "降温"
        return (
            f"最近窗口温度统计: 最新={self.latest_temp:.2f}C, 平均={self.mean_temp:.2f}C, "
            f"最小={self.min_temp:.2f}C, 最大={self.max_temp:.2f}C, 趋势={direction}({self.slope_per_min:.3f} C/min)"
        )


def get_cabin_temp(csv_path: str, window: int = 8) -> str:
    """读取客室温度 CSV，并返回滑动窗口统计结果。"""
    path = Path(csv_path)
    if not path.exists():
        return f"未找到温度文件: {path}"

    df = pd.read_csv(path)
    required_cols = {"timestamp", "temp_c"}
    if not required_cols.issubset(df.columns):
        return f"CSV 缺少必需字段: {required_cols}"

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "temp_c"]).sort_values("timestamp")

    if len(df) < 2:
        return "有效温度数据不足，无法进行趋势分析。"

    window_df = df.tail(max(2, window)).reset_index(drop=True)
    minutes = (window_df["timestamp"].iloc[-1] - window_df["timestamp"].iloc[0]).total_seconds() / 60
    minutes = max(minutes, 1e-6)

    slope = (window_df["temp_c"].iloc[-1] - window_df["temp_c"].iloc[0]) / minutes
    stats = TempWindowStats(
        latest_temp=float(window_df["temp_c"].iloc[-1]),
        mean_temp=float(window_df["temp_c"].mean()),
        min_temp=float(window_df["temp_c"].min()),
        max_temp=float(window_df["temp_c"].max()),
        slope_per_min=float(slope),
    )
    return stats.as_text()
