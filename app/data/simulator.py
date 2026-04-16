from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class CabinTempSimulator:
    base_temp: float = 24.0
    drift_per_min: float = 0.15
    noise_std: float = 0.2

    def generate(self, minutes: int = 30, start_time: datetime | None = None) -> pd.DataFrame:
        if start_time is None:
            start_time = datetime.now().replace(second=0, microsecond=0)

        times: Iterable[datetime] = (start_time + timedelta(minutes=i) for i in range(minutes))
        trend = np.array([self.base_temp + i * self.drift_per_min for i in range(minutes)])
        noise = np.random.normal(0, self.noise_std, size=minutes)

        return pd.DataFrame(
            {
                "timestamp": list(times),
                "temp_c": (trend + noise).round(2),
            }
        )


def save_simulated_csv(output_path: Path, minutes: int = 30) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    simulator = CabinTempSimulator()
    df = simulator.generate(minutes=minutes)
    df.to_csv(output_path, index=False)
    return output_path
