from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class CostTracker:
    total_cost_usd: float = 0.0
    total_duration_seconds: float = 0.0

    def add(self, cost_usd: float, duration_seconds: float) -> None:
        self.total_cost_usd += cost_usd
        self.total_duration_seconds += duration_seconds

    @property
    def cost_per_minute(self) -> float:
        if self.total_duration_seconds <= 0:
            return 0.0
        return self.total_cost_usd / self.total_duration_seconds * 60


def timestamped_filename(prefix: str, suffix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}{suffix}"