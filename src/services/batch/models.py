from dataclasses import dataclass, field
from typing import Optional

from src.core.models.analysis import AudioAnalysisResult
from src.services.validation_service import ValidationResult


@dataclass
class FileResult:
    filename: str
    status: str                              # "success" | "failed" | "missing"
    analysis: Optional[AudioAnalysisResult] = None
    error: Optional[str] = None


@dataclass
class BatchResult:
    total: int
    succeeded: int
    failed: int
    results: list[FileResult] = field(default_factory=list)
    validation: Optional[ValidationResult] = None
    total_cost_usd: float = 0.0
    total_duration_seconds: float = 0.0
    total_processing_seconds: float = 0.0
    output_csv: Optional[str] = None
    output_json: Optional[str] = None

    @property
    def avg_cost_per_minute(self) -> float:
        if self.total_duration_seconds <= 0:
            return 0.0
        return self.total_cost_usd / self.total_duration_seconds * 60