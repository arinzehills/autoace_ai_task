from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AudioAnalysisResult:
    # Required output schema (9 fields)
    emotional_tone: str          # neutral | satisfied | frustrated | upset | distressed
    emotional_intensity: str     # low | medium | high
    background_noise_present: bool
    background_noise_type: str   # empty string when no noise
    background_noise_severity: str  # none | low | medium | high
    audio_quality: str           # clear | slightly_impaired | severely_impaired
    speaker_overlap_present: bool
    long_silence_present: bool
    confidence: float            # 0.0 – 1.0

    # Metadata (not in output schema)
    filename: Optional[str] = None
    duration_seconds: Optional[float] = None
    transcript: Optional[str] = None
    cost_usd: Optional[float] = None
    processing_seconds: Optional[float] = None
    error: Optional[str] = None

    def to_output_dict(self) -> dict:
        """Returns only the required 9-field schema."""
        return {
            "emotional_tone": self.emotional_tone,
            "emotional_intensity": self.emotional_intensity,
            "background_noise_present": self.background_noise_present,
            "background_noise_type": self.background_noise_type,
            "background_noise_severity": self.background_noise_severity,
            "audio_quality": self.audio_quality,
            "speaker_overlap_present": self.speaker_overlap_present,
            "long_silence_present": self.long_silence_present,
            "confidence": self.confidence,
        }

    def to_full_dict(self) -> dict:
        """Returns output schema plus metadata."""
        return {
            **self.to_output_dict(),
            "filename": self.filename,
            "duration_seconds": self.duration_seconds,
            "transcript": self.transcript,
            "cost_usd": self.cost_usd,
            "processing_seconds": self.processing_seconds,
            "error": self.error,
        }