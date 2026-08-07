"""
Orchestrates: acoustic features → transcription → classification → AudioAnalysisResult.
Run directly: python -m src.core.processors.audio_processor path/to/call.wav
"""
import json
import time
import sys
from pathlib import Path

from src.core.tools.acoustic_tool import extract_acoustic_features
from src.core.tools.transcription_tool import transcribe
from src.core.tools.classifier_tool import classify
from src.core.models.analysis import AudioAnalysisResult


_VALID_EMOTIONAL_TONE = {"neutral", "satisfied", "frustrated", "upset", "distressed"}
_VALID_EMOTIONAL_INTENSITY = {"low", "medium", "high"}
_VALID_NOISE_SEVERITY = {"none", "low", "medium", "high"}
_VALID_AUDIO_QUALITY = {"clear", "slightly_impaired", "severely_impaired"}


def _coerce_enum(value: str, valid: set, default: str) -> str:
    """Return value if it's a valid enum member, otherwise the default."""
    return value if value in valid else default


def analyze(audio_path: str) -> AudioAnalysisResult:
    path = Path(audio_path)
    start = time.time()

    # Step 1 — acoustic features (free, local)
    acoustic = extract_acoustic_features(audio_path)

    # Step 2 — transcription (free, local whisper)
    transcript = transcribe(audio_path)

    # Step 3 — speaking rate (words/min) requires both transcript + duration
    word_count = len(transcript.split()) if transcript else 0
    duration_min = acoustic.get("duration_seconds", 1) / 60.0
    acoustic["speaking_rate_wpm"] = round(word_count / duration_min, 1) if duration_min > 0 else 0.0

    # Step 4 — LLM classification
    result_dict, cost_usd = classify(transcript, acoustic)

    processing_seconds = round(time.time() - start, 2)
    duration = acoustic.get("duration_seconds", 0)

    raw_noise_type = result_dict.get("background_noise_type", "")
    noise_present = bool(result_dict.get("background_noise_present", False))

    return AudioAnalysisResult(
        emotional_tone=_coerce_enum(
            result_dict.get("emotional_tone", "neutral"), _VALID_EMOTIONAL_TONE, "neutral"
        ),
        emotional_intensity=_coerce_enum(
            result_dict.get("emotional_intensity", "low"), _VALID_EMOTIONAL_INTENSITY, "low"
        ),
        background_noise_present=noise_present,
        background_noise_type=raw_noise_type if isinstance(raw_noise_type, str) else "",
        background_noise_severity=_coerce_enum(
            result_dict.get("background_noise_severity", "none"), _VALID_NOISE_SEVERITY, "none"
        ),
        audio_quality=_coerce_enum(
            result_dict.get("audio_quality", "clear"), _VALID_AUDIO_QUALITY, "clear"
        ),
        speaker_overlap_present=bool(result_dict.get("speaker_overlap_present", False)),
        long_silence_present=bool(result_dict.get("long_silence_present", False)),
        confidence=min(1.0, max(0.0, float(result_dict.get("confidence", 0.5)))),
        filename=path.name,
        duration_seconds=duration,
        transcript=transcript,
        cost_usd=cost_usd,
        processing_seconds=processing_seconds,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.core.processors.audio_processor <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    print(f"\nAnalyzing: {audio_file}\n")

    result = analyze(audio_file)

    print("=== OUTPUT (9-field schema) ===")
    print(json.dumps(result.to_output_dict(), indent=2))

    print("\n=== METADATA ===")
    print(f"  Duration:    {result.duration_seconds}s")
    print(f"  Transcript:  {result.transcript[:120]}..." if result.transcript and len(result.transcript) > 120 else f"  Transcript:  {result.transcript}")
    print(f"  Cost:        ${result.cost_usd:.6f} ({result.cost_usd / result.duration_seconds * 60:.6f}/min)")
    print(f"  Processed:   {result.processing_seconds}s")