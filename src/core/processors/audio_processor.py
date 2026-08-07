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


def analyze(audio_path: str) -> AudioAnalysisResult:
    path = Path(audio_path)
    start = time.time()

    # Step 1 — acoustic features (free, local)
    acoustic = extract_acoustic_features(audio_path)

    # Step 2 — transcription (free, local whisper)
    transcript = transcribe(audio_path)

    # Step 3 — LLM classification
    result_dict, cost_usd = classify(transcript, acoustic)

    processing_seconds = round(time.time() - start, 2)
    duration = acoustic.get("duration_seconds", 0)
    cost_per_minute = (cost_usd / duration * 60) if duration > 0 else 0

    return AudioAnalysisResult(
        emotional_tone=result_dict.get("emotional_tone", "neutral"),
        emotional_intensity=result_dict.get("emotional_intensity", "low"),
        background_noise_present=bool(result_dict.get("background_noise_present", False)),
        background_noise_type=result_dict.get("background_noise_type", ""),
        background_noise_severity=result_dict.get("background_noise_severity", "none"),
        audio_quality=result_dict.get("audio_quality", "clear"),
        speaker_overlap_present=bool(result_dict.get("speaker_overlap_present", False)),
        long_silence_present=bool(result_dict.get("long_silence_present", False)),
        confidence=float(result_dict.get("confidence", 0.5)),
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