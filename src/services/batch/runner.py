"""
Batch runner — orchestrates input resolution, validation, processing, and output.
Each concern is delegated to its own module.
"""
from pathlib import Path
from typing import Callable, Optional

from src.common.utils import CostTracker
from src.core.models.analysis import AudioAnalysisResult
from src.core.processors.audio_processor import analyze
from src.services.batch.models import BatchResult, FileResult
from src.services.batch.output_writer import save_results
from src.services.validation_service import (
    ValidationResult,
    find_audio_files,
    find_manifest,
    validate_batch,
)


def run_batch(folder: Path, on_file_complete: Optional[Callable[[int, int, FileResult], None]] = None) -> BatchResult:
    manifest = find_manifest(folder)
    validation, files_to_process = _resolve_files(folder, manifest)

    print(f"  {'File':<25} {'Tone':<22} {'Noise':<14} {'Cost':>10}  {'Time':>6}")
    print(f"  {'─' * 80}")

    results: list[FileResult] = []
    cost_tracker = CostTracker()

    for idx, audio_path in enumerate(files_to_process):
        file_result = _process_file(audio_path)
        results.append(file_result)
        if on_file_complete:
            on_file_complete(idx + 1, len(files_to_process), file_result)
        if file_result.analysis:
            cost_tracker.add(
                file_result.analysis.cost_usd or 0.0,
                file_result.analysis.duration_seconds or 0.0,
            )

    if validation:
        for name in validation.missing_files:
            results.append(FileResult(filename=name, status="missing", error="File not found on disk"))

    succeeded = sum(1 for r in results if r.status == "success")
    total_processing = sum(
        r.analysis.processing_seconds or 0.0 for r in results if r.analysis
    )

    output_csv, output_json = save_results(results)

    return BatchResult(
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        results=results,
        validation=validation,
        total_cost_usd=cost_tracker.total_cost_usd,
        total_duration_seconds=cost_tracker.total_duration_seconds,
        total_processing_seconds=total_processing,
        output_csv=output_csv,
        output_json=output_json,
    )


def _resolve_files(
    folder: Path, manifest: Optional[Path]
) -> tuple[Optional[ValidationResult], list[Path]]:
    if not manifest:
        return None, find_audio_files(folder)

    validation = validate_batch(folder, manifest)

    for name in validation.missing_files:
        print(f"  ⚠  {name:<25} not found on disk — skipping")
    for name in validation.unmatched_files:
        print(f"  ⚠  {name:<25} not in manifest — skipping")

    return validation, [folder / name for name in validation.matched]


def _process_file(audio_path: Path) -> FileResult:
    try:
        result = analyze(str(audio_path))
        noise_label = result.background_noise_type or "no noise"
        tone_label = f"{result.emotional_tone}/{result.emotional_intensity}"
        print(
            f"  ✓ {audio_path.name:<25} {tone_label:<22} {noise_label:<14}"
            f" ${result.cost_usd:.6f}  {result.processing_seconds}s"
        )
        return FileResult(filename=audio_path.name, status="success", analysis=result)

    except Exception as exc:
        print(f"  ✗ {audio_path.name:<25} ERROR: {exc}")
        return FileResult(filename=audio_path.name, status="failed", error=str(exc))