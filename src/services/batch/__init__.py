"""
Batch processing module.
Public API: run_batch, BatchResult, FileResult.
"""
import sys

from src.services.batch.input_handler import resolve_input
from src.services.batch.models import BatchResult, FileResult
from src.services.batch.runner import run_batch as _run_batch


def run_batch(input_path: str) -> BatchResult:
    """Entry point — accepts a folder path or ZIP archive."""
    with resolve_input(input_path) as folder:
        return _run_batch(folder)


__all__ = ["run_batch", "BatchResult", "FileResult"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.services.batch <folder_or_zip>")
        sys.exit(1)

    print(f"\nProcessing batch: {sys.argv[1]}\n")
    batch = run_batch(sys.argv[1])

    print(f"\n  {'─' * 80}")
    print(f"  Batch complete   {batch.succeeded}/{batch.total} succeeded   {batch.failed} failed")
    if batch.total_duration_seconds > 0:
        print(f"  Total cost:      ${batch.total_cost_usd:.6f}  |  Avg: ${batch.avg_cost_per_minute:.6f}/min")
    print(f"  Total time:      {batch.total_processing_seconds:.1f}s")
    if batch.output_csv:
        print(f"  Output:          {batch.output_csv}")
        print(f"                   {batch.output_json}")

    sys.exit(0 if batch.failed == 0 else 1)