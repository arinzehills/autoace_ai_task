"""
Persists batch results to CSV and JSON in the results/ directory.
"""
import csv
import json
from pathlib import Path

from src.common.utils import timestamped_filename
from src.services.batch.models import FileResult

_RESULTS_DIR = Path("results")


def save_results(results: list[FileResult]) -> tuple[str, str]:
    """Saves results to timestamped CSV and JSON files. Returns (csv_path, json_path)."""
    _RESULTS_DIR.mkdir(exist_ok=True)

    base = timestamped_filename("batch_results", "")
    csv_path = _RESULTS_DIR / f"{base}.csv"
    json_path = _RESULTS_DIR / f"{base}.json"

    _write_csv(results, csv_path)
    _write_json(results, json_path)

    return str(csv_path), str(json_path)


def _write_csv(results: list[FileResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "result_json", "error"])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "name": r.filename,
                "result_json": json.dumps(r.analysis.to_output_dict()) if r.analysis else "",
                "error": r.error or "",
            })


def _write_json(results: list[FileResult], path: Path) -> None:
    payload = [
        {
            "name": r.filename,
            "status": r.status,
            "result": r.analysis.to_output_dict() if r.analysis else None,
            "metadata": {
                "duration_seconds": r.analysis.duration_seconds,
                "cost_usd": r.analysis.cost_usd,
                "processing_seconds": r.analysis.processing_seconds,
                "transcript": r.analysis.transcript,
            } if r.analysis else None,
            "error": r.error,
        }
        for r in results
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")