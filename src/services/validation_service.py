import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.config.settings import SUPPORTED_AUDIO_EXTENSIONS


@dataclass
class ValidationResult:
    matched: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)    # in CSV but no audio file found
    unmatched_files: list[str] = field(default_factory=list)  # audio file found but not in CSV
    manifest_path: Optional[str] = None


def find_manifest(folder: Path) -> Optional[Path]:
    """Return the CSV manifest in a folder, preferring labels.csv or manifest.csv."""
    candidates = list(folder.glob("*.csv"))
    if not candidates:
        return None
    preferred = next(
        (c for c in candidates if c.name in {"labels.csv", "manifest.csv"}),
        None,
    )
    return preferred or candidates[0]


def find_audio_files(folder: Path) -> list[Path]:
    return sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )


def validate_batch(folder: Path, manifest: Path) -> ValidationResult:
    with manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        manifest_names = {row["name"] for row in reader if "name" in row}

    audio_names = {f.name for f in find_audio_files(folder)}

    return ValidationResult(
        manifest_path=str(manifest),
        matched=sorted(manifest_names & audio_names),
        missing_files=sorted(manifest_names - audio_names),
        unmatched_files=sorted(audio_names - manifest_names),
    )