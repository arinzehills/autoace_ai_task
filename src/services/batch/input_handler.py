"""
Resolves batch input — accepts a folder path or a ZIP archive.
Extracts ZIP to a temp directory and returns the working folder.
"""
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


@contextmanager
def resolve_input(input_path: str) -> Generator[Path, None, None]:
    """
    Yields a Path to the folder containing audio files and manifest.
    Handles both plain folders and ZIP archives transparently.
    """
    path = Path(input_path)

    if path.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(path) as zf:
                zf.extractall(tmp)
            yield Path(tmp)
    elif path.is_dir():
        yield path
    else:
        raise ValueError(f"Input must be a folder or .zip file: {input_path}")