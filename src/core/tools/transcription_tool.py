"""
Local Whisper transcription — free, audio never leaves the machine.
Downloads model on first run (~140MB for base).
"""
import whisper
from src.config.settings import WHISPER_MODEL

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = whisper.load_model(WHISPER_MODEL)
    return _model


def transcribe(audio_path: str) -> str:
    model = _get_model()
    result = model.transcribe(audio_path, language="en", fp16=False)
    return result.get("text", "").strip()