"""
Extracts acoustic features from an audio file using librosa.
Features are passed to the classifier as supporting evidence — the LLM makes the final call.
"""
from typing import Any

import numpy as np
import librosa

from src.config.settings import (
    LONG_SILENCE_THRESHOLD_SECONDS,
    BACKGROUND_NOISE_RMS_THRESHOLD,
    SILENCE_RMS_THRESHOLD,
)

_CLIPPING_THRESHOLD = 0.98
_CLIPPING_MIN_RATIO = 0.001
_PITCH_FMIN_HZ = 75
_PITCH_FMAX_HZ = 400
_NOISE_FLOOR_PERCENTILE = 0.10
_OVERLAP_WINDOW_SECONDS = 0.5
_IMPULSE_SIGMA_THRESHOLD = 3
_EPSILON = 1e-8


def extract_acoustic_features(audio_path: str) -> dict[str, Any]:
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    hop_length = 512
    frame_duration = hop_length / sr

    # --- Energy ---
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_mean = float(np.mean(rms))
    rms_max = float(np.max(rms))
    rms_std = float(np.std(rms))

    # --- Silence detection ---
    is_silent = rms < SILENCE_RMS_THRESHOLD
    is_speech = ~is_silent
    silence_ratio = float(np.mean(is_silent))

    max_silence_s = _longest_continuous_silence(is_silent, frame_duration)
    long_silence_present = max_silence_s >= LONG_SILENCE_THRESHOLD_SECONDS

    # --- Pitch (voiced frames only, vectorised) ---
    pitches, magnitudes = librosa.piptrack(
        y=y, sr=sr, fmin=_PITCH_FMIN_HZ, fmax=_PITCH_FMAX_HZ, hop_length=hop_length
    )
    peak_idx = magnitudes.argmax(axis=0)
    voiced_mask = pitches[peak_idx, np.arange(pitches.shape[1])] > 0
    voiced_pitches = pitches[peak_idx, np.arange(pitches.shape[1])][voiced_mask]

    if voiced_pitches.size > 0:
        pitch_mean = float(np.mean(voiced_pitches))
        pitch_std = float(np.std(voiced_pitches))
        pitch_range = float(voiced_pitches.max() - voiced_pitches.min())
    else:
        pitch_mean = pitch_std = pitch_range = 0.0

    # --- SNR estimate (noise floor = quietest 10% of frames) ---
    noise_floor_idx = max(1, int(len(rms) * _NOISE_FLOOR_PERCENTILE))
    noise_floor = float(np.mean(np.sort(rms)[:noise_floor_idx]))
    snr_db = (
        float(20 * np.log10(rms_mean / noise_floor))
        if noise_floor > 0 and rms_mean > 0
        else 0.0
    )

    # --- Background noise (energy in non-speech frames) ---
    non_speech_rms = float(np.mean(rms[is_silent])) if np.any(is_silent) else 0.0
    has_background_noise_signal = non_speech_rms > BACKGROUND_NOISE_RMS_THRESHOLD

    # --- Audio quality ---
    clipping_ratio = float(np.mean(np.abs(y) > _CLIPPING_THRESHOLD))
    clipping_detected = clipping_ratio > _CLIPPING_MIN_RATIO

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]

    n_sf = min(len(spectral_flatness), len(is_speech))
    flatness_in_silence = float(np.mean(spectral_flatness[is_silent])) if np.any(is_silent) else 0.0
    flatness_in_speech = (
        float(np.mean(spectral_flatness[:n_sf][is_speech[:n_sf]]))
        if np.any(is_speech)
        else 0.0
    )

    # --- Speaker overlap heuristic (coefficient of variation in short windows) ---
    short_hop = 128
    rms_short = librosa.feature.rms(y=y, hop_length=short_hop)[0]
    window_frames = int(sr * _OVERLAP_WINDOW_SECONDS / short_hop)
    overlap_score = _energy_variance_score(rms_short, window_frames)

    # --- Impulse noise (sharp static shows as spiky RMS distribution) ---
    rms_kurtosis = float(
        np.mean((rms - rms_mean) ** 4)
        / (np.mean((rms - rms_mean) ** 2) ** 2 + _EPSILON)
    )
    rms_peak_to_mean = round(float(rms_max / (rms_mean + _EPSILON)), 2)
    impulse_frames = int(np.sum(rms > rms_mean + _IMPULSE_SIGMA_THRESHOLD * rms_std))

    return {
        "duration_seconds": round(duration, 2),
        "rms_mean": round(rms_mean, 5),
        "rms_max": round(rms_max, 5),
        "rms_std": round(rms_std, 5),
        "pitch_mean_hz": round(pitch_mean, 1),
        "pitch_std_hz": round(pitch_std, 1),
        "pitch_range_hz": round(pitch_range, 1),
        "silence_ratio": round(silence_ratio, 3),
        "long_silence_present": long_silence_present,
        "long_silence_max_seconds": round(max_silence_s, 2),
        "snr_db": round(snr_db, 1),
        "noise_floor_rms": round(noise_floor, 6),
        "non_speech_noise_rms": round(non_speech_rms, 6),
        "has_background_noise_signal": has_background_noise_signal,
        "clipping_detected": clipping_detected,
        "clipping_ratio": round(clipping_ratio, 5),
        "spectral_centroid_mean_hz": round(float(np.mean(spectral_centroid)), 1),
        "spectral_flatness_mean": round(float(np.mean(spectral_flatness)), 5),
        "spectral_flatness_in_silence": round(flatness_in_silence, 5),
        "spectral_flatness_in_speech": round(flatness_in_speech, 5),
        "zero_crossing_rate_mean": round(float(np.mean(zcr)), 5),
        "speaker_overlap_score": round(overlap_score, 3),
        "rms_kurtosis": round(rms_kurtosis, 2),
        "rms_peak_to_mean": rms_peak_to_mean,
        "impulse_frames": impulse_frames,
    }


def _longest_continuous_silence(is_silent: np.ndarray, frame_duration: float) -> float:
    max_s = current_s = 0.0
    for silent in is_silent:
        if silent:
            current_s += frame_duration
            max_s = max(max_s, current_s)
        else:
            current_s = 0.0
    return max_s


def _energy_variance_score(rms_short: np.ndarray, window_frames: int) -> float:
    scores = []
    step = window_frames // 2
    for i in range(0, len(rms_short) - window_frames, step):
        window = rms_short[i : i + window_frames]
        mean_e = np.mean(window)
        if mean_e > SILENCE_RMS_THRESHOLD:
            scores.append(np.std(window) / (mean_e + _EPSILON))
    return float(np.mean(scores)) if scores else 0.0