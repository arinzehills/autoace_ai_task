"""
Extracts acoustic features from an audio file using librosa.
These features are passed to the classifier as supporting evidence — the LLM makes the final call.
"""
import numpy as np
import librosa
from src.config.settings import (
    LONG_SILENCE_THRESHOLD_SECONDS,
    BACKGROUND_NOISE_RMS_THRESHOLD,
    SILENCE_RMS_THRESHOLD,
)


def extract_acoustic_features(audio_path: str) -> dict:
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    hop_length = 512
    frame_duration = hop_length / sr

    # Energy
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_mean = float(np.mean(rms))
    rms_max = float(np.max(rms))
    rms_std = float(np.std(rms))

    # Silence detection
    is_silent = rms < SILENCE_RMS_THRESHOLD
    silence_ratio = float(np.mean(is_silent))

    # Longest continuous silence
    max_silence_s = 0.0
    current_silence_s = 0.0
    for silent in is_silent:
        if silent:
            current_silence_s += frame_duration
            max_silence_s = max(max_silence_s, current_silence_s)
        else:
            current_silence_s = 0.0

    long_silence_present = max_silence_s >= LONG_SILENCE_THRESHOLD_SECONDS

    # Pitch — voiced frame estimation via piptrack
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=75, fmax=400, hop_length=hop_length)
    voiced = []
    for t in range(pitches.shape[1]):
        idx = magnitudes[:, t].argmax()
        p = pitches[idx, t]
        if p > 0:
            voiced.append(p)

    if voiced:
        pitch_mean = float(np.mean(voiced))
        pitch_std = float(np.std(voiced))
        pitch_range = float(np.max(voiced) - np.min(voiced))
    else:
        pitch_mean = pitch_std = pitch_range = 0.0

    # SNR estimate — noise floor = mean of quietest 10% of frames
    rms_sorted = np.sort(rms)
    noise_floor = float(np.mean(rms_sorted[: max(1, len(rms_sorted) // 10)]))
    snr_db = float(20 * np.log10(rms_mean / noise_floor)) if noise_floor > 0 and rms_mean > 0 else 0.0

    # Background noise — energy in non-speech frames + spectral flatness during silence
    non_speech_rms = float(np.mean(rms[is_silent])) if np.any(is_silent) else 0.0
    has_background_noise_signal = non_speech_rms > BACKGROUND_NOISE_RMS_THRESHOLD

    # Audio quality signals
    clipping_ratio = float(np.mean(np.abs(y) > 0.98))
    clipping_detected = clipping_ratio > 0.001  # >0.1% of samples clipped
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]

    # Speaker overlap heuristic — RMS energy variance in short windows during speech
    # Overlapping speakers create rapid irregular energy fluctuations
    is_speech = ~is_silent
    short_hop = 128
    rms_short = librosa.feature.rms(y=y, hop_length=short_hop)[0]
    window_size = int(sr * 0.5 / short_hop)  # 0.5s windows
    energy_variance_scores = []
    for i in range(0, len(rms_short) - window_size, window_size // 2):
        window = rms_short[i : i + window_size]
        mean_e = np.mean(window)
        if mean_e > SILENCE_RMS_THRESHOLD:  # only in speech frames
            cv = np.std(window) / (mean_e + 1e-8)  # coefficient of variation
            energy_variance_scores.append(cv)
    overlap_score = float(np.mean(energy_variance_scores)) if energy_variance_scores else 0.0

    # Impulse noise detection — sharp static shows as spiky RMS distribution
    rms_kurtosis = float(np.mean((rms - rms_mean) ** 4) / (np.mean((rms - rms_mean) ** 2) ** 2 + 1e-10))
    rms_peak_to_mean = round(float(rms_max / (rms_mean + 1e-8)), 2)
    impulse_frames = int(np.sum(rms > rms_mean + 3 * rms_std))  # frames >3 sigma above mean

    # Spectral flatness in silence (broadband noise = high) and in speech (static in speech = high)
    flatness_in_silence = float(np.mean(spectral_flatness[is_silent])) if np.any(is_silent) else 0.0
    n_sf = min(len(spectral_flatness), len(is_speech))
    flatness_in_speech = float(np.mean(spectral_flatness[:n_sf][is_speech[:n_sf]])) if np.any(is_speech) else 0.0

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
        "speaker_overlap_score": round(overlap_score, 3),        "rms_kurtosis": round(rms_kurtosis, 2),
        "rms_peak_to_mean": rms_peak_to_mean,
        "impulse_frames": impulse_frames,
    }