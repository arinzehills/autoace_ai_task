# Failure Modes, Limitations & Next Steps

## Known Failure Modes

**1. Non-English calls**
Whisper transcribes non-English audio but may produce artifacts. The classifier was tuned to avoid reading Spanish transcription noise as frustration, but other languages may still produce misclassifications. Mitigation: detect language via Whisper's language detection and apply language-specific prompts.

**2. Faint background noise**
The acoustic noise floor heuristic (non-speech RMS threshold) misses very quiet or intermittent background noise (e.g. brief static bursts). These are statistically washed out in the average. Mitigation: compute percentile-based noise metrics and peak impulse detection instead of mean RMS.

**3. Speaker overlap detection**
Reliable overlap detection requires speaker diarization (e.g. pyannote.audio). The current heuristic uses energy variance and transcript garbling as proxies, which underperforms on subtle overlap. Mitigation: integrate a lightweight diarization model.

**4. Long calls with natural pauses**
A 7-second pause in a 3-minute call may be an agent looking something up — not dead air. The long silence threshold is calibrated proportionally but may still flag normal agent processing time on longer calls.

**5. Emotional intensity calibration**
Intensity (low/medium/high) is the weakest field — it relies heavily on acoustic energy which doesn't cleanly map to perceived intensity. With only 3 training examples, calibration is limited.

**6. Confidence is fixed**
The current confidence score reflects the LLM's self-reported confidence which trends toward 0.85 regardless of actual uncertainty. Proper calibration would require a held-out validation set.

## Limitations

- Only 3 labeled training examples — limited ability to validate generalization
- No fine-tuned model — relies on GPT-4o-mini zero-shot reasoning
- CPU-only Whisper — slower than GPU-accelerated alternatives
- No streaming support — batch only, not real-time

## Next Steps with More Data or Time

1. **More labeled data** — 50+ labeled calls would enable proper train/validation split, per-class F1, and prompt calibration
2. **Diarization** — integrate pyannote.audio for accurate speaker overlap detection
3. **Faster transcription** — switch to faster-whisper (CTranslate2) for 3-5x speed improvement at zero cost
4. **Confidence calibration** — use Platt scaling or temperature scaling on a held-out set
5. **Language handling** — auto-detect language and apply language-specific classification prompts
6. **Fine-tuning** — with 500+ labeled examples, fine-tune a small classifier on acoustic features to replace the LLM for the cheaper/faster fields (noise, silence, overlap)