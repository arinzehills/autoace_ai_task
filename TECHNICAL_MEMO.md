# Technical Memo — AutoAce Audio Analyzer

## The Business Problem This Solves

Every call center has the same blind spot — supervisors can only listen to a fraction of calls manually. That means upset customers, poor audio quality, and agents talking over customers go unnoticed until it's too late.

This system automatically listens to every call and answers five questions that directly impact business outcomes:

- **Is the customer upset or distressed?** → Flag for immediate supervisor follow-up before the customer churns
- **Is there background noise on the call?** → Identify agents working in noisy environments affecting call quality
- **Is the audio technically degraded?** → Detect infrastructure or equipment issues before they affect more calls
- **Are two people talking over each other?** → Surface training opportunities for agents who interrupt customers
- **Was there dead air on the call?** → Catch hold time problems or call flow failures in real time

Instead of a supervisor spending 10 minutes reviewing one call, this system processes an entire day's batch overnight at less than $0.001 per call — and surfaces only the ones that need human attention. The result is better customer retention, faster QA, and data-driven coaching — without adding headcount.

---

## Approaches Tested

**Approach 1 — LLM-only (transcript + prompt)**
Transcribed audio with Whisper, passed the transcript to GPT-4o-mini, and asked it to classify all 9 fields from text alone. This worked well for emotional tone on clear English calls but failed on background noise and speaker overlap detection since those are acoustic phenomena not captured in transcription.

**Approach 2 — Acoustic features only**
Extracted RMS energy, pitch, SNR, spectral flatness, and silence duration using librosa. Attempted to classify emotional tone purely from acoustics. This reliably detected silence, clipping, and some noise but was too coarse for nuanced emotion classification — the same acoustic profile can map to multiple emotional states depending on context.

**Approach 3 — Hybrid acoustic + transcript + LLM (final)**
Combined both: librosa extracts ~20 acoustic features (energy, pitch, SNR, noise floor, impulse events, spectral flatness, overlap heuristic), Whisper transcribes the audio locally, and GPT-4o-mini classifies all 9 fields using both the transcript and acoustic features as structured input. The LLM is guided by explicit rules (e.g. "do not infer frustration from loudness alone") to avoid common misclassifications.

## Final Architecture

```
Audio file
  → librosa (acoustic feature extraction)   [free, local]
  → openai-whisper base (transcription)      [free, local, ~5-15s/clip]
  → GPT-4o-mini (structured classification) [~$0.0003/min]
  → AudioAnalysisResult (9-field schema)
```

The hybrid approach outperformed both alternatives on the 3 labeled calls, particularly for background noise detection and emotional tone nuance.

## Why This Architecture

- **Cost:** local Whisper and librosa are free; GPT-4o-mini costs ~$0.0003/min, 10x under the $0.003/min ceiling
- **Privacy:** audio never leaves the machine for transcription — Whisper runs locally
- **Accuracy:** LLM reasoning over both text and acoustic signals outperforms either alone
- **Reproducibility:** no fine-tuning, no external datasets, no random state — deterministic given the same model version
- **Latency:** 10-20 seconds per clip is acceptable for batch analysis