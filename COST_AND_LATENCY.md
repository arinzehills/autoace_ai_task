# Cost & Latency Analysis

## Cost Analysis

| Component | Model | Pricing | Cost per clip |
|-----------|-------|---------|---------------|
| Transcription | openai-whisper (local) | Free | $0.00 |
| Acoustic features | librosa (local) | Free | $0.00 |
| Classification | GPT-4o-mini | $0.15/1M input, $0.60/1M output | ~$0.0003 |

**Measured costs on 3 labeled calls:**

| File | Duration | Cost | Cost/min |
|------|----------|------|----------|
| call_001.ogg | 30.9s | $0.000281 | $0.000545/min |
| call_002.ogg | 35.0s | $0.000289 | $0.000496/min |
| call_003.ogg | 171.9s | $0.000335 | $0.000117/min |
| **Average** | — | — | **$0.000228/min** |

**Cost ceiling:** $0.003/min  
**Actual cost:** ~$0.000228/min  
**Headroom:** ~13x under the ceiling

### Assumptions
- GPT-4o-mini pricing as of August 2026: $0.15/1M input tokens, $0.60/1M output tokens
- Each classification call uses ~700-900 input tokens (transcript + acoustic features) and ~150-200 output tokens
- Whisper base model runs on CPU — no GPU cost
- No API costs for transcription or acoustic analysis

---

## Latency Analysis

| Component | Time per clip |
|-----------|--------------|
| Acoustic extraction (librosa) | 0.5-2s |
| Transcription (Whisper base, CPU) | 5-15s |
| Classification (GPT-4o-mini API) | 2-4s |
| **Total per clip** | **~10-20s** |

**Measured on 3 labeled calls:**

| File | Processing time |
|------|----------------|
| call_001.ogg (31s audio) | ~10s |
| call_002.ogg (35s audio) | ~8s |
| call_003.ogg (172s audio) | ~20s |

Latency scales roughly with audio duration due to Whisper transcription time. A 3-minute call takes ~20 seconds to process. For batch use cases this is well within acceptable range.

### Bottleneck
Whisper transcription on CPU is the primary bottleneck. Using a GPU or upgrading to `faster-whisper` with CTranslate2 would reduce latency by ~3-5x with no cost increase.