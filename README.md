# AutoAce Audio Analyzer

A production-grade system for classifying emotional tone and detecting background noise in call center audio. Built for the AutoAce AI Technical Trial.

## Live Dashboard

**URL:** *(add your Streamlit URL here)*  
**Username:** autoace  
**Password:** *(provided separately)*

## What It Does

For each audio clip, the system returns a structured JSON result:

```json
{
  "emotional_tone": "frustrated",
  "emotional_intensity": "medium",
  "background_noise_present": true,
  "background_noise_type": "office chatter",
  "background_noise_severity": "low",
  "audio_quality": "clear",
  "speaker_overlap_present": false,
  "long_silence_present": false,
  "confidence": 0.85
}
```

## Architecture

```
Audio file
  → librosa       (acoustic features — free, local)
  → Whisper base  (transcription — free, local)
  → GPT-4o-mini   (classification — ~$0.0003/min)
```

**Cost:** ~$0.0003/min average — 10x under the $0.003/min ceiling.

```
src/
  config/           settings and constants
  core/
    models/         AudioAnalysisResult schema
    tools/          acoustic_tool, transcription_tool, classifier_tool
    processors/     audio_processor (orchestrator)
  services/
    batch/          batch runner, input handler, output writer, models
    validation_service.py
  common/           shared utilities
dashboard/
  app.py            Streamlit web dashboard
data/
  labeled/          3 labeled production calls + ground truth CSV
```

## Local Setup

**Requirements:** Python 3.13, ffmpeg

```bash
# Install ffmpeg (Mac)
brew install ffmpeg

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Running

**Single file:**
```bash
python -m src.core.processors.audio_processor path/to/call.wav
```

**Batch (folder or ZIP):**
```bash
python -m src.services.batch data/labeled/
python -m src.services.batch evaluation_batch.zip
```

**Dashboard (local):**
```bash
streamlit run dashboard/app.py
```

## Batch Input Format

Upload a ZIP containing audio files at the root plus a CSV manifest:

```
evaluation_batch/
  call_001.wav
  call_002.mp3
  labels.csv
```

**CSV format:**
```
name,result_json
call_001.wav,"{""emotional_tone"":""frustrated"",...}"
call_002.mp3,""
```

Supported formats: `.wav`, `.mp3`, `.ogg`, `.m4a`, `.flac`, `.webm`

## Cost & Latency

| Component | Cost | Latency |
|-----------|------|---------|
| Whisper (local) | $0.00/min | ~5–15s/clip |
| GPT-4o-mini | ~$0.0003/min | ~2–3s/clip |
| **Total** | **~$0.0003/min** | **~10–20s/clip** |

Ceiling: $0.003/min — system runs at ~10% of the limit.