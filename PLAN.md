# AutoAce Audio Analyzer — Build Plan

## Folder Structure

```
autoace-audio-analyzer/
│
├── src/
│   ├── config/
│   │   └── settings.py              # API keys, model names, cost ceiling const
│   │
│   ├── core/
│   │   ├── models/
│   │   │   └── analysis.py          # AudioAnalysisResult dataclass (9-field schema)
│   │   │
│   │   ├── tools/
│   │   │   ├── acoustic_tool.py     # librosa → RMS, pitch, SNR, silence, overlap detection
│   │   │   ├── transcription_tool.py# openai-whisper local → transcript text
│   │   │   └── classifier_tool.py   # GPT-4o-mini prompt → parses final JSON output
│   │   │
│   │   └── processors/
│   │       └── audio_processor.py   # orchestrates: acoustic → transcribe → classify → result
│   │
│   ├── services/
│   │   ├── batch_service.py         # reads ZIP + CSV manifest, loops files, collects results
│   │   └── validation_service.py    # manifest vs files cross-check, reports missing/unmatched
│   │
│   └── common/
│       └── utils.py                 # cost tracker, timer, file helpers
│
├── dashboard/
│   └── app.py                       # Streamlit: login → upload → progress → results → download
│
├── data/
│   └── labeled/                     # 3 labeled production calls + ground truth
│       ├── call_001.ogg
│       ├── call_002.ogg
│       ├── call_003.ogg
│       └── labels.csv
│
├── tests/
│   └── test_labeled_calls.py        # run on 3 labeled calls, print accuracy + confusion matrix
│
├── requirements.txt
├── .env.example
└── PLAN.md
```

---

## Vertical Slices (build order)

Each slice is fully testable end-to-end before moving to the next.
Never revisit a completed slice.

---

### Slice 1 — Single File Analyzer (Core)
**What it covers:** `settings.py` + `analysis.py` + `acoustic_tool.py` + `transcription_tool.py` + `classifier_tool.py` + `audio_processor.py`

**Test:** Run `python -m src.core.processors.audio_processor path/to/call.wav`  
**Output:** Prints the full 9-field JSON to terminal for one audio file  
**Done when:** All 9 fields return valid values on a real audio file

---

### Slice 2 — Batch Processor (CLI)
**What it covers:** `batch_service.py` + `validation_service.py` + `utils.py`

**Test:** Run `python -m src.services.batch_service path/to/folder_or.zip`  
**Output:** Prints per-file results + summary table (cost, latency, errors) to terminal  
**Done when:** Processes all 3 labeled calls, handles a bad file gracefully, prints cost per minute

---

### Slice 3 — Streamlit Dashboard (UI)
**What it covers:** `dashboard/app.py` — full UI wired to batch_service

**Test:** Run `streamlit run dashboard/app.py` locally  
**Output:** Login → ZIP upload → progress bar → results table → download CSV/JSON  
**Done when:** Can log in, upload the labeled calls ZIP, see results, download output

---

### Slice 4 — Deploy
**What it covers:** `requirements.txt`, Streamlit Cloud config, `.env` secrets setup

**Test:** Hit the public URL, log in, upload, download  
**Done when:** AutoAce can access the live URL with provided credentials

---

### Slice 5 — Validation & Write-ups
**What it covers:** `tests/test_labeled_calls.py` + technical memo + cost/latency analysis

**Test:** Run `python tests/test_labeled_calls.py` → prints accuracy, F1, confusion matrix  
**Done when:** All deliverable write-ups are complete and ready to submit

---

## Cost Target
- Local Whisper (base model): $0.00/min
- GPT-4o-mini classification: ~$0.0002/min
- **Total: ~$0.0002–0.0005/min** (target ceiling: $0.003/min)

## Key Constraints
- Emotion tone: do NOT infer frustration from loudness alone
- Background noise: do NOT infer from poor audio quality alone
- Each file failure must be isolated — batch must continue
- No customer audio uploaded to unapproved services (whisper runs locally)