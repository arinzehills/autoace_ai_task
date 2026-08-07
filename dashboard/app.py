import json
import sys
import tempfile
from pathlib import Path

# Ensure project root is on the path regardless of where streamlit is launched from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src.config.settings import DASHBOARD_PASSWORD, DASHBOARD_USERNAME
from src.services.batch import BatchResult, FileResult, run_batch

st.set_page_config(
    page_title="AutoAce Audio Analyzer",
    page_icon="🎙️",
    layout="wide",
)

_TONE_COLORS = {
    "upset": "🔴",
    "distressed": "🔴",
    "frustrated": "🟠",
    "neutral": "⚪",
    "satisfied": "🟢",
}


def main() -> None:
    if not st.session_state.get("authenticated"):
        _render_login()
    else:
        _render_dashboard()


# ── Auth ─────────────────────────────────────────────────────────────────────

def _render_login() -> None:
    col = st.columns([1, 2, 1])[1]
    with col:
        st.title("🎙️ArinzeHills AutoAce Audio Analyzer")
        st.markdown("---")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Log in", use_container_width=True):
            if username == DASHBOARD_USERNAME and password == DASHBOARD_PASSWORD:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Invalid credentials")


# ── Dashboard ─────────────────────────────────────────────────────────────────

def _render_dashboard() -> None:
    st.title("🎙️ArinzeHills AutoAce Audio Analyzer")
    with st.sidebar:
        st.markdown(f"Logged in as **{DASHBOARD_USERNAME}**")
        if st.button("Log out"):
            st.session_state.clear()
            st.rerun()

    _render_upload_section()

    if "batch_result" in st.session_state:
        _render_results(st.session_state["batch_result"])


def _render_upload_section() -> None:
    st.subheader("Upload Evaluation Batch")
    st.caption(
        "Upload a ZIP archive containing audio files and a CSV manifest "
        "(labels.csv with `name` and `result_json` columns)."
    )

    uploaded = st.file_uploader(
        "Drop ZIP here or click to browse",
        type=["zip"],
        label_visibility="collapsed",
    )

    if uploaded and st.button("▶  Process Batch", type="primary"):
        _run_and_store(uploaded)


def _run_and_store(uploaded_file) -> None:
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    progress_bar = st.progress(0, text="Starting…")
    status_area = st.empty()
    log_lines: list[str] = []

    def on_progress(done: int, total: int, result: FileResult) -> None:
        pct = done / total
        progress_bar.progress(pct, text=f"Processing {done}/{total} files…")
        icon = "✓" if result.status == "success" else "✗"
        tone = (
            f"{result.analysis.emotional_tone}/{result.analysis.emotional_intensity}"
            if result.analysis else "—"
        )
        log_lines.append(f"{icon} {result.filename}  →  {tone}")
        status_area.code("\n".join(log_lines))

    batch = run_batch(tmp_path, on_file_complete=on_progress)

    progress_bar.progress(1.0, text="Complete!")
    status_area.empty()
    st.session_state["batch_result"] = batch


# ── Results ───────────────────────────────────────────────────────────────────

def _render_results(batch: BatchResult) -> None:
    st.markdown("---")
    st.subheader("Results")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total files", batch.total)
    col2.metric("Succeeded", batch.succeeded)
    col3.metric("Failed", batch.failed)
    col4.metric("Avg cost / min", f"${batch.avg_cost_per_minute:.6f}")

    st.markdown("####")

    df = _build_results_df(batch)
    st.dataframe(df, use_container_width=True, hide_index=True)

    _render_downloads(batch)


def _build_results_df(batch: BatchResult) -> pd.DataFrame:
    rows = []
    for r in batch.results:
        if r.analysis:
            a = r.analysis
            tone_icon = _TONE_COLORS.get(a.emotional_tone, "⚪")
            rows.append({
                "File": r.filename,
                "Tone": f"{tone_icon} {a.emotional_tone}",
                "Intensity": a.emotional_intensity,
                "Noise": "Yes" if a.background_noise_present else "No",
                "Noise Type": a.background_noise_type or "—",
                "Noise Severity": a.background_noise_severity,
                "Audio Quality": a.audio_quality,
                "Overlap": "Yes" if a.speaker_overlap_present else "No",
                "Long Silence": "Yes" if a.long_silence_present else "No",
                "Confidence": a.confidence,
                "Cost USD": f"${a.cost_usd:.6f}",
                "Status": "✓",
            })
        else:
            rows.append({
                "File": r.filename,
                "Tone": "—", "Intensity": "—", "Noise": "—",
                "Noise Type": "—", "Noise Severity": "—", "Audio Quality": "—",
                "Overlap": "—", "Long Silence": "—", "Confidence": "—",
                "Cost USD": "—", "Status": f"✗ {r.error}",
            })
    return pd.DataFrame(rows)


def _render_downloads(batch: BatchResult) -> None:
    st.markdown("#### Download Results")

    csv_rows = []
    for r in batch.results:
        csv_rows.append({
            "name": r.filename,
            "result_json": json.dumps(r.analysis.to_output_dict()) if r.analysis else "",
            "error": r.error or "",
        })

    csv_df = pd.DataFrame(csv_rows)
    json_payload = [
        {
            "name": r.filename,
            "status": r.status,
            "result": r.analysis.to_output_dict() if r.analysis else None,
            "error": r.error,
        }
        for r in batch.results
    ]

    col1, col2 = st.columns(2)
    col1.download_button(
        label="⬇  Download CSV",
        data=csv_df.to_csv(index=False),
        file_name="autoace_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
    col2.download_button(
        label="⬇  Download JSON",
        data=json.dumps(json_payload, indent=2),
        file_name="autoace_results.json",
        mime="application/json",
        use_container_width=True,
    )


main()