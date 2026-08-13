
import streamlit as st
import io
import json
import math
import soundfile as sf
import torch
import torchaudio
from app import predict, device

st.set_page_config(page_title="Caption-Craft Transcriber", page_icon="🎧", layout="centered")

CSS = """
<style>
:root{--bg:#0b1220;--card:#0f1724;--muted:#9fb0c8;--accent:#4f46e5}
body { background: linear-gradient(180deg,#071020,#0b1220); color: #e6eef8; }
.container { max-width: 980px; margin: 0 auto; }
.hero { padding: 28px 0 8px 0; }
.title { font-size: 40px; font-weight: 800; margin: 0 0 6px 0; letter-spacing: -0.5px }
.subtitle { color: var(--muted); margin: 0 0 18px 0; }
.card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border: 1px solid rgba(255,255,255,0.03); padding: 18px; border-radius: 14px; }
.features { display:flex; gap:12px; margin-top:14px; }
.feature { flex:1; padding:16px; border-radius:12px; background: rgba(255,255,255,0.02); text-align:center; box-shadow: 0 6px 20px rgba(2,6,23,0.6); }
.feature strong { display:block; font-size:18px }
.upload-area { border: 2px dashed rgba(255,255,255,0.04); padding: 28px; border-radius: 12px; text-align:center; }
.transcript { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding:16px; border-radius:12px; margin-top:14px; color:#dbeafe }
.btn { background:linear-gradient(90deg,var(--accent),#60a5fa); color:white; padding:10px 18px; border-radius:10px; border:none; font-weight:600 }
.muted { color:var(--muted); font-size:14px }
.meta { color: #b9cfe6; font-size:13px; margin-top:8px }
.small { font-size:12px; color:#9fb0c8 }
.flex { display:flex; gap:12px; align-items:center }
.pill { background:rgba(255,255,255,0.03); padding:8px 12px; border-radius:999px; font-weight:600 }
.center { display:flex; justify-content:center }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='hero'><div class='title'>Caption-Craft — Speech Transcription</div><div class='subtitle'>Upload audio and get accurate captions quickly.</div></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='feature card'><strong>📝 High Accuracy</strong><div class='muted'>State-of-the-art model</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='feature card'><strong>⚡ Fast</strong><div class='muted'>Quick audio processing</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='feature card'><strong>📄 Export</strong><div class='muted'>Copy or download transcripts</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    upload_col, action_col = st.columns([3,1])

    with upload_col:
        st.markdown("<div class='card upload-area'>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop or choose an audio file", type=["wav","flac","mp3","m4a","ogg"], help="200MB per file")
        st.markdown("<div class='muted'>Supported: WAV, FLAC, MP3, M4A, OGG</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with action_col:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        transcribe_btn = st.button("Transcribe", key="transcribe", use_container_width=True)

    file_meta = None
    if uploaded is not None:
        st.audio(uploaded)
        # gather metadata
        uploaded.seek(0)
        raw = uploaded.read()
        uploaded.seek(0)
        try:
            info = sf.info(io.BytesIO(raw))
            duration = info.frames / info.samplerate if info.frames and info.samplerate else None
            size_mb = len(raw) / (1024*1024)
            file_meta = {
                "name": getattr(uploaded, 'name', 'audio'),
                "size_mb": round(size_mb, 2),
                "samplerate": info.samplerate,
                "duration_s": round(duration, 2) if duration else None,
            }
        except Exception:
            file_meta = {"name": getattr(uploaded, 'name', 'audio')}

    transcription = None
    if transcribe_btn and uploaded is not None:
        with st.spinner("Transcribing audio — this may take a moment..."):
            p = st.progress(0)
            try:
                audio_bytes = uploaded.read()
                audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
                p.progress(20)

                # Convert to mono if needed
                audio_np = audio_data
                if audio_np.ndim == 2:
                    audio_np = audio_np.mean(axis=1)
                p.progress(35)

                # Convert to tensor shape (1, samples)
                waveform = torch.from_numpy(audio_np).float().unsqueeze(0)

                target_sr = 16000
                if sample_rate != target_sr:
                    try:
                        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
                        waveform = resampler(waveform)
                        sample_rate = target_sr
                    except Exception:
                        # fallback: proceed without resampling and warn
                        st.warning(f"Could not resample from {sample_rate}Hz to {target_sr}Hz — continuing with original rate")
                p.progress(65)

                # move to device
                waveform = waveform.to(device)
                p.progress(80)

                transcription = predict(waveform)
                p.progress(100)
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                p.progress(0)

    if transcription:
        st.markdown("<div class='transcript card'><strong>Transcription</strong></div>", unsafe_allow_html=True)
        st.write("")
        st.text_area("Transcription", value=transcription, height=220, label_visibility="collapsed")
        st.download_button("Download .txt", transcription, file_name="transcription.txt")
        # Copy button via inline HTML+JS (avoids deprecated components.html)
        safe_text = json.dumps(transcription)
        copy_button_html = f"""
            <div style='margin-top:8px'>
              <button class='btn' onclick='navigator.clipboard.writeText({safe_text})'>Copy to clipboard</button>
            </div>
        """
        st.markdown(copy_button_html, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.caption("Tip: Shorter clips transcribe faster. Model loads on first run.")
