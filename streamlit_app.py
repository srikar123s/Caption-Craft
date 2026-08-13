import streamlit as st
import io
import time
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from faster_whisper import WhisperModel
from groq import Groq

from app import predict, device


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Caption Craft",
    page_icon="🎙️",
    layout="wide"
)


# =========================================================
# WHISPER MODEL LOADER
# =========================================================

@st.cache_resource
def load_whisper():

    model_path = "./whisper-tiny.en"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Whisper model not found at: {model_path}"
        )

    if torch.cuda.is_available():
        whisper_device = "cuda"
        compute_type = "float16"
    else:
        whisper_device = "cpu"
        compute_type = "int8"

    print(
        f"Loading Whisper on {whisper_device} "
        f"with {compute_type}..."
    )

    model = WhisperModel(
        model_path,
        device=whisper_device,
        compute_type=compute_type
    )

    return model


# =========================================================
# GROQ CLIENT
# =========================================================

@st.cache_resource
def load_ai_client():

    api_key = st.secrets.get(
        "GROQ_API_KEY"
    )

    if not api_key:
        return None

    return Groq(
        api_key=api_key
    )


# =========================================================
# HYBRID TRANSCRIPT FUSION
# =========================================================

def create_hybrid_transcript(
    custom_transcript,
    whisper_transcript
):
    """
    Combine the BiLSTM and Whisper outputs
    into one final transcript.

    The individual model outputs are never
    displayed to the user.
    """

    custom_transcript = (
        custom_transcript or ""
    ).strip()

    whisper_transcript = (
        whisper_transcript or ""
    ).strip()


    # -----------------------------------------------------
    # If one model produced no result
    # -----------------------------------------------------

    if not custom_transcript and not whisper_transcript:
        return ""

    if not custom_transcript:
        return whisper_transcript

    if not whisper_transcript:
        return custom_transcript


    # -----------------------------------------------------
    # Use Groq to intelligently fuse both outputs
    # -----------------------------------------------------

    client = load_ai_client()

    if client is None:

        # Safe fallback
        return whisper_transcript


    prompt = f"""
You are a speech transcription correction system.

Two speech recognition models transcribed the SAME audio.

Model A - Custom BiLSTM + CTC:
{custom_transcript}

Model B - Whisper:
{whisper_transcript}

Create ONE final, clean transcription.

Rules:

1. Do not combine the two transcripts by simply joining them.
2. Do not repeat sentences or phrases.
3. Preserve the meaning of the original speech.
4. Use Model A when it provides wording that is more accurate
   or useful for the speech.
5. Use Model B when it is clearer or more grammatically reliable.
6. Resolve differences intelligently.
7. Do not invent information that is absent from both transcripts.
8. Keep the natural wording of the speaker.
9. Return ONLY the final transcript.
10. Do not add explanations, labels, or comments.
"""

    try:

        response = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional speech "
                        "transcription fusion assistant."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.1,

            max_tokens=2000
        )

        final_text = (
            response.choices[0]
            .message.content
            .strip()
        )

        if final_text:
            return final_text

    except Exception as e:

        print(
            f"Hybrid fusion failed: {e}"
        )


    # -----------------------------------------------------
    # Fallback
    # -----------------------------------------------------

    return whisper_transcript

# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

# =========================================================
# AI TRANSCRIPT INTELLIGENCE
# =========================================================

def analyze_transcript(transcript):

    client = load_ai_client()

    if client is None:
        raise ValueError(
            "GROQ_API_KEY is not configured."
        )


    prompt = f"""
You are an AI transcript analysis assistant.

Analyze the following final speech transcript.

Return the result using exactly these four sections:

SUMMARY:
Give a concise 3-5 sentence summary.

KEY POINTS:
Give 3-6 important points as bullet points.

ACTION ITEMS:
List important tasks, decisions, or follow-up actions
mentioned in the transcript.

If there are no action items, write:
No specific action items identified.

QUALITY:
Give a score from 0 to 100 and briefly explain the
transcript quality.

Consider:
- clarity
- completeness
- repetition
- possible transcription errors

Transcript:
{transcript}
"""


    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional transcript "
                    "analysis assistant. "
                    "Be concise, factual, and easy to understand."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2,

        max_tokens=1500
    )


    return (
        response
        .choices[0]
        .message
        .content
    )


# =========================================================
# AI ANALYSIS PARSER
# =========================================================


def parse_ai_analysis(raw_text):

    text = (raw_text or "").strip()

    if not text:
        return {
            "summary": "No summary available.",
            "key_points": [],
            "action_items": [],
            "quality": "Quality not available."
        }

    sections = {
        "summary": "",
        "key_points": [],
        "action_items": [],
        "quality": ""
    }

    text = text.replace("\r\n", "\n")

    summary_match = re.search(
        r"SUMMARY:\s*(.*?)(?=\n\s*KEY POINTS:|\n\s*ACTION ITEMS:|\n\s*QUALITY:|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if summary_match:
        sections["summary"] = summary_match.group(1).strip()

    key_points_match = re.search(
        r"KEY POINTS:\s*(.*?)(?=\n\s*ACTION ITEMS:|\n\s*QUALITY:|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if key_points_match:
        key_block = key_points_match.group(1).strip()
        sections["key_points"] = [
            item.strip("- *•\n ")
            for item in key_block.split("\n")
            if item.strip()
        ]

    action_items_match = re.search(
        r"ACTION ITEMS:\s*(.*?)(?=\n\s*QUALITY:|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if action_items_match:
        action_block = action_items_match.group(1).strip()
        if action_block.lower().strip() == "no specific action items identified.":
            sections["action_items"] = []
        else:
            sections["action_items"] = [
                item.strip("- *•\n ")
                for item in action_block.split("\n")
                if item.strip()
            ]

    quality_match = re.search(
        r"QUALITY:\s*(.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if quality_match:
        sections["quality"] = quality_match.group(1).strip()

    return sections


def render_ai_analysis(raw_text):

    analysis = parse_ai_analysis(raw_text)

    quality_score = 0
    quality_match = re.search(r"(\d{1,3})", analysis["quality"])
    if quality_match:
        quality_score = int(quality_match.group(1))

    key_count = len(analysis["key_points"])
    action_count = len(analysis["action_items"])

    st.markdown(
        """
        <style>
        .ai-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px;
            margin: 18px 0 22px 0;
        }
        .ai-metric-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 14px;
            padding: 18px 16px;
        }
        .ai-metric-label {
            font-size: 12px;
            color: #9aa8bd;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
        }
        .ai-metric-value {
            font-size: 28px;
            font-weight: 700;
            line-height: 1.2;
        }
        .ai-section-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 18px 18px 10px 18px;
            margin-bottom: 16px;
        }
        .ai-section-card h4 {
            margin-top: 0;
            margin-bottom: 10px;
            color: #f3f6fb;
        }
        .ai-section-card ul {
            margin-top: 8px;
            margin-left: 18px;
            padding-left: 0;
        }
        .ai-section-card li {
            margin-bottom: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="ai-metrics">
            <div class="ai-metric-card">
                <div class="ai-metric-label">Quality</div>
                <div class="ai-metric-value">{quality_score}/100</div>
            </div>
            <div class="ai-metric-card">
                <div class="ai-metric-label">Key Points</div>
                <div class="ai-metric-value">{key_count}</div>
            </div>
            <div class="ai-metric-card">
                <div class="ai-metric-label">Action Items</div>
                <div class="ai-metric-value">{action_count}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="ai-section-card">
            <h4>📝 Summary</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(analysis["summary"])

    st.markdown(
        """
        <div class="ai-section-card">
            <h4>🔑 Key Points</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if analysis["key_points"]:
        st.markdown("\n".join(f"- {item}" for item in analysis["key_points"]))
    else:
        st.write("No key points identified.")

    st.markdown(
        """
        <div class="ai-section-card">
            <h4>✅ Action Items</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if analysis["action_items"]:
        st.markdown("\n".join(f"- {item}" for item in analysis["action_items"]))
    else:
        st.write("No specific action items identified.")

    st.markdown(
        """
        <div class="ai-section-card">
            <h4>📊 Transcript Quality</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(analysis["quality"])


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 44px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #9aa8bd;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .final-box {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.04);
        margin-bottom: 15px;
    }

    .status-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 12px;
    }

    .status-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(46, 204, 113, 0.12);
        border: 1px solid rgba(46, 204, 113, 0.35);
        color: #dfffe7;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .status-badge.neutral {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.10);
        color: #dfe7f5;
    }

    div[data-testid="stTextArea"] > div {
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.02);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }

    textarea {
        background: transparent !important;
        color: #f4f7fb !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    "<div class='main-title'>🎙️ Caption Craft</div>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='subtitle'>
        Hybrid Indian Accent Speech Recognition
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# AUDIO UPLOAD
# =========================================================

st.markdown(
    "### 🎧 Upload Audio"
)

uploaded = st.file_uploader(
    "Choose an audio file",

    type=[
        "wav",
        "flac",
        "mp3",
        "m4a",
        "ogg"
    ],

    help="Upload an audio file for transcription."
)


# =========================================================
# AUDIO PREVIEW
# =========================================================

if uploaded is not None:

    audio_bytes = uploaded.getvalue()

    st.audio(audio_bytes)


# =========================================================
# TRANSCRIBE BUTTON
# =========================================================

if st.button(
    "🚀 Transcribe Audio",
    type="primary",
    use_container_width=True
):

    if uploaded is None:

        st.warning(
            "Please upload an audio file first."
        )

        st.stop()


    try:

        # =================================================
        # READ AUDIO
        # =================================================

        uploaded.seek(0)

        audio_bytes = uploaded.read()


        audio_data, sample_rate = sf.read(
            io.BytesIO(audio_bytes),
            dtype="float32"
        )


        # =================================================
        # STEREO → MONO
        # =================================================

        if audio_data.ndim == 2:

            audio_data = audio_data.mean(
                axis=1
            )


        audio_data = np.asarray(
            audio_data,
            dtype=np.float32
        )


        # =================================================
        # CONVERT TO TORCH
        # =================================================

        waveform = torch.from_numpy(
            audio_data
        ).float().unsqueeze(0)


        # =================================================
        # RESAMPLE TO 16 kHz
        # =================================================

        target_sample_rate = 16000


        if sample_rate != target_sample_rate:

            resampler = (
                torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=target_sample_rate
                )
            )


            waveform = resampler(
                waveform
            )


            sample_rate = target_sample_rate


        # Convert back to float32 NumPy
        # for Whisper

        audio_data = (
            waveform
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )


        # =================================================
        # STAGE 1 — CUSTOM BiLSTM
        # =================================================

        with st.spinner(
            "🧠 Processing with custom model..."
        ):

            custom_start = time.perf_counter()


            custom_transcript = predict(
                waveform.to(device)
            )


            custom_time = (
                time.perf_counter()
                - custom_start
            )


        # =================================================
        # STAGE 2 — WHISPER
        # =================================================

        with st.spinner(
            "🤖 Processing with speech recognition model..."
        ):

            whisper_model = load_whisper()


            whisper_start = time.perf_counter()


            segments, info = (
                whisper_model.transcribe(

                    audio_data,

                    language="en",

                    beam_size=5,

                    vad_filter=True
                )
            )


            whisper_transcript = " ".join(
                segment.text.strip()
                for segment in segments
            )


            whisper_time = (
                time.perf_counter()
                - whisper_start
            )


        # =================================================
        # STAGE 3 — FINAL TRANSCRIPT SELECTION
        # =================================================

        with st.spinner(
            "🔄 Comparing final transcript outputs..."
        ):

            compare_start = time.perf_counter()

            custom_normalized = normalize_text(
                custom_transcript
            )

            whisper_normalized = normalize_text(
                whisper_transcript
            )

            if custom_normalized == whisper_normalized:

                # Both models produced the same result
                final_transcript = custom_transcript

            else:

                # Models produced different results
                # Use Whisper as the final output
                final_transcript = whisper_transcript

            comparison_time = (
                time.perf_counter()
                - compare_start
            )


        # =================================================
        # CLEAN FINAL TRANSCRIPT
        # =================================================

        final_transcript = (
            final_transcript
            .strip()
        )


        # =================================================
        # CHECK RESULT
        # =================================================

        if not final_transcript:

            st.warning(
                "⚠️ No speech was detected "
                "in the audio."
            )

        else:

            # =============================================
            # SAVE ONLY FINAL RESULT
            # =============================================

            st.session_state[
                "final_transcript"
            ] = final_transcript


            st.session_state[
                "transcription_time"
            ] = (
                custom_time
                + whisper_time
                + comparison_time
            )


            st.session_state[
                "language"
            ] = getattr(
                info,
                "language",
                "en"
            )


            # Clear previous AI analysis

            st.session_state.pop(
                "ai_analysis",
                None
            )


            st.success(
                "✅ Hybrid transcription completed!"
            )


    except Exception as e:

        st.error(
            f"❌ Transcription failed: {e}"
        )

        st.exception(e)


# =========================================================
# FINAL TRANSCRIPT
# =========================================================

if "final_transcript" in st.session_state:

    st.divider()


    st.markdown(
        "## 📝 Final Transcript"
    )

    st.markdown(
        """
        <div class='status-row'>
            <span class='status-badge'>Ready</span>
            <span class='status-badge neutral'>Final Output</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.text_area(
        "Final transcription",

        value=st.session_state[
            "final_transcript"
        ],

        height=250,

        label_visibility="collapsed"
    )


    # =====================================================
    # PROCESSING INFORMATION
    # =====================================================

    col1, col2 = st.columns(2)


    with col1:

        st.caption(
            f"⏱️ Total processing time: "
            f"{st.session_state['transcription_time']:.2f} seconds"
        )


    with col2:

        st.caption(
            f"🌐 Detected language: "
            f"{st.session_state['language']}"
        )


    # =====================================================
    # DOWNLOAD
    # =====================================================

    st.download_button(

        "⬇️ Download Final Transcript",

        data=st.session_state[
            "final_transcript"
        ],

        file_name="caption_craft_transcription.txt",

        mime="text/plain",

        use_container_width=True
    )


    # =====================================================
    # AI TRANSCRIPT INTELLIGENCE
    # =====================================================

    st.divider()


    st.markdown(
        "## 🧠 AI Transcript Intelligence"
    )

    st.caption(
        "✨ AI summary, key points, action items, and transcript quality"
    )

    if st.button(
        "✨ Analyze Transcript",
        type="primary",
        use_container_width=True
    ):

        transcript = st.session_state.get(
            "final_transcript",
            ""
        )


        if not transcript.strip():

            st.warning(
                "Please generate a transcript first."
            )

        else:

            try:

                with st.spinner(
                    "🧠 AI is analyzing your transcript..."
                ):

                    ai_analysis = (
                        analyze_transcript(
                            transcript
                        )
                    )


                st.session_state[
                    "ai_analysis"
                ] = ai_analysis


            except Exception as e:

                st.error(
                    f"❌ AI analysis failed: {e}"
                )


    # =====================================================
    # DISPLAY AI ANALYSIS
    # =====================================================

    if "ai_analysis" in st.session_state:

        render_ai_analysis(
            st.session_state["ai_analysis"]
        )