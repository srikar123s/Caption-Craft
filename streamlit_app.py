import streamlit as st
import io
import time
import os
import json
import re
import hashlib
import tempfile
import importlib
import textwrap
from pathlib import Path


# Keep the app startup lightweight by importing ML packages only when needed.
# This avoids Streamlit scanning heavy libraries on every refresh and reduces
# the time it takes the app to open.


def _ensure_audio_runtime():
    import torch
    import torchaudio
    from faster_whisper import WhisperModel
    return torch, torchaudio, WhisperModel


def _load_module(module_name, package_name=None):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        install_name = package_name or module_name
        raise ModuleNotFoundError(
            f"{module_name} is missing. Install it in the venv with: "
            f"pip install {install_name}"
        ) from exc


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
    import torch
    from faster_whisper import WhisperModel

    model_path = Path(__file__).resolve().parent / "whisper-tiny.en"

    if not model_path.exists():
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

    _, _, WhisperModel = _ensure_audio_runtime()

    model = WhisperModel(
        str(model_path),
        device=whisper_device,
        compute_type=compute_type
    )

    return model


# =========================================================
# GROQ CLIENT
# =========================================================

@st.cache_resource
def load_ai_client():
    from groq import Groq

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
# AI TRANSCRIPT INTELLIGENCE — STRUCTURED EXTRACTION
# =========================================================

def analyze_transcript(transcript):
    """Analyze the final transcript and return structured JSON."""

    client = load_ai_client()

    if client is None:
        raise ValueError(
            "GROQ_API_KEY is not configured. "
            "Add it to Streamlit secrets."
        )

    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty.")

    prompt = f"""
You are a precise information extraction assistant.

Analyze the following FINAL speech transcript and return ONLY valid JSON.

TRANSCRIPT:
{transcript}

Extract:

1. summary
   - Concise 3-5 sentence summary.

2. key_points
   - Important points discussed.

3. action_items
   - Tasks or follow-up actions explicitly mentioned.
   - Each item must contain task, owner, and deadline.
   - Use "" if owner/deadline is not mentioned.

4. people
   - People explicitly mentioned.

5. organizations
   - Companies, institutions, teams, or organizations explicitly mentioned.

6. dates
   - Dates, deadlines, days, months, years, or explicit time references.

7. numbers
   - Important numerical information with its context.
   - Preserve values exactly as stated.

8. decisions
   - Explicit decisions, agreements, conclusions, or choices.

9. quality
   - score: integer from 0 to 100.
   - explanation: short explanation based on clarity,
     completeness, repetition, and possible transcription errors.

STRICT RULES:
- Use ONLY information present in the transcript.
- Never invent information.
- If a category is absent, return an empty list.
- Preserve numerical values exactly as stated.
- Return ONLY valid JSON.
- Do not use markdown code fences.

Return exactly:

{{
    "summary": "",
    "key_points": [],
    "action_items": [
        {{
            "task": "",
            "owner": "",
            "deadline": ""
        }}
    ],
    "people": [],
    "organizations": [],
    "dates": [],
    "numbers": [],
    "decisions": [],
    "quality": {{
        "score": 0,
        "explanation": ""
    }}
}}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise transcript information "
                    "extraction system. Return valid JSON only."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=2000
    )

    raw = response.choices[0].message.content.strip()

    # Remove accidental markdown fences.
    if raw.startswith("```"):
        raw = re.sub(
            r"^```(?:json)?\s*",
            "",
            raw,
            flags=re.IGNORECASE
        )
        raw = re.sub(r"\s*```$", "", raw).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise ValueError(
                "Groq returned invalid JSON."
            )
        data = json.loads(match.group(0))

    if not isinstance(data, dict):
        raise ValueError(
            "Groq returned an invalid JSON object."
        )

    # Safe defaults.
    defaults = {
        "summary": "",
        "key_points": [],
        "action_items": [],
        "people": [],
        "organizations": [],
        "dates": [],
        "numbers": [],
        "decisions": [],
        "quality": {
            "score": 0,
            "explanation": ""
        }
    }

    for key, value in defaults.items():
        data.setdefault(key, value)

    # Normalize list fields.
    list_fields = [
        "key_points",
        "people",
        "organizations",
        "dates",
        "numbers",
        "decisions"
    ]

    for field in list_fields:
        if not isinstance(data[field], list):
            data[field] = [str(data[field])]

    # Normalize action items.
    cleaned_actions = []

    if isinstance(data["action_items"], list):
        for item in data["action_items"]:
            if isinstance(item, dict):
                cleaned_actions.append({
                    "task": str(item.get("task", "")).strip(),
                    "owner": str(item.get("owner", "")).strip(),
                    "deadline": str(item.get("deadline", "")).strip()
                })
            else:
                cleaned_actions.append({
                    "task": str(item).strip(),
                    "owner": "",
                    "deadline": ""
                })

    data["action_items"] = cleaned_actions

    # Normalize quality.
    if not isinstance(data["quality"], dict):
        data["quality"] = {
            "score": 0,
            "explanation": str(data["quality"])
        }

    try:
        score = int(data["quality"].get("score", 0))
    except (TypeError, ValueError):
        score = 0

    data["quality"]["score"] = max(
        0,
        min(100, score)
    )

    data["quality"]["explanation"] = str(
        data["quality"].get("explanation", "")
    ).strip()

    data["summary"] = str(
        data["summary"]
    ).strip()

    return data


def render_ai_analysis(analysis):
    """Render structured AI analysis in a clean Streamlit UI."""

    if not isinstance(analysis, dict):
        st.error("Invalid AI analysis format.")
        return

    summary = str(
        analysis.get("summary", "")
    ).strip()

    key_points = analysis.get("key_points", [])
    action_items = analysis.get("action_items", [])
    people = analysis.get("people", [])
    organizations = analysis.get("organizations", [])
    dates = analysis.get("dates", [])
    numbers = analysis.get("numbers", [])
    decisions = analysis.get("decisions", [])
    quality = analysis.get("quality", {})

    try:
        quality_score = int(
            quality.get("score", 0)
        )
    except (TypeError, ValueError):
        quality_score = 0

    quality_score = max(
        0,
        min(100, quality_score)
    )

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "Transcript Quality",
            f"{quality_score}/100"
        )

    with m2:
        st.metric(
            "Key Points",
            len(key_points)
        )

    with m3:
        st.metric(
            "Action Items",
            len(action_items)
        )

    with m4:
        st.metric(
            "Entities",
            len(people) + len(organizations)
        )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📝 Summary</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        textwrap.dedent(f"""
        <div class="info-card">
            {summary or '<span class="info-empty">No summary available.</span>'}
        </div>
        """),
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # KEY POINTS
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">🔑 Key Points</div>',
        unsafe_allow_html=True
    )

    if key_points:

        key_text = "".join(
            f"<li>{str(item)}</li>"
            for item in key_points
        )

        st.markdown(
            textwrap.dedent(f"""
            <div class="info-card">
                <ul style="margin-bottom:0;">
                    {key_text}
                </ul>
            </div>
            """),
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="info-card">'
            '<span class="info-empty">No key points identified.</span>'
            '</div>',
            unsafe_allow_html=True
        )

    # -----------------------------------------------------
    # PEOPLE + ORGANIZATIONS
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        people_html = ""

        if people:
            people_html = "".join(
                f"<li>{str(person)}</li>"
                for person in people
            )
        else:
            people_html = (
                '<span class="info-empty">'
                'No people identified.'
                '</span>'
            )

        st.markdown(
            textwrap.dedent(f"""
            <div class="info-card">
                <h4>👤 People</h4>
                {"<ul>" + people_html + "</ul>" if people else people_html}
            </div>
            """),
            unsafe_allow_html=True
        )

    with col2:

        org_html = ""

        if organizations:
            org_html = "".join(
                f"<li>{str(org)}</li>"
                for org in organizations
            )
        else:
            org_html = (
                '<span class="info-empty">'
                'No organizations identified.'
                '</span>'
            )

        st.markdown(
            textwrap.dedent(f"""
            <div class="info-card">
                <h4>🏢 Organizations</h4>
                {"<ul>" + org_html + "</ul>" if organizations else org_html}
            </div>
            """),
            unsafe_allow_html=True
        )

    # -----------------------------------------------------
    # DATES + NUMBERS
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        date_html = ""

        if dates:
            date_html = "".join(
                f"<li>{str(date)}</li>"
                for date in dates
            )
        else:
            date_html = (
                '<span class="info-empty">'
                'No dates or deadlines identified.'
                '</span>'
            )

        st.markdown(
            textwrap.dedent(f"""
            <div class="info-card">
                <h4>📅 Dates / Time References</h4>
                {"<ul>" + date_html + "</ul>" if dates else date_html}
            </div>
            """),
            unsafe_allow_html=True
        )

    with col2:

        number_html = ""

        if numbers:
            number_html = "".join(
                f"<li>{str(number)}</li>"
                for number in numbers
            )
        else:
            number_html = (
                '<span class="info-empty">'
                'No important numbers identified.'
                '</span>'
            )

        st.markdown(
            textwrap.dedent(f"""
            <div class="info-card">
                <h4>🔢 Important Numbers</h4>
                {"<ul>" + number_html + "</ul>" if numbers else number_html}
            </div>
            """),
            unsafe_allow_html=True
        )

    # -----------------------------------------------------
    # DECISIONS
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">✅ Decisions</div>',
        unsafe_allow_html=True
    )

    if decisions:

        decision_html = "".join(
            f"<li>{str(decision)}</li>"
            for decision in decisions
        )

        st.markdown(
            textwrap.dedent(f"""
            <div class="info-card">
                <ul style="margin-bottom:0;">
                    {decision_html}
                </ul>
            </div>
            """),
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="info-card">'
            '<span class="info-empty">No explicit decisions identified.</span>'
            '</div>',
            unsafe_allow_html=True
        )

    # -----------------------------------------------------
    # ACTION ITEMS
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📌 Action Items</div>',
        unsafe_allow_html=True
    )

    if action_items:

        for item in action_items:

            if not isinstance(item, dict):
                continue

            task = str(
                item.get("task", "")
            ).strip()

            owner = str(
                item.get("owner", "")
            ).strip()

            deadline = str(
                item.get("deadline", "")
            ).strip()

            if not task:
                continue

            meta = []

            if owner:
                meta.append(
                    f"Owner: {owner}"
                )

            if deadline:
                meta.append(
                    f"Deadline: {deadline}"
                )

            meta_html = (
                " • ".join(meta)
                if meta
                else "Owner/deadline not specified"
            )

            st.markdown(
                textwrap.dedent(f"""
                <div class="action-card">
                    <div class="action-task">📌 {task}</div>
                    <div class="action-meta">{meta_html}</div>
                </div>
                """),
                unsafe_allow_html=True
            )

    else:

        st.markdown(
            '<div class="info-card">'
            '<span class="info-empty">No specific action items identified.</span>'
            '</div>',
            unsafe_allow_html=True
        )

    # -----------------------------------------------------
    # QUALITY
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📊 Transcript Quality</div>',
        unsafe_allow_html=True
    )

    st.progress(
        quality_score / 100
    )

    st.write(
        f"**Quality Score: {quality_score}/100**"
    )

    explanation = str(
        quality.get("explanation", "")
    ).strip()

    if explanation:
        st.caption(explanation)



# =========================================================
# TRANSCRIPT RAG
# =========================================================

@st.cache_resource
def load_rag_embeddings():
    """
    Load a small local sentence-transformer embedding model.

    The model is downloaded once and then cached by Streamlit.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )


@st.cache_resource
def build_transcript_vectorstore(transcript):
    """
    Split the final transcript into overlapping chunks and
    index them in an in-memory Chroma vector store.

    The transcript itself is used as the cache key, so a new
    transcript gets a new vector store automatically.
    """

    if not transcript or not transcript.strip():
        raise ValueError(
            "Cannot build RAG index from an empty transcript."
        )

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False
    )

    documents = splitter.create_documents(
        [transcript]
    )

    for index, document in enumerate(documents):
        document.metadata["chunk_id"] = index + 1
        document.metadata["source"] = "final_transcript"

    embeddings = load_rag_embeddings()

    collection_suffix = hashlib.sha256(
        transcript.encode("utf-8")
    ).hexdigest()[:12]

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=f"caption_craft_{collection_suffix}"
    )

    return vectorstore


def answer_from_transcript(question, transcript):
    """
    Retrieve the most relevant transcript chunks and ask Groq
    to answer using only those retrieved chunks.
    """

    client = load_ai_client()

    if client is None:
        raise ValueError(
            "GROQ_API_KEY is not configured. "
            "Add it to Streamlit secrets."
        )

    if not transcript or not transcript.strip():
        raise ValueError(
            "Generate a transcript before asking questions."
        )

    if not question or not question.strip():
        raise ValueError(
            "Please enter a question."
        )

    vectorstore = build_transcript_vectorstore(
        transcript
    )

    results = vectorstore.similarity_search_with_score(
        question.strip(),
        k=4
    )

    if not results:
        return {
            "answer": (
                "I couldn't find enough relevant information "
                "in the transcript to answer that question."
            ),
            "sources": []
        }

    context_parts = []
    sources = []

    for document, score in results:

        chunk_id = document.metadata.get(
            "chunk_id",
            "?"
        )

        context_parts.append(
            f"[Transcript Chunk {chunk_id}]\n"
            f"{document.page_content}"
        )

        sources.append({
            "chunk_id": chunk_id,
            "text": document.page_content,
            "score": float(score)
        })

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are a retrieval-grounded transcript assistant.

Answer the user's question using ONLY the transcript
context provided below.

If the answer is not supported by the context, say:
"I couldn't find that information in the transcript."

Do not invent facts.
Do not use outside knowledge.
Keep the answer concise but complete.
When useful, mention the relevant transcript chunk number.

TRANSCRIPT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions only from retrieved "
                    "transcript context. Never hallucinate."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=700
    )

    answer = (
        response.choices[0]
        .message.content
        .strip()
    )

    return {
        "answer": answer,
        "sources": sources
    }


def render_rag_sources(sources):
    """
    Display the transcript chunks retrieved by the vector search.
    """

    if not sources:
        return

    st.markdown(
        "### 📚 Retrieved Transcript Context"
    )

    for source in sources:

        chunk_id = source["chunk_id"]
        text = source["text"]

        with st.expander(
            f"Transcript Chunk {chunk_id}"
        ):
            st.write(text)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    textwrap.dedent(
        """
        <style>
        :root {
            --bg: #02192a;
            --bg-2: #061f32;
            --panel: rgba(8, 26, 37, 0.94);
            --panel-soft: rgba(12, 26, 36, 0.88);
            --panel-elevated: rgba(17, 31, 42, 0.98);
            --line: rgba(150, 170, 198, 0.16);
            --line-strong: rgba(150, 170, 198, 0.3);
            --text: #edf4ff;
            --muted: #a5b8cf;
            --muted-2: #7e93ad;
            --accent-red: #f86f6b;
            --accent-red-strong: #ff7d78;
            --accent-blue: #4ea7ff;
            --success: #31d66b;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, var(--bg) 0%, #061c2d 100%);
            color: var(--text);
        }

        .stApp {
            background: linear-gradient(180deg, var(--bg) 0%, #061c2d 100%);
        }

        .block-container {
            max-width: 1360px;
            padding-top: 0.5rem;
            padding-bottom: 2.5rem;
        }

        .browser-shell {
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 16px;
            overflow: hidden;
            background: rgba(5, 16, 27, 0.9);
            box-shadow: 0 18px 32px rgba(1, 5, 10, 0.28);
             margin: 3rem 0 1rem;
        }

        .browser-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            background: linear-gradient(180deg, #1b1226 0%, #120d1f 100%);
            border-bottom: 1px solid rgba(148, 163, 184, 0.12);
            padding: 10px 16px;
            min-height: 48px;
        }

        .browser-left,
        .browser-right {
            display: flex;
            align-items: center;
            gap: 10px;
            min-width: 120px;
        }

        .browser-right {
            justify-content: flex-end;
        }

        .browser-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            box-shadow: inset 0 1px 1px rgba(255,255,255,0.18);
        }

        .browser-dot.red { background: #ff6a5f; }
        .browser-dot.yellow { background: #ffbf2f; }
        .browser-dot.green { background: #36c85c; }

        .browser-address {
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .browser-pill {
            min-width: 260px;
            max-width: 520px;
            width: 100%;
            padding: 7px 14px;
            border-radius: 12px;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(148, 163, 184, 0.12);
            color: rgba(214, 224, 240, 0.85);
            font-size: 0.78rem;
            text-align: center;
            letter-spacing: 0.02em;
        }

        .browser-action {
            width: 22px;
            height: 22px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: #dfeaf9;
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.07);
            font-size: 0.72rem;
        }

        .browser-tabstrip {
            display: flex;
            align-items: flex-end;
            gap: 10px;
            padding: 10px 12px 0;
            background: rgba(7, 18, 31, 0.9);
            border-bottom: 1px solid rgba(148, 163, 184, 0.12);
            overflow-x: auto;
        }

        .browser-tab {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 14px 10px 12px;
            background: rgba(255,255,255,0.02);
            border: 1px solid transparent;
            border-bottom: none;
            border-radius: 10px 10px 0 0;
            color: rgba(191, 205, 224, 0.8);
            font-size: 0.76rem;
            opacity: 0.82;
        }

        .browser-tab.active {
            background: rgba(9, 24, 35, 1);
            border-color: rgba(148, 163, 184, 0.14);
            opacity: 1;
            color: #edf4ff;
        }

        .browser-tab .favicon {
            width: 12px;
            height: 12px;
            border-radius: 4px;
            background: linear-gradient(135deg, #dff3ff, #3b82f6);
            display: inline-block;
        }

        .browser-tab.active .favicon {
            background: linear-gradient(135deg, #ffffff, #a5d8ff);
        }

        .hero-shell {
            background: linear-gradient(180deg, rgba(4, 17, 27, 0.9), rgba(2, 14, 22, 0.98));
            border-top: 1px solid rgba(148, 163, 184, 0.1);
            min-height: 220px;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1.5rem 1.2rem 1.4rem;
        }

        .hero-inner {
            width: min(100%, 820px);
            min-height: 172px;
            background: rgba(7, 16, 25, 0.8);
            border: 1px solid rgba(148, 163, 184, 0.10);
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 0.35rem;
        }

        .hero-title {
            font-size: 2.7rem;
            font-weight: 800;
            letter-spacing: -0.06em;
            margin: 0;
            color: var(--text);
            line-height: 1.08;
        }

        .hero-subtitle {
            color: #d8e4f5;
            font-size: 1rem;
            letter-spacing: 0.02em;
            margin: 0;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        }

        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text);
            margin: 1.3rem 0 0.6rem 0;
        }

        .section-subtitle {
            color: var(--muted-2);
            font-size: 0.82rem;
            margin-bottom: 0.9rem;
        }

        .upload-card {
            background: rgba(9, 19, 28, 0.72);
            border: 1px solid rgba(150, 170, 198, 0.18);
            border-radius: 18px;
            padding: 0.1rem 0.2rem;
            margin-bottom: 0.2rem;
        }

        [data-testid="stFileUploader"] > section {
            background: rgba(9, 19, 28, 0.72);
            border: 1px solid rgba(150, 170, 198, 0.18);
            border-radius: 16px;
            padding: 0.3rem 0.4rem;
        }

        [data-testid="stFileUploader"] label {
            color: var(--text) !important;
            font-weight: 600;
        }

        .upload-box {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            padding: 0.9rem 1.1rem;
            border-radius: 14px;
            background: rgba(12, 20, 28, 0.9);
            border: 1px solid rgba(150, 170, 198, 0.18);
            color: var(--muted);
            min-height: 52px;
        }

        .upload-box .left {
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 600;
        }

        .upload-box .meta {
            font-size: 0.8rem;
            color: var(--muted-2);
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 18px !important;
            border: none !important;
            font-weight: 700 !important;
            letter-spacing: 0.01em;
            transition: all 0.2s ease;
            box-shadow: 0 10px 24px rgba(248, 111, 107, 0.24);
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform: translateY(-1px);
            filter: brightness(1.05);
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, var(--accent-red) 0%, var(--accent-red-strong) 100%) !important;
            color: white !important;
            border-radius: 16px !important;
            min-height: 52px;
        }

        .stDownloadButton > button {
            background: linear-gradient(90deg, #1f8ac8, #4ea7ff) !important;
            color: white !important;
        }

        .status-row {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin: 0.8rem 0 0.9rem 0;
        }

        .status-badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(49, 214, 107, 0.12);
            border: 1px solid rgba(49, 214, 107, 0.28);
            color: #dffef0;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .status-badge.neutral {
            background: rgba(78, 167, 255, 0.12);
            border: 1px solid rgba(78, 167, 255, 0.25);
            color: #dfeeff;
        }

        div[data-testid="stTextArea"] > div {
            border-radius: 14px;
            border: 1px solid rgba(150, 170, 198, 0.18);
            background: rgba(7, 17, 27, 0.86);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }

        textarea {
            background: transparent !important;
            color: #edf5ff !important;
            font-size: 15px !important;
            line-height: 1.6 !important;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace !important;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(120px, 1fr));
            gap: 12px;
            margin: 16px 0 24px 0;
        }

        .metric-card {
            padding: 18px 16px;
            border-radius: 16px;
            border: 1px solid rgba(150, 170, 198, 0.12);
            background: rgba(9, 20, 29, 0.74);
        }

        .metric-label {
            color: var(--muted-2);
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .metric-value {
            margin-top: 8px;
            color: var(--text);
            font-size: 1.7rem;
            font-weight: 800;
            line-height: 1.2;
        }

        .info-card {
            min-height: 145px;
            padding: 18px;
            border-radius: 16px;
            border: 1px solid rgba(150, 170, 198, 0.12);
            background: rgba(7, 18, 28, 0.72);
            margin-bottom: 14px;
        }

        .info-card h4 {
            margin: 0 0 10px 0;
            color: var(--text);
            font-size: 1rem;
        }

        .info-empty {
            color: var(--muted-2);
        }

        .action-card {
            padding: 15px 17px;
            border-radius: 14px;
            border: 1px solid rgba(150, 170, 198, 0.12);
            background: rgba(8, 20, 28, 0.7);
            margin-bottom: 10px;
        }

        .action-task {
            color: var(--text);
            font-weight: 700;
            font-size: 15px;
            margin-bottom: 5px;
        }

        .action-meta {
            color: var(--muted);
            font-size: 13px;
        }

        .stProgress > div > div {
            background: linear-gradient(90deg, var(--accent-blue), #90d0ff);
        }

        @media (max-width: 800px) {
            .hero-title { font-size: 2.1rem; }
            .browser-pill { min-width: 180px; }
            .metric-grid { grid-template-columns: repeat(2, 1fr); }
        }
        </style>
        """
    ),
    unsafe_allow_html=True,
)


# =========================================================
# HEADER
# =========================================================

st.html("""
<div class="browser-shell">
    <div class="browser-bar">
        <div class="browser-left">
            <span class="browser-dot red"></span>
            <span class="browser-dot yellow"></span>
            <span class="browser-dot green"></span>
        </div>

        <div class="browser-address">
            <div class="browser-pill">localhost:8501</div>
        </div>

        <div class="browser-right">
            <span class="browser-action">＋</span>
            <span class="browser-action">↻</span>
        </div>
    </div>

    <div class="browser-tabstrip">
        <div class="browser-tab active">
            <span class="favicon"></span>
            <span>Caption Craft</span>
        </div>

        <div class="browser-tab">
            <span class="favicon"></span>
            <span>Caption Craft</span>
        </div>

        <div class="browser-tab">
            <span class="favicon"></span>
            <span>Caption Craft</span>
        </div>
    </div>

    <div class="hero-shell">
        <div class="hero-inner">
            <h1 class="hero-title">Caption Craft</h1>
            <div class="hero-subtitle">Hybrid Indian Accent Speech Recognition</div>
        </div>
    </div>
</div>
""")

# AUDIO UPLOAD
# =========================================================

st.markdown(
    '<div class="section-title">🎧 Upload Audio</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="upload-card">',
    unsafe_allow_html=True
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
    help="Upload an audio file for transcription.",
    label_visibility="collapsed"
)

st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# AUDIO PREVIEW
# =========================================================

if uploaded is not None:

    st.audio(
        uploaded.getvalue()
    )

    st.caption(
        f"Selected file: **{uploaded.name}**"
    )


# =========================================================
# TRANSCRIBE BUTTON
# =========================================================

transcribe_col1, transcribe_col2, transcribe_col3 = st.columns(
    [1, 8, 1]
)

with transcribe_col2:
    transcribe_clicked = st.button(
        "🎙️ Transcribe Audio",
        type="primary",
        use_container_width=True
    )


if transcribe_clicked:

    if uploaded is None:

        st.warning(
            "Please upload an audio file first."
        )

        st.stop()


    try:
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio

        try:
            from app import predict, device
        except ModuleNotFoundError as exc:
            if exc.name == "fastapi":
                st.error(
                    "FastAPI is missing in this deployment. "
                    "Add fastapi, uvicorn, and python-multipart to requirements.txt "
                    "and redeploy."
                )
                st.stop()
            raise

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
        # STAGE 3 — HYBRID TRANSCRIPT FUSION
        # =================================================

        with st.spinner(
            "🔄 Creating final hybrid transcript..."
        ):

            compare_start = time.perf_counter()

            # Intelligently reconcile the Custom BiLSTM and
            # Whisper outputs instead of automatically discarding
            # the custom model whenever the outputs differ.
            final_transcript = create_hybrid_transcript(
                custom_transcript,
                whisper_transcript
            )

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
        '<div class="section-title">📝 Final Transcript</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """<div class="status-row">
    <span class="status-badge">✓ Ready</span>
    <span class="status-badge neutral">Hybrid Output</span>
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown(
        """<div style="margin-bottom: 10px; color: #c9d9ef; font-size: 0.8rem; letter-spacing: 0.06em; text-transform: uppercase; opacity: 0.8;">
    Final Transcript
</div>
""",
        unsafe_allow_html=True
    )

    st.text_area(
        "Final transcription",
        value=st.session_state["final_transcript"],
        height=220,
        label_visibility="collapsed"
    )

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.caption(
            f"⏱️ Processing: "
            f"{st.session_state['transcription_time']:.2f} sec"
        )

    with info_col2:
        st.caption(
            f"🌐 Language: "
            f"{st.session_state['language']}"
        )

    with info_col3:
        word_count = len(
            st.session_state["final_transcript"].split()
        )
        st.caption(
            f"📝 Words: {word_count}"
        )

    download_col1, download_col2, download_col3 = st.columns(
        [1, 2, 1]
    )

    with download_col2:
        st.download_button(
            "⬇️ Download Final Transcript",
            data=st.session_state["final_transcript"],
            file_name="caption_craft_transcription.txt",
            mime="text/plain",
            use_container_width=True
        )


    # =====================================================
    # AI TRANSCRIPT INTELLIGENCE
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-title">🧠 AI Transcript Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Convert the transcript into structured, actionable information.'
        '</div>',
        unsafe_allow_html=True
    )

    analyze_col1, analyze_col2, analyze_col3 = st.columns(
        [1, 2, 1]
    )

    with analyze_col2:
        analyze_clicked = st.button(
            "✨ Analyze Transcript",
            type="primary",
            use_container_width=True
        )

    if analyze_clicked:

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
                    "🧠 Extracting structured information..."
                ):

                    ai_analysis = analyze_transcript(
                        transcript
                    )

                st.session_state[
                    "ai_analysis"
                ] = ai_analysis

                st.success(
                    "✅ Transcript analysis completed."
                )

            except Exception as e:

                st.error(
                    f"❌ AI analysis failed: {e}"
                )


    # =====================================================
    # DISPLAY AI ANALYSIS
    # =====================================================

    if "ai_analysis" in st.session_state:

        analysis = st.session_state[
            "ai_analysis"
        ]


        render_ai_analysis(
            analysis
        )


    # =====================================================
    # ASK YOUR TRANSCRIPT — RAG
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-title">🔎 Ask Your Transcript</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Ask questions about the transcript. Caption Craft retrieves '
        'the most relevant transcript sections before generating an answer.'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "💡 Example: **What did Rahul agree to do?**  "
        "or **What financial numbers were mentioned?**"
    )

    question = st.text_input(
        "Ask a question",
        placeholder="What was the main decision discussed?",
        key="rag_question"
    )

    ask_col1, ask_col2, ask_col3 = st.columns(
        [1, 2, 1]
    )

    with ask_col2:

        ask_clicked = st.button(
            "🔎 Ask Transcript",
            type="primary",
            use_container_width=True
        )

    if ask_clicked:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            try:

                with st.spinner(
                    "🔍 Retrieving relevant transcript context..."
                ):

                    rag_result = answer_from_transcript(
                        question,
                        st.session_state["final_transcript"]
                    )

                st.session_state[
                    "rag_result"
                ] = rag_result

                st.success(
                    "✅ Answer generated from the transcript."
                )

            except Exception as e:

                st.error(
                    f"❌ RAG question failed: {e}"
                )


    # =====================================================
    # DISPLAY RAG ANSWER
    # =====================================================

    if "rag_result" in st.session_state:

        rag_result = st.session_state[
            "rag_result"
        ]

        st.markdown(
            "### 💬 Answer"
        )

        st.html(f"""
<div class="info-card">
    {rag_result.get("answer", "No answer available.")}
</div>
""")

        render_rag_sources(
            rag_result.get("sources", [])
        )