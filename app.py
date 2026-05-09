"""
eSafeRide Safeguarding Companion
RAG-Based Policy Question-Answering System for Makerere University
"""

import os
import re
import time
import requests
import numpy as np
import pandas as pd
import nltk
import torch
import streamlit as st
from datetime import datetime
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

GITHUB_PDF_URLS = [
    "https://raw.githubusercontent.com/madrinejean123/MACHINE-LEARNING--CHATBOT-GROUP-O/madrine/data/Makerere-Safeguarding-Policy.pdf",
    "https://raw.githubusercontent.com/madrinejean123/MACHINE-LEARNING--CHATBOT-GROUP-O/madrine/data/Policy-and-Regulations-Against-Sexual-Harassment-2018.pdf",
    "https://raw.githubusercontent.com/madrinejean123/MACHINE-LEARNING--CHATBOT-GROUP-O/madrine/data/Makerere-Policy-on-Persons-Living-With-Disabilities.pdf",
    "https://raw.githubusercontent.com/madrinejean123/MACHINE-LEARNING--CHATBOT-GROUP-O/madrine/data/FINAL-REVISED-NATIONAL-POLICY-ON-PWDs-2023.pdf",
    "https://raw.githubusercontent.com/madrinejean123/MACHINE-LEARNING--CHATBOT-GROUP-O/madrine/data/HIV_AIDS_Policy.pdf",
    "https://raw.githubusercontent.com/madrinejean123/MACHINE-LEARNING--CHATBOT-GROUP-O/madrine/data/UTAMU-Disability-Policy.pdf",
]

DATA_FOLDER          = "data"
CHUNK_CSV            = "policy_chunks.csv"
EMBEDDINGS_NPY       = "chunk_embeddings.npy"
CHUNK_MAX_WORDS      = 150
CHUNK_OVERLAP        = 1
TOP_K                = 5
SIMILARITY_THRESHOLD = 0.25

STOP_WORDS = {
    "the", "and", "for", "are", "that", "this", "with", "how",
    "what", "who", "can", "you", "was", "has", "have", "been",
    "hello", "hi", "hey", "please", "thanks", "thank",
}

GREETINGS = {
    "hi", "hello", "hey", "hie", "howdy",
    "good morning", "good afternoon", "good evening", "greetings",
}

GREETING_RESPONSE = (
    "Hello! Welcome to the Safeguarding Companion.\n\n"
    "I'm here to help you understand university policies on safeguarding, "
    "disability rights, sexual harassment, and more — in plain, simple English.\n\n"
    "You can ask things like:\n"
    "- How do I report harassment?\n"
    "- What rights do students with disabilities have?\n"
    "- How do I file a complaint?\n\n"
    "What would you like to know?"
)

SUGGESTED_QUESTIONS = [
    "How do I report harassment?",
    "Rights for students with disabilities",
    "How do I file a complaint?",
    "What is the HIV/AIDS policy?",
    "Support for persons with disabilities",
]

# ---------------------------------------------------------------------------
# DOCUMENT PROCESSING
# ---------------------------------------------------------------------------

def download_pdfs(urls, folder):
    os.makedirs(folder, exist_ok=True)
    paths = []
    for url in urls:
        filename = url.split("/")[-1]
        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            paths.append(filepath)
            continue
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(response.content)
            paths.append(filepath)
        except Exception as e:
            print(f"Could not download {filename}: {e}")
    return paths


def extract_text_from_pdf(filepath):
    import PyPDF2
    text = ""
    try:
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
        if len(text.strip()) > 50:
            return text.strip()
    except Exception as e:
        print(f"PyPDF2 failed for {os.path.basename(filepath)}: {e}")

    try:
        from pdf2image import convert_from_path
        import pytesseract
        images = convert_from_path(filepath, dpi=200)
        ocr_text = ""
        for img in images:
            ocr_text += pytesseract.image_to_string(img, lang="eng") + " "
        if ocr_text.strip():
            return ocr_text.strip()
    except Exception as e:
        print(f"OCR failed for {os.path.basename(filepath)}: {e}")

    return ""


def clean_text(text):
    fixes = {
        "har- assment": "harassment",
        "dis- ability": "disability",
        "re- port":     "report",
        "com- plaint":  "complaint",
    }
    for broken, fixed in fixes.items():
        text = text.replace(broken, fixed)
    text = re.sub(r"\b[a-zA-Z]\)\s*", "", text)
    text = re.sub(r"\b\d+\.\s*", "", text)
    text = re.sub(r"\b[ivxlIVXL]+\.\s*", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,;:()\-\/ ]", "", text)
    text = re.sub(r"\(\d+\)", "", text)
    return text.strip()


def chunk_text(text, max_words=CHUNK_MAX_WORDS, overlap=CHUNK_OVERLAP):
    sentences = sent_tokenize(text)
    chunks, current, current_len = [], [], 0
    for sentence in sentences:
        wc = len(sentence.split())
        if current_len + wc <= max_words:
            current.append(sentence)
            current_len += wc
        else:
            if current:
                chunks.append(" ".join(current))
            tail = current[-overlap:] if overlap > 0 else []
            current = tail + [sentence]
            current_len = sum(len(s.split()) for s in current)
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_dataset(folder):
    records = []
    pdf_files = [f for f in os.listdir(folder) if f.endswith(".pdf")]
    if not pdf_files:
        return pd.DataFrame()
    for filename in pdf_files:
        filepath = os.path.join(folder, filename)
        raw     = extract_text_from_pdf(filepath)
        if not raw:
            continue
        cleaned = clean_text(raw)
        chunks  = chunk_text(cleaned)
        for idx, chunk in enumerate(chunks):
            records.append({
                "chunk_id":        f"{filename}_chunk_{idx}",
                "source_document": filename,
                "chunk_index":     idx,
                "text":            chunk,
                "word_count":      len(chunk.split()),
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# RETRIEVAL
# ---------------------------------------------------------------------------

def keyword_filter(df, query):
    keywords = [
        w.lower() for w in re.findall(r"\b\w+\b", query)
        if len(w) > 2 and w.lower() not in STOP_WORDS
    ]
    if not keywords:
        return df
    mask     = df["text"].apply(lambda x: any(k in str(x).lower() for k in keywords))
    filtered = df[mask]
    return filtered if len(filtered) > 0 else df


def retrieve_top_k(query, model, embeddings, df, k=TOP_K, threshold=SIMILARITY_THRESHOLD):
    filtered = keyword_filter(df, query)
    indices  = filtered.index.tolist()
    f_embeds = embeddings[indices]
    q_vec    = model.encode([query], normalize_embeddings=True)
    scores   = cosine_similarity(q_vec, f_embeds)[0]
    sorted_i = np.argsort(scores)[::-1]
    top_i    = [i for i in sorted_i[:k] if scores[i] >= threshold]
    if not top_i:
        top_i = sorted_i[:3]
    results = filtered.iloc[top_i].copy()
    results["similarity_score"] = scores[top_i]
    return results


# ---------------------------------------------------------------------------
# GENERATION
# ---------------------------------------------------------------------------

def is_greeting(text):
    cleaned = text.strip().lower().rstrip("!.,?")
    words   = cleaned.split()
    return cleaned in GREETINGS or (len(words) <= 3 and words[0] in GREETINGS)


def clean_answer(text):
    text = re.sub(r"^[\*\-•]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\b[a-zA-Z]\)\s*", "", text)
    text = re.sub(r"\b\d+[\.\)]\s*", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def generate_answer(query, retrieved, tokenizer, model, device):
    if retrieved.empty:
        return (
            "I could not find specific information about that in the policy documents. "
            "Please try rephrasing your question, or contact the Gender Mainstreaming "
            "Directorate directly for assistance."
        )

    context_parts = []
    for _, row in retrieved.head(3).iterrows():
        source = (
            row["source_document"]
            .replace(".pdf", "")
            .replace("-", " ")
            .replace("_", " ")
        )
        context_parts.append(f"[{source}]: {row['text']}")
    context = "\n\n".join(context_parts)

    prompt = (
        f"You are a friendly university support assistant. "
        f"A student asked: \"{query}\"\n\n"
        f"Using the policy excerpts below, write a clear, helpful answer "
        f"in 3 to 5 short sentences. "
        f"Do NOT copy sentences from the policy word-for-word. "
        f"Do NOT use bullet letters like a) b) or numbers like 1. 2. "
        f"Write naturally, as if explaining to a friend. "
        f"Use plain English — no legal jargon.\n\n"
        f"Policy excerpts:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Helpful answer:"
    )

    inputs = tokenizer(
        prompt,
        max_length=600,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=220,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        do_sample=False,
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = clean_answer(answer)

    if len(answer.split()) < 5:
        return (
            "The policy documents have relevant information but I could not "
            "generate a clear summary. Please contact the Gender Mainstreaming "
            "Directorate for assistance."
        )
    return answer


def apply_simplified_language(text):
    replacements = {
        "pursuant to":    "according to",
        "stipulates":     "says",
        "thereof":        "of it",
        "aforementioned": "mentioned above",
        "provisions":     "rules",
        "shall":          "must",
        "whilst":         "while",
        "herein":         "in this document",
    }
    for formal, plain in replacements.items():
        text = text.replace(formal, plain)
    return text


def format_response(answer):
    answer = clean_answer(answer)
    if "\n" in answer:
        return answer
    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 8]
    if not sentences:
        return answer
    return " ".join(s + "." for s in sentences)


def nice_source_name(raw):
    return raw.replace(".pdf", "").replace("-", " ").replace("_", " ").strip()


def get_chat_title(messages):
    """Generate a short title from the first user message."""
    for msg in messages:
        if msg["role"] == "user":
            text = msg["content"][:50]
            return text if len(msg["content"]) <= 50 else text + "…"
    return "New conversation"


# ---------------------------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_everything():
    download_pdfs(GITHUB_PDF_URLS, DATA_FOLDER)

    if os.path.exists(CHUNK_CSV) and os.path.exists(EMBEDDINGS_NPY):
        df         = pd.read_csv(CHUNK_CSV)
        embeddings = np.load(EMBEDDINGS_NPY)
    else:
        df = build_dataset(DATA_FOLDER)
        if df.empty:
            raise RuntimeError("No documents could be processed.")
        df.to_csv(CHUNK_CSV, index=False)
        tmp = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = tmp.encode(
            df["text"].astype(str).tolist(),
            show_progress_bar=False,
            batch_size=32,
            normalize_embeddings=True,
        )
        np.save(EMBEDDINGS_NPY, embeddings)

    emb_model   = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer   = AutoTokenizer.from_pretrained("google/flan-t5-base")
    gen_model   = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_model.to(device)
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

    return df, embeddings, emb_model, tokenizer, gen_model, device, transcriber


# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Safeguarding Companion",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# GLOBAL CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600&family=Playfair+Display:wght@700&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif !important;
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #21262d !important;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

.sidebar-logo {
    display: flex; align-items: center; gap: 10px;
    padding: 4px 0 20px;
    border-bottom: 1px solid #21262d;
    margin-bottom: 16px;
}
.sidebar-logo-icon {
    width: 34px; height: 34px; border-radius: 10px;
    background: linear-gradient(135deg,#238636,#1f6feb);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0;
}
.sidebar-logo-text { font-size: 0.9rem; font-weight: 600; color: #e6edf3 !important; }
.sidebar-logo-sub  { font-size: 0.65rem; color: #8b949e !important; }

.sidebar-section-label {
    font-size: 0.65rem; font-weight: 600; letter-spacing: .08em;
    text-transform: uppercase; color: #8b949e !important;
    padding: 12px 0 6px; margin-bottom: 2px;
}

.history-item {
    display: flex; align-items: center; gap: 8px;
    padding: 9px 12px; border-radius: 10px; cursor: pointer;
    border: 1px solid transparent; margin-bottom: 3px;
    transition: background .15s;
    background: transparent;
}
.history-item:hover { background: #1c2128; border-color: #30363d; }
.history-item.active { background: #1c2128; border-color: #1f6feb44; }
.history-item-icon  { font-size: 0.85rem; flex-shrink: 0; }
.history-item-text  { font-size: 0.78rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #c9d1d9 !important; }
.history-item-time  { font-size: 0.62rem; color: #8b949e !important; margin-top: 1px; }

.new-chat-btn {
    display: flex; align-items: center; justify-content: center; gap: 8px;
    width: 100%; padding: 10px; border-radius: 10px;
    background: linear-gradient(135deg,#238636,#1f6feb);
    color: #fff !important; font-size: 0.85rem; font-weight: 600;
    border: none; cursor: pointer; margin-bottom: 20px;
    transition: opacity .2s;
}
.new-chat-btn:hover { opacity: .88; }

/* ── Main area ── */
.main-header {
    display: flex; align-items: center; gap: 14px;
    padding: 0 0 18px;
    border-bottom: 1px solid #21262d;
    margin-bottom: 24px;
}
.main-logo {
    width: 42px; height: 42px; border-radius: 12px;
    background: linear-gradient(135deg,#238636,#1f6feb);
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; flex-shrink: 0;
}
.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem; color: #e6edf3;
}
.main-sub { font-size: 0.7rem; color: #8b949e; margin-top: 2px; }
.online-badge {
    margin-left: auto; font-size: 0.68rem; padding: 4px 12px;
    border-radius: 20px; background: rgba(35,134,54,0.15);
    color: #3fb950; border: 1px solid rgba(63,185,80,0.3);
}

/* ── Welcome ── */
.welcome-card { text-align: center; padding: 40px 16px 28px; }
.welcome-card h2 {
    font-family: 'Playfair Display', serif;
    font-size: 1.7rem; color: #e6edf3; margin-bottom: 12px;
}
.welcome-card p {
    color: #8b949e; font-size: 0.88rem;
    line-height: 1.8; max-width: 500px; margin: 0 auto;
}
.pill-row {
    display: flex; flex-wrap: wrap; gap: 8px;
    justify-content: center; margin-top: 24px;
}
.pill {
    font-size: 0.78rem; padding: 9px 18px; border-radius: 20px;
    background: #161b22; border: 1px solid #30363d; color: #8b949e;
    cursor: pointer; transition: border-color .2s, color .2s;
}
.pill:hover { border-color: #58a6ff; color: #58a6ff; }

/* ── Source tags ── */
.src-tag {
    display: inline-block; margin: 4px 4px 0 0;
    font-size: 0.68rem; padding: 3px 10px; border-radius: 20px;
    background: rgba(31,111,235,.12); color: #58a6ff;
    border: 1px solid rgba(88,166,255,.2);
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: #1c2128 !important;
    border: 1.5px solid #30363d !important;
    border-radius: 28px !important;
    color: #e6edf3 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 14px 20px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,.1) !important;
}
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg,#238636,#1f6feb) !important;
    border-radius: 50% !important; border: none !important;
}

/* ── Message bubbles ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 4px 0 !important;
}

/* ── Divider ── */
hr { border-color: #21262d !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #58a6ff !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SESSION STATE INIT
# ---------------------------------------------------------------------------

if "conversations" not in st.session_state:
    # List of {id, title, messages, timestamp}
    st.session_state.conversations = []

if "active_conv_id" not in st.session_state:
    st.session_state.active_conv_id = None

if "pending_input" not in st.session_state:
    st.session_state.pending_input = None


def new_conversation():
    conv_id = str(int(time.time() * 1000))
    st.session_state.conversations.insert(0, {
        "id":        conv_id,
        "title":     "New conversation",
        "messages":  [],
        "timestamp": datetime.now().strftime("%H:%M"),
    })
    st.session_state.active_conv_id = conv_id


def get_active_conv():
    for conv in st.session_state.conversations:
        if conv["id"] == st.session_state.active_conv_id:
            return conv
    return None


# Make sure there's always at least one conversation
if not st.session_state.conversations:
    new_conversation()
if st.session_state.active_conv_id is None and st.session_state.conversations:
    st.session_state.active_conv_id = st.session_state.conversations[0]["id"]


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <div class="sidebar-logo-icon">🛡️</div>
      <div>
        <div class="sidebar-logo-text">Safeguarding</div>
        <div class="sidebar-logo-sub">Makerere University</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # New chat button
    if st.button("✦  New conversation", use_container_width=True, key="new_chat_btn"):
        new_conversation()
        st.rerun()

    st.markdown('<div class="sidebar-section-label">Recent chats</div>', unsafe_allow_html=True)

    if not st.session_state.conversations:
        st.markdown('<div style="font-size:0.78rem;color:#8b949e;padding:8px 0;">No conversations yet.</div>', unsafe_allow_html=True)
    else:
        for conv in st.session_state.conversations:
            is_active = conv["id"] == st.session_state.active_conv_id
            active_cls = "active" if is_active else ""
            label = conv["title"]
            ts    = conv["timestamp"]

            clicked = st.button(
                f"💬  {label}",
                key=f"conv_{conv['id']}",
                use_container_width=True,
            )
            if clicked:
                st.session_state.active_conv_id = conv["id"]
                st.rerun()

    st.markdown("---")
    st.markdown('<div style="font-size:0.68rem;color:#8b949e;line-height:1.6;">Answers drawn from official Makerere University policy documents.</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# MAIN CONTENT
# ---------------------------------------------------------------------------

# Header
st.markdown("""
<div class="main-header">
  <div class="main-logo">🛡️</div>
  <div>
    <div class="main-title">Safeguarding Companion</div>
    <div class="main-sub">Makerere University · Policy Q&A</div>
  </div>
  <div class="online-badge">● Online</div>
</div>
""", unsafe_allow_html=True)

# Load models
with st.spinner("Loading policy documents and models — first run takes a minute…"):
    df, embeddings, emb_model, gen_tokenizer, gen_model, device, transcriber = load_everything()

# Get active conversation
active_conv = get_active_conv()

# Welcome screen (no messages yet)
if active_conv and not active_conv["messages"]:
    st.markdown("""
    <div class="welcome-card">
      <div style="font-size:3.2rem;margin-bottom:14px">🛡️</div>
      <h2>How can I help you today?</h2>
      <p>Ask me anything about Makerere University's safeguarding policies,
      disability rights, sexual harassment procedures, and student protections.
      All answers come from official policy documents.</p>
      <div class="pill-row">
        <span class="pill">How do I report harassment?</span>
        <span class="pill">Rights for students with disabilities</span>
        <span class="pill">How do I file a complaint?</span>
        <span class="pill">What is the HIV/AIDS policy?</span>
        <span class="pill">Support for persons with disabilities</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# Render chat history for active conversation
if active_conv:
    for msg in active_conv["messages"]:
        with st.chat_message(msg["role"], avatar="🛡️" if msg["role"] == "assistant" else "🧑"):
            st.markdown(msg["content"], unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------------------------

user_input = st.chat_input("Ask anything about university policies…")

if user_input and active_conv:
    # Show user message
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    active_conv["messages"].append({"role": "user", "content": user_input})

    # Update conversation title from first user message
    if active_conv["title"] == "New conversation":
        active_conv["title"] = user_input[:45] + ("…" if len(user_input) > 45 else "")

    # Generate response
    with st.chat_message("assistant", avatar="🛡️"):
        with st.spinner("Searching policy documents…"):

            if is_greeting(user_input):
                answer  = GREETING_RESPONSE
                sources = []
            else:
                retrieved = retrieve_top_k(user_input, emb_model, embeddings, df)
                raw       = generate_answer(user_input, retrieved, gen_tokenizer, gen_model, device)
                raw       = apply_simplified_language(raw)
                answer    = format_response(raw)
                sources   = (
                    list(retrieved["source_document"].unique())
                    if not retrieved.empty else []
                )

        st.markdown(answer)

        if sources:
            tags = "".join(
                f'<span class="src-tag">📄 {nice_source_name(s)}</span>'
                for s in sources
            )
            st.markdown(tags, unsafe_allow_html=True)

        full_content = answer
        if sources:
            full_content += "<br>" + "".join(
                f'<span class="src-tag">📄 {nice_source_name(s)}</span>'
                for s in sources
            )

    active_conv["messages"].append({"role": "assistant", "content": full_content})
    st.rerun()