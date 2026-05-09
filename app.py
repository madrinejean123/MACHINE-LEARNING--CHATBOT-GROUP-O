"""
eSafeRide Safeguarding Companion
RAG-Based Policy Question-Answering System for Makerere University
"""

import os
import re
import tempfile
import requests
import numpy as np
import pandas as pd
import nltk
import torch
import gradio as gr
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
    "I am here to help you understand university policies on safeguarding, "
    "disability rights, sexual harassment, and more - in plain, simple English.\n\n"
    "You can ask things like:\n"
    "  - How do I report harassment?\n"
    "  - What rights do students with disabilities have?\n"
    "  - How do I file a complaint?\n\n"
    "What would you like to know?"
)

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
            print(f"Downloaded: {filename}")
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
        images = convert_from_path(filepath)
        ocr_text = ""
        for img in images:
            ocr_text += pytesseract.image_to_string(img) + " "
        if ocr_text.strip():
            return ocr_text.strip()
    except Exception as e:
        print(f"OCR fallback failed for {os.path.basename(filepath)}: {e}")

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
        print("No PDF files found.")
        return pd.DataFrame()
    for filename in pdf_files:
        filepath = os.path.join(folder, filename)
        print(f"Processing: {filename}")
        raw     = extract_text_from_pdf(filepath)
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
        "You are a helpful assistant explaining university policies in simple, "
        "friendly language for students including those with disabilities.\n\n"
        "Using ONLY the policy context below, answer the question with 3 to 5 "
        "short points. Use plain English. Avoid legal jargon. Be clear and kind.\n\n"
        f"Policy Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    inputs = tokenizer(
        prompt,
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=200,
        do_sample=False,
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if len(answer.split()) < 5:
        return (
            "The policy documents contain relevant information but I could not "
            "generate a clear summary. Please contact the Gender Mainstreaming "
            "Directorate for assistance."
        )
    return answer


def format_response(answer):
    if "\n" in answer or answer.startswith(("-", "*")):
        return answer
    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    return "\n".join(f"- {s}." for s in sentences)


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


def transcribe_audio(audio_path):
    try:
        result = transcriber(audio_path)
        return result["text"].strip()
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""


# ---------------------------------------------------------------------------
# STARTUP
# ---------------------------------------------------------------------------

print("Downloading policy documents...")
download_pdfs(GITHUB_PDF_URLS, DATA_FOLDER)

if os.path.exists(CHUNK_CSV) and os.path.exists(EMBEDDINGS_NPY):
    print("Loading cached chunks and embeddings...")
    df         = pd.read_csv(CHUNK_CSV)
    embeddings = np.load(EMBEDDINGS_NPY)
else:
    print("Building document dataset...")
    df = build_dataset(DATA_FOLDER)
    if df.empty:
        raise RuntimeError("No documents could be processed. Check PDF URLs.")
    df.to_csv(CHUNK_CSV, index=False)
    print("Generating embeddings...")
    _tmp_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = _tmp_model.encode(
        df["text"].astype(str).tolist(),
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True,
    )
    np.save(EMBEDDINGS_NPY, embeddings)
    print(f"Embeddings saved. Shape: {embeddings.shape}")

print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading generation model...")
gen_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
gen_model     = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen_model.to(device)

print("Loading Whisper for voice transcription...")
transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
)

print("All models ready.")


# ---------------------------------------------------------------------------
# QUERY HANDLER
# ---------------------------------------------------------------------------

def handle_query(text_query, audio_input, simplified_mode):
    query = ""

    if audio_input is not None:
        query = transcribe_audio(audio_input)

    if not query:
        query = text_query or ""

    if not query.strip():
        return "Please type or speak your question.", ""

    if is_greeting(query):
        return GREETING_RESPONSE, ""

    retrieved = retrieve_top_k(query, embedding_model, embeddings, df)
    answer    = generate_answer(query, retrieved, gen_tokenizer, gen_model, device)

    if simplified_mode:
        answer = apply_simplified_language(answer)

    formatted = format_response(answer)

    sources = ""
    if not retrieved.empty:
        source_names = [
            s.replace(".pdf", "").replace("-", " ").replace("_", " ")
            for s in retrieved["source_document"].unique()
        ]
        sources = "Policy Sources:\n" + "\n".join(f"  - {s}" for s in source_names)

    return formatted, sources


# ---------------------------------------------------------------------------
# GRADIO INTERFACE
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# GRADIO INTERFACE  — replace everything from CSS = """ to demo.launch()
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600&family=Playfair+Display:wght@600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    font-family: 'Sora', sans-serif !important;
    background: #0d1117 !important;
    color: #e6edf3 !important;
    min-height: 100vh;
}

/* ── hide default gradio chrome ── */
footer, .gr-form > .label-wrap { display: none !important; }
.gradio-container { padding: 0 !important; max-width: 100% !important; }
.contain { padding: 0 !important; }

/* ── top bar ── */
.top-bar {
    position: sticky; top: 0; z-index: 100;
    display: flex; align-items: center; gap: 14px;
    padding: 14px 28px;
    background: rgba(13,17,23,0.85);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid #21262d;
}
.top-bar-logo {
    width: 38px; height: 38px; border-radius: 10px;
    background: linear-gradient(135deg, #238636, #1f6feb);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0;
}
.top-bar-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem; color: #e6edf3; letter-spacing: 0.01em;
}
.top-bar-subtitle {
    font-size: 0.72rem; color: #8b949e; margin-top: 1px;
}
.top-bar-badge {
    margin-left: auto;
    font-size: 0.68rem; font-weight: 500; letter-spacing: 0.05em;
    padding: 3px 10px; border-radius: 20px;
    background: rgba(35,134,54,0.15); color: #3fb950;
    border: 1px solid rgba(63,185,80,0.3);
}

/* ── chat window ── */
.chat-wrap {
    max-width: 820px; margin: 0 auto;
    padding: 24px 20px 200px;   /* bottom pad = input bar height */
    display: flex; flex-direction: column; gap: 20px;
}

/* ── bubbles ── */
.bubble-row { display: flex; gap: 12px; align-items: flex-end; }
.bubble-row.user { flex-direction: row-reverse; }

.avatar {
    width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 600;
}
.avatar.bot  { background: linear-gradient(135deg,#238636,#1f6feb); color:#fff; }
.avatar.user { background: #21262d; color: #8b949e; border:1px solid #30363d; }

.bubble {
    max-width: 72%; padding: 13px 16px;
    border-radius: 18px; line-height: 1.65;
    font-size: 0.9rem; position: relative;
    animation: fadeUp 0.3s ease both;
}
.bubble.bot {
    background: #161b22; border: 1px solid #21262d;
    border-bottom-left-radius: 4px; color: #c9d1d9;
}
.bubble.user {
    background: linear-gradient(135deg,#1f6feb,#388bfd);
    border-bottom-right-radius: 4px; color: #fff;
}
.bubble ul { padding-left: 18px; margin-top: 6px; }
.bubble li { margin-bottom: 5px; }
.bubble strong { color: #e6edf3; }

.source-tag {
    display: inline-block; margin-top: 10px; margin-right: 6px;
    font-size: 0.68rem; padding: 2px 8px; border-radius: 20px;
    background: rgba(31,111,235,0.12); color: #58a6ff;
    border: 1px solid rgba(88,166,255,0.2);
}

@keyframes fadeUp {
    from { opacity:0; transform:translateY(8px); }
    to   { opacity:1; transform:translateY(0); }
}

/* ── typing indicator ── */
.typing { display: flex; gap: 5px; padding: 14px 16px; }
.typing span {
    width: 7px; height: 7px; border-radius: 50%;
    background: #58a6ff; animation: blink 1.2s infinite;
}
.typing span:nth-child(2) { animation-delay: 0.2s; }
.typing span:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink {
    0%,80%,100% { opacity:0.2; transform:scale(0.85); }
    40%          { opacity:1;   transform:scale(1); }
}

/* ── welcome card ── */
.welcome-card {
    text-align: center; padding: 40px 20px;
    animation: fadeUp 0.5s ease both;
}
.welcome-card .shield { font-size: 3rem; margin-bottom: 14px; }
.welcome-card h2 {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem; color: #e6edf3; margin-bottom: 8px;
}
.welcome-card p { color: #8b949e; font-size: 0.88rem; line-height: 1.7; max-width: 460px; margin: 0 auto 24px; }
.pill-wrap { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; }
.pill {
    font-size: 0.78rem; padding: 7px 14px; border-radius: 20px;
    background: #161b22; border: 1px solid #30363d; color: #8b949e;
    cursor: pointer; transition: all 0.2s;
}
.pill:hover { border-color: #58a6ff; color: #58a6ff; background: rgba(88,166,255,0.06); }

/* ── fixed input bar ── */
.input-bar {
    position: fixed; bottom: 0; left: 0; right: 0; z-index: 200;
    padding: 16px 20px 20px;
    background: linear-gradient(to top, #0d1117 70%, transparent);
}
.input-inner {
    max-width: 820px; margin: 0 auto;
    display: flex; align-items: flex-end; gap: 10px;
    background: #161b22; border: 1px solid #30363d;
    border-radius: 16px; padding: 10px 12px;
    transition: border-color 0.2s;
}
.input-inner:focus-within { border-color: #58a6ff; }

/* gradio textbox overrides */
.input-inner .gr-textbox, .input-inner textarea {
    background: transparent !important; border: none !important;
    box-shadow: none !important; color: #e6edf3 !important;
    font-family: 'Sora', sans-serif !important; font-size: 0.9rem !important;
    resize: none !important; outline: none !important;
    flex: 1; min-height: 24px; max-height: 120px;
    padding: 2px 4px !important;
}
.input-inner textarea::placeholder { color: #484f58 !important; }

.send-btn {
    width: 38px; height: 38px; border-radius: 10px; flex-shrink: 0;
    background: linear-gradient(135deg,#238636,#1f6feb);
    border: none; cursor: pointer; display: flex;
    align-items: center; justify-content: center;
    transition: opacity 0.2s, transform 0.1s;
    font-size: 16px;
}
.send-btn:hover { opacity: 0.85; }
.send-btn:active { transform: scale(0.93); }

.input-tools {
    display: flex; align-items: center; gap: 8px; margin-top: 8px;
}
.tool-btn {
    font-size: 0.72rem; padding: 4px 10px; border-radius: 20px;
    background: transparent; border: 1px solid #30363d; color: #8b949e;
    cursor: pointer; display: flex; align-items: center; gap: 5px;
    transition: all 0.2s;
}
.tool-btn:hover { border-color: #58a6ff; color: #58a6ff; }
.tool-btn.active { background: rgba(35,134,54,0.12); border-color:#3fb950; color:#3fb950; }

/* audio component shrink */
#audio-wrap { display: none; }
#audio-wrap.visible { display: block; padding: 8px 0 0; }
"""

# ── JavaScript for chat behaviour ────────────────────────────────────────────
JS = """
function initChat() {
    // pill click → fill textarea
    document.querySelectorAll('.pill').forEach(p => {
        p.addEventListener('click', () => {
            const ta = document.querySelector('#text-input textarea');
            if (ta) {
                ta.value = p.textContent;
                ta.dispatchEvent(new Event('input', {bubbles:true}));
            }
        });
    });

    // mic toggle
    const micBtn = document.getElementById('mic-toggle');
    const audioWrap = document.getElementById('audio-wrap');
    if (micBtn && audioWrap) {
        micBtn.addEventListener('click', () => {
            audioWrap.classList.toggle('visible');
            micBtn.classList.toggle('active');
        });
    }
}
setTimeout(initChat, 800);
"""

WELCOME_HTML = """
<div class="welcome-card">
  <div class="shield">🛡️</div>
  <h2>How can I help you today?</h2>
  <p>
    Ask me anything about Makerere University's safeguarding policies,
    disability rights, sexual harassment procedures, and student protections.
    Answers are grounded in official policy documents.
  </p>
  <div class="pill-wrap">
    <div class="pill">How do I report harassment?</div>
    <div class="pill">Rights for students with disabilities</div>
    <div class="pill">How do I file a complaint?</div>
    <div class="pill">What is the HIV/AIDS policy?</div>
    <div class="pill">Support for persons with disabilities</div>
  </div>
</div>
"""


def render_message(role, content, sources=""):
    if role == "user":
        return f"""
        <div class="bubble-row user">
          <div class="avatar user">U</div>
          <div class="bubble user">{content}</div>
        </div>"""

    # bot: convert bullet lines to <ul>
    lines = content.strip().split("\n")
    html_parts = []
    in_ul = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- "):
            if not in_ul:
                html_parts.append("<ul>"); in_ul = True
            html_parts.append(f"<li>{stripped[2:]}</li>")
        else:
            if in_ul:
                html_parts.append("</ul>"); in_ul = False
            if stripped:
                html_parts.append(f"<p>{stripped}</p>")
    if in_ul:
        html_parts.append("</ul>")

    source_html = ""
    if sources:
        tags = "".join(
            f'<span class="source-tag">📄 {s.strip()}</span>'
            for s in sources.replace("Policy Sources:", "").split("\n")
            if s.strip().startswith("-")
        )
        if tags:
            source_html = f"<div style='margin-top:10px'>{tags}</div>"

    inner = "\n".join(html_parts) + source_html
    return f"""
    <div class="bubble-row bot">
      <div class="avatar bot">🛡</div>
      <div class="bubble bot">{inner}</div>
    </div>"""


def chat_fn(user_msg, audio_input, simplified, history_state):
    if not user_msg.strip() and audio_input is None:
        return history_state, build_chat_html(history_state), ""

    # resolve query (voice or text)
    query = ""
    if audio_input is not None:
        query = transcribe_audio(audio_input)
    if not query:
        query = user_msg.strip()

    if not query:
        return history_state, build_chat_html(history_state), ""

    # add user bubble
    history_state = history_state + [("user", query, "")]

    # greeting short-circuit
    if is_greeting(query):
        history_state = history_state + [("bot", GREETING_RESPONSE, "")]
        return history_state, build_chat_html(history_state), ""

    # RAG pipeline
    retrieved = retrieve_top_k(query, embedding_model, embeddings, df)
    answer    = generate_answer(query, retrieved, gen_tokenizer, gen_model, device)
    if simplified:
        answer = apply_simplified_language(answer)
    formatted = format_response(answer)

    sources = ""
    if not retrieved.empty:
        names = [
            s.replace(".pdf", "").replace("-", " ").replace("_", " ")
            for s in retrieved["source_document"].unique()
        ]
        sources = "Policy Sources:\n" + "\n".join(f"  - {n}" for n in names)

    history_state = history_state + [("bot", formatted, sources)]
    return history_state, build_chat_html(history_state), ""


def build_chat_html(history):
    if not history:
        return WELCOME_HTML
    parts = []
    for role, content, sources in history:
        parts.append(render_message(role, content, sources))
    return "\n".join(parts)


# ── Gradio blocks ─────────────────────────────────────────────────────────────

with gr.Blocks(css=CSS, title="eSafeRide — Safeguarding Companion") as demo:

    # top bar
    gr.HTML("""
    <div class="top-bar">
      <div class="top-bar-logo">🛡️</div>
      <div>
        <div class="top-bar-title">Safeguarding Companion</div>
        <div class="top-bar-subtitle">Makerere University · Policy Q&A</div>
      </div>
      <div class="top-bar-badge">● Online</div>
    </div>
    """)

    # chat display area
    chat_display = gr.HTML(value=WELCOME_HTML, elem_id="chat-display")

    # hidden state
    history_state = gr.State([])

    # ── fixed input bar (rendered as HTML + real Gradio inputs overlaid) ──
    gr.HTML("""
    <div class="input-bar">
      <div class="input-inner" id="input-inner-wrap">
    """)

    with gr.Row(elem_id="input-row"):
        text_input = gr.Textbox(
            placeholder="Ask about policies, rights, reporting...",
            show_label=False,
            lines=1,
            max_lines=4,
            elem_id="text-input",
            scale=9,
            container=False,
        )
        submit_btn = gr.Button("➤", elem_id="send-btn", scale=1, min_width=42, variant="primary")

    gr.HTML("""
      </div>
      <div class="input-tools">
        <button class="tool-btn active" id="simplified-hint" style="pointer-events:none">
          🌿 Plain English mode
        </button>
        <button class="tool-btn" id="mic-toggle">🎙 Voice input</button>
      </div>
    </div>
    """)

    # hidden simplified toggle (always-on for clean UX; expose if needed)
    simplified_toggle = gr.Checkbox(value=True, visible=False)

    # audio hidden by default
    with gr.Row(visible=False) as audio_row:
        audio_input = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Speak your question",
            show_label=False,
        )

    # wire up
    submit_btn.click(
        fn=chat_fn,
        inputs=[text_input, audio_input, simplified_toggle, history_state],
        outputs=[history_state, chat_display, text_input],
    )
    text_input.submit(
        fn=chat_fn,
        inputs=[text_input, audio_input, simplified_toggle, history_state],
        outputs=[history_state, chat_display, text_input],
    )

    gr.HTML(f"<script>{JS}</script>")

demo.launch(ssr_mode=False)