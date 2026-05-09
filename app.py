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
    "disability rights, sexual harassment, and more — in plain, simple English.\n\n"
    "You can ask things like:\n"
    "• How do I report harassment?\n"
    "• What rights do students with disabilities have?\n"
    "• How do I file a complaint?\n\n"
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
    """Try PyPDF2 first; fall back to OCR via pdf2image + pytesseract."""
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

    # OCR fallback — requires poppler (packages.txt) and tesseract (packages.txt)
    try:
        from pdf2image import convert_from_path
        import pytesseract
        print(f"  → Trying OCR for {os.path.basename(filepath)} ...")
        images = convert_from_path(filepath, dpi=200)
        ocr_text = ""
        for img in images:
            ocr_text += pytesseract.image_to_string(img, lang="eng") + " "
        if ocr_text.strip():
            print(f"  → OCR succeeded ({len(ocr_text.split())} words)")
            return ocr_text.strip()
    except Exception as e:
        print(f"  → OCR failed for {os.path.basename(filepath)}: {e}")

    return ""


def clean_text(text):
    """Remove artefacts, fix hyphenation, normalise whitespace."""
    fixes = {
        "har- assment": "harassment",
        "dis- ability": "disability",
        "re- port":     "report",
        "com- plaint":  "complaint",
    }
    for broken, fixed in fixes.items():
        text = text.replace(broken, fixed)

    # remove bullet/numbering artefacts like "a)", "b)", "1.", "i."
    text = re.sub(r"\b[a-zA-Z]\)\s*", "", text)        # a) b) c)
    text = re.sub(r"\b\d+\.\s*", "", text)              # 1. 2. 3.
    text = re.sub(r"\b[ivxlIVXL]+\.\s*", "", text)      # i. ii. iii.
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
        if not raw:
            print(f"  ⚠ No text extracted from {filename}, skipping.")
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
    """
    Remove leftover policy numbering artefacts and tidy up the model output.
    e.g.  '* b) Students must...' → 'Students must...'
    """
    # strip leading *, -, bullet chars
    text = re.sub(r"^[\*\-•]\s*", "", text, flags=re.MULTILINE)
    # strip lettered sub-items like "a)" "b)" at start of line or inline
    text = re.sub(r"\b[a-zA-Z]\)\s*", "", text)
    # strip numbered items
    text = re.sub(r"\b\d+[\.\)]\s*", "", text)
    # collapse extra whitespace
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
        "You are a friendly university support assistant. "
        "A student asked: \"{query}\"\n\n"
        "Using the policy excerpts below, write a clear, helpful answer "
        "in 3 to 5 short sentences. "
        "Do NOT copy sentences from the policy word-for-word. "
        "Do NOT use bullet letters like a) b) or numbers like 1. 2. "
        "Write naturally, as if explaining to a friend. "
        "Use plain English — no legal jargon.\n\n"
        f"Policy excerpts:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Helpful answer:"
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


def format_response(answer):
    """Turn the answer into clean readable sentences."""
    answer = clean_answer(answer)
    # if already has newlines leave it
    if "\n" in answer:
        return answer
    # split on full stops into short sentences
    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 8]
    if not sentences:
        return answer
    return " ".join(s + "." for s in sentences)


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
        raise RuntimeError("No documents could be processed. Check PDF URLs and poppler install.")
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
# CHAT LOGIC
# ---------------------------------------------------------------------------

def nice_source_name(raw):
    return (
        raw.replace(".pdf", "")
           .replace("-", " ")
           .replace("_", " ")
           .strip()
    )


def render_bubble(role, content, sources=""):
    """Return HTML for a single chat bubble."""
    if role == "user":
        safe = content.replace("<", "&lt;").replace(">", "&gt;")
        return f"""
        <div class="bubble-row user">
          <div class="avatar user-av">You</div>
          <div class="bubble user-bubble">{safe}</div>
        </div>"""

    # bot — clean up content
    safe = content.replace("<", "&lt;").replace(">", "&gt;")
    # convert newlines to <br>
    safe = safe.replace("\n", "<br>")

    source_html = ""
    if sources:
        lines = [l.strip() for l in sources.split("\n") if l.strip() and l.strip() != "Policy Sources:"]
        tags  = "".join(
            f'<span class="src-tag">📄 {nice_source_name(l.lstrip("- "))}</span>'
            for l in lines if l
        )
        if tags:
            source_html = f'<div class="src-row">{tags}</div>'

    return f"""
    <div class="bubble-row bot">
      <div class="avatar bot-av">🛡</div>
      <div class="bubble bot-bubble">{safe}{source_html}</div>
    </div>"""


WELCOME_HTML = """
<div class="welcome-card">
  <div class="w-icon">🛡️</div>
  <h2>Safeguarding Companion</h2>
  <p>Ask me anything about Makerere University's safeguarding policies,
  disability rights, sexual harassment procedures, and student protections.
  All answers come from official policy documents.</p>
  <div class="pill-row">
    <span class="pill" onclick="fillQuery(this)">How do I report harassment?</span>
    <span class="pill" onclick="fillQuery(this)">Rights for students with disabilities</span>
    <span class="pill" onclick="fillQuery(this)">How do I file a complaint?</span>
    <span class="pill" onclick="fillQuery(this)">What is the HIV/AIDS policy?</span>
    <span class="pill" onclick="fillQuery(this)">Support for persons with disabilities</span>
  </div>
</div>"""


def build_chat_html(history):
    if not history:
        return WELCOME_HTML
    return "\n".join(render_bubble(r, c, s) for r, c, s in history)


def chat_fn(user_msg, audio_input, history_state):
    query = ""
    if audio_input is not None:
        query = transcribe_audio(audio_input)
    if not query:
        query = (user_msg or "").strip()
    if not query:
        return history_state, build_chat_html(history_state), ""

    history_state = list(history_state) + [("user", query, "")]

    if is_greeting(query):
        history_state.append(("bot", GREETING_RESPONSE, ""))
        return history_state, build_chat_html(history_state), ""

    retrieved = retrieve_top_k(query, embedding_model, embeddings, df)
    answer    = generate_answer(query, retrieved, gen_tokenizer, gen_model, device)
    answer    = apply_simplified_language(answer)
    formatted = format_response(answer)

    sources = ""
    if not retrieved.empty:
        names   = retrieved["source_document"].unique()
        sources = "Policy Sources:\n" + "\n".join(f"- {n}" for n in names)

    history_state.append(("bot", formatted, sources))
    return history_state, build_chat_html(history_state), ""


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600&family=Playfair+Display:wght@700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    font-family: 'Sora', sans-serif !important;
    background: #0d1117 !important;
    color: #e6edf3 !important;
    min-height: 100vh;
}

footer, .gr-form > .label-wrap { display: none !important; }
.gradio-container { padding: 0 !important; max-width: 100% !important; }
.contain { padding: 0 !important; }

/* ── TOP BAR ── */
.top-bar {
    position: sticky; top: 0; z-index: 100;
    display: flex; align-items: center; gap: 14px;
    padding: 14px 28px;
    background: rgba(13,17,23,0.92);
    backdrop-filter: blur(14px);
    border-bottom: 1px solid #21262d;
}
.top-logo {
    width: 40px; height: 40px; border-radius: 12px;
    background: linear-gradient(135deg,#238636,#1f6feb);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; flex-shrink: 0;
}
.top-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem; color: #e6edf3;
}
.top-sub { font-size: 0.7rem; color: #8b949e; margin-top: 2px; }
.online-badge {
    margin-left: auto; font-size: 0.68rem; padding: 3px 12px;
    border-radius: 20px; background: rgba(35,134,54,0.15);
    color: #3fb950; border: 1px solid rgba(63,185,80,0.3);
    font-weight: 500;
}

/* ── CHAT AREA ── */
.chat-scroll {
    max-width: 780px; margin: 0 auto;
    padding: 28px 20px 160px;
    display: flex; flex-direction: column; gap: 22px;
}

/* ── WELCOME ── */
.welcome-card {
    text-align: center; padding: 48px 20px 32px;
    animation: fadeUp .45s ease both;
}
.w-icon { font-size: 3.2rem; margin-bottom: 14px; }
.welcome-card h2 {
    font-family: 'Playfair Display', serif;
    font-size: 1.7rem; color: #e6edf3; margin-bottom: 10px;
}
.welcome-card p {
    color: #8b949e; font-size: 0.88rem; line-height: 1.75;
    max-width: 480px; margin: 0 auto 26px;
}
.pill-row { display: flex; flex-wrap: wrap; gap: 9px; justify-content: center; }
.pill {
    font-size: 0.78rem; padding: 8px 16px; border-radius: 20px;
    background: #161b22; border: 1px solid #30363d; color: #8b949e;
    cursor: pointer; transition: all .2s; user-select: none;
}
.pill:hover { border-color: #58a6ff; color: #58a6ff; background: rgba(88,166,255,.07); }

/* ── BUBBLES ── */
.bubble-row {
    display: flex; gap: 12px; align-items: flex-end;
    animation: fadeUp .3s ease both;
}
.bubble-row.user { flex-direction: row-reverse; }

.avatar {
    width: 34px; height: 34px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 700; letter-spacing: -.3px;
}
.bot-av  { background: linear-gradient(135deg,#238636,#1f6feb); color:#fff; font-size:16px; }
.user-av { background: #21262d; color: #8b949e; border: 1px solid #30363d; font-size: 10px; }

.bubble {
    max-width: 74%; padding: 14px 18px;
    font-size: 0.895rem; line-height: 1.72;
    border-radius: 20px; position: relative;
}
.bot-bubble {
    background: #161b22; border: 1px solid #21262d;
    border-bottom-left-radius: 5px; color: #c9d1d9;
}
.user-bubble {
    background: linear-gradient(135deg,#1f6feb,#388bfd);
    border-bottom-right-radius: 5px; color: #fff;
}

/* source tags inside bot bubble */
.src-row { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 6px; }
.src-tag {
    font-size: 0.67rem; padding: 3px 9px; border-radius: 20px;
    background: rgba(31,111,235,.12); color: #58a6ff;
    border: 1px solid rgba(88,166,255,.2);
}

@keyframes fadeUp {
    from { opacity:0; transform:translateY(10px); }
    to   { opacity:1; transform:translateY(0); }
}

/* ── FIXED INPUT BAR ── */
/* Gradio renders gr.Row as a div with the elem_id directly on it */
#input-bar {
    position: fixed !important;
    bottom: 0 !important; left: 0 !important; right: 0 !important;
    z-index: 200 !important;
    padding: 14px 20px 22px !important;
    background: linear-gradient(to top, #0d1117 80%, transparent) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0 !important;
}

/* inner pill shell — wrap the row's children */
#input-bar > .wrap,
#input-bar > div {
    max-width: 780px;
    width: 100%;
    display: flex !important;
    align-items: center !important;
    background: #1c2128 !important;
    border: 1.5px solid #30363d !important;
    border-radius: 28px !important;
    padding: 8px 10px 8px 20px !important;
    gap: 8px !important;
    transition: border-color .2s;
    min-height: 58px;
}
#input-bar > .wrap:focus-within,
#input-bar > div:focus-within {
    border-color: #58a6ff !important;
}

/* textbox */
#qbox {
    flex: 1 !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
}
#qbox .wrap { border: none !important; box-shadow: none !important; background: transparent !important; padding: 0 !important; }
#qbox label { display: none !important; }
#qbox textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #e6edf3 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.97rem !important;
    resize: none !important;
    outline: none !important;
    min-height: 28px !important;
    max-height: 140px !important;
    padding: 4px 0 !important;
    line-height: 1.55 !important;
}
#qbox textarea::placeholder { color: #484f58 !important; }

/* mic audio widget — compact */
#mic-widget {
    flex-shrink: 0 !important;
    width: 54px !important;
    min-width: 54px !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
#mic-widget .wrap,
#mic-widget label { display: none !important; }
/* show just the mic icon button from Gradio's audio widget */
#mic-widget button {
    width: 40px !important; height: 40px !important;
    border-radius: 50% !important;
    background: transparent !important;
    border: 1px solid #30363d !important;
    color: #8b949e !important;
    font-size: 18px !important;
    cursor: pointer !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    transition: all .2s !important;
}
#mic-widget button:hover { border-color: #58a6ff !important; color: #58a6ff !important; }

/* send button */
#send-btn {
    flex-shrink: 0 !important;
    width: 44px !important; height: 44px !important;
    border-radius: 50% !important;
    background: linear-gradient(135deg,#238636,#1f6feb) !important;
    color: #fff !important;
    font-size: 18px !important;
    border: none !important;
    padding: 0 !important;
    min-width: 44px !important;
    transition: opacity .2s, transform .1s !important;
}
#send-btn:hover { opacity: .85 !important; }
#send-btn:active { transform: scale(.92) !important; }
"""

# ---------------------------------------------------------------------------
# JS — pill click fills the textarea; auto-scroll on new messages
# ---------------------------------------------------------------------------
JS = """
<script>
function fillQuery(el) {
    const ta = document.querySelector('#qbox textarea');
    if (!ta) return;
    ta.value = el.textContent.trim();
    ta.dispatchEvent(new Event('input', {bubbles: true}));
    ta.focus();
}

// auto-scroll to bottom whenever chat content changes
const _obs = new MutationObserver(() => {
    window.scrollTo(0, document.body.scrollHeight);
});
setTimeout(() => {
    const el = document.getElementById('chat-display');
    if (el) _obs.observe(el, { childList: true, subtree: true });
}, 1200);
</script>
"""

# ---------------------------------------------------------------------------
# GRADIO BLOCKS
# ---------------------------------------------------------------------------

with gr.Blocks(title="Safeguarding Companion") as demo:

    # ── TOP BAR
    gr.HTML("""
    <div class="top-bar">
      <div class="top-logo">🛡️</div>
      <div>
        <div class="top-title">Safeguarding Companion</div>
        <div class="top-sub">Makerere University · Policy Q&A</div>
      </div>
      <div class="online-badge">● Online</div>
    </div>
    """)

    # ── CHAT DISPLAY
    with gr.Column(elem_classes=["chat-scroll"]):
        chat_display = gr.HTML(value=WELCOME_HTML, elem_id="chat-display")

    # ── HIDDEN STATE
    history_state = gr.State([])

    # ── FIXED INPUT BAR — one proper Gradio Row, styled entirely via CSS
    with gr.Row(elem_id="input-bar"):
        text_input = gr.Textbox(
            placeholder="Ask anything about university policies...",
            show_label=False,
            lines=1,
            max_lines=5,
            elem_id="qbox",
            scale=8,
            container=False,
        )
        audio_input = gr.Audio(
            sources=["microphone"],
            type="filepath",
            show_label=False,
            elem_id="mic-widget",
            scale=1,
            min_width=54,
        )
        send_btn = gr.Button(
            "➤",
            elem_id="send-btn",
            scale=1,
            min_width=54,
        )

    # ── WIRE UP
    send_btn.click(
        fn=chat_fn,
        inputs=[text_input, audio_input, history_state],
        outputs=[history_state, chat_display, text_input],
    )
    text_input.submit(
        fn=chat_fn,
        inputs=[text_input, audio_input, history_state],
        outputs=[history_state, chat_display, text_input],
    )

    # inject JS last
    gr.HTML(JS)

demo.launch(ssr_mode=False, css=CSS)