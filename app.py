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
CHUNK_CSV            = "/home/user/app/policy_chunks.csv"
EMBEDDINGS_NPY       = "/home/user/app/chunk_embeddings.npy"
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

print("All models ready.")

# ---------------------------------------------------------------------------
# MEMORY
# ---------------------------------------------------------------------------

conversation_history = {}

def get_history(session_id="default"):
    return conversation_history.get(session_id, [])

def update_history(session_id, role, message):
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    conversation_history[session_id].append((role, message))
    conversation_history[session_id] = conversation_history[session_id][-12:]


# ---------------------------------------------------------------------------
# STREAM CHAT (CHATGPT STYLE TYPING)
# ---------------------------------------------------------------------------

def chat_stream(user_msg, audio_input, history_state):

    if history_state is None:
        history_state = []

    # voice input
    if audio_input:
        user_msg = transcribe_audio(audio_input)

    if not user_msg:
        yield history_state, render_chat(history_state), ""
        return

    # save user message
    history_state.append(("user", user_msg))
    update_history("default", "user", user_msg)

    yield history_state, render_chat(history_state), ""

    # retrieval + answer
    retrieved = retrieve_top_k(user_msg, embedding_model, embeddings, df)
    answer = generate_answer(user_msg, retrieved, gen_tokenizer, gen_model, device)
    answer = apply_simplified_language(answer)
    answer = format_response(answer)

    # STREAMING EFFECT
    streamed = ""
    for word in answer.split():
        streamed += word + " "
        temp = history_state + [("bot", streamed)]
        yield temp, render_chat(temp), ""

    history_state.append(("bot", answer))
    update_history("default", "bot", answer)


# ---------------------------------------------------------------------------
# SIMPLE CHAT RENDER (CLEAN CHATGPT STYLE)
# ---------------------------------------------------------------------------

def render_chat(history):
    if not history:
        return """
        <div style='text-align:center;color:#8b949e;margin-top:40px;'>
            Ask a question to begin
        </div>
        """

    html = ""

    for role, msg in history:
        msg = msg.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")

        if role == "user":
            html += f"""
            <div style="display:flex;justify-content:flex-end;margin:10px;">
              <div style="
                background:linear-gradient(135deg,#1f6feb,#388bfd);
                color:white;padding:12px 14px;
                border-radius:18px;
                max-width:70%;
                font-size:0.95rem;">
                {msg}
              </div>
            </div>
            """
        else:
            html += f"""
            <div style="display:flex;justify-content:flex-start;margin:10px;">
              <div style="
                background:#161b22;
                border:1px solid #30363d;
                color:#c9d1d9;
                padding:12px 14px;
                border-radius:18px;
                max-width:70%;
                font-size:0.95rem;">
                {msg}
              </div>
            </div>
            """

    return html


# ---------------------------------------------------------------------------
# GRADIO UI (CLEAN CHATGPT STYLE)
# ---------------------------------------------------------------------------

import gradio as gr

CSS = """
body {
    background:#0d1117 !important;
    color:#e6edf3 !important;
    font-family: system-ui;
}

/* chat container */
#chat {
    height:75vh;
    overflow-y:auto;
    padding:20px;
}

/* bottom bar */
#bar {
    position:fixed;
    bottom:0;
    width:100%;
    display:flex;
    gap:10px;
    padding:14px;
    background:#161b22;
    border-top:1px solid #30363d;
}

/* textbox */
#msg textarea {
    border-radius:20px !important;
    padding:12px !important;
    font-size:1rem !important;
}

/* buttons */
#send {
    width:48px;
    border-radius:50%;
    background:#238636;
    color:white;
    font-size:18px;
}
"""

with gr.Blocks(css=CSS, title="Safeguarding Companion") as demo:

    gr.Markdown("# 🛡 Safeguarding Companion")

    history_state = gr.State([])

    chat_display = gr.HTML(render_chat([]), elem_id="chat")

    with gr.Row(elem_id="bar"):

        audio_input = gr.Audio(
            sources=["microphone"],
            type="filepath",
            scale=1
        )

        text_input = gr.Textbox(
            placeholder="Ask a question...",
            label="",
            elem_id="msg",
            scale=6
        )

        send_btn = gr.Button("➤", elem_id="send", scale=1)

    send_btn.click(
        chat_stream,
        inputs=[text_input, audio_input, history_state],
        outputs=[history_state, chat_display, text_input]
    )

    text_input.submit(
        chat_stream,
        inputs=[text_input, audio_input, history_state],
        outputs=[history_state, chat_display, text_input]
    )

demo.launch()