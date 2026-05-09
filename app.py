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

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');

body, .gradio-container {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #f4f1eb !important;
}

.app-header {
    background: #1b3a2d;
    color: #f4f1eb;
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}

.app-header h1 {
    color: #c8e6c9;
    font-size: 2rem;
    margin: 0 0 0.4rem 0;
    font-family: 'DM Serif Display', serif;
}

.app-header p {
    color: #a5d6a7;
    margin: 0;
    font-size: 0.95rem;
    line-height: 1.5;
}

footer { display: none !important; }
"""

with gr.Blocks(css=CSS, title="Safeguarding Companion") as demo:

    gr.HTML("""
    <div class="app-header">
        <h1>Safeguarding Companion</h1>
        <p>
            Ask questions about safeguarding policies, disability rights, sexual harassment
            reporting, and student protection procedures.
            Answers are grounded in official Makerere University policy documents
            and written in plain, simple English.
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            text_input = gr.Textbox(
                lines=2,
                placeholder="Type your question here...",
                label="Your Question",
            )
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Or speak your question",
            )

    simplified_toggle = gr.Checkbox(
        label="Simplified language mode (plainer English)",
        value=False,
    )

    submit_btn = gr.Button("Get Answer", variant="primary")

    with gr.Row():
        with gr.Column(scale=3):
            text_output = gr.Textbox(
                lines=10,
                label="Answer",
                interactive=False,
            )
        with gr.Column(scale=1):
            sources_output = gr.Textbox(
                lines=6,
                label="Policy Sources",
                interactive=False,
            )

    gr.Examples(
        examples=[
            ["How do I report sexual harassment?", None, False],
            ["What rights do students with disabilities have?", None, False],
            ["How do I file a complaint?", None, True],
            ["What support is available for persons with disabilities?", None, False],
            ["What is the university HIV/AIDS policy?", None, False],
        ],
        inputs=[text_input, audio_input, simplified_toggle],
        label="Example questions - click any to try",
    )

    submit_btn.click(
        fn=handle_query,
        inputs=[text_input, audio_input, simplified_toggle],
        outputs=[text_output, sources_output],
    )

    text_input.submit(
        fn=handle_query,
        inputs=[text_input, audio_input, simplified_toggle],
        outputs=[text_output, sources_output],
    )

    gr.HTML("""
    <div style="margin-top:1.5rem; padding:1rem; background:#e8f5e9;
                border-radius:8px; font-size:0.85rem; color:#1b3a2d;">
        <strong>Accessibility features:</strong>
        Full keyboard navigation. Screen-reader compatible.
        Voice input via microphone.
        Simplified language mode for plainer English responses.
    </div>
    """)

demo.launch(ssr_mode=False)