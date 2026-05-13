"""
Run this ONCE locally to pre-build chunks and embeddings.
Then commit policy_chunks.csv and chunk_embeddings.npy to your GitHub repo.
After that, app.py will download them instead of rebuilding every time.

Usage:
    pip install sentence-transformers PyPDF2 nltk pandas numpy requests
    python precompute.py
"""

import os
import re
import requests
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Same URLs as app.py ──────────────────────────────────────────────────────
GITHUB_PDF_URLS = [
    "https://raw.githubusercontent.com/madrinejean123/MACHINE-LEARNING--CHATBOT-GROUP-O/madrine/data/Makerere-Safeguarding-Policy.pdf",
    "https://raw.githubusercontent.com/madrinejean123/MACHINE-LEARNING--CHATBOT-GROUP-O/madrine/data/Policy-and-Regulations-Against-Sexual-Harassment-2018.pdf",
    "https://raw.githubusercontent.com/madrinejean123/MACHINE-LEARNING--CHATBOT-GROUP-O/madrine/data/Makerere-Policy-on-Persons-Living-With-Disabilities.pdf",
    "https://raw.githubusercontent.com/madrinejean123/MACHINE-LEARNING--CHATBOT-GROUP-O/madrine/data/FINAL-REVISED-NATIONAL-POLICY-ON-PWDs-2023.pdf",
    "https://raw.githubusercontent.com/madrinejean123/MACHINE-LEARNING--CHATBOT-GROUP-O/madrine/data/HIV_AIDS_Policy.pdf",
    "https://raw.githubusercontent.com/madrinejean123/MACHINE-LEARNING--CHATBOT-GROUP-O/madrine/data/UTAMU-Disability-Policy.pdf",
]

DATA_FOLDER    = "data"
CHUNK_CSV      = "policy_chunks.csv"
EMBEDDINGS_NPY = "chunk_embeddings.npy"
CHUNK_MAX_WORDS = 150
CHUNK_OVERLAP   = 1

STOP_WORDS = {
    "the", "and", "for", "are", "that", "this", "with", "how",
    "what", "who", "can", "you", "was", "has", "have", "been",
}

# ── Helpers (same as app.py) ─────────────────────────────────────────────────

def download_pdfs(urls, folder):
    os.makedirs(folder, exist_ok=True)
    paths = []
    for url in urls:
        filename = url.split("/")[-1]
        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            print(f"  ✓ Already exists: {filename}")
            paths.append(filepath)
            continue
        print(f"  ↓ Downloading {filename} ...")
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(r.content)
            paths.append(filepath)
            print(f"    ✓ Done ({len(r.content)//1024} KB)")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
    return paths


def extract_text_from_pdf(filepath):
    import PyPDF2
    text = ""
    try:
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + " "
        if len(text.strip()) > 50:
            return text.strip()
    except Exception as e:
        print(f"  PyPDF2 failed for {os.path.basename(filepath)}: {e}")

    # OCR fallback
    try:
        from pdf2image import convert_from_path
        import pytesseract
        print(f"  → OCR for {os.path.basename(filepath)} ...")
        images = convert_from_path(filepath, dpi=200)
        ocr_text = ""
        for img in images:
            ocr_text += pytesseract.image_to_string(img, lang="eng") + " "
        if ocr_text.strip():
            return ocr_text.strip()
    except Exception as e:
        print(f"  → OCR failed: {e}")
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
    for filename in pdf_files:
        print(f"\n  Processing {filename} ...")
        filepath = os.path.join(folder, filename)
        raw      = extract_text_from_pdf(filepath)
        if not raw:
            print(f"  ✗ No text extracted, skipping.")
            continue
        cleaned = clean_text(raw)
        chunks  = chunk_text(cleaned)
        print(f"  ✓ {len(chunks)} chunks")
        for idx, chunk in enumerate(chunks):
            records.append({
                "chunk_id":        f"{filename}_chunk_{idx}",
                "source_document": filename,
                "chunk_index":     idx,
                "text":            chunk,
                "word_count":      len(chunk.split()),
            })
    return pd.DataFrame(records)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Step 1: Download PDFs ===")
    download_pdfs(GITHUB_PDF_URLS, DATA_FOLDER)

    print("\n=== Step 2: Build chunks ===")
    df = build_dataset(DATA_FOLDER)
    if df.empty:
        raise RuntimeError("No text extracted from any PDF. Check your files.")
    df.to_csv(CHUNK_CSV, index=False)
    print(f"\n✓ Saved {len(df)} chunks to {CHUNK_CSV}")

    print("\n=== Step 3: Build embeddings ===")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        df["text"].astype(str).tolist(),
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True,
    )
    np.save(EMBEDDINGS_NPY, embeddings)
    print(f"✓ Saved embeddings ({embeddings.shape}) to {EMBEDDINGS_NPY}")

    print("\n✅ Done! Now commit these two files to your GitHub repo:")
    print(f"   git add {CHUNK_CSV} {EMBEDDINGS_NPY}")
    print( "   git commit -m 'Add pre-built chunks and embeddings'")
    print( "   git push")
    print("\nHugging Face will use them directly — no rebuilding on startup.")
