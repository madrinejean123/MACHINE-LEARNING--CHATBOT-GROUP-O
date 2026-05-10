"""
document.py — PDF downloading, text extraction, cleaning and chunking
"""

import os
import re
import requests
from nltk.tokenize import sent_tokenize
from config import CHUNK_MAX_WORDS, CHUNK_OVERLAP


def download_pdfs(urls: list, folder: str) -> list:
    os.makedirs(folder, exist_ok=True)
    paths = []
    for url in urls:
        filename = url.split("/")[-1]
        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            paths.append(filepath)
            continue
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(r.content)
            paths.append(filepath)
        except Exception as e:
            print(f"Could not download {filename}: {e}")
    return paths


def extract_text_from_pdf(filepath: str) -> str:
    import PyPDF2
    text = ""
    try:
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                pt = page.extract_text()
                if pt:
                    text += pt + " "
        if len(text.strip()) > 50:
            return text.strip()
    except Exception as e:
        print(f"PyPDF2 failed for {os.path.basename(filepath)}: {e}")

    # OCR fallback
    try:
        from pdf2image import convert_from_path
        import pytesseract
        images  = convert_from_path(filepath, dpi=200)
        ocr     = ""
        for img in images:
            ocr += pytesseract.image_to_string(img, lang="eng") + " "
        if ocr.strip():
            return ocr.strip()
    except Exception as e:
        print(f"OCR failed for {os.path.basename(filepath)}: {e}")

    return ""


def clean_text(text: str) -> str:
    ocr_fixes = {
        "mustbe": "must be", "mustset": "must set", "mustnot": "must not",
        "shouldbe": "should be", "canbe": "can be", "willbe": "will be",
        "shallbe": "shall be", "togo": "to go", "todo": "to do",
        "tothe": "to the", "ofthe": "of the", "inthe": "in the",
        "forthe": "for the", "andthe": "and the", "bythe": "by the",
        "onthe": "on the", "atthe": "at the", "withthe": "with the",
        "fromthe": "from the",
        "Directorateof": "Directorate of",
        "GenderMainstreaming": "Gender Mainstreaming",
        "Deanof": "Dean of", "complaintsof": "complaints of",
        "evidenceof": "evidence of", "rightto": "right to",
        "entitledto": "entitled to", "accessto": "access to",
        "subjectto": "subject to",
        "har assment": "harassment", "dis ability": "disability",
        "com plaint": "complaint",
        "har- assment": "harassment", "dis- ability": "disability",
    }
    for broken, fixed in ocr_fixes.items():
        text = re.sub(re.escape(broken), fixed, text, flags=re.IGNORECASE)

    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    text = re.sub(r"\b[a-zA-Z]\)\s*", "", text)
    text = re.sub(r"\b\d+\.\s*", "", text)
    text = re.sub(r"\b[ivxlIVXL]+\.\s*", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,;:()\-\/ ]", "", text)
    text = re.sub(r"\(\d+\)", "", text)
    return text.strip()


def chunk_text(text: str, max_words=CHUNK_MAX_WORDS, overlap=CHUNK_OVERLAP) -> list:
    sentences   = sent_tokenize(text)
    chunks, current, current_len = [], [], 0
    for sentence in sentences:
        wc = len(sentence.split())
        if current_len + wc <= max_words:
            current.append(sentence)
            current_len += wc
        else:
            if current:
                chunks.append(" ".join(current))
            tail        = current[-overlap:] if overlap > 0 else []
            current     = tail + [sentence]
            current_len = sum(len(s.split()) for s in current)
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_dataset(folder: str):
    import pandas as pd
    records   = []
    pdf_files = [f for f in os.listdir(folder) if f.endswith(".pdf")]
    if not pdf_files:
        return pd.DataFrame()
    for filename in pdf_files:
        filepath = os.path.join(folder, filename)
        raw      = extract_text_from_pdf(filepath)
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