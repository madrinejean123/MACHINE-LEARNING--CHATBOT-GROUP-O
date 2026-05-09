"""
eSafeRide Safeguarding Companion
RAG-Based Policy Question-Answering System for Makerere University
- Uses Groq (Llama 3) for free, fast, clean answer generation
"""

import os
import re
import time
import requests
import numpy as np
import pandas as pd
import nltk
import streamlit as st
from datetime import datetime
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
CHUNK_MAX_WORDS      = 250
CHUNK_OVERLAP        = 2
TOP_K                = 7
SIMILARITY_THRESHOLD = 0.20

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

SUGGESTIONS = [
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
    # Expanded OCR fix dictionary — covers all common merged-word patterns
    ocr_fixes = {
        # Must/should/can merges
        "mustbe": "must be",
        "mustset": "must set",
        "mustnot": "must not",
        "shouldbe": "should be",
        "canbe": "can be",
        "willbe": "will be",
        "shallbe": "shall be",
        # Common preposition merges
        "togo": "to go",
        "todo": "to do",
        "tothe": "to the",
        "ofthe": "of the",
        "inthe": "in the",
        "forthe": "for the",
        "andthe": "and the",
        "bythe": "by the",
        "onthe": "on the",
        "atthe": "at the",
        "withthe": "with the",
        "fromthe": "from the",
        "intothe": "into the",
        # Institution / department merges
        "Directorateof": "Directorate of",
        "Mainstreamingor": "Mainstreaming or",
        "Mainstreamingto": "Mainstreaming to",
        "GenderMainstreaming": "Gender Mainstreaming",
        "Deanof": "Dean of",
        # Complaint / report merges
        "complaintsof": "complaints of",
        "complaintof": "complaint of",
        "complaintshould": "complaint should",
        "evidenceof": "evidence of",
        "casesof": "cases of",
        "formsof": "forms of",
        "typesof": "types of",
        "reportof": "report of",
        # Person / victim merges
        "victimsto": "victims to",
        "victimof": "victim of",
        "personin": "person in",
        "personwho": "person who",
        "personwith": "person with",
        # Action merges
        "actioncan": "action can",
        "actionmust": "action must",
        "stepsto": "steps to",
        "ableto": "able to",
        "failsto": "fails to",
        "needto": "need to",
        "hasto": "has to",
        "wantto": "want to",
        "doingany": "doing any",
        "takingany": "taking any",
        "subjectto": "subject to",
        "accessto": "access to",
        "rightto": "right to",
        "entitledto": "entitled to",
        # Word-break hyphen OCR artifacts
        "specialcir cumstances": "special circumstances",
        "specialcircumstances": "special circumstances",
        "cir cumstances": "circumstances",
        "har assment": "harassment",
        "haras sment": "harassment",
        "dis ability": "disability",
        "com plaint": "complaint",
        "investiga tion": "investigation",
        "com mittee": "committee",
        "har- assment": "harassment",
        "dis- ability": "disability",
        "re- port": "report",
        "com- plaint": "complaint",
        # Number / list prefix cleanup already in regex below
    }

    for broken, fixed in ocr_fixes.items():
        text = re.sub(re.escape(broken), fixed, text, flags=re.IGNORECASE)

    # Fix hyphenated line-break merges e.g. "harass-\nment"
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)

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

QUERY_EXPANSIONS = {
    "harass":    "report complaint procedure steps lodge file officer committee investigation",
    "assault":   "report complaint procedure steps lodge file officer committee investigation",
    "abuse":     "report complaint procedure steps lodge file officer committee investigation",
    "harassed":  "report complaint procedure steps lodge file officer committee investigation",
    "disabilit": "rights accommodation access support services equal opportunity",
    "disabled":  "rights accommodation access support services equal opportunity",
    "pwd":       "rights accommodation access support services equal opportunity",
    "hiv":       "rights confidentiality treatment support non-discrimination policy",
    "aids":      "rights confidentiality treatment support non-discrimination policy",
    "report":    "complaint procedure steps lodge file officer committee investigation",
    "complain":  "complaint procedure steps lodge file officer committee investigation",
    "file":      "complaint procedure steps lodge file officer committee investigation",
    "right":     "rights responsibilities protection policy entitlement",
    "protect":   "safeguarding protection rights policy procedure",
}


def expand_query(query):
    q_lower = query.lower()
    extras  = set()
    for trigger, expansion in QUERY_EXPANSIONS.items():
        if trigger in q_lower:
            extras.update(expansion.split())
    if extras:
        return query + " " + " ".join(extras)
    return query


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
    expanded = expand_query(query)
    filtered = keyword_filter(df, query)
    indices  = filtered.index.tolist()
    f_embeds = embeddings[indices]

    q_vec  = model.encode([expanded], normalize_embeddings=True)
    scores = cosine_similarity(q_vec, f_embeds)[0]

    sorted_i = np.argsort(scores)[::-1]
    top_i    = [i for i in sorted_i[:k] if scores[i] >= threshold]
    if not top_i:
        top_i = sorted_i[:5]

    results = filtered.iloc[top_i].copy()
    results["similarity_score"] = scores[top_i]

    ACTION_WORDS = ["report", "complain", "lodge", "file", "contact", "procedure",
                    "steps", "committee", "officer", "directorate", "submit", "notify",
                    "support", "rights", "entitled", "must", "shall", "access"]

    def action_score(text):
        t = str(text).lower()
        return sum(1 for w in ACTION_WORDS if w in t)

    results["action_boost"] = results["text"].apply(action_score)
    results = results.sort_values(
        by=["action_boost", "similarity_score"],
        ascending=[False, False]
    ).drop(columns=["action_boost"])

    return results


# ---------------------------------------------------------------------------
# GROQ GENERATION  (free, fast, clean — needs GROQ_API_KEY secret in Space)
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.1-8b-instant"   # current active free model on Groq (2026)


def polish_with_groq(question: str, raw_policy_text: str) -> str:
    """
    Sends retrieved policy chunks to Groq (free Llama 3).
    Rewrites them into clean, plain English with numbered steps.
    Returns polished string, or None if the call fails.
    """
    if not GROQ_API_KEY:
        print("No GROQ_API_KEY found — skipping Groq polish.")
        return None

    system_prompt = (
        "You are a helpful university safeguarding assistant for Makerere University students. "
        "Your job is to explain university policies in plain, simple English that any student can understand. "
        "Rules:\n"
        "1. Fix OCR errors like merged words (e.g. 'mustbe' -> 'must be', 'Directorateof' -> 'Directorate of').\n"
        "2. If the answer describes steps or a procedure, use a numbered list (1. 2. 3.).\n"
        "3. If it is general information, use clear bullet points starting with •.\n"
        "4. Use ONLY the policy text provided. Do not add anything extra.\n"
        "5. Remove repeated sentences.\n"
        "6. Keep it short and friendly.\n"
        "7. Always end with: 'For more help, contact the Directorate of Gender Mainstreaming at gendermainstreaming@mak.ac.ug or call +256 (0)414 532 631.'"
    )

    user_prompt = (
        f"A student asked: \"{question}\"\n\n"
        f"Here is the raw policy text retrieved from official documents:\n"
        f"---\n{raw_policy_text[:3000]}\n---\n\n"
        f"Write a clear, simple answer for the student now:"
    )

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":       GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                "max_tokens":  600,
                "temperature": 0.3,
            },
            timeout=30,
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"].strip()
        return answer if answer else None

    except Exception as e:
        print(f"Groq polish failed: {e}")
        return None


# ---------------------------------------------------------------------------
# FALLBACK FORMATTER  (used when flan-t5 generation fails)
# ---------------------------------------------------------------------------

def is_greeting(text):
    cleaned = text.strip().lower().rstrip("!.,?")
    words   = cleaned.split()
    return cleaned in GREETINGS or (len(words) <= 3 and words[0] in GREETINGS)


def nice_source_name(raw):
    return raw.replace(".pdf", "").replace("-", " ").replace("_", " ").strip()


def clean_sentence(s):
    s = re.sub(
        r'\b([a-zA-Z]{2,})\s([a-z]{1,3})\b',
        lambda m: m.group(1) + m.group(2) if m.group(2) not in STOP_WORDS else m.group(0),
        s
    )
    s = re.sub(r'^\s*[\(\[]?[a-zA-Z0-9]+[\)\]\.]\s*', '', s)
    s = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def split_into_sentences(text):
    raw  = re.split(r'(?<=[.!?])\s+', text)
    seen = set()
    sentences = []
    for s in raw:
        s = clean_sentence(s)
        if len(s.split()) < 6:
            continue
        key = re.sub(r'\s+', ' ', s.lower().strip())
        if key in seen:
            continue
        seen.add(key)
        sentences.append(s)
    return sentences


SKIP_PHRASES = [
    "there is no documented", "current evidence", "principles underpinning",
    "it should be noted", "as quasi-judicial", "no current evidence",
    "standard procedures concerning", "enjoy relative flexibility",
]


def format_chunks_as_bullets(retrieved, query=""):
    """Fallback formatter when flan-t5 generation is unavailable."""
    ACTION_WORDS = {
        "report", "lodge", "file", "contact", "collect", "document",
        "record", "seek", "notify", "submit", "communicate", "keep",
        "note", "familiarize", "request", "support", "access", "entitled",
        "rights", "must", "should", "procedure", "steps", "committee",
        "directorate", "officer", "complaint", "evidence", "witness",
        "can", "will", "shall", "ensure", "provide", "receive",
    }

    grouped = {}
    for _, row in retrieved.iterrows():
        src = nice_source_name(row["source_document"])
        grouped.setdefault(src, []).append(row["text"].strip())

    parts         = []
    total_bullets = 0

    for src, texts in grouped.items():
        combined  = " ".join(texts)
        sentences = split_into_sentences(combined)
        sentences = [s for s in sentences if not any(p in s.lower() for p in SKIP_PHRASES)]
        if not sentences:
            continue

        scored = [(sum(1 for w in ACTION_WORDS if w in s.lower()), s) for s in sentences]
        scored.sort(key=lambda x: x[0], reverse=True)
        top   = [s for _, s in scored[:8]]
        order = {s: i for i, s in enumerate(sentences)}
        top.sort(key=lambda s: order.get(s, 999))

        parts.append(f"\n**📋 {src}**\n")
        for s in top:
            if not s.endswith(('.', '!', '?')):
                s += '.'
            parts.append(f"- {s}")
            total_bullets += 1

    if total_bullets == 0:
        return (
            "I found related policy sections but could not extract clear steps.\n\n"
            "Please contact the **Directorate of Gender Mainstreaming** directly.\n"
            "📞 +256 (0)414 532 631 · 📧 gendermainstreaming@mak.ac.ug"
        )

    header = f"Here is what the policies say about **{query.strip()}**:\n\n" if query else ""
    return header + "\n".join(parts)


# ---------------------------------------------------------------------------
# MAIN ANSWER GENERATOR
# ---------------------------------------------------------------------------

def format_as_bullets(answer):
    """Ensure flan-t5 output displays as clean bullet points."""
    if '\n' in answer or answer.startswith('•') or answer.startswith('-') or answer.startswith('*'):
        return answer
    sentences = [s.strip() for s in answer.split('.') if len(s.strip()) > 10]
    return '\n'.join(f"• {s}." for s in sentences)


def generate_answer(question: str, retrieved) -> str:
    """
    1. Collects raw text from retrieved policy chunks.
    2. Sends to Groq (free Llama 3) to rewrite into plain English.
    3. Falls back to bullet formatter if Groq call fails.
    """
    if retrieved is None or retrieved.empty:
        return (
            "I could not find specific information about that in the policy documents.\n\n"
            "Please contact the **Directorate of Gender Mainstreaming** directly:\n"
            "📞 +256 (0)414 532 631 · 📧 gendermainstreaming@mak.ac.ug"
        )

    # Build raw policy context from all retrieved chunks
    raw_policy_text = "\n\n".join(
        f"[Source: {nice_source_name(row['source_document'])}]\n{row['text'].strip()}"
        for _, row in retrieved.iterrows()
    )

    # Try Groq first
    polished = polish_with_groq(question, raw_policy_text)
    if polished:
        return polished

    # Fallback to bullet formatter
    return format_chunks_as_bullets(retrieved, query=question)


def apply_simplified_language(text):
    replacements = {
        "pursuant to": "according to", "stipulates": "says",
        "thereof": "of it", "aforementioned": "mentioned above",
        "provisions": "rules", "shall": "must", "whilst": "while",
        "herein": "in this document", "hereunder": "below", "aforesaid": "mentioned",
    }
    for formal, plain in replacements.items():
        text = text.replace(formal, plain)
    return text


def fix_list_spacing(text):
    """
    Ensures every bullet point (• or -) and numbered item (1. 2. 3.)
    is on its own separate line with a blank line before it for breathing room.
    """
    lines = text.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        # Numbered list item e.g. "1." "2." "10."
        if re.match(r'^\d+\.', stripped):
            if result and result[-1] != "":
                result.append("")
            result.append(stripped)
        # Bullet point starting with • or -
        elif stripped.startswith("•") or stripped.startswith("-"):
            if result and result[-1] != "":
                result.append("")
            result.append(stripped)
        else:
            result.append(line)

    # Remove duplicate blank lines
    final = []
    prev_blank = False
    for line in result:
        if line == "":
            if not prev_blank:
                final.append(line)
            prev_blank = True
        else:
            final.append(line)
            prev_blank = False

    return "\n".join(final)


def format_response(answer):
    answer = apply_simplified_language(answer)
    answer = fix_list_spacing(answer)
    return answer


# ---------------------------------------------------------------------------
# LOAD EMBEDDING MODEL + PRE-BUILT CHUNKS
# ---------------------------------------------------------------------------

GITHUB_RAW_BASE    = (
    "https://raw.githubusercontent.com/"
    "madrinejean123/MACHINE-LEARNING--CHATBOT-GROUP-O/madrine"
)
CHUNK_CSV_URL      = f"{GITHUB_RAW_BASE}/policy_chunks.csv"
EMBEDDINGS_NPY_URL = f"{GITHUB_RAW_BASE}/chunk_embeddings.npy"

_CACHE_DIR  = "/data" if os.path.isdir("/data") else "."
_CHUNK_PATH = os.path.join(_CACHE_DIR, "policy_chunks.csv")
_EMBED_PATH = os.path.join(_CACHE_DIR, "chunk_embeddings.npy")


def _download_if_missing(url, local_path):
    """Always re-download to ensure latest GitHub version is used."""
    if os.path.exists(local_path):
        os.remove(local_path)
        print(f"Refreshing {os.path.basename(local_path)} ...")
    print(f"Downloading {os.path.basename(local_path)} from GitHub ...")
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
        print(f"  Saved ({len(r.content)//1024} KB)")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


@st.cache_resource(show_spinner=False)
def load_everything():
    csv_ok = _download_if_missing(CHUNK_CSV_URL,      _CHUNK_PATH)
    npy_ok = _download_if_missing(EMBEDDINGS_NPY_URL, _EMBED_PATH)

    if not (csv_ok and npy_ok):
        raise RuntimeError(
            "Could not load pre-built chunks from GitHub. "
            "Check that policy_chunks.csv and chunk_embeddings.npy "
            "are pushed to the madrine branch."
        )

    df         = pd.read_csv(_CHUNK_PATH)
    embeddings = np.load(_EMBED_PATH)
    print(f"Loaded {len(df)} chunks, embeddings shape {embeddings.shape}")

    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    return df, embeddings, emb_model


# ---------------------------------------------------------------------------
# PAGE CONFIG  (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Safeguarding Companion",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------

if "dark_mode"       not in st.session_state: st.session_state.dark_mode       = True
if "contrast"        not in st.session_state: st.session_state.contrast        = 100
if "font_size"       not in st.session_state: st.session_state.font_size       = 16
if "conversations"   not in st.session_state: st.session_state.conversations   = []
if "active_conv_id"  not in st.session_state: st.session_state.active_conv_id  = None
if "suggested_query" not in st.session_state: st.session_state.suggested_query = None

# ---------------------------------------------------------------------------
# DYNAMIC CSS
# ---------------------------------------------------------------------------

_dark  = st.session_state.dark_mode
_cont  = st.session_state.contrast / 100
_fsize = st.session_state.font_size

if _dark:
    BG, SIDEBAR, BORDER = "#0d1117", "#161b22", "#21262d"
    TEXT    = f"rgba(230,237,243,{min(_cont,1)})"
    SUBTEXT = f"rgba(139,148,158,{min(_cont,1)})"
    INPUT_BG, INPUT_BOR = "#1c2128", "#30363d"
else:
    BG, SIDEBAR, BORDER = "#ffffff", "#f6f8fa", "#d0d7de"
    TEXT    = f"rgba(31,35,40,{min(_cont,1)})"
    SUBTEXT = f"rgba(87,96,106,{min(_cont,1)})"
    INPUT_BG, INPUT_BOR = "#ffffff", "#d0d7de"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600&family=Playfair+Display:wght@700&display=swap');
html, body, [class*="css"] {{
    font-family: 'Sora', sans-serif !important;
    background-color: {BG} !important;
    color: {TEXT} !important;
    font-size: {_fsize}px !important;
}}
[data-testid="stAppViewContainer"] {{ filter: contrast({_cont}); }}
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stSidebar"] {{
    background-color: {SIDEBAR} !important;
    border-right: 1px solid {BORDER} !important;
}}
[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
.sidebar-logo {{
    display:flex; align-items:center; gap:10px;
    padding:4px 0 20px; border-bottom:1px solid {BORDER}; margin-bottom:16px;
}}
.sidebar-logo-icon {{
    width:34px; height:34px; border-radius:10px;
    background:linear-gradient(135deg,#238636,#1f6feb);
    display:flex; align-items:center; justify-content:center;
    font-size:18px; flex-shrink:0;
}}
.sidebar-logo-text {{ font-size:0.9rem; font-weight:600; }}
.sidebar-logo-sub  {{ font-size:0.65rem; color:{SUBTEXT} !important; }}
.sidebar-section-label {{
    font-size:0.65rem; font-weight:600; letter-spacing:.08em;
    text-transform:uppercase; color:{SUBTEXT} !important; padding:12px 0 6px;
}}
.main-header {{
    display:flex; align-items:center; gap:14px;
    padding:0 0 18px; border-bottom:1px solid {BORDER}; margin-bottom:24px;
}}
.main-logo {{
    width:42px; height:42px; border-radius:12px;
    background:linear-gradient(135deg,#238636,#1f6feb);
    display:flex; align-items:center; justify-content:center;
    font-size:22px; flex-shrink:0;
}}
.main-title {{ font-family:'Playfair Display',serif; font-size:1.15rem; color:{TEXT}; }}
.main-sub   {{ font-size:0.7rem; color:{SUBTEXT}; margin-top:2px; }}
.online-badge {{
    margin-left:auto; font-size:0.68rem; padding:4px 12px; border-radius:20px;
    background:rgba(35,134,54,0.15); color:#3fb950;
    border:1px solid rgba(63,185,80,0.3);
}}
.welcome-card {{ text-align:center; padding:40px 16px 28px; }}
.welcome-card h2 {{
    font-family:'Playfair Display',serif; font-size:1.7rem;
    margin-bottom:12px; color:{TEXT};
}}
.welcome-card p {{
    color:{SUBTEXT}; font-size:0.88rem;
    line-height:1.8; max-width:500px; margin:0 auto;
}}
.src-tag {{
    display:inline-block; margin:4px 4px 0 0; font-size:0.68rem;
    padding:3px 10px; border-radius:20px;
    background:rgba(31,111,235,.12); color:#58a6ff;
    border:1px solid rgba(88,166,255,.2);
}}
[data-testid="stChatInput"] textarea {{
    background:{INPUT_BG} !important; border:1.5px solid {INPUT_BOR} !important;
    border-radius:28px !important; color:{TEXT} !important;
    font-family:'Sora',sans-serif !important;
    font-size:0.95rem !important; padding:14px 20px !important;
}}
[data-testid="stChatInput"] textarea:focus {{
    border-color:#58a6ff !important;
    box-shadow:0 0 0 3px rgba(88,166,255,.1) !important;
}}
[data-testid="stChatInput"] button {{
    background:linear-gradient(135deg,#238636,#1f6feb) !important;
    border-radius:50% !important; border:none !important;
}}
[data-testid="stChatMessage"] {{
    background:transparent !important; border:none !important; padding:4px 0 !important;
}}
hr {{ border-color:{BORDER} !important; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CONVERSATION HELPERS
# ---------------------------------------------------------------------------

def new_conversation():
    conv_id = str(int(time.time() * 1000))
    st.session_state.conversations.insert(0, {
        "id": conv_id, "title": "New conversation",
        "messages": [], "timestamp": datetime.now().strftime("%H:%M"),
    })
    st.session_state.active_conv_id = conv_id


def get_active_conv():
    for conv in st.session_state.conversations:
        if conv["id"] == st.session_state.active_conv_id:
            return conv
    return None


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

    if st.button("✦  New conversation", use_container_width=True, key="new_chat_btn"):
        new_conversation()
        st.rerun()

    st.markdown('<div class="sidebar-section-label">Recent chats</div>', unsafe_allow_html=True)

    if not st.session_state.conversations:
        st.markdown('<div style="font-size:0.78rem;padding:8px 0;">No conversations yet.</div>', unsafe_allow_html=True)
    else:
        for conv in st.session_state.conversations:
            if st.button(f"💬  {conv['title']}", key=f"conv_{conv['id']}", use_container_width=True):
                st.session_state.active_conv_id = conv["id"]
                st.rerun()

    st.markdown("---")

    st.markdown('<div class="sidebar-section-label">⚙️ Settings</div>', unsafe_allow_html=True)

    toggled = st.toggle(
        "🌙 Dark mode" if st.session_state.dark_mode else "☀️ Light mode",
        value=st.session_state.dark_mode, key="theme_toggle"
    )
    if toggled != st.session_state.dark_mode:
        st.session_state.dark_mode = toggled
        st.rerun()

    st.markdown('<div style="font-size:0.72rem;margin-top:10px;margin-bottom:4px;">🔆 Contrast</div>', unsafe_allow_html=True)
    nc = st.slider("contrast", 50, 150, st.session_state.contrast, 5,
                   label_visibility="collapsed", key="contrast_slider")
    if nc != st.session_state.contrast:
        st.session_state.contrast = nc
        st.rerun()

    st.markdown('<div style="font-size:0.72rem;margin-top:10px;margin-bottom:4px;">🔡 Font size</div>', unsafe_allow_html=True)
    nf = st.slider("font", 12, 22, st.session_state.font_size, 1,
                   label_visibility="collapsed", key="font_slider")
    if nf != st.session_state.font_size:
        st.session_state.font_size = nf
        st.rerun()

    st.markdown("---")

    st.markdown("""
<div style="background:rgba(255,77,77,0.08);border:1px solid rgba(255,77,77,0.3);
border-radius:10px;padding:12px 14px;margin-bottom:12px;">
<div style="font-size:0.72rem;font-weight:700;color:#ff6b6b;margin-bottom:8px;">
🚨 NEED IMMEDIATE HELP?
</div>
<div style="font-size:0.72rem;line-height:1.9;">
<b>Gender Mainstreaming Directorate</b><br>
📞 +256 (0)414 532 631<br>
📧 gendermainstreaming@mak.ac.ug<br><br>
<b>Dean of Students Office</b><br>
📞 +256 (0)414 531 543<br>
📧 deanofstudents@mak.ac.ug<br><br>
<b>Security / Emergency</b><br>
📞 +256 (0)414 530 903<br><br>
<span style="color:#8b949e;">Mon–Fri · 8 AM – 5 PM<br>
Frank Kalimuzo Building</span>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.65rem;line-height:1.6;">Answers drawn from official Makerere University policy documents.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MAIN CONTENT
# ---------------------------------------------------------------------------

st.markdown("""
<div class="main-header">
  <div class="main-logo">🛡️</div>
  <div>
    <div class="main-title">Safeguarding Companion</div>
    <div class="main-sub">Makerere University · Policy Q&amp;A</div>
  </div>
  <div class="online-badge">● Online</div>
</div>
""", unsafe_allow_html=True)

# Load embedding model + chunks
with st.spinner("Loading policy documents — first run takes a moment…"):
    df, embeddings, emb_model = load_everything()

active_conv = get_active_conv()

# ── Welcome screen ────────────────────────────────────────────────────────────
if active_conv and not active_conv["messages"]:
    st.markdown("""
    <div class="welcome-card">
      <div style="font-size:3.2rem;margin-bottom:14px">🛡️</div>
      <h2>How can I help you today?</h2>
      <p>Ask me anything about Makerere University's safeguarding policies,
      disability rights, sexual harassment procedures, and student protections.
      All answers come from official policy documents.</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(len(SUGGESTIONS))
    for col, suggestion in zip(cols, SUGGESTIONS):
        with col:
            if st.button(suggestion, use_container_width=True, key=f"sug_{suggestion[:20]}"):
                st.session_state.suggested_query = suggestion
                st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
if active_conv:
    for msg in active_conv["messages"]:
        with st.chat_message(msg["role"], avatar="🛡️" if msg["role"] == "assistant" else "🧑"):
            st.markdown(msg["content"], unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------------------------

user_input = st.chat_input("Ask anything about university policies…")

if st.session_state.suggested_query and not user_input:
    user_input = st.session_state.suggested_query
    st.session_state.suggested_query = None

if user_input and active_conv:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    active_conv["messages"].append({"role": "user", "content": user_input})

    if active_conv["title"] == "New conversation":
        active_conv["title"] = user_input[:45] + ("…" if len(user_input) > 45 else "")

    with st.chat_message("assistant", avatar="🛡️"):
        with st.spinner("Searching policy documents and generating answer…"):
            if is_greeting(user_input):
                answer    = GREETING_RESPONSE
                sources   = []
                retrieved = None
                groq_used = None
            else:
                retrieved = retrieve_top_k(user_input, emb_model, embeddings, df)

                # Build raw context
                raw_policy_text = "\n\n".join(
                    f"[Source: {nice_source_name(row['source_document'])}]\n{row['text'].strip()}"
                    for _, row in retrieved.iterrows()
                )

                # Try Groq and track whether it worked
                groq_result = polish_with_groq(user_input, raw_policy_text)
                if groq_result:
                    groq_used = True
                    raw = groq_result
                else:
                    groq_used = False
                    raw = format_chunks_as_bullets(retrieved, query=user_input)

                answer  = format_response(raw)
                sources = (
                    list(retrieved["source_document"].unique())
                    if retrieved is not None and not retrieved.empty else []
                )

        # Debug badge — shows clearly whether Groq ran or not
        if groq_used is True:
            st.success("✅ Groq (Llama 3) generated this answer")
        elif groq_used is False:
            key_present = bool(GROQ_API_KEY)
            st.warning(
                f"⚠️ Groq failed — using fallback. "
                f"API key loaded: {'YES' if key_present else 'NO — check Space secrets'}"
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