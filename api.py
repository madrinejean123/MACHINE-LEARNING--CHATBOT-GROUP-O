"""
api.py — FastAPI wrapper around the Safeguarding Companion RAG pipeline,
served from the SAME Hugging Face Space as the Streamlit UI (see nginx.conf
and start.sh — nginx routes /ask, /health, /suggestions here and everything
else to Streamlit on the same public URL).

Reuses retrieval.py / generation.py / utils.py / config.py unchanged from
the existing Streamlit app. Only the model loading from models.py is
reimplemented here without Streamlit's @st.cache_resource, since this
process has no Streamlit runtime — it just loads once at startup instead.
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
import requests as http_requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from config import CHUNK_CSV_URL, EMBEDDINGS_NPY_URL, GREETING_RESPONSE, SUGGESTIONS
from generation import format_response, generate_answer
from retrieval import retrieve_top_k
from utils import is_greeting, nice_source_name

# Optional shared-secret so random callers can't run up your Groq bill.
# Leave SAFEGUARDING_API_KEY unset to keep the API open.
API_KEY = os.environ.get("SAFEGUARDING_API_KEY", "")

_CACHE_DIR = "/data" if os.path.isdir("/data") else "."
_CHUNK_PATH = os.path.join(_CACHE_DIR, "policy_chunks.csv")
_EMBED_PATH = os.path.join(_CACHE_DIR, "chunk_embeddings.npy")

_state: dict = {}


def _download_file(url: str, local_path: str) -> None:
    r = http_requests.get(url, timeout=120)
    r.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(r.content)


def _load_everything():
    _download_file(CHUNK_CSV_URL, _CHUNK_PATH)
    _download_file(EMBEDDINGS_NPY_URL, _EMBED_PATH)
    df = pd.read_csv(_CHUNK_PATH)
    embeddings = np.load(_EMBED_PATH)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return df, embeddings, model


@asynccontextmanager
async def lifespan(app: FastAPI):
    df, embeddings, model = _load_everything()
    _state["df"] = df
    _state["embeddings"] = embeddings
    _state["model"] = model
    print(f"[api.py] Loaded {len(df)} policy chunks.")
    yield
    _state.clear()


app = FastAPI(title="Safeguarding Companion API", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    is_greeting: bool


def _check_api_key(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
def health():
    return {"status": "ok", "chunks_loaded": len(_state.get("df", []))}


@app.get("/suggestions")
def suggestions():
    return {"suggestions": SUGGESTIONS}


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, x_api_key: Optional[str] = Header(default=None)):
    _check_api_key(x_api_key)

    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    if is_greeting(question):
        return AskResponse(answer=GREETING_RESPONSE, sources=[], is_greeting=True)

    retrieved = retrieve_top_k(question, _state["model"], _state["embeddings"], _state["df"])
    answer = format_response(generate_answer(question, retrieved))

    sources: list[str] = []
    if retrieved is not None and not retrieved.empty:
        for src in retrieved["source_document"]:
            name = nice_source_name(src)
            if name not in sources:
                sources.append(name)

    return AskResponse(answer=answer, sources=sources, is_greeting=False)
