"""
models.py — loads embedding model and downloads pre-built chunks from GitHub
"""

import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from config import CHUNK_CSV_URL, EMBEDDINGS_NPY_URL

_CACHE_DIR  = "/data" if os.path.isdir("/data") else "."
_CHUNK_PATH = os.path.join(_CACHE_DIR, "policy_chunks.csv")
_EMBED_PATH = os.path.join(_CACHE_DIR, "chunk_embeddings.npy")


def _download_file(url: str, local_path: str) -> bool:
    if os.path.exists(local_path):
        os.remove(local_path)
    print(f"Downloading {os.path.basename(local_path)} ...")
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
        print(f"  Saved ({len(r.content) // 1024} KB)")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


@st.cache_resource(show_spinner=False)
def load_everything():
    """
    Downloads pre-built chunks + embeddings from GitHub,
    then loads the SentenceTransformer embedding model.
    Returns (df, embeddings, emb_model).
    """
    csv_ok = _download_file(CHUNK_CSV_URL,      _CHUNK_PATH)
    npy_ok = _download_file(EMBEDDINGS_NPY_URL, _EMBED_PATH)

    if not (csv_ok and npy_ok):
        raise RuntimeError(
            "Could not load pre-built chunks from GitHub. "
            "Make sure policy_chunks.csv and chunk_embeddings.npy "
            "are committed to the madrine branch."
        )

    df         = pd.read_csv(_CHUNK_PATH)
    embeddings = np.load(_EMBED_PATH)
    print(f"Loaded {len(df)} chunks, embeddings shape {embeddings.shape}")

    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    return df, embeddings, emb_model