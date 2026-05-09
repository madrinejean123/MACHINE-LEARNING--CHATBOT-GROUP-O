"""
eSafeRide Safeguarding Companion
RAG-Based Policy Question-Answering System for Makerere University

Pipeline:
 1. Download policy PDFs from GitHub
 2. Extract and chunk text
 3. Generate embeddings (all-MiniLM-L6-v2)
 4. Hybrid retrieval: keyword filter + cosine similarity
 5. Answer generation (flan-t5-base)
 6. Accessibility: voice input, text-to-speech, simplified language
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
from gtts import gTTS

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

DATA_FOLDER      = "data"
CHUNK_CSV        = "policy_chunks.csv"
EMBEDDINGS_NPY   = "chunk_embeddings.npy"
CHUNK_MAX_WORDS  = 150
CHUNK_OVERLAP    = 1
TOP_K            = 5
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
   "disability rights, sexual harassment, and more click any to try",
   )

   submit_btn.click(
       fn=handle_query,
       inputs=[text_input, audio_input, simplified_toggle],
       outputs=[text_output, audio_output, sources_output],
   )

   text_input.submit(
       fn=handle_query,
       inputs=[text_input, audio_input, simplified_toggle],
       outputs=[text_output, audio_output, sources_output],
   )

   gr.HTML("""
   <div style="margin-top:1.5rem; padding:1rem; background:#e8f5e9;
               border-radius:8px; font-size:0.85rem; color:#1b3a2d;">
       <strong>Accessibility features:</strong>
       Full keyboard navigation. Screen-reader compatible.
       Voice input via microphone. All answers read aloud via audio player.
       Simplified language mode for plainer English responses.
   </div>
   """)


demo.launch()