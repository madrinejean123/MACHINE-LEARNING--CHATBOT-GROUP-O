"""
eSafeRide Safeguarding Companion
RAG-Based Policy Question-Answering System for Makerere University

Retrieves answers from official university policy documents using
semantic search and neural text generation.
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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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

DATA_FOLDER = "data"
CHUNK_CSV = "policy_chunks.csv"
EMBEDDINGS_NPY = "chunk_embeddings.npy"
CHUNK_SIZE_WORDS = 150
OVERLAP_SENTENCES = 1
TOP_K = 5
SIMILARITY_THRESHOLD = 0.25

STOP_WORDS = {
   "the", "and", "for", "are", "that", "this", "with", "how",
   "what", "who", "can", "you", "was", "has", "have", "been",
   "hello", "hi", "hey", "please", "thanks", "thank",
}

GREETINGS = {
   "hi", "hello", "hey", "hie", "howdy",
   "good morning", "good afternoon", "good evening",
   "greetings", "sup", "wassup",
}

GREETING_RESPONSE = (
   "Hello! Welcome to the Safeguarding Companion.\n\n"
   "I am here to help you understand university policies on safeguarding, "
   "disability rights, sexual harassment, and more ", "-", "*")):
       return answer
   sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
   return "\n".join(f"*\-]", "", text)
       tts = gTTS(text=clean, lang="en", slow=False)
       tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
       tts.save(tmp.name)
       return tmp.name
   except Exception as e:
       print(f"Text-to-speech error: {e}")
       return None


def transcribe_audio(audio_path: str) -> str:
   try:
       from transformers import pipeline as hf_pipeline
       transcriber = hf_pipeline(
           "automatic-speech-recognition",
           model="openai/whisper-tiny",
       )
       result = transcriber(audio_path)
       return result["text"].strip()
   except Exception as e:
       print(f"Transcription error: {e}")
       return ""


# ---------------------------------------------------------------------------
# STARTUP click any to try",
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
   <div style="margin-top:2rem; padding:1rem; background:#e8f5e9; border-radius:8px;
               font-size:0.85rem; color:#1b3a2d;">
       <strong>Accessibility features:</strong>
       Full keyboard navigation supported. Screen-reader compatible.
       Use simplified language mode for plainer English responses.
       High contrast mode available for visual accessibility.
       Voice input supported via microphone button.
       All answers read aloud via audio player.
   </div>
   """)


demo.launch()