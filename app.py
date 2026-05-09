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
import gradio as gr

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)
from gtts import gTTS

nltk.download("punkt", quiet=True)

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

TOP_K = 3

GREETINGS = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
}

GREETING_RESPONSE = """
Hello! Welcome to the Safeguarding Companion.

I can help you understand:
- Safeguarding policies
- Sexual harassment reporting
- Disability rights
- Student protection procedures
- Support services

Ask me a question to begin.
"""

# ---------------------------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------------------------

print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading text generation model...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

generator_model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-base"
)

print("Loading Whisper...")
transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny"
)

# ---------------------------------------------------------------------------
# SAMPLE KNOWLEDGE BASE
# ---------------------------------------------------------------------------

documents = [
    "Students can report sexual harassment through university safeguarding offices.",
    "Students with disabilities have the right to accessible learning environments.",
    "Makerere University provides safeguarding support services for students.",
    "Complaints should be handled confidentially and respectfully.",
    "University safeguarding policies protect students from abuse and discrimination.",
]

doc_embeddings = embedding_model.encode(documents)

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def simplify_answer(text):
    sentences = text.split(".")
    cleaned = []

    for s in sentences:
        s = s.strip()

        if len(s) > 10:
            cleaned.append(f"- {s}")

    return "\n".join(cleaned[:5])


def text_to_speech(text):
    try:
        clean = re.sub(r"[*_#>`-]", "", text)

        tts = gTTS(
            text=clean,
            lang="en",
            slow=False
        )

        tmp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        )

        tts.save(tmp.name)

        return tmp.name

    except Exception as e:
        print("TTS Error:", e)
        return None


def transcribe_audio(audio_path):
    try:
        result = transcriber(audio_path)
        return result["text"]

    except Exception as e:
        print("Transcription Error:", e)
        return ""


def retrieve_context(query):
    query_embedding = embedding_model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        doc_embeddings
    )[0]

    top_indices = similarities.argsort()[-TOP_K:][::-1]

    contexts = []

    for idx in top_indices:
        contexts.append(documents[idx])

    return "\n".join(contexts)


def generate_answer(query, context):
    prompt = f"""
    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = generator_model.generate(
        **inputs,
        max_new_tokens=120
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer


# ---------------------------------------------------------------------------
# MAIN CHAT FUNCTION
# ---------------------------------------------------------------------------

def handle_query(text_query, audio_input, simplified):

    if audio_input is not None:
        text_query = transcribe_audio(audio_input)

    if not text_query:
        return (
            "Please enter a question.",
            None,
            ""
        )

    lower_query = text_query.lower().strip()

    if lower_query in GREETINGS:
        audio_file = text_to_speech(GREETING_RESPONSE)

        return (
            GREETING_RESPONSE,
            audio_file,
            "Greeting detected"
        )

    context = retrieve_context(text_query)

    answer = generate_answer(
        text_query,
        context
    )

    if simplified:
        answer = simplify_answer(answer)

    audio_file = text_to_speech(answer)

    return (
        answer,
        audio_file,
        context
    )

# ---------------------------------------------------------------------------
# GRADIO UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Safeguarding Companion") as demo:

    gr.Markdown(
        """
        # 🛡️ Safeguarding Companion

        Ask questions about:
        - Safeguarding
        - Disability rights
        - Sexual harassment reporting
        - Student protection policies
        """
    )

    with gr.Row():
        text_input = gr.Textbox(
            label="Ask a question",
            placeholder="Type your question here..."
        )

        audio_input = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Voice Input"
        )

    simplified_toggle = gr.Checkbox(
        label="Simplified Language Mode",
        value=False
    )

    submit_btn = gr.Button("Submit")

    text_output = gr.Textbox(
        label="Response",
        lines=10
    )

    audio_output = gr.Audio(
        label="Audio Response"
    )

    sources_output = gr.Textbox(
        label="Retrieved Context",
        lines=5
    )

    submit_btn.click(
        fn=handle_query,
        inputs=[
            text_input,
            audio_input,
            simplified_toggle
        ],
        outputs=[
            text_output,
            audio_output,
            sources_output
        ]
    )

    text_input.submit(
        fn=handle_query,
        inputs=[
            text_input,
            audio_input,
            simplified_toggle
        ],
        outputs=[
            text_output,
            audio_output,
            sources_output
        ]
    )

    gr.HTML(
        """
        <div style="margin-top:2rem; padding:1rem;
                    background:#e8f5e9;
                    border-radius:8px;
                    font-size:0.85rem;
                    color:#1b3a2d;">

        <strong>Accessibility Features:</strong><br>
        - Voice input supported<br>
        - Text-to-speech enabled<br>
        - Keyboard navigation compatible<br>
        - Simplified language mode available

        </div>
        """
    )

# ---------------------------------------------------------------------------
# LAUNCH
# ---------------------------------------------------------------------------

demo.launch()