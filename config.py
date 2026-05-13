"""
config.py — all constants and configuration for the Safeguarding Companion
"""

# ---------------------------------------------------------------------------
# GitHub PDF sources
# ---------------------------------------------------------------------------
GITHUB_PDF_URLS = [
    "https://raw.githubusercontent.com/madrinejean123/MACHINE-LEARNING--CHATBOT-GROUP-O/madrine/data/Makerere-Safeguarding-Policy.pdf",
    "https://raw.githubusercontent.com/madrinejean123/MACHINE-LEARNING--CHATBOT-GROUP-O/madrine/data/Policy-and-Regulations-Against-Sexual-Harassment-2018.pdf",
    "https://raw.githubusercontent.com/madrinejean123/MACHINE-LEARNING--CHATBOT-GROUP-O/madrine/data/Makerere-Policy-on-Persons-Living-With-Disabilities.pdf",
    "https://raw.githubusercontent.com/madrinejean123/MACHINE-LEARNING--CHATBOT-GROUP-O/madrine/data/FINAL-REVISED-NATIONAL-POLICY-ON-PWDs-2023.pdf",
    "https://raw.githubusercontent.com/madrinejean123/MACHINE-LEARNING--CHATBOT-GROUP-O/madrine/data/HIV_AIDS_Policy.pdf",
    "https://raw.githubusercontent.com/madrinejean123/MACHINE-LEARNING--CHATBOT-GROUP-O/madrine/data/UTAMU-Disability-Policy.pdf",
]

# ---------------------------------------------------------------------------
# Pre-built chunks on GitHub (downloaded at startup)
# ---------------------------------------------------------------------------
GITHUB_RAW_BASE    = "https://raw.githubusercontent.com/madrinejean123/MACHINE-LEARNING--CHATBOT-GROUP-O/madrine"
CHUNK_CSV_URL      = f"{GITHUB_RAW_BASE}/policy_chunks.csv"
EMBEDDINGS_NPY_URL = f"{GITHUB_RAW_BASE}/chunk_embeddings.npy"

# ---------------------------------------------------------------------------
# RAG settings
# ---------------------------------------------------------------------------
DATA_FOLDER          = "data"
CHUNK_MAX_WORDS      = 250
CHUNK_OVERLAP        = 2
TOP_K                = 7
SIMILARITY_THRESHOLD = 0.20

# ---------------------------------------------------------------------------
# NLP helpers
# ---------------------------------------------------------------------------
STOP_WORDS = {
    "the","and","for","are","that","this","with","how",
    "what","who","can","you","was","has","have","been",
    "hello","hi","hey","please","thanks","thank",
}

GREETINGS = {
    "hi","hello","hey","hie","howdy",
    "good morning","good afternoon","good evening","greetings",
}

GREETING_RESPONSE = (
    "Hello! Welcome to the Safeguarding Companion.\n\n"
    "I'm here to help you understand university policies on safeguarding, "
    "disability rights, sexual harassment etc.\n\n"
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
# Groq API (key loaded from environment — set in HF Space secrets)
# ---------------------------------------------------------------------------
import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.1-8b-instant"

# ---------------------------------------------------------------------------
# Phrases to skip when formatting retrieved chunks
# ---------------------------------------------------------------------------
SKIP_PHRASES = [
    "there is no documented","current evidence","principles underpinning",
    "it should be noted","as quasi-judicial","no current evidence",
    "standard procedures concerning","enjoy relative flexibility",
]