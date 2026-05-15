
# Safeguarding Companion
# Link to demo #:
https://www.google.com/url?q=https%3A%2F%2Fmadrine-safeguarding-companion.hf.space
# Open the notebook here:

https://colab.research.google.com/drive/1epT64vaRd4VwFXo7ObaCusNkx3KCr5l4?usp=sharing

A RAG-based chatbot for inclusive university safeguarding support at Makerere University.

## What it does

Answers questions about university safeguarding policies, disability rights, sexual harassment reporting, and complaint procedures — grounded in official policy documents and written in plain English.

## Features

- Retrieval-Augmented Generation (RAG) pipeline
- Semantic search over 6 official Makerere University policy documents
- Voice input (speech-to-text via Whisper-tiny)
- Text-to-speech output
- Simplified language mode for cognitive accessibility
- High contrast mode for visual accessibility
- Full keyboard navigation and screen-reader compatible layout

## Policy Documents

- Makerere University Safeguarding Policy (2024)
- Policy and Regulations Against Sexual Harassment (2018)
- Policy on Students with Disabilities
- National Policy on Persons with Disabilities (2023)
- HIV/AIDS Policy
- UTAMU Disability Policy

## Architecture

1. PDF documents downloaded and chunked (150 words, 1-sentence overlap)
2. Chunks embedded using all-MiniLM-L6-v2
3. Hybrid retrieval: keyword filter + cosine similarity
4. Answer generation using google/flan-t5-large
5. Accessibility layer: TTS, voice input, simplified language
