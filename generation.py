"""
generation.py — answer generation: Groq polish + fallback bullet formatter
"""

import re
import requests
from config import GROQ_API_KEY, GROQ_API_URL, GROQ_MODEL, SKIP_PHRASES
from utils import (
    nice_source_name, split_into_sentences,
    apply_simplified_language, fix_list_spacing,
)

ACTION_WORDS = {
    "report", "lodge", "file", "contact", "collect", "document", "record",
    "seek", "notify", "submit", "communicate", "keep", "note", "familiarize",
    "request", "support", "access", "entitled", "rights", "must", "should",
    "procedure", "steps", "committee", "directorate", "officer", "complaint",
    "evidence", "witness", "can", "will", "shall", "ensure", "provide", "receive",
}

NO_ANSWER_MSG = (
    "I found related policy sections but could not extract clear steps.\n\n"
    "Please contact the **Directorate of Gender Mainstreaming** directly.\n"
    "📞 +256 (0)414 532 631 · 📧 gendermainstreaming@mak.ac.ug\n\n"
    "**Office of the Dean of Students**\n"
    "📞 +256 (0)414 531 543 · 📧 deanofstudents@mak.ac.ug"
)

NOT_FOUND_MSG = (
    "I could not find specific information about that in the policy documents. "
    "Please try rephrasing your question.\n\n"
    "**Contact directly:**\n\n"
    "**Directorate of Gender Mainstreaming**\n"
    "📞 +256 (0)414 532 631 · 📧 gendermainstreaming@mak.ac.ug\n\n"
    "**Office of the Dean of Students**\n"
    "📞 +256 (0)414 531 543 · 📧 deanofstudents@mak.ac.ug"
)


# ---------------------------------------------------------------------------
# Groq polish
# ---------------------------------------------------------------------------

def polish_with_groq(question: str, raw_policy_text: str) -> str | None:
    """
    Send retrieved policy text to Groq (Llama 3) to rewrite it cleanly.
    Returns polished answer string, or None if Groq is unavailable.
    """
    if not GROQ_API_KEY:
        print("No GROQ_API_KEY — using fallback formatter.")
        return None

    system_prompt = (
        "You are a helpful university safeguarding assistant for Makerere University students. "
        "Explain university policies in plain, simple English.\n"
        "Rules:\n"
        "1. Fix OCR merged words (e.g. 'mustbe' -> 'must be').\n"
        "2. Use numbered list (1. 2. 3.) for procedures.\n"
        "3. Use bullet points (•) for general info.\n"
        "4. Use ONLY the provided policy text — do not add new information.\n"
        "5. Remove repeated sentences.\n"
        "6. Keep it short and friendly.\n"
        "7. End with: 'For more help, contact the Directorate of Gender Mainstreaming at "
        "gendermainstreaming@mak.ac.ug or call +256 (0)414 532 631.'"
    )

    user_prompt = (
        f"A student asked: \"{question}\"\n\n"
        f"Raw policy text:\n---\n{raw_policy_text[:3000]}\n---\n\n"
        f"Write a clear, simple answer:"
    )

    try:
        r = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":    GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                "max_tokens":  600,
                "temperature": 0.3,
            },
            timeout=30,
        )
        r.raise_for_status()
        answer = r.json()["choices"][0]["message"]["content"].strip()
        return answer if answer else None
    except Exception as e:
        print(f"Groq polish failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Fallback bullet formatter (used when Groq is not available)
# ---------------------------------------------------------------------------

def format_chunks_as_bullets(retrieved, query: str = "") -> str:
    grouped = {}
    for _, row in retrieved.iterrows():
        src = nice_source_name(row["source_document"])
        grouped.setdefault(src, []).append(row["text"].strip())

    parts, total_bullets = [], 0

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
        return NO_ANSWER_MSG

    header = f"Here is what the policies say about **{query.strip()}**:\n\n" if query else ""
    return header + "\n".join(parts)


# ---------------------------------------------------------------------------
# Main entry point called by app.py
# ---------------------------------------------------------------------------

def generate_answer(question: str, retrieved) -> str:
    """
    1. Try Groq (Llama 3) for a polished, clean answer.
    2. Fall back to bullet formatter if Groq fails or is not configured.
    """
    if retrieved is None or retrieved.empty:
        return NOT_FOUND_MSG

    raw_policy_text = "\n\n".join(
        f"[Source: {nice_source_name(row['source_document'])}]\n{row['text'].strip()}"
        for _, row in retrieved.iterrows()
    )

    groq_result = polish_with_groq(question, raw_policy_text)
    if groq_result:
        return groq_result

    return format_chunks_as_bullets(retrieved, query=question)


def format_response(answer: str) -> str:
    """Apply simplified language and fix list spacing."""
    answer = apply_simplified_language(answer)
    answer = fix_list_spacing(answer)
    return answer