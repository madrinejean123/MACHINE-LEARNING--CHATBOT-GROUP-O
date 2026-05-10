"""
utils.py — small helper functions used across the app
"""

import re
from config import GREETINGS, STOP_WORDS


def is_greeting(text: str) -> bool:
    cleaned = text.strip().lower().rstrip("!.,?")
    words   = cleaned.split()
    return cleaned in GREETINGS or (len(words) <= 3 and words[0] in GREETINGS)


def nice_source_name(raw: str) -> str:
    return raw.replace(".pdf", "").replace("-", " ").replace("_", " ").strip()


def apply_simplified_language(text: str) -> str:
    replacements = {
        "pursuant to":    "according to",
        "stipulates":     "says",
        "thereof":        "of it",
        "aforementioned": "mentioned above",
        "provisions":     "rules",
        "shall":          "must",
        "whilst":         "while",
        "herein":         "in this document",
        "hereunder":      "below",
        "aforesaid":      "mentioned",
    }
    for formal, plain in replacements.items():
        text = text.replace(formal, plain)
    return text


def fix_list_spacing(text: str) -> str:
    lines  = text.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^\d+\.', stripped) or stripped.startswith("•") or stripped.startswith("-"):
            if result and result[-1] != "":
                result.append("")
            result.append(stripped)
        else:
            result.append(line)
    final, prev_blank = [], False
    for line in result:
        if line == "":
            if not prev_blank:
                final.append(line)
            prev_blank = True
        else:
            final.append(line)
            prev_blank = False
    return "\n".join(final)


def clean_sentence(s: str) -> str:
    s = re.sub(r'^\s*[\(\[]?[a-zA-Z0-9]+[\)\]\.]\s*', '', s)
    s = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def split_into_sentences(text: str) -> list:
    raw  = re.split(r'(?<=[.!?])\s+', text)
    seen, sentences = set(), []
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