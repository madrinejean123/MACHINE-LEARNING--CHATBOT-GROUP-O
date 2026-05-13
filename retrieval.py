"""
retrieval.py — query expansion, keyword filtering and semantic retrieval
"""

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import STOP_WORDS, TOP_K, SIMILARITY_THRESHOLD

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

ACTION_WORDS = [
    "report", "complain", "lodge", "file", "contact", "procedure",
    "steps", "committee", "officer", "directorate", "submit", "notify",
    "support", "rights", "entitled", "must", "shall", "access",
]


def expand_query(query: str) -> str:
    q_lower = query.lower()
    extras  = set()
    for trigger, expansion in QUERY_EXPANSIONS.items():
        if trigger in q_lower:
            extras.update(expansion.split())
    return query + " " + " ".join(extras) if extras else query


def keyword_filter(df, query: str):
    keywords = [
        w.lower() for w in re.findall(r"\b\w+\b", query)
        if len(w) > 2 and w.lower() not in STOP_WORDS
    ]
    if not keywords:
        return df
    mask     = df["text"].apply(lambda x: any(k in str(x).lower() for k in keywords))
    filtered = df[mask]
    return filtered if len(filtered) > 0 else df


def retrieve_top_k(query: str, model, embeddings, df, k=TOP_K, threshold=SIMILARITY_THRESHOLD):
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
    results["action_boost"]     = results["text"].apply(
        lambda t: sum(1 for w in ACTION_WORDS if w in str(t).lower())
    )
    results = results.sort_values(
        by=["action_boost", "similarity_score"], ascending=[False, False]
    ).drop(columns=["action_boost"])

    return results