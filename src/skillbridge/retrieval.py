"""Sentence embeddings, FAISS index, dense and hybrid search.

Owner: Model / Retrieval Engineer (Recep Kamil Fırat)

Interface (same shape as baselines.py):
    build_index(jobs)                         -> index
    dense_search(cv_text, index, k)           -> top-k jobs, each with "score"
    hybrid_search(cv_text, jobs, index, k)    -> top-k jobs, each with "score"

Design:
- Model: sentence-transformers/all-MiniLM-L6-v2 (open source, 384-dim, runs on CPU).
  Override with the SKILLBRIDGE_EMBED_MODEL environment variable.
- Embeddings are L2-normalized, so FAISS inner product == cosine similarity.
- CVs and job posts are long; the model reads at most 256 tokens. Each text is
  split into overlapping word windows, embedded, and mean-pooled into one vector.
- Hybrid = Reciprocal Rank Fusion (Cormack et al., 2009) of BM25 and dense ranks.
  RRF uses only ranks, so the two very different score scales need no tuning.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

MODEL_NAME = os.getenv("SKILLBRIDGE_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_WORDS = 180      # ~240 tokens, under the model's 256-token limit
CHUNK_OVERLAP = 40
RRF_K = 60             # standard constant from Cormack et al. (2009)

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _chunks(text: str) -> list[str]:
    words = text.split()
    if len(words) <= CHUNK_WORDS:
        return [" ".join(words) or " "]
    step = CHUNK_WORDS - CHUNK_OVERLAP
    return [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, len(words) - CHUNK_OVERLAP, step)]


def embed(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """One L2-normalized vector per text (mean of its chunk vectors)."""
    pieces, owner = [], []
    for i, t in enumerate(texts):
        for c in _chunks(t):
            pieces.append(c)
            owner.append(i)
    vecs = _get_model().encode(pieces, batch_size=batch_size, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=len(pieces) > 500)
    out = np.zeros((len(texts), vecs.shape[1]), dtype="float32")
    np.add.at(out, np.array(owner), vecs)
    out /= np.linalg.norm(out, axis=1, keepdims=True).clip(min=1e-12)
    return out


@dataclass
class JobIndex:
    jobs: list[dict]
    faiss_index: object
    bm25: object = field(default=None)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9+#.]+", text.lower())


def build_index(jobs: list[dict]) -> JobIndex:
    """Embed every job posting once and return a searchable index."""
    import faiss
    from rank_bm25 import BM25Okapi

    vecs = embed([j["text"] for j in jobs])
    index = faiss.IndexFlatIP(vecs.shape[1])   # exact search; pool is ~1k jobs
    index.add(vecs)
    return JobIndex(jobs=jobs, faiss_index=index, bm25=BM25Okapi([_tokens(j["text"]) for j in jobs]))


def dense_search(cv_text: str, index: JobIndex, k: int = 20) -> list[dict]:
    """Return the top-k jobs by embedding similarity, each with a "score"."""
    q = embed([cv_text])
    k = min(k, len(index.jobs))
    scores, ids = index.faiss_index.search(q, k)
    return [{**index.jobs[i], "score": float(s)} for s, i in zip(scores[0], ids[0]) if i >= 0]


def hybrid_search(cv_text: str, jobs: list[dict], index: JobIndex, k: int = 20) -> list[dict]:
    """Combine BM25 and dense results into one top-k list (Reciprocal Rank Fusion)."""
    n = len(jobs)
    depth = min(n, max(k * 5, 100))
    dense = dense_search(cv_text, index, depth)
    bm25_scores = index.bm25.get_scores(_tokens(cv_text))
    bm25_order = np.argsort(-bm25_scores)[:depth]

    fused: dict[int, float] = {}
    id_pos = {j["id"]: p for p, j in enumerate(jobs)}
    for rank, job in enumerate(dense, start=1):
        p = id_pos[job["id"]]
        fused[p] = fused.get(p, 0.0) + 1.0 / (RRF_K + rank)
    for rank, p in enumerate(bm25_order, start=1):
        fused[int(p)] = fused.get(int(p), 0.0) + 1.0 / (RRF_K + rank)

    best = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [{**jobs[p], "score": s} for p, s in best]


if __name__ == "__main__":
    demo = [
        {"id": "j1", "text": "Junior Python developer. Pandas, SQL, REST APIs."},
        {"id": "j2", "text": "Data analyst intern. Excel, SQL, dashboards, Power BI."},
        {"id": "j3", "text": "Frontend developer. React, TypeScript, CSS."},
        {"id": "j4", "text": "Registered nurse for an intensive care unit."},
    ]
    idx = build_index(demo)
    cv = "Computer engineering student. Built web scrapers in Python, analysed data with pandas, wrote SQL queries."
    for name, res in [("dense", dense_search(cv, idx, 4)), ("hybrid", hybrid_search(cv, demo, idx, 4))]:
        print(name, [(r["id"], round(r["score"], 3)) for r in res])
