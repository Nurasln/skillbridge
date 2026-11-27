"""TF-IDF and BM25 baseline rankers.

Owner: Research & Literature

Both rankers treat the CV as the query and the job texts as the documents:
- tfidf_rank: TF-IDF vectors + cosine similarity (scikit-learn)
- bm25_rank:  Okapi BM25 (rank-bm25)
"""
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _tokens(text: str) -> list[str]:
    """Lowercase words (keeps c++, c#, .net), without English stop words."""
    words = re.findall(r"[a-z0-9][a-z0-9+#.]*", (text or "").lower())
    return [w.strip(".") for w in words if w.strip(".") and w not in ENGLISH_STOP_WORDS]


def _top_k(scores, jobs: list[dict], k: int) -> list[dict]:
    """Best k jobs first, each a copy of the job dict plus a float "score"."""
    order = np.argsort(-scores, kind="stable")[:k]
    return [{**jobs[i], "score": float(scores[i])} for i in order]


def tfidf_rank(cv_text: str, jobs: list[dict], k: int = 20) -> list[dict]:
    """Return the top-k jobs by TF-IDF cosine similarity, each with a "score"."""
    if not jobs or k <= 0:
        return []
    if not any(_tokens(j["text"]) for j in jobs):  # every job text is empty
        return _top_k(np.zeros(len(jobs)), jobs, k)
    vectorizer = TfidfVectorizer(tokenizer=_tokens, lowercase=False, token_pattern=None)
    job_vectors = vectorizer.fit_transform([j["text"] for j in jobs])
    cv_vector = vectorizer.transform([cv_text])
    scores = cosine_similarity(cv_vector, job_vectors).ravel()
    return _top_k(scores, jobs, k)


def bm25_rank(cv_text: str, jobs: list[dict], k: int = 20) -> list[dict]:
    """Return the top-k jobs by BM25 score, each with a "score"."""
    if not jobs or k <= 0:
        return []
    corpus = [_tokens(j["text"]) for j in jobs]
    if not any(corpus):  # every job text is empty
        return _top_k(np.zeros(len(jobs)), jobs, k)
    scores = np.array(BM25Okapi(corpus).get_scores(_tokens(cv_text)))
    return _top_k(scores, jobs, k)
