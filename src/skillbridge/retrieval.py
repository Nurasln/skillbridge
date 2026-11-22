"""Sentence embeddings, FAISS index, dense and hybrid search.

Owner: Model / Retrieval Engineer
"""


def build_index(jobs: list[dict]):
    """Embed every job posting once and return a searchable index."""
    raise NotImplementedError


def dense_search(cv_text: str, index, k: int = 20) -> list[dict]:
    """Return the top-k jobs by embedding similarity, each with a "score"."""
    raise NotImplementedError


def hybrid_search(cv_text: str, jobs: list[dict], index, k: int = 20) -> list[dict]:
    """Combine BM25 and dense results into one top-k list."""
    raise NotImplementedError
