"""TF-IDF and BM25 baseline rankers.

Owner: Research & Literature
"""


def tfidf_rank(cv_text: str, jobs: list[dict], k: int = 20) -> list[dict]:
    """Return the top-k jobs by TF-IDF cosine similarity, each with a "score"."""
    raise NotImplementedError


def bm25_rank(cv_text: str, jobs: list[dict], k: int = 20) -> list[dict]:
    """Return the top-k jobs by BM25 score, each with a "score"."""
    raise NotImplementedError
