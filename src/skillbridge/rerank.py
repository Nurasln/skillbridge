"""Cross-encoder reranker.

Owner: AI / LLM Engineer

Interface agreed with the team (do not change without telling everyone):
    rerank(cv_text, jobs) -> the same jobs, each with a "score", best first.
"""


def rerank(cv_text: str, jobs: list[dict]) -> list[dict]:
    """jobs: [{"id": ..., "text": ...}] -> same list with "score" added, sorted.

    Placeholder: keeps the input order so the UI can be built before the
    real cross-encoder is ready.
    """
    return [{**job, "score": 0.0} for job in jobs]
