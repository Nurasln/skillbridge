"""Cross-encoder reranker.

Owner: AI / LLM Engineer

Interface agreed with the team (do not change without telling everyone):
    rerank(cv_text, jobs) -> the same jobs, each with a "score", best first.
"""


def rerank(cv_text: str, jobs: list[dict]) -> list[dict]:
    """jobs: [{"id": ..., "text": ...}] -> same list with "score" added, sorted.

    Not implemented yet: raises NotImplementedError so the pipeline keeps the
    retrieval order and reports the fallback (AGENTS.md: never return fake numbers).
    """
    raise NotImplementedError
