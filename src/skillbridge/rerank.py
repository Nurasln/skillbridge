"""Cross-encoder reranker.

Owner: AI / LLM Engineer (Asiye Nur Aslan)

Interface agreed with the team (do not change without telling everyone):
    rerank(cv_text, jobs) -> the same jobs, each with a "score", best first.

How it works:
- A bi-encoder (retrieval.py) embeds the CV and each job separately, which is
  fast but coarse. A cross-encoder reads the CV and ONE job together in a single
  pass and outputs a relevance score, so it can compare them word by word.
  It is too slow for all jobs, so it only reorders the top 20 from hybrid_search.
- Model: set by SKILLBRIDGE_RERANK_MODEL (default below, chosen on the dev split).
- Uses the GPU when one is available, otherwise the CPU.
- If the model cannot be loaded (offline demo machine with no cache, typo in
  the model name) we raise NotImplementedError, so the pipeline keeps the
  hybrid order instead of crashing.
- Limitation: CV + job share MAX_LENGTH tokens, so with long CVs much of the
  job text is cut. The dev-split numbers were measured with this setting.
"""

from __future__ import annotations

import os

MODEL_NAME = os.getenv("SKILLBRIDGE_RERANK_MODEL", "BAAI/bge-reranker-base")
MAX_LENGTH = int(os.getenv("SKILLBRIDGE_RERANK_MAX_LENGTH", "512"))  # tokens for CV + job together

_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import CrossEncoder
            _model = CrossEncoder(MODEL_NAME, max_length=MAX_LENGTH)
        except Exception as e:  # offline, no cache, wrong name: keep the retrieval order
            raise NotImplementedError(f"reranker {MODEL_NAME} could not be loaded: {type(e).__name__}: {e}") from e
    return _model


def rerank(cv_text: str, jobs: list[dict]) -> list[dict]:
    """jobs: [{"id": ..., "text": ...}] -> same list with "score" added, sorted best first.

    The incoming "score" (from retrieval) is replaced by the cross-encoder score.
    """
    if not jobs:
        return []
    pairs = [(cv_text, job["text"]) for job in jobs]
    scores = _get_model().predict(pairs, batch_size=8, show_progress_bar=False)
    scored = [{**job, "score": float(s)} for job, s in zip(jobs, scores)]
    return sorted(scored, key=lambda job: job["score"], reverse=True)
