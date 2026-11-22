"""Evaluation: Recall@k, MRR, latency and cost on the gold test set.

Owner: Evaluation & Testing
"""


def recall_at_k(ranked_ids: list, relevant_ids: set, k: int) -> float:
    """1.0 if any relevant job is in the top k, else 0.0."""
    return float(any(job_id in relevant_ids for job_id in ranked_ids[:k]))


def reciprocal_rank(ranked_ids: list, relevant_ids: set) -> float:
    """1 / rank of the first relevant job, 0.0 if none is found."""
    for rank, job_id in enumerate(ranked_ids, start=1):
        if job_id in relevant_ids:
            return 1.0 / rank
    return 0.0
