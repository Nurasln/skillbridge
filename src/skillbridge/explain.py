"""LLM-generated match explanations.

Owner: AI / LLM Engineer

Interface agreed with the team (do not change without telling everyone):
    explain(cv_text, job_text) -> {"matched": [...], "missing": [...], "summary": "..."}
"""


def explain(cv_text: str, job_text: str) -> dict:
    """Explain why a job fits a CV.

    Placeholder: returns an empty explanation so the UI can be built before
    the real LLM call is ready.
    """
    return {"matched": [], "missing": [], "summary": "Explanation not implemented yet."}
