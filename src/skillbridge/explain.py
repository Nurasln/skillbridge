"""LLM-generated match explanations.

Owner: AI / LLM Engineer

Interface agreed with the team (do not change without telling everyone):
    explain(cv_text, job_text) -> {"matched": [...], "missing": [...], "summary": "..."}
"""


def explain(cv_text: str, job_text: str) -> dict:
    """Explain why a job fits a CV.

    Not implemented yet: raises NotImplementedError so the pipeline shows an
    empty explanation and reports the fallback (AGENTS.md: never return fake numbers).
    """
    raise NotImplementedError
