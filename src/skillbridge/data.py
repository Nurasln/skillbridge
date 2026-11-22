"""Dataset loading, CV text extraction and cleaning.

Owner: Data Engineer
"""


def load_jobs() -> list[dict]:
    """Return the job posting pool as [{"id": ..., "text": ...}]."""
    raise NotImplementedError


def extract_cv_text(uploaded_file) -> str:
    """Turn an uploaded PDF or text file into plain CV text."""
    raise NotImplementedError


def clean_text(text: str) -> str:
    """Normalize whitespace, casing and symbols."""
    raise NotImplementedError
