"""LLM-generated match explanations.

Owner: AI / LLM Engineer (Asiye Nur Aslan)

Interface agreed with the team (do not change without telling everyone):
    explain(cv_text, job_text) -> {"matched": [...], "missing": [...], "summary": "..."}

How it works:
- One Gemini call per (CV, job) pair. The model must answer in a fixed JSON
  shape (response_schema), so we never parse free text.
- The prompt allows only facts written in the two texts; temperature 0 keeps
  answers stable between runs.
- Needs GEMINI_API_KEY in .env (see .env.example). Without a key the function
  raises NotImplementedError, so the pipeline falls back to an empty
  explanation instead of crashing for teammates who have no key.
- Model: set by SKILLBRIDGE_LLM_MODEL (default below).
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel

MODEL_NAME = os.getenv("SKILLBRIDGE_LLM_MODEL", "gemini-2.5-flash")
MAX_CV_CHARS = 6000    # long texts are cut to keep cost and latency low
MAX_JOB_CHARS = 4000
MAX_ITEMS = 6

PROMPT = """You help a student see how well their CV fits a job posting.
Use only facts that are written in the CV and the job posting below. Do not guess or invent.

Return:
- matched: up to {n} requirements or skills from the job posting that the CV clearly shows (short phrases).
- missing: up to {n} requirements from the job posting that the CV does not show (short phrases).
- summary: one or two plain sentences on the overall fit.

CV:
<<<
{cv}
>>>

JOB POSTING:
<<<
{job}
>>>"""


class _Explanation(BaseModel):
    matched: list[str]
    missing: list[str]
    summary: str


_client = None


def _get_client():
    global _client
    if _client is None:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        if not os.getenv("GEMINI_API_KEY"):
            raise NotImplementedError("GEMINI_API_KEY is not set (copy .env.example to .env)")
        from google import genai
        _client = genai.Client()  # reads GEMINI_API_KEY from the environment
    return _client


def explain(cv_text: str, job_text: str) -> dict:
    """Explain why a job fits a CV: matched skills, missing skills, short summary."""
    client = _get_client()
    prompt = PROMPT.format(n=MAX_ITEMS, cv=cv_text[:MAX_CV_CHARS], job=job_text[:MAX_JOB_CHARS])
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": _Explanation,
            "temperature": 0,
        },
    )
    result: _Explanation = response.parsed
    return {
        "matched": result.matched[:MAX_ITEMS],
        "missing": result.missing[:MAX_ITEMS],
        "summary": result.summary.strip(),
    }
