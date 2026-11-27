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
- Model: set by SKILLBRIDGE_LLM_MODEL. Default gemini-3.1-flash-lite: on
  2026-11-26 gemini-2.5-flash returned 404 for new keys and the larger flash
  models returned 503 (high demand); flash-lite answered in ~2 s.
- If the Gemini call fails for any reason (503 busy, quota, dropped network,
  timeout) we raise NotImplementedError too, so the pipeline shows an empty
  explanation and lists the fallback instead of the whole app crashing during
  a demo. Network errors come from the HTTP library (httpx2.ConnectError,
  not genai's APIError), so the call is guarded with a broad except.

Token usage (for the cost per query in the report):
- Every successful Gemini call records its token counts (from the API's own
  usage_metadata) and its duration. explain() itself returns exactly the same
  dict as before; the counts are read separately:
      reset_usage()   # before a measurement run
      ... call explain() / Pipeline().match() ...
      get_usage()     # -> {"calls", "prompt_tokens", "output_tokens", ..., "per_call": {...}}
- Cost = tokens x the price on Google's pricing page; no price is stored here.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

from pydantic import BaseModel

log = logging.getLogger(__name__)

MODEL_NAME = os.getenv("SKILLBRIDGE_LLM_MODEL", "gemini-3.1-flash-lite")
TIMEOUT_MS = 20_000    # per request; a hung call must not freeze the demo
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
_warned = False  # log the missing-key warning only once


def _get_client():
    global _client, _warned
    if _client is None:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        if not os.getenv("GEMINI_API_KEY"):
            if not _warned:
                _warned = True
                log.warning("GEMINI_API_KEY is not set: explanations stay empty (copy .env.example to .env)")
            raise NotImplementedError("GEMINI_API_KEY is not set (copy .env.example to .env)")
        from google import genai
        # reads GEMINI_API_KEY from the environment
        _client = genai.Client(http_options={"timeout": TIMEOUT_MS})
    return _client


def explain(cv_text: str, job_text: str) -> dict:
    """Explain why a job fits a CV: matched skills, missing skills, short summary."""
    from google.genai import errors

    client = _get_client()
    prompt = PROMPT.format(n=MAX_ITEMS, cv=cv_text[:MAX_CV_CHARS], job=job_text[:MAX_JOB_CHARS])
    start = time.perf_counter()
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": _Explanation,
                "temperature": 0,
            },
        )
    except errors.APIError as e:
        raise NotImplementedError(f"Gemini call failed ({e.code}): {e.message}") from e
    except Exception as e:  # network errors, timeouts, bad replies: never crash the app
        raise NotImplementedError(f"Gemini request failed: {type(e).__name__}: {e}") from e
    _record_usage(response, time.perf_counter() - start)  # tokens are spent even if parsing fails
    result: _Explanation | None = response.parsed
    if result is None:
        raise NotImplementedError("Gemini returned no parsable JSON")
    return {
        "matched": result.matched[:MAX_ITEMS],
        "missing": result.missing[:MAX_ITEMS],
        "summary": result.summary.strip(),
    }


# --------------------------------------------------------------------------- #
# Token usage
# --------------------------------------------------------------------------- #

_TOKEN_FIELDS = {          # our name -> usage_metadata field from the Gemini API
    "prompt_tokens": "prompt_token_count",
    "output_tokens": "candidates_token_count",
    "thinking_tokens": "thoughts_token_count",
    "total_tokens": "total_token_count",
}

_usage: list[dict] = []
_usage_lock = threading.Lock()   # safe if the pipeline ever explains jobs in parallel


def _record_usage(response, seconds: float) -> None:
    meta = getattr(response, "usage_metadata", None)
    row = {name: int(getattr(meta, field, None) or 0) for name, field in _TOKEN_FIELDS.items()}
    row["seconds"] = seconds
    with _usage_lock:
        _usage.append(row)
    log.info("gemini %s: %d prompt + %d output tokens, %.2f s",
             MODEL_NAME, row["prompt_tokens"], row["output_tokens"], seconds)


def reset_usage() -> None:
    """Forget all recorded calls (call this before a measurement run)."""
    with _usage_lock:
        _usage.clear()


def get_usage() -> dict:
    """Totals and per-call means of all successful Gemini calls since the last reset."""
    with _usage_lock:
        rows = list(_usage)
    n = len(rows)
    totals = {k: sum(r[k] for r in rows) for k in (*_TOKEN_FIELDS, "seconds")}
    return {
        "model": MODEL_NAME,
        "calls": n,
        **totals,
        "per_call": {k: (v / n if n else 0.0) for k, v in totals.items()},
    }
