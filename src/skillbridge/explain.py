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
- Server errors (5xx, e.g. 504 deadline) and network errors/timeouts are
  retried once (20 s timeout per attempt); 4xx errors (key, quota, model) are not.
- If the Gemini call still fails (503 busy, quota, dropped network,
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
MAX_ATTEMPTS = 2       # one retry on 5xx / timeouts: ~8% of calls hit 504 on 2026-11-27
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
    for attempt in range(1, MAX_ATTEMPTS + 1):
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
            break
        except Exception as e:  # network errors, timeouts, API errors: never crash the app
            if attempt < MAX_ATTEMPTS and _is_retryable(e, errors):
                log.info("gemini attempt %d failed (%s); retrying once", attempt, _describe(e, errors))
                continue
            raise NotImplementedError(f"Gemini call failed: {_describe(e, errors)}") from e
    # tokens are spent even if parsing fails; seconds include a retry, i.e. what the user waited
    _record_usage(response, time.perf_counter() - start, attempt)
    result: _Explanation | None = response.parsed
    if result is None:
        raise NotImplementedError("Gemini returned no parsable JSON")
    return {
        "matched": result.matched[:MAX_ITEMS],
        "missing": result.missing[:MAX_ITEMS],
        "summary": result.summary.strip(),
    }


def _is_retryable(e: Exception, errors) -> bool:
    """Server-side errors (5xx, e.g. 503 busy, 504 deadline) and network/timeout errors.

    4xx errors (bad key, quota 429, unknown model 404) fail the same way again,
    so they are not retried.
    """
    if isinstance(e, errors.APIError):
        return (e.code or 0) >= 500
    return True   # httpx2 timeouts / dropped connections are not APIError


def _describe(e: Exception, errors) -> str:
    if isinstance(e, errors.APIError):
        return f"({e.code}) {e.message}"
    return f"{type(e).__name__}: {e}"


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


def _record_usage(response, seconds: float, attempts: int = 1) -> None:
    meta = getattr(response, "usage_metadata", None)
    row = {"seconds": seconds, "attempts": attempts, "has_usage": meta is not None}
    row.update({name: int(getattr(meta, field, None) or 0) for name, field in _TOKEN_FIELDS.items()})
    with _usage_lock:
        _usage.append(row)
    if meta is None:
        log.info("gemini %s: no usage_metadata in reply, %.2f s", MODEL_NAME, seconds)
    else:
        log.info("gemini %s: %d prompt + %d output tokens, %.2f s",
                 MODEL_NAME, row["prompt_tokens"], row["output_tokens"], seconds)


def reset_usage() -> None:
    """Forget all recorded calls (call this before a measurement run)."""
    with _usage_lock:
        _usage.clear()


def get_usage() -> dict:
    """Totals and per-call means of all successful Gemini calls since the last reset.

    Token means are taken over the calls whose reply carried usage_metadata
    ("calls_with_usage"), so a reply without it does not pull the mean down.
    "seconds" includes the retry when one was needed; "retried" counts those calls.
    """
    with _usage_lock:
        rows = list(_usage)
    n = len(rows)
    with_usage = [r for r in rows if r["has_usage"]]
    m = len(with_usage)
    tokens = {k: sum(r[k] for r in with_usage) for k in _TOKEN_FIELDS}
    seconds = sum(r["seconds"] for r in rows)
    per_call = {k: (v / m if m else 0.0) for k, v in tokens.items()}
    per_call["seconds"] = seconds / n if n else 0.0
    return {
        "model": MODEL_NAME,
        "calls": n,
        "calls_with_usage": m,
        "retried": sum(1 for r in rows if r["attempts"] > 1),
        **tokens,
        "seconds": seconds,
        "per_call": per_call,
    }
