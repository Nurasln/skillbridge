"""End-to-end SkillBridge pipeline: one place where all modules meet.

Owner: Team Leader (integration). Module owners only fill in their own file.

Data contract (every module speaks this shape):
    Job    = {"id": str, "text": str, ...optional fields like "title"}
    Ranked = Job + {"score": float}
    Match  = Ranked + {"explanation": {"matched": [...], "missing": [...], "summary": str}}

Flow:
    CV file/text -> data.extract_cv_text -> data.clean_text
    jobs = data.load_jobs()  (cleaned once, indexed once by retrieval.build_index)
    retrieval.hybrid_search  -> top RETRIEVE_K   (BM25 + dense, RRF)
    rerank.rerank            -> top FINAL_K      (cross-encoder)
    explain.explain          -> matched / missing skills per job

Methods (match(..., method=...)), so the eval script and the app compare them
through one entry point:
    "tfidf"          baselines.tfidf_rank
    "bm25"           baselines.bm25_rank
    "dense"          retrieval.dense_search
    "hybrid"         retrieval.hybrid_search
    "hybrid_rerank"  hybrid top RETRIEVE_K -> rerank.rerank   (default, the app's method)
A method whose ranker is not implemented raises NotImplementedError: the pipeline
never swaps in another method behind your back. With strict=True (use it in
evaluation) a rerank fallback raises too, so a "hybrid_rerank" number is never
silently a "hybrid" number.

Modules that are not implemented yet (raise NotImplementedError) fall back to a
safe default, so the app and tests keep running while the team works in parallel.
rerank and explain depend on models and a remote API, so any runtime error there
(quota, timeout, missing model) also falls back instead of crashing the app; the
reason is logged and listed in Result.fallbacks.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from . import baselines, data, explain, rerank, retrieval

log = logging.getLogger(__name__)

RETRIEVE_K = 20
FINAL_K = 5
METHODS = ("tfidf", "bm25", "dense", "hybrid", "hybrid_rerank")
DEFAULT_METHOD = "hybrid_rerank"

SAMPLE_JOBS = [
    {"id": "j1", "text": "Junior Python developer. Pandas, SQL, REST APIs."},
    {"id": "j2", "text": "Data analyst intern. Excel, SQL, dashboards, Power BI."},
    {"id": "j3", "text": "Frontend developer. React, TypeScript, CSS."},
    {"id": "j4", "text": "Registered nurse for an intensive care unit."},
]


def _empty_explanation() -> dict:
    """A fresh empty explanation per job (a shared dict would leak edits across cards)."""
    return {"matched": [], "missing": [], "summary": ""}


def _try(fn, *args, fallback):
    try:
        return fn(*args), True
    except NotImplementedError:
        return fallback, False


def _guarded(fn, *args, fallback):
    """Like _try, but also survives runtime errors (API down, quota, missing model).

    Returns (value, None) on success, otherwise (fallback, reason).
    """
    try:
        return fn(*args), None
    except NotImplementedError as err:
        # explain/rerank also use NotImplementedError for API or model failures; keep their message
        reason = str(err).strip() or "not implemented"
        log.warning("%s.%s fell back: %s", fn.__module__, fn.__name__, reason)
        return fallback, reason[:120]
    except Exception as err:  # noqa: BLE001 - an external service must not crash the demo
        log.exception("%s.%s failed; using fallback", fn.__module__, fn.__name__)
        return fallback, f"error: {type(err).__name__}"


@dataclass
class Result:
    matches: list[dict]
    timings: dict[str, float] = field(default_factory=dict)
    fallbacks: list[str] = field(default_factory=list)


class Pipeline:
    """Build once (load + index jobs), then call match() for each CV."""

    def __init__(self, jobs: list[dict] | None = None):
        self.fallbacks: list[str] = []
        if jobs is None:
            jobs, ok = _try(data.load_jobs, fallback=SAMPLE_JOBS)
            if not ok:
                self.fallbacks.append("data.load_jobs -> sample jobs")
        self.jobs = [{**j, "text": self._clean(j["text"])} for j in jobs]
        self.index = retrieval.build_index(self.jobs)

    def _clean(self, text: str) -> str:
        out, ok = _try(data.clean_text, text, fallback=" ".join(text.split()))
        if not ok and "data.clean_text -> whitespace only" not in self.fallbacks:
            self.fallbacks.append("data.clean_text -> whitespace only")
        return out

    def _rank(self, cv: str, method: str, retrieve_k: int, final_k: int, strict: bool,
              t: dict, fallbacks: list) -> list[dict]:
        """Ranked jobs (best first, each with "score") for one method."""
        s = time.perf_counter()
        if method == "tfidf":
            ranked = baselines.tfidf_rank(cv, self.jobs, final_k)
        elif method == "bm25":
            ranked = baselines.bm25_rank(cv, self.jobs, final_k)
        elif method == "dense":
            ranked = retrieval.dense_search(cv, self.index, final_k)
        else:  # "hybrid" and "hybrid_rerank" share the first stage
            k = final_k if method == "hybrid" else retrieve_k
            ranked = retrieval.hybrid_search(cv, self.jobs, self.index, k)
        t["retrieval"] = time.perf_counter() - s

        if method == "hybrid_rerank":
            s = time.perf_counter()
            reranked, why = _guarded(rerank.rerank, cv, ranked, fallback=ranked)
            if why:
                if strict:
                    raise RuntimeError(f"rerank failed in strict mode: {why}")
                fallbacks.append(f"rerank.rerank -> retrieval order ({why})")
            ranked = reranked
            t["rerank"] = time.perf_counter() - s
        return ranked[:final_k]

    def match(self, cv_text: str, retrieve_k: int = RETRIEVE_K, final_k: int = FINAL_K, *,
              method: str = DEFAULT_METHOD, with_explanations: bool = True, strict: bool = False) -> Result:
        """Rank jobs for one CV.

        method: one of METHODS. with_explanations=False skips the LLM (fast and free,
        for evaluation). strict=True raises instead of falling back in ranking.
        """
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; choose one of {METHODS}")
        t = {}
        start = time.perf_counter()
        cv = self._clean(cv_text)
        fallbacks = list(self.fallbacks)
        ranked = self._rank(cv, method, retrieve_k, final_k, strict, t, fallbacks)

        s = time.perf_counter()
        matches, why = [], None
        for job in ranked:
            exp = None
            if with_explanations and why is None:   # after the first failure, stop calling the API
                exp, why = _guarded(explain.explain, cv, job["text"], fallback=None)
            # missing keys in an LLM answer must not break the UI
            exp = {**_empty_explanation(), **exp} if isinstance(exp, dict) else _empty_explanation()
            matches.append({**job, "explanation": exp})
        if why:
            fallbacks.append(f"explain.explain -> empty ({why})")
        if with_explanations:
            t["explain"] = time.perf_counter() - s

        t["total"] = time.perf_counter() - start
        return Result(matches=matches, timings=t, fallbacks=fallbacks)


def read_cv(uploaded_file) -> str:
    """Uploaded PDF/text -> raw CV text (falls back to UTF-8 decode)."""
    text, ok = _try(data.extract_cv_text, uploaded_file, fallback=None)
    if not ok:
        raw = uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file
        text = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)
    return text


if __name__ == "__main__":
    p = Pipeline()
    demo_cv = "Computer engineering student. Python scrapers, pandas analysis, SQL queries."
    r = p.match(demo_cv)
    for m in r.matches:
        print(m["id"], round(m["score"], 3), m["explanation"]["summary"])
    print("timings", {k: round(v, 3) for k, v in r.timings.items()})
    print("fallbacks", r.fallbacks)
    print("methods:")
    for name in METHODS:
        try:
            top = p.match(demo_cv, final_k=3, method=name, with_explanations=False).matches
            print(f"  {name:14s} ok   top-3 {[m['id'] for m in top]}")
        except NotImplementedError:
            print(f"  {name:14s} not implemented yet")
