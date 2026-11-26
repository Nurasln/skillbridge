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

Modules that are not implemented yet (raise NotImplementedError) fall back to a
safe default, so the app and tests keep running while the team works in parallel.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import data, explain, rerank, retrieval

RETRIEVE_K = 20
FINAL_K = 5

SAMPLE_JOBS = [
    {"id": "j1", "text": "Junior Python developer. Pandas, SQL, REST APIs."},
    {"id": "j2", "text": "Data analyst intern. Excel, SQL, dashboards, Power BI."},
    {"id": "j3", "text": "Frontend developer. React, TypeScript, CSS."},
    {"id": "j4", "text": "Registered nurse for an intensive care unit."},
]

EMPTY_EXPLANATION = {"matched": [], "missing": [], "summary": ""}


def _try(fn, *args, fallback):
    try:
        return fn(*args), True
    except NotImplementedError:
        return fallback, False


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

    def match(self, cv_text: str, retrieve_k: int = RETRIEVE_K, final_k: int = FINAL_K) -> Result:
        t = {}
        start = time.perf_counter()
        cv = self._clean(cv_text)

        s = time.perf_counter()
        candidates = retrieval.hybrid_search(cv, self.jobs, self.index, retrieve_k)
        t["retrieval"] = time.perf_counter() - s

        s = time.perf_counter()
        ranked, ok = _try(rerank.rerank, cv, candidates, fallback=candidates)
        ranked = ranked[:final_k]
        fallbacks = list(self.fallbacks)
        if not ok:
            fallbacks.append("rerank.rerank -> retrieval order")
        t["rerank"] = time.perf_counter() - s

        s = time.perf_counter()
        matches = []
        for job in ranked:
            exp, ok = _try(explain.explain, cv, job["text"], fallback=EMPTY_EXPLANATION)
            if not ok and "explain.explain -> empty" not in fallbacks:
                fallbacks.append("explain.explain -> empty")
            matches.append({**job, "explanation": exp})
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
    r = p.match("Computer engineering student. Python scrapers, pandas analysis, SQL queries.")
    for m in r.matches:
        print(m["id"], round(m["score"], 3), m["explanation"]["summary"])
    print("timings", {k: round(v, 3) for k, v in r.timings.items()})
    print("fallbacks", r.fallbacks)
