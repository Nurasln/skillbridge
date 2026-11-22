# Rules for every contributor and every AI agent (Claude, Codex, Cursor...)

Read this before changing anything. Several people's agents work on this repo in parallel.
The goal: nobody's agent breaks someone else's work.

## 1. Who owns what

| Role | Person | Owns (only edit these) | Delivers |
| --- | --- | --- | --- |
| Team Leader / PM | Dilara Gültekin | `README.md`, `pipeline.py` (integration), `AGENTS.md` | Wiring, merges, report |
| Data Engineer | TBD | `src/skillbridge/data.py`, `data/` scripts | `load_jobs()`, `extract_cv_text()`, `clean_text()`, gold test set |
| Research & Literature | TBD | `src/skillbridge/baselines.py`, `docs/` | `tfidf_rank()`, `bm25_rank()`, related work |
| Model / Retrieval | Recep Kamil Fırat | `src/skillbridge/retrieval.py` | `build_index()`, `dense_search()`, `hybrid_search()` (done) |
| AI / LLM Engineer | TBD | `src/skillbridge/rerank.py`, `src/skillbridge/explain.py` | cross-encoder `rerank()`, Gemini `explain()` |
| Evaluation & Testing | TBD | `src/skillbridge/evaluate.py`, `tests/` | Recall@5/10, MRR, latency, cost table |
| Frontend / UI | TBD | `app.py` | Streamlit screens |

Fill in TBD names in this table (leader's job). If your task needs a change in a file you do not own,
**do not edit it**: open an issue or tell the owner.

## 2. Contracts (frozen, change only with team agreement)

```
Job    = {"id": str, "text": str, ...optional "title", "company"}
data.load_jobs() -> list[Job]
data.extract_cv_text(uploaded_file) -> str
data.clean_text(text) -> str
baselines.tfidf_rank / bm25_rank(cv_text, jobs, k) -> list[Job + "score"], best first
retrieval.build_index(jobs) -> index
retrieval.dense_search(cv_text, index, k) -> list[Job + "score"]
retrieval.hybrid_search(cv_text, jobs, index, k) -> list[Job + "score"]
rerank.rerank(cv_text, jobs) -> same jobs + "score", best first
explain.explain(cv_text, job_text) -> {"matched": [str], "missing": [str], "summary": str}
evaluate.recall_at_k(ranked_ids, relevant_ids, k) -> float
evaluate.reciprocal_rank(ranked_ids, relevant_ids) -> float
```

`src/skillbridge/pipeline.py` calls these in order. Until a function is done it must
`raise NotImplementedError` (pipeline then uses a safe fallback), never return fake numbers.

## 3. Git workflow

1. `git pull --rebase` before you start and before you push.
2. Work on a branch: `feat/<role>-<short-topic>` (e.g. `feat/data-load-jobs`). Open a PR to `main`.
3. One PR = only your own files. A PR that touches someone else's file needs their approval.
4. Never force-push `main`. Never rewrite others' commits.
5. Before pushing, run: `cd src && python -m skillbridge.pipeline` (must still run end to end).
6. New dependency? Add it to `requirements.txt` in the same PR and say so in the PR text.

## 4. Never

- Commit API keys (`.env` only), datasets, model weights or `.venv`.
- Rename/move files or change a contract signature on your own.
- Reformat or "clean up" files you do not own.
- Put numbers in the report that did not come from `evaluate.py` runs.
