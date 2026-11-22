# SkillBridge: Explainable CV–Job Matching

SWE015 – Introduction to Large Language Models, final project (Topic 10: Semantic Search and Matching).

A student uploads a CV; SkillBridge ranks job postings by fit and explains each match
(matched skills / missing skills).

## Pipeline

1. **CV upload** (Streamlit, PDF or text)
2. **Text extraction and cleaning**
3. **Candidate retrieval**: BM25 + dense embeddings (FAISS), top 20
4. **Cross-encoder reranker**: top 20 → top 5
5. **LLM explanation**: matched / missing skills
6. **Result screen**: ranked jobs with explanation cards

Baselines compared on the same test set: TF-IDF, BM25, dense, hybrid + reranker
(Recall@5, Recall@10, MRR, latency, cost).

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then put your own API key in .env
```

## Run

```bash
streamlit run app.py
```

## Module contract

Every module passes jobs as `{"id", "text", ...}` dicts; rankers add `"score"`,
`explain` returns `{"matched", "missing", "summary"}`. `pipeline.py` calls the modules in
order and falls back to safe defaults for anything still `NotImplementedError`, so each
member can work on their own file independently. Quick check: `cd src && python -m skillbridge.pipeline`.

## Repository layout

```
app.py                  Streamlit interface (display only)
src/skillbridge/
    pipeline.py         wires all modules together; shared data contract
    data.py             dataset loading, CV text extraction, cleaning
    baselines.py        TF-IDF and BM25 rankers
    retrieval.py        embeddings, FAISS index, dense + hybrid search
    rerank.py           cross-encoder reranker
    explain.py          LLM match explanations
    evaluate.py         Recall@k, MRR, latency, cost
data/                   local data (not committed)
```

## Team

| Member | Role | Component |
| --- | --- | --- |
| TBD | Team Leader / Project Manager | Pipeline integration, repository, README |
| TBD | Data Engineer | `data.py` |
| TBD | Research & Literature | `baselines.py` |
| TBD | Model / Retrieval Engineer | `retrieval.py` |
| TBD | AI / LLM Engineer | `rerank.py`, `explain.py` |
| TBD | Evaluation & Testing | `evaluate.py` |
| TBD | Frontend / UI Developer | `app.py` |

## Rules for this repo

**Team and AI agents: read [AGENTS.md](AGENTS.md) first** (who owns which file, frozen interfaces, branch/PR workflow).


- Never commit API keys: keep them in `.env` (already in `.gitignore`).
- No model weights or large data files in git; link to Hugging Face Hub instead.
- Every number in the report must come from our own runs (`evaluate.py`).
