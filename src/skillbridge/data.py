"""Dataset loading, CV text extraction and cleaning.

Owner: Data Engineer (Arda Kocadoru)

Public interface (contract in AGENTS.md; the extra arguments are optional and
keep the frozen signatures working):
    load_jobs(pool="eval")       -> [{"id", "text", "source", optional "title"}]
    extract_cv_text(file)        -> plain CV text (PDF, DOCX or text file)
    clean_text(text)             -> normalised text (idempotent, case preserved)
    load_test_set(split="gold")  -> [{"id", "text", "relevant", "partial", "verified"}]

Where the data comes from (all Hugging Face, pinned to a commit so every
teammate gets identical rows; nothing is committed to git):
    cnamuangtoun/resume-job-description-fit  original CV-JD pairs, 3 labels.
        test split: 1,759 pairs, 477 CVs, 71 job descriptions.
    Philseok/resume_fit_data                 334 rows; every row is a pair from
        the cnamuangtoun test split with the same label (verified in `report`).
    manohar005/resume-job-fit-merged-v1      the same CVs/JDs re-labelled by
        Qwen2.5-32B; used only as a second opinion in the gold review sheet.
    rahulrajpu676/job-resume-matching        1,167 LinkedIn-style job posts
        (MIT); only used for the optional "extended" pool.

Pools:
    "eval"      71 unique JDs of the cnamuangtoun test split (superset of
                Philseok's 65). Every gold/dev CV is labelled against this pool,
                so evaluation runs on it. This is the default.
    "extended"  eval + 280 cnamuangtoun train JDs + rahul JDs, near-duplicates
                removed (~1.4k). For the demo only: these extra jobs have no
                labels for our CVs, so do not compute metrics on it.

Command line (run from src/):
    python -m skillbridge.data report        dataset statistics -> data/dataset_report.md
    python -m skillbridge.data export        cleaned pools -> data/jobs_<pool>.jsonl
    python -m skillbridge.data review-sheet  blind review workbook -> data/gold/
    python -m skillbridge.data cross-sheet   second-reviewer tabs -> data/gold/review_cross.xlsx
    python -m skillbridge.data gold --sheet X.xlsx
                                             filled workbook -> data/gold/gold_labels.csv
                                             (+ agreement.md)
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import unicodedata
import zipfile
from functools import lru_cache
from pathlib import Path
from xml.etree import ElementTree

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("SKILLBRIDGE_DATA_DIR", REPO_ROOT / "data"))
GOLD_DIR = DATA_DIR / "gold"

# (repo id, commit sha). Bump the sha only on purpose: it changes every number.
CNAM = ("cnamuangtoun/resume-job-description-fit", "08978e21714984bb417547d2c0f9b477f5298163")
PHILSEOK = ("Philseok/resume_fit_data", "c6b76785c15c945b785d70958ec3cf296d6d5c70")
MANOHAR = ("manohar005/resume-job-fit-merged-v1", "b8d56708674ad99cf419a07208fae5a09ee79699")
RAHUL = ("rahulrajpu676/job-resume-matching", "2be32581a2b5dc8346618e7e15d3acd4dabc0d5d")

GOOD, POTENTIAL, NO = "Good Fit", "Potential Fit", "No Fit"
LABELS = (GOOD, POTENTIAL, NO)

GOLD_SIZE = 49          # 7 reviewers x 7 CVs (project guide: "~50 CV, herkes ~7")
N_REVIEWERS = 7
CROSS_PER_REVIEWER = 1  # CVs each reviewer also labels for another reviewer (agreement check)
POOL_DEPTH = 5          # BM25 / dense top-k added to each CV's review candidates
NEAR_DUP_COSINE = 0.95  # TF-IDF cosine above which two job posts count as one
GOLD_SEED = "skillbridge-gold-v1"


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #

# CamelCase product names that must not be split by the glued-word repair below.
_KEEP_CAMEL = {
    t.lower() for t in """
    JavaScript TypeScript CoffeeScript ActionScript PowerShell PowerPoint PowerApps
    PowerPivot SharePoint OneDrive OneNote QuickBooks UltraTax NetSuite PeopleSoft
    SuccessFactors ServiceNow ServiceTitan SalesForce HubSpot WordPress WooCommerce
    LinkedIn GitHub GitLab BitBucket DevOps DevSecOps GitOps FinOps DataOps
    TensorFlow PyTorch PySpark PyCharm NumPy SciPy SymPy LaTeX YouTube FaceBook
    WhatsApp PayPal DocuSign DropBox SolidWorks AutoDesk QlikView QlikSense
    SnapLogic DataStage InfoSphere WebSphere WebLogic WebSockets JetBrains PhpStorm
    WebStorm CloudFormation CloudWatch CloudFront ElasticSearch LogStash FireBase
    FireStore SageMaker MicroStation MicroStrategy SolarWinds AppDynamics NewRelic
    DataDog DataBricks SnowFlake ClickHouse OpenShift OpenStack MacBook AirBnB
    PlayStation TikTok LoadRunner QuickTest HackerRank LeetCode AppFolio JavaBeans
    BigQuery BigTable RedShift MariaDB SpringBoot HashiCorp KeyCloak NodeJs
    """.split()
}

_MOJIBAKE_HINT = re.compile("[ÂÃÄÅâ][\u0080-¿‘-›€™]")
_INVISIBLE = re.compile("[­​-‏  ⁠﻿]")
_BULLETS = re.compile("[•‣⁃∙▪-▫●◦◼▶➢♦❖✓✔✅]")
_RULE_LINES = re.compile(r"([-_=*~.·])\1{3,}")              # ----- ____ ..... separators
_HYPHEN_BREAK = re.compile(r"(?<=[a-zçğıöşü])-[ \t]*\n[ \t]*(?=[a-zçğıöşü])")
_MISSING_SPACE = re.compile(r"(?<=[a-zçğıöşü]{2})([.!?;:,])(?=[A-ZÇĞİÖŞÜ][a-zçğıöşü])")
_WORD = re.compile(r"[^\W\d_]{4,}")
_CAMEL_HINT = re.compile(r"[a-zçğıöşüà-ÿ]{2}[A-ZÇĞİÖŞÜÀ-Þ][a-zçğıöşüà-ÿ]")

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"(?:https?://|www\.)\S+|\b(?:linkedin|github|gitlab)\.com/\S+", re.I)
_PHONE = re.compile(
    r"(?<![\w+])(?:\+\d{1,3}[ .-]?)?(?:\(\d{1,4}\)[ .-]?)?\d{2,4}(?:[ .-]\d{2,4}){1,4}(?![\w])"
    r"|(?<![\w+])\+\d{10,13}(?!\w)"
    r"|\b0\d{10}\b"
)
_YEAR = re.compile(r"(19[5-9]\d|20[0-3]\d)")


def _split_glued_word(word: str) -> str:
    """'SummaryForward' -> 'Summary Forward', but keep 'JavaScript'.

    The source datasets lost their line breaks, so a heading and the next word
    were glued ("SkillsAgile", "plansLed"). We split at lower->Upper+lower
    boundaries with at least two lowercase letters before, unless the boundary
    lies inside a known CamelCase product name.
    """
    low = word.lower()
    owner = [-1] * len(word)
    for n, term in enumerate(_KEEP_CAMEL):
        start = low.find(term)
        while start != -1:
            for i in range(start, start + len(term)):
                owner[i] = n
            start = low.find(term, start + 1)
    out = [word[0]]
    for i in range(1, len(word)):
        boundary = (
            i >= 2 and i + 1 < len(word)
            and word[i - 2].islower() and word[i - 1].islower()
            and word[i].isupper() and word[i + 1].islower()
        )
        if boundary and not (owner[i - 1] != -1 and owner[i - 1] == owner[i]):
            out.append(" ")
        out.append(word[i])
    return "".join(out)


def _mask_phone(m: re.Match) -> str:
    s = m.group(0)
    digits = re.sub(r"\D", "", s)
    groups = re.findall(r"\d+", s)
    if not 10 <= len(digits) <= 15:
        return s
    if all(len(g) == 4 and _YEAR.fullmatch(g) for g in groups):   # "2016 2017 2018"
        return s
    return "[PHONE]"


def mask_pii(text: str) -> str:
    """Replace e-mail addresses, URLs and phone numbers with placeholders.

    Names and street addresses cannot be removed reliably by rules, so real
    CVs must still be anonymised by hand (project privacy rule).
    """
    text = _EMAIL.sub("[EMAIL]", text)
    text = _URL.sub("[URL]", text)
    return _PHONE.sub(_mask_phone, text)


def clean_text(text: str) -> str:
    """Normalise a CV or job text for retrieval, reranking and display.

    Steps: repair mojibake (ftfy), Unicode NFKC (ligatures, non-breaking spaces),
    drop invisible characters, bullets and separator lines, re-join words
    hyphenated at line breaks, add missing spaces after punctuation, split
    glued words, mask e-mail/URL/phone, collapse all whitespace to one space.

    Case is preserved on purpose: the text is shown in the UI and sent to the
    LLM, BM25 lowercases on its own and MiniLM is an uncased model. Turkish
    letters are kept. The function is idempotent (clean(clean(x)) == clean(x)),
    so the pipeline may call it on text that load_jobs() already cleaned.
    """
    if text is None:
        return ""
    text = str(text)
    if _MOJIBAKE_HINT.search(text):
        import ftfy
        text = ftfy.fix_text(text, normalization="NFC")
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = mask_pii(text)                                 # before any step that could split an address
    text = _HYPHEN_BREAK.sub("", text)
    text = _BULLETS.sub(" ", text)
    text = _RULE_LINES.sub(" ", text)
    text = re.sub(r"\bandor\b", "and/or", text)          # "and/or" lost its slash in the source
    text = _MISSING_SPACE.sub(r"\1 ", text)
    text = _WORD.sub(lambda m: _split_glued_word(m.group(0)) if _CAMEL_HINT.search(m.group(0)) else m.group(0), text)
    text = "".join(ch if ch in "\n\t" or unicodedata.category(ch)[0] != "C" else " " for ch in text)
    text = "".join(" " if unicodedata.category(ch) == "So" else ch for ch in text)   # emoji, ®-like symbols
    return " ".join(text.split())


def fingerprint(text: str) -> str:
    """Content hash used for ids and duplicate detection (case/space-insensitive)."""
    return hashlib.md5(" ".join(clean_text(text).lower().split()).encode("utf-8")).hexdigest()


def job_id(text: str) -> str:
    return "j_" + fingerprint(text)[:10]


def cv_id(text: str) -> str:
    return "cv_" + fingerprint(text)[:10]


# --------------------------------------------------------------------------- #
# CV upload -> text
# --------------------------------------------------------------------------- #

def _read_upload(uploaded_file) -> tuple[str, bytes]:
    """Return (lowercase file name, raw bytes) for the input types we accept."""
    if isinstance(uploaded_file, (bytes, bytearray)):
        return "", bytes(uploaded_file)
    if isinstance(uploaded_file, (str, Path)):
        path = Path(uploaded_file)
        try:
            if path.is_file():
                return path.name.lower(), path.read_bytes()
        except (OSError, ValueError):   # e.g. a long pasted CV is not a valid path
            pass
        return "", str(uploaded_file).encode("utf-8")
    name = str(getattr(uploaded_file, "name", "") or "").lower()
    if hasattr(uploaded_file, "getvalue"):            # Streamlit UploadedFile, BytesIO
        raw = uploaded_file.getvalue()
    elif hasattr(uploaded_file, "read"):
        raw = uploaded_file.read()
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
    else:
        raise TypeError(f"Unsupported CV input: {type(uploaded_file).__name__}")
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    return name, bytes(raw)


def _pdf_to_text(raw: bytes) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("This PDF is password-protected. Please upload an unlocked copy.")
        pages = [page.extract_text() or "" for page in reader.pages]
    except PdfReadError as err:
        raise ValueError(f"Could not read this PDF ({err}). Try exporting it again or paste the text.") from err
    text = "\n\n".join(pages)
    if len(text.strip()) < 30:
        raise ValueError("No text found in this PDF (it looks scanned). Please upload a text-based PDF or paste the text.")
    return text


def _docx_to_text(raw: bytes) -> str:
    """Paragraph text of a .docx using only the standard library."""
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        root = ElementTree.fromstring(z.read("word/document.xml"))
    paragraphs = []
    for p in root.iter(f"{ns}p"):
        parts = []
        for node in p.iter():
            if node.tag == f"{ns}t" and node.text:
                parts.append(node.text)
            elif node.tag in (f"{ns}tab", f"{ns}br"):
                parts.append(" ")
        paragraphs.append("".join(parts))
    return "\n".join(paragraphs)


def _decode_text(raw: bytes) -> str:
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", errors="replace")
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace")
    for enc in ("utf-8", "cp1254"):        # cp1254 = Turkish Windows (Notepad "ANSI")
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def extract_cv_text(uploaded_file) -> str:
    """Turn an uploaded PDF, DOCX or text file into plain CV text.

    Accepts a Streamlit UploadedFile, any file-like object, raw bytes or a path.
    The file type is detected from the content, not the extension. Returns the
    text with line breaks; call clean_text() before matching (pipeline does).
    Raises ValueError with a user-facing message for scanned, encrypted or
    broken PDFs.
    """
    name, raw = _read_upload(uploaded_file)
    if b"%PDF-" in raw[:1024]:
        return _pdf_to_text(raw)
    if raw.startswith(b"PK"):
        try:
            return _docx_to_text(raw)
        except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as err:
            raise ValueError("Unsupported file. Please upload a PDF, DOCX or plain-text CV.") from err
    if name.endswith(".doc") or raw.startswith(b"\xd0\xcf\x11\xe0"):
        raise ValueError("Old .doc files are not supported. Please save the CV as PDF or DOCX.")
    return _decode_text(raw)


# --------------------------------------------------------------------------- #
# Hugging Face sources
# --------------------------------------------------------------------------- #

def _download(source: tuple[str, str], filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    repo_id, revision = source
    return Path(hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision))


@lru_cache(maxsize=None)
def _cnam(split: str):
    import pandas as pd

    df = pd.read_csv(_download(CNAM, f"{split}.csv"))
    return df.rename(columns={"resume_text": "resume", "job_description_text": "jd"})


@lru_cache(maxsize=None)
def _philseok():
    import pandas as pd

    df = pd.read_parquet(_download(PHILSEOK, "data/train-00000-of-00001.parquet"))
    return df.rename(columns={"resume_text": "resume", "job_description_text": "jd"})


@lru_cache(maxsize=None)
def _manohar_test():
    import pandas as pd

    return pd.read_parquet(_download(MANOHAR, "data/test-00000-of-00001.parquet"))


@lru_cache(maxsize=None)
def _rahul_jobs():
    import pandas as pd

    cols = ["job_id", "job_title", "category", "job_description"]
    return pd.read_csv(_download(RAHUL, "job.csv"), usecols=cols)


def _with_ids(df, cv_col: str = "resume", job_col: str = "jd"):
    """Add cleaned text and content ids. Cleaning runs once per unique text."""
    df = df.copy()
    for col, prefix in ((cv_col, "cv"), (job_col, "job")):
        uniq = {t: clean_text(t) for t in df[col].unique()}
        df[f"{prefix}_text"] = df[col].map(uniq)
        hashes = {c: hashlib.md5(" ".join(c.lower().split()).encode("utf-8")).hexdigest()[:10]
                  for c in set(uniq.values())}
        df[f"{prefix}_id"] = df[f"{prefix}_text"].map(lambda c: ("cv_" if prefix == "cv" else "j_") + hashes[c])
    return df


@lru_cache(maxsize=None)
def _labelled_test_pairs():
    """cnamuangtoun test split with ids, original label and Qwen label."""
    pairs = _with_ids(_cnam("test"))
    qwen = _with_ids(_manohar_test())[["cv_id", "job_id", "label"]].rename(columns={"label": "qwen_label"})
    pairs = pairs.merge(qwen, on=["cv_id", "job_id"], how="left")
    assert not pairs.duplicated(["cv_id", "job_id"]).any(), "duplicate CV-job pairs after cleaning"
    return pairs


# --------------------------------------------------------------------------- #
# Job pool
# --------------------------------------------------------------------------- #

_ROLE_WORDS = re.compile(
    r"\b(analyst|engineer|developer|accountant|manager|architect|scientist|designer|specialist|"
    r"consultant|director|intern|associate|administrator|coordinator|controller|programmer|"
    r"lead|officer|representative|executive|technician|auditor|bookkeeper)s?\b", re.I)
_TITLE_FIELD = re.compile(
    r"(?:job title|position title|title|position|role)\s*[:\-–]\s*(.{3,70}?)"
    r"(?=\s*(?:location|duration|job type|job status|type|status|exp\w*|salary|pay|department|"
    r"reports to|start|client|work|$|\n|\||\(on|\(remote|\(hybrid))", re.I)


def _plausible_title(s: str) -> str | None:
    s = re.sub(r"^\d+\s+", "", clean_text(s)).strip(" -:|,.")     # drop leading requisition numbers
    if not (1 <= len(s.split()) <= 8) or not _ROLE_WORDS.search(s):
        return None
    if re.search(r"\b(we|our|you|are|is|looking|seeking|with)\b", s, re.I):
        return None
    return s


def _extract_title(raw_text: str) -> str | None:
    """Best-effort job title from an explicit 'Title:'/'Role:' field or a short first line.

    Most source JDs carry no explicit title, so this returns None for many jobs;
    the UI should then fall back to the first words of the text.
    """
    m = _TITLE_FIELD.search(raw_text[:400])
    if m and (t := _plausible_title(m.group(1))):
        return t
    first = raw_text.strip().split("\n", 1)[0]
    if len(first.split()) <= 7:
        return _plausible_title(first)
    return None


def _sentence_case(text: str) -> str:
    """For all-caps sources: lowercase, then capitalise sentence starts."""
    text = text.lower()
    return re.sub(r"(^|[.!?]\s+|\n\s*)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


def _drop_near_duplicates(jobs: list[dict]) -> list[dict]:
    """Keep the first of every group of job texts with TF-IDF cosine >= NEAR_DUP_COSINE."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform([j["text"] for j in jobs])
    sims = (X @ X.T).tocsr()
    dropped: set[int] = set()
    for i in range(len(jobs)):
        if i in dropped:
            continue
        row = sims.getrow(i)
        for j, s in zip(row.indices, row.data):
            if j > i and s >= NEAR_DUP_COSINE:
                dropped.add(j)
    return [j for n, j in enumerate(jobs) if n not in dropped]


@lru_cache(maxsize=None)
def _build_pool(pool: str) -> tuple[dict, ...]:
    if pool not in ("eval", "extended"):
        raise ValueError(f"unknown pool {pool!r}; use 'eval' or 'extended'")

    test = _labelled_test_pairs()
    philseok_jobs = set(_with_ids(_philseok())["job_id"])
    jobs: list[dict] = []
    for raw, jid, text in test[["jd", "job_id", "job_text"]].drop_duplicates("job_id").itertuples(index=False):
        job = {"id": jid, "text": text, "source": "cnamuangtoun:test"}
        if jid in philseok_jobs:
            job["source"] += "+philseok"
        if title := _extract_title(raw):
            job["title"] = title
        jobs.append(job)
    missing = philseok_jobs - {j["id"] for j in jobs}
    if missing:
        log.warning("%d Philseok jobs not found in the cnamuangtoun test split", len(missing))
    if pool == "eval":
        return tuple(sorted(jobs, key=lambda j: j["id"]))

    seen = {j["id"] for j in jobs}
    train = _with_ids(_cnam("train"))
    for raw, jid, text in train[["jd", "job_id", "job_text"]].drop_duplicates("job_id").itertuples(index=False):
        if jid not in seen:
            seen.add(jid)
            job = {"id": jid, "text": text, "source": "cnamuangtoun:train"}
            if title := _extract_title(raw):
                job["title"] = title
            jobs.append(job)
    for title, category, desc in _rahul_jobs()[["job_title", "category", "job_description"]].itertuples(index=False):
        desc = str(desc)
        letters = [c for c in desc if c.isalpha()]
        if letters and sum(c.isupper() for c in letters) / len(letters) > 0.8:
            desc = _sentence_case(desc)          # every rahul description is upper-cased
        text = clean_text(desc)
        jid = job_id(text)
        if jid not in seen and len(text.split()) >= 30:
            seen.add(jid)
            jobs.append({"id": jid, "text": text, "source": "rahulrajpu676", "title": clean_text(title),
                         "category": str(category)})
    # eval jobs come first, so a near-duplicate never removes a labelled job
    return tuple(_drop_near_duplicates(jobs))


def load_jobs(pool: str = "eval") -> list[dict]:
    """Return the job posting pool as [{"id", "text", "source", optional "title"}].

    pool="eval" (default): the 71 labelled job descriptions; use for metrics.
    pool="extended": adds unlabelled jobs for a richer demo (~1.4k).
    Texts are already cleaned with clean_text(). The first call downloads the
    source files from Hugging Face (cached afterwards, works offline).
    """
    return [dict(j) for j in _build_pool(pool)]


# --------------------------------------------------------------------------- #
# Gold / dev test set
# --------------------------------------------------------------------------- #

def _stable_order(key: str) -> str:
    return hashlib.sha256(f"{GOLD_SEED}:{key}".encode()).hexdigest()


@lru_cache(maxsize=None)
def _query_split() -> dict[str, str]:
    """cv_id -> "gold" | "dev" for every test CV with at least one Good Fit.

    Deterministic (hash order, no RNG) and stratified by the CV domain that the
    manohar dataset assigns, so the 49 gold CVs mirror the domain mix.
    """
    test = _labelled_test_pairs()
    eligible = sorted(set(test.loc[test.label == GOOD, "cv_id"]))
    dom = _with_ids(_manohar_test()).drop_duplicates("cv_id").set_index("cv_id")["resume_domain"]
    domains = {c: dom.get(c, "other") for c in eligible}
    groups: dict[str, list[str]] = {}
    for c in sorted(eligible, key=_stable_order):
        groups.setdefault(domains[c], []).append(c)
    # largest-remainder allocation of GOLD_SIZE slots
    total = len(eligible)
    quota = {d: GOLD_SIZE * len(v) / total for d, v in groups.items()}
    take = {d: int(q) for d, q in quota.items()}
    for d in sorted(quota, key=lambda d: (-(quota[d] - take[d]), d))[: GOLD_SIZE - sum(take.values())]:
        take[d] += 1
    split = {c: "dev" for c in eligible}
    for d, members in groups.items():
        for c in members[: take[d]]:
            split[c] = "gold"
    return split


def _gold_labels_file() -> Path:
    return GOLD_DIR / "gold_labels.csv"


def load_test_set(split: str = "gold") -> list[dict]:
    """Evaluation queries: [{"id", "text", "relevant": [job ids], "partial": [job ids], "verified"}].

    split="gold": 49 CVs held out for the final results table.
    split="dev":  the other 184 CVs; use these for tuning, never the gold ones.
    "relevant" = Good Fit, "partial" = Potential Fit, both within the eval pool.
    If data/gold/gold_labels.csv exists (the team's manual review), gold labels
    come from it and "verified" is True; otherwise the original cnamuangtoun
    labels are used and "verified" is False.
    """
    if split not in ("gold", "dev"):
        raise ValueError("split must be 'gold' or 'dev'")
    import pandas as pd

    test = _labelled_test_pairs()
    which = _query_split()
    ids = sorted((c for c, s in which.items() if s == split), key=_stable_order)
    labels = test[["cv_id", "job_id", "label"]]
    verified = False
    if split == "gold" and _gold_labels_file().exists():
        labels = pd.read_csv(_gold_labels_file(), dtype=str)
        labels = labels.rename(columns={"judgement": "label"})[["cv_id", "job_id", "label"]]
        verified = True
    elif split == "gold":
        log.warning("gold labels not reviewed yet; using original cnamuangtoun labels")
    texts = test.drop_duplicates("cv_id").set_index("cv_id")["cv_text"]
    by_cv = labels.groupby("cv_id")
    out = []
    for c in ids:
        rows = by_cv.get_group(c) if c in by_cv.groups else labels.iloc[0:0]
        out.append({
            "id": c,
            "text": texts[c],
            "relevant": sorted(rows.loc[rows.label == GOOD, "job_id"]),
            "partial": sorted(rows.loc[rows.label == POTENTIAL, "job_id"]),
            "verified": verified,
        })
    return out


def gold_cv_ids() -> set[str]:
    """CV ids that must never be used for training or tuning (leakage guard).

    Note: 476 of the 477 test CVs also appear in the cnamuangtoun *train* split
    (paired with other jobs). Anyone fine-tuning on train must drop these ids.
    """
    return {c for c, s in _query_split().items() if s == "gold"}


# --------------------------------------------------------------------------- #
# Command line: report, export, review sheet, gold labels
# --------------------------------------------------------------------------- #

def _report() -> str:
    import pandas as pd

    cnam_test, cnam_train = _with_ids(_cnam("test")), _with_ids(_cnam("train"))
    phil, qwen = _with_ids(_philseok()), _with_ids(_manohar_test())
    rahul = _rahul_jobs()
    test = _labelled_test_pairs()
    split = _query_split()

    def dist(s):
        return ", ".join(f"{k} {v}" for k, v in s.value_counts().reindex(LABELS).fillna(0).astype(int).items())

    phil_in_test = phil.merge(cnam_test[["cv_id", "job_id", "label"]], on=["cv_id", "job_id"], suffixes=("", "_c"))
    both = test.dropna(subset=["qwen_label"])
    agree = (both.label == both.qwen_label).mean()
    good_to_no = ((both.label == GOOD) & (both.qwen_label == NO)).sum()
    cv_overlap = len(set(cnam_test.cv_id) & set(cnam_train.cv_id))
    jd_overlap = len(set(cnam_test.job_id) & set(cnam_train.job_id))
    eval_pool, ext_pool = load_jobs("eval"), load_jobs("extended")
    gold, dev = load_test_set("gold"), load_test_set("dev")
    ext_sources = pd.Series([j["source"].split(":")[0].split("+")[0] for j in ext_pool]).value_counts().to_dict()
    titled = sum("title" in j for j in eval_pool)
    words = pd.Series([len(j["text"].split()) for j in eval_pool])
    cv_words = test.drop_duplicates("cv_id").cv_text.str.split().str.len()
    cv_domain = qwen.drop_duplicates("cv_id").set_index("cv_id")["resume_domain"]
    gold_dom = pd.Series([cv_domain.get(c, "other") for c, s in split.items() if s == "gold"])

    lines = [
        "# SkillBridge dataset report",
        "",
        "Generated by `python -m skillbridge.data report` from the pinned Hugging Face revisions.",
        "",
        "## Sources",
        "",
        "| Dataset | Rows | Licence on card | Role in SkillBridge |",
        "| --- | ---: | --- | --- |",
        f"| {CNAM[0]} | test {len(cnam_test):,} / train {len(cnam_train):,} | none declared | job pool (test JDs) + original labels |",
        f"| {PHILSEOK[0]} | {len(phil):,} | none declared | subset of the cnamuangtoun test split (checked below) |",
        f"| {MANOHAR[0]} | test {len(qwen):,} (train 80,017 not used) | none declared | Qwen2.5-32B labels as second opinion |",
        f"| {RAHUL[0]} | jobs {len(rahul):,} (cv/matches not used) | MIT (tag) | extra jobs for the extended demo pool |",
        "",
        "## Findings",
        "",
        f"- Philseok: {len(phil)} rows, {phil.cv_id.nunique()} CVs, {phil.job_id.nunique()} JDs; labels {dist(phil.label)}. "
        f"{len(phil_in_test)}/{len(phil)} rows are pairs of the cnamuangtoun test split and "
        f"{(phil_in_test.label == phil_in_test.label_c).sum()} keep the same label.",
        f"- cnamuangtoun test: {len(cnam_test):,} pairs, {cnam_test.cv_id.nunique()} CVs, {cnam_test.job_id.nunique()} JDs; "
        f"labels {dist(cnam_test.label)}; no duplicate or conflicting pairs.",
        f"- Train/test leakage in the source: JDs shared {jd_overlap}, CVs shared {cv_overlap} of {cnam_test.cv_id.nunique()}. "
        "We therefore split our own gold/dev sets by CV.",
        f"- manohar (Qwen) vs original labels on the {len(both):,} pairs present in both: agreement {agree:.1%}; "
        f"{good_to_no} original Good Fit pairs are No Fit for Qwen. Labels are noisy, so the gold set is reviewed by hand.",
        "- rahul: every job description is upper-cased (converted to sentence case); match labels are positive-only "
        "and partly derived from an embedding similarity, so they are not used as ground truth.",
        "",
        "## Outputs",
        "",
        f"- Eval pool: {len(eval_pool)} jobs (JD length median {int(words.median())} words, max {words.max()}); "
        f"{titled} have an extracted title.",
        f"- Extended pool: {len(ext_pool):,} jobs after exact and near-duplicate removal (cosine >= {NEAR_DUP_COSINE}); "
        f"by source {ext_sources}.",
        f"- Queries: {len(gold) + len(dev)} test CVs with at least one Good Fit -> gold {len(gold)}, dev {len(dev)} "
        f"(CV length median {int(cv_words.median())} words). Gold domains: {gold_dom.value_counts().to_dict()}.",
        f"- Relevant jobs per gold CV (before review): mean {sum(len(q['relevant']) for q in gold) / len(gold):.2f}.",
        "",
    ]
    return "\n".join(lines)


def _review_candidates():
    """Union of original labels, Qwen Good Fits and BM25/dense top-k for each gold CV."""
    import pandas as pd
    from rank_bm25 import BM25Okapi

    test = _labelled_test_pairs()
    jobs = load_jobs("eval")
    gold = sorted(gold_cv_ids(), key=_stable_order)
    texts = test.drop_duplicates("cv_id").set_index("cv_id")["cv_text"]
    tok = lambda t: re.findall(r"[a-z0-9+#.]+", t.lower())  # noqa: E731
    bm25 = BM25Okapi([tok(j["text"]) for j in jobs])
    cands: dict[tuple[str, str], set[str]] = {}

    def add(c, j, why):
        cands.setdefault((c, j), set()).add(why)

    sub = test[test.cv_id.isin(gold)]
    for c, j in sub[["cv_id", "job_id"]].itertuples(index=False):
        add(c, j, "original")
    for c, j in sub.loc[sub.qwen_label == GOOD, ["cv_id", "job_id"]].itertuples(index=False):
        add(c, j, "qwen")
    for c in gold:
        scores = bm25.get_scores(tok(texts[c]))
        for i in scores.argsort()[::-1][:POOL_DEPTH]:
            add(c, jobs[i]["id"], "bm25")
    try:
        from . import retrieval
        index = retrieval.build_index(jobs)
        for c in gold:
            for hit in retrieval.dense_search(texts[c], index, POOL_DEPTH):
                add(c, hit["id"], "dense")
    except Exception as err:  # retrieval is another member's module; review still works without it
        log.warning("dense pooling skipped: %s", err)
    rows = [{"cv_id": c, "job_id": j, "found_by": "+".join(sorted(w))} for (c, j), w in cands.items()]
    df = pd.DataFrame(rows)
    lab = test[["cv_id", "job_id", "label", "qwen_label"]].rename(columns={"label": "original_label"})
    return df.merge(lab, on=["cv_id", "job_id"], how="left")


def _review_rows():
    """Review rows in sheet order, with the CV owner (R1..R7) and the hidden source labels."""
    cands = _review_candidates()
    gold = sorted(gold_cv_ids(), key=_stable_order)
    reviewer = {c: f"R{i % N_REVIEWERS + 1}" for i, c in enumerate(gold)}
    cands["reviewer"] = cands.cv_id.map(reviewer)
    cands["order"] = (cands.cv_id + cands.job_id).map(_stable_order)
    return cands.sort_values(["reviewer", "cv_id", "order"]).drop(columns="order")


def _review_key():
    """The rows the shared workbook was built from (review_key.csv); recomputed if the file is missing."""
    import pandas as pd

    path = GOLD_DIR / "review_key.csv"
    if path.exists():
        return pd.read_csv(path, dtype=str)
    log.warning("%s not found; recomputing the review rows", path)
    return _review_rows()


def _cross_assignment(key) -> dict[str, str]:
    """cv_id -> second reviewer: reviewer R(k+1) also labels CROSS_PER_REVIEWER CVs of R(k)."""
    owner = key.drop_duplicates("cv_id").set_index("cv_id")["reviewer"]
    out = {}
    for k in range(1, N_REVIEWERS + 1):
        mine = sorted((c for c, r in owner.items() if r == f"R{k}"), key=lambda c: _stable_order(c + ":cross"))
        for c in mine[:CROSS_PER_REVIEWER]:
            out[c] = f"R{k % N_REVIEWERS + 1}"
    return out


def _add_review_tab(wb, name: str, part, jobs: dict) -> None:
    from openpyxl.styles import Font
    from openpyxl.worksheet.datavalidation import DataValidation

    sh = wb.create_sheet(name)
    sh.append(["cv_id", "job_id", "job_title", "job_start", "judgement", "note"])
    for c in sh[1]:
        c.font = Font(bold=True)
    for row in part.itertuples(index=False):
        job = jobs[row.job_id]
        sh.append([row.cv_id, row.job_id, job.get("title", ""), job["text"][:160], None, None])
    dv = DataValidation(type="list", formula1=f'"{GOOD},{POTENTIAL},{NO}"', allow_blank=True)
    sh.add_data_validation(dv)
    dv.add(f"E2:E{len(part) + 1}")
    for col, width in zip("ABCDEF", (16, 16, 30, 70, 16, 30)):
        sh.column_dimensions[col].width = width
    sh.freeze_panes = "A2"


def _write_review_sheet() -> Path:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    cands = _review_rows()
    cands.to_csv(GOLD_DIR / "review_key.csv", index=False)   # labels kept out of the sheet
    gold = sorted(gold_cv_ids(), key=_stable_order)
    reviewer = cands.drop_duplicates("cv_id").set_index("cv_id")["reviewer"]

    test = _labelled_test_pairs()
    cv_text = test.drop_duplicates("cv_id").set_index("cv_id")["cv_text"]
    jobs = {j["id"]: j for j in load_jobs("eval")}
    wb = Workbook()
    ws = wb.active
    ws.title = "README"
    for line in [
        "SkillBridge gold test set - blind review",
        "",
        "Every row is one (CV, job) pair. Read the CV (sheet CVs) and the job (sheet Jobs),",
        "then pick a judgement: Good Fit / Potential Fit / No Fit.",
        "Good Fit = the student could apply and be shortlisted; Potential Fit = related but clear gaps;",
        "No Fit = wrong field or major requirements missing.",
        "Labels from the source datasets are hidden on purpose so they do not bias you.",
        f"{len(gold)} CVs, {len(cands)} rows, {N_REVIEWERS} reviewers (R1..R{N_REVIEWERS}). Fill only your own sheet.",
        "When done: save, then run  python -m skillbridge.data gold  (from src/).",
    ]:
        ws.append([line])
    ws["A1"].font = Font(bold=True, size=13)
    ws.column_dimensions["A"].width = 110

    for r in sorted(cands.reviewer.unique()):
        _add_review_tab(wb, r, cands[cands.reviewer == r], jobs)

    for name, rows in (("CVs", [(c, reviewer[c], cv_text[c]) for c in gold]),
                       ("Jobs", [(j["id"], j.get("title", ""), j["text"]) for j in jobs.values()])):
        sh = wb.create_sheet(name)
        sh.append(["id", "reviewer" if name == "CVs" else "title", "text"])
        for row in rows:
            sh.append(list(row))
        sh.column_dimensions["A"].width, sh.column_dimensions["B"].width, sh.column_dimensions["C"].width = 16, 24, 140
        for cell in sh["C"]:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    path = GOLD_DIR / "review_sheet.xlsx"
    wb.save(path)
    return path


def _write_cross_sheet() -> Path:
    """Second-reviewer tabs (R<k>_cross) to import into the shared workbook as new sheets."""
    from openpyxl import Workbook
    from openpyxl.styles import Font

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    key = _review_key()
    cross = _cross_assignment(key)
    jobs = {j["id"]: j for j in load_jobs("eval")}
    rows = key[key.cv_id.isin(cross)].copy()
    rows["reviewer"] = rows.cv_id.map(cross)
    rows["order"] = (rows.cv_id + rows.job_id + ":cross").map(_stable_order)
    rows = rows.sort_values(["reviewer", "cv_id", "order"])

    wb = Workbook()
    ws = wb.active
    ws.title = "README_cross"
    for line in [
        "SkillBridge gold review - cross-check tabs",
        "",
        "Your R<k>_cross tab holds CVs that another reviewer also labels. Same rules and drop-down as your own tab.",
        "Find the CV by its cv_id in the CVs sheet (Ctrl+F); its reviewer column shows the other person, that is expected.",
        "Do not open the other reviewer's tab and do not discuss these rows: the two judgements must be independent.",
        "Your own R<k> tab is the gold label; these rows only measure how often two reviewers agree.",
    ]:
        ws.append([line])
    ws["A1"].font = Font(bold=True, size=13)
    ws.column_dimensions["A"].width = 110
    for r in sorted(rows.reviewer.unique()):
        _add_review_tab(wb, f"{r}_cross", rows[rows.reviewer == r], jobs)
    path = GOLD_DIR / "review_cross.xlsx"
    wb.save(path)
    return path


def _check_tabs(sheets: dict, pattern: str, expected, strict: bool):
    """Validate the tabs whose name matches `pattern` against the rows they were built with.

    Deleted, added, duplicated or moved rows, edited ids, renamed columns and
    values typed outside the drop-down all corrupt the gold set silently, so they
    are errors. Unjudged rows are errors only when `strict`.
    Returns (rows [cv_id, job_id, judgement, tab], errors).
    """
    import pandas as pd

    def clean(v):
        return v.strip() if isinstance(v, str) and v.strip() else None

    def show(pairs):
        return ", ".join(f"{c}/{j}" for c, j in sorted(pairs)[:3])

    errors, parts = [], []
    tabs = {n: df for n, df in sheets.items() if re.fullmatch(pattern, n)}
    for name in sorted(set(expected.tab) - set(tabs)):
        errors.append(f"{name}: tab missing (renamed or deleted?)")
    for name, df in sorted(tabs.items()):
        lost = {"cv_id", "job_id", "judgement"} - set(df.columns)
        if lost:
            errors.append(f"{name}: column(s) {sorted(lost)} not found (renamed?)")
            continue
        df = df[["cv_id", "job_id", "judgement"]].astype(object).map(clean)
        df = df[df.cv_id.notna() | df.job_id.notna()].assign(tab=name)
        got = set(df[["cv_id", "job_id"]].itertuples(index=False, name=None))
        want = set(expected.loc[expected.tab == name, ["cv_id", "job_id"]].itertuples(index=False, name=None))
        if want - got:
            errors.append(f"{name}: {len(want - got)} row(s) deleted or ids edited, e.g. {show(want - got)}")
        if got - want:
            errors.append(f"{name}: {len(got - want)} unknown row(s), e.g. {show(got - want)}")
        if df.duplicated(["cv_id", "job_id"]).any():
            errors.append(f"{name}: {int(df.duplicated(['cv_id', 'job_id']).sum())} duplicated row(s)")
        bad = df.judgement.notna() & ~df.judgement.isin(LABELS)
        if bad.any():
            errors.append(f"{name}: {int(bad.sum())} invalid judgement(s), e.g. {df.judgement[bad].iloc[0]!r}")
        empty = int(df.judgement.isna().sum())
        if empty and strict:
            errors.append(f"{name}: {empty} row(s) not judged yet")
        elif empty:
            log.warning("%s: %d row(s) not judged yet; left out of the agreement", name, empty)
        parts.append(df)
    rows = pd.concat(parts) if parts else pd.DataFrame(columns=["cv_id", "job_id", "judgement", "tab"])
    return rows, errors


def _kappa(a: list, b: list) -> float:
    """Cohen's kappa for two raters over the same items."""
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(k) / n) * (b.count(k) / n) for k in set(a) | set(b))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def _agreement_report(gold, second, key) -> str:
    def compare(title, x, y):
        x, y = list(x), list(y)
        if not x:
            return f"- {title}: no overlapping pairs."
        acc = sum(p == q for p, q in zip(x, y)) / len(x)
        good_x, good_y = [v == GOOD for v in x], [v == GOOD for v in y]
        return (f"- {title}: {len(x)} pairs, exact agreement {acc:.1%}, Cohen's kappa {_kappa(x, y):.2f} "
                f"(Good Fit vs rest: {_kappa(good_x, good_y):.2f}).")

    per_cv = gold.assign(good=gold.judgement == GOOD).groupby("cv_id").good.sum()
    before = key.assign(good=key.original_label == GOOD).groupby("cv_id").good.sum()
    lines = [
        "# Gold review summary",
        "",
        f"- Gold labels: {len(gold)} pairs, {gold.cv_id.nunique()} CVs; {gold.judgement.value_counts().to_dict()}.",
        f"- Good Fit jobs per CV: mean {per_cv.mean():.2f} after review, {before.mean():.2f} in the original labels. "
        f"CVs with no Good Fit after review: {int((per_cv == 0).sum())} (they have no relevant job for Recall/MRR).",
    ]
    if second is not None:
        both = gold.merge(second.dropna(subset=["judgement"]), on=["cv_id", "job_id"], suffixes=("", "_2"))
        lines.append(compare(f"Two reviewers on the cross-check CVs ({both.cv_id.nunique()} CVs)",
                             both.judgement, both.judgement_2))
    src = gold.merge(key[["cv_id", "job_id", "original_label", "qwen_label"]], on=["cv_id", "job_id"])
    orig = src.dropna(subset=["original_label"])
    qwen = src.dropna(subset=["qwen_label"])
    lines.append(compare("Team vs original cnamuangtoun labels", orig.judgement, orig.original_label))
    lines.append(compare("Team vs Qwen2.5-32B labels (manohar)", qwen.judgement, qwen.qwen_label))
    return "\n".join(lines) + "\n"


def _sheet_to_gold(path: Path) -> Path:
    import pandas as pd

    sheets = pd.read_excel(path, sheet_name=None, dtype=str)
    key = _review_key()
    gold, errors = _check_tabs(sheets, r"R\d+", key.assign(tab=key.reviewer), strict=True)
    second = None
    if any(re.fullmatch(r"R\d+_cross", n) for n in sheets):
        cross = _cross_assignment(key)
        expected = key[key.cv_id.isin(cross)]
        expected = expected.assign(tab=expected.cv_id.map(cross) + "_cross")
        second, more = _check_tabs(sheets, r"R\d+_cross", expected, strict=False)
        errors += more
    else:
        log.warning("no R*_cross tabs in %s; reviewer agreement not computed", path)
    if errors:
        raise SystemExit("review sheet has problems, nothing written:\n  " + "\n  ".join(errors))
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    out = _gold_labels_file()
    gold[["cv_id", "job_id", "judgement"]].sort_values(["cv_id", "job_id"]).to_csv(out, index=False)
    report = _agreement_report(gold, second, key)
    (GOLD_DIR / "agreement.md").write_text(report, encoding="utf-8")
    print(report)
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="python -m skillbridge.data", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("report", help="dataset statistics -> data/dataset_report.md")
    sub.add_parser("export", help="cleaned job pools -> data/jobs_<pool>.jsonl")
    sub.add_parser("review-sheet", help="blind review workbook for the gold set -> data/gold/")
    sub.add_parser("cross-sheet", help="second-reviewer tabs -> data/gold/review_cross.xlsx")
    g = sub.add_parser("gold", help="filled review workbook -> data/gold/gold_labels.csv + agreement.md")
    g.add_argument("--sheet", type=Path, default=GOLD_DIR / "review_sheet.xlsx")
    args = ap.parse_args(argv)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.cmd == "report":
        text = _report()
        (DATA_DIR / "dataset_report.md").write_text(text, encoding="utf-8")
        print(text)
    elif args.cmd == "export":
        for pool in ("eval", "extended"):
            path = DATA_DIR / f"jobs_{pool}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for job in load_jobs(pool):
                    f.write(json.dumps(job, ensure_ascii=False) + "\n")
            print("wrote", path)
    elif args.cmd == "review-sheet":
        print("wrote", _write_review_sheet(), "and", GOLD_DIR / "review_key.csv")
    elif args.cmd == "cross-sheet":
        print("wrote", _write_cross_sheet())
    elif args.cmd == "gold":
        print("wrote", _sheet_to_gold(args.sheet))


if __name__ == "__main__":
    main()
