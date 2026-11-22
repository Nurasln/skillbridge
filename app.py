"""SkillBridge Streamlit interface.

Owner: Frontend / UI Developer
Run: streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from skillbridge.explain import explain  # noqa: E402
from skillbridge.rerank import rerank  # noqa: E402

# Placeholder jobs until data.load_jobs() is ready.
SAMPLE_JOBS = [
    {"id": "j1", "text": "Junior Python developer. Pandas, SQL, REST APIs."},
    {"id": "j2", "text": "Data analyst intern. Excel, SQL, dashboards."},
    {"id": "j3", "text": "Frontend developer. React, TypeScript, CSS."},
]

st.title("SkillBridge")
st.caption("Explainable CV–job matching")

cv_text = st.text_area("Paste your CV text", height=200)

if st.button("Find matching jobs") and cv_text.strip():
    for job in rerank(cv_text, SAMPLE_JOBS):
        result = explain(cv_text, job["text"])
        with st.container(border=True):
            st.markdown(f"**{job['id']}** · score {job['score']:.2f}")
            st.write(job["text"])
            st.write(result["summary"])
