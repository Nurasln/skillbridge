"""SkillBridge Streamlit interface.

Owner: Frontend / UI Developer
Run: streamlit run app.py
All matching logic lives in src/skillbridge/pipeline.py; this file only shows results.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from skillbridge.pipeline import Pipeline, read_cv  # noqa: E402


@st.cache_resource
def get_pipeline() -> Pipeline:
    return Pipeline()  # loads and indexes jobs once per server


st.title("SkillBridge")
st.caption("Explainable CV–job matching")

uploaded = st.file_uploader("Upload your CV (PDF or text)", type=["pdf", "txt"])
cv_text = st.text_area("...or paste your CV text", height=200)

if st.button("Find matching jobs"):
    text = read_cv(uploaded) if uploaded else cv_text
    if not text.strip():
        st.warning("Please upload or paste a CV.")
        st.stop()
    result = get_pipeline().match(text)
    if result.fallbacks:
        st.info("Placeholders in use: " + ", ".join(result.fallbacks))
    for rank, job in enumerate(result.matches, start=1):
        exp = job["explanation"]
        with st.container(border=True):
            st.markdown(f"**#{rank} · {job.get('title', job['id'])}** · score {job['score']:.3f}")
            st.write(job["text"][:500])
            st.write(exp["summary"])
            if exp["matched"]:
                st.markdown("✅ " + ", ".join(exp["matched"]))
            if exp["missing"]:
                st.markdown("❌ " + ", ".join(exp["missing"]))
    st.caption(" · ".join(f"{k} {v*1000:.0f} ms" for k, v in result.timings.items()))
