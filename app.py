"""
Clinical Trial RAG — Streamlit UI.

Run with: streamlit run app.py
"""

import streamlit as st
from src.rag import (
    load_chunks,
    load_index,
    load_model,
    make_groq_client,
    generate,
    DEFAULT_THRESHOLD,
    DEFAULT_K,
)


# ---- Page config ----
st.set_page_config(
    page_title="Clinical Trial Intelligence",
    page_icon="🔬",
    layout="wide",
)


# ---- One-time resource loading (cached across reruns) ----
@st.cache_resource(show_spinner="Loading PubMedBERT (~440MB, first run only)...")
def _load_model():
    return load_model()


@st.cache_resource(show_spinner="Loading FAISS index...")
def _load_index():
    return load_index()


@st.cache_resource(show_spinner="Loading 3,264 trial chunks...")
def _load_chunks():
    return load_chunks()


@st.cache_resource
def _load_groq_client():
    return make_groq_client()


# Load everything at startup
model = _load_model()
index = _load_index()
chunks = _load_chunks()
client = _load_groq_client()


# ---- UI ----
st.title("Clinical Trial Intelligence")
st.caption(
    "RAG over 484 ClinicalTrials.gov studies (oncology, diabetes, cardiovascular). "
    "PubMedBERT embeddings, FAISS retrieval, Groq Llama 3.3 70B with two-layer safety."
)

# Sidebar with knobs and example queries
with st.sidebar:
    st.header("Settings")
    k = st.slider("Sources to retrieve (top-k)", 1, 10, DEFAULT_K)
    threshold = st.slider(
        "Similarity threshold (Layer 1 safety)",
        0.0, 1.0, DEFAULT_THRESHOLD, 0.05,
        help="Below this cosine similarity, queries are refused before the LLM is called."
    )

    st.divider()
    st.header("Try these")
    examples = [
        "What trials are testing pembrolizumab for non-small cell lung cancer?",
        "BRAF mutation melanoma treatment options",
        "HER2 positive breast cancer trials",
        "Heart failure ejection fraction studies",
        "How do I bake sourdough bread?",  # OOD test
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["query"] = ex


# Main search
query = st.text_input(
    "Ask a question about clinical trials:",
    value=st.session_state.get("query", ""),
    placeholder="e.g. What trials are studying BRAF mutations in melanoma?",
)

if query:
    with st.spinner("Retrieving relevant trials and generating answer..."):
        result = generate(
            query, client, model, index, chunks, k=k, threshold=threshold
        )

    # Top score + refusal indicator
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Top similarity", f"{result['top_score']:.3f}")
    with col2:
        if result["refused_at"] == "threshold":
            st.warning(
                f"Refused at Layer 1 (threshold): top score below {threshold}")
        elif "outside the scope" in result["answer"]:
            st.warning("Refused at Layer 2 (LLM): out-of-scope query")
        elif "don't have enough information" in result["answer"]:
            st.warning(
                "Refused at Layer 2 (LLM): retrieved chunks don't answer the question")
        else:
            st.success("Answered with cited sources")

    # Answer
    st.subheader("Answer")
    st.write(result["answer"])

    # Sources (only shown when not refused at threshold)
    if result["sources"]:
        st.subheader("Sources retrieved")
        for src in result["sources"]:
            with st.expander(
                f"#{src['rank']}  {src['nct_id']}  (similarity: {src['score']:.3f})  —  {src['title'][:80]}"
            ):
                st.markdown(
                    f"**[View on ClinicalTrials.gov](https://clinicaltrials.gov/study/{src['nct_id']})**"
                )
                st.text(src["text"])
