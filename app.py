"""
Clinical Trial RAG — Streamlit UI.

Two reasoning modes:
- Agentic (CRAG): LangGraph orchestration with document grading,
  query rewriting on poor retrieval, and bounded retries.
- Baseline (single-shot): original retrieve -> threshold -> generate flow.

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
from src.agent import build_agent, run_agent


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


@st.cache_resource(show_spinner="Compiling LangGraph agent...")
def _load_agent():
    """Build the agent once at app startup. Reuses the cached deps."""
    return build_agent(
        client=_load_groq_client(),
        model=_load_model(),
        index=_load_index(),
        chunks=_load_chunks(),
    )


# Load everything at startup.
model = _load_model()
index = _load_index()
chunks = _load_chunks()
client = _load_groq_client()
agent_graph = _load_agent()


# ---- UI ----
st.title("Clinical Trial Intelligence")
st.caption(
    "RAG over 484 ClinicalTrials.gov studies (oncology, diabetes, cardiovascular). "
    "PubMedBERT embeddings, FAISS retrieval, Groq Llama 3.3 70B, LangGraph orchestration."
)


# ---- Sidebar ----
with st.sidebar:
    st.header("Reasoning mode")
    mode = st.radio(
        "Mode",
        ["Agentic (CRAG)", "Baseline (single-shot)"],
        index=0,
        label_visibility="collapsed",
        help=(
            "Agentic: LangGraph with document grading and query rewriting "
            "on poor retrieval.\n\n"
            "Baseline: Single-shot retrieve -> threshold -> generate."
        ),
    )
    is_agentic = mode == "Agentic (CRAG)"

    st.divider()
    st.header("Baseline settings")
    if is_agentic:
        st.caption("These knobs apply only in Baseline mode.")
    k = st.slider(
        "Sources to retrieve (top-k)", 1, 10, DEFAULT_K,
        disabled=is_agentic,
    )
    threshold = st.slider(
        "Similarity threshold (Layer 1)",
        0.0, 1.0, DEFAULT_THRESHOLD, 0.05,
        disabled=is_agentic,
        help="Below this cosine similarity, queries are refused before the LLM is called.",
    )

    st.divider()
    st.header("Try these")
    examples = [
        "What conditions are pembrolizumab clinical trials targeting?",
        "BRAF mutation melanoma treatment options",
        "HER2 positive breast cancer trials",
        "Heart failure ejection fraction studies",
        "What are the side effects of pembrolizumab?",
        "How do I bake sourdough bread?",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["query"] = ex


# ---- Main search ----
query = st.text_input(
    "Ask a question about clinical trials:",
    value=st.session_state.get("query", ""),
    placeholder="e.g. What trials are studying BRAF mutations in melanoma?",
)

if query:
    spinner_text = (
        "Running agent (retrieve -> grade -> generate)..."
        if is_agentic
        else "Retrieving and generating..."
    )
    with st.spinner(spinner_text):
        if is_agentic:
            result = run_agent(query, agent_graph)
        else:
            result = generate(
                query, client, model, index, chunks, k=k, threshold=threshold,
            )

    # ---- Status row: similarity + outcome ----
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Top similarity", f"{result['top_score']:.3f}")
    with col2:
        refused_at = result.get("refused_at")
        if refused_at == "threshold":
            st.warning("Refused: similarity below threshold (Layer 1 OOD gate)")
        elif refused_at == "generation":
            st.warning(
                "Refused: retrieved trials don't actually contain the answer to "
                "this specific question (LLM refusal clause fired)"
            )
        elif refused_at == "max_retries_exhausted":
            st.warning(
                "Refused: grader rejected documents even after a query rewrite "
                "(LangGraph CRAG retry exhausted)"
            )
        elif refused_at is None and result.get("answer"):
            st.success("Answered with cited sources")
        else:
            # Defensive fallback for baseline-only string-matched refusals.
            ans = result.get("answer", "")
            if "outside the scope" in ans:
                st.warning("Refused: out-of-scope query")
            elif "don't have enough information" in ans:
                st.warning("Refused: LLM declined the question")
            else:
                st.success("Answered")

    # ---- Agent telemetry (agentic mode only) ----
    if is_agentic:
        retries = result.get("retry_count", 0)
        relevance = result.get("relevance")
        info_parts = []
        if relevance:
            info_parts.append(f"**Document relevance:** {relevance}")
        if retries > 0:
            info_parts.append(f"**Query rewrites:** {retries}")
        if info_parts:
            st.caption("  ·  ".join(info_parts))

    # ---- Answer ----
    st.subheader("Answer")
    st.write(result["answer"])

    # ---- Rewritten query (agentic mode, only if a rewrite happened) ----
    if is_agentic and result.get("retry_count", 0) > 0:
        with st.expander("Agent's rewritten query"):
            st.caption("Original question:")
            st.text(result.get("original_query", query))
            st.caption("Query after rewrite (used for final retrieval):")
            st.text(result.get("current_query", query))

    # ---- Sources ----
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
