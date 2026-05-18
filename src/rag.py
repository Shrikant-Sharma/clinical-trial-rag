"""
Clinical Trial RAG — core retrieval and generation logic.

Loads pre-built FAISS index and chunks from disk, exposes retrieve()
and generate() for use by the Streamlit app.
"""

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv(override=True)

# ---- Config ----
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
LLM_MODEL = "llama-3.3-70b-versatile"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_THRESHOLD = 0.50
DEFAULT_K = 5
FETCH_MULTIPLIER = 4

SYSTEM_PROMPT = """You are a clinical trial research assistant. You answer questions \
strictly using the trial excerpts provided in the context. Follow these rules:

1. If the context does not contain enough information to answer, say:
   "I don't have enough information in the retrieved trials to answer that question."
   Do not guess. Do not use general medical knowledge.

2. Cite every claim by NCT ID in square brackets, e.g. [NCT01234567]. If a claim
   is supported by multiple trials, cite all of them.

3. If the question is not about clinical trials or medical research, say:
   "This question is outside the scope of the clinical trial database."

4. Be concise. Two to four sentences unless the question requires more.
"""


# ---- One-time loaders (called once at app startup) ----
def load_chunks():
    """Load the chunks list from disk."""
    with open(os.path.join(DATA_DIR, "chunks.pkl"), "rb") as f:
        return pickle.load(f)


def load_index():
    """Load the FAISS index from disk."""
    return faiss.read_index(os.path.join(DATA_DIR, "faiss.index"))


def load_model():
    """Load the PubMedBERT embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL)


def load_reranker():
    """Load the cross-encoder reranker used for second-stage scoring.

    Caches at app level via @st.cache_resource in app.py. First call
    downloads ~80MB to ~/.cache/huggingface/hub/.
    """
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER_MODEL)


def make_groq_client():
    """Create a Groq client. Reads GROQ_API_KEY from environment."""
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


# ---- Retrieval and generation ----
def retrieve(query, model, index, chunks, k=DEFAULT_K, fetch_multiplier=FETCH_MULTIPLIER, reranker=None):
    """
    Encode query, fetch (k * fetch_multiplier) chunks from FAISS, optionally
    rerank with a cross-encoder, dedup by nct_id, return top-k unique sources.

    When `reranker` is None the function behaves identically to the original
    single-stage retrieval. When a cross-encoder is provided, all fetched
    candidates are rescored BEFORE dedup so the best chunk per trial (per the
    cross-encoder) is the one that survives, not just the best FAISS chunk.
    The cosine `score` field is preserved for the agent's threshold gate.
    """
    query_vec = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    fetch_k = k * fetch_multiplier
    distances, indices = index.search(query_vec, fetch_k)

    # Materialize all candidates with cosine similarity (no dedup yet).
    candidates = []
    for dist, idx in zip(distances[0], indices[0]):
        chunk = chunks[idx]
        cosine_sim = 1 - (dist / 2)
        candidates.append({
            "score": float(cosine_sim),
            "nct_id": chunk["nct_id"],
            "title": chunk["title"],
            "text": chunk["text"],
        })

    # Optional cross-encoder rerank BEFORE dedup. Reranking after dedup
    # would only re-order the already-chosen chunks; reranking before lets
    # the cross-encoder choose which chunk per trial is most relevant.
    if reranker is not None:
        candidates = rerank(query, candidates, reranker)

    # Dedup by nct_id, walking in current sort order (FAISS or reranked).
    seen_nct_ids = set()
    results = []
    for cand in candidates:
        if cand["nct_id"] in seen_nct_ids:
            continue
        seen_nct_ids.add(cand["nct_id"])
        cand["rank"] = len(results) + 1
        results.append(cand)
        if len(results) >= k:
            break
    return results


def rerank(query: str, candidates: list, reranker) -> list:
    """Re-score retrieved candidates against the query with a cross-encoder.

    Each candidate dict must have a 'text' field. Adds a 'rerank_score'
    field (float) to every candidate and returns the list sorted by
    rerank_score descending. The cross-encoder reads both query and
    document together, which catches relevance that bi-encoder cosine
    similarity dilutes (this is the whole point of two-stage retrieval).
    """
    if not candidates:
        return candidates
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)


def build_context(results):
    """Concatenate retrieved chunks with source markers for the LLM."""
    blocks = [f"[{r['nct_id']}] {r['title']}\n{r['text']}" for r in results]
    return "\n\n---\n\n".join(blocks)


def generate_answer(query, results, client):
    """
    Call Groq with retrieved context and the system prompt. Returns just
    the answer text. Does NOT include retrieval, threshold checking, or
    refusal logic — those are the caller's responsibility.

    Shared between rag.generate() (baseline single-shot RAG) and
    agent._generate_node (agentic path). Centralizing the LLM call here
    means a future tweak to temperature, max_tokens, or the system prompt
    flows to both paths automatically.
    """
    context = build_context(results)
    user_message = f"Context:\n\n{context}\n\n---\n\nQuestion: {query}"
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=400,
    )
    return response.choices[0].message.content


def generate(query, client, model, index, chunks, k=DEFAULT_K, threshold=DEFAULT_THRESHOLD, reranker=None):
    """
    Full RAG pipeline: retrieve -> threshold check (Layer 1) ->
    Groq generation with refusal clause (Layer 2).

    Returns dict with: answer, sources, top_score, refused_at.
    """
    results = retrieve(query, model, index, chunks, k=k, reranker=reranker)
    top_score = results[0]["score"] if results else 0.0

    # Layer 1: similarity threshold (catches catastrophic OOD)
    if top_score < threshold:
        return {
            "answer": "I don't have enough information in the retrieved trials to answer that question.",
            "sources": [],
            "top_score": top_score,
            "refused_at": "threshold",
        }

    # Layer 2: LLM with refusal clause in system prompt
    answer = generate_answer(query, results, client)
    canonical_refusal = "I don't have enough information in the retrieved trials"
    is_refusal = canonical_refusal in answer
    return {
        "answer": answer,
        "sources": [] if is_refusal else results,
        "top_score": top_score,
        "refused_at": "generation" if is_refusal else None,
    }
