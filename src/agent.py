"""
Clinical Trial RAG — agentic orchestration with LangGraph.

Wraps the retrieve + generate primitives from rag.py in a self-corrective
RAG agent (CRAG pattern). The agent retrieves, grades document relevance,
rewrites the query and retries on poor retrieval, and falls back to an
honest refusal when retries are exhausted.

Design choices:
- Core retrieval (FAISS + PubMedBERT) and generation (Groq Llama 3.3 70B)
  live in rag.py and are LangChain-free. LangGraph only orchestrates.
- The two-layer safety from rag.generate() is preserved: the similarity
  threshold catches catastrophic OOD queries; the LLM refusal clause is the
  backstop. This file adds a third layer: relevance grading with retry.
- All refusal paths set state["refused_at"] to a specific gate name, so
  downstream telemetry can attribute refusals: "threshold" (cosine too low),
  "max_retries_exhausted" (grader rejected docs after final retry).

Public API:
    AgentState            TypedDict schema for graph state.
    build_agent(...)      Returns a compiled LangGraph agent with deps baked in.
    run_agent(query, ...) Convenience wrapper around build_agent + invoke.
"""

from typing import TypedDict, Optional

from rag import retrieve, DEFAULT_THRESHOLD, build_context, generate_answer, LLM_MODEL


# ---- Config ----
MAX_RETRIES = 1
# One rewrite is enough to recover from vague-question failures, which are
# the dominant cause of poor retrieval. Beyond one, the question is almost
# always genuinely out of scope and refusal is the right response, not more
# retries. Increasing this trades latency and LLM cost for marginal recall.


# ---- Prompts ----
GRADER_PROMPT = """You are grading whether retrieved clinical trial excerpts \
are relevant to a user's question. Reply with exactly one word: "relevant" or \
"not_relevant".

A document is relevant if it contains information that could be cited to answer \
the question, even partially. A document is not_relevant if it is on a different \
medical topic, a different trial design, or otherwise cannot support an answer.

Question: {query}

Retrieved excerpts:
{context}

Reply with one word only."""


REWRITER_PROMPT = """You rewrite a clinical trial search query to improve \
retrieval. The previous query did not return relevant results.

Rewrite rules:
1. Keep the medical intent of the original question.
2. Add specific clinical vocabulary (drug names, trial phases, disease \
   subtypes) if the original was vague.
3. Remove conversational filler ("what about", "tell me", etc.).
4. Output the rewritten query as a single line. No explanation.

Original question: {original_query}
Previous query (did not work): {current_query}

Rewritten query:"""


# ---- State schema ----
class AgentState(TypedDict):
    """
    State passed between nodes in the agent graph.

    Lifecycle:
        Entry             caller sets original_query, current_query, retry_count=0
        After retrieve    populates documents, top_score
        After grade       populates relevance
        After rewrite     updates current_query, increments retry_count
        After generate    populates answer, sources (success path)
        After refuse      populates answer, sets refused_at (failure paths)
    """
    original_query: str            # User's original question. Never modified.
    # Active query for retrieval. Rewritten on retry.
    current_query: str
    # Output of rag.retrieve(). Empty before first hop.
    documents: list[dict]
    top_score: float               # Cosine similarity of top chunk.
    # "relevant" | "not_relevant" | None (pre-grade).
    relevance: Optional[str]
    retry_count: int               # Number of rewrites performed.
    answer: str                    # Final answer or refusal message.
    sources: list[dict]            # Cited chunks. Empty on refusal.
    refused_at: Optional[str]      # None on success; gate name on refusal.


# ---- Nodes ----
def _retrieve_node(state: AgentState, model, index, chunks) -> dict:
    """
    Retrieve top-k documents for the current query.

    Wraps rag.retrieve() and adds the retrieved documents and the top
    cosine score to state. Does NOT decide whether to refuse — that
    decision belongs to the conditional edge below.

    Note: deps (model, index, chunks) come in as extra args. build_agent()
    in a later step binds them via functools.partial so LangGraph sees
    a 1-arg callable.
    """
    results = retrieve(state["current_query"], model, index, chunks)
    top_score = results[0]["score"] if results else 0.0
    return {"documents": results, "top_score": top_score}


def _refuse_node(state: AgentState) -> dict:
    """
    Terminal node producing the refusal answer.

    Attributes the refusal to a specific gate by examining state:
        top_score below threshold                  → "threshold"
        relevance not_relevant + retries exhausted → "max_retries_exhausted"
        otherwise (shouldn't happen)               → "unknown"

    Reuses the canonical refusal phrase from rag.SYSTEM_PROMPT so users
    see a consistent "I don't know" voice regardless of which gate fired.
    """
    if state["top_score"] < DEFAULT_THRESHOLD:
        refused_at = "threshold"
    elif state.get("relevance") == "not_relevant" and state["retry_count"] >= MAX_RETRIES:
        refused_at = "max_retries_exhausted"
    else:
        refused_at = "unknown"

    return {
        "answer": "I don't have enough information in the retrieved trials to answer that question.",
        "sources": [],
        "refused_at": refused_at,
    }


# ---- Conditional edges ----
def _route_after_retrieve(state: AgentState) -> str:
    """
    First gate: refuse if retrieval similarity is below the OOD threshold.

    Preserves Layer 1 safety from the original rag.generate(): catastrophic
    OOD queries get caught here, before any LLM call. Pure routing — no
    state mutation; _refuse_node infers the gate from state.
    """
    if state["top_score"] < DEFAULT_THRESHOLD:
        return "refuse"
    return "grade"


def _generate_node(state: AgentState, client) -> dict:
    """
    Generate the final answer from retrieved documents.

    Wraps rag.generate_answer() so the agentic and baseline paths share
    the same LLM call. Returns answer + sources; refused_at stays None
    because this node is reached only on the success path.
    """
    answer = generate_answer(
        state["current_query"], state["documents"], client)
    return {
        "answer": answer,
        "sources": state["documents"],
        "refused_at": None,
    }


def _grade_documents(state: AgentState, client) -> dict:
    """
    Grade retrieved documents for relevance to the current query.

    Calls the LLM with GRADER_PROMPT and parses the one-word response.
    Sets state["relevance"] to "relevant" or "not_relevant". On unexpected
    LLM output, defaults to "relevant" (fail-open) so a flaky grader does
    not spuriously refuse a valid query — the threshold gate and the LLM
    refusal clause in generate_answer() are independent backstops.
    """
    context = build_context(state["documents"])
    prompt = GRADER_PROMPT.format(
        query=state["current_query"], context=context)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,   # Deterministic for grading.
        max_tokens=10,     # One word answer.
    )
    output = response.choices[0].message.content.strip().lower()

    # Normalize: check "not_relevant" before "relevant" since both contain
    # the substring "relevant".
    if "not_relevant" in output or "not relevant" in output:
        relevance = "not_relevant"
    elif "relevant" in output:
        relevance = "relevant"
    else:
        # Defensive default: fail-open on unexpected LLM output.
        relevance = "relevant"

    return {"relevance": relevance}
