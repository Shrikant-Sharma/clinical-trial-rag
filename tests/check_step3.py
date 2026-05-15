"""
Smoke test for Step 3 of the LangGraph agent build.

Verifies:
1. _retrieve_node returns documents and a numeric top_score for a real
   in-scope query (integration test against FAISS + PubMedBERT).
2. _route_after_retrieve routes correctly based on top_score relative to
   DEFAULT_THRESHOLD (synthetic states; deterministic routing logic).
3. _refuse_node attributes a threshold-triggered refusal correctly.

Design note: Test 2 uses synthetic states rather than real OOD queries.
PubMedBERT has a high cosine-similarity baseline (most natural-language
queries score 0.6-0.9 regardless of semantic relevance), so finding an
OOD query that reliably scores below the 0.50 threshold is fragile.
The threshold gate exists for catastrophic OOD only; the LLM grader
added in Step 4 is the real defense for borderline cases.

Run from the repo root with PYTHONPATH set:
    PowerShell: $env:PYTHONPATH = "src"; python tests/check_step3.py
    Bash:       PYTHONPATH=src python tests/check_step3.py
"""
from rag import load_model, load_index, load_chunks, DEFAULT_THRESHOLD
from agent import _retrieve_node, _route_after_retrieve, _refuse_node


def main():
    print("Loading deps...")
    model = load_model()
    index = load_index()
    chunks = load_chunks()
    print(f"  Loaded {len(chunks)} chunks.\n")

    # ---- Test 1: real in-scope retrieval works end-to-end ----
    print("Test 1: in-scope clinical query (integration test)")
    state = {"current_query": "What are the side effects of pembrolizumab?"}
    state.update(_retrieve_node(state, model, index, chunks))
    print(f"  top_score: {state['top_score']:.3f}")
    print(f"  docs returned: {len(state['documents'])}")
    if state["documents"]:
        print(f"  top doc NCT: {state['documents'][0]['nct_id']}")
    assert state["top_score"] > DEFAULT_THRESHOLD, \
        f"in-scope query should exceed threshold; got {state['top_score']:.3f}"
    assert len(state["documents"]
               ) > 0, "retrieve should return at least one document"
    print("  PASS\n")

    # ---- Test 2: routing logic (synthetic states) ----
    print("Test 2: routing logic with synthetic states")

    low_state = {"top_score": 0.10}
    route_low = _route_after_retrieve(low_state)
    print(f"  top_score=0.10 -> route={route_low}")
    assert route_low == "refuse", f"expected 'refuse' below threshold, got '{route_low}'"

    high_state = {"top_score": 0.95}
    route_high = _route_after_retrieve(high_state)
    print(f"  top_score=0.95 -> route={route_high}")
    assert route_high == "grade", f"expected 'grade' above threshold, got '{route_high}'"

    boundary_state = {"top_score": DEFAULT_THRESHOLD}
    route_boundary = _route_after_retrieve(boundary_state)
    print(
        f"  top_score={DEFAULT_THRESHOLD} (boundary) -> route={route_boundary}")
    assert route_boundary == "grade", \
        f"boundary value should not trigger refuse, got '{route_boundary}'"
    print("  PASS\n")

    # ---- Test 3: refuse_node attributes the refusal correctly ----
    print("Test 3: refuse_node attribution for threshold refusal")
    refuse_state = {
        "top_score": 0.10,
        "relevance": None,
        "retry_count": 0,
    }
    result = _refuse_node(refuse_state)
    print(f"  answer: {result['answer'][:60]}...")
    print(f"  refused_at: {result['refused_at']}")
    print(f"  sources: {result['sources']}")
    assert result["refused_at"] == "threshold", \
        f"expected 'threshold', got '{result['refused_at']}'"
    print("  PASS\n")

    print("All Step 3 checks passed.")


if __name__ == "__main__":
    main()
