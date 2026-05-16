"""
Smoke test for Step 5 of the LangGraph agent build.

Verifies:
1. _route_after_grade routes correctly for all three branches
   (relevant -> generate; not_relevant + retries left -> rewrite;
   not_relevant + retries exhausted -> refuse).
2. _rewrite_query produces a non-empty rewritten query that differs
   from the input, and increments retry_count.
3. _generate_node detects the canonical refusal phrase and surfaces
   refused_at = "generation" with sources cleared.

Run from the repo root with PYTHONPATH set:
    PowerShell: $env:PYTHONPATH = "src"; python tests/check_step5.py
    Bash:       PYTHONPATH=src python tests/check_step5.py
"""
from rag import load_model, load_index, load_chunks, make_groq_client
from agent import (
    MAX_RETRIES,
    _retrieve_node,
    _generate_node,
    _rewrite_query,
    _route_after_grade,
)


def main():
    print("Loading deps...")
    model = load_model()
    index = load_index()
    chunks = load_chunks()
    client = make_groq_client()
    print(f"  Loaded {len(chunks)} chunks + Groq client.\n")

    # ---- Test 1: _route_after_grade routing (synthetic states) ----
    print("Test 1: _route_after_grade three-way routing")

    state = {"relevance": "relevant", "retry_count": 0}
    route = _route_after_grade(state)
    print(f"  relevant -> route={route}")
    assert route == "generate", f"expected 'generate', got '{route}'"

    state = {"relevance": "not_relevant", "retry_count": 0}
    route = _route_after_grade(state)
    print(f"  not_relevant, retries=0 -> route={route}")
    assert route == "rewrite", f"expected 'rewrite', got '{route}'"

    state = {"relevance": "not_relevant", "retry_count": MAX_RETRIES}
    route = _route_after_grade(state)
    print(f"  not_relevant, retries={MAX_RETRIES} -> route={route}")
    assert route == "refuse", f"expected 'refuse', got '{route}'"

    print("  PASS\n")

    # ---- Test 2: _rewrite_query sharpens a vague query ----
    print("Test 2: _rewrite_query on a vague query")
    state = {
        "original_query": "tell me about pembrolizumab",
        "current_query": "tell me about pembrolizumab",
        "retry_count": 0,
    }
    result = _rewrite_query(state, client)
    print(f"  original:  '{state['current_query']}'")
    print(f"  rewritten: '{result['current_query']}'")
    print(f"  retry_count: {result['retry_count']}")
    assert result["retry_count"] == 1, f"retry_count should be 1"
    assert len(result["current_query"]
               ) > 0, "rewritten query should not be empty"
    assert result["current_query"] != state["current_query"], \
        "rewriter should produce a different query"
    print("  PASS\n")

    # ---- Test 3: _generate_node detects generator-level refusal ----
    print("Test 3: _generate_node detects canonical refusal")
    # The side-effects question retrieves on-topic docs (protocols mention
    # pembrolizumab) but the protocols don't contain side-effect findings
    # (those live in trial results). The LLM should use the refusal clause
    # from rag.SYSTEM_PROMPT, and _generate_node should mark refused_at
    # accordingly.
    state = {"current_query": "What are the side effects of pembrolizumab?"}
    state.update(_retrieve_node(state, model, index, chunks))
    result = _generate_node(state, client)
    print(f"  answer:     {result['answer'][:80]}...")
    print(f"  refused_at: {result['refused_at']}")
    print(f"  sources:    {len(result['sources'])}")
    assert result["refused_at"] == "generation", (
        f"expected refused_at='generation', got '{result['refused_at']}'. "
        "If this fails, the LLM may be answering using general medical "
        "knowledge instead of refusing — investigate whether the system "
        "prompt's refusal clause is still active in rag.SYSTEM_PROMPT."
    )
    assert result["sources"] == [], "refused answer should drop sources"
    print("  PASS\n")

    print("All Step 5 checks passed.")


if __name__ == "__main__":
    main()
