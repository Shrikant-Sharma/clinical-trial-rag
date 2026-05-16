"""
Smoke test for Step 4 of the LangGraph agent build.

Verifies:
0. rag.generate() still works after the generate_answer() refactor
   (regression test for the rag.py extract).
1. _generate_node produces a non-empty answer string for an in-scope
   query with retrieved documents.
2. _grade_documents returns "relevant" when documents actually answer
   the query.
3. _grade_documents returns "not_relevant" when documents are clearly
   off-topic relative to the query.

Run from the repo root with PYTHONPATH set:
    PowerShell: $env:PYTHONPATH = "src"; python tests/check_step4.py
    Bash:       PYTHONPATH=src python tests/check_step4.py
"""
from rag import load_model, load_index, load_chunks, make_groq_client, generate
from agent import _retrieve_node, _generate_node, _grade_documents


def main():
    print("Loading deps...")
    model = load_model()
    index = load_index()
    chunks = load_chunks()
    client = make_groq_client()
    print(f"  Loaded {len(chunks)} chunks + Groq client.\n")

    # ---- Test 0: rag.generate() unchanged after refactor ----
    print("Test 0: rag.generate() regression check")
    result = generate(
        "What conditions are pembrolizumab clinical trials targeting?",
        client, model, index, chunks,
    )
    print(f"  answer (first 80 chars): {result['answer'][:80]}...")
    print(f"  sources returned: {len(result['sources'])}")
    print(f"  refused_at: {result['refused_at']}")
    assert result["refused_at"] is None, "in-scope query should not refuse"
    assert len(result["answer"]) > 0, "answer should not be empty"
    assert len(result["sources"]) > 0, "should have sources"
    print("  PASS\n")

    # ---- Test 1: _generate_node produces an answer ----
    print("Test 1: _generate_node end-to-end")
    state = {
        "current_query": "What conditions are pembrolizumab clinical trials targeting?"}
    state.update(_retrieve_node(state, model, index, chunks))
    result = _generate_node(state, client)
    print(f"  answer (first 80 chars): {result['answer'][:80]}...")
    print(f"  sources returned: {len(result['sources'])}")
    assert len(result["answer"]) > 0, "answer should not be empty"
    assert result["refused_at"] is None, "success path should have refused_at=None"
    assert len(result["sources"]) > 0, "sources should be carried through"
    print("  PASS\n")

    # ---- Test 2: grader recognizes relevant documents ----
    print("Test 2: _grade_documents with relevant docs")
    # Reuse state from Test 1 (real retrieved pembrolizumab chunks).
    grade_result = _grade_documents(state, client)
    print(f"  relevance: {grade_result['relevance']}")
    assert grade_result["relevance"] == "relevant", \
        f"expected 'relevant' for pembrolizumab query + pembrolizumab docs"
    print("  PASS\n")

    # ---- Test 3: grader recognizes irrelevant documents ----
    print("Test 3: _grade_documents with synthesized off-topic docs")
    irrelevant_state = {
        "current_query": "What are the side effects of pembrolizumab?",
        "documents": [
            {
                "rank": 1, "score": 0.5,
                "nct_id": "NCT00000000",
                "title": "Effect of dough hydration on artisan sourdough crumb structure",
                "text": "We tested three hydration levels (65%, 75%, 85%) and "
                        "measured crumb porosity. Higher hydration produced more "
                        "open crumb but required longer proofing times.",
            },
        ],
    }
    grade_result = _grade_documents(irrelevant_state, client)
    print(f"  relevance: {grade_result['relevance']}")
    assert grade_result["relevance"] == "not_relevant", \
        f"expected 'not_relevant' for pembrolizumab query + sourdough docs"
    print("  PASS\n")

    print("All Step 4 checks passed.")


if __name__ == "__main__":
    main()
