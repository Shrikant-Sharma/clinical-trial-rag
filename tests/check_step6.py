"""
End-to-end smoke test for Step 6 of the LangGraph agent build.

Verifies:
1. build_agent() compiles without error.
2. In-scope clinical query routes through retrieve -> grade -> generate
   and returns a substantive answer with sources.
3. On-topic-but-unanswerable query (side effects vs. protocol-only corpus)
   routes through retrieve -> grade -> generate and surfaces generator
   refusal via refused_at = "generation".
4. The agent terminates on degenerate input within MAX_RETRIES bound (no
   infinite loops, no crash). Specific refused_at value depends on the
   embedding's behavior for the input, so we assert termination only.

Run from the repo root with PYTHONPATH set:
    PowerShell: $env:PYTHONPATH = "src"; python tests/check_step6.py
"""
from rag import load_model, load_index, load_chunks, make_groq_client
from agent import build_agent, run_agent, MAX_RETRIES


def main():
    print("Loading deps and building agent...")
    model = load_model()
    index = load_index()
    chunks = load_chunks()
    client = make_groq_client()
    graph = build_agent(client, model, index, chunks)
    print(f"  Loaded {len(chunks)} chunks; graph compiled.\n")

    # ---- Test 1: in-scope query → end-to-end success ----
    print("Test 1: in-scope clinical query, full success path")
    query = "What conditions are pembrolizumab clinical trials targeting?"
    result = run_agent(query, graph)
    print(f"  query:      '{query}'")
    print(f"  top_score:  {result['top_score']:.3f}")
    print(f"  relevance:  {result['relevance']}")
    print(f"  retries:    {result['retry_count']}")
    print(f"  refused_at: {result['refused_at']}")
    print(f"  sources:    {len(result['sources'])}")
    print(f"  answer:     {result['answer'][:120]}...")
    assert result["refused_at"] is None, \
        f"expected refused_at=None, got '{result['refused_at']}'"
    assert len(result["sources"]) > 0, "success path should carry sources"
    assert len(result["answer"]) > 50, "answer should be substantive"
    print("  PASS\n")

    # ---- Test 2: on-topic but unanswerable → generator refusal ----
    print("Test 2: on-topic-but-unanswerable, generator refusal path")
    query = "What are the side effects of pembrolizumab?"
    result = run_agent(query, graph)
    print(f"  query:      '{query}'")
    print(f"  top_score:  {result['top_score']:.3f}")
    print(f"  relevance:  {result['relevance']}")
    print(f"  retries:    {result['retry_count']}")
    print(f"  refused_at: {result['refused_at']}")
    print(f"  sources:    {len(result['sources'])}")
    print(f"  answer:     {result['answer'][:120]}...")
    assert result["refused_at"] == "generation", \
        f"expected 'generation', got '{result['refused_at']}'"
    assert result["sources"] == [], "refused answer should drop sources"
    print("  PASS\n")

    # ---- Test 3: degenerate input → agent terminates ----
    # Demonstrates the agent doesn't loop forever and respects MAX_RETRIES.
    # The specific gate that catches degenerate input depends on embedding
    # behavior, so we assert only on termination + bounded retry count.
    print("Test 3: degenerate input, bounded termination")
    query = "asdf qwer zxcv lkjh"
    result = run_agent(query, graph)
    print(f"  query:      '{query}'")
    print(f"  top_score:  {result['top_score']:.3f}")
    print(f"  retries:    {result['retry_count']}")
    print(f"  refused_at: {result['refused_at']}")
    print(f"  answer:     {result['answer'][:120]}...")
    assert result["answer"], "agent should always return some answer"
    assert result["retry_count"] <= MAX_RETRIES, \
        f"retry_count {result['retry_count']} exceeded MAX_RETRIES={MAX_RETRIES}"
    print("  PASS\n")

    print("All Step 6 checks passed. Agent is wired and operational.")


if __name__ == "__main__":
    main()
