"""Smoke test: retrieve with/without cross-encoder reranker on a real query."""
from src.rag import load_model, load_index, load_chunks, load_reranker, retrieve

print("Loading deps...")
model = load_model()
index = load_index()
chunks = load_chunks()
print(f"  Loaded {len(chunks)} chunks.")
reranker = load_reranker()
print("  Loaded reranker.")
print()

# A query the README flagged as having retrieval ambiguity — perfect rerank target
query = "antibody-drug conjugates for HER2-positive breast cancer"
print(f"Query: {query}")
print()

# Baseline (no rerank)
r1 = retrieve(query, model, index, chunks, k=5)
print("=== BASELINE (no rerank) ===")
for r in r1:
    print(
        f"  rank {r['rank']} | cos {r['score']:.3f} | {r['nct_id']} | {r['title'][:60]}")

# With reranker
r2 = retrieve(query, model, index, chunks, k=5, reranker=reranker)
print()
print("=== WITH RERANKER ===")
for r in r2:
    rerank_score = r.get("rerank_score")
    rerank_str = f" | rerank {rerank_score:+.3f}" if rerank_score is not None else ""
    print(
        f"  rank {r['rank']} | cos {r['score']:.3f}{rerank_str} | {r['nct_id']} | {r['title'][:60]}")

# Comparison
nct1 = [r["nct_id"] for r in r1]
nct2 = [r["nct_id"] for r in r2]
print()
print(f"Baseline NCT order: {nct1}")
print(f"Rerank NCT order:   {nct2}")
print(f"Same order: {nct1 == nct2}")
print(f"Same set:   {set(nct1) == set(nct2)}")

# Sanity: rerank scores should be in descending order
if r2 and "rerank_score" in r2[0]:
    scores = [r["rerank_score"] for r in r2]
    is_sorted = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    print(f"Rerank scores descending: {is_sorted}")
