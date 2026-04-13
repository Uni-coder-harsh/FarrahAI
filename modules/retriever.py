"""
FarrahAI — Module 5: Retrieval
================================
Retrieves the most relevant note chunks for a student's question.
This is the core of the offline RAG system.

Flow:
  query → embed query → FAISS similarity search → top-k chunks → return
"""

import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def retrieve(query: str,
             subject: str,
             index_dir: str,
             top_k: int = 5,
             model_name: str = "all-MiniLM-L6-v2") -> list[dict]:
    """
    Retrieve top-k relevant note chunks for a query.

    Args:
        query: student's question
        subject: subject room name
        index_dir: path to FAISS indexes
        top_k: number of chunks to retrieve
        model_name: embedding model

    Returns:
        list of { 'chunk_id', 'text', 'score', 'rank' }
    """
    from modules.embedder import embed_query, load_index

    # Load index for this subject
    index, chunks = load_index(index_dir, subject)

    # Embed the query
    query_vec = embed_query(query, model_name=model_name)

    # Search FAISS
    scores, indices = index.search(query_vec, top_k)

    # Build results
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx == -1:   # FAISS returns -1 for empty slots
            continue
        chunk = chunks[idx].copy()
        chunk["score"] = round(float(score), 4)
        chunk["rank"]  = rank + 1
        results.append(chunk)

    logger.info(f"Retrieved {len(results)} chunks for query: '{query[:60]}...'")
    return results


def format_retrieved_context(results: list[dict]) -> str:
    """
    Format retrieved chunks into a context string for the LLM.
    """
    if not results:
        return "No relevant notes found."

    lines = []
    for r in results:
        lines.append(f"[Source {r['rank']} | Score: {r['score']}]")
        lines.append(r["text"])
        lines.append("")

    return "\n".join(lines)


# ── Evaluation Metrics ────────────────────────────────────────────────────────

def recall_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    """
    Recall@k — what fraction of relevant chunks were retrieved in top-k.
    """
    retrieved_top_k = set(retrieved_ids[:k])
    relevant_set    = set(relevant_ids)
    if not relevant_set:
        return 0.0
    return round(len(retrieved_top_k & relevant_set) / len(relevant_set), 4)


def mean_reciprocal_rank(retrieved_ids: list, relevant_ids: list) -> float:
    """
    MRR — position of first relevant result. Higher is better.
    """
    relevant_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_set:
            return round(1.0 / rank, 4)
    return 0.0


def evaluate_retrieval(test_queries: list[dict],
                        subject: str,
                        index_dir: str,
                        top_k: int = 5) -> dict:
    """
    Evaluate retrieval quality on a set of test queries.

    test_queries format:
        [ { 'query': str, 'relevant_chunk_ids': [int, ...] }, ... ]

    Returns dict with Recall@k and MRR.
    """
    recall_scores = []
    mrr_scores    = []

    for item in test_queries:
        results = retrieve(item["query"], subject, index_dir, top_k=top_k)
        retrieved_ids = [r["chunk_id"] for r in results]

        r_at_k = recall_at_k(retrieved_ids, item["relevant_chunk_ids"], top_k)
        mrr    = mean_reciprocal_rank(retrieved_ids, item["relevant_chunk_ids"])

        recall_scores.append(r_at_k)
        mrr_scores.append(mrr)

    metrics = {
        f"Recall@{top_k}": round(float(np.mean(recall_scores)), 4),
        "MRR":             round(float(np.mean(mrr_scores)), 4),
        "n_queries":       len(test_queries),
    }

    print(f"\n── Retrieval Evaluation ──────────────────────")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return metrics
