"""
FarrahAI — Module 4: Embeddings + FAISS Index
==============================================
Converts text chunks into dense vector embeddings
and stores them in a FAISS index for fast similarity search.

Model: all-MiniLM-L6-v2 (offline, 80MB, very fast)
  - 384-dimensional embeddings
  - Runs fully locally, no API needed

One FAISS index per subject room.
"""

import os
import json
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Lazy imports — avoids crash if not installed
_sentence_model = None
_faiss = None

def _get_model(model_name: str = "all-MiniLM-L6-v2"):
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {model_name}")
        _sentence_model = SentenceTransformer(model_name)
    return _sentence_model


def _get_faiss():
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2",
                batch_size: int = 64) -> np.ndarray:
    """
    Convert list of strings to embedding matrix.

    Returns:
        numpy array of shape (n_texts, embedding_dim)
    """
    model = _get_model(model_name)
    logger.info(f"Embedding {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True  # important for cosine similarity
    )
    return embeddings.astype(np.float32)


def embed_query(query: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Embed a single query string.
    Returns shape (1, embedding_dim)
    """
    model = _get_model(model_name)
    emb = model.encode([query], normalize_embeddings=True)
    return emb.astype(np.float32)


# ── FAISS Index ───────────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> object:
    """
    Build an L2/cosine FAISS index from embeddings.
    Using IndexFlatIP (inner product = cosine similarity for normalized vectors).
    """
    faiss = _get_faiss()
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product for normalized vectors
    index.add(embeddings)
    logger.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def save_index(index, chunks: list[dict], index_dir: str, subject: str):
    """
    Save FAISS index + chunk metadata to disk.

    Files saved:
      {index_dir}/{subject}.index   — FAISS binary
      {index_dir}/{subject}_meta.json — chunk texts and metadata
    """
    faiss = _get_faiss()
    path = Path(index_dir)
    path.mkdir(parents=True, exist_ok=True)

    index_path = path / f"{subject}.index"
    meta_path  = path / f"{subject}_meta.json"

    faiss.write_index(index, str(index_path))

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved index → {index_path}")
    logger.info(f"Saved metadata → {meta_path}")


def load_index(index_dir: str, subject: str):
    """
    Load FAISS index + chunk metadata from disk.

    Returns:
        (index, chunks_list)
    """
    faiss = _get_faiss()
    path  = Path(index_dir)

    index_path = path / f"{subject}.index"
    meta_path  = path / f"{subject}_meta.json"

    if not index_path.exists():
        raise FileNotFoundError(
            f"No FAISS index found for subject '{subject}' at {index_path}\n"
            f"Upload notes first: python main.py --action upload --subject {subject}"
        )

    index  = faiss.read_index(str(index_path))
    with open(meta_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    logger.info(f"Loaded index '{subject}': {index.ntotal} vectors, {len(chunks)} chunks")
    return index, chunks


# ── Main: Index a subject's knowledge base ───────────────────────────────────

def index_subject(chunks: list[dict],
                  subject: str,
                  index_dir: str,
                  model_name: str = "all-MiniLM-L6-v2"):
    """
    Full pipeline: chunks → embeddings → FAISS index → save.

    Args:
        chunks: list of chunk dicts (from chunker.py)
        subject: subject name (used as filename)
        index_dir: where to save index files
        model_name: sentence-transformers model
    """
    texts      = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, model_name=model_name)
    index      = build_faiss_index(embeddings)
    save_index(index, chunks, index_dir, subject)

    logger.info(f"Subject '{subject}' indexed successfully: {len(chunks)} chunks")
    return index
