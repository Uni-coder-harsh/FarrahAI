"""
FarrahAI — Module 10: Room Manager
=====================================
Manages subject "server rooms."

Each room is a subject-specific folder containing:
  - raw uploaded files
  - processed text
  - FAISS index (knowledge base)
  - room metadata (students, subject, assigned teacher)

Teacher profiles are GLOBAL (shared across all rooms).
Knowledge bases are LOCAL (per subject room).
"""

import json
import logging
import shutil
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

ROOMS_MANIFEST = "rooms_manifest.json"


# ── Room Operations ───────────────────────────────────────────────────────────

def create_room(subject: str, teacher_name: str, base_dir: str) -> dict:
    """
    Create a new subject room.

    Args:
        subject: e.g. "AI_ML", "DBMS", "Maths"
        teacher_name: assigned teacher (must exist or will be created in teacher DB)
        base_dir: root data directory

    Returns:
        room metadata dict
    """
    base = Path(base_dir)
    room_dir = base / "rooms" / subject
    room_dir.mkdir(parents=True, exist_ok=True)
    (room_dir / "raw").mkdir(exist_ok=True)
    (room_dir / "processed").mkdir(exist_ok=True)

    metadata = {
        "subject":       subject,
        "teacher":       teacher_name,
        "created_at":    datetime.now().isoformat(),
        "files_uploaded": 0,
        "chunks_indexed": 0,
        "students":      [],
        "last_indexed":  None,
        "room_dir":      str(room_dir),
    }

    meta_path = room_dir / "room_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Update global manifest
    _update_manifest(base_dir, subject, teacher_name)

    logger.info(f"Room created: '{subject}' → {room_dir}")
    print(f"✓ Subject room created: {subject}  (Teacher: {teacher_name})")
    return metadata


def load_room(subject: str, base_dir: str) -> dict:
    """Load room metadata."""
    meta_path = Path(base_dir) / "rooms" / subject / "room_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Room '{subject}' does not exist.\n"
            f"Create it first: python main.py --action create_room --subject {subject}"
        )
    with open(meta_path) as f:
        return json.load(f)


def update_room_meta(subject: str, base_dir: str, updates: dict):
    """Update specific fields in room metadata."""
    room = load_room(subject, base_dir)
    room.update(updates)
    meta_path = Path(base_dir) / "rooms" / subject / "room_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(room, f, indent=2)


def list_rooms(base_dir: str) -> list[dict]:
    """List all subject rooms."""
    manifest_path = Path(base_dir) / ROOMS_MANIFEST
    if not manifest_path.exists():
        return []
    with open(manifest_path) as f:
        return json.load(f).get("rooms", [])


def _update_manifest(base_dir: str, subject: str, teacher: str):
    manifest_path = Path(base_dir) / ROOMS_MANIFEST
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"rooms": []}

    # Update or add entry
    existing = [r for r in manifest["rooms"] if r["subject"] == subject]
    if existing:
        existing[0]["teacher"] = teacher
    else:
        manifest["rooms"].append({
            "subject": subject,
            "teacher": teacher,
            "created_at": datetime.now().isoformat()
        })

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


# ── File Upload Pipeline ──────────────────────────────────────────────────────

def upload_and_index(file_path: str,
                      subject: str,
                      base_dir: str,
                      ocr_engine: str = "paddleocr",
                      chunk_method: str = "words",
                      embedding_model: str = "all-MiniLM-L6-v2"):
    """
    Full upload pipeline for one file:
      1. Copy to room raw dir
      2. Detect type (image vs PDF)
      3. Preprocess + OCR (images) or direct extract (PDF)
      4. Clean + chunk text
      5. Add to subject FAISS index

    Args:
        file_path: path to uploaded file
        subject: subject room name
        base_dir: root data directory
    """
    from modules.preprocess import preprocess_image
    from modules.ocr import extract_text
    from modules.chunker import process_text, extract_text_from_pdf
    from modules.embedder import embed_texts, build_faiss_index, load_index, save_index

    import faiss as faiss_lib
    import numpy as np

    room_dir = Path(base_dir) / "rooms" / subject
    raw_dir  = room_dir / "raw"
    proc_dir = room_dir / "processed"
    idx_dir  = Path(base_dir) / "embeddings"

    # 1. Copy file
    src = Path(file_path)
    dest = raw_dir / src.name
    shutil.copy2(src, dest)
    logger.info(f"Copied {src.name} → {dest}")

    # 2. Extract text
    suffix = src.suffix.lower()
    if suffix in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        proc_path = str(proc_dir / f"proc_{src.name}")
        preprocess_image(str(dest), save_path=proc_path)
        result = extract_text(proc_path, engine=ocr_engine)
        raw_text = result["text"]
    elif suffix == ".pdf":
        raw_text = extract_text_from_pdf(str(dest))
    elif suffix == ".txt":
        with open(dest, 'r', encoding='utf-8', errors='replace') as f:
            raw_text = f.read()
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    if not raw_text.strip():
        logger.warning(f"No text extracted from {src.name}")
        return

    # 3. Chunk
    chunks = process_text(raw_text, method=chunk_method)
    for c in chunks:
        c["source_file"] = src.name
        c["subject"]     = subject

    # 4. Append to FAISS index
    texts = [c["text"] for c in chunks]
    new_embeddings = embed_texts(texts, model_name=embedding_model)

    try:
        index, existing_chunks = load_index(str(idx_dir), subject)
        # Merge
        index.add(new_embeddings)
        # Re-number chunk IDs
        offset = max(c["chunk_id"] for c in existing_chunks) + 1
        for c in chunks:
            c["chunk_id"] += offset
        all_chunks = existing_chunks + chunks
    except FileNotFoundError:
        # First upload for this subject
        index      = build_faiss_index(new_embeddings)
        all_chunks = chunks

    save_index(index, all_chunks, str(idx_dir), subject)

    # 5. Update room metadata
    update_room_meta(subject, base_dir, {
        "files_uploaded": load_room(subject, base_dir)["files_uploaded"] + 1,
        "chunks_indexed": len(all_chunks),
        "last_indexed":   datetime.now().isoformat(),
    })

    print(f"✓ Uploaded & indexed: {src.name}")
    print(f"  Chunks added: {len(chunks)} | Total in KB: {len(all_chunks)}")
