"""
FarrahAI — Global Configuration
All paths, model names, and constants live here.
Change paths here if your setup differs.
"""

import os
from pathlib import Path

# ── Base paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR          = BASE_DIR / "data"
RAW_DIR           = DATA_DIR / "raw"
PROCESSED_DIR     = DATA_DIR / "processed"
EMBEDDINGS_DIR    = DATA_DIR / "embeddings"
KNOWLEDGE_BASE_DIR= DATA_DIR / "knowledge_base"
QP_DIR            = DATA_DIR / "question_papers"
MODELS_DIR        = BASE_DIR / "models"
OUTPUTS_DIR       = BASE_DIR / "outputs"

# Create dirs if missing
for d in [RAW_DIR, PROCESSED_DIR, EMBEDDINGS_DIR,
          KNOWLEDGE_BASE_DIR, QP_DIR, MODELS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── OCR ─────────────────────────────────────────────────────────────────────
OCR_ENGINE        = "tesseract"   # "paddleocr" | "tesseract"
TESSERACT_CMD     = "/usr/bin/tesseract"   # update if different on your system
OCR_LANG          = "en"

# ── Preprocessing ───────────────────────────────────────────────────────────
CLAHE_CLIP_LIMIT  = 2.0
CLAHE_TILE_SIZE   = (8, 8)
DENOISE_STRENGTH  = 10

# ── Chunking ────────────────────────────────────────────────────────────────
CHUNK_SIZE        = 300    # tokens / words per chunk
CHUNK_OVERLAP     = 50

# ── Embeddings ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"   # fast, offline, good quality
EMBEDDING_DIM     = 384

# ── FAISS ───────────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL   = 5

# ── Ollama ──────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL   = "http://localhost:11434"
OLLAMA_MODEL      = "mistral:7b-instruct-q4_0"   # change to whatever you have: llama3, phi3, etc.
OLLAMA_TIMEOUT    = 120

# ── ML ──────────────────────────────────────────────────────────────────────
RANDOM_STATE      = 42
TEST_SIZE         = 0.2

# ── Database ────────────────────────────────────────────────────────────────
DB_PATH           = BASE_DIR / "farrahai.db"

# ── Teacher profile ─────────────────────────────────────────────────────────
TEACHER_DB_PATH   = BASE_DIR / "data" / "teacher_profiles.json"

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL         = "INFO"
