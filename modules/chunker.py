"""
FarrahAI — Module 3: Text Cleaning & Chunking
==============================================
After OCR, text is messy: broken lines, repeated headers, noise.
This module cleans and splits text into chunks suitable for:
  - embedding
  - retrieval
  - topic classification
"""

import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Text Cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean raw OCR text:
    - Remove non-printable characters
    - Fix broken hyphenated words
    - Normalize whitespace
    - Remove repeated separators
    """
    # Remove non-printable characters (keep newlines)
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)

    # Fix hyphenated line breaks (word- \n word → word word)
    text = re.sub(r'-\s*\n\s*', '', text)

    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)

    # Collapse more than 2 newlines → double newline
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove lines that are just noise (single chars, numbers, dashes)
    lines = text.split('\n')
    lines = [l.strip() for l in lines if len(l.strip()) > 3]
    text = '\n'.join(lines)

    return text.strip()


def remove_headers_footers(text: str, min_repeat: int = 3) -> str:
    """
    Detect and remove lines that repeat many times (likely headers/footers).
    """
    lines = text.split('\n')
    from collections import Counter
    line_counts = Counter(lines)

    cleaned = [line for line in lines
               if line_counts[line] < min_repeat or len(line.strip()) < 5]
    return '\n'.join(cleaned)


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_by_words(text: str, chunk_size: int = 300, overlap: int = 50) -> list[dict]:
    """
    Split text into overlapping word-based chunks.

    Args:
        text: cleaned text
        chunk_size: words per chunk
        overlap: words of overlap between chunks

    Returns:
        list of { 'chunk_id': int, 'text': str, 'word_count': int }
    """
    words = text.split()
    chunks = []
    i = 0
    chunk_id = 0

    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunk_text  = " ".join(chunk_words)
        chunks.append({
            "chunk_id":   chunk_id,
            "text":       chunk_text,
            "word_count": len(chunk_words),
            "start_word": i,
        })
        chunk_id += 1
        i += chunk_size - overlap  # slide forward with overlap

    return chunks


def chunk_by_paragraphs(text: str, max_words: int = 400) -> list[dict]:
    """
    Split by paragraphs, merging short ones and splitting long ones.
    Better for structured notes with clear sections.
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    chunk_id = 0
    buffer = ""

    for para in paragraphs:
        if len(buffer.split()) + len(para.split()) <= max_words:
            buffer = (buffer + "\n\n" + para).strip()
        else:
            if buffer:
                chunks.append({
                    "chunk_id":   chunk_id,
                    "text":       buffer,
                    "word_count": len(buffer.split()),
                })
                chunk_id += 1
            buffer = para

    if buffer:
        chunks.append({
            "chunk_id":   chunk_id,
            "text":       buffer,
            "word_count": len(buffer.split()),
        })

    return chunks


# ── PDF Text Extraction ───────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF using pdfplumber.
    Works well for digital PDFs (not scanned).
    For scanned PDFs, use OCR pipeline instead.
    """
    try:
        import pdfplumber
        text_pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_pages.append(f"[Page {i+1}]\n{page_text}")
        return "\n\n".join(text_pages)
    except ImportError:
        logger.warning("pdfplumber not installed, trying PyMuPDF")
        return extract_text_from_pdf_fitz(pdf_path)


def extract_text_from_pdf_fitz(pdf_path: str) -> str:
    """Fallback: extract text using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        pages = []
        for i, page in enumerate(doc):
            pages.append(f"[Page {i+1}]\n{page.get_text()}")
        return "\n\n".join(pages)
    except ImportError:
        raise ImportError("Install pdfplumber or PyMuPDF: pip install pdfplumber PyMuPDF")


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def process_text(raw_text: str,
                 method: str = "words",
                 chunk_size: int = 300,
                 overlap: int = 50) -> list[dict]:
    """
    Full text processing pipeline: clean → chunk.

    Args:
        raw_text: text from OCR or PDF
        method: "words" | "paragraphs"
        chunk_size: words per chunk (for word method)
        overlap: overlap in words

    Returns:
        list of chunk dicts
    """
    cleaned = clean_text(raw_text)
    cleaned = remove_headers_footers(cleaned)

    if method == "paragraphs":
        chunks = chunk_by_paragraphs(cleaned, max_words=chunk_size)
    else:
        chunks = chunk_by_words(cleaned, chunk_size=chunk_size, overlap=overlap)

    logger.info(f"Text processed → {len(chunks)} chunks using '{method}' method")
    return chunks


def save_chunks(chunks: list[dict], output_path: str, subject: str = "unknown"):
    """Save chunks as a simple text file for inspection."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"Subject: {subject}\n")
        f.write(f"Total chunks: {len(chunks)}\n")
        f.write("=" * 60 + "\n\n")
        for chunk in chunks:
            f.write(f"[Chunk {chunk['chunk_id']}] ({chunk['word_count']} words)\n")
            f.write(chunk['text'])
            f.write("\n\n" + "─" * 40 + "\n\n")

    logger.info(f"Saved {len(chunks)} chunks to {output_path}")
