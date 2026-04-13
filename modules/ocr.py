"""
FarrahAI — Module 2: OCR + Quality Evaluation
==============================================
Extracts text from preprocessed images.
Also measures extraction quality using CER and WER.

CER = Character Error Rate  (lower is better)
WER = Word Error Rate       (lower is better)

These metrics require a ground truth text to compare against.
For evaluation, you manually type the correct text for 10-15 pages,
then compare with OCR output. This is what you show in your PPT.
"""

import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# ── OCR backends ─────────────────────────────────────────────────────────────

def ocr_with_paddleocr(image_path: str) -> dict:
    """
    Run OCR using PaddleOCR.
    Returns: { 'text': str, 'confidence': float, 'boxes': list }
    """
    try:
        from paddleocr import PaddleOCR
        ocr_engine = PaddleOCR(lang='en')
        result = ocr_engine.ocr(image_path)

        lines = []
        confidences = []
        boxes = []

        if result and result[0]:
            for line in result[0]:
                box, (text, conf) = line
                lines.append(text)
                confidences.append(conf)
                boxes.append(box)

        full_text = "\n".join(lines)
        avg_conf  = float(np.mean(confidences)) if confidences else 0.0

        return {
            "text": full_text,
            "confidence": round(avg_conf, 4),
            "boxes": boxes,
            "engine": "paddleocr"
        }
    except ImportError:
        logger.warning("PaddleOCR not installed, falling back to Tesseract")
        return ocr_with_tesseract(image_path)


def ocr_with_tesseract(image_path: str) -> dict:
    """
    Run OCR using Tesseract.
    Returns: { 'text': str, 'confidence': float }
    """
    try:
        import pytesseract
        from PIL import Image

        img  = Image.open(image_path)
        data = pytesseract.image_to_data(
            img, output_type=pytesseract.Output.DICT
        )
        words  = [w for w in data['text'] if w.strip()]
        confs  = [c for c, w in zip(data['conf'], data['text'])
                  if w.strip() and c != -1]
        text   = " ".join(words)
        avg_c  = float(np.mean(confs)) / 100.0 if confs else 0.0

        return {
            "text": text,
            "confidence": round(avg_c, 4),
            "boxes": [],
            "engine": "tesseract"
        }
    except ImportError:
        raise ImportError("Neither PaddleOCR nor Tesseract is available. "
                          "Install one: pip install paddleocr OR sudo apt install tesseract-ocr")


# ── Main OCR function ─────────────────────────────────────────────────────────

def extract_text(image_path: str, engine: str = "paddleocr") -> dict:
    """
    Extract text from an image file.

    Args:
        image_path: path to image (preprocessed or raw)
        engine: "paddleocr" | "tesseract"

    Returns:
        dict with 'text', 'confidence', 'engine'
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    logger.info(f"Running OCR [{engine}] on: {Path(image_path).name}")

    if engine == "paddleocr":
        return ocr_with_paddleocr(image_path)
    elif engine == "tesseract":
        return ocr_with_tesseract(image_path)
    else:
        raise ValueError(f"Unknown OCR engine: {engine}. Use 'paddleocr' or 'tesseract'")


# ── Quality Metrics: CER and WER ─────────────────────────────────────────────

def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate.
    Uses edit distance at character level.
    range: 0.0 (perfect) to 1.0+ (very bad)
    """
    try:
        from jiwer import cer
        return round(cer(reference, hypothesis), 4)
    except ImportError:
        # manual fallback using edit distance
        return _edit_distance_cer(reference, hypothesis)


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Word Error Rate.
    Standard metric for OCR and ASR evaluation.
    range: 0.0 (perfect) to 1.0+ (very bad)
    """
    try:
        from jiwer import wer
        return round(wer(reference, hypothesis), 4)
    except ImportError:
        return _edit_distance_wer(reference, hypothesis)


def _edit_distance_cer(ref: str, hyp: str) -> float:
    """Manual CER using dynamic programming."""
    ref, hyp = ref.replace(" ", ""), hyp.replace(" ", "")
    n, m = len(ref), len(hyp)
    if n == 0:
        return 1.0
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        new_dp = [i]
        for j in range(1, m + 1):
            if ref[i-1] == hyp[j-1]:
                new_dp.append(dp[j-1])
            else:
                new_dp.append(1 + min(dp[j], new_dp[-1], dp[j-1]))
        dp = new_dp
    return round(dp[m] / n, 4)


def _edit_distance_wer(ref: str, hyp: str) -> float:
    """Manual WER using word-level edit distance."""
    ref_words = ref.split()
    hyp_words = hyp.split()
    n, m = len(ref_words), len(hyp_words)
    if n == 0:
        return 1.0
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        new_dp = [i]
        for j in range(1, m + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                new_dp.append(dp[j-1])
            else:
                new_dp.append(1 + min(dp[j], new_dp[-1], dp[j-1]))
        dp = new_dp
    return round(dp[m] / n, 4)


def evaluate_ocr(ground_truth_text: str,
                 ocr_text: str,
                 label: str = "eval") -> dict:
    """
    Full OCR quality evaluation.

    Args:
        ground_truth_text: manually verified correct text
        ocr_text: text extracted by OCR
        label: label for this evaluation (e.g. "raw", "preprocessed")

    Returns:
        dict with CER, WER, and summary
    """
    cer_val = compute_cer(ground_truth_text, ocr_text)
    wer_val = compute_wer(ground_truth_text, ocr_text)

    result = {
        "label": label,
        "cer":   cer_val,
        "wer":   wer_val,
        "cer_%": round(cer_val * 100, 2),
        "wer_%": round(wer_val * 100, 2),
    }

    logger.info(f"[{label}] CER: {result['cer_%']}%  WER: {result['wer_%']}%")
    return result


def compare_preprocessing_effect(image_path_raw: str,
                                  image_path_processed: str,
                                  ground_truth: str,
                                  engine: str = "paddleocr") -> dict:
    """
    Compare OCR quality BEFORE and AFTER preprocessing.
    This is exactly the table you show in your PPT:

    Method          CER      WER
    Raw OCR         0.31     0.44
    Preprocessed    0.17     0.26

    Returns both results side by side.
    """
    raw_result  = extract_text(image_path_raw, engine=engine)
    proc_result = extract_text(image_path_processed, engine=engine)

    raw_eval  = evaluate_ocr(ground_truth, raw_result["text"],  label="Raw OCR")
    proc_eval = evaluate_ocr(ground_truth, proc_result["text"], label="Preprocessed OCR")

    comparison = {
        "raw":          raw_eval,
        "preprocessed": proc_eval,
        "cer_improvement_%": round((raw_eval["cer"] - proc_eval["cer"]) * 100, 2),
        "wer_improvement_%": round((raw_eval["wer"] - proc_eval["wer"]) * 100, 2),
    }

    print("\n── OCR Quality Comparison ──────────────────")
    print(f"{'Method':<20} {'CER':>8} {'WER':>8}")
    print(f"{'─'*40}")
    print(f"{'Raw OCR':<20} {raw_eval['cer']:>8.4f} {raw_eval['wer']:>8.4f}")
    print(f"{'Preprocessed OCR':<20} {proc_eval['cer']:>8.4f} {proc_eval['wer']:>8.4f}")
    print(f"{'─'*40}")
    print(f"Improvement → CER: {comparison['cer_improvement_%']}%  WER: {comparison['wer_improvement_%']}%")

    return comparison
