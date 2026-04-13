"""
FarrahAI — Module 1: Image Preprocessing
=========================================
Converts raw note images into clean, OCR-friendly images.

Steps:
  1. Load image
  2. Convert to grayscale
  3. Denoise
  4. CLAHE (contrast enhancement)
  5. Adaptive thresholding
  6. Deskew
  7. Optional: morphological cleanup

All these steps improve OCR accuracy.
They do NOT make the model "understand" the content —
they make the image readable enough for the OCR engine.
"""

import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_image(image_path: str) -> np.ndarray:
    """Load image from path. Raises FileNotFoundError if missing."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    logger.info(f"Loaded image: {path.name} | shape: {img.shape}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    if len(img.shape) == 2:
        return img  # already grayscale
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(img: np.ndarray, strength: int = 10) -> np.ndarray:
    """
    Apply fast non-local means denoising.
    strength: higher = more smoothing (good for heavy noise)
    """
    return cv2.fastNlMeansDenoising(img, h=strength)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0,
                tile_size: tuple = (8, 8)) -> np.ndarray:
    """
    CLAHE — Contrast Limited Adaptive Histogram Equalization.
    Improves local contrast — very useful for uneven lighting in phone camera shots.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(img)


def threshold(img: np.ndarray) -> np.ndarray:
    """
    Adaptive thresholding — converts to clean black/white.
    Better than global threshold for handwritten notes.
    """
    return cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )


def deskew(img: np.ndarray) -> np.ndarray:
    """
    Deskew the image by detecting and correcting rotation angle.
    Handles slightly tilted scans / phone photos.
    """
    coords = np.column_stack(np.where(img > 0))
    if len(coords) == 0:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def morphological_cleanup(img: np.ndarray) -> np.ndarray:
    """
    Remove small noise dots using morphological opening.
    Use only when image has heavy speckle noise.
    """
    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


def preprocess_image(image_path: str,
                     save_path: str = None,
                     apply_morph: bool = False,
                     clip_limit: float = 2.0,
                     denoise_strength: int = 10) -> np.ndarray:
    """
    Full preprocessing pipeline.

    Args:
        image_path: path to raw image
        save_path: if given, saves the processed image here
        apply_morph: apply morphological cleanup (optional)
        clip_limit: CLAHE clip limit
        denoise_strength: denoising strength

    Returns:
        processed image as numpy array
    """
    img = load_image(image_path)
    img = to_grayscale(img)
    img = denoise(img, strength=denoise_strength)
    img = apply_clahe(img, clip_limit=clip_limit)
    img = threshold(img)
    img = deskew(img)
    if apply_morph:
        img = morphological_cleanup(img)

    if save_path:
        cv2.imwrite(save_path, img)
        logger.info(f"Saved processed image to: {save_path}")

    return img


def batch_preprocess(input_dir: str, output_dir: str,
                     extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp")) -> list:
    """
    Preprocess all images in a folder.

    Returns:
        list of (original_path, processed_path) tuples
    """
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []
    files = [f for f in input_path.iterdir()
             if f.suffix.lower() in extensions]

    logger.info(f"Found {len(files)} images to preprocess")

    for f in files:
        out_file = output_path / f"proc_{f.name}"
        try:
            preprocess_image(str(f), save_path=str(out_file))
            results.append((str(f), str(out_file)))
            logger.info(f"✓ {f.name}")
        except Exception as e:
            logger.error(f"✗ {f.name}: {e}")

    logger.info(f"Preprocessing complete: {len(results)}/{len(files)} successful")
    return results
