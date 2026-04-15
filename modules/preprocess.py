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


def deskew(img: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    """
    Safely deskew a document image using horizontal projection profiles.

    Key fix over the old version:
    - Old version: used minAreaRect on ALL white pixels → catastrophic rotation
      when thresholded image has text blocks at various positions.
    - New version: uses Hough line detection to estimate skew angle from
      actual text lines, then CLAMPS to ±max_angle degrees.

    max_angle=10.0 means we only correct genuine slight tilts (phone photos,
    hand-placed scans). We refuse to rotate more than that — protecting against
    the 90° catastrophic rotation you observed.

    Args:
        img: grayscale or binary image
        max_angle: maximum degrees to rotate (default 10°)

    Returns:
        deskewed image, or original if angle outside safe range
    """
    # Invert if needed: Hough works on edges of dark text on white background
    # After adaptive threshold, text is black (0) on white (255)
    # We invert to get white text on black for edge detection
    if img.mean() > 127:
        working = cv2.bitwise_not(img)
    else:
        working = img.copy()

    # Detect edges
    edges = cv2.Canny(working, 50, 150, apertureSize=3)

    # Hough line detection
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=100,
        minLineLength=img.shape[1] // 4,   # lines must span at least 1/4 width
        maxLineGap=20
    )

    if lines is None or len(lines) == 0:
        logger.debug("deskew: no lines detected, returning original")
        return img

    # Compute angle for each detected line
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue   # vertical line — skip
        angle_deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle_deg)

    if not angles:
        return img

    # Use median angle to be robust against outliers
    median_angle = float(np.median(angles))

    # Clamp: only correct if angle is a small tilt
    if abs(median_angle) > max_angle:
        logger.warning(
            f"deskew: detected angle {median_angle:.1f}° exceeds max_angle "
            f"({max_angle}°) — skipping rotation to prevent catastrophic flip"
        )
        return img

    # Apply rotation
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    logger.info(f"deskew: corrected {median_angle:.2f}°")
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
                     apply_deskew: bool = True,
                     clip_limit: float = 1.5,
                     denoise_strength: int = 7,
                     max_deskew_angle: float = 10.0) -> np.ndarray:
    """
    Full preprocessing pipeline — corrected order and safe defaults.

    Correct pipeline order:
        grayscale → CLAHE → denoise → threshold → deskew → (optional morph)

    Why this order?
        - CLAHE before denoise: enhance contrast while detail still exists
        - Denoise after CLAHE: smooth out CLAHE artifacts
        - Threshold after denoise: cleaner binary from smooth input
        - Deskew last on binary: Hough lines work best on clean B&W

    Args:
        image_path:        path to raw image
        save_path:         if given, saves the processed image here
        apply_morph:       apply morphological cleanup (default off)
        apply_deskew:      apply deskew correction (default on, safe ±10°)
        clip_limit:        CLAHE clip limit — lower = less aggressive (1.5 default)
        denoise_strength:  denoising strength — lower = less blurring (7 default)
        max_deskew_angle:  max degrees to correct — prevents catastrophic rotation

    Returns:
        processed image as numpy array
    """
    img = load_image(image_path)
    img = to_grayscale(img)
    img = apply_clahe(img, clip_limit=clip_limit)        # enhance contrast first
    img = denoise(img, strength=denoise_strength)         # then smooth
    img = threshold(img)                                  # then binarize
    if apply_deskew:
        img = deskew(img, max_angle=max_deskew_angle)    # safe deskew last
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
