"""
image_utils.py
---------------
Phase 1 — Foundation.

This module handles everything about getting an image INTO the app safely:
- validating the uploaded file
- loading it with PIL
- converting it into NumPy arrays we can do math on
- extracting basic information and statistics

Keeping this separate from app.py (the UI) means later phases (Phase 2+)
can import these same functions without duplicating logic.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, UnidentifiedImageError

# Formats we explicitly support. Point operations are defined on raster
# pixel data, so we stick to common raster formats.
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}

# A sane upper bound so a huge image doesn't stall the app in the browser.
MAX_DIMENSION = 4000  # pixels, on the longer side


def validate_image(uploaded_file) -> tuple[bool, str]:
    """
    Check that an uploaded file is something we can safely process.

    Parameters
    ----------
    uploaded_file : a Streamlit UploadedFile object (has .name, and is
        file-like / seekable).

    Returns
    -------
    (is_valid, message)
        is_valid : True if the file passed all checks.
        message  : Empty string if valid, otherwise a human-readable
                   explanation of what went wrong (safe to show the user).
    """
    if uploaded_file is None:
        return False, "No file was uploaded."

    ext = uploaded_file.name.rsplit(".", 1)[-1].lower() if "." in uploaded_file.name else ""
    if ext not in ALLOWED_EXTENSIONS:
        return False, (
            f"Unsupported file type '.{ext}'. "
            f"Please upload one of: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )

    # Try to actually open it — this catches corrupted files or files that
    # merely have an image-like extension but aren't valid images.
    try:
        uploaded_file.seek(0)
        img = Image.open(uploaded_file)
        img.verify()  # checks integrity without fully decoding
    except (UnidentifiedImageError, OSError):
        return False, "This file could not be read as a valid image."
    finally:
        uploaded_file.seek(0)  # reset pointer so it can be read again later

    # Re-open (verify() leaves the image object unusable for further reads)
    img = Image.open(uploaded_file)
    width, height = img.size
    if max(width, height) > MAX_DIMENSION:
        return False, (
            f"Image is too large ({width}x{height}). "
            f"Please use an image where the longer side is under {MAX_DIMENSION}px."
        )
    uploaded_file.seek(0)

    return True, ""


def load_image(uploaded_file) -> dict:
    """
    Load a validated uploaded file into the representations we need
    throughout the app.

    Returns a dict with:
        'pil_original'  : PIL.Image, exactly as uploaded (mode preserved)
        'pil_rgb'       : PIL.Image, converted to RGB for consistent display
        'array_color'   : np.ndarray (H, W, 3), dtype=uint8
        'array_gray'    : np.ndarray (H, W), dtype=uint8
                           (this is what most point operations act on)
        'was_grayscale' : bool, True if the original upload was already
                           single-channel (grayscale) rather than color
        'filename'      : original filename
    """
    uploaded_file.seek(0)
    pil_original = Image.open(uploaded_file)
    pil_original.load()  # force full decode now, while the file is open

    was_grayscale = pil_original.mode in ("L", "1", "I", "F")

    pil_rgb = pil_original.convert("RGB")
    array_color = np.array(pil_rgb, dtype=np.uint8)

    pil_gray = pil_original.convert("L")
    array_gray = np.array(pil_gray, dtype=np.uint8)

    return {
        "pil_original": pil_original,
        "pil_rgb": pil_rgb,
        "array_color": array_color,
        "array_gray": array_gray,
        "was_grayscale": was_grayscale,
        "filename": uploaded_file.name,
    }


def get_image_info(image_data: dict) -> dict:
    """
    Collect display-friendly metadata about the loaded image.
    Used for the 'Image Information' section of the UI and reports.
    """
    pil_original = image_data["pil_original"]
    array_gray = image_data["array_gray"]
    width, height = pil_original.size

    return {
        "filename": image_data["filename"],
        "width": width,
        "height": height,
        "original_mode": pil_original.mode,
        "was_grayscale": image_data["was_grayscale"],
        "channels_processed_as": "Grayscale (single channel)",
        "bit_depth": "8-bit (0-255 per channel)",
        "total_pixels": width * height,
        "dtype": str(array_gray.dtype),
    }


def describe_brightness(mean: float) -> str:
    """
    Classify an image's overall brightness from its mean pixel intensity.
    Used to generate image-specific (not just generic) explanations.
    Thresholds split the 0-255 range into rough thirds.
    """
    if mean < 85:
        return "predominantly dark"
    elif mean > 170:
        return "predominantly bright"
    return "balanced (mid-range brightness)"


def get_basic_stats(array: np.ndarray) -> dict:
    """
    Compute basic pixel-intensity statistics for a single-channel array.
    These numbers are reused everywhere: info panels, before/after
    comparisons, and reports.
    """
    return {
        "min": int(array.min()),
        "max": int(array.max()),
        "mean": round(float(array.mean()), 2),
        "std": round(float(array.std()), 2),
        "median": float(np.median(array)),
    }


def otsu_threshold(array: np.ndarray) -> int:
    """
    Compute an automatic threshold using Otsu's method: the threshold
    that minimizes intra-class intensity variance (equivalently,
    maximizes between-class variance) between "foreground" and
    "background" pixels. Used to suggest a sensible default for the
    Thresholding operation instead of an arbitrary fixed number.
    """
    hist, _ = np.histogram(array.ravel(), bins=256, range=(0, 256))
    total = array.size
    sum_total = float(np.dot(np.arange(256), hist))

    weight_bg = 0.0
    sum_bg = 0.0
    best_variance = -1.0
    best_threshold = 0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between_variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between_variance > best_variance:
            best_variance = between_variance
            best_threshold = t

    return best_threshold
