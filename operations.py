"""
operations.py
--------------
All point operation implementations, plus the metadata the UI needs to
explain each one (definition, purpose, formula, adjustable parameters,
pixel examples, and an image-specific "what changed" narrative).

Design note: OPERATIONS is a registry (dict). Every function has the
signature `function(array, params) -> array` even when it takes no
parameters, so app.py can call every operation the same way:

    processed = OPERATIONS[name]["function"](original_array, params)

`params_spec` describes what UI controls app.py should render for an
operation. Each spec is one of:
    {"key", "label", "type": "slider", "min", "max", "step",
     "default"} or {..., "default_fn": callable(array) -> number}
    {"key", "label", "type": "select", "options": [...], "default"}

`default_fn` lets a parameter's default be computed FROM the uploaded
image (e.g. Otsu's threshold, or percentiles) instead of being a fixed
guess that may not suit every image.
"""

from __future__ import annotations

import numpy as np

from image_utils import describe_brightness, otsu_threshold

# A common set of input values used to build "pixel examples" tables
# and to sanity-check a transformation's behavior across the range.
EXAMPLE_INPUTS = [0, 32, 64, 96, 128, 160, 192, 224, 255]


# ------------------------------------------------------------------
# 1. Image Negative
# ------------------------------------------------------------------
def apply_negative(array: np.ndarray, params: dict) -> np.ndarray:
    """s = (L-1) - r  →  s = 255 - r for an 8-bit image."""
    return (255 - array.astype(np.int16)).astype(np.uint8)


def _describe_negative(before: dict, after: dict, params: dict) -> str:
    before_desc = describe_brightness(before["mean"])
    after_desc = describe_brightness(after["mean"])
    return (
        f"The original image was {before_desc} (mean {before['mean']}). "
        f"After negation, it became {after_desc} (mean {after['mean']}). "
        f"Every pixel was mirrored around the midpoint 127.5 — "
        f"e.g. {before['min']} → {255 - before['min']} and "
        f"{before['max']} → {255 - before['max']}. The histogram is "
        f"effectively flipped left-to-right."
    )


# ------------------------------------------------------------------
# 2. Log Transformation
# ------------------------------------------------------------------
def apply_log(array: np.ndarray, params: dict) -> np.ndarray:
    """s = c * log(1 + r). Expands dark tones, compresses bright ones."""
    c = params.get("c", 1.0)
    r = array.astype(np.float64)
    s = c * np.log1p(r)
    return np.clip(s, 0, 255).astype(np.uint8)


def _default_log_c(array: np.ndarray) -> float:
    """Choose c so the image's own max intensity maps close to 255."""
    max_val = float(array.max())
    denom = np.log1p(max_val)
    return round(255.0 / denom, 2) if denom > 0 else 1.0


def _describe_log(before: dict, after: dict, params: dict) -> str:
    c = params.get("c", 1.0)
    return (
        f"With c={c}, the log transform compressed the wide range of bright "
        f"values while stretching apart the darker ones. Mean intensity "
        f"moved from {before['mean']} to {after['mean']}, and the standard "
        f"deviation changed from {before['std']} to {after['std']} — a drop "
        f"here means bright details got squeezed closer together, which is "
        f"the expected effect of the log curve flattening out at high r."
    )


# ------------------------------------------------------------------
# 3. Gamma (Power-Law) Transformation
# ------------------------------------------------------------------
def apply_gamma(array: np.ndarray, params: dict) -> np.ndarray:
    """s = c * (r/255)^gamma * 255."""
    gamma = params.get("gamma", 1.0)
    c = params.get("c", 1.0)
    r_norm = array.astype(np.float64) / 255.0
    s = c * np.power(r_norm, gamma) * 255.0
    return np.clip(s, 0, 255).astype(np.uint8)


def _describe_gamma(before: dict, after: dict, params: dict) -> str:
    gamma = params.get("gamma", 1.0)
    if gamma < 1:
        effect = "brightened the image overall, expanding detail in dark regions"
    elif gamma > 1:
        effect = "darkened the image overall, expanding detail in bright regions"
    else:
        effect = "left the image essentially unchanged (gamma = 1 is the identity mapping)"
    return (
        f"With gamma={gamma}, this operation {effect}. Mean intensity moved "
        f"from {before['mean']} to {after['mean']}."
    )


# ------------------------------------------------------------------
# 4. Contrast Stretching (piecewise-linear, two control points)
# ------------------------------------------------------------------
def apply_contrast_stretch(array: np.ndarray, params: dict) -> np.ndarray:
    """
    Piecewise-linear stretch through control points (r1,s1) and (r2,s2):
      r in [0, r1)   -> s = (s1/r1) * r                      (or 0 if r1=0)
      r in [r1, r2)  -> s = ((s2-s1)/(r2-r1)) * (r-r1) + s1
      r in [r2, 255] -> s = ((255-s2)/(255-r2)) * (r-r2) + s2 (or s2 if r2=255)
    """
    r1 = float(params.get("r1", 50))
    s1 = float(params.get("s1", 0))
    r2 = float(params.get("r2", 200))
    s2 = float(params.get("s2", 255))
    r = array.astype(np.float64)
    s = np.zeros_like(r)

    seg1 = r < r1
    seg2 = (r >= r1) & (r < r2)
    seg3 = r >= r2

    s[seg1] = (s1 / r1) * r[seg1] if r1 > 0 else 0
    if r2 > r1:
        s[seg2] = ((s2 - s1) / (r2 - r1)) * (r[seg2] - r1) + s1
    else:
        s[seg2] = s1
    if r2 < 255:
        s[seg3] = ((255 - s2) / (255 - r2)) * (r[seg3] - r2) + s2
    else:
        s[seg3] = s2

    return np.clip(s, 0, 255).astype(np.uint8)


def _default_stretch_points(array: np.ndarray) -> tuple[float, float]:
    """Use the 5th/95th percentiles as sensible default control points."""
    r1 = float(np.percentile(array, 5))
    r2 = float(np.percentile(array, 95))
    if r2 <= r1:
        r2 = r1 + 1
    return round(r1), round(r2)


def _describe_contrast_stretch(before: dict, after: dict, params: dict) -> str:
    r1, r2 = params.get("r1"), params.get("r2")
    return (
        f"Intensities between {r1} and {r2} were stretched to fill the full "
        f"0-255 range, increasing contrast in that band. Standard deviation "
        f"(a measure of contrast/spread) changed from {before['std']} to "
        f"{after['std']} — a higher value means more visible contrast."
    )


# ------------------------------------------------------------------
# 5. Thresholding
# ------------------------------------------------------------------
def apply_threshold(array: np.ndarray, params: dict) -> np.ndarray:
    """s = 255 if r >= T else 0 — converts the image to pure black & white."""
    t = params.get("t", 127)
    return np.where(array >= t, 255, 0).astype(np.uint8)


def _describe_threshold(before: dict, after: dict, params: dict) -> str:
    t = params.get("t")
    pct_white = round((after["mean"] / 255.0) * 100, 1)
    return (
        f"Every pixel at or above {t} became pure white (255); everything "
        f"below became pure black (0). About {pct_white}% of the image's "
        f"total brightness is now white, turning this into a binary "
        f"(two-level) image."
    )


# ------------------------------------------------------------------
# 6. Gray-Level Slicing
# ------------------------------------------------------------------
def apply_gray_level_slicing(array: np.ndarray, params: dict) -> np.ndarray:
    """
    Highlight intensities within [A, B].
    mode='Preserve background': pixels outside the range keep their
        original value; pixels inside become 255.
    mode='Suppress background': pixels outside the range become 0;
        pixels inside become 255.
    """
    a = params.get("a", 100)
    b = params.get("b", 180)
    mode = params.get("mode", "Preserve background")
    in_range = (array >= a) & (array <= b)

    if mode == "Suppress background":
        result = np.zeros_like(array)
    else:
        result = array.copy()

    result = np.where(in_range, 255, result).astype(np.uint8)
    return result


def _default_slicing_range(array: np.ndarray) -> tuple[int, int]:
    """Center a default [A, B] window around the image's own mean intensity."""
    mean = float(array.mean())
    std = float(array.std()) or 20.0
    a = int(max(0, mean - std / 2))
    b = int(min(255, mean + std / 2))
    if b <= a:
        b = min(255, a + 20)
    return a, b


def _describe_slicing(before: dict, after: dict, params: dict) -> str:
    a, b = params.get("a"), params.get("b")
    mode = params.get("mode")
    return (
        f"Pixels with intensity between {a} and {b} were highlighted as pure "
        f"white; {'the rest of the image kept its original values' if mode == 'Preserve background' else 'everything else was pushed to black'}. "
        f"This makes it easy to visually isolate that specific brightness band."
    )


# ------------------------------------------------------------------
# Operation registry
# ------------------------------------------------------------------
OPERATIONS = {
    "Image Negative": {
        "function": apply_negative,
        "what": (
            "Image Negative reverses every pixel's brightness. Dark areas "
            "become bright, and bright areas become dark — like a "
            "photographic film negative."
        ),
        "why": (
            "Useful for revealing detail hidden in dark regions of an image "
            "(e.g. subtle structures in dark medical or astronomical images "
            "become easier to see once they're bright)."
        ),
        "formula_latex": r"s = (L-1) - r \;\;\Rightarrow\;\; s = 255 - r",
        "formula_plain": "s = 255 - r",
        "params_spec": [],
        "describe_change": _describe_negative,
    },
    "Log Transformation": {
        "function": apply_log,
        "what": (
            "The log transformation compresses a wide range of pixel values "
            "into a narrower one, expanding dark pixel values while "
            "compressing bright ones."
        ),
        "why": (
            "Useful when an image has a very large dynamic range (e.g. "
            "Fourier spectra, or images with a few very bright spots) and "
            "you want to reveal detail in the darker majority of pixels."
        ),
        "formula_latex": r"s = c \cdot \log(1 + r)",
        "formula_plain": "s = c * log(1 + r)",
        "params_spec": [
            {
                "key": "c",
                "label": "Scaling constant (c)",
                "type": "slider",
                "min": 0.1,
                "max": 100.0,
                "step": 0.1,
                "default_fn": _default_log_c,
            }
        ],
        "describe_change": _describe_log,
    },
    "Gamma (Power-Law) Transformation": {
        "function": apply_gamma,
        "what": (
            "The gamma (power-law) transformation raises normalized pixel "
            "values to a power gamma (γ). γ < 1 brightens the image; γ > 1 "
            "darkens it."
        ),
        "why": (
            "Used for gamma correction — compensating for how displays, "
            "cameras, and human vision respond non-linearly to brightness."
        ),
        "formula_latex": r"s = c \cdot r^{\gamma}",
        "formula_plain": "s = c * (r/255)^gamma * 255",
        "params_spec": [
            {
                "key": "gamma",
                "label": "Gamma (γ)",
                "type": "slider",
                "min": 0.1,
                "max": 5.0,
                "step": 0.1,
                "default": 0.5,
            },
            {
                "key": "c",
                "label": "Scaling constant (c)",
                "type": "slider",
                "min": 0.1,
                "max": 3.0,
                "step": 0.1,
                "default": 1.0,
            },
        ],
        "describe_change": _describe_gamma,
    },
    "Contrast Stretching": {
        "function": apply_contrast_stretch,
        "what": (
            "Contrast Stretching maps a chosen input range [r1, r2] onto the "
            "full [0, 255] output range using a piecewise-linear function, "
            "increasing contrast within that band."
        ),
        "why": (
            "Useful for low-contrast images where most pixel values are "
            "clustered in a narrow band — stretching that band improves "
            "visibility."
        ),
        "formula_latex": r"s = \frac{s_2-s_1}{r_2-r_1}(r-r_1) + s_1 \ \text{for } r_1 \le r \le r_2",
        "formula_plain": "s = ((s2-s1)/(r2-r1)) * (r-r1) + s1, piecewise",
        "params_spec": [
            {
                "key": "r1",
                "label": "r1 (lower input bound)",
                "type": "slider",
                "min": 0,
                "max": 254,
                "step": 1,
                "default_fn": lambda arr: _default_stretch_points(arr)[0],
            },
            {
                "key": "s1",
                "label": "s1 (output for r1)",
                "type": "slider",
                "min": 0,
                "max": 255,
                "step": 1,
                "default": 0,
            },
            {
                "key": "r2",
                "label": "r2 (upper input bound)",
                "type": "slider",
                "min": 1,
                "max": 255,
                "step": 1,
                "default_fn": lambda arr: _default_stretch_points(arr)[1],
            },
            {
                "key": "s2",
                "label": "s2 (output for r2)",
                "type": "slider",
                "min": 0,
                "max": 255,
                "step": 1,
                "default": 255,
            },
        ],
        "describe_change": _describe_contrast_stretch,
    },
    "Thresholding": {
        "function": apply_threshold,
        "what": (
            "Thresholding converts the image to pure black and white: every "
            "pixel at or above a threshold T becomes white, everything else "
            "becomes black."
        ),
        "why": (
            "The basic building block of image segmentation — separating "
            "'objects' from 'background' based on brightness."
        ),
        "formula_latex": r"s = \begin{cases} 255 & r \ge T \\ 0 & r < T \end{cases}",
        "formula_plain": "s = 255 if r >= T else 0",
        "params_spec": [
            {
                "key": "t",
                "label": "Threshold (T)",
                "type": "slider",
                "min": 0,
                "max": 255,
                "step": 1,
                "default_fn": otsu_threshold,
            }
        ],
        "describe_change": _describe_threshold,
    },
    "Gray-Level Slicing": {
        "function": apply_gray_level_slicing,
        "what": (
            "Gray-Level Slicing highlights a specific band of intensities "
            "[A, B] by making them stand out (pure white), either keeping "
            "or discarding everything outside that band."
        ),
        "why": (
            "Useful for emphasizing a particular feature that occupies a "
            "known intensity range, such as flaws in an X-ray or specific "
            "tissue in a medical scan."
        ),
        "formula_latex": r"s = \begin{cases} 255 & A \le r \le B \\ r \text{ or } 0 & \text{otherwise} \end{cases}",
        "formula_plain": "s = 255 if A <= r <= B else (r or 0, depending on mode)",
        "params_spec": [
            {
                "key": "a",
                "label": "A (lower bound)",
                "type": "slider",
                "min": 0,
                "max": 254,
                "step": 1,
                "default_fn": lambda arr: _default_slicing_range(arr)[0],
            },
            {
                "key": "b",
                "label": "B (upper bound)",
                "type": "slider",
                "min": 1,
                "max": 255,
                "step": 1,
                "default_fn": lambda arr: _default_slicing_range(arr)[1],
            },
            {
                "key": "mode",
                "label": "Background handling",
                "type": "select",
                "options": ["Preserve background", "Suppress background"],
                "default": "Preserve background",
            },
        ],
        "describe_change": _describe_slicing,
    },
}
