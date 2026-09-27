"""
report_utils.py
-----------------
Builds a report as a Markdown string (with embedded before/after images)
that gets posted directly as a chat message — no separate downloadable
file. Generating this is pure local formatting of already-computed
numbers, so it costs no AI API call.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


def _array_to_base64_png(array: np.ndarray) -> str:
    """Encode a uint8 grayscale array as a base64 PNG data URI."""
    img = Image.fromarray(array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def generate_report_markdown(
    *,
    filename: str,
    image_info: dict,
    operation_name: str,
    operation_meta: dict,
    params_used: dict,
    original_array: np.ndarray,
    processed_array: np.ndarray,
    original_stats: dict,
    processed_stats: dict,
    change_summary: str,
) -> str:
    """Build a Markdown report string, ready to post as a chat message."""
    before_uri = _array_to_base64_png(original_array)
    after_uri = _array_to_base64_png(processed_array)
    params_str = ", ".join(f"{k}={v}" for k, v in params_used.items()) if params_used else "None"

    return f"""#### 📄 Report — {operation_name}

**Image:** {filename} ({image_info['width']}×{image_info['height']}px, {image_info['original_mode']})

**Formula:** `{operation_meta['formula_plain']}`
**Parameters used:** {params_str}

| Statistic | Before | After |
|---|---|---|
| Min | {original_stats['min']} | {processed_stats['min']} |
| Max | {original_stats['max']} | {processed_stats['max']} |
| Mean | {original_stats['mean']} | {processed_stats['mean']} |
| Std Dev | {original_stats['std']} | {processed_stats['std']} |
| Median | {original_stats['median']} | {processed_stats['median']} |

**What happened:** {change_summary}

![Before]({before_uri}) ![After]({after_uri})
"""
