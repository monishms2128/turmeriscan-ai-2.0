"""Optical quality and capture validation module for TurmeriScan AI.

Evaluates ambient lighting, sharpness (blur detection), and sample framing
to ensure reliable model inference from camera or mobile uploads.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def assess_image_quality(pil_img: Image.Image) -> dict[str, object]:
    """Analyze lighting, sharpness, and framing to ensure reliable screening.

    Returns:
        Dictionary containing quality metrics, status ('Optimal', 'Warning', 'Poor'),
        and actionable recommendations for the user.
    """
    img_gray = np.array(pil_img.convert("L"))
    h, w = img_gray.shape

    # 1. Sharpness / Blur Detection (Laplacian Variance)
    laplacian_var = float(cv2.Laplacian(img_gray, cv2.CV_64F).var())
    is_sharp = laplacian_var >= 45.0

    # 2. Illumination / Brightness (Mean Luminance)
    mean_brightness = float(np.mean(img_gray))
    is_well_lit = 35.0 <= mean_brightness <= 225.0
    is_too_dark = mean_brightness < 35.0
    is_overexposed = mean_brightness > 225.0

    # 3. Sample Presence & Contrast (Histogram dynamic range)
    contrast = float(np.std(img_gray))
    has_contrast = contrast >= 20.0

    # Determine overall status and advice
    warnings = []
    if not is_sharp:
        warnings.append("Image appears blurry. Hold camera steady and refocus.")
    if is_too_dark:
        warnings.append("Illumination is too dim. Add direct white light.")
    elif is_overexposed:
        warnings.append("Overexposed / glare detected. Avoid direct camera flash.")
    if not has_contrast:
        warnings.append("Low background contrast. Place turmeric on a dark/black surface.")

    if not warnings:
        status = "Optimal"
        badge_color = "#136338"
        summary = "Optimal illumination and focus for screening."
    elif len(warnings) == 1:
        status = "Acceptable"
        badge_color = "#986500"
        summary = warnings[0]
    else:
        status = "Suboptimal"
        badge_color = "#b83725"
        summary = " · ".join(warnings)

    return {
        "status": status,
        "badge_color": badge_color,
        "summary": summary,
        "is_reliable": len(warnings) <= 1,
        "sharpness_score": round(laplacian_var, 1),
        "brightness_score": round(mean_brightness, 1),
        "contrast_score": round(contrast, 1),
        "warnings": warnings,
    }
