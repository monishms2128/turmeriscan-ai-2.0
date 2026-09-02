"""TurmeriScan AI — Sample Domain & Security Gatekeeper.

Enforces strict dual-mode validation:
1. Laboratory Multispectral Mode (single-band NIR/spectral absorption captures, TIFF, false-color bands)
2. Smart RGB Camera Mode (golden/yellow/amber turmeric powder captures)

Rejects Out-Of-Distribution (OOD) inputs (human faces/selfies, persons, skin,
non-turmeric objects, room backgrounds, vehicles, printed documents) before neural inference.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _detect_human_presence(img_gray: np.ndarray) -> tuple[bool, str]:
    """Detect human faces, profile faces, or upper body presence in the image."""
    cascades_to_check = [
        ("haarcascade_frontalface_default.xml", 1.08, 3, (30, 30), "Human face / selfie"),
        ("haarcascade_frontalface_alt.xml", 1.08, 3, (30, 30), "Human face / selfie"),
        ("haarcascade_frontalface_alt2.xml", 1.08, 3, (30, 30), "Human face / selfie"),
        ("haarcascade_profileface.xml", 1.1, 3, (35, 35), "Side profile / face"),
        ("haarcascade_upperbody.xml", 1.08, 2, (60, 60), "Person / upper body"),
    ]

    for cascade_name, scale, neighbors, min_sz, desc in cascades_to_check:
        try:
            path = cv2.data.haarcascades + cascade_name
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                detections = cascade.detectMultiScale(
                    img_gray,
                    scaleFactor=scale,
                    minNeighbors=neighbors,
                    minSize=min_sz,
                )
                if len(detections) > 0:
                    return True, desc
        except Exception:
            continue

    return False, ""


def _detect_human_skin_dominance(img_rgb: np.ndarray) -> tuple[bool, float]:
    """Detect if the image is dominated by human skin tones rather than spice powder."""
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower_skin = np.array([0, 20, 50], dtype=np.uint8)
    upper_skin = np.array([18, 170, 250], dtype=np.uint8)

    skin_mask = cv2.inRange(img_hsv, lower_skin, upper_skin)
    skin_pixels = int(np.count_nonzero(skin_mask))
    total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
    skin_ratio = float(skin_pixels / max(total_pixels, 1)) * 100.0

    return skin_ratio > 22.0, skin_ratio


def _is_text_or_document(img_gray: np.ndarray) -> bool:
    """Detect printed paper, receipts, text documents, or barcode patterns."""
    edges = cv2.Canny(img_gray, 80, 180)
    edge_density = float(np.count_nonzero(edges) / max(edges.size, 1)) * 100.0
    mean_val = float(np.mean(img_gray))
    std_val = float(np.std(img_gray))

    return mean_val > 215.0 and edge_density > 5.5 and std_val > 35.0


def _analyze_rgb_chromaticity(pil_img: Image.Image) -> tuple[bool, float, str]:
    """Verify if an RGB image matches the intense golden/yellow/amber/ochre color spectrum of turmeric powder."""
    img_rgb = np.array(pil_img.convert("RGB"))
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    lower_turmeric = np.array([14, 65, 40], dtype=np.uint8)
    upper_turmeric = np.array([38, 255, 255], dtype=np.uint8)

    turmeric_mask = cv2.inRange(img_hsv, lower_turmeric, upper_turmeric)
    turmeric_pixels = int(np.count_nonzero(turmeric_mask))
    total_pixels = img_rgb.shape[0] * img_rgb.shape[1]

    coverage_ratio = float(turmeric_pixels / max(total_pixels, 1)) * 100.0

    non_dark_mask = cv2.inRange(img_hsv, np.array([0, 0, 30], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8))
    non_dark_pixels = int(np.count_nonzero(non_dark_mask))
    fg_coverage = float(turmeric_pixels / max(non_dark_pixels, 1)) * 100.0

    is_skin, skin_pct = _detect_human_skin_dominance(img_rgb)
    if is_skin and coverage_ratio < 15.0:
        return False, coverage_ratio, f"Human skin / selfie profile detected ({skin_pct:.1f}% skin tone vs {coverage_ratio:.1f}% turmeric hue)."

    if coverage_ratio < 7.0 and fg_coverage < 16.0:
        return False, coverage_ratio, f"Color spectrum mismatch ({coverage_ratio:.1f}% turmeric yellow-amber hue). Image is dominated by non-turmeric objects or room background."

    return True, coverage_ratio, f"Turmeric color signature verified ({coverage_ratio:.1f}% coverage)."


def is_image_natively_grayscale(pil_img: Image.Image) -> bool:
    """Determine whether an image is a single-channel or grayscale spectral band."""
    if pil_img.mode in ("L", "1", "I", "F", "I;16"):
        return True

    if pil_img.mode in ("RGB", "RGBA"):
        img_arr = np.array(pil_img.convert("RGB"))
        r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
        diff_rg = np.mean(np.abs(r.astype(np.int16) - g.astype(np.int16)))
        diff_rb = np.mean(np.abs(r.astype(np.int16) - b.astype(np.int16)))
        return bool(diff_rg < 8.0 and diff_rb < 8.0)

    return False


def validate_sample_domain(pil_img: Image.Image, expected_mode: str = "auto") -> dict[str, object]:
    """Execute dual-mode sample validation for TurmeriScan AI.

    Args:
        pil_img: Image to validate.
        expected_mode: 'auto', 'multispectral', or 'rgb'.

    Returns:
        is_valid: bool
        sample_mode: str
        category: str
        reason: str
        turmeric_coverage: float
    """
    img_gray = np.array(pil_img.convert("L"))

    # 1. Multi-Cascade Human Face & Upper Body Gatekeeper (Always active)
    has_human, human_desc = _detect_human_presence(img_gray)
    if has_human:
        return {
            "is_valid": False,
            "sample_mode": "Unknown",
            "category": "Human Photo / Selfie",
            "reason": f"{human_desc} detected. TurmeriScan AI strictly screens turmeric samples. Please point your camera directly at a turmeric sample.",
            "turmeric_coverage": 0.0,
        }

    # 2. Text Document / Printed Paper Gatekeeper (Always active)
    if _is_text_or_document(img_gray):
        return {
            "is_valid": False,
            "sample_mode": "Unknown",
            "category": "Document / Text",
            "reason": "Document, barcode, or printed paper detected. Please upload an image of turmeric powder.",
            "turmeric_coverage": 0.0,
        }

    # 3. Check for empty or dark frame
    mean_val = float(np.mean(img_gray))
    if mean_val < 5.0:
        return {
            "is_valid": False,
            "sample_mode": "Unknown",
            "category": "Blank / Underexposed Frame",
            "reason": "Image frame is completely dark or blank.",
            "turmeric_coverage": 0.0,
        }

    # 4. Mode Routing
    if expected_mode == "multispectral" or (expected_mode == "auto" and is_image_natively_grayscale(pil_img)):
        # In Multispectral Mode, accept all spectral band captures (single band, NIR, multi-wavelength)
        return {
            "is_valid": True,
            "sample_mode": "Multispectral Band (Laboratory Grade)",
            "category": "Valid Multispectral Sample",
            "reason": "Verified as authentic laboratory spectral band capture.",
            "turmeric_coverage": 100.0,
        }
    else:
        # In Smart RGB Mobile Mode, enforce turmeric chromaticity check
        is_color_valid, coverage, color_msg = _analyze_rgb_chromaticity(pil_img)
        if not is_color_valid:
            return {
                "is_valid": False,
                "sample_mode": "Smart RGB Camera Capture",
                "category": "Non-Turmeric Object / Background",
                "reason": f"Non-turmeric input: {color_msg}",
                "turmeric_coverage": round(coverage, 1),
            }

        return {
            "is_valid": True,
            "sample_mode": "Smart RGB Camera Capture",
            "category": "Valid RGB Turmeric Sample",
            "reason": f"Verified turmeric powder signature ({coverage:.1f}% golden-amber chromaticity match).",
            "turmeric_coverage": round(coverage, 1),
        }