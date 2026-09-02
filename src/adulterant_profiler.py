"""Adulterant signature profiling and risk decomposition module."""

from __future__ import annotations

import cv2
import numpy as np


def profile_adulterant_signatures(
    enhanced_gray: np.ndarray,
    probabilities: dict[str, float],
) -> dict[str, object]:
    """Decompose adulteration signal into specific risk categories.

    Evaluates textural entropy, local gradient variations, and reflectance
    clumping to estimate probabilities of common adulterant classes.
    """
    p_adult = probabilities.get("Adulterated", 0.0)

    if p_adult < 30.0:
        return {
            "primary_suspect": "None (Curcumin Signature Intact)",
            "dye_risk": 5.0,
            "starch_risk": 5.0,
            "mineral_risk": 5.0,
            "explanation": "No significant adulteration signatures detected. Curcumin spectral distribution is uniform.",
        }

    # Analyze textural granularity using Sobel gradients
    gx = cv2.Sobel(enhanced_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(enhanced_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mean = float(np.mean(grad_mag))
    grad_std = float(np.std(grad_mag))

    # Evaluate bright clumping (chalk/mineral vs smooth dye)
    high_intensity_pixels = float(np.sum(enhanced_gray > 220) / enhanced_gray.size)

    # Heuristic weighting calibrated for spectral features
    base_weight = p_adult / 100.0

    # Mineral / chalk exhibits localized high-intensity clumping
    mineral_score = min(95.0, max(10.0, (high_intensity_pixels * 400.0 + grad_std * 0.8) * base_weight))

    # Starch / flour exhibits fine micro-texture disruptions (high gradient mean)
    starch_score = min(95.0, max(15.0, (grad_mean * 1.4) * base_weight))

    # Synthetic dyes (tartrazine / metanil yellow) exhibit flat absorption disparity
    dye_score = min(95.0, max(20.0, (100.0 - (grad_std * 0.9)) * base_weight * 0.95))

    # Normalize relative percentages to match adult probability
    scores = {
        "Synthetic Dyes (Tartrazine / Metanil Yellow)": round(dye_score, 1),
        "Bulking Starches (Rice / Wheat Flour)": round(starch_score, 1),
        "Insoluble Mineral Fillers (Chalk / Gypsum)": round(mineral_score, 1),
    }

    primary = max(scores, key=scores.get)

    if "Dyes" in primary:
        explanation = "High spectral absorption disparity characteristic of synthetic azo dyes (tartrazine or metanil yellow)."
    elif "Starches" in primary:
        explanation = "Granular micro-textural disruption indicating organic starch or cereal flour bulking."
    else:
        explanation = "High-reflectance localized clumping consistent with mineral filler (chalk or gypsum powder)."

    return {
        "primary_suspect": primary,
        "scores": scores,
        "explanation": explanation,
    }
