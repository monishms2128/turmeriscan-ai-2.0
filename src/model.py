"""Model loading and inference services for TurmeriScan AI."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
import tensorflow as tf

from src.config import CLASS_NAMES, DEFAULT_CONFIDENCE_THRESHOLD, MODEL_PATH
from src.preprocessing import preprocess_image_pipeline


def load_screening_model(model_path: str | Path = MODEL_PATH) -> tf.keras.Model:
    """Load the trained MobileNetV2 screening model."""
    path_obj = Path(model_path)
    if not path_obj.exists():
        raise FileNotFoundError(
            f"Screening model artifact not found at: {path_obj.resolve()}"
        )
    return tf.keras.models.load_model(str(path_obj))


def predict_sample(
    model: tf.keras.Model,
    input_tensor: np.ndarray,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Execute forward inference on a single preprocessed tensor.

    Args:
        model: Loaded Keras model.
        input_tensor: Tensor of shape (1, 224, 224, 3).
        threshold: Minimum confidence (%) required for definitive verdict.

    Returns:
        Dictionary containing predicted class, status verdict, confidence %, and probabilities.
    """
    raw_preds = model.predict(input_tensor, verbose=0)
    probabilities = np.asarray(raw_preds[0]).reshape(-1)

    if probabilities.size != len(CLASS_NAMES):
        raise ValueError(
            f"Model output dimension ({probabilities.size}) does not match class names ({len(CLASS_NAMES)})"
        )

    pred_idx = int(np.argmax(probabilities))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(probabilities[pred_idx] * 100)

    status = pred_label if confidence >= threshold else "Inconclusive"

    return {
        "status": status,
        "predicted_label": pred_label,
        "predicted_idx": pred_idx,
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[i]: float(probabilities[i] * 100) for i in range(len(CLASS_NAMES))
        },
        "raw_probabilities": probabilities,
    }


def process_and_classify_image(
    model: tf.keras.Model,
    pil_image: Image.Image,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Run end-to-end preprocessing, inference, and feature extraction for a PIL image."""
    input_tensor, enhanced_gray, spectral_heatmap = preprocess_image_pipeline(pil_image)
    prediction = predict_sample(model, input_tensor, threshold=threshold)

    return {
        **prediction,
        "input_tensor": input_tensor,
        "enhanced_gray": enhanced_gray,
        "spectral_heatmap": spectral_heatmap,
    }
