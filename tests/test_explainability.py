"""Unit tests for the Grad-CAM visual explainability module."""

import numpy as np
import pytest

from src.config import MODEL_PATH
from src.explainability import generate_gradcam_heatmap, overlay_gradcam
from src.model import load_screening_model


@pytest.fixture(scope="module")
def loaded_model():
    """Load model once for Grad-CAM test suite."""
    return load_screening_model(MODEL_PATH)


def test_generate_gradcam_heatmap(loaded_model):
    """Verify Grad-CAM heatmap generation on nested MobileNetV2 graph."""
    dummy_input = np.random.uniform(-1, 1, (1, 224, 224, 3)).astype(np.float32)
    heatmap = generate_gradcam_heatmap(loaded_model, dummy_input, target_class_idx=0)

    assert heatmap.ndim == 2
    assert heatmap.shape == (7, 7)
    assert np.min(heatmap) >= 0.0
    assert np.max(heatmap) <= 1.0


def test_overlay_gradcam():
    """Verify overlay blending produces RGB image of matching base dimensions."""
    dummy_heatmap = np.random.uniform(0, 1, (7, 7)).astype(np.float32)
    base_gray = np.full((180, 240), 128, dtype=np.uint8)

    overlay = overlay_gradcam(dummy_heatmap, base_gray, alpha=0.5)

    assert overlay.shape == (180, 240, 3)
    assert overlay.dtype == np.uint8
