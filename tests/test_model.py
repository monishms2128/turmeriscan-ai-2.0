"""Unit tests for model loading, prediction, and reporting services."""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from src.config import CLASS_NAMES, MODEL_PATH
from src.model import load_screening_model, predict_sample, process_and_classify_image
from src.report import compute_kpi_summary, generate_csv_bytes, generate_summary_dataframe


@pytest.fixture(scope="module")
def loaded_model():
    """Load model once for test suite."""
    assert MODEL_PATH.exists(), f"Model file must exist at {MODEL_PATH}"
    return load_screening_model(MODEL_PATH)


def test_model_loading(loaded_model):
    """Verify model loads and has expected input/output shapes."""
    assert loaded_model is not None
    assert loaded_model.input_shape == (None, 224, 224, 3)
    assert loaded_model.output_shape == (None, 2)


def test_predict_sample(loaded_model):
    """Verify prediction output structure and probability constraints."""
    dummy_input = np.random.uniform(-1, 1, (1, 224, 224, 3)).astype(np.float32)
    result = predict_sample(loaded_model, dummy_input, threshold=65.0)

    assert "status" in result
    assert "predicted_label" in result
    assert result["predicted_label"] in CLASS_NAMES
    assert 0.0 <= result["confidence"] <= 100.0

    raw_probs = result["raw_probabilities"]
    assert np.isclose(np.sum(raw_probs), 1.0, atol=1e-4)


def test_confidence_threshold_gating(loaded_model):
    """Verify high threshold triggers Inconclusive status."""
    dummy_input = np.random.uniform(-1, 1, (1, 224, 224, 3)).astype(np.float32)
    # Threshold at 99.99% will almost certainly yield Inconclusive
    result = predict_sample(loaded_model, dummy_input, threshold=99.99)
    assert result["status"] == "Inconclusive"


def test_process_and_classify_image(loaded_model):
    """Verify end-to-end classification from PIL image."""
    img = Image.new("L", (200, 200), color=150)
    result = process_and_classify_image(loaded_model, img, threshold=65.0)

    assert "input_tensor" in result
    assert "enhanced_gray" in result
    assert "spectral_heatmap" in result
    assert result["input_tensor"].shape == (1, 224, 224, 3)


def test_reporting_utilities():
    """Verify KPI computation, DataFrame generation, and CSV serialization."""
    mock_results = [
        {
            "filename": "sample1.png",
            "status": "Pure",
            "confidence": 98.5,
            "probabilities": {"Pure": 98.5, "Adulterated": 1.5},
        },
        {
            "filename": "sample2.png",
            "status": "Adulterated",
            "confidence": 92.1,
            "probabilities": {"Pure": 7.9, "Adulterated": 92.1},
        },
        {
            "filename": "sample3.png",
            "status": "Inconclusive",
            "confidence": 55.0,
            "probabilities": {"Pure": 55.0, "Adulterated": 45.0},
        },
    ]

    kpis = compute_kpi_summary(mock_results)
    assert kpis["total"] == 3
    assert kpis["pure"] == 1
    assert kpis["adulterated"] == 1
    assert kpis["inconclusive"] == 1

    df = generate_summary_dataframe(mock_results)
    assert len(df) == 3
    assert "Verdict" in df.columns

    csv_bytes = generate_csv_bytes(df)
    assert isinstance(csv_bytes, bytes)
    assert len(csv_bytes) > 0
