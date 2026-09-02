"""Unit tests for optical quality validator, adulterant profiler, and PDF certificate generator."""

import numpy as np
from PIL import Image

from src.adulterant_profiler import profile_adulterant_signatures
from src.certificate import generate_pdf_certificate
from src.quality_validator import assess_image_quality


def test_assess_image_quality_sharp():
    """Verify quality assessor on a sharp sample."""
    img = np.zeros((200, 200), dtype=np.uint8)
    img[50:150, 50:150] = 180
    pil_img = Image.fromarray(img)

    quality = assess_image_quality(pil_img)
    assert "status" in quality
    assert "sharpness_score" in quality
    assert quality["status"] in ["Optimal", "Acceptable", "Suboptimal"]


def test_adulterant_profiler_pure():
    """Verify profiler returns clean curcumin profile for pure samples."""
    gray = np.full((200, 200), 120, dtype=np.uint8)
    probs = {"Pure": 98.0, "Adulterated": 2.0}

    profile = profile_adulterant_signatures(gray, probs)
    assert "Curcumin" in profile["primary_suspect"]


def test_adulterant_profiler_adulterated():
    """Verify profiler returns risk scores for adulterated samples."""
    gray = np.random.randint(50, 240, (200, 200), dtype=np.uint8)
    probs = {"Pure": 10.0, "Adulterated": 90.0}

    profile = profile_adulterant_signatures(gray, probs)
    assert "scores" in profile
    assert len(profile["scores"]) == 3
    assert profile["primary_suspect"] != "None"


def test_pdf_certificate_generation():
    """Verify PDF certificate produces valid binary PDF data."""
    orig = Image.new("RGB", (200, 200), color=(180, 140, 20))
    gradcam = np.zeros((200, 200, 3), dtype=np.uint8)
    probs = {"Pure": 95.0, "Adulterated": 5.0}

    pdf_bytes = generate_pdf_certificate(
        sample_name="Lab_Sample_01.png",
        status="Pure",
        confidence=95.0,
        probabilities=probs,
        original_img=orig,
        gradcam_img=gradcam,
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    # PDF standard magic header
    assert pdf_bytes.startswith(b"%PDF")
