"""Unit tests for the TurmeriScan AI Sample Domain & Security Validator."""

import cv2
import numpy as np
from PIL import Image
import pytest

from src.sample_validator import (
    is_image_natively_grayscale,
    validate_sample_domain,
)


def test_valid_rgb_turmeric_image():
    """Verify that a synthetic RGB turmeric patch is accepted in RGB mode."""
    img_arr = np.zeros((200, 200, 3), dtype=np.uint8)
    # Fill center with golden yellow (RGB: 225, 175, 20 -> HSV Hue ~22, Saturation ~230)
    cv2.circle(img_arr, (100, 100), 70, (225, 175, 20), -1)
    noise = np.random.randint(-5, 5, (200, 200, 3), dtype=np.int16)
    img_arr = np.clip(img_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img_arr)
    result = validate_sample_domain(pil_img, expected_mode="rgb")

    assert result["is_valid"] is True
    assert result["sample_mode"] == "Smart RGB Camera Capture"
    assert result["turmeric_coverage"] > 10.0


def test_reject_non_turmeric_blue_object():
    """Verify that a blue non-turmeric image is rejected in RGB mode."""
    img_arr = np.zeros((200, 200, 3), dtype=np.uint8)
    img_arr[:, :] = [20, 40, 220]

    pil_img = Image.fromarray(img_arr)
    result = validate_sample_domain(pil_img, expected_mode="rgb")

    assert result["is_valid"] is False
    assert result["category"] == "Non-Turmeric Object / Background"


def test_valid_multispectral_dish():
    """Verify that a grayscale laboratory capture is accepted in Multispectral mode."""
    img_arr = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(img_arr, (100, 100), 70, 160, -1)
    noise = np.random.normal(0, 15, (200, 200)).astype(np.int16)
    img_arr = np.clip(img_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img_arr)
    result = validate_sample_domain(pil_img, expected_mode="multispectral")

    assert result["is_valid"] is True
    assert result["sample_mode"] == "Multispectral Band (Laboratory Grade)"


def test_reject_blank_or_dark_frame():
    """Verify that an all-black image is rejected in both modes."""
    black_img = Image.fromarray(np.zeros((100, 100), dtype=np.uint8))
    result = validate_sample_domain(black_img, expected_mode="multispectral")
    assert result["is_valid"] is False


def test_is_image_natively_grayscale_helper():
    """Verify detection of grayscale vs 3-channel RGB image."""
    gray_img = Image.fromarray(np.zeros((50, 50), dtype=np.uint8))
    assert is_image_natively_grayscale(gray_img) is True

    equal_rgb = Image.fromarray(np.stack([np.zeros((50, 50), dtype=np.uint8)] * 3, axis=-1))
    assert is_image_natively_grayscale(equal_rgb) is True

    color_rgb = Image.fromarray(np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8))
    assert is_image_natively_grayscale(color_rgb) is False
