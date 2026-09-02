"""Unit tests for the image preprocessing pipeline."""

import cv2
import numpy as np
from PIL import Image
import pytest

from src.preprocessing import (
    apply_clahe,
    crop_foreground,
    preprocess_image_pipeline,
    pseudo_color_map,
)


def test_crop_foreground_normal():
    """Verify foreground cropping correctly trims dark borders."""
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), 200, -1)
    cropped = crop_foreground(img, threshold_val=5, padding=10)
    assert cropped.shape[0] < 200
    assert cropped.shape[1] < 200
    assert cropped.shape[0] >= 100
    assert cropped.shape[1] >= 100


def test_crop_foreground_all_black():
    """Verify fallback handling when no bright foreground is found."""
    img = np.zeros((100, 100), dtype=np.uint8)
    cropped = crop_foreground(img)
    assert cropped.shape == (100, 100)


def test_crop_foreground_tiny():
    """Verify fallback when detected bounding box is smaller than min dimension."""
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(img, (50, 50), 3, 255, -1)  # 6x6 pixel dot
    cropped = crop_foreground(img, padding=0, min_dim=30)
    assert cropped.shape == (100, 100)


def test_apply_clahe():
    """Verify CLAHE preserves image shape and enhances dynamic range."""
    img = np.full((128, 128), 100, dtype=np.uint8)
    img[32:96, 32:96] = 120
    enhanced = apply_clahe(img)
    assert enhanced.shape == (128, 128)
    assert enhanced.dtype == np.uint8


def test_pseudo_color_map():
    """Verify false-color mapping converts 2D grayscale to 3D RGB."""
    img = np.linspace(0, 255, 10000, dtype=np.uint8).reshape((100, 100))
    colored = pseudo_color_map(img)
    assert colored.shape == (100, 100, 3)
    assert colored.dtype == np.uint8


def test_preprocess_image_pipeline():
    """Verify full end-to-end pipeline returns expected tensor shape and value ranges."""
    pil_img = Image.new("RGB", (300, 300), color=(180, 140, 30))
    tensor, gray_preview, false_color = preprocess_image_pipeline(pil_img)

    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.float32
    # MobileNetV2 normalizes values in range [-1.0, 1.0]
    assert np.min(tensor) >= -1.0
    assert np.max(tensor) <= 1.0
    assert len(gray_preview.shape) == 2
    assert false_color.shape[2] == 3
