"""Image preprocessing and enhancement pipeline for spectral turmeric screening."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

from src.config import (
    BINARY_BG_THRESHOLD,
    CLAHE_CLIP_LIMIT,
    CLAHE_GRID_SIZE,
    CROP_PADDING_PX,
    INPUT_SHAPE,
    MIN_CROP_DIMENSION_PX,
)


def crop_foreground(
    gray_img: np.ndarray,
    threshold_val: int = BINARY_BG_THRESHOLD,
    padding: int = CROP_PADDING_PX,
    min_dim: int = MIN_CROP_DIMENSION_PX,
) -> np.ndarray:
    """Isolate the sample area from dark background with padding and dimension safety."""
    if gray_img.size == 0:
        return gray_img

    _, binary_mask = cv2.threshold(gray_img, threshold_val, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary_mask)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(gray_img.shape[1], x + w + padding)
        y1 = min(gray_img.shape[0], y + h + padding)
        cropped = gray_img[y0:y1, x0:x1]

        if cropped.shape[0] >= min_dim and cropped.shape[1] >= min_dim:
            return cropped

    return gray_img


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    grid_size: tuple[int, int] = CLAHE_GRID_SIZE,
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization to accentuate spectral textures."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)


def pseudo_color_map(
    gray_img: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO
) -> np.ndarray:
    """Convert a grayscale spectral image into a false-color absorption heatmap."""
    normalized = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, colormap)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def preprocess_image_pipeline(
    pil_img: Image.Image,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute complete end-to-end preprocessing pipeline for MobileNetV2.

    Returns:
        input_tensor: Shape (1, 224, 224, 3) ready for model prediction
        enhanced_gray: CLAHE-enhanced grayscale 2D array
        spectral_heatmap: False-color RGB array for visual inspection
    """
    # Convert input to single-channel luminance / spectral band
    gray = np.array(pil_img.convert("L"))

    # Crop foreground
    cropped = crop_foreground(gray)

    # Local contrast enhancement
    enhanced_gray = apply_clahe(cropped)

    # Generate false-color spectral view
    spectral_heatmap = pseudo_color_map(enhanced_gray)

    # Replicate into 3-channel RGB for MobileNetV2
    rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(rgb, (INPUT_SHAPE[0], INPUT_SHAPE[1]))

    # MobileNetV2 scaling to [-1, 1]
    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(
        resized.astype(np.float32)
    )
    input_tensor = np.expand_dims(preprocessed, axis=0)

    return input_tensor, enhanced_gray, spectral_heatmap
