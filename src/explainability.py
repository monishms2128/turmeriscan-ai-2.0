"""Grad-CAM visual explainability module for TurmeriScan AI."""

from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf


def generate_gradcam_heatmap(
    model: tf.keras.Model,
    input_tensor: np.ndarray,
    target_class_idx: int | None = None,
) -> np.ndarray:
    """Generate a Grad-CAM heatmap highlighting spatial regions influencing the model's verdict.

    Args:
        model: Trained Keras functional model containing nested MobileNetV2 backbone.
        input_tensor: Preprocessed tensor of shape (1, 224, 224, 3).
        target_class_idx: Target class index (0 for Adulterated, 1 for Pure).
                          If None, uses the predicted class.

    Returns:
        2D numpy array with normalized heatmap values in [0, 1].
    """
    backbone_name = "mobilenetv2_1.00_224"
    try:
        backbone = model.get_layer(backbone_name)
    except ValueError:
        # Fallback search for any nested functional model layer
        backbone = next(
            (layer for layer in model.layers if isinstance(layer, tf.keras.Model)), None
        )
        if backbone is None:
            raise ValueError(f"Could not locate backbone layer in model.")

    # Target the last convolution layer in MobileNetV2
    last_conv_name = "Conv_1"
    try:
        last_conv_layer = backbone.get_layer(last_conv_name)
    except ValueError:
        # Fallback to finding the last 4D output layer in backbone
        conv_layers = [l for l in backbone.layers if len(l.output_shape) == 4]
        if not conv_layers:
            raise ValueError("No 4D convolutional layers found in backbone.")
        last_conv_layer = conv_layers[-1]

    # Build intermediate feature extractor for backbone
    backbone_grad_model = tf.keras.models.Model(
        inputs=backbone.input,
        outputs=[last_conv_layer.output, backbone.output],
    )

    # Collect head layers following the backbone
    head_layers = []
    reached_backbone = False
    for layer in model.layers:
        if layer.name == backbone.name:
            reached_backbone = True
            continue
        if reached_backbone:
            head_layers.append(layer)

    with tf.GradientTape() as tape:
        conv_outputs, backbone_out = backbone_grad_model(input_tensor)
        tape.watch(conv_outputs)

        x = backbone_out
        for layer in head_layers:
            x = layer(x)
        predictions = x

        if target_class_idx is None:
            target_class_idx = int(tf.argmax(predictions[0]))

        target_score = predictions[:, target_class_idx]

    # Calculate gradients of the class score with respect to feature maps
    grads = tape.gradient(target_score, conv_outputs)
    if grads is None:
        return np.zeros((7, 7), dtype=np.float32)

    # Pool gradients across spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature channels
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU and normalize
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def overlay_gradcam(
    heatmap: np.ndarray,
    base_image: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Superimpose Grad-CAM heatmap onto base image.

    Args:
        heatmap: 2D array [0, 1] from `generate_gradcam_heatmap`.
        base_image: 2D grayscale or 3D RGB/BGR image.
        alpha: Transparency weight for the heatmap overlay.
        colormap: OpenCV colormap enum (e.g. cv2.COLORMAP_JET).

    Returns:
        RGB image array with overlay.
    """
    h, w = base_image.shape[:2]
    resized_heatmap = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * np.clip(resized_heatmap, 0, 1))
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)

    if len(base_image.shape) == 2:
        base_rgb = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    elif base_image.shape[2] == 3:
        base_rgb = cv2.cvtColor(base_image, cv2.COLOR_RGB2BGR)
    else:
        base_rgb = base_image

    overlaid = cv2.addWeighted(colored_heatmap, alpha, base_rgb, 1 - alpha, 0)
    return cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB)
