"""
Turmeric Adulteration Detector - Live Demo App
------------------------------------------------
Run locally:
    pip install streamlit tensorflow pillow opencv-python-headless numpy
    streamlit run app.py

Make sure `turmeric_binary_final.h5` (downloaded from Colab) is in the
same folder as this script.
"""

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from gradcam import make_gradcam_heatmap, overlay_heatmap

st.set_page_config(page_title="Turmeric Adulteration Detector", page_icon="🟡", layout="centered")

# ----------------------------
# Load model (cached so it only loads once)
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("turmeric_binary_final.h5")

model = load_model()
CLASS_NAMES = ["Adulterated", "Pure"]  # alphabetical order, matches training

# ----------------------------
# Preprocessing - must match training pipeline exactly
# ----------------------------
def preprocess_image(pil_img):
    img = np.array(pil_img.convert("L"))  # grayscale

    # crop to bright region (same as training preprocessing)
    _, thresh = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = 20
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
        cropped = img[y0:y1, x0:x1]
        if cropped.shape[0] < 30 or cropped.shape[1] < 30:
            cropped = img
    else:
        cropped = img

    # CLAHE contrast enhancement (same as training)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(cropped)

    # convert back to RGB (3-channel) and resize
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(rgb, (224, 224))

    # MobileNetV2 preprocessing
    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(resized.astype(np.float32))
    return np.expand_dims(preprocessed, axis=0), enhanced


# ----------------------------
# Grad-CAM: shows which part of the image the model focused on
# ----------------------------
def find_base_backbone(keras_model):
    """Find the nested MobileNetV2 sub-model inside our functional model."""
    for layer in keras_model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    return None

def find_last_conv_layer(backbone):
    """Find the last layer with a 4D output (conv-like) in the backbone."""
    for layer in reversed(backbone.layers):
        if len(layer.output_shape) == 4:
            return layer.name
    return None

def make_gradcam_heatmap(input_tensor, model, pred_index):
    backbone = find_base_backbone(model)
    if backbone is None:
        return None
    last_conv_name = find_last_conv_layer(backbone)
    if last_conv_name is None:
        return None

    grad_model = tf.keras.models.Model(
        inputs=backbone.input,
        outputs=[backbone.get_layer(last_conv_name).output, backbone.output],
    )

    # rebuild the head (layers after the backbone) to connect to final prediction
    with tf.GradientTape() as tape:
        conv_output, backbone_output = grad_model(input_tensor)
        tape.watch(conv_output)
        x = backbone_output
        for layer in model.layers:
            if layer.name == backbone.name:
                continue
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
        # re-run through the head layers manually
        head_started = False
        for layer in model.layers:
            if layer is backbone:
                head_started = True
                continue
            if not head_started:
                continue
            x = layer(x)
        preds = x
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    if grads is None:
        return None
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(heatmap, original_bgr_or_gray, alpha=0.45):
    if heatmap is None:
        return None
    heatmap_resized = cv2.resize(heatmap, (original_bgr_or_gray.shape[1], original_bgr_or_gray.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    if len(original_bgr_or_gray.shape) == 2:
        base = cv2.cvtColor(original_bgr_or_gray, cv2.COLOR_GRAY2BGR)
    else:
        base = original_bgr_or_gray

    overlaid = cv2.addWeighted(heatmap_color, alpha, base, 1 - alpha, 0)
    return cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB)


# ----------------------------
# UI
# ----------------------------
st.title("🟡 Turmeric Adulteration Detector")
st.markdown(
    "Upload a spectral/grayscale image of a turmeric sample to check whether "
    "it's **pure** or **adulterated** (e.g. with rice flour, tartrazine, or other fillers)."
)

uploaded_file = st.file_uploader("Upload turmeric sample image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, caption="Original Upload", use_container_width=True)

    with st.spinner("Analyzing sample..."):
        input_tensor, processed_preview = preprocess_image(pil_img)

        with col2:
            st.image(processed_preview, caption="Processed (cropped + enhanced)", use_container_width=True, clamp=True)

        preds = model.predict(input_tensor)[0]
        pred_idx = int(np.argmax(preds))
        pred_label = CLASS_NAMES[pred_idx]
        confidence = preds[pred_idx] * 100

    st.markdown("---")
    if pred_label == "Pure":
        st.success(f"✅ **PURE TURMERIC** — Confidence: {confidence:.1f}%")
    else:
        st.error(f"⚠️ **ADULTERATED** — Confidence: {confidence:.1f}%")

    st.markdown("#### Prediction breakdown")
    for i, cls in enumerate(CLASS_NAMES):
        st.progress(float(preds[i]), text=f"{cls}: {preds[i]*100:.1f}%")

    with st.expander("🔍 Show Grad-CAM (where the model is looking)"):
        with st.spinner("Generating heatmap..."):
            try:
                heatmap = make_gradcam_heatmap(input_tensor, model, pred_idx)
                overlay_rgb = overlay_heatmap(heatmap, processed_preview)
                if overlay_rgb is not None:
                    st.image(overlay_rgb, caption="Grad-CAM: red/yellow = high influence on decision",
                              use_container_width=True)
                else:
                    st.warning("Could not generate Grad-CAM for this model.")
            except Exception as e:
                st.warning(f"Grad-CAM unavailable: {e}")

    st.caption(
        "Note: model trained on spectral-band grayscale images captured under "
        "controlled lab conditions. Accuracy may vary on ordinary phone camera photos."
    )
else:
    st.info("👆 Upload an image to get started")

st.markdown("---")
st.caption("Built for hackathon demo — CNN transfer learning (MobileNetV2) on multispectral turmeric imagery.")
