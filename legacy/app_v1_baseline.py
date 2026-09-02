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

    st.caption(
        "Note: model trained on spectral-band grayscale images captured under "
        "controlled lab conditions. Accuracy may vary on ordinary phone camera photos."
    )
else:
    st.info("👆 Upload an image to get started")

st.markdown("---")
st.caption("Built for hackathon demo — CNN transfer learning (MobileNetV2) on multispectral turmeric imagery.")
