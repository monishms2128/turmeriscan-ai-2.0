"""
Turmeric Adulteration Detector - Live Demo App
------------------------------------------------
Run locally:
    pip install streamlit tensorflow pillow opencv-python-headless numpy
    streamlit run app.py

Make sure `turmeric_binary_final.h5` and `gradcam.py` are in the
same folder as this script.
"""

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from gradcam import make_gradcam_heatmap, overlay_heatmap

st.set_page_config(page_title="TurmeriScan AI", page_icon="🟡", layout="centered")

# ----------------------------
# Styling
# ----------------------------
st.markdown("""
<style>
    .main { background-color: #FFFDF7; }
    .stApp { font-family: 'Segoe UI', sans-serif; }
    h1 { color: #B8860B; }
    .verdict-pure {
        background: linear-gradient(90deg, #E8F5E9, #C8E6C9);
        border-left: 6px solid #2E7D32;
        padding: 18px; border-radius: 10px; margin: 10px 0;
    }
    .verdict-adulterated {
        background: linear-gradient(90deg, #FFEBEE, #FFCDD2);
        border-left: 6px solid #C62828;
        padding: 18px; border-radius: 10px; margin: 10px 0;
    }
    .verdict-uncertain {
        background: linear-gradient(90deg, #FFF8E1, #FFECB3);
        border-left: 6px solid #F9A825;
        padding: 18px; border-radius: 10px; margin: 10px 0;
    }
    .footer-note { color: #888; font-size: 0.85em; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load model (cached so it only loads once)
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("turmeric_binary_final.h5")

model = load_model()
CLASS_NAMES = ["Adulterated", "Pure"]  # alphabetical order, matches training
CONFIDENCE_THRESHOLD = 65.0  # below this -> flag as inconclusive

# ----------------------------
# Preprocessing - must match training pipeline exactly
# ----------------------------
def preprocess_image(pil_img):
    img = np.array(pil_img.convert("L"))  # grayscale

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

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(cropped)

    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(rgb, (224, 224))

    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(resized.astype(np.float32))
    return np.expand_dims(preprocessed, axis=0), enhanced


# ----------------------------
# Header
# ----------------------------
st.markdown("<h1 style='text-align:center;'>🟡 TurmeriScan AI</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#666; font-size:1.05em;'>"
    "AI-powered turmeric purity check using multispectral imaging"
    "</p>", unsafe_allow_html=True
)

with st.expander("ℹ️ How this works"):
    st.markdown("""
    This tool analyzes **multispectral images** of turmeric samples — captured using a
    special camera that sees light beyond the human visible range — to detect adulteration
    (e.g. rice flour, tartrazine, chalk powder) that can be invisible to the naked eye or a
    normal camera.

    **Pipeline:** Spectral Image → Crop + Contrast Enhancement → CNN (MobileNetV2 transfer
    learning) → Pure / Adulterated verdict, with a Grad-CAM explanation of what the model focused on.

    *Model accuracy: 97.4% on held-out test data, 100% recall on adulterated samples.*
    """)

st.markdown("---")

# ----------------------------
# Upload
# ----------------------------
uploaded_file = st.file_uploader("📤 Upload a turmeric sample image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, caption="Original Upload", use_container_width=True)

    with st.spinner("Analyzing sample..."):
        input_tensor, processed_preview = preprocess_image(pil_img)

        with col2:
            st.image(processed_preview, caption="Processed (cropped + enhanced)",
                      use_container_width=True, clamp=True)

        preds = model.predict(input_tensor, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        pred_label = CLASS_NAMES[pred_idx]
        confidence = preds[pred_idx] * 100

    st.markdown("---")

    if confidence < CONFIDENCE_THRESHOLD:
        st.markdown(
            f"<div class='verdict-uncertain'>⚠️ <b>INCONCLUSIVE</b> — Confidence only "
            f"{confidence:.1f}%. Please retest with a clearer sample image.</div>",
            unsafe_allow_html=True,
        )
    elif pred_label == "Pure":
        st.markdown(
            f"<div class='verdict-pure'>✅ <b>PURE TURMERIC</b> — Confidence: {confidence:.1f}%</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='verdict-adulterated'>⚠️ <b>ADULTERATED</b> — Confidence: {confidence:.1f}%</div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Prediction breakdown")
    for i, cls in enumerate(CLASS_NAMES):
        st.progress(float(preds[i]), text=f"{cls}: {preds[i]*100:.1f}%")

    with st.expander("🔍 Show Grad-CAM (where the model is looking)"):
        with st.spinner("Generating heatmap..."):
            try:
                heatmap = make_gradcam_heatmap(input_tensor, model, pred_idx)
                overlay_rgb = overlay_heatmap(heatmap, processed_preview)
                if overlay_rgb is not None:
                    st.image(overlay_rgb, caption="Red/yellow = high influence on the decision",
                              use_container_width=True)
                else:
                    st.warning("Could not generate Grad-CAM for this image.")
            except Exception as e:
                st.warning(f"Grad-CAM unavailable: {e}")

    st.caption(
        "Note: model trained on spectral-band grayscale images captured under "
        "controlled lab conditions. Accuracy may vary on ordinary phone camera photos."
    )
else:
    st.info("👆 Upload an image to get started")

st.markdown("---")
st.markdown(
    "<p class='footer-note' style='text-align:center;'>"
    "TurmeriScan AI — CNN transfer learning (MobileNetV2) on multispectral turmeric imagery."
    "</p>", unsafe_allow_html=True
)
