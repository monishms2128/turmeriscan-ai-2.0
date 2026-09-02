"""
Turmeric Adulteration Detector - Live Demo App (Polished UI)
----------------------------------------------------------------
Run locally:
    streamlit run app.py

Make sure `turmeric_binary_final.h5` is in the same folder as this script.
"""

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf

st.set_page_config(
    page_title="TurmeriScan AI | Adulteration Detector",
    page_icon="🟡",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    .main {
        background-color: #FFFDF7;
    }
    .hero {
        background: linear-gradient(135deg, #F4A300 0%, #E67E22 100%);
        padding: 2.2rem 1.5rem;
        border-radius: 18px;
        text-align: center;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 24px rgba(230, 126, 34, 0.25);
    }
    .hero h1 {
        color: white;
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #FFF3E0;
        font-size: 1.02rem;
        margin: 0;
    }
    .stat-card {
        background: white;
        border-radius: 14px;
        padding: 1rem 0.5rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border: 1px solid #F0E6D2;
    }
    .stat-number {
        font-size: 1.6rem;
        font-weight: 800;
        color: #E67E22;
    }
    .stat-label {
        font-size: 0.78rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .upload-section {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 2px 14px rgba(0,0,0,0.06);
        border: 1px solid #F0E6D2;
        margin-bottom: 1.5rem;
    }
    .result-pure {
        background: linear-gradient(135deg, #2ECC71 0%, #27AE60 100%);
        color: white;
        padding: 1.3rem;
        border-radius: 14px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 800;
        box-shadow: 0 6px 18px rgba(46, 204, 113, 0.3);
        margin: 1rem 0;
    }
    .result-adulterated {
        background: linear-gradient(135deg, #E74C3C 0%, #C0392B 100%);
        color: white;
        padding: 1.3rem;
        border-radius: 14px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 800;
        box-shadow: 0 6px 18px rgba(231, 76, 60, 0.3);
        margin: 1rem 0;
    }
    .footer-note {
        text-align: center;
        color: #999;
        font-size: 0.82rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #F4A300;
        border-radius: 12px;
        padding: 0.5rem;
        background: #FFF9EE;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Load model
# ============================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("turmeric_binary_final.h5")

model = load_model()
CLASS_NAMES = ["Adulterated", "Pure"]

# ============================================================
# Preprocessing - must match training pipeline exactly
# ============================================================
def preprocess_image(pil_img):
    img = np.array(pil_img.convert("L"))

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


def pseudo_color(gray_img):
    """Apply a false-color heatmap for a more visually striking preview."""
    normalized = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("### 🟡 TurmeriScan AI")
    st.markdown("**Multispectral Adulteration Detection**")
    st.markdown("---")
    st.markdown("#### How it works")
    st.markdown(
        "1. 📷 Multispectral camera captures sample\n"
        "2. 🔧 Image cropped + contrast-enhanced\n"
        "3. 🧠 CNN (MobileNetV2 transfer learning) classifies\n"
        "4. ✅ Instant Pure / Adulterated verdict"
    )
    st.markdown("---")
    st.markdown("#### Model performance")
    st.metric("Test Accuracy", "97.4%")
    st.metric("Adulterant Recall", "100%")
    st.markdown("---")
    st.caption("Built for detecting common adulterants: rice flour, tartrazine, chalk powder, and synthetic dyes.")

# ============================================================
# Hero header
# ============================================================
st.markdown("""
<div class="hero">
    <h1>🟡 TurmeriScan AI</h1>
    <p>AI-powered turmeric purity detection using multispectral imaging</p>
</div>
""", unsafe_allow_html=True)

# Stat cards
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="stat-card"><div class="stat-number">97.4%</div><div class="stat-label">Accuracy</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-card"><div class="stat-number">100%</div><div class="stat-label">Recall</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-card"><div class="stat-number">&lt;1s</div><div class="stat-label">Prediction Time</div></div>', unsafe_allow_html=True)

st.write("")

# ============================================================
# Upload section
# ============================================================
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("#### 📤 Upload a Turmeric Sample Image")
st.caption("Upload a multispectral/grayscale image captured from a turmeric sample.")
uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)

    with st.spinner("🔬 Analyzing spectral signature..."):
        input_tensor, processed_preview = preprocess_image(pil_img)
        colored_preview = pseudo_color(processed_preview)
        preds = model.predict(input_tensor, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        pred_label = CLASS_NAMES[pred_idx]
        confidence = preds[pred_idx] * 100

    st.markdown("#### 🖼️ Sample Views")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(pil_img, caption="Original Upload", use_container_width=True)
    with col2:
        st.image(processed_preview, caption="Enhanced (crop + CLAHE)", use_container_width=True, clamp=True)
    with col3:
        st.image(colored_preview, caption="Spectral Heatmap View", use_container_width=True)

    if pred_label == "Pure":
        st.markdown(f'<div class="result-pure">✅ PURE TURMERIC — Confidence: {confidence:.1f}%</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-adulterated">⚠️ ADULTERATED — Confidence: {confidence:.1f}%</div>', unsafe_allow_html=True)

    st.markdown("##### Prediction Breakdown")
    for i, cls in enumerate(CLASS_NAMES):
        st.progress(float(preds[i]), text=f"{cls}: {preds[i]*100:.1f}%")

else:
    st.info("👆 Upload a sample image above to run detection")

st.markdown("""
<div class="footer-note">
    Model trained on spectral-band imagery under controlled lab conditions using transfer learning (MobileNetV2).<br>
    Built for hackathon demo — TurmeriScan AI Team.
</div>
""", unsafe_allow_html=True)
