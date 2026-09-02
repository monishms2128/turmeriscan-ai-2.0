"""
Turmeric Adulteration Detector - Hackathon Demo UI (v2)
-----------------------------------------------------------
Run locally:
    streamlit run app.py

Make sure `turmeric_binary_final.h5` is in the same folder as this script.
"""

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import time

st.set_page_config(
    page_title="TurmeriScan AI | Adulteration Detector",
    page_icon="🟡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.hero-banner {
    background: linear-gradient(135deg, #F4A300 0%, #E67E22 50%, #C0392B 100%);
    padding: 2.5rem 2rem;
    border-radius: 20px;
    margin-bottom: 1.8rem;
    box-shadow: 0 10px 30px rgba(230, 126, 34, 0.25);
}
.hero-title {
    font-family: 'Poppins', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: white;
    margin: 0;
    text-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.hero-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 1.05rem;
    color: rgba(255,255,255,0.92);
    margin-top: 0.5rem;
    font-weight: 500;
}
.hero-badges { margin-top: 1rem; }
.badge {
    display: inline-block;
    background: rgba(255,255,255,0.22);
    backdrop-filter: blur(4px);
    color: white;
    padding: 0.35rem 0.9rem;
    border-radius: 50px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-right: 0.5rem;
    border: 1px solid rgba(255,255,255,0.3);
}

.section-card {
    background: white;
    border-radius: 16px;
    padding: 1.6rem;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
    border: 1px solid #f0f0f0;
    margin-bottom: 1.2rem;
}

.upload-label {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.15rem;
    color: #2c2c2c;
    margin-bottom: 0.3rem;
}
.upload-hint {
    color: #888;
    font-size: 0.88rem;
    margin-bottom: 1rem;
}

.result-card-pure {
    background: linear-gradient(135deg, #2ECC71 0%, #27AE60 100%);
    border-radius: 18px;
    padding: 1.8rem 2rem;
    text-align: center;
    box-shadow: 0 8px 24px rgba(46, 204, 113, 0.35);
}
.result-card-adult {
    background: linear-gradient(135deg, #E74C3C 0%, #C0392B 100%);
    border-radius: 18px;
    padding: 1.8rem 2rem;
    text-align: center;
    box-shadow: 0 8px 24px rgba(231, 76, 60, 0.35);
}
.result-icon { font-size: 3rem; margin-bottom: 0.3rem; }
.result-label {
    font-family: 'Poppins', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    color: white;
    margin: 0.2rem 0;
}
.result-confidence {
    color: rgba(255,255,255,0.9);
    font-size: 1rem;
    font-weight: 600;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fffaf0 0%, #fff5e6 100%);
}

.img-caption {
    text-align: center;
    color: #666;
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 0.4rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("turmeric_binary_final.h5")

model = load_model()
CLASS_NAMES = ["Adulterated", "Pure"]

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

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🟡 TurmeriScan AI")
    st.markdown("---")
    st.markdown("#### 📊 Model Performance")
    st.metric("Test Accuracy", "97.4%")
    st.metric("Recall (Adulterated)", "100%")
    st.markdown("---")
    st.markdown("#### 🔬 How it works")
    st.markdown("""
    1. **Capture** — multispectral image of turmeric sample
    2. **Preprocess** — crop + contrast enhancement
    3. **Classify** — CNN (MobileNetV2 transfer learning)
    4. **Verdict** — Pure or Adulterated, with confidence
    """)
    st.markdown("---")
    st.markdown("#### ⚠️ Detects adulterants like")
    st.markdown("- Rice flour\n- Tartrazine dye\n- Starch / chalk powder")
    st.markdown("---")
    st.caption("Built for Hackathon 2026 · CNN Transfer Learning on Multispectral Imagery")

# ============================================================
# HERO BANNER
# ============================================================
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🟡 TurmeriScan AI</div>
    <div class="hero-subtitle">Detecting turmeric adulteration using multispectral imaging & deep learning — catching what the naked eye can't see.</div>
    <div class="hero-badges">
        <span class="badge">🎯 97.4% Accuracy</span>
        <span class="badge">🛡️ 100% Adulterant Recall</span>
        <span class="badge">⚡ Instant Results</span>
        <span class="badge">🔬 Multispectral Imaging</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MAIN CONTENT
# ============================================================
col_main, col_stats = st.columns([2.2, 1])

with col_main:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="upload-label">📤 Upload a Turmeric Sample Image</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-hint">Accepts multispectral/grayscale sample images (.jpg, .png)</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)

        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(pil_img, use_container_width=True)
            st.markdown('<div class="img-caption">📷 Original Sample</div>', unsafe_allow_html=True)

        with st.spinner("🔍 Analyzing spectral signature..."):
            time.sleep(0.6)
            input_tensor, processed_preview = preprocess_image(pil_img)
            preds = model.predict(input_tensor, verbose=0)[0]
            pred_idx = int(np.argmax(preds))
            pred_label = CLASS_NAMES[pred_idx]
            confidence = preds[pred_idx] * 100

        with img_col2:
            st.image(processed_preview, use_container_width=True, clamp=True)
            st.markdown('<div class="img-caption">⚙️ Enhanced (Cropped + CLAHE)</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if pred_label == "Pure":
            st.markdown(f"""
            <div class="result-card-pure">
                <div class="result-icon">✅</div>
                <div class="result-label">PURE TURMERIC</div>
                <div class="result-confidence">Confidence: {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card-adult">
                <div class="result-icon">⚠️</div>
                <div class="result-label">ADULTERATED</div>
                <div class="result-confidence">Confidence: {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Prediction Breakdown**")
        b1, b2 = st.columns(2)
        with b1:
            st.progress(float(preds[0]), text=f"⚠️ Adulterated: {preds[0]*100:.1f}%")
        with b2:
            st.progress(float(preds[1]), text=f"✅ Pure: {preds[1]*100:.1f}%")

        st.caption("ℹ️ Model trained on spectral-band imagery captured under controlled lab conditions.")
    else:
        st.info("👆 Upload a sample image above to run the detector")

with col_stats:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### 📈 Why Multispectral?")
    st.markdown("""
    Adulterants like tartrazine dye can perfectly mimic turmeric's yellow color to the human eye —
    but they reflect light differently in the near-infrared spectrum. Our model catches this
    invisible signature that no ordinary camera can see.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### 🧪 Common Adulterants")
    st.markdown("""
    | Adulterant | Purpose |
    |---|---|
    | Rice flour | Increase weight |
    | Tartrazine | Enhance color |
    | Chalk powder | Increase weight |
    | Metanil yellow | Enhance color |
    """)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("🎓 Hackathon 2026 Project · CNN Transfer Learning (MobileNetV2) on Multispectral Turmeric Imagery")
