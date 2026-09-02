"""
TurmeriScan AI - Turmeric Adulteration Detector
--------------------------------------------------
Run locally:
    pip install streamlit tensorflow pillow opencv-python-headless numpy pandas
    streamlit run app.py

Make sure `turmeric_binary_final.h5` is in the same folder as this script.
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import tensorflow as tf

st.set_page_config(page_title="TurmeriScan AI", page_icon="🟡", layout="wide")

# ============================================================
# Styling
# ============================================================
st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #FFFBF0 0%, #FFF8E7 100%); font-family: 'Segoe UI', sans-serif; }

    .hero {
        background: linear-gradient(135deg, #F4A300 0%, #D4820A 100%);
        padding: 36px 24px; border-radius: 18px; margin-bottom: 24px;
        text-align: center; box-shadow: 0 8px 24px rgba(212,130,10,0.25);
    }
    .hero h1 { color: white; font-size: 2.4em; margin: 0; font-weight: 800; }
    .hero p { color: #FFF3D6; font-size: 1.05em; margin-top: 8px; }

    .stat-box {
        background: white; border-radius: 14px; padding: 16px; text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06); border: 1px solid #F0E6D2;
    }
    .stat-number { font-size: 1.8em; font-weight: 800; color: #B8860B; }
    .stat-label { color: #888; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.5px; }

    .result-card {
        background: white; border-radius: 16px; padding: 18px; margin-bottom: 16px;
        box-shadow: 0 3px 14px rgba(0,0,0,0.07); border: 1px solid #F0E6D2;
    }
    .badge-pure {
        background: linear-gradient(90deg, #E8F5E9, #C8E6C9); color: #1B5E20;
        padding: 8px 16px; border-radius: 30px; font-weight: 700; display: inline-block;
        border: 1px solid #A5D6A7;
    }
    .badge-adulterated {
        background: linear-gradient(90deg, #FFEBEE, #FFCDD2); color: #B71C1C;
        padding: 8px 16px; border-radius: 30px; font-weight: 700; display: inline-block;
        border: 1px solid #EF9A9A;
    }
    .badge-uncertain {
        background: linear-gradient(90deg, #FFF8E1, #FFECB3); color: #E65100;
        padding: 8px 16px; border-radius: 30px; font-weight: 700; display: inline-block;
        border: 1px solid #FFD54F;
    }
    .footer-note { color: #999; font-size: 0.85em; text-align: center; margin-top: 30px; }
    div[data-testid="stFileUploader"] { background: white; padding: 20px; border-radius: 14px;
        border: 2px dashed #E8C468; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Model
# ============================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("turmeric_binary_final.h5")

model = load_model()
CLASS_NAMES = ["Adulterated", "Pure"]
CONFIDENCE_THRESHOLD = 65.0

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

def classify(pil_img):
    input_tensor, processed_preview = preprocess_image(pil_img)
    preds = model.predict(input_tensor, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(preds[pred_idx] * 100)
    if confidence < CONFIDENCE_THRESHOLD:
        status = "Inconclusive"
    else:
        status = pred_label
    return status, confidence, processed_preview, preds

# ============================================================
# Header
# ============================================================
st.markdown("""
<div class="hero">
    <h1>🟡 TurmeriScan AI</h1>
    <p>AI-powered turmeric purity detection using multispectral imaging</p>
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️  How this works"):
    st.markdown("""
    This tool analyzes **multispectral images** of turmeric — captured using a camera that
    sees light beyond the human visible range — to detect adulteration (rice flour,
    tartrazine, chalk powder, etc.) that can be invisible to the naked eye or a normal camera.

    **Pipeline:** Spectral Image → Crop + Contrast Enhancement → CNN (MobileNetV2 transfer
    learning) → Pure / Adulterated verdict.

    **Model performance:** 97.4% test accuracy · 100% recall on adulterated samples.
    """)

st.markdown("### 📤 Upload turmeric sample image(s)")
uploaded_files = st.file_uploader(
    "You can select multiple images at once",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

# ============================================================
# Results
# ============================================================
if uploaded_files:
    results = []
    with st.spinner(f"Analyzing {len(uploaded_files)} sample(s)..."):
        for f in uploaded_files:
            pil_img = Image.open(f)
            status, confidence, processed_preview, preds = classify(pil_img)
            results.append({
                "filename": f.name,
                "image": pil_img,
                "processed": processed_preview,
                "status": status,
                "confidence": confidence,
                "preds": preds,
            })

    # ---- Summary stats ----
    n_pure = sum(1 for r in results if r["status"] == "Pure")
    n_adult = sum(1 for r in results if r["status"] == "Adulterated")
    n_unsure = sum(1 for r in results if r["status"] == "Inconclusive")

    st.markdown("### 📊 Summary")
    c1, c2, c3, c4 = st.columns(4)
    for col, num, label in [
        (c1, len(results), "Total Samples"),
        (c2, n_pure, "✅ Pure"),
        (c3, n_adult, "⚠️ Adulterated"),
        (c4, n_unsure, "❓ Inconclusive"),
    ]:
        with col:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{num}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔬 Individual Results")

    badge_map = {
        "Pure": ("badge-pure", "✅ PURE TURMERIC"),
        "Adulterated": ("badge-adulterated", "⚠️ ADULTERATED"),
        "Inconclusive": ("badge-uncertain", "❓ INCONCLUSIVE"),
    }

    # grid layout: 3 cards per row
    cols_per_row = 3
    for i in range(0, len(results), cols_per_row):
        row_results = results[i:i + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, r in zip(cols, row_results):
            with col:
                badge_class, badge_text = badge_map[r["status"]]
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.image(r["image"], use_container_width=True)
                st.markdown(f'<span class="{badge_class}">{badge_text}</span>', unsafe_allow_html=True)
                st.caption(f"**{r['filename']}** · Confidence: {r['confidence']:.1f}%")
                st.progress(r["confidence"] / 100)
                st.markdown('</div>', unsafe_allow_html=True)

    # ---- Table view ----
    with st.expander("📋 View as table"):
        df = pd.DataFrame([{
            "File": r["filename"],
            "Verdict": r["status"],
            "Confidence (%)": round(r["confidence"], 1),
        } for r in results])
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.caption(
        "Note: model trained on spectral-band grayscale images captured under "
        "controlled lab conditions. Accuracy may vary on ordinary phone camera photos."
    )
else:
    st.info("👆 Upload one or more turmeric sample images to get started")

st.markdown(
    '<p class="footer-note">TurmeriScan AI — CNN transfer learning (MobileNetV2) on multispectral turmeric imagery.</p>',
    unsafe_allow_html=True,
)
