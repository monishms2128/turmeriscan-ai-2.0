"""
TurmeriScan AI — Turmeric Adulteration Screening

Run locally:
    pip install streamlit tensorflow pillow opencv-python-headless numpy pandas
    streamlit run turmeriscan_refined.py

Place `turmeric_binary_final.h5` in the same directory as this script.
"""

from __future__ import annotations

import html
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st
import tensorflow as tf


# -----------------------------------------------------------------------------
# Page configuration and visual system
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TurmeriScan | Quality Screening",
    page_icon="TS",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        :root {
            --ink: #17211b;
            --muted: #627068;
            --line: #dce5da;
            --paper: #fbfdf9;
            --saffron: #eea51b;
            --saffron-deep: #b96c08;
            --leaf: #20613d;
            --pure: #17603a;
            --alert: #b83725;
            --caution: #986500;
        }

        .stApp {
            background:
                radial-gradient(circle at 88% -6%, rgba(250, 209, 100, 0.24), transparent 28rem),
                radial-gradient(circle at -5% 42%, rgba(164, 211, 167, 0.14), transparent 24rem),
                var(--paper);
            color: var(--ink);
        }
        .block-container { max-width: 1240px; padding-top: 2.1rem; padding-bottom: 2.5rem; }
        header[data-testid="stHeader"] { background: rgba(251, 253, 249, 0.78); }

        .app-header {
            display: flex; align-items: center; justify-content: space-between; gap: 1rem;
            padding: 1.15rem 0 1.9rem; border-bottom: 1px solid var(--line); margin-bottom: 1.5rem;
        }
        .brand-wrap { display: flex; align-items: center; gap: 0.82rem; }
        .brand-mark {
            display: grid; place-items: center; width: 42px; height: 42px; border-radius: 13px;
            color: #fffaf0; background: linear-gradient(145deg, #f6bc35, #d47b08);
            box-shadow: 0 8px 20px rgba(191, 113, 0, 0.22); font-size: 0.78rem; font-weight: 800; letter-spacing: .08em;
        }
        .brand-name { color: var(--ink); font-size: 1.18rem; font-weight: 750; letter-spacing: -0.03em; }
        .brand-subtitle { color: var(--muted); font-size: .78rem; margin-top: .06rem; }
        .header-tag {
            color: var(--leaf); background: #eef6ef; border: 1px solid #d3e7d5; border-radius: 999px;
            padding: .42rem .72rem; font-size: .69rem; font-weight: 750; letter-spacing: .075em;
        }

        .hero-panel {
            position: relative; overflow: hidden; padding: 2.35rem; border-radius: 22px;
            background: linear-gradient(122deg, #153c29 0%, #1d6140 57%, #36784d 100%);
            box-shadow: 0 18px 42px rgba(25, 73, 45, 0.18); margin-bottom: 1.5rem;
        }
        .hero-panel::after {
            content: ""; position: absolute; width: 260px; height: 260px; right: -62px; top: -105px;
            border-radius: 50%; border: 38px solid rgba(247, 194, 73, 0.18);
        }
        .eyebrow { color: #f8ca65; font-size: .7rem; font-weight: 800; letter-spacing: .15em; margin-bottom: .68rem; }
        .hero-panel h1 { color: #ffffff; font-weight: 750; font-size: clamp(1.9rem, 3.7vw, 3rem); letter-spacing: -.055em; line-height: 1.04; margin: 0; max-width: 760px; }
        .hero-panel p { position: relative; z-index: 1; color: #dcebdd; font-size: 1rem; line-height: 1.55; max-width: 680px; margin: .95rem 0 0; }

        .section-kicker { color: var(--saffron-deep); font-size: .7rem; font-weight: 800; letter-spacing: .13em; margin-bottom: .28rem; }
        h2.section-title { color: var(--ink); font-size: 1.32rem; letter-spacing: -.035em; margin: 0 0 .22rem; }
        .section-lede { color: var(--muted); font-size: .92rem; margin: 0 0 1.05rem; }

        .how-card, .empty-state, .model-note {
            background: rgba(255,255,255,.82); border: 1px solid var(--line); border-radius: 16px;
        }
        .how-card { padding: 1.15rem 1.2rem; margin: .4rem 0 1.05rem; }
        .how-title { color: var(--ink); font-size: .92rem; font-weight: 750; margin-bottom: .7rem; }
        .process-row { display: flex; align-items: center; flex-wrap: wrap; gap: .45rem; }
        .process-step { color: #3e5145; background: #f1f6ef; border: 1px solid #deeadc; border-radius: 8px; padding: .45rem .62rem; font-size: .76rem; font-weight: 650; }
        .process-arrow { color: #9aab9c; font-size: .85rem; }

        div[data-testid="stFileUploader"] {
            padding: 1.1rem; background: rgba(255,255,255,.86); border: 1.5px dashed #c6a34e;
            border-radius: 16px;
        }
        div[data-testid="stFileUploader"] section { background: #fffdf7; }
        div[data-testid="stFileUploader"] button { border-radius: 8px; }

        .metric-card {
            min-height: 106px; padding: 1rem 1rem .9rem; background: rgba(255,255,255,.88);
            border: 1px solid var(--line); border-radius: 15px;
        }
        .metric-label { color: var(--muted); font-size: .7rem; font-weight: 750; letter-spacing: .09em; text-transform: uppercase; }
        .metric-value { color: var(--ink); font-size: 1.85rem; line-height: 1.1; font-weight: 760; letter-spacing: -.05em; padding-top: .4rem; }
        .metric-card.pure .metric-value { color: var(--pure); }
        .metric-card.alert .metric-value { color: var(--alert); }
        .metric-card.caution .metric-value { color: var(--caution); }

        .result-shell {
            background: rgba(255,255,255,.9); border: 1px solid var(--line); border-radius: 16px;
            padding: .9rem; margin-bottom: .9rem;
        }
        .result-top { display: flex; align-items: flex-start; justify-content: space-between; gap: .55rem; margin-bottom: .72rem; }
        .file-name { color: var(--ink); font-weight: 720; font-size: .88rem; overflow-wrap: anywhere; line-height: 1.3; }
        .sample-number { color: var(--muted); font-size: .71rem; margin-bottom: .17rem; letter-spacing: .06em; text-transform: uppercase; }
        .status-pill { display: inline-block; padding: .32rem .56rem; border-radius: 999px; font-size: .67rem; font-weight: 800; letter-spacing: .055em; white-space: nowrap; }
        .status-pure { color: #0c5631; background: #e7f5e9; border: 1px solid #bde0c3; }
        .status-adulterated { color: #9e2a1c; background: #ffebe8; border: 1px solid #f2c7c1; }
        .status-inconclusive { color: #825500; background: #fff5d8; border: 1px solid #efdaa0; }
        .confidence-label { display: flex; justify-content: space-between; color: var(--muted); font-size: .76rem; margin: .68rem 0 .28rem; }
        .verdict-copy { color: #405046; font-size: .79rem; line-height: 1.4; min-height: 2.3rem; margin-top: .55rem; }
        .result-shell .stImage img { border-radius: 10px; border: 1px solid #edf0eb; }
        .result-shell div[data-testid="stProgress"] > div > div { background-color: var(--saffron); }

        .empty-state { padding: 2rem 1.2rem; text-align: center; color: var(--muted); }
        .empty-state strong { display: block; color: var(--ink); font-size: 1.03rem; margin-bottom: .34rem; }
        .model-note { padding: .9rem 1rem; color: #5a675e; font-size: .8rem; line-height: 1.52; margin-top: 1rem; }
        .footer { color: #7b8880; border-top: 1px solid var(--line); margin-top: 2.2rem; padding-top: 1rem; font-size: .73rem; text-align: center; }
        div[data-testid="stDownloadButton"] button { width: 100%; border-radius: 9px; }
        .stTabs [data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid var(--line); }
        .stTabs [data-baseweb="tab"] { color: var(--muted); font-size: .86rem; font-weight: 650; padding: .62rem .82rem; }
        .stTabs [aria-selected="true"] { color: var(--leaf); }

        @media (max-width: 700px) {
            .block-container { padding-top: 1.2rem; }
            .app-header { padding-bottom: 1.15rem; }
            .header-tag { display: none; }
            .hero-panel { padding: 1.55rem; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Model and image processing
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "turmeric_binary_final.h5"
CLASS_NAMES = ["Adulterated", "Pure"]
CONFIDENCE_THRESHOLD = 65.0


@st.cache_resource(show_spinner="Loading screening model…")
def load_model(model_path: str):
    """Load the trained model once for the active Streamlit session."""
    return tf.keras.models.load_model(model_path)


def preprocess_image(pil_img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Crop foreground, enhance grayscale contrast, then prepare a MobileNetV2 tensor."""
    image_gray = np.array(pil_img.convert("L"))
    _, threshold = cv2.threshold(image_gray, 5, 255, cv2.THRESH_BINARY)
    coordinates = cv2.findNonZero(threshold)

    if coordinates is not None:
        x, y, width, height = cv2.boundingRect(coordinates)
        padding = 20
        left, top = max(0, x - padding), max(0, y - padding)
        right = min(image_gray.shape[1], x + width + padding)
        bottom = min(image_gray.shape[0], y + height + padding)
        cropped = image_gray[top:bottom, left:right]
        if cropped.shape[0] < 30 or cropped.shape[1] < 30:
            cropped = image_gray
    else:
        cropped = image_gray

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(cropped)
    rgb_image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(rgb_image, (224, 224))
    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(resized.astype(np.float32))
    return np.expand_dims(preprocessed, axis=0), enhanced


def classify(pil_img: Image.Image, model: tf.keras.Model) -> tuple[str, float, np.ndarray, np.ndarray]:
    """Return a thresholded verdict, confidence score, processed preview, and raw probabilities."""
    input_tensor, processed_preview = preprocess_image(pil_img)
    probabilities = np.asarray(model.predict(input_tensor, verbose=0)[0]).reshape(-1)

    if probabilities.size != len(CLASS_NAMES):
        raise ValueError(
            "The model output does not match the expected two-class configuration "
            "(Adulterated, Pure)."
        )

    predicted_index = int(np.argmax(probabilities))
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index] * 100)
    status = predicted_label if confidence >= CONFIDENCE_THRESHOLD else "Inconclusive"
    return status, confidence, processed_preview, probabilities


# -----------------------------------------------------------------------------
# Presentation helpers
# -----------------------------------------------------------------------------
def metric_card(value: int, label: str, accent: str = "") -> str:
    accent_class = f" {accent}" if accent else ""
    return (
        f'<div class="metric-card{accent_class}">'
        f'<div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{value}</div>'
        "</div>"
    )


def status_details(status: str) -> tuple[str, str, str]:
    mapping = {
        "Pure": (
            "status-pure",
            "PURE",
            "The sample is classified as pure at the configured confidence threshold.",
        ),
        "Adulterated": (
            "status-adulterated",
            "ADULTERATED",
            "The sample shows an adulteration signal and should be reviewed or retested.",
        ),
        "Inconclusive": (
            "status-inconclusive",
            "INCONCLUSIVE",
            "The prediction did not meet the confidence threshold; capture another image and retest.",
        ),
    }
    return mapping[status]


# -----------------------------------------------------------------------------
# Application interface
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <div class="brand-wrap">
            <div class="brand-mark">TS</div>
            <div>
                <div class="brand-name">TurmeriScan AI</div>
                <div class="brand-subtitle">Spectral-assisted quality screening</div>
            </div>
        </div>
        <div class="header-tag">MOBILENETV2 SCREENING MODEL</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero-panel">
        <div class="eyebrow">TURMERIC QUALITY SCREENING</div>
        <h1>Screen turmeric samples with a clear, confidence-led result.</h1>
        <p>Upload one or more compatible sample images to obtain a model-assisted purity classification and a concise review summary.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

analysis_tab, method_tab = st.tabs(["Screen samples", "Method and scope"])

with analysis_tab:
    st.markdown('<div class="section-kicker">STEP 1</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Add sample images</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-lede">Choose clear JPG or PNG images. Multiple samples can be screened in one run.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="how-card">
            <div class="how-title">Screening workflow</div>
            <div class="process-row">
                <span class="process-step">1. Upload sample</span><span class="process-arrow">→</span>
                <span class="process-step">2. Enhance image</span><span class="process-arrow">→</span>
                <span class="process-step">3. Model prediction</span><span class="process-arrow">→</span>
                <span class="process-step">4. Review result</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload turmeric sample images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Use the imaging setup for which the screening model was trained whenever possible.",
        label_visibility="collapsed",
    )

    if uploaded_files:
        if not MODEL_PATH.exists():
            st.error(
                "Screening model not found. Place `turmeric_binary_final.h5` in the same folder as this application, then restart it."
            )
        else:
            try:
                model = load_model(str(MODEL_PATH))
            except Exception as error:
                st.error(f"The screening model could not be loaded: {error}")
            else:
                results: list[dict[str, object]] = []
                invalid_files: list[str] = []

                with st.spinner(f"Screening {len(uploaded_files)} sample(s)…"):
                    for uploaded_file in uploaded_files:
                        try:
                            sample_image = Image.open(uploaded_file)
                            sample_image.load()
                            status, confidence, preview, probabilities = classify(sample_image, model)
                            results.append(
                                {
                                    "filename": uploaded_file.name,
                                    "image": sample_image.copy(),
                                    "processed": preview,
                                    "status": status,
                                    "confidence": confidence,
                                    "probabilities": probabilities,
                                }
                            )
                        except (UnidentifiedImageError, OSError, ValueError) as error:
                            invalid_files.append(f"{uploaded_file.name} ({error})")

                if invalid_files:
                    st.warning("Some files could not be screened: " + "; ".join(invalid_files))

                if results:
                    pure_count = sum(result["status"] == "Pure" for result in results)
                    adulterated_count = sum(result["status"] == "Adulterated" for result in results)
                    inconclusive_count = sum(result["status"] == "Inconclusive" for result in results)

                    st.markdown('<div style="height:1.55rem"></div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-kicker">STEP 2</div>', unsafe_allow_html=True)
                    st.markdown('<h2 class="section-title">Run summary</h2>', unsafe_allow_html=True)
                    st.markdown(
                        '<p class="section-lede">Counts below reflect the configured confidence threshold.</p>',
                        unsafe_allow_html=True,
                    )

                    metric_columns = st.columns(4, gap="small")
                    summary_metrics = [
                        (len(results), "Samples screened", ""),
                        (pure_count, "Classified pure", "pure"),
                        (adulterated_count, "Flagged adulterated", "alert"),
                        (inconclusive_count, "Needs review", "caution"),
                    ]
                    for column, (value, label, accent) in zip(metric_columns, summary_metrics):
                        with column:
                            st.markdown(metric_card(value, label, accent), unsafe_allow_html=True)

                    st.markdown('<div style="height:1.45rem"></div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-kicker">STEP 3</div>', unsafe_allow_html=True)
                    st.markdown('<h2 class="section-title">Sample results</h2>', unsafe_allow_html=True)
                    st.markdown(
                        '<p class="section-lede">Review individual classifications and inspect the enhanced image used for screening.</p>',
                        unsafe_allow_html=True,
                    )

                    columns_per_row = 3
                    for row_start in range(0, len(results), columns_per_row):
                        row_results = results[row_start:row_start + columns_per_row]
                        result_columns = st.columns(columns_per_row, gap="medium")
                        for sample_number, (column, result) in enumerate(
                            zip(result_columns, row_results), start=row_start + 1
                        ):
                            status_class, status_label, verdict_copy = status_details(str(result["status"]))
                            safe_filename = html.escape(str(result["filename"]))
                            confidence = float(result["confidence"])

                            with column:
                                with st.container(border=True):
                                    st.markdown(
                                        f"""
                                        <div class="result-top">
                                            <div>
                                                <div class="sample-number">Sample {sample_number:02d}</div>
                                                <div class="file-name">{safe_filename}</div>
                                            </div>
                                            <span class="status-pill {status_class}">{status_label}</span>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                                    st.image(result["image"], use_container_width=True)
                                    st.markdown(
                                        f"""
                                        <div class="confidence-label"><span>Model confidence</span><strong>{confidence:.1f}%</strong></div>
                                        <div class="verdict-copy">{verdict_copy}</div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                                    st.progress(max(0.0, min(confidence / 100, 1.0)))
                                    with st.expander("View enhanced image"):
                                        st.image(
                                            result["processed"],
                                            caption="Contrast-enhanced grayscale image used as model input.",
                                            use_container_width=True,
                                        )

                    table_rows = [
                        {
                            "Sample": sample_number,
                            "File": result["filename"],
                            "Verdict": result["status"],
                            "Confidence (%)": round(float(result["confidence"]), 1),
                        }
                        for sample_number, result in enumerate(results, start=1)
                    ]
                    results_table = pd.DataFrame(table_rows)
                    csv_data = results_table.to_csv(index=False).encode("utf-8")

                    with st.expander("Open result table and export"):
                        st.dataframe(results_table, use_container_width=True, hide_index=True)
                        st.download_button(
                            "Download screening summary (CSV)",
                            data=csv_data,
                            file_name="turmeriscan_screening_summary.csv",
                            mime="text/csv",
                        )

                    st.markdown(
                        """
                        <div class="model-note">
                            <strong>Interpret results carefully.</strong> This application is a screening aid, not a laboratory certification. It is intended for images captured under conditions comparable to the training data; lighting, camera characteristics, image quality, and sample preparation can affect predictions.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.error("None of the uploaded files could be processed. Please upload valid JPG or PNG image files.")
    else:
        st.markdown(
            """
            <div class="empty-state">
                <strong>Ready when your samples are.</strong>
                Upload one or more turmeric images above to begin a screening run.
            </div>
            """,
            unsafe_allow_html=True,
        )

with method_tab:
    st.markdown('<div class="section-kicker">ABOUT THE SCREEN</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">How the classification is produced</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        The application crops the detected image foreground, applies local contrast enhancement to its grayscale representation, and resizes the result for a MobileNetV2-based classification model. The highest model probability becomes the displayed confidence. A result below the configured **65% threshold** is labelled **Inconclusive** rather than being forced into a purity verdict.
        """
    )
    st.markdown(
        """
        <div class="model-note">
            <strong>Recommended use:</strong> retain the original files, use a consistent imaging procedure, and confirm any operational or regulatory decision with an appropriate laboratory method. The displayed result reflects the model's prediction for the uploaded image, not a conclusive chemical analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    "<div class='footer'>TurmeriScan AI · Model-assisted turmeric sample screening</div>",
    unsafe_allow_html=True,
)
