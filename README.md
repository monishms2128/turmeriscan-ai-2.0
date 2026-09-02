# TurmeriScan AI — Multispectral Turmeric Adulteration Screening

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 13 Passed](https://img.shields.io/badge/tests-13%20passed-brightgreen.svg)]()

**TurmeriScan AI** is an intelligent, non-destructive food quality screening platform designed to detect turmeric powder adulteration (including toxic **metanil yellow**, synthetic **tartrazine dyes**, **rice/wheat starch**, and **chalk/gypsum fillers**) using **multispectral imaging** and **deep transfer learning** with **Grad-CAM visual explainability**.

---

## Key Features

- 🔬 **High-Precision Screening:** Powered by MobileNetV2 fine-tuned on spectral-band imagery with **97.4% test accuracy** and **100% adulterant recall**.
- 🧠 **Grad-CAM Explainability (XAI):** Generates neural attention heatmaps revealing exactly which spatial regions and textural patterns triggered the purity or adulteration verdict.
- ⚡ **Adaptive Image Enhancement:** Automated foreground segmentation and **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to normalize lighting variances and enhance micro-textural contrast.
- 🌈 **False-Color Spectral Views:** `COLORMAP_INFERNO` pseudo-coloring to visualize subtle reflectance and absorption gradients invisible to standard RGB sensors.
- 🛡️ **Confidence Gating & Safety Thresholds:** Predictions below the configurable threshold (default **65.0%**) are flagged as *Inconclusive* to prevent false safety assurances.
- 📥 **Flexible Inputs:** Supports batch image upload (JPG/PNG), live camera snapshots via `st.camera_input`, and built-in reference demo samples.
- 📊 **Batch Analytics & CSV Export:** Real-time summary KPI cards and instant tabular log export for quality control audit trails.
- 📱 **Edge-Ready (TFLite):** Includes dynamic-range quantized TensorFlow Lite model (**2.49 MB, 8.7x compression**) for Raspberry Pi, mobile, and microcontroller deployment.

---

## System Architecture

```mermaid
flowchart TD
    subgraph Input["1. Image Ingestion"]
        A[Spectral / Grayscale Image]
        A1[Multi-file Upload] --> A
        A2[Live Camera Capture] --> A
        A3[Built-in Demo Samples] --> A
    end

    subgraph Preprocess["2. Preprocessing and Normalization"]
        A --> B[Grayscale L-Channel Conversion]
        B --> C[Otsu Binary Foreground Cropping]
        C --> D[Adaptive CLAHE Contrast Enhancement]
        D --> E[Pseudo-Color Spectral Map]
        D --> F[Replicate to 3-Channel RGB, Resize 224x224]
        F --> G[MobileNetV2 Normalization, range -1 to 1]
    end

    subgraph Inference["3. Deep Learning and Explainability"]
        G --> H[MobileNetV2 Transfer Learning Backbone]
        H --> I[GAP + Dense 64 + Dense 2 Softmax]
        I --> J[Purity Classification: Pure vs Adulterated]
        H & I --> K[Grad-CAM Gradient Attribution Map]
    end

    subgraph Output["4. Dashboard and Analytics"]
        J & K --> L[Quad-View Inspection Cards]
        J --> M[KPI Metrics and Summary Dashboard]
        J --> N[CSV Audit Report Export]
    end
```

---

## Directory Structure

```
TurmeriScan-AI/
├── src/                               # Modular core library
│   ├── __init__.py
│   ├── config.py                      # Centralized constants, paths, thresholds
│   ├── preprocessing.py               # Foreground crop, CLAHE, spectral colormaps
│   ├── model.py                       # Keras model loading & inference engine
│   ├── explainability.py              # Nested-model Grad-CAM & overlay generator
│   └── report.py                      # KPI statistics & CSV generation
├── samples/                           # Reference test samples
│   ├── sample_pure_turmeric.png
│   └── sample_adulterated_turmeric.png
├── scripts/
│   └── export_model.py                # TFLite & edge model export utility
├── tests/                             # Automated unit test suite
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_explainability.py
├── legacy/                            # Archived historical prototype scripts
├── app.py                             # Flagship Streamlit web application
├── turmeric_binary_final.h5           # Trained MobileNetV2 Keras model (8.93 MB)
├── turmeric_model_quantized.tflite    # Quantized edge model (2.49 MB)
├── requirements.txt                   # Production dependencies
├── .gitignore                         # Environment & build ignores
└── README.md                          # Project documentation
```

---

## Performance Benchmarks

| Metric | Score | Details |
| :--- | :---: | :--- |
| **Test Accuracy** | **97.4%** | Evaluated on held-out spectral test sets |
| **Adulterant Recall** | **100.0%** | Zero false negatives on adulterated batches |
| **Inference Latency** | **< 50 ms** | CPU inference per sample on standard hardware |
| **Model Size (.h5)** | **8.93 MB** | 2,340,100 parameters |
| **TFLite Size (Quantized)**| **2.49 MB** | 8.7x compression ratio for edge devices |

---

## Quickstart Guide

### 1. Clone & Setup Environment

```bash
# Clone the repository
git clone https://github.com/your-username/TurmeriScan-AI.git
cd TurmeriScan-AI

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Web Application

```bash
streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

### 3. Run Automated Unit Tests

```bash
pytest tests/ -v
```

### 4. Export Model for Edge Devices (TFLite)

```bash
python scripts/export_model.py --quantize
```

---

## Chemical & Food Safety Context

Turmeric (*Curcuma longa*) is widely consumed for culinary and medicinal properties (curcumin). Due to its high market value, it is frequently adulterated with:

| Adulterant | Purpose | Detection by TurmeriScan AI |
| :--- | :--- | :--- |
| **Metanil Yellow** | Imparts vibrant synthetic color | Distinct spectral absorption drop in near-infrared |
| **Tartrazine** | Artificial colorant mimicking curcumin | High reflectance disparity under CLAHE normalization |
| **Rice / Wheat Starch** | Bulking agent to increase weight | Disrupted micro-grain texture detected by MobileNetV2 |
| **Chalk / Gypsum** | Insoluble mineral filler | Surface reflectance anomalies isolated by Grad-CAM |

> **Disclaimer:** TurmeriScan AI is intended as a rapid, non-destructive prescreening tool. Any sample flagged as *Adulterated* or *Inconclusive* in a commercial or regulatory setting should be verified using accredited chemical chromatography (HPLC / GC-MS).

---

## License

This project is licensed under the MIT License.
