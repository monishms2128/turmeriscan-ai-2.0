"""PDF Quality Inspection Certificate Generator for TurmeriScan AI."""

from __future__ import annotations

import io
import time
from typing import Any
import cv2
import numpy as np
from PIL import Image

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _pil_or_array_to_bytes(img_obj: Image.Image | np.ndarray) -> io.BytesIO:
    """Convert a PIL Image or numpy array to a PNG BytesIO stream for ReportLab."""
    buf = io.BytesIO()
    if isinstance(img_obj, np.ndarray):
        if img_obj.ndim == 2:
            pil_img = Image.fromarray(img_obj)
        elif img_obj.shape[2] == 3:
            pil_img = Image.fromarray(img_obj.astype(np.uint8))
        else:
            pil_img = Image.fromarray(img_obj)
    else:
        pil_img = img_obj

    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def generate_pdf_certificate(
    sample_name: str,
    status: str,
    confidence: float,
    probabilities: dict[str, float],
    original_img: Image.Image | np.ndarray,
    gradcam_img: np.ndarray,
    adulterant_profile: dict[str, Any] | None = None,
    quality_metrics: dict[str, Any] | None = None,
) -> bytes:
    """Generate a high-resolution, print-ready PDF Inspection Certificate."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CertTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=HexColor("#133a27"),
        alignment=1,  # Center
    )
    subtitle_style = ParagraphStyle(
        "CertSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=HexColor("#5a6b61"),
        alignment=1,
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading3"],
        fontSize=12,
        leading=16,
        textColor=HexColor("#133a27"),
    )
    body_style = ParagraphStyle(
        "CertBody",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=HexColor("#17211b"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=7.5,
        leading=10.5,
        textColor=HexColor("#7a8a80"),
        alignment=1,
    )

    story = []

    # 1. Header & Title
    story.append(Paragraph("TURMERISCAN AI — QUALITY INSPECTION CERTIFICATE", title_style))
    story.append(Paragraph("Non-Destructive Multispectral Screening & Neural Explainability Audit", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=HexColor("#eea51b"), spaceAfter=12))

    # 2. Metadata Table
    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S UTC")
    cert_id = f"TS-{int(time.time())}-{np.random.randint(1000, 9999)}"

    meta_data = [
        [
            Paragraph(f"<b>Certificate ID:</b> {cert_id}", body_style),
            Paragraph(f"<b>Sample Name:</b> {sample_name}", body_style),
        ],
        [
            Paragraph(f"<b>Inspection Date:</b> {timestamp_str}", body_style),
            Paragraph("<b>Model Architecture:</b> MobileNetV2 + Grad-CAM XAI", body_style),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[270, 270])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), HexColor("#f8faf8")),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("BOX", (0, 0), (-1, -1), 0.5, HexColor("#d8e2d6")),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # 3. Verdict Banner
    if status == "Pure":
        verdict_text = "✅ PURE TURMERIC (CURCUMIN SIGNATURE VERIFIED)"
        banner_bg = HexColor("#e5f5e7")
        banner_border = HexColor("#136338")
        banner_text_color = HexColor("#0b542e")
    elif status == "Adulterated":
        verdict_text = "⚠️ ADULTERATED SAMPLE DETECTED"
        banner_bg = HexColor("#ffe9e6")
        banner_border = HexColor("#b83725")
        banner_text_color = HexColor("#9c2719")
    else:
        verdict_text = "❓ INCONCLUSIVE — RETEST RECOMMENDED"
        banner_bg = HexColor("#fff3d1")
        banner_border = HexColor("#986500")
        banner_text_color = HexColor("#7f5200")

    verdict_style = ParagraphStyle(
        "VerdictText",
        parent=styles["Normal"],
        fontSize=13,
        leading=17,
        fontName="Helvetica-Bold",
        textColor=banner_text_color,
        alignment=1,
    )

    verdict_table = Table([[Paragraph(verdict_text, verdict_style)]], colWidths=[540])
    verdict_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), banner_bg),
                ("BOX", (0, 0), (-1, -1), 1.5, banner_border),
                ("PADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(verdict_table)
    story.append(Spacer(1, 12))

    # 4. Visual Evidence Section (Images side-by-side)
    story.append(Paragraph("Visual Evidence & Neural Attention Attribution", section_heading))
    story.append(Spacer(1, 6))

    orig_buf = _pil_or_array_to_bytes(original_img)
    cam_buf = _pil_or_array_to_bytes(gradcam_img)

    img_orig = RLImage(orig_buf, width=2.6 * inch, height=2.3 * inch)
    img_cam = RLImage(cam_buf, width=2.6 * inch, height=2.3 * inch)

    img_table = Table(
        [
            [img_orig, img_cam],
            [
                Paragraph("<b>Figure 1:</b> Captured Sample", body_style),
                Paragraph("<b>Figure 2:</b> Grad-CAM Attention Heatmap", body_style),
            ],
        ],
        colWidths=[270, 270],
    )
    img_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(img_table)
    story.append(Spacer(1, 12))

    # 5. Quantitative Analysis Table
    story.append(Paragraph("Quantitative Purity & Optical Telemetry", section_heading))
    story.append(Spacer(1, 6))

    pure_prob = probabilities.get("Pure", 0.0)
    adult_prob = probabilities.get("Adulterated", 0.0)
    quality_status = quality_metrics.get("status", "N/A") if quality_metrics else "N/A"

    metrics_rows = [
        [Paragraph("<b>Parameter</b>", body_style), Paragraph("<b>Measured Value</b>", body_style), Paragraph("<b>Status / Reference</b>", body_style)],
        [Paragraph("Screening Verdict", body_style), Paragraph(status, body_style), Paragraph("Threshold Gated", body_style)],
        [Paragraph("Model Confidence", body_style), Paragraph(f"{confidence:.2f}%", body_style), Paragraph("Safety Threshold: 65.0%", body_style)],
        [Paragraph("Curcumin Probability", body_style), Paragraph(f"{pure_prob:.2f}%", body_style), Paragraph("Natural Absorption Band", body_style)],
        [Paragraph("Adulterant Probability", body_style), Paragraph(f"{adult_prob:.2f}%", body_style), Paragraph("Disrupted Spectrum Band", body_style)],
        [Paragraph("Optical Capture Quality", body_style), Paragraph(quality_status, body_style), Paragraph("Illumination & Focus Check", body_style)],
    ]

    if adulterant_profile and status == "Adulterated":
        suspect = adulterant_profile.get("primary_suspect", "Unknown")
        metrics_rows.append(
            [Paragraph("Primary Adulterant Suspect", body_style), Paragraph(suspect, body_style), Paragraph("Texture & Gradient Profile", body_style)]
        )

    metrics_table = Table(metrics_rows, colWidths=[180, 160, 200])
    metrics_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#e8efe9")),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#d8e2d6")),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(metrics_table)
    story.append(Spacer(1, 14))

    # 6. Sign-off & Regulatory Disclaimer
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#d8e2d6"), spaceAfter=8))
    story.append(
        Paragraph(
            "<b>Disclaimer:</b> TurmeriScan AI is a rapid prescreening instrument utilizing deep transfer learning on spectral imagery. "
            "This certificate constitutes an automated screening evaluation and does not replace statutory laboratory assays (HPLC / GC-MS). "
            "Issued electronically by TurmeriScan AI Quality Control Engine.",
            disclaimer_style,
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
