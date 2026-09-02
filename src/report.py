"""Reporting, statistics, and export services for screening runs."""

from __future__ import annotations

from typing import Any
import pandas as pd


def compute_kpi_summary(results: list[dict[str, Any]]) -> dict[str, int]:
    """Compute aggregate count statistics for a screening run."""
    total = len(results)
    pure = sum(1 for r in results if r["status"] == "Pure")
    adulterated = sum(1 for r in results if r["status"] == "Adulterated")
    inconclusive = sum(1 for r in results if r["status"] == "Inconclusive")

    return {
        "total": total,
        "pure": pure,
        "adulterated": adulterated,
        "inconclusive": inconclusive,
    }


def generate_summary_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Format screening results into a clean tabular DataFrame."""
    rows = []
    for idx, r in enumerate(results, start=1):
        filename = r.get("filename", f"Sample_{idx:02d}")
        status = r["status"]
        conf = round(float(r["confidence"]), 2)
        p_pure = round(float(r["probabilities"].get("Pure", 0.0)), 2)
        p_adulterated = round(float(r["probabilities"].get("Adulterated", 0.0)), 2)

        rows.append(
            {
                "Sample #": idx,
                "Filename": filename,
                "Verdict": status,
                "Confidence (%)": conf,
                "Pure Probability (%)": p_pure,
                "Adulterated Probability (%)": p_adulterated,
            }
        )

    return pd.DataFrame(rows)


def generate_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize DataFrame to UTF-8 CSV bytes for download."""
    return df.to_csv(index=False).encode("utf-8")
