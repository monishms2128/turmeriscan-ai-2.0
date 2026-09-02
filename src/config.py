"""Configuration settings and constants for TurmeriScan AI."""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "turmeric_binary_final.h5"
SAMPLES_DIR = PROJECT_ROOT / "samples"

# Classification Constants
CLASS_NAMES = ["Adulterated", "Pure"]  # Index 0: Adulterated, Index 1: Pure
DEFAULT_CONFIDENCE_THRESHOLD = 65.0  # Percentage minimum to declare a confident verdict
INPUT_SHAPE = (224, 224, 3)

# Image Preprocessing Hyperparameters
CLAHE_CLIP_LIMIT = 4.0
CLAHE_GRID_SIZE = (8, 8)
CROP_PADDING_PX = 20
BINARY_BG_THRESHOLD = 5
MIN_CROP_DIMENSION_PX = 30

# UI Theme Tokens
COLORS = {
    "ink": "#17211b",
    "muted": "#627068",
    "line": "#dce5da",
    "paper": "#fbfdf9",
    "saffron": "#eea51b",
    "saffron_deep": "#b96c08",
    "leaf": "#20613d",
    "pure": "#17603a",
    "alert": "#b83725",
    "caution": "#986500",
}
