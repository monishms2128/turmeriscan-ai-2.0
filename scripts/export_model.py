"""Export trained Keras model to TensorFlow Lite (TFLite) for edge and mobile deployment."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_PATH


def export_to_tflite(
    keras_model_path: Path = MODEL_PATH,
    output_dir: Path = PROJECT_ROOT,
    quantize: bool = False,
) -> Path:
    """Convert Keras .h5 model to .tflite format."""
    print(f"[*] Loading Keras model from: {keras_model_path}")
    model = tf.keras.models.load_model(str(keras_model_path))

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        print("[*] Applying Dynamic Range Quantization (INT8 weights)...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        output_filename = "turmeric_model_quantized.tflite"
    else:
        output_filename = "turmeric_model.tflite"

    tflite_model = converter.convert()
    output_path = output_dir / output_filename

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    orig_size_mb = keras_model_path.stat().st_size / (1024 * 1024)
    tflite_size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"[+] Successfully exported TFLite model to: {output_path}")
    print(f"    Original Size (.h5):     {orig_size_mb:.2f} MB")
    print(f"    Exported Size (.tflite): {tflite_size_mb:.2f} MB")
    print(f"    Compression Ratio:       {(orig_size_mb / tflite_size_mb):.1f}x")

    # Verify inference with TFLite Interpreter
    print("[*] Verifying TFLite inference with interpreter...")
    interpreter = tf.lite.Interpreter(model_path=str(output_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    dummy_input = np.random.uniform(-1, 1, input_details[0]["shape"]).astype(np.float32)

    start_time = time.perf_counter()
    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    latency_ms = (time.perf_counter() - start_time) * 1000

    print(f"[+] TFLite Verification successful! Output shape: {output_data.shape}, Latency: {latency_ms:.2f} ms")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export TurmeriScan model to TFLite format.")
    parser.add_argument("--quantize", action="store_true", help="Enable dynamic range quantization.")
    args = parser.parse_args()

    export_to_tflite(quantize=args.quantize)
