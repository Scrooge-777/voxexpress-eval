#!/usr/bin/env python3
"""
ONNX export utility for ExpressEval.
Prepares model weights and computation graphs for high-speed CPU/GPU inference.
"""

import argparse
import os
import json
import numpy as np


def export_ridge_onnx(weights_path: str, output_path: str):
    """Generates an ONNX-compatible specification for the learned RidgeAggregator."""
    if not os.path.exists(weights_path):
        print(f"Weights file not found: {weights_path}. Train the model first via make train.")
        return

    with open(weights_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    weights = np.array(meta.get("feature_weights", []), dtype=np.float32)
    bias = float(meta.get("model_bias", 0.0))

    spec = {
        "format": "expresseval-linear-graph-v1",
        "input_dim": len(weights),
        "output_dim": 1,
        "weights": weights.tolist(),
        "bias": bias,
        "activation": "clip(1.0, 5.0)",
        "target_runtime": "onnxruntime",
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)

    print(f"Exported inference specification to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export ExpressEval models for ONNX inference")
    parser.add_argument("--weights", "-w", default="checkpoints/aggregator_weights.json", help="Path to weights JSON")
    parser.add_argument("--output", "-o", default="checkpoints/aggregator_onnx_spec.json", help="Path to output export")
    args = parser.parse_args()

    export_ridge_onnx(args.weights, args.output)


if __name__ == "__main__":
    main()
