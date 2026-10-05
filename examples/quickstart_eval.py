#!/usr/bin/env python3
"""
ExpressEval - Quickstart Runnable Evaluation Demo
Demonstrates programmatic batch evaluation and Ridge aggregation using ExpressEval.
"""

import os
import sys
import numpy as np

# Ensure expresseval is importable from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from expresseval.pipeline import EvaluationPipeline
from expresseval.io.manifest import ManifestItem
from expresseval.aggregator.model import RidgeAggregator


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 68)
    print(" [*] ExpressEval: Multi-Axis Expressive TTS Evaluation Demo")
    print("=" * 68)

    # 1. Initialize Pipeline with configuration
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    pipeline = EvaluationPipeline(config_path)
    print(f"[*] Evaluation Pipeline initialized with config: {os.path.basename(config_path)}")

    # 2. Formulate ManifestItem pointing to existing sample audio
    sample_wav = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "samples", "sample_en.wav"))
    ref_wav = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "samples", "sample_ref.wav"))

    item = ManifestItem(
        id="demo_sample_01",
        audio_path=sample_wav,
        ref_audio_path=ref_wav,
        text="The quick brown fox jumps over the lazy dog.",
        language="en",
        system_id="ChatTTS-Conversational",
        emotion="neutral"
    )
    print(f"[*] Evaluating sample item: {item.id} ({item.audio_path})")

    # 3. Execute multi-axis evaluation
    record = pipeline.evaluate_item(item)

    print("\n" + "-" * 50)
    print(" [Results] Multi-Axis Metric Scores:")
    print("-" * 50)
    for k, v in record.items():
        if isinstance(v, float):
            print(f"  * {k:<30}: {v:.4f}")
        else:
            print(f"  * {k:<30}: {v}")

    # 4. Ridge Aggregator Demonstration
    aggregator = RidgeAggregator(alpha=0.5)
    # Simulate calibration training on multi-axis metrics vs human MOS ratings
    np.random.seed(42)
    X_train = np.random.uniform(0.0, 1.0, size=(100, 8)).astype(np.float32)
    y_train = (1.5 + 0.8 * X_train[:, 0] + 1.2 * X_train[:, 6] + 0.9 * X_train[:, 7]).astype(np.float32)
    aggregator.fit(X_train, y_train)

    # Feature vector normalized: [wer, f0_range, pitch_vel, mcd, emotion, crosslingual, mos, sim]
    features = np.array([[
        1.0 - (record.get("intelligibility_score", 100.0) / 100.0),
        min(record.get("prosody_score", 12.0) / 20.0, 1.0),
        0.55,
        min(record.get("spectral_score", 0.0) / 10.0, 1.0),
        record.get("emotion_score", 85.0) / 100.0,
        record.get("crosslingual_score", 60.0) / 100.0,
        record.get("naturalness_score", 66.0) / 100.0,
        record.get("speaker_sim_score", 96.0) / 100.0,
    ]], dtype=np.float32)

    predicted_composite = aggregator.predict(features)[0]
    print("-" * 50)
    print(f" [Composite] Calibrated Overall MOS Score: {predicted_composite:.2f} / 5.0")
    print("=" * 68)
    print("[+] Demo completed successfully.")


if __name__ == "__main__":
    main()
