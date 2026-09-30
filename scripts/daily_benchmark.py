#!/usr/bin/env python3
"""
Autonomous Daily Benchmark Runner for VoxExpress-Eval.
Runs daily evaluation checks, logs acoustic telemetry, and appends to BENCHMARK_LOG.md.
"""

from datetime import datetime, timezone
import json
import os
import sys

# Ensure local package import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from voxeval.core.synthetic import generate_synthetic_prosody_profile
from voxeval.metrics.prosody_basic import compute_basic_prosody


def run_benchmark():
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"[{timestamp}] Starting VoxExpress-Eval Autonomous Daily Benchmark...")

    test_profiles = [
        {"name": "ChatTTS_Expressive_Sim", "base_pitch": 195.0, "mod_depth": 42.0, "expr": 1.3},
        {"name": "ElevenLabs_Multilingual_Sim", "base_pitch": 165.0, "mod_depth": 30.0, "expr": 1.1},
        {"name": "Standard_Neural_Neutral_Sim", "base_pitch": 175.0, "mod_depth": 18.0, "expr": 0.6},
        {"name": "Flat_Monotone_Baseline_Sim", "base_pitch": 150.0, "mod_depth": 4.0, "expr": 0.1},
    ]

    results = []

    for profile in test_profiles:
        f0, energy = generate_synthetic_prosody_profile(
            duration_s=4.0,
            base_pitch_hz=profile["base_pitch"],
            pitch_mod_depth=profile["mod_depth"],
            expressiveness_factor=profile["expr"],
        )
        metrics = compute_basic_prosody(f0, energy)

        # Baseline Expressiveness Index (normalized 0 - 100)
        # Scaled by pitch range (semitones) and pitch velocity
        raw_score = (metrics.f0_range_semitones * 5.0) + (metrics.pitch_velocity_mean * 0.4)
        express_score = min(max(raw_score, 0.0), 100.0)

        results.append({
            "model": profile["name"],
            "f0_mean_hz": round(metrics.f0_mean_hz, 2),
            "pitch_range_semitones": round(metrics.f0_range_semitones, 2),
            "pitch_velocity_mean": round(metrics.pitch_velocity_mean, 2),
            "voicing_ratio": round(metrics.voicing_ratio, 3),
            "energy_dynamic_range_db": round(metrics.energy_dynamic_range_db, 2),
            "expressiveness_index": round(express_score, 2),
        })

    # Sort results by expressiveness index descending
    results.sort(key=lambda x: x["expressiveness_index"], reverse=True)

    # Save JSON telemetry
    os.makedirs("reports", exist_ok=True)
    telemetry_path = os.path.join("reports", "daily_telemetry.json")
    with open(telemetry_path, "w", encoding="utf-8") as f:
        json.dump({
            "last_updated": timestamp,
            "benchmark_version": "0.1.0",
            "evaluations": results,
        }, f, indent=2)

    # Update BENCHMARK_LOG.md
    log_path = "BENCHMARK_LOG.md"
    log_entry = f"\n### 📊 Daily Benchmark Execution: `{date_str}`\n"
    log_entry += f"*Executed at: `{timestamp}`*\n\n"
    log_entry += "| Model / Profile | F0 Range (Semitones) | Pitch Velocity | Voicing Ratio | Expressiveness Index |\n"
    log_entry += "| :--- | :---: | :---: | :---: | :---: |\n"

    for r in results:
        log_entry += f"| **{r['model']}** | {r['pitch_range_semitones']} st | {r['pitch_velocity_mean']} st/s | {int(r['voicing_ratio']*100)}% | **{r['expressiveness_index']} / 100** |\n"

    if os.path.exists(log_path):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_entry)
    else:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("# 📈 VoxExpress-Eval Autonomous Daily Benchmark Log\n" + log_entry)

    print(f"[{timestamp}] Benchmark completed successfully. Log updated at {log_path}.")


if __name__ == "__main__":
    run_benchmark()
