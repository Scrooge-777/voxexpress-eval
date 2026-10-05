# 🚀 ExpressEval Quickstart & Practical Usage Guide

Welcome to **ExpressEval** (`voxexpress-eval`), an objective evaluation benchmark suite for multilingual and expressive text-to-speech (TTS) systems.

This guide provides everything you need to start evaluating generated speech waveforms in under 5 minutes, covering both the **Command-Line Interface (CLI)** and the **Python API**.

---

## 📋 Table of Contents
1. [Installation](#1-installation)
2. [Quick Verification](#2-quick-verification)
3. [CLI Usage & Manifest Batch Evaluation](#3-cli-usage--manifest-batch-evaluation)
4. [Python API Usage](#4-python-api-usage)
5. [Understanding the 7 Evaluation Dimensions](#5-understanding-the-7-evaluation-dimensions)
6. [Training & Using the Ridge Aggregator](#6-training--using-the-ridge-aggregator)
7. [Inspecting Telemetry & Benchmark Leaderboards](#7-inspecting-telemetry--benchmark-leaderboards)

## 🧭 Which Workflow is Best for You?

| Use Case | Best Approach | What to Run | Why It's Best |
| :--- | :--- | :--- | :--- |
| **Instant Verification** | 🌟 **Interactive Demo** | `python examples/quickstart_eval.py` | Runs immediately with bundled sample audio; outputs all 7 metrics & predicted MOS in 2s. |
| **Visual Audio Testing** | 🎨 **Gradio Web Interface** | `python app/gradio_app.py` | Drag-and-drop audio files in your browser and view ASCII / graphical radar breakdowns. |
| **Dataset & Pipeline Scoring** | 🚀 **CLI Batch Pipeline** | `expresseval run --manifest <path>` | Evaluates hundreds of generated audio files and exports CSV / Parquet score reports. |
| **Model Development & Training**| 🐍 **Python Library API** | `from expresseval.pipeline import ...` | Plugs directly into PyTorch training loops or automated validation callbacks. |

---

## 1. Installation

### From Source (Editable Mode)
```bash
# Clone the repository
git clone https://github.com/Scrooge-777/voxexpress-eval.git
cd voxexpress-eval

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies and the expresseval package
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## 2. Quick Verification

Confirm your installation and verify all acoustic evaluation modules:

```bash
# Run the complete test suite (15 unit tests)
pytest tests/ -v
```

All 15 tests across speech intelligibility, prosody, spectral distance, emotion, cross-lingual consistency, naturalness, and speaker verification will execute and pass in ~2 seconds.

---

## 3. CLI Usage & Manifest Batch Evaluation

ExpressEval operates on **evaluation manifests** (`.csv` or `.jsonl`), pairing synthesized audio files with ground-truth references and transcript text.

### Manifest Format (`manifest.csv`)
```csv
id,audio_path,reference_audio_path,text,language
sample_01,data/samples/sample_en.wav,data/samples/sample_ref.wav,"The quick brown fox jumps over the lazy dog.",en
```

### Running Batch Evaluation via CLI
```bash
expresseval evaluate \
  --manifest data/manifests/sample_manifest.csv \
  --config configs/default.yaml \
  --output reports/eval_results.csv
```

### CLI Command Options
| Flag | Description | Default |
| :--- | :--- | :--- |
| `--manifest` | Path to CSV or JSONL evaluation manifest. | *Required* |
| `--config` | Path to YAML metric weights configuration. | `configs/default.yaml` |
| `--output` | Destination path for evaluated score metrics. | `reports/eval_results.csv` |
| `--device` | Execution device (`cpu` or `cuda`). | `cpu` |

---

## 4. Python API Usage

ExpressEval can be embedded directly into PyTorch pipelines, evaluation loops, or Jupyter notebooks.

```python
from expresseval.pipeline import EvaluationPipeline
from expresseval.io.manifest import ManifestItem

# 1. Initialize the multi-axis evaluation pipeline
pipeline = EvaluationPipeline("configs/default.yaml")

# 2. Formulate input manifest item
item = ManifestItem(
    id="sample_01",
    audio_path="data/samples/sample_en.wav",
    ref_audio_path="data/samples/sample_ref.wav",
    text="The quick brown fox jumps over the lazy dog.",
    language="en",
    system_id="My-TTS-System"
)

# 3. Compute metrics across all 7 dimensions
record = pipeline.evaluate_item(item)

# 4. Inspect scores
print("Intelligibility Score:", record.get("intelligibility_score"))
print("Prosody Score:        ", record.get("prosody_score"))
print("Speaker Sim Score:    ", record.get("speaker_sim_score"))
print("Naturalness Score:    ", record.get("naturalness_score"))
print("Overall Composite:    ", record.get("overall_score"))
```

---

## 5. Understanding the 7 Evaluation Dimensions

| Dimension | Key Objective Metrics | What It Measures |
| :--- | :--- | :--- |
| **Intelligibility** | Word Error Rate (`WER`), Character Error Rate (`CER`) | Phonetic accuracy and word transcription correctness. |
| **Prosody** | F0 Range (`semitones`), Pitch Velocity (`st/s`), Energy Dynamics | Pitch variation, dynamic inflection, and cadence without monotone flatness. |
| **Spectral Distance** | Mel-Cepstral Distortion (`MCD`), Log-Spectral Distance | Timbral fidelity relative to human acoustic reference. |
| **Emotion Alignment** | Valence / Arousal / Dominance distance | Expressive emotional color and contextual tone. |
| **Cross-Lingual** | Phoneme transition stability across language pairs | Accent preservation and bilingual speech consistency. |
| **Naturalness** | Non-intrusive MOS prediction (`1.0 - 5.0`) | Perceptual audio quality and absence of robotic artifacts. |
| **Speaker Similarity** | Speaker embedding cosine similarity (`0.0 - 1.0`) | Voice preservation and speaker verification accuracy. |

---

## 6. Training & Using the Ridge Aggregator

ExpressEval includes a **Ridge Regression Composite Aggregator** that maps the 7 metric dimensions into a calibrated human MOS score.

### Training the Aggregator:
```bash
python -m expresseval.aggregator.train \
  --config configs/train_aggregator.yaml \
  --output models/aggregator.pt
```

### Exporting to ONNX for High-Throughput Inference:
```bash
python scripts/export_onnx.py \
  --model models/aggregator.pt \
  --output models/aggregator.onnx
```

---

## 7. Inspecting Telemetry & Benchmark Leaderboards

The repository continuously evaluates benchmark baselines (`ChatTTS`, `ElevenLabs`, `MMS-TTS`, `Acoustic-Monotone`).

To view the latest live evaluation results:
- Open [`BENCHMARK_LOG.md`](../BENCHMARK_LOG.md) to inspect historical benchmark runs.
- Inspect [`reports/telemetry.json`](../reports/telemetry.json) for raw machine-readable JSON metrics.
- Run a benchmark locally at any time:
  ```bash
  python scripts/run_benchmark.py
  ```
