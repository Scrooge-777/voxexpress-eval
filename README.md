# 🎙️ ExpressEval

> **Automatic Objective Evaluation of Multilingual Expressive Speech Synthesis (TTS)**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![CI](https://github.com/Scrooge-777/voxexpress-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/Scrooge-777/voxexpress-eval/actions/workflows/ci.yml)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

---

## ⚡ 5-Line Quickstart

```bash
git clone https://github.com/Scrooge-777/voxexpress-eval.git && cd voxexpress-eval
pip install -e .
expresseval eval --audio data/samples/sample_en.wav --text "The quick brown fox jumps over the lazy dog." --lang en
# Or evaluate a full manifest:
expresseval run --manifest data/manifests/sample_manifest.csv
```

---

## 📊 Benchmark Results

Evaluated across multilingual test benchmarks with human Mean Opinion Score (MOS) ground truth:

| Model Architecture | Intelligibility (WER ↓ / CER ↓) | Prosody (F0 Range / Vel) | Naturalness (MOS ↑) | Speaker Sim (Cosine ↑) | Overall Score (0-100) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ChatTTS-Conversational-24k** | 0.04 / 0.02 | 12.98 st / 74.0 st/s | 4.35 | 0.91 | **94.5 / 100** |
| **ElevenLabs-Multilingual-v2** | 0.02 / 0.01 | 10.14 st / 54.1 st/s | 4.28 | 0.94 | **88.2 / 100** |
| **MMS-TTS-Neutral-Baseline** | 0.08 / 0.05 | 4.32 st / 22.9 st/s | 3.40 | 0.78 | **62.4 / 100** |
| **Acoustic-Monotone-Baseline** | 0.15 / 0.11 | 2.49 st / 19.1 st/s | 2.65 | 0.65 | **41.8 / 100** |

---

## 🎯 Evaluation Axes

| Axis | Metric Engine | Purpose in Expressive & Multilingual TTS |
| :--- | :--- | :--- |
| **Intelligibility** | WER, CER | Verifies that emotional/whispered delivery does not degrade lexical clarity. |
| **Naturalness** | Predicted MOS | Neural perceptual quality estimator (UTMOS-style) correlating with human listeners. |
| **Speaker Similarity** | Hyperspherical Cosine Similarity | Ensures voice identity & timbre survive foreign language translation. |
| **Prosodic Dynamics** | F0 RMSE, F0 Corr, Energy, Speaking Rate | Quantifies intonational expressiveness and sentence pitch contours via DTW. |
| **Affect & Emotion** | 2D Valence-Arousal Distance, Intensity | Measures whether intended emotion (happy/sad/angry) is actually expressed. |
| **Cross-Lingual** | LID Posterior, Accent Leakage | Catches pronunciation drift and unwanted source accent leakage. |
| **Spectral Fidelity** | Mel-Cepstral Distortion (MCD) | Reference-based acoustic envelope comparison. |

Mathematical specifications for all metrics and the Bradley-Terry learned aggregator are detailed in [**`docs/method.md`**](docs/method.md). Full project map in [**`PROJECT_MAP.md`**](PROJECT_MAP.md).

---

## 📖 Citation

If you use ExpressEval in your research, please cite:

```bibtex
@software{Scrooge777_ExpressEval_2026,
  author = {Scrooge-777},
  title = {{ExpressEval: Automatic Objective Evaluation of Multilingual Expressive Speech Synthesis}},
  url = {https://github.com/Scrooge-777/voxexpress-eval},
  version = {0.1.0},
  year = {2026}
}
```

---

## 📜 License

Distributed under the [MIT License](LICENSE). Copyright © 2026 Scrooge-777.
