# Model & Benchmark Card: ExpressEval

## Overview
- **Name:** ExpressEval
- **Task:** Automatic Objective Evaluation of Multilingual Expressive Speech Synthesis (TTS)
- **Inputs:** Synthesized speech waveform (`.wav`, `.flac`), target text transcript, ISO-639-1 language code, optional reference audio and emotion tags.
- **Outputs:** Multi-axis acoustic scores (`prosody`, `intelligibility`, `naturalness`, `speaker_sim`, `spectral`, `emotion`, `crosslingual`) and learned composite MOS prediction.

## Supported Languages
English (`en`), Spanish (`es`), Mandarin Chinese (`zh`), Hindi (`hi`), French (`fr`), German (`de`), Japanese (`ja`).

## Ethical Considerations
ExpressEval is intended for scientific evaluation, model benchmarking, and quality assurance in speech synthesis research. It does not generate audio or clone voices directly.
