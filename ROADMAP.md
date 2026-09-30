# 🗺️ 60-Day Phased Research & Engineering Roadmap

> **Project:** VoxExpress-Eval  
> **Repository:** [Scrooge-777/voxexpress-eval](https://github.com/Scrooge-777/voxexpress-eval)  
> **Evolution Cadence:** Daily incremental module & theoretical updates

---

## 📅 Phase 1: Mathematical Foundations & Classical Acoustic Signal Processing (Days 1–15)

- [x] **Day 1 (Foundation):** Mathematical Foundations Whitepaper (`VOX-THEORY-001`), core architecture, and autonomous cloud evolution workflow.
- [ ] **Day 2 (Audio I/O & Normalization):** High-precision 24-bit audio pipeline, EBU R128 loudness normalization, and zero-crossing rate analysis.
- [ ] **Day 3 (Voice Activity Detection):** Adaptive energy and spectral entropy Voice Activity Detection (VAD) with hysteresis gating.
- [ ] **Day 4 (STFT & Mel-Filterbanks):** Multi-resolution Short-Time Fourier Transform (STFT) and auditory Mel-scale filterbank tensor generators.
- [ ] **Day 5 (Fundamental Frequency Tracking):** Robust continuous $F_0$ extraction using the probabilistic YIN (pYIN) algorithm with unvoiced interpolation.
- [ ] **Day 6 (Prosodic Dynamic Range):** Semitone-scale pitch range, dynamic pitch velocity ($\Delta F_0$), and acceleration ($\Delta\Delta F_0$) metrics.
- [ ] **Day 7 (Continuous Wavelet Transform):** Implementation of 10-scale dyadic Mexican Hat CWT for multi-scale pitch decomposition.
- [ ] **Day 8 (Energy Dynamics):** Instantaneous Root Mean Square (RMS) energy contour, crest factor, and dynamic loudness range (LRA).
- [ ] **Day 9 (Speech Rhythm & Timing):** Normalized Pairwise Variability Index (nPVI) for vowel durations and raw PVI (rPVI) for consonantal intervals.
- [ ] **Day 10 (Pause Architecture):** Hesitation analysis, micro-pause detection (<200ms), and macro-syntactic boundary pause distributions.
- [ ] **Day 11 (Vocal Perturbation Metrics):** Local jitter (frequency perturbation) and shimmer (amplitude perturbation) estimators for vocal stability.
- [ ] **Day 12 (Spectral Tilt & Formants):** Formant tracking ($F_1, F_2, F_3$) and spectral tilt ($\alpha$-ratio) quantifying vocal effort and breathiness.
- [ ] **Day 13 (Harmonic-to-Noise Ratio):** Cepstral Peak Prominence (CPP) and Harmonic-to-Noise Ratio (HNR) extraction.
- [ ] **Day 14 (Classical Verification Suite):** Comprehensive unit tests, numerical stability benchmarks, and mock audio synthetic generators.
- [ ] **Day 15 (Phase 1 Synthesis):** Phase 1 consolidation report, initial CLI evaluation runner (`voxeval evaluate --phase 1`).

---

## 📅 Phase 2: Neural Cross-Lingual Alignment & Speaker Manifold Verification (Days 16–30)

- [ ] **Day 16 (Multilingual Speech Representation Theory):** Theoretical paper on Self-Supervised Speech Representations (WavLM, XLS-R) across language families.
- [ ] **Day 17 (ECAPA-TDNN Architecture):** Neural speaker embedding extraction with Squeeze-and-Excitation channel attention.
- [ ] **Day 18 (Cross-Lingual Manifold Projection):** Orthogonal language subspace de-biasing ($(\mathbf{I} - \mathbf{P}_{\mathcal{V}_{\text{lang}}})\mathbf{e}$) for pure timbre comparison.
- [ ] **Day 19 (Cosine & Angular Distance):** Geodesic distance on unit hypersphere $\mathbb{S}^{d-1}$ for cross-lingual voice cloning verification.
- [ ] **Day 20 (Multilingual Phoneme Sets):** International Phonetic Alphabet (IPA) mapping and phoneme distance tables across target languages.
- [ ] **Day 21 (CTC Forced Alignment):** Connectionist Temporal Classification (CTC) character-level and phoneme-level segmentation engine.
- [ ] **Day 22 (Phoneme Error Rate):** Cross-lingual Levenshtein distance on phoneme sequences with phonetic confusion matrices.
- [ ] **Day 23 (Intelligibility Scoring):** Neural ASR-based Word Error Rate (WER) and Character Error Rate (CER) integration (Whisper/MMS).
- [ ] **Day 24 (Pronunciation Quality):** Goodness of Pronunciation (GOP) scoring derived from neural posterior probabilities.
- [ ] **Day 25 (Non-Intrusive Perceptual Quality):** Multi-task neural Mean Opinion Score (MOS) predictor architecture.
- [ ] **Day 26 (Noise & Artifact Robustness):** Spectral glitch detection, phase discontinuity estimation, and vocoder buzz quantification.
- [ ] **Day 27 (Cross-Lingual Batch Pipeline):** Parallelized batch processing for paired reference and multi-speaker synthetic datasets.
- [ ] **Day 28 (Language-Specific Normalization):** Calibrated scoring curves adjusting for tonal languages (Mandarin) vs stress-timed languages (English/German).
- [ ] **Day 29 (Phase 2 Test Suite):** Integration tests for neural embedding extractors and phonetic alignment pipelines.
- [ ] **Day 30 (Phase 2 Synthesis):** Phase 2 milestone release and cross-lingual benchmark validation report.

---

## 📅 Phase 3: Emotional Expressiveness & Semantic-Prosodic Coupling (Days 31–45)

- [ ] **Day 31 (Affective Computing Foundations):** Theoretical paper on dimensional emotion theory (Valence-Arousal-Dominance) in acoustic phonetics.
- [ ] **Day 32 (Speech Emotion Recognition):** Multi-modal Speech Emotion Recognition (SER) embedding extractor.
- [ ] **Day 33 (Arousal & Pitch Range Correlation):** Mathematical formulation linking acoustic arousal to pitch dynamics and energy slope.
- [ ] **Day 34 (Valence Manifold Mapping):** Mahalanobis distance in 2D affective circumplex space against target emotional centroids.
- [ ] **Day 35 (Wasserstein-1 Optimal Transport):** Earth Mover's Distance implementation on 2D joint pitch-energy empirical distributions.
- [ ] **Day 36 (Prosodic Entropy & Surprise):** Shannon entropy and conditional surprise metrics for expressive intonation arcs.
- [ ] **Day 37 (Semantic-Prosodic Alignment):** Text sentiment extraction vs acoustic pitch emphasis cross-modal mutual information.
- [ ] **Day 38 (Punctuation & Syntax Coupling):** Evaluating pitch inflection behavior at syntactic markers (question marks, commas, exclamations).
- [ ] **Day 39 (Conversational Spontaneity):** Laughter, breath intake, and vocalized filler detection and appropriateness scoring.
- [ ] **Day 40 (Emphasis & Focal Accentuation):** Automated detection of focal contrastive stress in synthesized sentences.
- [ ] **Day 41 (Affective Degradation Metric):** Measuring emotional attenuation (synthetic flattening) across long paragraph synthesis.
- [ ] **Day 42 (Subjective MOS Correlation):** Statistical calibration of VoxExpress metric weights against public Blizzard Challenge MOS datasets.
- [ ] **Day 43 (Multi-Scale Emotional Radar):** 8-axis acoustic radar visualization of expressive characteristics.
- [ ] **Day 44 (Phase 3 Test Suite):** Rigorous verification of optimal transport solvers and affective classification pipelines.
- [ ] **Day 45 (Phase 3 Synthesis):** Complete expressive evaluation module release.

---

## 📅 Phase 4: Autonomous Cloud Benchmarking, Open Leaderboard & Public Paper (Days 46–60+)

- [ ] **Day 46 (Cloud Autonomous Engine):** Hardened GitHub Actions distributed matrix runners for cloud evaluations.
- [ ] **Day 47 (Model Evaluator - ChatTTS):** Automated ingestion and benchmark evaluation of ChatTTS conversational audio.
- [ ] **Day 48 (Model Evaluator - CosyVoice):** Automated evaluation of CosyVoice multilingual expressive synthesis.
- [ ] **Day 49 (Model Evaluator - F5-TTS):** Non-autoregressive flow-matching TTS evaluation and prosodic fidelity benchmark.
- [ ] **Day 50 (Model Evaluator - ElevenLabs):** Benchmark comparison with state-of-the-art commercial expressive voices.
- [ ] **Day 51 (Public Open Leaderboard):** Auto-updating Markdown leaderboard table embedded in README with daily dynamic rankings.
- [ ] **Day 52 (Interactive HTML Diagnostic Report):** Self-contained HTML report with embedded WebAudio player and pitch/spectrogram visualizer.
- [ ] **Day 53 (REST API & FastAPI Service):** Lightweight async API for on-demand evaluation of uploaded audio files.
- [ ] **Day 54 (Docker & Containerization):** Multi-stage Dockerfile with GPU acceleration support for cloud deployments.
- [ ] **Day 55 (PyPI Package Preparation):** Setup `pyproject.toml` and packaging metadata for `pip install voxexpress-eval`.
- [ ] **Day 56 (Extensive Documentation & Tutorials):** Jupyter notebook walkthroughs and API documentation on GitHub Pages.
- [ ] **Day 57 (Scientific Paper Draft - Abstract & Intro):** LaTeX preprint draft: Problem statement, motivation, and mathematical framework.
- [ ] **Day 58 (Scientific Paper Draft - Experiments):** Correlation results with human listener MOS, benchmark tables, and ablation studies.
- [ ] **Day 59 (Scientific Paper Draft - Conclusion & Release):** Finalizing research paper for submission to arXiv / Interspeech.
- [ ] **Day 60 (v1.0.0 Global Release):** Official v1.0.0 milestone release, community contribution guidelines, and ongoing roadmap.
