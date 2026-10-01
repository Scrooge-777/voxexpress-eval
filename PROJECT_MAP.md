# ExpressEval — Project Map
**Automatic Objective Evaluation of Multilingual Expressive Speech Synthesis (TTS)**

Input: synthesized audio + text + language (+ optional reference audio / target emotion)
Output: per-axis scores + one learned overall score that predicts human MOS, with confidence intervals.

---

## 1. File system

```
expresseval/
├── PROJECT_MAP.md              # this file
├── README.md                   # 5-line quickstart, results table, citation
├── LICENSE
├── pyproject.toml              # pip-installable, CLI entry point
├── Dockerfile
├── Makefile                    # make setup | eval | train | test | demo
├── CITATION.cff
│
├── configs/
│   ├── default.yaml            # metrics on/off, batch size, fp16, device
│   ├── languages.yaml          # language -> ASR model, G2P, LID code
│   └── train_aggregator.yaml   # loss weights, folds, seeds
│
├── src/expresseval/
│   ├── cli.py                  # `expresseval run --audio-dir ... --lang hi`
│   ├── pipeline.py             # loads audio once, fans out to metrics, caches
│   ├── io/                     # audio loading, resampling, manifest (CSV/Parquet)
│   ├── metrics/
│   │   ├── base.py             # Metric interface: name, axis, __call__(batch)
│   │   ├── intelligibility.py  # WER/CER via faster-whisper
│   │   ├── naturalness.py      # MOS predictor (UTMOS-style) wrapper
│   │   ├── speaker_sim.py      # cosine sim of speaker embeddings
│   │   ├── prosody.py          # F0 RMSE/corr, energy, duration, speaking rate, DTW
│   │   ├── spectral.py         # MCD, log-spectral distance
│   │   ├── emotion.py          # emotion embedding distance + classifier accuracy
│   │   └── crosslingual.py     # language-ID consistency, accent leakage
│   ├── aggregator/
│   │   ├── features.py         # stack metric outputs, z-normalize per language
│   │   ├── model.py            # ridge / small MLP -> predicted MOS
│   │   ├── losses.py           # MSE + pairwise ranking (Bradley-Terry)
│   │   └── train.py
│   ├── stats/
│   │   ├── correlation.py      # Pearson, Spearman, Kendall tau (utt + system level)
│   │   ├── bootstrap.py        # 95% CIs
│   │   ├── significance.py     # Fisher z / Steiger test for comparing metrics
│   │   └── agreement.py        # Krippendorff alpha for human raters
│   └── report/
│       ├── tables.py           # per-language, per-emotion tables
│       └── plots.py            # scatter vs human MOS, radar per system
│
├── data/
│   ├── README.md               # sources + licenses (no raw data committed)
│   ├── manifests/              # train/dev/test CSVs: path, text, lang, emotion, mos
│   └── cache/                  # embeddings, ASR outputs (gitignored)
│
├── scripts/
│   ├── download_data.sh
│   ├── build_manifests.py
│   └── export_onnx.py          # speed: export heavy models
│
├── experiments/
│   ├── 01_metric_correlations.ipynb
│   ├── 02_aggregator_ablation.ipynb
│   └── results/                # Parquet + final tables
│
├── app/
│   └── gradio_app.py           # upload audio -> scores + radar chart (HF Space)
│
├── tests/
│   ├── test_metrics.py         # known-answer tests (e.g. WER on toy strings)
│   ├── test_stats.py           # correlation / bootstrap sanity
│   └── test_cli.py
│
├── docs/
│   ├── method.md               # math definitions of every metric
│   ├── model_card.md
│   └── paper/                  # LaTeX draft
│
└── .github/workflows/ci.yml    # lint + tests on push
```

---

## 2. Evaluation axes (what gets measured)

| Axis | Metrics | Why it matters for *expressive multilingual* TTS |
|---|---|---|
| Intelligibility | WER, CER (CER for scripts without clear word boundaries) | Expressive speech often hurts clarity |
| Naturalness | Predicted MOS | Proxy for human listening score |
| Speaker similarity | Embedding cosine | Voice must survive emotion + language change |
| Prosody | F0 RMSE, F0 correlation, energy, duration, speaking rate | Core of "expressive" |
| Emotion | Emotion-embedding distance, classifier agreement with target | Did the intended emotion actually come through |
| Cross-lingual | LID posterior of target language, accent leakage score | Catches code-mixing / wrong-accent failures |
| Spectral | MCD | Needs a reference; classic baseline |

---

## 3. Math layer (the "reasoning" part)

**Metric definitions** (in `docs/method.md`)
- WER = (S + D + I) / N
- MCD = (10 / ln 10) * sqrt(2 * sum_d (c_d - c'_d)^2), averaged over DTW-aligned frames
- Speaker/emotion similarity = cos(e_ref, e_syn)
- F0 correlation computed over voiced frames only, after DTW alignment

**Learned aggregator**
- Features: per-language z-scored metric vector x
- Prediction: y_hat = f(x), where f is ridge regression first, small MLP second
- Loss: L = MSE(y, y_hat) + lambda * pairwise ranking loss (Bradley-Terry: -log sigmoid(y_hat_i - y_hat_j) for pairs where human y_i > y_j)
- Per-language and per-emotion calibration so one language doesn't dominate

**Validation**
- Utterance-level and system-level Pearson, Spearman, Kendall tau
- Bootstrap 95% CIs (resample systems, not just utterances, for system-level)
- Fisher z / Steiger test: is the aggregator significantly better than the best single metric?
- Krippendorff alpha on human ratings = ceiling for any automatic metric
- Leave-one-language-out CV = does it generalize to unseen languages

---

## 4. Speed plan
1. Load and resample audio once, share across all metrics
2. Cache every embedding/ASR output keyed by file hash (re-runs are near-instant)
3. Batch + fp16 on GPU; faster-whisper instead of vanilla Whisper
4. Export heavy encoders to ONNX
5. Lazy-load metrics: only the ones enabled in config get imported
6. Write results as Parquet, not CSV

---

## 5. Publish checklist
- [ ] `pip install -e .` and `expresseval run` work on a clean machine
- [ ] Docker image builds and runs the demo manifest
- [ ] Tests + CI green
- [ ] Hugging Face Space (Gradio) live
- [ ] GitHub release + PyPI package
- [ ] Zenodo DOI + CITATION.cff
- [ ] Model card + dataset license table
- [ ] arXiv / Interspeech-style paper draft in `docs/paper/`

---

## 6. Topic-specific additions (ranked)
1. **Language-fair normalization**: per-language z-scoring so scores are comparable across languages
2. **Emotion-intensity score**: not just "right emotion" but "how strongly", with a monotonic check vs intensity labels
3. **Expressiveness vs intelligibility Pareto plot**: where each TTS system sits on the trade-off
4. **Code-mixing test set**: sentences mixing two languages, scored for LID consistency
5. **Prosody transfer score**: when a reference style audio is given, how closely the output follows it
6. **Metric-failure detector**: flag clips where metrics disagree strongly (usually a metric bug or a weird clip)
7. **Rank-agreement leaderboard**: score a set of public TTS systems and show whether metric ranking matches human ranking

---

## 7. Build order
1. Manifest format + `Metric` interface + CLI skeleton
2. WER/CER, speaker sim, F0 prosody
3. Stats module (correlations + bootstrap)
4. Naturalness + emotion + cross-lingual metrics
5. Aggregator training + ablations
6. Gradio demo, Docker, CI
7. Paper + release
