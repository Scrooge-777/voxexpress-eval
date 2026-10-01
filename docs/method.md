# Mathematical & Algorithmic Method Specification

> **Specification:** ExpressEval Quantitative Metrics & Learned Aggregator  
> **Document:** `docs/method.md`

---

## 1. Evaluation Metric Definitions

### 1.1 Intelligibility
- **Word Error Rate (WER):**
  $$\text{WER} = \frac{S + D + I}{N_{\text{ref}}}$$
  where $S$ is substitutions, $D$ is deletions, $I$ is insertions, and $N_{\text{ref}}$ is the number of words in the reference text.
- **Character Error Rate (CER):**
  $$\text{CER} = \frac{S_{\text{char}} + D_{\text{char}} + I_{\text{char}}}{N_{\text{chars}}}$$
  Applied particularly for languages without clear whitespace tokenization (Mandarin, Japanese).

### 1.2 Spectral Distance
- **Mel-Cepstral Distortion (MCD):**
  Computed over Dynamic Time Warping (DTW) aligned frame pairs:
  $$\text{MCD} = \frac{10\sqrt{2}}{\ln 10} \frac{1}{T} \sum_{t=1}^T \sqrt{\sum_{d=1}^D \left( c_d^{\text{syn}}(t) - c_d^{\text{ref}}(\phi(t)) \right)^2}$$
  where $\phi(t)$ is the optimal DTW warping alignment between synthesized and reference frames.

### 1.3 Speaker & Emotion Similarity
- **Cosine Embedding Distance:**
  Given embedding vectors $\mathbf{e}_{\text{syn}}$ and $\mathbf{e}_{\text{ref}}$ on the unit hypersphere:
  $$\mathcal{S}_{\text{cosine}} = \frac{\mathbf{e}_{\text{syn}} \cdot \mathbf{e}_{\text{ref}}}{\|\mathbf{e}_{\text{syn}}\| \|\mathbf{e}_{\text{ref}}\|}$$

### 1.4 Prosodic Dynamics
- **$F_0$ Correlation ($r_{F_0}$):**
  Computed exclusively over mutual voiced frames following DTW alignment:
  $$r_{F_0} = \frac{\sum_{t \in \mathcal{V}} (f_0^{\text{syn}}(t) - \bar{f}_0^{\text{syn}})(f_0^{\text{ref}}(t) - \bar{f}_0^{\text{ref}})}{\sqrt{\sum_{t \in \mathcal{V}} (f_0^{\text{syn}}(t) - \bar{f}_0^{\text{syn}})^2} \sqrt{\sum_{t \in \mathcal{V}} (f_0^{\text{ref}}(t) - \bar{f}_0^{\text{ref}})^2}}$$
- **$F_0$ Dynamic Range:** Expressed in semitones relative to reference pitch $f_{\text{ref}} = 440\text{ Hz}$:
  $$\text{Range}_{\text{st}} = 12 \log_2\left(\frac{\max(F_0)}{\min(F_0)}\right)$$

---

## 2. Learned Aggregator (Bradley-Terry Formulation)

1. **Language-Fair Normalization:**
   Features are z-scored per language group $L$:
   $$z_{i, d} = \frac{x_{i, d} - \mu_{L, d}}{\sigma_{L, d} + \epsilon}$$
2. **Aggregator Prediction:**
   $$\hat{y}_i = f(\mathbf{z}_i) = \mathbf{w}^T \mathbf{z}_i + b$$
3. **Loss Function:**
   $$\mathcal{L} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2 + \lambda \mathcal{L}_{\text{BT}}$$
   where $\mathcal{L}_{\text{BT}}$ is the Bradley-Terry pairwise ranking loss over pairs where human $y_i > y_j$:
   $$\mathcal{L}_{\text{BT}} = - \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} \log \sigma(\hat{y}_i - \hat{y}_j - \text{margin})$$

---

## 3. Statistical Validation Protocol

1. **System-Level & Utterance-Level Correlations:** Pearson $r$, Spearman $\rho$, Kendall $\tau$.
2. **Bootstrap Confidence Intervals:** 1,000 resamples generating 95% CIs.
3. **Steiger's Z-Test:** Proving significant correlation superiority over baseline metrics.
4. **Krippendorff's Alpha:** Measuring human rater consensus as the empirical upper bound.
