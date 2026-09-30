# Mathematical & Signal Processing Foundations of Expressive Speech Evaluation

> **Document ID:** `VOX-THEORY-001`  
> **Author:** Scrooge-777  
> **Version:** 1.0.0 (Day 1 Foundation)  
> **Target System:** VoxExpress-Eval Framework

---

## 1. Introduction & Problem Formalization

A speech signal $s(t) \in L^2(\mathbb{R})$ conveys information across three orthogonal dimensions:
1. **Linguistic Information ($\mathcal{L}$):** Phonemes, syllables, syntax, and semantics.
2. **Paralinguistic / Expressive Information ($\mathcal{E}$):** Emotion, mood, emphasis, attitude, and conversational pauses.
3. **Extralinguistic Information ($\mathcal{X}$):** Speaker anatomy, vocal tract geometry, age, gender, and accent.

In standard Text-to-Speech (TTS), objective evaluation has historically relied on point-wise acoustic distances such as **Mel-Cepstral Distortion (MCD)**:
$$\text{MCD} = \frac{10\sqrt{2}}{\ln 10} \frac{1}{T} \sum_{t=1}^{T} \sqrt{\sum_{d=1}^{D} \left( c_d^{\text{ref}}(t) - c_d^{\text{synth}}(t) \right)^2}$$

### Why MCD Fails for Expressive & Multilingual TTS:
- **Temporal Non-Alignment:** Expressive speech features organic lengthening, hesitations, and micro-pauses that cause point-wise Euclidean Euclidean distances to blow up even when the synthesis is natural.
- **Multilingual Spectral Mismatch:** Transferring a speaker voice across languages alters formant transitions due to phonotactic constraints, rendering raw cepstral distances uninformative.
- **Independence of Prosody:** MCD is largely invariant to pitch trajectory ($F_0$) dynamics and energy envelope modulation.

VoxExpress-Eval resolves this by decomposing evaluation into **invariant manifold geometries** and **scale-separated dynamical metrics**.

---

## 2. Multi-Scale Prosodic Dynamics via Continuous Wavelet Transform (CWT)

Prosody operates simultaneously across multiple temporal hierarchies:
- **Phonetic level:** Duration $10 - 50\text{ ms}$ (micro-prosodic perturbations, consonant transitions).
- **Syllabic / Word level:** Duration $100 - 300\text{ ms}$ (lexical stress, pitch accents).
- **Phrase / Sentence level:** Duration $1 - 5\text{ s}$ (intonation group, question rise, terminal fall).

### 2.1 Continuous Wavelet Formulation
Given an interpolated, zero-mean, continuous fundamental frequency trajectory $f_0(t)$:
$$\mathcal{W}_{\psi}[f_0](a, b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} f_0(t) \, \psi^*\left(\frac{t - b}{a}\right) dt$$
where:
- $a \in \mathbb{R}^+$ is the scale parameter (dilation/compression, inversely proportional to frequency).
- $b \in \mathbb{R}$ is the translation parameter (temporal position).
- $\psi(t)$ is the **Mexican Hat Wavelet** (second derivative of a Gaussian):
$$\psi(t) = \frac{2}{\sqrt{3\sigma} \pi^{1/4}} \left(1 - \left(\frac{t}{\sigma}\right)^2\right) \exp\left(-\frac{t^2}{2\sigma^2}\right)$$

### 2.2 Discrete Octave Decomposition
We decompose the pitch contour into $J = 10$ dyadic octaves with scale intervals $\tau_j = \tau_0 \cdot 2^{j/2}$:
$$f_0(t) \approx \sum_{j=1}^{J} \mathcal{W}_{\psi}[f_0](\tau_j, t) + r(t)$$

The **Wavelet Prosodic Energy Distance (WPED)** between synthetic and reference prosody is defined as:
$$\mathcal{D}_{\text{CWT}} = \sum_{j=1}^{J} w_j \cdot \left| \log \int |\mathcal{W}[f_0^{\text{synth}}](\tau_j, t)|^2 dt - \log \int |\mathcal{W}[f_0^{\text{ref}}](\tau_j, t)|^2 dt \right|$$
where $w_j$ weights sentence-level intonational arcs over high-frequency vocal micro-jitter.

---

## 3. Optimal Transport & The Wasserstein Metric for Expressiveness

Expressive speech synthesis should match the **joint distribution of pitch and energy dynamics** without requiring rigid frame-to-frame alignment.

Let $\mathbf{z}(t) = [F_0(t), E_{\text{RMS}}(t)]^T \in \mathbb{R}^2$ represent the instantaneous pitch-energy state vector at time $t$.

### 3.1 Wasserstein-1 Distance (Earth Mover's Distance)
Let $P_{\text{synth}}$ and $P_{\text{ref}}$ be the probability measures over the acoustic state space $\Omega \subset \mathbb{R}^2$:
$$\mathcal{W}_1(P_{\text{synth}}, P_{\text{ref}}) = \inf_{\gamma \in \Pi(P_{\text{synth}}, P_{\text{ref}})} \iint_{\Omega \times \Omega} \|\mathbf{u} - \mathbf{v}\|_2 \, d\gamma(\mathbf{u}, \mathbf{v})$$
where $\Pi(P_{\text{synth}}, P_{\text{ref}})$ denotes the set of all joint distributions on $\Omega \times \Omega$ with marginals $P_{\text{synth}}$ and $P_{\text{ref}}$.

By the Kantorovich-Rubinstein duality:
$$\mathcal{W}_1(P_{\text{synth}}, P_{\text{ref}}) = \sup_{\|\phi\|_{\text{Lip}} \le 1} \left( \mathbb{E}_{\mathbf{u} \sim P_{\text{synth}}}[\phi(\mathbf{u})] - \mathbb{E}_{\mathbf{v} \sim P_{\text{ref}}}[\phi(\mathbf{v})] \right)$$

This metric guarantees that an expressive utterance that varies its dynamic range smoothly receives high reward, while monotone or flat speech is heavily penalized.

---

## 4. Cross-Lingual Speaker Manifold Geometry

Cross-lingual voice cloning requires maintaining the speaker identity $\mathcal{X}$ while synthesizing in a foreign target language $L_t \ne L_s$.

### 4.1 Hyperspherical Latent Projections
Let $f_{\theta}: \mathcal{S} \to \mathbb{S}^{d-1}$ be a deep neural feature extractor (ECAPA-TDNN) parameterized by $\theta$, mapping an audio utterance to the $(d-1)$-dimensional unit hypersphere:
$$\mathbf{e} = \frac{f_{\theta}(s)}{\|f_{\theta}(s)\|_2}, \quad \|\mathbf{e}\|_2 = 1$$

### 4.2 Cross-Lingual Identity Preservation Score (C-LIPS)
Given a source reference utterance $s_{\text{ref}}^{(L_s)}$ and a synthesized multilingual utterance $\hat{s}_{\text{synth}}^{(L_t)}$:
$$\mathcal{S}_{\text{C-LIPS}} = \cos(\mathbf{e}_{\text{ref}}, \mathbf{e}_{\text{synth}}) = \mathbf{e}_{\text{ref}}^T \mathbf{e}_{\text{synth}}$$

To eliminate language-specific acoustic bias from the embedding space, we project onto the orthogonal complement of the language subspace $\mathcal{V}_{\text{lang}}$:
$$\tilde{\mathbf{e}} = (\mathbf{I} - \mathbf{P}_{\mathcal{V}_{\text{lang}}}) \mathbf{e}$$
where $\mathbf{P}_{\mathcal{V}_{\text{lang}}} = \mathbf{V} (\mathbf{V}^T \mathbf{V})^{-1} \mathbf{V}^T$ is the projection matrix formed by average language vectors.

---

## 5. Affective Circumplex & Speech Emotion Recognition (SER) Distance

Human emotional expression in speech can be parameterized via the **Russell Circumplex Model**:
$$\mathbf{a} = [v, a]^T \in [-1, 1]^2$$
- $v$: **Valence** (pleasantness: sad/angry $\to$ happy/serene)
- $a$: **Arousal** (activation/energy: bored/sleepy $\to$ excited/furious)

### 5.1 Affective Euclidean Divergence
For a target emotion class $c \in \{\text{Neutral, Happy, Angry, Sad, Surprise}\}$ with centroid $[\mu_v(c), \mu_a(c)]^T$ and covariance $\mathbf{\Sigma}_c$:
$$d_{\text{Mahalanobis}}(\hat{\mathbf{a}}, c) = \sqrt{(\hat{\mathbf{a}} - \boldsymbol{\mu}_c)^T \mathbf{\Sigma}_c^{-1} (\hat{\mathbf{a}} - \boldsymbol{\mu}_c)}$$

---

## 6. The Unified VoxExpress Composite Score

We synthesize these multi-dimensional metrics into a single calibrated benchmark score $\mathcal{V} \in [0, 100]$:

$$\mathcal{V}_{\text{Express}} = 100 \cdot \left[ \alpha \cdot \mathcal{S}_{\text{Prosody}} + \beta \cdot \mathcal{S}_{\text{Emotion}} + \gamma \cdot \mathcal{S}_{\text{Speaker}} + \delta \cdot \mathcal{S}_{\text{Intelligibility}} \right]$$

subject to:
$$\alpha + \beta + \gamma + \delta = 1, \quad \alpha, \beta, \gamma, \delta > 0$$

- $\alpha = 0.30$ (Prosodic Wavelet & Dynamic Range)
- $\beta = 0.25$ (Affective Wasserstein Coupling)
- $\gamma = 0.25$ (Cross-Lingual Identity Preservation)
- $\delta = 0.20$ (Phonetic Alignment & Intelligibility)

---

## 7. References
1. **Möllering et al. (2023):** *Continuous Wavelet Transform for Multi-Scale Speech Rhythm and Intonation Analysis.* Interspeech.
2. **Villani, C. (2009):** *Optimal Transport: Old and New.* Springer-Verlag Berlin Heidelberg.
3. **Desplanques et al. (2020):** *ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation for Speaker Verification.* Interspeech.
4. **Russell, J. A. (1980):** *A Circumplex Model of Affect.* Journal of Personality and Social Psychology, 39(6), 1161–1178.
