# Multi-head SepVAE Diagnostic Manual

A comprehensive framework for interpreting model health during training and evaluation of the Multi-head Salient Variational Autoencoder with CheSS backbone.

---

## Table of Contents

1. [Loss Term Interpretation & Convergence Guidelines](#1-loss-term-interpretation--convergence-guidelines)
2. [Structural Reconstruction & Artifact Evaluation](#2-structural-reconstruction--artifact-evaluation)
3. [Latent Manifold Diagnostics](#3-latent-manifold-diagnostics)
4. [Compositional Geometry for Multi-Disease Synthesis](#4-compositional-geometry-for-multi-disease-synthesis)
5. [Quick Reference Tables](#5-quick-reference-tables)

---

## 1. Loss Term Interpretation & Convergence Guidelines

### 1.1 Reconstruction Loss (MSE)

The reconstruction loss measures pixel-level fidelity between input images and their reconstructions. In the SepVAE architecture, this operates on 512×512 grayscale chest X-rays normalized to [-1, 1].

#### Expected Training Trajectory

| Phase | Epoch Range | Expected Behavior | Typical Values |
|-------|-------------|-------------------|----------------|
| **Early** | 1-10 | Rapid decrease as decoder learns global structure | 0.15 → 0.05 |
| **Mid** | 10-50 | Gradual refinement of anatomical details | 0.05 → 0.02 |
| **Late** | 50+ | Plateau with minor fluctuations | 0.015 → 0.01 |

#### Diagnostic Criteria

| Status | Characteristics | Visual Indicators |
|--------|-----------------|-------------------|
| **Good** | Sharp anatomical boundaries, preserved lung textures, clear cardiac silhouette | Bone edges are crisp; lung markings visible; no ghosting artifacts |
| **Stable** | Loss fluctuates ±10% around plateau | Minor epoch-to-epoch variation; consistent across disease classes |
| **Problematic** | Loss stagnates early (>0.05) or oscillates wildly | Blurred reconstructions; loss of fine detail; class-dependent quality variance |

#### Failure Mode Analysis

```
Symptom: Reconstruction loss stuck at high plateau (>0.08 after epoch 20)
├── Cause 1: Decoder capacity insufficient
│   └── Action: Increase FPN channels (--fpn_channels 768)
├── Cause 2: Learning rate too low for decoder
│   └── Action: Increase --lr_vae (try 2e-4)
└── Cause 3: Aggressive KL regularization
    └── Action: Reduce --weight_kl_common/disease or enable --free_bits 1.0

Symptom: High-frequency noise in reconstructions
├── Cause 1: Adversarial loss dominates too early
│   └── Action: Delay --disc_start_epoch or reduce --weight_adversarial
└── Cause 2: Insufficient gradient clipping
    └── Action: Reduce --grad_clip (try 0.5)
```

#### Example Interpretation

| Scenario | Loss Value | Interpretation |
|----------|------------|----------------|
| MSE = 0.008 at epoch 80 | Excellent | Near pixel-perfect reconstruction; ready for downstream LDM |
| MSE = 0.025 at epoch 80 | Acceptable | Sufficient for disentanglement; may need decoder refinement |
| MSE = 0.070 at epoch 80 | Poor | Fundamental learning issue; check backbone integration |

---

### 1.2 KL Divergence (Common & Disease)

The KL divergence terms regularize the latent spaces toward their priors. The architecture uses:
- **Common space**: 64×64×4 spatial latents (anatomy)
- **Disease spaces**: 64×64×2 spatial latents each (pathology-specific)

#### Free-Bits Mechanism

The `--free_bits` parameter prevents posterior collapse by allowing a minimum information rate per channel:

```
KL_effective = max(0, KL_raw - free_bits)
```

| Free-Bits Setting | Effect | Recommended Use Case |
|-------------------|--------|----------------------|
| 0.0 (default) | Full KL penalty | When latent utilization is already high |
| 0.5 | Mild protection | General-purpose training |
| 1.0 | Moderate protection | When observing posterior collapse |
| 2.0+ | Strong protection | Debugging only; may cause underfitting |

#### Expected Value Ranges

| Metric | Early Training | Converged (Good) | Posterior Collapse | Over-regularized |
|--------|----------------|------------------|-------------------|------------------|
| `kl_total` (raw) | 50-200 | 10-50 | <1.0 | >500 |
| `kl_weighted` | 0.01-0.05 | 0.005-0.02 | <0.0001 | >0.1 |
| `kl_common` | 30-150 | 8-40 | <0.5 | >400 |
| `kl_disease` | 10-50 per head | 2-15 per active head | <0.2 | >100 |

#### KL Warmup Interpretation

When using `--kl_warmup_epochs`, the annealing factor `β(t)` scales KL contributions:

| Epoch (assuming 10-epoch warmup) | β(t) | Expected Behavior |
|----------------------------------|------|-------------------|
| 1 | 0.1 | Reconstruction-dominated; latents may be unstructured |
| 5 | 0.5 | Balance emerges; latent organization begins |
| 10+ | 1.0 | Full regularization; latent space should be smooth |

#### Diagnostic Decision Tree

```
Is kl_total < 5.0?
├── YES → Posterior Collapse Detected
│   ├── Check: Are reconstructions still good?
│   │   ├── YES → Decoder bypassing latents (autoencoder mode)
│   │   │   └── Action: Increase --weight_kl_* by 10x; add noise to decoder
│   │   └── NO → Model not learning at all
│   │       └── Action: Reduce --weight_rec; check data pipeline
│   └── Immediate Action: Enable --free_bits 1.0
│
└── NO → Is kl_total > 200?
    ├── YES → Over-regularization
    │   ├── Symptom: Blurry reconstructions despite low MSE
    │   └── Action: Reduce --weight_kl_* by 5x
    └── NO → Healthy Range (10-100)
        └── Monitor: Ensure disease KL < common KL for healthy samples
```

---

### 1.3 Nulling Loss

The nulling loss enforces that disease-specific latent heads remain inactive (close to prior) for healthy control samples. This is critical for the "common + salient" disentanglement.

#### Mechanism

For samples labeled as "Normal" (class 0):
- Both disease heads (Effusion, Cardiomegaly) should output μ ≈ 0, σ ≈ σ_inactive
- The nulling loss penalizes deviation from this prior

#### Expected Trajectory

| Phase | Epoch Range | Expected `loss/nulling` | Interpretation |
|-------|-------------|-------------------------|----------------|
| **Early** | 1-5 | 0.5 - 2.0 | Routing logic learning; heads may fire indiscriminately |
| **Mid** | 5-20 | 0.1 - 0.5 | Disease heads learning to deactivate for normals |
| **Converged** | 20+ | <0.05 | Proper routing established; minimal leakage |

#### σ_inactive Hyperparameter

| Setting | Effect | Diagnostic Implication |
|---------|--------|------------------------|
| 1.0 (default) | Standard prior; heads output N(0,1) when inactive | Balanced regularization |
| 0.5 | Tight prior; stronger nulling constraint | Faster convergence but may over-constrain |
| 2.0 | Loose prior; permissive nulling | Slower convergence; may allow information leakage |

#### Failure Modes

| Symptom | Nulling Value | Root Cause | Corrective Action |
|---------|---------------|------------|-------------------|
| Persistent high nulling | >0.3 at epoch 50 | Disease features leaking into anatomy | Increase `--weight_null` to 5e-3 |
| Nulling near zero but reconstructions poor | <0.01 | Over-nulling; useful information suppressed | Reduce `--weight_null` to 5e-4 |
| Nulling oscillates | Varies ±50% | Unstable routing gradients | Reduce `--lr_vae`; add gradient clipping |

#### Diagnostic Checklist

- [ ] Is nulling loss decreasing monotonically in early training?
- [ ] Does nulling loss for diseased samples remain higher than for normals?
- [ ] At convergence, is `loss/nulling` < 0.05?
- [ ] Do reconstruction grids show normal samples without disease artifacts?

---

### 1.4 Mutual Information (MI) Penalty

The MI penalty ensures latent independence between the common (anatomical) and salient (disease) spaces. This uses a discriminator-based estimator trained adversarially.

#### Two-Player Dynamics

| Component | Objective | Update Frequency |
|-----------|-----------|------------------|
| **MI Discriminator** | Maximize ability to detect statistical dependence | Every step (`mi_disc_step`) |
| **Encoder** | Minimize MI by making z_c and z_d independent | Via `loss/mi_penalty` term |

#### Expected Equilibrium

| Metric | Early Training | Equilibrium (Good) | Encoder Winning | Discriminator Winning |
|--------|----------------|-------------------|-----------------|----------------------|
| `loss/mi_penalty` | 0.5 - 2.0 | 0.3 - 0.8 | <0.1 | >2.0 |
| `loss/mi_disc_update` | High variance | Stable ~0.5 | Stuck near 0 | Increasing |

#### Interpretation Framework

```
Equilibrium Analysis:

              MI_penalty decreasing
                      │
         ┌────────────┼────────────┐
         │            │            │
    Too Fast      Healthy      Too Slow
   (<0.1 by      (0.3-0.8)    (>1.5 at
   epoch 20)                  epoch 50)
         │            │            │
         ▼            ▼            ▼
   Encoder may     Good        Shared info
   be ignoring   separation    between
   MI signal                   latents
         │            │            │
         ▼            ▼            ▼
   Check if        Ready      Increase
   disentan-       for        --weight_mi
   glement         LDM
   works
```

#### Healthy Dynamics Table

| Epoch | MI Penalty | MI Disc Loss | Interpretation |
|-------|------------|--------------|----------------|
| 5 | 1.2 | 0.7 | Both learning; initial exploration |
| 20 | 0.6 | 0.5 | Approaching equilibrium |
| 50 | 0.4 | 0.45 | Stable equilibrium; good disentanglement |
| 50 | 0.05 | 0.1 | WARNING: Possible mode collapse in latents |
| 50 | 1.8 | 0.9 | WARNING: Insufficient MI penalty weight |

---

### 1.5 Perceptual & Adversarial (PatchGAN) Dynamics

These auxiliary losses improve reconstruction quality by matching high-level features (perceptual) and local texture statistics (adversarial).

#### Perceptual Loss (Backbone-Based)

Uses the frozen CheSS backbone to compare feature representations:

| Layer Compared | Sensitivity | Expected Contribution |
|----------------|-------------|----------------------|
| Early conv | Edges, textures | High (0.3-0.5 of perceptual) |
| Mid layers | Anatomical structures | Medium (0.3-0.4) |
| Deep layers | Semantic content | Low (0.1-0.2) |

#### PatchGAN Discriminator Schedule

The discriminator activates at `--disc_start_epoch` to prevent early training instability:

| Epoch (assuming disc_start=10) | `disc_factor` | Expected Behavior |
|--------------------------------|---------------|-------------------|
| 1-9 | 0.0 | Pure reconstruction + regularization |
| 10 | 1.0 | Discriminator activates; may cause loss spike |
| 11-20 | 1.0 | Discriminator and generator find balance |
| 20+ | 1.0 | Stable adversarial dynamics |

#### Adversarial Balance Indicators

| Metric | Healthy Range | Generator Dominates | Discriminator Dominates |
|--------|---------------|---------------------|------------------------|
| `loss/gen_adversarial` | 0.5 - 2.0 | <0.1 (D fooled completely) | >4.0 (D always wins) |
| `loss/patch_disc` | 0.3 - 0.7 | >0.9 (D can't learn) | <0.1 (D overfitting) |
| Ratio (gen/disc) | 1.0 - 3.0 | <0.5 | >5.0 |

#### Failure Mode Diagnosis

```
Symptom: Reconstruction loss spikes when discriminator activates
├── Normal if spike is <2x baseline and recovers within 5 epochs
└── Problematic if:
    ├── Spike is >3x baseline
    │   └── Action: Reduce --weight_adversarial (try 0.05)
    ├── Loss doesn't recover
    │   └── Action: Delay --disc_start_epoch by 10 more epochs
    └── Oscillations begin
        └── Action: Reduce --lr_patch_disc (try 2e-4)

Symptom: Checkerboard patterns emerge after discriminator starts
├── Cause: Discriminator focusing on frequency artifacts
└── Action: Ensure --upsample_method is 'bilinear' or 'subpixel'
```

#### Recommended Weight Relationships

| Training Stage | Reconstruction | Perceptual | Adversarial | Notes |
|----------------|---------------|------------|-------------|-------|
| Early (1-10) | 1.0 | 0.1 | 0.0 | Focus on structure |
| Mid (10-30) | 1.0 | 0.1 | 0.05 | Introduce realism gradually |
| Late (30+) | 1.0 | 0.1 | 0.1 | Full multi-objective |

---

## 2. Structural Reconstruction & Artifact Evaluation

### 2.1 Upsampling Artifact Detection

The decoder upsamples from 64×64 spatial latents to 512×512 output. The `--upsample_method` critically affects artifact patterns.

#### Method Comparison

| Method | Mechanism | Artifact Risk | Quality | Memory |
|--------|-----------|---------------|---------|--------|
| `nearest` | Replicate pixels | **HIGH** - Checkerboard | Low | Low |
| `bilinear` | Weighted average | LOW - Smooth | Medium | Low |
| `subpixel` | Learned shuffle | LOWEST | High | +10% |

#### Checkerboard Detection Checklist

When evaluating reconstruction grids from `samples/recon_epochXXXX.png`:

| Inspection Area | Good | Problematic | Diagnostic Action |
|-----------------|------|-------------|-------------------|
| Lung fields | Smooth gradients, natural texture | Grid pattern at 2x2 or 4x4 scale | Switch to `bilinear` |
| Rib edges | Sharp, continuous boundaries | Jagged staircase artifacts | Enable `subpixel` |
| Cardiac border | Smooth silhouette | Pixelated outline | Check decoder conv layers |
| Background (outside body) | Uniform intensity | Mottled pattern | Reduce adversarial weight |

#### Visual Reference

```
GOOD (bilinear/subpixel):          BAD (nearest + checkerboard):
┌────────────────────┐              ┌────────────────────┐
│  ░░▒▒▓▓██▓▓▒▒░░   │              │  █░█░█░█░█░█░█░█  │
│ ░▒▒▓▓████▓▓▒▒░    │              │ ░█░█░█░█░█░█░█░   │
│░▒▓▓████████▓▓▒░   │              │█░█░█░██░█░█░█░█   │
│▒▓▓██████████▓▓▒   │              │░█░██░██░██░█░█░   │
│ Smooth gradients  │              │  Checker pattern   │
└────────────────────┘              └────────────────────┘
```

#### Aliasing in Frequency Domain

For advanced diagnostics, compute FFT of reconstruction residual:

| FFT Pattern | Interpretation | Action |
|-------------|----------------|--------|
| Low-frequency dominated | Normal reconstruction error | None |
| Spike at Nyquist (high-freq corners) | Aliasing from upsampling | Switch upsample method |
| Periodic peaks | Decoder learned periodic artifacts | Retrain with spectral normalization |

---

### 2.2 Medical Fidelity Assessment

Evaluating whether reconstructions preserve clinically relevant features.

#### Anatomical Structure Checklist

| Structure | Good Reconstruction | Degraded Reconstruction | Critical? |
|-----------|---------------------|------------------------|-----------|
| **Clavicles** | Visible, symmetric | Blurred or asymmetric | Medium |
| **Ribs** | Countable, defined spacing | Merged or missing | High |
| **Cardiac silhouette** | Clear borders, CTR measurable | Fuzzy edges | High |
| **Costophrenic angles** | Sharp, well-defined | Blunted or obscured | High |
| **Lung parenchyma** | Vascular markings visible | Homogeneous opacity | High |
| **Mediastinum** | Defined trachea, aortic knob | Loss of definition | Medium |
| **Diaphragm** | Smooth dome, clear interface | Irregular or absent | High |

#### CheSS Backbone Feature Alignment

The frozen CheSS backbone extracts features optimized for chest X-ray understanding. Verify alignment:

| Backbone Layer | Expected Sensitivity | Verification Method |
|----------------|---------------------|---------------------|
| `layer1` (256ch) | Edges, local contrast | Compare Sobel magnitudes |
| `layer2` (512ch) | Texture patterns | Gram matrix similarity |
| `layer3` (1024ch) | Anatomical regions | Activation map overlap |
| `layer4` (2048ch) | Global semantics | Feature cosine similarity |

#### Quantitative Fidelity Metrics

| Metric | Computation | Good Threshold | Clinical Relevance |
|--------|-------------|----------------|-------------------|
| SSIM | Structural similarity | >0.85 | Preserves anatomy |
| PSNR | Peak signal-to-noise | >25 dB | Detail preservation |
| LPIPS | Perceptual similarity | <0.2 | Visual quality |
| FID (features) | Backbone feature distance | <50 | Distribution match |

---

### 2.3 Pathological Veracity

Ensuring disease features are correctly represented when active.

#### Disease-Specific Feature Tables

**Pleural Effusion (Class 1):**

| Feature | Good Reconstruction | Bad Reconstruction |
|---------|--------------------|--------------------|
| Fluid level | Sharp meniscus visible | Diffuse haziness |
| Costophrenic blunting | Smooth obliteration | Artificial step edges |
| Density gradient | Homogeneous opacity | Patchy or mottled |
| Laterality | Correct side affected | Mirrored to wrong side |

**Cardiomegaly (Class 2):**

| Feature | Good Reconstruction | Bad Reconstruction |
|---------|--------------------|--------------------|
| CTR (cardiothoracic ratio) | >0.5, measurable | Unclear boundaries |
| Cardiac borders | Enlarged but defined | Blurred or irregular |
| Chamber proportions | Plausible enlargement | Asymmetric distortion |
| Pulmonary vasculature | Appropriate redistribution | Missing congestion signs |

#### Cross-Class Leakage Detection

| Test Case | Expected Output | Leakage Indicator |
|-----------|-----------------|-------------------|
| Normal → Reconstruct | No disease features | Effusion/cardiomegaly visible |
| Effusion → Reconstruct | Only effusion features | Cardiomegaly also present |
| Cardiomegaly → Reconstruct | Only enlarged heart | Effusion also present |

#### Diagnostic Protocol

1. **Visual inspection of class-stratified grids:**
   - Row 0: Normal originals vs reconstructions
   - Row 1: Effusion originals vs reconstructions
   - Row 2: Cardiomegaly originals vs reconstructions

2. **Cross-reconstruction test:**
   - Feed normal images through diseased latent paths
   - Should remain normal (nulling loss working)

3. **Feature activation maps:**
   - Compare backbone activations for original vs reconstruction
   - High correlation = good fidelity

---

## 3. Latent Manifold Diagnostics

### 3.1 Common vs. Salient Separation

The architectural goal is to isolate anatomical variation in `z_common` (64×64×4) and disease variation in `z_disease` (64×64×2 per head).

#### Visualization Protocol

| Visualization | Tool | What to Look For |
|---------------|------|------------------|
| PCA of `z_common` | `sepvae_analysis.py` | No clustering by disease class |
| PCA of `z_disease` | `sepvae_analysis.py` | Clear separation by disease class |
| t-SNE of concatenated | External | No mixed clusters |

#### Separation Quality Metrics

| Metric | Computation | Good | Poor |
|--------|-------------|------|------|
| **Silhouette (common)** | Cluster coherence for disease labels | <0.1 (no clusters) | >0.3 (leakage) |
| **Silhouette (salient)** | Cluster coherence for disease labels | >0.5 (clear clusters) | <0.2 (entangled) |
| **Inter-class distance ratio** | mean(between)/mean(within) | >2.0 (salient), <1.5 (common) | Inverted ratios |

#### Expected PCA Structure

**z_common (Good):**
```
        PC2
         │     ● Normal
         │   ● Effusion
    ─────┼───● Cardiomegaly──── PC1
         │ ●   ●  ●
         │   ●  ●
         │ (All classes mixed = anatomy only)
```

**z_disease (Good):**
```
        PC2
         │
    ●●●  │        ▲▲▲
    ●●●  │       ▲▲▲▲
    Normal────────────────── PC1
         │    ■■■■
         │   ■■■■
         │ (Distinct clusters for each disease)

Legend: ● Normal, ▲ Effusion, ■ Cardiomegaly
```

#### Failure Mode Analysis

| Observation | Diagnosis | Corrective Action |
|-------------|-----------|-------------------|
| Disease clusters in common space | Anatomy contaminated | Increase MI penalty |
| Normal scattered in salient space | Nulling failure | Increase nulling weight |
| All classes overlapping everywhere | Complete entanglement | Re-check backbone integration |
| Perfect separation in both spaces | Redundant encoding | May be acceptable; check recon |

---

### 3.2 Latent Smoothness

For downstream LDM training, the latent manifold must be continuous and well-distributed.

#### Continuity Diagnostics

| Test | Method | Good Indicator |
|------|--------|----------------|
| **Linear interpolation** | Interpolate between two latents | Smooth transition in pixel space |
| **Latent perturbation** | Add small Gaussian noise | Minor, coherent changes |
| **Neighborhood consistency** | k-NN in latent vs pixel space | High rank correlation |

#### Distribution Coverage

| Metric | Description | Good Range | Action if Bad |
|--------|-------------|------------|---------------|
| **Variance ratio** | Var(z)/Prior_var | 0.7 - 1.3 | Adjust KL weight |
| **Kurtosis** | Tail behavior | -0.5 to 0.5 | Check for mode collapse |
| **Coverage (%) ** | % of prior volume occupied | >60% | Increase free_bits |

#### Interpolation Quality Table

| Interpolation Type | What Changes | What Should Stay Constant |
|-------------------|--------------|---------------------------|
| z_common interpolation | Patient anatomy, pose, body habitus | Disease presence/absence |
| z_disease interpolation | Disease severity | Anatomical identity |
| Cross-class interpolation | N/A (invalid test) | Should not be performed |

#### LDM Readiness Checklist

- [ ] Latent distribution is approximately Gaussian (Q-Q plot)
- [ ] No isolated clusters or voids in the manifold
- [ ] Interpolations produce plausible intermediate images
- [ ] Variance matches prior (avoid collapsed or exploded latents)
- [ ] Disease heads produce interpretable gradients when varied

---

### 3.3 Cluster Purity

Evaluating how well background (normal) and target (diseased) samples separate in the salient space.

#### Purity Metrics

| Metric | Formula | Threshold (Good) | Interpretation |
|--------|---------|------------------|----------------|
| **Cluster purity** | max_class / total per cluster | >0.9 | Clean separation |
| **Adjusted Rand Index** | Corrected for chance | >0.7 | Strong agreement |
| **Normalized MI** | MI / max(H_true, H_pred) | >0.6 | Information preserved |
| **Margin** | min inter-class / max intra-class | >1.5 | Well-separated |

#### Expected Purity by Training Stage

| Epoch | Normal Purity | Effusion Purity | Cardiomegaly Purity | Notes |
|-------|---------------|-----------------|---------------------|-------|
| 5 | 0.5 | 0.5 | 0.5 | Random baseline |
| 20 | 0.7 | 0.65 | 0.6 | Emerging structure |
| 50 | 0.9 | 0.85 | 0.8 | Good separation |
| 100 | 0.95+ | 0.9+ | 0.85+ | Converged |

#### Margin Analysis

The margin between classes indicates robustness to noise and generalization:

```
Margin = d(closest_different_class) - d(farthest_same_class)

Good (margin > 0):           Poor (margin < 0):

   ●●●         ▲▲▲              ●●▲ ●▲●
  ●●●●●      ▲▲▲▲▲             ●▲●▲ ▲●●
   ●●●●       ▲▲▲               ▲●●▲ ●▲
     │←margin→│                    │overlap│
```

#### Diagnostic Commands

```python
# Example: Compute cluster purity from latents
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans

# Extract latents
z_salient = model.encode_salient(images, labels)
z_flat = z_salient.reshape(len(z_salient), -1)

# Cluster and evaluate
kmeans = KMeans(n_clusters=3)
pred_labels = kmeans.fit_predict(z_flat)

ari = adjusted_rand_score(true_labels, pred_labels)
nmi = normalized_mutual_info_score(true_labels, pred_labels)

print(f"ARI: {ari:.3f}, NMI: {nmi:.3f}")
```

---

## 4. Compositional Geometry for Multi-Disease Synthesis

This section addresses a critical question: **How must the SepVAE latent manifold be structured to enable meaningful composition of multiple diseases (e.g., cardiomegaly + effusion) via the downstream Latent Diffusion Model?**

The LDM operates by denoising in the latent space. For disease composition to work, the manifold geometry must support additive or navigable combinations of disease factors while preserving anatomical coherence.

---

### 4.1 The Composition Problem

Our goal is to generate synthetic chest X-rays exhibiting **both cardiomegaly and effusion simultaneously**, starting from:
- A common anatomical representation `z_c`
- Disease-specific salient representations `z_eff` (effusion) and `z_card` (cardiomegaly)

The composition operation can be expressed as:

```
z_composed = f(z_c, z_eff, z_card)
```

For this to produce clinically plausible images, the latent manifold must satisfy specific geometric constraints.

---

### 4.2 Manifold Structure Taxonomy

Below are five distinct manifold configurations that can emerge from SepVAE training, with analysis of their compositional properties.

#### Structure A: Entangled Overlap (FAILS)

```
        z_salient PC2
              │
              │    ●▲■ ●▲
              │   ▲●■▲●■
         ─────┼────●▲■●▲────── PC1
              │   ■●▲■●▲
              │    ▲●■●
              │
        (All classes overlapping)

Legend: ● Normal, ▲ Effusion, ■ Cardiomegaly
```

| Property | Value | Impact on Composition |
|----------|-------|----------------------|
| Silhouette score | <0.1 | Cannot distinguish disease signals |
| Inter-class margin | Negative | Diseases encoded redundantly |
| Composition outcome | **FAILURE** | Adding z_eff + z_card = noise |

**Why it fails:** When diseases overlap in the salient space, there is no distinct "effusion direction" or "cardiomegaly direction" to combine. The encoder has failed to learn disentangled disease representations—attempting composition produces incoherent features.

**SepVAE dynamics causing this:**
- Insufficient `weight_mi` (MI penalty too weak)
- `weight_null` too low (disease signal leaks everywhere)
- Posterior collapse in disease heads

---

#### Structure B: Single-Axis Collapse (FAILS)

```
        z_salient PC2
              │
              │
              │
         ─────●────▲▲▲────■■■■── PC1
              │
              │    (All variation on PC1)
              │

Legend: ● Normal, ▲ Effusion, ■ Cardiomegaly
```

| Property | Value | Impact on Composition |
|----------|-------|----------------------|
| Effective dimensions | 1 | Only one disease axis exists |
| Orthogonality | 0.0 | Diseases mutually exclusive |
| Composition outcome | **FAILURE** | z_eff + z_card = cancellation or interpolation |

**Why it fails:** Both diseases are encoded along the same axis but in different directions. Composition becomes interpolation—you get "partial effusion + partial cardiomegaly" rather than both fully expressed. The manifold lacks the degrees of freedom to represent co-occurrence.

**SepVAE dynamics causing this:**
- Disease heads share too much encoder capacity
- Insufficient `z_channels_disease` (only 2 channels per head)
- Over-regularized KL collapsing variance to single direction

---

#### Structure C: Isolated Clusters with Gaps (FAILS for LDM)

```
        z_salient PC2
              │
        ●●●   │              ▲▲▲
       ●●●●●  │             ▲▲▲▲▲
         ─────┼───────────────────── PC1
              │
              │    ■■■■■
              │   ■■■■■■
              │

Legend: ● Normal, ▲ Effusion, ■ Cardiomegaly
```

| Property | Value | Impact on Composition |
|----------|-------|----------------------|
| Cluster separation | High (good!) | Diseases distinguishable |
| Manifold coverage | <30% | Large "dead zones" between clusters |
| Composition outcome | **FAILS for LDM** | Diffusion paths cross invalid regions |

**Why it fails:** Although diseases are well-separated (good for classification!), the space between clusters is empty. When the LDM tries to denoise toward a composition point (between effusion and cardiomegaly regions), it traverses regions with no training support. The decoder produces artifacts or mode collapse.

**SepVAE dynamics causing this:**
- Excessive KL weight (over-regularization)
- Insufficient `free_bits` allowing collapse
- Disease-only training without composition examples

**The critical insight:** Good classification structure ≠ good generative structure. LDMs need **continuous manifolds**, not isolated islands.

---

#### Structure D: Orthogonal Factorized (IDEAL)

```
        z_salient PC2 (Effusion axis)
              │
              │▲▲▲▲▲
              │▲▲▲▲▲▲
              │▲▲▲▲▲
         ●●●●●┼●●●●●●●●●●●●●●●●●●● PC1 (Cardiomegaly axis)
         ●●●●●│●●●●●●●●●●●●●●●●●●●
         ●●●●●│         ■■■■■■■■■
              │         ■■■■■■■■■
              │         ■■■■■■■■■
              │
                    ★ = z_eff + z_card
                       (valid composition target)

Legend: ● Normal (origin), ▲ Effusion, ■ Cardiomegaly, ★ Composition
```

| Property | Value | Impact on Composition |
|----------|-------|----------------------|
| Axis orthogonality | >0.8 | Independent disease factors |
| Manifold coverage | >70% | Continuous traversal possible |
| Composition outcome | **SUCCESS** | z_eff + z_card lands in valid region |

**Why it works:** Effusion and cardiomegaly occupy **orthogonal subspaces**. The effusion direction (PC2) is independent of the cardiomegaly direction (PC1). Composition is additive:

```
z_composed = z_normal + Δz_eff + Δz_card
```

The composed point lies in a region the decoder can interpret because:
1. The manifold is dense (no gaps)
2. Each disease contributes independently
3. Normal samples anchor the origin

**SepVAE dynamics producing this:**
- Balanced MI penalty (enough for independence, not too much)
- Separate disease heads with sufficient capacity
- Nulling loss anchoring normal at origin
- Appropriate KL with free-bits for coverage

---

#### Structure E: Curved Compositional Manifold (REQUIRES RIEMANNIAN)

```
        z_salient PC2
              │
              │    ▲▲▲▲
              │   ▲▲  ▲▲
              │  ▲      ▲
         ─────┼─●●●      ╲──────── PC1
              │  ●●●●     ╲
              │    ●●●●    ■■■
              │      ●●●●  ■■■■
              │         ●●●■■■
              │            ■■■

        Curved path ╲ = geodesic (valid)
        Straight line = Euclidean (INVALID - crosses gap)
```

| Property | Value | Impact on Composition |
|----------|-------|----------------------|
| Manifold curvature | High | Euclidean interpolation fails |
| Geodesic path | Valid | Riemannian navigation required |
| Composition outcome | **CONDITIONAL** | Works only with geometry-aware LDM |

**Why Euclidean fails, Riemannian works:** The latent manifold is curved—diseases don't lie on a flat plane. Straight-line interpolation (Euclidean) cuts through regions of low probability density, causing:
- Blurry intermediate samples
- Anatomical inconsistencies
- Mode collapse artifacts

**Geodesic paths** follow the manifold's curvature, staying in high-density regions throughout the composition trajectory.

---

### 4.3 Riemannian Geometry for Latent Navigation

When the SepVAE learns a curved manifold, Euclidean operations (addition, linear interpolation) become invalid. This section explains why and how to diagnose/address it.

#### The Problem: Euclidean vs. Geodesic Paths

```
EUCLIDEAN INTERPOLATION (Flat assumption):
─────────────────────────────────────────

    z_eff ●━━━━━━━━━━━━━━━━━● z_card
              │
              │ ✗ Crosses low-density void
              │ ✗ Decoder sees OOD inputs
              │ ✗ Artifacts and mode mixing
              ▼
         [Blurry mess]


GEODESIC INTERPOLATION (Manifold-aware):
────────────────────────────────────────

    z_eff ●                    ● z_card
           ╲                  ╱
            ╲   ●    ●    ●  ╱
             ╲  │    │    │ ╱
              ╲─┴────┴────┴╱
               ✓ Stays on manifold
               ✓ High-density path
               ✓ Plausible intermediates
                      │
                      ▼
              [Smooth transition]
```

#### Metric Tensor Interpretation

The Riemannian metric tensor `G(z)` at each point describes local geometry:

| Metric Property | Interpretation | Diagnostic |
|-----------------|----------------|------------|
| `det(G) ≈ 1` everywhere | Flat manifold (Euclidean OK) | Interpolation test |
| `det(G)` varies significantly | Curved manifold (Riemannian needed) | Jacobian analysis |
| `det(G) → 0` in regions | Manifold boundary/void | Coverage maps |

#### Practical Geodesic Computation

For SepVAE latents, approximate geodesics using:

```python
def geodesic_interpolation(z_start, z_end, decoder, n_steps=10):
    """
    Compute geodesic path using decoder Jacobian.

    The key insight: geodesics minimize path length in OUTPUT space,
    not input space. We want smooth image transitions.
    """
    # Initialize with Euclidean (as starting guess)
    path = [z_start + t * (z_end - z_start)
            for t in np.linspace(0, 1, n_steps)]

    # Iterative refinement toward geodesic
    for iteration in range(100):
        for i in range(1, n_steps - 1):
            # Compute local metric from decoder Jacobian
            J = jacobian(decoder, path[i])
            G = J.T @ J  # Pullback metric

            # Geodesic equation: move toward weighted midpoint
            G_inv = np.linalg.inv(G + 1e-6 * np.eye(G.shape[0]))
            midpoint = 0.5 * (path[i-1] + path[i+1])
            path[i] = path[i] + 0.1 * G_inv @ (midpoint - path[i])

    return path
```

#### When is Riemannian Geometry Necessary?

| Scenario | Euclidean OK? | Recommendation |
|----------|---------------|----------------|
| Structure D (orthogonal) | ✓ Yes | Simple addition works |
| Structure E (curved) | ✗ No | Use geodesic sampling |
| Mixed (partially curved) | Sometimes | Test interpolations first |

#### Diagnostic: Interpolation Smoothness Test

```
Test Protocol:
1. Sample z_eff from effusion cluster
2. Sample z_card from cardiomegaly cluster
3. Linear interpolate: z(t) = (1-t)*z_eff + t*z_card
4. Decode all z(t) → images
5. Evaluate:

PASS (Euclidean OK):
┌─────┬─────┬─────┬─────┬─────┐
│ Eff │ ↘   │  ↘  │  ↘  │Card │  Smooth transition
│     │     │     │     │     │  No blur spikes
└─────┴─────┴─────┴─────┴─────┘
  t=0   0.25  0.5   0.75  1.0

FAIL (Riemannian needed):
┌─────┬─────┬─────┬─────┬─────┐
│ Eff │ ░░░ │█████│ ░░░ │Card │  Middle frames blurry
│     │blur │VOID │blur │     │  or artifact-filled
└─────┴─────┴─────┴─────┴─────┘
  t=0   0.25  0.5   0.75  1.0
```

---

### 4.4 SepVAE Training Dynamics Affecting Composition

The geometry of the learned manifold is determined by training dynamics. This table maps hyperparameters to compositional outcomes.

#### Hyperparameter → Geometry → Composition Mapping

| Hyperparameter | Low Value Effect | High Value Effect | Ideal for Composition |
|----------------|------------------|-------------------|----------------------|
| `weight_mi` | Entangled (Structure A) | Over-separated (Structure C) | 1e-3 to 5e-3 |
| `weight_null` | Disease leakage | Origin collapse | 1e-3 (balanced anchoring) |
| `weight_kl_disease` | Scattered clusters | Collapsed to point | 1e-4 with free_bits |
| `free_bits` | Sparse coverage (C) | Overly uniform | 0.5 to 1.0 |
| `z_channels_disease` | Single-axis (B) | Sufficient DoF | ≥2 per disease |
| `sigma_inactive` | Tight origin | Dispersed normals | 1.0 (unit Gaussian) |

#### Training Phase Recommendations

| Phase | Focus | Key Metrics to Monitor |
|-------|-------|----------------------|
| **Epochs 1-20** | Reconstruction + Disentanglement | MSE↓, Nulling↓, MI stabilizing |
| **Epochs 20-50** | Manifold coverage | KL in range, no collapse |
| **Epochs 50-80** | Compositional structure | Interpolation smoothness test |
| **Epochs 80+** | Fine-tuning | Orthogonality, coverage maps |

#### Diagnostic: Compositional Readiness Checklist

Before training the downstream LDM, verify:

- [ ] **Orthogonality test**: cos(mean_z_eff, mean_z_card) < 0.3
- [ ] **Coverage test**: >60% of unit hypercube contains samples
- [ ] **Interpolation test**: No blurry frames in disease→disease paths
- [ ] **Additivity test**: z_normal + Δz_eff + Δz_card decodes plausibly
- [ ] **Jacobian test**: det(G) varies <10x across manifold (near-flat)

---

### 4.5 Composition Architectures

Given different manifold geometries, here are recommended LDM composition strategies:

#### Strategy 1: Additive Composition (For Structure D)

```
Architecture:
┌─────────────┐
│   z_common  │ ← Anatomy (frozen during composition)
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│  z_salient = z_eff + z_card         │ ← Simple addition
│            = Δeff + Δcard + z_origin │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────┐
│   Decoder   │ → X-ray with both diseases
└─────────────┘
```

| Condition | Requirement |
|-----------|-------------|
| Manifold structure | Orthogonal (Structure D) |
| Nulling quality | Excellent (normal ≈ origin) |
| Implementation | Direct latent arithmetic |

#### Strategy 2: Classifier-Free Guidance Composition (For Structure C/E)

```
Architecture:
┌────────────────────────────────────────────────────┐
│  LDM with disease conditioning                      │
│                                                     │
│  ε_θ(z_t, t, c_eff, c_card)                        │
│                                                     │
│  Guidance: ε = ε_uncond + w_eff*(ε_eff - ε_uncond) │
│                        + w_card*(ε_card - ε_uncond)│
└────────────────────────────────────────────────────┘
```

| Condition | Requirement |
|-----------|-------------|
| Manifold structure | Any (guidance navigates) |
| Training data | Needs some co-occurrence examples |
| Implementation | Conditional diffusion with CFG |

#### Strategy 3: Geodesic Diffusion (For Structure E)

```
Architecture:
┌─────────────────────────────────────────────────────┐
│  Riemannian Score Matching                          │
│                                                     │
│  Score: ∇_z log p(z) computed on manifold          │
│  Diffusion: dz = -G⁻¹(z)∇_z log p(z)dt + noise     │
│                                                     │
│  G(z) = J(z)ᵀJ(z) from decoder Jacobian            │
└─────────────────────────────────────────────────────┘
```

| Condition | Requirement |
|-----------|-------------|
| Manifold structure | Curved (Structure E) |
| Computational cost | High (Jacobian per step) |
| Implementation | Custom diffusion with metric |

---

### 4.6 Failure Case Gallery

Visual examples of composition failures and their manifold causes:

#### Case 1: Ghosting Artifacts (Entangled Manifold)

```
Input intent: Effusion + Cardiomegaly
Manifold: Structure A (entangled)

Result:
┌─────────────────────────────────┐
│     ░░░▓▓▓░░░                   │
│   ░▓███░░███▓░    ← Ghost       │
│  ▓████░░░████▓      effusion    │
│  ████░░░░░████   ← Doubled      │
│  ████░░░░░████      cardiac     │
│   ▓███░░███▓        border      │
│     ░░▓▓▓░░                     │
│                                 │
│  Diagnosis: z_eff and z_card    │
│  encode overlapping features    │
└─────────────────────────────────┘
```

#### Case 2: Feature Cancellation (Single-Axis Collapse)

```
Input intent: Effusion + Cardiomegaly
Manifold: Structure B (single axis)

Result:
┌─────────────────────────────────┐
│                                 │
│        ┌─────────┐              │
│       ╱           ╲             │
│      │   Normal-   │  ← Neither │
│      │   looking   │    disease │
│      │   heart     │    visible │
│       ╲           ╱             │
│        └─────────┘              │
│                                 │
│  Diagnosis: Diseases cancel     │
│  when on opposite ends of       │
│  same axis                      │
└─────────────────────────────────┘
```

#### Case 3: Void Artifacts (Sparse Clusters)

```
Input intent: Effusion + Cardiomegaly
Manifold: Structure C (gaps)

Result:
┌─────────────────────────────────┐
│     ▒▒▒▒▒▒▒▒▒▒▒                │
│   ▒▒███████████▒▒   ← Blurry   │
│  ▒▒█████████████▒▒    mess     │
│  ▒▒█████████████▒▒             │
│  ▒▒█████████████▒▒  ← No clear │
│   ▒▒███████████▒▒     anatomy  │
│     ▒▒▒▒▒▒▒▒▒▒▒                │
│                                 │
│  Diagnosis: Composition point   │
│  falls in untrained void region │
└─────────────────────────────────┘
```

#### Case 4: Successful Composition (Orthogonal Manifold)

```
Input intent: Effusion + Cardiomegaly
Manifold: Structure D (orthogonal)

Result:
┌─────────────────────────────────┐
│                                 │
│      ┌───────────────┐          │
│     ╱                 ╲         │
│    │   ████████████    │ ← Clear│
│    │   ████████████    │ cardio-│
│    │   ████████████    │ megaly │
│     ╲_________________╱         │
│    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ ← Clear│
│    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  effusion│
│                                 │
│  Both diseases clearly present  │
│  with preserved anatomy         │
└─────────────────────────────────┘
```

---

### 4.7 Manifold Diagnostic Commands

```python
# === Orthogonality Test ===
def test_orthogonality(z_eff_samples, z_card_samples, z_normal_samples):
    """Test if disease directions are orthogonal."""
    # Compute disease directions from normal origin
    delta_eff = z_eff_samples.mean(0) - z_normal_samples.mean(0)
    delta_card = z_card_samples.mean(0) - z_normal_samples.mean(0)

    # Cosine similarity (should be near 0 for orthogonal)
    cos_sim = np.dot(delta_eff.flatten(), delta_card.flatten()) / (
        np.linalg.norm(delta_eff) * np.linalg.norm(delta_card)
    )

    print(f"Disease axis cosine similarity: {cos_sim:.3f}")
    print(f"Orthogonality: {'GOOD' if abs(cos_sim) < 0.3 else 'POOR'}")
    return cos_sim

# === Coverage Test ===
def test_coverage(z_samples, n_bins=10):
    """Test manifold coverage via histogram occupancy."""
    z_flat = z_samples.reshape(len(z_samples), -1)

    # Normalize to unit cube
    z_min, z_max = z_flat.min(0), z_flat.max(0)
    z_norm = (z_flat - z_min) / (z_max - z_min + 1e-8)

    # Count occupied bins
    occupied = set()
    for z in z_norm:
        bin_idx = tuple((z * n_bins).astype(int).clip(0, n_bins-1))
        occupied.add(bin_idx)

    coverage = len(occupied) / (n_bins ** min(z_flat.shape[1], 4))  # Cap at 4D
    print(f"Manifold coverage: {coverage*100:.1f}%")
    print(f"Coverage: {'GOOD' if coverage > 0.6 else 'SPARSE'}")
    return coverage

# === Interpolation Smoothness Test ===
def test_interpolation_smoothness(z_start, z_end, decoder, n_steps=10):
    """Test if interpolation produces smooth transitions."""
    path = [z_start + t * (z_end - z_start)
            for t in np.linspace(0, 1, n_steps)]

    images = [decoder(z) for z in path]

    # Compute frame-to-frame differences
    diffs = [np.abs(images[i+1] - images[i]).mean()
             for i in range(len(images)-1)]

    # Check for spikes (indicates void crossing)
    mean_diff = np.mean(diffs)
    max_diff = np.max(diffs)
    spike_ratio = max_diff / (mean_diff + 1e-8)

    print(f"Interpolation spike ratio: {spike_ratio:.2f}")
    print(f"Smoothness: {'GOOD' if spike_ratio < 2.0 else 'VOID DETECTED'}")
    return spike_ratio, images

# === Additivity Test ===
def test_additivity(z_normal, z_eff, z_card, decoder):
    """Test if z_normal + delta_eff + delta_card decodes plausibly."""
    delta_eff = z_eff - z_normal
    delta_card = z_card - z_normal

    z_composed = z_normal + delta_eff + delta_card

    img_composed = decoder(z_composed)
    img_eff = decoder(z_eff)
    img_card = decoder(z_card)

    # Check that composed image has features of both
    # (Manual inspection recommended)
    return z_composed, img_composed
```

---

### 4.8 Summary: Manifold Requirements for Composition

| Requirement | Metric | Threshold | Why It Matters |
|-------------|--------|-----------|----------------|
| **Orthogonality** | cos(Δz_eff, Δz_card) | <0.3 | Ensures additive composition |
| **Coverage** | % bins occupied | >60% | Prevents void artifacts |
| **Smoothness** | Interpolation spike ratio | <2.0 | Validates continuous manifold |
| **Anchoring** | ‖z_normal - 0‖ | <0.5 | Enables origin-based arithmetic |
| **Flatness** | max/min det(G) | <10 | Euclidean approximation valid |

When all criteria pass, the SepVAE latent space is **compositionally ready** for multi-disease synthesis via LDM.

---

## 5. Quick Reference Tables

### 5.1 Loss Value Reference Card

| Loss Term | Early (1-10) | Mid (10-50) | Converged (50+) | Alert Threshold |
|-----------|--------------|-------------|-----------------|-----------------|
| `loss/total` | 0.5 - 2.0 | 0.1 - 0.5 | 0.05 - 0.15 | >0.3 at epoch 80 |
| `loss/reconstruction` | 0.1 - 0.2 | 0.02 - 0.05 | 0.01 - 0.02 | >0.05 at epoch 50 |
| `loss/kl_total` | 50 - 200 | 20 - 80 | 10 - 50 | <5 or >200 |
| `loss/kl_weighted` | 0.01 - 0.05 | 0.005 - 0.02 | 0.003 - 0.015 | <0.0001 |
| `loss/nulling` | 0.5 - 2.0 | 0.1 - 0.5 | <0.05 | >0.2 at epoch 40 |
| `loss/mi_penalty` | 0.5 - 2.0 | 0.3 - 0.8 | 0.2 - 0.6 | <0.1 or >1.5 |
| `loss/perceptual` | 0.1 - 0.5 | 0.05 - 0.2 | 0.03 - 0.1 | >0.3 at epoch 50 |
| `loss/gen_adversarial` | N/A (0) | 0.5 - 2.0 | 0.5 - 1.5 | >3.0 |
| `loss/patch_disc` | N/A (0) | 0.4 - 0.7 | 0.3 - 0.6 | <0.1 or >0.9 |

### 5.2 Hyperparameter Adjustment Guide

| Problem | Primary Adjustment | Secondary Adjustment |
|---------|-------------------|---------------------|
| Blurry reconstructions | ↓ `weight_kl_*` by 5x | ↑ `weight_rec` or ↑ `lr_vae` |
| Posterior collapse | ↑ `free_bits` to 1.0 | ↓ `weight_kl_*` by 10x |
| Checkerboard artifacts | Set `upsample_method=bilinear` | ↓ `weight_adversarial` |
| Disease leakage to normals | ↑ `weight_null` by 3x | ↓ `sigma_inactive` to 0.5 |
| Entangled latents | ↑ `weight_mi` by 3x | Check MI discriminator LR |
| Unstable GAN training | ↑ `disc_start_epoch` by 10 | ↓ `lr_patch_disc` by 2x |
| High variance in losses | ↓ `lr_vae` and `lr_disc` | ↑ `grad_clip` to 0.5 |
| Out of memory | ↓ `batch_size` by 2 | Enable `gradient_checkpointing` |

### 5.3 Training Checklist by Epoch

#### Epoch 5 Checkpoint
- [ ] Reconstruction loss < 0.15
- [ ] Backbone features loading correctly (check weight norms)
- [ ] All three disease classes represented in batches
- [ ] No NaN/Inf in any loss term

#### Epoch 20 Checkpoint
- [ ] Reconstruction loss < 0.05
- [ ] KL divergence in healthy range (10-100)
- [ ] Nulling loss < 0.3
- [ ] First reconstruction grid shows recognizable anatomy

#### Epoch 50 Checkpoint
- [ ] Reconstruction loss < 0.025
- [ ] KL stable, not collapsed
- [ ] Nulling loss < 0.1
- [ ] MI penalty in equilibrium range
- [ ] PCA shows separation in salient space

#### Final Evaluation
- [ ] Reconstruction loss ≈ 0.015
- [ ] No posterior collapse (KL > 5)
- [ ] Clean disease head routing (nulling < 0.05)
- [ ] Disentanglement verified via interpolation
- [ ] Reconstruction grids show faithful disease features
- [ ] Ready for LDM: smooth latent manifold, Gaussian-like distribution

---

## Appendix A: Command Reference

### Training Commands by GPU

```bash
# RTX 3090 (24GB) - Conservative
python -m run.train_sep_vae \
    --batch_size 4 \
    --gradient_checkpointing \
    --upsample_method bilinear \
    --weight_kl_common 1e-4 \
    --weight_kl_disease 1e-4 \
    --kl_warmup_epochs 10

# A100 (80GB) - Full Quality
python -m run.train_sep_vae \
    --batch_size 16 \
    --half_precision bf16 \
    --upsample_method subpixel \
    --use_fpn \
    --fpn_channels 512 \
    --weight_perceptual 0.1 \
    --weight_adversarial 0.1 \
    --disc_start_epoch 15 \
    --free_bits 0.5

# Resume from checkpoint
python -m run.train_sep_vae \
    --resume runs_sepvae/exp-name/checkpoints/checkpoint_epoch0050.pkl \
    --wandb \
    --wandb_run_id EXISTING_RUN_ID
```

### Diagnostic Commands

```bash
# Enable verbose backbone diagnostics
python -m run.train_sep_vae \
    --verbose_backbone \
    --verbose_n_samples 24 \
    --epochs 1  # Just for diagnostics

# Quick sanity check (few epochs, small batch)
python -m run.train_sep_vae \
    --batch_size 2 \
    --epochs 3 \
    --log_every 10 \
    --sample_every 1
```

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **CheSS** | Contrastive learning framework for chest X-ray self-supervision (frozen backbone) |
| **CTR** | Cardiothoracic ratio - cardiac width / thoracic width |
| **FPN** | Feature Pyramid Network - multi-scale feature aggregation |
| **Free-bits** | Per-channel KL floor to prevent posterior collapse |
| **MI Penalty** | Mutual information penalty for latent independence |
| **Nulling Loss** | Constraint forcing disease heads to prior for normal samples |
| **PatchGAN** | Multi-scale discriminator for local texture realism |
| **Posterior Collapse** | Pathology where encoder ignores latent, decoder reconstructs from nothing |
| **Salient Space** | Disease-specific latent subspace (z_disease) |
| **Spatial Latents** | 64×64 feature maps instead of flat vectors |

---

*Generated from analysis of `run/train_sep_vae.py` for the Multi-head SepVAE project.*
