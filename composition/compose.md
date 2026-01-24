# SuperDiff Composition: Experiment Configurations & Interpretations

This document outlines the three primary sampler configurations for composing the **Normal** and **TB** diffusion models. It explains the purpose of each experiment, the exact arguments to run it, and how to interpret the resulting visual diagnostics.

---

## 1. The Baseline: Product of Experts (PoE)

**Hypothesis:** Simply adding the score vectors ($\nabla \log p(x) = \nabla \log p_A(x) + \nabla \log p_B(x)$) will force features from both models to appear, but may result in "burnt" images or unstable textures because the optimal mixing weight is unknown.

### Run Command
```bash
python compose.py \
    --run_dir_normal "runs_ldm/ldm-normal-composition-preencode-latents-ancestral-8b4bb7d-20260120-111058" \
    --run_dir_tb "runs_ldm/ldm-tb-composition-preencode-latents-ancestral-8b4bb7d-20260120-123239" \
    --output_path "results/superdiff_poe_result.png" \
    --steps 200 \
    --seed 42 \
    --batch_size 4 \
    --sample_images True \
    --sampler PoE

```

### Interpretation of Results

* **Log Trajectories:** Expect the log-likelihoods of Model A and Model B to **diverge** or slope independently. Since there is no "locking" mechanism (like ), the models drift apart as they fight for control.
* **PCA/t-SNE:** PoE samples will likely form a cluster that is **extreme** or far removed from both the Normal and TB clusters, potentially indicating "out of distribution" artifacts due to gradient summation.
* **Image Quality:** Look for high contrast, saturation artifacts, or "deep-fried" textures.

---

## 2. The Gold Standard: SuperDiff Faithful (DDPM Ancestral)

**Hypothesis:** Using the **Itô Density Estimator** to solve for the optimal mixing weight  will create a mathematically consistent "intersection" of the two distributions. The **DDPM Ancestral Sampler** will ensure textures remain sharp and realistic (correct variance).

### Run Command

```bash
python compose.py \
    --run_dir_normal "runs_ldm/ldm-normal-composition-preencode-latents-ancestral-8b4bb7d-20260120-111058" \
    --run_dir_tb "runs_ldm/ldm-tb-composition-preencode-latents-ancestral-8b4bb7d-20260120-123239" \
    --output_path "results/superdiff_faithful_result.png" \
    --steps 200 \
    --seed 42 \
    --batch_size 4 \
    --sample_images True \
    --sampler Ancestral \
    --lift 0.0

```

### Interpretation of Results

* **Log Trajectories:** Expect the lines for Model A and Model B to move **roughly parallel** to each other. This confirms that the algorithm is successfully satisfying the condition .
* **Kappa ():** Should oscillate around **0.5**. If it saturates at 0.0 or 1.0 constantly, the composition has failed (collapsed to one model).
* **PCA/t-SNE:** Samples should form a distinct cluster **between** or **overlapping** the Normal and TB clusters, representing a valid "hybrid" distribution.
* **Image Quality:** High fidelity, sharp details, with semantic features of TB (opacities) smoothly integrated into the Normal lung structure.

---

## 3. The Original: SuperDiff Stochastic (Euler)

**Hypothesis:** This reproduces the original paper's algorithm using an **Implicit Noise Estimator** and a standard **Euler-Maruyama SDE solver**. It is computationally faster per-step (no extra gradient passes) but may be noisier or "softer" due to the SDE discretization.

### Run Command

```bash
python compose.py \
    --run_dir_normal "runs_ldm/ldm-normal-composition-preencode-latents-ancestral-8b4bb7d-20260120-111058" \
    --run_dir_tb "runs_ldm/ldm-tb-composition-preencode-latents-ancestral-8b4bb7d-20260120-123239" \
    --output_path "results/superdiff_euler_result.png" \
    --steps 200 \
    --seed 42 \
    --batch_size 4 \
    --sample_images True \
    --sampler Euler \
    --lift 0.0

```

### Interpretation of Results

* **Log Trajectories:** We use a "Shadow Logger" here. The trajectories might be **noisier** or jagged compared to the Faithful version because the  calculation has higher variance.
* **Kappa ():** Expect more spikes or extreme values compared to the smooth Faithful version.
* **Image Quality:** Images may appear slightly **blurry** or "washed out" compared to DDPM Ancestral, as Euler steps often struggle to maintain perfect texture variance without many steps ().

---

## Comparative Summary

| Feature | **PoE** (Baseline) | **SuperDiff Faithful** (Recommended) | **SuperDiff Stochastic** (Legacy) |
| --- | --- | --- | --- |
| **Mixing Logic** | Simple Sum | Analytic Solve (Itô Estimator) | Implicit Noise Estimate |
| **Sampler** | DDPM Ancestral | DDPM Ancestral | Euler-Maruyama SDE |
| **Computational Cost** | Low | Medium (Dot Products) | Low |
| **Stability** | Low (Risk of Burn) | **High** (Optimal ) | Medium (Noisy ) |
| **Texture Quality** | Sharp / Artifacts | **Sharp / Realistic** | Soft / Blurry |

## How to Read the Plots

1. **`log_trajectories.png`**:
* **Goal:** Parallel lines.
* **Failure:** Crossing lines or exponential divergence.


2. **`kappa_trajectory.png`**:
* **Goal:** Wiggle around 0.5.
* **Failure:** Flatline at 2.0 (clipping) or 0.0/1.0 (mode collapse).


3. **`latent_pca.png` / `latent_tsne.png**`:
* **Goal:** Green points (Composition) sitting in the empty space between Blue (Normal) and Orange (TB).
* **Failure:** Green points completely covering the Orange cluster (means it just ignored the Normal model).

