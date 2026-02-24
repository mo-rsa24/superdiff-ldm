# Research Blueprint: Semantic vs Logical Composition in Latent Diffusion Models

> A conceptual blueprint of the research program, grounded in our experimental scripts,
> clarifying what we are testing, what constitutes evidence, and what the intellectual contribution is.

---

## 1. Core Research Objective

**Problem.** Compositional image generation currently operates under two paradigms — semantic composition (monolithic natural-language prompts) and logical composition (score-based operators such as SuperDiff AND) — but these are routinely treated as interchangeable alternatives rather than as structurally distinct mechanisms.

**Gap investigated.** We ask: do semantic AND and logical AND produce the same outcome in latent space? If not, what is the nature and magnitude of the gap, and can it be closed?

---

## 2. Central Hypothesis

**Semantic composition** (monolithic prompt `"c₁ and c₂"`) conflates the two concepts into a single conditioning vector; the model resolves the composition implicitly through attention, biased by training-corpus co-occurrence statistics.

**Logical composition** (SuperDiff AND: `v_c1 ∧ v_c2`) composes score functions directly, enforcing joint probability mass rather than a learned blend. This is a geometry-first operation, indifferent to semantic adjacency.

**Geometric suspicion.** From a common noise origin `x_T`, the two paradigms carve out distinct trajectories through the latent flow field, converging to different terminal basins. The monolithic trajectory is pulled toward a corpus-mode attractor; the SuperDiff trajectory navigates a region of genuinely higher joint density. The gap between them is therefore not noise — it is the signature of two different inductive biases operating in the same denoising manifold.

---

## 3. Key Claims

| Claim | Evidence required |
|---|---|
| Semantic AND ≠ logical AND | Non-trivial terminal latent MSE (`d_T_mono` >> 0) between monolithic and SuperDiff, replicated across pairs and seeds |
| SuperDiff occupies a distinct geometric regime | PCA/MDS trajectories diverge early and terminate in spatially separated regions; divergence is consistent across pairs |
| Solo conditioning is a weak proxy for composition | `d_T_c1` and `d_T_c2` are both large and similar to `d_T_mono`; composition is not reducible to either marginal |
| The gap is not symmetric across concept pairs | Per-pair distributions differ in mean and variance, reflecting pair-specific composability difficulty |
| An inverter can partially bridge the gap | `d_T_pstar` is measurably smaller than `d_T_mono`; JS² divergence between p* and AND distributions is lower than JS² between mono and AND |
| The two paradigms are complementary, not substitutes | A guided hybrid `(1−α)·v_mono + α·v_superdiff` outperforms either alone on perceptual metrics |

**Falsification conditions.** If `d_T_mono ≈ d_T_pstar ≈ 0` (all conditions converge to the same terminal), semantic and logical AND are equivalent and the inverter has nothing to teach. If trajectory projections overlap, the geometric-regime hypothesis fails.

---

## 4. Methodology (Script-Level Framing)

### `trajectory_dynamics_experiment.py` — Geometric evidence

Runs all conditioning strategies from a fixed `x_T`, records full latent trajectories, and projects them jointly via PCA/MDS. This is the foundational geometric experiment: it shows *how* the paths diverge, not just *that* the endpoints differ. The trajectory is the argument.

### `measure_composability_gap.py` — Quantitative gap measurement

Operationalises the gap as per-element MSE in raw latent space, computed per seed and per step, anchored to SuperDiff AND as the reference. Introduces `d_T_pstar` as the critical quantity: how close can a semantically-driven SD3.5 run (with inverted conditioning) get to the SuperDiff terminal? This script generates the raw numbers that plots 00–13 visualise.

### `plot_gap_analysis.py` — Statistical narrative

Transforms the raw records into a 14-plot statistical story: raw distributions (00–05) establish the gap is real and consistent; temporal stacks (06–10) localise *when* divergence accumulates during denoising; closing-the-gap plots (11–13) evaluate whether p* recovers the AND basin. Together they constitute a complete empirical argument.

### `train_inverter.py` — Semantic proxy learning

Trains `f_θ(image) → (pooled_embeds, seq_embeds)` so that an AND image can be re-expressed as an SD3.5 conditioning vector. This is a reverse-engineering experiment: if `f_θ` succeeds, then the AND image's semantic content can be captured by a standard prompt embedding, and we learn what that prompt would be. The quality of this capture, measured by `d_T_pstar`, tells us how much of the logical composition is semantically representable.

---

## 5. Paper Section Framing

### Two Distinct Composition Spaces

**Purpose.** Establish that semantic and logical AND are geometrically non-equivalent.

**Experiment.** PCA/MDS trajectory visualisation (`trajectory_dynamics_experiment.py`).

**Figure.** Time-coloured 2D trajectory map showing monolithic and SuperDiff paths diverging from a common `x_T`; terminal clusters well-separated in PC space. Include for multiple concept pairs to show consistency.

---

### The Composability Gap

**Purpose.** Quantify the discrepancy between composition paradigms across pairs and seeds.

**Experiment.** Terminal latent MSE and trajectory-resolved divergence (`measure_composability_gap.py`).

**Figure.** Plots 00/01 (strip + bar by pair) for the terminal gap; plots 08/10 (temporal detail) showing when in the denoising process the paths separate — the early-vs-late divergence structure has interpretive weight.

---

### Closing the Gap via Prompt Inversion

**Purpose.** Test whether the AND outcome is semantically representable, and by how much.

**Experiment.** Inverter training and evaluation (`train_inverter.py` + `measure_composability_gap.py`).

**Figure.** Plot 11 (p* strip alongside mono, c1, c2) — the key result plot. Plot 12 (CLIP cosine: AND↔p* vs AND↔mono) and plot 13 (JS² divergence) for distributional evidence.

---

### Guided Hybrid Composition

**Purpose.** Show that integrating semantic and logical signals outperforms either alone.

**Experiment.** `superdiff_guided` mode in `trajectory_dynamics_experiment.py`; α-sweep on perceptual metrics.

**Figure.** Trajectory map showing the guided path threading between the monolithic and SuperDiff terminals; CLIP/LPIPS vs α curve showing a non-trivial optimum.

---

## 6. Expected Results

### Confirmatory pattern

- `d_T_mono >> d_T_pstar > 0`: the inverter reduces but does not eliminate the gap, indicating partial semantic representability of logical composition.
- Trajectory PCA shows early divergence (steps 0–20) as the dominant contributor to terminal gap, with late-stage trajectories running roughly parallel — the inductive bias is injected early.
- JS²(P_p\*, P_mono) < JS²(P_AND, P_mono): p* is a better proxy for AND than the monolithic prompt is.
- Guided hybrid with α ≈ 0.3–0.5 achieves best CLIP similarity to AND on held-out pairs.

### Falsifying pattern

- `d_T_pstar ≈ d_T_mono`: inverter gains nothing → logical composition is not semantically decomposable; prompts cannot recover AND.
- Late-stage divergence dominates: composition is happening in fine-detail, not structural, formation — different implications for where interventions should be applied.
- No pair-to-pair variation in gap magnitude: suggests a universal failure mode rather than a pair-difficulty effect (undermines the geometric-regime story).

---

## 7. Interpretation & Implications

**Latent geometry.** SuperDiff AND defines a constraint surface in latent space — a locus of points with non-negligible probability under both marginal scores. Monolithic prompts define a different attractor shaped by token co-occurrence in the training distribution. The gap is the Hausdorff distance between these surfaces, and it varies by pair as a function of semantic proximity between concepts.

**Inductive bias in SD3.5.** SD3.5's text conditioning encodes *pragmatic* meaning (how concepts co-occur in captioned images) rather than *logical* meaning (what it means for both to be simultaneously present). A monolithic prompt `"a cat and a dog"` does not instruct the model to maximise joint likelihood; it provides a single point in embedding space that the model maps to its corpus mode for that phrase. SuperDiff bypasses this by operating directly on the score field.

**Limits of pure logical composition.** SuperDiff AND is formally correct but perceptually impoverished when concepts are semantically distant: the joint density basin it finds may be low-density under the training distribution, producing technically-compositional but unnaturally-rendered images. The semantic component of a monolithic prompt provides the distributional prior that anchors the image to the learned manifold.

---

## 8. Recommendation

> **Logical (score-based) composition and semantic (prompt-based) composition operate in geometrically distinct but functionally complementary regimes within the latent flow field of large-scale text-to-image models. Logical composition provides formal correctness — enforcing joint density constraints independently of training corpus statistics — while semantic composition provides distributional plausibility, anchoring the output to the learned image manifold. Practical compositional generation systems should integrate both: using score-based AND to enforce the logical constraint and prompt-based conditioning to regularise against low-density regions. We demonstrate that these regimes are empirically separable, quantify the gap between them, and show that a learned semantic proxy can partially but not fully recover the logical composition outcome — establishing the gap as an irreducible consequence of the fundamental difference between probability-matching and score-field geometry.**

The concise version for an abstract or conclusion:

> Semantic and logical AND are not interchangeable: they occupy distinct geometric regimes in the latent flow field, produce measurably different terminal distributions, and encode different inductive biases. Bridging this gap requires hybridising both paradigms, not substituting one for the other.

---

## 9. Framing Note: The Inverter as a Semantic Representability Probe

The inverter experiment (`train_inverter.py` + p* evaluation) is the strongest differentiator of this work. Most compositionality papers either benchmark logical operators or train semantic models — we use one as a *probe of the other*. The fact that `f_θ` can partially but not fully recover AND in semantic space is itself the theoretical contribution: it operationalises the gap as a **semantic representability horizon**.

The paper should frame p* not as "can we cheat SuperDiff with a prompt" but as: *how much of logical composition is semantically articulable, and what remains beyond that horizon?* The residual gap — the portion of AND that no prompt embedding can recover — is the direct empirical signature of the fundamental distinction between these two composition regimes.
