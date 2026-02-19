# ECCV 2026 Experiment Structure

All commands use:

```bash
ROOT=experiments/eccv2026
SCRIPT=scripts/trajectory_dynamics_experiment.py
```

## Directory Layout

```
experiments/eccv2026/
├── failures/                        # Sec 3: SD3.5 compositional failure taxonomy
│   └── sd35_compositional/
│       ├── solo_monolithic/         # Baseline: SD3.5 standard CFG
│       └── decomposed_pairwise_rescue/  # Can decomposition help?
│
├── trajectory_dynamics/             # Sec 4: Latent trajectory analysis
│   ├── verification/                # SD3.5 SuperDIFF AND implementation validation
│   ├── clip_vs_superdiff/           # CLIP AND vs SuperDIFF AND divergence
│   ├── spatial/                     # Spatial conditioning trajectories
│   └── multi_prompt_dynamics/       # 3+ prompt logical composition
│
├── guidance/                        # Sec 5: SuperDIFF-guided composition
│   ├── superdiff_guided/
│   │   └── alpha_sweep/             # α ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 1.0}
│   ├── negation/
│   │   └── composable_vs_superdiff_not/  # Eq 13 vs dynamic κ_NOT
│   └── clip_to_superdiff_geometry/  # How α reshapes the latent manifold
│
└── ablations/                       # Sec 6 / Appendix: Sensitivity analysis
    ├── lift/                        # SuperDIFF lift parameter ℓ
    ├── guidance_scale/              # CFG scale gs
    ├── step_schedule/               # Num inference steps T
    ├── seed_robustness/             # Multiple seeds for statistical stability
    ├── projection/                  # PCA vs MDS comparison
    └── superdiff_variant/           # ours vs author_det vs author_stoch vs fm_ode
```

## Naming Convention

```
{prompt_slug}__{param1}__{param2}__...{paramN}
```

| Token | Meaning | Example |
|-------|---------|---------|
| `pair_X_Y` | 2-prompt mode | `pair_dog_cat` |
| `multi_X_Y_Z` | 3+ prompt mode | `multi_portrait_oil_lighting` |
| `solo_X` | Solo mode | `solo_woman_hat` |
| `g{X}p{Y}` | guidance_scale | `g4p5` = 4.5 |
| `t{N}` | steps | `t50` |
| `s{NNNN}` | seed | `s0042` |
| `l{X}p{Y}` | lift | `l0p1` = 0.1 |
| `a{vals}` | alpha(s) | `a0p1-0p3-0p5` |
| `var-{X}` | superdiff variant | `var-fm_ode` |
| `ns{X}p{Y}` | neg_scale | `ns1p0` |
| `nl{X}p{Y}` | neg_lambda | `nl1p0` |
| `proj-{X}` | projection | `proj-pca` |

---

## Prompt Taxonomy (Specific)

Use this table to classify prompt sets before choosing a run block.

| Category | Subtype | Prompt Pattern | Example Prompt(s) | Primary Failure Signal | Recommended Block |
|---------|---------|----------------|-------------------|------------------------|-------------------|
| Co-presence baseline | Simple noun conjunction | `A and B` | `a dog and a cat` | Semantic-vs-logical trajectory gap | `trajectory_dynamics/clip_vs_superdiff/` |
| Attribute binding (single) | Object-color/part binding | `attr1 obj1 with attr2 obj2` | `a red car next to a blue truck` | Attribute swap/leakage | `failures/solo_monolithic/` |
| Attribute binding (cross-object) | Multi-object scene binding | `attr obj in relation to attr obj` | `a brown bench in front of a white building` | Cross-object attribute bleed | `failures/solo_monolithic/`, `guidance/superdiff_guided/alpha_sweep/` |
| Spatial relation (binary) | Left/right grounding | `A on the left, B on the right` | `a book on the left and a bird on the right` | Positional collapse/inversion | `trajectory_dynamics/spatial/`, `guidance/superdiff_guided/alpha_sweep/` |
| Spatial relation (relative) | Front/behind/near | `A in front of/behind B` | `a bench in front of a building` | Relative-depth confusion | `trajectory_dynamics/clip_vs_superdiff/` |
| Counting/combinatorics | Cardinality constraints | `exactly n A and m B` | `exactly three red apples` / `three dogs and four cats` | Count mismatch/crowding | `failures/solo_monolithic/`, `failures/decomposed_pairwise_rescue/` |
| Negation | Explicit suppression | `A without B` or `A NOT B` | `a woman wearing a hat without glasses` | Negated concept persists | `guidance/negation/composable_vs_superdiff_not/` |
| Logic trap | Conjunction + negation | `A with B but without C` | `a cake with strawberries but without chocolate` | Contradictory concept mixing | `failures/solo_monolithic/`, `guidance/negation/` |
| Multi-constraint composition | 3+ prompt factorization | `[co-presence], [constraint1], [constraint2]` | `"a cat and a dog" + "cat on the left" + "dog on the right"` | Constraint interference between clauses | `trajectory_dynamics/multi_prompt_dynamics/` |
| Guided bridge probes | Hybrid semantic-logical interpolation | same base prompts with alpha sweep | `two blue apples` + `a red table` with `--guided --alpha ...` | Best alpha trade-off point | `guidance/superdiff_guided/alpha_sweep/`, `guidance/clip_to_superdiff_geometry/` |

---

## Run Commands

### 1. failures/sd35_compositional/

**Hypothesis**: SD3.5 fails at attribute binding, spatial relations, negation, and counting.

These runs intentionally stress known failure modes using `--solo` monolithic prompting.

#### 1.1 Solo monolithic failures

```bash
# Attribute binding — test if SD3.5 preserves color-object bindings
python $SCRIPT \
  --solo --monolithic "a red car parked next to a blue truck" \
  --no-clip-probe --steps 50 --guidance 4.5 --seed 42 \
  --output-dir $ROOT/failures/sd35_compositional/solo_monolithic/red_car_blue_truck__g4p5__t50__s0042

# Spatial relations — test directional grounding ("left of")
python $SCRIPT \
  --solo --monolithic "a book on the left of a bird" \
  --no-clip-probe --steps 50 --guidance 4.5 --seed 42 \
  --output-dir $ROOT/failures/sd35_compositional/solo_monolithic/book_left_bird__g4p5__t50__s0042

# Negation — test "without" suppression
python $SCRIPT \
  --solo --monolithic "a woman without glasses wearing a hat" \
  --no-clip-probe --steps 50 --guidance 4.5 --seed 42 \
  --output-dir $ROOT/failures/sd35_compositional/solo_monolithic/woman_no_glasses_hat__g4p5__t50__s0042

# Counting — test strict object multiplicity
python $SCRIPT \
  --solo --monolithic "exactly three red apples on a wooden table" \
  --no-clip-probe --steps 50 --guidance 4.5 --seed 42 \
  --output-dir $ROOT/failures/sd35_compositional/solo_monolithic/three_red_apples__g4p5__t50__s0042

# Multi-attribute binding — test cross-object attribute leakage
python $SCRIPT \
  --solo --monolithic "a brown bench in front of a white building" \
  --no-clip-probe --steps 50 --guidance 4.5 --seed 42 \
  --output-dir $ROOT/failures/sd35_compositional/solo_monolithic/brown_bench_white_building__g4p5__t50__s0042

# Logic trap — test conjunction + negation in one prompt
python $SCRIPT \
  --solo --monolithic "a cake with strawberries but without chocolate" \
  --no-clip-probe --steps 50 --guidance 4.5 --seed 42 \
  --output-dir $ROOT/failures/sd35_compositional/solo_monolithic/cake_strawberry_no_chocolate__g4p5__t50__s0042
```

#### 1.2 Decomposed pairwise rescue

Compare monolithic failure vs decomposed pairwise composition on the same semantic target.

```bash
python $SCRIPT \
  --prompt-a "three dogs" --prompt-b "four cats, no overlaps" \
  --monolithic "Three dogs sitting next to four cats, no overlaps" \
  --superdiff-variant fm_ode --no-poe \
  --steps 50 --guidance 4.5 --seed 42 \
  --output-dir $ROOT/failures/sd35_compositional/decomposed_pairwise_rescue/three_dogs_four_cats__g4p5__t50__s0042
```

---

### 2. trajectory_dynamics/

These runs establish that CLIP AND and SuperDiff AND are different dynamical systems under shared noise.

#### 2.1 Implementation verification (`superdiff_variant=all`)

Sanity-check behavior of all SD3 SuperDiff variants on SD3.5 (`ours`, `author_det`, `fm_ode`).

```bash
python $SCRIPT \
  --prompt-a "a dog" --prompt-b "a cat" \
  --superdiff-variant all --no-poe \
  --steps 50 --guidance 4.5 --seed 42 \
  --output-dir $ROOT/trajectory_dynamics/verification/sd35_superdiff_and_impl/pair_dog_cat__var-all__g4p5__t50__s0042
```

#### 2.2 CLIP AND vs SuperDiff AND

Measure trajectory divergence and endpoint geometry.

```bash
# Standard pair
python $SCRIPT \
  --prompt-a "a dog" --prompt-b "a cat" \
  --monolithic "a dog and a cat" \
  --superdiff-variant fm_ode --no-poe --uniform-color \
  --steps 50 --guidance 4.5 --seed 42 \
  --output-dir $ROOT/trajectory_dynamics/clip_vs_superdiff/pair_dog_cat__g4p5__t50__s0042

# Spatial pair — test divergence on relational prompts
python $SCRIPT \
  --prompt-a "cat on the left" --prompt-b "dog on the right" \
  --monolithic "cat on the left and dog on the right" \
  --superdiff-variant fm_ode --no-poe --uniform-color \
  --steps 50 --guidance 4.5 --seed 42 \
  --output-dir $ROOT/trajectory_dynamics/clip_vs_superdiff/catleft_dogright__g4p5__t50__s0042

# Attribute binding pair
python $SCRIPT \
  --prompts "two blue apples" "a red table" \
  --steps 50 --guidance 4.5 --seed 42 --uniform-color \
  --output-dir $ROOT/trajectory_dynamics/clip_vs_superdiff/blue_apples_red_table__g4p5__t50__s0042

# MDS variant (pairwise distance preservation)
python $SCRIPT \
  --prompt-a "a dog" --prompt-b "a cat" \
  --projection mds --steps 50 --guidance 4.5 --seed 42 --uniform-color \
  --output-dir $ROOT/trajectory_dynamics/clip_vs_superdiff/pair_dog_cat__proj-mds__g4p5__t50__s0042
```

#### 2.3 Spatial extension block (`--spatial`)

Generate both base trajectories and spatial extension diagnostics in one run.

```bash
python $SCRIPT \
  --prompt-a "a cat on the left" --prompt-b "a dog on the right" \
  --monolithic "a cat on the left and a dog on the right" \
  --spatial \
  --spatial-a "a cat on the left" \
  --spatial-b "a dog on the right" \
  --spatial-mono "a cat on the left and a dog on the right" \
  --superdiff-variant fm_ode --no-poe \
  --steps 50 --guidance 4.5 --seed 42 \
  --output-dir $ROOT/trajectory_dynamics/spatial/catleft_dogright__g4p5__t50__s0042
```

#### 2.4 Multi-prompt trajectory dynamics (3 prompts)

Compare monolithic conjunction vs multi-AND composition with semantic anchoring prompt.

```bash
python $SCRIPT \
  --prompts "a cat and a dog" "cat on the left" "dog on the right" \
  --monolithic "a cat and a dog and cat on the left and dog on the right" \
  --steps 50 --guidance 4.5 --seed 42 \
  --output-dir $ROOT/trajectory_dynamics/multi_prompt_dynamics/catdog_spatial__g4p5__t50__s0042
```

---

### 3. guidance/

These runs study how guidance controls the trade-off between linguistic coherence and logical balancing.

#### 3.1 Alpha sweep — SuperDiff Guided

**Hypothesis**: SuperDIFF-Guided hybrid (1-α)·v_mono + α·v_sd interpolates between SD3.5 and SuperDIFF.

```bash
# Attribute binding rescue
python $SCRIPT \
  --prompts "two blue apples" "a red table" \
  --monolithic "two blue apples and a red table" \
  --guided --alpha 0.0 0.1 0.3 0.5 0.7 1.0 \
  --superdiff-variant fm_ode --no-poe --uniform-color \
  --steps 50 --guidance 4.5 --seed 42 \
  --output-dir $ROOT/guidance/superdiff_guided/alpha_sweep/blue_apples_red_table__a0p0-0p1-0p3-0p5-0p7-1p0__g4p5__t50__s0042

# Spatial rescue
python $SCRIPT \
  --prompts "a book on the left" "a bird on the right" \
  --monolithic "a book on the left and a bird on the right" \
  --guided --alpha 0.0 0.1 0.3 0.5 0.7 1.0 \
  --superdiff-variant fm_ode --no-poe --uniform-color \
  --steps 50 --guidance 4.5 --seed 42 \
  --output-dir $ROOT/guidance/superdiff_guided/alpha_sweep/book_bird__a0p0-0p1-0p3-0p5-0p7-1p0__g4p5__t50__s0042

# Multi-attribute rescue
python $SCRIPT \
  --prompts "a brown bench" "a white building" \
  --monolithic "a brown bench in front of a white building" \
  --guided --alpha 0.0 0.1 0.3 0.5 0.7 1.0 \
  --superdiff-variant fm_ode --no-poe --uniform-color \
  --steps 50 --guidance 4.5 --seed 42 \
  --output-dir $ROOT/guidance/superdiff_guided/alpha_sweep/brown_bench_white_building__a0p0-0p1-0p3-0p5-0p7-1p0__g4p5__t50__s0042
```

#### 3.2 CLIP-to-SuperDiff geometry probe

**Hypothesis**: Increasing α bends the latent manifold from CLIP AND's trajectory toward SuperDIFF AND's.

```bash
python $SCRIPT \
  --prompts "a cat on the left" "a dog on the right" \
  --monolithic "a cat on the left and a dog on the right" \
  --guided --alpha 0.1 0.3 0.5 0.7 \
  --no-poe --uniform-color \
  --steps 50 --guidance 4.5 --seed 42 \
  --output-dir $ROOT/guidance/clip_to_superdiff_geometry/catdog_spatial__a0p1-0p3-0p5-0p7__g4p5__t50__s0042
```

#### 3.3 Negation: Composable NOT vs SuperDiff NOT

**Hypothesis**: SuperDIFF NOT (dynamic κ) outperforms Composable NOT (Eq 13, fixed weight) at suppressing negated concepts.

```bash
# Negation showcase: hat without glasses
python $SCRIPT \
  --prompt-a "a woman wearing a hat" --neg-prompt "glasses" \
  --solo --no-poe --steps 50 --guidance 4.5 --seed 42 \
  --output-dir $ROOT/guidance/negation/composable_vs_superdiff_not/hat_not_glasses__ns1p0__nl1p0__g4p5__t50__s0042

# Logic trap: cake without chocolate
python $SCRIPT \
  --prompt-a "a cake with strawberries" --neg-prompt "chocolate" \
  --solo --no-poe --steps 50 --guidance 4.5 --seed 42 \
  --output-dir $ROOT/guidance/negation/composable_vs_superdiff_not/cake_not_chocolate__ns1p0__nl1p0__g4p5__t50__s0042

# Lambda sweep (suppression strength)
for lam in 0.5 1.0 2.0 3.0; do
  slug=$(echo $lam | tr '.' 'p')
  python $SCRIPT \
    --prompt-a "a woman wearing a hat" --neg-prompt "glasses" \
    --neg-lambda $lam --solo --no-poe \
    --steps 50 --guidance 4.5 --seed 42 \
    --output-dir $ROOT/guidance/negation/composable_vs_superdiff_not/hat_not_glasses__ns1p0__nl${slug}__g4p5__t50__s0042
done
```

---

### 4. ablations/

**Parameter sweeps** — one factor changes at a time; all other controls fixed.

#### 4.1 Guidance scale sweep

Test sensitivity to classifier-free guidance strength.

```bash
for gs in 3.0 4.5 6.0 7.5; do
  slug=$(echo $gs | tr '.' 'p')
  python $SCRIPT \
    --prompt-a "a dog" --prompt-b "a cat" \
    --superdiff-variant fm_ode --no-poe \
    --steps 50 --guidance $gs --seed 42 \
    --output-dir $ROOT/ablations/guidance_scale/pair_dog_cat__g${slug}__t50__s0042
done
```

#### 4.2 Step schedule sweep

Inspect whether divergence and composition quality are schedule-dependent.

```bash
for steps in 30 40 50 75 100; do
  python $SCRIPT \
    --prompt-a "a dog" --prompt-b "a cat" \
    --superdiff-variant fm_ode --no-poe \
    --steps $steps --guidance 4.5 --seed 42 \
    --output-dir $ROOT/ablations/step_schedule/pair_dog_cat__g4p5__t${steps}__s0042
done
```

#### 4.3 Lift sweep

Test the effect of SuperDiff lift regularization on trajectory behavior.

```bash
for lift in 0.0 0.1 0.3 0.5; do
  slug=$(echo $lift | tr '.' 'p')
  python $SCRIPT \
    --prompt-a "a dog" --prompt-b "a cat" \
    --superdiff-variant fm_ode --no-poe \
    --lift $lift --steps 50 --guidance 4.5 --seed 42 \
    --output-dir $ROOT/ablations/lift/pair_dog_cat__l${slug}__g4p5__t50__s0042
done
```

#### 4.4 Projection sweep

Verify geometric conclusions are stable across PCA and MDS views.

```bash
for proj in pca mds; do
  python $SCRIPT \
    --prompt-a "a dog" --prompt-b "a cat" \
    --superdiff-variant fm_ode --no-poe \
    --projection $proj --steps 50 --guidance 4.5 --seed 42 \
    --output-dir $ROOT/ablations/projection/pair_dog_cat__proj-${proj}__g4p5__t50__s0042
done
```

#### 4.5 Seed robustness

Estimate whether observed divergence trends hold across independent seeds.

```bash
for seed in 42 43 44 45 46; do
  python $SCRIPT \
    --prompt-a "a dog" --prompt-b "a cat" \
    --superdiff-variant fm_ode --no-poe \
    --steps 50 --guidance 4.5 --seed $seed \
    --output-dir $ROOT/ablations/seed_robustness/pair_dog_cat__g4p5__t50__s$(printf "%04d" $seed)
done
```

#### 4.6 SuperDiff variant sweep

Compare `ours`, `author_det`, `author_stoch`, and `fm_ode` under the same prompt pair.
For `author_stoch`, baseline conditions (`prompt_a`, `prompt_b`, monolithic, PoE) use SD3.5,
while only the `SuperDIFF (author, stoch)` condition uses the notebook-compatible backend
(`CompVis/stable-diffusion-v1-4`).

```bash
for variant in ours author_det author_stoch fm_ode; do
  python $SCRIPT \
    --prompt-a "a dog" --prompt-b "a cat" \
    --superdiff-variant $variant --no-poe \
    --steps 50 --guidance 4.5 --seed 42 \
    --output-dir $ROOT/ablations/superdiff_variant/pair_dog_cat__var-${variant}__g4p5__t50__s0042
done
```

---

## Flag → Experiment Category Mapping

| Flag(s) | Category | Subdirectory |
|---------|----------|-------------|
| `--solo --monolithic` | Failure taxonomy | `failures/sd35_compositional/solo_monolithic/` |
| `--superdiff-variant all` | Implementation verification | `trajectory_dynamics/verification/` |
| `--prompt-a --prompt-b --no-poe` | CLIP vs SuperDIFF comparison | `trajectory_dynamics/clip_vs_superdiff/` |
| `--spatial` | Spatial trajectory analysis | `trajectory_dynamics/spatial/` |
| `--prompts` (3+) | Multi-prompt dynamics | `trajectory_dynamics/multi_prompt_dynamics/` |
| `--guided --alpha` | Guided alpha sweep | `guidance/superdiff_guided/alpha_sweep/` |
| `--guided --alpha` + geometry focus | Manifold geometry | `guidance/clip_to_superdiff_geometry/` |
| `--neg-prompt` | Negation experiments | `guidance/negation/` |
| `--lift` sweep | Lift ablation | `ablations/lift/` |
| `--guidance` sweep | CFG scale ablation | `ablations/guidance_scale/` |
| `--steps` sweep | Step schedule ablation | `ablations/step_schedule/` |
| seed loop | Seed robustness | `ablations/seed_robustness/` |
| `--projection pca\|mds` | Projection comparison | `ablations/projection/` |
| `--superdiff-variant` sweep | Variant comparison | `ablations/superdiff_variant/` |

## Paper Section → Directory Mapping

| Section | Experiment Directory | Key Figures |
|---------|---------------------|-------------|
| Fig 1 (teaser) | `failures/solo_monolithic/` + `guidance/alpha_sweep/` | SD3.5 fail → guided fix |
| Sec 3 (Problem) | `failures/sd35_compositional/` | Failure taxonomy grid |
| Sec 4 (Method) | `trajectory_dynamics/verification/` | Kappa evolution, trajectory PCA |
| Sec 5.1 (AND) | `trajectory_dynamics/clip_vs_superdiff/` | Manifold divergence |
| Sec 5.2 (Guided) | `guidance/superdiff_guided/alpha_sweep/` | α sweep manifold |
| Sec 5.3 (NOT) | `guidance/negation/` | Negation comparison |
| Sec 6 (Ablations) | `ablations/` | Sensitivity grids |
| Appendix | `ablations/seed_robustness/`, `ablations/projection/`, `ablations/superdiff_variant/` | Statistical stability |

---

## Reproducibility Checklist

1. Always set `--output-dir` explicitly.
2. Keep `--seed`, `--steps`, and `--guidance` fixed across method comparisons.
3. Use `--no-poe` when isolating CLIP AND vs SuperDiff AND only.
4. Use `--uniform-color` for cleaner side-by-side manifold figure interpretation.
5. Store generated `summary.json` with figures for each run directory.
