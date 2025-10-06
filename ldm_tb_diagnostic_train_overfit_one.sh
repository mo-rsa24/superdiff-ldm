#!/usr/bin/env bash
#
# Launch a fast LDM diagnostic run that OVERFITS on ONE sample.
# - Uses the AE trained on the full dataset (point AE_* vars below)
# - Samples EVERY EPOCH with a TINY sample batch for quick visual feedback
# - Logs frequently so you can see if ε̂ std wakes up
#
# Tip for speed while debugging:
#   - Keep SAMPLE_BATCH_SIZE small (e.g., 2–4)
#   - Keep SAMPLE_EVERY=1 (so you see progress each epoch)
#   - If you also reduced n_steps inside Euler_Maruyama_sampler to ~100–150,
#     sampling will be MUCH faster per epoch.

set -euo pipefail

# ────────────────────────────────────────────────────────────────────────────────
# ENV / PATHS
# ────────────────────────────────────────────────────────────────────────────────
export ENV_NAME="jax115"

# ❗ Point these at your confirmed AE run (full-dataset VAE) ❗
export AE_CKPT_PATH="runs/ae_tb_full_kl_1.0e-5_zchannels_3/20251003-125631/ckpts/last.flax"
export AE_CONFIG_PATH="runs/ae_tb_full_kl_1.0e-5_zchannels_3/20251003-125631/run_meta.json"

# Your data root & task

export DATA_ROOT="/datasets/mmolefe/cleaned"
export TASK="TB"
export SPLIT="train"
export CLASS_FILTER="1"            # 1 = diseased, 0 = normal (keeps it consistent)
export IMG_SIZE="256"              # matches AE training

# ────────────────────────────────────────────────────────────────────────────────
# LDM DIAGNOSTIC: overfit ONE image; fastest possible feedback
# ────────────────────────────────────────────────────────────────────────────────
export OVERFIT_ONE="1"             # enable fixed-latent path in ldm.py
export OVERFIT_K="0"
export REPEAT_LEN="200"            # virtual length for the repeated single sample

# IMPORTANT: use COMMAS (ldm.py splits on ',')
export LDM_CH_MULTS="1,2,4,4"
export LDM_BASE_CH="128"
export LDM_NUM_RES_BLOCKS="2"
export LDM_ATTN_RES="16,8"

# Scale factor you measured for the overfit image
export LATENT_SCALE_FACTOR="2.051733"

# ────────────────────────────────────────────────────────────────────────────────
# OPTIMIZER / SCHEDULING (tuned for one-image overfit)
# ────────────────────────────────────────────────────────────────────────────────
export LR="2e-4"                   # higher LR so ε̂ starts moving fast
export WEIGHT_DECAY="0.0"          # no WD on a single-sample fit
export GRAD_CLIP="1.0"

export EPOCHS="30"                 # short sanity run
export BATCH_PER_DEVICE="1"        # keep tiny per device
export LOG_EVERY="25"              # frequent feedback in logs

# Sampling every epoch with a tiny sample batch ⇒ QUICK visual checks
export SAMPLE_EVERY="1"            # sample at the end of EVERY epoch
export SAMPLE_BATCH_SIZE="4"       # tiny grid for speed (2–4 is good)

# If you edited Euler_Maruyama_sampler default n_steps to ~100–150,
# sampling per epoch will be much faster. (No CLI flag for this in ldm.py.)

# ────────────────────────────────────────────────────────────────────────────────
# W&B (off by default for speed; enable if you want remote logs)
# ────────────────────────────────────────────────────────────────────────────────
export USE_WANDB="0"
export WANDB_PROJECT="cxr-ldm"
export WANDB_TAGS="ldm:tb:overfit-one,fast"

# ────────────────────────────────────────────────────────────────────────────────
# RUN NAMING
# ────────────────────────────────────────────────────────────────────────────────
export EXP_NAME="cxr_ldm"
export RUN_NAME="ldm_tb_overfit_one_fast_$(date +%Y%m%d-%H%M%S)"

# ────────────────────────────────────────────────────────────────────────────────
# QUALITY-OF-LIFE: JAX/XLA env that often helps interactivity on clusters
# (safe to keep; doesn't change training math)
# ────────────────────────────────────────────────────────────────────────────────
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ────────────────────────────────────────────────────────────────────────────────
# SUBMIT
# The SLURM script reads these env vars and constructs the CLI for run/ldm.py
# ────────────────────────────────────────────────────────────────────────────────
echo "Submitting SLURM job: LDM Overfit ONE (fast feedback)"
echo "────────────────────────────────────────────────────"
printf "  AE_CKPT_PATH         : %s\n" "$AE_CKPT_PATH"
printf "  AE_CONFIG_PATH       : %s\n" "$AE_CONFIG_PATH"
printf "  LDM_CH_MULTS         : %s\n" "$LDM_CH_MULTS"
printf "  LDM_ATTN_RES         : %s\n" "$LDM_ATTN_RES"
printf "  LR / WD              : %s / %s\n" "$LR" "$WEIGHT_DECAY"
printf "  LOG_EVERY            : %s steps\n" "$LOG_EVERY"
printf "  SAMPLE_EVERY         : every %s epoch(s)\n" "$SAMPLE_EVERY"
printf "  SAMPLE_BATCH_SIZE    : %s\n" "$SAMPLE_BATCH_SIZE"
printf "  RUN_NAME             : %s\n" "$RUN_NAME"
echo "────────────────────────────────────────────────────"
sbatch cxr_ldm.slurm
