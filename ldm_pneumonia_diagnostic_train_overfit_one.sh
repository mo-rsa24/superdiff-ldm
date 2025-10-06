#!/usr/bin/env bash
#
# Launch a fast LDM diagnostic run that OVERFITS on ONE PNEUMONIA sample.
#

set -euo pipefail

# ────────────────────────────────────────────────────────────────────────────────
# ENV / PATHS
# ────────────────────────────────────────────────────────────────────────────────
export ENV_NAME="jax115"

# ❗ Point these at your confirmed AE run for the PNEUMONIA dataset ❗
export AE_CKPT_PATH="runs/ae_pneumonia_full/path/to/your/ckpt.flax"
export AE_CONFIG_PATH="runs/ae_pneumonia_full/path/to/your/run_meta.json"

# Your data root & task
export DATA_ROOT="../datasets/cleaned"
export TASK="Pneumonia"
export SPLIT="train"
export CLASS_FILTER="1"            # Assuming 1 = Pneumonia
export IMG_SIZE="256"

# ────────────────────────────────────────────────────────────────────────────────
# LDM DIAGNOSTIC: overfit ONE image
# ────────────────────────────────────────────────────────────────────────────────
export OVERFIT_ONE="1"
export OVERFIT_K="0"
export REPEAT_LEN="200"

# Model Architecture
export LDM_CH_MULTS="1,2,4,4"
export LDM_BASE_CH="128"
export LDM_NUM_RES_BLOCKS="2"
export LDM_ATTN_RES="16,8"

# You should calculate this for a single pneumonia image
export LATENT_SCALE_FACTOR="YOUR_PNEUMONIA_SCALE_FACTOR"

# ────────────────────────────────────────────────────────────────────────────────
# OPTIMIZER / SCHEDULING (tuned for one-image overfit)
# ────────────────────────────────────────────────────────────────────────────────
export LR="2e-4"
export WEIGHT_DECAY="0.0"
export GRAD_CLIP="1.0"

export EPOCHS="30"
export BATCH_PER_DEVICE="1"
export LOG_EVERY="25"

# Sampling every epoch for quick visual checks
export SAMPLE_EVERY="1"
export SAMPLE_BATCH_SIZE="4"

# ────────────────────────────────────────────────────────────────────────────────
# W&B
# ────────────────────────────────────────────────────────────────────────────────
export USE_WANDB="0"
export WANDB_PROJECT="cxr-ldm"
export WANDB_TAGS="ldm:pneumonia:overfit-one,fast"

# ────────────────────────────────────────────────────────────────────────────────
# RUN NAMING
# ────────────────────────────────────────────────────────────────────────────────
export EXP_NAME="cxr_ldm"
export RUN_NAME="ldm_pneumonia_overfit_one_fast_$(date +%Y%m%d-%H%M%S)"

# ────────────────────────────────────────────────────────────────────────────────
# SUBMIT
# ────────────────────────────────────────────────────────────────────────────────
echo "Submitting SLURM job: LDM Pneumonia Overfit ONE"
sbatch cxr_ldm.slurm