#!/usr/bin/env bash
#
# Launch a fast LDM diagnostic run that OVERFITS on 16 PNEUMONIA samples.
#

set -euo pipefail

# ────────────────────────────────────────────────────────────────────────────────
# ENV / PATHS
# ────────────────────────────────────────────────────────────────────────────────
export ENV_NAME="jax115"

export AE_CKPT_PATH="runs/ae_pneumonia_full/path/to/your/ckpt.flax"
export AE_CONFIG_PATH="runs/ae_pneumonia_full/path/to/your/run_meta.json"

export DATA_ROOT="../datasets/cleaned"
export TASK="Pneumonia"
export SPLIT="train"
export CLASS_FILTER="1"
export IMG_SIZE="256"

# ────────────────────────────────────────────────────────────────────────────────
# LDM DIAGNOSTIC: overfit 16 images
# ────────────────────────────────────────────────────────────────────────────────
export OVERFIT_ONE="0"
export OVERFIT_K="16"
export REPEAT_LEN="200"

export LDM_CH_MULTS="1,2,4,4"
export LDM_BASE_CH="128"
export LDM_NUM_RES_BLOCKS="2"
export LDM_ATTN_RES="16,8"

export LATENT_SCALE_FACTOR="YOUR_PNEUMONIA_SCALE_FACTOR"

# ────────────────────────────────────────────────────────────────────────────────
# OPTIMIZER / SCHEDULING
# ────────────────────────────────────────────────────────────────────────────────
export LR="2e-4"
export WEIGHT_DECAY="0.0"
export GRAD_CLIP="1.0"

export EPOCHS="100"
export BATCH_PER_DEVICE="4"
export LOG_EVERY="25"

export SAMPLE_EVERY="5"
export SAMPLE_BATCH_SIZE="16"

# ────────────────────────────────────────────────────────────────────────────────
# W&B
# ────────────────────────────────────────────────────────────────────────────────
export USE_WANDB="0"
export WANDB_PROJECT="cxr-ldm"
export WANDB_TAGS="ldm:pneumonia:overfit-16,fast"

# ────────────────────────────────────────────────────────────────────────────────
# RUN NAMING
# ────────────────────────────────────────────────────────────────────────────────
export EXP_NAME="cxr_ldm"
export RUN_NAME="ldm_pneumonia_overfit_16_fast_$(date +%Y%m%d-%H%M%S)"

# ────────────────────────────────────────────────────────────────────────────────
# SUBMIT
# ────────────────────────────────────────────────────────────────────────────────
echo "Submitting SLURM job: LDM Pneumonia Overfit 16"
sbatch cxr_ldm.slurm