#!/usr/bin/env bash
#
# Launch a fast LDM diagnostic run that OVERFITS on 32 samples.
#

set -euo pipefail

# ────────────────────────────────────────────────────────────────────────────────
# ENV / PATHS
# ────────────────────────────────────────────────────────────────────────────────
export ENV_NAME="jax115"

export AE_CKPT_PATH="runs/ae_tb_full_kl_1.0e-5_zchannels_3/20251003-125631/ckpts/last.flax"
export AE_CONFIG_PATH="runs/ae_tb_full_kl_1.0e-5_zchannels_3/20251003-125631/run_meta.json"

export DATA_ROOT="/datasets/mmolefe/cleaned"
export TASK="TB"
export SPLIT="train"
export CLASS_FILTER="1"
export IMG_SIZE="256"

# ────────────────────────────────────────────────────────────────────────────────
# LDM DIAGNOSTIC: overfit 32 images
# ────────────────────────────────────────────────────────────────────────────────
export OVERFIT_ONE="0"             # disable single-image overfit
export OVERFIT_K="32"              # overfit on the first 32 samples
export REPEAT_LEN="200"

export LDM_CH_MULTS="1,2,4,4"
export LDM_BASE_CH="128"
export LDM_NUM_RES_BLOCKS="2"
export LDM_ATTN_RES="16,8"

export LATENT_SCALE_FACTOR="2.051733" # Use the same factor

# ────────────────────────────────────────────────────────────────────────────────
# OPTIMIZER / SCHEDULING
# ────────────────────────────────────────────────────────────────────────────────
export LR="2e-4"
export WEIGHT_DECAY="0.0"
export GRAD_CLIP="1.0"

export EPOCHS="200"                # a bit longer again
export BATCH_PER_DEVICE="8"
export LOG_EVERY="25"

export SAMPLE_EVERY="5"
export SAMPLE_BATCH_SIZE="16"      # sample a grid of 16 out of the 32

# ────────────────────────────────────────────────────────────────────────────────
# W&B
# ────────────────────────────────────────────────────────────────────────────────
export USE_WANDB="0"
export WANDB_PROJECT="cxr-ldm"
export WANDB_TAGS="ldm:tb:overfit-32,fast"

# ────────────────────────────────────────────────────────────────────────────────
# RUN NAMING
# ────────────────────────────────────────────────────────────────────────────────
export EXP_NAME="cxr_ldm"
export RUN_NAME="ldm_tb_overfit_32_fast_$(date +%Y%m%d-%H%M%S)"

# ────────────────────────────────────────────────────────────────────────────────
# QUALITY-OF-LIFE
# ────────────────────────────────────────────────────────────────────────────────
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ────────────────────────────────────────────────────────────────────────────────
# SUBMIT
# ────────────────────────────────────────────────────────────────────────────────
echo "Submitting SLURM job: LDM Overfit 32 (fast feedback)"
echo "────────────────────────────────────────────────────"
printf "  AE_CKPT_PATH         : %s\n" "$AE_CKPT_PATH"
printf "  OVERFIT_K            : %s\n" "$OVERFIT_K"
printf "  LR / WD              : %s / %s\n" "$LR" "$WEIGHT_DECAY"
printf "  RUN_NAME             : %s\n" "$RUN_NAME"
echo "────────────────────────────────────────────────────"
sbatch cxr_ldm.slurm