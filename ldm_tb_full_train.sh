#!/usr/bin/env bash
#
# Launch a FULL LDM training run on the TB dataset.
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
# LDM: FULL TRAINING
# ────────────────────────────────────────────────────────────────────────────────
export OVERFIT_ONE="0"             # disable overfit
export OVERFIT_K="0"               # disable overfit
export REPEAT_LEN="1"              # not used

export LDM_CH_MULTS="1,2,4,4"
export LDM_BASE_CH="128"
export LDM_NUM_RES_BLOCKS="2"
export LDM_ATTN_RES="16,8"

# Use the new, more robust scale factor you calculated!
export LATENT_SCALE_FACTOR="YOUR_NEW_SCALE_FACTOR"

# ────────────────────────────────────────────────────────────────────────────────
# OPTIMIZER / SCHEDULING (tuned for full dataset)
# ────────────────────────────────────────────────────────────────────────────────
export LR="1e-4"                   # lower LR for full training
export WEIGHT_DECAY="0.01"         # add some weight decay
export GRAD_CLIP="1.0"

export EPOCHS="1000"               # train for many epochs
export BATCH_PER_DEVICE="16"       # as large as your GPU can handle
export LOG_EVERY="100"             # log less frequently

export SAMPLE_EVERY="10"           # sample every 10 epochs
export SAMPLE_BATCH_SIZE="16"

# ────────────────────────────────────────────────────────────────────────────────
# W&B
# ────────────────────────────────────────────────────────────────────────────────
export USE_WANDB="1"               # enable W&B for full runs
export WANDB_PROJECT="cxr-ldm"
export WANDB_TAGS="ldm:tb:full-train"

# ────────────────────────────────────────────────────────────────────────────────
# RUN NAMING
# ────────────────────────────────────────────────────────────────────────────────
export EXP_NAME="cxr_ldm"
export RUN_NAME="ldm_tb_full_train_$(date +%Y%m%d-%H%M%S)"

# ────────────────────────────────────────────────────────────────────────────────
# QUALITY-OF-LIFE
# ────────────────────────────────────────────────────────────────────────────────
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ────────────────────────────────────────────────────────────────────────────────
# SUBMIT
# ────────────────────────────────────────────────────────────────────────────────
echo "Submitting SLURM job: LDM Full Training"
echo "────────────────────────────────────────────────────"
printf "  AE_CKPT_PATH         : %s\n" "$AE_CKPT_PATH"
printf "  LR / WD              : %s / %s\n" "$LR" "$WEIGHT_DECAY"
printf "  BATCH_PER_DEVICE     : %s\n" "$BATCH_PER_DEVICE"
printf "  RUN_NAME             : %s\n" "$RUN_NAME"
echo "────────────────────────────────────────────────────"
sbatch cxr_ldm.slurm