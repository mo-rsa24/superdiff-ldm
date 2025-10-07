#!/usr/bin/env bash
set -euo pipefail

export ENV_NAME="jax115"
export DATA_ROOT="/datasets/mmolefe/cleaned"
export TASK="TB"
export IMG_SIZE="256"
export Z_CHANNELS="${1:-1}" # Default to 1, or take from first argument

# --- Training ---
export LR="1e-4"
export KL_WEIGHT="1e-6"
export EPOCHS="500"
export BATCH_PER_DEVICE="32"

# --- Run Naming ---
export RUN_NAME="ae_tb_z${Z_CHANNELS}_$(date +%Y%m%d-%H%M%S)"
export WANDB_PROJECT="cxr-vae"
export WANDB_TAGS="vae,tb,z${Z_CHANNELS}"

echo "Submitting VAE Training for TB with Z_CHANNELS=${Z_CHANNELS}"
sbatch slurm_scripts/cxr_ae.slurm