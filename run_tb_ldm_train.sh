#!/usr/bin/env bash
#
# Launches a full training job for the Latent Diffusion Model on NORMAL samples using cxr_ldm.slurm.
#

set -euo pipefail

# ----------------------------------------------------------------
#                ✅ --- USER CONFIGURATION --- ✅
# ----------------------------------------------------------------
# REQUIRED: Set the name of your conda or mamba environment
export ENV_NAME="jax115"

# REQUIRED: Set paths to your trained Autoencoder artifacts
# NOTE: This should likely be an AE trained on both normal and diseased, or just normal.
# Using the TB-trained one for now as an example.
export AE_CKPT_PATH="runs/ae_full_tb_b4_20250918/20250918-112409/ckpts/last.flax"
export AE_CONFIG_PATH="runs/ae_full_tb_b4_20250918/20250918-112409/run_meta.json"

# --- Key Training Parameters ---
export TASK="TB"                         # Dataset task (TB or PNEUMONIA) - NORMAL is a class within these
export DISEASE="0"                       # Set to 0 to train on the NORMAL subset
export EPOCHS=500
export BATCH_PER_DEVICE=4
export SAMPLE_EVERY=50
export LDM_BASE_CH=192
export LDM_CH_MULTS="1:2:3"
export WANDB_TAGS="ldm-full-train:slurm:normal" # Set W&B tags

# ----------------------------------------------------------------
#                🚀 --- JOB LAUNCH LOGIC --- 🚀
# ----------------------------------------------------------------
# Disable debugging/overfit modes for a full training run
export OVERFIT_ONE=0
export OVERFIT_K=0

# Create a descriptive name for the experiment run
export RUN_NAME="ldm_full_${TASK,,}_normal_b${BATCH_PER_DEVICE}_$(date +%Y%m%d)"

# --- Submit to Slurm ---
echo "Submitting SLURM job for LDM FULL TRAINING on NORMAL data..."
echo "------------------------------------------------"
echo "  ▶️  Run Name:         $RUN_NAME"
echo "  ▶️  Conda Env:        $ENV_NAME"
echo "  ▶️  Task:             $TASK (NORMAL subset)"
echo "  ▶️  Epochs:           $EPOCHS"
echo "  ▶️  Batch per Device: $BATCH_PER_DEVICE"
echo "  ▶️  AE Checkpoint:    $AE_CKPT_PATH"
echo "  ▶️  Overfit Mode:     DISABLED"
echo "------------------------------------------------"

sbatch cxr_ldm.slurm

echo "✅ Job successfully submitted!"