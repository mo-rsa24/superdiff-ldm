#!/usr/bin/env bash
#
# Launches a full training job for the autoencoder using cxr_ae.slurm.
# This script disables the default "overfit" mode.
#

set -euo pipefail

# ----------------------------------------------------------------
#                ✅ --- USER CONFIGURATION --- ✅
# ----------------------------------------------------------------
# REQUIRED: Set the name of your conda or mamba environment
export ENV_NAME="jax115"

# --- Key Training Parameters ---
export TASK="TB"                         # Set the dataset task (TB or PNEUMONIA)
export CLASS_FILTER="0"                  # 0 = Normal, 1 = Diseased
export EPOCHS=200                        # Keep this high for the full run
export BATCH_PER_DEVICE=16
export SAMPLE_EVERY=5
export CH_MULTS="128:256:512"
export Z_CHANNELS="4"
export WANDB_TAGS="ae:normal:full:fast"

# --- Add these to your script to override the defaults in cxr_ae.slurm ---
export PERCEPTUAL_WEIGHT="1.0"
export DISC_START="10000"

# ----------------------------------------------------------------
#                🚀 --- JOB LAUNCH LOGIC --- 🚀
# ----------------------------------------------------------------
# IMPORTANT: Disable debugging/overfit modes for a full training run
export OVERFIT_ONE=0  # Set to 0 to turn OFF overfitting
export OVERFIT_K=0    # Set to 0 to turn OFF tiny-K subset training

# Create a descriptive name for the experiment run
export RUN_NAME="ae_normal_full_b${BATCH_PER_DEVICE}_$(date +%Y%m%d)"

# --- Submit to Slurm ---
echo "Submitting SLURM job for AE FULL training on NORMAL data..."
echo "------------------------------------------------"
echo "  ▶️  Run Name:         $RUN_NAME"
echo "  ▶️  Task:             $TASK (Class: Normal)"
echo "  ▶️  Overfit Mode:     DISABLED"
echo "------------------------------------------------"
sbatch cxr_ae.slurm
echo "✅ Job successfully submitted!"