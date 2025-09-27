#!/usr/bin/env bash
#
# Launches a FULL training job for the autoencoder on PNEUMONIA samples.
#

set -euo pipefail

export ENV_NAME="jax115"

# --- Key Training Parameters ---
export TASK="PNEUMONIA"
export CLASS_FILTER="1"                  # 0 = Normal, 1 = Diseased
export EPOCHS=200
export BATCH_PER_DEVICE=8
export SAMPLE_EVERY=20
export CH_MULTS="64:128:256"
export WANDB_TAGS="ae:pneumonia:full"

# --- Disable debugging/overfit modes ---
export OVERFIT_ONE=0
export OVERFIT_K=0

export RUN_NAME="ae_pneumonia_full_b${BATCH_PER_DEVICE}_$(date +%Y%m%d)"

echo "Submitting SLURM job for AE FULL training on PNEUMONIA data..."
echo "------------------------------------------------"
echo "  ▶️  Run Name:         $RUN_NAME"
echo "  ▶️  Task:             $TASK (Class: Pneumonia)"
echo "  ▶️  Overfit Mode:     DISABLED"
echo "------------------------------------------------"
sbatch cxr_ae.slurm
echo "✅ Job successfully submitted!"