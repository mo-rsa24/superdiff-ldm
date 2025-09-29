#!/usr/bin/env bash
#
# Launches a DIAGNOSTIC training job for the autoencoder on a single NORMAL sample.
#

set -euo pipefail

export ENV_NAME="jax115"

# --- Key Training Parameters ---
export TASK="TB"
export CLASS_FILTER="0"
export EPOCHS=100
export REPEAT_LEN=100
export BATCH_PER_DEVICE=1
export SAMPLE_EVERY=10
export CH_MULTS="64:128:256"
export WANDB_TAGS="ae:normal:diagnostic"

# --- Enable overfitting on one sample ---
export OVERFIT_ONE=1
export OVERFIT_K=0

export RUN_NAME="ae_normal_diagnostic_$(date +%Y%m%d)"

echo "Submitting SLURM job for AE DIAGNOSTIC training on NORMAL data..."
echo "------------------------------------------------"
echo "  ▶️  Run Name:         $RUN_NAME"
echo "  ▶️  Task:             $TASK (Class: Normal)"
echo "  ▶️  Overfit Mode:     ENABLED (1 sample)"
echo "------------------------------------------------"
sbatch cxr_ae.slurm
echo "✅ Job successfully submitted!"