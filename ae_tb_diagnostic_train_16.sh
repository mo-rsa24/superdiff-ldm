#!/usr/bin/env bash

set -euo pipefail

export ENV_NAME="jax115"

# --- Key Training Parameters ---
export TASK="TB"
export CLASS_FILTER="1"
export EPOCHS=200
export BATCH_PER_DEVICE=4
export SAMPLE_EVERY=10

# --- ARCHITECTURE ---
export BASE_CH=128
export CH_MULTS="1,2,4,4"
export ATTN_RES="16"
export Z_CHANNELS=64
export EMBED_DIM=64
export NUM_RES_BLOCKS=2

# --- VAE Regularization ---
export KL_WEIGHT=1.0e-6

# --- W&B Logging ---
export WANDB=1
export WANDB_TAGS="ae:tb:diagnostic-overfit-16"

# --- MODIFIED: Overfit on a small batch of 16 instead of one ---
export OVERFIT_ONE=0 # Turn OFF single-sample overfitting
export OVERFIT_K=16  # Turn ON small-batch overfitting with 16 samples

export RUN_NAME="ae_tb_diagnostic_overfit16_$(date +%Y%m%d)"

echo "Submitting SLURM job for VAE DIAGNOSTIC training (overfitting on 16 samples)..."
echo "------------------------------------------------"
echo "  ▶️  Run Name:         $RUN_NAME"
echo "  ▶️  Task:             $TASK (Class: TB)"
echo "  ▶️  Overfit Mode:     ENABLED ($OVERFIT_K samples)"
echo "------------------------------------------------"
sbatch cxr_ae.slurm
echo "✅ Job successfully submitted!"