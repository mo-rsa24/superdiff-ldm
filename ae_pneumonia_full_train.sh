#!/usr/bin/env bash

set -euo pipefail

export ENV_NAME="jax115"

# --- Key Training Parameters (Aligned with CompVis) ---
export TASK="PNEUMONIA"
export CLASS_FILTER="1"
export EPOCHS=200
export BATCH_PER_DEVICE=8
export SAMPLE_EVERY=10

# --- ARCHITECTURE ---
export BASE_CH=64
export CH_MULTS="1,2,4,4"
export ATTN_RES="16"
export Z_CHANNELS=3
export EMBED_DIM=3
export NUM_RES_BLOCKS=2

# --- VAE Regularization ---
export KL_WEIGHT=1.0e-5

# --- W&B Logging ---
export WANDB=1
export WANDB_TAGS="ae:pneumonia:full"
export OVERFIT_ONE=0
export OVERFIT_K=0
export RUN_NAME="ae_pneumonia_full_kl_1.0e-5_zchannels_3"

echo "Submitting SLURM job for VAE DIAGNOSTIC training..."
echo "------------------------------------------------"
echo "  ▶️  Run Name:         $RUN_NAME"
echo "  ▶️  Task:             $TASK (Class: PNEUMONIA)"
echo "  ▶️  Latent Channels:  $Z_CHANNELS"
echo "  ▶️  Attention At:     $ATTN_RES"
echo "------------------------------------------------"
sbatch cxr_ae.slurm
echo "✅ Job successfully submitted!"