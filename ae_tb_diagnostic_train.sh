#!/usr/bin/env bash

set -euo pipefail

export ENV_NAME="jax115"

# --- Key Training Parameters (Aligned with CompVis) ---
export TASK="TB"
export CLASS_FILTER="1"
export EPOCHS=200
export BATCH_PER_DEVICE=1
export SAMPLE_EVERY=10

# --- ARCHITECTURE ---
export BASE_CH=128
export CH_MULTS="1:2:4:4"  # Deeper and wider
export ATTN_RES="16"       # Add attention at 16x16 resolution
export Z_CHANNELS=64       # CRITICAL: Match LDM's expected channels
export EMBED_DIM=64        # Explicitly set embedding dimension

# --- VAE Regularization ---
export KL_WEIGHT=1.0e-6    # Start with a small KL weight

export WANDB_TAGS="ae:tb:diagnostic-compvis-arch"
export OVERFIT_ONE=1
export OVERFIT_K=0
export RUN_NAME="ae_tb_diagnostic_compvis_$(date +%Y%m%d)"

echo "Submitting SLURM job for VAE DIAGNOSTIC training (CompVis-style)..."
echo "------------------------------------------------"
echo "  ▶️  Run Name:         $RUN_NAME"
echo "  ▶️  Task:             $TASK (Class: TB)"
echo "  ▶️  Latent Channels:  $Z_CHANNELS"
echo "  ▶️  Attention At:     $ATTN_RES"
echo "------------------------------------------------"
sbatch cxr_ae.slurm
echo "✅ Job successfully submitted!"