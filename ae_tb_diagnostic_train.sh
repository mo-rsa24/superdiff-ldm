#!/usr/bin/env bash
#
# Launches a DIAGNOSTIC training job for the autoencoder on a single TB sample
# with settings suitable for creating a normalized latent space.
#

set -euo pipefail

# REQUIRED: Set the name of your conda or mamba environment
export ENV_NAME="jax115"

# --- Key Training Parameters ---
export TASK="TB"
export CLASS_FILTER="1"
export EPOCHS=200 # Train a bit longer to ensure convergence
export BATCH_PER_DEVICE=1
export SAMPLE_EVERY=10
export CH_MULTS="64:128:256"

# --- MODIFIED: Set parameters for a proper VAE ---
export Z_CHANNELS=4          # Increased from 3 to 4 for more capacity
export KL_WEIGHT=1.0e-5      # Set a meaningful KL weight to regularize the latent space

export WANDB_TAGS="ae:tb:diagnostic-vae"

# --- Enable overfitting on one sample ---
export OVERFIT_ONE=1
export OVERFIT_K=0

export RUN_NAME="ae_tb_diagnostic_vae_$(date +%Y%m%d)"

echo "Submitting SLURM job for VAE DIAGNOSTIC training..."
echo "------------------------------------------------"
echo "  ▶️  Run Name:         $RUN_NAME"
echo "  ▶️  Task:             $TASK (Class: TB)"
echo "  ▶️  Overfit Mode:     ENABLED (1 sample)"
echo "  ▶️  Latent Channels:  $Z_CHANNELS"
echo "  ▶️  KL Weight:        $KL_WEIGHT"
echo "------------------------------------------------"
sbatch cxr_ae.slurm
echo "✅ Job successfully submitted!"
