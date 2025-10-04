#!/usr/bin/env bash
#
# Launches a speed-optimized LDM diagnostic job to overfit on 32 samples.
#

set -euo pipefail

export ENV_NAME="jax115"

# ❗ UPDATE THIS TO YOUR VERIFIED VAE CHECKPOINT ❗
export AE_CKPT_PATH="runs/ae_tb_full_kl_1.0e-5_zchannels_3/20251003-125631/ckpts/last.flax"
export AE_CONFIG_PATH="runs/ae_tb_full_kl_1.0e-5_zchannels_3/20251003-125631/run_meta.json"

# --- Key Training Parameters ---
export TASK="TB"
export DISEASE="1"
export EPOCHS=300 # More epochs needed for a larger set
export LATENT_SCALE_FACTOR="1.938884" # Use the value from your diagnostics
export SAMPLE_EVERY=25
export LOG_EVERY=10

# --- Overfitting & Speed Optimization ---
export OVERFIT_ONE=0
export OVERFIT_K=32
# For speed, set batch size to match the dataset size to process all data in one step.
export BATCH_PER_DEVICE=32

# --- LDM Architecture ---
export LDM_BASE_CH=256
export LDM_CH_MULTS="1:2:4:4"
export LDM_ATTN_RES="16:8"

# --- Experiment Naming ---
export WANDB_TAGS="ldm:tb:overfit-32"
export RUN_NAME="ldm_tb_overfit_32_$(date +%Y%m%d)"

echo "Submitting SLURM job: LDM Overfit on 32 Samples"
echo "------------------------------------------------"
echo "  ▶️  Run Name:         $RUN_NAME"
echo "  ▶️  Batch Size:       $BATCH_PER_DEVICE"
echo "  ▶️  Overfit Mode:     32 Samples"
echo "------------------------------------------------"
sbatch cxr_ldm.slurm
echo "✅ Job successfully submitted!"
