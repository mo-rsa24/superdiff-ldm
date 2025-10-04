#!/usr/bin/env bash
#
# Launches a DIAGNOSTIC training job for the LDM on a single TB latent.
#

set -euo pipefail

export ENV_NAME="jax115"

# ❗ UPDATE AFTER AE TRAINING ❗
export AE_CKPT_PATH="runs/ae_tb_full_kl_1.0e-5_zchannels_3/20251003-125631/ckpts/last.flax"
export AE_CONFIG_PATH="runs/ae_tb_full_kl_1.0e-5_zchannels_3/20251003-125631/run_meta.json"

# --- Key Training Parameters ---
export TASK="TB"
export DISEASE="1"
export EPOCHS=50
export LATENT_SCALE_FACTOR="1.9388840460107197"
export BATCH_PER_DEVICE=1
export SAMPLE_EVERY=5
export LOG_EVERY=10
export LDM_BASE_CH=256
export LDM_CH_MULTS="1:2:4:4"
export LDM_ATTN_RES="4:2:1" # Attention in the deeper layers

export WANDB_TAGS="ldm:tb:diagnostic"

# --- Enable overfitting on one sample ---
export OVERFIT_ONE=1
export OVERFIT_K=0

export RUN_NAME="ldm_tb_diagnostic_$(date +%Y%m%d)"

echo "Submitting SLURM job for LDM DIAGNOSTIC training on TB data..."
echo "------------------------------------------------"
echo "  ▶️  Run Name:         $RUN_NAME"
echo "  ▶️  Task:             $TASK (Class: TB)"
echo "  ▶️  Overfit Mode:     ENABLED (1 sample)"
echo "  ▶️  AE Checkpoint:    $AE_CKPT_PATH"
echo "------------------------------------------------"
sbatch cxr_ldm.slurm
echo "✅ Job successfully submitted!"