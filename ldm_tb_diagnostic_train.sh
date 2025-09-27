#!/usr/bin/env bash
#
# Launches a DIAGNOSTIC training job for the LDM on a single TB latent.
#

set -euo pipefail

export ENV_NAME="jax115"

# ❗ UPDATE AFTER AE TRAINING ❗
export AE_CKPT_PATH="runs/ae_full_tb_b4_20250918/20250918-112409/ckpts/last.flax"
export AE_CONFIG_PATH="runs/ae_full_tb_b4_20250918/20250918-112409/run_meta.json"

# --- Key Training Parameters ---
export TASK="TB"
export DISEASE="1"
export EPOCHS=50
export BATCH_PER_DEVICE=1
export SAMPLE_EVERY=5
export LDM_BASE_CH=192
export LDM_CH_MULTS="1:2:3"
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