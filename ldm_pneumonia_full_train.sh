#!/usr/bin/env bash
#
# Launches a FULL training job for the LDM on PNEUMONIA latents.
#

set -euo pipefail

export ENV_NAME="jax115"

# ❗ UPDATE AFTER AE TRAINING ❗
export AE_CKPT_PATH="runs/ae_full_pneumonia_b8_20250924/20250924-081619/ckpts/last.flax"
export AE_CONFIG_PATH="runs/ae_full_pneumonia_b8_20250924/20250924-081619/run_meta.json"

# --- Key Training Parameters ---
export TASK="PNEUMONIA"
export DISEASE="1"                       # 0 = Normal, 1 = Diseased
export EPOCHS=500
export BATCH_PER_DEVICE=4
export SAMPLE_EVERY=25
export LDM_BASE_CH=192
export LDM_CH_MULTS="1:2:3"
export WANDB_TAGS="ldm:pneumonia:full"

# --- Disable debugging/overfit modes ---
export OVERFIT_ONE=0
export OVERFIT_K=0

export RUN_NAME="ldm_pneumonia_full_b${BATCH_PER_DEVICE}_$(date +%Y%m%d)"

echo "Submitting SLURM job for LDM FULL training on PNEUMONIA data..."
echo "------------------------------------------------"
echo "  ▶️  Run Name:         $RUN_NAME"
echo "  ▶️  Task:             $TASK (Class: Pneumonia)"
echo "  ▶️  Overfit Mode:     DISABLED"
echo "  ▶️  AE Checkpoint:    $AE_CKPT_PATH"
echo "------------------------------------------------"
sbatch cxr_ldm.slurm
echo "✅ Job successfully submitted!"