#!/usr/bin/env bash
#
# Launches a diagnostic training job for the Latent Diffusion Model on NORMAL samples using cxr_ldm.slurm.
#

set -euo pipefail

# ----------------------------------------------------------------
#                ✅ --- USER CONFIGURATION --- ✅
# ----------------------------------------------------------------
# REQUIRED: Set the name of your conda or mamba environment
export ENV_NAME="jax115"
export AE_CKPT_PATH="runs/ae_full_tb_b4_20250918/20250918-112409/ckpts/last.flax"
export AE_CONFIG_PATH="runs/ae_full_tb_b4_20250918/20250918-112409/run_meta.json"

# --- Key Training Parameters ---
export TASK="TB"                         # Dataset task (TB or PNEUMONIA) - NORMAL is a class within these
export DISEASE="1"                       # Set to 0 to train on the NORMAL subset
export EPOCHS=100
export BATCH_PER_DEVICE=1
export SAMPLE_EVERY=10
export LDM_BASE_CH=192
export LDM_CH_MULTS="1:2:3"
export WANDB_TAGS="ldm-diagnostic-train:slurm:tb" # Set W&B tags

# ----------------------------------------------------------------
#                🚀 --- JOB LAUNCH LOGIC --- 🚀
# ----------------------------------------------------------------
# --- Enable overfitting on a single example ---
export OVERFIT_ONE=1
export OVERFIT_K=0

# Create a descriptive name for the experiment run
export RUN_NAME="ldm_diagnostic_${TASK,,}_normal_b${BATCH_PER_DEVICE}_$(date +%Y%m%d)"

# --- Submit to Slurm ---
echo "Submitting SLURM job for LDM Diagnostic TRAINING on NORMAL data..."
echo "------------------------------------------------"
echo "  ▶️  Run Name:         $RUN_NAME"
echo "  ▶️  Conda Env:        $ENV_NAME"
echo "  ▶️  Task:             $TASK (NORMAL subset)"
echo "  ▶️  Epochs:           $EPOCHS"
echo "  ▶️  Batch per Device: $BATCH_PER_DEVICE"
echo "  ▶️  AE Checkpoint:    $AE_CKPT_PATH"
echo "  ▶️  Overfit Mode:     ENABLED (1 sample)"
echo "------------------------------------------------"

sbatch cxr_ldm.slurm

echo "✅ Job successfully submitted!"