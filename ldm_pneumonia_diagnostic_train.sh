#!/usr/bin/env bash
#
# Launches a diagnostic training job for the Latent Diffusion Model using cxr_ldm.slurm.
#

set -euo pipefail

# ----------------------------------------------------------------
#                ✅ --- USER CONFIGURATION --- ✅
# ----------------------------------------------------------------
# REQUIRED: Set the name of your conda or mamba environment
export ENV_NAME="jax115"
export AE_CKPT_PATH="runs/ae_full_pneumonia_b8_20250924/20250924-081619/ckpts/last.flax"
export AE_CONFIG_PATH="runs/ae_full_pneumonia_b8_20250924/20250924-081619/run_meta.json"

# --- Key Training Parameters ---
export TASK="PNEUMONIA"                         # Dataset task (TB or PNEUMONIA)
export DISEASE="1"                       # Set to 0 to train on the NORMAL subset
export EPOCHS=100                  # Set a higher number of epochs for LDM
export BATCH_PER_DEVICE=1                # Adjust batch size
export SAMPLE_EVERY=10                   # Sample every 20 epochs
export LDM_BASE_CH=192                   # UNet base channels
export LDM_CH_MULTS="1:2:3"              # UNet channel multipliers (use colons)
export WANDB_TAGS="ldm-diagnostic-train:slurm:pneumonia" # Set W&B tags (use colons)

# ----------------------------------------------------------------
#                🚀 --- JOB LAUNCH LOGIC --- 🚀
# ----------------------------------------------------------------
# Disable debugging/overfit modes for a full training run
export OVERFIT_ONE=1
export OVERFIT_K=0

# Create a descriptive name for the experiment run
export RUN_NAME="ldm_diagnostic_${TASK,,}_b${BATCH_PER_DEVICE}_$(date +%Y%m%d)"

# --- Submit to Slurm ---
echo "Submitting SLURM job for LDM Diagnostic TRAINING..."
echo "------------------------------------------------"
echo "  ▶️  Run Name:         $RUN_NAME"
echo "  ▶️  Conda Env:        $ENV_NAME"
echo "  ▶️  Task:             $TASK"
echo "  ▶️  Epochs:           $EPOCHS"
echo "  ▶️  Batch per Device: $BATCH_PER_DEVICE"
echo "  ▶️  AE Checkpoint:    $AE_CKPT_PATH"
echo "  ▶️  Overfit Mode:     ENABLED (1 sample)"
echo "------------------------------------------------"

sbatch cxr_ldm.slurm

echo "✅ Job successfully submitted!"