#!/usr/bin/env bash
set -euo pipefail

# --- Defaults (can be overridden by command-line args) ---
export ENV_NAME="jax115"
export DATA_ROOT="../datasets/cleaned"
export TASK="All_CXR"
export IMG_SIZE="128"
export CLASS_FILTER="None"
export BASE_CH="32"
export CH_MULTS="1,2,4"
export Z_CHANNELS="4"
export EMBED_DIM="4"
export NUM_RES_BLOCKS="2"
export ATTN_RES="16"
export LR="1e-4"
export KL_WEIGHT="1.0e-5"
export EPOCHS="100"
export BATCH_PER_DEVICE="8"
export SAMPLE_EVERY="10"
export WANDB="1"
export WANDB_PROJECT="unified-cxr-vae"

# SLURM Defaults
export SLURM_PARTITION="bigbatch"
export SLURM_JOB_NAME="unified-ae-proto"

# --- Parse Command-Line Arguments ---
# This loop processes arguments like --partition, --job-name, etc.
# Any other arguments (e.g., --img_size, --base_ch) are passed to the python script via $@
OTHER_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --partition)
      export SLURM_PARTITION="$2"
      shift 2
      ;;
    --job-name)
      export SLURM_JOB_NAME="$2"
      shift 2
      ;;
    *)
      OTHER_ARGS+=("$1") # save unrecognized arg
      shift
      ;;
  esac
done

# --- Run Naming (uses final values) ---
export RUN_NAME="${SLURM_JOB_NAME}_z${Z_CHANNELS}_$(date +%Y%m%d-%H%M%S)"
export WANDB_TAGS="unified-ae,all-cxr,z${Z_CHANNELS}"

# --- Submit to SLURM ---
echo "Submitting Unified Autoencoder Training..."
sbatch --partition="$SLURM_PARTITION" --job-name="$SLURM_JOB_NAME" slurm_scripts/cxr_ae.slurm "${OTHER_ARGS[@]}"
echo "✅ Job successfully submitted!"