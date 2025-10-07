#!/usr/bin/env bash
set -euo pipefail

# --- Defaults ---
export TASK="TB"
export ENV_NAME="jax115"
export IMG_SIZE="256"
export TRAINING_MODE="${1:-full_train}"
export LR="1e-4"
export WEIGHT_DECAY="0.05"
export LDM_BASE_CH="96"
export GRAD_CLIP="1.0"
export BATCH_PER_DEVICE="16"
# SLURM Defaults
export SLURM_PARTITION="bigbatch"
export SLURM_JOB_NAME="cxr-ldm-${TASK,,}-${TRAINING_MODE}" # Default job name, e.g., cxr-ldm-tb-full_train

# --- VAE Checkpoint (❗ IMPORTANT: Update this path) ---
export AE_CKPT_PATH="runs/ae_tb_full_kl_1.0e-5_zchannels_3/20251003-125631/ckpts/last.flax"
export AE_CONFIG_PATH="runs/ae_tb_full_kl_1.0e-5_zchannels_3/20251003-125631/run_meta.json"

# --- Parse Command-Line Overrides ---
shift # Shift away the TRAINING_MODE argument
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --lr) LR="$2"; shift ;;
        --weight-decay) WEIGHT_DECAY="$2"; shift ;;
        --ldm-base-ch) LDM_BASE_CH="$2"; shift ;;
        --job-name) SLURM_JOB_NAME="$2"; shift ;;
        --partition) SLURM_PARTITION="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# --- Configure Training Mode ---
export OVERFIT_ONE="0"; export OVERFIT_K="0"
if [[ "$TRAINING_MODE" == "overfit_one" ]]; then export OVERFIT_ONE="1";
elif [[ "$TRAINING_MODE" == "overfit_16" ]]; then export OVERFIT_K="16";
elif [[ "$TRAINING_MODE" == "overfit_32" ]]; then export OVERFIT_K="32";
fi

# --- Run Naming ---
export RUN_NAME="ldm_${TASK,,}_${TRAINING_MODE}_lr${LR}_$(date +%Y%m%d-%H%M%S)"
export WANDB_PROJECT="cxr-ldm"
export WANDB_TAGS="ldm,${TASK,,},${TRAINING_MODE}"

# --- Submit to SLURM ---
echo "Submitting LDM Training for ${TASK} (${TRAINING_MODE})"
echo "  LR: ${LR}"
echo "  Weight Decay: ${WEIGHT_DECAY}"
echo "  LDM Base CH: ${LDM_BASE_CH}"
echo "  SLURM Job Name: ${SLURM_JOB_NAME}"
echo "  SLURM Partition: ${SLURM_PARTITION}"

sbatch --job-name="$SLURM_JOB_NAME" --partition="$SLURM_PARTITION" slurm_scripts/cxr_ldm.slurm