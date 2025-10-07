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
export EPOCHS="1000"
export LOG_EVERY="100"
export SAMPLE_EVERY="10"
export SAMPLE_BATCH_SIZE="16"
export WANDB="1" # Use 1 for 'true', 0 for 'false
export LATENT_SCALE_FACTOR="1.951791"
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
        --batch-per-device) BATCH_PER_DEVICE="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --log-every) LOG_EVERY="$2"; shift ;;
        --sample-every) SAMPLE_EVERY="$2"; shift ;;
        --job-name) SLURM_JOB_NAME="$2"; shift ;;
        --partition) SLURM_PARTITION="$2"; shift ;;
        --no-wandb) WANDB="0";;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# --- Configure Training Mode based on presets ---
if [[ "$TRAINING_MODE" == "overfit_one" ]]; then
    export OVERFIT_ONE="1"; export OVERFIT_K="0"; export EPOCHS="50"
    export BATCH_PER_DEVICE="1"; export LOG_EVERY="5"; export SAMPLE_EVERY="5";
elif [[ "$TRAINING_MODE" == "overfit_16" ]]; then
    export OVERFIT_ONE="0"; export OVERFIT_K="16";
    export BATCH_PER_DEVICE="16"; export LOG_EVERY="5"; export SAMPLE_EVERY="5"; export EPOCHS="50";
elif [[ "$TRAINING_MODE" == "overfit_32" ]]; then
    export OVERFIT_ONE="0"; export OVERFIT_K="32";
    export BATCH_PER_DEVICE="16"; export LOG_EVERY="5"; export SAMPLE_EVERY="5"; export EPOCHS="50";
else # full_train
    export OVERFIT_ONE="0"; export OVERFIT_K="0";
fi

# --- Run Naming ---
export RUN_NAME="ldm_${TASK,,}_${TRAINING_MODE}_lr${LR}_wd${WEIGHT_DECAY}_ch${LDM_BASE_CH}_$(date +%Y%m%d-%H%M%S)"
export WANDB_PROJECT="cxr-ldm"
export WANDB_TAGS="ldm,${TASK,,},${TRAINING_MODE}"

# --- Submit to SLURM ---
# --- Prettier Submit Message ---
CYN=$(printf '\033[36m'); BLU=$(printf '\033[34m'); BLD=$(printf '\033[1m'); RST=$(printf '\033[0m')
printf "\n${BLU}${BLD}== Submitting SLURM Job: LDM Training ==${RST}\n"
printf "  ${CYN}%-22s${RST} %s\n" "Task | Mode" "${TASK} | ${TRAINING_MODE}"
printf "  ${CYN}%-22s${RST} %s\n" "SLURM Job Name" "${SLURM_JOB_NAME}"
printf "  ${CYN}%-22s${RST} %s\n" "SLURM Partition" "${SLURM_PARTITION}"
printf -- "----------------------------------------\n"
printf "  ${CYN}%-22s${RST} %s\n" "Learning Rate" "${LR}"
printf "  ${CYN}%-22s${RST} %s\n" "Weight Decay" "${WEIGHT_DECAY}"
printf "  ${CYN}%-22s${RST} %s\n" "LDM Base CH" "${LDM_BASE_CH}"
printf "  ${CYN}%-22s${RST} %s\n" "Batch Per Device" "${BATCH_PER_DEVICE}"
printf "  ${CYN}%-22s${RST} %s\n" "Epochs" "${EPOCHS}"
printf "  ${CYN}%-22s${RST} %s\n" "Log Every" "${LOG_EVERY}"
printf "  ${CYN}%-22s${RST} %s\n" "Sample Every" "${SAMPLE_EVERY}"
printf "  ${CYN}%-22s${RST} %s\n" "W&B Enabled" "$( ((WANDB==1)) && echo 'Yes' || echo 'No' )"
printf -- "----------------------------------------\n"

sbatch --job-name="$SLURM_JOB_NAME" --partition="$SLURM_PARTITION" slurm_scripts/cxr_ldm.slurm