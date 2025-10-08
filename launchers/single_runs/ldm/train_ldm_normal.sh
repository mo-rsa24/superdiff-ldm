#!/usr/bin/env bash
set -euo pipefail

# --- Defaults ---
export TASK="TB"
export ENV_NAME="jax115"
export IMG_SIZE="256"
export TRAINING_MODE="${1:-full_train}"
export DISEASE="0"

# --- Hyperparameters ---
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
export LATENT_SCALE_FACTOR="8.97676989056676"
# SLURM Defaults
# --- SLURM Defaults ---
export SLURM_PARTITION="bigbatch"
export SLURM_JOB_NAME="ldm-${TASK,,}-proto-${TRAINING_MODE}"

# --- VAE Checkpoint (❗ IMPORTANT: Update this path) ---
export AE_CKPT_PATH="runs/unified-ae-128_z4_20251008-135847/20251008-140403/ckpts/last.flax"
export AE_CONFIG_PATH="runs/unified-ae-128_z4_20251008-135847/20251008-140403/run_meta.json"

# --- Parse Command-Line Overrides ---
OTHER_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --partition) export SLURM_PARTITION="$2"; shift 2 ;;
    --job-name) export SLURM_JOB_NAME="$2"; shift 2 ;;
    *) OTHER_ARGS+=("$1"); shift ;;
  esac
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
export RUN_NAME="${SLURM_JOB_NAME}_$(date +%Y%m%d-%H%M%S)"
export WANDB_PROJECT="cxr-ldm-composition"
export WANDB_TAGS="ldm,${TASK,,},proto"

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

# --- Submit to SLURM ---
echo "Submitting LDM Training for TASK: ${TASK} (Class Filter: ${DISEASE})"
sbatch --job-name="$SLURM_JOB_NAME" --partition="$SLURM_PARTITION" slurm_scripts/cxr_ldm.slurm
echo "✅ Job successfully submitted!"