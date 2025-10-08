#!/usr/bin/env bash
set -euo pipefail

# --- Defaults (can be overridden by command-line arguments) ---
export TASK="TB"
export ENV_NAME="jax115"
export IMG_SIZE="128"
export TRAINING_MODE="${1:-full_train}" # Reads mode (e.g., full_train) from the first argument
export DISEASE="1" # 1 for TB, 0 for Normal

# --- Hyperparameter Defaults ---
export LR="1e-4"
export WEIGHT_DECAY="0.05"
export LDM_BASE_CH="64"
export GRAD_CLIP="1.0"
export BATCH_PER_DEVICE="16"
export EPOCHS="300"
export LOG_EVERY="100"
export SAMPLE_EVERY="10"
export SAMPLE_BATCH_SIZE="16"
export LDM_CH_MULTS="1,2,4"
export LDM_NUM_RES_BLOCKS="2"
export LDM_ATTN_RES="16"
export WANDB="1"

# --- Shared VAE and Scale Factor (❗ IMPORTANT: Update these values) ---
export AE_CKPT_PATH="runs/unified-ae-128_z4_20251008-135847/20251008-140403/ckpts/last.flax"
export AE_CONFIG_PATH="runs/unified-ae-128_z4_20251008-135847/20251008-140403/run_meta.json"
export LATENT_SCALE_FACTOR="3.133603"

# --- SLURM Defaults ---
export SLURM_PARTITION="bigbatch"
export SLURM_JOB_NAME="ldm-${TASK,,}-proto"

# --- Robust Argument Parsing Loop ---
OTHER_ARGS=()
shift # Shift away the first argument (training_mode)

while [[ $# -gt 0 ]]; do
  case $1 in
    --partition)          export SLURM_PARTITION="$2"; shift 2 ;;
    --job-name)           export SLURM_JOB_NAME="$2"; shift 2 ;;
    --lr)                 export LR="$2"; shift 2 ;;
    --weight_decay)       export WEIGHT_DECAY="$2"; shift 2 ;;
    --ldm_base_ch)        export LDM_BASE_CH="$2"; shift 2 ;;
    --grad_clip)          export GRAD_CLIP="$2"; shift 2 ;;
    --epochs)             export EPOCHS="$2"; shift 2 ;;
    --batch_per_device)   export BATCH_PER_DEVICE="$2"; shift 2 ;;
    --ldm_ch_mults)       export LDM_CH_MULTS="$2"; shift 2 ;;
    --ldm_num_res_blocks) export LDM_NUM_RES_BLOCKS="$2"; shift 2 ;;
    --ldm_attn_res)       export LDM_ATTN_RES="$2"; shift 2 ;;
    --log_every)          export LOG_EVERY="$2"; shift 2 ;;
    --sample_every)       export SAMPLE_EVERY="$2"; shift 2 ;;
    --sample_batch_size)  export SAMPLE_BATCH_SIZE="$2"; shift 2 ;;
    *)                    OTHER_ARGS+=("$1"); shift ;; # Save unrecognized arg
  esac
done
# --- Run Naming (uses the final, potentially overridden values) ---
export RUN_NAME="${SLURM_JOB_NAME}_lr${LR}_wd${WEIGHT_DECAY}_ch${LDM_BASE_CH}_$(date +%Y%m%d-%H%M%S)"
export WANDB_PROJECT="cxr-ldm-composition"
export WANDB_TAGS="ldm,${TASK,,},proto"

# --- Prettier Submit Message ---
CYN=$(printf '\033[36m'); BLU=$(printf '\033[34m'); BLD=$(printf '\033[1m'); RST=$(printf '\033[0m')
kv(){ printf "  ${CYN}%-22s${RST} %s\n" "$1" "$2"; }
rule(){ printf "${BLU}%.0s" $(seq 1 60); printf "${RST}\n"; }

rule
printf "${BLD}${BLU}🚀 Submitting LDM Training Job${RST}\n"
rule
kv "SLURM Job Name" "${SLURM_JOB_NAME}"
kv "SLURM Partition" "${SLURM_PARTITION}"
printf "\n"
kv "📊 Dataset Task" "${TASK} (Class: ${DISEASE})"
kv "Image Size" "${IMG_SIZE}"
kv "Training Mode" "${TRAINING_MODE}"
printf "\n"
kv "🧠 Model Base CH" "${LDM_BASE_CH}"
kv "Model CH Multipliers" "${LDM_CH_MULTS}"
kv "Attention Resolutions" "${LDM_ATTN_RES}"
printf "\n"
kv "⚙️ Learning Rate" "${LR}"
kv "Epochs" "${EPOCHS}"
kv "Batch Size" "${BATCH_PER_DEVICE}"
kv "Log Every (Steps)" "${LOG_EVERY}"
kv "Sample Every (Epochs)" "${SAMPLE_EVERY}"
kv "Sample Batch Size" "${SAMPLE_BATCH_SIZE}"
rule

sbatch --partition="$SLURM_PARTITION" --job-name="$SLURM_JOB_NAME" slurm_scripts/cxr_ldm.slurm "${OTHER_ARGS[@]}"
echo "✅ Job successfully submitted!"