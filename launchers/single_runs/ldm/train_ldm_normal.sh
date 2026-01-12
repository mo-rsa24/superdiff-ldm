#!/usr/bin/env bash
set -euo pipefail

if [[ -t 1 ]]; then
  BOLD=$(tput bold); CYAN=$(tput setaf 6); BLUE=$(tput setaf 4); RED=$(tput setaf 1); RESET=$(tput sgr0)
else
  BOLD=""; CYAN=""; BLUE=""; RED=""; RESET=""
fi

status_line() { printf "${BLUE}▶${RESET} ${BOLD}%-20s${RESET} %s\n" "$1:" "${2:-}"; }
rule() { printf "${BLUE}%0.s-${RESET}" {1..50}; printf "\n"; }
header() { printf "\n${BLUE}${BOLD}# %s${RESET}\n" "$1"; rule; }

# --- Defaults (can be overridden by command-line arguments) ---
export TASK="TB"
export ENV_NAME="jaxstack"
export IMG_SIZE="256"
export TRAINING_MODE="${1:-full_train}" # Reads mode (e.g., full_train) from the first argument
export DISEASE="0" # 1 for TB, 0 for Normal

# --- Hyperparameter Defaults ---
export LR="1e-4"
export WEIGHT_DECAY="1e-4"
export LDM_BASE_CH="192"
export GRAD_CLIP="1.0"
export BATCH_PER_DEVICE="16"
export EPOCHS="1500"
export LOG_EVERY="100"
export SAMPLE_EVERY="250"
export SAMPLE_BATCH_SIZE="16"
export LDM_CH_MULTS="1,2,4,4"
export LDM_NUM_RES_BLOCKS="3"
export LDM_ATTN_RES="32,16,8"
export WANDB="1"
export WANDB_PROJECT="cxr-ldm-composition"
export WANDB_ENTITY=""
export WANDB_RUN_GROUP="ldm-normal"
export WANDB_TAGS="ldm,normal,256"

# --- Shared VAE and Scale Factor (❗ IMPORTANT: Update these values) ---
export AE_RUN_DIR="${AE_RUN_DIR:-}"
export AE_CKPT_PATH="${AE_CKPT_PATH:-}"
export AE_CONFIG_PATH="${AE_CONFIG_PATH:-}"
export LATENT_SCALE_FACTOR="0.99999905"

# --- SLURM Defaults ---
export SLURM_PARTITION="bigbatch"
export SLURM_JOB_NAME="ldm-${TASK,,}-normal"
export TIME_LIMIT="${TIME_LIMIT:-72:00:00}"
export STAGING_ROOT="${STAGING_ROOT:-${HOME}/cluster_staging}"
# --- EMA Configuration ---
export USE_EMA="1" # Use "1" for true, "0" for false
export EMA_DECAY="0.999"
# --- Robust Argument Parsing Loop ---
OTHER_ARGS=()
shift || true # Shift away the first argument (training_mode) if present

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
    --ae_ckpt_path)       export AE_CKPT_PATH="$2"; shift 2 ;;
    --ae_config_path)     export AE_CONFIG_PATH="$2"; shift 2 ;;
    --ae_run_dir)         export AE_RUN_DIR="$2"; shift 2 ;;
    --latent_scale_factor) export LATENT_SCALE_FACTOR="$2"; shift 2 ;;
    --wandb_project)      export WANDB_PROJECT="$2"; shift 2 ;;
    --wandb_name)         export WANDB_NAME="$2"; shift 2 ;;
    --wandb_tags)         export WANDB_TAGS="$2"; shift 2 ;;
    --wandb_group)        export WANDB_RUN_GROUP="$2"; shift 2 ;;
    --wandb_entity)       export WANDB_ENTITY="$2"; shift 2 ;;
    --time)               export TIME_LIMIT="$2"; shift 2 ;;
    --workdir)            export WORKDIR="$2"; shift 2 ;;
    *)                    OTHER_ARGS+=("$1"); shift ;; # Save unrecognized arg
  esac
done

if [[ -n "$AE_RUN_DIR" ]]; then
  export AE_CKPT_PATH="${AE_CKPT_PATH:-$AE_RUN_DIR/ckpts/last.flax}"
  export AE_CONFIG_PATH="${AE_CONFIG_PATH:-$AE_RUN_DIR/run_meta.json}"
fi

if [[ -z "$AE_CKPT_PATH" || -z "$AE_CONFIG_PATH" ]]; then
  echo "ERROR: AE_CKPT_PATH and AE_CONFIG_PATH must be set (or pass --ae_run_dir)."
  exit 1
fi

REPO_ROOT=$(pwd)
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
GIT_HASH=$(git rev-parse --short HEAD)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
GIT_PARENT=$(git rev-parse --short HEAD^ 2>/dev/null || echo "none")

export WANDB_NAME="${WANDB_NAME:-${SLURM_JOB_NAME}-${GIT_BRANCH}-${GIT_HASH}-${TIMESTAMP}}"
export RUN_NAME="${RUN_NAME:-$WANDB_NAME}"
export WANDB_TAGS="${WANDB_TAGS:-ldm,normal,${IMG_SIZE},${GIT_BRANCH},${GIT_HASH},parent-${GIT_PARENT}}"

JOB_NAME="${SLURM_JOB_NAME}-${GIT_HASH}"
STAGING_DIR="${STAGING_ROOT}/${JOB_NAME}_${TIMESTAMP}"

echo "Staging repo to ${STAGING_DIR}"
mkdir -p "$STAGING_DIR"
rsync -a \
  --exclude 'logs' \
  --exclude 'runs' \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'wandb' \
  --exclude 'playground' \
  --exclude 'composed_output_sweep' \
  --exclude 'composed_output_single' \
  --exclude 'composed_output' \
  --exclude 'runs_ldm' \
  --exclude 'runs' \
  --exclude 'and_out' \
  --exclude 'superdiff_and_output' \
  "$REPO_ROOT/" "$STAGING_DIR/"

status_line "✅ Snapshot" "Complete"

status_line "--------------------------------------------------------"
mkdir -p "${REPO_ROOT}/logs"
cd "$STAGING_DIR"
mkdir -p "${STAGING_DIR}/runs"
mkdir -p "${STAGING_DIR}/runs_ldm"
export WORKDIR="${WORKDIR:-$STAGING_DIR}"
status_line "--------------------------------------------------------"
status_line "📂 Workdir" "$WORKDIR"
status_line "📌 Commit"  "$GIT_HASH"
status_line "🏷️  Branch"  "$GIT_BRANCH"
status_line "🔗 Parent"  "$GIT_PARENT"
status_line "🪪 W&B"     "$WANDB_NAME"

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
kv "Latent Scale Factor" "${LATENT_SCALE_FACTOR}"
rule

JOB_ID=$(sbatch --partition="$SLURM_PARTITION" \
  --job-name="$JOB_NAME" \
  --time="$TIME_LIMIT" \
  --output="${REPO_ROOT}/logs/%x-%j.out" \
  --error="${REPO_ROOT}/logs/%x-%j.err" \
  --export=ALL,ENV_NAME="$ENV_NAME",WORKDIR="$WORKDIR",TASK="$TASK",IMG_SIZE="$IMG_SIZE",DISEASE="$DISEASE",AE_CKPT_PATH="$AE_CKPT_PATH",AE_CONFIG_PATH="$AE_CONFIG_PATH",LATENT_SCALE_FACTOR="$LATENT_SCALE_FACTOR",LR="$LR",WEIGHT_DECAY="$WEIGHT_DECAY",LDM_BASE_CH="$LDM_BASE_CH",GRAD_CLIP="$GRAD_CLIP",BATCH_PER_DEVICE="$BATCH_PER_DEVICE",EPOCHS="$EPOCHS",LOG_EVERY="$LOG_EVERY",SAMPLE_EVERY="$SAMPLE_EVERY",SAMPLE_BATCH_SIZE="$SAMPLE_BATCH_SIZE",LDM_CH_MULTS="$LDM_CH_MULTS",LDM_NUM_RES_BLOCKS="$LDM_NUM_RES_BLOCKS",LDM_ATTN_RES="$LDM_ATTN_RES",WANDB="$WANDB",WANDB_PROJECT="$WANDB_PROJECT",WANDB_ENTITY="$WANDB_ENTITY",WANDB_TAGS="$WANDB_TAGS",WANDB_RUN_GROUP="$WANDB_RUN_GROUP",WANDB_NAME="$WANDB_NAME",RUN_NAME="$RUN_NAME",TRAINING_MODE="$TRAINING_MODE",GIT_HASH="$GIT_HASH",GIT_BRANCH="$GIT_BRANCH",GIT_PARENT="$GIT_PARENT",USE_EMA="$USE_EMA",EMA_DECAY="$EMA_DECAY" \
  slurm_scripts/cxr_ldm.slurm "${OTHER_ARGS[@]}" | awk '{print $4}')
status_line "🎉 Submitted" "Job ID: $JOB_ID"
status_line "📝 Logs at" "${REPO_ROOT}/logs/${JOB_NAME}-${JOB_ID}.out"

# Run script
#./launchers/single_runs/ldm/train_ldm_normal.sh full_train \
#  --batch_per_device 2 \
#  --ae_ckpt_path runs/unified-ae-proto-increase-ae-autoencoder-1f2a36b-20260110-013819/20260110-013836/ckpts/last.flax \
#  --ae_config_path runs/unified-ae-proto-increase-ae-autoencoder-1f2a36b-20260110-013819/20260110-013836/run_meta.json \
#  --latent_scale_factor 0.99999905 \
#  --wandb_project cxr-ldm-composition \
#  --workdir "${HOME}/cluster_staging/ldm-increase-capacity-normal"