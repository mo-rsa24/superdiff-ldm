#!/usr/bin/env bash
#SBATCH --job-name=cxr-ae
#SBATCH --partition=bigbatch
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  RED=${RED-$(printf '\033[31m')}
  GRN=${GRN-$(printf '\033[32m')}
  YEL=${YEL-$(printf '\033[33m')}
  BLU=${BLU-$(printf '\033[34m')}
  MAG=${MAG-$(printf '\033[35m')}
  CYN=${CYN-$(printf '\033[36m')}
  WHT=${WHT-$(printf '\033[37m')}
  BLD=${BLD-$(printf '\033[1m')}
  RST=${RST-$(printf '\033[0m')}
else
  RED=${RED-}; GRN=${GRN-}; YEL=${YEL-}; BLU=${BLU-}; MAG=${MAG-}; CYN=${CYN-}; WHT=${WHT-}; BLD=${BLD-}; RST=${RST-}
fi
# ----------------------------
# Pretty printing helpers
# ----------------------------
RED=$'\e[31m'; GRN=$'\e[32m'; YLW=$'\e[33m'; BLU=$'\e[34m'; RST=$'\e[0m'; BLD=$'\e[1m';
banner(){ printf "\n${BLU}${BLD}== %s ==${RST}\n" "$*"; }
kv(){ printf "  ${CYN}%-22s${RST} %s\n" "$1" "$2"; }
ok(){ printf "${GRN}%s${RST}\n" "$*"; }

START_TS=$(date -Iseconds)

# ----------------------------
# Defaults (override via env)
# ----------------------------
WORKDIR="${WORKDIR:-$PWD}"
PYFILE="${PYFILE:-run/autoencoder.py}"
DATA_ROOT="/datasets/mmolefe/cleaned"

TASK="${TASK:-TB}"
SPLIT="${SPLIT:-train}"
IMG_SIZE="${IMG_SIZE:-256}"

# Overfit / tiny controls
OVERFIT_ONE="${OVERFIT_ONE:-1}"
OVERFIT_K="${OVERFIT_K:-0}"
REPEAT_LEN="${REPEAT_LEN:-500}"

# Model Arch
CH_MULTS="${CH_MULTS:-128,256,512}"
NUM_RES_BLOCKS="${NUM_RES_BLOCKS:-2}"
DROPOUT="${DROPOUT:-0.0}"
# --- MODIFIED: Increased default latent channels ---
Z_CHANNELS="${Z_CHANNELS:-4}"

# Loss settings
# --- MODIFIED: Increased KL weight to a meaningful value for regularization ---
KL_WEIGHT="${KL_WEIGHT:-1.0e-5}"
PIXEL_WEIGHT="${PIXEL_WEIGHT:-1.0}"
DISC_START="${DISC_START:-50001}"
DISC_WEIGHT="${DISC_WEIGHT:-0.5}"
PERCEPTUAL_WEIGHT="${PERCEPTUAL_WEIGHT:-0.0}"

# Training
LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
EPOCHS="${EPOCHS:-100}"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-4}"
SAMPLE_EVERY="${SAMPLE_EVERY:-1}"
SEED="${SEED:-0}"

# Experiment naming / output
OUTPUT_ROOT="${OUTPUT_ROOT:-runs}"
EXP_NAME="${EXP_NAME:-cxr_ae}"
RUN_NAME="${RUN_NAME:-}"
RESUME_DIR="${RESUME_DIR:-}"

# WandB
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-cxr-ae}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_TAGS="${WANDB_TAGS:-slurm}"
WANDB_ID="${WANDB_ID:-}"

# ----------------------------
# Environment & paths
# ----------------------------
mkdir -p "$WORKDIR/logs"
cd "$WORKDIR"
export PYTHONPATH="$WORKDIR${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CPP_MIN_LOG_LEVEL=3
source ~/.bashrc
mamba activate "${ENV_NAME:?Must set ENV_NAME}"
ok "Activated micromamba environment: ${ENV_NAME}"

# ----------------------------
# Normalize list-like vars
# ----------------------------
CH_MULTS="${CH_MULTS//:/,}"
WANDB_TAGS="${WANDB_TAGS//:/,}"

# ----------------------------
# Build args array safely
# ----------------------------
ARGS=(
  --data_root "$DATA_ROOT"
  --task "$TASK" --split "$SPLIT" --img_size "$IMG_SIZE"
  --ch_mults "$CH_MULTS" --num_res_blocks "$NUM_RES_BLOCKS" --dropout "$DROPOUT" --z_channels "$Z_CHANNELS"
  --kl_weight "$KL_WEIGHT" --pixel_weight "$PIXEL_WEIGHT" --disc_start "$DISC_START" --disc_weight "$DISC_WEIGHT" --perceptual_weight "$PERCEPTUAL_WEIGHT"
  --lr "$LR" --weight_decay "$WEIGHT_DECAY" --grad_clip "$GRAD_CLIP"
  --epochs "$EPOCHS" --batch_per_device "$BATCH_PER_DEVICE"
  --seed "$SEED" --sample_every "$SAMPLE_EVERY"
  --output_root "$OUTPUT_ROOT" --exp_name "$EXP_NAME"
   --base_ch "$BASE_CH" \
  --ch_mults "$CH_MULTS" \
  --attn_res "$ATTN_RES" \
  --z_channels "$Z_CHANNELS" \
  --embed_dim "$EMBED_DIM" \
  --kl_weight "$KL_WEIGHT"
)

# Overfit/tiny
if [[ "$OVERFIT_ONE" == "1" ]]; then
  ARGS+=( --overfit_one --repeat_len "$REPEAT_LEN" )
elif (( OVERFIT_K > 0 )); then
  ARGS+=( --overfit_k "$OVERFIT_K" )
fi

# Optional run name / resume
[[ -n "$RUN_NAME"    ]] && ARGS+=( --run_name "$RUN_NAME" )
[[ -n "$RESUME_DIR"  ]] && ARGS+=( --resume_dir "$RESUME_DIR" )

# WandB
if [[ "$WANDB" == "1" ]]; then
  ARGS+=( --wandb --wandb_project "$WANDB_PROJECT" )
  [[ -n "$WANDB_ENTITY" ]] && ARGS+=( --wandb_entity "$WANDB_ENTITY" )
  [[ -n "$WANDB_TAGS"   ]] && ARGS+=( --wandb_tags "$WANDB_TAGS" )
  [[ -n "$WANDB_ID"     ]] && ARGS+=( --wandb_id "$WANDB_ID" )
fi

# ----------------------------
# Info & Run
# ----------------------------
banner "SLURM Job Info"
kv "JOB_ID"         "${SLURM_JOB_ID:-<none>}"
kv "JOB_NAME"       "${SLURM_JOB_NAME:-<none>}"
kv "Node"           "$(hostname)"

banner "Training Info"
kv "PYFILE"         "$PYFILE"
kv "TASK"           "$TASK"
kv "IMG_SIZE"       "$IMG_SIZE"
kv "CH_MULTS"       "$CH_MULTS"
kv "Z_CHANNELS"     "$Z_CHANNELS"
kv "KL_WEIGHT"      "$KL_WEIGHT"
kv "LR"             "$LR"
kv "EPOCHS"         "$EPOCHS"
kv "BATCHxDEVS"     "$BATCH_PER_DEVICE x $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)"
kv "OVERFIT"        "(of1=$OVERFIT_ONE, tinyK=$OVERFIT_K)"
kv "W&B"            "$WANDB (${WANDB_PROJECT}) tags=[$WANDB_TAGS]"

banner "Run"
srun python -u "$PYFILE" "${ARGS[@]}"
RC=$?

banner "Done (ExitCode: $RC)"
[[ $RC -eq 0 ]] && ok "Success." || { err "Failure (exit $RC)"; exit "$RC"; }
