#!/usr/bin/env bash
set -euo pipefail

# This script is called by the `wandb agent`.
# It parses hyperparameters and submits a SLURM job.

# --- Fixed Defaults ---
export ENV_NAME="jax115"
export DATA_ROOT="/datasets/mmolefe/cleaned"
export BATCH_PER_DEVICE="16"
export EPOCHS="1000"

# --- Parse W&B agent args and export as env vars ---
# Example: --lr=0.001 -> export LR="0.001"
for arg in "$@"; do
    if [[ "$arg" == --program* ]]; then
        PY_PROGRAM="${arg#*=}"
        continue
    fi
    VAR_NAME=$(echo "$arg" | cut -d'=' -f1 | sed 's/--//' | tr '[:lower:]' '[:upper:]')
    VAR_VALUE="${arg#*=}"
    export "$VAR_NAME"="$VAR_VALUE"
done

# --- Determine SLURM script to use ---
SLURM_SCRIPT=""
if [[ "$PY_PROGRAM" == "run/autoencoder.py" ]]; then
    SLURM_SCRIPT="slurm_scripts/cxr_ae.slurm"
    export WANDB_PROJECT="cxr-vae-sweeps"
elif [[ "$PY_PROGRAM" == "run/ldm.py" ]]; then
    SLURM_SCRIPT="slurm_scripts/cxr_ldm.slurm"
    export WANDB_PROJECT="cxr-ldm-sweeps"
else
    echo "ERROR: Unknown program '$PY_PROGRAM'" >&2
    exit 1
fi

# --- SPECIAL LOGIC for LDM Z_CHANNELS SWEEP ---
# If Z_CHANNELS and VAE_RUNS_ROOT are set, find the corresponding VAE checkpoint.
if [[ -n "${Z_CHANNELS:-}" && -n "${VAE_RUNS_ROOT:-}" && "$PY_PROGRAM" == "run/ldm.py" ]]; then
    echo ">>> LDM z_channels sweep detected. Finding VAE artifacts..."
    TASK_LOWER=$(echo "$TASK" | tr '[:upper:]' '[:lower:]')

    # Find the most recent run directory for this task and z_channels
    # Example search pattern: "runs/ae_tb_z1_*"
    SEARCH_PATTERN="${VAE_RUNS_ROOT}/ae_${TASK_LOWER}_full_kl_*_zchannels_${Z_CHANNELS}/*"

    # Find the latest directory matching the pattern
    VAE_RUN_DIR=$(find $VAE_RUNS_ROOT -type d -name "ae_${TASK_LOWER}_full_kl_*_zchannels_${Z_CHANNELS}" | sort -r | head -n 1)

    if [[ -z "$VAE_RUN_DIR" || ! -d "$VAE_RUN_DIR" ]]; then
        echo "ERROR: Could not find a VAE run directory for TASK=${TASK} and Z_CHANNELS=${Z_CHANNELS}" >&2
        echo "Searched with pattern: ${SEARCH_PATTERN}" >&2
        exit 1
    fi

    export AE_CKPT_PATH="${VAE_RUN_DIR}/ckpts/last.flax"
    export AE_CONFIG_PATH="${VAE_RUN_DIR}/run_meta.json"
    SCALE_FACTOR_FILE="${VAE_RUN_DIR}/latent_scale_factor.txt"

    if [[ ! -f "$SCALE_FACTOR_FILE" ]]; then
        echo "ERROR: latent_scale_factor.txt not found in ${VAE_RUN_DIR}" >&2
        echo "Please run scripts/find_latent_scale.py for this VAE first." >&2
        exit 1
    fi
    export LATENT_SCALE_FACTOR=$(cat "$SCALE_FACTOR_FILE")

    echo "Found VAE Checkpoint: $AE_CKPT_PATH"
    echo "Found Latent Scale Factor: $LATENT_SCALE_FACTOR"
fi

# --- Generate a Unique Run Name for W&B ---
export RUN_NAME="sweep-$(date +%s)-${RANDOM}"

# --- Submit Job ---
echo "LAUNCHING SWEEP RUN via ${SLURM_SCRIPT}"
sbatch "$SLURM_SCRIPT"