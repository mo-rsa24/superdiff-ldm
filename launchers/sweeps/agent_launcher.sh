#!/usr/bin/env bash
set -euo pipefail

# This script is called by the `wandb agent`.
# It parses hyperparameters, generates tags, and submits a SLURM job.

# --- Fixed Defaults ---
export ENV_NAME="jax115"
# NOTE: DATA_ROOT is now expected from the sweep .yaml file

# --- Parse W&B agent args and export as env vars ---
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

# --- Generate W&B Tags from Sweep Parameters ---
# This section automatically creates descriptive tags for better organization.
TAGS="sweep"
TRAINING_MODE="full_train" # Default mode

# Determine training mode from overfit flags
if [[ "${OVERFIT_ONE}" == "true" ]]; then TRAINING_MODE="overfit_one";
elif [[ "${OVERFIT_K}" -gt 0 ]]; then TRAINING_MODE="overfit_${OVERFIT_K}";
fi

# Add task and mode to tags
[[ -n "${TASK}" ]] && TAGS+=",${TASK,,}" # Add task, e.g., ",tb"
TAGS+=",${TRAINING_MODE}"

# If this is a z_channels sweep, add that tag too
[[ -n "${Z_CHANNELS}" ]] && TAGS+=",z${Z_CHANNELS}"

export WANDB_TAGS="$TAGS"
# --- End of Tag Generation ---


# --- SPECIAL LOGIC for LDM Z_CHANNELS SWEEP ---
# (This part remains unchanged)
if [[ -n "${Z_CHANNELS:-}" && -n "${VAE_RUNS_ROOT:-}" && "$PY_PROGRAM" == "run/ldm.py" ]]; then
    # ... (logic to find VAE artifacts) ...
fi

# --- Generate a Unique Run Name for W&B ---
export RUN_NAME="sweep-${TASK,,}-${TRAINING_MODE}-$(date +%s)"

# --- Submit Job ---
echo "LAUNCHING SWEEP RUN with TAGS: ${WANDB_TAGS}"
sbatch "$SLURM_SCRIPT"
