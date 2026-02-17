#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TRAJ_SCRIPT="$PROJECT_ROOT/scripts/trajectory_dynamics_experiment.py"

PROMPT_A="a cat on the left"
PROMPT_B="a dog on the right"
BEST_PROMPT=""
OUTPUT_ROOT=""

BASE_STEPS=50
BASE_GUIDANCE="4.5"
BASE_LIFT="0.1"
BASE_NUM_SEEDS=4
BASE_SEED_START=42

SWEEP_STEPS=60
SWEEP_NUM_SEEDS=6
SWEEP_GUIDANCES=("3.8" "4.5" "5.2")
SWEEP_LIFTS=("0.0" "0.1" "0.2")

RUN_COARSE=1
RUN_SWEEP=1
RUN_TRIAD=1

CANDIDATES=()

usage() {
  cat <<'EOF'
Run manual semantic-prompt search + validation with trajectory_dynamics_experiment.py.

This script performs up to three stages:
1) Coarse candidate search over monolithic prompts C for A,B
2) Hyperparameter sweep for a selected best prompt C*
3) Triadic logical composition test: A ^ B ^ C*

Usage:
  bash scripts/run_semantic_prompt_tests.sh [options]

Options:
  --prompt-a TEXT               Base prompt A (default: "a cat on the left")
  --prompt-b TEXT               Base prompt B (default: "a dog on the right")
  --candidate TEXT              Add monolithic candidate prompt C (repeatable)
  --candidate-file PATH         Load candidate prompts from file (one per line, '#' comments allowed)
  --best-prompt TEXT            Prompt C* used for sweep/triad (defaults to first candidate)
  --output-root PATH            Root output directory (default: experiments/semantic_prompt_search_<timestamp>)

  --steps N                     Coarse steps (default: 50)
  --guidance X                  Coarse guidance (default: 4.5)
  --lift X                      Coarse lift (default: 0.1)
  --num-seeds N                 Coarse seed count (default: 4)
  --seed-start N                Starting seed (default: 42)

  --sweep-steps N               Sweep steps (default: 60)
  --sweep-num-seeds N           Sweep seed count (default: 6)
  --sweep-guidances CSV         Sweep guidances (default: 3.8,4.5,5.2)
  --sweep-lifts CSV             Sweep lifts (default: 0.0,0.1,0.2)

  --skip-coarse                 Skip coarse candidate stage
  --skip-sweep                  Skip hyperparameter sweep stage
  --skip-triad                  Skip triadic A^B^C* stage
  --coarse-only                 Run only coarse stage
  -h, --help                    Show this help

Examples:
  bash scripts/run_semantic_prompt_tests.sh \
    --prompt-a "a cat on the left" \
    --prompt-b "a dog on the right" \
    --candidate "a cat on the left and a dog on the right" \
    --candidate "two animals in one frame, cat on left, dog on right"

  bash scripts/run_semantic_prompt_tests.sh \
    --candidate-file /tmp/candidates.txt \
    --best-prompt "a cat on the left and a dog on the right"
EOF
}

slugify() {
  local s="$1"
  s="$(printf '%s' "$s" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//; s/__+/_/g')"
  [[ -z "$s" ]] && s="prompt"
  printf '%.80s' "$s"
}

csv_to_array() {
  local csv="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a out_ref <<<"$csv"
}

load_candidate_file() {
  local file="$1"
  if [[ ! -f "$file" ]]; then
    echo "ERROR: candidate file not found: $file" >&2
    exit 1
  fi
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    line="$(printf '%s' "$line" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')"
    [[ -z "$line" ]] && continue
    CANDIDATES+=("$line")
  done <"$file"
}

run_pairwise_case() {
  local monolithic="$1"
  local out_dir="$2"
  local seed="$3"
  local steps="$4"
  local guidance="$5"
  local lift="$6"

  mkdir -p "$out_dir"
  echo "[RUN] pairwise seed=$seed guidance=$guidance lift=$lift"
  echo "      A=\"$PROMPT_A\""
  echo "      B=\"$PROMPT_B\""
  echo "      C=\"$monolithic\""
  echo "      out=$out_dir"

  python "$TRAJ_SCRIPT" \
    --prompt-a "$PROMPT_A" \
    --prompt-b "$PROMPT_B" \
    --monolithic "$monolithic" \
    --superdiff-variant fm_ode \
    --steps "$steps" \
    --guidance "$guidance" \
    --lift "$lift" \
    --seed "$seed" \
    --num-seeds 1 \
    --batch-size 1 \
    --no-poe \
    --output-dir "$out_dir"
}

run_triad_case() {
  local best_prompt="$1"
  local out_dir="$2"
  local seed="$3"
  local steps="$4"
  local guidance="$5"
  local lift="$6"

  mkdir -p "$out_dir"
  echo "[RUN] triad seed=$seed guidance=$guidance lift=$lift"
  echo "      prompts=(\"$PROMPT_A\" \"${PROMPT_B}\" \"${best_prompt}\")"
  echo "      out=$out_dir"

  python "$TRAJ_SCRIPT" \
    --prompts "$PROMPT_A" "$PROMPT_B" "$best_prompt" \
    --steps "$steps" \
    --guidance "$guidance" \
    --lift "$lift" \
    --seed "$seed" \
    --num-seeds 1 \
    --batch-size 1 \
    --no-poe \
    --output-dir "$out_dir"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt-a) PROMPT_A="$2"; shift 2 ;;
    --prompt-b) PROMPT_B="$2"; shift 2 ;;
    --candidate) CANDIDATES+=("$2"); shift 2 ;;
    --candidate-file) load_candidate_file "$2"; shift 2 ;;
    --best-prompt) BEST_PROMPT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;

    --steps) BASE_STEPS="$2"; shift 2 ;;
    --guidance) BASE_GUIDANCE="$2"; shift 2 ;;
    --lift) BASE_LIFT="$2"; shift 2 ;;
    --num-seeds) BASE_NUM_SEEDS="$2"; shift 2 ;;
    --seed-start|--seed) BASE_SEED_START="$2"; shift 2 ;;

    --sweep-steps) SWEEP_STEPS="$2"; shift 2 ;;
    --sweep-num-seeds) SWEEP_NUM_SEEDS="$2"; shift 2 ;;
    --sweep-guidances) csv_to_array "$2" SWEEP_GUIDANCES; shift 2 ;;
    --sweep-lifts) csv_to_array "$2" SWEEP_LIFTS; shift 2 ;;

    --skip-coarse) RUN_COARSE=0; shift ;;
    --skip-sweep) RUN_SWEEP=0; shift ;;
    --skip-triad) RUN_TRIAD=0; shift ;;
    --coarse-only) RUN_SWEEP=0; RUN_TRIAD=0; shift ;;

    -h|--help) usage; exit 0 ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$TRAJ_SCRIPT" ]]; then
  echo "ERROR: trajectory script not found: $TRAJ_SCRIPT" >&2
  exit 1
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  OUTPUT_ROOT="$PROJECT_ROOT/experiments/semantic_prompt_search_$ts"
fi
mkdir -p "$OUTPUT_ROOT"

if (( RUN_COARSE == 1 )) && [[ ${#CANDIDATES[@]} -eq 0 ]]; then
  CANDIDATES=(
    "$PROMPT_A and $PROMPT_B"
    "two subjects in one frame, $PROMPT_A and $PROMPT_B"
    "a realistic composition with $PROMPT_A and $PROMPT_B"
    "a coherent scene showing $PROMPT_A and $PROMPT_B"
  )
fi

if [[ -z "$BEST_PROMPT" ]] && (( RUN_COARSE == 1 )) && [[ ${#CANDIDATES[@]} -gt 0 ]]; then
  BEST_PROMPT="${CANDIDATES[0]}"
  echo "[INFO] --best-prompt not provided; defaulting to first candidate:"
  echo "       $BEST_PROMPT"
fi

if (( (RUN_SWEEP == 1 || RUN_TRIAD == 1) )) && [[ -z "$BEST_PROMPT" ]]; then
  echo "ERROR: BEST_PROMPT is required when sweep/triad is enabled." >&2
  echo "Set --best-prompt or run coarse search with candidates." >&2
  exit 1
fi

echo "============================================================"
echo "Semantic Prompt Test Runner"
echo "Output root: $OUTPUT_ROOT"
echo "Prompt A: $PROMPT_A"
echo "Prompt B: $PROMPT_B"
echo "Run coarse: $RUN_COARSE"
echo "Run sweep:  $RUN_SWEEP"
echo "Run triad:  $RUN_TRIAD"
echo "============================================================"

if (( RUN_COARSE == 1 )); then
  echo
  echo "=== Stage 1: Coarse Candidate Search ==="
  echo "Candidates: ${#CANDIDATES[@]}"
  for i in "${!CANDIDATES[@]}"; do
    idx="$(printf '%02d' "$((i + 1))")"
    cand="${CANDIDATES[$i]}"
    cand_slug="$(slugify "$cand")"
    base_dir="$OUTPUT_ROOT/coarse/${idx}_${cand_slug}"
    for ((k = 0; k < BASE_NUM_SEEDS; k++)); do
      seed="$((BASE_SEED_START + k))"
      run_pairwise_case \
        "$cand" \
        "$base_dir/seed_${seed}" \
        "$seed" \
        "$BASE_STEPS" \
        "$BASE_GUIDANCE" \
        "$BASE_LIFT"
    done
  done
fi

if (( RUN_SWEEP == 1 )); then
  echo
  echo "=== Stage 2: Hyperparameter Sweep on Best Prompt ==="
  echo "Best prompt: $BEST_PROMPT"
  for g in "${SWEEP_GUIDANCES[@]}"; do
    for l in "${SWEEP_LIFTS[@]}"; do
      glabel="g${g//./p}_l${l//./p}"
      base_dir="$OUTPUT_ROOT/sweep/$glabel"
      for ((k = 0; k < SWEEP_NUM_SEEDS; k++)); do
        seed="$((BASE_SEED_START + k))"
        run_pairwise_case \
          "$BEST_PROMPT" \
          "$base_dir/seed_${seed}" \
          "$seed" \
          "$SWEEP_STEPS" \
          "$g" \
          "$l"
      done
    done
  done
fi

if (( RUN_TRIAD == 1 )); then
  echo
  echo "=== Stage 3: Triadic A ^ B ^ C* Validation ==="
  echo "Best prompt: $BEST_PROMPT"
  triad_slug="$(slugify "$BEST_PROMPT")"
  base_dir="$OUTPUT_ROOT/triad/$triad_slug"
  for ((k = 0; k < SWEEP_NUM_SEEDS; k++)); do
    seed="$((BASE_SEED_START + k))"
    run_triad_case \
      "$BEST_PROMPT" \
      "$base_dir/seed_${seed}" \
      "$seed" \
      "$SWEEP_STEPS" \
      "$BASE_GUIDANCE" \
      "$BASE_LIFT"
  done
fi

echo
echo "Done. Results saved under:"
echo "  $OUTPUT_ROOT"
echo
echo "Inspect each run's summary at:"
echo "  <run_dir>/summary.json"

