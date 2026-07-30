#!/usr/bin/env bash
# CFG Phase-0 sweep driver: 7 prompts x lambda {1,2} at a fixed design length.
#
# Measures how much of each coordinate step the SPA prompt accounts for, which is the quantity a
# classifier-free-guidance scale would multiply. See scripts/eval/probe_cfg_delta.py for the method.
#
# NOTE ON THE DESIGN: a fixed L isolates the prompt, but it also introduces a prompt-length confound
# (measured: Spearman rho(N, ratio) = -0.571 at lambda=1, -0.714 at lambda=2). Do not read the
# between-prompt spread as fold-to-fold variation without accounting for length.
#
# Usage (all knobs are env-overridable):
#   bash scripts/eval/run_cfg_phase0.sh
#   OUT=/tmp/p0 LENGTH=150 K=2 SEED=42 LAMBDAS="1.0 2.0" bash scripts/eval/run_cfg_phase0.sh
#
# Aggregate the JSONs it writes with the analysis script in the companion planning repo.
set -u

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"

# Prompt sources. Defaults point at a local CDDB/eval-external layout; override for another machine.
PDB_DIR="${PDB_DIR:-/home/user1/projects/spa/training_data/proteina-atomistica_data_vrelease/atomistica_data_release/pdb}"
EXTERNAL_DIR="${EXTERNAL_DIR:-/home/user1/projects/spa/training_data/eval_external}"

OUT="${OUT:-outputs/eval/cfg_phase0}"
CKPT="${CKPT:-models/spa-Nx1536-uncond.pt}"
LENGTH="${LENGTH:-150}"
K="${K:-2}"
SEED="${SEED:-42}"
LAMBDAS="${LAMBDAS:-1.0 2.0}"
CONDA_ENV="${CONDA_ENV:-spa-dev}"

mkdir -p "$OUT"

declare -A P=(
  [A0A522W419]="$PDB_DIR/AF-A0A522W419-F1-model_v4_esmfold_v1.pdb"
  [A0A7S3EB45]="$PDB_DIR/AF-A0A7S3EB45-F1-model_v4_esmfold_v1.pdb"
  [A0A1X7NTP0]="$PDB_DIR/AF-A0A1X7NTP0-F1-model_v4_esmfold_v1.pdb"
  [A0A090ME36]="$PDB_DIR/AF-A0A090ME36-F1-model_v4_esmfold_v1.pdb"
  [A0A6A0D1E8]="$PDB_DIR/AF-A0A6A0D1E8-F1-model_v4_esmfold_v1.pdb"
  [A0A2X2KHU0]="$PDB_DIR/AF-A0A2X2KHU0-F1-model_v4_esmfold_v1.pdb"
  [8SIU]="$EXTERNAL_DIR/8SIU.pdb"
)
TAGS="${TAGS:-A0A522W419 A0A7S3EB45 8SIU A0A1X7NTP0 A0A090ME36 A0A6A0D1E8 A0A2X2KHU0}"

one() {  # tag lambda
  local tag=$1 lam=$2
  local out="$OUT/p0_${tag}_lam${lam}.json"
  [ -f "$out" ] && { echo "SKIP $tag lam=$lam (exists)"; return; }
  if [ ! -f "${P[$tag]}" ]; then
    echo "MISSING prompt for $tag at ${P[$tag]} (set PDB_DIR / EXTERNAL_DIR)"; return
  fi
  echo "=== $tag  lambda=$lam ==="
  conda run -n "$CONDA_ENV" python scripts/eval/probe_cfg_delta.py \
    --prompt-pdb "${P[$tag]}" --ckpt "$CKPT" --length "$LENGTH" -K "$K" --seed "$SEED" \
    --lam "$lam" --out "$out" 2>&1 | grep -E "ratio_median|snr_median|prompt_shift_rms_A_max"
}

for lam in $LAMBDAS; do
  for t in $TAGS; do one "$t" "$lam"; done
done
echo "BATCH DONE -> $OUT"
