#!/usr/bin/env bash
# Set up a conda environment for running SPA inference: torch, RFdiffusion3 (via foundry),
# atomworks, ESM3, and this package itself. Downloads the RFdiffusion3 base checkpoint too.
#
# Usage:
#   bash scripts/setup/install_env.sh
#
# Options:
#   --env-name NAME     conda environment name (default: spa)
#   --with-clss         also install CLSS (only needed for the 1x32 variant)
#   --skip-checkpoint   skip downloading the RFdiffusion3 base checkpoint
#
# Requires: conda (or miniconda), an NVIDIA GPU with a CUDA 12.4-compatible driver.

set -euo pipefail

ENV_NAME="spa"
WITH_CLSS=0
SKIP_CHECKPOINT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --with-clss) WITH_CLSS=1; shift ;;
    --skip-checkpoint) SKIP_CHECKPOINT=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# SPA_PROJECT_ROOT: the same portable, env-var-driven root that configs/paths/default.yaml uses
# for rfd3_ckpt / proteinmpnn_repo / openfold3_ckpt (default: $HOME/projects/spa). Placing things
# here — not inside this repo — means the Hydra configs find them with zero path overrides.
SPA_PROJECT_ROOT="${SPA_PROJECT_ROOT:-$HOME/projects/spa}"
DEPS_DIR="$SPA_PROJECT_ROOT/needed_repos"
MODELS_ROOT="$SPA_PROJECT_ROOT/models"

if [[ "$REPO_ROOT" != "$SPA_PROJECT_ROOT/structure-prompt-adapter" ]]; then
  echo "Warning: this repo is at $REPO_ROOT, not \$SPA_PROJECT_ROOT/structure-prompt-adapter" >&2
  echo "         ($SPA_PROJECT_ROOT/structure-prompt-adapter). Some Hydra config defaults" >&2
  echo "         (e.g. openfold3_runner_yaml) assume that exact layout — either move this" >&2
  echo "         checkout there, or set SPA_PROJECT_ROOT to this repo's actual parent directory." >&2
fi

# Pinned dependency commits — this is the exact combination validated to work together
# (torch 2.5.1+cu124, no cuequivariance). Do not float these to "latest main".
FOUNDRY_COMMIT="e412591cd651093badd68ca78b93591f2d40f46f"
ATOMWORKS_COMMIT="54f17711d70244a3be53ec9c8267edf8a5e8c0d6"
ESM_COMMIT="82ee35553d39169d678f784c8d3f8712ffd7d2c4"
CLSS_COMMIT="b0a42ff41028665c9418bad5afe2413b137b2ac1"

RFD3_CKPT_URL="https://files.ipd.uw.edu/pub/rfd3/rfd3_foundry_2025_12_01_remapped.ckpt"

command -v conda >/dev/null 2>&1 || {
  echo "conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html" >&2
  exit 1
}

echo "==> Creating conda env '$ENV_NAME' (python 3.12)"
conda create -y -n "$ENV_NAME" -c conda-forge python=3.12

echo "==> Installing torch 2.5.1+cu124"
conda run -n "$ENV_NAME" pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

mkdir -p "$DEPS_DIR"

clone_pinned () {
  local url="$1" dir="$2" commit="$3"
  if [[ -d "$dir/.git" ]]; then
    echo "==> $dir already present, skipping clone"
  else
    echo "==> Cloning $url @ $commit"
    git clone "$url" "$dir"
    git -C "$dir" checkout --quiet "$commit"
  fi
}

clone_pinned "https://github.com/RosettaCommons/atomworks.git" "$DEPS_DIR/atomworks" "$ATOMWORKS_COMMIT"
clone_pinned "https://github.com/RosettaCommons/foundry.git" "$DEPS_DIR/foundry" "$FOUNDRY_COMMIT"
clone_pinned "https://github.com/Biohub/esm.git" "$DEPS_DIR/esm" "$ESM_COMMIT"

echo "==> Installing atomworks (editable)"
conda run -n "$ENV_NAME" pip install -e "$DEPS_DIR/atomworks"

echo "==> Installing RFdiffusion3 (rc-foundry[rfd3], editable)"
conda run -n "$ENV_NAME" pip install -e "$DEPS_DIR/foundry[rfd3]"

echo "==> Installing ESM3 (editable)"
conda run -n "$ENV_NAME" pip install -e "$DEPS_DIR/esm"

if [[ "$WITH_CLSS" -eq 1 ]]; then
  clone_pinned "https://github.com/guyyanai/CLSS.git" "$DEPS_DIR/CLSS" "$CLSS_COMMIT"
  echo "==> Installing CLSS (editable) — needed for the 1x32 variant"
  conda run -n "$ENV_NAME" pip install -e "$DEPS_DIR/CLSS"
fi

echo "==> Installing SPA itself (editable)"
conda run -n "$ENV_NAME" pip install -e "$REPO_ROOT"

if [[ "$SKIP_CHECKPOINT" -eq 0 ]]; then
  RFD3_CKPT_DIR="$MODELS_ROOT/rfdiffusion3"
  mkdir -p "$RFD3_CKPT_DIR"
  if [[ -f "$RFD3_CKPT_DIR/rfd3_latest.ckpt" ]]; then
    echo "==> RFdiffusion3 checkpoint already present, skipping download"
  else
    echo "==> Downloading RFdiffusion3 base checkpoint (multi-GB, may take a while)"
    curl -fL "$RFD3_CKPT_URL" -o "$RFD3_CKPT_DIR/rfd3_latest.ckpt"
  fi
fi

cat <<EOF

Done. Next steps:
  conda activate $ENV_NAME

SPA adapter weights are already in models/ (part of this repo).
RFdiffusion3 base checkpoint: $MODELS_ROOT/rfdiffusion3/rfd3_latest.ckpt
  (this matches configs/paths/default.yaml's default \$SPA_PROJECT_ROOT/models/rfdiffusion3/ —
  no path override needed when running scripts/eval/generate.py etc., as long as SPA_PROJECT_ROOT
  is unset or set the same way here and at run time)

Note: ESM3 weights auto-download from Hugging Face on first use. Even though they're
MIT-licensed / ungated, you still need a free Hugging Face account and a Read token:
  huggingface-cli login
EOF
