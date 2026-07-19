#!/usr/bin/env bash
# Set up a conda environment for the SPA validation pipeline: ProteinMPNN (inverse folding) and
# OpenFold3 (refold + designability scoring). Kept separate from the inference env ("spa", see
# install_env.sh) since OpenFold3's dependencies (cuequivariance-ops-torch-cu12, pytorch-lightning)
# don't need to coexist with RFdiffusion3/ESM3, and OpenFold3 wants substantially more VRAM.
#
# The env name matches spa.eval.openfold3.OF3Refolder's own default (conda_env="spa-verify-of3"),
# so run_flywheel.py's OF3 step needs NO conda_env override — only eval.proteinmpnn.conda_env does,
# since ProteinMPNN's default is "current interpreter" (see configs/eval/default.yaml).
#
# Usage:
#   bash scripts/setup/install_validation_env.sh
#
# Options:
#   --env-name NAME     conda environment name (default: spa-verify-of3)
#   --skip-checkpoint   skip downloading the OpenFold3 checkpoint
#
# Requires: conda, an NVIDIA GPU with 32GB+ VRAM recommended (OpenFold3's own docs cite an
# A100 40GB as typical) — this is a materially bigger requirement than the inference tier.

set -euo pipefail

ENV_NAME="spa-verify-of3"
SKIP_CHECKPOINT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --skip-checkpoint) SKIP_CHECKPOINT=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SPA_PROJECT_ROOT="${SPA_PROJECT_ROOT:-$HOME/projects/spa}"
DEPS_DIR="$SPA_PROJECT_ROOT/needed_repos"
OF3_MODELS_DIR="$SPA_PROJECT_ROOT/models/openfold3"

# Pinned dependency commits — do not float these to "latest main".
PROTEINMPNN_COMMIT="8907e6671bfbfc92303b5f79c4b5e6ce47cdef57"
OPENFOLD3_COMMIT="e583ecee04c1cc34ed85dba60d0ce52ba330ed0d"

command -v conda >/dev/null 2>&1 || {
  echo "conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html" >&2
  exit 1
}

echo "==> Creating conda env '$ENV_NAME' (python 3.12)"
conda create -y -n "$ENV_NAME" -c conda-forge python=3.12

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

# ProteinMPNN: no pip install — it's invoked as a script directly (no setup.py/pyproject.toml in
# the repo), with weights bundled in-repo. This path matches configs/paths/default.yaml's
# proteinmpnn_repo default ($SPA_PROJECT_ROOT/needed_repos/ProteinMPNN) exactly — zero overrides.
clone_pinned "https://github.com/dauparas/ProteinMPNN.git" "$DEPS_DIR/ProteinMPNN" "$PROTEINMPNN_COMMIT"

# OpenFold3: editable-installed like the inference-tier deps.
clone_pinned "https://github.com/aqlaboratory/openfold-3.git" "$DEPS_DIR/openfold-3" "$OPENFOLD3_COMMIT"
echo "==> Installing OpenFold3 (editable)"
conda run -n "$ENV_NAME" pip install -e "$DEPS_DIR/openfold-3"

if [[ "$SKIP_CHECKPOINT" -eq 0 ]]; then
  mkdir -p "$OF3_MODELS_DIR"
  echo "==> Downloading OpenFold3 checkpoint into $OF3_MODELS_DIR"
  # setup_openfold has no --param-directory flag; redirect via a small --config JSON instead of
  # its ~/.openfold3 default, to match configs/paths/default.yaml's openfold3_ckpt expectation.
  OF3_SETUP_CONFIG="$(mktemp)"
  cat > "$OF3_SETUP_CONFIG" <<JSON
{"openfold_cache": "$OF3_MODELS_DIR", "param_directory": "$OF3_MODELS_DIR",
 "selected_parameters": "default", "force_download_parameters": false,
 "run_integration_tests": false}
JSON
  conda run -n "$ENV_NAME" setup_openfold --config "$OF3_SETUP_CONFIG"
  rm -f "$OF3_SETUP_CONFIG"
  echo "==> NOTE: verify the downloaded checkpoint filename matches paths.openfold3_ckpt's default"
  echo "    (\$SPA_PROJECT_ROOT/models/openfold3/of3-p2-155k.pt) — not independently confirmed by"
  echo "    this script; override paths.openfold3_ckpt=<actual path> if it differs."
fi

cat <<EOF

Done.

ProteinMPNN: $DEPS_DIR/ProteinMPNN (matches paths.proteinmpnn_repo default — no override needed)
OpenFold3 checkpoint dir: $OF3_MODELS_DIR

Run the full RFD3+SPA -> ProteinMPNN -> OpenFold3 flywheel from the INFERENCE env ('spa'), letting
it shell out to this env for both downstream steps:

  conda activate spa
  python scripts/eval/run_flywheel.py \\
      variant=C_n_by_1536 eval.ckpt=$REPO_ROOT/models/spa-Nx1536-uncond.pt \\
      eval.prompt_pdb=/path/to/prompt.pdb 'eval.conditions=[baseline,spa]' \\
      'eval.lambda_scale=[0.5,1.0]' eval.num_designs=8 eval.length=100 \\
      eval.proteinmpnn.conda_env=$ENV_NAME \\
      +eval.flywheel.refolder._target_=spa.eval.openfold3.OF3Refolder \\
      +eval.flywheel.refolder.ckpt_path=\${paths.openfold3_ckpt} \\
      +eval.flywheel.refolder.runner_yaml=\${paths.openfold3_runner_yaml} \\
      +eval.flywheel.refolder.out_dir=\${eval.out_dir}

(OpenFold3's own conda_env defaults to '$ENV_NAME' already, so no override needed there if you
kept the default --env-name.)

Reminder: OpenFold3 wants substantially more VRAM than the inference tier — 32GB+ recommended
(its own docs cite an A100 40GB as typical).
EOF
