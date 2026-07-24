#!/usr/bin/env bash
# Set up the SPA TRAINING tier. This builds ON the inference tier ("spa" env, see install_env.sh):
# it reuses that env, adds the experiment tracker (wandb, the [tracking] extra), points you at the
# CDDB dataset (public via NVIDIA NGC), and verifies the pipeline with a toy ESM3-cache + a short
# local training run.
#
# Scope note: FULL-SCALE training is a cloud / H100 job by design — the ~251 GB ESM3 prompt cache and
# the real ~30k-step training run live in scripts/cloud/ (run_cache_gen.sh + run_train.sh, submitted via
# submit_*_job.sh). Locally the A5000 (24 GB) only makes a small test cache and runs short training
# sanity checks. This script sets up that local path and documents the cloud one.
#
# The training pipeline is:
#   1) CDDB dataset            -> training_data/proteina-atomistica_data_vrelease/... (NVIDIA NGC; see below)
#   2) build_splits.py         -> train/validate/test manifests under training_data/  (needs full CDDB)
#   3) gen_esm3_cache.py       -> per-residue (N,1536) ESM3 prompts  (data=toy local / data=cddb cloud)
#   4) train.py                -> frozen RFdiffusion3 host + trainable SPA adapter
# The toy path (steps 3-4 with data=toy) needs NO CDDB download — it uses training_data/toy/.
#
# Usage:
#   bash scripts/setup/install_train_env.sh                  # env + wandb + toy smoke; GUIDE the CDDB pull
#   bash scripts/setup/install_train_env.sh --download-data  # ALSO download+untar CDDB from NGC
#
# Options:
#   --env-name NAME     conda env to reuse/create (default: spa) — the same env install_env.sh builds
#   --download-data     download + untar the full CDDB dataset from NGC (needs NGC auth; see the note)
#   --skip-verify       skip the toy ESM3-cache + short local training smoke test
#
# Requires: conda; an NVIDIA GPU with a CUDA 12.4-compatible driver. For --download-data: the NGC CLI
# plus a free NVIDIA account API key (a human step this script cannot automate — it prints how).

set -euo pipefail

ENV_NAME="spa"
DOWNLOAD_DATA=0
SKIP_VERIFY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --download-data) DOWNLOAD_DATA=1; shift ;;
    --skip-verify) SKIP_VERIFY=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPA_PROJECT_ROOT="${SPA_PROJECT_ROOT:-$HOME/projects/spa}"
DATA_ROOT="$SPA_PROJECT_ROOT/training_data"

# CDDB on NGC — the exact resource scripts/cloud/{run_cache_gen,run_train}.sh pull (10.96 GB tarball).
NGC_RESOURCE="nvidia/clara/proteina-atomistica_data:release"

command -v conda >/dev/null 2>&1 || {
  echo "conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html" >&2
  exit 1
}

# 1) The inference env is a hard prerequisite (training freezes the same RFdiffusion3 host + reuses the
#    whole torch/foundry/esm stack). Build it via install_env.sh if it isn't there yet — that also fetches
#    the RFD3 base checkpoint the frozen host loads during training.
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "==> Reusing existing conda env '$ENV_NAME' (inference tier)"
else
  echo "==> conda env '$ENV_NAME' not found -> building the inference tier first via install_env.sh"
  bash "$SETUP_DIR/install_env.sh" --env-name "$ENV_NAME"
fi

# 2) Training extras: wandb (experiment tracker, the [tracking] extra). [eval] is kept for parity with
#    the inference tier (idempotent). tracker defaults to null in configs/train/default.yaml, so wandb is
#    only used if you pass train.tracker=wandb (then run `wandb login` once).
echo "==> Installing SPA training extras ([eval,tracking] -> tmtools + wandb)"
conda run -n "$ENV_NAME" pip install -e "$REPO_ROOT[eval,tracking]"

# 3) CDDB dataset (NVIDIA NGC). ~455k single-chain structures; ~11 GB tarball -> tens of GB untarred.
#    Extracted to match configs/data/cddb.yaml's pdb_dir default (no path override needed):
#      $DATA_ROOT/proteina-atomistica_data_vrelease/atomistica_data_release/pdb
if [[ "$DOWNLOAD_DATA" -eq 1 ]]; then
  echo "==> Downloading CDDB from NGC ($NGC_RESOURCE)"
  command -v ngc >/dev/null 2>&1 || {
    echo "ngc CLI not found. Install it (https://org.ngc.nvidia.com/setup/installers/cli), then run" >&2
    echo "  ngc config set   # paste the API key from a FREE NVIDIA account (org: nvidia)" >&2
    echo "and re-run with --download-data. (Or set NGC_CLI_API_KEY + NGC_CLI_ORG=nvidia in the env.)" >&2
    exit 1
  }
  if [[ -d "$DATA_ROOT/proteina-atomistica_data_vrelease/atomistica_data_release/pdb" ]]; then
    echo "==> CDDB already present under $DATA_ROOT, skipping download"
  else
    # GATE: resource reachable (fails fast on missing/no-auth NGC creds before the big download).
    ngc registry resource info "$NGC_RESOURCE" --files >/dev/null
    mkdir -p "$DATA_ROOT"
    ngc registry resource download-version "$NGC_RESOURCE" --dest "$DATA_ROOT"
    TARBALL="$(find "$DATA_ROOT" -name 'atomistica_cd_dataset.tar.gz' | head -1)"
    echo "==> Untarring $TARBALL"
    tar -xzf "$TARBALL" -C "$(dirname "$TARBALL")"
  fi
else
  cat <<'NOTE'
==> Skipping CDDB download (default). The toy smoke below needs no dataset.
    To train on the full CDDB dataset:
      1. Create a FREE NVIDIA account and get an NGC API key: https://ngc.nvidia.com/setup/api-key
      2. Install the NGC CLI (https://org.ngc.nvidia.com/setup/installers/cli) and run `ngc config set`
         (org: nvidia), OR export NGC_CLI_API_KEY=<key> NGC_CLI_ORG=nvidia
      3. Re-run this script with --download-data  (downloads + untars ~11 GB into training_data/)
    Then build splits and the ESM3 cache:
      conda run -n <env> python scripts/build_splits.py                       # train/validate/test manifests
      conda run -n <env> python scripts/gen_esm3_cache.py data=cddb hardware=cloud_h100   # full cache (cloud)
    Full-scale training runs on a cloud H100 (see scripts/cloud/run_train.sh + submit_train_job.sh).
NOTE
fi

# 4) Verify the training pipeline with a TOY end-to-end run (no CDDB needed): cache the 4 training_data/toy/
#    structures, then run 2 optimizer steps of real training on the frozen RFD3 host + SPA adapter.
if [[ "$SKIP_VERIFY" -eq 0 ]]; then
  SMOKE_DIR="$(mktemp -d)"
  echo "==> Toy ESM3 cache-gen (data=toy)"
  ( cd "$REPO_ROOT" && conda run -n "$ENV_NAME" python scripts/gen_esm3_cache.py data=toy )
  echo "==> Short training smoke (data=toy, 2 optimizer steps on the A5000)"
  # resume=false so a stale checkpoint from prior training on this box can't derail the smoke; ckpt_dir
  # + hydra.run.dir isolated to a temp so nothing lands in the repo's checkpoints/ or outputs/.
  ( cd "$REPO_ROOT" && conda run -n "$ENV_NAME" python scripts/train.py \
      variant=C_n_by_1536 data=toy hardware=local_a5000 \
      train.max_steps=2 train.warmup_steps=0 train.val_every_steps=0 \
      train.ckpt_every_steps=1000000 train.log_every_steps=1 train.tracker=null \
      train.resume=false hardware.grad_accum=1 \
      "train.ckpt_dir=$SMOKE_DIR/ckpt" "hydra.run.dir=$SMOKE_DIR/run" )
  rm -rf "$SMOKE_DIR"
  echo "==> Toy training smoke OK."
fi

cat <<EOF

Done. The '$ENV_NAME' env now has the training stack (inference tier + wandb).

Toy pipeline (no CDDB needed) — sanity-check any change:
  conda run -n $ENV_NAME python scripts/gen_esm3_cache.py data=toy
  conda run -n $ENV_NAME python scripts/train.py variant=C_n_by_1536 data=toy hardware=local_a5000 train.max_steps=50

Full pipeline (CDDB): download the dataset (--download-data), then
  conda run -n $ENV_NAME python scripts/build_splits.py
  conda run -n $ENV_NAME python scripts/gen_esm3_cache.py data=cddb hardware=cloud_h100   # ~251 GB, cloud H100
  conda run -n $ENV_NAME python scripts/train.py variant=C_n_by_1536 data=cddb hardware=cloud_h100
Full cache-gen + training are cloud/H100 jobs — see scripts/cloud/{run_cache_gen,run_train}.sh and the
submit_*_job.sh wrappers. To log to Weights & Biases, add train.tracker=wandb and run \`wandb login\` first.
EOF
