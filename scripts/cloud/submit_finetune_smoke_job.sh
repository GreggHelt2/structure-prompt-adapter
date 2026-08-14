#!/usr/bin/env bash
# LAUNCH Option B's fine-tuning SMOKE TEST (dev plan/69 SS4.8 item 1) as a Vertex AI Custom Job: 1x
# H100 (a3-highgpu-1g), on-demand, spa-worker SA, runs scripts/cloud/run_finetune_smoke.sh, stages the
# train log to GCS, AUTO-TERMINATES. Mirrors submit_smoke_job.sh. Validates that unfreezing RFD3 via
# harness.py's build_optimizer (commit 620e750) trains stably (no NaN, no blowup) and that the new
# host-checkpoint plumbing actually produces its artifacts -- NOT the real Option B experiment, which
# still needs the fold-choice decision (plan/69 SS4.8 item 2, Gregg's call, deliberately decoupled).
#
# PREREQUISITES:
#   - PUBLIC repo PUSHED @ $REPO_REF, including the build_optimizer change (the job clones from GitHub)
#   - spa-combined image already in Artifact Registry (used by every other cloud job in this project)
#   - PINNED GCS: $BUCKET/weights/rfd3_latest.ckpt
#   - Run the test-suite regression job (submit_test_suite_job.sh) clean FIRST -- this smoke test
#     assumes the unfreeze mechanism already has CPU + GPU test coverage passing, it does not re-derive
#     correctness, only convergence behavior on the real model.
#
# Usage:
#   DRY_RUN=1 ./submit_finetune_smoke_job.sh                          # print the spec; create NOTHING
#   ./submit_finetune_smoke_job.sh                                     # real launch, default 200 steps
#   HOST_LR=5e-6 MAX_STEPS=500 ./submit_finetune_smoke_job.sh          # override the smoke-test knobs
#
# Cost: toy path only (no CDDB/ESM3-cache/split data to fetch), 200 steps of a tiny synthetic-overfit
# example. Dominated by H100 provisioning (~5-7 min); expect single-digit dollars total.
set -euo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
REGION="${REGION:-us-west1}"                       # the region with real H100 capacity (dev plan/69 SS5: us-central1 hit a resource error 2026-08-14)
IMAGE="${IMAGE:-us-central1-docker.pkg.dev/spa-dev-499900/spa/spa-combined:0.1.0}"
SA="${SA:-spa-worker@spa-dev-499900.iam.gserviceaccount.com}"
REPO_URL="${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}"
REPO_REF="${REPO_REF:-main}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
HOST_LR="${HOST_LR:-1.0e-5}"
MAX_STEPS="${MAX_STEPS:-200}"
CKPT_EVERY="${CKPT_EVERY:-50}"
DISK_GB="${DISK_GB:-150}"                          # matches submit_smoke_job.sh; no training cache needed
STRATEGY="${STRATEGY:-ONDEMAND}"
NAME="${NAME:-spa-finetune-smoke-$(date -u +%Y%m%d-%H%M%S)}"
GCLOUD="${GCLOUD:-gcloud}"                          # PATH lookup, portable across machines; override to a full path if not on PATH

# Pin the code revision and container digest (see _pin_run_env.sh). Must come AFTER IMAGE / REPO_URL /
# GCLOUD are set and BEFORE BOOT is built.
. "$(dirname "${BASH_SOURCE[0]}")/_pin_run_env.sh"

BOOT="set -e; ${BOOT_CHECKOUT} && bash /opt/spa/scripts/cloud/run_finetune_smoke.sh"

CFG="$(mktemp)"   # no --suffix: GNU-only flag, absent on macOS/BSD mktemp; gcloud --config doesn't need the extension
cat > "${CFG}" <<YAML
workerPoolSpecs:
  - machineSpec:
      machineType: a3-highgpu-1g
      acceleratorType: NVIDIA_H100_80GB
      acceleratorCount: 1
    replicaCount: 1
    diskSpec:
      bootDiskType: pd-ssd
      bootDiskSizeGb: ${DISK_GB}
    containerSpec:
      imageUri: ${IMAGE}
      command: ["bash", "-c"]
      args: ["${BOOT}"]
      env:
YAML

add_env(){ [ -z "${2:-}" ] && return 0; printf '        - name: %s\n          value: "%s"\n' "$1" "$2" >> "${CFG}"; }
add_env PROJECT "${PROJECT}"
add_env BUCKET "${BUCKET}"
add_env RFD3_CKPT_URI "${RFD3_CKPT_URI}"
add_env HOST_LR "${HOST_LR}"
add_env MAX_STEPS "${MAX_STEPS}"
add_env CKPT_EVERY "${CKPT_EVERY}"
add_env REPO_REF "${REPO_REF}"

case "${STRATEGY}" in
  ONDEMAND|STANDARD|on-demand|"") : ;;
  *) printf 'scheduling:\n  strategy: %s\n' "${STRATEGY}" >> "${CFG}" ;;
esac

echo ">>> CustomJobSpec (${CFG}):"; sed 's/^/    /' "${CFG}"
echo ">>> name=${NAME} region=${REGION} sa=${SA} image=${IMAGE}"
echo ">>> host_lr=${HOST_LR} max_steps=${MAX_STEPS} ckpt_every=${CKPT_EVERY} disk=${DISK_GB}GB strategy=${STRATEGY}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo ">>> DRY_RUN=1 — not submitting. Real command would be:"
  echo "    ${GCLOUD} ai custom-jobs create --project=${PROJECT} --region=${REGION} --display-name=${NAME} --service-account=${SA} --config=${CFG}"
  exit 0
fi

echo ">>> Submitting Vertex Custom Job (this provisions the H100) ..."
exec "${GCLOUD}" ai custom-jobs create \
  --project="${PROJECT}" --region="${REGION}" \
  --display-name="${NAME}" \
  --service-account="${SA}" \
  --config="${CFG}"
