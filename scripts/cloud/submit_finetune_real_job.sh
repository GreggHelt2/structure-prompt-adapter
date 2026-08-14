#!/usr/bin/env bash
# LAUNCH Option B's REAL fine-tune (dev plan/69 SS4.11/4.12) as a Vertex AI Custom Job: 1x H100
# (a3-highgpu-1g), on-demand, spa-worker SA, runs scripts/cloud/run_finetune_real.sh, stages the
# checkpoints (not just a log) to GCS, AUTO-TERMINATES. Mirrors submit_finetune_smoke_job.sh, extended
# for the real data path: fine-tunes on the ~50-structure all-beta CDDB neighborhood of A0A7S3EB45,
# not the toy synthetic-overfit example the smoke test used.
#
# PREREQUISITES:
#   - PUBLIC repo PUSHED @ $REPO_REF, including scripts/gen_esm3_cache.py and the neighborhood-aware
#     run_finetune_real.sh (the job clones from GitHub)
#   - spa-combined image already in Artifact Registry
#   - PINNED GCS: $BUCKET/weights/rfd3_latest.ckpt
#   - The neighborhood staged at $BUCKET/eval/finetune_neighborhood/A0A7S3EB45/ (50 PDBs + a custom
#     train/manifest.parquet) -- done 2026-08-14, see docs/plan/69 SS4.11 in the dev repo
#   - The smoke test (submit_finetune_smoke_job.sh) already validated the unfreeze mechanism trains
#     stably; this run answers a different, harder question (does it learn something real from actual
#     CDDB structures), not "does the plumbing work"
#
# Usage:
#   DRY_RUN=1 ./submit_finetune_real_job.sh                          # print the spec; create NOTHING
#   ./submit_finetune_real_job.sh                                     # real launch, 1000 steps
#   MAX_STEPS=200 ./submit_finetune_real_job.sh                       # a shorter first look
#
# Cost: an ESM3 cache-gen pass over 50 tiny structures (minutes) plus up to 1,000 training steps on
# real CDDB data (not the trivial toy path) -- meaningfully more compute than the smoke test, but still
# far under the Option B budget (~$50-150 total, dev plan/69 SS4.2). Dominated by H100 provisioning
# (~5-7 min) plus real step time; expect tens of dollars, not the smoke test's single digits.
set -euo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
REGION="${REGION:-us-west1}"                       # the region with real H100 capacity (dev plan/69 SS5)
IMAGE="${IMAGE:-us-central1-docker.pkg.dev/spa-dev-499900/spa/spa-combined:0.1.0}"
SA="${SA:-spa-worker@spa-dev-499900.iam.gserviceaccount.com}"
REPO_URL="${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}"
REPO_REF="${REPO_REF:-main}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
NEIGHBORHOOD_URI="${NEIGHBORHOOD_URI:-$BUCKET/eval/finetune_neighborhood/A0A7S3EB45}"
HOST_LR="${HOST_LR:-1.0e-5}"
MAX_STEPS="${MAX_STEPS:-1000}"
CKPT_EVERY="${CKPT_EVERY:-100}"
DISK_GB="${DISK_GB:-150}"
STRATEGY="${STRATEGY:-ONDEMAND}"
NAME="${NAME:-spa-finetune-real-$(date -u +%Y%m%d-%H%M%S)}"
GCLOUD="${GCLOUD:-gcloud}"

# Pin the code revision and container digest (see _pin_run_env.sh). Must come AFTER IMAGE / REPO_URL /
# GCLOUD are set and BEFORE BOOT is built.
. "$(dirname "${BASH_SOURCE[0]}")/_pin_run_env.sh"

BOOT="set -e; ${BOOT_CHECKOUT} && bash /opt/spa/scripts/cloud/run_finetune_real.sh"

CFG="$(mktemp)"
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
add_env NEIGHBORHOOD_URI "${NEIGHBORHOOD_URI}"
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
echo ">>> neighborhood=${NEIGHBORHOOD_URI} host_lr=${HOST_LR} max_steps=${MAX_STEPS} ckpt_every=${CKPT_EVERY} disk=${DISK_GB}GB strategy=${STRATEGY}"

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
