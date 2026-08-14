#!/usr/bin/env bash
# LAUNCH the Option B checkpoint reconstruction (dev plan/69 SS4.13, generalized 2026-08-14) as a
# Vertex AI Custom Job: 1x H100 (a3-highgpu-1g), on-demand, spa-worker SA, runs
# scripts/cloud/run_extract_finetuned_checkpoint.sh, stages the result to GCS, AUTO-TERMINATES.
# Mirrors submit_check_esm3_cache_job.sh. No GPU work is actually done (pure CPU torch.load/torch.save
# dict surgery) -- reuses the H100 image because it is the fastest already-proven path to something
# working, not because the task needs a GPU; swap to a cheaper CPU-only machine type if this becomes a
# frequent operation.
#
# PREREQUISITES:
#   - PUBLIC repo PUSHED @ $REPO_REF
#   - spa-combined image already in Artifact Registry
#   - SPA_CKPT_URI must point at a full-state SPA checkpoint saved WITH a fine-tuned host (i.e. from a
#     run with train.finetune_host_lr set) -- spa_<X>_last.pt or spa_<X>_stepN.pt, NOT a
#     spa_<X>_final.pt adapter-only export, which never carries a "host" key.
#
# Usage:
#   DRY_RUN=1 SPA_CKPT_URI=gs://.../spa_C_last.pt OUT_CKPT_URI=gs://.../rfd3_finetuned_full.ckpt \
#     ./submit_extract_finetuned_checkpoint_job.sh                # print the spec only
#   SPA_CKPT_URI=gs://.../spa_C_last.pt OUT_CKPT_URI=gs://.../rfd3_finetuned_full.ckpt \
#     ./submit_extract_finetuned_checkpoint_job.sh                # real launch
#
# Cost: no GPU compute, two same-region GCS->GCE downloads (SPA checkpoint + base RFD3 checkpoint,
# both a few GB) plus a CPU dict-splice and one upload, all at GCS-internal bandwidth. Dominated by
# H100 provisioning time; expect a few dollars, well under the week's cap.
set -euo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
REGION="${REGION:-us-west1}"
IMAGE="${IMAGE:-us-central1-docker.pkg.dev/spa-dev-499900/spa/spa-combined:0.1.0}"
SA="${SA:-spa-worker@spa-dev-499900.iam.gserviceaccount.com}"
REPO_URL="${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}"
REPO_REF="${REPO_REF:-main}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
SPA_CKPT_URI="${SPA_CKPT_URI:?set SPA_CKPT_URI to the SPA full-state fine-tune checkpoint}"
BASE_RFD3_CKPT_URI="${BASE_RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
OUT_CKPT_URI="${OUT_CKPT_URI:?set OUT_CKPT_URI to where the reconstructed checkpoint should land}"
DISK_GB="${DISK_GB:-100}"                          # two ~2.5 GB checkpoints in flight, generous headroom
STRATEGY="${STRATEGY:-ONDEMAND}"
NAME="${NAME:-spa-extract-finetuned-ckpt-$(date -u +%Y%m%d-%H%M%S)}"
GCLOUD="${GCLOUD:-gcloud}"

. "$(dirname "${BASH_SOURCE[0]}")/_pin_run_env.sh"

BOOT="set -e; ${BOOT_CHECKOUT} && bash /opt/spa/scripts/cloud/run_extract_finetuned_checkpoint.sh"

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
add_env SPA_CKPT_URI "${SPA_CKPT_URI}"
add_env BASE_RFD3_CKPT_URI "${BASE_RFD3_CKPT_URI}"
add_env OUT_CKPT_URI "${OUT_CKPT_URI}"
add_env REPO_REF "${REPO_REF}"

case "${STRATEGY}" in
  ONDEMAND|STANDARD|on-demand|"") : ;;
  *) printf 'scheduling:\n  strategy: %s\n' "${STRATEGY}" >> "${CFG}" ;;
esac

echo ">>> CustomJobSpec (${CFG}):"; sed 's/^/    /' "${CFG}"
echo ">>> name=${NAME} region=${REGION} sa=${SA} image=${IMAGE}"
echo ">>> spa_ckpt=${SPA_CKPT_URI} base_ckpt=${BASE_RFD3_CKPT_URI} out=${OUT_CKPT_URI} disk=${DISK_GB}GB strategy=${STRATEGY}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo ">>> DRY_RUN=1: not submitting. Real command would be:"
  echo "    ${GCLOUD} ai custom-jobs create --project=${PROJECT} --region=${REGION} --display-name=${NAME} --service-account=${SA} --config=${CFG}"
  exit 0
fi

echo ">>> Submitting Vertex Custom Job (this provisions the H100) ..."
exec "${GCLOUD}" ai custom-jobs create \
  --project="${PROJECT}" --region="${REGION}" \
  --display-name="${NAME}" \
  --service-account="${SA}" \
  --config="${CFG}"
