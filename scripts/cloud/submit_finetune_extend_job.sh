#!/usr/bin/env bash
# LAUNCH Option B's fine-tune EXTEND (dev plan/69 SS4.15) as a Vertex AI Custom Job: 1x H100
# (a3-highgpu-1g), on-demand, spa-worker SA, runs scripts/cloud/run_finetune_extend.sh, resumes the
# existing checkpoint and continues to a larger MAX_STEPS, stages results back to the SAME checkpoint
# directory, AUTO-TERMINATES. Mirrors submit_finetune_real_job.sh.
#
# PREREQUISITES:
#   - PUBLIC repo PUSHED @ $REPO_REF, including run_finetune_extend.sh
#   - spa-combined image already in Artifact Registry
#   - PINNED GCS: $BUCKET/weights/rfd3_latest.ckpt
#   - The neighborhood staged at $BUCKET/eval/finetune_neighborhood/A0A7S3EB45/
#   - The existing checkpoint at $BUCKET/checkpoints/finetune-A0A7S3EB45-20260814-045849/spa_Nx1536_last.pt
#     (renamed 2026-08-14 from spa_C_last.pt; configs/variant/C_n_by_1536.yaml's name: field changed
#     C -> Nx1536 in the same session, public ed4847b, so this is the exact filename harness.py's
#     auto-resume now looks for)
#
# Usage:
#   DRY_RUN=1 MAX_STEPS=3000 ./submit_finetune_extend_job.sh    # print the spec; create NOTHING
#   MAX_STEPS=3000 ./submit_finetune_extend_job.sh              # real launch, resume 1000 -> 3000
#   MAX_STEPS=5000 ./submit_finetune_extend_job.sh              # a later call, resume 3000 -> 5000
#
# Cost: only the INCREMENTAL steps (resume, not restart) -- ~2000 more steps at roughly the original
# run's rate (~1 h / ~$11 for 1000 steps), so ~$20-25 for 1000->3000, another ~$20-25 for 3000->5000.
set -euo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
REGION="${REGION:-us-west1}"
IMAGE="${IMAGE:-us-central1-docker.pkg.dev/spa-dev-499900/spa/spa-combined:0.1.0}"
SA="${SA:-spa-worker@spa-dev-499900.iam.gserviceaccount.com}"
REPO_URL="${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}"
REPO_REF="${REPO_REF:-main}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
NEIGHBORHOOD_URI="${NEIGHBORHOOD_URI:-$BUCKET/eval/finetune_neighborhood/A0A7S3EB45}"
CKPT_OUT_URI="${CKPT_OUT_URI:-$BUCKET/checkpoints/finetune-A0A7S3EB45-20260814-045849}"
RESUME_FROM_CKPT_URI="${RESUME_FROM_CKPT_URI:-$CKPT_OUT_URI/spa_Nx1536_last.pt}"
HOST_LR="${HOST_LR:-1.0e-5}"
MAX_STEPS="${MAX_STEPS:?set MAX_STEPS to the new target total step count (e.g. 3000)}"
CKPT_EVERY="${CKPT_EVERY:-100}"
DISK_GB="${DISK_GB:-150}"
STRATEGY="${STRATEGY:-ONDEMAND}"
NAME="${NAME:-spa-finetune-extend-$(date -u +%Y%m%d-%H%M%S)}"
GCLOUD="${GCLOUD:-gcloud}"

. "$(dirname "${BASH_SOURCE[0]}")/_pin_run_env.sh"

BOOT="set -e; ${BOOT_CHECKOUT} && bash /opt/spa/scripts/cloud/run_finetune_extend.sh"

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
add_env CKPT_OUT_URI "${CKPT_OUT_URI}"
add_env RESUME_FROM_CKPT_URI "${RESUME_FROM_CKPT_URI}"
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
echo ">>> resume_from=${RESUME_FROM_CKPT_URI} max_steps=${MAX_STEPS} ckpt_every=${CKPT_EVERY} disk=${DISK_GB}GB strategy=${STRATEGY}"

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
