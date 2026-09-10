#!/usr/bin/env bash
# LAUNCH the CROSS-GPU K=1 portability check as a Vertex AI Custom Job: 1x H100 (a3-highgpu-1g),
# on-demand, runs as spa-worker, boots run_k1_cross_gpu_h100.sh, AUTO-TERMINATES.
#
# WHAT IT ANSWERS, and why the 2026-09-09 determinism job did not. That job verified both shims
# WITHIN Hopper and never compared an H100 structure to an A5000 one. dev results/44 section 4 still
# lists GPU model in the reproducibility identity as "yes, ASSUMED - untested across either", while
# dev plan/95 states it as settled. The reasoning is sound; the measurement does not exist.
#
# ⛔ ITS ARTIFACTS CANNOT BE REUSED. verdict.json kept check_b as COUNTS, not hashes or coordinates,
# and its OpenFold3 hash used of3_triton.yml against the A5000's of3_nokernel.yml, so that pair is
# confounded by kernel config. Triton cannot run on sm_86 at all, so that particular comparison is
# impossible rather than merely missing.
#
# ⛔ AND K=4 DESIGN 0 IS NOT A K=1 DESIGN: results/44 section 5c.6 measured them 0.506 A apart.
#
# ⭐ THE A5000 HALF ALREADY EXISTS AND IS VALIDATED: L=100, K=1, seed 0, deterministic gives md5
# 8fd129f939f6 across two independent invocations. Generate the rest with
# dev scripts/analysis/make_k1_cross_gpu_refs.sh BEFORE submitting, so both sides cover the same
# lengths through the same code path.
#
# ⭐ THIS JOB STAGES THE PDBs, not only a verdict. A verdict JSON is not an artifact.
#
# COST. Six single-design generations, so the bill is provisioning and image pull, not compute.
# Expect ~30-45 min billed, roughly $6-10 on a3-highgpu-1g.
#
# ⚠️ The job clones REPO_REF from GitHub, so run_k1_cross_gpu_h100.sh must be PUSHED before
#    submitting. Verify with: git -C <public repo> status -sb   (must not say "ahead").
#
# Usage:
#   DRY_RUN=1 ./submit_k1_cross_gpu_job.sh                 # print the CustomJobSpec; create NOTHING
#   ./submit_k1_cross_gpu_job.sh                           # submit, us-central1
#   REGION=us-west1 ./submit_k1_cross_gpu_job.sh           # quota is 1 job PER REGION, not 2 fungible
#   LENGTHS="100 150 208" ./submit_k1_cross_gpu_job.sh     # must match the A5000 refs
set -euo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
REGION="${REGION:-us-central1}"
TAG="${TAG:-0.1.0}"
IMAGE="${IMAGE:-us-central1-docker.pkg.dev/spa-dev-499900/spa/spa-combined:${TAG}}"
SA="${SA:-spa-worker@spa-dev-499900.iam.gserviceaccount.com}"
REPO_URL="${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}"
REPO_REF="${REPO_REF:-main}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
LENGTHS="${LENGTHS:-100 150 208}"
RESULTS_URI="${RESULTS_URI:-}"                      # empty -> run_k1_cross_gpu_h100.sh's own default
DISK_GB="${DISK_GB:-150}"
STRATEGY="${STRATEGY:-ONDEMAND}"
NAME="${NAME:-spa-k1xgpu-h100-$(date -u +%Y%m%d-%H%M%S)}"
GCLOUD="${GCLOUD:-gcloud}"

. "$(dirname "${BASH_SOURCE[0]}")/_pin_run_env.sh"

BOOT="set -e; ${BOOT_CHECKOUT} && bash /opt/spa/scripts/cloud/run_k1_cross_gpu_h100.sh"

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
add_env LENGTHS "${LENGTHS}"
add_env RESULTS_URI "${RESULTS_URI}"
add_env REPO_REF "${REPO_REF}"

case "${STRATEGY}" in
  ONDEMAND|STANDARD|on-demand|"") : ;;
  *) printf 'scheduling:\n  strategy: %s\n' "${STRATEGY}" >> "${CFG}" ;;
esac

echo ">>> CustomJobSpec (${CFG}):"; sed 's/^/    /' "${CFG}"
echo ">>> name=${NAME} region=${REGION} sa=${SA} image=${IMAGE} ref=${REPO_REF}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo ">>> DRY_RUN=1, not submitting. Real command would be:"
  echo "    ${GCLOUD} ai custom-jobs create --project=${PROJECT} --region=${REGION} --display-name=${NAME} --service-account=${SA} --config=${CFG}"
  exit 0
fi

echo ">>> Submitting Vertex Custom Job (this provisions the H100, ~\$6-10) ..."
exec "${GCLOUD}" ai custom-jobs create \
  --project="${PROJECT}" --region="${REGION}" \
  --display-name="${NAME}" \
  --service-account="${SA}" \
  --config="${CFG}"
