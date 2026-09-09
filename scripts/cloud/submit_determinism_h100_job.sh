#!/usr/bin/env bash
# LAUNCH the H100 determinism-portability check (dev plan/91 §6.2) as a Vertex AI Custom Job:
# 1x H100 (a3-highgpu-1g), on-demand, runs as spa-worker, boots run_determinism_h100.sh,
# AUTO-TERMINATES. Mirrors submit_m10_bench_job.sh.
#
# WHAT IT ANSWERS. The determinism shims are architecture-agnostic code, but every measurement behind
# them is one A5000 in one process. Three checks, and A is the gate:
#   A  cuBLAS: are GEMMs bit-exact run to run on Hopper with CUBLAS_WORKSPACE_CONFIG UNSET? The RFD3
#      shim's whole cost-neutrality rests on yes. cuBLAS may pick split-k with atomics for some shapes.
#   B  RFD3 generation end to end, 2 processes per arm.
#   C  OpenFold3 on of3_triton.yml, the CLOUD runner-yaml the A5000 work never tested.
#
# COST. Deliberately tiny: K=4 at L=100 and one 76 aa refold, so the bill is startup and image pull,
# not compute. Expect ~30-45 min billed, roughly $6-10 on a3-highgpu-1g.
#
# ⚠️ The job clones REPO_REF from GitHub, so the shims must be PUSHED before submitting. Verify with
#    git -C <public repo> status -sb  (should not say "ahead").
#
# Usage:
#   DRY_RUN=1 ./submit_determinism_h100_job.sh              # print the CustomJobSpec; create NOTHING
#   ./submit_determinism_h100_job.sh                        # submit, us-central1
#   REGION=us-west1 ./submit_determinism_h100_job.sh        # quota is 1 job PER REGION, not 2 fungible
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
OF3_CKPT_URI="${OF3_CKPT_URI:-$BUCKET/weights/of3-p2-155k.pt}"
RESULTS_URI="${RESULTS_URI:-}"                      # empty -> run_determinism_h100.sh's own default
DISK_GB="${DISK_GB:-150}"
STRATEGY="${STRATEGY:-ONDEMAND}"
NAME="${NAME:-spa-det-h100-$(date -u +%Y%m%d-%H%M%S)}"
GCLOUD="${GCLOUD:-gcloud}"

. "$(dirname "${BASH_SOURCE[0]}")/_pin_run_env.sh"

BOOT="set -e; ${BOOT_CHECKOUT} && bash /opt/spa/scripts/cloud/run_determinism_h100.sh"

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
add_env OF3_CKPT_URI "${OF3_CKPT_URI}"
add_env RESULTS_URI "${RESULTS_URI}"
add_env REPO_REF "${REPO_REF}"

case "${STRATEGY}" in
  ONDEMAND|STANDARD|on-demand|"") : ;;
  *) printf 'scheduling:\n  strategy: %s\n' "${STRATEGY}" >> "${CFG}" ;;
esac

echo ">>> CustomJobSpec (${CFG}):"; sed 's/^/    /' "${CFG}"
echo ">>> name=${NAME} region=${REGION} sa=${SA} image=${IMAGE} ref=${REPO_REF}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo ">>> DRY_RUN=1 — not submitting. Real command would be:"
  echo "    ${GCLOUD} ai custom-jobs create --project=${PROJECT} --region=${REGION} --display-name=${NAME} --service-account=${SA} --config=${CFG}"
  exit 0
fi

echo ">>> Submitting Vertex Custom Job (this provisions the H100, ~\$6-10) ..."
exec "${GCLOUD}" ai custom-jobs create \
  --project="${PROJECT}" --region="${REGION}" \
  --display-name="${NAME}" \
  --service-account="${SA}" \
  --config="${CFG}"
