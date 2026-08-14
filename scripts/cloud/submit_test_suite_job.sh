#!/usr/bin/env bash
# LAUNCH the full test-suite run (dev plan/69 W1/Option-B safety net) as a Vertex AI Custom Job: 1x H100
# (a3-highgpu-1g), on-demand, spa-worker SA, runs scripts/cloud/run_test_suite.sh, stages the pytest log
# to GCS, AUTO-TERMINATES. Mirrors submit_smoke_job.sh. Used as a before/after baseline around the
# harness.py host-unfreeze change so a regression is caught by CI, not asserted from a diff read.
#
# PREREQUISITES:
#   - PUBLIC repo PUSHED @ $REPO_REF (the job git-clones run_test_suite.sh + the package)
#   - spa-combined image already in Artifact Registry (used by every other cloud job in this project)
#   - PINNED GCS: $BUCKET/weights/rfd3_latest.ckpt
#
# Usage:
#   DRY_RUN=1 ./submit_test_suite_job.sh                # print the spec + gcloud cmd; create NOTHING
#   ./submit_test_suite_job.sh                           # real launch
#
# Cost: no real training/eval compute, just pytest against a loaded checkpoint (test_harness.py's
# `trained` fixture is a 60-step toy overfit, shared across its 4 tests; test_train_resumes_end_to_end
# is 10 toy steps). Dominated by H100 provisioning (~5-7 min per the trivial-baseline job); expect
# single-digit dollars total.
set -euo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
REGION="${REGION:-us-central1}"                    # same idle H100 region submit_smoke_job.sh defaults to
IMAGE="${IMAGE:-us-central1-docker.pkg.dev/spa-dev-499900/spa/spa-combined:0.1.0}"
SA="${SA:-spa-worker@spa-dev-499900.iam.gserviceaccount.com}"
REPO_URL="${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}"
REPO_REF="${REPO_REF:-main}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
DISK_GB="${DISK_GB:-150}"                          # matches submit_smoke_job.sh; no training cache needed
STRATEGY="${STRATEGY:-ONDEMAND}"
NAME="${NAME:-spa-test-suite-$(date -u +%Y%m%d-%H%M%S)}"
GCLOUD="${GCLOUD:-gcloud}"                          # PATH lookup, portable across machines; override to a full path if not on PATH

# Pin the code revision and container digest (see _pin_run_env.sh). Must come AFTER IMAGE / REPO_URL /
# GCLOUD are set and BEFORE BOOT is built.
. "$(dirname "${BASH_SOURCE[0]}")/_pin_run_env.sh"

BOOT="set -e; ${BOOT_CHECKOUT} && bash /opt/spa/scripts/cloud/run_test_suite.sh"

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
add_env REPO_REF "${REPO_REF}"

case "${STRATEGY}" in
  ONDEMAND|STANDARD|on-demand|"") : ;;
  *) printf 'scheduling:\n  strategy: %s\n' "${STRATEGY}" >> "${CFG}" ;;
esac

echo ">>> CustomJobSpec (${CFG}):"; sed 's/^/    /' "${CFG}"
echo ">>> name=${NAME} region=${REGION} sa=${SA} image=${IMAGE}"
echo ">>> ckpt=${RFD3_CKPT_URI} disk=${DISK_GB}GB strategy=${STRATEGY}"

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
