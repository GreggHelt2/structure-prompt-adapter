#!/usr/bin/env bash
# LAUNCH the ESM3 cache audit (dev plan/69, prompted by Gregg 2026-08-14) as a Vertex AI Custom Job: 1x
# H100 (a3-highgpu-1g), on-demand, spa-worker SA, runs scripts/cloud/run_check_esm3_cache.sh, stages the
# full tar listing to GCS, AUTO-TERMINATES. Mirrors every other submit_*.sh this week. No GPU work is
# actually done (tar -tf only lists entries) -- reuses the H100 image because it is the fastest already-
# proven path to something working, per Gregg's own instruction, not because the task needs a GPU.
#
# PREREQUISITES:
#   - PUBLIC repo PUSHED @ $REPO_REF
#   - spa-combined image already in Artifact Registry
#   - DISK_GB must exceed 234.67 GB (the tar itself) with real headroom; default below is generous
#
# Usage:
#   DRY_RUN=1 ./submit_check_esm3_cache_job.sh                                    # print the spec only
#   ./submit_check_esm3_cache_job.sh                                               # real launch
#   NEIGHBORHOOD_TSV_URI=/opt/spa-data/neighborhood.tsv ./submit_check_esm3_cache_job.sh  # also check coverage
#
# Cost: no GPU compute, just a same-region GCS->GCE download (234.67 GB, should be fast on GCP's
# internal network) plus a sequential tar listing. Dominated by H100 provisioning + download time;
# expect low tens of dollars for the provisioned H100 minutes even though it does no GPU work.
set -euo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
REGION="${REGION:-us-west1}"
IMAGE="${IMAGE:-us-central1-docker.pkg.dev/spa-dev-499900/spa/spa-combined:0.1.0}"
SA="${SA:-spa-worker@spa-dev-499900.iam.gserviceaccount.com}"
REPO_URL="${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}"
REPO_REF="${REPO_REF:-main}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
CACHE_TAR_URI="${CACHE_TAR_URI:-$BUCKET/esm3_cache.tar}"
NEIGHBORHOOD_TSV_URI="${NEIGHBORHOOD_TSV_URI:-}"
DISK_GB="${DISK_GB:-300}"                          # tar is 234.67 GB; real headroom for the listing file
STRATEGY="${STRATEGY:-ONDEMAND}"
NAME="${NAME:-spa-esm3-cache-audit-$(date -u +%Y%m%d-%H%M%S)}"
GCLOUD="${GCLOUD:-gcloud}"

. "$(dirname "${BASH_SOURCE[0]}")/_pin_run_env.sh"

BOOT="set -e; ${BOOT_CHECKOUT} && bash /opt/spa/scripts/cloud/run_check_esm3_cache.sh"

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
add_env CACHE_TAR_URI "${CACHE_TAR_URI}"
add_env NEIGHBORHOOD_TSV_URI "${NEIGHBORHOOD_TSV_URI}"
add_env REPO_REF "${REPO_REF}"

case "${STRATEGY}" in
  ONDEMAND|STANDARD|on-demand|"") : ;;
  *) printf 'scheduling:\n  strategy: %s\n' "${STRATEGY}" >> "${CFG}" ;;
esac

echo ">>> CustomJobSpec (${CFG}):"; sed 's/^/    /' "${CFG}"
echo ">>> name=${NAME} region=${REGION} sa=${SA} image=${IMAGE}"
echo ">>> cache_tar=${CACHE_TAR_URI} disk=${DISK_GB}GB strategy=${STRATEGY}"

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
