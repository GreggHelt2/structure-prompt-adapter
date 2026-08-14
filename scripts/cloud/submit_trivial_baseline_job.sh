#!/usr/bin/env bash
# LAUNCH the TRIVIAL DOMAIN BASELINE (dev plan/69 W1 closure, comparator 4) as a Vertex AI Custom Job:
# 1x H100 (a3-highgpu-1g), on-demand, spa-worker SA, runs scripts/cloud/run_trivial_baseline.sh, stages
# results to GCS, AUTO-TERMINATES. Mirrors submit_smoke_job.sh. No SPA checkpoint is fetched or needed.
#
# PREREQUISITES:
#   - PUBLIC repo PUSHED @ $REPO_REF (the job git-clones run_trivial_baseline.sh + the package)
#   - spa-combined image already in Artifact Registry (used by every other cloud job in this project)
#   - PINNED GCS: $BUCKET/weights/{rfd3_latest.ckpt,of3-p2-155k.pt}
#   - The target PDB already staged at $BUCKET/eval/b1_full/prep/A0A7S3EB45.pdb (reused from the
#     existing B1-full prep, dev plan/69 §3.3 -- verify it is actually there before DRY_RUN=0; if not,
#     override SOURCE_PDB_URI to point at wherever the CDDB copy of A0A7S3EB45 has been staged instead)
#
# Usage:
#   DRY_RUN=1 ./submit_trivial_baseline_job.sh                # print the spec + gcloud cmd; create NOTHING
#   REGION=us-west1 ./submit_trivial_baseline_job.sh           # real launch
#
# Cost (dev plan/69 §3.8): K=8/N=8 on one 125-residue fold, single condition, no SPA. Raw compute is a
# few dollars by this project's own $0.30/design cost-basis; budget under $20 including instance
# spin-up and checkpoint transfer, well under the week's $300 cap.
set -euo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
REGION="${REGION:-us-central1}"                    # same idle H100 region submit_smoke_job.sh defaults to
IMAGE="${IMAGE:-us-central1-docker.pkg.dev/spa-dev-499900/spa/spa-combined:0.1.0}"
SA="${SA:-spa-worker@spa-dev-499900.iam.gserviceaccount.com}"
REPO_URL="${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}"
REPO_REF="${REPO_REF:-main}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
OF3_CKPT_URI="${OF3_CKPT_URI:-$BUCKET/weights/of3-p2-155k.pt}"
SOURCE_PDB_URI="${SOURCE_PDB_URI:-$BUCKET/eval/b1_full/prep/A0A7S3EB45.pdb}"
CONTIG="${CONTIG:-A1-125}"
K="${K:-8}"; NSEQ="${NSEQ:-8}"; SEED="${SEED:-42}"
DISK_GB="${DISK_GB:-150}"                          # matches submit_smoke_job.sh; no training cache needed
STRATEGY="${STRATEGY:-ONDEMAND}"
NAME="${NAME:-spa-trivial-baseline-$(date -u +%Y%m%d-%H%M%S)}"
GCLOUD="${GCLOUD:-gcloud}"                          # override to the full SDK path if not on PATH

# Pin the code revision and container digest (see _pin_run_env.sh). Must come AFTER IMAGE / REPO_URL /
# GCLOUD are set and BEFORE BOOT is built.
. "$(dirname "${BASH_SOURCE[0]}")/_pin_run_env.sh"

BOOT="set -e; ${BOOT_CHECKOUT} && bash /opt/spa/scripts/cloud/run_trivial_baseline.sh"

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
add_env OF3_CKPT_URI "${OF3_CKPT_URI}"
add_env SOURCE_PDB_URI "${SOURCE_PDB_URI}"
add_env CONTIG "${CONTIG}"
add_env K "${K}"
add_env NSEQ "${NSEQ}"
add_env SEED "${SEED}"
add_env REPO_REF "${REPO_REF}"

case "${STRATEGY}" in
  ONDEMAND|STANDARD|on-demand|"") : ;;
  *) printf 'scheduling:\n  strategy: %s\n' "${STRATEGY}" >> "${CFG}" ;;
esac

echo ">>> CustomJobSpec (${CFG}):"; sed 's/^/    /' "${CFG}"
echo ">>> name=${NAME} region=${REGION} sa=${SA} image=${IMAGE}"
echo ">>> source_pdb=${SOURCE_PDB_URI} contig=${CONTIG} K=${K} N=${NSEQ} seed=${SEED} disk=${DISK_GB}GB strategy=${STRATEGY}"

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
