#!/usr/bin/env bash
# LAUNCH the THREE-WAY A⊕B⊕C sweep as a Vertex AI Custom Job (dev docs/plan/23): 1x H100 (a3-highgpu-1g),
# on-demand, runs as spa-worker, boots run_threeway_sweep.sh (RFD3±SPA generate + adherence, and — for
# STAGE=designability — ProteinMPNN→OF3 scRMSD) on the spa-combined image, then AUTO-TERMINATES. Mirrors
# submit_smoke_job.sh but forwards the three-way sweep knobs. On-demand H100 = 1/region (Spot/Flex quota=0).
#
# PREREQUISITES:
#   - spa-combined:${TAG} pushed to Artifact Registry
#   - PUBLIC repo pushed to GitHub @ $REPO_REF (the job git-clones it for run_threeway_sweep.sh + the drivers)
#   - PINNED GCS artifacts: weights/{rfd3_latest.ckpt,of3-p2-155k.pt}, checkpoints/spa-Nx1536-multigran/spa_C_final.pt
#   - eval/threeway/prep/ staged with the motif-source + target-fold PDBs (scripts/eval/prep_threeway.py)
#
# Usage:
#   DRY_RUN=1 ./submit_threeway_job.sh                                   # print the CustomJobSpec; create NOTHING
#   STAGE=adherence REGION=us-west1 MOTIFS=... FOLDS=... ./submit_threeway_job.sh   # Stage-1 screen
#   STAGE=designability WINNERS=... REGION=us-central1 ./submit_threeway_job.sh     # Stage-2 on winners
set -euo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
REGION="${REGION:-us-west1}"
TAG="${TAG:-0.1.0}"
IMAGE="${IMAGE:-us-central1-docker.pkg.dev/spa-dev-499900/spa/spa-combined:${TAG}}"
SA="${SA:-spa-worker@spa-dev-499900.iam.gserviceaccount.com}"
REPO_URL="${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}"
REPO_REF="${REPO_REF:-main}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
DISK_GB="${DISK_GB:-150}"                    # image + RFD3 (2.9G) + OF3 (2.3G) + SPA multigran + designs
STRATEGY="${STRATEGY:-ONDEMAND}"
NAME="${NAME:-spa-threeway-${STAGE:-adherence}-$(date -u +%Y%m%d-%H%M%S)}"
GCLOUD="${GCLOUD:-/home/user1/google-cloud-sdk/bin/gcloud}"

BOOT="set -e; git clone --depth 1 --branch ${REPO_REF} ${REPO_URL} /opt/spa && bash /opt/spa/scripts/cloud/run_threeway_sweep.sh"

CFG="$(mktemp --suffix=.yaml)"
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
add_env REPO_REF "${REPO_REF}"
# three-way sweep knobs (all omitted-when-empty -> the run script's defaults apply):
add_env STAGE "${STAGE:-}"
add_env MOTIFS "${MOTIFS:-}"
add_env FOLDS "${FOLDS:-}"
add_env LAYOUTS "${LAYOUTS:-}"
add_env LAMBDAS "${LAMBDAS:-}"
add_env ULEN "${ULEN:-}"
add_env CLEN "${CLEN:-}"
add_env K "${K:-}"
add_env NSEQ "${NSEQ:-}"
add_env SEEDS "${SEEDS:-}"
add_env WINNERS "${WINNERS:-}"
add_env PROTEINMPNN_SEED "${PROTEINMPNN_SEED:-}"
add_env LEAN_RESULTS "${LEAN_RESULTS:-}"

case "${STRATEGY}" in
  ONDEMAND|STANDARD|on-demand|"") : ;;
  *) printf 'scheduling:\n  strategy: %s\n' "${STRATEGY}" >> "${CFG}" ;;
esac

echo ">>> CustomJobSpec (${CFG}):"; sed 's/^/    /' "${CFG}"
echo ">>> name=${NAME} region=${REGION} sa=${SA} image=${IMAGE} stage=${STAGE:-adherence}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo ">>> DRY_RUN=1 — not submitting. Real command would be:"
  echo "    ${GCLOUD} ai custom-jobs create --project=${PROJECT} --region=${REGION} --display-name=${NAME} --service-account=${SA} --config=${CFG}"
  exit 0
fi

echo ">>> Submitting Vertex Custom Job (provisions the H100) ..."
exec "${GCLOUD}" ai custom-jobs create \
  --project="${PROJECT}" --region="${REGION}" \
  --display-name="${NAME}" \
  --service-account="${SA}" \
  --config="${CFG}"
