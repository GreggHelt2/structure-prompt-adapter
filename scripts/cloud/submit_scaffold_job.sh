#!/usr/bin/env bash
# LAUNCH the SCAFFOLDING big-run (dev 17 §7 / 16 §9.5) as a Vertex AI Custom Job: 1x H100
# (a3-highgpu-1g), on-demand, spa-worker SA, runs scripts/cloud/run_scaffold_eval.sh, stages results to
# GCS, AUTO-TERMINATES. Multigran vs base full-prompt SPA on sub-region-masked held-out prompts, with OF3
# designability. Mirrors submit_train_job.sh but boots the eval runner on the COMBINED image (RFD3+SPA+
# ProteinMPNN+OF3), like run_variant_desig/B1-full.
#
# PREREQUISITES:
#   - PUBLIC repo PUSHED @ $REPO_REF (the job git-clones run_scaffold_eval.sh + the package)
#   - prep staged: scripts/eval/prep_scaffold.py --gcs-uri $PREP_URI  (<id>.pt + <id>.pdb + scaffold_resolved.json)
#   - PINNED GCS: $BUCKET/weights/{rfd3_latest.ckpt,of3-p2-155k.pt} + both SPA ckpts (already in GCS)
#
# Usage:
#   DRY_RUN=1 ./submit_scaffold_job.sh                       # print the spec + gcloud cmd; create NOTHING
#   REGION=us-west1 K=4 NSEQ=4 GRANS=domain,segment_small ./submit_scaffold_job.sh   # real launch
#
# Cost (defaults K=4/N=4, 17 folds × 2 grans × {baseline+spa_mg, spa_base}): ~4-6 h H100, ~$50-70.
set -euo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
REGION="${REGION:-us-west1}"                       # the idle H100 region (us-central1 = B4 as of 2026-07-02)
IMAGE="${IMAGE:-us-central1-docker.pkg.dev/spa-dev-499900/spa/spa-combined:0.1.0}"   # combined: RFD3+SPA+MPNN+OF3
SA="${SA:-spa-worker@spa-dev-499900.iam.gserviceaccount.com}"
REPO_URL="${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}"
REPO_REF="${REPO_REF:-main}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"

PREP_URI="${PREP_URI:-$BUCKET/eval/scaffold/prep}"
RESULTS_URI="${RESULTS_URI:-$BUCKET/eval/scaffold/results}"
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
OF3_CKPT_URI="${OF3_CKPT_URI:-$BUCKET/weights/of3-p2-155k.pt}"
MG_CKPT_URI="${MG_CKPT_URI:-$BUCKET/checkpoints/spa-Nx1536-multigran/spa_C_final.pt}"
BASE_CKPT_URI="${BASE_CKPT_URI:-$BUCKET/checkpoints/spa-Nx1536-uncond/spa_C_final.pt}"
VARIANT="${VARIANT:-C_n_by_1536}"
K="${K:-4}"; NSEQ="${NSEQ:-4}"; LAM="${LAM:-1}"; GRANS="${GRANS:-domain,segment_small}"
LEAN_RESULTS="${LEAN_RESULTS:-}"                    # 1 -> metrics JSON only (no design PDBs)
DISK_GB="${DISK_GB:-200}"                           # ckpts + OF3 working set (no 250 GB cache here)
STRATEGY="${STRATEGY:-ONDEMAND}"
NAME="${NAME:-spa-scaffold-$(date -u +%Y%m%d-%H%M%S)}"
GCLOUD="${GCLOUD:-/home/user1/google-cloud-sdk/bin/gcloud}"

BOOT="set -e; git clone --depth 1 --branch ${REPO_REF} ${REPO_URL} /opt/spa && bash /opt/spa/scripts/cloud/run_scaffold_eval.sh"

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
add_env PREP_URI "${PREP_URI}"
add_env RESULTS_URI "${RESULTS_URI}"
add_env RFD3_CKPT_URI "${RFD3_CKPT_URI}"
add_env OF3_CKPT_URI "${OF3_CKPT_URI}"
add_env MG_CKPT_URI "${MG_CKPT_URI}"
add_env BASE_CKPT_URI "${BASE_CKPT_URI}"
add_env VARIANT "${VARIANT}"
add_env K "${K}"
add_env NSEQ "${NSEQ}"
add_env LAM "${LAM}"
add_env GRANS "${GRANS}"
add_env REPO_REF "${REPO_REF}"
add_env LEAN_RESULTS "${LEAN_RESULTS}"

case "${STRATEGY}" in
  ONDEMAND|STANDARD|on-demand|"") : ;;
  *) printf 'scheduling:\n  strategy: %s\n' "${STRATEGY}" >> "${CFG}" ;;
esac

echo ">>> CustomJobSpec (${CFG}):"; sed 's/^/    /' "${CFG}"
echo ">>> name=${NAME} region=${REGION} image=${IMAGE}"
echo ">>> prep=${PREP_URI} results=${RESULTS_URI} K=${K} N=${NSEQ} λ=${LAM} grans=${GRANS}"

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
