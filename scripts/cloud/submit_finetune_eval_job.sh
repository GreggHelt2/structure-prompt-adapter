#!/usr/bin/env bash
# LAUNCH the Option B fine-tune EVAL (dev plan/69 SS4.13) as a Vertex AI Custom Job: 1x H100
# (a3-highgpu-1g), on-demand, spa-worker SA, runs scripts/cloud/run_finetune_eval.sh, stages results
# to GCS, AUTO-TERMINATES. Mirrors submit_trivial_baseline_job.sh.
#
# PREREQUISITES:
#   - PUBLIC repo PUSHED @ $REPO_REF
#   - spa-combined image already in Artifact Registry
#   - PINNED GCS: $BUCKET/weights/of3-p2-155k.pt
#   - $BUCKET/checkpoints/finetune-A0A7S3EB45-20260814-045849/{rfd3_C_finetuned_full.ckpt,spa_C_final.pt}
#   - $BUCKET/eval/finetune_neighborhood/A0A7S3EB45/target.pdb
#
# Usage:
#   DRY_RUN=1 ./submit_finetune_eval_job.sh                # print the spec + gcloud cmd; create NOTHING
#   ./submit_finetune_eval_job.sh                           # real launch
#
# Cost: K=8/N=8 on one 125-residue fold, TWO conditions (baseline + spa lambda=[1,2], so 3 generation
# arms total: baseline, spa@1, spa@2), with OF3 refold on all of them. Similar shape to the b1_full /
# trivial-baseline runs already priced at $10-20; budget under $30.
set -euo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
REGION="${REGION:-us-west1}"
IMAGE="${IMAGE:-us-central1-docker.pkg.dev/spa-dev-499900/spa/spa-combined:0.1.0}"
SA="${SA:-spa-worker@spa-dev-499900.iam.gserviceaccount.com}"
REPO_URL="${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}"
REPO_REF="${REPO_REF:-main}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
RFD3_FINETUNED_CKPT_URI="${RFD3_FINETUNED_CKPT_URI:-$BUCKET/checkpoints/finetune-A0A7S3EB45-20260814-045849/rfd3_C_finetuned_full.ckpt}"
SPA_ADAPTER_CKPT_URI="${SPA_ADAPTER_CKPT_URI:-$BUCKET/checkpoints/finetune-A0A7S3EB45-20260814-045849/spa_C_final.pt}"
OF3_CKPT_URI="${OF3_CKPT_URI:-$BUCKET/weights/of3-p2-155k.pt}"
TARGET_PDB_URI="${TARGET_PDB_URI:-$BUCKET/eval/finetune_neighborhood/A0A7S3EB45/target.pdb}"
LENGTH="${LENGTH:-125}"
K="${K:-8}"; NSEQ="${NSEQ:-8}"; SEED="${SEED:-42}"
LAMBDAS="${LAMBDAS:-[1,2]}"
DISK_GB="${DISK_GB:-150}"
STRATEGY="${STRATEGY:-ONDEMAND}"
NAME="${NAME:-spa-finetune-eval-$(date -u +%Y%m%d-%H%M%S)}"
GCLOUD="${GCLOUD:-gcloud}"

. "$(dirname "${BASH_SOURCE[0]}")/_pin_run_env.sh"

BOOT="set -e; ${BOOT_CHECKOUT} && bash /opt/spa/scripts/cloud/run_finetune_eval.sh"

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
add_env RFD3_FINETUNED_CKPT_URI "${RFD3_FINETUNED_CKPT_URI}"
add_env SPA_ADAPTER_CKPT_URI "${SPA_ADAPTER_CKPT_URI}"
add_env OF3_CKPT_URI "${OF3_CKPT_URI}"
add_env TARGET_PDB_URI "${TARGET_PDB_URI}"
add_env LENGTH "${LENGTH}"
add_env K "${K}"
add_env NSEQ "${NSEQ}"
add_env SEED "${SEED}"
add_env LAMBDAS "${LAMBDAS}"
add_env REPO_REF "${REPO_REF}"

case "${STRATEGY}" in
  ONDEMAND|STANDARD|on-demand|"") : ;;
  *) printf 'scheduling:\n  strategy: %s\n' "${STRATEGY}" >> "${CFG}" ;;
esac

echo ">>> CustomJobSpec (${CFG}):"; sed 's/^/    /' "${CFG}"
echo ">>> name=${NAME} region=${REGION} sa=${SA} image=${IMAGE}"
echo ">>> length=${LENGTH} K=${K} N=${NSEQ} seed=${SEED} lambdas=${LAMBDAS} disk=${DISK_GB}GB strategy=${STRATEGY}"

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
