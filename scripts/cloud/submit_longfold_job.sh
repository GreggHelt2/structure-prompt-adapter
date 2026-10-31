#!/usr/bin/env bash
# LAUNCH the plan/153 LONG-PROMPT STEERING sweep as a Vertex AI Custom Job: 1x H100 (a3-highgpu-1g),
# on-demand, runs as spa-worker, boots run_longfold_steer.sh on the spa-combined image, then
# auto-terminates. Mirrors submit_threeway_job.sh. On-demand H100 = 1/region (Spot/Flex quota=0), so
# split the 28 folds across the two regions with SUBSET_IDS (us-west1 + us-central1).
#
# PREREQUISITES:
#   - spa-combined:${TAG} pushed to Artifact Registry
#   - PUBLIC repo pushed to GitHub @ $REPO_REF (the job git-clones it for run_longfold_steer.sh)
#   - PINNED GCS artifacts: weights/{rfd3_latest.ckpt,of3-p2-155k.pt}, checkpoints/spa-Nx1536-uncond/spa_C_final.pt
#   - eval/longfold_steer/prep/ staged: the 28 .pt (from esm3_cache.tar) + .pdb + longfold_steer_resolved.json
#     (scripts/eval/prep_longfold_steer.py; GPU-free, run on a GCS-connected box)
#
# Usage:
#   DRY_RUN=1 ./submit_longfold_job.sh                                             # print the CustomJobSpec; create NOTHING
#   STAGE=adherence REGION=us-west1    SUBSET_IDS=<14 ids> ./submit_longfold_job.sh  # Stage-1 half A
#   STAGE=adherence REGION=us-central1 SUBSET_IDS=<14 ids> ./submit_longfold_job.sh  # Stage-1 half B
#   STAGE=designability REGION=us-west1 SUBSET_IDS=<ids> ./submit_longfold_job.sh    # Stage-2 (gated)
#   SUBSET_IDS=<1 per class> STAGE=adherence ./submit_longfold_job.sh                # smoke
#   LAMBDAS=1.5,2 STAGE=adherence REGION=us-west1 SUBSET_IDS=<strong-steering ids> ./submit_longfold_job.sh  # G0 λ-extend sweep (SPA-only, reuses the {0,1} baseline; plan/153 §9 #4)
set -euo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
REGION="${REGION:-us-west1}"
TAG="${TAG:-0.1.0}"
IMAGE="${IMAGE:-us-central1-docker.pkg.dev/spa-dev-499900/spa/spa-combined:${TAG}}"
SA="${SA:-spa-worker@spa-dev-499900.iam.gserviceaccount.com}"
REPO_URL="${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}"
REPO_REF="${REPO_REF:-main}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
DISK_GB="${DISK_GB:-150}"
STRATEGY="${STRATEGY:-ONDEMAND}"
STAGE="${STAGE:-adherence}"
NAME="${NAME:-spa-longfold-${STAGE}-$(date -u +%Y%m%d-%H%M%S)}"
GCLOUD="${GCLOUD:-gcloud}"

RUN_SCRIPT="${RUN_SCRIPT:-run_longfold_steer.sh}"
. "$(dirname "${BASH_SOURCE[0]}")/_pin_run_env.sh"

BOOT="set -e; ${BOOT_CHECKOUT} && bash /opt/spa/scripts/cloud/${RUN_SCRIPT}"

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
add_env REPO_REF "${REPO_REF}"
# long-fold steering knobs (omitted-when-empty -> run_longfold_steer.sh defaults apply):
add_env STAGE "${STAGE:-}"
add_env SUBSET_IDS "${SUBSET_IDS:-}"   # comma list of fold ids -> run only these (region split / smoke); empty = all 28
add_env ARM "${ARM:-}"                 # default rfd3 (200/0.6) in the runner
add_env SEEDS "${SEEDS:-}"             # default [0,1] adherence / [0] designability in the runner
add_env LAMBDAS "${LAMBDAS:-}"         # comma list -> eval.lambda_scale sweep, e.g. "1.5,2" (empty = manifest default 1); plan/153 §9 #4 (G0)
add_env CONDITIONS "${CONDITIONS:-}"   # empty: a LAMBDAS sweep runs SPA-only [spa] (reuse the {0,1} baseline), else [baseline,spa]
add_env PREP_URI "${PREP_URI:-}"
add_env RESULTS_URI "${RESULTS_URI:-}"
# tensor-extract job knobs (run_extract_prompt_tensors.sh):
add_env GENERATE_ONLY "${GENERATE_ONLY:-}"   # 1 = skip hydrate + CDDB extract + b4 pull, re-run only the 17-fold ESM3 generate
add_env GEN_IDS "${GEN_IDS:-}"               # comma list of recognizable PDB ids to ESM3-encode (default = the 17)
add_env B4_IDS "${B4_IDS:-}"
add_env EXTRA_IDS "${EXTRA_IDS:-}"
# G4-A driver knobs (run_g4a_recognizable.sh):
add_env G4A_GEN_IDS "${G4A_GEN_IDS:-}"
add_env G4A_B4_IDS "${G4A_B4_IDS:-}"
add_env LAM "${LAM:-}"

case "${STRATEGY}" in
  ONDEMAND|STANDARD|on-demand|"") : ;;
  *) printf 'scheduling:\n  strategy: %s\n' "${STRATEGY}" >> "${CFG}" ;;
esac

echo ">>> CustomJobSpec (${CFG}):"; sed 's/^/    /' "${CFG}"
echo ">>> name=${NAME} region=${REGION} sa=${SA} image=${IMAGE} stage=${STAGE}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo ">>> DRY_RUN=1 — not submitting. Real command would be:"
  echo "    ${GCLOUD} ai custom-jobs create --project=${PROJECT} --region=${REGION} --display-name=${NAME} --service-account=${SA} --config=${CFG}"
  exit 0
fi

echo ">>> Submitting Vertex Custom Job (provisions the H100) ..."
exec "${GCLOUD}" ai custom-jobs create \
  --project="${PROJECT}" --region="${REGION}" \
  --display-name="${NAME}" --service-account="${SA}" --config="${CFG}"
