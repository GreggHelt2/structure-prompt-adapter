#!/usr/bin/env bash
# LAUNCH the M10 inference-overhead bench (dev plan/73) as a Vertex AI Custom Job: 1x H100
# (a3-highgpu-1g), on-demand, runs as spa-worker, boots run_m10_bench.sh, AUTO-TERMINATES.
# Mirrors submit_smoke_job.sh / submit_scaffold_job.sh.
#
# ALWAYS run MODE=smoke first (the default here). This harness (scripts/eval/bench_m10_overhead.py)
# has never executed against a real GPU or the cloud environment -- static review and cross-checking
# against source caught several real bugs before this point (see the commit history), but nothing
# replaces an actual run. A smoke pass is a few minutes of H100 (~$0.5-1); a malformed harness is
# cheap to find there rather than 2-3 hours into the full sweep.
#
# Usage:
#   DRY_RUN=1 ./submit_m10_bench_job.sh                    # print the CustomJobSpec; create NOTHING
#   ./submit_m10_bench_job.sh                               # MODE=smoke (default), us-central1
#   MODE=full REGION=us-west1 ./submit_m10_bench_job.sh      # the full plan/73 sweep, after smoke passes
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
SPA_CKPT_URI="${SPA_CKPT_URI:-$BUCKET/checkpoints/spa-Nx1536-uncond/spa_C_final.pt}"
PREP_URI="${PREP_URI:-$BUCKET/eval/b1_full/prep}"
RESULTS_URI="${RESULTS_URI:-}"                      # empty -> run_m10_bench.sh's own default
DISK_GB="${DISK_GB:-150}"
STRATEGY="${STRATEGY:-ONDEMAND}"
NAME="${NAME:-spa-m10-$(date -u +%Y%m%d-%H%M%S)}"
GCLOUD="${GCLOUD:-gcloud}"

# M10 bench knobs (run_m10_bench.sh)
MODE="${MODE:-smoke}"                               # smoke | full — ALWAYS smoke first, see header
VARIANT="${VARIANT:-C_n_by_1536}"
PROMPT_IDS="${PROMPT_IDS:-}"                         # empty -> run_m10_bench.sh's own 6-prompt default
LENGTHS="${LENGTHS:-}"                               # empty -> run_m10_bench.sh's own default
K="${K:-}"
LAM="${LAM:-}"

. "$(dirname "${BASH_SOURCE[0]}")/_pin_run_env.sh"

BOOT="set -e; ${BOOT_CHECKOUT} && MODE=${MODE} bash /opt/spa/scripts/cloud/run_m10_bench.sh"

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
add_env SPA_CKPT_URI "${SPA_CKPT_URI}"
add_env PREP_URI "${PREP_URI}"
add_env RESULTS_URI "${RESULTS_URI}"
add_env REPO_REF "${REPO_REF}"
add_env MODE "${MODE}"
add_env VARIANT "${VARIANT}"
add_env PROMPT_IDS "${PROMPT_IDS}"
add_env LENGTHS "${LENGTHS}"
add_env K "${K}"
add_env LAM "${LAM}"

case "${STRATEGY}" in
  ONDEMAND|STANDARD|on-demand|"") : ;;
  *) printf 'scheduling:\n  strategy: %s\n' "${STRATEGY}" >> "${CFG}" ;;
esac

echo ">>> CustomJobSpec (${CFG}):"; sed 's/^/    /' "${CFG}"
echo ">>> name=${NAME} region=${REGION} sa=${SA} image=${IMAGE} mode=${MODE}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo ">>> DRY_RUN=1 — not submitting. Real command would be:"
  echo "    ${GCLOUD} ai custom-jobs create --project=${PROJECT} --region=${REGION} --display-name=${NAME} --service-account=${SA} --config=${CFG}"
  exit 0
fi

if [ "${MODE}" = "full" ]; then
  echo ">>> MODE=full: this is the ~2-3h / ~\$22-33 run (dev plan/73 SS6). Confirm a MODE=smoke pass"
  echo "    already succeeded for this exact commit before proceeding."
fi

echo ">>> Submitting Vertex Custom Job (this provisions the H100) ..."
exec "${GCLOUD}" ai custom-jobs create \
  --project="${PROJECT}" --region="${REGION}" \
  --display-name="${NAME}" \
  --service-account="${SA}" \
  --config="${CFG}"
