#!/usr/bin/env bash
# LAUNCH Phase-2 SPA training (or the probe-cache dress rehearsal) as a Vertex AI Custom Job.
# 1x H100 (a3-highgpu-1g), on-demand, runs as the spa-worker SA, writes SPA checkpoints to GCS,
# AUTO-TERMINATES. dev 04 §11 / 08 step 10 / 08 §3b workstream A. Mirrors submit_cache_job.sh.
#
# PREREQUISITES (must hold before launch):
#   - PUBLIC repo PUSHED to GitHub @ $REPO_REF  (the job git-clones it for run_train.sh + the package)
#   - env-only image in Artifact Registry                                              (done, cache-gen)
#   - secret spa-ngc-key staged + spa-worker secretAccessor                            (done)
#   - PINNED GCS artifacts: $BUCKET/weights/rfd3_latest.ckpt and $BUCKET/splits/$SPLIT_ID/  (stage once)
#   - the ESM3 cache tar at $CACHE_TAR (full: esm3_cache.tar from step 9; probe: esm3_cache_probe1k.tar)
#
# Usage:
#   DRY_RUN=1 ./submit_train_job.sh                 # print the CustomJobSpec + gcloud cmd; create NOTHING
#   # B2 PROBE dress rehearsal on the idle us-central1 H100 (short, cheap):
#   REGION=us-central1 CACHE_TAR=gs://genomancer-spa-cache/esm3_cache_probe1k.tar \
#     REQUIRE_CACHED_PROMPT=true MAX_STEPS=20 CHECKPOINT_SEC=120 DISK_GB=200 MIN_FREE_GB=120 \
#     ./submit_train_job.sh
#   # FULL run (after step 9's esm3_cache.tar lands), us-west1:
#   REGION=us-west1 CACHE_TAR=gs://genomancer-spa-cache/esm3_cache.tar DISK_GB=650 ./submit_train_job.sh
#
# Safety: a malformed spec is rejected by the Vertex API *before* any machine is provisioned (free).
set -euo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
REGION="${REGION:-us-west1}"                       # full run = us-west1; probe rehearsal = us-central1 (idle)
IMAGE="${IMAGE:-us-central1-docker.pkg.dev/spa-dev-499900/spa/spa-cloud:0.3.0}"
SA="${SA:-spa-worker@spa-dev-499900.iam.gserviceaccount.com}"
REPO_URL="${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}"
REPO_REF="${REPO_REF:-main}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"

CACHE_TAR="${CACHE_TAR:-$BUCKET/esm3_cache.tar}"   # probe rehearsal: $BUCKET/esm3_cache_probe1k.tar
SPLIT_ID="${SPLIT_ID:-v1_seed0_8-1-1_lenstrat}"
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
VARIANT="${VARIANT:-C_n_by_1536}"
CONDITIONING="${CONDITIONING:-unconditional}"
CHECKPOINT_SEC="${CHECKPOINT_SEC:-3600}"
DISK_GB="${DISK_GB:-650}"                          # full run peaks high; probe: DISK_GB=200
STRATEGY="${STRATEGY:-ONDEMAND}"                   # ONDEMAND -> the approved H100 quota (07 W5.6); omit scheduling
NAME="${NAME:-spa-train-$(date -u +%Y%m%d-%H%M%S)}"
CKPT_OUT_URI="${CKPT_OUT_URI:-$BUCKET/checkpoints/$NAME}"
GCLOUD="${GCLOUD:-/home/user1/google-cloud-sdk/bin/gcloud}"

# Optional knobs — omitted from the env block when empty (W5.7: Vertex rejects an empty env value).
REQUIRE_CACHED_PROMPT="${REQUIRE_CACHED_PROMPT:-}" # true for the probe rehearsal
LENGTH_CAP="${LENGTH_CAP:-}"                        # data.length_cap override (keep <= 384 for prompt alignment)
MAX_STEPS="${MAX_STEPS:-}"                          # short budget for the rehearsal
TRACKER="${TRACKER:-}"                              # wandb (workstream C)
MIN_FREE_GB="${MIN_FREE_GB:-}"                      # probe: set ~120 (full default 540 in run_train.sh)
FETCH_HF="${FETCH_HF:-}"                            # 1 only for the variant-A/CLSS path
RUN_MODE="${RUN_MODE:-}"                            # profile -> run scripts/profile_step.py (step timing) not train.py
N_STEPS="${N_STEPS:-}"                              # profile: # steps to time (profiler default 40)
NUM_WORKERS="${NUM_WORKERS:-}"                      # dataloader workers (train.num_workers override; H100 wants more)

BOOT="set -e; git clone --depth 1 --branch ${REPO_REF} ${REPO_URL} /opt/spa && bash /opt/spa/scripts/cloud/run_train.sh"

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
# Always-present:
add_env PROJECT "${PROJECT}"
add_env BUCKET "${BUCKET}"
add_env CACHE_TAR "${CACHE_TAR}"
add_env SPLIT_ID "${SPLIT_ID}"
add_env RFD3_CKPT_URI "${RFD3_CKPT_URI}"
add_env VARIANT "${VARIANT}"
add_env CONDITIONING "${CONDITIONING}"
add_env CHECKPOINT_SEC "${CHECKPOINT_SEC}"
add_env CKPT_OUT_URI "${CKPT_OUT_URI}"
add_env NAME "${NAME}"
# Optional (omitted when empty):
add_env REQUIRE_CACHED_PROMPT "${REQUIRE_CACHED_PROMPT}"
add_env LENGTH_CAP "${LENGTH_CAP}"
add_env MAX_STEPS "${MAX_STEPS}"
add_env TRACKER "${TRACKER}"
add_env MIN_FREE_GB "${MIN_FREE_GB}"
add_env FETCH_HF "${FETCH_HF}"
add_env RUN_MODE "${RUN_MODE}"
add_env N_STEPS "${N_STEPS}"
add_env NUM_WORKERS "${NUM_WORKERS}"

# On-demand (default) OMITS scheduling (Vertex defaults to on-demand). SPOT/FLEX_START draw the separate
# preemptible H100 quota (=0 here) -> opt in only with a quota bump.
case "${STRATEGY}" in
  ONDEMAND|STANDARD|on-demand|"") : ;;
  *) printf 'scheduling:\n  strategy: %s\n' "${STRATEGY}" >> "${CFG}" ;;
esac

echo ">>> CustomJobSpec (${CFG}):"; sed 's/^/    /' "${CFG}"
echo ">>> name=${NAME} region=${REGION} sa=${SA} variant=${VARIANT} cond=${CONDITIONING} cache=${CACHE_TAR}"
echo ">>> split_id=${SPLIT_ID} disk=${DISK_GB}GB strategy=${STRATEGY} ckpt_out=${CKPT_OUT_URI}"

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
