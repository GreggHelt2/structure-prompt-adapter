#!/usr/bin/env bash
# LAUNCH Phase-1 ESM3 cache-gen (or the ~1k benchmark) as a Vertex AI Custom Job.
# 1x H100 (a3-highgpu-1g), Spot, us-central1, runs as the spa-worker SA, writes to GCS, AUTO-TERMINATES.
# dev 04 §10 / 08 step 9.
#
# PREREQUISITES (all no-cost; must hold before launch):
#   - PUBLIC repo PUSHED to GitHub @ $REPO_REF  (the job git-clones it for run_cache_gen.sh + producer fix)
#   - image 0.2.0 in Artifact Registry                                   (done)
#   - secrets spa-hf-token / spa-ngc-key staged + spa-worker secretAccessor (done)
#
# Usage:
#   DRY_RUN=1 ./submit_cache_job.sh        # print the CustomJobSpec + gcloud cmd; create NOTHING
#   LIMIT=1000 CHECKPOINT_SEC=30 DISK_GB=650 REGION=us-west1 ./submit_cache_job.sh   # PROBE (~$3, ~13min)
#   LIMIT= DISK_GB=650 REGION=us-west1 ./submit_cache_job.sh                          # FULL run (on-demand, all 455k)
#
# Safety: a malformed spec is rejected by the Vertex API *before* any machine is provisioned (free),
# so the residual uncertainty in the scheduling/disk field names is safe to discover by submitting.
set -euo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
REGION="${REGION:-us-central1}"
IMAGE="${IMAGE:-us-central1-docker.pkg.dev/spa-dev-499900/spa/spa-cloud:0.3.0}"
SA="${SA:-spa-worker@spa-dev-499900.iam.gserviceaccount.com}"
REPO_URL="${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}"
REPO_REF="${REPO_REF:-main}"
LIMIT="${LIMIT-1000}"               # default benchmark; LIMIT= (empty) -> full run
DISK_GB="${DISK_GB:-200}"           # probe ~65GB; full run (cap=512) peaks ~550GB -> use DISK_GB=650 (or local SSD/W5.3)
STRATEGY="${STRATEGY:-ONDEMAND}"    # ONDEMAND -> "Custom model training Nvidia H100 GPUs" quota (approved 1+1).
                                    # SPOT | FLEX_START use the *preemptible* H100 quota (separate bucket, =0 here).
NAME="${NAME:-spa-cachegen-$(date -u +%Y%m%d-%H%M%S)}"
GCLOUD="${GCLOUD:-/home/user1/google-cloud-sdk/bin/gcloud}"

BOOT="set -e; git clone --depth 1 --branch ${REPO_REF} ${REPO_URL} /opt/spa && bash /opt/spa/scripts/cloud/run_cache_gen.sh"

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
        - name: LENGTH_CAP
          value: "${LENGTH_CAP:-512}"
        - name: CHECKPOINT_SEC
          value: "${CHECKPOINT_SEC:-3600}"
YAML

# LIMIT only when non-empty: Vertex rejects an empty env value, and an ABSENT LIMIT makes
# run_cache_gen.sh ("${LIMIT:-}") process the FULL dataset. The probe/benchmark set LIMIT=1000.
if [ -n "${LIMIT}" ]; then
  printf '        - name: LIMIT\n          value: "%s"\n' "${LIMIT}" >> "${CFG}"
fi

# On-demand (default) uses the non-preemptible "Custom model training Nvidia H100 GPUs" quota and OMITS
# the scheduling block (Vertex defaults to on-demand). SPOT/FLEX_START use the separate *preemptible*
# H100 quota — opt in via STRATEGY=SPOT|FLEX_START.
case "${STRATEGY}" in
  ONDEMAND|STANDARD|on-demand|"") : ;;
  *) printf 'scheduling:\n  strategy: %s\n' "${STRATEGY}" >> "${CFG}" ;;
esac

echo ">>> CustomJobSpec (${CFG}):"; sed 's/^/    /' "${CFG}"
echo ">>> name=${NAME}  region=${REGION}  sa=${SA}  limit=${LIMIT:-<all>}  cap=${LENGTH_CAP:-512}  ckpt=${CHECKPOINT_SEC:-3600}s  strategy=${STRATEGY}  disk=${DISK_GB}GB"

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
