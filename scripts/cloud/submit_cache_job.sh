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
#   ./submit_cache_job.sh                   # BENCHMARK: LIMIT=1000, 200GB, Spot  (FIRST H100 SPEND)
#   LIMIT= DISK_GB=400 STRATEGY=FLEX_START ./submit_cache_job.sh   # FULL run (LIMIT empty = all)
#
# Safety: a malformed spec is rejected by the Vertex API *before* any machine is provisioned (free),
# so the residual uncertainty in the scheduling/disk field names is safe to discover by submitting.
set -euo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
REGION="${REGION:-us-central1}"
IMAGE="${IMAGE:-us-central1-docker.pkg.dev/spa-dev-499900/spa/spa-cloud:0.2.0}"
SA="${SA:-spa-worker@spa-dev-499900.iam.gserviceaccount.com}"
REPO_URL="${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}"
REPO_REF="${REPO_REF:-main}"
LIMIT="${LIMIT-1000}"               # default benchmark; LIMIT= (empty) -> full run
DISK_GB="${DISK_GB:-200}"           # benchmark uses ~65GB; full run ~302GB (bump, or local SSD per W5.3)
STRATEGY="${STRATEGY:-SPOT}"        # SPOT (benchmark) | FLEX_START (full run, uninterrupted)
NAME="${NAME:-spa-cachegen-$(date -u +%Y%m%d-%H%M%S)}"
GCLOUD="${GCLOUD:-/home/user1/google-cloud-sdk/bin/gcloud}"

BOOT="set -e; git clone --depth 1 --branch ${REPO_REF} ${REPO_URL} /opt/spa && bash /opt/spa/scripts/cloud/run_cache_gen.sh"

CFG="$(mktemp --suffix=.yaml)"
cat > "${CFG}" <<YAML
workerPoolSpecs:
  - machineSpec:
      machineType: a3-highgpu-1g
    replicaCount: 1
    diskSpec:
      bootDiskType: pd-ssd
      bootDiskSizeGb: ${DISK_GB}
    containerSpec:
      imageUri: ${IMAGE}
      command: ["bash", "-c"]
      args: ["${BOOT}"]
      env:
        - name: LIMIT
          value: "${LIMIT}"
scheduling:
  strategy: ${STRATEGY}
YAML

echo ">>> CustomJobSpec (${CFG}):"; sed 's/^/    /' "${CFG}"
echo ">>> name=${NAME}  region=${REGION}  sa=${SA}  limit=${LIMIT:-<all>}  strategy=${STRATEGY}  disk=${DISK_GB}GB"

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
