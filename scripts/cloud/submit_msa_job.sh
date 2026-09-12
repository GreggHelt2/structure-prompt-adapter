#!/usr/bin/env bash
# LAUNCH GPU-MMseqs2 MSA generation as a Vertex Custom Job (dev plan/96 §5.9). Mirrors submit_cache_job.sh:
# 1x H100 (a3-highgpu-1g), us-central1, spa-worker SA, git-clones the public repo, runs run_msa_gen.sh,
# AUTO-TERMINATES. GPU here is USED (MMseqs2-GPU), unlike a CPU-only route.
#
# ⛔ NOT LAUNCHED / DRAFT. Reconcile against submit_cache_job.sh before real use.
#
# PREREQUISITES:
#   - PUBLIC repo PUSHED @ $REPO_REF (the job clones it for run_msa_gen.sh)
#   - the deduped input FASTA staged to $FASTA_GCS  (built locally from the run dirs' seqs/*.fa)
#   - VERIFY: the spa-cloud image has ColabFold + an MMseqs2-GPU (CUDA) binary, OR run_msa_gen.sh installs them
#
# Usage:
#   DRY_RUN=1 FASTA_GCS=gs://... OUT_GCS=gs://... ./submit_msa_job.sh          # print spec, create nothing
#   PROBE_ONLY=1 FASTA_GCS=... OUT_GCS=... ./submit_msa_job.sh                 # DB fetch + probe only (§5.8b gate)
#   FASTA_GCS=... OUT_GCS=... ./submit_msa_job.sh                             # full run after the probe rate is OK
set -euo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
REGION="${REGION:-us-central1}"
IMAGE="${IMAGE:-us-central1-docker.pkg.dev/spa-dev-499900/spa/spa-msa:0.1.0}"
SA="${SA:-spa-worker@spa-dev-499900.iam.gserviceaccount.com}"
REPO_URL="${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}"
REPO_REF="${REPO_REF:-main}"
# DECISION: ~1.5 TB DBs + makepaddedseqdb output + working set. 2000 GB pd-ssd is the starting point.
DISK_GB="${DISK_GB:-2000}"
PROBE_N="${PROBE_N:-8}"; PROBE_ONLY="${PROBE_ONLY:-0}"; DB_SET="${DB_SET:-full}"
# DB source + one-time cache knobs (see run_msa_gen.sh):
#   DB_SRC=ngc  + DB_GCS + CACHE_ONLY=1  -> pull uniref30 from NGC ONCE, mirror to GCS, stop (the cache job)
#   DB_SRC=gcs  + DB_GCS                 -> hydrate the DB from that GCS cache (fast), then probe/search
DB_SRC="${DB_SRC:-ngc}"; DB_GCS="${DB_GCS:-}"; CACHE_ONLY="${CACHE_ONLY:-0}"
if [ "$CACHE_ONLY" = 1 ]; then
  # cache-once: no query FASTA / output prefix needed; DB_GCS is the mirror target.
  FASTA_GCS="${FASTA_GCS:-gs://unused}"; OUT_GCS="${OUT_GCS:-gs://unused}"
  [ -n "$DB_GCS" ] || { echo "FATAL: set DB_GCS=gs://... (cache target) when CACHE_ONLY=1" >&2; exit 1; }
else
  FASTA_GCS="${FASTA_GCS:?set FASTA_GCS=gs://... (the deduped design-sequence FASTA)}"
  OUT_GCS="${OUT_GCS:?set OUT_GCS=gs://... (destination prefix for the .a3m)}"
  [ "$DB_SRC" != gcs ] || [ -n "$DB_GCS" ] || { echo "FATAL: DB_SRC=gcs needs DB_GCS=gs://..." >&2; exit 1; }
fi
NAME="${NAME:-spa-msagen-$(date -u +%Y%m%d-%H%M%S)}"
GCLOUD="${GCLOUD:-gcloud}"

. "$(dirname "${BASH_SOURCE[0]}")/_pin_run_env.sh"
BOOT="set -e; ${BOOT_CHECKOUT} && bash /opt/spa/scripts/cloud/run_msa_gen.sh"

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
        - name: FASTA_GCS
          value: "${FASTA_GCS}"
        - name: OUT_GCS
          value: "${OUT_GCS}"
        - name: PROBE_N
          value: "${PROBE_N}"
        - name: PROBE_ONLY
          value: "${PROBE_ONLY}"
        - name: DB_SET
          value: "${DB_SET}"
        - name: DB_SRC
          value: "${DB_SRC}"
        - name: DB_GCS
          value: "${DB_GCS}"
        - name: CACHE_ONLY
          value: "${CACHE_ONLY}"
YAML

echo ">>> CustomJobSpec (${CFG}):"; sed 's/^/    /' "${CFG}"
echo ">>> name=${NAME} region=${REGION} disk=${DISK_GB}GB fasta=${FASTA_GCS} out=${OUT_GCS} probe_only=${PROBE_ONLY}"
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo ">>> DRY_RUN=1: not submitting. Real command:"
  echo "    ${GCLOUD} ai custom-jobs create --project=${PROJECT} --region=${REGION} --display-name=${NAME} --service-account=${SA} --config=${CFG}"
  exit 0
fi
exec "${GCLOUD}" ai custom-jobs create \
  --project="${PROJECT}" --region="${REGION}" \
  --display-name="${NAME}" --service-account="${SA}" --config="${CFG}"
