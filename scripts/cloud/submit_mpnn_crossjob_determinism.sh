#!/usr/bin/env bash
# Submit the OpenFold3 PRECISION bench to a Vertex a3-highgpu-1g (H100), in OUR spa-combined image.
# Dev docs/plan/115 §1.1a step 0 ("an arm for OUR stack"); results -> dev docs/results/71.
#
# What it answers: what does bf16-mixed precision, the reachable half of Anthropic's OpenFold3
# `default`, buy US in speed and cost US in Angstrom, on the stage that owns 90.5% of local wall-clock?
# Four arms (fp32 x2, bf16 x2) at two lengths, P=1. The doubled arms are the control: our refolds are
# deterministic by default, so fp32-vs-fp32 should be bit-identical and pins the rerun floor at 0.
#
# ⛔ H100, NOT the A5000, and that is a VALIDITY constraint not a cost preference: every Anthropic arm in
# dev results/71 was measured on an H100, and both speed and bit-identity are properties of the GPU
# model (dev plan/91), so an A5000 arm could not be compared to any of them.
#
# ⛔ REAL H100 SPEND (~$11/h on-demand, dev plan/07 W5.5/W5.6). Measured anchor: dev results/71 §5 puts
# provisioning at 3.3-4.6 min, and §7.7's four short OF3 arms took 4.5 min total. Four arms at two
# lengths plus the repo/ckpt pull => expect ~15-25 min, roughly $3-5.
# ⚠️ Quota is 1 a3-highgpu-1g PER REGION (us-central1 + us-west1), not 2 fungible, and us-central1 has
# refused H100 capacity repeatedly, so us-west1 is the default (dev plan/105 §3).
#
# Usage:
#   DRY_RUN=1 bash scripts/cloud/submit_mpnn_crossjob.sh
#   bash scripts/cloud/submit_mpnn_crossjob.sh
set -euo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
REGION="${REGION:-us-west1}"
TAG="${TAG:-0.1.0}"
IMAGE="${IMAGE:-us-central1-docker.pkg.dev/spa-dev-499900/spa/spa-combined:${TAG}}"
SA="${SA:-spa-worker@spa-dev-499900.iam.gserviceaccount.com}"
REPO_URL="${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}"
# ⛔ NO `REPO_REF="${REPO_REF:-main}"` DEFAULT HERE, AND THAT IS THE POINT.
# `_pin_run_env.sh`'s `_pin_repo_ref` pins REPO_REF to the SUBMITTING MACHINE'S HEAD SHA, but only if
# the caller has not already set it (`if [ -n "${REPO_REF:-}" ]; then ... return`). A `:-main` default
# here therefore SKIPS the pin and hands the job a moving branch, which is exactly what the runner's
# own header promises does not happen. Caught by a dry run printing
# "[pin] REPO_REF explicitly set: main" instead of a sha. ⇒ leave it unset and let the pin work;
# `REPO_REF=main bash submit_...` still overrides deliberately if a branch is ever wanted.
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
OF3_CKPT_URI="${OF3_CKPT_URI:-$BUCKET/weights/of3-p2-155k.pt}"
DISK_GB="${DISK_GB:-150}"
NAME="${NAME:-spa-mpnn-crossjob-$(date -u +%Y%m%d-%H%M%S)}"
GCLOUD="${GCLOUD:-/home/user1/google-cloud-sdk/bin/gcloud}"

. "$(dirname "${BASH_SOURCE[0]}")/_pin_run_env.sh"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER_LOCAL="${HERE}/run_mpnn_crossjob_determinism.sh"
BUCKET_NAME="${BUCKET#gs://}"
RUNNER_GCS="${BUCKET}/eval/mpnn_crossjob/runner-$(date -u +%Y%m%d-%H%M%S).sh"
RUNNER_FUSE="/gcs/${BUCKET_NAME}/${RUNNER_GCS#${BUCKET}/}"

[ -r "$RUNNER_LOCAL" ] || { echo "missing runner: $RUNNER_LOCAL" >&2; exit 1; }
bash -n "$RUNNER_LOCAL" || { echo "runner has a syntax error, refusing to submit" >&2; exit 1; }

# ⛔ THE RUNNER COMES FROM GCS, NOT THE PINNED CLONE, AND THAT IS DELIBERATE.
# $BOOT_CHECKOUT fetches from GITHUB, so anything committed only locally is invisible to the job. This
# runner is new and the public repo is not pushed without Gregg's say-so, so a job that ran it from the
# clone would fail with "No such file". ⚠️ The DEPENDENCIES are a different matter and are fine in the
# clone: bench_of3_length.py, configs/of3/of3_triton.yml and score.py::ca_rmsd were all verified present
# in origin/main before submitting, and HEAD == origin/main, so the pin resolves. This is the same
# staging pattern the Anthropic arms used (dev results/71 §6a) and for the same reason.
echo ">>> staging runner to ${RUNNER_GCS}"
"${GCLOUD}" storage cp "$RUNNER_LOCAL" "$RUNNER_GCS"

BOOT="set -e; ${BOOT_CHECKOUT} && test -r ${RUNNER_FUSE} && bash ${RUNNER_FUSE}"

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
add_env BACKBONE "${BACKBONE:-}"
add_env NUM_SEQS "${NUM_SEQS:-}"
add_env SEED "${SEED:-}"
add_env REPO_REF "${REPO_REF}"

echo ">>> CustomJobSpec (${CFG}):"; sed 's/^/    /' "${CFG}"
echo ">>> name=${NAME} region=${REGION} image=${IMAGE}"
echo ">>> backbone=${BACKBONE:-<default>}  num_seqs=${NUM_SEQS:-<default>}  seed=${SEED:-<default>}  repo_ref=${REPO_REF}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo ">>> DRY_RUN=1, not submitting. Real command:"
  echo "    ${GCLOUD} ai custom-jobs create --project=${PROJECT} --region=${REGION} --display-name=${NAME} --service-account=${SA} --config=${CFG}"
  exit 0
fi

echo ">>> submitting: this PROVISIONS AN H100 and starts billing (~\$11/h) ..."
exec "${GCLOUD}" ai custom-jobs create \
  --project="${PROJECT}" --region="${REGION}" \
  --display-name="${NAME}" \
  --service-account="${SA}" \
  --config="${CFG}"
