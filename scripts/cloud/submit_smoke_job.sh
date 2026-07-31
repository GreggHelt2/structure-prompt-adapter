#!/usr/bin/env bash
# LAUNCH the COMBINED-image smoke test as a Vertex AI Custom Job: 1x H100 (a3-highgpu-1g), on-demand, runs
# as spa-worker, boots run_smoke.sh (full flywheel end-to-end on a tiny unconditional design -> validates
# both stacks coexist in one image + OF3 triton works on Hopper), then AUTO-TERMINATES. Mirrors
# submit_train_job.sh. Cheap: a few minutes of H100 (~$0.2–0.5); a malformed spec is rejected free.
#
# PREREQUISITES:
#   - spa-combined:${TAG} pushed to Artifact Registry     (build_and_push_combined.sh)
#   - PUBLIC repo pushed to GitHub @ $REPO_REF             (the job git-clones it for run_smoke.sh)
#   - PINNED GCS artifacts: $BUCKET/weights/rfd3_latest.ckpt and $BUCKET/weights/of3-p2-155k.pt
#
# Usage:
#   DRY_RUN=1 ./submit_smoke_job.sh          # print the CustomJobSpec + gcloud cmd; create NOTHING
#   REGION=us-central1 ./submit_smoke_job.sh # launch on the idle us-central1 H100
set -euo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
REGION="${REGION:-us-central1}"                    # smoke on the idle us-central1 H100 by default
TAG="${TAG:-0.1.0}"
IMAGE="${IMAGE:-us-central1-docker.pkg.dev/spa-dev-499900/spa/spa-combined:${TAG}}"
SA="${SA:-spa-worker@spa-dev-499900.iam.gserviceaccount.com}"
REPO_URL="${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}"
REPO_REF="${REPO_REF:-main}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
OF3_CKPT_URI="${OF3_CKPT_URI:-$BUCKET/weights/of3-p2-155k.pt}"
DISK_GB="${DISK_GB:-150}"                           # image + RFD3 (2.9G) + OF3 (2.3G) ckpts + tiny run
RUN_SCRIPT="${RUN_SCRIPT:-run_smoke.sh}"            # in-container script under scripts/cloud/ (run_smoke.sh | run_calib.sh)
LENGTHS="${LENGTHS:-}"                              # calibration only: comma lengths for run_calib.sh (ignored otherwise)
STRATEGY="${STRATEGY:-ONDEMAND}"                    # ONDEMAND -> the approved H100 quota; omit scheduling
NAME="${NAME:-spa-smoke-$(date -u +%Y%m%d-%H%M%S)}"
GCLOUD="${GCLOUD:-/home/user1/google-cloud-sdk/bin/gcloud}"

# Pin the code revision and container digest (see _pin_run_env.sh). Must come AFTER IMAGE /
# REPO_URL / GCLOUD are set and BEFORE BOOT is built.
. "$(dirname "${BASH_SOURCE[0]}")/_pin_run_env.sh"

BOOT="set -e; ${BOOT_CHECKOUT} && bash /opt/spa/scripts/cloud/${RUN_SCRIPT}"

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
add_env RFD3_CKPT_URI "${RFD3_CKPT_URI}"
add_env OF3_CKPT_URI "${OF3_CKPT_URI}"
add_env REPO_REF "${REPO_REF}"
add_env LENGTHS "${LENGTHS}"
# run_eval.sh (B1-full) knobs — omitted when empty (full run uses manifest defaults / all prompts):
add_env NUM_TIMESTEPS "${NUM_TIMESTEPS:-}"
add_env SUBSET_IDS "${SUBSET_IDS:-}"
add_env K_OVERRIDE "${K_OVERRIDE:-}"
add_env NSEQ_OVERRIDE "${NSEQ_OVERRIDE:-}"
# run_of3_batch_bench.sh knobs (dev plan/23 §7). KERNEL matters: the triton Evoformer kernel asserts a bias
# batch-dim of 1, so it CANNOT batch — a bs>1 bench must run KERNEL=nokernel or it reports "no folds".
add_env KERNEL "${KERNEL:-}"
add_env BATCH_SIZES "${BATCH_SIZES:-}"
add_env DESIGN_ID "${DESIGN_ID:-}"
add_env N_SEQS "${N_SEQS:-}"
# run_m6_calibration.sh knobs (review M6). ARM selects the sampler arm: rfd3 (200/0.6) vs ours (100/0.8).
add_env ARM "${ARM:-}"
add_env K "${K:-}"
add_env NSEQ "${NSEQ:-}"
add_env SEED "${SEED:-}"
add_env SCRMSD_ATOMS "${SCRMSD_ATOMS:-}"
add_env SCRMSD_CUTOFF "${SCRMSD_CUTOFF:-}"
# run_sampler_interaction.sh knobs (dev plan/36). ARM/K/NSEQ/SEED/SCRMSD_* above are shared with M6.
# CONDITIONS and LAMBDAS are the factors this run sweeps; PREP_URI lets it reuse the b1_full prep,
# and RESULTS_URI keeps a smoke from writing into the real results prefix.
add_env CONDITIONS "${CONDITIONS:-}"
add_env LAMBDAS "${LAMBDAS:-}"
add_env PREP_URI "${PREP_URI:-}"
add_env RESULTS_URI "${RESULTS_URI:-}"

case "${STRATEGY}" in
  ONDEMAND|STANDARD|on-demand|"") : ;;
  *) printf 'scheduling:\n  strategy: %s\n' "${STRATEGY}" >> "${CFG}" ;;
esac

echo ">>> CustomJobSpec (${CFG}):"; sed 's/^/    /' "${CFG}"
echo ">>> name=${NAME} region=${REGION} sa=${SA} image=${IMAGE}"
echo ">>> rfd3=${RFD3_CKPT_URI} of3=${OF3_CKPT_URI} disk=${DISK_GB}GB strategy=${STRATEGY}"

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
