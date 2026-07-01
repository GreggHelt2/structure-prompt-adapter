#!/usr/bin/env bash
# OF3-triton image — local CPU build + push to Artifact Registry. Unlocks the cloud OF3 eval track
# (B1-B4: designability at scale, curated-50, 1CTT, recognizable-fold demos). OF3 is a PUBLIC repo,
# so NO SSH forwarding is needed (unlike docker/build_and_push.sh for the spa-cloud image).
#
# Prereqs (already set up for spa-cloud): the AR docker repo `spa` + `gcloud auth configure-docker
# us-central1-docker.pkg.dev`. Usage:  ./build_and_push_of3.sh   (build only) | PUSH=1 ./build_and_push_of3.sh
set -euo pipefail

REGION="${REGION:-us-central1}"
PROJECT="${PROJECT:-spa-dev-499900}"
REPO="${REPO:-spa}"
TAG="${TAG:-0.1.0}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/of3-triton:${TAG}"

cd "$(dirname "$0")"

echo ">>> Building ${IMAGE} (CPU build; the build-time import check gates a broken OF3/triton install)"
DOCKER_BUILDKIT=1 docker build -f Dockerfile.of3 -t "${IMAGE}" .

if [[ "${PUSH:-0}" == "1" ]]; then
    echo ">>> Pushing ${IMAGE}"
    docker push "${IMAGE}"
    echo ">>> Pushed. Smoke-test on an H100 (run_openfold on ubiquitin + of3_triton.yml) before a matrix."
else
    echo ">>> Built (not pushed). Re-run with PUSH=1 to push."
fi
