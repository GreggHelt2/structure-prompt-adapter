#!/usr/bin/env bash
# SPA cloud image — local CPU build (BuildKit SSH forwarding) + push to Artifact Registry.
# W5.1 env-only image; serves Phase-1 cache-gen + Phase-2 training. (dev 07 W5.1 / 08 step 2)
#
# Prerequisites:
#   1) ssh-agent loaded with your GitHub key (the deps are private repos):   ssh-add -l
#   2) Artifact Registry docker repo (one-time):
#        gcloud artifacts repositories create spa \
#          --repository-format=docker --location=us-central1 \
#          --description="SPA cloud images"
#   3) Docker auth to AR (one-time):
#        gcloud auth configure-docker us-central1-docker.pkg.dev
#
# Usage:
#   ./build_and_push.sh            # CPU build only
#   PUSH=1 ./build_and_push.sh     # build + push (after steps 2-3 above)
set -euo pipefail

REGION="${REGION:-us-central1}"
PROJECT="${PROJECT:-spa-dev-499900}"
REPO="${REPO:-spa}"
TAG="${TAG:-0.3.0}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/spa-cloud:${TAG}"

cd "$(dirname "$0")"

echo ">>> Building ${IMAGE} (CPU build; private deps cloned via SSH forwarding)"
DOCKER_BUILDKIT=1 docker build --ssh default -t "${IMAGE}" .

if [[ "${PUSH:-0}" == "1" ]]; then
    echo ">>> Pushing ${IMAGE}"
    docker push "${IMAGE}"
else
    echo ">>> Built (not pushed). Re-run with PUSH=1 once the AR repo + docker auth exist."
fi
