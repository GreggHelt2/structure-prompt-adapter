#!/usr/bin/env bash
# COMBINED cloud image (spa-cloud + OpenFold3/triton conda env) — local CPU build + push to Artifact
# Registry. Lets the validation flywheel run END-TO-END in one Vertex job (RFD3±SPA -> ProteinMPNN ->
# OF3 refold -> score), no two-stage GCS handoff. Builds FROM the spa-cloud image, so NO SSH forwarding
# is needed (the private deps are already in the base; OF3 is public).
#
# Prereq: the spa-cloud base image must be pullable (it is — pushed by build_and_push.sh), plus the
# AR docker repo + `gcloud auth configure-docker us-central1-docker.pkg.dev` (already set up).
# Usage:  ./build_and_push_combined.sh   (build only) | PUSH=1 ./build_and_push_combined.sh
set -euo pipefail

REGION="${REGION:-us-central1}"
PROJECT="${PROJECT:-spa-dev-499900}"
REPO="${REPO:-spa}"
TAG="${TAG:-0.1.0}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/spa-combined:${TAG}"

cd "$(dirname "$0")"

echo ">>> Building ${IMAGE} (FROM spa-cloud + OF3 conda env; the build-time OF3 import check gates it)"
DOCKER_BUILDKIT=1 docker build -f Dockerfile.combined -t "${IMAGE}" .

if [[ "${PUSH:-0}" == "1" ]]; then
    echo ">>> Pushing ${IMAGE}"
    docker push "${IMAGE}"
    echo ">>> Pushed. Smoke-test on an H100: full flywheel end-to-end on a tiny example (validates OF3 triton on Hopper)."
else
    echo ">>> Built (not pushed). Re-run with PUSH=1 to push."
fi
