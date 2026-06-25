#!/usr/bin/env bash
# Poll a Vertex AI Custom Job: print its state + meaningful container logs.
# A single allowlistable command so the /loop poller runs without a permission prompt each fire.
# Usage: poll_job.sh <jobNumericId> [region] [project]
set -uo pipefail
JOBID="${1:?usage: poll_job.sh <jobNumericId> [region] [project]}"
REGION="${2:-us-central1}"
PROJECT="${3:-spa-dev-499900}"
GC=/home/user1/google-cloud-sdk/bin/gcloud

echo "state: $("$GC" ai custom-jobs describe "$JOBID" --region="$REGION" --project="$PROJECT" --format='value(state)' 2>/dev/null)"
echo "=== container logs (framework-provisioning lines filtered) ==="
"$GC" logging read "resource.type=\"ml_job\" AND resource.labels.job_id=\"$JOBID\"" \
  --project="$PROJECT" --order=asc --limit=200 --format="value(textPayload)" 2>&1 \
  | sed '/^[[:space:]]*$/d' \
  | grep -vaE "provisioning job running framework" \
  | tail -30
