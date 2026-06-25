#!/usr/bin/env bash
# Poll a Vertex AI Custom Job: print its state + meaningful container logs.
# A single allowlistable command so the /loop poller runs without a permission prompt each fire.
# Usage: poll_job.sh <jobNumericId> [region] [project]
set -uo pipefail
JOBID="${1:?usage: poll_job.sh <jobNumericId> [region] [project]}"
REGION="${2:-us-central1}"
PROJECT="${3:-spa-dev-499900}"
GC=/home/user1/google-cloud-sdk/bin/gcloud

read -r STATE CREATE < <("$GC" ai custom-jobs describe "$JOBID" --region="$REGION" --project="$PROJECT" --format='value(state,createTime)' 2>/dev/null)
echo "state: ${STATE:-?}   (elapsed ~$(( ( $(date +%s) - $(date -d "${CREATE:-now}" +%s 2>/dev/null || date +%s) ) / 60 )) min since create)"
echo "=== container logs (framework-provisioning lines filtered) ==="
"$GC" logging read "resource.type=\"ml_job\" AND resource.labels.job_id=\"$JOBID\"" \
  --project="$PROJECT" --order=asc --limit=200 --format="value(textPayload)" 2>&1 \
  | sed '/^[[:space:]]*$/d' \
  | grep -vaE "provisioning job running framework" \
  | tail -30
