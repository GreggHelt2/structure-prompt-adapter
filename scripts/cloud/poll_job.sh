#!/usr/bin/env bash
# Poll a Vertex AI Custom Job: state + the *meaningful* container log lines (progress / checkpoints /
# summary / errors), ANSI-stripped and de-noised. ONE allowlistable command (matches Bash(bash *)) so the
# /loop poller runs with NO permission prompt per fire. Self-detects terminal state + prints SUMMARY/error.
# Usage: poll_job.sh <jobNumericId> [region] [project]
set -uo pipefail
JOBID="${1:?usage: poll_job.sh <jobNumericId> [region] [project]}"
REGION="${2:-us-west1}"
PROJECT="${3:-spa-dev-499900}"
GC=/home/user1/google-cloud-sdk/bin/gcloud

read -r STATE CREATE < <("$GC" ai custom-jobs describe "$JOBID" --region="$REGION" --project="$PROJECT" --format='value(state,createTime)' 2>/dev/null)
echo "state: ${STATE:-?}   (~$(( ( $(date +%s) - $(date -d "${CREATE:-now}" +%s 2>/dev/null || date +%s) ) / 60 )) min since create)"

logs(){ "$GC" logging read "resource.type=\"ml_job\" AND resource.labels.job_id=\"$JOBID\"" \
  --project="$PROJECT" --order=desc --limit="${1:-200}" --format='value(textPayload)' 2>&1 \
  | sed -r 's/\x1b\[[0-9;]*[a-zA-Z]//g'; }

echo "--- latest progress ---";    logs 200  | grep -aE '\[progress\]' | grep -avaE 'Remaining:|Completed:|Elapsed:' | head -1
echo "--- recent checkpoints ---"; logs 1500 | grep -aE 'checkpoint #' | head -3
W=$(logs 200 | grep -aE 'FATAL|\[warn\]' | head -5); [ -n "$W" ] && { echo "--- warnings/errors ---"; echo "$W"; }

case "${STATE:-}" in
  JOB_STATE_SUCCEEDED)
    echo "--- SUMMARY ---"; logs 60 | grep -aE 'SUMMARY|structures cached|cache size|cache-gen wall|throughput|full-cache|uploaded single tar|DONE' | head -10
    echo "--- tar in GCS ---"; "$GC" storage ls -l "gs://genomancer-spa-cache/esm3_cache.tar" 2>&1 | head -2
    echo "TERMINAL: SUCCEEDED" ;;
  JOB_STATE_FAILED|JOB_STATE_CANCELLED|JOB_STATE_EXPIRED)
    echo "--- error/traceback ---"; logs 150 | grep -aiE 'error|traceback|exception|FATAL|assert|no CUDA' | head -12
    echo "TERMINAL: ${STATE}" ;;
esac
