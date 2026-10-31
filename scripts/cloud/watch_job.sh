#!/usr/bin/env bash
# Live-follow a Vertex AI Custom Job by SURFACING NEW SCRIPT OUTPUT as it is produced — a `tail -f` over
# Cloud Logging — instead of re-printing "still RUNNING". Each tick prints only the container log lines
# that appeared since the previous tick (ANSI-stripped); a quiet tick prints nothing but a single dot, so
# the signal is the script's own progress, not the poller's heartbeat.
#
# ONE allowlistable command (matches Bash(bash *)): run under `!` to watch live, or detached
# (run_in_background) so it wakes the session on the TERMINAL verdict.
#
# ⛔ The verdict is CONTENT-BASED, not JOB_STATE: a driver can swallow an error and still let Vertex report
#    JOB_STATE_SUCCEEDED (measured 2026-10-31: an ESM3 401 → `EXTRACT DONE: 0/0` → job SUCCEEDED). So OK is
#    declared only on a real success sentinel, and FAIL on a known failure line, regardless of job state.
#    (When the drivers gain a standard `=== RESULT: OK|FAIL ... ===` line, that becomes the sole sentinel.)
#
# Usage: watch_job.sh <jobNumericId> [region] [project]
#   INTERVAL=45   seconds between ticks (default 45)
#   MAX_TICKS=80  safety cap (default 80 ≈ 1 h at 45 s)
#   ARTIFACT=gs://...   optional object to check for existence on a terminal OK
set -uo pipefail
JOBID="${1:?usage: watch_job.sh <jobNumericId> [region] [project]}"
REGION="${2:-us-west1}"
PROJECT="${3:-spa-dev-499900}"
GC="${GC:-gcloud}"                 # PATH lookup, portable across machines
INTERVAL="${INTERVAL:-45}"
MAX_TICKS="${MAX_TICKS:-80}"
ARTIFACT="${ARTIFACT:-}"

JPATH="projects/$PROJECT/locations/$REGION/customJobs/$JOBID"
strip(){ sed -r 's/\x1b\[[0-9;]*[a-zA-Z]//g'; }

# Known sentinels for the current cloud drivers (run_extract_prompt_tensors / run_longfold_steer / run_g4a /
# run_cache_gen). Superseded by a standard RESULT line once the drivers emit one.
# ⚠️ `EXTRACT DONE: 0/0` is NOT a failure under GENERATE_ONLY=1 (the $ok counter tracks only the skipped
#    CDDB loop; the generated .pt still uploads) — verified on the 2OMF smoke 2026-10-31, where a good run
#    printed `EXTRACT DONE: 0/0` right after `ESM3 encode OK`. So do not treat `0/0` as FAIL.
FAIL_RE='encode FAILED|no \.pt produced|RepositoryNotFound|signature verification failed|no CUDA|^\s*Traceback|FATAL|=== RESULT: FAIL'
OK_RE='ESM3 encode OK|EXTRACT DONE: [1-9]|gen [0-9A-Za-z]+: \([0-9]+, 1536\)|uploaded single tar|===== SUMMARY =====|STAGE .* DONE|=== RESULT: OK'

# init the incremental bound to the job's createTime so the first tick shows everything so far
read -r STATE LAST < <("$GC" ai custom-jobs describe "$JPATH" --region="$REGION" --project="$PROJECT" --format='value(state,createTime)' 2>/dev/null)
echo ">>> watching $JOBID ($REGION)  state=${STATE:-?}  interval=${INTERVAL}s"
LAST="${LAST:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"

VERDICT=""
for ((i=1; i<=MAX_TICKS; i++)); do
  # new lines strictly after the last-seen timestamp, oldest-first, so output reads top-to-bottom
  NEW=$("$GC" logging read \
        "resource.type=\"ml_job\" AND resource.labels.job_id=\"$JOBID\" AND timestamp>\"$LAST\"" \
        --project="$PROJECT" --order=asc --limit=1000 \
        --format='value(timestamp,textPayload)' 2>/dev/null | strip)
  if [ -n "$NEW" ]; then
    # advance the bound to the newest timestamp we just printed (first field of the last line)
    NEWLAST=$(printf '%s\n' "$NEW" | tail -1 | awk '{print $1}')
    [ -n "$NEWLAST" ] && LAST="$NEWLAST"
    # drop the leading timestamp column for readability; keep the script's own [HH:MM:SS] prefix
    printf '%s\n' "$NEW" | sed -E 's/^[0-9T:.Z-]+[[:space:]]+//'
    echo "$NEW" | grep -qiE "$FAIL_RE" && VERDICT="FAIL"
    echo "$NEW" | grep -qE  "$OK_RE"   && [ -z "$VERDICT" ] && VERDICT="OK"
  else
    printf '.'   # quiet tick: one dot, no "still RUNNING" spam
  fi

  STATE=$("$GC" ai custom-jobs describe "$JPATH" --region="$REGION" --project="$PROJECT" --format='value(state)' 2>/dev/null)
  case "$STATE" in
    JOB_STATE_SUCCEEDED|JOB_STATE_FAILED|JOB_STATE_CANCELLED|JOB_STATE_EXPIRED)
      echo; echo ">>> job state TERMINAL: $STATE" ; break ;;
  esac
  sleep "$INTERVAL"
done

# final sweep of the tail for a sentinel the incremental loop may have raced past
TAIL=$("$GC" logging read "resource.type=\"ml_job\" AND resource.labels.job_id=\"$JOBID\"" \
       --project="$PROJECT" --order=desc --limit=60 --format='value(textPayload)' 2>/dev/null | strip)
[ -z "$VERDICT" ] && echo "$TAIL" | grep -qiE "$FAIL_RE" && VERDICT="FAIL"
[ -z "$VERDICT" ] && echo "$TAIL" | grep -qE  "$OK_RE"   && VERDICT="OK"

echo ">>> CONTENT VERDICT: ${VERDICT:-UNKNOWN}  (job state: ${STATE:-?})"
[ "${VERDICT:-}" != "OK" ] && { echo "--- tail (newest first) ---"; printf '%s\n' "$TAIL" | head -15; }
if [ -n "$ARTIFACT" ]; then
  echo "--- artifact ---"; "$GC" storage ls "$ARTIFACT" 2>&1 | head -3
fi
