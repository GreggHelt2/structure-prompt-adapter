#!/usr/bin/env bash
# Pin WHAT CODE and WHAT IMAGE a cloud job runs. Sourced by every scripts/cloud/submit_*.sh.
#
# WHY. An audit of all 93 Vertex jobs this project has ever launched found that NONE of them pinned
# either one: 46 passed REPO_REF=main, 47 left it unset (also main), and all 93 referenced the
# container by a mutable tag rather than a digest. So the code and environment a given job ran are
# pinned only by its timestamp, and recovering them means guessing from `git log --before=<date>`.
# Training runs were luckier: the harness stamps `git_commit` into every checkpoint. Eval runs, which
# produce every published number, had nothing.
#
# WHAT THIS SETS (both overridable, both echoed at submit time so the value lands in the job record):
#   REPO_REF   default = the LOCAL public repo's HEAD sha, but only if that commit is on the remote.
#              Falls back to `main` with a loud warning otherwise, because pinning to a commit the job
#              cannot clone would turn a provenance improvement into a launch failure.
#   IMAGE      default = the caller's tag resolved to an immutable `@sha256:...` digest. Falls back to
#              the tag with a warning if the registry cannot be reached.
#
# ⚠️ `git clone --branch` accepts branch and tag names but NOT arbitrary shas, so pinning a sha also
# requires changing how the job checks out. Callers must use $BOOT_CHECKOUT below rather than
# hand-rolling a clone; it handles branch, tag and sha uniformly.

# --- REPO_REF ---------------------------------------------------------------------------------
_pin_repo_ref() {
  local repo_root ref
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  if [ -n "${REPO_REF:-}" ]; then
    echo ">>> [pin] REPO_REF explicitly set: ${REPO_REF}" >&2
    return
  fi
  if ! ref="$(git -C "$repo_root" rev-parse HEAD 2>/dev/null)"; then
    echo ">>> [pin] WARNING: not a git repo at ${repo_root}; falling back to REPO_REF=main" >&2
    REPO_REF=main
    return
  fi
  # The job clones from GitHub, so a sha only works if it has been pushed. `git branch -r --contains`
  # is the cheap local check; it needs a reasonably fresh remote ref, so refresh quietly first.
  git -C "$repo_root" fetch --quiet origin 2>/dev/null || true
  if [ -n "$(git -C "$repo_root" branch -r --contains "$ref" 2>/dev/null)" ]; then
    REPO_REF="$ref"
    echo ">>> [pin] REPO_REF pinned to local HEAD: ${REPO_REF}" >&2
    if ! git -C "$repo_root" diff --quiet HEAD 2>/dev/null; then
      echo ">>> [pin] ⚠️  WORKING TREE IS DIRTY. The job runs ${REPO_REF} as pushed, NOT your" >&2
      echo ">>> [pin]     uncommitted changes. Commit and push first if you meant to include them." >&2
    fi
  else
    echo ">>> [pin] ⚠️  HEAD ${ref} is NOT on any remote branch (unpushed?). Falling back to" >&2
    echo ">>> [pin]     REPO_REF=main, so this job will NOT be reproducibly pinned." >&2
    REPO_REF=main
  fi
}

# --- IMAGE ------------------------------------------------------------------------------------
_pin_image() {
  local digest base
  case "${IMAGE}" in
    *@sha256:*) echo ">>> [pin] IMAGE already digest-pinned" >&2; return ;;
  esac
  digest="$("${GCLOUD:-gcloud}" artifacts docker images describe "${IMAGE}" \
             --format='value(image_summary.digest)' 2>/dev/null)" || true
  if [ -n "$digest" ]; then
    base="${IMAGE%:*}"
    echo ">>> [pin] IMAGE ${IMAGE} -> ${base}@${digest}" >&2
    IMAGE="${base}@${digest}"
  else
    echo ">>> [pin] ⚠️  could not resolve a digest for ${IMAGE}; using the MUTABLE tag." >&2
  fi
}

# --- GUARD: the resolved REPO_REF must be a FETCHABLE remote ref ------------------------------
# WHY (2026-11-15, after a REPO_REF=<sha> smoke died in BOOT). $REPO_REF is interpolated into
# $BOOT_CHECKOUT as `git fetch --depth 1 origin $REPO_REF`, run on the Vertex VM. GitHub's shallow
# fetch resolves a BRANCH or TAG name but NOT a bare commit SHA, so a REPO_REF=<sha> dies on the VM
# with `fatal: couldn't find remote ref <sha>` AFTER the job has provisioned an H100 (BOOT exit 128).
# Every historical cloud run used REPO_REF=main (a branch); a session that "pins" a SHA for
# reproducibility silently breaks instead. Catch it at SUBMIT, where it costs nothing, not on the VM.
# ⛔ This REJECTS SHA pinning rather than making it work: real SHA pinning would need BOOT_CHECKOUT to
# do a full (non-shallow) fetch then checkout the sha, a separate change. Since shallow SHA-fetch is
# broken anyway, a clear rejection is the right behavior.
_validate_repo_ref() {
  local repo_root
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  if ! git -C "$repo_root" ls-remote --exit-code origin "$REPO_REF" >/dev/null 2>&1; then
    echo "FATAL: REPO_REF='${REPO_REF}' is not a branch or tag on origin." >&2
    echo "       The Vertex BOOT runs 'git fetch --depth 1 origin ${REPO_REF}', and GitHub's shallow" >&2
    echo "       fetch cannot resolve a bare commit SHA, so the job would die with" >&2
    echo "       'couldn't find remote ref ${REPO_REF}' AFTER provisioning the H100." >&2
    echo "       Use a branch or tag name, or leave REPO_REF=main (the default every prior run used)." >&2
    exit 2
  fi
  echo ">>> [pin] REPO_REF '${REPO_REF}' verified as a fetchable remote ref" >&2
}

# --- GUARD: refuse a 2nd job in a region that already has one (H100 quota = 1 per region) ------
# WHY (2026-11-15, after 6 jobs were submitted at once against the quota). H100 quota is 1
# a3-highgpu-1g PER REGION (the always-loaded `h100-quota-per-region` memory): a 2nd job in a region
# whose slot is taken just sits quota-blocked in us-west1, and in us-central1 FAILS after ~80 min with
# a capacity timeout. The convention is to CHAIN one job per region (watch_job.sh -> launch next). This
# catches a double-submit at SUBMIT time, at zero cost, before a job is created.
# ⛔ Checks THIS project's jobs in the region; a separate GCP project sharing the account is not seen.
# Skipped under DRY_RUN (previewing creates nothing). Deliberate queue: SPA_ALLOW_REGION_BUSY=<reason 12+ chars>.
_validate_region_free() {
  [ "${DRY_RUN:-0}" = "1" ] && return 0
  local region="${REGION:-}" reason="${SPA_ALLOW_REGION_BUSY:-}" busy
  [ -n "$region" ] || { echo ">>> [pin] REGION unset at guard time; skipping region-occupancy check" >&2; return 0; }
  if [ -n "$reason" ]; then
    [ "${#reason}" -ge 12 ] || { echo "FATAL: SPA_ALLOW_REGION_BUSY must state a reason of 12+ chars." >&2; exit 2; }
    echo ">>> [pin] SPA_ALLOW_REGION_BUSY set ($reason); skipping the region-occupancy check" >&2
    return 0
  fi
  busy="$("${GCLOUD:-gcloud}" ai custom-jobs list --project="${PROJECT:-spa-dev-499900}" \
            --region="$region" --filter='state=JOB_STATE_RUNNING OR state=JOB_STATE_PENDING' \
            --format='value(name)' 2>/dev/null | head -3)"
  if [ -n "$busy" ]; then
    echo "FATAL: region $region already has a RUNNING/PENDING job; H100 quota is 1 per region." >&2
    echo "       A 2nd job just sits quota-blocked (us-west1) or FAILS after ~80 min (us-central1)." >&2
    echo "       CHAIN instead: watch the running job (scripts/cloud/watch_job.sh <id> $region)," >&2
    echo "       then submit the next when it finishes. Occupying job(s):" >&2
    echo "$busy" | sed 's/^/         /' >&2
    echo "       To queue deliberately anyway: SPA_ALLOW_REGION_BUSY='<reason, 12+ chars>'." >&2
    exit 2
  fi
  echo ">>> [pin] region $region is free (no running/pending job)" >&2
}

_pin_repo_ref
_validate_repo_ref
_pin_image
_validate_region_free

# Checkout that works for a branch, a tag OR a sha. `git clone --branch` does not accept a sha, which
# is why this replaces the previous clone line rather than parameterizing it.
#
# ⚠️ THIS STRING MUST CONTAIN NO DOUBLE QUOTES AND NO SQUARE BRACKETS. Callers embed it in $BOOT,
# which the submit scripts interpolate into a YAML double-quoted flow scalar (`args: ["${BOOT}"]`);
# a literal `"` terminates that scalar and produces invalid YAML that gcloud rejects at submit time.
# Square brackets would be a bash glob in the container's unquoted `echo`. Hence the bare
# KEY=value form below, which is also greppable in the job log.
BOOT_CHECKOUT="git init -q /opt/spa && git -C /opt/spa remote add origin ${REPO_URL} \
&& git -C /opt/spa fetch -q --depth 1 origin ${REPO_REF} && git -C /opt/spa checkout -q --detach FETCH_HEAD \
&& echo PINNED_CODE_REV=\$(git -C /opt/spa rev-parse HEAD)"
