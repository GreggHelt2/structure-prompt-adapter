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

_pin_repo_ref
_pin_image

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
