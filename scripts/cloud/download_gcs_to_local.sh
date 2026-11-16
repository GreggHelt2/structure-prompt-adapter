#!/usr/bin/env bash
# download_gcs_to_local.sh — download an H100 cloud run's results from GCS to the LOCAL mirror, COPY-ONLY.
# ⇕ COUNTERPART of the dev-repo scripts/tools/promote_run.sh: that one goes LOCAL-origin -> census + GCS
#   archive (and re-uploads); THIS one is the cloud -> local direction and NEVER re-uploads. For a
#   cloud-origin run (already in gs://.../eval/), use THIS, not promote_run.sh (which would 2x it in GCS).
#
# WHY THIS EXISTS (2026-11-16). Cloud (Vertex H100) jobs push their outputs to gs://$BUCKET/eval/<family>
# (and /results/), which is the primary GCS copy. The convention for getting them local for analysis is a
# copy-only `gcloud storage rsync` into outputs/_cloud_mirror/ (dev plan/30 §2.1) -- GCS untouched, no
# re-upload. But that rsync was only ever run BY HAND, in batches (2026-09-11, 2026-11-14), so a run
# finishing after the last manual pull sits GCS-only until someone remembers. This script is that missing
# automation: one conventional command to pull a run local.
#
# ⛔ THIS IS NOT promote_run.sh. Promotion re-archives the run as a tarball to gs://.../outputs-archive/,
#    which for a CLOUD-origin run (already in gs://.../eval/) is a redundant SECOND GCS copy. This script
#    only ever COPIES DOWN: the GCS source is never modified, deleted, or re-uploaded.
#
# Usage:
#   download_gcs_to_local.sh <family> [--prefix eval|results] [--dry-run]
#       mirror gs://$BUCKET/<prefix>/<family>  ->  $MIRROR/<prefix>/<family>   (prefix default: eval)
#   download_gcs_to_local.sh --gcs <gs://...> --dest <localdir> [--dry-run]
#       mirror an arbitrary GCS prefix to an arbitrary local dir (same copy-only semantics)
#
# Env: BUCKET (default gs://genomancer-spa-cache), SPA_PROJECT_ROOT (default /home/user1/projects/spa),
#      MIRROR (default $SPA_PROJECT_ROOT/outputs/_cloud_mirror), GCLOUD (default: gcloud on PATH).
set -uo pipefail

BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
SPA_PROJECT_ROOT="${SPA_PROJECT_ROOT:-/home/user1/projects/spa}"
MIRROR="${MIRROR:-$SPA_PROJECT_ROOT/outputs/_cloud_mirror}"
GCLOUD="${GCLOUD:-gcloud}"
PREFIX="eval"; DRY=0; FAMILY=""; GCS=""; DEST=""

while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --gcs) GCS="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    -*) echo "download_gcs_to_local.sh: unknown flag $1" >&2; exit 2 ;;
    *) FAMILY="$1"; shift ;;
  esac
done

# Resolve source + dest.
if [ -n "$GCS" ]; then
  [ -n "$DEST" ] || { echo "FATAL: --gcs requires --dest" >&2; exit 2; }
  SRC="$GCS"
else
  [ -n "$FAMILY" ] || { echo "FATAL: give a <family> (or --gcs/--dest). See --help." >&2; exit 2; }
  case "$PREFIX" in eval|results) : ;; *) echo "FATAL: --prefix must be eval or results" >&2; exit 2 ;; esac
  SRC="$BUCKET/$PREFIX/$FAMILY"
  DEST="$MIRROR/$PREFIX/$FAMILY"
fi

# Guard: the source must exist in GCS (a typo'd family would otherwise create an empty local dir).
if ! "$GCLOUD" storage ls "$SRC" >/dev/null 2>&1; then
  echo "FATAL: GCS source not found: $SRC" >&2
  echo "       Check the family name (gcloud storage ls $BUCKET/$PREFIX/)." >&2
  exit 2
fi

echo ">>> download_gcs_to_local: COPY-ONLY mirror (GCS source is never modified)"
echo "    source: $SRC"
echo "    dest:   $DEST"
mkdir -p "$DEST"

# Copy-only: no --delete-unmatched-destination-objects, so nothing local is ever removed and GCS is
# never written. --dry-run lists what WOULD be copied (prints to stderr) and transfers nothing.
if [ "$DRY" = "1" ]; then
  echo ">>> DRY_RUN: listing what would be copied (nothing transferred) ..."
  "$GCLOUD" storage rsync -r --dry-run "$SRC" "$DEST" 2>&1 | grep -E "^Would copy" | sed 's/^/    /' | head -50
  n=$("$GCLOUD" storage rsync -r --dry-run "$SRC" "$DEST" 2>&1 | grep -cE "^Would copy")
  echo ">>> DRY_RUN: $n object(s) would be copied."
  exit 0
fi

echo ">>> rsync (copy-only) ..."
"$GCLOUD" storage rsync -r "$SRC" "$DEST"
echo ">>> download_gcs_to_local DONE: $SRC -> $DEST (GCS unchanged)"
