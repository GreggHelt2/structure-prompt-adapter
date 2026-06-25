#!/usr/bin/env bash
# Pull the ESM3 tensor cache from GCS to local NVMe for Phase-2 training: download the single tar,
# untar it to the destination (the RAID0-striped NVMe mount, which shows up as one device, e.g. /mnt/nvme).
# Pairs with run_cache_gen.sh step 6 (which uploads ONE tar instead of ~358k objects).
#
# Usage: load_cache.sh <tar_gcs_uri> <dest_dir>
#   load_cache.sh gs://genomancer-spa-cache/esm3_cache.tar /mnt/nvme/esm3_cache
#
# Uses sliced/parallel download (faster than streaming) then untar; needs ~2x the cache size transiently
# (tar + extracted) — fine on the ~750 GB RAID0 NVMe. For a tight-disk alternative, stream instead:
#   gcloud storage cp <tar_gcs_uri> - | tar -xf - -C <dest_dir>   (sequential -> no sliced parallelism)
set -euo pipefail
TAR_URI="${1:?usage: load_cache.sh <tar_gcs_uri> <dest_dir>}"
DEST="${2:?usage: load_cache.sh <tar_gcs_uri> <dest_dir>}"
GC="${GCLOUD:-/usr/local/bin/gcloud}"   # in-container gcloud (image 0.2.0+); override for other hosts
command -v "$GC" >/dev/null 2>&1 || GC=gcloud

mkdir -p "$DEST"
TMP_TAR="${DEST%/}.tar"   # alongside DEST, on the same NVMe
echo "[load_cache] downloading $TAR_URI -> $TMP_TAR"
t0=$SECONDS
"$GC" storage cp "$TAR_URI" "$TMP_TAR"
echo "[load_cache] untarring -> $DEST"
tar -xf "$TMP_TAR" -C "$DEST"
rm -f "$TMP_TAR"
echo "[load_cache] done in $((SECONDS-t0))s: $(find "$DEST" -name '*.pt' | wc -l) .pt files in $DEST"
