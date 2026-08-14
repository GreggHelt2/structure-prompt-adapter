#!/usr/bin/env bash
# In-container ESM3 cache AUDIT (dev plan/69, prompted by Gregg's question 2026-08-14): what is
# actually inside gs://genomancer-spa-cache/esm3_cache.tar (234.67 GB)? The expanded esm3_cache/
# GCS "directory" only has 797 objects -- far short of the full CDDB corpus, the 94,914-structure
# training draw, or anything else expected -- so it is almost certainly a small probe, not the real
# cache. `tar -tf` lists entries without extracting, so this answers the question without ever writing
# 234 GB of individual files to disk. No GPU needed at all; runs on this project's existing H100 image
# only because that is the fastest path to something already working, per Gregg's own instruction.
set -uo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
CACHE_TAR_URI="${CACHE_TAR_URI:-$BUCKET/esm3_cache.tar}"
OUT="${OUT:-/workspace/cache_audit}"
NEIGHBORHOOD_TSV_URI="${NEIGHBORHOOD_TSV_URI:-}"   # optional: a GCS URI, fetched below if set

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

mkdir -p "$OUT"
NEIGHBORHOOD_TSV=""
if [ -n "$NEIGHBORHOOD_TSV_URI" ]; then
  NEIGHBORHOOD_TSV="$OUT/neighborhood.tsv"
  log "fetching $NEIGHBORHOOD_TSV_URI -> $NEIGHBORHOOD_TSV"
  gcloud storage cp "$NEIGHBORHOOD_TSV_URI" "$NEIGHBORHOOD_TSV"
fi
log "fetching $CACHE_TAR_URI -> $OUT/esm3_cache.tar (same-region GCS transfer, not to a local machine)"
gcloud storage cp "$CACHE_TAR_URI" "$OUT/esm3_cache.tar"
log "download done, listing tar contents (no extraction)"

tar -tf "$OUT/esm3_cache.tar" > "$OUT/listing.txt"
N=$(wc -l < "$OUT/listing.txt")
log "===== TOTAL ENTRIES IN esm3_cache.tar: $N ====="
log "first 5 entries:"; head -5 "$OUT/listing.txt"
log "last 5 entries:"; tail -5 "$OUT/listing.txt"

if [ -n "$NEIGHBORHOOD_TSV" ] && [ -f "$NEIGHBORHOOD_TSV" ]; then
  log "checking neighborhood coverage against $NEIGHBORHOOD_TSV"
  HIT=0; MISS=0
  while IFS=$'\t' read -r id pdb_file rest; do
    [ "$id" = "id" ] && continue  # header
    stem="${pdb_file%.pdb}"
    if grep -qF "${stem}.pt" "$OUT/listing.txt"; then
      HIT=$((HIT+1))
    else
      MISS=$((MISS+1))
    fi
  done < "$NEIGHBORHOOD_TSV"
  log "neighborhood coverage: $HIT hit, $MISS miss (of $((HIT+MISS)))"
fi

gcloud storage cp "$OUT/listing.txt" "$BUCKET/eval/esm3_cache_audit/listing_$(date -u +%Y%m%d-%H%M%S).txt" 2>/dev/null \
  && log "full listing staged to gs://genomancer-spa-cache/eval/esm3_cache_audit/"
log "===== CACHE AUDIT DONE: $N entries ====="
