#!/usr/bin/env bash
# Extract the ESM3 prompt tensors for the plan/153 long-fold set from the full cache tar, for the V2
# attention-dilution analysis (plan/153 §9 item 3 / §6 V2). The 28 tensors live ONLY inside
# esm3_cache.tar (234 GiB, no individual GCS objects) and the orchestration Mac has no torch, so this
# runs on a GCS-connected a3 box: hydrate the cache to NVMe (the established load_cache.sh pattern),
# copy each fold's .pt, dump a numpy .npy (so the Mac can load it without torch), and upload BOTH to
# gs://.../eval/longfold_steer/prompt_tensors/. No GPU compute (the a3 is for NVMe + bandwidth).
# Launch via: RUN_SCRIPT=run_extract_prompt_tensors.sh REGION=us-west1 bash submit_longfold_job.sh
set -uo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
CACHE_TAR="${CACHE_TAR:-$BUCKET/esm3_cache.tar}"
OUT_URI="${OUT_URI:-$BUCKET/eval/longfold_steer/prompt_tensors}"
SPA_REPO="${SPA_REPO:-/opt/spa}"
MANIFEST_REL="${MANIFEST_REL:-configs/eval/manifest_longfold_steer.json}"
SCRATCH="${SCRATCH:-/workspace}"
ESM3_CACHE_DIR="${ESM3_CACHE_DIR:-$SCRATCH/esm3_cache}"
OUT="$SCRATCH/prompt_tensors_out"; mkdir -p "$OUT"

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

# --- SPA repo (for load_cache.sh + the manifest) ---
[ -d "$SPA_REPO/.git" ] || git clone --depth 1 --branch "${REPO_REF:-main}" "${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}" "$SPA_REPO"
MAN="$SPA_REPO/$MANIFEST_REL"
[ -f "$MAN" ] || { log "FATAL: manifest $MAN not found"; exit 1; }

# --- Hydrate the full ESM3 cache to NVMe (established load_cache.sh pattern) ---
if [ ! -d "$ESM3_CACHE_DIR" ] || [ -z "$(ls -A "$ESM3_CACHE_DIR" 2>/dev/null)" ]; then
  log "hydrate ESM3 cache: $CACHE_TAR -> $ESM3_CACHE_DIR"
  GCLOUD="$(command -v gcloud)" bash "$SPA_REPO/scripts/cloud/load_cache.sh" "$CACHE_TAR" "$ESM3_CACHE_DIR"
fi
log "ESM3 cache: $(find "$ESM3_CACHE_DIR" -name '*.pt' 2>/dev/null | wc -l) .pt present"

# --- Fold ids (all folds in the manifest; SUBSET_IDS narrows for a smoke) ---
python -c "import json,os
sub=set(x for x in os.environ.get('SUBSET_IDS','').split(',') if x)
for p in json.load(open('$MAN'))['prompts']:
    if not sub or p['id'] in sub: print(p['id'])" > "$OUT/ids.txt"
log "extracting $(wc -l < "$OUT/ids.txt") fold tensors"

n=0; ok=0
while read -r id; do
  [ -z "$id" ] && continue
  n=$((n+1)); stem="AF-$id-F1-model_v4_esmfold_v1"; pt="$ESM3_CACHE_DIR/$stem.pt"
  if [ ! -f "$pt" ]; then log "[$n] $id MISSING ($pt)"; continue; fi
  cp "$pt" "$OUT/$id.pt"
  # torch -> numpy; the cache holds a [L,1536] fp16 tensor (bare or in a dict). Find the 2-D [*,1536] array.
  python - "$pt" "$OUT/$id.npy" "$id" <<'PY'
import sys, numpy as np, torch
src, dst, fid = sys.argv[1], sys.argv[2], sys.argv[3]
obj = torch.load(src, map_location="cpu")
def find(o):
    if torch.is_tensor(o):
        return o if (o.ndim == 2 and o.shape[-1] == 1536) else None
    if isinstance(o, dict):
        for v in o.values():
            r = find(v)
            if r is not None: return r
    if isinstance(o, (list, tuple)):
        for v in o:
            r = find(v)
            if r is not None: return r
    return None
t = find(obj)
if t is None:
    print(f"  {fid}: WARN no [L,1536] tensor found (type {type(obj).__name__})"); sys.exit(0)
a = t.float().cpu().numpy()
np.save(dst, a)
print(f"  {fid}: {a.shape} {a.dtype} -> {dst}")
PY
  ok=$((ok+1))
done < "$OUT/ids.txt"

log "uploading $ok .pt + .npy -> $OUT_URI/"
gcloud storage cp "$OUT"/*.pt  "$OUT_URI/" 2>/dev/null
gcloud storage cp "$OUT"/*.npy "$OUT_URI/" 2>/dev/null
log "===== EXTRACT DONE: $ok/$n tensors -> $OUT_URI ====="
