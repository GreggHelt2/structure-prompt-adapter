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
pip install -e "$SPA_REPO" --no-deps -q   # REQUIRED so the generate step's precompute_prompts.py can `import spa` (same as run_cache_gen.sh / the flywheel runners; its omission is why the first generate failed with ModuleNotFoundError: No module named 'spa')
n=0; ok=0
MAN="$SPA_REPO/$MANIFEST_REL"
[ -f "$MAN" ] || { log "FATAL: manifest $MAN not found"; exit 1; }

if [ -z "${GENERATE_ONLY:-}" ]; then   # GENERATE_ONLY=1 skips hydrate + CDDB extract + b4 pull (those 34 tensors already landed); re-runs only the 17-fold generate
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
# EXTRA_IDS: additional CDDB accessions to extract from the hydrated cache (free, prep dominates).
# e.g. the short/medium folds so the V2 dilution-vs-length analysis spans the full 52-373 aa range,
# not just the long band. CDDB only (external PDBs come via the b4/prep pull below).
for x in ${EXTRA_IDS:+${EXTRA_IDS//,/ }}; do [ -n "$x" ] && echo "$x"; done >> "$OUT/ids.txt"
sort -u "$OUT/ids.txt" -o "$OUT/ids.txt"
log "extracting $(wc -l < "$OUT/ids.txt") CDDB fold tensors (28 stratified + EXTRA_IDS)"

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

# --- Named recognizable-fold prompts (plan/153 §9 #13 / G4): already ESM3-encoded by the b4 run and
#     staged individually at eval/b4/prep (these are REAL PDB, NOT in the CDDB tar). Pull + convert to
#     .npy so the same V2 dilution analysis can include the recognizable folds. 1CEX/1EMA are not in
#     b4/prep (separate encodes) and are skipped. ---
B4_IDS="${B4_IDS:-1QY3,6E6R,7RPN,7SBD,7YII,8SIU}"
for bid in ${B4_IDS//,/ }; do
  [ -z "$bid" ] && continue
  if gcloud storage cp "$BUCKET/eval/b4/prep/$bid.pt" "$OUT/named_$bid.pt" 2>/dev/null; then
    python - "$OUT/named_$bid.pt" "$OUT/named_$bid.npy" "$bid" <<'PY'
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
    print(f"  named {fid}: WARN no [L,1536] tensor"); sys.exit(0)
a = t.float().cpu().numpy(); np.save(dst, a); print(f"  named {fid}: {a.shape} -> {dst}")
PY
    log "named $bid pulled from b4/prep + converted"
  else
    log "named $bid NOT in b4/prep (skipped)"
  fi
done

fi   # end GENERATE_ONLY guard (hydrate + CDDB extract + b4 pull)

# --- NEW recognizable >250 PDBs (plan/153 §9 #13 / G4): REAL PDB, in no cache, so FRESH ESM3 N×1536
#     generation. Download from RCSB, extract the LONGEST polymer chain (the fold-bearing one: chain A
#     for homo-oligomers, the LRR chain for the 1DFJ complex), then ONE ESM3 encode over all of them
#     via precompute_prompts.py (loads ESM3 once). 15 folds: TIM(1GOX,2MNR) Rossmann(1GPD,6ADH,6LDH)
#     propeller(2BAT) porin(2OMF) P450(2CPP) LRR(1DFJ) β-helix(1AIR,1DAB) α-solenoid(2BCT,1B3U)
#     serpin(7API,1QLP) α-globular(1AO6 serum albumin, 1ANN annexin). 17 folds. ---
GEN_IDS="${GEN_IDS:-1GOX,2MNR,1GPD,6ADH,6LDH,2BAT,2OMF,2CPP,1DFJ,1AIR,1DAB,2BCT,1B3U,7API,1QLP,1AO6,1ANN}"
GENDIR="$OUT/gen"; mkdir -p "$GENDIR"
for gid in ${GEN_IDS//,/ }; do
  [ -z "$gid" ] && continue
  if ! curl -fsSL "https://files.rcsb.org/download/$gid.pdb" -o "$GENDIR/${gid}_full.pdb"; then
    log "gen $gid: RCSB download FAILED (skip)"; continue; fi
  python - "$GENDIR/${gid}_full.pdb" "$GENDIR/$gid.pdb" "$gid" <<'PY'
import sys
from collections import defaultdict
inp, outp, gid = sys.argv[1], sys.argv[2], sys.argv[3]
res = defaultdict(set); lines = defaultdict(list)
for ln in open(inp):
    if ln.startswith("ATOM"):
        ch = ln[21]; res[ch].add(ln[22:27]); lines[ch].append(ln)
if not res:
    print(f"  gen {gid}: no ATOM records"); sys.exit(1)
best = max(res, key=lambda c: len(res[c]))
with open(outp, "w") as f:
    f.writelines(lines[best]); f.write("END\n")
print(f"  gen {gid}: chain {best}, {len(res[best])} residues -> longest-chain PDB")
PY
done
CHAINS=$(ls "$GENDIR"/*.pdb 2>/dev/null | grep -v '_full')
if [ -n "$CHAINS" ]; then
  log "ESM3-encoding $(echo "$CHAINS" | wc -w) recognizable PDBs (fresh N×1536) via precompute_prompts.py"
  python "$SPA_REPO/scripts/eval/precompute_prompts.py" --pdbs $CHAINS --out-dir "$OUT" --device cuda \
    && log "ESM3 encode OK" || log "ESM3 encode FAILED (continuing to upload what landed)"
  for gid in ${GEN_IDS//,/ }; do
    [ -f "$OUT/$gid.pt" ] || { log "gen $gid: no .pt produced"; continue; }
    python - "$OUT/$gid.pt" "$OUT/$gid.npy" "$gid" <<'PY'
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
    print(f"  gen {fid}: WARN no [L,1536] tensor"); sys.exit(0)
a = t.float().cpu().numpy(); np.save(dst, a); print(f"  gen {fid}: {a.shape} -> {dst}")
PY
  done
  for c in $CHAINS; do gcloud storage cp "$c" "$OUT_URI/" 2>/dev/null; done
  log "uploaded $(echo "$CHAINS" | wc -w) reference chain PDBs -> $OUT_URI/ (for G4-A TM scoring)"
fi

log "uploading $ok CDDB + named + generated .pt + .npy -> $OUT_URI/"
gcloud storage cp "$OUT"/*.pt  "$OUT_URI/" 2>/dev/null
gcloud storage cp "$OUT"/*.npy "$OUT_URI/" 2>/dev/null
log "===== EXTRACT DONE: $ok/$n tensors -> $OUT_URI ====="
