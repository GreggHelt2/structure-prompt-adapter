#!/usr/bin/env bash
# In-container Option B REAL fine-tune (dev plan/69 SS4.11/4.12): fine-tunes RFD3's own weights on the
# ~50-structure all-beta CDDB neighborhood of A0A7S3EB45 (found by find_finetune_neighborhood.py,
# dev repo), NOT the toy path the smoke test (run_finetune_smoke.sh) used. Three things the smoke test
# did not need, all handled here: (1) a custom, tiny train split pointing at exactly the neighborhood,
# not the full 364k-row CDDB train split; (2) an ESM3 prompt cache built fresh for just those 50
# structures (data=cddb reads prompts from a cache, no live-encode fallback,
# src/spa/data/dataset.py:255-257); (3) staging the resulting checkpoints (both spa_*_final.pt AND
# rfd3_*_finetuned.pt) to GCS -- the smoke test only staged its log, a gap this fixes.
set -uo pipefail   # NOT -e: a training divergence must show up in the summary, not abort silently

PROJECT="${PROJECT:-spa-dev-499900}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
NEIGHBORHOOD_URI="${NEIGHBORHOOD_URI:-$BUCKET/eval/finetune_neighborhood/A0A7S3EB45}"
SPA_REPO="${SPA_REPO:-/opt/spa}"
HOST_LR="${HOST_LR:-1.0e-5}"          # dev plan/69 SS4.2's conservative starting guess, same as the smoke test
MAX_STEPS="${MAX_STEPS:-1000}"        # dev plan/69 SS4.11's recommendation
CKPT_EVERY="${CKPT_EVERY:-100}"       # inspect-and-stop-early cadence (SS4.11)
OUT="${OUT:-/workspace/finetune_real_out}"
CKPT_OUT_URI="${CKPT_OUT_URI:-$BUCKET/checkpoints/finetune-A0A7S3EB45-$(date -u +%Y%m%d-%H%M%S)}"

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

# --- GPU driver-path fix (verbatim from run_smoke.sh) ---
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/nvidia/bin:${PATH}"
ldconfig 2>/dev/null || true
( command -v nvidia-smi >/dev/null && nvidia-smi -L ) || echo "  nvidia-smi: n/a"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('GPU', torch.cuda.get_device_name(0), 'CUDA', torch.version.cuda)"

# --- SPA repo ---
[ -d "$SPA_REPO/.git" ] || git clone --depth 1 --branch "${REPO_REF:-main}" "${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}" "$SPA_REPO"
pip install -e "$SPA_REPO" --no-deps -q

# --- HF token, needed for ESM3 (run_cache_gen.sh's pattern) ---
export HF_TOKEN="$(gcloud secrets versions access latest --secret=spa-hf-token --project="$PROJECT")"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# --- Fetch the neighborhood: 50 PDBs + the custom train/manifest.parquet (dev plan/69 SS4.11/4.12) ---
STAGE="$OUT/neighborhood"
mkdir -p "$STAGE/pdb" "$STAGE/splits/train" "$OUT"
log "fetching neighborhood: $NEIGHBORHOOD_URI -> $STAGE"
gcloud storage rsync -r "$NEIGHBORHOOD_URI/pdb" "$STAGE/pdb"
gcloud storage cp "$NEIGHBORHOOD_URI/train/manifest.parquet" "$STAGE/splits/train/manifest.parquet"
N_PDB=$(ls "$STAGE/pdb" | wc -l)
log "  $N_PDB PDBs staged"

# --- Build the ESM3 cache for exactly these 50 structures (spa.prompt.build_cache via gen_esm3_cache.py;
# data=cddb's dataloader reads a cache with NO live-encode fallback, dataset.py _load_prompt) ---
CACHE_DIR="$OUT/esm3_cache"
mkdir -p "$CACHE_DIR"
log "building ESM3 cache for the neighborhood -> $CACHE_DIR"
python "$SPA_REPO/scripts/gen_esm3_cache.py" \
  data=cddb hardware=cloud_h100 \
  data.pdb_dir="$STAGE/pdb" \
  out_dir="$CACHE_DIR"

# --- Fetch the RFD3 checkpoint ---
mkdir -p /workspace/weights
gcloud storage cp "$RFD3_CKPT_URI" /workspace/weights/rfd3_latest.ckpt

log "config: host_lr=$HOST_LR max_steps=$MAX_STEPS ckpt_every=$CKPT_EVERY neighborhood=$N_PDB structures"

# --- Train: data=cddb, pointed at the custom (tiny) split rather than the real 364k-row train split.
# data.length_cap must cover the neighborhood's longest structure (up to 150, dev plan/69 SS4.11's own
# filter band) -- the cloud_h100 default of 256 already does, left unset deliberately (no override
# needed). data.require_cached_prompt is not needed either: every structure in this split has a cache
# entry by construction (the cache was just built from exactly this PDB set). ---
cd "$SPA_REPO"
python scripts/train.py \
  variant=C_n_by_1536 hardware=cloud_h100 data=cddb \
  paths.rfd3_ckpt=/workspace/weights/rfd3_latest.ckpt \
  paths.esm3_cache_dir="$CACHE_DIR" \
  data.pdb_dir="$STAGE/pdb" \
  data.splits_root="$STAGE/splits" \
  train.finetune_host_lr="$HOST_LR" \
  train.max_steps="$MAX_STEPS" \
  train.ckpt_every_steps="$CKPT_EVERY" \
  train.log_every_steps=10 \
  train.ckpt_dir="$OUT/ckpt" \
  hydra.run.dir="$OUT/hydra" \
  2>&1 | tee "$OUT/train.log"
STATUS=${PIPESTATUS[0]}

log "===== REAL FINE-TUNE TRAINING DONE (train.py exit=$STATUS) ====="

# --- Sanity checks, same shape as run_finetune_smoke.sh's verdict logic ---
python - "$OUT/train.log" "$OUT/ckpt" <<'PY'
import re, sys, os, glob
log_path, ckpt_dir = sys.argv[1], sys.argv[2]
lines = open(log_path).read().splitlines()
losses = [float(m.group(1)) for l in lines if (m := re.search(r"loss ([\d.eE+-]+) \| lr", l))]
print(f"[verdict] {len(losses)} logged loss values")
if not losses:
    print("[verdict] FAIL: no loss values parsed from the log at all")
    sys.exit(1)
if any(l != l for l in losses):
    print("[verdict] FAIL: NaN loss encountered")
    sys.exit(1)
if losses[-1] > losses[0] * 10:
    print(f"[verdict] FAIL: loss exploded ({losses[0]:.4f} -> {losses[-1]:.4f})")
    sys.exit(1)
print(f"[verdict] loss trajectory OK: {losses[0]:.4f} -> {losses[-1]:.4f} (min {min(losses):.4f})")
host_exports = glob.glob(os.path.join(ckpt_dir, "rfd3_*_finetuned.pt"))
if not host_exports:
    print("[verdict] FAIL: no rfd3_*_finetuned.pt export found")
    sys.exit(1)
print(f"[verdict] host export present: {host_exports}")
PY
VERDICT_STATUS=$?

# --- Stage EVERYTHING to GCS, not just the log (the smoke test's gap): the SPA adapter export, the
# fine-tuned host export, and the full checkpoint dir (so a crash mid-run is still recoverable). ---
log "staging checkpoints + log -> $CKPT_OUT_URI"
gcloud storage rsync -r "$OUT/ckpt" "$CKPT_OUT_URI" || log "  (checkpoint rsync issues)"
gcloud storage cp "$OUT/train.log" "$CKPT_OUT_URI/train.log" 2>/dev/null

log "===== REAL FINE-TUNE DONE (train=$STATUS, verdict=$VERDICT_STATUS) -> $CKPT_OUT_URI ====="
exit $(( STATUS != 0 ? STATUS : VERDICT_STATUS ))
