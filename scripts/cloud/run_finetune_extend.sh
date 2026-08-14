#!/usr/bin/env bash
# In-container Option B fine-tune EXTEND (dev plan/69 SS4.15): resumes the existing fine-tune
# checkpoint and continues training to a larger MAX_STEPS, rather than restarting from scratch. Same
# neighborhood/host_lr/config as run_finetune_real.sh; the only new piece is downloading the existing
# spa_Nx1536_last.pt into the local ckpt_dir BEFORE calling train.py, so harness.py's auto-resume
# (harness.py:474-478: resumes iff a file literally named spa_{cfg.variant.name}_last.pt already
# exists at cfg.train.ckpt_dir) picks it up. build_scheduler (harness.py:67-88) rebuilds the
# warmup->cosine-decay schedule fresh from whatever MAX_STEPS is passed at invocation time, so
# resuming with a larger MAX_STEPS correctly re-shapes the LR curve around the new total rather than
# continuing at the near-zero LR the original 1000-step schedule ended at (verified by reading the
# code, not assumed; this is the same mechanism behind this project's already-validated 30k->45k
# SPA-adapter extend-run).
set -uo pipefail   # NOT -e: a training divergence must show up in the summary, not abort silently

PROJECT="${PROJECT:-spa-dev-499900}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
NEIGHBORHOOD_URI="${NEIGHBORHOOD_URI:-$BUCKET/eval/finetune_neighborhood/A0A7S3EB45}"
CKPT_OUT_URI="${CKPT_OUT_URI:-$BUCKET/checkpoints/finetune-A0A7S3EB45-20260814-045849}"
RESUME_FROM_CKPT_URI="${RESUME_FROM_CKPT_URI:-$CKPT_OUT_URI/spa_Nx1536_last.pt}"
SPA_REPO="${SPA_REPO:-/opt/spa}"
HOST_LR="${HOST_LR:-1.0e-5}"          # unchanged from the original run
MAX_STEPS="${MAX_STEPS:?set MAX_STEPS to the new target total step count (e.g. 3000)}"
CKPT_EVERY="${CKPT_EVERY:-100}"
OUT="${OUT:-/workspace/finetune_extend_out}"

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

# --- GPU driver-path fix (verbatim from run_finetune_real.sh) ---
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/nvidia/bin:${PATH}"
ldconfig 2>/dev/null || true
( command -v nvidia-smi >/dev/null && nvidia-smi -L ) || echo "  nvidia-smi: n/a"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('GPU', torch.cuda.get_device_name(0), 'CUDA', torch.version.cuda)"

# --- SPA repo ---
[ -d "$SPA_REPO/.git" ] || git clone --depth 1 --branch "${REPO_REF:-main}" "${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}" "$SPA_REPO"
pip install -e "$SPA_REPO" --no-deps -q

# --- Fetch the neighborhood (same 50 PDBs + splits as the original run) ---
STAGE="$OUT/neighborhood"
mkdir -p "$STAGE/pdb" "$STAGE/splits/train" "$STAGE/splits/validate" "$OUT"
log "fetching neighborhood: $NEIGHBORHOOD_URI -> $STAGE"
gcloud storage rsync -r "$NEIGHBORHOOD_URI/pdb" "$STAGE/pdb"
gcloud storage cp "$NEIGHBORHOOD_URI/train/manifest.parquet" "$STAGE/splits/train/manifest.parquet"
cp "$STAGE/splits/train/manifest.parquet" "$STAGE/splits/validate/manifest.parquet"
N_PDB=$(ls "$STAGE/pdb" | wc -l)
log "  $N_PDB PDBs staged"

# --- Build the ESM3 cache fresh (cheap, ~50 structures; ephemeral container has none of it) ---
CACHE_DIR="$OUT/esm3_cache"
mkdir -p "$CACHE_DIR"
log "building ESM3 cache for the neighborhood -> $CACHE_DIR"
python "$SPA_REPO/scripts/gen_esm3_cache.py" \
  data=cddb hardware=cloud_h100 \
  data.pdb_dir="$STAGE/pdb" \
  out_dir="$CACHE_DIR"

N_CACHED=$(ls "$CACHE_DIR"/*.pt 2>/dev/null | wc -l)
if [ "$N_CACHED" -lt "$N_PDB" ]; then
  log "FATAL: ESM3 cache has $N_CACHED/$N_PDB structures -- cache-gen did not complete cleanly"
  exit 1
fi
log "ESM3 cache OK: $N_CACHED/$N_PDB structures"

# --- Fetch the base RFD3 checkpoint (still needed to construct net/shadow before resume overrides
# the relevant weights from the checkpoint's own "host"/"adapter" state, harness.py:476-478) ---
mkdir -p /workspace/weights
gcloud storage cp "$RFD3_CKPT_URI" /workspace/weights/rfd3_latest.ckpt

# --- Resolve foundry's training-transform cfg dir (verbatim from run_finetune_real.sh) ---
log "resolving foundry train-cfg dir"
FOUNDRY_TRAIN_CFG_DIR="$(python - <<'PY'
import os, glob
try:
    import rfd3
except Exception:
    print(""); raise SystemExit
seen, hit = set(), ""
start = os.path.dirname(rfd3.__file__)
roots = [start, os.path.dirname(start), os.path.dirname(os.path.dirname(start))]
for r in roots:
    if not r or r in seen: continue
    seen.add(r)
    for p in glob.glob(os.path.join(r, "**", "datasets", "train", "pdb", "base_transform_args.yaml"), recursive=True):
        hit = os.path.dirname(os.path.dirname(p)); break
    if hit: break
print(hit)
PY
)"
if [ -z "$FOUNDRY_TRAIN_CFG_DIR" ] || [ ! -f "$FOUNDRY_TRAIN_CFG_DIR/pdb/base_transform_args.yaml" ] \
   || [ ! -f "$FOUNDRY_TRAIN_CFG_DIR/rfd3_monomer_distillation.yaml" ]; then
  log "FATAL: could not locate foundry train-cfg dir (needs pdb/base_transform_args.yaml + rfd3_monomer_distillation.yaml)"
  exit 1
fi
log "FOUNDRY_TRAIN_CFG_DIR=$FOUNDRY_TRAIN_CFG_DIR"

# --- Fetch the existing checkpoint to resume from, into train.ckpt_dir under the EXACT filename
# harness.py's auto-resume looks for (spa_{cfg.variant.name}_last.pt = spa_Nx1536_last.pt, now that
# configs/variant/C_n_by_1536.yaml's name: field was changed C -> Nx1536, public ed4847b). ---
mkdir -p "$OUT/ckpt"
log "fetching resume checkpoint: $RESUME_FROM_CKPT_URI -> $OUT/ckpt/spa_Nx1536_last.pt"
gcloud storage cp "$RESUME_FROM_CKPT_URI" "$OUT/ckpt/spa_Nx1536_last.pt"

log "config: host_lr=$HOST_LR max_steps=$MAX_STEPS ckpt_every=$CKPT_EVERY neighborhood=$N_PDB structures resume_from=$RESUME_FROM_CKPT_URI"

# --- Train: resume is automatic once spa_Nx1536_last.pt exists at train.ckpt_dir (harness.py:475). ---
cd "$SPA_REPO"
python scripts/train.py \
  variant=C_n_by_1536 hardware=cloud_h100 data=cddb \
  paths.rfd3_ckpt=/workspace/weights/rfd3_latest.ckpt \
  paths.esm3_cache_dir="$CACHE_DIR" \
  paths.foundry_train_cfg_dir="$FOUNDRY_TRAIN_CFG_DIR" \
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

log "===== EXTEND TRAINING DONE (train.py exit=$STATUS) ====="

# --- Sanity checks: confirm the resume actually happened (not a silent restart-from-scratch), then
# the usual loss-trajectory + host-export checks. ---
python - "$OUT/train.log" "$OUT/ckpt" <<'PY'
import re, sys, os, glob
log_path, ckpt_dir = sys.argv[1], sys.argv[2]
lines = open(log_path).read().splitlines()
resumed = [l for l in lines if "resumed from" in l]
if not resumed:
    print("[verdict] FAIL: no 'resumed from' line found -- this run silently started from scratch")
    sys.exit(1)
print(f"[verdict] {resumed[0]}")
losses = [float(m.group(1)) for l in lines if (m := re.search(r"loss ([\d.eE+-]+) \| lr", l))]
print(f"[verdict] {len(losses)} logged loss values this invocation")
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

# --- Stage everything back to the SAME checkpoint directory (rolling last.pt updates in place, new
# numbered snapshots like spa_Nx1536_step3000.pt are added alongside the existing ones). ---
log "staging checkpoints + log -> $CKPT_OUT_URI"
gcloud storage rsync -r "$OUT/ckpt" "$CKPT_OUT_URI" || log "  (checkpoint rsync issues)"
gcloud storage cp "$OUT/train.log" "$CKPT_OUT_URI/train_extend_to_${MAX_STEPS}.log" 2>/dev/null

log "===== EXTEND DONE (train=$STATUS, verdict=$VERDICT_STATUS) -> $CKPT_OUT_URI ====="
exit $(( STATUS != 0 ? STATUS : VERDICT_STATUS ))
