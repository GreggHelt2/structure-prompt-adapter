#!/usr/bin/env bash
# In-container orchestration for Phase-2 SPA training (or the probe-cache dress rehearsal via CACHE_TAR).
# Runs inside the SPA cloud image on a Vertex Custom Job (a3-highgpu-1g, as the spa-worker SA).
# dev 04 §11 / 08 step 10 / 08 §3b workstream A. Mirrors run_cache_gen.sh and reuses its proven pieces
# (scratch auto-detect -> auto-RAID0 /workspace, LD_LIBRARY_PATH GPU fix, Secret Manager fetch, free-space
# precheck, env-only image + git-clone bootstrap, bg-rsync checkpoint).
#
# KEY DIFFERENCE FROM CACHE-GEN: cache-gen only ran ESM3. The training dataloader (CDDBPromptDataset) does
# LIVE RFD3 featurization, so this job needs FIVE runtime inputs, not one (dev 04 §11):
#   1) ESM3 prompt cache   -> load_cache.sh (tar -> NVMe), paths.esm3_cache_dir
#   2) raw CDDB PDBs       -> NGC pull + untar (featurization reads the source structures), data.pdb_dir
#   3) split manifests     -> pinned GCS artifact splits/$SPLIT_ID/ , data.splits_root
#   4) RFD3 checkpoint     -> pinned GCS artifact weights/ (frozen host AND build_train_transform's train_cfg)
#   5) foundry train-cfg   -> the in-image foundry install (the needed_repos/ default is absent here) <- the
#                             single most likely break; resolved + asserted below (B2 de-risks it).
# Training reads CACHED prompts -> NO ESM3, NO HF token needed for variant C (dev: ESM3 weights local==cloud).
#
# Env knobs (set by the Vertex job; submit_train_job.sh omits empty ones per W5.7):
#   PROJECT      spa-dev-499900
#   BUCKET       gs://genomancer-spa-cache
#   CACHE_TAR    $BUCKET/esm3_cache.tar         (probe rehearsal: $BUCKET/esm3_cache_probe1k.tar)
#   SPLIT_ID     v1_seed0_8-1-1_lenstrat        -> splits dir = $BUCKET/splits/$SPLIT_ID
#   RFD3_CKPT_URI $BUCKET/weights/rfd3_latest.ckpt
#   VARIANT      C_n_by_1536 | B_1_by_1536 | A_1_by_32
#   CONDITIONING unconditional (Run A) | island (Run B, hard native + soft SPA)
#   LENGTH_CAP   (optional) data.length_cap override; keep <= 384 (crop_size) for prompt alignment
#   REQUIRE_CACHED_PROMPT  true for the probe rehearsal (restricts the train split to cached structures)
#   MAX_STEPS    (optional) short step budget for the rehearsal; unset -> use config (full run = C harness work)
#   TRACKER      (optional) wandb (needs spa-wandb-key staged + harness logging — workstream C)
#   CHECKPOINT_SEC  bg-rsync cadence for the SPA ckpt dir (default 3600; rehearsal sets smaller)
#   CKPT_OUT_URI    $BUCKET/checkpoints/<run>   (where SPA adapter ckpts are rsynced)
#   FETCH_HF     1 to also fetch the HF token (only the variant-A/CLSS path needs it)
#   SPA_REPO     /opt/spa
set -euo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
CACHE_TAR="${CACHE_TAR:-$BUCKET/esm3_cache.tar}"
SPLIT_ID="${SPLIT_ID:-v1_seed0_8-1-1_lenstrat}"
SPLITS_URI="${SPLITS_URI:-$BUCKET/splits/$SPLIT_ID}"
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
VARIANT="${VARIANT:-C_n_by_1536}"
CONDITIONING="${CONDITIONING:-unconditional}"
REQUIRE_CACHED_PROMPT="${REQUIRE_CACHED_PROMPT:-false}"
TRACKER="${TRACKER:-}"
SPA_REPO="${SPA_REPO:-/opt/spa}"
NGC_RESOURCE="nvidia/clara/proteina-atomistica_data:release"
CHECKPOINT_SEC="${CHECKPOINT_SEC:-3600}"

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
trap 'log "FAILED at line $LINENO"' ERR

# --- 0) Disk diagnostic + scratch auto-detect (same as cache-gen: the a3 local-SSD RAID0 surfaces here) --
log "DISK DIAGNOSTIC (lsblk / df / ssd-nvme-md mounts):"
{ lsblk -o NAME,SIZE,TYPE,MOUNTPOINT 2>/dev/null; echo "-- df -h --"; df -h 2>/dev/null; \
  echo "-- ssd/nvme/md mounts --"; mount 2>/dev/null | grep -iE "ssd|nvme|md[0-9]|local"; } | sed 's/^/  /' || true
if [ -z "${SCRATCH:-}" ]; then
  for cand in /mnt/local_ssd /mnt/disks/local_ssd /mnt/disks/ssd0 /mnt/stateful_partition; do
    if mount 2>/dev/null | grep -q " on $cand " && [ -w "$cand" ]; then SCRATCH="$cand"; break; fi
  done
fi
SCRATCH="${SCRATCH:-/workspace}"
log "SCRATCH=$SCRATCH ($(df -h "$SCRATCH" 2>/dev/null | awk 'NR==2{print $4" free / "$2}'))"
DATA_DIR="${DATA_DIR:-$SCRATCH/data}"               # CDDB PDB download + untar
ESM3_CACHE_DIR="${ESM3_CACHE_DIR:-$SCRATCH/esm3_cache}"  # untarred ESM3 prompt cache (per-file .pt)
WEIGHTS_DIR="${WEIGHTS_DIR:-$SCRATCH/weights}"
SPLITS_DIR="${SPLITS_DIR:-$SCRATCH/splits}"
CKPT_DIR="${CKPT_DIR:-$SCRATCH/checkpoints}"
CKPT_OUT_URI="${CKPT_OUT_URI:-$BUCKET/checkpoints/${NAME:-spa-train}}"
mkdir -p "$DATA_DIR" "$ESM3_CACHE_DIR" "$WEIGHTS_DIR" "$SPLITS_DIR" "$CKPT_DIR"

# Fail FAST if SCRATCH can't hold the ESM3 cache (+ its tar transiently) + CDDB PDBs -> never die late.
# Full cache ~237 GB, peak ~2x during load (tar+extract) + CDDB PDBs ~51 GB. Probe cache is tiny (~0.5 GB).
NEED_GB="${MIN_FREE_GB:-540}"
FREE_GB=$(df -BG "$SCRATCH" 2>/dev/null | awk 'NR==2{gsub(/[A-Za-z]/,"",$4); print $4}')
log "free-space check: $SCRATCH has ${FREE_GB:-?} GB free, need >= ${NEED_GB} GB (override MIN_FREE_GB for the probe)"
if [ -n "${FREE_GB:-}" ] && [ "$FREE_GB" -lt "$NEED_GB" ]; then
  log "FATAL: scratch too small (${FREE_GB} < ${NEED_GB} GB) -- bump DISK_GB or set MIN_FREE_GB for the probe. Aborting before spend."
  exit 1
fi

# --- 1) Secrets (job runs as spa-worker; ADC via the metadata server). Training needs only NGC (CDDB PDBs).
log "fetching secrets from Secret Manager"
export NGC_CLI_API_KEY="$(gcloud secrets versions access latest --secret=spa-ngc-key --project="$PROJECT")"
export NGC_CLI_ORG=nvidia
export NGC_CLI_FORMAT_TYPE=ascii
if [ "${FETCH_HF:-0}" = "1" ]; then   # only the variant-A/CLSS path touches HF/ESM3
  export HF_TOKEN="$(gcloud secrets versions access latest --secret=spa-hf-token --project="$PROJECT")"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
if [ "$TRACKER" = "wandb" ]; then     # workstream C: stage spa-wandb-key first; harness logging lands in C
  if WK="$(gcloud secrets versions access latest --secret=spa-wandb-key --project="$PROJECT" 2>/dev/null)"; then
    export WANDB_API_KEY="$WK"; log "W&B key loaded"
  else
    log "WARN: TRACKER=wandb but spa-wandb-key not readable -> disabling tracker"; TRACKER=""
  fi
fi

# --- 2) GATE A — NGC must be reachable before we spend on GPU/download (mirror cache-gen) --------------
log "GATE A: NGC resource reachable ($NGC_RESOURCE)"
ngc registry resource info "$NGC_RESOURCE" --files >/dev/null && log "  NGC OK"

# --- 3) GPU sanity (+ driver path fix) — verbatim from cache-gen (the libcuda mount fix) ---------------
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/nvidia/bin:${PATH}"
ldconfig 2>/dev/null || true
( command -v nvidia-smi >/dev/null && nvidia-smi -L ) || echo "  nvidia-smi: not available"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('GPU:', torch.cuda.get_device_name(0), 'CUDA', torch.version.cuda)"

# --- 4) Install SPA (env-only image; heavy stack already present) --------------------------------------
log "pip install -e SPA (--no-deps)"
pip install -e "$SPA_REPO" --no-deps -q

# --- 5) Pinned GCS artifacts: RFD3 ckpt + split manifests (NOT rebuilt from code -> reproducible) ------
log "fetch RFD3 ckpt: $RFD3_CKPT_URI -> $WEIGHTS_DIR/rfd3_latest.ckpt"
gcloud storage cp "$RFD3_CKPT_URI" "$WEIGHTS_DIR/rfd3_latest.ckpt"
RFD3_CKPT="$WEIGHTS_DIR/rfd3_latest.ckpt"
log "fetch split manifests: $SPLITS_URI/ -> $SPLITS_DIR (split_id=$SPLIT_ID)"
gcloud storage rsync -r "$SPLITS_URI" "$SPLITS_DIR"
for s in train validate test; do
  [ -f "$SPLITS_DIR/$s/manifest.parquet" ] || { log "FATAL: missing $SPLITS_DIR/$s/manifest.parquet in split artifact"; exit 1; }
done
[ -f "$SPLITS_DIR/split_meta.json" ] && log "  split_meta: $(tr -d '\n' < "$SPLITS_DIR/split_meta.json" | cut -c1-200)"

# --- 6) Fetch + untar CDDB PDBs from NGC (live featurization reads the source structures) --------------
log "downloading CDDB from NGC -> $DATA_DIR"
t_dl=$SECONDS
ngc registry resource download-version "$NGC_RESOURCE" --dest "$DATA_DIR"
TARBALL="$(find "$DATA_DIR" -name 'atomistica_cd_dataset.tar.gz' | head -1)"
log "untarring $TARBALL"
tar -xzf "$TARBALL" -C "$DATA_DIR"
PDB_DIR="$(find "$DATA_DIR" -type d -name pdb | head -1)"
log "download+untar $((SECONDS-t_dl))s; PDB dir: $PDB_DIR ($(ls "$PDB_DIR" | wc -l) files)"

# --- 7) Load the ESM3 prompt cache (single tar -> NVMe) via the shared loader -------------------------
log "load ESM3 cache: $CACHE_TAR -> $ESM3_CACHE_DIR"
GCLOUD="$(command -v gcloud)" bash "$SPA_REPO/scripts/cloud/load_cache.sh" "$CACHE_TAR" "$ESM3_CACHE_DIR"

# --- 8) Resolve foundry's training-transform cfg dir (THE integration risk; dev 04 §11 input 5) -------
# The env-only image installed foundry via pip, so configs/paths.foundry_train_cfg_dir (a needed_repos/
# path) is absent. Find the dir holding base_transform_args.yaml + rfd3_monomer_distillation.yaml in the
# installed package; fail LOUD if absent (that means the yamls aren't shipped in the wheel -> image fix).
log "resolving foundry train-cfg dir"
if [ -z "${FOUNDRY_TRAIN_CFG_DIR:-}" ]; then
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
        hit = os.path.dirname(os.path.dirname(p)); break  # -> .../datasets/train
    if hit: break
print(hit)
PY
)"
fi
if [ -z "$FOUNDRY_TRAIN_CFG_DIR" ] || [ ! -f "$FOUNDRY_TRAIN_CFG_DIR/pdb/base_transform_args.yaml" ] \
   || [ ! -f "$FOUNDRY_TRAIN_CFG_DIR/rfd3_monomer_distillation.yaml" ]; then
  log "FATAL: could not locate foundry train-cfg dir (needs pdb/base_transform_args.yaml +"
  log "       rfd3_monomer_distillation.yaml). dev 04 §11 input 5 -- the foundry-cfg integration risk."
  log "       -> set FOUNDRY_TRAIN_CFG_DIR explicitly, or the yamls aren't in the pip wheel (image fix needed)."
  exit 1
fi
log "FOUNDRY_TRAIN_CFG_DIR=$FOUNDRY_TRAIN_CFG_DIR"

# --- 8b) RESUME: pull any prior checkpoint for THIS run DOWN from GCS so the harness (resume=auto)
# continues after a Vertex restart/preemption (workstream C). First run -> nothing to pull. The harness
# then finds $CKPT_DIR/spa_<variant>_last.pt and resumes at the next optimizer step. ------------------
log "resume: checking for a prior checkpoint at $CKPT_OUT_URI"
if gcloud storage ls "$CKPT_OUT_URI/**" >/dev/null 2>&1; then
  gcloud storage rsync -r "$CKPT_OUT_URI" "$CKPT_DIR" \
    && log "  prior checkpoint restored -> $CKPT_DIR ($(find "$CKPT_DIR" -maxdepth 1 -name '*_last.pt' | wc -l) last.pt)" \
    || log "  resume rsync issues (continuing fresh)"
else
  log "  no prior checkpoint at $CKPT_OUT_URI (fresh run)"
fi

# --- 9) Background checkpoint: rsync the SPA ckpt dir UP every CHECKPOINT_SEC (mirror cache-gen) -------
# The harness now checkpoints periodically (rolling last.pt + numbered snapshots) and resumes from
# last.pt (workstream C); this bg-rsync ships them to GCS, and step 8b pulled any prior ones DOWN, so a
# preempted/restarted job continues where it left off.
CKPT_PID=""
if [ "${CHECKPOINT_SEC:-0}" -gt 0 ]; then
  ( while sleep "$CHECKPOINT_SEC"; do
      if compgen -G "$CKPT_DIR/*" >/dev/null 2>&1; then
        gcloud storage rsync -r "$CKPT_DIR" "$CKPT_OUT_URI" >/dev/null 2>&1 \
          && log "checkpoint rsync OK -> $CKPT_OUT_URI" || log "checkpoint rsync FAILED (retry next tick)"
      fi
    done ) &
  CKPT_PID=$!
  trap '[ -n "${CKPT_PID:-}" ] && kill "$CKPT_PID" 2>/dev/null || true' EXIT
  log "background checkpoint every ${CHECKPOINT_SEC}s (pid $CKPT_PID) -> $CKPT_OUT_URI"
fi

# --- 10) Train: scripts/train.py with all hardware/path knobs as Hydra overrides ----------------------
# Empty optionals are simply not appended (so Hydra keeps the config default).
OVERRIDES=( data=cddb hardware=cloud_h100 "variant=$VARIANT"
  "paths.esm3_cache_dir=$ESM3_CACHE_DIR" "data.pdb_dir=$PDB_DIR" "data.splits_root=$SPLITS_DIR"
  "paths.rfd3_ckpt=$RFD3_CKPT" "paths.foundry_train_cfg_dir=$FOUNDRY_TRAIN_CFG_DIR"
  "train.ckpt_dir=$CKPT_DIR" "data.conditioning=$CONDITIONING"
  "data.require_cached_prompt=$REQUIRE_CACHED_PROMPT" )
[ -n "${LENGTH_CAP:-}" ]  && OVERRIDES+=( "data.length_cap=$LENGTH_CAP" )
[ -n "${MAX_STEPS:-}" ]   && OVERRIDES+=( "train.max_steps=$MAX_STEPS" )
[ -n "${NUM_WORKERS:-}" ] && OVERRIDES+=( "train.num_workers=$NUM_WORKERS" )
[ -n "$TRACKER" ]         && OVERRIDES+=( "train.tracker=$TRACKER" )
[ -n "${DIFFUSION_BATCH_SIZE:-}" ] && OVERRIDES+=( "data.diffusion_batch_size=$DIFFUSION_BATCH_SIZE" )  # D (recipe=8)
[ -n "${MATMUL_PRECISION:-}" ]     && OVERRIDES+=( "train.matmul_precision=$MATMUL_PRECISION" )
# Arbitrary extra Hydra overrides (space-separated), for knobs without a dedicated env var — e.g. the
# extend-run's `train.sample_with_replacement=false` (dev 28 §2). Word-split intentionally.
[ -n "${EXTRA_OVERRIDES:-}" ] && OVERRIDES+=( ${EXTRA_OVERRIDES} )
# Distinct, resumable W&B run per job: name + stable id from $NAME so a Vertex restart REATTACHES the
# same run (Run A vs Run B differ by $NAME), matching the ckpt resume above.
if [ "$TRACKER" = "wandb" ] && [ -n "${NAME:-}" ]; then
  OVERRIDES+=( "run_name=$NAME" "train.wandb_id=$NAME" )
fi

# RUN_MODE=profile runs the step profiler (scripts/profile_step.py) instead of training — same input
# gathering above, so we measure steady-state step time + breakdown on the real H100 (dev 08 §6).
ENTRY="scripts/train.py"; LABEL="SPA training"
[ "${RUN_MODE:-train}" = "profile" ] && { ENTRY="scripts/profile_step.py"; LABEL="SPA step PROFILE"; }
log "$LABEL: ${OVERRIDES[*]}"
t_train=$SECONDS
python "$SPA_REPO/$ENTRY" "${OVERRIDES[@]}"
log "wall: $((SECONDS-t_train))s"

# --- 11) Final checkpoint rsync + summary -------------------------------------------------------------
if [ -n "$CKPT_PID" ]; then kill "$CKPT_PID" 2>/dev/null || true; wait "$CKPT_PID" 2>/dev/null || true; fi
log "final checkpoint rsync ($(find "$CKPT_DIR" -type f | wc -l) files) -> $CKPT_OUT_URI"
gcloud storage rsync -r "$CKPT_DIR" "$CKPT_OUT_URI" || log "  (final rsync issues)"
log "===== SUMMARY ====="
log "  variant=$VARIANT  conditioning=$CONDITIONING  cache=$CACHE_TAR  split_id=$SPLIT_ID"
log "  SPA checkpoints -> $CKPT_OUT_URI"
log "DONE"
