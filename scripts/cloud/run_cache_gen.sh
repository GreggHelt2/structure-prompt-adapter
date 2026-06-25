#!/usr/bin/env bash
# In-container orchestration for Phase-1 ESM3 cache-gen (or the ~1k benchmark via LIMIT).
# Runs inside the SPA cloud image on a Vertex Custom Job (a3-highgpu-1g, as the spa-worker SA).
# dev 04 §10 / 08 step 9. Gate A (ESM3 + NGC canaries) runs BEFORE any GPU/download work, so a bad
# credential fails fast and cheap.
#
# Env knobs (set by the Vertex job):
#   PROJECT    spa-dev-499900
#   BUCKET     gs://genomancer-spa-cache      cache dest = $BUCKET/$GCS_PREFIX
#   GCS_PREFIX esm3_cache
#   LIMIT      ""=all, or e.g. 1000 for the benchmark
#   DATA_DIR   /workspace/data    (CDDB tarball download + untar)
#   CACHE_DIR  /workspace/cache   (local .pt output before rsync)
#   SPA_REPO   /opt/spa           (the repo cloned by the job bootstrap)
set -euo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
GCS_PREFIX="${GCS_PREFIX:-esm3_cache}"
LIMIT="${LIMIT:-}"
SPA_REPO="${SPA_REPO:-/opt/spa}"
NGC_RESOURCE="nvidia/clara/proteina-atomistica_data:release"

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
trap 'log "FAILED at line $LINENO"' ERR

# --- 0) Disk diagnostic + scratch auto-detect (resolves W5.3: does the a3 local-SSD RAID0 reach us?) -
log "DISK DIAGNOSTIC (lsblk / df / ssd-nvme-md mounts):"
{ lsblk -o NAME,SIZE,TYPE,MOUNTPOINT 2>/dev/null; echo "-- df -h --"; df -h 2>/dev/null; \
  echo "-- ssd/nvme/md mounts --"; mount 2>/dev/null | grep -iE "ssd|nvme|md[0-9]|local"; } | sed 's/^/  /' || true
# Prefer a large local-SSD scratch if Vertex's DLVM auto-RAID0 exposed one to our container; else /workspace.
if [ -z "${SCRATCH:-}" ]; then
  for cand in /mnt/local_ssd /mnt/disks/local_ssd /mnt/disks/ssd0 /mnt/stateful_partition; do
    if mount 2>/dev/null | grep -q " on $cand " && [ -w "$cand" ]; then SCRATCH="$cand"; break; fi
  done
fi
SCRATCH="${SCRATCH:-/workspace}"
log "SCRATCH=$SCRATCH ($(df -h "$SCRATCH" 2>/dev/null | awk 'NR==2{print $4" free / "$2}'))"
DATA_DIR="${DATA_DIR:-$SCRATCH/data}"
CACHE_DIR="${CACHE_DIR:-$SCRATCH/cache}"

# Fail FAST if SCRATCH can't hold the cache + its tar (peak ~2x cache) -> never die AFTER hours of compute.
N_EST="${LIMIT:-455473}"
NEED_GB="${MIN_FREE_GB:-$(( N_EST * 11 / 10000 + 40 ))}"   # ~0.55 MB/struct cache, x2 for the tar, +40 slack
FREE_GB=$(df -BG "$SCRATCH" 2>/dev/null | awk 'NR==2{gsub(/[A-Za-z]/,"",$4); print $4}')
log "free-space check: $SCRATCH has ${FREE_GB:-?} GB free, need >= ${NEED_GB} GB"
if [ -n "${FREE_GB:-}" ] && [ "$FREE_GB" -lt "$NEED_GB" ]; then
  log "FATAL: scratch too small (${FREE_GB} < ${NEED_GB} GB) for cache + tar -- bump DISK_GB or use the local NVMe. Aborting before any spend."
  exit 1
fi

# --- 1) Secrets (job runs as spa-worker; ADC via the metadata server) -----------------------------
log "fetching secrets from Secret Manager"
export HF_TOKEN="$(gcloud secrets versions access latest --secret=spa-hf-token --project="$PROJECT")"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export NGC_CLI_API_KEY="$(gcloud secrets versions access latest --secret=spa-ngc-key --project="$PROJECT")"
export NGC_CLI_ORG=nvidia
export NGC_CLI_FORMAT_TYPE=ascii

# --- 2) GATE A — credentials must work before we spend on GPU/download ----------------------------
log "GATE A: ESM3 gated-weight canary (biohub/esm3-sm-open-v1/config.json)"
python -c "from huggingface_hub import hf_hub_download; print('  ESM3 OK ->', hf_hub_download('biohub/esm3-sm-open-v1','config.json'))"
log "GATE A: NGC resource reachable ($NGC_RESOURCE)"
ngc registry resource info "$NGC_RESOURCE" --files >/dev/null && log "  NGC OK"

# --- 3) GPU sanity (+ driver path fix + diagnostics) ----------------------------------------------
# Vertex's nvidia-container runtime MOUNTS the host driver (libcuda.so.1) into /usr/local/nvidia, but the
# python:3.12-slim base doesn't put that on the library path -> torch can't find libcuda -> "no CUDA"
# (this failed the 0.2.0/0.3.0 runs even with NVIDIA_DRIVER_CAPABILITIES=compute). Add the path:
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/nvidia/bin:${PATH}"
ldconfig 2>/dev/null || true
log "GPU diag: NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>} CAPS=${NVIDIA_DRIVER_CAPABILITIES:-<unset>}"
( ls /usr/local/nvidia/lib64 2>/dev/null | grep -iE "libcuda|libnvidia-ml" | head ) || echo "  (no driver libs in /usr/local/nvidia/lib64)"
( ls -l /dev/nvidia* 2>&1 | head -6 ) || true
( command -v nvidia-smi >/dev/null && nvidia-smi -L ) || echo "  nvidia-smi: not available"
python -c "import torch; print('  torch',torch.__version__,'built-cuda',torch.version.cuda,'avail',torch.cuda.is_available(),'count',torch.cuda.device_count())" || true
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('GPU:', torch.cuda.get_device_name(0), 'CUDA', torch.version.cuda)"

# --- 4) Fetch + untar CDDB from NGC ---------------------------------------------------------------
mkdir -p "$DATA_DIR" "$CACHE_DIR"
log "downloading CDDB from NGC -> $DATA_DIR"
t_dl=$SECONDS
ngc registry resource download-version "$NGC_RESOURCE" --dest "$DATA_DIR"
TARBALL="$(find "$DATA_DIR" -name 'atomistica_cd_dataset.tar.gz' | head -1)"
log "untarring $TARBALL"
tar -xzf "$TARBALL" -C "$DATA_DIR"
PDB_DIR="$(find "$DATA_DIR" -type d -name pdb | head -1)"
log "download+untar ${SECONDS}-${t_dl}=$((SECONDS-t_dl))s; PDB dir: $PDB_DIR ($(ls "$PDB_DIR" | wc -l) files)"

# --- 5) Install SPA (env-only image; heavy stack already present) and run cache-gen ----------------
log "pip install -e SPA (--no-deps)"
pip install -e "$SPA_REPO" --no-deps -q
LIMIT_ARG=""; [ -n "$LIMIT" ] && LIMIT_ARG="limit=$LIMIT"
log "ESM3 cache-gen (limit=${LIMIT:-all}) -> $CACHE_DIR"
t_gen=$SECONDS
python "$SPA_REPO/scripts/gen_esm3_cache.py" \
    data=cddb hardware=cloud_h100 \
    data.pdb_dir="$PDB_DIR" out_dir="$CACHE_DIR" data.length_cap="${LENGTH_CAP:-512}" $LIMIT_ARG
gen_secs=$((SECONDS-t_gen))
n_pt=$(find "$CACHE_DIR" -name '*.pt' | wc -l)
cache_mb=$(du -sm "$CACHE_DIR" | cut -f1)

# --- 6) Pack into ONE tar (to disk) and upload to GCS ---------------------------------------------
# ~358k tiny .pt objects make the later GCS->NVMe pull request-rate-bound (per-object GET + listing) on
# EVERY training run; one tar makes it bandwidth-bound. Tar to disk FIRST (not stream) so the upload is a
# *resumable, parallel* gcloud cp of a single file — robust after the ~11h run (a blip resumes vs restarts;
# the tar persists for re-upload). Costs ~1x extra disk transiently -> size the boot disk ~500GB (cheap, ~$1).
TAR_URI="${TAR_URI:-$BUCKET/${GCS_PREFIX}.tar}"
TAR_LOCAL="${TAR_LOCAL:-$(dirname "$CACHE_DIR")/esm3_cache.tar}"
# Free the CDDB PDBs (~62 GB) before taring -> lowers peak disk to ~2x cache (the precheck assumes this).
log "freeing CDDB data ($DATA_DIR) before tar"
rm -rf "${DATA_DIR:?}"/* 2>/dev/null || true
log "tar $n_pt .pt files -> $TAR_LOCAL"
tar -cf "$TAR_LOCAL" -C "$CACHE_DIR" .
# Upload with retries: a single transient blip here, after hours of compute, must not lose the run.
for attempt in 1 2 3 4 5; do
  log "upload $TAR_LOCAL ($(du -h "$TAR_LOCAL" 2>/dev/null | cut -f1)) -> $TAR_URI (attempt $attempt/5)"
  if gcloud storage cp "$TAR_LOCAL" "$TAR_URI"; then log "uploaded single tar: $TAR_URI"; break; fi
  [ "$attempt" = 5 ] && { log "FATAL: upload failed after 5 attempts (tar kept at $TAR_LOCAL)"; exit 1; }
  log "upload failed; retry in 30s"; sleep 30
done

# --- 7) Benchmark summary (feeds the Gate-B extrapolation) ----------------------------------------
log "===== SUMMARY ====="
log "  structures cached : $n_pt"
log "  cache size        : ${cache_mb} MB"
log "  cache-gen wall    : ${gen_secs}s"
python -c "n=$n_pt; s=$gen_secs; print(f'  throughput        : {n/max(s,1):.1f} prot/s  ->  455473 would take ~{455473/max(n/max(s,1),1e-9)/3600:.2f} h')"
python -c "n=$n_pt; mb=$cache_mb; print(f'  full-cache est    : ~{mb/max(n,1)*455473/1024:.0f} GB (scaling MB/struct x 455473)')"
log "DONE"
