#!/usr/bin/env bash
# In-container Option B SMOKE TEST (dev plan/69 SS4.8 item 1): validates the host-unfreeze training
# path (harness.py's build_optimizer, commit 620e750) converges without diverging, on the cheap
# `toy` synthetic-overfit path -- no CDDB data, no ESM3 cache tar, no split manifests needed, mirrors
# run_smoke.sh's minimal bootstrap. This is a PLUMBING check, not the real Option B experiment: it
# answers "does fine-tuning the host train stably at all", not "which target fold(s)" (dev plan/69
# section 4.8 item 2, still open, Gregg's call, deliberately decoupled from this).
set -uo pipefail   # NOT -e: a training divergence must show up in the summary, not abort silently

PROJECT="${PROJECT:-spa-dev-499900}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
SPA_REPO="${SPA_REPO:-/opt/spa}"
HOST_LR="${HOST_LR:-1.0e-5}"          # dev plan/69 SS4.2's conservative starting guess (~10x below train.lr)
MAX_STEPS="${MAX_STEPS:-200}"
CKPT_EVERY="${CKPT_EVERY:-50}"        # exercise the host-checkpoint save/load path a few times in one run
OUT="${OUT:-/workspace/finetune_smoke_out}"

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

# --- GPU driver-path fix (verbatim from run_smoke.sh) ---
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/nvidia/bin:${PATH}"
ldconfig 2>/dev/null || true
( command -v nvidia-smi >/dev/null && nvidia-smi -L ) || echo "  nvidia-smi: n/a"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('GPU', torch.cuda.get_device_name(0), 'CUDA', torch.version.cuda)"

# --- SPA repo (the Vertex BOOT command already clones $SPA_REPO; clone only if absent) ---
[ -d "$SPA_REPO/.git" ] || git clone --depth 1 --branch "${REPO_REF:-main}" "${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}" "$SPA_REPO"
pip install -e "$SPA_REPO" --no-deps -q

# --- Fetch the RFD3 checkpoint. The toy path (configs/train.yaml's data=toy default) needs nothing
# else: no ESM3 cache, no CDDB PDBs, no split manifests, unlike run_train.sh's full production path. ---
mkdir -p /workspace/weights "$OUT"
gcloud storage cp "$RFD3_CKPT_URI" /workspace/weights/rfd3_latest.ckpt

log "config: host_lr=$HOST_LR max_steps=$MAX_STEPS ckpt_every=$CKPT_EVERY"

cd "$SPA_REPO"
python scripts/train.py \
  variant=C_n_by_1536 hardware=cloud_h100 \
  paths.rfd3_ckpt=/workspace/weights/rfd3_latest.ckpt \
  train.finetune_host_lr="$HOST_LR" \
  train.max_steps="$MAX_STEPS" \
  train.ckpt_every_steps="$CKPT_EVERY" \
  train.log_every_steps=10 \
  train.ckpt_dir="$OUT/ckpt" \
  hydra.run.dir="$OUT/hydra" \
  2>&1 | tee "$OUT/train.log"
STATUS=${PIPESTATUS[0]}

log "===== SMOKE TEST TRAINING DONE (train.py exit=$STATUS) ====="

# --- Sanity checks on the log + artifacts, so the summary is a verdict, not a raw log dump: no NaN,
# no order-of-magnitude blowup, and the new host-checkpoint plumbing actually produced its artifacts. ---
python - "$OUT/train.log" "$OUT/ckpt" <<'PY'
import re, sys, os, glob
log_path, ckpt_dir = sys.argv[1], sys.argv[2]
lines = open(log_path).read().splitlines()
losses = [float(m.group(1)) for l in lines if (m := re.search(r"loss ([\d.eE+-]+) \| lr", l))]
print(f"[verdict] {len(losses)} logged loss values")
if not losses:
    print("[verdict] FAIL: no loss values parsed from the log at all")
    sys.exit(1)
if any(l != l for l in losses):  # NaN check (NaN != NaN is the standard idiom)
    print("[verdict] FAIL: NaN loss encountered")
    sys.exit(1)
if losses[-1] > losses[0] * 10:
    print(f"[verdict] FAIL: loss exploded ({losses[0]:.4f} -> {losses[-1]:.4f})")
    sys.exit(1)
print(f"[verdict] loss trajectory OK: {losses[0]:.4f} -> {losses[-1]:.4f} (min {min(losses):.4f})")

host_exports = glob.glob(os.path.join(ckpt_dir, "rfd3_*_finetuned.pt"))
last_ckpt = os.path.join(ckpt_dir, "spa_C_last.pt")
print(f"[verdict] host export present: {bool(host_exports)} {host_exports}")
if not os.path.exists(last_ckpt):
    print(f"[verdict] FAIL: no last.pt at {last_ckpt}")
    sys.exit(1)
import torch
ck = torch.load(last_ckpt, map_location="cpu", weights_only=False)
if "host" not in ck:
    print("[verdict] FAIL: last.pt has no 'host' key despite finetune_host_lr being set")
    sys.exit(1)
print(f"[verdict] last.pt 'host' key present, {len(ck['host'])} tensors")
PY
VERDICT_STATUS=$?

gcloud storage cp "$OUT/train.log" "$BUCKET/eval/finetune_smoke_runs/train_$(date -u +%Y%m%d-%H%M%S).log" 2>/dev/null \
  && log "log staged to gs://genomancer-spa-cache/eval/finetune_smoke_runs/"

log "===== FINETUNE SMOKE TEST DONE (train=$STATUS, verdict=$VERDICT_STATUS) ====="
exit $(( STATUS != 0 ? STATUS : VERDICT_STATUS ))
