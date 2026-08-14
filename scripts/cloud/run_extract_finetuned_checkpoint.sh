#!/usr/bin/env bash
# In-container Option B checkpoint reconstruction (dev plan/69 SS4.13, generalized 2026-08-14 from the
# A0A7S3EB45 run). Turns a SPA full-state fine-tune checkpoint (spa_<X>_last.pt / spa_<X>_stepN.pt,
# saved by harness.py's save_checkpoint with net= set) into a plain RFD3-loadable checkpoint, entirely
# GCS-to-GCS. No GPU is used at all (pure CPU torch.load/torch.save dict surgery); this reuses the
# project's proven H100 image only because that is the fastest already-working path, exactly like
# run_check_esm3_cache.sh's own reasoning -- swap to a cheaper CPU-only machine type if this becomes a
# frequent operation.
#
# WHY THIS EXISTS: the first real run (2026-08-14, A0A7S3EB45) did this exact reconstruction on a
# bandwidth-constrained local Mac -- 2.1 GB SPA checkpoint down, 2.5 GB base RFD3 checkpoint down, 2.5
# GB result up, all at local link speed (~15-35 MiB/s measured) instead of GCS-internal speed (~800
# MiB/s measured by run_check_esm3_cache.sh's own tar pull). Every artifact this needs is already in
# GCS; there was never a reason to round-trip through a local machine. Do the reconstruction here for
# every future fine-tune run instead.
#
# THE TWO TRANSFORMS, mechanical and total (dev plan/69 SS4.13):
#   1. Extract the "host" key from the SPA full-state checkpoint (net.state_dict(), still SPA-wrapped:
#      keys are a mix of ...attention_pair_bias.orig.<param> [the real, fine-tuned RFD3 weights],
#      ...attention_pair_bias.spa.<param> [SPA's own cross-attention, irrelevant to a vanilla-RFD3
#      checkpoint], and everything else unchanged).
#   2. Strip the SPA wrapping (drop .spa. keys, rename .orig. away) and splice the result into a COPY
#      of the base RFD3 checkpoint's "shadow.*" keys (not "model.*" -- RFdiffusion3 inference always
#      dispatches to the EMA shadow net at eval() time, needed_repos/foundry's
#      foundry/training/EMA.py:62-67, confirmed by direct read 2026-08-14), leaving everything else
#      (optimizer, scheduler_cfg, train_cfg, global_step, current_epoch, the raw "model.*" weights)
#      untouched. The result is structurally identical in shape to a checkpoint the loader already
#      handles correctly (foundry/trainers/fabric.py:781-785's ckpt["model"] with "model."/"shadow."
#      prefixed keys) -- a bare bare state dict pointed directly at paths.rfd3_ckpt would NOT load.
set -uo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
SPA_CKPT_URI="${SPA_CKPT_URI:?set SPA_CKPT_URI to the SPA full-state fine-tune checkpoint (spa_<X>_last.pt or spa_<X>_stepN.pt)}"
BASE_RFD3_CKPT_URI="${BASE_RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
OUT_CKPT_URI="${OUT_CKPT_URI:?set OUT_CKPT_URI to where the reconstructed checkpoint should land}"
OUT="${OUT:-/workspace/extract_finetuned_checkpoint}"

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
trap 'log "CHECKPOINT EXTRACTION FAILED at line $LINENO"' ERR

mkdir -p "$OUT"
log "fetching SPA checkpoint: $SPA_CKPT_URI"
gcloud storage cp "$SPA_CKPT_URI" "$OUT/spa_finetuned.pt"
log "fetching base RFD3 checkpoint: $BASE_RFD3_CKPT_URI"
gcloud storage cp "$BASE_RFD3_CKPT_URI" "$OUT/rfd3_base.ckpt"

python - "$OUT/spa_finetuned.pt" "$OUT/rfd3_base.ckpt" "$OUT/rfd3_finetuned_full.ckpt" <<'PY'
import sys
import torch

spa_path, base_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

print(f"[extract] loading SPA checkpoint: {spa_path}")
spa_ckpt = torch.load(spa_path, map_location="cpu", weights_only=False)
if "host" not in spa_ckpt:
    raise SystemExit(
        "FATAL: no 'host' key in the SPA checkpoint -- this run did not fine-tune the host "
        "(train.finetune_host_lr was unset), nothing to extract."
    )
raw = spa_ckpt["host"]
print(f"[extract] host state dict: {len(raw)} tensors "
      f"(step={spa_ckpt.get('step')}, variant={spa_ckpt.get('variant')})")

# Step 1: strip the SPA wrapping (SPAWrappedAttention(orig=<RFD3>, spa=<cross-attn>)).
hostonly = {}
n_dropped_spa = 0
n_renamed = 0
for k, v in raw.items():
    if ".attention_pair_bias.spa." in k:
        n_dropped_spa += 1
        continue
    if ".attention_pair_bias.orig." in k:
        k = k.replace(".attention_pair_bias.orig.", ".attention_pair_bias.")
        n_renamed += 1
    hostonly[k] = v
print(f"[extract] unwrapped -> {len(hostonly)} tensors "
      f"(dropped {n_dropped_spa} .spa. keys, renamed {n_renamed} .orig. keys)")

# Step 2: splice into the base checkpoint's shadow.* keys (the EMA net actually used at inference).
print(f"[extract] loading base RFD3 checkpoint: {base_path}")
base = torch.load(base_path, map_location="cpu", weights_only=False)
model_dict = base["model"]

n_replaced = n_shape_mismatch = n_missing_in_base = 0
for k, v in hostonly.items():
    bk = f"shadow.{k}"
    if bk not in model_dict:
        n_missing_in_base += 1
        continue
    if model_dict[bk].shape != v.shape:
        n_shape_mismatch += 1
        continue
    model_dict[bk] = v
    n_replaced += 1

print(f"[extract] replaced: {n_replaced}  shape_mismatch: {n_shape_mismatch}  "
      f"missing_in_base: {n_missing_in_base}  hostonly_total: {len(hostonly)}")

if n_replaced == 0:
    raise SystemExit("FATAL: replaced 0 tensors -- the key mapping did not match anything in the base checkpoint")
if n_shape_mismatch or n_missing_in_base:
    print(f"[extract] WARNING: {n_shape_mismatch + n_missing_in_base} tensor(s) did not splice cleanly, "
          "check the SPA variant / RFD3 checkpoint versions match")

torch.save(base, out_path)
print(f"[extract] wrote -> {out_path}")
PY

log "staging result -> $OUT_CKPT_URI"
gcloud storage cp "$OUT/rfd3_finetuned_full.ckpt" "$OUT_CKPT_URI"
log "===== CHECKPOINT EXTRACTION DONE -> $OUT_CKPT_URI ====="
