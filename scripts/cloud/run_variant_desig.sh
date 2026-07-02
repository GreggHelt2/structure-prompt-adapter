#!/usr/bin/env bash
# In-container VARIANT SOFT-designability: for each SPA variant (C=N×1536 / B=1×1536 / A=1×32-CLSS) run the
# flywheel BASELINE vs SPA (SOFT-only — NO hard motif, so no pooled-variant non-overlap question, #2 parked)
# on the curated-15, with OF3-triton refold → designability (scRMSD / d_succ) + adherence (TM vs prompt, ~free
# via prompt_struct). Completes the "≥2 variants + CLSS-encoder" story on the designability axis (today's
# variant comparison is adherence-only). Reuses the B1-full prep .pt embeddings — one ESM3 [N,1536] cache
# serves all 3 variants (front-ends pool / CLSS internally; projectors.py). Per (variant,prompt); results
# staged incrementally. NB: NOT `set -e` — one flywheel failing must not abort the loops.
set -uo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
PREP_URI="${PREP_URI:-$BUCKET/eval/b1_full/prep}"            # reuse B1-full's ESM3 prompt .pt + source .pdb
RESULTS_URI="${RESULTS_URI:-$BUCKET/eval/variant_desig}"     # per (variant,prompt) results out
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
OF3_CKPT_URI="${OF3_CKPT_URI:-$BUCKET/weights/of3-p2-155k.pt}"
SPA_REPO="${SPA_REPO:-/opt/spa}"
MPNN_REPO="${MPNN_REPO:-/opt/ProteinMPNN}"
# variant:ckpt-relpath entries (space-separated). C is primary; B/A are the pooled variants.
VARIANTS="${VARIANTS:-C_n_by_1536:spa-Nx1536-uncond/spa_C_final.pt B_1_by_1536:spa-1x1536-uncond/spa_B_final.pt A_1_by_32:spa-1x32-uncond/spa_A_final.pt}"
K="${K:-4}"; NSEQ="${NSEQ:-4}"; LAM="${LAM:-1}"
BAND="${BAND:-le256}"   # curated-15 = the b1_full manifest's le256 band
PREP=/workspace/prep; OUT=/workspace/vdesig

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/nvidia/bin:${PATH}"; ldconfig 2>/dev/null || true
python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
conda run -n spa-verify-of3 python -c "import triton; print('OF3 env: triton', triton.__version__)"

[ -d "$SPA_REPO/.git" ] || git clone --depth 1 --branch "${REPO_REF:-main}" "${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}" "$SPA_REPO"
pip install -e "$SPA_REPO" --no-deps -q
[ -d "$MPNN_REPO" ] || git clone --depth 1 https://github.com/dauparas/ProteinMPNN "$MPNN_REPO"

mkdir -p /workspace/weights "$PREP" "$OUT"
gcloud storage cp "$RFD3_CKPT_URI" /workspace/weights/rfd3_latest.ckpt
gcloud storage cp "$OF3_CKPT_URI"  /workspace/weights/of3-p2-155k.pt
gcloud storage cp "$PREP_URI/*" "$PREP/"
MAN="$PREP/b1_full_resolved.json"
[ -f "$MAN" ] || { log "FATAL: $MAN not found"; exit 1; }

# prompts of the chosen band: id<TAB>len
export BAND
python -c "import json,os
b=os.environ['BAND']
for p in json.load(open('$MAN'))['prompts']:
    if p['band']==b: print(p['id']+chr(9)+str(p['len']))" > "$OUT/prompts.tsv"
NP=$(wc -l < "$OUT/prompts.tsv")
NV=$(echo "$VARIANTS" | wc -w)
log "variant SOFT-designability: ${NV} variants × ${NP} prompts (band=$BAND), K=$K N=$NSEQ λ=$LAM"

for entry in $VARIANTS; do
  vname="${entry%%:*}"; ckpt_rel="${entry#*:}"
  gcloud storage cp "$BUCKET/checkpoints/$ckpt_rel" "/workspace/weights/spa_${vname}.pt"
  log "=== variant $vname (ckpt $ckpt_rel) ==="
  n=0; ok=0
  while IFS=$'\t' read -r id len; do
    n=$((n+1)); po="$OUT/$vname/$id"
    log "  [$vname $n/$NP] $id (len $len)"
    python "$SPA_REPO/scripts/eval/run_flywheel.py" \
      variant="$vname" hardware=cloud_h100 \
      'eval.conditions=[baseline,spa]' "eval.lambda_scale=[$LAM]" \
      eval.num_designs="$K" eval.proteinmpnn.num_seqs="$NSEQ" \
      eval.length="$len" \
      eval.ckpt="/workspace/weights/spa_${vname}.pt" \
      eval.prompt_cache="$PREP/$id.pt" \
      +eval.flywheel.prompt_struct="$PREP/$id.pdb" \
      paths.rfd3_ckpt=/workspace/weights/rfd3_latest.ckpt \
      paths.proteinmpnn_repo="$MPNN_REPO" \
      +eval.flywheel.refolder._target_=spa.eval.openfold3.OF3Refolder \
      +eval.flywheel.refolder.ckpt_path=/workspace/weights/of3-p2-155k.pt \
      +eval.flywheel.refolder.runner_yaml="$SPA_REPO/configs/of3/of3_triton.yml" \
      +eval.flywheel.refolder.out_dir="$po" \
      eval.out_dir="$po" </dev/null \
      && { ok=$((ok+1)); gcloud storage cp "$po/flywheel_results.json" "$RESULTS_URI/$vname/$id.json" 2>/dev/null; \
           # Keep the actual design STRUCTURES by DEFAULT (generate.py already wrote
           # {id}_{cond}_lambda{λ}_{idx}.pdb to out_dir — the compute happened, staging is ~free, and NOT
           # keeping them was the B4 bug). LEAN_RESULTS=1 opts out (metrics-only) for giant sweeps where
           # GCS object count matters; it does NOT change the metrics JSON either way.
           [ "${LEAN_RESULTS:-0}" = "1" ] || gcloud storage cp "$po"/*.pdb "$RESULTS_URI/$vname/$id/" 2>/dev/null; \
           log "  [$vname $n] $id OK -> staged"; } \
      || log "  [$vname $n] $id FAILED (continuing)"
  done < "$OUT/prompts.tsv"
  log "=== variant $vname done: $ok/$NP staged ==="
done
log "===== VARIANT DESIGNABILITY DONE -> $RESULTS_URI ====="
