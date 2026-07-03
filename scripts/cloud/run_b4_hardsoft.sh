#!/usr/bin/env bash
# In-container B4 HARD⊕SOFT: for the named folds, run the flywheel BASELINE vs SPA with a native
# coordinate-pinned motif (auto-carved SSE) AND the soft SPA fold prompt in the same design — the
# hard⊕soft headline, on recognizable external folds. Per-fold motif contig comes from the manifest
# (motif_contig field); the motif is masked out of the SPA prompt (non-overlap) by the eval.motif path,
# so N×1536 (variant C) only. Reuses the b4 prep .pt embeddings (renumbered A1..N sources). Results
# staged incrementally. NB: NOT `set -e` — one flywheel failing must not abort the loop.
set -uo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
PREP_URI="${PREP_URI:-$BUCKET/eval/b4_hardsoft/prep}"
RESULTS_URI="${RESULTS_URI:-$BUCKET/eval/b4_hardsoft/results}"
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
OF3_CKPT_URI="${OF3_CKPT_URI:-$BUCKET/weights/of3-p2-155k.pt}"
SPA_REPO="${SPA_REPO:-/opt/spa}"
MPNN_REPO="${MPNN_REPO:-/opt/ProteinMPNN}"
MAN_NAME="${MAN_NAME:-b4_hardsoft_resolved.json}"
VARIANTS="${VARIANTS:-C_n_by_1536:spa-Nx1536-uncond/spa_C_final.pt}"   # N×1536 only (hard⊕soft mask)
K="${K:-8}"; NSEQ="${NSEQ:-8}"; LAM="${LAM:-0.5,1,2}"
BAND="${BAND:-le384}"
PREP=/workspace/prep; OUT=/workspace/b4hs

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
MAN="$PREP/$MAN_NAME"
[ -f "$MAN" ] || { log "FATAL: $MAN not found"; exit 1; }

# prompts of the chosen band: id<TAB>len<TAB>motif_contig
export BAND
python -c "import json,os
b=os.environ['BAND']
for p in json.load(open('$MAN'))['prompts']:
    if p['band']==b: print(p['id']+chr(9)+str(p['len'])+chr(9)+p.get('motif_contig',''))" > "$OUT/prompts.tsv"
NP=$(wc -l < "$OUT/prompts.tsv")
log "B4 HARD⊕SOFT: ${NP} folds (band=$BAND), variant=C, K=$K N=$NSEQ λ=$LAM"

for entry in $VARIANTS; do
  vname="${entry%%:*}"; ckpt_rel="${entry#*:}"
  gcloud storage cp "$BUCKET/checkpoints/$ckpt_rel" "/workspace/weights/spa_${vname}.pt"
  log "=== variant $vname (ckpt $ckpt_rel) ==="
  n=0; ok=0
  while IFS=$'\t' read -r id len contig; do
    n=$((n+1)); po="$OUT/$vname/$id"
    [ -z "$contig" ] && { log "  [$vname $n] $id SKIP (no motif_contig)"; continue; }
    log "  [$vname $n/$NP] $id (len $len) motif=$contig"
    python "$SPA_REPO/scripts/eval/run_flywheel.py" \
      variant="$vname" hardware=cloud_h100 \
      'eval.conditions=[baseline,spa]' "eval.lambda_scale=[$LAM]" \
      eval.num_designs="$K" eval.proteinmpnn.num_seqs="$NSEQ" \
      eval.length="$len" \
      eval.ckpt="/workspace/weights/spa_${vname}.pt" \
      eval.prompt_cache="$PREP/$id.pt" \
      +eval.flywheel.prompt_struct="$PREP/$id.pdb" \
      +eval.motif.source_pdb="$PREP/$id.pdb" \
      +eval.motif.contig="'$contig'" \
      paths.rfd3_ckpt=/workspace/weights/rfd3_latest.ckpt \
      paths.proteinmpnn_repo="$MPNN_REPO" \
      +eval.flywheel.refolder._target_=spa.eval.openfold3.OF3Refolder \
      +eval.flywheel.refolder.ckpt_path=/workspace/weights/of3-p2-155k.pt \
      +eval.flywheel.refolder.runner_yaml="$SPA_REPO/configs/of3/of3_triton.yml" \
      +eval.flywheel.refolder.out_dir="$po" \
      eval.out_dir="$po" </dev/null \
      && { ok=$((ok+1)); gcloud storage cp "$po/flywheel_results.json" "$RESULTS_URI/$vname/$id.json" 2>/dev/null; \
           [ "${LEAN_RESULTS:-0}" = "1" ] || gcloud storage cp "$po"/*.pdb "$RESULTS_URI/$vname/$id/" 2>/dev/null; \
           log "  [$vname $n] $id OK -> staged"; } \
      || log "  [$vname $n] $id FAILED (continuing)"
  done < "$OUT/prompts.tsv"
  log "=== variant $vname done: $ok/$NP staged ==="
done
log "===== B4 HARD⊕SOFT DONE -> $RESULTS_URI ====="
