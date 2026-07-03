#!/usr/bin/env bash
# In-container SCAFFOLDING big-run (dev 17 §7 / 16 §9.5): does the multigranularity ("editing") SPA
# scaffold a sub-region-MASKED prompt BETTER than the base full-prompt SPA? For each held-out prompt ×
# granularity (domain / segment_small), run the flywheel BASELINE vs SPA with SPA conditioned on S only
# (eval.subregion.keep_range) + OF3-triton refold -> sub-region motif-RMSD + designability (scRMSD) +
# whole-fold TM, for BOTH the multigran ckpt and the base-uncond ckpt on the SAME masks. Mirrors
# run_variant_desig.sh (precomputed ESM3 .pt prompts from prep -> no ESM3 on the cloud image). NOT set -e:
# one flywheel failing must not abort the loops.
set -uo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
PREP_URI="${PREP_URI:-$BUCKET/eval/scaffold/prep}"           # <id>.pt + <id>.pdb + scaffold_resolved.json (prep_scaffold.py)
RESULTS_URI="${RESULTS_URI:-$BUCKET/eval/scaffold/results}"
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
OF3_CKPT_URI="${OF3_CKPT_URI:-$BUCKET/weights/of3-p2-155k.pt}"
MG_CKPT_URI="${MG_CKPT_URI:-$BUCKET/checkpoints/spa-Nx1536-multigran/spa_C_final.pt}"   # the "editing" ckpt
BASE_CKPT_URI="${BASE_CKPT_URI:-$BUCKET/checkpoints/spa-Nx1536-uncond/spa_C_final.pt}"  # base full-prompt SPA
SPA_REPO="${SPA_REPO:-/opt/spa}"
MPNN_REPO="${MPNN_REPO:-/opt/ProteinMPNN}"
VARIANT="${VARIANT:-C_n_by_1536}"
K="${K:-4}"; NSEQ="${NSEQ:-4}"; LAM="${LAM:-1}"
GRANS="${GRANS:-domain,segment_small}"
PREP=/workspace/prep; OUT=/workspace/scaffold

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
gcloud storage cp "$MG_CKPT_URI"   /workspace/weights/spa_multigran.pt
gcloud storage cp "$BASE_CKPT_URI" /workspace/weights/spa_base.pt
gcloud storage cp "$PREP_URI/*" "$PREP/"
MAN="$PREP/scaffold_resolved.json"
[ -f "$MAN" ] || { log "FATAL: $MAN not found"; exit 1; }

# Flatten to: id <TAB> len <TAB> gran <TAB> start <TAB> end   (only the requested GRANS)
export GRANS
python -c "import json,os
grans=set(os.environ['GRANS'].split(','))
for p in json.load(open('$MAN'))['prompts']:
    for g,(s,e) in p['grans'].items():
        if g in grans: print('\t'.join([p['id'],str(p['len']),g,str(s),str(e)]))" > "$OUT/work.tsv"
NW=$(wc -l < "$OUT/work.tsv")
log "scaffolding big-run: $NW (prompt×gran) units, grans=$GRANS, K=$K N=$NSEQ λ=$LAM; multigran vs base"

# ckpt-groups as PARALLEL ARRAYS (not a space-split string): the conds contain glob brackets
# `[baseline,spa]`/`[spa]`, so an unquoted `for x in $STR` would pathname-expand them into garbage.
# Arrays with quoted access never word-split or glob. multigran runs baseline+spa (baseline is
# ckpt-independent, scored once per unit); base runs spa only (== multigran's baseline).
G_TAGS=(multigran base)
G_CKPTS=(/workspace/weights/spa_multigran.pt /workspace/weights/spa_base.pt)
G_CONDS=('[baseline,spa]' '[spa]')

run_one(){  # $1=tag $2=ckpt $3=conds $4=id $5=len $6=gran $7=start $8=end
  local tag="$1" ckpt="$2" conds="$3" id="$4" len="$5" gran="$6" s="$7" e="$8"
  local po="$OUT/$tag/$gran/$id"
  python "$SPA_REPO/scripts/eval/run_flywheel.py" \
    variant="$VARIANT" hardware=cloud_h100 \
    "eval.conditions=$conds" "eval.lambda_scale=[$LAM]" \
    eval.num_designs="$K" eval.proteinmpnn.num_seqs="$NSEQ" eval.length="$len" \
    eval.ckpt="$ckpt" \
    eval.prompt_cache="$PREP/$id.pt" \
    +eval.flywheel.prompt_struct="$PREP/$id.pdb" \
    "+eval.subregion.keep_range=[$s,$e]" \
    paths.rfd3_ckpt=/workspace/weights/rfd3_latest.ckpt \
    paths.proteinmpnn_repo="$MPNN_REPO" \
    +eval.flywheel.refolder._target_=spa.eval.openfold3.OF3Refolder \
    +eval.flywheel.refolder.ckpt_path=/workspace/weights/of3-p2-155k.pt \
    +eval.flywheel.refolder.runner_yaml="$SPA_REPO/configs/of3/of3_triton.yml" \
    +eval.flywheel.refolder.out_dir="$po" \
    eval.out_dir="$po" </dev/null \
    && { gcloud storage cp "$po/flywheel_results.json" "$RESULTS_URI/$tag/$gran/$id.json" 2>/dev/null; \
         [ "${LEAN_RESULTS:-0}" = "1" ] || gcloud storage cp "$po"/*.pdb "$RESULTS_URI/$tag/$gran/$id/" 2>/dev/null; \
         OK=$((OK+1)); log "  [$tag] $id/$gran OK -> staged"; } \
    || log "  [$tag] $id/$gran FAILED (continuing)"
}

OK=0
for gi in "${!G_TAGS[@]}"; do
  tag="${G_TAGS[$gi]}"; ckpt="${G_CKPTS[$gi]}"; conds="${G_CONDS[$gi]}"
  log "=== ckpt-group $tag ($ckpt) conds=$conds ==="
  n=0
  while IFS=$'\t' read -r id len gran s e; do
    n=$((n+1)); log "  [$tag $n/$NW] $id len=$len gran=$gran S=[$s,$e)"
    run_one "$tag" "$ckpt" "$conds" "$id" "$len" "$gran" "$s" "$e"
  done < "$OUT/work.tsv"
done
TOTAL=$(( NW * ${#G_TAGS[@]} ))
log "===== SCAFFOLDING BIG-RUN DONE: $OK/$TOTAL units staged -> $RESULTS_URI ====="
# Fail LOUDLY if nothing staged (else Vertex reports a vacuous SUCCEEDED, as the glob bug did).
[ "$OK" -gt 0 ] || { log "FATAL: 0/$TOTAL units staged — every flywheel call failed (see errors above)."; exit 1; }
