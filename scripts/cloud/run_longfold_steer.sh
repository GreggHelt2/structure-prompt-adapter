#!/usr/bin/env bash
# In-container LONG-PROMPT STEERING sweep on the H100 (dev docs/plan/153 section 10): the
# character-stratified soft-only self-prompt experiment. For each of 28 CDDB test-split folds (7 per
# class: pure-a / pure-b / mixed / irregular, 262-373 aa) run the flywheel BASELINE (lambda=0,
# unconditional RFD3) vs SPA (lambda=1, soft-only self-prompt, NO motif) with eval.length = the fold's
# own length.
#   STAGE=adherence     : gen-only (inverse_fold=false, NO refolder) -> tm_norm_prompt adherence (+ P1
#                         from the baseline pool). 16 draws/arm (2 seeds x K=8). Core readout; cheap.
#   STAGE=designability : + ProteinMPNN -> OpenFold3 (nokernel, bs=8 via of3_batch_patch shim) -> scRMSD.
#                         8 draws/arm (seeds=[0]), gated on Stage 1.
# Mirrors run_eval.sh (b1_full, EXPERIMENTS.md:185) EXACTLY except: soft-only (no motif contig;
# eval.length instead), the 28-fold manifest, eval.seeds=[0,1], Stage-2 draws=8. Self-contained H100
# epoch: determinism RELAXED to fixed-seeds-only (plan/153 section 10.2); the RFD3 generation shim is
# kept (free, eval.deterministic=true), OF3 bs=8 accepts per-row non-reproducibility (plan/97: 0
# designability verdict flips). Per-prompt (one failure does not sink the run). Bootstraps like run_eval.sh.
# NB: NOT `set -e`. DRAFT: validate with DRY_RUN=1, then a tiny smoke (1 fold per class) first.
set -uo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
PREP_URI="${PREP_URI:-$BUCKET/eval/longfold_steer/prep}"       # .pt (from esm3_cache.tar) + .pdb + longfold_steer_resolved.json
RESULTS_URI="${RESULTS_URI:-$BUCKET/eval/longfold_steer/results}"
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
OF3_CKPT_URI="${OF3_CKPT_URI:-$BUCKET/weights/of3-p2-155k.pt}"
SPA_REPO="${SPA_REPO:-/opt/spa}"
MPNN_REPO="${MPNN_REPO:-/opt/ProteinMPNN}"
STAGE="${STAGE:-adherence}"                 # adherence | designability
# RFD3 sampler: ARM=rfd3 = the RFdiffusion3 paper's 200 / gamma_0 0.6 (plan/153 section 10.3), NOT the
# checkpoint's 100 / 0.8. _sampler_arm.sh expands $ARM into SAMPLER_ARGS (same mechanism as run_eval.sh).
ARM="${ARM:-rfd3}"
. "$(dirname "${BASH_SOURCE[0]}")/../_sampler_arm.sh"
PREP=/workspace/prep
OUT=/workspace/longfold_out

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

# --- GPU sanity ---
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/nvidia/bin:${PATH}"; ldconfig 2>/dev/null || true
python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
[ "$STAGE" = "designability" ] && conda run -n spa-verify-of3 python -c "import torch, triton; print('OF3 env: triton', triton.__version__)"

# --- SPA repo + ProteinMPNN (MPNN only needed for designability) ---
[ -d "$SPA_REPO/.git" ] || git clone --depth 1 --branch "${REPO_REF:-main}" "${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}" "$SPA_REPO"
pip install -e "$SPA_REPO" --no-deps -q
[ "$STAGE" = "designability" ] && { [ -d "$MPNN_REPO" ] || git clone --depth 1 https://github.com/dauparas/ProteinMPNN "$MPNN_REPO"; }

# --- Fetch weights + prep artifacts ---
mkdir -p /workspace/weights "$PREP" "$OUT"
gcloud storage cp "$RFD3_CKPT_URI" /workspace/weights/rfd3_latest.ckpt
[ "$STAGE" = "designability" ] && gcloud storage cp "$OF3_CKPT_URI" /workspace/weights/of3-p2-155k.pt
gcloud storage cp "$PREP_URI/*" "$PREP/"

MAN="$PREP/longfold_steer_resolved.json"
[ -f "$MAN" ] || { log "FATAL: $MAN not found (did prep_longfold_steer.py stage to $PREP_URI?)"; exit 1; }
SPA_CKPT_REL=$(python -c "import json;print(json.load(open('$MAN'))['spa_ckpt'])")
LAM=$(python -c "import json;print(json.load(open('$MAN'))['lambda_scale'])")
K=$(python -c "import json;print(json.load(open('$MAN'))['num_designs'])")
NSEQ=$(python -c "import json;print(json.load(open('$MAN'))['num_seqs'])")
gcloud storage cp "$BUCKET/checkpoints/$SPA_CKPT_REL" /workspace/weights/spa.pt

# Seeds: Stage 1 uses the manifest's seeds (2 seeds x K=8 = 16 draws); Stage 2 subsets to 8 draws (seed 0).
if [ "$STAGE" = "designability" ]; then
  SEEDS="${SEEDS:-[0]}"
else
  SEEDS="${SEEDS:-$(python -c "import json;print(json.load(open('$MAN'))['seeds'])")}"
fi
[ -n "${K_OVERRIDE:-}" ] && K="$K_OVERRIDE"
export SUBSET_IDS="${SUBSET_IDS:-}"   # comma list -> run only these folds (smoke/validation); empty = all 28
log "config: stage=$STAGE spa_ckpt=$SPA_CKPT_REL lambda=$LAM K=$K N=$NSEQ seeds=$SEEDS arm=$ARM subset=${SUBSET_IDS:-ALL}"

# OF3 bs=8 batch shim (designability only). nokernel runner: the triton kernels cannot batch
# (evoformer.py:915). Same mechanism run_threeway_sweep.sh uses (plan/153 section 10.6; dev plan/24).
if [ "$STAGE" = "designability" ]; then
  export OF3_BATCH_SHIM="$SPA_REPO/scripts/eval/of3_batch_patch.py"
  export OF3_BATCH_SIZE="${OF3_BATCH_SIZE:-8}"
fi

# --- Per-fold loop (id<TAB>length; soft-only, so the manifest carries length not a motif contig) ---
python -c "import json,os
sub=set(x for x in os.environ.get('SUBSET_IDS','').split(',') if x)
for p in json.load(open('$MAN'))['prompts']:
    if not sub or p['id'] in sub: print(p['id']+chr(9)+str(p['length']))" > "$OUT/prompts.tsv"
n=0; ok=0
while IFS=$'\t' read -r id len; do
  n=$((n+1)); po="$OUT/$id"
  log "[$n] fold $id (L=$len, soft-only self-prompt, stage=$STAGE)"
  ARGS=( variant=C_n_by_1536 hardware=cloud_h100
         "eval.conditions=[baseline,spa]" "eval.lambda_scale=[$LAM]"
         "eval.num_designs=$K" "eval.seeds=$SEEDS"
         "${SAMPLER_ARGS[@]}"
         eval.deterministic=true
         "eval.ckpt=/workspace/weights/spa.pt"
         "eval.prompt_cache=$PREP/$id.pt" "eval.prompt_id=$id"
         "eval.length=$len"
         "+eval.flywheel.prompt_struct=$PREP/$id.pdb"
         "paths.rfd3_ckpt=/workspace/weights/rfd3_latest.ckpt"
         "eval.out_dir=$po" )
  if [ "$STAGE" = "designability" ]; then
    ARGS+=( "eval.proteinmpnn.num_seqs=$NSEQ" "paths.proteinmpnn_repo=$MPNN_REPO"
            "+eval.flywheel.refolder._target_=spa.eval.openfold3.OF3Refolder"
            "+eval.flywheel.refolder.deterministic=false"
            "+eval.flywheel.refolder.ckpt_path=/workspace/weights/of3-p2-155k.pt"
            "+eval.flywheel.refolder.runner_yaml=$SPA_REPO/configs/of3/of3_nokernel.yml"
            "+eval.flywheel.refolder.out_dir=$po" )
    # ⚠️ TODO-VERIFY before launch: how OF3Refolder receives bs=8. run_threeway_sweep.sh passes it via
    # the OF3_BATCH_SHIM + OF3_BATCH_SIZE env (set above) which injects data_module_args.batch_size into
    # the runner yaml; confirm OF3Refolder honors that env here, or add the batch_size as a refolder arg.
  else
    ARGS+=( "eval.flywheel.inverse_fold=false" )
  fi
  if [ -n "${DRY_RUN:-}" ]; then
    python "$SPA_REPO/scripts/eval/run_flywheel.py" --cfg job "${ARGS[@]}" >/dev/null 2>"$OUT/$id.dry.err" \
      && { log "[$n] $id DRY_RUN parse OK"; ok=$((ok+1)); } || { log "[$n] $id DRY_RUN parse FAILED"; tail -5 "$OUT/$id.dry.err" >&2; }
    continue
  fi
  python "$SPA_REPO/scripts/eval/run_flywheel.py" "${ARGS[@]}" </dev/null \
    && { ok=$((ok+1)); gcloud storage cp "$po/flywheel_results.json" "$RESULTS_URI/$STAGE/$id.json" 2>/dev/null && log "[$n] $id OK -> $RESULTS_URI/$STAGE/$id.json"; } \
    || log "[$n] $id FAILED (continuing)"
done < "$OUT/prompts.tsv"
log "===== LONGFOLD-STEER $STAGE DONE: $ok/$n folds -> $RESULTS_URI/$STAGE ====="
