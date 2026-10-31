#!/usr/bin/env bash
# G4-A: soft-only SPA steering on the RECOGNIZABLE long-fold panel (plan/153 §9 #13 / §9a G4-A).
# 16 folds (the 20-fold panel minus the 4 redundant a+b: 2MNR,1GPD,6LDH,1QLP). Two arms: baseline
# (lambda=0, unconditional RFD3 at prompt length) and spa (lambda=1, soft-only self-prompt). Adherence
# gen-only (STAGE=adherence); designability is a gated follow-on (STAGE=designability, not default).
# Prompts are REAL PDB (not CDDB), so there is NO esm3_cache.tar hydrate: each fold's ESM3 prompt tensor
# + reference chain PDB are pulled individually from GCS (staged by run_extract_prompt_tensors.sh):
#   - 13 generated folds: gs://.../eval/longfold_steer/prompt_tensors/<ID>.{pt,pdb}
#   - 3 b4 folds (7RPN,7YII,8SIU):  gs://.../eval/b4/prep/<ID>.{pt,pdb}
# eval.length is derived per fold from the tensor shape (prompt = design length, self-prompt matched).
# Launch via: RUN_SCRIPT=run_g4a_recognizable.sh REGION=us-west1 bash submit_longfold_job.sh
# NB: NOT set -e; validate DRY_RUN=1 then a 1-per-class smoke first.
set -uo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
TENS_URI="${TENS_URI:-$BUCKET/eval/longfold_steer/prompt_tensors}"
B4_URI="${B4_URI:-$BUCKET/eval/b4/prep}"
RESULTS_URI="${RESULTS_URI:-$BUCKET/eval/g4a_recognizable/results}"
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
OF3_CKPT_URI="${OF3_CKPT_URI:-$BUCKET/weights/of3-p2-155k.pt}"
SPA_CKPT_REL="${SPA_CKPT_REL:-checkpoints/spa-Nx1536-uncond/spa_C_final.pt}"
SPA_REPO="${SPA_REPO:-/opt/spa}"
MPNN_REPO="${MPNN_REPO:-/opt/ProteinMPNN}"
STAGE="${STAGE:-adherence}"
ARM="${ARM:-rfd3}"                       # 200 / gamma_0 0.6
LAM="${LAM:-1}"
K="${K:-8}"
SEEDS="${SEEDS:-[0,1]}"                   # 2 seeds x K=8 = 16 draws/arm (adherence); designability override below
NSEQ="${NSEQ:-8}"
# 16-fold panel (override to trim/extend). Generated (from prompt_tensors) and b4 (from b4/prep):
G4A_GEN_IDS="${G4A_GEN_IDS:-2BCT,1B3U,1AO6,1ANN,2BAT,2OMF,1AIR,1DAB,1GOX,6ADH,2CPP,1DFJ,7API}"
G4A_B4_IDS="${G4A_B4_IDS:-7RPN,7YII,8SIU}"
SCRATCH="${SCRATCH:-/workspace}"; OUT="$SCRATCH/g4a_out"; mkdir -p "$OUT" "$SCRATCH/prompts"
[ "$STAGE" = "designability" ] && SEEDS="${SEEDS_DES:-[0]}"

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/nvidia/bin:${PATH}"; ldconfig 2>/dev/null || true
python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"

[ -d "$SPA_REPO/.git" ] || git clone --depth 1 --branch "${REPO_REF:-main}" "${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}" "$SPA_REPO"
pip install -e "$SPA_REPO" --no-deps -q
[ "$STAGE" = "designability" ] && { [ -d "$MPNN_REPO" ] || git clone --depth 1 https://github.com/dauparas/ProteinMPNN "$MPNN_REPO"; }
mkdir -p /workspace/weights
gcloud storage cp "$RFD3_CKPT_URI" /workspace/weights/rfd3_latest.ckpt
gcloud storage cp "$BUCKET/$SPA_CKPT_REL" /workspace/weights/spa.pt
[ "$STAGE" = "designability" ] && gcloud storage cp "$OF3_CKPT_URI" /workspace/weights/of3-p2-155k.pt
OF3_BATCH_SHIM="$SPA_REPO/scripts/eval/of3_batch_patch.py"
. "$(dirname "${BASH_SOURCE[0]}")/../_sampler_arm.sh"   # $ARM -> SAMPLER_ARGS (file is scripts/_sampler_arm.sh, NOT scripts/cloud/; matches run_longfold_steer.sh:112)
{ [ -n "${SAMPLER_ARGS+x}" ] && [ "${#SAMPLER_ARGS[@]}" -eq 3 ]; } || { echo "FATAL: _sampler_arm.sh did not load (wrong include path or missing file) -- refusing to run at the inherited checkpoint sampler instead of the requested ARM. See the run_g4a 2026-10-31 bug." >&2; exit 2; }

# build the fold table: id<TAB>pt_uri<TAB>pdb_uri
: > "$OUT/folds.tsv"
for id in ${G4A_GEN_IDS//,/ }; do [ -n "$id" ] && echo -e "$id\t$TENS_URI/$id.pt\t$TENS_URI/$id.pdb" >> "$OUT/folds.tsv"; done
for id in ${G4A_B4_IDS//,/ };  do [ -n "$id" ] && echo -e "$id\t$B4_URI/$id.pt\t$B4_URI/$id.pdb"     >> "$OUT/folds.tsv"; done
log "G4-A $STAGE: $(wc -l < "$OUT/folds.tsv") folds, lambda=$LAM, conditions=[baseline,spa], K=$K seeds=$SEEDS arm=$ARM"

n=0; ok=0
while IFS=$'\t' read -r id pt_uri pdb_uri; do
  n=$((n+1)); po="$OUT/$id"
  pt="$SCRATCH/prompts/$id.pt"; pdb="$SCRATCH/prompts/$id.pdb"
  gcloud storage cp "$pt_uri" "$pt" 2>/dev/null   || { log "[$n] $id FAILED: tensor missing ($pt_uri)"; continue; }
  gcloud storage cp "$pdb_uri" "$pdb" 2>/dev/null || { log "[$n] $id FAILED: ref PDB missing ($pdb_uri)"; continue; }
  # length = prompt tensor row count (self-prompt matched)
  len=$(python - "$pt" <<'PY'
import sys, torch
o=torch.load(sys.argv[1], map_location="cpu")
def find(x):
    if torch.is_tensor(x): return x if (x.ndim==2 and x.shape[-1]==1536) else None
    if isinstance(x,dict):
        for v in x.values():
            r=find(v)
            if r is not None: return r
    if isinstance(x,(list,tuple)):
        for v in x:
            r=find(v)
            if r is not None: return r
    return None
t=find(o); print(t.shape[0] if t is not None else 0)
PY
)
  [ "${len:-0}" -ge 1 ] || { log "[$n] $id FAILED: bad length ($len)"; continue; }
  log "[$n] $id (L=$len, soft-only self-prompt, stage=$STAGE)"
  ARGS=( variant=C_n_by_1536 hardware=cloud_h100
         "eval.conditions=[baseline,spa]" "eval.lambda_scale=[$LAM]"
         "eval.num_designs=$K" "eval.seeds=$SEEDS" "${SAMPLER_ARGS[@]}"
         eval.deterministic=true "eval.ckpt=/workspace/weights/spa.pt"
         "eval.prompt_cache=$pt" "eval.prompt_id=$id" "eval.length=$len"
         "+eval.flywheel.prompt_struct=$pdb"
         "paths.rfd3_ckpt=/workspace/weights/rfd3_latest.ckpt" "eval.out_dir=$po" )
  if [ "$STAGE" = "designability" ]; then
    ARGS+=( "eval.proteinmpnn.num_seqs=$NSEQ" "paths.proteinmpnn_repo=$MPNN_REPO"
            "+eval.flywheel.refolder._target_=spa.eval.openfold3.OF3Refolder"
            "+eval.flywheel.refolder.deterministic=false"
            "+eval.flywheel.refolder.ckpt_path=/workspace/weights/of3-p2-155k.pt"
            "+eval.flywheel.refolder.runner_yaml=$SPA_REPO/configs/of3/of3_nokernel_bs8.yml"
            "+eval.flywheel.refolder.batch_patch_shim=$OF3_BATCH_SHIM"
            "+eval.flywheel.refolder.out_dir=$po" )
  else
    ARGS+=( "eval.flywheel.inverse_fold=false" )
  fi
  if [ -n "${DRY_RUN:-}" ]; then
    python "$SPA_REPO/scripts/eval/run_flywheel.py" --cfg job "${ARGS[@]}" >/dev/null 2>"$OUT/$id.dry.err" \
      && { log "[$n] $id DRY_RUN parse OK"; ok=$((ok+1)); } || { log "[$n] $id DRY_RUN FAILED"; tail -5 "$OUT/$id.dry.err" >&2; }
    continue
  fi
  python "$SPA_REPO/scripts/eval/run_flywheel.py" "${ARGS[@]}" </dev/null \
    && { ok=$((ok+1)); gcloud storage cp "$po/flywheel_results.json" "$RESULTS_URI/$STAGE/$id.json" 2>/dev/null && log "[$n] $id OK -> $RESULTS_URI/$STAGE/$id.json"; } \
    || log "[$n] $id FAILED (continuing)"
done < "$OUT/folds.tsv"
log "===== G4-A $STAGE DONE: $ok/$n folds -> $RESULTS_URI/$STAGE ====="
