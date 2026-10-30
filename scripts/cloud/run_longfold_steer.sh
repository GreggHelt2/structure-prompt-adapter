#!/usr/bin/env bash
# In-container LONG-PROMPT STEERING sweep on the H100 (dev docs/plan/153 section 10): the
# character-stratified soft-only self-prompt experiment. For each of 28 CDDB test-split folds (7 per
# class: pure-a / pure-b / mixed / irregular, 262-373 aa) run the flywheel BASELINE (lambda=0,
# unconditional RFD3) vs SPA (lambda=1, soft-only self-prompt, NO motif) at eval.length = fold length.
#   STAGE=adherence     : gen-only (inverse_fold=false, NO refolder) -> tm_norm_prompt adherence (+ P1
#                         from the baseline pool). 16 draws/arm (2 seeds x K=8). Core readout; cheap.
#   STAGE=designability : + ProteinMPNN -> OpenFold3 (nokernel, bs=8 via batch_patch_shim) -> scRMSD.
#                         8 draws/arm (seeds=[0]), gated on Stage 1.
#
# DATA: uses the ESTABLISHED H100 pattern (run_train.sh / run_cache_gen.sh), NOT a bespoke prep:
#   - ESM3 prompts: load_cache.sh untars the full esm3_cache.tar to NVMe once, then each fold's tensor
#     is a lookup $ESM3_CACHE_DIR/AF-<id>-F1-model_v4_esmfold_v1.pt (the exact cached tensors; no regen,
#     no GPU, no subset extraction). All 28 verified present in the cache (plan/153 section 10.1).
#   - Reference PDBs: the full CDDB is pulled + untarred from NGC to $PDB_DIR (the established
#     run_train.sh pattern); each fold's .pdb is a lookup there. (No bespoke staging.)
# Self-contained H100 epoch: determinism RELAXED to fixed-seeds-only (plan/153 section 10.2); RFD3 gen
# shim kept (free, eval.deterministic=true); OF3 bs=8 accepts per-row non-reproducibility (plan/97: 0
# designability verdict flips). Mirrors run_eval.sh's per-fold flywheel call (b1_full, EXPERIMENTS.md:185).
# NB: NOT `set -e`. DRAFT: validate with DRY_RUN=1, then a tiny smoke (1 fold per class) first.
set -uo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
CACHE_TAR="${CACHE_TAR:-$BUCKET/esm3_cache.tar}"               # the full ESM3 prompt cache (all CDDB splits)
NGC_RESOURCE="${NGC_RESOURCE:-nvidia/clara/proteina-atomistica_data:release}"  # CDDB PDBs (run_train pattern)
RESULTS_URI="${RESULTS_URI:-$BUCKET/eval/longfold_steer/results}"
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
OF3_CKPT_URI="${OF3_CKPT_URI:-$BUCKET/weights/of3-p2-155k.pt}"
SPA_CKPT_REL="${SPA_CKPT_REL:-checkpoints/spa-Nx1536-uncond/spa_C_final.pt}"
SPA_REPO="${SPA_REPO:-/opt/spa}"
MPNN_REPO="${MPNN_REPO:-/opt/ProteinMPNN}"
STAGE="${STAGE:-adherence}"                 # adherence | designability
ARM="${ARM:-rfd3}"                          # rfd3 = 200 / gamma_0 0.6 (RFD3 paper sampler; plan/153 section 10.3)
MANIFEST_REL="${MANIFEST_REL:-configs/eval/manifest_longfold_steer.json}"
# Scratch = the a3 local-SSD RAID0 (mounts at /workspace on this image; auto-RAID0, ~750 GB), same as
# run_train.sh / run_cache_gen.sh. The 234 GiB cache untars here at ~1 GB/s.
SCRATCH="${SCRATCH:-/workspace}"
ESM3_CACHE_DIR="${ESM3_CACHE_DIR:-$SCRATCH/esm3_cache}"
DATA_DIR="${DATA_DIR:-$SCRATCH/data}"       # CDDB download + untar (PDB_DIR detected under it)
OUT="$SCRATCH/longfold_out"

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

# --- GPU sanity ---
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/nvidia/bin:${PATH}"; ldconfig 2>/dev/null || true
python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
[ "$STAGE" = "designability" ] && conda run -n spa-verify-of3 python -c "import torch, triton; print('OF3 env: triton', triton.__version__)"

# --- SPA repo + ProteinMPNN (MPNN only for designability) ---
[ -d "$SPA_REPO/.git" ] || git clone --depth 1 --branch "${REPO_REF:-main}" "${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}" "$SPA_REPO"
pip install -e "$SPA_REPO" --no-deps -q
[ "$STAGE" = "designability" ] && { [ -d "$MPNN_REPO" ] || git clone --depth 1 https://github.com/dauparas/ProteinMPNN "$MPNN_REPO"; }

# --- Secrets (NGC for the CDDB pull, as run_train.sh) ---
export NGC_CLI_API_KEY="$(gcloud secrets versions access latest --secret=spa-ngc-key --project="$PROJECT")"
export NGC_CLI_ORG=nvidia
export NGC_CLI_FORMAT_TYPE=ascii

# --- Weights ---
mkdir -p /workspace/weights "$DATA_DIR" "$OUT"
gcloud storage cp "$RFD3_CKPT_URI" /workspace/weights/rfd3_latest.ckpt
gcloud storage cp "$BUCKET/$SPA_CKPT_REL" /workspace/weights/spa.pt
[ "$STAGE" = "designability" ] && gcloud storage cp "$OF3_CKPT_URI" /workspace/weights/of3-p2-155k.pt

# --- CDDB PDBs: pull + untar from NGC (ESTABLISHED run_train.sh pattern; reference for prompt_struct) ---
log "downloading CDDB from NGC -> $DATA_DIR"
ngc registry resource download-version "$NGC_RESOURCE" --dest "$DATA_DIR"
TARBALL="$(find "$DATA_DIR" -name 'atomistica_cd_dataset.tar.gz' | head -1)"
log "untarring $TARBALL"
tar -xzf "$TARBALL" -C "$DATA_DIR"
PDB_DIR="$(find "$DATA_DIR" -type d -name pdb | head -1)"
log "CDDB PDB dir: $PDB_DIR ($(ls "$PDB_DIR" 2>/dev/null | wc -l) files)"

# --- ESM3 prompt cache: hydrate the full tar to NVMe once (ESTABLISHED pattern; load_cache.sh) ---
if [ ! -d "$ESM3_CACHE_DIR" ] || [ -z "$(ls -A "$ESM3_CACHE_DIR" 2>/dev/null)" ]; then
  log "load ESM3 cache: $CACHE_TAR -> $ESM3_CACHE_DIR"
  GCLOUD="$(command -v gcloud)" bash "$SPA_REPO/scripts/cloud/load_cache.sh" "$CACHE_TAR" "$ESM3_CACHE_DIR"
fi
log "ESM3 cache: $(find "$ESM3_CACHE_DIR" -name '*.pt' 2>/dev/null | wc -l) .pt present"

# --- Config from the manifest (in the cloned repo) ---
MAN="$SPA_REPO/$MANIFEST_REL"
[ -f "$MAN" ] || { log "FATAL: manifest $MAN not found"; exit 1; }
LAM=$(python -c "import json;print(json.load(open('$MAN'))['lambda_scale'])")
K=$(python -c "import json;print(json.load(open('$MAN'))['num_designs'])")
NSEQ=$(python -c "import json;print(json.load(open('$MAN'))['num_seqs'])")
if [ "$STAGE" = "designability" ]; then SEEDS="${SEEDS:-[0]}"; else SEEDS="${SEEDS:-$(python -c "import json;print(json.load(open('$MAN'))['seeds'])")}"; fi
[ -n "${K_OVERRIDE:-}" ] && K="$K_OVERRIDE"
export SUBSET_IDS="${SUBSET_IDS:-}"   # comma list of fold ids -> run only these (region split / smoke); empty = all 28
OF3_BATCH_SHIM="$SPA_REPO/scripts/eval/of3_batch_patch.py"
STEM(){ echo "AF-$1-F1-model_v4_esmfold_v1"; }
log "config: stage=$STAGE spa_ckpt=$SPA_CKPT_REL lambda=$LAM K=$K N=$NSEQ seeds=$SEEDS arm=$ARM subset=${SUBSET_IDS:-ALL}"

# --- Per-fold loop (id<TAB>length) ---
python -c "import json,os
sub=set(x for x in os.environ.get('SUBSET_IDS','').split(',') if x)
for p in json.load(open('$MAN'))['prompts']:
    if not sub or p['id'] in sub: print(p['id']+chr(9)+str(p['length']))" > "$OUT/prompts.tsv"
. "$(dirname "${BASH_SOURCE[0]}")/../_sampler_arm.sh"   # expands $ARM -> SAMPLER_ARGS (same as run_eval.sh)
n=0; ok=0
while IFS=$'\t' read -r id len; do
  n=$((n+1)); po="$OUT/$id"; stem="$(STEM "$id")"
  pt="$ESM3_CACHE_DIR/$stem.pt"; pdb="$PDB_DIR/$stem.pdb"
  [ -f "$pt" ]  || { log "[$n] $id FAILED: cache tensor missing ($pt)"; continue; }
  [ -f "$pdb" ] || { log "[$n] $id FAILED: reference PDB missing ($pdb)"; continue; }
  log "[$n] fold $id (L=$len, soft-only self-prompt, stage=$STAGE)"
  ARGS=( variant=C_n_by_1536 hardware=cloud_h100
         "eval.conditions=[baseline,spa]" "eval.lambda_scale=[$LAM]"
         "eval.num_designs=$K" "eval.seeds=$SEEDS"
         "${SAMPLER_ARGS[@]}"
         eval.deterministic=true
         "eval.ckpt=/workspace/weights/spa.pt"
         "eval.prompt_cache=$pt" "eval.prompt_id=$id"
         "eval.length=$len"
         "+eval.flywheel.prompt_struct=$pdb"
         "paths.rfd3_ckpt=/workspace/weights/rfd3_latest.ckpt"
         "eval.out_dir=$po" )
  if [ "$STAGE" = "designability" ]; then
    # bs=8: batch_patch_shim + of3_nokernel_bs8.yml (data_module_args.batch_size=8) + deterministic=false
    # (OF3Refolder RAISES if deterministic=true with a shim; src/spa/eval/openfold3.py:133-140).
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
      && { log "[$n] $id DRY_RUN parse OK"; ok=$((ok+1)); } || { log "[$n] $id DRY_RUN parse FAILED"; tail -5 "$OUT/$id.dry.err" >&2; }
    continue
  fi
  python "$SPA_REPO/scripts/eval/run_flywheel.py" "${ARGS[@]}" </dev/null \
    && { ok=$((ok+1)); gcloud storage cp "$po/flywheel_results.json" "$RESULTS_URI/$STAGE/$id.json" 2>/dev/null && log "[$n] $id OK -> $RESULTS_URI/$STAGE/$id.json"; } \
    || log "[$n] $id FAILED (continuing)"
done < "$OUT/prompts.tsv"
log "===== LONGFOLD-STEER $STAGE DONE: $ok/$n folds -> $RESULTS_URI/$STAGE ====="
