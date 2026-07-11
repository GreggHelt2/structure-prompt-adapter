#!/usr/bin/env bash
# In-container TWO-STEER designability on the H100 (dev: two-steer Panel-7 build): score PRE-GENERATED
# two-steer backbones (R1|M|R2, motif pinned) with ProteinMPNN(N,seed42) -> OpenFold3 refold -> best-of-N
# scRMSD (<2 Å = designable) + motif-survival, batched OF3 (nokernel, ~2.5x @ bs=8; dev 23 §7.8).
#
# Unlike run_threeway_sweep.sh (STAGE=designability), this does NOT regenerate — RFD3 is not bitwise
# reproducible (dev RFD3_irreproducibility.md), so we fold the EXACT staged PDBs. Mirrors
# run_of3_batch_bench.sh's setup (weights pull, MPNN clone, git-fetch-reset+SHA, error surfacing) but
# calls score_threeway_designability.py per cell. The scorer is layout-agnostic (only needs the contig).
#
# Inputs staged to $PREP_URI/<candidate>_s<seed>/<arm>/*.pdb + $PREP_URI/AF-<MOTIF_ID>-...pdb (motif source).
# Full per-cell tree (design PDBs + MPNN FASTAs + OF3 refold CIFs + designability.json) -> $RESULTS_URI.
set -uo pipefail
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
PREP_URI="${PREP_URI:-$BUCKET/eval/twosteer/prep}"
RESULTS_URI="${RESULTS_URI:-$BUCKET/eval/twosteer/results}"
OF3_CKPT_URI="${OF3_CKPT_URI:-$BUCKET/weights/of3-p2-155k.pt}"
SPA_REPO="${SPA_REPO:-/opt/spa}"; MPNN_REPO="${MPNN_REPO:-/opt/ProteinMPNN}"
NSEQ="${NSEQ:-16}"; PROTEINMPNN_SEED="${PROTEINMPNN_SEED:-42}"; OF3_BATCH_SIZE="${OF3_BATCH_SIZE:-8}"
MOTIF_ID="${MOTIF_ID:-A0A2X2KHU0}"
PREP=/workspace/prep; OUT=/workspace/twosteer
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

# Which cell arms to score, over both candidates × both seeds. Default = the headliner + baseline;
# override e.g. ARMS="g1_free free_g2" for the single-steer follow-up. contig = R1_len,motif_seg,R2_len
# (the design's build_contig BAC string) — per-candidate, since R1_len differs.
ARMS="${ARMS:-g1_g2 free_free}"
declare -A CONTIG=( ["A0A843G012"]="54,A2-20,47" ["A0A3P5VTL4"]="88,A2-20,47" )
COMBOS=()
for cand in A0A843G012 A0A3P5VTL4; do
  for seed in 17 0; do
    for arm in $ARMS; do
      COMBOS+=("$cand $seed $arm ${CONTIG[$cand]}")
    done
  done
done
log "ARMS=$ARMS  ->  ${#COMBOS[@]} cells"

export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/nvidia/bin:${PATH}"; ldconfig 2>/dev/null || true
python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
conda run -n spa-verify-of3 python -c "import triton; print('OF3 env ok: triton', triton.__version__)" || true

# ALWAYS refresh /opt/spa (a warm clone would silently run STALE code) + record the SHA that actually ran.
REPO_REF="${REPO_REF:-main}"; REPO_URL="${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}"
if [ -d "$SPA_REPO/.git" ]; then
  git -C "$SPA_REPO" fetch --depth 1 origin "$REPO_REF" && git -C "$SPA_REPO" reset --hard FETCH_HEAD
else
  git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$SPA_REPO"
fi
pip install -e "$SPA_REPO" --no-deps -q
log "SPA repo @ $REPO_REF  ->  SHA $(git -C "$SPA_REPO" rev-parse HEAD)"
[ -d "$MPNN_REPO" ] || git clone --depth 1 https://github.com/dauparas/ProteinMPNN "$MPNN_REPO"

mkdir -p /workspace/weights "$PREP" "$OUT"
gcloud storage cp "$OF3_CKPT_URI" /workspace/weights/of3-p2-155k.pt
gcloud storage cp "$PREP_URI/AF-${MOTIF_ID}-F1-model_v4_esmfold_v1.pdb" "$PREP/"

export OF3_CKPT=/workspace/weights/of3-p2-155k.pt
export OF3_RUNNER_YAML="$SPA_REPO/configs/of3/of3_nokernel.yml"   # bs>1 forces nokernel anyway (kernel can't batch)
export PROTEINMPNN_REPO="$MPNN_REPO"
export OF3_BATCH_SHIM="$SPA_REPO/scripts/eval/of3_batch_patch.py"

OK=0; TOTAL=${#COMBOS[@]}
for c in "${COMBOS[@]}"; do
  read -r cand seed arm contig <<< "$c"
  key="${cand}_s${seed}_${arm}"
  cdir="$PREP/$key"; po="$OUT/$key"; mkdir -p "$cdir" "$po"
  if ! gcloud storage cp "$PREP_URI/${cand}_s${seed}/${arm}/*.pdb" "$cdir/" 2>/dev/null; then
    log "[des] $key: no PDBs staged at $PREP_URI/${cand}_s${seed}/${arm}/ — skipping"; continue
  fi
  pdbs=$(ls "$cdir"/*.pdb 2>/dev/null)
  [ -n "$pdbs" ] || { log "[des] $key: no local PDBs"; continue; }
  log "[des] $key  contig=$contig  n=$(echo "$pdbs" | wc -w)  N=$NSEQ bs=$OF3_BATCH_SIZE"
  if python "$SPA_REPO/scripts/eval/score_threeway_designability.py" \
       --pdbs $pdbs --contig "$contig" \
       --motif-source "$PREP/AF-${MOTIF_ID}-F1-model_v4_esmfold_v1.pdb" \
       --num-seqs "$NSEQ" --proteinmpnn-seed "$PROTEINMPNN_SEED" \
       --of3-batch-size "$OF3_BATCH_SIZE" --out-dir "$po/desig" </dev/null; then
    gcloud storage cp "$po/desig/designability.json" "$RESULTS_URI/designability/${key}.json" 2>/dev/null
    gcloud storage cp -r "$po" "$RESULTS_URI/designability/cells/" 2>/dev/null || log "  [warn] tree push failed for $key"
    OK=$((OK+1)); log "  [des] $key OK -> json + full tree staged"
  else
    log "  [des] $key SCORE FAILED (continuing)"
  fi
  # surface any SWALLOWED OF3 predict_step tracebacks (bs>1 "no folds" = OF3 exit-0-after-error; dev 23 §7)
  errlogs=$(find "$po" -name "predict_err_rank*.log" 2>/dev/null)
  if [ -n "$errlogs" ]; then
    echo "$errlogs" | while read -r f; do echo "===== $f ====="; sed -n "1,50p" "$f"; \
      gcloud storage cp "$f" "$RESULTS_URI/designability/cells/$key/logs/" 2>/dev/null || true; done
  fi
done
log "===== TWO-STEER DESIGNABILITY DONE: $OK/$TOTAL cell(s) staged -> $RESULTS_URI ====="
[ "$OK" -gt 0 ] || { log "FATAL: 0/$TOTAL staged — every cell failed."; exit 1; }
