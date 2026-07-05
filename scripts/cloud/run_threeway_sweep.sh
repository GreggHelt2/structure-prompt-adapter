#!/usr/bin/env bash
# In-container THREE-WAY A⊕B⊕C sweep on the H100 (dev docs/plan/23; the cloud continuation of dev 21).
# Sweeps fixed motif M × target fold G × layout × λ with the multigranularity SPA ckpt, in two stages:
#   STAGE=adherence     : probe_hard_soft_free.py over --layouts × --lambdas per (motif,fold) -> grid_result.json
#                         (cheap; no OF3). The screen — rank cells by net-steer with the pin held.
#   STAGE=designability : for each WINNER cell (motif:seg:fold:layout:lambda), regenerate that cell then
#                         score_threeway_designability.py (ProteinMPNN->OF3 scRMSD) -> designability.json.
# Mirrors run_scaffold_eval.sh (spa-combined image: RFD3+SPA+ESM3+tmtools in system python, OF3 in the
# spa-verify-of3 conda env; ProteinMPNN git-cloned; of3_triton.yml refolder). NOT set -e: one cell failing
# must not abort the sweep. **DRAFT — validate with DRY_RUN then a tiny smoke (1 motif × 1 fold) first.**
set -uo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
PREP_URI="${PREP_URI:-$BUCKET/eval/threeway/prep}"            # AF-<id>-...pdb for every motif source + fold (prep_threeway.py)
RESULTS_URI="${RESULTS_URI:-$BUCKET/eval/threeway/results}"
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
OF3_CKPT_URI="${OF3_CKPT_URI:-$BUCKET/weights/of3-p2-155k.pt}"
MG_CKPT_URI="${MG_CKPT_URI:-$BUCKET/checkpoints/spa-Nx1536-multigran/spa_C_final.pt}"
SPA_REPO="${SPA_REPO:-/opt/spa}"
MPNN_REPO="${MPNN_REPO:-/opt/ProteinMPNN}"

STAGE="${STAGE:-adherence}"                                  # adherence | designability
MOTIFS="${MOTIFS:-A0A2X2KHU0:A2-20,A0A7C9GW19:A30-50}"       # <uniprot>:<segment> list
FOLDS="${FOLDS:-A0A090ME36,A0A3P5VTL4,A0A6A0D1E8}"           # target-fold G ids (embedded live by the image's ESM3)
LAYOUTS="${LAYOUTS:-BAC,ABC,CAB}"; LAMBDAS="${LAMBDAS:-2,3}"
ULEN="${ULEN:-90}"; CLEN="${CLEN:-120}"; K="${K:-8}"; NSEQ="${NSEQ:-8}"; SEEDS="${SEEDS:-0}"
WINNERS="${WINNERS:-}"                                       # designability: motif:seg:fold:layout:lambda[,...]
PREP=/workspace/prep; OUT=/workspace/threeway

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
gcloud storage cp "$MG_CKPT_URI"   /workspace/weights/spa_multigran.pt
[ "$STAGE" = "designability" ] && gcloud storage cp "$OF3_CKPT_URI" /workspace/weights/of3-p2-155k.pt
gcloud storage cp "$PREP_URI/*" "$PREP/" 2>/dev/null || gcloud storage cp -r "$PREP_URI/." "$PREP/"

# Cloud paths -> the drivers' env fallbacks (see the --*-ckpt / $ENV args added in dev 23 §2).
export RFD3_CKPT=/workspace/weights/rfd3_latest.ckpt
export OF3_CKPT=/workspace/weights/of3-p2-155k.pt
export OF3_RUNNER_YAML="$SPA_REPO/configs/of3/of3_triton.yml"
export PROTEINMPNN_REPO="$MPNN_REPO"
SPA_CKPT=/workspace/weights/spa_multigran.pt
COMMON=(--ckpt "$SPA_CKPT" --rfd3-ckpt "$RFD3_CKPT" --pdb-dir "$PREP" --u-len "$ULEN" --c-len "$CLEN")

OK=0
if [ "$STAGE" = "adherence" ]; then
  IFS=',' read -ra MO <<< "$MOTIFS"; IFS=',' read -ra FO <<< "$FOLDS"
  TOTAL=$(( ${#MO[@]} * ${#FO[@]} * $(tr ',' ' ' <<< "$SEEDS" | wc -w) ))
  log "ADHERENCE screen: ${#MO[@]} motifs × ${#FO[@]} folds × seeds[$SEEDS], layouts=$LAYOUTS λ=$LAMBDAS K=$K"
  for ms in "${MO[@]}"; do
    mid="${ms%%:*}"; seg="${ms##*:}"
    for f in "${FO[@]}"; do
      for s in $(tr ',' ' ' <<< "$SEEDS"); do
        po="$OUT/adherence/${mid}_${seg}_${f}_s${s}"
        python "$SPA_REPO/scripts/eval/probe_hard_soft_free.py" "${COMMON[@]}" \
          --motif-source "$mid" --motif-seg "$seg" --target "$f" \
          --layouts "$LAYOUTS" --lambdas "$LAMBDAS" --num-designs "$K" --seed "$s" \
          --out-dir "$po" </dev/null \
          && { gcloud storage cp "$po/grid_result.json" "$RESULTS_URI/adherence/${mid}_${seg}_${f}_s${s}.json" 2>/dev/null; \
               OK=$((OK+1)); log "  [adh] ${mid}:${seg} × ${f} s${s} OK -> staged"; } \
          || log "  [adh] ${mid}:${seg} × ${f} s${s} FAILED (continuing)"
      done
    done
  done

elif [ "$STAGE" = "designability" ]; then
  [ -n "$WINNERS" ] || { log "FATAL: STAGE=designability needs WINNERS=motif:seg:fold:layout:lambda[,...]"; exit 1; }
  IFS=',' read -ra WS <<< "$WINNERS"; TOTAL=${#WS[@]}
  log "DESIGNABILITY on ${#WS[@]} winner cell(s): K=$K N=$NSEQ"
  for w in "${WS[@]}"; do
    IFS=':' read -r mid seg f layout lam <<< "$w"
    po="$OUT/desig/${mid}_${seg}_${f}_${layout}_l${lam}"
    # regenerate this one cell (same seed 0), then score its localized designs
    python "$SPA_REPO/scripts/eval/probe_hard_soft_free.py" "${COMMON[@]}" \
      --motif-source "$mid" --motif-seg "$seg" --target "$f" \
      --layouts "$layout" --lambdas "$lam" --num-designs "$K" --seed 0 --out-dir "$po" </dev/null || { log "  [des] $w GEN FAILED"; continue; }
    contig=$(cd "$SPA_REPO/scripts/eval" && python -c "from probe_hard_soft_free import build_contig; print(build_contig('$seg',$ULEN,$CLEN,'$layout')[0])")
    pdbs=$(ls "$po/${layout}_${mid}_${f}_"*/localized_l${lam}_*.pdb 2>/dev/null)
    [ -n "$pdbs" ] || { log "  [des] $w: no localized PDBs found"; continue; }
    python "$SPA_REPO/scripts/eval/score_threeway_designability.py" \
      --pdbs $pdbs --contig "$contig" --motif-source "$PREP/AF-${mid}-F1-model_v4_esmfold_v1.pdb" \
      --num-seqs "$NSEQ" --out-dir "$po/desig" </dev/null \
      && { gcloud storage cp "$po/desig/designability.json" "$RESULTS_URI/designability/${mid}_${seg}_${f}_${layout}_l${lam}.json" 2>/dev/null; \
           OK=$((OK+1)); log "  [des] $w OK -> staged"; } \
      || log "  [des] $w SCORE FAILED (continuing)"
  done
else
  log "FATAL: unknown STAGE=$STAGE (adherence|designability)"; exit 1
fi

log "===== THREE-WAY SWEEP ($STAGE) DONE: $OK/$TOTAL cell(s) staged -> $RESULTS_URI ====="
[ "$OK" -gt 0 ] || { log "FATAL: 0/$TOTAL staged — every cell failed (see errors above)."; exit 1; }
