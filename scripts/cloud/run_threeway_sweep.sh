#!/usr/bin/env bash
# In-container THREE-WAY A⊕B⊕C sweep on the H100 (dev docs/plan/23; the cloud continuation of dev 21).
# Sweeps fixed motif M × target fold G × layout × λ with the multigranularity SPA ckpt, in two stages:
#   STAGE=adherence     : probe_hard_soft_free.py over --layouts × --lambdas per (motif,fold) -> grid_result.json
#                         (cheap; no OF3). The screen — rank cells by net-steer with the pin held.
#   STAGE=designability : for each WINNER cell (motif:seg:fold:layout:lambda), regenerate that cell then
#                         score_threeway_designability.py (ProteinMPNN->OF3 scRMSD) -> designability.json.
# PERSISTENCE (dev 23): per cell we push the FULL artifact tree to $RESULTS/<stage>/cells/<key>/ — the
# RFD3+SPA design PDBs, ProteinMPNN FASTAs, OF3 refold CIFs, the per-cell json, AND a manifest.json
# (fixed motif + SPA-prompt fold + layout/λ/seed + exact input-file refs) — so 3D figures are makeable after.
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
OF3_BATCH_SIZE="${OF3_BATCH_SIZE:-8}"                        # designability: OF3 refold batch_size (nokernel; ~2.5x@bs=8; dev 23 §7.8; set 1 to disable)
PREP=/workspace/prep; OUT=/workspace/threeway

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
# Write a provenance manifest (json): conditioning + exact input/result file refs, so every saved
# structure is traceable to its fixed motif + SPA-prompt fold. $1=outfile; rest = key=val pairs.
manifest(){ local out="$1"; shift; python -c 'import json,sys; open(sys.argv[1],"w").write(json.dumps(dict(kv.split("=",1) for kv in sys.argv[2:]),indent=2)+"\n")' "$out" "$@"; }

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
# Designability is the only OF3 consumer (adherence has no OF3). OF3 batching (bs>1) REQUIRES nokernel —
# the triton kernels can't batch (evoformer.py:915). So use nokernel for designability, triton otherwise.
if [ "$STAGE" = "designability" ]; then
  export OF3_RUNNER_YAML="$SPA_REPO/configs/of3/of3_nokernel.yml"
else
  export OF3_RUNNER_YAML="$SPA_REPO/configs/of3/of3_triton.yml"
fi
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
        po="$OUT/adherence/${mid}_${seg}_${f}_s${s}"; key="${mid}_${seg}_${f}_s${s}"
        if python "$SPA_REPO/scripts/eval/probe_hard_soft_free.py" "${COMMON[@]}" \
             --motif-source "$mid" --motif-seg "$seg" --target "$f" \
             --layouts "$LAYOUTS" --lambdas "$LAMBDAS" --num-designs "$K" --seed "$s" \
             --out-dir "$po" </dev/null; then
          gcloud storage cp "$po/grid_result.json" "$RESULTS_URI/adherence/${key}.json" 2>/dev/null
          manifest "$po/manifest.json" stage=adherence motif_source="$mid" motif_segment="$seg" \
            target_fold="$f" layouts="$LAYOUTS" lambdas="$LAMBDAS" seed="$s" u_len="$ULEN" c_len="$CLEN" \
            num_designs="$K" input_motif_pdb="$PREP_URI/AF-${mid}-F1-model_v4_esmfold_v1.pdb" \
            input_fold_pdb="$PREP_URI/AF-${f}-F1-model_v4_esmfold_v1.pdb" rfd3_ckpt="$RFD3_CKPT_URI" \
            spa_ckpt="$MG_CKPT_URI" gcs_cell_prefix="$RESULTS_URI/adherence/cells/${key}"
          # FULL cell tree (RFD3+SPA design PDBs + grid_result.json + manifest), path-preserved -> GCS
          gcloud storage cp -r "$po" "$RESULTS_URI/adherence/cells/" 2>/dev/null || log "  [warn] structure push failed for $key"
          OK=$((OK+1)); log "  [adh] ${mid}:${seg} × ${f} s${s} OK -> json+structures+manifest staged"
        else
          log "  [adh] ${mid}:${seg} × ${f} s${s} FAILED (continuing)"
        fi
      done
    done
  done

elif [ "$STAGE" = "designability" ]; then
  [ -n "$WINNERS" ] || { log "FATAL: STAGE=designability needs WINNERS=motif:seg:fold:layout:lambda[,...]"; exit 1; }
  IFS=',' read -ra WS <<< "$WINNERS"; TOTAL=${#WS[@]}
  log "DESIGNABILITY on ${#WS[@]} winner cell(s): K=$K N=$NSEQ"
  for w in "${WS[@]}"; do
    IFS=':' read -r mid seg f layout lam <<< "$w"
    po="$OUT/desig/${mid}_${seg}_${f}_${layout}_l${lam}"; key="${mid}_${seg}_${f}_${layout}_l${lam}"
    # regenerate this one cell (same seed 0), then score its localized designs
    python "$SPA_REPO/scripts/eval/probe_hard_soft_free.py" "${COMMON[@]}" \
      --motif-source "$mid" --motif-seg "$seg" --target "$f" \
      --layouts "$layout" --lambdas "$lam" --num-designs "$K" --seed 0 --out-dir "$po" </dev/null || { log "  [des] $w GEN FAILED"; continue; }
    contig=$(cd "$SPA_REPO/scripts/eval" && python -c "from probe_hard_soft_free import build_contig; print(build_contig('$seg',$ULEN,$CLEN,'$layout')[0])")
    pdbs=$(ls "$po/${layout}_${mid}_${f}_"*/localized_l${lam}_*.pdb 2>/dev/null)
    [ -n "$pdbs" ] || { log "  [des] $w: no localized PDBs found"; continue; }
    if python "$SPA_REPO/scripts/eval/score_threeway_designability.py" \
         --pdbs $pdbs --contig "$contig" --motif-source "$PREP/AF-${mid}-F1-model_v4_esmfold_v1.pdb" \
         --num-seqs "$NSEQ" --of3-batch-size "$OF3_BATCH_SIZE" --out-dir "$po/desig" </dev/null; then
      gcloud storage cp "$po/desig/designability.json" "$RESULTS_URI/designability/${key}.json" 2>/dev/null
      manifest "$po/manifest.json" stage=designability motif_source="$mid" motif_segment="$seg" \
        target_fold="$f" layout="$layout" lambda="$lam" seed=0 u_len="$ULEN" c_len="$CLEN" contig="$contig" \
        num_designs="$K" num_seqs="$NSEQ" input_motif_pdb="$PREP_URI/AF-${mid}-F1-model_v4_esmfold_v1.pdb" \
        input_fold_pdb="$PREP_URI/AF-${f}-F1-model_v4_esmfold_v1.pdb" rfd3_ckpt="$RFD3_CKPT_URI" \
        spa_ckpt="$MG_CKPT_URI" of3_ckpt="$OF3_CKPT_URI" gcs_cell_prefix="$RESULTS_URI/designability/cells/${key}"
      # FULL cell tree: RFD3+SPA design PDBs + ProteinMPNN FASTAs + OF3 refold CIFs + designability.json + manifest
      gcloud storage cp -r "$po" "$RESULTS_URI/designability/cells/" 2>/dev/null || log "  [warn] structure push failed for $key"
      OK=$((OK+1)); log "  [des] $w OK -> json+structures+manifest staged"
    else
      log "  [des] $w SCORE FAILED (continuing)"
    fi
  done
else
  log "FATAL: unknown STAGE=$STAGE (adherence|designability)"; exit 1
fi

log "===== THREE-WAY SWEEP ($STAGE) DONE: $OK/$TOTAL cell(s) staged -> $RESULTS_URI ====="
[ "$OK" -gt 0 ] || { log "FATAL: 0/$TOTAL staged — every cell failed (see errors above)."; exit 1; }
