#!/usr/bin/env bash
# In-container M1-BREADTH regional two-steer sweep on the H100 (dev docs/plan/160, final-push Phase B).
# Ports the A5000 pre-reg two-steer pipeline (row47/51, probe_two_steer.py) to cloud, modeled on
# run_threeway_sweep.sh. Geometry: [R1 | spacer | Fixed Motif | spacer | R2], R1 steered toward a soft
# G1 prompt (the breadth fold), R2 toward G2 (held), Fixed Motif hard-pinned between them.
#   STAGE=adherence     : gen-only (probe_two_steer, NO OF3) -> design PDBs + per-cell provenance. The
#                         POSITIONAL claim (strand/helix/SS-identity/TM) is design-side, scored later.
#   STAGE=designability : score_threeway_designability.py (ProteinMPNN -> OF3 scRMSD) on the g1_g2 PDBs.
# ⭐ PERSISTENCE (Gregg, 2026-11-21): EVERY artifact to GCS — design PDBs, ProteinMPNN FASTAs, OF3 refold
#   CIFs, probe_two_steer's own RUN_PROVENANCE.json (the spa_conditioning block: prompt->region mapping +
#   per-region lambda + motif->design span), AND a per-cell manifest.json (every prompt, the G1->R1 / G2->R2
#   mapping, lambda values, motif, contig, seed). This is the gap that lost results/104/105's structures.
# NOT set -e: one cell failing must not abort the sweep. DRAFT -> validate DRY_RUN then a 1-cell smoke first.
set -uo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
PREP_URI="${PREP_URI:-$BUCKET/eval/m1breadth/prep}"              # AF-<id>-...pdb for every soft G1 fold + G2 + every motif source
RESULTS_URI="${RESULTS_URI:-$BUCKET/eval/m1breadth/results}"
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
OF3_CKPT_URI="${OF3_CKPT_URI:-$BUCKET/weights/of3-p2-155k.pt}"
MG_CKPT_URI="${MG_CKPT_URI:-$BUCKET/checkpoints/spa-Nx1536-multigran/spa_C_final.pt}"
SPA_REPO="${SPA_REPO:-/opt/spa}"
MPNN_REPO="${MPNN_REPO:-/opt/ProteinMPNN}"

STAGE="${STAGE:-adherence}"                                      # adherence | designability
FOLDS="${FOLDS:-A0A850FC24}"                                     # soft G1 prompt (breadth) fold ids
MOTIFS="${MOTIFS:-A0A2X2KHU0:A2-20}"                             # Fixed Motif <uniprot>:<seg> list (hard-pinned between R1/R2)
G2="${G2:-A0A841ZMR1}"                                           # R2 target fold (held from pre-reg)
LAMBDAS="${LAMBDAS:-2.5}"                                        # steered R1 lambda list; R2 lambda is 0.6*R1 (pre-reg 2.5/1.5) unless R2_LAMBDAS set
R2_LAMBDAS="${R2_LAMBDAS:-}"                                     # optional explicit R2 lambda list (parallel to LAMBDAS); default = each * 0.6
INNER_SPACER="${INNER_SPACER:-16}"; TERM_SPACER="${TERM_SPACER:-0}"
K="${K:-1}"; NSEQ="${NSEQ:-4}"; SEEDS="${SEEDS:-0}"
OF3_BATCH_SIZE="${OF3_BATCH_SIZE:-8}"                            # designability OF3 refold batch (nokernel; ~2.5x@bs=8)
ARM="${ARM:-rfd3}"                                              # sampler arm: rfd3 = 200/0.6/1.5 (pre-reg). NOT 'ours' (100/0.8).
PREP=/workspace/prep; OUT=/workspace/m1breadth

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
manifest(){ local out="$1"; shift; python -c 'import json,sys; open(sys.argv[1],"w").write(json.dumps(dict(kv.split("=",1) for kv in sys.argv[2:]),indent=2)+"\n")' "$out" "$@"; }
r2lam(){ python -c "print(round(float('$1')*0.6,4))"; }        # pre-reg relation: R2 lambda = 0.6 * R1 lambda (2.5->1.5, 1->0.6)
glen(){ python -c "import sys;sys.path.insert(0,'$SPA_REPO/src');from spa.eval.score import _as_struct,_ca_array;print(len(_ca_array(_as_struct('$PREP/AF-$1-F1-model_v4_esmfold_v1.pdb'))))"; }

export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/nvidia/bin:${PATH}"; ldconfig 2>/dev/null || true
python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
[ "$STAGE" = "designability" ] && conda run -n spa-verify-of3 python -c "import triton; print('OF3 env: triton', triton.__version__)"

[ -d "$SPA_REPO/.git" ] || git clone --depth 1 --branch "${REPO_REF:-main}" "${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}" "$SPA_REPO"
pip install -e "$SPA_REPO" --no-deps -q
[ -d "$MPNN_REPO" ] || git clone --depth 1 https://github.com/dauparas/ProteinMPNN "$MPNN_REPO"

mkdir -p /workspace/weights "$PREP" "$OUT"
gcloud storage cp "$RFD3_CKPT_URI" /workspace/weights/rfd3_latest.ckpt
gcloud storage cp "$MG_CKPT_URI"   /workspace/weights/spa_multigran.pt
[ "$STAGE" = "designability" ] && gcloud storage cp "$OF3_CKPT_URI" /workspace/weights/of3-p2-155k.pt
gcloud storage cp "$PREP_URI/*" "$PREP/" 2>/dev/null || gcloud storage cp -r "$PREP_URI/." "$PREP/"

export RFD3_CKPT=/workspace/weights/rfd3_latest.ckpt
export OF3_CKPT=/workspace/weights/of3-p2-155k.pt
if [ "$STAGE" = "designability" ]; then export OF3_RUNNER_YAML="$SPA_REPO/configs/of3/of3_nokernel.yml"; fi
export PROTEINMPNN_REPO="$MPNN_REPO"
# ⭐ SHARED ESM3 prompt cache: embed each fold ONCE, not once per cell (dev plan/161; the per-cell
# ESM3 reload made the sweep ~12x too slow). probe_two_steer/_precompute_prompt reuses this across cells.
export SPA_PROMPT_CACHE=/workspace/prompt_cache; mkdir -p "$SPA_PROMPT_CACHE"
SPA_CKPT=/workspace/weights/spa_multigran.pt
R2_LEN=$(glen "$G2")
log "R2=$G2 |R2|=$R2_LEN ; sampler arm=$ARM (200/0.6/1.5) ; K=$K N=$NSEQ ; stage=$STAGE"

IFS=',' read -ra FO <<< "$FOLDS"; IFS=',' read -ra MO <<< "$MOTIFS"
IFS=',' read -ra LA <<< "$LAMBDAS"; IFS=',' read -ra R2LA <<< "${R2_LAMBDAS:-}"
OK=0; TOTAL=0

for f in "${FO[@]}"; do
  R1_LEN=$(glen "$f") || { log "  [skip] cannot read |$f|"; continue; }
  for ms in "${MO[@]}"; do
    mid="${ms%%:*}"; seg="${ms##*:}"
    CONTIG="${R1_LEN},${INNER_SPACER},${seg},${INNER_SPACER},${R2_LEN}"
    for s in $(tr ',' ' ' <<< "$SEEDS"); do
      for i in "${!LA[@]}"; do
        r1lam="${LA[$i]}"; r2lam="${R2LA[$i]:-$(r2lam "$r1lam")}"
        TOTAL=$((TOTAL+1))
        key="${f}_${mid}_${seg}_l${r1lam}_s${s}"
        po="$OUT/$STAGE/$key"; mkdir -p "$po"
        if [ "$STAGE" = "adherence" ]; then
          # First lambda also emits the free:free baseline; later lambdas emit g1:g2 only.
          CELLS="g1:g2"; [ "$i" -eq 0 ] && CELLS="free:free,g1:g2"
          if python "$SPA_REPO/scripts/eval/probe_two_steer.py" \
               --ckpt "$SPA_CKPT" --rfd3-ckpt "$RFD3_CKPT" --motif-source "$mid" --motif-seg "$seg" \
               --g1 "$f" --r1-len "$R1_LEN" --g2 "$G2" --r2-len "$R2_LEN" \
               --r1-lambda "$r1lam" --r2-lambda "$r2lam" --inner-spacer "$INNER_SPACER" --term-spacer "$TERM_SPACER" \
               --cells "$CELLS" --num-designs "$K" --seed "$s" \
               --deterministic --sampler-arm "$ARM" --pdb-dir "$PREP" --out-dir "$po" </dev/null; then
            manifest "$po/manifest.json" stage=adherence g1_soft_prompt="$f" r1_len="$R1_LEN" \
              g2_soft_prompt="$G2" r2_len="$R2_LEN" motif_source="$mid" motif_segment="$seg" \
              r1_lambda="$r1lam" r2_lambda="$r2lam" seed="$s" inner_spacer="$INNER_SPACER" \
              term_spacer="$TERM_SPACER" contig="$CONTIG" sampler_arm="$ARM" num_designs="$K" \
              cells="$CELLS" input_g1_pdb="$PREP_URI/AF-${f}-F1-model_v4_esmfold_v1.pdb" \
              input_g2_pdb="$PREP_URI/AF-${G2}-F1-model_v4_esmfold_v1.pdb" \
              input_motif_pdb="$PREP_URI/AF-${mid}-F1-model_v4_esmfold_v1.pdb" \
              rfd3_ckpt="$RFD3_CKPT_URI" spa_ckpt="$MG_CKPT_URI" gcs_cell_prefix="$RESULTS_URI/adherence/cells/$key"
            gcloud storage cp -r "$po" "$RESULTS_URI/adherence/cells/" 2>/dev/null || log "  [warn] push failed $key"
            OK=$((OK+1)); log "  [adh] $key OK"
          else log "  [adh] $key FAILED (continuing)"; fi
        else
          pdbs=$(ls "$po"/g1_g2_*.pdb 2>/dev/null)
          [ -n "$pdbs" ] || { ADH="$OUT/adherence/$key"; gcloud storage cp -r "$RESULTS_URI/adherence/cells/$key" "$ADH" 2>/dev/null && pdbs=$(ls "$ADH"/g1_g2_*.pdb 2>/dev/null); }
          [ -n "$pdbs" ] || { log "  [des] $key: no g1_g2 PDBs (run adherence first)"; continue; }
          if python "$SPA_REPO/scripts/eval/score_threeway_designability.py" \
               --pdbs $pdbs --contig "$CONTIG" --motif-source "$PREP/AF-${mid}-F1-model_v4_esmfold_v1.pdb" \
               --num-seqs "$NSEQ" --of3-batch-size "$OF3_BATCH_SIZE" --deterministic --out-dir "$po/desig" </dev/null; then
            manifest "$po/manifest.json" stage=designability g1_soft_prompt="$f" motif_source="$mid" \
              motif_segment="$seg" r1_lambda="$r1lam" seed="$s" contig="$CONTIG" num_seqs="$NSEQ" \
              of3_batch_size="$OF3_BATCH_SIZE" gcs_cell_prefix="$RESULTS_URI/designability/cells/$key"
            gcloud storage cp -r "$po" "$RESULTS_URI/designability/cells/" 2>/dev/null || log "  [warn] push failed $key"
            OK=$((OK+1)); log "  [des] $key OK"
          else log "  [des] $key SCORE FAILED (continuing)"; fi
        fi
      done
    done
  done
done

log "===== M1-BREADTH ($STAGE) DONE: $OK/$TOTAL cell(s) staged -> $RESULTS_URI ====="
[ "$OK" -gt 0 ] || { log "FATAL: 0/$TOTAL staged — every cell failed."; exit 1; }
