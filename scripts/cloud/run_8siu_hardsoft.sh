#!/usr/bin/env bash
# In-container 8SIU HARD⊕SOFT designability eval — a SINGLE-FOLD adaptation of run_b4_hardsoft.sh for the
# 368-res β-propeller 8SIU (candidate 3rd row of poster Panel 2). Runs the flywheel BASELINE (λ0) vs SPA
# (λ1) with a native coordinate-pinned 2-segment β-strand motif (HARD) AND the soft SPA fold prompt (SOFT)
# in the same design — the hard⊕soft headline, on 8SIU.
#
# WHY THIS IS SEPARATE FROM run_b4_hardsoft.sh (two reasons):
#   1) 8SIU was EXCLUDED from the b4 hard⊕soft sweep because its pre-staged ESM3 .pt didn't match the
#      cleaned chain length (N != ptN). THE FIX: build the SPA prompt at RUNTIME from the cleaned PDB via
#      eval.prompt_pdb (runtime ESM3 on the exact chain), NOT a cached .pt — so the prompt length N equals
#      the contig design length L by construction (see the contig ↓, Σ = 368 = 8SIU chain length). This is
#      the self_prompt (§4) hard⊕soft case: the SPA prompt IS 8SIU's own structure (N==L), motif rows
#      masked out of the prompt (non-overlap) → the per-residue identity projector, i.e. N×1536 (variant C) only.
#   2) 368 res > the A5000 OF3 ≤256-res ceiling, so designability MUST run on the H100 with OF3 NOKERNEL
#      (combined image, calibrated to 384 res; the triton kernels are avoided here — of3_nokernel.yml).
#
# Metrics: adherence (TM→8SIU + motif-RMSD) AND designability (OF3 refold scRMSD, best-of-N). Results
# (flywheel_results.json + the full structure tree for figures) staged to GCS. NB: NOT `set -e` — a single
# non-zero from a cleanup cp must not mask the real exit status; the flywheel result is gated explicitly.
set -uo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
PROMPT_PDB_URI="${PROMPT_PDB_URI:-$BUCKET/eval/b4/prep/8SIU.pdb}"    # cleaned 368-res chain; runtime ESM3 → [368,1536]
RESULTS_URI="${RESULTS_URI:-$BUCKET/eval/8siu_hardsoft}"
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
OF3_CKPT_URI="${OF3_CKPT_URI:-$BUCKET/weights/of3-p2-155k.pt}"
SPA_CKPT_URI="${SPA_CKPT_URI:-$BUCKET/checkpoints/spa-Nx1536-uncond/spa_C_final.pt}"  # Run-A uncond N×1536 (variant C)
SPA_REPO="${SPA_REPO:-/opt/spa}"
MPNN_REPO="${MPNN_REPO:-/opt/ProteinMPNN}"

ID="${ID:-8SIU}"
VARIANT="${VARIANT:-C_n_by_1536}"                                   # N×1536 — hard⊕soft non-overlap mask needs the per-residue identity projector
MOTIF_CONTIG="${MOTIF_CONTIG:-78,A79-85,168,A254-260,108}"          # 2-seg β-strand motif; 78+7+168+7+108 = 368 = 8SIU chain length (N==L)
K="${K:-8}"; NSEQ="${NSEQ:-16}"                                     # K designs/(cond,λ); N ProteinMPNN seqs/backbone (best-of-N designability)
LAM="${LAM:-1}"                                                     # SPA λ sweep = λ1 (operating point). baseline auto-runs at λ0 → λ∈{0,1}.
EVAL_SEED="${EVAL_SEED:-0}"                                         # RFD3 fixed seed (genuine 0 → reproducible + paired baseline-vs-SPA noise)
MPNN_SEED="${MPNN_SEED:-42}"                                        # ProteinMPNN FIXED seed (0 == RANDOM each run; results/05 E8). 42 = reproducible.
OUT=/workspace/8siuhs; PO="$OUT/$VARIANT/$ID"

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/nvidia/bin:${PATH}"; ldconfig 2>/dev/null || true
python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
conda run -n spa-verify-of3 python -c "import triton; print('OF3 env: triton', triton.__version__)"

[ -d "$SPA_REPO/.git" ] || git clone --depth 1 --branch "${REPO_REF:-main}" "${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}" "$SPA_REPO"
pip install -e "$SPA_REPO" --no-deps -q
[ -d "$MPNN_REPO" ] || git clone --depth 1 https://github.com/dauparas/ProteinMPNN "$MPNN_REPO"

mkdir -p /workspace/weights "$OUT" "$PO"
gcloud storage cp "$RFD3_CKPT_URI" /workspace/weights/rfd3_latest.ckpt
gcloud storage cp "$OF3_CKPT_URI"  /workspace/weights/of3-p2-155k.pt
gcloud storage cp "$SPA_CKPT_URI"  "/workspace/weights/spa_${VARIANT}.pt"
gcloud storage cp "$PROMPT_PDB_URI" "$OUT/$ID.pdb"
[ -f "$OUT/$ID.pdb" ] || { log "FATAL: prompt PDB $OUT/$ID.pdb not staged from $PROMPT_PDB_URI"; exit 1; }

log "8SIU HARD⊕SOFT: variant=$VARIANT K=$K N=$NSEQ λ=[$LAM] (baseline λ0 + SPA λ$LAM) motif=$MOTIF_CONTIG seeds rfd3=$EVAL_SEED mpnn=$MPNN_SEED"

python "$SPA_REPO/scripts/eval/run_flywheel.py" \
  variant="$VARIANT" hardware=cloud_h100 \
  'eval.conditions=[baseline,spa]' "eval.lambda_scale=[$LAM]" \
  eval.num_designs="$K" eval.proteinmpnn.num_seqs="$NSEQ" \
  eval.seed="$EVAL_SEED" eval.proteinmpnn.seed="$MPNN_SEED" \
  eval.length=368 \
  eval.ckpt="/workspace/weights/spa_${VARIANT}.pt" \
  eval.prompt_pdb="$OUT/$ID.pdb" \
  +eval.flywheel.prompt_struct="$OUT/$ID.pdb" \
  +eval.motif.source_pdb="$OUT/$ID.pdb" \
  +eval.motif.contig="'$MOTIF_CONTIG'" \
  +eval.motif.self_prompt=true \
  paths.rfd3_ckpt=/workspace/weights/rfd3_latest.ckpt \
  paths.proteinmpnn_repo="$MPNN_REPO" \
  +eval.flywheel.refolder._target_=spa.eval.openfold3.OF3Refolder \
  +eval.flywheel.refolder.ckpt_path=/workspace/weights/of3-p2-155k.pt \
  +eval.flywheel.refolder.runner_yaml="$SPA_REPO/configs/of3/of3_nokernel.yml" \
  +eval.flywheel.refolder.out_dir="$PO" \
  eval.out_dir="$PO" </dev/null \
  || { log "FATAL: 8SIU flywheel FAILED"; exit 1; }

log "8SIU flywheel OK -> staging results to $RESULTS_URI"
gcloud storage cp "$PO/flywheel_results.json" "$RESULTS_URI/flywheel_results.json" 2>/dev/null || log "  [warn] results json push failed"
# FULL cell tree (RFD3+SPA design PDBs + ProteinMPNN FASTAs + OF3 refold CIFs + results json) → GCS for figures.
gcloud storage cp -r "$PO" "$RESULTS_URI/cells/" 2>/dev/null || log "  [warn] structure tree push failed"
log "===== 8SIU HARD⊕SOFT DONE -> $RESULTS_URI ====="
