#!/usr/bin/env bash
# In-container PIPELINE CALIBRATION (review item M6): generate VANILLA RFdiffusion3 unconditionally and
# score designability, so our absolute rates can finally be compared to RFD3's published 98%.
#
# WHY: our baseline designable rates run 0.62-0.875 while RFD3 reports 98% for unconditional generation,
# and ours is measured under a LOOSER criterion (CA-only at 2.0 A vs their N,CA,C at <=1.5 A). Three
# differences could explain the gap: sampler settings, refolding oracle (OpenFold3 MSA-free vs AF3), and
# length regime. Until this runs, NO absolute designability number in the paper is anchored to anything
# external (review B5). It is a prerequisite for the bioRxiv submission.
#
# TWO ARMS, same lengths and counts, so the sampler contribution is separable from the oracle's:
#   ARM=rfd3  -> RFD3's own published settings: 200 steps, gamma_0 0.6, step_scale 1.5
#   ARM=ours  -> what this project has been running: checkpoint defaults, 100 steps, gamma_0 0.8
# Neither arm uses SPA. `eval.conditions=[baseline]` is vanilla RFD3 (SPA's zero-init gate is a bit-exact
# identity, and no SPA checkpoint is loaded here at all).
#
# SCORING: this run scores with RFD3's criterion (N,CA,C at 1.5 A) because that is the comparable number.
# ALL structures are retained and staged, so the project's usual CA-at-2.0 criterion can be recomputed
# offline from the same refolds without re-running anything (dev scripts/analysis/).
#
# NB: NOT `set -e` -- one length's failure must not abort the sweep.
set -uo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
RESULTS_URI="${RESULTS_URI:-$BUCKET/eval/m6_calibration/results}"
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
OF3_CKPT_URI="${OF3_CKPT_URI:-$BUCKET/weights/of3-p2-155k.pt}"
SPA_REPO="${SPA_REPO:-/opt/spa}"
MPNN_REPO="${MPNN_REPO:-/opt/ProteinMPNN}"

# --- experiment knobs (defaults mirror RFD3's own unconditional benchmark) ---
ARM="${ARM:-rfd3}"                      # rfd3 | ours
LENGTHS="${LENGTHS:-100,150,250}"       # RFD3 supplementary: 32 backbones each at L in {100,150,250}
K="${K:-32}"                            # backbones per length
NSEQ="${NSEQ:-8}"                       # RFD3: 8 ProteinMPNN sequences per backbone
SEED="${SEED:-42}"                      # fixed, recorded, NON-ZERO (root CLAUDE.md reproducibility rule)
SCRMSD_ATOMS="${SCRMSD_ATOMS:-N,CA,C}"  # RFD3's designability atom set
SCRMSD_CUTOFF="${SCRMSD_CUTOFF:-1.5}"   # RFD3's threshold
KERNEL="${KERNEL:-nokernel}"            # triton cannot batch (bias batch-dim asserted to 1); dev 23 §7

case "$ARM" in
  rfd3) TS=200; GAMMA0=0.6; STEP_SCALE=1.5 ;;
  ours) TS=100; GAMMA0=0.8; STEP_SCALE=1.5 ;;
  *) echo "FATAL: ARM must be 'rfd3' or 'ours' (got '$ARM')"; exit 2 ;;
esac

OUT=/workspace/m6_out
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/nvidia/bin:${PATH}"; ldconfig 2>/dev/null || true
python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
conda run -n spa-verify-of3 python -c "import torch; print('OF3 env cuda:', torch.cuda.is_available())"

REPO_REF="${REPO_REF:-main}"; REPO_URL="${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}"
if [ -d "$SPA_REPO/.git" ]; then
  git -C "$SPA_REPO" fetch --depth 1 origin "$REPO_REF" && git -C "$SPA_REPO" reset --hard FETCH_HEAD
else
  git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$SPA_REPO"
fi
pip install -e "$SPA_REPO" --no-deps -q
log "SPA repo @ $(git -C "$SPA_REPO" rev-parse HEAD)"   # records which code ACTUALLY ran
[ -d "$MPNN_REPO" ] || git clone --depth 1 https://github.com/dauparas/ProteinMPNN "$MPNN_REPO"

mkdir -p /workspace/weights "$OUT"
gcloud storage cp "$RFD3_CKPT_URI" /workspace/weights/rfd3_latest.ckpt
gcloud storage cp "$OF3_CKPT_URI"  /workspace/weights/of3-p2-155k.pt

log "M6 arm=$ARM  lengths=$LENGTHS  K=$K  N=$NSEQ  seed=$SEED"
log "  sampler: num_timesteps=$TS gamma_0=$GAMMA0 step_scale=$STEP_SCALE"
log "  scoring: atoms=$SCRMSD_ATOMS cutoff=${SCRMSD_CUTOFF}A  (RFD3's criterion)"
log "  NOTE: generate.py asserts the overrides actually reached the live sampler and will ABORT if not."

n=0; ok=0
for L in ${LENGTHS//,/ }; do
  n=$((n+1))
  po="$OUT/${ARM}_L${L}"
  log "[$n] arm=$ARM length=$L (K=$K backbones)"
  python "$SPA_REPO/scripts/eval/run_flywheel.py" \
    variant=C_n_by_1536 hardware=cloud_h100 \
    'eval.conditions=[baseline]' \
    eval.num_designs="$K" eval.length="$L" eval.seed="$SEED" \
    eval.proteinmpnn.num_seqs="$NSEQ" \
    eval.num_timesteps="$TS" \
    eval.gamma_0="$GAMMA0" eval.step_scale="$STEP_SCALE" \
    "eval.score.scrmsd_atoms='$SCRMSD_ATOMS'" \
    eval.score.scrmsd_cutoff="$SCRMSD_CUTOFF" \
    paths.rfd3_ckpt=/workspace/weights/rfd3_latest.ckpt \
    paths.proteinmpnn_repo="$MPNN_REPO" \
    +eval.flywheel.refolder._target_=spa.eval.openfold3.OF3Refolder \
    +eval.flywheel.refolder.ckpt_path=/workspace/weights/of3-p2-155k.pt \
    +eval.flywheel.refolder.runner_yaml="$SPA_REPO/configs/of3/of3_${KERNEL}.yml" \
    +eval.flywheel.refolder.out_dir="$po" \
    eval.out_dir="$po" </dev/null \
    && { ok=$((ok+1))
         gcloud storage cp "$po/flywheel_results.json" "$RESULTS_URI/${ARM}_L${L}.json" 2>/dev/null
         # retain ALL structures so the CA-at-2.0 criterion can be recomputed offline without re-running
         gcloud storage cp -r "$po" "$RESULTS_URI/structures/${ARM}_L${L}/" 2>/dev/null
         log "[$n] arm=$ARM L=$L OK -> $RESULTS_URI/${ARM}_L${L}.json"; } \
    || log "[$n] arm=$ARM L=$L FAILED (continuing)"
done

log "===== M6 ($ARM) DONE: $ok/$n lengths succeeded -> $RESULTS_URI ====="
[ "$ok" -gt 0 ] || exit 1
