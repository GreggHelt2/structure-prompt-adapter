#!/usr/bin/env bash
# In-container SAMPLER-INTERACTION CHECK (dev docs/plan/36): do the paper's DELTAS survive RFdiffusion3's
# own published sampler settings?
#
# WHY: M6 (dev docs/results/14) measured a MAIN EFFECT. Vanilla RFD3 scores 0.823 designable at RFD3's
# settings (200 steps, gamma_0 0.6) versus 0.719 at the settings inherited from the released checkpoint
# (100 steps, gamma_0 0.8), and the gap WIDENS with length: +0.06 at L=100/150 but +0.19 at L=250.
# Every SPA number in the paper was generated at the `ours` settings. The paper's claims are paired
# within-condition, so a UNIFORM offset cancels in the delta and every relative conclusion stands. Nothing
# has tested whether the offset IS uniform. This run measures that interaction at ~1/50th the cost of
# redoing the evaluation program.
#
# TWO ARMS, identical in every other respect (same prompts, K, N, seeds, checkpoint):
#   ARM=rfd3  -> RFD3's published settings: 200 steps, gamma_0 0.6, step_scale 1.5
#   ARM=ours  -> the checkpoint's inherited defaults: 100 steps, gamma_0 0.8, step_scale 1.5
# Only those two knobs differ; gamma_min (1.0) and step_scale/eta (1.5) are identical in both. Submit once
# per arm, as M6 did, so the two can run concurrently in different regions.
#
# CELLS: reuses the b1_full prep wholesale (prompt caches, PDBs, and pinned motif contigs already staged),
# so this run stages NOTHING new. Four prompts, chosen to cover the exposed claims and to bracket the
# length effect. All four are hard(+)soft by construction, because the b1_full path always carves a motif:
#   A0A7S3EB45  125  all-b   the all-beta cell behind section 5.2's 1/8 -> 4/8 (seed-42: 2/8 -> 4/8)
#   A0A522W419  150  a-b     the alpha/beta with-the-grain workhorse (7/8 -> 8/8, dTM +0.337)
#   A0A1X0IID6  247  a-b     long scaffold; exactly the top of section 6's "181 to 247 residues" band,
#                            and the length where M6 measured the +0.19 penalty
#   A0A090ME36   90  all-b   short all-beta; the section 5.5 headliner, brackets the length effect
#
# SCORING: scores under RFD3's criterion (N,CA,C at 1.5 A). ALL structures are retained and staged so the
# project's usual CA-at-2.0 criterion can be recomputed offline from the SAME refolds, giving both cells of
# the settings x criterion 2x2 without re-running (dev results/14 section 3).
#
# NB: NOT `set -e` -- one prompt's failure must not abort the sweep.
set -uo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
PREP_URI="${PREP_URI:-$BUCKET/eval/b1_full/prep}"
RESULTS_URI="${RESULTS_URI:-$BUCKET/eval/sampler_interaction/results}"
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
OF3_CKPT_URI="${OF3_CKPT_URI:-$BUCKET/weights/of3-p2-155k.pt}"
SPA_REPO="${SPA_REPO:-/opt/spa}"
MPNN_REPO="${MPNN_REPO:-/opt/ProteinMPNN}"

ARM="${ARM:-rfd3}"                       # rfd3 | ours
SUBSET_IDS="${SUBSET_IDS:-A0A7S3EB45,A0A522W419,A0A1X0IID6,A0A090ME36}"
CONDITIONS="${CONDITIONS:-[baseline,spa]}"
LAMBDAS="${LAMBDAS:-[1,2]}"
K="${K:-16}"                             # raised from 8 (dev plan/36 section 3b): Wilson width 0.57 -> 0.33
NSEQ="${NSEQ:-16}"
SEED="${SEED:-42}"                       # fixed, recorded, NON-ZERO (root CLAUDE.md reproducibility rule)
SCRMSD_ATOMS="${SCRMSD_ATOMS:-N,CA,C}"
SCRMSD_CUTOFF="${SCRMSD_CUTOFF:-1.5}"
KERNEL="${KERNEL:-nokernel}"             # triton cannot batch (bias batch-dim asserted to 1); dev 23 section 7

case "$ARM" in
  rfd3) TS=200; GAMMA0=0.6; STEP_SCALE=1.5 ;;
  ours) TS=100; GAMMA0=0.8; STEP_SCALE=1.5 ;;
  *) echo "FATAL: ARM must be 'rfd3' or 'ours' (got '$ARM')"; exit 2 ;;
esac

OUT=/workspace/si_out
PREP=/workspace/prep
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

mkdir -p /workspace/weights "$OUT" "$PREP"
gcloud storage cp "$RFD3_CKPT_URI" /workspace/weights/rfd3_latest.ckpt
gcloud storage cp "$OF3_CKPT_URI"  /workspace/weights/of3-p2-155k.pt
gcloud storage cp -r "$PREP_URI/*" "$PREP/"

MAN="$PREP/b1_full_resolved.json"
[ -f "$MAN" ] || { echo "FATAL: no manifest at $MAN"; exit 2; }
SPA_CKPT_REL=$(python -c "import json;print(json.load(open('$MAN'))['spa_ckpt'])")
gcloud storage cp "$BUCKET/checkpoints/$SPA_CKPT_REL" /workspace/weights/spa.pt

log "SAMPLER-INTERACTION arm=$ARM  K=$K  N=$NSEQ  seed=$SEED"
log "  sampler:  num_timesteps=$TS gamma_0=$GAMMA0 step_scale=$STEP_SCALE"
log "  cells:    $SUBSET_IDS"
log "  factors:  conditions=$CONDITIONS lambdas=$LAMBDAS"
log "  scoring:  atoms=$SCRMSD_ATOMS cutoff=${SCRMSD_CUTOFF}A (RFD3's criterion; CA@2.0 recomputed offline)"
log "  spa_ckpt: $SPA_CKPT_REL"
log "  NOTE: generate.py asserts the sampler overrides actually reached the live sampler; it ABORTS if not."

export SUBSET_IDS
python -c "
import json, os, sys
sub=set(x for x in os.environ.get('SUBSET_IDS','').split(',') if x)
ps=[p for p in json.load(open('$MAN'))['prompts'] if p['id'] in sub]
missing=sub-{p['id'] for p in ps}
if missing:
    sys.exit('FATAL: requested prompts not in the manifest: %s' % sorted(missing))
for p in ps: print(p['id']+chr(9)+p['contig'])
" > "$OUT/prompts.tsv" || exit 2
log "resolved $(wc -l < "$OUT/prompts.tsv") of $(echo "$SUBSET_IDS" | tr ',' '\n' | wc -l) requested prompts"

n=0; ok=0
while IFS=$'\t' read -r id contig; do
  [ -n "$id" ] || continue
  n=$((n+1))
  po="$OUT/${ARM}_${id}"
  log "[$n] arm=$ARM prompt=$id (motif contig $contig)"
  python "$SPA_REPO/scripts/eval/run_flywheel.py" \
    variant=C_n_by_1536 hardware=cloud_h100 \
    "eval.conditions=$CONDITIONS" "eval.lambda_scale=$LAMBDAS" \
    eval.num_designs="$K" eval.seed="$SEED" \
    eval.proteinmpnn.num_seqs="$NSEQ" \
    eval.num_timesteps="$TS" \
    eval.gamma_0="$GAMMA0" eval.step_scale="$STEP_SCALE" \
    "eval.score.scrmsd_atoms='$SCRMSD_ATOMS'" \
    eval.score.scrmsd_cutoff="$SCRMSD_CUTOFF" \
    eval.ckpt=/workspace/weights/spa.pt \
    eval.prompt_cache="$PREP/$id.pt" \
    +eval.motif.source_pdb="$PREP/$id.pdb" \
    "+eval.motif.contig='$contig'" \
    paths.rfd3_ckpt=/workspace/weights/rfd3_latest.ckpt \
    paths.proteinmpnn_repo="$MPNN_REPO" \
    +eval.flywheel.refolder._target_=spa.eval.openfold3.OF3Refolder \
    +eval.flywheel.refolder.ckpt_path=/workspace/weights/of3-p2-155k.pt \
    +eval.flywheel.refolder.runner_yaml="$SPA_REPO/configs/of3/of3_${KERNEL}.yml" \
    +eval.flywheel.refolder.out_dir="$po" \
    eval.out_dir="$po" </dev/null \
    && { ok=$((ok+1))
         gcloud storage cp "$po/flywheel_results.json" "$RESULTS_URI/${ARM}_${id}.json" 2>/dev/null
         # retain ALL structures: the CA-at-2.0 criterion is recomputed offline from these same refolds
         gcloud storage cp -r "$po" "$RESULTS_URI/structures/${ARM}_${id}/" 2>/dev/null
         log "[$n] arm=$ARM $id OK -> $RESULTS_URI/${ARM}_${id}.json"; } \
    || log "[$n] arm=$ARM $id FAILED (continuing)"
done < "$OUT/prompts.tsv"

log "===== SAMPLER-INTERACTION ($ARM) DONE: $ok/$n prompts succeeded -> $RESULTS_URI ====="
[ "$ok" -gt 0 ] || exit 1
