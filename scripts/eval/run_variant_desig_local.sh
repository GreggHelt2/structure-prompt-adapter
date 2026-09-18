#!/usr/bin/env bash
# LOCAL (A5000) SOFT-only designability sweep over SPA variants and/or λ. The on-workstation twin of
# scripts/cloud/run_variant_desig.sh, which is the driver behind TWO rows of dev docs/plan/81 §5:
#
#   row 2.3  variant designability   VARIANTS=all three, LAM=1        (paper §5.4)
#   row 2.1  17-fold λ sweep         VARIANTS=C only,   LAM=0.5,1,2   (paper §5.2, Fig 5)
#
# Both are SOFT-ONLY: no hard motif is carved into the design, so the pooled 1×1536 and 1×32 front-ends
# have no non-overlap mask to honour (dev docs/plan/15 §3; that is why the variant comparison is
# deliberately confined to the unconditional regime). Adherence comes ~free via `prompt_struct`.
#
# ⚠️ NOT `set -e`: one (variant, prompt) failing must not abort the rest of the grid.
#
# ⭐ REFOLDS ARE RETAINED. Same requirement and same reason as run_b1_full_local.sh: the scored JSON
# keeps only the per-design MINIMUM scRMSD, so re-scoring under RFdiffusion3's N,Cα,C ≤ 1.5 Å criterion
# is possible only from the structures on disk (dev docs/results/22 §1b, docs/plan/81 §3).
#
# USAGE
#   conda activate spa-dev
#   # row 2.3, variant designability at RFdiffusion3's sampler:
#   ARM=rfd3 bash scripts/eval/run_variant_desig_local.sh
#   # row 2.1, the 17-fold λ sweep (its own prompt set, so its own prep dir):
#   ARM=rfd3 VARIANTS='C_n_by_1536:spa-Nx1536-uncond/spa_C_final.pt' LAM=0.5,1,2 \
#     MANIFEST=configs/eval/manifest_lambda_sweep.yaml PREP_DIR=$SPA_OUTPUTS/_prep/lambda_sweep \
#     RUN_NAME=lambda_sweep_local bash scripts/eval/run_variant_desig_local.sh
#
#   ARM=ours|rfd3   sampler configuration (default ours)
#   VARIANTS        space-separated `variant:ckpt-relpath` entries; ckpt-relpath resolves under
#                   <repo>/checkpoints/, which mirrors the GCS checkpoints/ layout
#   LAM             λ, or a comma list for a sweep (default 1). λ=0 is the baseline arm and is added
#                   automatically by `eval.conditions=[baseline,spa]`; do not list it here.
#   BAND            all | le256 | gt256 (default le256 = the curated-15, which is what §5.4 reports)
#   K, NSEQ         designs per condition and ProteinMPNN sequences per design (default 4 / 4)
#   SUBSET_IDS      comma list of prompt ids, for a smoke pass
#   SEED            RFD3 sampler seed (default 0). Change it ONLY to extend an existing run with
#                   fresh draws; see the note beside the variable.
#   SEEDS           comma list of seeds, e.g. 0,1,2,3. Sets eval.num_designs=1 and passes the list to
#                   eval.seeds, which is the K=1 deterministic convention (dev plan/100 §9). Overrides K.
#   DETERMINISTIC   true -> eval.deterministic=true. Unset by default, so existing runs are unchanged.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root (this file is scripts/eval/)
PROJECT_ROOT="${SPA_PROJECT_ROOT:-$HOME/projects/spa}"

MANIFEST="${MANIFEST:-$REPO/configs/eval/manifest_b1_full.yaml}"
PDB_DIR="${PDB_DIR:-$PROJECT_ROOT/training_data/proteina-atomistica_data_vrelease/atomistica_data_release/pdb}"
PREP="${PREP_DIR:-$PROJECT_ROOT/outputs/_prep/b1_full}"
RUN_NAME="${RUN_NAME:-variant_desig_local}"
OUT="${OUT_DIR:-$PROJECT_ROOT/outputs/_incoming/$(date +%F)__${RUN_NAME}_${ARM}}"
# C is primary; B and A are the pooled variants. Same three checkpoints the cloud driver names.
VARIANTS="${VARIANTS:-C_n_by_1536:spa-Nx1536-uncond/spa_C_final.pt B_1_by_1536:spa-1x1536-uncond/spa_B_final.pt A_1_by_32:spa-1x32-uncond/spa_A_final.pt}"
K="${K:-4}"; NSEQ="${NSEQ:-4}"; LAM="${LAM:-1}"
# RFD3 sampler seed. Default 0 matches configs/eval/default.yaml, so behaviour is unchanged when
# unset. It is exposed so a run can be EXTENDED: `eval.seed` genuinely fixes the initial noise
# (dev docs/plan/RFD3_irreproducibility.md), so a second run at the SAME seed redraws the SAME
# backbones and pooling it would report 2K while carrying the information of K. A second run at a
# DIFFERENT seed draws K new ones and pools honestly to 2K. Both arms of one run must share a seed,
# because the paired baseline-vs-SPA comparison relies on them sharing initial noise.
SEED="${SEED:-0}"
# ⭐ OPTIONAL, added 2026-09-16 for dev plan/106 row 25. Both default to UNSET and append nothing when
# unset, so every existing invocation of this harness stays byte-identical.
# ⛔ WHY SEEDS EXISTS. K is the RFD3 DIFFUSION BATCH (eval.num_designs), and batch size sits inside the
# REPRODUCIBILITY IDENTITY: D is the batch dimension of every matmul, so a different K selects
# different cuBLAS kernels, and design *i* at K=4 differs from the same design at K=8 by up to 2.433 A.
# A run at one K is therefore not comparable to a run at another. The deterministic convention is K=1
# with one seed per draw, and generate.py already implements the seed list natively (_normalize_seeds,
# :889-907, seed loop :1116-1165) with FOUR drivers passing it. This harness simply never exposed it.
# ⇒ Setting SEEDS pins eval.num_designs=1 and hands the list through, so design identity no longer
# depends on how many designs were asked for, at ONE model load per cell because the loop lives inside
# generate.py rather than out here.
#
# ⛔ CORRECTION 2026-09-16, same day this comment was first committed. It originally read "K>1 forfeits
# determinism: its rows share one initial-noise draw", and said SEEDS made draws "independent rather
# than correlated batch rows". That mechanism is WRONG: the shared-noise hypothesis was tested against
# torch.randn(8,100,3)[0] at a fixed seed and rejected, and a K=25 cell measured 25 of 25 DISTINCT
# structures. Every batch row gets its own noise. Only the reasoning changed; K=1 still stands, and
# nothing about this script's behaviour changed with the correction.
# ⚠️ NOT to be confused with the true claim just above about SEED: two ARMS at the same seed do share
# initial noise, which is what makes the paired baseline-vs-SPA comparison valid.
SEEDS="${SEEDS:-}"
DETERMINISTIC="${DETERMINISTIC:-}"
# ⛔ OF3 REFOLD DETERMINISM, DEFAULT ON since 2026-09-18. This variable exists because its ABSENCE
# caused queue row 25 to generate deterministically and then refold on STOCK OpenFold3, which
# results/63 §2 then labelled "contract v2". DETERMINISTIC above is the RFdiffusion3 flag ALONE and
# never reached the refolder. Set REFOLD_DETERMINISTIC=false only for a deliberate non-deterministic
# refold, and say so in the run's plan/106 row.
REFOLD_DETERMINISTIC="${REFOLD_DETERMINISTIC:-true}"
BAND="${BAND:-le256}"
OF3_ENV="${OF3_ENV:-spa-verify-of3}"
SCRMSD_ATOMS="${SCRMSD_ATOMS:-CA}"
SCRMSD_CUTOFF="${SCRMSD_CUTOFF:-2.0}"
# `conda run` buffers child output and hides progress on a multi-hour job (dev memory
# dont-saturate-the-shared-workstation), so call python directly from an activated spa-dev.
PYTHON="${PYTHON:-python}"

log(){ echo "[$(date '+%H:%M:%S')] $*"; }
die(){ echo "FATAL: $*" >&2; exit 2; }

case "$SCRMSD_ATOMS:$SCRMSD_CUTOFF" in
  CA:2.0|N,CA,C:1.5) ;;
  *) die "SCRMSD_ATOMS='$SCRMSD_ATOMS' with SCRMSD_CUTOFF='$SCRMSD_CUTOFF' is not a published pairing.
       Use CA + 2.0 (this project) or 'N,CA,C' + 1.5 (RFdiffusion3)." ;;
esac

"$PYTHON" - <<'PY' || die "the spa package is not importable by '$PYTHON'. Activate spa-dev, or set PYTHON=<path>."
import sys
import spa, torch                                     # noqa: F401
assert torch.cuda.is_available(), "no CUDA device visible"
print(f"[preflight] {torch.cuda.get_device_name(0)} | torch {torch.__version__} | {sys.executable}")
PY
[ -f "$PROJECT_ROOT/models/rfdiffusion3/rfd3_latest.ckpt" ] || die "no RFD3 checkpoint at $PROJECT_ROOT/models/rfdiffusion3/"
[ -f "$PROJECT_ROOT/models/openfold3/of3-p2-155k.pt" ]      || die "no OpenFold3 checkpoint at $PROJECT_ROOT/models/openfold3/"

# Sampler arm resolved AFTER the preflight, so an unimportable `spa` reports the friendlier
# "activate spa-dev" message above rather than the helper's import traceback.
ARM="${ARM:-ours}"
. "$REPO/scripts/_sampler_arm.sh"

MAN="$PREP/b1_full_resolved.json"
if [ ! -f "$MAN" ]; then
  log "no resolved manifest at $MAN; building prep from $MANIFEST (loads ESM3 once)"
  [ -d "$PDB_DIR" ] || die "PDB_DIR does not exist: $PDB_DIR"
  mkdir -p "$PREP"
  "$PYTHON" "$REPO/scripts/eval/prep_b1_full.py" \
      --manifest "$MANIFEST" --pdb-dir "$PDB_DIR" --out-dir "$PREP" || die "prep_b1_full.py failed"
else
  log "reusing existing prep at $PREP"
fi
mkdir -p "$OUT"

export BAND SUBSET_IDS="${SUBSET_IDS:-}"
"$PYTHON" -c "
import json, os, sys
band = os.environ['BAND']
sub  = set(x for x in os.environ.get('SUBSET_IDS','').split(',') if x)
ps   = json.load(open('$MAN'))['prompts']
sel  = [p for p in ps if (band == 'all' or p['band'] == band) and (not sub or p['id'] in sub)]
missing = sub - {p['id'] for p in ps}
if missing:
    sys.exit('FATAL: requested prompt ids not in the manifest: %s' % sorted(missing))
if not sel:
    sys.exit('FATAL: no prompts selected (band=%s subset=%s)' % (band, sorted(sub) or 'ALL'))
for p in sel:
    print(p['id'] + chr(9) + str(p['len']))
" > "$OUT/prompts.tsv" || die "prompt selection failed"
NP=$(wc -l < "$OUT/prompts.tsv")
NV=$(echo "$VARIANTS" | wc -w)

log "===== VARIANT / λ SOFT-DESIGNABILITY, LOCAL ====="
log "  sampler:  ARM=$ARM -> ${SAMPLER_ARGS[*]}"
log "  scoring:  scrmsd_atoms=$SCRMSD_ATOMS cutoff=${SCRMSD_CUTOFF}A  (refolds RETAINED)"
log "  grid:     $NV variant(s) x $NP prompt(s) (band=$BAND) x lambda=[$LAM], K=$K N=$NSEQ"
log "  prep:     $PREP"
log "  out:      $OUT"

# Draw arguments, assembled once. Default path is exactly what it always was.
DRAW_ARGS=(eval.num_designs="$K" eval.seed="$SEED")
if [ -n "$SEEDS" ]; then
  DRAW_ARGS=(eval.num_designs=1 "eval.seeds=[$SEEDS]")
  log "  draws:    K=1 x seeds [$SEEDS]  (deterministic convention; NOT a K=$(echo "$SEEDS" | tr ',' '\n' | wc -l) batch)"
else
  log "  draws:    K=$K at seed $SEED  (diffusion BATCH; K>1 is not deterministic)"
fi
[ -n "$DETERMINISTIC" ] && DRAW_ARGS+=(eval.deterministic="$DETERMINISTIC") \
  && log "  determinism: eval.deterministic=$DETERMINISTIC"

TOTAL=$(( NV * NP )); done_n=0; ok=0
for entry in $VARIANTS; do
  vname="${entry%%:*}"; ckpt_rel="${entry#*:}"
  ckpt="$REPO/checkpoints/$ckpt_rel"
  if [ ! -f "$ckpt" ]; then
    log "SKIPPING variant $vname: no checkpoint at $ckpt"
    log "  (fetch: gcloud storage cp gs://genomancer-spa-cache/checkpoints/$ckpt_rel $ckpt)"
    done_n=$(( done_n + NP )); continue
  fi
  log "=== variant $vname (ckpt $ckpt_rel) ==="
  while IFS=$'\t' read -r id len; do
    [ -n "$id" ] || continue
    done_n=$((done_n+1)); po="$OUT/$vname/$id"
    log "  [$done_n/$TOTAL] $vname / $id (len $len)"
    "$PYTHON" "$REPO/scripts/eval/run_flywheel.py" \
      variant="$vname" hardware=local_a5000 \
      'eval.conditions=[baseline,spa]' "eval.lambda_scale=[$LAM]" \
      "${DRAW_ARGS[@]}" eval.proteinmpnn.num_seqs="$NSEQ" \
      "${SAMPLER_ARGS[@]}" \
      "eval.score.scrmsd_atoms='$SCRMSD_ATOMS'" \
      eval.score.scrmsd_cutoff="$SCRMSD_CUTOFF" \
      eval.length="$len" \
      eval.ckpt="$ckpt" \
      eval.prompt_cache="$PREP/$id.pt" \
      +eval.flywheel.prompt_struct="$PREP/$id.pdb" \
      +eval.flywheel.refolder._target_=spa.eval.openfold3.OF3Refolder \
      +eval.flywheel.refolder.ckpt_path="$PROJECT_ROOT/models/openfold3/of3-p2-155k.pt" \
      +eval.flywheel.refolder.runner_yaml="$REPO/configs/of3/of3_nokernel.yml" \
      +eval.flywheel.refolder.conda_env="$OF3_ENV" \
      +eval.flywheel.refolder.deterministic="$REFOLD_DETERMINISTIC" \
      +eval.flywheel.refolder.out_dir="$po" \
      eval.out_dir="$po" </dev/null \
      && { ok=$((ok+1)); log "  [$done_n/$TOTAL] $vname / $id OK"; } \
      || log "  [$done_n/$TOTAL] $vname / $id FAILED (continuing)"
  done < "$OUT/prompts.tsv"
done

log "===== DONE: $ok/$TOTAL cells succeeded -> $OUT ====="
log "Aggregate (variants): $PYTHON $REPO/scripts/eval/aggregate_variant_desig.py --results-dir $OUT"
log "Aggregate (lambda):   $PYTHON $REPO/scripts/eval/aggregate_lambda_desig.py --results-dir $OUT --manifest $MAN"
[ "$ok" -eq "$TOTAL" ] || exit 1
