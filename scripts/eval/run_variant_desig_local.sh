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
      eval.num_designs="$K" eval.proteinmpnn.num_seqs="$NSEQ" \
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
