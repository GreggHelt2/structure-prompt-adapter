#!/usr/bin/env bash
# LOCAL (A5000) B1-full hard⊕soft designability. The on-workstation twin of scripts/cloud/run_eval.sh.
#
# WHY THIS EXISTS. b1-full is the run behind the paper's headline hard⊕soft designability claim, and
# until now its ONLY execution path was an in-container cloud script (dev docs/plan/81 §5b). That made
# "run the redo locally at RFdiffusion3's own sampler" impossible for the single most important row.
# This driver is run_eval.sh minus the cloud: no `gcloud`, no `git clone`, no weight staging, because
# every input already lives on this machine.
#
#   RFD3 ckpt   models/rfdiffusion3/rfd3_latest.ckpt         (configs/paths/default.yaml)
#   OF3 ckpt    models/openfold3/of3-p2-155k.pt              (ditto)
#   SPA ckpt    <repo>/checkpoints/<rel>  where <rel> is the manifest's `spa_ckpt`. The local
#               checkpoints/ tree MIRRORS the GCS gs://.../checkpoints/ layout, so the manifest field
#               resolves unchanged on both sides.
#   ProteinMPNN needed_repos/ProteinMPNN                     (configs/paths/default.yaml)
#   prompts     built by scripts/eval/prep_b1_full.py from the local CDDB PDBs; auto-built below if
#               absent. ESM3 embeddings are byte-identical to the cloud's (dev memory
#               esm3-weights-byte-identical-local-cloud), so this is a rebuild, not a substitute.
#
# ⚠️ NOT `set -e`: one prompt failing must not abort the other 24.
#
# ⭐ REFOLDS ARE RETAINED, deliberately. `+eval.flywheel.refolder.out_dir` puts every OpenFold3 output
# under the run dir, and nothing deletes them. The scored JSON keeps only the per-design MINIMUM
# scRMSD, so re-scoring under RFdiffusion3's own N,Cα,C ≤ 1.5 Å criterion is possible ONLY from the
# retained structures (dev docs/results/22 §1b did exactly that, reproducing all 384 recorded values).
# Dev docs/plan/81 §3 makes this a standing requirement of any regeneration. Do not add a cleanup step.
#
# USAGE
#   conda activate spa-dev                       # see the PYTHON note below
#   ARM=rfd3 bash scripts/eval/run_b1_full_local.sh
#
#   ARM=ours|rfd3        sampler configuration (default ours = the checkpoint's 100 / γ₀ 0.8)
#   BAND=all|le256|gt256 which prompts (default all 25; see the length note below)
#   SUBSET_IDS=a,b,c     run only these prompt ids (smoke/validation)
#   K_OVERRIDE=2         override the manifest's K (8) for a cheap pass
#   NSEQ_OVERRIDE=2      override the manifest's ProteinMPNN N (8)
#   OUT_DIR=...          default outputs/_incoming/<date>__b1_full_local_<arm>/
#
# LENGTH NOTE. BAND defaults to `all`, including the 10 `gt256` prompts the manifest used to call
# "H100-only". ✅ **Verified end to end on this A5000 2026-08-31**: `A0A2V8GJC1` at L=374, the longest
# entry, generated 16/16 backbones (25.2 s/backbone at 200 steps) and completed all 128 refolds, 1 h 13
# min. It failed on a first attempt, but on a memory leak in our own pipeline rather than the card: the
# parent held 21.70 GiB of reserved-but-unallocated arena when the OF3 subprocess launched. That is
# fixed in `spa.eval.flywheel` (see `release_gpu_memory`), and the manifest note is corrected.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root (this file is scripts/eval/)
PROJECT_ROOT="${SPA_PROJECT_ROOT:-$HOME/projects/spa}"

MANIFEST="${MANIFEST:-$REPO/configs/eval/manifest_b1_full.yaml}"
PDB_DIR="${PDB_DIR:-$PROJECT_ROOT/training_data/proteina-atomistica_data_vrelease/atomistica_data_release/pdb}"
PREP="${PREP_DIR:-$PROJECT_ROOT/outputs/_prep/b1_full}"
OUT="${OUT_DIR:-$PROJECT_ROOT/outputs/_incoming/$(date +%F)__b1_full_local_${ARM}}"
BAND="${BAND:-all}"
OF3_ENV="${OF3_ENV:-spa-verify-of3}"
# Paired scoring criterion. CA@2.0 is this project's; N,CA,C@1.5 is RFdiffusion3's. configs/eval's own
# comment says to SET BOTH TOGETHER, same reasoning as the sampler arm, so this refuses a half-change.
SCRMSD_ATOMS="${SCRMSD_ATOMS:-CA}"
SCRMSD_CUTOFF="${SCRMSD_CUTOFF:-2.0}"
# ⚠️ `conda run` BUFFERS child output, which hides progress on a multi-hour job (dev memory
# dont-saturate-the-shared-workstation). So this driver calls python DIRECTLY and expects to be run
# from an activated spa-dev. Override with PYTHON=/path/to/python if you prefer.
PYTHON="${PYTHON:-python}"

log(){ echo "[$(date '+%H:%M:%S')] $*"; }
die(){ echo "FATAL: $*" >&2; exit 2; }

case "$SCRMSD_ATOMS:$SCRMSD_CUTOFF" in
  CA:2.0|N,CA,C:1.5) ;;
  *) die "SCRMSD_ATOMS='$SCRMSD_ATOMS' with SCRMSD_CUTOFF='$SCRMSD_CUTOFF' is not a published pairing.
       Use CA + 2.0 (this project) or 'N,CA,C' + 1.5 (RFdiffusion3). Changing one alone is not
       comparable to anything (configs/eval/default.yaml, score.scrmsd_atoms)." ;;
esac

# --- preflight ----------------------------------------------------------------------------------
"$PYTHON" - <<'PY' || die "the spa package is not importable by '$PYTHON'. Activate spa-dev, or set PYTHON=<path>."
import sys
import spa, torch                                     # noqa: F401
assert torch.cuda.is_available(), "no CUDA device visible"
print(f"[preflight] {torch.cuda.get_device_name(0)} | torch {torch.__version__} | {sys.executable}")
PY
[ -f "$PROJECT_ROOT/models/rfdiffusion3/rfd3_latest.ckpt" ] || die "no RFD3 checkpoint at $PROJECT_ROOT/models/rfdiffusion3/"
[ -f "$PROJECT_ROOT/models/openfold3/of3-p2-155k.pt" ]      || die "no OpenFold3 checkpoint at $PROJECT_ROOT/models/openfold3/"
[ -d "$PROJECT_ROOT/needed_repos/ProteinMPNN" ]             || die "no ProteinMPNN at $PROJECT_ROOT/needed_repos/ProteinMPNN"

# Sampler arm resolved AFTER the preflight, so an unimportable `spa` reports the friendlier
# "activate spa-dev" message above rather than the helper's import traceback.
ARM="${ARM:-ours}"
. "$REPO/scripts/_sampler_arm.sh"

# --- prep (idempotent; the ESM3 encode is the only slow part and it runs once) ---------------------
MAN="$PREP/b1_full_resolved.json"
if [ ! -f "$MAN" ]; then
  log "no resolved manifest at $MAN; building prep from $MANIFEST (loads ESM3 once)"
  [ -d "$PDB_DIR" ] || die "PDB_DIR does not exist: $PDB_DIR"
  mkdir -p "$PREP"
  "$PYTHON" "$REPO/scripts/eval/prep_b1_full.py" \
      --manifest "$MANIFEST" --pdb-dir "$PDB_DIR" --out-dir "$PREP" || die "prep_b1_full.py failed"
  [ -f "$MAN" ] || die "prep_b1_full.py did not write $MAN"
else
  log "reusing existing prep at $PREP"
fi

SPA_CKPT_REL=$("$PYTHON" -c "import json;print(json.load(open('$MAN'))['spa_ckpt'])")
LAM=$("$PYTHON" -c "import json;print(json.load(open('$MAN'))['lambda_scale'])")
K=$("$PYTHON" -c "import json;print(json.load(open('$MAN'))['num_designs'])")
NSEQ=$("$PYTHON" -c "import json;print(json.load(open('$MAN'))['num_seqs'])")
SPA_CKPT="$REPO/checkpoints/$SPA_CKPT_REL"
[ -f "$SPA_CKPT" ] || die "SPA checkpoint not found: $SPA_CKPT
       (the manifest names '$SPA_CKPT_REL'; the local checkpoints/ tree mirrors the GCS layout, so
        fetch it with: gcloud storage cp gs://genomancer-spa-cache/checkpoints/$SPA_CKPT_REL $SPA_CKPT)"

[ -n "${K_OVERRIDE:-}" ] && K="$K_OVERRIDE"
[ -n "${NSEQ_OVERRIDE:-}" ] && NSEQ="$NSEQ_OVERRIDE"
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
    print(p['id'] + chr(9) + p['contig'])
" > "$OUT/prompts.tsv" || die "prompt selection failed"
NP=$(wc -l < "$OUT/prompts.tsv")

log "===== B1-FULL LOCAL ====="
log "  sampler:   ARM=$ARM -> ${SAMPLER_ARGS[*]}"
log "  scoring:   scrmsd_atoms=$SCRMSD_ATOMS cutoff=${SCRMSD_CUTOFF}A  (refolds RETAINED for offline re-scoring)"
log "  spa_ckpt:  $SPA_CKPT_REL"
log "  config:    lambda=$LAM K=$K N=$NSEQ band=$BAND subset=${SUBSET_IDS:-ALL} -> $NP prompt(s)"
log "  out:       $OUT"

n=0; ok=0
while IFS=$'\t' read -r id contig; do
  [ -n "$id" ] || continue
  n=$((n+1)); po="$OUT/$id"
  log "[$n/$NP] $id (motif contig $contig)"
  # The contig is single-quoted AT THE HYDRA LEVEL so its commas stay a string, not a Hydra list.
  "$PYTHON" "$REPO/scripts/eval/run_flywheel.py" \
    variant=C_n_by_1536 hardware=local_a5000 \
    'eval.conditions=[baseline,spa]' "eval.lambda_scale=[$LAM]" \
    eval.num_designs="$K" eval.proteinmpnn.num_seqs="$NSEQ" \
    "${SAMPLER_ARGS[@]}" \
    "eval.score.scrmsd_atoms='$SCRMSD_ATOMS'" \
    eval.score.scrmsd_cutoff="$SCRMSD_CUTOFF" \
    eval.ckpt="$SPA_CKPT" \
    eval.prompt_cache="$PREP/$id.pt" \
    +eval.motif.source_pdb="$PREP/$id.pdb" \
    "+eval.motif.contig='$contig'" \
    +eval.flywheel.prompt_struct="$PREP/$id.pdb" \
    +eval.flywheel.refolder._target_=spa.eval.openfold3.OF3Refolder \
    +eval.flywheel.refolder.ckpt_path="$PROJECT_ROOT/models/openfold3/of3-p2-155k.pt" \
    +eval.flywheel.refolder.runner_yaml="$REPO/configs/of3/of3_nokernel.yml" \
    +eval.flywheel.refolder.conda_env="$OF3_ENV" \
    +eval.flywheel.refolder.out_dir="$po" \
    eval.out_dir="$po" </dev/null \
    && { ok=$((ok+1)); log "[$n/$NP] $id OK -> $po/flywheel_results.json"; } \
    || log "[$n/$NP] $id FAILED (continuing)"
done < "$OUT/prompts.tsv"

log "===== B1-FULL LOCAL DONE: $ok/$n prompts succeeded -> $OUT ====="
log "Aggregate:  $PYTHON $REPO/scripts/eval/aggregate_b1_full.py --results-dir $OUT --manifest $MAN"
[ "$ok" -eq "$n" ] || exit 1
