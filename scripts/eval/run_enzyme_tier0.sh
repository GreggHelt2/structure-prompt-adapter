#!/usr/bin/env bash
# Tier-0 ENZYME hard⊕soft driver — cutinase (1CEX) catalytic-triad TIP-atom motif ⊕ a foreign fold prompt.
# Spec: dev docs/plan/26 §4 (Tier-0) + §8. Runs the two compatibility arms through the §4 flywheel
# (config configs/eval/enzyme_tier0.yaml), coarse→fine per the A5000 plan:
#   STAGE=adherence     (default) — generate + score TM-to-G + tip-atom motif-RMSD, NO OF3 ($0, fast)
#   STAGE=designability          — + ProteinMPNN→OF3 (nokernel, A5000) on the chosen arm/λ
#
# Arms (both share the motif; only the SPA prompt G differs — dev 26 §4 parent-fold self-consistent):
#   compatible   : G = 1CEX itself (cutinase's own α/β-hydrolase fold)  → expect fold-steer, pin holds
#   incompatible : G = A0A7S3EB45  (held-out all-β; dev 05 C12)         → expect designability collapse
#
# Usage:  bash scripts/eval/run_enzyme_tier0.sh
#   env: CKPT=<multigran N×1536 ckpt>  STAGE=adherence|designability  ARM=both|compatible|incompatible
#        LAMBDAS=0,0.5,1,2  K=8  OUT=outputs/eval/enzyme_tier0  INPUTS=<dir for 1cex.pdb>
#        CDDB_DIR=<CDDB pdb dir>  INCOMPAT_ID=A0A7S3EB45  OF3_CKPT=<of3 ckpt>  OF3_YAML=<runner yaml>
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

CKPT="${CKPT:-checkpoints/spa_C_multigran_final.pt}"
STAGE="${STAGE:-adherence}"
ARM="${ARM:-both}"
LAMBDAS="${LAMBDAS:-0,0.5,1,2}"
K="${K:-8}"
OUT="${OUT:-outputs/eval/enzyme_tier0}"
INPUTS="${INPUTS:-$REPO/outputs/eval/enzyme_tier0/inputs}"
CDDB_DIR="${CDDB_DIR:-/home/user1/projects/spa/training_data/proteina-atomistica_data_vrelease/atomistica_data_release/pdb}"
INCOMPAT_ID="${INCOMPAT_ID:-A0A7S3EB45}"
INCOMPAT_PDB="${INCOMPAT_PDB:-$CDDB_DIR/AF-${INCOMPAT_ID}-F1-model_v4_esmfold_v1.pdb}"
ENV="${ENV:-spa-dev}"
EVAL_CFG="${EVAL_CFG:-enzyme_tier0}"   # enzyme_tier0 (indexed shape B) | enzyme_tier0_shapeA (unindexed shape A)

# --- stage the motif/compatible-prompt structure: cutinase 1CEX (public, ungated) --------------------
mkdir -p "$INPUTS"
CEX="$INPUTS/1cex.pdb"
if [ ! -s "$CEX" ]; then
  echo "[tier0] downloading 1CEX cutinase -> $CEX"
  curl -fsSL --max-time 60 https://files.rcsb.org/download/1CEX.pdb -o "$CEX"
fi
[ -s "$INCOMPAT_PDB" ] || { echo "[tier0] ERROR: incompatible-fold PDB not found: $INCOMPAT_PDB"; exit 1; }
echo "[tier0] motif/compatible G = $CEX ; incompatible G = $INCOMPAT_PDB ; ckpt = $CKPT ; STAGE=$STAGE"

# LAMBDAS as a Hydra list literal, e.g. [0.0,0.5,1.0,2.0]
LAM_LIST="[$(echo "$LAMBDAS" | sed 's/ //g')]"

# OF3 refolder overrides (only for the designability stage; nokernel on the A5000 — dev 23 §7)
OF3_OVR=()
if [ "$STAGE" = "designability" ]; then
  OF3_CKPT="${OF3_CKPT:-/home/user1/projects/spa/models/openfold3/of3-p2-155k.pt}"
  OF3_YAML="${OF3_YAML:-$REPO/configs/of3/of3_nokernel.yml}"
  OF3_ENV="${OF3_ENV:-spa-verify-of3}"
  OF3_OVR=(
    +eval.flywheel.refolder._target_=spa.eval.openfold3.OF3Refolder
    +eval.flywheel.refolder.ckpt_path="$OF3_CKPT"
    +eval.flywheel.refolder.runner_yaml="$OF3_YAML"
    +eval.flywheel.refolder.conda_env="$OF3_ENV"
  )
fi

run_arm () {   # $1 = arm label, $2 = prompt G pdb
  local arm="$1" gpdb="$2"          # NOTE: separate `local` — `set -u` expands all args before assigning,
  local odir="$OUT/${STAGE}_${arm}" # so ${arm} must be set on a prior line before it is referenced here.
  local of3_out=()                  # OF3Refolder needs out_dir; it is per-arm so it's set here, not in OF3_OVR.
  [ "${#OF3_OVR[@]}" -gt 0 ] && of3_out=("+eval.flywheel.refolder.out_dir=$odir/of3")
  echo "[tier0] === arm=$arm  G=$gpdb  λ=$LAM_LIST  K=$K  -> $odir ==="
  conda run -n "$ENV" python scripts/eval/run_flywheel.py "eval=$EVAL_CFG" \
    eval.ckpt="$CKPT" \
    eval.motif.source_pdb="$CEX" \
    eval.prompt_pdb="$gpdb" \
    eval.lambda_scale="$LAM_LIST" \
    eval.num_designs="$K" \
    eval.out_dir="$odir" \
    ${OF3_OVR[@]+"${OF3_OVR[@]}"} \
    ${of3_out[@]+"${of3_out[@]}"}
}

case "$ARM" in
  both)         run_arm compatible "$CEX"; run_arm incompatible "$INCOMPAT_PDB" ;;
  compatible)   run_arm compatible "$CEX" ;;
  incompatible) run_arm incompatible "$INCOMPAT_PDB" ;;
  *) echo "[tier0] ERROR: ARM must be both|compatible|incompatible"; exit 1 ;;
esac
echo "[tier0] done — results under $OUT/${STAGE}_*/flywheel_results.json"
