#!/usr/bin/env bash
# In-container M10 INFERENCE OVERHEAD bench (dev plan/73): SPA's cold-vs-cached ESM3 cost, the
# wrapper's architectural tax, both sampler settings, and model residency across separate process
# launches. Runs scripts/eval/bench_m10_overhead.py inside the combined image.
#
# MODE=smoke  -> tiny pipeline-validation pass (K=1, one length, arms+load-breakdown only), the
#                 project's own "$2 smoke before a real run" convention (results/14 SS4). Run this
#                 FIRST -- this harness has never executed against a real GPU/environment before.
# MODE=full   -> the whole plan/73 spec (arms x both samplers, k-sweep, m-sweep, residency).
#
# Prompts: reuses the ALREADY-STAGED b1_full prep artifacts (gs://.../eval/b1_full/prep/<ID>.{pdb,pt}),
# no new prep needed -- they have both a structure file and a matching precomputed [N,1536] cache,
# exactly what this harness's cold/warm/residency arms need. The `contig` metadata in that manifest
# (hard-motif indices) is irrelevant here: this harness never sets eval.motif, so the .pdb files are
# used purely as plain structures.
set -uo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
RESULTS_URI="${RESULTS_URI:-$BUCKET/eval/m10_bench/results}"
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
SPA_CKPT_URI="${SPA_CKPT_URI:-$BUCKET/checkpoints/spa-Nx1536-uncond/spa_C_final.pt}"
PREP_URI="${PREP_URI:-$BUCKET/eval/b1_full/prep}"
SPA_REPO="${SPA_REPO:-/opt/spa}"

MODE="${MODE:-smoke}"                                   # smoke | full
VARIANT="${VARIANT:-C_n_by_1536}"
# 6 real, already-staged b1_full prompts spanning lengths 52-110. First is the "primary" prompt
# used for the arms/k-sweep/m-sweep phases; all are used for the residency phase (needs >=2 DIFFERENT
# prompts by construction).
PROMPT_IDS="${PROMPT_IDS:-A0A1X7NTP0,A0A6A0D1E8,A0A3P5VTL4,A0A090ME36,A0A7S1B8G4,A0A820JRM2}"
LENGTHS="${LENGTHS:-100,150,250}"                        # full mode only; smoke forces one length internally
K="${K:-8}"
LAM="${LAM:-1}"

OUT=/workspace/m10_out
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/nvidia/bin:${PATH}"; ldconfig 2>/dev/null || true
python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"

REPO_REF="${REPO_REF:-main}"; REPO_URL="${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}"
[ -d "$SPA_REPO/.git" ] || git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$SPA_REPO"
pip install -e "$SPA_REPO" --no-deps -q

mkdir -p /workspace/weights /workspace/prompts "$OUT"
log "staging weights + prompts from GCS"
gcloud storage cp "$RFD3_CKPT_URI" /workspace/weights/rfd3_latest.ckpt
gcloud storage cp "$SPA_CKPT_URI"  "/workspace/weights/spa_${VARIANT}.pt"

RESIDENCY_ARG=""
FIRST_PDB=""; FIRST_PT=""
IFS=',' read -ra IDS <<< "$PROMPT_IDS"
for id in "${IDS[@]}"; do
  gcloud storage cp "$PREP_URI/${id}.pdb" "/workspace/prompts/${id}.pdb"
  gcloud storage cp "$PREP_URI/${id}.pt"  "/workspace/prompts/${id}.pt"
  [ -f "/workspace/prompts/${id}.pdb" ] && [ -f "/workspace/prompts/${id}.pt" ] || {
    log "FATAL: prompt $id not staged from $PREP_URI"; exit 1; }
  if [ -z "$FIRST_PDB" ]; then FIRST_PDB="/workspace/prompts/${id}.pdb"; FIRST_PT="/workspace/prompts/${id}.pt"; fi
  RESIDENCY_ARG="${RESIDENCY_ARG}/workspace/prompts/${id}.pdb:/workspace/prompts/${id}.pt,"
done
RESIDENCY_ARG="${RESIDENCY_ARG%,}"                       # trim trailing comma

log "M10 bench: MODE=$MODE variant=$VARIANT K=$K lengths=$LENGTHS prompts=${#IDS[@]} primary=$FIRST_PDB"

COMMON_ARGS=(
  --variant "$VARIANT" --hardware cloud_h100
  --rfd3-ckpt /workspace/weights/rfd3_latest.ckpt
  --ckpt "/workspace/weights/spa_${VARIANT}.pt"
  --prompt-pdb "$FIRST_PDB" --prompt-cache "$FIRST_PT"
  --residency-prompts "$RESIDENCY_ARG"
  --k "$K" --lam "$LAM" --out-dir "$OUT" --json "$OUT/m10_results.json"
)

if [ "$MODE" = "smoke" ]; then
  python "$SPA_REPO/scripts/eval/bench_m10_overhead.py" "${COMMON_ARGS[@]}" --lengths "$LENGTHS" --smoke \
    || { log "FATAL: M10 smoke pass FAILED"; exit 1; }
else
  python "$SPA_REPO/scripts/eval/bench_m10_overhead.py" "${COMMON_ARGS[@]}" --lengths "$LENGTHS" --phase all \
    || { log "FATAL: M10 full bench FAILED"; exit 1; }
fi

log "M10 bench OK -> staging results to $RESULTS_URI"
gcloud storage cp "$OUT/m10_results.json" "$RESULTS_URI/m10_results_${MODE}.json" 2>/dev/null \
  || log "  [warn] results json push failed"
log "===== M10 BENCH DONE ($MODE) -> $RESULTS_URI ====="
