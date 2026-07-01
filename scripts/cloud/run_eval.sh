#!/usr/bin/env bash
# In-container B1-full designability (hard⊕soft, K=8/N=8, λ=1): for each pinned prompt (b1_full_resolved.json,
# staged by scripts/eval/prep_b1_full.py) run the flywheel BASELINE vs SPA with the carved hard motif + the
# OF3-triton refold -> scRMSD / designable-rate / motif-survival. Per-prompt (one failure doesn't sink the
# run); each prompt's results are staged to GCS as it completes. Bootstraps like run_smoke.sh.
# NB: NOT `set -e` — a single prompt's failure must not abort the loop.
set -uo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
PREP_URI="${PREP_URI:-$BUCKET/eval/b1_full/prep}"           # prep artifacts (.pt/.pdb + b1_full_resolved.json)
RESULTS_URI="${RESULTS_URI:-$BUCKET/eval/b1_full/results}"  # per-prompt flywheel_results.json out
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
OF3_CKPT_URI="${OF3_CKPT_URI:-$BUCKET/weights/of3-p2-155k.pt}"
SPA_REPO="${SPA_REPO:-/opt/spa}"
MPNN_REPO="${MPNN_REPO:-/opt/ProteinMPNN}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-}"   # empty -> flywheel default (RFD3 engine default 200)
PREP=/workspace/prep
OUT=/workspace/b1_out

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

# --- GPU sanity (both stacks) ---
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/nvidia/bin:${PATH}"; ldconfig 2>/dev/null || true
python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
conda run -n spa-verify-of3 python -c "import torch, triton; print('OF3 env: triton', triton.__version__, '| cuda', torch.cuda.is_available())"

# --- SPA repo + ProteinMPNN ---
[ -d "$SPA_REPO/.git" ] || git clone --depth 1 --branch "${REPO_REF:-main}" "${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}" "$SPA_REPO"
pip install -e "$SPA_REPO" --no-deps -q
[ -d "$MPNN_REPO" ] || git clone --depth 1 https://github.com/dauparas/ProteinMPNN "$MPNN_REPO"

# --- Fetch weights + prep artifacts ---
mkdir -p /workspace/weights "$PREP" "$OUT"
gcloud storage cp "$RFD3_CKPT_URI" /workspace/weights/rfd3_latest.ckpt
gcloud storage cp "$OF3_CKPT_URI"  /workspace/weights/of3-p2-155k.pt
gcloud storage cp "$PREP_URI/*" "$PREP/"

MAN="$PREP/b1_full_resolved.json"
[ -f "$MAN" ] || { log "FATAL: $MAN not found (did prep_b1_full.py stage to $PREP_URI?)"; exit 1; }
SPA_CKPT_REL=$(python -c "import json;print(json.load(open('$MAN'))['spa_ckpt'])")
LAM=$(python -c "import json;print(json.load(open('$MAN'))['lambda_scale'])")
K=$(python -c "import json;print(json.load(open('$MAN'))['num_designs'])")
NSEQ=$(python -c "import json;print(json.load(open('$MAN'))['num_seqs'])")
gcloud storage cp "$BUCKET/checkpoints/$SPA_CKPT_REL" /workspace/weights/spa.pt
log "config: spa_ckpt=$SPA_CKPT_REL  lambda=$LAM  K=$K  N=$NSEQ  timesteps=${NUM_TIMESTEPS:-default}"

# --- Per-prompt loop (id<TAB>contig; TAB-delimited so the comma-bearing contig stays one field) ---
python -c "import json;[print(p['id']+chr(9)+p['contig']) for p in json.load(open('$MAN'))['prompts']]" > "$OUT/prompts.tsv"
TS_ARG=""; [ -n "$NUM_TIMESTEPS" ] && TS_ARG="eval.num_timesteps=$NUM_TIMESTEPS"
n=0; ok=0
while IFS=$'\t' read -r id contig; do
  n=$((n+1))
  po="$OUT/$id"
  log "[$n] prompt $id (motif contig $contig)"
  # NB: contig is single-quoted AT THE HYDRA LEVEL ('...') so its commas are a string, not a Hydra list.
  python "$SPA_REPO/scripts/eval/run_flywheel.py" \
    variant=C_n_by_1536 hardware=cloud_h100 \
    'eval.conditions=[baseline,spa]' "eval.lambda_scale=[$LAM]" \
    eval.num_designs="$K" eval.proteinmpnn.num_seqs="$NSEQ" $TS_ARG \
    eval.ckpt=/workspace/weights/spa.pt \
    eval.prompt_cache="$PREP/$id.pt" \
    +eval.motif.source_pdb="$PREP/$id.pdb" \
    "+eval.motif.contig='$contig'" \
    paths.rfd3_ckpt=/workspace/weights/rfd3_latest.ckpt \
    paths.proteinmpnn_repo="$MPNN_REPO" \
    +eval.flywheel.refolder._target_=spa.eval.openfold3.OF3Refolder \
    +eval.flywheel.refolder.ckpt_path=/workspace/weights/of3-p2-155k.pt \
    +eval.flywheel.refolder.runner_yaml="$SPA_REPO/configs/of3/of3_triton.yml" \
    +eval.flywheel.refolder.out_dir="$po" \
    eval.out_dir="$po" </dev/null \
    && { ok=$((ok+1)); gcloud storage cp "$po/flywheel_results.json" "$RESULTS_URI/$id.json" 2>/dev/null && log "[$n] $id OK -> staged $RESULTS_URI/$id.json"; } \
    || log "[$n] $id FAILED (continuing)"
done < "$OUT/prompts.tsv"
log "===== B1-FULL DONE: $ok/$n prompts succeeded -> $RESULTS_URI ====="
