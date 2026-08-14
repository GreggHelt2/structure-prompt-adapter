#!/usr/bin/env bash
# In-container Option B eval (dev plan/69 SS4.13): scores the fine-tuned RFD3 host built by
# run_finetune_real.sh against A0A7S3EB45, at both:
#   row 3 (primary): fine-tuned host, UNCONDITIONAL generation (condition=baseline, no SPA prompt at
#     all) -- does fine-tuning alone shift RFD3's own prior toward the target fold?
#   row 4 (stretch, nearly free once row 3 runs): fine-tuned host + its own co-trained SPA adapter
#     (condition=spa, eval.ckpt=spa_C_final.pt) -- does fine-tuning help ON TOP of SPA too?
# Both conditions run in one flywheel call so ProteinMPNN/OF3 setup is paid once.
#
# eval.prompt_pdb is what makes adherence scoring work for the UNCONDITIONED baseline arm: flywheel.py
# scores every design (any condition) against whatever eval.prompt_pdb resolves to -- score_design()
# is condition-agnostic (src/spa/eval/score.py:591-617, src/spa/eval/flywheel.py:154-219, verified by
# direct read 2026-08-14). The plan doc's draft `+eval.tm_reference_pdb` guess was never a real gap:
# eval.prompt_pdb is an EXISTING declared key (configs/eval/default.yaml:25), not a new one.
#
# paths.rfd3_ckpt points at rfd3_C_finetuned_full.ckpt, NOT the raw net.state_dict() export. RFD3
# inference always dispatches to the EMA "shadow" net at eval() time (needed_repos/foundry's
# foundry/training/EMA.py:62-67), and the checkpoint loader expects the full Lightning-style container
# (ckpt["model"] with "model."/"shadow."-prefixed keys, foundry/trainers/fabric.py:781-785) -- a bare
# state dict pointed directly at paths.rfd3_ckpt would not load correctly. rfd3_C_finetuned_full.ckpt
# was built by taking rfd3_latest.ckpt and substituting only the "shadow.*" tensors with the fine-tuned
# host weights (extracted + SPA-unwrapped per this doc's own extraction rule), so it is loadable by the
# exact same code path that already works for every other eval in this project.
set -uo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
RFD3_FINETUNED_CKPT_URI="${RFD3_FINETUNED_CKPT_URI:-$BUCKET/checkpoints/finetune-A0A7S3EB45-20260814-045849/rfd3_C_finetuned_full_cloudgen.ckpt}"
SPA_ADAPTER_CKPT_URI="${SPA_ADAPTER_CKPT_URI:-$BUCKET/checkpoints/finetune-A0A7S3EB45-20260814-045849/spa_C_final.pt}"
OF3_CKPT_URI="${OF3_CKPT_URI:-$BUCKET/weights/of3-p2-155k.pt}"
TARGET_PDB_URI="${TARGET_PDB_URI:-$BUCKET/eval/finetune_neighborhood/A0A7S3EB45/target.pdb}"
LENGTH="${LENGTH:-125}"                # matches the target's own length (dev plan/69 SS4.13)
K="${K:-8}"; NSEQ="${NSEQ:-8}"; SEED="${SEED:-42}"
LAMBDAS="${LAMBDAS:-[1,2]}"            # row 4 only; matches the existing SPA lambda=1/2 comparison points
SPA_REPO="${SPA_REPO:-/opt/spa}"
MPNN_REPO="${MPNN_REPO:-/opt/ProteinMPNN}"
OUT="${OUT:-/workspace/finetune_eval_out}"
RESULTS_URI="${RESULTS_URI:-$BUCKET/eval/finetune_comparison/A0A7S3EB45}"

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
trap 'log "FINETUNE EVAL FAILED at line $LINENO"' ERR

# --- GPU driver-path fix (verbatim from run_smoke.sh) ---
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/nvidia/bin:${PATH}"
ldconfig 2>/dev/null || true
( command -v nvidia-smi >/dev/null && nvidia-smi -L ) || echo "  nvidia-smi: n/a"
log "GPU sanity, system python (SPA stack):"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('  GPU', torch.cuda.get_device_name(0), 'CUDA', torch.version.cuda)"
log "GPU sanity, OF3 conda env (triton):"
conda run -n spa-verify-of3 python -c "import torch, triton; print('  OF3 env: torch', torch.__version__, '| triton', triton.__version__, '| cuda', torch.cuda.is_available())"

# --- SPA repo + ProteinMPNN ---
[ -d "$SPA_REPO/.git" ] || git clone --depth 1 --branch "${REPO_REF:-main}" "${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}" "$SPA_REPO"
pip install -e "$SPA_REPO" --no-deps -q
[ -d "$MPNN_REPO" ] || git clone --depth 1 https://github.com/dauparas/ProteinMPNN "$MPNN_REPO"

# --- Fetch weights + the target PDB ---
mkdir -p /workspace/weights "$OUT"
log "fetching fine-tuned RFD3 checkpoint: $RFD3_FINETUNED_CKPT_URI"
gcloud storage cp "$RFD3_FINETUNED_CKPT_URI" /workspace/weights/rfd3_finetuned_full.ckpt
log "fetching co-trained SPA adapter: $SPA_ADAPTER_CKPT_URI"
gcloud storage cp "$SPA_ADAPTER_CKPT_URI" /workspace/weights/spa_C_final.pt
gcloud storage cp "$OF3_CKPT_URI" /workspace/weights/of3-p2-155k.pt
gcloud storage cp "$TARGET_PDB_URI" /workspace/target.pdb

log "config: length=$LENGTH K=$K N=$NSEQ seed=$SEED lambdas=$LAMBDAS"

# --- Row 3 (baseline, unconditional) + row 4 (spa, fine-tuned host's own adapter) in one flywheel call ---
python "$SPA_REPO/scripts/eval/run_flywheel.py" \
  variant=C_n_by_1536 hardware=cloud_h100 \
  "eval.conditions=[baseline,spa]" \
  eval.lambda_scale="$LAMBDAS" \
  eval.num_designs="$K" eval.length="$LENGTH" \
  eval.proteinmpnn.num_seqs="$NSEQ" eval.proteinmpnn.seed="$SEED" eval.seed="$SEED" \
  eval.prompt_pdb=/workspace/target.pdb \
  eval.ckpt=/workspace/weights/spa_C_final.pt \
  paths.rfd3_ckpt=/workspace/weights/rfd3_finetuned_full.ckpt \
  paths.proteinmpnn_repo="$MPNN_REPO" \
  +eval.flywheel.refolder._target_=spa.eval.openfold3.OF3Refolder \
  +eval.flywheel.refolder.ckpt_path=/workspace/weights/of3-p2-155k.pt \
  +eval.flywheel.refolder.runner_yaml="$SPA_REPO/configs/of3/of3_triton.yml" \
  +eval.flywheel.refolder.out_dir="$OUT" \
  eval.out_dir="$OUT" \
  2>&1 | tee "$OUT/flywheel.log"
STATUS=${PIPESTATUS[0]}

log "===== FLYWHEEL DONE (exit=$STATUS) ====="

# --- Report the headline numbers per condition and stage results to GCS ---
python - "$OUT/flywheel_results.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
scores = d.get("scores", [])
by_cond = {}
for s in scores:
    key = (s.get("condition"), s.get("lambda_scale"))
    by_cond.setdefault(key, []).append(s)
print(f"[finetune_eval] {len(scores)} design(s) scored total")
for key in sorted(by_cond, key=lambda k: (str(k[0]), k[1] or 0)):
    items = by_cond[key]
    tm = [s["tm_score"] for s in items if s.get("tm_score") is not None]
    desig = [s for s in items if s.get("designable")]
    print(f"[finetune_eval] condition={key[0]} lambda={key[1]}: n={len(items)} "
          f"tm_score={tm} designable={len(desig)}/{len(items)}")
print("[finetune_eval] per-condition summaries (incl. diversity_tm):")
for s in d.get("summaries", []):
    print(f"  condition={s.get('condition')} lambda={s.get('lambda_scale')} "
          f"diversity_tm={s.get('diversity_tm')} n_designs={s.get('n_designs')} "
          f"n_designable={s.get('n_designable')} success_rate={s.get('success_rate')}")
PY

gcloud storage cp "$OUT/flywheel_results.json" "$RESULTS_URI/A0A7S3EB45_finetune_eval.json" 2>/dev/null
gcloud storage cp "$OUT/flywheel.log" "$RESULTS_URI/A0A7S3EB45_finetune_eval.log" 2>/dev/null
log "===== FINETUNE EVAL DONE -> ${RESULTS_URI}/A0A7S3EB45_finetune_eval.json ====="
exit $STATUS
