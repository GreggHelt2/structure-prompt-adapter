#!/usr/bin/env bash
# In-container OF3 BATCHED-inference benchmark on the H100 (dev 23): fold N same-length ProteinMPNN seqs at
# batch_size ∈ {1,8,16} via bench_of3_batch.py -> speedup + scRMSD-equivalence. Uses of3_triton.yml (Hopper)
# as the base runner-yaml; the bench injects data_module_args.batch_size. Mirrors run_scaffold_eval.sh setup.
set -uo pipefail
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
PREP_URI="${PREP_URI:-$BUCKET/eval/threeway/prep}"
OF3_CKPT_URI="${OF3_CKPT_URI:-$BUCKET/weights/of3-p2-155k.pt}"
SPA_REPO="${SPA_REPO:-/opt/spa}"; MPNN_REPO="${MPNN_REPO:-/opt/ProteinMPNN}"
DESIGN_ID="${DESIGN_ID:-A0A7C9GW19}"; N_SEQS="${N_SEQS:-16}"; BATCH_SIZES="${BATCH_SIZES:-1,8,16}"
PREP=/workspace/prep; OUT=/workspace/of3_batch_bench
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/nvidia/bin:${PATH}"; ldconfig 2>/dev/null || true
python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
conda run -n spa-verify-of3 python -c "import triton; print('OF3 env: triton', triton.__version__)"

REPO_REF="${REPO_REF:-main}"; REPO_URL="${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}"
if [ -d "$SPA_REPO/.git" ]; then
  # ALWAYS update (a warm/baked /opt/spa would otherwise silently run STALE code — the exact ambiguity
  # that made a "did it run the fix?" question un-answerable on the first re-bench).
  git -C "$SPA_REPO" fetch --depth 1 origin "$REPO_REF" && git -C "$SPA_REPO" reset --hard FETCH_HEAD
else
  git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$SPA_REPO"
fi
pip install -e "$SPA_REPO" --no-deps -q
log "SPA repo: $REPO_URL @ $REPO_REF  ->  SHA $(git -C "$SPA_REPO" rev-parse HEAD)"   # records which code ACTUALLY ran
[ -d "$MPNN_REPO" ] || git clone --depth 1 https://github.com/dauparas/ProteinMPNN "$MPNN_REPO"

mkdir -p /workspace/weights "$PREP" "$OUT"
gcloud storage cp "$OF3_CKPT_URI" /workspace/weights/of3-p2-155k.pt
gcloud storage cp "$PREP_URI/*" "$PREP/" 2>/dev/null || gcloud storage cp -r "$PREP_URI/." "$PREP/"

export OF3_CKPT=/workspace/weights/of3-p2-155k.pt
# Runner-yaml is overridable: default = triton (Hopper perf). If a batched run "no folds" traces to the
# triton kernels, set KERNEL=nokernel to use the A5000-VERIFIED stock-attention path (batched folds proven
# correct at 206-aa bs≤8; loses the per-fold triton speedup but keeps the batch amortization). dev 23 §7.
KERNEL="${KERNEL:-triton}"
export OF3_RUNNER_YAML="${OF3_RUNNER_YAML:-$SPA_REPO/configs/of3/of3_${KERNEL}.yml}"
export PROTEINMPNN_REPO="$MPNN_REPO"
export OF3_BATCH_SHIM="$SPA_REPO/scripts/eval/of3_batch_patch.py"   # bs>1 ragged-atom fix (dev 23 §7)
log "OF3 runner-yaml: $OF3_RUNNER_YAML"

log "OF3 batch bench: design=$DESIGN_ID N=$N_SEQS batch_sizes=$BATCH_SIZES"
python "$SPA_REPO/scripts/eval/bench_of3_batch.py" \
  --design "$PREP/AF-${DESIGN_ID}-F1-model_v4_esmfold_v1.pdb" \
  --n-seqs "$N_SEQS" --batch-sizes "$BATCH_SIZES" --out-dir "$OUT" </dev/null
rc=$?
# stage the runner-yamls + a small summary (the numbers are in the job logs)
gcloud storage cp "$OUT"/runner_bs*.yml "$BUCKET/eval/threeway/results/of3_batch_bench/" 2>/dev/null || true
# Surface + stage any SWALLOWED OF3 predict_step tracebacks — the "no folds" on a bs>1 run is OF3
# exiting 0 after logging the real error to predict_err_rank*.log (see dev 23 §7). Without this we are
# blind on the cloud. (bench_of3_batch.py + OF3Refolder also echo these to stdout now.)
errlogs=$(find "$OUT" -name "predict_err_rank*.log" 2>/dev/null)
if [ -n "$errlogs" ]; then
  log "⚠️  OF3 wrote predict_err logs (a swallowed bs>1 failure) — surfacing + staging:"
  echo "$errlogs" | while read -r f; do echo "===== $f ====="; sed -n "1,60p" "$f"; \
    gcloud storage cp "$f" "$BUCKET/eval/threeway/results/of3_batch_bench/logs/" 2>/dev/null || true; done
fi
log "OF3 batch bench done (exit $rc)"
[ "$rc" = "0" ] || exit 1
