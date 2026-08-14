#!/usr/bin/env bash
# In-container FULL TEST SUITE run (dev plan/69 W1/Option-B safety net): the same pytest suite that
# runs locally CPU-only (241 tests) PLUS the GPU-gated tests skipped there (test_harness.py's 4,
# test_train_loop.py's test_train_resumes_end_to_end) that need real CUDA + the real RFD3 checkpoint.
# Used as a before/after baseline around the harness.py host-unfreeze change (dev plan/69 SS4.6-4.8) so
# a regression is caught by CI rather than asserted from a diff read. Mirrors run_smoke.sh's bootstrap.
set -uo pipefail   # NOT -e: a real test failure must be visible in the summary, not abort the script

PROJECT="${PROJECT:-spa-dev-499900}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
SPA_REPO="${SPA_REPO:-/opt/spa}"
OUT="${OUT:-/workspace/test_suite_out}"

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

# --- GPU driver-path fix (verbatim from run_smoke.sh) ---
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/nvidia/bin:${PATH}"
ldconfig 2>/dev/null || true
( command -v nvidia-smi >/dev/null && nvidia-smi -L ) || echo "  nvidia-smi: n/a"
log "GPU sanity:"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('  GPU', torch.cuda.get_device_name(0), 'CUDA', torch.version.cuda)"

# --- SPA repo (the Vertex BOOT command already clones $SPA_REPO; clone only if absent) ---
[ -d "$SPA_REPO/.git" ] || git clone --depth 1 --branch "${REPO_REF:-main}" "${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}" "$SPA_REPO"
pip install -e "$SPA_REPO" --no-deps -q
# Test-only extras NOT in the base combined image by default (dev/eval optional-dependency groups,
# pyproject.toml): pytest, tmtools. NOT --no-deps here (unlike the line above) since these genuinely
# need their own small dependency trees; neither pulls in torch or anything already pinned.
pip install -q "pytest>=7" tmtools 2>&1 | tail -5 || log "note: tmtools install failed (known: needs a C++ toolchain); its 1 test will fail, unrelated to this change"

# --- Fetch the RFD3 checkpoint, point the tests at it via SPA_RFD3_CKPT (both test files respect this
# env var; see tests/test_harness.py / tests/test_train_loop.py) ---
mkdir -p /workspace/weights "$OUT"
gcloud storage cp "$RFD3_CKPT_URI" /workspace/weights/rfd3_latest.ckpt
export SPA_RFD3_CKPT=/workspace/weights/rfd3_latest.ckpt

log "config: repo_ref=${REPO_REF:-main} ckpt=$SPA_RFD3_CKPT"

# --- Run the full suite. test_conditions.py needs atomworks, not installed in this pytest-only
# smoke; excluded exactly as it was for the local CPU baseline (dev plan/69), not a new gap. ---
cd "$SPA_REPO"
python -m pytest tests/ --ignore=tests/test_conditions.py -v 2>&1 | tee "$OUT/pytest_output.log"
STATUS=${PIPESTATUS[0]}

log "===== TEST SUITE DONE (pytest exit=$STATUS) ====="
tail -20 "$OUT/pytest_output.log"
gcloud storage cp "$OUT/pytest_output.log" "$BUCKET/eval/test_suite_runs/pytest_$(date -u +%Y%m%d-%H%M%S).log" 2>/dev/null \
  && log "log staged to gs://genomancer-spa-cache/eval/test_suite_runs/"
exit "$STATUS"
