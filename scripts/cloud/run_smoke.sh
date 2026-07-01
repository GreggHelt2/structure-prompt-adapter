#!/usr/bin/env bash
# In-container SMOKE TEST for the COMBINED image (spa-cloud + OpenFold3/triton conda env). Runs the full
# validation flywheel END-TO-END on a tiny unconditional design: RFD3 generate -> ProteinMPNN -> OF3 refold
# (TRITON kernel, spa-verify-of3 env) -> scRMSD. Validates (a) both stacks coexist + run in one image and
# (b) OF3 triton works on Hopper (upstream marks it "to test"). Inference only — no cache/splits/NGC needed.
# Mirrors run_train.sh's proven pieces (the GPU libcuda-path fix, git-clone bootstrap).
set -euo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
OF3_CKPT_URI="${OF3_CKPT_URI:-$BUCKET/weights/of3-p2-155k.pt}"
SPA_REPO="${SPA_REPO:-/opt/spa}"
MPNN_REPO="${MPNN_REPO:-/opt/ProteinMPNN}"
OUT="${OUT:-/workspace/smoke_out}"

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
trap 'log "SMOKE FAILED at line $LINENO"' ERR

# --- GPU driver-path fix (verbatim from run_train.sh: without it the libcuda driver libs are not mounted
# into the container -> torch.cuda.is_available()=False) ---
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/nvidia/bin:${PATH}"
ldconfig 2>/dev/null || true
( command -v nvidia-smi >/dev/null && nvidia-smi -L ) || echo "  nvidia-smi: n/a"
log "GPU sanity — system python (SPA stack):"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('  GPU', torch.cuda.get_device_name(0), 'CUDA', torch.version.cuda)"
log "GPU sanity — OF3 conda env (triton):"
conda run -n spa-verify-of3 python -c "import torch, triton; print('  OF3 env: torch', torch.__version__, '| triton', triton.__version__, '| cuda', torch.cuda.is_available())"

# --- SPA repo + ProteinMPNN (a bundled script repo, not a pip dep; the eval flywheel shells it).
# The Vertex BOOT command already clones $SPA_REPO before invoking this script, so clone only if absent. ---
[ -d "$SPA_REPO/.git" ] || git clone --depth 1 --branch "${REPO_REF:-main}" "${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}" "$SPA_REPO"
pip install -e "$SPA_REPO" --no-deps -q
[ -d "$MPNN_REPO" ] || git clone --depth 1 https://github.com/dauparas/ProteinMPNN "$MPNN_REPO"

# --- Fetch the frozen-host + refold weights (pinned GCS artifacts) ---
mkdir -p /workspace/weights
gcloud storage cp "$RFD3_CKPT_URI" /workspace/weights/rfd3_latest.ckpt
gcloud storage cp "$OF3_CKPT_URI"  /workspace/weights/of3-p2-155k.pt

# --- Full flywheel: unconditional baseline (K=2, len 80, 50 steps) -> MPNN -> OF3 refold (triton) -> scRMSD.
# Baseline still loads+attaches SPA (zero-init identity), so both stacks are exercised; OF3Refolder defaults
# to conda_env=spa-verify-of3, so the refold runs in the OF3 env with the TRITON runner-yaml. ---
log "running smoke flywheel (RFD3 -> ProteinMPNN -> OF3 triton refold)"
python "$SPA_REPO/scripts/eval/run_flywheel.py" \
  variant=C_n_by_1536 hardware=cloud_h100 \
  'eval.conditions=[baseline]' eval.num_designs=2 eval.length=80 eval.num_timesteps=50 \
  eval.proteinmpnn.num_seqs=2 \
  paths.rfd3_ckpt=/workspace/weights/rfd3_latest.ckpt \
  paths.proteinmpnn_repo="$MPNN_REPO" \
  +eval.flywheel.refolder._target_=spa.eval.openfold3.OF3Refolder \
  +eval.flywheel.refolder.ckpt_path=/workspace/weights/of3-p2-155k.pt \
  +eval.flywheel.refolder.runner_yaml="$SPA_REPO/configs/of3/of3_triton.yml" \
  +eval.flywheel.refolder.out_dir="$OUT" \
  eval.out_dir="$OUT"

# --- Validate OF3 actually produced a scRMSD (the triton refold ran end-to-end) ---
python - "$OUT/flywheel_results.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
scr = [s["scrmsd"] for s in d.get("scores", []) if s.get("scrmsd") is not None]
print(f"[smoke] scored {len(d.get('scores', []))} design(s); {len(scr)} with scRMSD (OF3 ran)")
assert scr, "NO scRMSD -> OF3 triton refold did not run"
print(f"[smoke] OF3 TRITON OK on Hopper — e.g. scRMSD {scr[0]:.2f} A")
PY
log "===== SMOKE OK: combined image runs the full flywheel end-to-end incl OF3 triton on the H100 ====="
