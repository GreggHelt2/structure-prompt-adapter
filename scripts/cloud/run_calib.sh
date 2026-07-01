#!/usr/bin/env bash
# In-container OF3 refold-timing CALIBRATION (drives scripts/eval/calib_of3.py). Bootstraps like
# run_smoke.sh (GPU libcuda-path fix, SPA clone, OF3 ckpt fetch) then times OF3 refolds at LENGTHS to pin
# the cloud-eval cost model. No RFD3/MPNN/RFD3-ckpt needed — it folds synthetic sequences directly.
set -euo pipefail

BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
OF3_CKPT_URI="${OF3_CKPT_URI:-$BUCKET/weights/of3-p2-155k.pt}"
SPA_REPO="${SPA_REPO:-/opt/spa}"
OUT="${OUT:-/workspace/calib_out}"
LENGTHS="${LENGTHS:-256,384}"

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
trap 'log "CALIB FAILED at line $LINENO"' ERR

export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/nvidia/bin:${PATH}"
ldconfig 2>/dev/null || true
( command -v nvidia-smi >/dev/null && nvidia-smi -L ) || echo "  nvidia-smi n/a"
conda run -n spa-verify-of3 python -c "import torch, triton; print('OF3 env: torch', torch.__version__, '| triton', triton.__version__, '| cuda', torch.cuda.is_available())"

[ -d "$SPA_REPO/.git" ] || git clone --depth 1 --branch "${REPO_REF:-main}" "${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}" "$SPA_REPO"
pip install -e "$SPA_REPO" --no-deps -q

mkdir -p /workspace/weights "$OUT"
gcloud storage cp "$OF3_CKPT_URI" /workspace/weights/of3-p2-155k.pt

log "OF3 refold-timing calibration at lengths: $LENGTHS"
python "$SPA_REPO/scripts/eval/calib_of3.py" \
  /workspace/weights/of3-p2-155k.pt "$SPA_REPO/configs/of3/of3_triton.yml" "$OUT" "$LENGTHS"
log "===== CALIB DONE ====="
cat "$OUT/calib.json"
