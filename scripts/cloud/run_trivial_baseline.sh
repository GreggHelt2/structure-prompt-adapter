#!/usr/bin/env bash
# In-container TRIVIAL DOMAIN BASELINE (dev plan/69 W1 closure, comparator 4): pin the ENTIRE target fold
# as one RFD3-native motif (not through SPA) and run RFD3 -> ProteinMPNN -> OF3 refold -> score. No SPA
# checkpoint is loaded or needed (eval.conditions=[baseline] only). Demonstrates that coordinate-copying
# through the motif channel trivially reaches TM~1 with near-zero design diversity, which is why it is a
# different problem from SPA's task rather than a competing solution to it. Mirrors run_smoke.sh's
# bootstrap (GPU driver-path fix, git-clone-if-absent, weight fetch); the only new piece is the
# full-chain motif override in the run_flywheel.py call.
set -euo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
OF3_CKPT_URI="${OF3_CKPT_URI:-$BUCKET/weights/of3-p2-155k.pt}"
# Reuses the PDB already staged for the B1-full run (dev plan/69 §3.3: same fold, so the new baseline
# row sits beside a number the paper already prints). No new prep/staging step needed.
SOURCE_PDB_URI="${SOURCE_PDB_URI:-$BUCKET/eval/b1_full/prep/A0A7S3EB45.pdb}"
CONTIG="${CONTIG:-A1-125}"            # whole-chain fixed motif; verify against SOURCE_PDB's actual chain/length if swapping folds
K="${K:-8}"                            # RFD3 backbone samples (dev plan/69 §3.3: project's standard K)
NSEQ="${NSEQ:-8}"                      # ProteinMPNN sequences per backbone
SEED="${SEED:-42}"                     # project's standard seed (root CLAUDE.md reproducibility rule)
SPA_REPO="${SPA_REPO:-/opt/spa}"
MPNN_REPO="${MPNN_REPO:-/opt/ProteinMPNN}"
OUT="${OUT:-/workspace/trivial_baseline_out}"
RESULTS_URI="${RESULTS_URI:-$BUCKET/eval/trivial_domain_baseline/results}"

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
trap 'log "TRIVIAL BASELINE FAILED at line $LINENO"' ERR

# --- GPU driver-path fix (verbatim from run_smoke.sh) ---
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/nvidia/bin:${PATH}"
ldconfig 2>/dev/null || true
( command -v nvidia-smi >/dev/null && nvidia-smi -L ) || echo "  nvidia-smi: n/a"
log "GPU sanity — system python (SPA stack):"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('  GPU', torch.cuda.get_device_name(0), 'CUDA', torch.version.cuda)"
log "GPU sanity — OF3 conda env (triton):"
conda run -n spa-verify-of3 python -c "import torch, triton; print('  OF3 env: torch', torch.__version__, '| triton', triton.__version__, '| cuda', torch.cuda.is_available())"

# --- SPA repo + ProteinMPNN (the Vertex BOOT command already clones $SPA_REPO; clone only if absent) ---
[ -d "$SPA_REPO/.git" ] || git clone --depth 1 --branch "${REPO_REF:-main}" "${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}" "$SPA_REPO"
pip install -e "$SPA_REPO" --no-deps -q
[ -d "$MPNN_REPO" ] || git clone --depth 1 https://github.com/dauparas/ProteinMPNN "$MPNN_REPO"

# --- Fetch the frozen-host + refold weights + the target PDB (no SPA checkpoint: baseline only) ---
mkdir -p /workspace/weights "$OUT"
gcloud storage cp "$RFD3_CKPT_URI" /workspace/weights/rfd3_latest.ckpt
gcloud storage cp "$OF3_CKPT_URI"  /workspace/weights/of3-p2-155k.pt
gcloud storage cp "$SOURCE_PDB_URI" /workspace/target.pdb

log "config: source_pdb=$SOURCE_PDB_URI contig=$CONTIG K=$K N=$NSEQ seed=$SEED"

# --- The whole-structure-as-motif run: baseline only, no SPA checkpoint loaded or referenced. ---
python "$SPA_REPO/scripts/eval/run_flywheel.py" \
  variant=C_n_by_1536 hardware=cloud_h100 \
  'eval.conditions=[baseline]' \
  eval.num_designs="$K" eval.proteinmpnn.num_seqs="$NSEQ" eval.proteinmpnn.seed="$SEED" eval.seed="$SEED" \
  +eval.motif.source_pdb=/workspace/target.pdb \
  "+eval.motif.contig='$CONTIG'" \
  +eval.motif.fixed_atoms=true \
  paths.rfd3_ckpt=/workspace/weights/rfd3_latest.ckpt \
  paths.proteinmpnn_repo="$MPNN_REPO" \
  +eval.flywheel.refolder._target_=spa.eval.openfold3.OF3Refolder \
  +eval.flywheel.refolder.ckpt_path=/workspace/weights/of3-p2-155k.pt \
  +eval.flywheel.refolder.runner_yaml="$SPA_REPO/configs/of3/of3_triton.yml" \
  +eval.flywheel.refolder.out_dir="$OUT" \
  eval.out_dir="$OUT"

# --- Report the three headline numbers (dev plan/69 §3.5's predictions) and stage results to GCS ---
python - "$OUT/flywheel_results.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
scores = d.get("scores", [])
tm = [s["tm_score"] for s in scores if s.get("tm_score") is not None]
scr = [s["scrmsd"] for s in scores if s.get("scrmsd") is not None]
desig = [s for s in scores if s.get("designable")]
div = d.get("diversity_tm")
print(f"[trivial_baseline] {len(scores)} design(s) scored")
print(f"[trivial_baseline] tm_score: {tm}")
print(f"[trivial_baseline] designable: {len(desig)}/{len(scores)}")
print(f"[trivial_baseline] diversity_tm: {div}")
PY

gcloud storage cp "$OUT/flywheel_results.json" "$RESULTS_URI/A0A7S3EB45_trivial.json"
log "===== TRIVIAL DOMAIN BASELINE DONE -> ${RESULTS_URI}/A0A7S3EB45_trivial.json ====="
