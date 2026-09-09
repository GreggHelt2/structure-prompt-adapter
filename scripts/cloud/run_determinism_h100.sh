#!/usr/bin/env bash
# In-container H100 PORTABILITY CHECK for the determinism shims (dev plan/91 §6.2).
#
# WHY. Both shims (spa.eval.determinism for RFD3, scripts/eval/of3_determinism_patch.py for OpenFold3)
# are architecture-agnostic Python, but EVERY measurement behind them is one A5000 in one process.
# Three things could differ on Hopper, and only a run can say:
#
#   A. cuBLAS. The RFD3 fix works because GEMMs are bit-exact run to run, measured on the A5000 as
#      1 distinct bit pattern in 50 calls WITHOUT CUBLAS_WORKSPACE_CONFIG. That is a property of which
#      kernel cuBLAS selects, not a guarantee of matmul: cuBLAS can pick split-k with atomic
#      accumulation for some shapes. If Hopper does, the one-hot matmul inherits the nondeterminism it
#      was written to remove. ⭐ THIS IS THE GATE: if A fails, B and C are expected to fail too.
#   B. RFD3 end to end, which is the thing we actually ship.
#   C. OpenFold3 on of3_triton.yml, the CLOUD runner-yaml. The A5000 work used of3_nokernel.yml;
#      triton mode has a ~4x larger stock spread and the scatter_add_-only route was NEVER tested on it.
#      Triton kernels can carry their own atomics.
#
# Inference only, no cache/splits/NGC. Mirrors run_smoke.sh's bootstrap and driver fix verbatim.
# Cheap by construction: K=4 at L=100 and one 76 aa refold, so the bill is startup, not compute.
set -euo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
OF3_CKPT_URI="${OF3_CKPT_URI:-$BUCKET/weights/of3-p2-155k.pt}"
SPA_REPO="${SPA_REPO:-/opt/spa}"
MPNN_REPO="${MPNN_REPO:-/opt/ProteinMPNN}"
OUT="${OUT:-/workspace/det_out}"
RESULTS_URI="${RESULTS_URI:-$BUCKET/results/determinism_h100/$(date -u +%Y%m%dT%H%M%SZ)}"

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
trap 'log "DETERMINISM CHECK FAILED at line $LINENO"' ERR
mkdir -p "$OUT"

# ⛔ CUBLAS_WORKSPACE_CONFIG must NOT be set: the whole point of check A is whether Hopper needs it.
unset CUBLAS_WORKSPACE_CONFIG || true

# --- GPU driver-path fix (verbatim from run_smoke.sh: without it libcuda is not mounted into the
# container -> torch.cuda.is_available()=False) ---
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/nvidia/bin:${PATH}"
ldconfig 2>/dev/null || true
( command -v nvidia-smi >/dev/null && nvidia-smi -L ) || echo "  nvidia-smi: n/a"

[ -d "$SPA_REPO/.git" ] || git clone --depth 1 --branch "${REPO_REF:-main}" "${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}" "$SPA_REPO"
pip install -e "$SPA_REPO" --no-deps -q
[ -d "$MPNN_REPO" ] || git clone --depth 1 https://github.com/dauparas/ProteinMPNN "$MPNN_REPO"

mkdir -p /workspace/weights
gcloud storage cp "$RFD3_CKPT_URI" /workspace/weights/rfd3_latest.ckpt
gcloud storage cp "$OF3_CKPT_URI"  /workspace/weights/of3-p2-155k.pt

# =====================================================================================
# CHECK A — the gate. Is a GEMM bit-exact run to run on Hopper, with no env var set?
# Also re-probes index_reduce and scatter_add_ so the A5000 numbers have a Hopper counterpart.
# =====================================================================================
log "CHECK A: op-level determinism (cuBLAS gate)"
python - "$OUT/check_a.json" <<'PY' | tee "$OUT/check_a.txt"
import json, sys, os, torch
assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ, "env var set; check A would be meaningless"
dev = torch.cuda.get_device_name(0)
L, I, C = 1200, 150, 384
src = torch.randn(1, L, C, device="cuda")
idx = torch.sort(torch.randint(0, I, (L,), device="cuda"))[0]
w   = torch.randn(C, C, device="cuda")
res = {"gpu": dev, "torch": torch.__version__, "cuda": torch.version.cuda, "ops": {}}

def distinct(fn, n=50):
    outs = [fn() for _ in range(n)]
    return len({o.float().cpu().numpy().tobytes() for o in outs})

for dt in (torch.float32, torch.bfloat16):
    name = str(dt).split(".")[-1]
    s, ww = src.to(dt), w.to(dt)
    res["ops"][f"matmul_{name}"] = distinct(lambda: s @ ww)
    z = torch.zeros(1, I, C, device="cuda", dtype=dt)
    res["ops"][f"index_reduce_{name}"] = distinct(
        lambda: z.index_reduce(-2, idx, s, "mean", include_self=False))
    ie = idx.view(1, L, 1).expand(1, L, C)
    res["ops"][f"scatter_add_{name}"] = distinct(
        lambda: torch.zeros(1, I, C, device="cuda", dtype=dt).scatter_add_(1, ie, s))
    # the shim's own replacement, end to end
    def onehot():
        oh = torch.zeros(I, L, device="cuda", dtype=dt)
        oh[idx, torch.arange(L, device="cuda")] = 1.0
        return (oh @ s) / oh.sum(-1, keepdim=True).clamp(min=1)
    res["ops"][f"onehot_matmul_{name}"] = distinct(onehot)

gate = max(res["ops"]["matmul_float32"], res["ops"]["matmul_bfloat16"],
           res["ops"]["onehot_matmul_float32"], res["ops"]["onehot_matmul_bfloat16"])
res["gate_pass"] = (gate == 1)
print(json.dumps(res, indent=2))
print("\nCHECK A:", "PASS (GEMMs bit-exact, shim transfers)" if res["gate_pass"]
      else "FAIL (GEMMs vary -> Hopper needs CUBLAS_WORKSPACE_CONFIG; cost-neutrality is A5000-only)")
json.dump(res, open(sys.argv[1], "w"), indent=2)
PY

# =====================================================================================
# CHECK B — RFD3 end to end. Two runs per arm, fresh process each, same seed.
# =====================================================================================
log "CHECK B: RFD3 generation bit-identity (deterministic vs stock)"
for arm in det stock; do
  [ "$arm" = det ] && D=true || D=false
  for rep in A B; do
    python "$SPA_REPO/scripts/eval/generate.py" \
      variant=C_n_by_1536 hardware=cloud_h100 \
      'eval.conditions=[baseline]' eval.num_designs=4 eval.length=100 eval.seed=0 \
      eval.deterministic=$D \
      paths.rfd3_ckpt=/workspace/weights/rfd3_latest.ckpt \
      eval.out_dir="$OUT/rfd3_${arm}_${rep}" \
      hydra.run.dir="$OUT/hyd_rfd3_${arm}_${rep}" > "$OUT/log_rfd3_${arm}_${rep}.txt" 2>&1
    log "  rfd3 $arm $rep done"
  done
done

# =====================================================================================
# CHECK C — OpenFold3 on the CLOUD runner-yaml (triton), which the A5000 work never tested.
# =====================================================================================
log "CHECK C: OpenFold3 refold bit-identity on of3_triton.yml"
for arm in det stock; do
  [ "$arm" = det ] && D=True || D=False
  for rep in A B; do
    python - "$SPA_REPO" "$OUT/of3_${arm}_${rep}" "$D" <<'PY' > "$OUT/log_of3_${arm}_${rep}.txt" 2>&1
import sys
from pathlib import Path
repo, out, det = sys.argv[1], sys.argv[2], sys.argv[3] == "True"
sys.path.insert(0, f"{repo}/src")
from spa.eval.openfold3 import OF3Refolder
from spa.eval.proteinmpnn import SequenceSet
SEQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
r = OF3Refolder(ckpt_path="/workspace/weights/of3-p2-155k.pt",
                runner_yaml=f"{repo}/configs/of3/of3_triton.yml",
                out_dir=out, conda_env="spa-verify-of3",
                num_diffusion_samples=1, seed=42, deterministic=det)
ss = SequenceSet(name="ubq", design_path=Path("/dev/null"), fasta_path=Path("/dev/null"),
                 sequences=[SEQ], n_residues=len(SEQ))
print("refolds:", r.refold(ss))
PY
    log "  of3 $arm $rep done"
  done
done

# =====================================================================================
# VERDICT
# =====================================================================================
log "collating"
python - "$OUT" <<'PY' | tee "$OUT/VERDICT.txt"
import glob, hashlib, json, os, sys
OUT = sys.argv[1]
a = json.load(open(f"{OUT}/check_a.json"))

def pdbs(d):
    return sorted(glob.glob(f"{d}/*.pdb"))
def same_files(A, B):
    fa, fb = pdbs(A), pdbs(B)
    if not fa or len(fa) != len(fb): return None, len(fa)
    n = sum(1 for x, y in zip(fa, fb) if open(x,'rb').read() == open(y,'rb').read())
    return n, len(fa)
def cif_md5(d):
    g = glob.glob(f"{d}/of3/ubq/q0/seed_42/*_model.cif")
    return hashlib.md5(open(g[0],'rb').read()).hexdigest()[:12] if g else None

print("="*74)
print(f"H100 DETERMINISM PORTABILITY CHECK  ({a['gpu']}, torch {a['torch']}, CUDA {a['cuda']})")
print("="*74)
print("\nCHECK A — op-level, CUBLAS_WORKSPACE_CONFIG unset (distinct bit patterns / 50):")
for k, v in a["ops"].items():
    flag = "ok" if (("matmul" in k and v == 1) or ("matmul" not in k and v > 1)) else "!!"
    print(f"   {k:<28} {v:>3}/50   {flag}")
print(f"\n   GATE: {'PASS' if a['gate_pass'] else 'FAIL'}"
      f"  ({'GEMMs bit-exact, the RFD3 shim transfers' if a['gate_pass'] else 'GEMMs vary on Hopper'})")

print("\nCHECK B — RFD3 generation, 2 processes per arm, same seed:")
res_b = {}
for arm in ("det", "stock"):
    n, tot = same_files(f"{OUT}/rfd3_{arm}_A", f"{OUT}/rfd3_{arm}_B")
    res_b[arm] = (n, tot)
    print(f"   deterministic={arm=='det'!s:<5}  {n}/{tot} designs bit-identical")

print("\nCHECK C — OpenFold3 refold on of3_triton.yml (CLOUD config), 2 processes per arm:")
res_c = {}
for arm in ("det", "stock"):
    ma, mb = cif_md5(f"{OUT}/of3_{arm}_A"), cif_md5(f"{OUT}/of3_{arm}_B")
    res_c[arm] = (ma, mb)
    print(f"   deterministic={arm=='det'!s:<5}  md5 {ma} / {mb}   {'IDENTICAL' if ma and ma==mb else 'DIFFER'}")

ok = (a["gate_pass"]
      and res_b["det"][0] == res_b["det"][1] and res_b["det"][1] > 0
      and res_c["det"][0] and res_c["det"][0] == res_c["det"][1])
print("\n" + "="*74)
print("OVERALL:", "PASS — both shims transfer to the H100" if ok
      else "FAIL — see the checks above; do NOT assume A5000 results carry")
print("="*74)
json.dump({"check_a": a, "check_b": res_b, "check_c": res_c, "overall_pass": ok},
          open(f"{OUT}/verdict.json", "w"), indent=2)
PY

log "uploading results to $RESULTS_URI"
gcloud storage cp -r "$OUT/VERDICT.txt" "$OUT/verdict.json" "$OUT/check_a.json" "$RESULTS_URI/" || \
  log "  (upload issues; the verdict is above in the job log regardless)"
log "===== DETERMINISM CHECK COMPLETE ====="
