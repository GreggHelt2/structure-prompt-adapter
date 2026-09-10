#!/usr/bin/env bash
# In-container H100 half of the CROSS-GPU K=1 portability test (dev plan/100, results/44 section 4b).
#
# THE QUESTION, and why the existing H100 job did not answer it. run_determinism_h100.sh verified both
# shims WITHIN Hopper: 4/4 designs bit-identical across two processes, with index_reduce and
# scatter_add_ giving 50/50 distinct outputs as a negative control proving the probe was not blind. It
# never compared an H100 structure to an A5000 one. results/44 section 4 still lists GPU model in the
# reproducibility identity as "yes, ASSUMED - untested across either".
#
# ⛔ AND ITS ARTIFACTS CANNOT BE REUSED, for two independent reasons:
#   1. verdict.json recorded check_b as COUNTS ({"det":[4,4]}), not hashes or coordinates. Nothing to
#      diff. The GCS prefix holds three JSON files and no structures.
#   2. Its OpenFold3 hash used of3_triton.yml while the A5000's used of3_nokernel.yml, so that pair is
#      confounded by kernel config. Triton cannot run on sm_86 at all (dev plan/23 section 7.7), so an
#      of3_triton comparison against the A5000 is not merely missing, it is impossible.
#
# ⛔ AND K=4 DESIGN 0 IS NOT A K=1 DESIGN. results/44 section 5c.6 measured them 0.506 A apart. Using
# the existing K=4 run's first entry would reintroduce the batch confound into a test whose whole
# purpose is to isolate architecture.
#
# ⭐ SO THIS RUNS K=1 THROUGH generate.py, the same code path as the A5000 half
# (dev scripts/analysis/make_k1_cross_gpu_refs.sh), and ⭐ STAGES THE PDBs THEMSELVES to GCS.
# A verdict JSON is not an artifact: the 2026-09-09 job answered its own question and destroyed the
# evidence, which is the same gap EXPERIMENTS.md already records for four paper-facing cloud runs.
#
# Cheap by construction: 6 single-design generations. The bill is provisioning, not compute.
set -euo pipefail

PROJECT="${PROJECT:-spa-dev-499900}"
BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
RFD3_CKPT_URI="${RFD3_CKPT_URI:-$BUCKET/weights/rfd3_latest.ckpt}"
SPA_REPO="${SPA_REPO:-/opt/spa}"
OUT="${OUT:-/workspace/k1_cross_gpu}"
RESULTS_URI="${RESULTS_URI:-$BUCKET/results/k1_cross_gpu_h100/$(date -u +%Y%m%dT%H%M%SZ)}"
LENGTHS="${LENGTHS:-100 208 374}"   # lengths with an existing A5000 reference

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
trap 'log "K=1 CROSS-GPU CHECK FAILED at line $LINENO"' ERR
mkdir -p "$OUT"

# ⛔ Must NOT be set: the A5000 reference was produced without it, and setting it here would make any
# difference attributable to the env var rather than the architecture.
unset CUBLAS_WORKSPACE_CONFIG || true

# --- GPU driver-path fix (verbatim from run_smoke.sh: without it libcuda is not mounted into the
# container -> torch.cuda.is_available()=False) ---
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/nvidia/bin:${PATH}"
ldconfig 2>/dev/null || true
( command -v nvidia-smi >/dev/null && nvidia-smi -L ) || echo "  nvidia-smi: n/a"

[ -d "$SPA_REPO/.git" ] || git clone --depth 1 --branch "${REPO_REF:-main}" "${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}" "$SPA_REPO"
pip install -e "$SPA_REPO" --no-deps -q

mkdir -p /workspace/weights
gcloud storage cp "$RFD3_CKPT_URI" /workspace/weights/rfd3_latest.ckpt

python - "$OUT/env.json" <<'PY' | tee "$OUT/env.txt"
import json, sys, torch
# The identity tuple includes the torch/CUDA build, so record it: a mismatch here would explain a
# difference without implicating the architecture at all.
e = {"gpu": torch.cuda.get_device_name(0),
     "capability": list(torch.cuda.get_device_capability(0)),
     "torch": torch.__version__, "cuda": torch.version.cuda,
     "cudnn": torch.backends.cudnn.version()}
print(json.dumps(e, indent=1)); json.dump(e, open(sys.argv[1], "w"), indent=1)
PY

log "generating K=1 designs, seed 0, deterministic, lengths: $LENGTHS"
: > "$OUT/refs.jsonl"
for L in $LENGTHS; do
  for rep in a b; do          # twice per length, so H100 self-consistency is established too
    d="$OUT/L${L}_${rep}"; rm -rf "$d"
    python "$SPA_REPO/scripts/eval/generate.py" \
      eval.conditions=[baseline] eval.num_designs=1 eval.length="$L" eval.seed=0 \
      eval.lambda_scale=0.0 eval.deterministic=true \
      paths.rfd3_ckpt=/workspace/weights/rfd3_latest.ckpt \
      eval.out_dir="$d" hydra.run.dir="$d/_hydra" > "$d.log" 2>&1
    f=$(ls "$d"/*.pdb 2>/dev/null | head -1)
    [ -n "$f" ] || { tail -20 "$d.log"; echo "no PDB for L=$L rep=$rep" >&2; exit 1; }
    m=$(md5sum "$f" | cut -c1-12)
    printf '{"platform":"H100","L":%d,"rep":"%s","md5":"%s","file":"%s"}\n' \
      "$L" "$rep" "$m" "$(basename "$f")" | tee -a "$OUT/refs.jsonl"
  done
done

log "H100 self-consistency (a vs b per length)"
# ⛔ NO PIPE HERE. In `python - args | tee file <<'PY'` the heredoc binds to TEE, not python, so
# VERDICT.txt received the SCRIPT SOURCE and python got nothing. Hit on 2026-09-09; the data in
# refs.jsonl was unaffected, but the job's own verdict was unreadable. Write, then tee separately.
python - "$OUT/refs.jsonl" > "$OUT/VERDICT.txt" <<'PY'
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
by = collections.defaultdict(dict)
for r in rows: by[r["L"]][r["rep"]] = r["md5"]
print("H100 K=1, generate.py, seed 0, deterministic\n")
ok = True
for L, d in sorted(by.items()):
    same = d.get("a") == d.get("b"); ok &= same
    print(f"  L={L:<5} {d.get('a')}   {'self-consistent' if same else 'DIFFERS ACROSS PROCESSES'}")
print("\n" + ("H100 is self-consistent at K=1. Diff these hashes against the A5000 refs."
              if ok else
              "H100 is NOT self-consistent; a cross-GPU comparison is meaningless until that is fixed."))
REF = {100: "8fd129f939f6", 208: "742aca051e76", 374: "50c4b7c7591e"}
print("\nCROSS-GPU COMPARISON against the A5000 references")
print("  L=100 is confirmed by THREE independent A5000 runs across two scripts; 208 and 374 by one each.\n")
for L, d in sorted(by.items()):
    r = REF.get(L)
    if r is None:
        print(f"  L={L:<5} no A5000 reference at this length; generate one with"); continue
    v = "PORTABLE, bit-identical across architectures" if d.get("a") == r else "DIFFERS across architectures"
    print(f"  L={L:<5} H100 {d.get('a')}  vs  A5000 {r}   -> {v}")
print("\n⚠️ A match is weak evidence FOR portability at one length and strong evidence AGAINST it if it fails.")
print("⚠️ Compare env.json's torch/CUDA build against the A5000's 2.5.1+cu124 / 12.4 before attributing")
print("   any difference to the architecture: the build sits in the same identity tuple.")
PY
cat "$OUT/VERDICT.txt"

# ⭐ STAGE THE STRUCTURES, not only the verdict. This is the thing the previous job got wrong.
log "uploading results AND PDBs to $RESULTS_URI"
find "$OUT" -name '*.pdb' -print0 | while IFS= read -r -d '' p; do
  gcloud storage cp "$p" "$RESULTS_URI/pdb/$(basename "$(dirname "$p")")__$(basename "$p")"
done
gcloud storage cp "$OUT/VERDICT.txt" "$OUT/refs.jsonl" "$OUT/env.json" "$RESULTS_URI/"
log "===== K=1 CROSS-GPU CHECK COMPLETE ====="
