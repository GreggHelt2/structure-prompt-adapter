#!/usr/bin/env bash
# In-container: what does OpenFold3 bf16-mixed precision buy US, and what does it cost scientifically?
# Runs on the H100 in the spa-combined image. Dev docs/plan/115 §1.1a step 0; results -> dev results/71.
#
# ⭐ WHY THIS EXISTS, AND IT IS ABOUT OUR PIPELINE, NOT ANTHROPIC'S CODE.
# Anthropic benchmark every speed-up against `default`, "the unmodified model run in the fastest correct
# way available to a knowledgeable user". For OpenFold3 their `default` is upstream `base` PLUS two
# settings reachable through upstream's own interface: **bf16-mixed precision** and cuEquivariance
# triangle kernels. Every cell of the dev results/71 matrix compares THEIR `off` to THEIR `exact` inside
# THEIR image, so nothing yet says what OUR stack gives up by sitting a rung lower. This is that arm.
#
# ⛔ SCOPE, stated precisely, because the phrase "the default arm" is broader than what runs here.
#   - RFdiffusion3 `default` is UNREACHABLE for us and that is measured, not assumed: `compile_model`
#     does not exist anywhere in our foundry checkout (it arrived in the 9 commits after our pin) and
#     apex is not installed and has no public wheel for our torch. Out of scope.
#   - ProteinMPNN `default` is eight MPS processes. Reachable in principle, a separate question, and note
#     it would NOT break the one-PDB-per-call invariance the way batching does, since each MPS process
#     still folds one backbone at a time. Not this run.
#   - OpenFold3 is the one that is both reachable and on the stage that matters: **90.5% of local
#     wall-clock** (dev plan/85 §2). Precision is a single runner-yaml key and needs no new package.
# ⇒ This run tests the PRECISION half of OpenFold3 `default`, alone.
#
# ⚠️ A CORRECTION THIS RUN DEPENDS ON. Dev plan/115 §1.1a says we are "exactly their `base`" for
# OpenFold3, citing configs/of3/of3_nokernel.yml. That is the LOCAL A5000 config. On the CLOUD we run
# of3_triton.yml, which already enables Triton triangle kernels, so on the H100 we are NOT at `base`:
# we are at base-plus-kernels, fp32. ⇒ holding of3_triton.yml fixed and moving ONLY precision is what
# isolates the precision lever here. Do not read these numbers as base-to-default.
#
# ⛔ KEY PATH VERIFIED FROM SOURCE, NOT GUESSED (the dev results/71 §7.3 lesson):
#     openfold3/entry_points/validator.py:121  class PlTrainerArgs
#     openfold3/entry_points/validator.py:127      precision: int | str = "32-true"
#     openfold3/entry_points/validator.py:298  InferenceExperimentConfig.pl_trainer_args
#     openfold3/entry_points/experiment_runner.py:108  is where it reaches the Trainer
#   and the bf16 literal is upstream's own spelling, "bf16-mixed" (tests/test_of3_model.py:77).
#
# ⭐ THE CONTROL IS WHAT MAKES THIS MEASURABLE, and it is free.
# Each precision runs TWICE. Our refolds are deterministic by default (OF3Refolder(deterministic=None)
# means ON), so fp32 rep A against fp32 rep B should be BIT-IDENTICAL. That pins the rerun floor at
# exactly 0, which means any fp32-vs-bf16 difference is attributable to precision ALONE rather than to
# refold noise. ⛔ If the control does NOT come back identical, the comparison is void and this script
# says so instead of reporting a delta.
#
# ⛔ THE SCIENTIFIC READOUT IS AN ANGSTROM NUMBER, NOT A HASH. bf16 is guaranteed to change bytes, so
# "differs" is not a finding. The finding is HOW FAR, against the 2.0 A designability threshold, which
# is why this computes score.py::ca_rmsd and not just a checksum. Dev results/68 is the precedent: a
# refold-epoch change moved 1,088 designs but flipped only 10 verdicts.
#
# ⛔ THIS RUNS AT P=1, DELIBERATELY, AND THAT IS WHAT OUR PRODUCTION PATH IS (Gregg asked, 2026-09-23).
# Verified from source rather than assumed:
#   - OF3Refolder.refold_all is the path the flywheel uses, and its own docstring says the model loads
#     once per subprocess but "OF3 still folds queries sequentially" (src/spa/eval/openfold3.py:346-352).
#     So query BATCHING into one process is not query CONCURRENCY.
#   - No cloud script sets lanes or P at all; a grep for lanes/P=3 over scripts/cloud/*.sh finds nothing.
#   - P=3 is a LOCAL A5000 practice applied from OUTSIDE the refolder, by the dev-repo lane drivers
#     (ligand_refold_lane.py, msa_refold_lane.py) sharding queries across three separate processes.
# ⭐ P=1 is also what dev plan/115 §5.1 requires for any comparison against Anthropic: "P=3 is our
# throughput trick and Anthropic's is batching, and mixing them measures neither". Running this at P=3
# would confound the precision lever with lane contention.
# ⚠️ CONSEQUENCE FOR READING THE SPEED NUMBER: it is PER REFOLD. It does not transfer to a local sweep
# that already takes 1.72x from P=3, and whether bf16 COMPOSES with P=3 or COMPETES with it is a
# separate question this run cannot answer. `bs=8` looked additive and measured 0.83x at P=3
# (dev plan/85 §4e), so assume nothing.
#
# ⚠️ AND WHATEVER IT SHOWS, ADOPTION IS A NEW EPOCH. Nothing refolded at bf16 pools with docs/results/.
set -uo pipefail

BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
OF3_CKPT_URI="${OF3_CKPT_URI:-$BUCKET/weights/of3-p2-155k.pt}"
SPA_REPO="${SPA_REPO:-/opt/spa}"
LENGTHS="${LENGTHS:-76,250}"
OUT="${OUT:-/workspace/of3_precision_bench}"
GCS_OUT="${GCS_OUT:-$BUCKET/eval/of3_precision_bench/$(date -u +%Y%m%d-%H%M%S)}"
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

# ⛔ Driver paths: run_determinism_h100.sh:41-42 has exported exactly these for months. A runner that
# trims them reports cuda_available False on a working H100 and dies at the gate (dev results/71 §4.1).
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/nvidia/bin:${PATH}"; ldconfig 2>/dev/null || true
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv 2>&1 | head -2 || log "note: nvidia-smi unavailable (advisory)"
python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))" \
  || { log "FATAL: torch cannot see the GPU"; exit 10; }

# ⛔ NO git fetch HERE, AND THAT IS DELIBERATE. The submit script's $BOOT_CHECKOUT (scripts/cloud/
# _pin_run_env.sh:80) has already checked out the PINNED code rev and detached, and _pin_repo_ref
# defaults REPO_REF to the submitting machine's HEAD sha precisely so the run is reproducible. A
# `fetch origin main && reset --hard` here would silently replace that pin with whatever main is at
# job START, turning a provenance guarantee into a race. ⇒ verify and RECORD, never re-fetch.
[ -d "$SPA_REPO/.git" ] || { log "FATAL: no repo at $SPA_REPO; BOOT_CHECKOUT must run first"; exit 12; }
log "PINNED_CODE_REV in container: $(git -C "$SPA_REPO" rev-parse HEAD)"
pip install -e "$SPA_REPO" --no-deps -q
python -c "import spa; print('spa import OK', spa.__file__)" || { log "FATAL: spa not importable"; exit 13; }
# ⛔ The three pieces this bench REUSES rather than reimplements. Fail loudly if the pinned tree lacks
# one, instead of discovering it 4 minutes into billed GPU time.
for need in scripts/eval/bench_of3_length.py configs/of3/of3_triton.yml; do
  [ -r "$SPA_REPO/$need" ] || { log "FATAL: pinned tree has no $need"; exit 14; }
done
# ⛔ THIS GATE RUNS IN THE DEFAULT ENV, NOT spa-verify-of3, AND THAT IS THE WHOLE ARCHITECTURE.
# openfold3.py:13-16 states it: OpenFold3 "is not importable here, it has its own heavy deps. We
# invoke it via subprocess ... run in spa-verify-of3 via conda run". So `spa` lives in the DEFAULT env
# and OF3Refolder(conda_env="spa-verify-of3") spawns the OF3 env ITSELF, per call.
# ⚠️ A first version wrapped this gate and the whole bench in `conda run -n spa-verify-of3`, where
# `spa` is not installed. The job died in 30 s (~$0.09) reporting "ca_rmsd missing from the pinned
# tree" while the tree was perfectly fine and the ENV was wrong. ⭐ The gate did its job, it caught my
# bug before any GPU work, but its MESSAGE sent the diagnosis in the wrong direction, which is the
# same defect as conflating a crash with a missing opt-in: distinct failures must not share a message.
python -c "from spa.eval.score import ca_rmsd; print('ca_rmsd OK (default env, where spa lives)')" \
  || { log "FATAL: cannot import spa.eval.score.ca_rmsd in the DEFAULT env."; \
       log "  Distinguish before acting: is 'spa' importable at all (the pip install -e above)?"; \
       log "  Is biotite present? Is ca_rmsd actually in the pinned tree? These are 3 different faults."; \
       exit 14; }
conda run -n spa-verify-of3 python -c "import triton; print('OF3 env: triton', triton.__version__)"

mkdir -p /workspace/weights "$OUT"
gcloud storage cp "$OF3_CKPT_URI" /workspace/weights/of3-p2-155k.pt
CKPT=/workspace/weights/of3-p2-155k.pt
log "OF3 ckpt sha256: $(sha256sum "$CKPT" | cut -c1-16)"

# ---------------------------------------------------------------- the two runner-yamls
# fp32 = our cloud config VERBATIM. bf16 = the same file plus ONE key, so the diff is auditable.
BASE_YAML="$SPA_REPO/configs/of3/of3_triton.yml"
[ -r "$BASE_YAML" ] || { log "FATAL: no $BASE_YAML"; exit 11; }
cp "$BASE_YAML" "$OUT/fp32.yml"
cp "$BASE_YAML" "$OUT/bf16.yml"
cat >> "$OUT/bf16.yml" <<'YML'

# ADDED by run_of3_precision_bench.sh: the precision half of Anthropic's OpenFold3 `default`.
# Key path from upstream source: PlTrainerArgs.precision (validator.py:127, default "32-true"),
# carried on InferenceExperimentConfig.pl_trainer_args (:298), reaching the Trainer at
# experiment_runner.py:108. The literal is upstream's own (tests/test_of3_model.py:77).
pl_trainer_args:
  precision: bf16-mixed
YML
log "--- the ONLY difference between the two arms ---"; diff "$OUT/fp32.yml" "$OUT/bf16.yml" || true

# ---------------------------------------------------------------- arms
# ⭐ Uses the EXISTING bench_of3_length.py rather than a new refold path: it already takes
# --of3-runner-yaml, goes through OF3Refolder, and records wall seconds + peak VRAM per length.
# Writing a second refold harness is the duplicate docs/CAPABILITIES.md exists to prevent.
run_arm() {                                   # run_arm <label> <yaml>
  [ $# -eq 2 ] || { log "FATAL: run_arm takes 2 args, got $#"; return 2; }
  # ⛔ TWO SEPARATE `local` STATEMENTS, DELIBERATELY. `local label=$1 d="$OUT/$label"` on ONE line dies
  # under `set -u` with "label: unbound variable", because every argument to the `local` builtin is
  # word-expanded BEFORE the builtin assigns any of them, so $label is still unset when $d expands.
  # ⚠️ `bash -n` does NOT catch this (it is a runtime expansion, not a syntax error) and it cost a third
  # H100 provisioning cycle. It reproduces locally in one line:
  #     bash -c 'set -u; f(){ local a=$1 b="X/$a"; echo "$b"; }; f hi'
  # ⇒ shellcheck catches it; run it on this file before submitting.
  local label=$1 yml=$2
  local d="$OUT/$label"
  mkdir -p "$d"
  log "=== ARM $label  ($(basename "$yml")) ==="
  # ⛔ DEFAULT ENV, not spa-verify-of3: the driver needs `spa`, and it passes --of3-conda-env down so
  # OF3Refolder spawns the OF3 env per refold (bench_of3_length.py:86,107). See the gate above.
  python "$SPA_REPO/scripts/eval/bench_of3_length.py" \
      --lengths "$LENGTHS" --of3-ckpt "$CKPT" --of3-runner-yaml "$yml" \
      --of3-conda-env spa-verify-of3 \
      --out-dir "$d" --json "$d/bench.json" > "$d/stdout.log" 2>&1
  local rc=$?
  log "ARM $label rc=$rc"
  grep -E "^\[bench\] L=" "$d/stdout.log" || { log "no per-length lines; tail:"; tail -20 "$d/stdout.log"; }
  [ $rc -eq 0 ] || return $rc
}

run_arm fp32_a "$OUT/fp32.yml"
run_arm fp32_b "$OUT/fp32.yml"
run_arm bf16_a "$OUT/bf16.yml"
run_arm bf16_b "$OUT/bf16.yml"

# ---------------------------------------------------------------- compare
log "=== COMPARISON ==="
# ⛔ DEFAULT ENV again: this imports spa.eval.score and biotite, neither of which is in spa-verify-of3.
python - "$OUT" "$LENGTHS" <<'PY' 2>&1 | tee "$OUT/VERDICT.txt"
import json, pathlib, subprocess, sys, hashlib

out, lengths = pathlib.Path(sys.argv[1]), [int(x) for x in sys.argv[2].split(",")]
ARMS = ["fp32_a", "fp32_b", "bf16_a", "bf16_b"]

def secs(arm, L):
    p = out / arm / "bench.json"
    if not p.is_file():
        return None
    for r in json.loads(p.read_text())["rows"]:
        if r["length"] == L and r["ok"]:
            return r["seconds"]
    return None

def cif(arm, L):
    """The refold OF3Refolder wrote for this length, whatever the seed dir is called."""
    hits = sorted((out / arm / f"L{L}").rglob("*.cif")) + sorted((out / arm / f"L{L}").rglob("*.cif.gz"))
    return hits[0] if hits else None

def canon(p):
    """Content hash: decompress, strip the generation timestamp (dev results/71 §3)."""
    raw = subprocess.run(["gunzip", "-c", str(p)], capture_output=True).stdout if p.suffix == ".gz" \
          else p.read_bytes()
    keep = [l for l in raw.splitlines() if not l.startswith((b"_entry.date", b"_entry.time"))]
    return hashlib.sha256(b"\n".join(keep)).hexdigest()[:12]

def ca(a, b):
    from spa.eval.score import ca_rmsd            # score.py::ca_rmsd, line 486
    import biotite.structure.io.pdbx as pdbx
    def load(p):
        f = pdbx.CIFFile.read(str(p))
        return pdbx.get_structure(f, model=1)
    return ca_rmsd(load(a), load(b))

print(f"{'L':>5} {'arm':>7} {'sec':>7}  hash")
for L in lengths:
    for a in ARMS:
        s, c = secs(a, L), cif(a, L)
        print(f"{L:>5} {a:>7} {('%.1f' % s) if s else '    ?':>7}  {canon(c) if c else '(no structure)'}")

print()
for L in lengths:
    fa, fb, ba = cif("fp32_a", L), cif("fp32_b", L), cif("bf16_a", L)
    # ⛔ THE CONTROL GATES EVERYTHING. An empty arm must never yield a verdict (dev plan/118 §3).
    if not (fa and fb and ba):
        print(f"L={L}: ⛔ INDETERMINATE, an arm produced no structure. NOT a difference.")
        continue
    ctrl_same = canon(fa) == canon(fb)
    print(f"L={L}: control fp32_a vs fp32_b = {'✅ BIT-IDENTICAL' if ctrl_same else '⛔ DIFFER'}"
          + ("" if ctrl_same else "  => the rerun floor is not 0, so the bf16 delta below is NOT attributable to precision"))
    try:
        d_ctrl = ca(fa, fb)
        d_bf16 = ca(fa, ba)
        print(f"L={L}: Ca RMSD  fp32 rerun (control) = {d_ctrl:.4f} A")
        print(f"L={L}: Ca RMSD  fp32 vs bf16         = {d_bf16:.4f} A   "
              f"({d_bf16 / 2.0 * 100:.1f}% of the 2.0 A designability threshold)")
    except Exception as e:                                        # noqa: BLE001
        print(f"L={L}: Ca RMSD unavailable: {type(e).__name__}: {e}")
    sa, sb = secs("fp32_a", L), secs("bf16_a", L)
    if sa and sb:
        print(f"L={L}: speed  fp32 {sa:.1f} s vs bf16 {sb:.1f} s  => {sa / sb:.2f}x")
print()
print("⚠️ Wall time here INCLUDES per-invocation start-up, so a ratio at one length is not a kernel")
print("   figure; dev results/71 §9.5 and §7.7 both measured start-up dominating a short arm.")
print("⛔ Whatever the numbers, bf16 refolds do NOT pool with docs/results/: adoption is a new epoch.")
PY

log "=== persisting to GCS ==="
gcloud storage cp -r "$OUT" "$GCS_OUT/" >/dev/null 2>&1 && log "artifacts at $GCS_OUT" || log "could not write $GCS_OUT"
log "=== DONE ==="
