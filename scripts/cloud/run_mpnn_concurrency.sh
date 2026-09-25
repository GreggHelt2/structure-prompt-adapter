#!/usr/bin/env bash
# In-container: does running OUR ProteinMPNN stage CONCURRENTLY change any sequence, and what does it buy?
# Dev docs/plan/115 §0.3b row 1; results -> dev docs/results/71 §13. Runs in spa-combined on the H100.
#
# ⛔ SCOPE, STATED PRECISELY, BECAUSE "ProteinMPNN default" IS BROADER THAN WHAT RUNS HERE.
# Anthropic's ProteinMPNN `default` is CUDA MPS packing EIGHT upstream processes onto one card, each still
# folding ONE BACKBONE AT A TIME, plus a fixed explicit seed. ⭐ Two separable parts:
#   (a) CONCURRENCY: several independent single-backbone processes sharing the GPU. ⭐ This is IN-CONVENTION
#       for us: plan/100 §0 says "Throughput is bought with P, never with K", and our OpenFold3 refolds
#       already run at P=3 this way. No Anthropic code, no MPS daemon, nothing new to adopt.
#   (b) THE MPS DAEMON itself, which reduces context-switch overhead between those processes.
# ⇒ THIS RUNNER TESTS (a) ONLY. It does not start nvidia-cuda-mps-control and makes no claim about MPS.
# Reporting a concurrency number as an "MPS result" would be the label error 115 §0 exists to prevent.
#
# ⭐ THE VALUABLE QUESTION IS INVARIANCE, NOT SPEED, and the speed ceiling is why.
# ProteinMPNN is 3.8% of local wall-clock (plan/85 §2), so even their claimed 2.8-4.8x is ~3% end to end.
# ⛔ But the invariance question is open and consequential: OpenFold3 FAILS partition invariance in every
# mode (results/71 §7.7), and Anthropic's test P showed ProteinMPNN's output depends on the BACKBONE SET
# when batched (§8). ⇒ If splitting our Stage-2 across concurrent processes changed any sequence, then
# P-way concurrency would be unsafe at Stage 2 exactly as query partitioning is unsafe at Stage 3, and
# that is a finding about OUR pipeline that matters whatever the speed says.
#
# TWO ARMS over the SAME 8 backbones, everything else held (seed 42, N=8, temp, batch_size, weights):
#   A  one inverse_fold.py call over a design_dir of 8   -> our PRODUCTION shape, internally serial
#   B  8 concurrent inverse_fold.py calls, 1 backbone each -> P=8 concurrency
# Per-backbone SEQ hashes must match between arms. Wall time is secondary and reported as such.
#
# ⛔ PREDICTION, PRE-REGISTERED so a null cannot be dressed up: sequences SHOULD match. Each process is
# independent, single-backbone, fixed-seed, so there is no shared reduction for concurrency to perturb.
# A DIFFERENCE would be the surprise and the real result.
set -uo pipefail

BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
PREP_URI="${PREP_URI:-$BUCKET/eval/threeway/prep}"
SPA_REPO="${SPA_REPO:-/opt/spa}"
NUM_SEQS="${NUM_SEQS:-8}"
SEED="${SEED:-42}"
P="${P:-8}"
OUT="${OUT:-/workspace/mpnn_conc}"
GCS_OUT="${GCS_OUT:-$BUCKET/eval/mpnn_concurrency/$(date -u +%Y%m%d-%H%M%S)}"
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/nvidia/bin:${PATH}"; ldconfig 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total --format=csv 2>&1 | head -2 || log "note: nvidia-smi unavailable"
python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))" \
  || { log "FATAL: torch cannot see the GPU"; exit 10; }

[ -d "$SPA_REPO/.git" ] || { log "FATAL: no repo at $SPA_REPO"; exit 12; }
log "PINNED_CODE_REV: $(git -C "$SPA_REPO" rev-parse HEAD)"
pip install -e "$SPA_REPO" --no-deps -q
python -c "import spa" || { log "FATAL: spa not importable"; exit 13; }
MPNN_REPO="${MPNN_REPO:-/opt/ProteinMPNN}"
[ -d "$MPNN_REPO" ] || git clone --depth 1 https://github.com/dauparas/ProteinMPNN "$MPNN_REPO"

mkdir -p "$OUT/designs"
gcloud storage cp "$PREP_URI/*.pdb" "$OUT/designs/" || { log "FATAL: no backbones"; exit 11; }
N_BB=$(find "$OUT/designs" -name '*.pdb' | wc -l)
log "backbones: $N_BB"
[ "$N_BB" -ge 2 ] || { log "FATAL: need >=2 backbones for a concurrency comparison, have $N_BB"; exit 11; }

one_call() {                       # one_call <outroot> <design_dir>
  [ $# -eq 2 ] || { log "FATAL: one_call needs 2 args, got $#"; return 2; }
  local root=$1
  local ddir=$2
  mkdir -p "$root"
  python "$SPA_REPO/scripts/eval/inverse_fold.py" \
      eval.proteinmpnn.design_dir="$ddir" \
      eval.proteinmpnn.out_dir="$root" \
      eval.proteinmpnn.num_seqs="$NUM_SEQS" \
      eval.proteinmpnn.seed="$SEED" \
      paths.proteinmpnn_repo="$MPNN_REPO" \
      hydra.run.dir="$root/hyd" > "$root/stdout.log" 2>&1
}

# ---------------------------------------------------------------- ARM A: production shape, serial
log "=== ARM A: ONE call over all $N_BB backbones (our production shape) ==="
tA0=$(date +%s); one_call "$OUT/A" "$OUT/designs"; rcA=$?; tA1=$(date +%s)
log "ARM A rc=$rcA in $((tA1-tA0))s"
[ $rcA -eq 0 ] || tail -20 "$OUT/A/stdout.log"

# ---------------------------------------------------------------- ARM B: P concurrent single-backbone calls
log "=== ARM B: $P CONCURRENT calls, one backbone each ==="
i=0
tB0=$(date +%s)
while IFS= read -r pdb; do
  d="$OUT/B/shard$i"; mkdir -p "$d/designs"
  cp "$pdb" "$d/designs/"
  one_call "$d" "$d/designs" &
  i=$((i+1))
  # ⛔ Cap in-flight jobs at P. A `for ... &` with no cap is how a box gets saturated (dev plan/98).
  while [ "$(jobs -rp | wc -l)" -ge "$P" ]; do sleep 1; done
done < <(find "$OUT/designs" -name '*.pdb' | sort)
wait
tB1=$(date +%s)
log "ARM B ($i shards, cap P=$P) in $((tB1-tB0))s"

# ---------------------------------------------------------------- compare
log "=== per-backbone SEQ hashes ==="
# ⛔ find, not a flat glob: ProteinMPNN appends seqs/ to out_dir itself (dev plan/112 rule 17, 3 instances).
hash_tree() {                      # hash_tree <root> <label>
  find "$1" -type f \( -name '*.fa' -o -name '*.fasta' \) | sort | while IFS= read -r f; do
    printf '%s  %s  %s\n' "$(grep -v '^>' "$f" | sha256sum | cut -c1-12)" "$(basename "$f")" "$2"
  done
}
hash_tree "$OUT/A" A | sort -k2 > "$OUT/A.HASHES"
hash_tree "$OUT/B" B | sort -k2 > "$OUT/B.HASHES"

# ⛔ PERSIST BEFORE ANY VERDICT (dev results/71 §4.2, and §12.6 where skipping it destroyed the evidence).
log "=== persisting to GCS (before the verdict, deliberately) ==="
gcloud storage cp -r "$OUT" "$GCS_OUT/" >/dev/null 2>&1 && log "artifacts at $GCS_OUT" || log "could not write $GCS_OUT"

nA=$(wc -l < "$OUT/A.HASHES"); nB=$(wc -l < "$OUT/B.HASHES")
log "arm A produced $nA fasta(s); arm B produced $nB"
paste <(awk '{print $2, $1}' "$OUT/A.HASHES") <(awk '{print $1}' "$OUT/B.HASHES") | sed 's/^/    /'

# ⛔ An empty or ragged result must never return a verdict (dev plan/118 §3).
if [ "$nA" -eq 0 ] || [ "$nB" -eq 0 ]; then
  log "⛔ VERDICT: INDETERMINATE, an arm produced no FASTA. NOT a match. Tree:"; find "$OUT" -name '*.fa' | head
  log "--- arm A stdout tail ---"; tail -20 "$OUT/A/stdout.log" 2>/dev/null
  exit 15
fi
if [ "$nA" -ne "$nB" ]; then
  log "⛔ VERDICT: INDETERMINATE, arm counts differ ($nA vs $nB). Comparing unequal sets proves nothing."
  exit 15
fi
if diff <(awk '{print $1, $2}' "$OUT/A.HASHES") <(awk '{print $1, $2}' "$OUT/B.HASHES") >/dev/null; then
  log "✅ VERDICT: every backbone's sequences are IDENTICAL serial vs P=$P."
  log "   ⇒ our Stage-2 is CONCURRENCY-INVARIANT, so P-way concurrency is safe here, unlike"
  log "     OpenFold3's query partitioning (§7.7) and unlike Anthropic's batched ProteinMPNN (§8, test P)."
else
  log "⛔ VERDICT: SEQUENCES DIFFER between serial and P=$P. This is the SURPRISE and the real finding:"
  log "   it would make P-way concurrency unsafe at Stage 2. Differences:"
  diff <(awk '{print $1, $2}' "$OUT/A.HASHES") <(awk '{print $1, $2}' "$OUT/B.HASHES") | head -12
fi

log "=== speed, reported as SECONDARY ==="
python3 -c "
a=$((tA1-tA0)); b=$((tB1-tB0))
print(f'    serial {a}s   P=$P {b}s   ratio {a/b:.2f}x' if b else '    (zero elapsed)')
print('    ⚠️ ProteinMPNN is 3.8% of local wall-clock, so even a large ratio here is ~3% end to end.')
print('    ⚠️ n=1 timing per arm, and arm A pays one process start while arm B pays $P of them.')
"
log "=== DONE ==="
