#!/usr/bin/env bash
# Is OUR ProteinMPNN Stage-2 byte-reproducible ACROSS GPU ARCHITECTURES? The A5000 (sm_86) half of the
# comparison whose H100 (sm_90) half is already banked. Spec: dev plan/115 §9.1e.
#
# ⭐ WHY THIS IS THE LAST CROSS-ARCHITECTURE GAP, stated by stage:
#   RFdiffusion3 generation  ⛔ measured NOT portable: up to 2.387 A at L=208, past the 2.0 A
#                               designability threshold (dev results/44 §4b), software build excluded.
#   OpenFold3 refold         ⛔ UNMEASURABLE with our configs: the cloud yaml enables Triton and Triton
#                               cannot run on sm_86 at all, so the two cards cannot execute one path.
#   ProteinMPNN              ⇒ THIS. No kernel obstacle: same vanilla weights, same code path, either card.
#
# ⚠️ PRE-REGISTERED PREDICTION (written before the run, dev plan/115 §9.1e): it PORTS. ProteinMPNN is a
# small network with no diffusion roll-out, so shape-dependent GEMM dispatch has far less to act on than in
# RFdiffusion3's 100-to-200-step trajectory. ⛔ If it does NOT port, then no stage of our pipeline is
# cross-architecture reproducible and every A5000-against-H100 comparison in the corpus is bounded by
# hardware rather than by method, which is a larger statement than §4b alone makes.
#
# ⛔ THE ROW'S OWN DESCRIPTION WAS WRONG AND WAS CHECKED: plan/115 §9.1e says "8 backbones". The banked
# H100 run used ONE (dev results/71 §12.5). The 8 belongs to §13's concurrency run. Verified before
# building this, per root CLAUDE.md's rule about never trusting a queued row's n.
#
# WHAT IS HELD FIXED, so a difference can only be the GPU: the backbone (md5 verified against the very GCS
# object the H100 pulled), num_seqs=8, seed=42, batch_size=1, sampling_temp=0.1, model v_48_020,
# ProteinMPNN @spa-pin 8907e66, torch 2.5.1+cu124, and the production entry point scripts/eval/inverse_fold.py.
# ⚠️ Seed 42 is load-bearing: upstream's `if args.seed:` makes 0 mean "pick a fresh RANDOM seed", so a zero
# seed would look nondeterministic and be correct.
#
# ⛔ IT DOES NOT TAKE THE WORKSTATION LOCK, and that is deliberate for TWO reasons. (1) As of this run the
# lock file is STALE, naming a dead pid from a crashed row-60 attempt, so taking it proves nothing about
# contention. (2) This is one backbone at 8 sequences: seconds of GPU at roughly 2 GB, against a card with
# ~19 GB free. ⚠️ It logs VRAM before and after so the cost is on the record rather than assumed.
set -uo pipefail

PROJ="${SPA_PROJECT_ROOT:-/home/user1/projects/spa}"
REPO="$PROJ/structure-prompt-adapter"
BACKBONE_SRC="${BACKBONE_SRC:-$PROJ/outputs/_cloud_mirror/eval/threeway/prep/AF-A0A1X7NTP0-F1-model_v4_esmfold_v1.pdb}"
BACKBONE_MD5_EXPECTED="${BACKBONE_MD5_EXPECTED:-7e40f2dfce905953479f59dad196a68a}"
NUM_SEQS="${NUM_SEQS:-8}"
SEED="${SEED:-42}"
ENV_NAME="${ENV_NAME:-spa-dev}"
STAMP="$(TZ=America/Los_Angeles date +%Y-%m-%d_%H%M%S)"
OUT="${OUT:-$PROJ/outputs/_incoming/${STAMP}__mpnn_crossarch_a5000}"

# The banked H100 values, both regions identical (dev results/71 §12.5).
H100_SEQ="2cf9edc81228"
H100_FILE="3bf60fa8c11b"

log(){ echo "[$(TZ=America/Los_Angeles date +%H:%M:%S\ %Z)] $*"; }
mkdir -p "$OUT/designs" || exit 1

log "===== ProteinMPNN CROSS-ARCHITECTURE determinism: A5000 (sm_86) against banked H100 (sm_90) ====="
log "  out: $OUT"

# ---- provenance, before anything runs ----
nvidia-smi --query-gpu=name,compute_cap,memory.used,memory.total --format=csv 2>&1 | sed 's/^/    /'
log "  torch: $(conda run -n "$ENV_NAME" python -c 'import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))' 2>/dev/null)"
log "  spa code rev: $(git -C "$REPO" rev-parse HEAD)"
log "  ProteinMPNN rev: $(git -C "$PROJ/needed_repos/ProteinMPNN" rev-parse --short HEAD 2>/dev/null || echo '(no git)')"

# ⛔ Verify the INPUT before trusting any output: a different backbone would produce a different hash for a
# reason that has nothing to do with the GPU, and would read as a portability failure.
[ -r "$BACKBONE_SRC" ] || { log "FATAL: no backbone at $BACKBONE_SRC"; exit 11; }
got=$(md5sum "$BACKBONE_SRC" | cut -d' ' -f1)
if [ "$got" != "$BACKBONE_MD5_EXPECTED" ]; then
  log "⛔ FATAL: backbone md5 $got != expected $BACKBONE_MD5_EXPECTED."
  log "   The H100 half pulled gs://genomancer-spa-cache/eval/threeway/prep/. A different input makes the"
  log "   comparison meaningless, so this refuses rather than producing a hash nobody can interpret."
  exit 12
fi
log "  backbone md5 ✅ $got (matches the GCS object the H100 pulled)"
cp "$BACKBONE_SRC" "$OUT/designs/"

# ---- the production Stage-2 entry point, not a hand-rolled call ----
log "=== running scripts/eval/inverse_fold.py (num_seqs=$NUM_SEQS seed=$SEED) ==="
t0=$(date +%s)
conda run -n "$ENV_NAME" python "$REPO/scripts/eval/inverse_fold.py" \
    eval.proteinmpnn.design_dir="$OUT/designs" \
    eval.proteinmpnn.out_dir="$OUT" \
    eval.proteinmpnn.num_seqs="$NUM_SEQS" \
    eval.proteinmpnn.seed="$SEED" \
    hydra.run.dir="$OUT/hyd" > "$OUT/stdout.log" 2>&1
RC=$?
log "  inverse_fold.py rc=$RC in $(( $(date +%s) - t0 ))s"
nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | sed 's/^/    VRAM after: /'
[ $RC -eq 0 ] || { log "⛔ non-zero rc, tail of stdout:"; tail -25 "$OUT/stdout.log"; }

# ---- hashes, computed exactly as the cloud runner computes them ----
# ⛔ RECURSIVE find, NOT a glob: ProteinMPNN appends `seqs/` to out_dir itself, and a flat glob missing the
# files made a successful run report "NO FASTA PRODUCED" (dev results/71 §12.6). Third time that class bit.
: > "$OUT/HASHES.txt"
while IFS= read -r f; do
  seqonly=$(grep -v '^>' "$f" | sha256sum | cut -c1-12)
  wholefile=$(sha256sum "$f" | cut -c1-12)
  nseq=$(grep -c '^>' "$f")
  printf 'SEQ %s  FILE %s  n=%s  %s\n' "$seqonly" "$wholefile" "$nseq" "${f#$OUT/}" >> "$OUT/HASHES.txt"
done < <(find "$OUT" -type f \( -name '*.fa' -o -name '*.fasta' \) | sort)

# ⛔ An empty result must never look like a verdict (dev plan/118 §3).
if [ ! -s "$OUT/HASHES.txt" ]; then
  log "⛔ NO FASTA FOUND. This is INDETERMINATE, not a determinism result. Tree:"
  find "$OUT" -type f | head -20
  log "--- tail of stdout ---"; tail -25 "$OUT/stdout.log" 2>/dev/null
  exit 15
fi
cat "$OUT/HASHES.txt" | sed 's/^/    /'

# ---- the verdict ----
a5000_seq=$(awk '{print $2; exit}' "$OUT/HASHES.txt")
a5000_file=$(awk '{print $4; exit}' "$OUT/HASHES.txt")
{
  echo "stage: ProteinMPNN (our Stage-2, scripts/eval/inverse_fold.py)"
  echo "question: cross-ARCHITECTURE byte reproducibility, A5000 sm_86 against H100 sm_90"
  echo "held fixed: backbone md5 $BACKBONE_MD5_EXPECTED, num_seqs=$NUM_SEQS, seed=$SEED, batch_size=1,"
  echo "            temp=0.1, v_48_020, ProteinMPNN 8907e66, torch 2.5.1+cu124"
  echo "H100  (banked, us-west1 6897038391506894848 and us-central1 4547990631225491456):"
  echo "  SEQ  $H100_SEQ"
  echo "  FILE $H100_FILE"
  echo "A5000 (this run):"
  echo "  SEQ  $a5000_seq"
  echo "  FILE $a5000_file"
} > "$OUT/VERDICT.txt"

echo "########## VERDICT ##########"
if [ "$a5000_seq" = "$H100_SEQ" ] && [ "$a5000_file" = "$H100_FILE" ]; then
  v="✅ PORTS COMPLETELY: SEQ and FILE both identical to the H100. ProteinMPNN Stage-2 is cross-architecture byte-reproducible."
elif [ "$a5000_seq" = "$H100_SEQ" ]; then
  v="⚠️ SEQUENCES PORT, headers differ: SEQ matches ($a5000_seq) but FILE does not ($a5000_file against $H100_FILE). ProteinMPNN headers carry sample index, score and seed, so this is a PROVENANCE difference, not a model one."
else
  v="⛔ DOES NOT PORT: SEQ $a5000_seq against H100 $H100_SEQ. ⇒ NO stage of our pipeline is cross-architecture byte-reproducible, and every A5000-against-H100 comparison in the corpus is bounded by hardware rather than by method. This is a larger statement than results/44 §4b makes and belongs in results/71 with that framing."
fi
echo "$v" | tee -a "$OUT/VERDICT.txt"
log "artifacts: $OUT"
log "===== DONE ====="
