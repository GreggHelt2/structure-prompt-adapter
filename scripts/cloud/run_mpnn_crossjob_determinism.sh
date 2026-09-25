#!/usr/bin/env bash
# In-container: is OUR ProteinMPNN stage byte-reproducible ACROSS JOBS on the cloud H100?
# Dev docs/results/71 §12 closed RFdiffusion3 and OpenFold3; this is the third leg. Runs in spa-combined.
#
# ⭐ WHY IT IS A GAP RATHER THAN A KNOWN. plan/91 found ProteinMPNN already bit-identical on CPU and CUDA
# at a fixed non-zero seed, ⚠️ but that was LOCAL, on the A5000. results/71 §8 did test determinism on the
# cloud, ⛔ but that was ANTHROPIC's ProteinMPNN package inside THEIR image, invoked through their run.sh.
# Neither covers our own Stage-2 in our own image across two containers, which is what every multi-job
# cloud comparison assumes.
#
# ⛔ NO BANKED BASELINE EXISTS, unlike the other two stages: §12's RFD3 arm had test D's hashes to
# reproduce and its OpenFold3 arm had the precision bench's. For ProteinMPNN there is nothing to compare
# against, so ⇒ THE COMPARISON IS BETWEEN TWO SUBMISSIONS OF THIS SAME RUNNER, one per region. That is
# also why it yields cross-REGION for free, as §12's OpenFold3 arm did by accident of the quota.
#
# ⭐ USES THE PRODUCTION ENTRY POINT, NOT A BESPOKE INVOCATION: scripts/eval/inverse_fold.py, the Hydra
# Stage-2 script the flywheel path uses. A hand-rolled ProteinMPNN call would measure something we do not
# run. Seed comes from the config default (42), which is load-bearing: `--seed 0` means "pick a fresh
# random seed" upstream (results/01 §4), so a zero seed here would look nondeterministic and be correct.
#
# ⚠️ WHAT A DIFFERENCE WOULD AND WOULD NOT MEAN. ProteinMPNN's `batch_size` is a RESULTS-affecting knob,
# not a throughput one (plan/91), and probe_mpnn_batch_composition.py asks the separate question of whether
# output depends on the batch composition. This arm holds the backbone set, N, temperature, seed and batch
# size all FIXED, so a difference could only come from the container, the host or the GPU instance.
set -uo pipefail

BUCKET="${BUCKET:-gs://genomancer-spa-cache}"
PREP_URI="${PREP_URI:-$BUCKET/eval/threeway/prep}"
BACKBONE="${BACKBONE:-AF-A0A1X7NTP0-F1-model_v4_esmfold_v1.pdb}"
SPA_REPO="${SPA_REPO:-/opt/spa}"
NUM_SEQS="${NUM_SEQS:-8}"
SEED="${SEED:-42}"
OUT="${OUT:-/workspace/mpnn_crossjob}"
GCS_OUT="${GCS_OUT:-$BUCKET/eval/mpnn_crossjob/$(date -u +%Y%m%d-%H%M%S)}"
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

# ⛔ Driver paths: run_determinism_h100.sh:41-42 has exported these for months; a trimmed preflight
# reports cuda_available False on a working H100 (dev results/71 §4.1).
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/nvidia/bin:${PATH}"; ldconfig 2>/dev/null || true
nvidia-smi --query-gpu=name,compute_cap --format=csv 2>&1 | head -2 || log "note: nvidia-smi unavailable (advisory)"
python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))" \
  || { log "FATAL: torch cannot see the GPU"; exit 10; }

# ⛔ No git fetch: BOOT_CHECKOUT already pinned the rev (_pin_run_env.sh:80). Re-fetching main would
# replace that pin with whatever main is at job start, which is a race dressed as provenance.
[ -d "$SPA_REPO/.git" ] || { log "FATAL: no repo at $SPA_REPO"; exit 12; }
log "PINNED_CODE_REV: $(git -C "$SPA_REPO" rev-parse HEAD)"
pip install -e "$SPA_REPO" --no-deps -q
python -c "import spa; print('spa OK')" || { log "FATAL: spa not importable"; exit 13; }

MPNN_REPO="${MPNN_REPO:-/opt/ProteinMPNN}"
[ -d "$MPNN_REPO" ] || git clone --depth 1 https://github.com/dauparas/ProteinMPNN "$MPNN_REPO"
log "ProteinMPNN repo: $(git -C "$MPNN_REPO" rev-parse --short HEAD 2>/dev/null || echo '(no git)')"

mkdir -p "$OUT/designs" "$OUT/seqs"
gcloud storage cp "$PREP_URI/$BACKBONE" "$OUT/designs/$BACKBONE" || { log "FATAL: no backbone"; exit 11; }
log "backbone sha256: $(sha256sum "$OUT/designs/$BACKBONE" | cut -c1-16)  ($(wc -l < "$OUT/designs/$BACKBONE") lines)"

log "=== running the PRODUCTION Stage-2 entry point ==="
python "$SPA_REPO/scripts/eval/inverse_fold.py" \
    eval.proteinmpnn.design_dir="$OUT/designs" \
    eval.proteinmpnn.out_dir="$OUT/seqs" \
    eval.proteinmpnn.num_seqs="$NUM_SEQS" \
    eval.proteinmpnn.seed="$SEED" \
    paths.proteinmpnn_repo="$MPNN_REPO" \
    hydra.run.dir="$OUT/hyd" > "$OUT/stdout.log" 2>&1
RC=$?
log "inverse_fold.py rc=$RC"
[ $RC -eq 0 ] || { log "⛔ FAILED, tail:"; tail -25 "$OUT/stdout.log"; }

log "=== the canonical hash ==="
# ⛔ HASH THE SEQUENCES, NOT THE FILE. ProteinMPNN's FASTA headers carry the sample index, the score and
# the seed, all of which are legitimately part of the result, ⚠️ but they can also carry a wall-clock or a
# path that differs between containers for reasons that are not the model. So: emit the SEQUENCE LINES
# only, in order, and hash those. Also emit the full-file hash separately so a header-only difference is
# distinguishable from a sequence difference rather than being silently folded together.
: > "$OUT/HASHES.txt"
shopt -s nullglob
for f in "$OUT/seqs"/*.fa "$OUT/seqs"/*.fasta; do
  seqonly=$(grep -v '^>' "$f" | sha256sum | cut -c1-12)
  wholefile=$(sha256sum "$f" | cut -c1-12)
  nseq=$(grep -c '^>' "$f")
  printf 'SEQ %s  FILE %s  n=%s  %s\n' "$seqonly" "$wholefile" "$nseq" "$(basename "$f")" >> "$OUT/HASHES.txt"
done
if [ ! -s "$OUT/HASHES.txt" ]; then
  # ⛔ An empty result must never look like a verdict (dev plan/118 §3).
  log "⛔ NO FASTA PRODUCED. This is INDETERMINATE, not a determinism result. Contents of $OUT/seqs:"
  ls -la "$OUT/seqs" 2>&1 | head
  exit 15
fi
cat "$OUT/HASHES.txt"
log "⇒ Compare the SEQ hash against the other region's job. Identical SEQ with differing FILE means only"
log "  the headers moved, which is a provenance difference, not a model one."

log "=== persisting to GCS ==="
gcloud storage cp -r "$OUT" "$GCS_OUT/" >/dev/null 2>&1 && log "artifacts at $GCS_OUT" || log "could not write $GCS_OUT"
log "=== DONE ==="
