#!/usr/bin/env bash
# ON-INSTANCE: GPU-accelerated MSA generation for the MSA-vs-MSA-free experiment (dev plan/96 §5.9).
# Runs on a Vertex Custom Job H100 (a3-highgpu-1g). CPU-search's slow half is GPU-accelerated here:
# colabfold_search --gpu 1 (MMseqs2-GPU, release >=16; full speed on Hopper). Writes ONE .a3m per input
# sequence, which OF3 consumes verbatim via main_msa_file_paths (proven: results/42 §7.8, TM 0.983).
#
# ⛔ NOT LAUNCHED / DRAFT. Open decisions marked DECISION:; things to confirm marked VERIFY:.
#
# FLOW: fetch ColabFold DBs on-instance (~1.5 TB, ~1 h at cloud bandwidth) -> makepaddedseqdb (GPU format)
#       -> THROUGHPUT PROBE on a handful (§5.8b: the one unmeasured number) -> full colabfold_search --gpu 1
#       -> push ONLY the ~12 GB of .a3m to GCS. DBs discarded with the instance (never stored locally).
set -euo pipefail
export TZ=America/Los_Angeles
say(){ echo "[$(date '+%F %T %Z')] $*"; }

# ---- inputs (env, set by submit_msa_job.sh) ----
FASTA_GCS="${FASTA_GCS:?gs:// path to the deduped input FASTA}"     # our design sequences, one FASTA
OUT_GCS="${OUT_GCS:?gs:// prefix to receive the .a3m output}"       # e.g. gs://genomancer-spa-cache/msa/2026-09-11__poster_redo
PROBE_N="${PROBE_N:-8}"                                             # §5.8b: measure per-seq rate BEFORE the full set
PROBE_ONLY="${PROBE_ONLY:-0}"                                       # 1 -> stop after the probe (gate the full run on the rate)
# ---- scratch: prefer the a3 local-NVMe RAID0 (fast, as train/cache-gen do); else pd-ssd /workspace ----
# Mirrors run_cache_gen.sh / run_train.sh: the a3 local SSD is auto-RAID0'd by the DLVM and is NOT declared
# in the Vertex spec (which only sets the pd-ssd boot disk). We detect it at runtime and PRINT a diagnostic
# so the real local-SSD size is visible. ⚠️ Our local NVMe is ~750 GB, which does NOT fit the 1.66 TB DBs,
# so this normally lands on the 2 TB pd-ssd (slower streaming; the PROBE below measures whether that hurts).
say "DISK DIAGNOSTIC:"; { lsblk -o NAME,SIZE,TYPE,MOUNTPOINT 2>/dev/null; df -h 2>/dev/null; \
  mount 2>/dev/null | grep -iE "ssd|nvme|md[0-9]|local"; } | sed 's/^/  /' || true
DB_SET="${DB_SET:-full}"
case "$DB_SET" in
  uniref30) DBS="uniref30_2302-m18v1";                              NEED_GB="${MIN_FREE_GB:-550}";;   # core; fits ~680 GB local NVMe
  full)     DBS="uniref30_2302-m18v1 colabfold_envdb_202108-m18v1"; NEED_GB="${MIN_FREE_GB:-1750}";;  # + metagenomic; needs the 2 TB pd-ssd
  *) say "FATAL: unknown DB_SET=$DB_SET (want uniref30 | full)"; exit 1;;
esac
say "DB_SET=$DB_SET  DBS='$DBS'  NEED_GB=$NEED_GB"
# Pick the FIRST writable mount with >= NEED_GB free, checking fast local NVMe first. MEASURED on the a3
# (2026-09-11): the ~750 GB local-NVMe RAID0 (md0) mounts at / and /cache (~680 GB free, too small), while
# the 2 TB pd-ssd (bootDiskSizeGb) surfaces at /var/log-storage (~2 TB free) -> the DBs land there.
freeof(){ df -BG "$1" 2>/dev/null | awk 'NR==2{gsub(/[A-Za-z]/,"",$4);print $4}'; }
if [ -z "${SCRATCH:-}" ]; then
  for cand in /mnt/local_ssd /mnt/disks/local_ssd /mnt/disks/ssd0 /cache /var/log-storage /workspace; do
    [ -d "$cand" ] && [ -w "$cand" ] || continue
    f=$(freeof "$cand"); [ -n "$f" ] || continue
    say "  candidate $cand: ${f} GB free (need ${NEED_GB})"
    if [ "$f" -ge "$NEED_GB" ]; then SCRATCH="$cand"; break; fi
  done
fi
SCRATCH="${SCRATCH:-/workspace}"
WORK="$SCRATCH/msa_work"; mkdir -p "$WORK" 2>/dev/null
DB_DIR="$WORK/colabfold_db"; MSA_OUT="$WORK/msas"; mkdir -p "$MSA_OUT" 2>/dev/null
FREE_GB=$(freeof "$SCRATCH")
say "SCRATCH=$SCRATCH  free=${FREE_GB:-?} GB  need>=${NEED_GB} GB"
if [ -z "${FREE_GB:-}" ] || [ "$FREE_GB" -lt "$NEED_GB" ]; then
  say "FATAL: no writable mount has >= ${NEED_GB} GB free -- bump DISK_GB (pd-ssd) or use a larger-local-SSD machine. Aborting before any spend."
  exit 1
fi

# ---- 0. tooling: the ESM3/RFD3 image ships NONE of these; install at runtime. ----
# MMseqs2-GPU is a CUDA STATIC BINARY (not pip); NGC CLI is a downloaded binary; ColabFold is pip.
# ⚠️ This runtime install is the least-validated part of the probe -> it runs FIRST so a failure costs
# seconds, not the ~1 h DB pull that follows.
BIN="$WORK/bin"; mkdir -p "$BIN"; export PATH="$BIN:$PATH"
# download helper: the spa-cloud image lacks wget -> curl, then wget, then python urllib.
dl(){ curl -fsSL "$1" -o "$2" 2>/dev/null || wget -q -O "$2" "$1" 2>/dev/null \
      || python3 -c "import sys,urllib.request;urllib.request.urlretrieve(sys.argv[1],sys.argv[2])" "$1" "$2"; }
unz(){ unzip -q -o "$1" -d "$2" 2>/dev/null || python3 -c "import zipfile,sys;zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$1" "$2"; }
if ! command -v mmseqs >/dev/null 2>&1; then
  say "installing MMseqs2-GPU static binary (mmseqs-linux-gpu)"
  dl https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz "$WORK/mmseqs.tar.gz" && tar xzf "$WORK/mmseqs.tar.gz" -C "$WORK"
  ln -sf "$WORK/mmseqs/bin/mmseqs" "$BIN/mmseqs"
fi
MMSEQS="${MMSEQS:-$BIN/mmseqs}"
command -v colabfold_search >/dev/null 2>&1 || { say "installing ColabFold (pip)"; pip install --quiet colabfold 2>&1 | tail -2; }
if ! command -v ngc >/dev/null 2>&1; then
  say "installing NGC CLI"
  dl https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/current/files/ngccli_linux.zip "$WORK/ngccli.zip" && unz "$WORK/ngccli.zip" "$WORK"
  ln -sf "$WORK/ngc-cli/ngc" "$BIN/ngc"
fi
# NGC auth: the SAME spa-ngc-key secret the cache-gen job used to pull CDDB from NGC.
export NGC_API_KEY="${NGC_API_KEY:-$(gcloud secrets versions access latest --secret=spa-ngc-key 2>/dev/null || true)}"
say "mmseqs=$MMSEQS  colabfold_search=$(command -v colabfold_search)  ngc=$(command -v ngc)  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
[ -x "$MMSEQS" ] || { say "FATAL: no mmseqs-gpu binary"; exit 1; }
command -v ngc >/dev/null 2>&1 || { say "FATAL: no ngc CLI"; exit 1; }
[ -n "${NGC_API_KEY:-}" ] || { say "FATAL: no NGC_API_KEY (secret spa-ngc-key) — needed to pull the NGC DBs"; exit 1; }

# ---- 1. databases: NVIDIA NGC PRE-INDEXED GPU-ready DBs -> skips setup_databases.sh AND makepaddedseqdb. ----
# NVIDIA NIM MSA-Search docs: these "come pre-indexed and optimized for GPU Server use. No additional
# indexing is required." Full depth (uniref30 + envdb), matching results/42 §2's measured alignments.
# NGC auth is already solved in this project: the spa-ngc-key secret + spa-worker secretAccessor (see
# submit_cache_job.sh prereqs; the cache-gen job pulled from NGC the same way).
if [ ! -f "$DB_DIR/.DB_OK" ]; then
  say "pulling NGC pre-indexed GPU DBs into $DB_DIR (no makepaddedseqdb needed)"
  mkdir -p "$DB_DIR"
  for DB in $DBS; do
    say "  ngc download nim/colabfold/msa-search:$DB"
    ngc registry model download-version "nim/colabfold/msa-search:$DB" --dest "$DB_DIR"
  done
  touch "$DB_DIR/.DB_OK"
fi
# VERIFY: confirm `colabfold_search --gpu 1` consumes the NGC-packaged DB layout directly (these are packaged
# for NVIDIA's MSA-Search NIM gpuserver). If not, the fallback is to run the MSA-Search NIM microservice on
# this H100 and POST query.fasta to it (it bundles these same pre-indexed DBs + the GPU search + an API).

# ---- 3. input FASTA ----
gcloud storage cp "$FASTA_GCS" "$WORK/query.fasta"
NSEQ=$(grep -c '^>' "$WORK/query.fasta"); say "query.fasta: $NSEQ sequences"

# ---- 4. THROUGHPUT PROBE (§5.8b): the number that decides hours-vs-weeks. Gate the full run on it. ----
head -$((PROBE_N*2)) "$WORK/query.fasta" > "$WORK/probe.fasta"
say "PROBE: colabfold_search --gpu 1 on $PROBE_N sequences"
t0=$(date +%s)
colabfold_search --mmseqs "$MMSEQS" --gpu 1 "$WORK/probe.fasta" "$DB_DIR" "$WORK/probe_msas"
dt=$(( $(date +%s) - t0 ))
say "PROBE done: ${dt}s for $PROBE_N seqs = $(awk "BEGIN{print $dt/$PROBE_N}") s/seq ; full set ~$(awk "BEGIN{print $dt/$PROBE_N*$NSEQ/3600}") h"
if [ "$PROBE_ONLY" = 1 ]; then say "PROBE_ONLY=1 -> stopping. Review the rate before the full run."; exit 0; fi

# ---- 5. full search ----
say "colabfold_search --gpu 1 over $NSEQ sequences"
colabfold_search --mmseqs "$MMSEQS" --gpu 1 "$WORK/query.fasta" "$DB_DIR" "$MSA_OUT"

# ---- 6. push ONLY the alignments (the durable artifact; DBs are discarded with the instance) ----
say "pushing .a3m to $OUT_GCS"
gcloud storage cp --recursive "$MSA_OUT" "$OUT_GCS/"
say "DONE. $(ls "$MSA_OUT"/*.a3m 2>/dev/null | wc -l) alignments -> $OUT_GCS"
