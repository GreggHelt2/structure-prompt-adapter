#!/usr/bin/env bash
# ON-INSTANCE: GPU-accelerated MSA generation for the MSA-vs-MSA-free experiment (dev plan/96 §5.9).
# Runs on a Vertex Custom Job H100 (a3-highgpu-1g). CPU-search's slow half is GPU-accelerated here:
# colabfold_search --gpu 1 (MMseqs2-GPU, release >=16; full speed on Hopper). Writes ONE .a3m per input
# sequence, which OF3 consumes verbatim via main_msa_file_paths (proven: results/42 §7.8, TM 0.983).
#
# Things to confirm are marked VERIFY:. The NGC download path is validated (CLI + spa-ngc-key + model
# resolve, 41 MB pulled in 20 s, 2026-09-11); the probe itself has not completed end to end yet.
#
# DB SOURCE (DB_SRC): the NGC DBs come pre-indexed for GPU, so NO makepaddedseqdb step. Two sources:
#   ngc  -> pull from NVIDIA NGC (~1 h full / ~15 min uniref30). Set DB_GCS to ALSO mirror the DB to GCS.
#   gcs  -> hydrate the DB from that GCS cache instead (same-region, ~minutes, free egress).
# CACHE_ONLY=1 (with DB_SRC=ngc + DB_GCS) makes this a one-time NGC->GCS cache job: pull, mirror, STOP.
#
# FLOW (search run): place DB (ngc pull or gcs hydrate) -> THROUGHPUT PROBE on a handful
#   (§5.8b: the one unmeasured number) -> full colabfold_search --gpu 1 -> push ONLY the ~12 GB of .a3m
#   to OUT_GCS. The DB itself is kept only in the GCS cache (via CACHE_ONLY), never re-pulled per run.
set -euo pipefail
export TZ=America/Los_Angeles
say(){ echo "[$(date '+%F %T %Z')] $*"; }

# ---- inputs (env, set by submit_msa_job.sh) ----
FASTA_GCS="${FASTA_GCS:?gs:// path to the deduped input FASTA}"     # our design sequences, one FASTA
OUT_GCS="${OUT_GCS:?gs:// prefix to receive the .a3m output}"       # e.g. gs://genomancer-spa-cache/msa/2026-09-11__poster_redo
PROBE_N="${PROBE_N:-8}"                                             # §5.8b: measure per-seq rate BEFORE the full set
PROBE_ONLY="${PROBE_ONLY:-0}"                                       # 1 -> stop after the probe (gate the full run on the rate)
# ---- DB source: pull the pre-indexed GPU DBs from NGC (default) OR hydrate from our GCS cache (fast, same-region). ----
DB_SRC="${DB_SRC:-ngc}"                                             # ngc | gcs. gcs = gcloud storage cp from DB_GCS (~minutes vs ~1h from NGC)
DB_GCS="${DB_GCS:-}"                                                # gs:// prefix for the cached GPU-indexed DBs (e.g. gs://genomancer-spa-cache/msa/db).
                                                                    #   used as the SOURCE when DB_SRC=gcs, and as the cache TARGET after an NGC pull when set + DB_SRC=ngc
CACHE_ONLY="${CACHE_ONLY:-0}"                                       # 1 -> after the DB is placed and cached to DB_GCS, STOP (the one-time NGC->GCS cache run; no search)
PROJECT="${PROJECT:-spa-dev-499900}"                               # for gcloud secrets access on the instance (bare container has no default project)
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
BIN="$WORK/bin"; mkdir -p "$BIN"; export PATH="$BIN:/usr/local/nvidia/bin:$PATH"  # nvidia-smi lives under /usr/local/nvidia/bin on the DLVM
# CACHE_ONLY / DB_SRC=gcs need only ngc+gcloud (cache) or gcloud (hydrate), NOT the GPU search tools.
NEED_SEARCH=1; [ "$CACHE_ONLY" = 1 ] && NEED_SEARCH=0
# download helper: the spa-cloud image lacks wget -> curl, then wget, then python urllib.
dl(){ curl -fsSL "$1" -o "$2" 2>/dev/null || wget -q -O "$2" "$1" 2>/dev/null \
      || python3 -c "import sys,urllib.request;urllib.request.urlretrieve(sys.argv[1],sys.argv[2])" "$1" "$2"; }
unz(){ unzip -q -o "$1" -d "$2" 2>/dev/null || python3 -c "import zipfile,sys;zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$1" "$2"; }
if [ "$NEED_SEARCH" = 1 ]; then
  if ! command -v mmseqs >/dev/null 2>&1; then
    say "installing MMseqs2-GPU static binary (mmseqs-linux-gpu)"
    dl https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz "$WORK/mmseqs.tar.gz" && tar xzf "$WORK/mmseqs.tar.gz" -C "$WORK"
    ln -sf "$WORK/mmseqs/bin/mmseqs" "$BIN/mmseqs"
  fi
  # the spa-msa image bakes mmseqs at /usr/local/bin/mmseqs; default to whatever is on PATH, NOT the (possibly absent) install path.
  MMSEQS="${MMSEQS:-$(command -v mmseqs || echo "$BIN/mmseqs")}"
  command -v colabfold_search >/dev/null 2>&1 || { say "installing ColabFold (pip)"; pip install --quiet colabfold 2>&1 | tail -2; }
fi
# NGC CLI + key are needed only when pulling FROM ngc (DB_SRC=ngc, including the cache-once run).
if [ "$DB_SRC" = ngc ]; then
  if ! command -v ngc >/dev/null 2>&1; then
    say "installing NGC CLI"
    dl https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.64.2/files/ngccli_linux.zip "$WORK/ngccli.zip" && unz "$WORK/ngccli.zip" "$WORK"
    ln -sf "$WORK/ngc-cli/ngc" "$BIN/ngc"
  fi
  # NGC auth: the SAME spa-ngc-key secret + pattern the cache-gen job uses (run_cache_gen.sh:58).
  # ⚠️ --project is REQUIRED: a bare Vertex container has no default gcloud project, so without it the
  # fetch errors and (previously, silenced by 2>/dev/null) left the key empty. The NGC CLI reads
  # NGC_CLI_API_KEY, not NGC_API_KEY, so set that one (keep NGC_API_KEY too for this script's guard).
  if [ -z "${NGC_CLI_API_KEY:-}" ]; then
    NGC_CLI_API_KEY="$(gcloud secrets versions access latest --secret=spa-ngc-key --project="$PROJECT" 2>"$WORK/secret.err")" \
      || { say "FATAL: could not read secret spa-ngc-key (project $PROJECT): $(cat "$WORK/secret.err" 2>/dev/null)"; exit 1; }
    export NGC_CLI_API_KEY
  fi
  export NGC_API_KEY="${NGC_API_KEY:-$NGC_CLI_API_KEY}"
  # NGC org: the colabfold NIM model's entitlement is ORG-SCOPED (cache-gen's org=nvidia works only for
  # public NVIDIA resources like CDDB). Fetched from Secret Manager (spa-ngc-org) to keep the private org
  # id out of the public repo; the download fails with "Missing org" without it.
  export NGC_CLI_ORG="${NGC_CLI_ORG:-$(gcloud secrets versions access latest --secret=spa-ngc-org --project="$PROJECT" 2>/dev/null || true)}"
  [ -n "${NGC_CLI_ORG:-}" ] || { say "FATAL: no NGC_CLI_ORG (secret spa-ngc-org): the NIM model download needs an org"; exit 1; }
fi
GPU="$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo n/a)"
say "DB_SRC=$DB_SRC CACHE_ONLY=$CACHE_ONLY  mmseqs=${MMSEQS:-n/a}  colabfold_search=$(command -v colabfold_search || echo n/a)  ngc=$(command -v ngc || echo n/a)  GPU=$GPU"
if [ "$NEED_SEARCH" = 1 ]; then [ -x "${MMSEQS:-}" ] || { say "FATAL: no mmseqs-gpu binary"; exit 1; }; fi
if [ "$DB_SRC" = ngc ]; then
  command -v ngc >/dev/null 2>&1 || { say "FATAL: no ngc CLI"; exit 1; }
  [ -n "${NGC_API_KEY:-}" ] || { say "FATAL: no NGC_API_KEY (secret spa-ngc-key): needed to pull the NGC DBs"; exit 1; }
fi
if { [ "$DB_SRC" = gcs ] || [ "$CACHE_ONLY" = 1 ]; } && [ -z "${DB_GCS:-}" ]; then
  say "FATAL: DB_GCS must be a gs:// prefix when DB_SRC=gcs or CACHE_ONLY=1"; exit 1
fi

# ---- 1. databases: NGC pre-indexed GPU DBs, OR hydrate a one-time copy from our GCS cache. ----
# These "come pre-indexed and optimized for GPU Server use" (NVIDIA NIM MSA-Search docs), so no
# setup_databases.sh / makepaddedseqdb. Two sources, selected by DB_SRC:
#   ngc  -> pull from NVIDIA NGC (~1 h for the full set at cloud bandwidth; the spa-ngc-key secret auth).
#   gcs  -> gcloud rsync a one-time copy from our same-region GCS cache (~minutes, free egress).
# The GPU-indexed files are byte-identical through a GCS round-trip, so colabfold_search consumes either.
if [ ! -f "$DB_DIR/.DB_OK" ]; then
  mkdir -p "$DB_DIR"
  if [ "$DB_SRC" = gcs ]; then
    say "hydrating GPU-indexed DBs from GCS cache $DB_GCS/$DB_SET (same-region, fast)"
    # refuse to hydrate a missing OR INCOMPLETE cache: .DB_OK is written LAST by the cache push, so its
    # presence means the whole DB is up. Without this a mid-push cache would hydrate partially.
    gcloud storage ls "$DB_GCS/$DB_SET/colabfold_db/.DB_OK" >/dev/null 2>&1 \
      || { say "FATAL: no .DB_OK at $DB_GCS/$DB_SET/colabfold_db/ — cache missing or still uploading (run CACHE_ONLY=1 DB_SRC=ngc first and let it reach SUCCEEDED)"; exit 1; }
    t=$(date +%s)
    gcloud storage rsync --recursive "$DB_GCS/$DB_SET/colabfold_db" "$DB_DIR" \
      || { say "FATAL: GCS hydrate failed from $DB_GCS/$DB_SET/colabfold_db"; exit 1; }
    say "  hydrate done in $(( $(date +%s) - t ))s"
  else
    say "pulling NGC pre-indexed GPU DBs into $DB_DIR (no makepaddedseqdb needed)"
    for DB in $DBS; do
      say "  ngc download nim/colabfold/msa-search:$DB"
      t=$(date +%s)
      ngc registry model download-version "nim/colabfold/msa-search:$DB" --dest "$DB_DIR"
      say "  $DB pulled in $(( $(date +%s) - t ))s"
    done
  fi
  touch "$DB_DIR/.DB_OK"
fi

# ---- 1a. one-time cache: after an NGC pull, mirror the GPU-indexed DB tree to GCS so later runs hydrate. ----
# ⚠️ .DB_OK is the completion sentinel and MUST be uploaded LAST: a plain rsync of the whole dir uploads
# it early (arbitrary order), so it can appear while the big DB blobs are still uploading, and a
# DB_SRC=gcs run would then hydrate a partial cache. So: rsync everything WITHOUT the sentinel, then cp
# the sentinel as the final object.
if [ "$DB_SRC" = ngc ] && [ -n "${DB_GCS:-}" ]; then
  DST="$DB_GCS/$DB_SET/colabfold_db"
  if gcloud storage ls "$DST/.DB_OK" >/dev/null 2>&1; then
    say "DB already cached at $DST/ (.DB_OK present): skipping push"
  else
    say "caching GPU-indexed DB tree to $DST/ (one-time; $(du -sh "$DB_DIR" 2>/dev/null | cut -f1) local)"
    t=$(date +%s)
    mv -f "$DB_DIR/.DB_OK" "$WORK/.DB_OK.pending" 2>/dev/null || true   # keep the sentinel OUT of the bulk rsync
    gcloud storage rsync --recursive "$DB_DIR" "$DST" \
      || { say "FATAL: cache push (rsync) to $DST/ failed"; exit 1; }
    mv -f "$WORK/.DB_OK.pending" "$DB_DIR/.DB_OK" 2>/dev/null || touch "$DB_DIR/.DB_OK"
    gcloud storage cp "$DB_DIR/.DB_OK" "$DST/.DB_OK" \
      || { say "FATAL: cache sentinel upload failed (DB is up but unmarked; a gcs hydrate will refuse it)"; exit 1; }
    say "  cache push done in $(( $(date +%s) - t ))s -> $DST/  (.DB_OK written LAST)"
  fi
fi
if [ "$CACHE_ONLY" = 1 ]; then
  say "CACHE_ONLY=1 -> DB cached to $DB_GCS/$DB_SET/. Stopping before any search. DONE."
  exit 0
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
