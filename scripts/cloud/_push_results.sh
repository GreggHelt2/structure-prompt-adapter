#!/usr/bin/env bash
# Shared RETENTION helper for cloud eval drivers (dev plan/156 §3 item 2, the durable fix).
#
# WHY THIS FILE EXISTS. A cloud run writes its backbones (.pdb), sequences (.fa) and refolds (.cif) to
# the VM's EPHEMERAL scratch; only what a driver pushes to GCS survives VM teardown. Five smoke-descended
# drivers pushed ONLY flywheel_results.json (the scored numbers), so ~15.5 H100 h of longfold/g4a
# structures were lost at teardown (dev plan/156 §1, 159a §2.4a/§2.4b). The fix is one sourced function
# every eval driver calls right after its per-item flywheel_results.json upload, so a future clone cannot
# silently drop retention the way the smoke-template clone did.
#
# CONTRACT:  spa_push_results <po> <dest_parent> <mode>
#   <po>          the flywheel out_dir for THIS item (the driver passes eval.out_dir, which equals
#                 refolder.out_dir). Verified layout: backbones at <po>/*.pdb (generate.py:1186),
#                 sequences at <po>/seqs/*.fa (proteinmpnn.py:209), refolds at <po>/of3*/.../*.cif
#                 (openfold3.py:30). mode=desig copies the whole <po> tree, so all three ride along.
#   <dest_parent> the GCS "directory" to drop this item under; the item lands at
#                 <dest_parent>/<basename po>/ for BOTH modes (so the leaf is uniform).
#   <mode>        gen   -> stage backbones only  (adherence / gen-only runs leave no .fa/.cif)
#                 desig -> stage the whole stage tree (pdb + seqs/*.fa + of3*/**/*.cif) via `cp -r`
#
# LEAN_RESULTS=1 opts OUT (metrics-JSON only), for giant sweeps where GCS object count matters. It
# defaults OFF: save everything. Same env name as the pre-existing run_variant_desig.sh:90 gate; it does
# NOT change the metrics JSON either way (that is uploaded separately by the driver).
#
# DRY_RUN (any non-empty value): PRINT the planned gcloud command instead of running it, so a driver is
# validatable with no GPU and no GCS writes (dev plan/156 §3 "validate by DRY_RUN").
#
# Never fatal: a failed structure push logs a WARN and returns 0, because the metrics JSON is already
# staged and one item's upload failure must not abort a multi-item sweep (the drivers are NOT `set -e`).

spa_push_results() {
  local po="${1:?spa_push_results needs <po> (the flywheel out_dir)}"
  local dest_parent="${2:?spa_push_results needs <dest_parent> (a GCS prefix)}"
  local mode="${3:-desig}"
  dest_parent="${dest_parent%/}"
  local leaf; leaf="$(basename "$po")"

  if [ "${LEAN_RESULTS:-0}" = "1" ]; then
    echo "[push] LEAN_RESULTS=1 -> metrics JSON only, structures NOT staged: $po"
    return 0
  fi

  local -a cmd
  case "$mode" in
    gen)   cmd=(gcloud storage cp    "$po"/*.pdb "$dest_parent/$leaf/") ;;
    desig) cmd=(gcloud storage cp -r "$po"       "$dest_parent/") ;;
    *) echo "[push] FATAL: unknown mode '$mode' (expected gen|desig)" >&2; return 2 ;;
  esac

  if [ -n "${DRY_RUN:-}" ]; then
    echo "[push][DRY_RUN] mode=$mode would run: ${cmd[*]}"
    return 0
  fi

  if "${cmd[@]}" 2>/dev/null; then
    echo "[push] staged $mode structures: $po -> $dest_parent/$leaf/"
  else
    echo "[push] WARN structure push failed: $po (metrics JSON already staged separately)"
  fi
  return 0
}
