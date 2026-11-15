#!/usr/bin/env bash
# Shared PREAMBLE helpers for cloud eval drivers (dev plan/156 §3/§4).
#
# WHY THIS FILE EXISTS. A hand-rolled cloud driver that diverges from the proven preamble CRASHES on
# Vertex after provisioning cost, invisible to `bash -n` and DRY_RUN (dev plan/156 §4: the 2026-10-31
# sampler-path bug that ran generation at the checkpoint default sampler; the extract-job trio's
# unbound-var / missing-install / missing-LD_LIBRARY_PATH crashes). These helpers are the byte-identical
# preamble steps every GPU-doing eval driver needs; sourcing them means a clone cannot drop or subtly
# mangle them. Each driver keeps its OWN torch/triton sanity prints and its own weight/secret fetches
# inline, since those legitimately differ; only the three invariant steps are factored here.

# GPU driver-path fix (verbatim from run_smoke.sh): make the container's CUDA libs/bins resolvable.
spa_gpu_ld_path() {
  export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH:-}"
  export PATH="/usr/local/nvidia/bin:${PATH}"
  ldconfig 2>/dev/null || true
}

# Source scripts/_sampler_arm.sh and REFUSE to run if it did not load. ARM must already be set by the
# caller (ARM=ours -> ckpt 100/0.8, ARM=rfd3 -> paper 200/0.6); this expands $ARM into the SAMPLER_ARGS
# array (3 elements). FATAL on a failed source, because an unsourced arm runs generation at the
# checkpoint default rather than the requested configuration (the run_g4a 2026-10-31 bug). Pass the
# DRIVER's own path as $1 ("${BASH_SOURCE[0]}") so ../_sampler_arm.sh resolves against the driver
# directory (scripts/cloud/), i.e. scripts/_sampler_arm.sh.
spa_require_sampler_arm() {
  local driver="${1:?spa_require_sampler_arm needs the driver path (\${BASH_SOURCE[0]})}"
  # shellcheck disable=SC1090
  . "$(dirname "$driver")/../_sampler_arm.sh"
  { [ -n "${SAMPLER_ARGS+x}" ] && [ "${#SAMPLER_ARGS[@]}" -eq 3 ]; } || {
    echo "FATAL: _sampler_arm.sh did not load (wrong include path or missing file) -- refusing to run at the inherited checkpoint sampler instead of the requested ARM. See the run_g4a 2026-10-31 bug." >&2
    exit 2
  }
}

# Clone the SPA repo if absent (pinned to $REPO_REF, default main) and editable-install it. $1 = the
# repo dir (default $SPA_REPO or /opt/spa).
spa_install_repo() {
  local repo="${1:-${SPA_REPO:-/opt/spa}}"
  [ -d "$repo/.git" ] || git clone --depth 1 --branch "${REPO_REF:-main}" \
    "${REPO_URL:-https://github.com/GreggHelt2/structure-prompt-adapter}" "$repo"
  pip install -e "$repo" --no-deps -q
}
