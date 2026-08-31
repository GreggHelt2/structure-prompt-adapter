#!/usr/bin/env bash
# Resolve the RFD3 sampler ARM into Hydra overrides. Sourced by the run_*.sh drivers
# that shell out to run_flywheel.py, and by scripts/eval/run_enzyme_tier0.sh.
#
# WHY THIS IS AN ARM AND NOT THREE ENV VARS. The full rationale is in the module this reads,
# src/spa/eval/sampler_arms.py. Short version: RFD3 has two published sampler configurations and a
# run that sets the step count but not gamma_0 matches neither. run_eval.sh exposed NUM_TIMESTEPS and
# nothing else for months, which is precisely that hazard. Selecting an arm makes a partial setting
# unrepresentable, and there is exactly ONE definition of what each arm means (the Python module),
# read here rather than duplicated.
#
# USAGE, from a driver in this directory:
#     ARM="${ARM:-ours}"; . "$(dirname "${BASH_SOURCE[0]}")/_sampler_arm.sh"
#     python .../run_flywheel.py ... "${SAMPLER_ARGS[@]}" ...
#
#   ARM=ours  (default) -> 100 steps, gamma_0 0.8, step_scale 1.5   [the released checkpoint's own]
#   ARM=rfd3            -> 200 steps, gamma_0 0.6, step_scale 1.5   [the RFdiffusion3 paper's]
#
# The default is `ours`, which is NUMERICALLY IDENTICAL to inheriting from the checkpoint but is
# *stated* rather than inherited: generate.py then logs a populated `requested overrides` instead of
# the empty dict, and it ABORTS if the override failed to reach the live sampler. A completed run
# becomes its own evidence of what it ran (dev docs/results/22 §1a).

ARM="${ARM:-ours}"

# Respect $PYTHON, because a LOCAL driver runs from an activated spa-dev but a bare `python` can still
# resolve to base conda, where `spa` is not installed. The cloud drivers set no PYTHON and get the
# container's `python`, unchanged. (Found by the first end-to-end smoke of run_b1_full_local.sh.)
_sampler_arm_py="${PYTHON:-python}"

_sampler_arm_out=""
if ! _sampler_arm_out="$("$_sampler_arm_py" -c '
import sys
from spa.eval.sampler_arms import hydra_overrides
sys.stdout.write("\n".join(hydra_overrides(sys.argv[1])))
' "$ARM" 2>&1)"; then
  echo "FATAL: could not resolve sampler ARM='$ARM' using '$_sampler_arm_py'." >&2
  echo "       $_sampler_arm_out" >&2
  echo "       (is the spa package importable here? this helper is the single source of truth and" >&2
  echo "        MUST NOT be worked around by hardcoding sampler values in a driver.)" >&2
  exit 2
fi

mapfile -t SAMPLER_ARGS <<< "$_sampler_arm_out"
unset _sampler_arm_out _sampler_arm_py

# Guard against a silently empty array, which would run at the checkpoint's inherited settings while
# the log claimed an arm had been selected. Three overrides, always: num_timesteps, gamma_0, step_scale.
if [ "${#SAMPLER_ARGS[@]}" -ne 3 ]; then
  echo "FATAL: expected 3 sampler overrides for ARM='$ARM', got ${#SAMPLER_ARGS[@]}: ${SAMPLER_ARGS[*]}" >&2
  exit 2
fi

export SAMPLER_ARM="$ARM"
echo "[sampler] ARM=$ARM -> ${SAMPLER_ARGS[*]}"
