"""Sample peak VRAM of a SUBPROCESS, on any platform, and say so when it cannot.

WHY THIS MODULE EXISTS. A refold or a generation runs in another conda env, so this process's
``torch.cuda.max_memory_allocated()`` cannot see it; ``nvidia-smi`` can. Two near-identical pollers had
grown in ``scripts/eval/bench_of3_length.py`` and ``scripts/eval/bench_m10_overhead.py``, and
⛔ **each was broken on the platform the other handled:**

* ``bench_of3_length`` hardcoded the local A5000's UUID, so on a cloud H100 ``nvidia-smi -i <uuid>``
  failed, a bare ``except: pass`` swallowed it, and every row reported ``0 MiB`` while looking fine.
  Measured 2026-09-23 on Vertex job ``5803754397590618112``: eight rows, all zero, no warning.
* ``bench_m10_overhead`` passed no ``-i`` at all and took the max over ALL GPUs, which on the local
  dual-GPU box can report the **display** card (the RTX 5060) instead of the compute card.

⭐ Two lessons are encoded here rather than left to be rediscovered:

1. **A hardcoded device UUID is a portability bug, and root ``CLAUDE.md`` already forbids it**
   ("Device selection must come from config/env, never a hardcoded UUID"). The UUID belongs to one
   workstation; this code runs on three platforms.
2. ⛔ **``0`` is a valid-looking measurement and must never stand for "I could not measure".** The old
   pollers initialised ``peak_mib = 0`` and returned it on total failure, so a broken poller was
   indistinguishable from an idle GPU. :attr:`GpuPoller.peak_mib` is ``None`` until a sample succeeds,
   and :meth:`GpuPoller.stop` warns loudly if none ever did. This is the same shape as an omitted
   opt-in: the evidence a reader wants is ABSENT rather than wrong, which no value check can catch.

Usage::

    p = GpuPoller(); p.start()
    ...run the subprocess...
    p.stop()
    p.peak_mib     # int MiB, or None meaning NOT MEASURED
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
import warnings

_UUID_RE = re.compile(r"^GPU-[0-9a-fA-F-]{8,}$")


def resolve_gpu_selector(explicit: str | None = None) -> tuple[str | None, str]:
    """Pick what to pass to ``nvidia-smi -i``, and say how it was chosen.

    Returns ``(selector, why)``. ``selector`` of ``None`` means "poll the only GPU present"; a
    :class:`RuntimeError` is raised when the machine has several GPUs and nothing says which is the
    compute card, because guessing is how the display card gets measured.
    """
    if explicit:
        return explicit, "explicit argument"

    # The local convention bakes the compute card's UUID into each conda env (root CLAUDE.md), so this
    # is both the right answer and self-documenting when present.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd and _UUID_RE.match(cvd.split(",")[0]):
        return cvd.split(",")[0], "CUDA_VISIBLE_DEVICES (UUID form)"

    uuids = _visible_gpu_uuids()
    if len(uuids) == 1:
        return uuids[0], "the only GPU present"
    if not uuids:
        raise RuntimeError("nvidia-smi listed no GPUs, so VRAM cannot be polled")
    # ⛔ Deliberately a refusal, not a max(). See the module docstring's second bullet.
    raise RuntimeError(
        f"{len(uuids)} GPUs present and no selector given. Polling all of them and taking the max "
        f"can report the DISPLAY card instead of the compute card. Set CUDA_VISIBLE_DEVICES to the "
        f"compute GPU's UUID, or pass one explicitly. Seen: {uuids}"
    )


def _visible_gpu_uuids() -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return []
    return [l.strip() for l in out.stdout.splitlines() if l.strip().startswith("GPU-")]


class GpuPoller(threading.Thread):
    """Poll ``memory.used`` on one GPU until :meth:`stop`. ``peak_mib`` is ``None`` if never sampled."""

    def __init__(self, interval: float = 0.5, gpu: str | None = None, *, quiet: bool = False):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_mib: int | None = None          # ⛔ None, NOT 0: see the module docstring
        self.samples = 0
        self.quiet = quiet
        self._halt = False                        # NB: never name this `_stop`; Thread has that method
        try:
            self.gpu, self.why = resolve_gpu_selector(gpu)
            self.error: str | None = None
        except RuntimeError as e:
            self.gpu, self.why, self.error = None, "unresolved", str(e)

    def run(self) -> None:
        if self.error:
            return
        cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        if self.gpu:
            cmd = ["nvidia-smi", "-i", self.gpu, *cmd[1:]]
        first_failure: str | None = None
        while not self._halt:
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                vals = [int(v) for v in r.stdout.split() if v.isdigit()]
                if vals:
                    self.peak_mib = max(vals + ([self.peak_mib] if self.peak_mib is not None else []))
                    self.samples += 1
                elif first_failure is None:
                    first_failure = (r.stderr or r.stdout or "nvidia-smi produced no number").strip()
            except (OSError, subprocess.SubprocessError, ValueError) as e:
                if first_failure is None:
                    first_failure = f"{type(e).__name__}: {e}"
            time.sleep(self.interval)
        if self.samples == 0 and self.error is None:
            self.error = first_failure or "no samples and no error captured"

    def stop(self) -> int | None:
        """Halt, join, and WARN if nothing was ever measured. Returns ``peak_mib``."""
        self._halt = True
        self.join(timeout=5)
        if self.peak_mib is None and not self.quiet:
            warnings.warn(
                f"VRAM was NOT MEASURED (selector={self.gpu!r} via {self.why}): {self.error}. "
                f"Reporting None rather than 0, because 0 would read as a real measurement.",
                RuntimeWarning, stacklevel=2,
            )
        return self.peak_mib


def fmt_mib(v: int | None) -> str:
    """Render a peak for a table: an explicit marker when unmeasured, never a bare 0."""
    return "n/a" if v is None else str(v)
