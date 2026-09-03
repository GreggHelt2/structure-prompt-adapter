"""A cross-process mutex over the GENERATION stage, so concurrent shards cannot collide on VRAM.

⛔ THE FAILURE THIS PREVENTS, measured twice in two days on the A5000. Peak VRAM is about 5x sustained
and lives entirely in generation: 16,253 MiB while RFD3 samples at K=16/L=227, against 3,202 MiB while
refolding (dev ``plan/85`` §4f). Three streams refolding is ~9.6 GB and fits comfortably; **two streams
GENERATING is ~30 GB and cannot**. So a sharded run is safe almost always and dies only when two
generation phases happen to overlap:

  * 2026-09-02 05:30, ``A0A1Q8BPK6`` died **21 s** into generation, 17-prompt run;
  * 2026-09-02 18:16, ``A0A2X2KHU0`` died **22 s** into generation, 9-prompt run.

⭐ WHY STAGGERING THE SHARD STARTS DOES NOT FIX IT, which is the lesson that produced this module.
``plan/85`` §4f recommended a start offset (the drivers used 240 s) so generation phases would not align.
**That does not survive drift.** In the 2026-09-02 run the three shards were deliberately BALANCED for
makespan, which keeps them in phase by construction, and prompts were ordered ascending by length, which
puts the longest ones in the final slot. The three shards entered their third prompt at 18:07, 18:16 and
18:20, i.e. **within 13 minutes of each other after 2.5 hours**, and a 19-minute generation phase swallowed
a 4-minute offset whole. Balancing for throughput and staggering for safety are in direct conflict.

✅ WHAT THIS DOES INSTEAD: serialize only the ~10-23% of a cycle that is generation, and let refolding,
which is 90% of wall-clock and fits three-wide, stay fully concurrent. The bound is therefore small and
the failure class is gone rather than made less likely.

⚠️ COST, stated honestly. With P shards the worst-case wait is (P-1) generation phases. At K=16/N=4 that
is roughly 15 to 20 minutes across a 4.5 h run, against the ~75 minutes a single OOM costs in retry plus
the human attention to notice it. It is a **bounded, deterministic** cost replacing an unbounded,
random one.

DEFAULT ON, deliberately. An uncontended lock costs microseconds, so a single-stream run is unaffected,
and a sharded run gets protection without anyone remembering to ask (dev ``plan/84``: what must be
remembered will not be). Escape hatches:

    SPA_GEN_LOCK=0                      disable entirely
    SPA_GEN_LOCK=/path/to/other.lock    use a different lock file (separate GPUs, separate locks)
    SPA_GEN_LOCK_TIMEOUT=<seconds>      how long to wait before giving up (default 3600)

⚠️ ON TIMEOUT IT PROCEEDS WITHOUT THE LOCK rather than failing. A multi-hour run must not hang because a
holder wedged; the worst case is then exactly today's behaviour, which is a risk of OOM, not a certainty.

The lock is released by the OS when the holder exits, so a crashed shard cannot wedge the others.
"""

from __future__ import annotations

import errno
import fcntl
import os
import time
from contextlib import contextmanager

DEFAULT_TIMEOUT = 3600.0
POLL = 5.0


def _lock_path() -> str | None:
    """Resolve the lock file, or None when disabled."""
    raw = os.environ.get("SPA_GEN_LOCK")
    if raw is not None:
        v = raw.strip()
        if v in ("0", "", "off", "false", "no"):
            return None
        if v not in ("1", "on", "true", "yes"):
            return v                      # an explicit path
    root = os.environ.get("SPA_OUTPUTS_ROOT") or os.path.join(
        os.environ.get("SPA_PROJECT_ROOT", os.path.expanduser("~/projects/spa")), "outputs")
    return os.path.join(root, "_meta", "generation.lock")


@contextmanager
def generation_lock(label: str = ""):
    """Hold an exclusive lock for the duration of the generation stage.

    No-op when disabled or when the lock file cannot be created (a missing outputs root must never
    stop a run). Logs the wait, so a shard that queued is visible in its driver log rather than
    looking mysteriously slow.
    """
    path = _lock_path()
    if path is None:
        yield
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fh = open(path, "a+")
    except OSError as exc:
        print(f"[gpu_lock] cannot open {path} ({exc}); proceeding WITHOUT the generation lock.",
              flush=True)
        yield
        return

    timeout = float(os.environ.get("SPA_GEN_LOCK_TIMEOUT", DEFAULT_TIMEOUT))
    t0 = time.time()
    held = False
    while True:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            held = True
            break
        except OSError as exc:
            if exc.errno not in (errno.EAGAIN, errno.EACCES):
                raise
            waited = time.time() - t0
            if waited >= timeout:
                print(f"[gpu_lock] ⚠️  waited {waited/60:.1f} min for the generation lock and gave up; "
                      f"proceeding WITHOUT it. OOM risk is back to baseline. holder={_holder(path)}",
                      flush=True)
                break
            if waited < POLL:               # announce once, on the first failed attempt
                print(f"[gpu_lock] another stream is generating (holder={_holder(path)}); "
                      f"waiting up to {timeout/60:.1f} min{f' [{label}]' if label else ''}.", flush=True)
            # Never sleep past the deadline: with a fixed POLL a short timeout would be overshot by up
            # to POLL seconds, so the timeout would not mean what it says. Irrelevant at the 3600 s
            # default, but the parameter should be honest at any value.
            time.sleep(max(0.05, min(POLL, timeout - waited)))

    if held:
        waited = time.time() - t0
        try:
            fh.seek(0)
            fh.truncate()
            fh.write(f"{os.getpid()} {label} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            fh.flush()
        except OSError:
            pass
        if waited >= POLL:
            print(f"[gpu_lock] acquired after waiting {waited/60:.1f} min.", flush=True)
    try:
        yield
    finally:
        try:
            if held:
                fcntl.flock(fh, fcntl.LOCK_UN)
            fh.close()
        except OSError:
            pass


def _holder(path: str) -> str:
    try:
        with open(path) as fh:
            return fh.read().strip() or "unknown"
    except OSError:
        return "unknown"
