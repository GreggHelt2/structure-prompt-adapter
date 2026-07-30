"""Where does OpenFold3 stop fitting on the local 24 GB A5000? Measure the ceiling, don't infer it.

WHY. OF3's own docs state a **32 GB VRAM minimum** (A100 40 GB typical), so every local refold this project
has done runs *below spec* on empirical headroom. Exactly one length has ever been measured locally: ~19 s
per refold at **229 residues** (dev `plan/23` §143). Our eval fold sets run to **247** residues and one
prompt is a **368**-residue β-propeller, both beyond anything measured.

That matters for planning any local designability run, because the alternative to knowing is a length cap
chosen for VRAM reasons that silently becomes a scientific scope decision (dev `plan/57`).

WHAT IT DOES. Refolds one synthetic single-chain sequence at each requested length, in the real pipeline
configuration (`spa-verify-of3` env, MSA-free, the **no-kernel** runner-yaml, which is stock attention and
therefore *less* memory-frugal than a fused path). Records success or failure, wall time, and peak GPU
memory sampled from `nvidia-smi` during the run. On failure it captures the error text so an OOM is
distinguishable from a config or data problem.

Sequence CONTENT is irrelevant to a memory ceiling, only length is, so the sequences are synthetic. They
are built from a repeating pattern of common residues rather than poly-alanine, so OF3 is not handed a
degenerate input.

Usage:
    conda run -n spa-dev python scripts/eval/bench_of3_length.py --lengths 229,250,300,368,450
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]          # structure-prompt-adapter
_PROJECT = _HERE.parents[3]       # spa/, where models/ lives

# a repeating, unremarkable pattern: no homopolymer, no obvious repeats OF3 could shortcut
_MOTIF = "AEKLVGTSDIPNQRFYWMH"


def _seq(n: int) -> str:
    return (_MOTIF * (n // len(_MOTIF) + 1))[:n]


class _GpuPoller(threading.Thread):
    """Sample used VRAM while a subprocess runs. The refold happens in another conda env, so torch's
    own max_memory_allocated cannot see it; nvidia-smi can."""

    # The A5000 by stable UUID, per root CLAUDE.md: numeric indices are unreliable on this box because
    # nvidia-smi orders by PCI bus (5060=0) while CUDA defaults to FASTEST_FIRST (A5000=0). Polling all
    # GPUs and taking the max would report the DISPLAY card's usage instead of the compute card's.
    A5000_UUID = "GPU-46586b6c-b6ad-480a-4e6d-ff908b4bc3cb"

    def __init__(self, interval=0.5, gpu=None):
        super().__init__(daemon=True)
        self.gpu = gpu or self.A5000_UUID
        # NB: do NOT name this `_stop` — threading.Thread has an internal _stop() method and a
        # bool would shadow it, giving "'bool' object is not callable" from the thread machinery.
        self.interval, self.peak_mib, self._halt = interval, 0, False

    def run(self):
        while not self._halt:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "-i", self.gpu, "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout.split()
                self.peak_mib = max([self.peak_mib] + [int(v) for v in out if v.isdigit()])
            except Exception:                                    # noqa: BLE001
                pass
            time.sleep(self.interval)

    def stop(self):
        self._halt = True
        self.join(timeout=5)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", default="229,250,300,368,450",
                    help="comma-separated residue counts; 229 is the known-good control")
    ap.add_argument("--of3-ckpt", default=None)
    ap.add_argument("--of3-runner-yaml", default=None)
    ap.add_argument("--of3-conda-env", default="spa-verify-of3")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    from spa.eval.openfold3 import OF3Refolder
    from spa.eval.proteinmpnn import SequenceSet

    ckpt = a.of3_ckpt or str(_PROJECT / "models" / "openfold3" / "of3-p2-155k.pt")
    yaml = a.of3_runner_yaml or str(_REPO / "configs" / "of3" / "of3_nokernel.yml")
    out_dir = Path(a.out_dir or (_PROJECT / "outputs" / "_incoming" / "of3_length_bench"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bench] ckpt   {ckpt}\n[bench] yaml   {yaml}\n[bench] outdir {out_dir}")
    print("[bench] OF3 docs state a 32 GB minimum; this card is 24 GB, so every row is below spec.\n")

    rows = []
    for n in [int(x) for x in a.lengths.split(",")]:
        rd = out_dir / f"L{n}"
        rd.mkdir(parents=True, exist_ok=True)
        refolder = OF3Refolder(ckpt_path=ckpt, runner_yaml=yaml, out_dir=rd,
                              conda_env=a.of3_conda_env, num_diffusion_samples=1, seed=42)
        poller = _GpuPoller(); poller.start()
        t0 = time.time()
        ok, err = True, ""
        try:
            # refold() takes a SequenceSet (it reads .name/.sequences), not a bare list:
            # passing a list silently yields 'no sequences to refold' and a 0.0 s false OOM-free pass.
            res = refolder.refold(SequenceSet(name=f"L{n}", design_path=None, fasta_path=None,
                                        sequences=[_seq(n)], n_residues=n, scores=[0.0]))
            ok = bool(res) and res[0] is not None
            if not ok:
                err = "refold returned no structure (see predict_err log in the run dir)"
        except Exception as e:                                   # noqa: BLE001
            ok, err = False, f"{type(e).__name__}: {e}"
        dt = time.time() - t0
        poller.stop()
        oom = any(k in err.lower() for k in ("out of memory", "oom", "cuda error"))
        rows.append({"length": n, "ok": ok, "seconds": round(dt, 1),
                     "peak_vram_mib": poller.peak_mib, "oom_suspected": oom,
                     "error": err[:400]})
        flag = "OK " if ok else ("OOM" if oom else "FAIL")
        print(f"[bench] L={n:>4}  {flag}  {dt:>6.1f} s  peak {poller.peak_mib:>6} MiB"
              + ("" if ok else f"\n           {err[:200]}"))

    print(f"\n{'len':>5} {'result':>7} {'sec':>7} {'peakMiB':>9}")
    for r in rows:
        print(f"{r['length']:>5} {('OK' if r['ok'] else ('OOM' if r['oom_suspected'] else 'FAIL')):>7} "
              f"{r['seconds']:>7.1f} {r['peak_vram_mib']:>9}")
    good = [r["length"] for r in rows if r["ok"]]
    bad = [r["length"] for r in rows if not r["ok"]]
    print(f"\n[bench] ⭐ largest length that FOLDED locally: {max(good) if good else 'none'}")
    if bad:
        print(f"[bench] ⛔ failed at: {bad}  => the local ceiling sits between "
              f"{max(good) if good else 0} and {min(bad)} residues")
    else:
        print("[bench] no failures in the tested range; the ceiling is ABOVE the largest length tried, "
              "so it is still unmeasured rather than absent")

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({"rows": rows, "ckpt": ckpt, "runner_yaml": yaml}, indent=2))
        print(f"[bench] wrote {a.json}")


if __name__ == "__main__":
    main()
