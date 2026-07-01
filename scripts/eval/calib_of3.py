"""One-off H100 calibration: OF3 refold wall-time vs design length, to pin the cloud-eval cost model.

The cloud-eval cost is dominated by OpenFold3 refolds, whose time is length-driven. The smoke measured
only len 80 (~5–8 s/fold); everything ≥200 res was extrapolated. This times ``OF3Refolder.refold_all``
on **synthetic random sequences** (fold time depends on LENGTH, not on realness/conditioning), which
isolates OF3 from RFD3/MPNN. For each length it runs **two fold-counts** so the fixed model-load
cancels out:  ``per_fold = (t[nhi] - t[nlo]) / (nhi - nlo)``  and  ``load = t[nlo] - nlo*per_fold``.
A length whose folds OOM is recorded (that itself bounds the max cloud-refoldable length).

Usage:  python calib_of3.py <of3_ckpt> <runner_yaml> <out_dir> <comma_lengths>
"""

from __future__ import annotations

import json
import random
import sys
import time
import types

from spa.eval.openfold3 import OF3Refolder

AA = "ACDEFGHIKLMNPQRSTVWY"
LOAD_GUESS = 45.0  # fallback (s) if only one fold-count succeeds at a length
NLO, NHI = 2, 8    # the two fold-counts per length (their difference cancels model-load)


def fake_seq(length: int, seed: int) -> str:
    random.seed(seed)
    return "".join(random.choice(AA) for _ in range(length))


def sset(name: str, seqs: list[str]):
    return types.SimpleNamespace(name=name, sequences=seqs)


def main() -> None:
    ckpt, yml, out, lengths = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    LENGTHS = [int(x) for x in lengths.split(",")]
    rf = OF3Refolder(ckpt_path=ckpt, runner_yaml=yml, out_dir=out, conda_env="spa-verify-of3")

    def timed(length: int, n: int, seed0: int):
        ss = [sset(f"L{length}_n{n}", [fake_seq(length, seed0 + i) for i in range(n)])]
        t = time.time()
        try:
            got = rf.refold_all(ss)
            dt = time.time() - t
            cnt = sum(len(v) for v in got.values())
            ok = cnt == n
            print(f"[CALIB]   L={length} n={n}: {dt:.1f}s ({cnt}/{n} refolds{'' if ok else ' — INCOMPLETE'})", flush=True)
            return dt if ok else None
        except Exception as e:  # OOM / OF3 failure at this length — record + continue to other lengths
            print(f"[CALIB]   L={length} n={n}: FAILED {type(e).__name__}: {str(e)[:180]}", flush=True)
            return None

    res: dict = {}
    load = None
    for L in LENGTHS:
        tlo = timed(L, NLO, 1000 * L)
        thi = timed(L, NHI, 2000 * L)
        if tlo is not None and thi is not None:
            pf = (thi - tlo) / (NHI - NLO)
            if load is None:
                load = max(0.0, tlo - NLO * pf)
            method = "two-count"
        elif tlo is not None or thi is not None:
            t, n = (tlo, NLO) if tlo is not None else (thi, NHI)
            pf = max(0.0, (t - LOAD_GUESS) / n)
            method = f"single-count(assume load={LOAD_GUESS:.0f}s)"
        else:
            print(f"[CALIB] L={L}: both fold-counts FAILED (max cloud-refoldable length exceeded?)", flush=True)
            res[str(L)] = {"failed": True}
            continue
        res[str(L)] = {
            "t_nlo": round(tlo, 1) if tlo else None,
            "t_nhi": round(thi, 1) if thi else None,
            "per_fold_s": round(pf, 1),
            "method": method,
        }
        print(f"[CALIB] L={L}: per-fold ~{pf:.1f}s  [{method}]", flush=True)

    res["load_s"] = round(load, 1) if load is not None else None
    print(f"[CALIB] OF3 model-load ~{res['load_s']}s (empirical)" if load is not None
          else "[CALIB] load not isolated (no two-count length fully succeeded)", flush=True)
    with open(f"{out}/calib.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print("[CALIB] wrote " + out + "/calib.json", flush=True)


if __name__ == "__main__":
    main()
