"""Aggregate the B4 named-fold demos — per fold: does SPA steer a designable design toward the fold?

Reads per-fold ``<results-dir>/C_n_by_1536/<pdb>.json`` (run_variant_desig, λ={0.5,1,2}) + the b4 prep
manifest (fold names/lens). Per fold reports baseline vs each λ (designable-rate + mean TM-to-fold), and
picks the **money-shot** — the single highest-TM *designable* design across all λ (adopts the recognizable
fold AND is foldable) — the poster-candidate per fold.

Usage: conda run -n spa-dev python scripts/eval/aggregate_b4.py --results-dir <dir> --manifest <resolved.json>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from statistics import mean

LAMS = [0.5, 1.0, 2.0]


def _des(s, cut):
    d = s.get("designable")
    return bool(d) if d is not None else (s.get("scrmsd") is not None and s["scrmsd"] < cut)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--variant", default="C_n_by_1536")
    ap.add_argument("--scrmsd-cutoff", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    man = {p["id"]: p for p in json.load(open(a.manifest))["prompts"]}
    report = {"scrmsd_cutoff": a.scrmsd_cutoff, "folds": {}}
    print(f"\nB4 named-fold demos — designability + fold-adherence (scRMSD<{a.scrmsd_cutoff}Å, {a.variant} Run A)")
    print(f"{'fold':<16}{'pdb':<6}{'len':>4} {'base d/TM':>10} | "
          + " | ".join(f"λ{l:g} d/TM".rjust(11) for l in LAMS) + " |  MONEY-SHOT (λ, TM, scRMSD)")

    for jf in sorted(glob.glob(os.path.join(a.results_dir, a.variant, "*.json"))):
        pid = os.path.splitext(os.path.basename(jf))[0]
        m = man.get(pid, {})
        by = defaultdict(list)
        for s in json.load(open(jf)).get("scores", []):
            tm = s.get("tm_norm_prompt")
            if tm is None:
                continue
            key = "baseline" if s.get("condition") == "baseline" else ("spa", float(s["lambda_scale"]))
            by[key].append({"tm": tm, "des": _des(s, a.scrmsd_cutoff), "scrmsd": s.get("scrmsd")})

        def cell(entries):
            return None if not entries else (mean(1.0 if e["des"] else 0.0 for e in entries), mean(e["tm"] for e in entries))

        b = cell(by.get("baseline"))
        cells = []
        for l in LAMS:
            c = cell(by.get(("spa", l)))
            cells.append(f"{c[0]:.2f}/{c[1]:.2f}".rjust(11) if c else "—".rjust(11))
        # money-shot: highest-TM DESIGNABLE design across all spa λ
        best = None
        for l in LAMS:
            for e in by.get(("spa", l), []):
                if e["des"] and (best is None or e["tm"] > best["tm"]):
                    best = {**e, "lam": l}
        ms = f"λ{best['lam']:g}, TM {best['tm']:.2f}, scRMSD {best['scrmsd']:.2f}Å" if best else "none designable"
        bs = f"{b[0]:.2f}/{b[1]:.2f}" if b else "—"
        print(f"{m.get('fold_name', pid):<16}{pid:<6}{m.get('len', '?'):>4} {bs:>10} | " + " | ".join(cells) + f" |  {ms}")
        report["folds"][pid] = {"fold_name": m.get("fold_name"), "len": m.get("len"),
                                "baseline": b, "lambdas": {l: cell(by.get(("spa", l))) for l in LAMS},
                                "money_shot": best}

    print("\n(d = designable rate scRMSD<cut over 8 designs; TM = mean prompt-adherence to the named fold; "
          "money-shot = highest-TM design that is also designable — the poster candidate.)")
    if a.out:
        json.dump(report, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
