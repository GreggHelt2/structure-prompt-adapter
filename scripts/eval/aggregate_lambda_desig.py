"""Aggregate the λ×designability tradeoff sweep into the fold-dependent λ sweet-spot table.

Reads the per-prompt ``<results-dir>/C_n_by_1536/<id>.json`` (staged by run_variant_desig.sh with
``lambda_scale=[0.5,1,2]``) + the prep manifest (``b1_full_resolved.json``, for fold/len labels), and
reports — per fold class and overall — the designable rate (scRMSD<cutoff) and prompt-adherence (TM) at
baseline (λ=0) and each spa λ, with Δ-vs-baseline, plus each fold's sweet-spot λ (max adherence gain that
stays designable). This is the dev-17 / 13 §7 payoff: is the adherence↔designability sweet spot
fold-dependent (β→λ≈1, α/β→λ≈2, irregular→?).

Usage:
  conda run -n spa-dev python scripts/eval/aggregate_lambda_desig.py \
    --results-dir <dir-with-C_n_by_1536/> --manifest <dir>/b1_full_resolved.json [--scrmsd-cutoff 2.0]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from statistics import mean

LAMS = [0.5, 1.0, 2.0]


def _designable(s, cut):
    d = s.get("designable")
    return bool(d) if d is not None else (s.get("scrmsd") is not None and s["scrmsd"] < cut)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, help="dir containing C_n_by_1536/<id>.json")
    ap.add_argument("--manifest", required=True, help="b1_full_resolved.json (fold/len labels)")
    ap.add_argument("--variant", default="C_n_by_1536")
    ap.add_argument("--scrmsd-cutoff", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    man = json.load(open(a.manifest))
    fold_of = {p["id"]: p["fold"] for p in man["prompts"]}

    # scores[fold][cond_key] = list of (designable_bool, tm_norm_prompt) pooled across that fold's prompts;
    # cond_key = "baseline" or ("spa", λ). Also track per-fold prompt count.
    scores: dict = defaultdict(lambda: defaultdict(list))
    prompts_seen: dict = defaultdict(set)
    n_json = 0
    for jf in sorted(glob.glob(os.path.join(a.results_dir, a.variant, "*.json"))):
        uid = os.path.splitext(os.path.basename(jf))[0]
        fold = fold_of.get(uid)
        if fold is None:
            print(f"[warn] {uid} not in manifest — skipping")
            continue
        n_json += 1
        prompts_seen[fold].add(uid)
        for s in json.load(open(jf)).get("scores", []):
            tm = s.get("tm_norm_prompt")
            if tm is None:
                continue
            des = _designable(s, a.scrmsd_cutoff)
            if s.get("condition") == "baseline":
                scores[fold]["baseline"].append((des, tm))
            elif s.get("condition") == "spa":
                scores[fold][("spa", float(s["lambda_scale"]))].append((des, tm))

    def agg(entries):
        if not entries:
            return None
        return {"n": len(entries), "d_succ": mean(1.0 if d else 0.0 for d, _ in entries),
                "tm": mean(t for _, t in entries)}

    folds_order = ["all-a", "all-b", "a-b", "irregular"]
    folds = [f for f in folds_order if f in scores] + [f for f in scores if f not in folds_order]

    report = {"scrmsd_cutoff": a.scrmsd_cutoff, "n_prompts": n_json, "by_fold": {}}
    print(f"\nλ×designability tradeoff — {n_json} prompts, scRMSD<{a.scrmsd_cutoff}Å, {a.variant} (Run A)")
    print(f"{'fold':<11}{'#p':>3} {'base d/TM':>12} | " + " | ".join(f"λ{l:g}: d/TM (ΔTM)".rjust(20) for l in LAMS) + " | sweet-λ")
    # accumulate an OVERALL bucket too
    overall = defaultdict(list)
    for f in folds:
        for k, v in scores[f].items():
            overall[k].extend(v)

    def render(label, sc, np_):
        b = agg(sc.get("baseline"))
        cells, best = [], None
        row = {"n_prompts": np_, "baseline": b, "lambdas": {}}
        for l in LAMS:
            g = agg(sc.get(("spa", l)))
            row["lambdas"][l] = g
            if g and b:
                dtm = g["tm"] - b["tm"]
                cells.append(f"{g['d_succ']:.2f}/{g['tm']:.2f} ({dtm:+.2f})".rjust(20))
                # sweet-spot: max adherence gain among λ that stay >= baseline d_succ - 0.15
                if g["d_succ"] >= b["d_succ"] - 0.15 and (best is None or dtm > best[1]):
                    best = (l, dtm)
            else:
                cells.append("—".rjust(20))
        bs = f"{b['d_succ']:.2f}/{b['tm']:.2f}" if b else "—"
        sw = f"λ{best[0]:g}" if best else "none"
        print(f"{label:<11}{np_:>3} {bs:>12} | " + " | ".join(cells) + f" | {sw}")
        row["sweet_lambda"] = best[0] if best else None
        return row

    for f in folds:
        report["by_fold"][f] = render(f, scores[f], len(prompts_seen[f]))
    report["overall"] = render("OVERALL", overall, n_json)
    print("\n(d = designable rate scRMSD<cut; TM = mean prompt-adherence; ΔTM vs baseline; "
          "sweet-λ = max ΔTM that keeps d_succ within 0.15 of baseline.)")

    if a.out:
        json.dump(report, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
