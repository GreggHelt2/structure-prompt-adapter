"""Render SPA eval flywheel results into poster panels (dev ``03_poster_narrative.md`` §5).

Turns one or more ``flywheel_results.json`` into the figures the poster reports:
- **Adherence vs λ** (top): mean prompt-TM with SEM, baseline anchored at λ=0 — H2/H3.
- **Designability vs λ** (bottom): the scRMSD<2 Å success rate — H1.
- **Motif-RMSD panel** (added only for hard⊕soft runs that scaffolded a native motif): the design-side
  motif Cα-RMSD per condition — should sit at ≈0 and be *equal* across baseline/spa (H5 "hard satisfied").

Two modes, auto-detected:
- **Single JSON** → adherence+designability for that run (+ the motif panel if a motif was present).
- **Multiple JSONs** → overlay their adherence/designability curves with a legend — the variant comparison
  (N×1536 / 1×1536 / 1×32 on one prompt) or the fold comparison (one variant across prompts).

Stats are computed from the per-design ``scores`` (mean ± SEM), not just the summaries, so error bars are
honest. CPU-only, headless (Agg backend); writes a PNG. This is *starting-point* styling — tune freely.

    conda run -n spa-dev python scripts/eval/plot_results.py \
        --json outputs/eval/flywheel/<run>/flywheel_results.json --out fig.png [--title ...]
    # overlay (variant comparison):
    conda run -n spa-dev python scripts/eval/plot_results.py \
        --json A/flywheel_results.json --json B/flywheel_results.json --json C/flywheel_results.json \
        --labels 1x32 1x1536 Nx1536 --out variants.png --title "Variant comparison — A0A522W419"
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _stats(values):
    """(mean, SEM, n) over the non-None / non-NaN values; (None, 0, 0) if empty."""
    vals = [float(v) for v in values if v is not None and float(v) == float(v)]
    if not vals:
        return None, 0.0, 0
    sem = (pstdev(vals) / len(vals) ** 0.5) if len(vals) > 1 else 0.0
    return mean(vals), sem, len(vals)


def load_curve(path: str) -> dict:
    """Aggregate a flywheel_results.json into per-(condition, λ) rows for plotting.

    Baseline is placed at x=0 (the λ=0 identity anchor); spa points at their λ. Returns the sorted rows,
    whether a native motif was scored, and a display name (the run dir).
    """
    d = json.loads(Path(path).read_text())
    groups: dict[tuple, list] = defaultdict(list)
    for s in d["scores"]:
        groups[(s["condition"], float(s["lambda_scale"]))].append(s)

    rows = []
    for (cond, lam), items in groups.items():
        tm_m, tm_sem, _ = _stats([s.get("tm_score") for s in items])
        des = [s.get("designable") for s in items if s.get("designable") is not None]
        drate = (sum(bool(b) for b in des) / len(des)) if des else None
        mr_m, _, _ = _stats([s.get("motif_rmsd") for s in items])
        rows.append({
            "cond": cond, "lam": lam,
            "x": 0.0 if cond == "baseline" else lam,
            "tm": tm_m, "tm_sem": tm_sem, "drate": drate, "motif_rmsd": mr_m,
        })
    rows.sort(key=lambda r: (r["x"], r["cond"]))
    return {
        "rows": rows,
        "has_motif": any(r["motif_rmsd"] is not None for r in rows),
        "name": Path(path).resolve().parent.name,
    }


def _xticklabels(xs):
    return ["baseline" if x == 0.0 else f"λ={x:g}" for x in xs]


def plot(curves: list[dict], out: str, title: str | None = None) -> None:
    single_motif = len(curves) == 1 and curves[0]["has_motif"]
    n_panels = 3 if single_motif else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(7, 3.0 * n_panels), sharex=not single_motif)
    ax_tm, ax_des = axes[0], axes[1]

    for c in curves:
        rows = c["rows"]
        xs = [r["x"] for r in rows if r["tm"] is not None]
        tms = [r["tm"] for r in rows if r["tm"] is not None]
        sems = [r["tm_sem"] for r in rows if r["tm"] is not None]
        ax_tm.errorbar(xs, tms, yerr=sems, marker="o", capsize=3, label=c["name"])
        dxs = [r["x"] for r in rows if r["drate"] is not None]
        dr = [r["drate"] for r in rows if r["drate"] is not None]
        ax_des.plot(dxs, dr, marker="s", linestyle="--", label=c["name"])

    ax_tm.set_ylabel("prompt-TM (adherence)")
    ax_tm.set_title(title or (curves[0]["name"] if len(curves) == 1 else "SPA eval"))
    ax_tm.grid(alpha=0.3)
    ax_des.set_ylabel("designable rate\n(scRMSD < 2 Å)")
    ax_des.set_ylim(-0.05, 1.05)
    ax_des.grid(alpha=0.3)
    if len(curves) > 1:
        ax_tm.legend(fontsize=8)
        ax_des.legend(fontsize=8)

    # consistent λ ticks from the union of x positions
    all_x = sorted({r["x"] for c in curves for r in c["rows"]})
    for ax in (ax_tm, ax_des):
        ax.set_xticks(all_x)
        ax.set_xticklabels(_xticklabels(all_x))
    (ax_des if not single_motif else ax_tm).set_xlabel("SPA scale λ")

    if single_motif:
        ax_m = axes[2]
        rows = curves[0]["rows"]
        labels = [("baseline" if r["cond"] == "baseline" else f"spa λ={r['lam']:g}") for r in rows]
        vals = [(r["motif_rmsd"] if r["motif_rmsd"] is not None else 0.0) for r in rows]
        ax_m.bar(range(len(vals)), vals, color="tab:green")
        ax_m.axhline(1.0, color="r", ls=":", lw=1, label="satisfied cutoff (1 Å)")
        ax_m.set_xticks(range(len(labels)))
        ax_m.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax_m.set_ylabel("motif-RMSD (Å)")
        ax_m.set_title("hard motif satisfied (≈0, SPA-independent)")
        ax_m.grid(alpha=0.3, axis="y")
        ax_m.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[plot] wrote {out}  ({n_panels} panels, {len(curves)} curve(s))")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="append", required=True, help="flywheel_results.json (repeatable)")
    ap.add_argument("--labels", nargs="*", default=None, help="display label per --json (default: run dir)")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    curves = [load_curve(p) for p in args.json]
    if args.labels:
        if len(args.labels) != len(curves):
            ap.error(f"{len(args.labels)} labels for {len(curves)} --json")
        for c, lbl in zip(curves, args.labels):
            c["name"] = lbl
    plot(curves, args.out, args.title)


if __name__ == "__main__":
    main()
