"""Structural 'headliner' — superpose RFD3±SPA designs onto the target fold (3D Cα trace, matplotlib).

Per prompt: auto-pick a representative (median-TM) **baseline** design and the best-TM **SPA** design from
the flywheel scores, TM-align each onto the target structure, and render 3D Cα traces (target in gray +
design colored), two panels (without SPA | with SPA). Shows the fold-steering at a glance: the baseline
diverges from the target fold, the SPA design matches it. matplotlib only (no PyMOL) — a DRAFT; final
poster renders are best hand-tuned in PyMOL/ChimeraX (cartoon + secondary structure).

    conda run -n spa-dev python scripts/eval/plot_structures.py \
        --prompt-dir outputs/eval/bigN_h5/runA_R7VVY2 \
        --target <pdb_dir>/AF-R7VVY2-F1-model_v4_esmfold_v1.pdb \
        --lam 2 --out outputs/eval/figures/struct_R7VVY2.png
"""

from __future__ import annotations

import argparse
import json
import os


def _ca_seq(pdb):
    from spa.eval.score import _as_struct, _ca_array, _coords64, _seq_of
    ca = _ca_array(_as_struct(pdb))
    return _coords64(ca), _seq_of(ca)


def _align(design_xyz, design_seq, target_xyz, target_seq):
    """TM-align design onto target; return (aligned design coords, prompt-normalized TM)."""
    import tmtools
    res = tmtools.tm_align(design_xyz, target_xyz, design_seq, target_seq)
    return design_xyz @ res.u.T + res.t, float(res.tm_norm_chain2)


def _pick(scores, cond, lam, mode):
    g = [s for s in scores if s["condition"] == cond and abs(s["lambda_scale"] - lam) < 1e-6
         and s.get("tm_score") is not None]
    if not g:
        return None
    g.sort(key=lambda s: s["tm_score"])
    return g[-1] if mode == "best" else g[len(g) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-dir", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = json.load(open(f"{args.prompt_dir}/flywheel_results.json"))["scores"]
    base = _pick(scores, "baseline", 0.0, "median")
    spa = _pick(scores, "spa", args.lam, "best")
    if base is None or spa is None:
        raise SystemExit("missing baseline or spa designs in scores")
    tgt_xyz, tgt_seq = _ca_seq(args.target)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig = plt.figure(figsize=(11, 5.2))
    for i, (label, s, col) in enumerate([("without SPA (baseline)", base, "#d62728"),
                                         (f"with SPA (λ={args.lam:g})", spa, "#1f77b4")]):
        dx, ds = _ca_seq(f"{args.prompt_dir}/{s['name']}.pdb")
        aligned, tm = _align(dx, ds, tgt_xyz, tgt_seq)
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.plot(*tgt_xyz.T, color="0.6", lw=2.5, label="target fold")
        ax.plot(*aligned.T, color=col, lw=2.0, label="design")
        ax.set_title(f"{label}\nTM-to-target = {tm:.2f}", fontsize=11)
        ax.set_axis_off()
        ax.legend(fontsize=8, loc="upper right")

    pid = os.path.basename(args.prompt_dir).replace("runA_", "")
    fig.suptitle(f"{pid}: fold steering — baseline diverges, SPA matches the target", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=150)
    print(f"[plot] {pid}: baseline TM {base['tm_score']:.2f} -> SPA TM {spa['tm_score']:.2f}  -> {args.out}")


if __name__ == "__main__":
    main()
