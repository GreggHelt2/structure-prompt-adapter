"""PyMOL cartoon render: RFD3±SPA designs superposed on the target fold (final poster quality).

Headless:
    conda run -n pymol pymol -cq scripts/eval/render_cartoon.py -- \
        --prompt-dir outputs/eval/bigN_h5/runA_R7VVY2 \
        --target <pdb_dir>/AF-R7VVY2-F1-model_v4_esmfold_v1.pdb \
        --lam 2 --out outputs/eval/figures/cartoon_R7VVY2.png

Reads the flywheel scores -> picks a representative (median-TM) baseline + the best-TM SPA design,
superposes each onto the target (cmd.super), assigns secondary structure (cmd.dss — RFD3 PDBs carry no
SS records), styles cartoon (target gray, design colored), and renders a 2-panel grid
(without SPA | with SPA), ray-traced. Prints the TM values for the caption. The companion to
plot_structures.py (the matplotlib Cα-trace draft); this is the publication-quality version.
"""

import json
import sys


def _argval(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def _pick(scores, cond, lam, mode):
    g = [s for s in scores if s["condition"] == cond and abs(s["lambda_scale"] - lam) < 1e-6
         and s.get("tm_score") is not None]
    if not g:
        return None
    g.sort(key=lambda s: s["tm_score"])
    return g[-1] if mode == "best" else g[len(g) // 2]


prompt_dir = _argval("--prompt-dir")
target = _argval("--target")
lam = float(_argval("--lam", "2"))
out = _argval("--out")

scores = json.load(open(f"{prompt_dir}/flywheel_results.json"))["scores"]
base = _pick(scores, "baseline", 0.0, "median")
spa = _pick(scores, "spa", lam, "best")

from pymol import cmd

cmd.bg_color("white")
cmd.set("ray_opaque_background", 1)
cmd.set("ray_shadows", 0)
cmd.set("cartoon_fancy_helices", 1)

cmd.load(target, "tgtB")
cmd.load(target, "tgtS")
cmd.load(f"{prompt_dir}/{base['name']}.pdb", "baseline")
cmd.load(f"{prompt_dir}/{spa['name']}.pdb", "withspa")

cmd.super("baseline", "tgtB")
cmd.super("withspa", "tgtS")

cmd.dss()                                   # assign secondary structure (RFD3 PDBs lack SS records)
cmd.hide("everything")
cmd.show("cartoon")
cmd.color("gray70", "tgtB or tgtS")
cmd.color("firebrick", "baseline")
cmd.color("marine", "withspa")

cmd.set("grid_mode", 1)                      # 2 panels: slot 1 = baseline, slot 2 = with-SPA
for obj, slot in (("tgtB", 1), ("baseline", 1), ("tgtS", 2), ("withspa", 2)):
    cmd.set("grid_slot", slot, obj)

cmd.orient()
cmd.ray(1800, 900)
cmd.png(out, dpi=150)
print(f"[pymol] wrote {out}  | without-SPA TM {base['tm_score']:.2f}  with-SPA TM {spa['tm_score']:.2f}")
