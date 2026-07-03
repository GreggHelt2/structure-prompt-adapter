"""Render native-vs-SPA hard⊕soft pairs with the pinned motif in GREEN (PyMOL headless).

For each entry in the pick JSON: load the renumbered source (grey ghost) + the native (firebrick) and
SPA (marine) designs, superpose BY THE MOTIF (so the pinned region overlaps the target exactly and the
scaffold's divergence is visible), colour the motif green, ray-trace a PNG per (id, condition). The
PNGs are the input to plot_hardsoft_montage.py.

    conda run -n pymol pymol -cq scripts/eval/render_hardsoft_pairs.py -- \
        --pick <pick.json> --source-dir <dir> --design-dir <dir> --out-dir <dir>

pick.json: { "<id>": {"native": {"name": "<pdb-stem>"}, "spa": {"name": "<pdb-stem>"},
                       "motif_segs": [[start, end], ...]} }
source-dir: renumbered <id>.pdb (chain A, residues 1..N).  design-dir: <name>.pdb for each name.
"""
import argparse
import json
import os
import sys

from pymol import cmd


def setup():
    cmd.bg_color("white"); cmd.set("ray_opaque_background", 1); cmd.set("antialias", 2)
    cmd.set("ray_shadows", 0); cmd.set("cartoon_fancy_helices", 1); cmd.set("specular", 0.2)
    cmd.set_color("motifgreen", [0.15, 0.70, 0.32])


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", required=True)
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--design-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    pick = json.load(open(a.pick))

    for pid, c in pick.items():
        sel = "resi " + "+".join(f"{s}-{e}" for s, e in c["motif_segs"])
        cmd.reinitialize(); setup()
        cmd.load(os.path.join(a.source_dir, pid + ".pdb"), "target"); cmd.dss("target")
        cmd.hide("everything"); cmd.show("cartoon", "target"); cmd.orient("target")
        view = cmd.get_view()
        for tag, col in (("native", "firebrick"), ("spa", "marine")):
            cmd.delete("design")
            cmd.load(os.path.join(a.design_dir, c[tag]["name"] + ".pdb"), "design"); cmd.dss("design")
            try:
                cmd.super(f"design and {sel}", f"target and {sel}")
            except Exception:
                cmd.cealign("target", "design")
            cmd.set_view(view)
            cmd.hide("everything"); cmd.show("cartoon", "target or design")
            cmd.color("grey70", "target"); cmd.set("cartoon_transparency", 0.5, "target")
            cmd.color(col, "design")
            cmd.color("motifgreen", f"design and {sel}")
            cmd.set("cartoon_transparency", 0.0, f"design and {sel}")
            out = os.path.join(a.out_dir, f"{pid}_{tag}.png")
            cmd.ray(1000, 800); cmd.png(out, dpi=150)
            print(f"{pid} {tag:7} -> {os.path.basename(out)}")
    print("DONE")


if __name__ == "__main__" or True:
    main()
