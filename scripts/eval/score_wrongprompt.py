"""Strong wrong-prompt scorer — TM-align Run-B designs to BOTH the target and the decoy (dev 16 §11.6).

Run B conditions SPA on the DECOY and the flywheel scored adherence-to-*target* (weak version:
adherence-to-target should collapse to ~baseline). This adds adherence-to-*decoy*: if SPA followed the
decoy, TM→decoy is high — the strong claim "SPA follows whichever prompt it is given."

Globs the design PDBs in --run-dir, parses (condition, λ) from each filename, and reports mean
target-normalized TM to the target vs to the decoy, per (condition, λ). Reuses spa.eval.score.tm_score
(so normalization matches the scorer). Pure CPU.

    conda run -n spa-dev python scripts/eval/score_wrongprompt.py \
        --run-dir outputs/eval/control_B_wrongprompt_A0A522W419_decoy_A0A2X2KHU0 \
        --target <TARGET.pdb> --decoy <DECOY.pdb>
"""

from __future__ import annotations

import argparse
import collections
import glob
import re
import statistics as st
from pathlib import Path

# {pid}_{condition}_lambda{λ}_{idx}.pdb  — pid may contain underscores, so anchor on the tail.
PAT = re.compile(r"_(baseline|spa|nullprompt)_lambda([0-9p.]+)_(\d+)\.pdb$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--target", required=True, help="target structure (adherence should be LOW)")
    ap.add_argument("--decoy", required=True, help="decoy structure SPA was prompted with (should be HIGH)")
    args = ap.parse_args()

    from spa.eval.score import tm_score

    groups: dict[tuple[str, float], list[tuple[float, float]]] = collections.defaultdict(list)
    for p in sorted(glob.glob(f"{args.run_dir}/*.pdb")):
        m = PAT.search(Path(p).name)
        if not m:
            continue
        cond = m.group(1)
        lam = float(m.group(2).replace("p", "."))
        tt = tm_score(p, args.target)[1]   # normalized by the target
        td = tm_score(p, args.decoy)[1]    # normalized by the decoy
        groups[(cond, lam)].append((tt, td))

    if not groups:
        raise SystemExit(f"no design PDBs matched in {args.run_dir} (has Run B finished writing?)")

    print(f"{'condition':11}{'lam':>5}{'n':>3}{'TM->target':>12}{'TM->decoy':>11}   read")
    print("-" * 56)
    for k in sorted(groups):
        vals = groups[k]
        mt = st.mean(v[0] for v in vals)
        md = st.mean(v[1] for v in vals)
        read = "followed DECOY" if (md - mt) > 0.10 else ("~baseline" if k[0] != "spa" else "ambiguous")
        print(f"{k[0]:11}{k[1]:>5.1f}{len(vals):>3}{mt:>12.3f}{md:>11.3f}   {read}")
    print("\nStrong wrong-prompt read: for the spa (decoy-prompted) rows, TM->decoy >> TM->target means "
          "SPA steered toward the structure it was given, not the (scored) target.")


if __name__ == "__main__":
    main()
