"""Stage an RCSB entry as a single-chain, amino-acid-only prompt PDB.

WHY THIS EXISTS. The prompt PDBs in ``training_data/eval_external/`` were staged ad hoc on 2026-09-03
(dev ``90`` §5.0c), and staging is where two errors in that slate were caught rather than in review: an
NMR structure that had reached a recommendation table, and a "cofactor-free" entry carrying Cu and MPD
across two chains. **Staging is the check**, so it should be a script that records what it checked.

What it does, in order:
  1. Fetch the entry metadata from the RCSB REST API and REFUSE anything that is not a crystal
     structure. The experimental method is queried, never recalled (dev ``90`` §5.0c.1).
  2. Download the PDB, keep model 1 only.
  3. Keep the requested chain, or the LONGEST amino-acid chain when none is named.
  4. Apply :func:`spa.eval.score.polymer_only`, which drops ligands, ions and waters while keeping
     modified residues such as MSE and SEP. A Ca ion's atom name is literally ``CA``, so an unfiltered
     cofactor is silently counted as a residue by every backbone metric.
  5. Write ``<ID>_<chain>_clean.pdb`` and print the residue and Ca counts.

⚠️ Prompt rows and scoreable Ca can legitimately differ: ``1TEN`` gives 90 rows against 89 Ca because
one residue lacks a Ca. Do not read that as a disagreement.

Usage:
    conda run -n spa-dev python scripts/eval/stage_prompt_pdb.py \
        --ids 1A2P 1QYS 2CI2 --out-dir /home/user1/projects/spa/training_data/eval_external
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

CRYSTAL = "X-RAY DIFFRACTION"


def entry_meta(pdb_id: str) -> dict:
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    with urllib.request.urlopen(url, timeout=30) as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", nargs="+", required=True, help="RCSB entry ids, e.g. 1A2P 1QYS")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chain", default=None,
                    help="chain id to keep; default is the longest amino-acid chain")
    ap.add_argument("--allow-non-crystal", action="store_true",
                    help="override the crystal-structure gate; states the method it accepted")
    a = ap.parse_args()

    sys.path.insert(0, "/home/user1/projects/spa/structure-prompt-adapter/src")
    from biotite.structure.io.pdb import PDBFile
    from spa.eval.score import polymer_only, select_chains

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for pid in a.ids:
        pid = pid.upper()
        meta = entry_meta(pid)
        methods = [m["method"] for m in meta.get("exptl", [])]
        res = meta.get("rcsb_entry_info", {}).get("resolution_combined")
        title = meta.get("struct", {}).get("title", "")
        if CRYSTAL not in methods and not a.allow_non_crystal:
            print(f"⛔ {pid}: method {methods} is not {CRYSTAL}. Refusing "
                  f"(pass --allow-non-crystal to override).")
            continue
        print(f"[stage] {pid}: {methods}, resolution {res}, {title[:70]}")

        raw = out / f"{pid}.pdb"
        if not raw.exists():
            urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pid}.pdb", raw)
        arr = PDBFile.read(str(raw)).get_structure(model=1)

        aa = polymer_only(arr)
        if a.chain:
            chain = a.chain
        else:
            counts = {c: int((aa.chain_id == c).sum()) for c in sorted(set(aa.chain_id))}
            chain = max(counts, key=counts.get)
            if len(counts) > 1:
                print(f"[stage]   {len(counts)} amino-acid chains {counts}; keeping longest: {chain}")
        sel = polymer_only(select_chains(aa, chain))
        n_res = len(set(zip(sel.chain_id.tolist(), sel.res_id.tolist())))
        n_ca = int((sel.atom_name == "CA").sum())

        dst = out / f"{pid}_{chain}_clean.pdb"
        f = PDBFile()
        f.set_structure(sel)
        f.write(str(dst))
        print(f"[stage]   -> {dst.name}: {n_res} residues, {n_ca} CA")
        written.append(str(dst))

    print("\n[stage] staged files:")
    for w in written:
        print(" ", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
