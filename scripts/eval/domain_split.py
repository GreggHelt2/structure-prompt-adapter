"""THROWAWAY ANALYSIS — quick contiguous 2-domain split via Cα contact map.

Used to pick a clean *two-domain* protein for the three-way localization probe: my first probe cut an
arbitrary mid-protein boundary, which slices THROUGH one fold and guarantees RFD3 completes coherently
(the confound). A real domain boundary is the regime where region-localized steering could actually work.

Method (no new dependency — biotite Cα + numpy): build the Cα–Cα contact map (<thresh Å), and for each
contiguous boundary b (≥ min_dom from each end) score **domain-ness = cross-domain contacts / min(intra1,
intra2)** — the inter-domain contacts relative to the *weaker* domain's internal contacts. Low ⇒ two
self-contained units joined by a thin linker; high ⇒ single domain (or a trivial end-flap). Lower is
cleaner. Pick the lowest-scoring structure + its boundary, then run the probe with eval.probe_boundary=b.

    conda run -n spa-dev python scripts/eval/domain_split.py \
        --pdb-dir /home/user1/projects/spa/training_data/proteina-atomistica_data_vrelease/atomistica_data_release/pdb \
        --uniprots A0A1F1QD24,A0A7C9GW19,A0A2G6NLK2,H1SDK8,W7QV56,A0A2W5NKK0,A0A1Q8BPK6,A0A536G7C3,A0A1X0IID6
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def detect_domains(pdb_path: str, contact_thresh: float = 8.0, min_dom: int = 40) -> dict:
    """Best contiguous 2-domain split + domain-ness score (lower = cleaner two-domain)."""
    from spa.eval.score import _as_struct, _ca_array, _coords64

    X = _coords64(_ca_array(_as_struct(pdb_path)))            # [N, 3] Cα
    n = len(X)
    if n < 2 * min_dom:
        return {"path": pdb_path, "n_res": n, "boundary": None, "score": None, "note": "too short"}

    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    c = (d < contact_thresh).astype(np.float64)
    np.fill_diagonal(c, 0.0)

    best_b, best_score = None, np.inf
    for b in range(min_dom, n - min_dom + 1):
        cross = c[:b, b:].sum()
        intra1 = c[:b, :b].sum() / 2.0
        intra2 = c[b:, b:].sum() / 2.0
        denom = min(intra1, intra2)
        if denom <= 0:
            continue
        score = cross / denom                                # inter / weaker-domain intra
        if score < best_score:
            best_score, best_b = score, b
    return {"path": pdb_path, "n_res": n, "boundary": int(best_b),
            "dom1": [0, int(best_b)], "dom2": [int(best_b), n],
            "dom1_size": int(best_b), "dom2_size": int(n - best_b),
            "score": round(float(best_score), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdbs", nargs="*", help="explicit PDB paths")
    ap.add_argument("--pdb-dir", default=None, help="dir for --uniprots")
    ap.add_argument("--uniprots", default=None, help="comma-sep UniProt accessions (CDDB filename pattern)")
    ap.add_argument("--pattern", default="AF-{u}-F1-model_v4_esmfold_v1.pdb")
    ap.add_argument("--contact", type=float, default=8.0)
    ap.add_argument("--min-dom", type=int, default=40)
    args = ap.parse_args()

    paths = list(args.pdbs)
    if args.uniprots:
        for u in args.uniprots.split(","):
            paths.append(os.path.join(args.pdb_dir, args.pattern.format(u=u.strip())))

    rows = []
    for p in paths:
        try:
            rows.append(detect_domains(p, args.contact, args.min_dom))
        except Exception as e:  # noqa: BLE001 — a screen, never fail the batch
            rows.append({"path": p, "error": str(e)})

    scored = [r for r in rows if r.get("score") is not None]
    scored.sort(key=lambda r: r["score"])
    print(f"{'score':>8}{'n_res':>7}{'bound':>7}{'d1':>6}{'d2':>6}  pdb   (lower score = cleaner 2-domain)")
    for r in scored:
        print(f"{r['score']:>8.3f}{r['n_res']:>7}{r['boundary']:>7}{r['dom1_size']:>6}{r['dom2_size']:>6}  "
              f"{os.path.basename(r['path'])}")
    for r in rows:
        if r.get("score") is None:
            print(f"  [skip] {os.path.basename(r['path'])}: {r.get('note') or r.get('error')}")
    if scored:
        print("\n[best two-domain candidate]")
        print(json.dumps(scored[0], indent=2))


if __name__ == "__main__":
    main()
