"""Auto-carve a self-motif contig from a structure via biotite SSE (no DSSP install).

For the big-n H5 eval: pick the longest 1-2 secondary-structure segments (helix/strand) as the hard
motif and emit an RFD3 contig (gap,A{start}-{end},gap,...) spanning the full length — the same shape
as §4's hand-built contigs (e.g. 5,A6-14,99,A114-119,31). Self-prompt: CDDB monomers are numbered
1..L, so positional index i -> author resid i+1. Caps total motif to ~max_frac of the chain so plenty
of diffused scaffold remains (and the non-overlap mask never masks every row).

    conda run -n spa-dev python scripts/eval/carve_motif.py \
        --pdb-dir <pdb_dir> --uniprots A0A522W419,A0A7S1B8G4,...
"""

from __future__ import annotations

import argparse
import json
import os


def _runs(sse) -> list[tuple[int, int, str]]:
    """Contiguous SSE runs as (start, end_inclusive, kind) for kind in {'a','b'} (helix/strand)."""
    out, i, n = [], 0, len(sse)
    while i < n:
        k = sse[i]
        if k in ("a", "b"):
            j = i
            while j + 1 < n and sse[j + 1] == k:
                j += 1
            out.append((i, j, k))
            i = j + 1
        else:
            i += 1
    return out


def carve(pdb_path: str, n_seg: int = 2, min_seg: int = 5, max_seg: int = 12,
          max_frac: float = 0.18) -> dict:
    import biotite.structure as struc

    from spa.eval.score import _as_struct, _ca_array

    arr = _as_struct(pdb_path)
    L = len(_ca_array(arr))
    sse = struc.annotate_sse(arr)                              # per-residue 'a'/'b'/'c', length L
    runs = [r for r in _runs(sse) if (r[1] - r[0] + 1) >= min_seg]
    runs.sort(key=lambda r: -(r[1] - r[0] + 1))               # longest first

    budget = max(min_seg, int(max_frac * L))
    chosen: list[tuple[int, int]] = []
    used = 0
    for s, e, _k in runs:
        if len(chosen) >= n_seg:
            break
        seg_len = min(e - s + 1, max_seg)                     # truncate a long SSE to a central window
        s2 = s + (e - s + 1 - seg_len) // 2
        e2 = s2 + seg_len - 1
        if used + seg_len > budget and chosen:               # keep total under the frac cap (always keep ≥1)
            continue
        chosen.append((s2, e2))
        used += seg_len
    if not chosen:                                            # no clean SSE -> fall back to a central window
        seg = min(max_seg, max(min_seg, budget))
        s2 = max(0, L // 2 - seg // 2)
        chosen = [(s2, s2 + seg - 1)]

    chosen.sort()
    toks, cursor = [], 0
    for s, e in chosen:
        if s > cursor:
            toks.append(str(s - cursor))                     # diffused scaffold gap
        toks.append(f"A{s + 1}-{e + 1}" if e > s else f"A{s + 1}")
        cursor = e + 1
    if cursor < L:
        toks.append(str(L - cursor))
    contig = ",".join(toks)
    n_motif = sum(e - s + 1 for s, e in chosen)
    return {"path": pdb_path, "len": L, "n_motif": n_motif, "contig": contig,
            "segments": [[s, e] for s, e in chosen]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdbs", nargs="*")
    ap.add_argument("--pdb-dir", default=None)
    ap.add_argument("--uniprots", default=None)
    ap.add_argument("--pattern", default="AF-{u}-F1-model_v4_esmfold_v1.pdb")
    args = ap.parse_args()

    items = []
    for p in args.pdbs:
        items.append((os.path.basename(p).split("-")[1] if "-" in os.path.basename(p) else p, p))
    if args.uniprots:
        for u in args.uniprots.split(","):
            items.append((u.strip(), os.path.join(args.pdb_dir, args.pattern.format(u=u.strip()))))

    results = {}
    for uid, path in items:
        try:
            r = carve(path)
            results[uid] = r
            print(f"CONTIG {uid} {r['len']} {r['n_motif']} {r['contig']}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR  {uid} {e}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
