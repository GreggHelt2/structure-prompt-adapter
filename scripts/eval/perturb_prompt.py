"""Build perturbed prompt caches whose ROW CONTENT changes, for the far end of the sensitivity ladder.

WHY THIS EXISTS. Dev ``results/42`` §11.4a established that ``shuffle`` is a **mathematical no-op**:
``SPACrossAttention`` takes a softmax over the M prompt keys and returns a weighted sum of the values,
with no positional encoding on the key side, so permuting the rows permutes ``k`` and ``v`` by the same
permutation and the output is identical. **SPA reads the prompt as an unordered SET.** ⇒ To perturb a
prompt meaningfully one must change the row CONTENT, and §11.4 named that as the informative direction.

⛔ ROW ORDER IS NOT A VARIABLE HERE, and no perturbation below uses it. Replacing "a random subset of
rows" and "the first k rows" are the same experiment for this architecture; a fixed seed is used only so
the artifact is reproducible, not because position carries anything.

Two perturbation families, both reported in §11.3a's metric so they extend the existing ladder:

    perturbation = median over rows of  || row_old - row_new ||2 / || row_old ||2

``noise``  Gaussian noise added to every row, scaled so the perturbation lands on a TARGET value.
           ⭐ This is the control §11.3a said was missing: it reaches the SAME magnitude as a single
           residue deletion (4.6% and 15.7% are the two measured points) by a DIFFERENT cause, which
           separates "how big is the change" from "what caused it".
``mix``    A fraction f of rows replaced by rows drawn from a DIFFERENT fold's prompt. This is a
           semantic perturbation rather than a numerical one, and f=1.0 is the other fold outright.
           ⚠️ Row counts need not match; rows are drawn without replacement, and the donor is
           resampled if it is shorter than the number of rows being replaced.

Usage:
    conda run -n spa-dev python scripts/eval/perturb_prompt.py \
        --base .../1TEN_A_bbcomplete.pt --out-dir .../ladder \
        --noise 0.05 0.16 0.50 1.00 --mix .../256B_A_clean.pt:0.25,0.50,1.00 --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path


def perturbation(old, new) -> tuple[float, float, float]:
    """Relative L2 change per row -> (median over ALL rows, mean over all, median over CHANGED rows).

    ⛔ §11.3a's metric is the FIRST of these, and it DEGENERATES on a sparse perturbation. Deleting a
    residue moves every surviving row, because ESM3 is a transformer, so a median over all rows is the
    right summary there. Replacing a fraction f < 0.5 of rows leaves the majority BIT-IDENTICAL, and the
    median is then exactly 0.0 while the prompt has genuinely changed. Measured, not predicted: the
    f=0.25 and f=0.50 mixes below both report a median of 0.0000.

    ⇒ The three are reported together, and a sparse perturbation must be read on the third.
    """
    import torch

    n = min(old.shape[0], new.shape[0])
    d = (old[:n] - new[:n]).norm(dim=1) / old[:n].norm(dim=1)
    changed = d[d > 0]
    med_changed = float(torch.median(changed)) if changed.numel() else 0.0
    return float(torch.median(d)), float(d.mean()), med_changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="the reference prompt cache, [N, c_kv]")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--noise", nargs="*", type=float, default=[],
                    help="target perturbation values, e.g. 0.05 0.16 0.50")
    ap.add_argument("--mix", default=None,
                    help="<donor .pt>:<f1,f2,...>, fractions of rows replaced by donor rows")
    ap.add_argument("--seed", type=int, default=42,
                    help="fixed and recorded; row CHOICE is reproducible, row ORDER is not a variable")
    a = ap.parse_args()

    import torch

    base = torch.load(a.base, map_location="cpu", weights_only=False)
    if not torch.is_tensor(base) or base.ndim != 2:
        raise SystemExit(f"⛔ {a.base} is not a [N, c_kv] tensor")
    N, D = base.shape
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.base).stem
    print(f"[perturb] base {stem}: {N} rows x {D}")
    made = []

    for t in a.noise:
        g = torch.Generator().manual_seed(a.seed)
        # || eps || ~ s*sqrt(D) for eps ~ N(0, s^2 I), so s = t*||row||/sqrt(D) hits the target per row.
        s = (t * base.norm(dim=1) / (D ** 0.5)).unsqueeze(1)
        new = base + s * torch.randn(base.shape, generator=g)
        p = perturbation(base, new)
        dst = out / f"{stem}__noise{t:g}.pt"
        torch.save(new, dst)
        print(f"[perturb] noise target {t:g}  -> median {p[0]:.4f} mean {p[1]:.4f} "
              f"median-of-changed {p[2]:.4f}  {dst.name}")
        made.append((dst, p, N))

    if a.mix:
        donor_path, fracs = a.mix.rsplit(":", 1)
        donor = torch.load(donor_path, map_location="cpu", weights_only=False)
        print(f"[perturb] donor {Path(donor_path).stem}: {donor.shape[0]} rows")
        for f in [float(x) for x in fracs.split(",")]:
            g = torch.Generator().manual_seed(a.seed)
            k = int(round(f * N))
            pos = torch.randperm(N, generator=g)[:k]
            idx = torch.randperm(donor.shape[0], generator=g)
            while idx.numel() < k:                      # donor shorter than the block being replaced
                idx = torch.cat([idx, torch.randperm(donor.shape[0], generator=g)])
            new = base.clone()
            new[pos] = donor[idx[:k]].to(base.dtype)
            p = perturbation(base, new)
            dst = out / f"{stem}__mix{f:g}_{Path(donor_path).stem}.pt"
            torch.save(new, dst)
            print(f"[perturb] mix f={f:g} ({k}/{N} rows) -> median {p[0]:.4f} mean {p[1]:.4f} "
                  f"median-of-changed {p[2]:.4f}  {dst.name}")
            made.append((dst, p, k))

    print("\n[perturb] the ladder. §11.3a's metric is 'median'; read a SPARSE row-replacement on")
    print("[perturb] 'med-changed' instead, and see the perturbation() docstring for why.")
    print(f"  {'median':>8} {'mean':>8} {'med-changed':>12} {'rows chg':>9}  file")
    for dst, p, k in sorted(made, key=lambda x: x[1][1]):
        print(f"  {p[0]:8.4f} {p[1]:8.4f} {p[2]:12.4f} {k:>9}  {dst.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
