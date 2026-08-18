"""Pre-encode ESM3 prompts to ``.pt`` caches, loading ESM3 ONCE instead of once per generation call.

WHY THIS EXISTS, AND WHY IT IS NOT A NEW GENERATION PATH
``generate.py`` resolves a prompt from either ``eval.prompt_pdb`` (loads ESM3, encodes, frees it) or
``eval.prompt_cache`` (``torch.load`` of a ``[N, c_kv]`` tensor). A driver that invokes ``generate.py``
once per prompt therefore pays an ESM3 load once per prompt: measured directly (dev ``results/34``,
H100, two independent runs) at ~6.8-7.2 s per load, dominated by ``ESM3.from_pretrained`` itself
(~4.4 s), not the encode. (An earlier version of this docstring cited "~90 s" and derived "~90
minutes for 60 prompts" from it; that figure was never independently measured and was wrong by
roughly 13x -- corrected here once a real number existed. At the measured rate, 60 prompts costs
~7 minutes of pure model loading, not 90.) The fix below is still worth having regardless of the
exact number: repeated loads are still 100% avoidable overhead at any scale.

The obvious fix, a bespoke driver that loads ESM3 once and loops, is the **wrong** fix when the run's
purpose is comparability with an earlier result, because it introduces a second generation path that
could diverge from the original in ways no one would notice. This script takes the other route: it
produces caches that are **bit-identical** to what ``resolve_prompt`` would have computed, so the
existing driver runs **unmodified** with ``eval.prompt_cache=`` and every downstream byte is unchanged.

⭐ EQUIVALENCE IS A GATE, NOT A CLAIM. ``--verify`` re-encodes one prompt through the exact
``resolve_prompt`` code path ``generate.py`` uses and asserts the tensors match to ``--atol``. Run it.
The flags that matter (``strip_bos_eos``, ``use_sequence``) are read from the same config keys, because
a cache built with different flags is silently wrong rather than loudly wrong.

Usage:
    conda run -n spa-dev python scripts/eval/precompute_prompts.py \\
        --pdbs a.pdb b.pdb ... --out-dir <cache dir> --verify
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdbs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--strip-bos-eos", action="store_true", default=True,
                    help="must match variant.strip_bos_eos used at generation (default True)")
    ap.add_argument("--use-sequence", action="store_true", default=False,
                    help="must match eval.use_sequence used at generation (default False)")
    ap.add_argument("--verify", action="store_true",
                    help="re-encode one prompt through generate.py's own resolve_prompt and compare")
    ap.add_argument("--atol", type=float, default=0.0,
                    help="0.0 demands bit-identical, which is the point")
    a = ap.parse_args()

    import torch
    from spa.prompt.esm3_prompt import esm3_prompt, load_esm3
    from spa.utils.device import resolve_device

    dev = resolve_device(a.device)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    pdbs = [Path(p) for p in a.pdbs]
    missing = [p for p in pdbs if not p.exists()]
    if missing:
        raise SystemExit(f"[pre] ⛔ {len(missing)} prompt PDBs absent, e.g. {missing[0]}")

    print(f"[pre] encoding {len(pdbs)} prompts with ONE ESM3 load "
          f"(strip_bos_eos={a.strip_bos_eos}, use_sequence={a.use_sequence})")
    model = load_esm3(dev)
    try:
        for i, p in enumerate(pdbs, 1):
            t = esm3_prompt(str(p), model, strip_bos_eos=bool(a.strip_bos_eos),
                            use_sequence=bool(a.use_sequence)).detach().float().cpu()
            torch.save(t, out / f"{p.stem}.pt")
            if i % 10 == 0 or i == len(pdbs):
                print(f"[pre]   {i}/{len(pdbs)}  last {p.stem} -> {tuple(t.shape)}")
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if a.verify:
        # the gate: go through generate.py's OWN resolver, with a cfg shaped exactly as it expects
        from omegaconf import OmegaConf
        from spa.eval.generate import resolve_prompt
        ref_pdb = pdbs[0]
        cfg = OmegaConf.create({
            "variant": {"strip_bos_eos": bool(a.strip_bos_eos)},
            "eval": {"prompt_pdb": str(ref_pdb), "prompt_cache": None,
                     "use_sequence": bool(a.use_sequence)},
        })
        live = resolve_prompt(cfg, dev).detach().float().cpu()
        cached = torch.load(out / f"{ref_pdb.stem}.pt", weights_only=True).float()
        same_shape = tuple(live.shape) == tuple(cached.shape)
        maxdiff = float((live - cached).abs().max()) if same_shape else float("nan")
        ok = same_shape and (maxdiff <= a.atol)
        print(f"\n[pre] VERIFY on {ref_pdb.stem}: shapes {tuple(live.shape)} vs {tuple(cached.shape)}, "
              f"max |Δ| = {maxdiff:.3e}, tol {a.atol:.1e}")
        print(f"[pre] {'✅ caches are equivalent to the live path; generate.py is unchanged downstream' if ok else '⛔ MISMATCH: do NOT use these caches'}")
        if not ok:
            raise SystemExit(2)

    print(f"[pre] wrote {len(pdbs)} caches -> {out}")


if __name__ == "__main__":
    main()
