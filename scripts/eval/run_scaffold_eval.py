"""Scaffolding eval driver — does the multigranularity ("editing") SPA scaffold BETTER from a
sub-region-MASKED prompt than the base full-prompt SPA? (dev 17 §7 / 16 §9.5, graduated-hybrid Step 1.)

For each held-out prompt we pick a deterministic contiguous sub-region S (segment / domain / global via
``spa.data.granularity.subregion_pad_mask``, **seeded per prompt so BOTH checkpoints see the SAME mask**),
mask the ESM3 prompt to S (``key_padding_mask`` on SPA's cross-attention, ``eval.subregion``), roll out
RFD3 ± SPA at ``eval.length == N``, and score:

  - **sub-region motif-RMSD** — the design's S region vs the prompt structure's S region (design-side =
    RFD3 backbone; refold-side = OF3, with ``--of3``). The headline "did it scaffold S?" metric.
  - **whole-fold adherence TM** — does masking to S still yield the fold? (informational)
  - **designability scRMSD** — best-of-K self-consistency (only with ``--of3``; ProteinMPNN → OF3).

Compares three groups on IDENTICAL masks per prompt:
  - ``baseline``      — vanilla RFD3 (checkpoint-independent) — the anchor.
  - ``spa-multigran`` — the multigranularity ("editing") ckpt (trained on sub-region-masked prompts).
  - ``spa-base``      — the base unconditional full-prompt SPA (never saw a masked prompt).

**Key claim:** ``spa-multigran`` realizes S (lower sub-region motif-RMSD) better than ``spa-base`` on the
same partial prompt. N×1536 (variant C) ONLY — pooled variants drop the per-residue mask (dev 17 §5).

Run (A5000; adherence + design-side sub-region motif-RMSD, no OF3):
    conda run -n spa-dev python scripts/eval/run_scaffold_eval.py \
        --prompts A0A7S1B8G4,A0A522W419,A0A820JRM2 --granularity segment \
        --multigran-ckpt checkpoints/spa-Nx1536-multigran/spa_C_final.pt \
        --base-ckpt      checkpoints/spa-Nx1536-uncond/spa_C_final.pt \
        --num-designs 4 --lambda 1.0 --out-dir outputs/eval/scaffold

Add ``--of3`` to also compute designability scRMSD (ProteinMPNN → OpenFold3; ≤256 res on the A5000).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

# Run-artifact root — absolute + env-overridable, mirroring configs/paths/default.yaml's
# `outputs_root: ${oc.env:SPA_OUTPUTS_ROOT,${paths.project_root}/outputs}`. A *relative* default
# resolved against the invoking cwd and sent output into whichever repo the script was launched
# from; a *shared* default made runs overwrite each other. See dev docs/plan/30 §6.
_OUTPUTS_ROOT = Path(os.environ.get(
    "SPA_OUTPUTS_ROOT",
    Path(os.environ.get("SPA_PROJECT_ROOT", Path.home() / "projects" / "spa")) / "outputs"))


DEFAULT_PDB_DIR = ("/home/user1/projects/spa/training_data/proteina-atomistica_data_vrelease/"
                   "atomistica_data_release/pdb")
DEFAULT_PATTERN = "AF-{id}-F1-model_v4_esmfold_v1.pdb"


def _prompt_seed(base_seed: int, prompt_id: str) -> int:
    """Stable per-prompt seed (independent of PYTHONHASHSEED) so S is reproducible across runs/ckpts."""
    h = int(hashlib.sha1(prompt_id.encode()).hexdigest(), 16)
    return (base_seed ^ (h & 0xFFFFFFFF)) & 0xFFFFFFFF


def _select_subregion(n: int, granularity: str, seed: int, min_seg: int, pdb_path: str):
    """Deterministic contiguous S for one prompt. Returns ``(g_effective, keep_indices)``.

    Forces the requested granularity by weighting it 1.0 (``subregion_pad_mask`` still degrades
    domain→segment when there is no clean 2-domain split, and segment→global when the window covers N).
    ``keep_indices`` is the sorted 0-based list of S residues (all of [0,N) for the global case).
    """
    import numpy as np

    from spa.data.granularity import subregion_pad_mask

    weights = {"global": 0.0, "segment": 0.0, "domain": 0.0}
    weights[granularity] = 1.0
    g_eff, pad = subregion_pad_mask(
        n, weights=weights, min_seg=min_seg, pdb_path=pdb_path,
        rng=np.random.RandomState(seed),
    )
    keep = list(range(n)) if pad is None else [int(i) for i in np.nonzero(~pad)[0]]
    return g_eff, keep


def _ca_count(pdb_path: str) -> int:
    from spa.eval.score import _as_struct, _ca_array

    return int(len(_ca_array(_as_struct(pdb_path))))


def _base_cfg(variant: str, device: str):
    """Compose the base eval config once (paths/model/variant/hardware/eval groups)."""
    from hydra import compose, initialize_config_dir

    cfg_dir = str((Path(__file__).resolve().parents[2] / "configs"))
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="eval", overrides=[f"variant={variant}"])
    cfg.hardware.device = device
    return cfg


def _precompute_prompts(prompts, pdb_dir, pattern, variant_strip, device, out_dir):
    """Embed every prompt structure with ESM3 ONCE (one model residency), cache ``[N,1536]`` .pt files.

    Returns ``{id: {"pdb": path, "cache": path, "n": N}}``; skips ids whose PDB is missing.
    """
    import torch

    from spa.prompt.esm3_prompt import esm3_prompt, load_esm3

    cache_dir = out_dir / "prompts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    info = {}
    for pid in prompts:
        pdb = Path(pdb_dir) / pattern.format(id=pid)
        if not pdb.exists():
            print(f"[scaffold] SKIP {pid}: no PDB at {pdb}")
            continue
        info[pid] = {"pdb": str(pdb), "cache": str(cache_dir / f"{pid}.pt"), "n": _ca_count(str(pdb))}

    if not info:
        raise SystemExit("[scaffold] no prompts with a readable PDB — nothing to do.")

    print(f"[scaffold] embedding {len(info)} prompt(s) with ESM3 (one residency)...")
    dev = torch.device(device) if isinstance(device, str) else device
    model = load_esm3(dev)
    try:
        for pid, rec in info.items():
            p = esm3_prompt(rec["pdb"], model, strip_bos_eos=variant_strip, use_sequence=False)
            p = p.detach().float().cpu()
            if p.shape[0] != rec["n"]:
                print(f"[scaffold] WARN {pid}: ESM3 prompt N={p.shape[0]} != CA count {rec['n']} "
                      f"(strip_bos_eos issue?) — using ESM3 N for eval.length.")
                rec["n"] = int(p.shape[0])
            torch.save(p, rec["cache"])
            print(f"[scaffold]   {pid}: prompt [{p.shape[0]},{p.shape[1]}] -> {rec['cache']}")
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return info


def _run_group(base_cfg, rec, keep, group, ckpt, conditions, lam, K, seed, timesteps,
               out_dir, refolder):
    """Run the flywheel for one (prompt, group); return the list of per-design score dicts (tagged)."""
    from omegaconf import OmegaConf

    from spa.eval.flywheel import run_flywheel

    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))
    ev = cfg.eval
    ev.ckpt = ckpt
    ev.conditions = list(conditions)
    ev.lambda_scale = float(lam)
    ev.num_designs = int(K)
    ev.length = int(rec["n"])
    ev.seed = int(seed)
    ev.num_timesteps = timesteps
    ev.prompt_cache = rec["cache"]
    ev.prompt_pdb = None
    ev.prompt_id = Path(rec["pdb"]).stem
    ev.subregion = {"keep": [int(i) for i in keep]}
    ev.out_dir = str(out_dir)
    ev.proteinmpnn.out_dir = str(out_dir / "seqs")   # per-run seqs dir (avoid cross-ckpt FASTA collision)
    ev.flywheel.prompt_struct = rec["pdb"]     # adherence + sub-region scoring reference (structure)

    out = run_flywheel(cfg, refolder=refolder)
    tagged = []
    for s in out["scores"]:
        d = s.__dict__ if hasattr(s, "__dict__") else dict(s)
        # relabel the SPA condition by which checkpoint produced it (baseline stays 'baseline').
        cond = d.get("condition")
        d = dict(d)
        d["group"] = group if cond == "spa" else cond
        tagged.append(d)
    return tagged


def _summ(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None and r[key] == r[key]]
    return (sum(vals) / len(vals)) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", required=True, help="comma-separated UniProt-style ids")
    ap.add_argument("--pdb-dir", default=DEFAULT_PDB_DIR)
    ap.add_argument("--pdb-pattern", default=DEFAULT_PATTERN)
    ap.add_argument("--multigran-ckpt", required=True)
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--variant", default="C_n_by_1536")
    ap.add_argument("--granularity", default="segment", choices=["segment", "domain", "global"])
    ap.add_argument("--min-seg", type=int, default=12)
    ap.add_argument("--num-designs", type=int, default=4)
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-timesteps", type=int, default=None, help="RFD3 sampler steps (None=200)")
    ap.add_argument("--of3", action="store_true", help="also score designability scRMSD (ProteinMPNN→OF3)")
    ap.add_argument("--out-dir", default=str(_OUTPUTS_ROOT / "_incoming" / "scaffold"))
    args = ap.parse_args()

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = _base_cfg(args.variant, args.device)
    variant_strip = bool(base_cfg.variant.get("strip_bos_eos", True))

    info = _precompute_prompts(prompts, args.pdb_dir, args.pdb_pattern, variant_strip, args.device, out_dir)

    refolder = None
    if args.of3:
        from spa.eval.openfold3 import OF3Refolder
        refolder = OF3Refolder(
            ckpt_path=base_cfg.paths.openfold3_ckpt,
            runner_yaml=base_cfg.paths.openfold3_runner_yaml,
            out_dir=str(out_dir / "of3"),
            conda_env="spa-verify-of3",
        )

    all_rows: list[dict] = []
    per_prompt: list[dict] = []
    for pid, rec in info.items():
        pseed = _prompt_seed(args.seed, pid)
        g_eff, keep = _select_subregion(rec["n"], args.granularity, pseed, args.min_seg, rec["pdb"])
        cov = f"{len(keep)}/{rec['n']}"
        print(f"\n[scaffold] === {pid} (N={rec['n']}) granularity={g_eff} S={cov} "
              f"[{min(keep)}..{max(keep)}] ===")

        # Call 1 (multigran ckpt): baseline + spa-multigran. Call 2 (base ckpt): spa-base.
        rows = _run_group(base_cfg, rec, keep, "spa-multigran", args.multigran_ckpt,
                          ["baseline", "spa"], args.lam, args.num_designs, args.seed,
                          args.num_timesteps, out_dir / pid / "multigran", refolder)
        rows += _run_group(base_cfg, rec, keep, "spa-base", args.base_ckpt,
                           ["spa"], args.lam, args.num_designs, args.seed,
                           args.num_timesteps, out_dir / pid / "base", refolder)
        for r in rows:
            r["prompt_id"] = pid
        all_rows += rows

        row = {"prompt_id": pid, "n": rec["n"], "granularity": g_eff, "coverage": cov,
               "keep": [min(keep), max(keep)]}
        for grp in ("baseline", "spa-multigran", "spa-base"):
            g = [r for r in rows if r["group"] == grp]
            row[grp] = {
                "n": len(g),
                "motif_rmsd_S": _summ(g, "motif_rmsd"),                  # design-side sub-region RMSD
                "motif_rmsd_S_refold": _summ(g, "motif_rmsd_refold"),   # OF3-side (with --of3)
                "tm_whole": _summ(g, "tm_score"),
                "scrmsd": _summ(g, "scrmsd"),
                "designable_rate": (sum(1 for r in g if r.get("designable")) / len(g)) if g else None,
            }
        per_prompt.append(row)

    # Report.
    print("\n" + "=" * 92)
    print("SCAFFOLDING EVAL — sub-region motif-RMSD (design-side): does multigran realize S from a masked prompt?")
    print("=" * 92)
    hdr = f"{'prompt':<12}{'N':>5}{'gran':>8}{'S':>8}   {'baseline':>10}{'spa-base':>10}{'spa-multi':>11}{'Δ(mg−base)':>12}"
    print(hdr); print("-" * len(hdr))
    for row in per_prompt:
        b = row["baseline"]["motif_rmsd_S"]; sb = row["spa-base"]["motif_rmsd_S"]; sm = row["spa-multigran"]["motif_rmsd_S"]
        d = (sm - sb) if (sm is not None and sb is not None) else None
        f = lambda x: "n/a" if x is None else f"{x:.3f}"
        print(f"{row['prompt_id']:<12}{row['n']:>5}{row['granularity']:>8}{row['coverage']:>8}   "
              f"{f(b):>10}{f(sb):>10}{f(sm):>11}{f(d):>12}")
    # overall means
    def _ov(grp):
        v = [row[grp]["motif_rmsd_S"] for row in per_prompt if row[grp]["motif_rmsd_S"] is not None]
        return (sum(v) / len(v)) if v else None
    ob, osb, osm = _ov("baseline"), _ov("spa-base"), _ov("spa-multigran")
    f = lambda x: "n/a" if x is None else f"{x:.3f}"
    print("-" * len(hdr))
    print(f"{'OVERALL':<12}{'':>5}{'':>8}{'':>8}   {f(ob):>10}{f(osb):>10}{f(osm):>11}"
          f"{f((osm - osb) if (osm is not None and osb is not None) else None):>12}")
    print("\n[read] lower sub-region motif-RMSD = better scaffolding of S. Claim holds if spa-multigran < spa-base.")

    payload = {"config": vars(args), "per_prompt": per_prompt, "scores": all_rows}
    (out_dir / "scaffold_results.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[scaffold] wrote {out_dir / 'scaffold_results.json'} ({len(all_rows)} design scores)")


if __name__ == "__main__":
    main()
