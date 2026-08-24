"""Phase A of the SPA guidance-scale experiment: sweep ω at fixed λ (dev ``docs/plan/56_cfg_guidance_for_spa.md``).

Runs, per fold, an unsteered baseline plus a λ=1 arm at each ω, and scores **adherence, diversity and
physicality together**. Adherence-only would make a knob that buys fold-similarity by wrecking geometry
look like a win, which is exactly what ``plan/56`` §8 forbids reporting.

⭐ THIS IS §6.3 STEP 1, SOFT-ONLY, AND DELIBERATELY SO
No native motif is placed. The hard ⊕ soft comparison is step 2, the mandatory risk-check for §6.2 (CFG
rewrites ``delta_L``, which is applied to every atom including pinned ones, so ω could silently degrade
a motif pin). Do not run it until soft-only clears, and never report a hard ⊕ soft ω cell without
``motif_rmsd`` beside it.

CFG IS ARMED ON THE LIVE OBJECTS, NOT THROUGH CONFIG
``spa.model.cfg_guidance.arm_cfg`` sets the five attributes the host reads at forward time. That keeps
``generate.py``, ``configs/eval/default.yaml`` and ``_assert_sampler_effective`` untouched, so a run
without CFG is unchanged because no shared code changed, not because a default was chosen carefully.
It also sidesteps a real trap: via the config route, ``cfg_scale=1.0`` leaves the net's flag False while
the sampler's stays True, and the sampler enters its CFG block with ``f_ref=None`` and raises. Arming
post-construction sets both flags explicitly, which is what makes ω=1.0 an exact, runnable identity.

⭐ WHY THERE ARE TWO CFG-OFF ARMS
``plan/56`` §8 makes ω=1.0 an **in-run** identity control. But this host is nondeterministic on GPU
(``07`` I.12: bf16 autocast, and none of the four determinism levers move it), so "identical" cannot mean
"bit-equal" and comparing against zero would be wrong. ``results/27`` learned the same lesson at the step
level and its answer was to measure the floor rather than assume it. So ``cfg_off`` runs **twice**, and
the ω=1.0 arm is read against that measured run-to-run floor. Without it the identity control is
uninterpretable.

Usage:
    conda run -n spa-dev python scripts/eval/probe_cfg_omega.py \\
        --manifest configs/eval/manifest_curated15.yaml \\
        --pdb-dir <cddb pdb dir> --ckpt models/spa-Nx1536-uncond.pt \\
        -K 8 --out-dir <dir> --json <results.json>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
_PROJECT = _HERE.parents[3]

# ω=1.0 is the in-run identity control, not a spacer: it runs the full second forward pass and the
# extrapolation, and (1.0-1.0) is exactly 0.0, so it must land on cfg_off within the measured floor.
DEFAULT_OMEGAS = "1.0,1.5,2.0,3.0"


def _seed_all(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------------------------------------
# physicality: failure mode 1 (plan/56 §6.1) needs a metric, not an impression
# ------------------------------------------------------------------------------------------------

def _physicality(atom_array) -> dict:
    """Cheap geometry sanity for a backbone: Cα-Cα virtual bonds and non-local Cα clashes.

    ``plan/56`` §6.1 is explicit that coordinate-space overshoot shows up as non-physical geometry
    rather than as the tolerable oversaturation image CFG produces, and §8 pre-registers "non-physical
    geometry at ω ≤ 1.5" as a stop condition. That needs a number. Consecutive Cα in a real protein sit
    at ~3.80 Å (trans peptide); a stretched or collapsed chain shows up immediately.

    Returns mean/sd of the consecutive Cα distance, the fraction of those outside 3.80 ± 0.5 Å, and the
    count of non-adjacent Cα pairs closer than 4.0 Å (a steric impossibility for Cα).
    """
    import numpy as np

    from spa.eval.score import _as_struct, _ca_array

    ca = _ca_array(_as_struct(atom_array))
    xyz = np.asarray(ca.coord, dtype=np.float64)
    if xyz.ndim == 3:                                     # a stack; take the first model
        xyz = xyz[0]
    n = len(xyz)
    if n < 3:
        return {"n_ca": n, "ca_bond_mean": None, "ca_bond_sd": None,
                "frac_bond_outlier": None, "n_ca_clashes": None}
    d = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    dist = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    iu = np.triu_indices(n, k=3)                          # |i-j| >= 3 is non-local
    return {
        "n_ca": int(n),
        "ca_bond_mean": float(d.mean()),
        "ca_bond_sd": float(d.std()),
        "frac_bond_outlier": float(np.mean(np.abs(d - 3.80) > 0.5)),
        "n_ca_clashes": int(np.sum(dist[iu] < 4.0)),
    }


def _paired_rmsd(arms: dict, a: str, b: str) -> float | None:
    """Mean Cα RMSD between same-index designs of two arms.

    Every arm is re-seeded to the same value before its rollout, so design ``i`` of arm A and design
    ``i`` of arm B start from the **same** initial noise (``generate.py``'s paired-noise convention).
    The residual is therefore the effect of the arm plus the GPU's own nondeterminism, which is why the
    floor arm exists to separate them.
    """
    from spa.eval.score import ca_rmsd

    if a not in arms or b not in arms:
        return None
    xa, xb = arms[a], arms[b]
    if not xa or not xb or len(xa) != len(xb):
        return None
    vals = [ca_rmsd(p, q) for p, q in zip(xa, xb)]
    return float(sum(vals) / len(vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(_REPO / "configs" / "eval" / "manifest_curated15.yaml"))
    ap.add_argument("--folds", default=None,
                    help="comma-separated UniProt ids; default = every entry in --manifest")
    ap.add_argument("--pdb-dir", default=str(_PROJECT / "training_data" /
                                             "proteina-atomistica_data_vrelease" /
                                             "atomistica_data_release" / "pdb"))
    ap.add_argument("--ckpt", default=str(_REPO / "models" / "spa-Nx1536-uncond.pt"))
    ap.add_argument("-K", "--num-designs", type=int, default=8)
    ap.add_argument("--omegas", default=DEFAULT_OMEGAS)
    ap.add_argument("--lambda-scale", type=float, default=1.0,
                    help="λ, held FIXED across the ω sweep. plan/56 §2.6a: two interacting strength "
                         "knobs is how a cheap screen becomes an expensive grid.")
    ap.add_argument("--timesteps", type=int, default=None, help="None -> checkpoint default (100)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cfg-t-max", type=float, default=None,
                    help="guidance applies only while c_t > this; None = every step")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=str(_PROJECT / "outputs" / "_incoming" / "cfg_omega"))
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-floor", action="store_true",
                    help="skip the repeated cfg_off arm. Saves ~9%% of the run and makes the ω=1.0 "
                         "identity control uninterpretable; do not use for the real run.")
    a = ap.parse_args()

    import torch
    import yaml
    from omegaconf import OmegaConf

    from spa.eval import provenance as _prov
    from spa.eval.generate import _run_once, build_eval_engine, load_adapter, write_pdb
    from spa.eval.score import _as_struct, adherence, pairwise_tm_diversity
    from spa.model.cfg_guidance import CFGPromptSwap, arm_cfg, disarm_cfg, read_cfg_state
    from spa.train.harness import frozen_rfd3_net
    from spa.utils.device import resolve_device

    out_root = Path(a.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    dev = resolve_device(a.device)
    omegas = [float(x) for x in a.omegas.split(",") if x.strip()]

    man = yaml.safe_load(Path(a.manifest).read_text())
    pattern = man["pdb_pattern"]
    entries = man["prompts"]
    if a.folds:
        want = [f.strip() for f in a.folds.split(",") if f.strip()]
        entries = [e for e in entries if e["id"] in want]
        missing = sorted(set(want) - {e["id"] for e in entries})
        if missing:
            raise SystemExit(f"--folds ids not in {a.manifest}: {missing}")
    print(f"[omega] {len(entries)} fold(s), ω ∈ {omegas}, λ={a.lambda_scale}, K={a.num_designs}, "
          f"seed={a.seed}\n")

    # --- prompts first, ESM3 loaded ONCE for every fold, then freed before RFD3 runs ----------------
    # resolve_prompt() loads and frees ESM3 per call, which would pay that cost 15 times and, worse,
    # risk ESM3 co-residing with RFD3 on a 24 GB card.
    from spa.prompt.esm3_prompt import esm3_prompt, load_esm3

    t0 = time.time()
    esm3 = load_esm3(dev)
    prompts = {}
    try:
        for e in entries:
            pdb = Path(a.pdb_dir) / pattern.format(id=e["id"])
            if not pdb.exists():
                raise SystemExit(f"missing prompt structure: {pdb}")
            p = esm3_prompt(str(pdb), esm3, strip_bos_eos=True, use_sequence=False)
            prompts[e["id"]] = (p.detach().float().to(dev), str(pdb))
            print(f"[omega] prompt {e['id']}: N={p.shape[0]}")
    finally:
        del esm3
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"[omega] {len(prompts)} prompt(s) in {time.time() - t0:.0f}s; ESM3 freed\n")

    all_folds = []
    for e in entries:
        fid = e["id"]
        prompt_raw, pdb_path = prompts[fid]
        N = int(prompt_raw.shape[0])
        fold_dir = out_root / fid
        fold_dir.mkdir(parents=True, exist_ok=True)

        cfg = OmegaConf.create({
            "paths": {"rfd3_ckpt": str(_PROJECT / "models" / "rfdiffusion3" / "rfd3_latest.ckpt")},
            "hardware": {"device": a.device},
            "model": {"c_query": 768, "c_kv": 1536, "c_model": 768, "n_head": 8, "shared_kv": True,
                      "zero_init_output": True, "lambda_init": 1.0, "input_rmsnorm": True},
            "variant": {"name": "C", "projector": "identity", "resampler_tokens": None,
                        "strip_bos_eos": True, "use_clss": False},
            # self-prompt convention (results/07): the design is the prompt's own length
            "eval": {"num_designs": a.num_designs, "length": N, "specification": None,
                     "num_timesteps": a.timesteps, "seed": a.seed, "ckpt": a.ckpt,
                     "prompt_pdb": pdb_path, "prompt_cache": None, "use_sequence": False},
        })
        _prov.write(fold_dir, cfg, started=_prov._now_pacific(),
                    purpose=f"CFG Phase A (dev plan/56 §6.3 step 1, soft-only): ω sweep, fold {fid}",
                    scope="phase-A adherence + diversity + physicality, no OpenFold3",
                    prompts=[fid])

        engine = build_eval_engine(cfg)
        net = frozen_rfd3_net(engine)
        adapter = load_adapter(net, cfg, dev); adapter.eval()
        adtype = next(adapter.parameters()).dtype
        prompt = prompt_raw[None].expand(a.num_designs, -1, -1).to(device=dev, dtype=adtype).contiguous()

        n_steps = int(getattr(net.inference_sampler.sampler, "num_timesteps", a.timesteps or 100))
        prompt_struct = _as_struct(pdb_path)

        # arm list: (label, lambda, omega or None). None = CFG entirely off.
        arm_spec = [("baseline", 0.0, None), ("cfg_off", a.lambda_scale, None)]
        if not a.no_floor:
            arm_spec.append(("cfg_off_rep", a.lambda_scale, None))
        arm_spec += [(f"omega{w:g}", a.lambda_scale, w) for w in omegas]

        rows, structs_by_arm = [], {}
        print(f"[omega] === {fid}  N={N}  {n_steps} steps  {len(arm_spec)} arms ===")
        for label, lam, omega in arm_spec:
            t_arm = time.time()
            if label == "baseline":
                adapter.clear_prompt()
            else:
                adapter.set_prompt(prompt)
                adapter.set_scale(lam)

            hook = original = prior = None
            if omega is not None:
                prior = arm_cfg(net, cfg_scale=omega, cfg_features=[], cfg_t_max=a.cfg_t_max)
                hook = CFGPromptSwap(net.diffusion_module, adapter, batch=a.num_designs)
                original = hook.install(net)
            cfg_state = read_cfg_state(net)

            try:
                _seed_all(a.seed)                    # paired noise across every arm
                with torch.no_grad():
                    outs = _run_once(engine)
            finally:
                if hook is not None:
                    CFGPromptSwap.uninstall(net, original)
                    disarm_cfg(net, prior)

            # ⭐ the only real proof CFG ran. A config readback cannot see a post-construction arming,
            # and an arm that silently skipped its reference pass is an ω=1 arm wearing another label.
            hs = hook.summary() if hook is not None else {"n_conditional": 0, "n_reference": 0,
                                                          "omega_min": None, "omega_max": None,
                                                          "omega_mean": None}
            if omega is not None and hs["n_reference"] == 0:
                raise RuntimeError(
                    f"{fid}/{label}: CFG was armed (cfg_scale={omega}) but the wrapper never saw a "
                    f"reference pass, so no guidance was applied. Refusing to record this arm as ω="
                    f"{omega}. Live CFG state was {cfg_state}.")

            adir = fold_dir / label; adir.mkdir(parents=True, exist_ok=True)
            paths, designs = [], []
            for i, o in enumerate(outs):
                p = adir / f"{fid}_{label}_{i}.pdb"
                write_pdb(o.atom_array, p)
                paths.append(p)
            structs_by_arm[label] = paths

            for i, o in enumerate(outs):
                adh = adherence(o.atom_array, prompt_struct, tm_norm="prompt")
                designs.append({"idx": i, "pdb": str(paths[i]), "tm_prompt": adh.tm_norm_prompt,
                                "tm_design": adh.tm_norm_design, **_physicality(o.atom_array)})
            tms = [d["tm_prompt"] for d in designs]
            div = pairwise_tm_diversity([o.atom_array for o in outs])
            phys_bad = sum(1 for d in designs if (d["frac_bond_outlier"] or 0) > 0.02
                           or (d["n_ca_clashes"] or 0) > 0)

            rows.append({
                "arm": label, "lambda": lam, "omega": omega, "n_designs": len(outs),
                "tm_prompt_mean": sum(tms) / len(tms), "tm_prompt_min": min(tms),
                "tm_prompt_max": max(tms), "diversity_tm": div,
                "ca_bond_mean": sum(d["ca_bond_mean"] for d in designs) / len(designs),
                "frac_bond_outlier_mean": sum(d["frac_bond_outlier"] for d in designs) / len(designs),
                "n_ca_clashes_total": sum(d["n_ca_clashes"] for d in designs),
                "n_designs_nonphysical": phys_bad,
                "cfg_state": cfg_state, "hook": hs, "n_sampler_steps": n_steps,
                "seconds": round(time.time() - t_arm, 1), "out_dir": str(adir), "designs": designs,
            })
            _f = lambda v: "  n/a" if v is None else f"{v:.3f}"
            print(f"[omega] {label:<12} ω={str(omega):<5} TM={_f(rows[-1]['tm_prompt_mean'])} "
                  f"div={_f(div)} bond={rows[-1]['ca_bond_mean']:.2f}Å "
                  f"clash={rows[-1]['n_ca_clashes_total']:<4} ref={hs['n_reference']}/{hs['n_conditional']} "
                  f"{rows[-1]['seconds']:>6.1f}s")

        # --- the in-run identity control, read against the MEASURED floor ---------------------------
        floor = _paired_rmsd(structs_by_arm, "cfg_off", "cfg_off_rep")
        # find the ω=1.0 arm by VALUE, not by its formatted label, so changing --omegas cannot
        # silently drop the identity control.
        ident_arm = next((r["arm"] for r in rows if r["omega"] == 1.0), None)
        ident = _paired_rmsd(structs_by_arm, "cfg_off", ident_arm) if ident_arm else None
        base_tm = next(r["tm_prompt_mean"] for r in rows if r["arm"] == "baseline")
        off_tm = next(r["tm_prompt_mean"] for r in rows if r["arm"] == "cfg_off")
        for r in rows:
            r["dtm_vs_baseline"] = r["tm_prompt_mean"] - base_tm
            r["dtm_vs_cfg_off"] = r["tm_prompt_mean"] - off_tm
            r["paired_ca_rmsd_vs_cfg_off"] = _paired_rmsd(structs_by_arm, "cfg_off", r["arm"])

        verdict = "not measured (no floor arm, or ω=1.0 not in the sweep)"
        if floor is not None and ident is not None:
            verdict = ("✅ ω=1.0 is within the run-to-run floor" if ident <= floor * 1.5 else
                       "⚠️ ω=1.0 deviates BEYOND the floor: the armed CFG path is not an identity")
        print(f"[omega] {fid}: identity control -> ω=1.0 vs cfg_off {_fmt(ident)} Å, "
              f"run-to-run floor {_fmt(floor)} Å  {verdict}")

        rec = {"fold": fid, "n_residues": N, "fold_class": e.get("fold"), "arms": rows,
               "identity_ca_rmsd": ident, "floor_ca_rmsd": floor, "identity_verdict": verdict,
               "config": {"ckpt": a.ckpt, "K": a.num_designs, "seed": a.seed,
                          "lambda": a.lambda_scale, "omegas": omegas, "num_timesteps": n_steps,
                          "cfg_t_max": a.cfg_t_max, "prompt_pdb": pdb_path, "tm_norm": "prompt",
                          "cfg_features": [], "motif": None}}
        (fold_dir / f"{fid}.json").write_text(json.dumps(rec, indent=2))
        all_folds.append(rec)
        print(f"[omega] {fid} done -> {fold_dir}\n")

        del engine, net, adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({"folds": all_folds, "run_dir": str(out_root)}, indent=2))
        print(f"[omega] wrote {a.json}")
    print(f"[omega] {len(all_folds)} fold(s) -> {out_root}")
    print("[omega] read: higher TM = more adherent; LOWER diversity_tm = more diverse. Report "
          "adherence, diversity and physicality TOGETHER (plan/56 §8), never adherence alone.")


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


if __name__ == "__main__":
    main()
