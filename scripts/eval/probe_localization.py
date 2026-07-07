"""Step-2 localization probe — FAIR cross-prompt version (dev 16 §7 / §6.5 / §9.7 Step 2; dev 19).

The successor to the confounded ``probe_threeway.py`` (2026-06-30), which was self-prompt +
full-structure prompt + a globally-trained SPA (16 §6.5's three confounds). Here we **steer a
sub-region S of a design toward a WHOLE FOREIGN fold G** (a different held-out structure), with SPA
**output-gated to S only** (``set_profile`` = 1 on S, 0 on F), on the **multigranularity ckpt**.
Using a foreign fold as the prompt removes two confounds *for free*: the conditioning carries **zero**
information about the free region F (Conception-2), and G is foreign to F so any F motion toward G is
**pure host-coherence drag**, not self-coherence. The multigran ckpt removes the third (globally-
trained). F gets **no direct SPA term** — its only path to move is RFD3's frozen-host S↔F coupling,
which is exactly the §6.5 crux we measure.

**Question (16 §7):** can SPA steer S while F stays free? Success = S adheres to G **and** F-drag ≈ 0;
a persistent F-drag even here is a **clean trained-negative** (host coherence dominates), publishable
unlike the confounded zero-shot.

Three FRAMES (dev 19 §... — the frame ≈ only sets L and the S/F split; F is unconditioned in all):
  - ``balanced``    : L = 2|G|, S = [0,|G|) steered→G, F = [|G|,2|G|) free. |S|=|F|=|G| (length-matched).
  - ``fixed_l``     : fixed L, S = [0,L/2), F = [L/2,L). Arbitrary mid-fold seam; |S|≠|G|.
  - ``host_domain`` : real 2-domain host H, L=|H|, S = domain-1 [0,b), F = domain-2 [b,L) (contact-map
                      split). Natural domain ratio, but the seam is notional (design is unconditional).

Conditions per example (paired noise — same seed ⇒ clean A/B):
  - ``free``            : SPA off (λ=0) — the unsteered reference.
  - ``localized`` (×λ)  : profile = step(1@S,0@F), prompt = G — the TEST (steer S, F gated off).
  - ``feather``  (×λ)   : profile = cosine ramp at the seam (soft-junction variant).      [--feather]
  - ``global``   (×λ)   : profile = uniform, prompt = G — positive control / F-drag CEILING.

Metrics (region TM to G, superposition-invariant; paired to ``free`` per design idx):
  - S-steer   = TM(loc[S]→G) − TM(free[S]→G)        — must be strongly positive, else the probe is void.
  - F-drag    = TM(loc[F]→G) − TM(free[F]→G)        — the headline (16 §7); want ≈ 0.
  - F-disp    = 1 − TM(loc[F], free[F]) (paired)    — how much F moved at all (diagnostic).
  - loc-index = 1 − F-drag(loc)/F-drag(global)      — 1.0 = perfect localization, 0 = fully dragged.

Adherence-only (no ProteinMPNN/OF3) ⇒ fast + A5000-friendly (≤~256 res). N×1536 (variant C) only.

Run (A5000):
    conda run -n spa-dev python scripts/eval/probe_localization.py \
        --ckpt checkpoints/spa-Nx1536-multigran/spa_C_final.pt \
        --foreign A0A1X7NTP0,A0A6A0D1E8,A0A090ME36,A0A7S1B8G4,A0A820JRM2 \
        --hosts   A0A522W419,A0A6J8EPQ1,A0A7C9GW19,H1SDK8,A0A1X0IID6 \
        --frames balanced,fixed_l,host_domain --lambdas 1,2 --feather \
        --num-designs 8 --out-dir outputs/eval/localization
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DEFAULT_PDB_DIR = ("/home/user1/projects/spa/training_data/proteina-atomistica_data_vrelease/"
                   "atomistica_data_release/pdb")
DEFAULT_PATTERN = "AF-{id}-F1-model_v4_esmfold_v1.pdb"


# --------------------------------------------------------------------------------------------------
# Structure / prompt helpers
# --------------------------------------------------------------------------------------------------


def _pdb(pdb_dir, pid):
    return str(Path(pdb_dir) / DEFAULT_PATTERN.format(id=pid))


def _ca(pid_pdb):
    from spa.eval.score import _as_struct, _ca_array

    return _ca_array(_as_struct(pid_pdb))


def _slice_tm(design_ca, lo, hi, target_ca):
    """max-normalized Cα TM of ``design[lo:hi]`` vs the whole ``target`` (superposition-invariant)."""
    import tmtools

    from spa.eval.score import _coords64, _seq_of

    d = design_ca[lo:hi]
    res = tmtools.tm_align(_coords64(d), _coords64(target_ca), _seq_of(d), _seq_of(target_ca))
    return max(float(res.tm_norm_chain1), float(res.tm_norm_chain2))


def _pair_tm(ca_a, ca_b):
    """TM between two equal-length Cα slices (design[F] localized vs free) — displacement = 1 − this."""
    import tmtools

    from spa.eval.score import _coords64, _seq_of

    res = tmtools.tm_align(_coords64(ca_a), _coords64(ca_b), _seq_of(ca_a), _seq_of(ca_b))
    return max(float(res.tm_norm_chain1), float(res.tm_norm_chain2))


def _build_profile(L, b, kind, feather_w):
    """Per-residue λ weight [L]: ``step`` = 1 on [0,b)/0 on [b,L); ``feather`` = cosine ramp at b."""
    import torch

    w = torch.ones(L)
    if kind == "step":
        w[b:] = 0.0
        return w
    lo, hi = max(0, b - feather_w), min(L, b + feather_w)
    for i in range(L):
        if i < lo:
            w[i] = 1.0
        elif i >= hi:
            w[i] = 0.0
        else:
            t = (i - lo) / max(1, (hi - lo))
            w[i] = 0.5 * (1.0 + math.cos(math.pi * t))
    return w


# --------------------------------------------------------------------------------------------------
# Frame -> example list  (each example: name, L, boundary b, foreign fold id, S/F slices)
# --------------------------------------------------------------------------------------------------


def _examples(frame, foreign, hosts, fixed_len, pdb_dir, nca, ratios=None):
    """Build the (name, L, b, foreign_id) example list for a frame. b = S/F boundary (S=[0,b), F=[b,L))."""
    ex = []
    if frame == "balanced":
        for g in foreign:
            n = nca[g]
            ex.append({"name": f"bal_{g}", "L": 2 * n, "b": n, "G": g})
    elif frame == "fixed_l":
        for g in foreign:
            ex.append({"name": f"fix_{g}", "L": int(fixed_len), "b": int(fixed_len) // 2, "G": g})
    elif frame == "ratio_sweep":
        # ISOLATION probe (dev discussion 2026-07-02): sweep the S:F ratio at FIXED L on an ARBITRARY
        # (non-domain) mid-fold split. If F-drag/S-steer drops as S shrinks, "small S localizes better"
        # is a RATIO effect independent of any domain boundary (disentangles host_domain's confound).
        L = int(fixed_len)
        for g in foreign:
            for r in (ratios or [0.15, 0.3, 0.5, 0.7]):
                b = min(max(12, round(float(r) * L)), L - 12)   # keep both S and F >= 12 residues
                ex.append({"name": f"r{float(r):g}_{g}", "L": L, "b": int(b), "G": g, "ratio": float(r)})
    elif frame == "host_domain":
        from spa.data.granularity import detect_two_domains

        pairs = [(hosts[i], foreign[i % len(foreign)]) for i in range(len(hosts))]
        for h, g in pairs:
            if h == g:
                print(f"[loc] host_domain: host {h} == foreign {g}; skip (need distinct)."); continue
            info = detect_two_domains(_pdb(pdb_dir, h))
            if info["boundary"] is None:
                print(f"[loc] host_domain: {h} has no clean 2-domain split (score {info['score']}); skip.")
                continue
            ex.append({"name": f"host_{h}_x_{g}", "L": nca[h], "b": int(info["boundary"]), "G": g,
                       "host": h, "dom_score": info["score"]})
    else:
        raise ValueError(f"unknown frame {frame!r}")
    return ex


# --------------------------------------------------------------------------------------------------
# Precompute ESM3 prompts (one residency) for every foreign fold used.
# --------------------------------------------------------------------------------------------------


def _precompute(ids, pdb_dir, strip, device, out_dir):
    import torch

    from spa.prompt.esm3_prompt import esm3_prompt, load_esm3

    cdir = out_dir / "prompts"; cdir.mkdir(parents=True, exist_ok=True)
    caches = {}
    dev = torch.device(device) if isinstance(device, str) else device
    model = load_esm3(dev)
    try:
        for pid in ids:
            p = esm3_prompt(_pdb(pdb_dir, pid), model, strip_bos_eos=strip, use_sequence=False)
            p = p.detach().float().cpu()
            torch.save(p, cdir / f"{pid}.pt")
            caches[pid] = str(cdir / f"{pid}.pt")
            print(f"[loc]   ESM3 {pid}: [{p.shape[0]},{p.shape[1]}]")
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return caches


# --------------------------------------------------------------------------------------------------
# Run one example (all conditions) on a freshly-built engine
# --------------------------------------------------------------------------------------------------


def _run_example(ex, prompt_cache_pt, ckpt, lambdas, K, seed, timesteps, feather, out_dir, device, nca_G):
    import torch

    from spa.eval.generate import _run_once, _seed_all, build_eval_engine, load_adapter, write_pdb
    from spa.eval.score import _ca_array
    from spa.train.harness import frozen_rfd3_net
    from spa.utils.device import resolve_device
    from omegaconf import OmegaConf

    L, b, G = ex["L"], ex["b"], ex["G"]
    dev = resolve_device(device)
    # minimal composed cfg for the engine (variant C, identity projector).
    cfg = OmegaConf.create({
        "paths": {"rfd3_ckpt": _rfd3_ckpt()},
        "hardware": {"device": device},
        "model": {"c_query": 768, "c_kv": 1536, "c_model": 768, "n_head": 8, "shared_kv": True,
                  "zero_init_output": True, "lambda_init": 1.0, "input_rmsnorm": True},
        "variant": {"name": "C", "projector": "identity", "resampler_tokens": None,
                    "strip_bos_eos": True, "use_clss": False},
        "eval": {"num_designs": K, "length": L, "specification": None, "num_timesteps": timesteps,
                 "seed": seed, "ckpt": ckpt, "out_dir": str(out_dir)},
    })
    engine = build_eval_engine(cfg)
    net = frozen_rfd3_net(engine)
    adapter = load_adapter(net, cfg, dev)
    adapter.eval()
    adtype = next(adapter.parameters()).dtype

    p = torch.load(prompt_cache_pt, weights_only=True).float().to(dev)   # [|G|, 1536]
    prompt = p[None].expand(K, -1, -1).to(device=dev, dtype=adtype).contiguous()
    target_ca = nca_G  # Cα of the foreign fold G (scoring reference)

    step_prof = _build_profile(L, b, "step", 0)
    feath_prof = _build_profile(L, b, "feather", max(6, L // 12))

    # (condition_name, mode, profile, lambda)
    conds = [("free", "off", None, 0.0)]
    for lam in lambdas:
        conds.append((f"localized_l{lam:g}", "spa", step_prof, lam))
        if feather:
            conds.append((f"feather_l{lam:g}", "spa", feath_prof, lam))
        conds.append((f"global_l{lam:g}", "spa", None, lam))

    edir = out_dir / ex["name"]; edir.mkdir(parents=True, exist_ok=True)
    per_design = []          # rows: condition, idx, tm_S, tm_F
    free_F_ca = {}           # idx -> Cα of free design's F region (for displacement pairing)
    for (cname, mode, prof, lam) in conds:
        if mode == "off":
            adapter.clear_prompt()
        else:
            adapter.set_prompt(prompt)
            adapter.set_scale(float(lam))
            adapter.set_profile(None if prof is None else prof.to(dev))
        _seed_all(seed)
        with torch.no_grad():
            outs = _run_once(engine)
        for idx, o in enumerate(outs):
            aa = o.atom_array
            write_pdb(aa, edir / f"{cname}_{idx}.pdb")
            dca = _ca_array(aa)
            tmS = _slice_tm(dca, 0, b, target_ca)
            tmF = _slice_tm(dca, b, L, target_ca)
            disp = None
            if cname == "free":
                free_F_ca[idx] = dca[b:L]
            elif idx in free_F_ca:
                disp = 1.0 - _pair_tm(dca[b:L], free_F_ca[idx])
            per_design.append({"condition": cname, "idx": idx, "tm_S": tmS, "tm_F": tmF, "F_disp": disp})
        print(f"[loc]   {ex['name']:<22} {cname:<14} "
              f"S->G {sum(r['tm_S'] for r in per_design if r['condition']==cname)/K:.3f}  "
              f"F->G {sum(r['tm_F'] for r in per_design if r['condition']==cname)/K:.3f}", flush=True)

    del engine, adapter, net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return per_design


_RFD3 = None
def _rfd3_ckpt():
    global _RFD3
    if _RFD3 is None:
        _RFD3 = str(Path.home() / "projects/spa/models/rfdiffusion3/rfd3_latest.ckpt")
    return _RFD3


# --------------------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------------------


def _mean(rows, key, cond=None):
    vals = [r[key] for r in rows if r[key] is not None and (cond is None or r["condition"] == cond)]
    return (sum(vals) / len(vals)) if vals else None


def _summarize(ex, rows, lambdas):
    """Per example: free anchors + per-λ S-steer / F-drag / F-disp / loc-index."""
    free_S = _mean(rows, "tm_S", "free"); free_F = _mean(rows, "tm_F", "free")
    out = {"name": ex["name"], "L": ex["L"], "b": ex["b"], "G": ex["G"], "ratio": ex.get("ratio"),
           "free_S->G": free_S, "free_F->G": free_F, "lambdas": {}}
    for lam in lambdas:
        loc, glob = f"localized_l{lam:g}", f"global_l{lam:g}"
        locS, locF = _mean(rows, "tm_S", loc), _mean(rows, "tm_F", loc)
        gloS, gloF = _mean(rows, "tm_S", glob), _mean(rows, "tm_F", glob)
        disp = _mean(rows, "F_disp", loc)
        s_steer = (locS - free_S) if (locS is not None and free_S is not None) else None
        f_drag = (locF - free_F) if (locF is not None and free_F is not None) else None
        f_ceil = (gloF - free_F) if (gloF is not None and free_F is not None) else None
        loc_idx = (1.0 - f_drag / f_ceil) if (f_drag is not None and f_ceil not in (None, 0)) else None
        out["lambdas"][f"{lam:g}"] = {
            "loc_S->G": locS, "loc_F->G": locF, "glob_S->G": gloS, "glob_F->G": gloF,
            "S_steer": s_steer, "F_drag": f_drag, "F_ceiling": f_ceil, "F_disp": disp,
            "loc_index": loc_idx,
        }
    return out


def _print_frame(frame, summ, lambdas):
    f = lambda x: "n/a" if x is None else f"{x:+.3f}"
    g = lambda x: "n/a" if x is None else f"{x:.3f}"
    print(f"\n{'='*100}\nFRAME: {frame}  — S-steer (want +) | F-drag (want ~0) | loc-index (want ~1) | F-disp\n{'='*100}")
    for lam in lambdas:
        print(f"\n  λ={lam:g}")
        hdr = f"  {'example':<24}{'freeS→G':>9}{'freeF→G':>9}{'S-steer':>9}{'F-drag':>9}{'F-ceil':>9}{'loc-idx':>9}{'F-disp':>9}"
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        agg = {"S_steer": [], "F_drag": [], "F_ceiling": [], "loc_index": [], "F_disp": []}
        for s in summ:
            d = s["lambdas"][f"{lam:g}"]
            print(f"  {s['name']:<24}{g(s['free_S->G']):>9}{g(s['free_F->G']):>9}"
                  f"{f(d['S_steer']):>9}{f(d['F_drag']):>9}{f(d['F_ceiling']):>9}"
                  f"{g(d['loc_index']):>9}{g(d['F_disp']):>9}")
            for k in agg:
                if d[k] is not None:
                    agg[k].append(d[k])
        m = {k: (sum(v) / len(v) if v else None) for k, v in agg.items()}
        print("  " + "-" * (len(hdr) - 2))
        print(f"  {'MEAN':<24}{'':>9}{'':>9}{f(m['S_steer']):>9}{f(m['F_drag']):>9}"
              f"{f(m['F_ceiling']):>9}{g(m['loc_index']):>9}{g(m['F_disp']):>9}")


def _print_ratio_trend(summ, lambdas):
    """For the ratio_sweep frame: aggregate by S:F ratio to ISOLATE 'small S localizes better' from any
    domain-boundary effect (these splits are arbitrary). drag/steer ↓ and loc-idx ↑ as S/L ↓ ⇒ the
    asymmetric-S mechanism is real and boundary-independent."""
    f = lambda x: "n/a" if x is None else f"{x:+.3f}"
    g = lambda x: "n/a" if x is None else f"{x:.3f}"
    ratios = sorted({s["ratio"] for s in summ if s.get("ratio") is not None})
    print(f"\n{'='*80}\nRATIO-ISOLATION TREND — does localization improve as S shrinks? (arbitrary split)\n{'='*80}")
    for lam in lambdas:
        print(f"\n  λ={lam:g}   (want: drag/steer ↓ and loc-idx ↑ as S/L ↓)")
        hdr = f"  {'S/L':>6}{'n':>4}{'S-steer':>9}{'F-drag':>9}{'drag/steer':>11}{'loc-idx':>9}"
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for r in ratios:
            ds = [s["lambdas"][f"{lam:g}"] for s in summ if s.get("ratio") == r]
            ss = [d["S_steer"] for d in ds if d["S_steer"] is not None]
            fd = [d["F_drag"] for d in ds if d["F_drag"] is not None]
            li = [d["loc_index"] for d in ds if d["loc_index"] is not None]
            ms = (sum(ss) / len(ss)) if ss else None
            mf = (sum(fd) / len(fd)) if fd else None
            ratio_ds = (mf / ms) if (ms and mf is not None and ms != 0) else None
            ml = (sum(li) / len(li)) if li else None
            print(f"  {r:>6.2f}{len(ds):>4}{f(ms):>9}{f(mf):>9}{g(ratio_ds):>11}{g(ml):>9}")


# --------------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="multigranularity SPA checkpoint (variant C)")
    ap.add_argument("--foreign", required=True, help="comma ids used as steering targets G (≤128 for balanced)")
    ap.add_argument("--hosts", default="", help="comma ids for the host_domain frame (2-domain structures)")
    ap.add_argument("--frames", default="balanced,fixed_l,host_domain")
    ap.add_argument("--fixed-len", type=int, default=200)
    ap.add_argument("--ratios", default="0.15,0.3,0.5,0.7", help="S:F ratios for the ratio_sweep frame")
    ap.add_argument("--lambdas", default="1,2")
    ap.add_argument("--num-designs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-timesteps", type=int, default=None)
    ap.add_argument("--feather", action="store_true")
    ap.add_argument("--pdb-dir", default=DEFAULT_PDB_DIR)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default="outputs/eval/localization")
    args = ap.parse_args()

    foreign = [s.strip() for s in args.foreign.split(",") if s.strip()]
    hosts = [s.strip() for s in args.hosts.split(",") if s.strip()]
    frames = [s.strip() for s in args.frames.split(",") if s.strip()]
    ratios = [float(x) for x in args.ratios.split(",") if x.strip()]
    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    out_dir = Path(args.out_dir).expanduser().resolve(); out_dir.mkdir(parents=True, exist_ok=True)

    # Cα counts for every structure referenced (foreign always; hosts if host_domain).
    need = set(foreign) | (set(hosts) if "host_domain" in frames else set())
    nca = {pid: len(_ca(_pdb(args.pdb_dir, pid))) for pid in need}
    nca_G = {g: _ca(_pdb(args.pdb_dir, g)) for g in foreign}   # scoring-reference Cα for each foreign fold
    print("[loc] Cα counts:", {k: nca[k] for k in sorted(nca)})

    caches = _precompute(foreign, args.pdb_dir, True, args.device, out_dir)

    results = {"config": vars(args), "frames": {}}
    for frame in frames:
        exs = _examples(frame, foreign, hosts, args.fixed_len, args.pdb_dir, nca, ratios)
        exs = [e for e in exs if e["L"] <= 320]   # A5000 sanity cap (no OF3 here, so a bit past 256 is ok)
        print(f"\n[loc] frame={frame}: {len(exs)} example(s): "
              + ", ".join(f"{e['name']}(L={e['L']},b={e['b']})" for e in exs))
        summ = []
        for ex in exs:
            rows = _run_example(ex, caches[ex["G"]], args.ckpt, lambdas, args.num_designs,
                                args.seed, args.num_timesteps, args.feather, out_dir, args.device, nca_G[ex["G"]])
            summ.append(_summarize(ex, rows, lambdas))
        _print_frame(frame, summ, lambdas)
        if frame == "ratio_sweep":
            _print_ratio_trend(summ, lambdas)
        results["frames"][frame] = summ

    (out_dir / "localization_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[loc] wrote {out_dir / 'localization_results.json'}")
    print("[read] localization holds where S-steer is strongly + AND F-drag ≈ 0 / loc-index ≈ 1.")


if __name__ == "__main__":
    main()
