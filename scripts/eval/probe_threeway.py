"""THROWAWAY PROBE — three-way region-specific steering (states A/B/C), adherence-only.

Tests whether the new per-residue λ profile (``SPAAdapter.set_profile``) lets SPA steer a
*subregion* of the design toward the prompt fold (state B) while leaving the rest to RFD3's prior
(state C) — the original "region-specific topological steering" vision (dev: submitted-abstract
discussion). This is a decisive, fast, A5000-friendly check BEFORE committing to a big-n axis:
RFD3 generate ± SPA + per-region TM-to-prompt, NO ProteinMPNN/OF3 (the §1/§3 adherence-only path).

Four conditions on one prompt (default A0A522W419, len 150, N×1536), K designs each, paired noise:
  - baseline : clear_prompt (vanilla RFD3)               → the RFD3-prior anchor
  - full     : SPA λ, profile=None (uniform)             → reproduces §2-style whole-fold steering
  - threeway : SPA λ, profile = 1 on [0,B) / 0 on [B,L)  → state B steered, state C silent  ← the question
  - feather  : SPA λ, profile = cosine ramp 1→0 over B   → does tapering run clean (no junction artifact)

Success = the steering LOCALIZES: threeway's B-region TM rises toward `full` while its C-region TM
stays near `baseline`. Per-region TM = tm_align over the residue slice (rotation-invariant).

Run (A5000):
    conda run -n spa-dev python scripts/eval/probe_threeway.py \
        variant=C_n_by_1536 hardware=local_a5000 \
        eval.ckpt=checkpoints/spa_C_last.pt \
        eval.prompt_pdb=/home/user1/projects/spa/training_data/proteina-atomistica_data_vrelease/atomistica_data_release/pdb/AF-A0A522W419-F1-model_v4_esmfold_v1.pdb \
        eval.length=150 eval.num_designs=4 eval.lambda_scale=2.0 \
        eval.out_dir=./outputs/eval/probe_threeway
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import hydra
from omegaconf import DictConfig


def _build_profiles(length: int, boundary: int, feather_w: int):
    """Return ``{name: tensor[L] | None}`` for the four conditions (None ⇒ uniform scalar λ)."""
    import torch

    threeway = torch.ones(length)
    threeway[boundary:] = 0.0                                  # state B = [0,B), state C = [B,L)

    feather = torch.ones(length)                               # 1 → 0 cosine ramp centered at `boundary`
    lo, hi = boundary - feather_w, boundary + feather_w
    for i in range(length):
        if i < lo:
            feather[i] = 1.0
        elif i >= hi:
            feather[i] = 0.0
        else:
            t = (i - lo) / (2 * feather_w)                     # 0..1 across the ramp
            feather[i] = 0.5 * (1.0 + math.cos(math.pi * t))   # 1 at lo → 0 at hi
    return {"full": None, "threeway": threeway, "feather": feather}


def _region_tm(design_aa, prompt_ca, lo: int, hi: int) -> float:
    """Prompt-normalized Cα TM-score over the residue slice ``[lo:hi]`` (rotation-invariant)."""
    import tmtools

    from spa.eval.score import _as_struct, _ca_array, _coords64, _seq_of

    d = _ca_array(_as_struct(design_aa))[lo:hi]
    p = prompt_ca[lo:hi]
    res = tmtools.tm_align(_coords64(d), _coords64(p), _seq_of(d), _seq_of(p))
    return float(res.tm_norm_chain2)                           # chain2 = prompt


@hydra.main(version_base=None, config_path="../../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    import torch

    from spa.eval.generate import (
        _run_once,
        _seed_all,
        build_eval_engine,
        load_adapter,
        resolve_prompt,
        write_pdb,
    )
    from spa.eval.score import _as_struct, _ca_array, tm_score
    from spa.train.harness import frozen_rfd3_net
    from spa.utils.device import resolve_device

    ev = cfg.eval
    device = resolve_device(cfg.hardware.device)
    L = int(ev.length)
    K = int(ev.num_designs)
    seed = int(ev.get("seed", 0))
    lam = float(ev.lambda_scale if isinstance(ev.lambda_scale, (int, float)) else list(ev.lambda_scale)[0])
    boundary = int(ev.get("probe_boundary", L // 2))
    feather_w = int(ev.get("probe_feather_w", max(8, L // 10)))
    out_dir = Path(str(ev.out_dir)).expanduser()
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[probe] L={L} K={K} λ={lam} boundary={boundary} feather_w={feather_w}")
    print(f"[probe] state B = residues [0,{boundary}) steered; state C = [{boundary},{L}) silent")

    # Engine + adapter (loads the trained N×1536 ckpt) + the prompt, batched to [K,N,c_kv].
    engine = build_eval_engine(cfg)
    net = frozen_rfd3_net(engine)
    adapter = load_adapter(net, cfg, device)
    adapter.eval()
    adapter_dtype = next(adapter.parameters()).dtype
    p = resolve_prompt(cfg, device)                            # [N, c_kv]
    if p.shape[0] != L:
        raise ValueError(f"prompt length N={p.shape[0]} != eval.length L={L} (use the len-{L} source).")
    prompt_batched = p[None].expand(K, -1, -1).to(device=device, dtype=adapter_dtype).contiguous()
    prompt_ca = _ca_array(_as_struct(str(ev.prompt_pdb)))      # scoring reference (the source fold)

    profiles = _build_profiles(L, boundary, feather_w)
    # combo: query-gate C off (profile) AND key-mask the C-region PROMPT rows, so the B-region is
    # steered toward B's LOCAL structure only (not the whole fold). Tests whether localization is
    # recoverable when B no longer "sees" the full prompt. B prompt rows [0,boundary) stay unmasked
    # ⇒ every query keeps ≥boundary keys ⇒ no all-masked (NaN) softmax row.
    key_mask_BC = torch.zeros(K, L, dtype=torch.bool, device=device)
    key_mask_BC[:, boundary:] = True
    conditions = [("baseline", "clear", None, None), ("full", "spa", profiles["full"], None),
                  ("threeway", "spa", profiles["threeway"], None),
                  ("feather", "spa", profiles["feather"], None),
                  ("combo", "spa", profiles["threeway"], key_mask_BC)]

    rows = []
    per_design = []
    for name, mode, profile, key_mask in conditions:
        if mode == "clear":
            adapter.clear_prompt()
        else:
            adapter.set_prompt(prompt_batched, key_padding_mask=key_mask)
            adapter.set_scale(lam)
            adapter.set_profile(profile if profile is None else profile.to(device))

        _seed_all(seed)                                       # paired noise across conditions
        with torch.no_grad():
            outs = _run_once(engine)

        ov, br, cr = [], [], []
        for idx, rfd3_out in enumerate(outs):
            aa = rfd3_out.atom_array
            write_pdb(aa, out_dir / f"{name}_{idx}.pdb")
            _, tm_p = tm_score(aa, prompt_ca)                 # overall prompt-normalized TM
            b = _region_tm(aa, prompt_ca, 0, boundary)        # state-B region
            c = _region_tm(aa, prompt_ca, boundary, L)        # state-C region
            ov.append(tm_p); br.append(b); cr.append(c)
            per_design.append({"condition": name, "idx": idx, "tm_overall": tm_p, "tm_B": b, "tm_C": c})

        mean = lambda xs: sum(xs) / len(xs)
        rows.append((name, mean(ov), mean(br), mean(cr)))
        print(f"[probe] {name:<9} overall={mean(ov):.3f}  B[0:{boundary})={mean(br):.3f}  C[{boundary}:{L})={mean(cr):.3f}")

    # Summary table + the decisive read.
    print("\n=== three-way probe (mean prompt-TM over K designs) ===")
    print(f"{'condition':<10}{'overall':>9}{'B-region':>10}{'C-region':>10}")
    for name, o, b, c in rows:
        print(f"{name:<10}{o:>9.3f}{b:>10.3f}{c:>10.3f}")
    base = {r[0]: r for r in rows}
    if {"baseline", "full", "threeway"} <= set(base):
        tb, tf = base["threeway"], base["full"]
        bl = base["baseline"]
        print("\n[read] localization check (threeway):")
        print(f"  B-region: threeway {tb[2]:.3f} vs full {tf[2]:.3f} vs baseline {bl[2]:.3f}  "
              f"(want ≈full, >>baseline)")
        print(f"  C-region: threeway {tb[3]:.3f} vs baseline {bl[3]:.3f} vs full {tf[3]:.3f}  "
              f"(want ≈baseline, <<full)")

    (out_dir / "probe_results.json").write_text(json.dumps(
        {"config": {"L": L, "K": K, "lambda": lam, "boundary": boundary, "feather_w": feather_w,
                    "ckpt": str(ev.get("ckpt")), "prompt_pdb": str(ev.prompt_pdb)},
         "means": [{"condition": n, "tm_overall": o, "tm_B": b, "tm_C": c} for (n, o, b, c) in rows],
         "per_design": per_design}, indent=2))
    print(f"\n[probe] wrote PDBs + probe_results.json to {out_dir}")


if __name__ == "__main__":
    main()
