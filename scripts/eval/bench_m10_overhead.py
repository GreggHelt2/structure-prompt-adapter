"""M10: SPA inference overhead (dev `plan/73`) — cold vs. cached ESM3, the wrapper's architectural
tax, both sampler settings, and model residency across separate process launches.

WHY. The review calls inference overhead "a conspicuous absence for an adapter paper"
(`draft_review_opus5_2026-07-24a.md`:748). This measures it, and specifically distinguishes costs a
single opaque "seconds/design" number would blur together (dev `plan/73` SS1/SS3, all traced to
source, not assumed):

- **ESM3 prompt encode vs. ESM3 model LOAD.** The full 455,473-structure ESM3 cache-gen job measures
  the actual encode at ~0.08 s/structure (`EXPERIMENTS.md`:122-123). The ~90 s figure sometimes cited
  is model-*load* time (weights -> GPU), paid once per fresh process, not per structure. This script
  separately times both (`--phase load-breakdown`).
- **The wrapper's tax when SPA does nothing** (`conditions=[baseline]`, no prompt) vs. when it steers
  (`conditions=[spa]`).
- **Cold** (`eval.prompt_pdb`, ESM3 loaded fresh every process) vs. **cached** (`eval.prompt_cache`,
  ESM3 skipped entirely) prompt resolution.
- **Two amortization levels**: designs sharing one process (`--phase k-sweep`) vs. separate process
  launches for the SAME prompt (`--phase m-sweep`).
- **Four residency strategies for DIFFERENT prompts** (`--phase residency`), none of which this
  project's own sweep drivers already cover (verified against `scripts/eval/ladder_sweep.py`, which
  keeps the engine warm only for swapping SPA *checkpoints* on one fixed prompt, and
  `scripts/cloud/run_threeway_sweep.sh`, whose bash `for`-loop launches a fresh subprocess per
  cell): (a) fully separate subprocess launches (each pays a fresh RFD3 load; ESM3 held constant via
  a pre-staged cache); (b) one warm process, RFD3+SPA stay resident, looping the SAME pre-staged
  caches; (c) **ESM3 also stays resident** alongside RFD3+SPA, the opposite of `resolve_prompt()`'s
  own default (it deliberately frees ESM3 "so it does not co-reside with RFD3 during the sampler",
  `generate.py`:223-225) — added 2026-08-18 to test whether that caution, plausibly tuned for the
  24GB local A5000, still matters on an 80GB H100; (d) **precompute all M prompts with ESM3 first,
  free it completely, then build RFD3 fresh** (the `precompute_prompts.py` pattern, but timed
  end-to-end including the encode step, not assumed to happen outside the clock) — added the same
  day, the opposite extreme from (c).
- **Both sampler settings** (checkpoint-inherited 100-step/gamma0=0.8 vs RFD3's published
  200-step/gamma0=0.6), to see whether the overhead *fraction* changes with step count.

WHAT THIS DOES NOT MEASURE. Adherence or designability at either sampler setting — this is wall-clock
and peak VRAM only (dev `plan/73` SS2a). Not a substitute for `results/22` SS9's costed decision about
regenerating the paper's evidence base.

USAGE (local A5000, needs a real trained SPA checkpoint and at least one real prompt structure with
a matching precomputed [N,1536] .pt cache -- see scripts/cloud/run_m10_bench.sh for staging real
prompts from gs://genomancer-spa-cache/eval/b1_full/prep/):

    conda run -n spa-dev python scripts/eval/bench_m10_overhead.py \\
        --ckpt /path/spa_C_final.pt \\
        --prompt-pdb /path/prompt0.pdb --prompt-cache /path/prompt0.pt \\
        --residency-prompts /path/p1.pdb:/path/p1.pt,/path/p2.pdb:/path/p2.pt,... \\
        --phase all --json /path/m10_results.json

    # cheap pipeline-validation pass before spending on the full sweep (dev plan/73 SS6 "$2 smoke"
    # convention, results/14 SS4):
    conda run -n spa-dev python scripts/eval/bench_m10_overhead.py ... --smoke
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]          # structure-prompt-adapter


# --------------------------------------------------------------------------------------------------
# GPU-agnostic VRAM poller (bench_of3_length.py's pattern, generalized: no hardcoded A5000 UUID, so
# this runs unmodified on a single-GPU H100 cloud instance or the local dual-GPU A5000 box).
# --------------------------------------------------------------------------------------------------
class _GpuPoller(threading.Thread):
    """Sample used VRAM (nvidia-smi) while a subprocess runs. A child process's torch allocator is
    invisible to this process's own torch.cuda.max_memory_allocated(), so external polling is the
    only way to see it for the subprocess-based arms."""

    def __init__(self, interval=0.5, gpu=None):
        super().__init__(daemon=True)
        self.gpu = gpu                                            # None -> poll the default/only GPU
        self.interval, self.peak_mib, self._halt = interval, 0, False

    def run(self):
        cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        if self.gpu:
            cmd = ["nvidia-smi", "-i", self.gpu] + cmd[1:]
        while not self._halt:
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.split()
                self.peak_mib = max([self.peak_mib] + [int(v) for v in out if v.isdigit()])
            except Exception:                                     # noqa: BLE001
                pass
            time.sleep(self.interval)

    def stop(self):
        self._halt = True
        self.join(timeout=5)


# --------------------------------------------------------------------------------------------------
# One subprocess launch of the UNMODIFIED generate.py CLI. sys.executable, not a hardcoded conda
# env name, so this works both locally (invoked via `conda run -n spa-dev python bench_m10...`) and
# in the cloud combined image (SPA/RFD3 on system python, no conda prefix -- verified against
# scripts/cloud/run_smoke.sh, which never prefixes `conda run -n spa-dev`).
# --------------------------------------------------------------------------------------------------
def _run_generate(overrides: list[str], *, gpu=None, timeout=900, script="generate.py") -> dict:
    cmd = [sys.executable, str(_REPO / "scripts" / "eval" / script)] + overrides
    poller = _GpuPoller(gpu=gpu)
    poller.start()
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        ok, rc, out, err = proc.returncode == 0, proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        ok, rc, out, err = False, -1, e.stdout or "", f"TIMEOUT after {timeout}s"
    dt = time.time() - t0
    poller.stop()
    sampler_line = next((l for l in out.splitlines() if l.startswith("[sampler] EFFECTIVE:")), None)
    return {
        "seconds": round(dt, 2), "peak_vram_mib": poller.peak_mib, "ok": ok, "returncode": rc,
        "sampler_effective": sampler_line,
        "stderr_tail": ("" if ok else err[-2000:]),
        "cmd": " ".join(overrides),
    }


def _base_overrides(*, variant, out_dir, length=None, num_designs=8, ckpt=None,
                     num_timesteps=None, gamma_0=None, seed=0, conditions=None, lam=1.0,
                     hardware="local_a5000", rfd3_ckpt=None):
    # NB: no shell-quote characters around the list override below. This project's own docstrings
    # show 'eval.conditions=[...]' because that is what a BASH PROMPT needs (the shell would
    # otherwise glob-expand the brackets); subprocess.run() with a list argument bypasses the shell
    # entirely, so each element here is passed literally to argv, and embedding quote characters
    # would hand Hydra the literal string "'eval.conditions=[...]'", quotes and all.
    ov = [f"variant={variant}", f"hardware={hardware}", f"eval.out_dir={out_dir}",
          f"eval.num_designs={num_designs}", f"eval.seed={seed}"]
    if length is not None:
        ov.append(f"eval.length={length}")
    if ckpt:
        ov.append(f"eval.ckpt={ckpt}")
    if rfd3_ckpt:
        # configs/paths/default.yaml's rfd3_ckpt defaults to $SPA_PROJECT_ROOT/models/..., which
        # only exists on the local box (root CLAUDE.md Model Weights). The cloud combined image
        # stages weights under /workspace/weights/ instead (run_8siu_hardsoft.sh's own pattern),
        # so a cloud invocation MUST override paths.rfd3_ckpt explicitly or generation fails
        # looking for a checkpoint that was never staged there.
        ov.append(f"paths.rfd3_ckpt={rfd3_ckpt}")
    if num_timesteps is not None:
        ov.append(f"eval.num_timesteps={num_timesteps}")
    if gamma_0 is not None:
        ov.append(f"eval.gamma_0={gamma_0}")
    if conditions:
        ov.append("eval.conditions=[" + ",".join(conditions) + "]")
    if "spa" in (conditions or []):
        ov.append(f"eval.lambda_scale={lam}")
    return ov


# --------------------------------------------------------------------------------------------------
# SS2/SS2a: arms table (baseline-inert / cold / warm) x both sampler settings.
# --------------------------------------------------------------------------------------------------
_SAMPLERS = {"100": dict(num_timesteps=100, gamma_0=0.8), "200": dict(num_timesteps=200, gamma_0=0.6)}


def phase_arms(a) -> list[dict]:
    rows = []
    lengths = [int(x) for x in a.lengths.split(",")]
    for sampler_name, sampler_kw in _SAMPLERS.items():
        for length in lengths:
            out = f"{a.out_dir}/arms/s{sampler_name}_L{length}"
            # host-only: SPA never attached at all (generate_rfd3_only.py never imports spa.model),
            # the TRUE zero-overhead floor -- distinct from "inert" below, which has the wrapper
            # attached but silent. Added 2026-08-18: this row was in plan/73 SS2 from the first
            # draft of the spec and should have been built from the start; it was not, and this is
            # the fix, not a new idea.
            r = _run_generate(_base_overrides(
                variant=a.variant, hardware=a.hardware, rfd3_ckpt=a.rfd3_ckpt, out_dir=f"{out}/host_only",
                num_designs=a.k, length=length, **sampler_kw), gpu=a.gpu, script="generate_rfd3_only.py")
            rows.append({"arm": "host_only", "sampler": sampler_name, "length": length, **r})
            print(f"[m10:arms] host_only s={sampler_name} L={length}  {r['seconds']:>7.1f}s  "
                  f"peak {r['peak_vram_mib']:>6} MiB  ok={r['ok']}")

            # inert: wrapper attached (adapter present via --ckpt if given, but no prompt -> baseline
            # condition == clear_prompt, an identity gate; isolates the architectural tax alone)
            r = _run_generate(_base_overrides(
                variant=a.variant, hardware=a.hardware, rfd3_ckpt=a.rfd3_ckpt, out_dir=f"{out}/inert", num_designs=a.k, ckpt=a.ckpt,
                length=length, conditions=["baseline"], **sampler_kw), gpu=a.gpu)
            rows.append({"arm": "inert", "sampler": sampler_name, "length": length, **r})
            print(f"[m10:arms] inert   s={sampler_name} L={length}  {r['seconds']:>7.1f}s  "
                  f"peak {r['peak_vram_mib']:>6} MiB  ok={r['ok']}")

            if a.ckpt and a.prompt_pdb:
                r = _run_generate(_base_overrides(
                    variant=a.variant, hardware=a.hardware, rfd3_ckpt=a.rfd3_ckpt, out_dir=f"{out}/cold", num_designs=a.k, ckpt=a.ckpt,
                    length=length, conditions=["spa"], lam=a.lam, **sampler_kw)
                    + [f"eval.prompt_pdb={a.prompt_pdb}"], gpu=a.gpu)
                rows.append({"arm": "cold", "sampler": sampler_name, "length": length, **r})
                print(f"[m10:arms] cold    s={sampler_name} L={length}  {r['seconds']:>7.1f}s  "
                      f"peak {r['peak_vram_mib']:>6} MiB  ok={r['ok']}")

            if a.ckpt and a.prompt_cache:
                r = _run_generate(_base_overrides(
                    variant=a.variant, hardware=a.hardware, rfd3_ckpt=a.rfd3_ckpt, out_dir=f"{out}/warm", num_designs=a.k, ckpt=a.ckpt,
                    length=length, conditions=["spa"], lam=a.lam, **sampler_kw)
                    + [f"eval.prompt_cache={a.prompt_cache}"], gpu=a.gpu)
                rows.append({"arm": "warm", "sampler": sampler_name, "length": length, **r})
                print(f"[m10:arms] warm    s={sampler_name} L={length}  {r['seconds']:>7.1f}s  "
                      f"peak {r['peak_vram_mib']:>6} MiB  ok={r['ok']}")
    return rows


# --------------------------------------------------------------------------------------------------
# SS3 item 1: within-process K-sweep (cold path, 100-step only, one length).
# --------------------------------------------------------------------------------------------------
def phase_ksweep(a) -> list[dict]:
    if not (a.ckpt and a.prompt_pdb):
        print("[m10:k-sweep] SKIPPED: needs --ckpt and --prompt-pdb"); return []
    length = int(a.lengths.split(",")[0])
    rows = []
    for k in [1, 8, 16, 32]:
        r = _run_generate(_base_overrides(
            variant=a.variant, hardware=a.hardware, rfd3_ckpt=a.rfd3_ckpt, out_dir=f"{a.out_dir}/ksweep/k{k}", num_designs=k, ckpt=a.ckpt,
            length=length, conditions=["spa"], lam=a.lam, **_SAMPLERS["100"])
            + [f"eval.prompt_pdb={a.prompt_pdb}"], gpu=a.gpu)
        rows.append({"k": k, "length": length, **r})
        print(f"[m10:k-sweep] K={k:>3}  {r['seconds']:>7.1f}s  ({r['seconds']/k:>6.2f}s/design)  "
              f"peak {r['peak_vram_mib']:>6} MiB  ok={r['ok']}")
    return rows


# --------------------------------------------------------------------------------------------------
# SS3 item 2: cross-invocation M-sweep, SAME prompt, cold (prompt_pdb) vs warm (prompt_cache),
# each M separate subprocess launches (100-step only).
# --------------------------------------------------------------------------------------------------
def phase_msweep(a) -> list[dict]:
    if not (a.ckpt and a.prompt_pdb and a.prompt_cache):
        print("[m10:m-sweep] SKIPPED: needs --ckpt, --prompt-pdb, and --prompt-cache"); return []
    length = int(a.lengths.split(",")[0])
    rows = []
    for label, prompt_flag in (("cold", f"eval.prompt_pdb={a.prompt_pdb}"),
                                ("warm", f"eval.prompt_cache={a.prompt_cache}")):
        for m in (1, 5, 20):
            total = 0.0
            peak = 0
            for i in range(m):
                r = _run_generate(_base_overrides(
                    variant=a.variant, hardware=a.hardware, rfd3_ckpt=a.rfd3_ckpt, out_dir=f"{a.out_dir}/msweep/{label}_m{m}/{i}", num_designs=1,
                    ckpt=a.ckpt, length=length, conditions=["spa"], lam=a.lam, **_SAMPLERS["100"])
                    + [prompt_flag], gpu=a.gpu)
                total += r["seconds"]
                peak = max(peak, r["peak_vram_mib"])
                if not r["ok"]:
                    print(f"[m10:m-sweep] {label} m={m} invocation {i} FAILED: {r['stderr_tail'][:300]}")
            rows.append({"regime": label, "m": m, "total_seconds": round(total, 2),
                         "seconds_per_invocation": round(total / m, 2), "peak_vram_mib": peak})
            print(f"[m10:m-sweep] {label:>4} M={m:>3}  total {total:>7.1f}s  "
                  f"({total/m:>6.2f}s/invocation)  peak {peak:>6} MiB")
    return rows


# --------------------------------------------------------------------------------------------------
# SS3 item 3: model residency across DIFFERENT prompts. (a) M separate subprocess launches at
# prompt_cache (isolates the RFD3-load cost, ESM3 held constant since both use cached tensors).
# (b) ONE warm process looping over the same M prompts, engine+adapter built once -- the
# ladder_sweep.py idiom (build_eval_engine / load_adapter / generate(engine=,adapter=)), generalized
# from swapping checkpoints to swapping prompts. This is the one arm needing new (not reused) code.
# --------------------------------------------------------------------------------------------------
def phase_residency(a) -> dict:
    if not (a.ckpt and a.residency_prompts):
        print("[m10:residency] SKIPPED: needs --ckpt and --residency-prompts "
              "(pdb:cache,pdb:cache,...)"); return {}
    pairs = [p.split(":") for p in a.residency_prompts.split(",")]
    length = int(a.lengths.split(",")[0])

    # (a) separate launches, one per prompt, prompt_cache (cold-process, warm-ESM3-since-cached)
    sep_total, sep_peak = 0.0, 0
    for i, (_pdb, cache) in enumerate(pairs):
        r = _run_generate(_base_overrides(
            variant=a.variant, hardware=a.hardware, rfd3_ckpt=a.rfd3_ckpt, out_dir=f"{a.out_dir}/residency/separate/{i}", num_designs=1,
            ckpt=a.ckpt, length=length, conditions=["spa"], lam=a.lam, **_SAMPLERS["100"])
            + [f"eval.prompt_cache={cache}"], gpu=a.gpu)
        sep_total += r["seconds"]
        sep_peak = max(sep_peak, r["peak_vram_mib"])
        if not r["ok"]:
            print(f"[m10:residency] separate invocation {i} FAILED: {r['stderr_tail'][:300]}")
    print(f"[m10:residency] separate  M={len(pairs)}  total {sep_total:>7.1f}s  "
          f"({sep_total/len(pairs):>6.2f}s/prompt)  peak {sep_peak:>6} MiB")

    # (b) one warm process, engine+adapter built ONCE, looping generate(engine=,adapter=) over the
    # SAME M prompts via their precomputed caches (the ladder_sweep.py idiom, generalized).
    warm_result = _run_warm_residency(a, pairs, length)

    # (c) ESM3 ALSO stays resident (never freed), alongside the warm RFD3 engine, starting from RAW
    # pdbs (not pre-staged caches) -- added 2026-08-18 at Gregg's request, to test the design choice
    # resolve_prompt() deliberately does NOT take (it frees ESM3 after every encode "so it does not
    # co-reside with RFD3 during the sampler", generate.py:223-225): does that VRAM caution actually
    # matter on an 80GB H100, or is it a leftover A5000 (24GB) constraint applied everywhere?
    esm3_resident_result = _run_esm3_resident_residency(a, pairs, length)

    # (d) precompute ALL prompts with ESM3 first (one load, M encodes), free ESM3+VRAM completely,
    # THEN build RFD3 fresh and generate -- the precompute_prompts.py pattern, but timed end-to-end
    # (including the encode step) rather than assuming it happened outside the clock.
    precompute_result = _run_precompute_then_free_residency(a, pairs, length)

    return {
        "separate": {"m": len(pairs), "total_seconds": round(sep_total, 2),
                     "seconds_per_prompt": round(sep_total / len(pairs), 2), "peak_vram_mib": sep_peak},
        "warm_process": warm_result,
        "esm3_resident": esm3_resident_result,
        "precompute_then_free": precompute_result,
    }


def _run_warm_residency(a, pairs, length) -> dict:
    """Runs IN this process (no subprocess): build the RFD3 engine + SPA adapter ONCE, then loop
    generate() over each cached prompt, timing only the per-call marginal cost. Mirrors
    src/spa/eval/ladder.py's run_ladder(), which does exactly this for checkpoint-swapping; this is
    the prompt-swapping generalization dev plan/73 SS7 flagged as possibly needing new code."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    overrides = [
        f"variant={a.variant}", f"hardware={a.hardware}", f"eval.ckpt={a.ckpt}",
        f"eval.length={length}", f"eval.num_designs=1", f"eval.conditions=[spa]",
        f"eval.lambda_scale={a.lam}",
        f"eval.num_timesteps={_SAMPLERS['100']['num_timesteps']}",
        f"eval.gamma_0={_SAMPLERS['100']['gamma_0']}",
    ]
    if a.rfd3_ckpt:
        overrides.append(f"paths.rfd3_ckpt={a.rfd3_ckpt}")
    config_dir = str(_REPO / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="eval", overrides=overrides)
    OmegaConf.set_struct(cfg, False)

    from spa.eval.generate import build_eval_engine, load_adapter, generate
    from spa.train.harness import frozen_rfd3_net
    from spa.utils.device import resolve_device

    device = resolve_device(cfg.hardware.device)
    t_engine0 = time.time()
    engine = build_eval_engine(cfg)                       # RFD3 load -- paid ONCE for all M prompts
    t_engine = time.time() - t_engine0
    net = frozen_rfd3_net(engine)
    adapter = load_adapter(net, cfg, device)
    adapter.eval()

    per_prompt = []
    for i, (_pdb, cache) in enumerate(pairs):
        cfg.eval.prompt_cache = cache
        cfg.eval.out_dir = f"{a.out_dir}/residency/warm/{i}"
        t0 = time.time()
        designs = generate(cfg, engine=engine, adapter=adapter)
        dt = time.time() - t0
        per_prompt.append(round(dt, 2))
        print(f"[m10:residency] warm[{i}]  {dt:>6.2f}s  n_designs={len(designs)}")

    import torch
    peak_mib = int(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0
    total = t_engine + sum(per_prompt)
    print(f"[m10:residency] warm-process  engine-build {t_engine:>7.1f}s  "
          f"+ {len(pairs)} prompts avg {sum(per_prompt)/len(per_prompt):>6.2f}s/prompt  "
          f"total {total:>7.1f}s  peak {peak_mib:>6} MiB (torch, in-process)")
    return {"engine_build_seconds": round(t_engine, 2), "per_prompt_seconds": per_prompt,
            "total_seconds": round(total, 2), "peak_vram_mib_torch": peak_mib}


def _compose_residency_cfg(a, length):
    """Shared by the two new residency arms: same overrides `_run_warm_residency` composes."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    overrides = [
        f"variant={a.variant}", f"hardware={a.hardware}", f"eval.ckpt={a.ckpt}",
        f"eval.length={length}", f"eval.num_designs=1", f"eval.conditions=[spa]",
        f"eval.lambda_scale={a.lam}",
        f"eval.num_timesteps={_SAMPLERS['100']['num_timesteps']}",
        f"eval.gamma_0={_SAMPLERS['100']['gamma_0']}",
    ]
    if a.rfd3_ckpt:
        overrides.append(f"paths.rfd3_ckpt={a.rfd3_ckpt}")
    config_dir = str(_REPO / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="eval", overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    return cfg


def _run_esm3_resident_residency(a, pairs, length) -> dict:
    """Arm (c), added 2026-08-18 at Gregg's request: does NOT free ESM3 between (or during) calls --
    the opposite of resolve_prompt()'s own default behaviour (generate.py:223-225, "so it does not
    co-reside with RFD3 during the sampler"). ESM3 + the RFD3 engine + SPA adapter are ALL built
    once and stay resident in VRAM for the whole loop, encoding each of the M RAW prompt pdbs
    on-demand via the SAME already-loaded ESM3 instance. Reports peak VRAM directly, since the
    question this answers is empirical ("is this fine on the 80GB H100?"), not asserted."""
    import time as _time

    import torch

    from spa.eval.generate import build_eval_engine, generate, load_adapter
    from spa.prompt.esm3_prompt import esm3_prompt, load_esm3
    from spa.train.harness import frozen_rfd3_net
    from spa.utils.device import resolve_device

    cfg = _compose_residency_cfg(a, length)
    device = resolve_device(cfg.hardware.device)

    t0 = _time.time()
    engine = build_eval_engine(cfg)
    net = frozen_rfd3_net(engine)
    adapter = load_adapter(net, cfg, device)
    adapter.eval()
    esm3_model = load_esm3(device)          # loaded ONCE; deliberately never freed/deleted below
    t_setup = _time.time() - t0

    tmp_dir = Path(f"{a.out_dir}/residency/esm3_resident_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    encode_times, generate_times = [], []
    for i, (pdb, _cache) in enumerate(pairs):
        t_e0 = _time.time()
        p = esm3_prompt(pdb, esm3_model, strip_bos_eos=True).detach().cpu()
        encode_times.append(round(_time.time() - t_e0, 2))
        cache_path = tmp_dir / f"{i}.pt"
        torch.save(p, cache_path)
        cfg.eval.prompt_cache = str(cache_path)
        cfg.eval.prompt_pdb = None
        cfg.eval.out_dir = f"{a.out_dir}/residency/esm3_resident/{i}"
        t_g0 = _time.time()
        designs = generate(cfg, engine=engine, adapter=adapter)
        generate_times.append(round(_time.time() - t_g0, 2))
        print(f"[m10:residency] esm3-resident[{i}]  encode {encode_times[-1]:>5.2f}s  "
              f"generate {generate_times[-1]:>6.2f}s  n_designs={len(designs)}")

    peak_mib = int(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0
    total = t_setup + sum(encode_times) + sum(generate_times)
    print(f"[m10:residency] esm3-resident  setup(engine+adapter+ESM3) {t_setup:>7.1f}s  "
          f"+ {len(pairs)} prompts avg encode {sum(encode_times)/len(encode_times):>5.2f}s "
          f"generate {sum(generate_times)/len(generate_times):>6.2f}s  "
          f"total {total:>7.1f}s  peak {peak_mib:>6} MiB (ESM3+RFD3+SPA co-resident throughout)")
    return {"setup_seconds": round(t_setup, 2), "encode_seconds": encode_times,
            "generate_seconds": generate_times, "total_seconds": round(total, 2),
            "peak_vram_mib_torch": peak_mib}


def _run_precompute_then_free_residency(a, pairs, length) -> dict:
    """Arm (d), added 2026-08-18 at Gregg's request: the precompute_prompts.py pattern, timed
    end-to-end. Phase A loads ESM3 ONCE, encodes ALL M raw prompt pdbs, then explicitly frees ESM3
    (del + empty_cache) before RFD3 is even built -- so ESM3 and RFD3 never co-reside in VRAM at
    all, the opposite extreme from the esm3_resident arm above. Phase B builds RFD3 fresh and
    generates from the M now-cached tensors, engine warm across all of them (matching the
    warm_process arm's RFD3-residency assumption, so the only new variable vs that arm is whether
    the encode cost was paid inside this same timed measurement)."""
    import time as _time

    import torch

    from spa.eval.generate import build_eval_engine, generate, load_adapter
    from spa.prompt.esm3_prompt import esm3_prompt, load_esm3
    from spa.train.harness import frozen_rfd3_net
    from spa.utils.device import resolve_device

    cfg = _compose_residency_cfg(a, length)
    device = resolve_device(cfg.hardware.device)
    tmp_dir = Path(f"{a.out_dir}/residency/precompute_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase A: ESM3 only, all M prompts, then freed ---
    t_a0 = _time.time()
    esm3_model = load_esm3(device)
    vram_after_esm3_load = (int(torch.cuda.max_memory_allocated() / (1024 * 1024))
                            if torch.cuda.is_available() else 0)
    cache_paths = []
    encode_times = []
    for i, (pdb, _cache) in enumerate(pairs):
        t_e0 = _time.time()
        p = esm3_prompt(pdb, esm3_model, strip_bos_eos=True).detach().cpu()
        encode_times.append(round(_time.time() - t_e0, 2))
        cache_path = tmp_dir / f"{i}.pt"
        torch.save(p, cache_path)
        cache_paths.append(cache_path)
        print(f"[m10:residency] precompute[{i}]  encode {encode_times[-1]:>5.2f}s")
    del esm3_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    phase_a_total = _time.time() - t_a0
    print(f"[m10:residency] precompute Phase A (ESM3, {len(pairs)} prompts, then FREED): "
          f"{phase_a_total:>7.1f}s  peak-during {vram_after_esm3_load:>6} MiB")

    # --- Phase B: RFD3 fresh (built AFTER ESM3 is freed), warm across all M cached prompts ---
    t_b0 = _time.time()
    engine = build_eval_engine(cfg)
    net = frozen_rfd3_net(engine)
    adapter = load_adapter(net, cfg, device)
    adapter.eval()
    t_engine = _time.time() - t_b0

    generate_times = []
    for i, cache_path in enumerate(cache_paths):
        cfg.eval.prompt_cache = str(cache_path)
        cfg.eval.prompt_pdb = None
        cfg.eval.out_dir = f"{a.out_dir}/residency/precompute/{i}"
        t_g0 = _time.time()
        designs = generate(cfg, engine=engine, adapter=adapter)
        generate_times.append(round(_time.time() - t_g0, 2))
        print(f"[m10:residency] precompute generate[{i}]  {generate_times[-1]:>6.2f}s  n_designs={len(designs)}")

    peak_mib_b = int(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0
    phase_b_total = t_engine + sum(generate_times)
    total = phase_a_total + phase_b_total
    print(f"[m10:residency] precompute Phase B (RFD3 fresh + {len(pairs)} prompts warm): "
          f"engine-build {t_engine:>7.1f}s  avg generate {sum(generate_times)/len(generate_times):>6.2f}s  "
          f"total {phase_b_total:>7.1f}s  peak {peak_mib_b:>6} MiB")
    print(f"[m10:residency] precompute COMBINED total {total:>7.1f}s")
    return {"phase_a_esm3_seconds": round(phase_a_total, 2), "phase_a_encode_seconds": encode_times,
            "phase_a_vram_mib_torch": vram_after_esm3_load,
            "phase_b_engine_build_seconds": round(t_engine, 2), "phase_b_generate_seconds": generate_times,
            "phase_b_vram_mib_torch": peak_mib_b, "total_seconds": round(total, 2)}


# --------------------------------------------------------------------------------------------------
# SS1: load-time breakdown. Process start -> torch import -> ESM3 loaded (from_pretrained returns)
# -> first structure encoded. Decomposes the "~90s" docstring claim rather than repeating it (dev
# plan/73 SS1: that figure is unverified against a specific measured job log).
# --------------------------------------------------------------------------------------------------
def phase_load_breakdown(a) -> dict:
    if not a.prompt_pdb:
        print("[m10:load-breakdown] SKIPPED: needs --prompt-pdb"); return {}
    t_start = time.time()
    script = f"""
import time, sys, json
t0 = {t_start!r}
t_launch = time.time()
import torch
t_torch = time.time()
from spa.prompt.esm3_prompt import esm3_prompt, load_esm3
t_import = time.time()
model = load_esm3(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
t_loaded = time.time()
p = esm3_prompt({a.prompt_pdb!r}, model, strip_bos_eos=True)
t_encoded = time.time()
print(json.dumps({{
    "launch_to_torch_import_s": round(t_torch - t_launch, 2),
    "torch_import_to_spa_import_s": round(t_import - t_torch, 2),
    "spa_import_to_esm3_loaded_s": round(t_loaded - t_import, 2),
    "esm3_loaded_to_first_encode_s": round(t_encoded - t_loaded, 2),
    "total_from_launch_s": round(t_encoded - t_launch, 2),
    "prompt_shape": list(p.shape),
}}))
"""
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        print(f"[m10:load-breakdown] FAILED: {proc.stderr[-2000:]}")
        return {"ok": False, "stderr_tail": proc.stderr[-2000:]}
    try:
        breakdown = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as e:                                 # noqa: BLE001
        print(f"[m10:load-breakdown] could not parse output: {e}\n{proc.stdout}")
        return {"ok": False, "raw_stdout": proc.stdout}
    print(f"[m10:load-breakdown] {json.dumps(breakdown, indent=2)}")
    return {"ok": True, **breakdown}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="all",
                    choices=["arms", "k-sweep", "m-sweep", "residency", "load-breakdown", "all"])
    ap.add_argument("--variant", default="C_n_by_1536")
    ap.add_argument("--hardware", default="local_a5000", choices=["local_a5000", "cloud_h100"],
                    help="matches every other cloud script's convention of an explicit hardware= override "
                         "(configs/hardware/*.yaml); both use device=cuda:0 so this does not change "
                         "correctness here, only consistency with the rest of this project's cloud scripts")
    ap.add_argument("--rfd3-ckpt", default=None,
                    help="overrides paths.rfd3_ckpt; REQUIRED in the cloud combined image, where weights "
                         "stage to /workspace/weights/rfd3_latest.ckpt rather than the local default path "
                         "(configs/paths/default.yaml). Omit locally, where the default already resolves.")
    ap.add_argument("--ckpt", default=None, help="trained SPA checkpoint (required for cold/warm/steering arms)")
    ap.add_argument("--prompt-pdb", default=None, help="a structure file, for cold-path + k-sweep + m-sweep + load-breakdown")
    ap.add_argument("--prompt-cache", default=None, help="matching precomputed [N,1536] .pt, for warm-path + m-sweep")
    ap.add_argument("--residency-prompts", default=None,
                    help="comma-separated pdb:cache pairs, >=2 DIFFERENT prompts, for --phase residency")
    ap.add_argument("--lengths", default="100,150,250", help="comma-separated; first value used for k-sweep/m-sweep/residency")
    ap.add_argument("--k", type=int, default=8, help="K for the SS2 arms table")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--gpu", default=None, help="nvidia-smi GPU id/UUID; omit on a single-GPU box (cloud H100)")
    ap.add_argument("--out-dir", default=None,
                    help="ABSOLUTE (dev plan/30's cwd-spill lesson). Default: "
                         "$SPA_OUTPUTS_ROOT/_incoming/m10_bench, matching configs/paths/default.yaml's "
                         "own outputs_root convention (falls back to $HOME/projects/spa/outputs if "
                         "SPA_OUTPUTS_ROOT is unset).")
    ap.add_argument("--json", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny scale (K=1, one length, skip k/m-sweep) to validate the pipeline before the full run")
    a = ap.parse_args()

    if not a.out_dir:
        import os
        project_root = os.environ.get("SPA_PROJECT_ROOT", str(Path.home() / "projects" / "spa"))
        outputs_root = os.environ.get("SPA_OUTPUTS_ROOT", str(Path(project_root) / "outputs"))
        a.out_dir = str(Path(outputs_root) / "_incoming" / "m10_bench")

    if a.smoke:
        a.k = 1
        a.lengths = a.lengths.split(",")[0]
        if a.residency_prompts:
            # trim to 2 prompts: enough to exercise the warm-process code path (build_eval_engine /
            # load_adapter / generate(engine=,adapter=), never run before this) without paying for
            # all 6 staged prompts at smoke scale.
            a.residency_prompts = ",".join(a.residency_prompts.split(",")[:2])
        print(f"[m10] SMOKE MODE: K=1, length={a.lengths} only, "
              f"arms + load-breakdown + residency (2 prompts) only")

    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"[m10] out_dir={a.out_dir}  phase={a.phase}  ckpt={a.ckpt}  gpu={a.gpu or '(default)'}")

    results = {}
    phases = ["arms", "k-sweep", "m-sweep", "residency", "load-breakdown"] if a.phase == "all" else [a.phase]
    if a.smoke:
        # k-sweep/m-sweep reuse the SAME _run_generate/_base_overrides path the arms phase already
        # validated (just different loop parameters); residency's warm-process arm does NOT (direct
        # Hydra compose + engine reuse, never exercised), so it belongs in smoke validation and
        # k-sweep/m-sweep do not need to be.
        phases = [p for p in phases if p in ("arms", "load-breakdown", "residency")]

    for p in phases:
        print(f"\n=== phase: {p} ===")
        if p == "arms":
            results["arms"] = phase_arms(a)
        elif p == "k-sweep":
            results["k_sweep"] = phase_ksweep(a)
        elif p == "m-sweep":
            results["m_sweep"] = phase_msweep(a)
        elif p == "residency":
            results["residency"] = phase_residency(a)
        elif p == "load-breakdown":
            results["load_breakdown"] = phase_load_breakdown(a)

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(results, indent=2, default=str))
        print(f"\n[m10] wrote {a.json}")


if __name__ == "__main__":
    main()
