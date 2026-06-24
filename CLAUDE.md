# CLAUDE.md — Structure Prompt Adapter (SPA) — public implementation repo

This is the **public** repo: code, trained models, and public-facing docs only. It is a
lean operating guide — **not** the design record. The design rationale, decisions, and
specifications live in the private dev repo's planning docs (see below); do not restate them
here.

## What this is

SPA is a parameter-efficient **sidecar adapter for RFdiffusion3** (RFD3). It adds a *decoupled
cross-attention* term to RFD3's token track, keyed/valued on an external **ESM3 structural
prompt**, so a user can softly steer fold topology without freezing atomic coordinates.
Modeled on IP-Adapter (image diffusion). RFD3 and ESM3 stay **frozen**; only SPA trains.

## Design source of truth

`../structure-prompt-adapter-dev/docs/plan/` is authoritative for *why* and *what*:

| Topic | Doc |
|------|-----|
| Readiness, ordered build steps, blockers | `08_implementation_kickoff.md` (living) |
| RFD3 internals, ESM3/CLSS tensors, IP-Adapter blueprint | `01_codebase_analysis.md` |
| Injection points, wrapper, prompt side-channel | `02_attachment_points.md` |
| SPA module, 3 variants, zero-init, param counts, CFG | `03_spa_architecture.md` |
| Data/filter/splits, loss, caching, cloud Phases 1–2 | `04_training_strategy.md` |
| Validation flywheel (ProteinMPNN → OpenFold3), metrics | `05_validation_pipeline.md` |
| Hypotheses, baseline, ablations, 3-phase ISMB plan | `06_experiments.md` |
| Open questions / decisions / contradictions log | `07_open_questions.md` |

**Code is ground truth.** When code and a plan doc disagree, the code wins — fix the doc in
the dev repo and note it in `07_open_questions.md`. Don't duplicate plan content into this repo.

## Repo layout

```
structure-prompt-adapter/
├── CLAUDE.md                       # this file
├── README.md                       # poster-facing (public) — leave as the public landing page
├── pyproject.toml                  # package metadata; heavy deps are env-managed (see below)
├── configs/                        # Hydra config tree — ALL hardware/VRAM/variant knobs
│   ├── train.yaml / infer.yaml     # top-level (defaults lists)
│   ├── model/spa.yaml              # SPA hyperparams (c_model, n_head, shared K/V, zero-init…)
│   ├── variant/{C,B,A}_*.yaml      # CLSS-integration variant = a config switch (C is primary)
│   ├── data/{toy,cddb}.yaml        # dataset + cache locations
│   ├── hardware/{local_a5000,cloud_h100}.yaml   # device + VRAM-driven batch/protein caps
│   ├── train/default.yaml          # optimizer, lr, epochs, CFG drop-rate, λ
│   └── paths/default.yaml          # weight/cache/data roots
├── src/spa/                        # the package (src layout, import name `spa`)
│   ├── model/   cross_attention.py · wrapper.py · projectors.py · loader.py
│   ├── prompt/  esm3_prompt.py     # ESM3 → per-residue (N,1536) prompt producer
│   ├── data/    dataset.py         # cache-backed dataloader
│   ├── train/   harness.py         # own training loop (RFD3/ESM3 frozen, grad on SPA)
│   └── utils/   device.py          # device from config (NEVER a hardcoded UUID)
├── scripts/     train.py · gen_esm3_cache.py   # Hydra entry points
└── tests/       test_identity_at_init.py       # the standing correctness gate (below)
```

## Environments & how to run

Two conda envs (created in dev Task 0; see root dev `CLAUDE.md`). `conda activate` does **not**
persist across Claude Code's fresh-shell-per-command model — **always use `conda run`**:

| Env | Purpose |
|-----|---------|
| `spa-dev` | **primary** — editable installs of the full stack from `../needed_repos/` (torch 2.5.1+cu124, rc-foundry`[rfd3]`, atomworks, esm, clss-model). Develop + train here. |
| `spa-verify` | clean-room torch-only verify env (optional). |
| `spa-verify-of3` | OpenFold3 validator (downstream; separate env). |

```bash
conda run -n spa-dev pip install -e .                       # editable-install this package
conda run -n spa-dev python scripts/train.py --cfg job      # print composed Hydra config
conda run -n spa-dev python scripts/train.py variant=C_n_by_1536 hardware=local_a5000
conda run -n spa-dev python scripts/gen_esm3_cache.py data=toy
conda run -n spa-dev pytest                                 # run tests (incl. the identity gate)
```

**A5000 UUID masking is baked into both envs' vars**
(`CUDA_VISIBLE_DEVICES=GPU-46586b6c-…`), so `conda run -n spa-dev python …` automatically
targets the RTX A5000 and the display-only RTX 5060 is invisible — after masking the A5000 is
simply `cuda:0`. This masking is a **local, env-level concern**: it must never appear in source
code. Device selection comes from `configs/hardware/*.yaml` (`device: cuda:0`), so the
local-A5000 → cloud-H100 move is a **config change, not a code rewrite**.

## Standing correctness gate — the identity invariant

> **Wrapped-no-prompt must equal vanilla RFdiffusion3, bit-for-bit.**

With SPA attached but **no prompt stashed** on the wrappers (equivalently `λ=0`, or the
zero-initialized output projection `Wo=0`), the SPA term is exactly zero and the model must
reproduce vanilla RFD3 *exactly*. This is three things at once:

1. the **identity-at-init** guarantee that makes training stable (ControlNet-style "grow in"),
2. the **experiment baseline** (RFD3-without-SPA = wrapped-no-prompt), and
3. an enforced unit test: `tests/test_identity_at_init.py`.

**Keep that test green.** Any change that breaks it is a bug unless you are *deliberately*
changing the adapter's identity-at-init contract (and then the test, the baseline, and the
docs all move together). The mechanism is **zero-init of `Wo`** (SPA cannot warm-start like
IP-Adapter — the prompt is 1536-d, RFD3 K/V are 768-d; see `02 §4`, `03 §5`).

## Invariants / house rules

- **Dependency repos under `../needed_repos/` are READ-ONLY.** SPA patches RFD3 at runtime by
  **wrapping** the 18 `attention_pair_bias` modules — it never edits upstream source.
- **Repo separation.** Code lives only here; planning artifacts only in the dev repo. The two
  never exchange files. This repo is **public** — never commit weights, caches, data, or `.env`.
- **Frozen vs trainable.** RFD3 + ESM3 (and CLSS when the 1×32 variant is used) are frozen;
  grad flows **only** to SPA params, gathered in a `ModuleList` for optimize/checkpoint.
- **Prompt is a side-channel.** RFD3 has no forward slot for an external prompt, so the
  projected prompt is **stashed on the wrappers** (via `SPAContext`) once per design and reused
  — it is **constant across all 200 diffusion steps × 18 blocks** (ESM3 run once, cached).
- **Parameterize hardware.** Device, VRAM-driven sizes (batch, protein caps), paths, wheel/CUDA
  → config/env only, never hardcoded constants. cu124 + UUID masking are local-only;
  H100-only optimizations (FP8/TransformerEngine) stay optional + feature-flagged.

## Resolved decisions that the code relies on

- **ESM3 prompts come from LOCAL HuggingFace weights** (`ESM3.from_pretrained("esm3_sm_open_v1")`
  → HF cache, already present), **not** the Forge API — no `ESM_API_KEY` needed.
  *(Resolved: dev `07` C2 / `01 §3.5`.)*
- **esm2 is not used by SPA.** The `../needed_repos/esm2/` repo is never installed; CLSS's ESM2
  *sequence* tower only feeds the 1×32 contrastive path, which SPA's structure prompts do not
  touch. Only build/install ESM2-via-CLSS if the 1×32 **sequence** variant is ever exercised.
  *(Resolved: dev `07` Q0.1 / Q2.1.)*

## Commit workflow

This repo has its **own** git history and remote
(`github.com/GreggHelt2/structure-prompt-adapter`, **PUBLIC**). Commit code changes here; commit
the corresponding planning/status updates separately in the dev repo. See the dev root
`CLAUDE.md` → Commit Workflow. Commit/push only when asked.
