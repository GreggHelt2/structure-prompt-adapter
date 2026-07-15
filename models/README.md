# SPA trained models

Published SPA adapter weights. Each file is an **adapter-only inference export** — just the
trained SPA cross-attention sidecar's `state_dict` (~90-130 MB). It does **not** include optimizer/
scheduler state and is not resumable for further training; it's the ready-to-load inference artifact.

## Naming change (2026-07-15)

These files were renamed when published here, from their internal research names. The internal
names used a variant letter (`A`/`B`/`C`) that is **unrelated to, and easily confused with**, the
separate "Run A" / "Run B" terminology used in this project's training logs (Run A/Run B refer to two
different *training runs* of the same `C`/N×1536 architecture, differing only in native RFdiffusion3
conditioning — not to the variant letters). To avoid that confusion, published files are named directly
after the **[architecture] × [conditioning]** matrix instead of the internal letter:

| Published filename | Architecture (internal letter) | Trained on (native RFD3 conditioning) | Original internal path | SHA-256 |
|---|---|---|---|---|
| `spa-Nx1536-uncond.pt` | N×1536, per-residue identity projector (`C`) | `unconditional` only ("Run A") | `checkpoints/spa-Nx1536-uncond/spa_C_final.pt` | `e25c8377302a2f79ae60f2b0b13371f6176cf8b991aa818f5172b41289846665` |
| `spa-Nx1536-motif.pt` | N×1536, per-residue identity projector (`C`) | `island` motif conditioning, ~50/50 mixed with unconditional ("Run B") | `checkpoints/spa-Nx1536-motif/spa_C_final.pt` | `1c7a043d7763f6a6f97785d9318f7e19bf3fded3ac349ecae65cc809e8c7139a` |
| `spa-1x32-uncond.pt` | 1×32, frozen CLSS `structure_adapter` bottleneck (`A`) | `unconditional` only | `checkpoints/spa-1x32-uncond/spa_A_final.pt` | `c71a8858baef7e85e192f1148e12b7ea1d83717170311939aca046352b6469a6` |
| `spa-Nx1536-multigran.pt` | N×1536, per-residue identity projector (`C`) | `unconditional` (native side) + sub-region/multigranularity prompt-masking curriculum on the SPA side | `checkpoints/spa-Nx1536-multigran/spa_C_final.pt` | `9e14ce7c836ff0e825c0a9abad486b8098302061e6d51709971f0d2aabce60d4` |
| *(not yet published)* `spa-1x1536-uncond.pt` | 1×1536, mean-pool + fan-out projector (`B`) | `unconditional` only | `checkpoints/spa-1x1536-uncond/spa_B_final.pt` | `59a8652a67d27c567f4cf5e8817fbb72b41ffa6b67d9f9069e0f0d178bef1c71` |

**`spa-1x1536-uncond.pt` is held back for now** — at ~126 MB it exceeds GitHub's 100 MB hard limit for
a plain `git add`/push, and needs Git LFS (or another distribution path) before it can be added. Do not
assume it's present in `models/` until this note is removed.

## Which one do I want?

- **`spa-Nx1536-uncond`** — the default / general-purpose choice. Used for nearly all of the published
  poster/paper results, including applying it zero-shot to hard-motif (native RFdiffusion3) conditioning.
- **`spa-Nx1536-motif`** — trained directly on hard-motif conditioning. Included primarily for
  reproducibility of the head-to-head comparison showing it performs about the same as
  `spa-Nx1536-uncond` on that task ("emergent zero-shot") — not because it's the better default.
- **`spa-1x32-uncond`** — the CLSS-framed, most compressed variant. Performs comparably to the N×1536
  variant on both adherence and designability at a fraction of the prompt-representation cost.
- **`spa-Nx1536-multigran`** — trained on a sub-region/partial-prompt masking curriculum. Use this one
  for multi-region or localized/composable conditioning (steering only part of a structure). Also
  performs comparably to `spa-Nx1536-uncond` when the latter is given a masked prompt zero-shot, but
  this is the checkpoint actually used to generate the composability figures.
