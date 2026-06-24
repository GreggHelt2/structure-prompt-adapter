"""SPA training harness — own loop, RFD3/ESM3 frozen, gradient on SPA params only.

Kickoff steps 6–7. Spec: dev ``04_training_strategy.md`` §5–§6; ``02`` §5 (own harness needed —
RFD3's Lightning inference engine runs under ``torch.no_grad()``).

Per step:
  1. Pull ``(target, prompt)`` from the cache-backed loader; apply SE(3) augmentation to target.
  2. Project the prompt -> shared K/V; stash on the ``SPAContext`` (CFG: drop to zero-prompt with
     prob ~5–10% so the wrapper returns base only — dev ``03`` §8 / Q1.1).
  3. Run the RFD3 diffusion forward WITH grad enabled on the SPA params (host frozen).
  4. RFD3-native loss; backward; step the optimizer over the SPA ``ModuleList`` only.

Checkpoints save the SPA ``ModuleList`` (not the frozen host). All knobs (lr, epochs, batch,
protein cap, CFG drop-rate, λ, device) come from Hydra config so local-A5000 -> cloud-H100 is a
config change. Step 7 = overfit a tiny subset and confirm loss falls + λ responds.
"""

from __future__ import annotations


def train(cfg) -> None:
    """Run SPA training from a composed Hydra config. See module docstring for the loop."""
    # TODO(step 6): build frozen RFD3, attach SPA (spa.model.loader.attach_spa), build loader,
    # optimizer over SPA params, run the loop above, checkpoint SPA params.
    raise NotImplementedError(
        "train() is a step-1 scaffold; implement in kickoff steps 6–7 "
        "(dev 04_training_strategy.md §5–§6)."
    )
