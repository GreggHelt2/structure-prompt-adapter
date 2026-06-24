"""The standing correctness gate: wrapped-no-prompt == vanilla RFD3, bit-for-bit.

Kickoff step 4 (dev 02_attachment_points.md §8, 03_spa_architecture.md §1). This is THE
invariant for the whole project (see repo CLAUDE.md → "Standing correctness gate"): with SPA
attached but no prompt stashed (zero-init Wo), the model must reproduce vanilla RFdiffusion3
exactly — it is both the training-stability guarantee and the experiment baseline.

Placeholder until the wrapper/loader land (kickoff steps 2–3); skipped so the suite stays green.
"""

import pytest


@pytest.mark.skip(reason="kickoff step 4 — needs SPACrossAttention + wrapper/loader (steps 2–3)")
def test_wrapped_no_prompt_equals_vanilla_rfd3():
    # Plan (dev 02 §8):
    #   1. Build frozen RFD3 from cfg.paths.rfd3_ckpt; run a fixed-seed design -> coords_vanilla.
    #   2. attach_spa(model, cfg) with zero-init Wo; DO NOT stash a prompt on the context.
    #   3. Run the same fixed-seed design -> coords_wrapped.
    #   4. assert torch.equal(coords_vanilla, coords_wrapped)   # bit-for-bit identity at init
    ...
