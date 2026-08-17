"""Generate with the RAW RFD3 engine, SPA never attached at all (dev `plan/73` M10).

WHY THIS SCRIPT EXISTS, NOT JUST `conditions=[baseline]` IN `generate.py`. `generate()` in
`spa/eval/generate.py` *always* calls `load_adapter()` (which calls `attach_spa()`, wrapping the
18 token-track blocks) even for `conditions=[baseline]` — that condition only zeroes the SPA
*output* (`clear_prompt`), it does not skip attaching the wrapper. So `conditions=[baseline]`
measures "wrapper attached, silent," not "wrapper never inserted into the model at all," and the
two are not necessarily the same cost. This script calls `build_eval_engine()` +
`spa.eval.generate._run_once()` directly and imports nothing else from `spa.model` in the whole
process, so there is no code path by which SPA could be attached.

`build_eval_engine()` / `_run_once()` are pure RFD3 (`attach_spa` is imported lazily, inside
`load_adapter()`, which this script never calls -- verified by reading `spa/model/loader.py` and
`spa/eval/generate.py` directly, not assumed).

Usage (mirrors generate.py's own overrides):
    conda run -n spa-dev python scripts/eval/generate_rfd3_only.py \\
        variant=C_n_by_1536 eval.length=100 eval.num_designs=8 eval.out_dir=/abs/out
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    from spa.eval.generate import _resolve_out_dir, _run_once, build_eval_engine

    engine = build_eval_engine(cfg)          # pure RFD3; SPA is never imported, let alone attached
    out_dir = _resolve_out_dir(cfg.eval.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = _run_once(engine, spec=None)   # unconditional, K = eval.num_designs
    print(f"rfd3-only: generated {len(outputs)} design(s) under {out_dir} "
          f"(SPA never attached this process)")


if __name__ == "__main__":
    main()
