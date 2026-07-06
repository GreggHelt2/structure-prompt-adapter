"""Enable OpenFold3 (pinned e583ecee) batch_size>1 for HOMOGENEOUS same-length batches — a runtime
monkeypatch, NO edit to the read-only OF3 dep (dev docs/plan/23; crash-hunt subagent 2026-07-05).

OF3's inference is already batch-generic (trunk/diffusion/confidence/writer all carry a real batch dim),
EXCEPT three per-sample `[B]` bool guards used in `if <tensor>:`, which crash at bs>1 with
"Boolean value of Tensor with more than one value is ambiguous" (fine at bs=1 = 1-element tensor):
  - OpenFold3AllAtom.predict_step        (runner.py:917 `if not valid_sample or is_repeated_sample:`)
  - OF3OutputWriter.on_predict_batch_end (writer.py:314 `if batch.get("repeated_sample"):`)
  - PredictTimer.on_predict_batch_end    (callbacks.py:67 same pattern)
We COERCE those guard tensors to scalars before the original method runs — correct for our case (all
samples valid, none padding/`repeated`: same-length single-seq refolds, no distributed even-sharding).
NOT correct for mixed valid/invalid batches (would all-or-nothing) — same-length homogeneous only.

Run OF3 through this shim instead of the bare console script:
    conda run -n spa-verify-of3 python of3_batch_patch.py predict --query-json … --runner-yaml <bs=B>.yml …
(OF3Refolder does this automatically when batch_patch_shim is set.)

**Upstream fix candidate:** this is an OF3 issue+PR — replace the 3 `if <tensor>:` guards with `.any()`/
`.all()` (+ handle the `reseed(seed[0])` bs=1 TODO at runner.py:926). See dev docs.
"""
import sys


def apply_patches():
    from openfold3.core.utils.callbacks import PredictTimer
    from openfold3.core.runners.writer import OF3OutputWriter
    from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom

    def _coerce(batch):
        """Scalarize the [B]-bool guard tensors in-place (all-valid / none-repeated for our batches)."""
        if not isinstance(batch, dict):
            return
        v = batch.get("valid_sample")
        if v is not None and hasattr(v, "numel"):
            batch["valid_sample"] = bool(v.reshape(-1).all())
        r = batch.get("repeated_sample")
        if r is not None and hasattr(r, "numel"):
            batch["repeated_sample"] = bool(r.reshape(-1).any())

    _orig_predict = OpenFold3AllAtom.predict_step

    def predict_step(self, batch, *a, **k):
        _coerce(batch)
        return _orig_predict(self, batch, *a, **k)

    OpenFold3AllAtom.predict_step = predict_step

    def _wrap_callback(cls):
        orig = cls.on_predict_batch_end

        def on_predict_batch_end(self, *a, **k):
            for arg in list(a) + list(k.values()):           # Lightning passes `batch` positionally
                if isinstance(arg, dict) and ("repeated_sample" in arg or "valid_sample" in arg):
                    _coerce(arg)
            return orig(self, *a, **k)

        cls.on_predict_batch_end = on_predict_batch_end

    _wrap_callback(OF3OutputWriter)
    _wrap_callback(PredictTimer)
    print("[of3_batch_patch] batch>1 guards coerced (predict_step + OF3OutputWriter/PredictTimer callbacks)",
          file=sys.stderr, flush=True)


def main():
    apply_patches()
    from openfold3.run_openfold import cli
    cli()


if __name__ == "__main__":
    main()
