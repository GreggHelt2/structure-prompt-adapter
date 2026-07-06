"""Enable OpenFold3 (pinned e583ecee) batch_size>1 for SAME-LENGTH batches (incl. RAGGED-ATOM ones,
e.g. different ProteinMPNN designs of one backbone) — a runtime monkeypatch, NO edit to the read-only
OF3 dep (dev docs/plan/23; crash-hunt + A5000 debug subagent 2026-07-05).

OF3's trunk/diffusion FORWARD is genuinely batch-generic (it runs fine at bs>1 — that's where the
speedup is). Three *separate* things break at bs>1, in order; we patch all three:

(1) **[B]-bool guards** — three per-sample `[B]` bool tensors used in `if <tensor>:` crash with
    "Boolean value of Tensor with more than one value is ambiguous" (fine at bs=1 = 1-elem tensor):
      - OpenFold3AllAtom.predict_step        (runner.py:917 `if not valid_sample or is_repeated_sample:`)
      - OF3OutputWriter.on_predict_batch_end (writer.py:314 `if batch.get("repeated_sample"):`)
      - PredictTimer.on_predict_batch_end    (callbacks.py:67 same pattern)
    We coerce them to scalars (all samples valid, none `repeated` — same-length single-seq refolds, no
    distributed even-sharding). NOT correct for mixed valid/invalid batches — same-length only.

(2) **Ragged-atom confidence bug** — same-LENGTH sequences (same #tokens) have DIFFERENT #atoms
    (sidechain-dependent), so the collator pads the atom dim to the batch max. In the per-sample
    confidence loop, `atom_mask`/`x` are atom-PADDED (max, e.g. 233) but the token→atom broadcast
    (`num_atoms_per_token`) is the TRUE count (e.g. 230): they collide at
    `atomize_utils.get_token_frame_atoms` (`pair_mask * atom_asym_id_mask`, 233 vs 230). Crash is
    SWALLOWED by predict_step's try/except → exit 0, ZERO cifs. We truncate `atom_mask`/`x` to the true
    atom count (real atoms lead, padding trails) so the frame math is self-consistent. bs=1 = no-op.

(3) **Writer padding** — `atom_positions_predicted`/`plddt` are atom-padded (max), but the biotite
    `atom_array` is the TRUE unpadded structure; `set_annotation`/`.coord` demand equal length. We slice
    coords/plddt to `len(atom_array)` before writing. bs=1 = no-op.

**Equivalence (A5000-validated, 8 seqs, bs=1 vs bs=8):** batched refolds are VALID folds and the
designability statistics are preserved — best-of-K matched (0.29 vs 0.32 Å) and designable-rate matched
(7/8 both) — at a **3.3× speedup**. They are NOT bit-identical to bs=1 per sample, though: `predict_step`
does `reseed(seed[0])` once (runner.py:926 "TODO bs=1"), so only batch-row 0 reproduces bs=1 exactly;
other rows get a different (equally valid) diffusion draw (~0.9 Å Cα off). Fine for best-of-K designability
YIELD; do NOT rely on it for bit-reproducible single-sample refolds. Per-sample reseeding is part of the
upstream fix below.

Run OF3 through this shim instead of the bare console script:
    conda run -n spa-verify-of3 python of3_batch_patch.py predict --query-json … --runner-yaml <bs=B>.yml …
(OF3Refolder does this automatically when batch_patch_shim is set.)

**Upstream fix candidate:** this is an OF3 issue+PR — (1) replace the 3 `if <tensor>:` guards with
`.any()`/`.all()` + the `reseed(seed[0])` bs=1 TODO (runner.py:926); (2) align atom-dim tensors to the
true atom count in `get_token_frame_atoms` (or unpad per sample in `get_confidence_scores`); (3) slice
padded coords/plddt to the atom_array length in `write_structure_prediction`. See dev docs/plan/23 §7.
"""
import sys


def apply_patches():
    from openfold3.core.utils.callbacks import PredictTimer
    from openfold3.core.runners.writer import OF3OutputWriter
    import openfold3.projects.of3_all_atom.runner as _runner
    from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom
    import openfold3.core.metrics.aggregate_confidence_ranking as _acr

    # -- (1) [B]-bool guards ------------------------------------------------------------------------
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

    # -- (1b) SURFACE swallowed predict_step exceptions to stderr -----------------------------------
    # predict_step wraps forward+confidence in a try/except (runner.py:954) that logs to
    # <out>/logs/predict_err_rank*.log and returns None — so ANY bs>1 failure looks like "exit 0, zero
    # cifs" (the H100 "no folds" mystery: we were blind because the captured subprocess stderr is
    # discarded on exit-0). Mirror the full traceback to stderr so it lands in captured output / cloud
    # job logs. Read-only OF3 is untouched; we only wrap its exception logger.
    import traceback as _tb
    _orig_logexc = OpenFold3AllAtom._log_predict_exception

    def _log_predict_exception(self, e, query_id):
        try:
            print("[of3_batch_patch] SWALLOWED OF3 predict_step exception (surfaced to stderr):\n"
                  + _tb.format_exc(), file=sys.stderr, flush=True)
        except Exception:
            pass
        return _orig_logexc(self, e, query_id)

    OpenFold3AllAtom._log_predict_exception = _log_predict_exception

    # -- (2) ragged-atom confidence: unpad each sample to its TRUE atom count ------------------------
    # Same-length seqs have different #atoms; the collator atom-pads to the batch max A. OF3's
    # confidence loop keeps that padding, but the token->atom broadcast (num_atoms_per_token) uses the
    # TRUE count n, so padded (A) and true (n) atom tensors collide EVERYWHERE downstream
    # (get_token_frame_atoms, compute_has_clash, ...). Rather than patch each site, we reimplement
    # get_confidence_scores: per sample, slice + truncate every atom-dim tensor (dim==A) to n, run OF3's
    # own _get_confidence_scores (now fully self-consistent at n), then zero-pad the atom-dim OUTPUTS
    # back to A so the per-sample dicts still torch.stack uniformly. Writer patch (3) slices those
    # padded outputs back to each atom_array's true length. bs=1 delegates to the original (no-op).
    import torch as _torch
    from openfold3.core.utils.tensor_utils import dict_multimap as _dmm, tensor_tree_map as _ttm
    _orig_gcs = _acr.get_confidence_scores
    _get_one = _acr._get_confidence_scores

    def _resize_atom_dim(t, cur, new):
        """Slice (new<cur) or zero-pad (new>cur) the FIRST dim of size `cur` to length `new`."""
        if not isinstance(t, _torch.Tensor) or cur == new:
            return t
        d = next((i for i in range(t.ndim) if t.shape[i] == cur), None)
        if d is None:
            return t
        if new < cur:
            idx = [slice(None)] * t.ndim
            idx[d] = slice(0, new)
            return t[tuple(idx)]
        pad = list(t.shape)
        pad[d] = new - cur
        return _torch.cat([t, t.new_zeros(pad)], dim=d)

    def _gcs(batch, outputs, config, compute_per_sample=False):
        ap = outputs["atom_positions_predicted"]
        B, A = ap.size(0), ap.size(-2)
        if B <= 1:
            return _orig_gcs(batch, outputs, config, compute_per_sample)

        per = []
        for bi in range(B):
            def _slice_squeeze(x, _bi=bi):              # mirror the original per-batch slice (+squeeze sample dim)
                y = x[_bi]
                return y.squeeze(0) if (isinstance(y, _torch.Tensor) and y.ndim >= 1) else y
            cb = _ttm(_slice_squeeze, batch, strict_type=False)
            cb["atom_array"] = batch["atom_array"][bi]
            co = _ttm(lambda x, _bi=bi: x[_bi], outputs, strict_type=False)
            n = int(cb["num_atoms_per_token"].sum())
            cb = _ttm(lambda x: _resize_atom_dim(x, A, n), cb, strict_type=False)   # A -> n (truncate padding)
            co = _ttm(lambda x: _resize_atom_dim(x, A, n), co, strict_type=False)
            m = _get_one(batch=cb, outputs=co, config=config)                        # consistent at n
            per.append(_ttm(lambda x: _resize_atom_dim(x, n, A), m, strict_type=False))  # n -> A (pad for stack)
        return _dmm(_torch.stack, per)

    # runner.py did `from ...aggregate_confidence_ranking import get_confidence_scores` (its own binding),
    # so patch the RUNNER module's attribute (patching _acr's would miss). Patch both for good measure.
    _runner.get_confidence_scores = _gcs
    _acr.get_confidence_scores = _gcs

    # -- (3) writer: slice atom-padded coords/plddt down to the unpadded atom_array length ----------
    _orig_wsp = OF3OutputWriter.write_structure_prediction  # underlying func (a @staticmethod)

    def _wsp(atom_array, predicted_coords, plddt, *a, **k):
        n = len(atom_array)
        if getattr(predicted_coords, "shape", (n,))[0] != n:
            predicted_coords = predicted_coords[:n]
        if getattr(plddt, "shape", (n,))[0] != n:
            plddt = plddt[:n]
        return _orig_wsp(atom_array, predicted_coords, plddt, *a, **k)

    OF3OutputWriter.write_structure_prediction = staticmethod(_wsp)

    print("[of3_batch_patch] batch>1 patched: [B]-guards + ragged-atom confidence + writer unpad",
          file=sys.stderr, flush=True)


def main():
    apply_patches()
    from openfold3.run_openfold import cli
    cli()


if __name__ == "__main__":
    main()
