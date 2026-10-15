"""Cancellation signal for the Step-1 preprocessing engine.

napari-free BY DESIGN (mirrors the rest of ``packages/epicycle_preprocessing``):
stdlib only, no Qt / napari.
"""

from __future__ import annotations


class PreprocessingCancelled(BaseException):
    """Raised when the user cancels a whole-volume preprocessing run.

    Derives from :class:`BaseException` — **not** :class:`Exception` — on purpose
    (the project's cancel pattern, mirroring
    ``packages/epicycle_training/exceptions.py::TrainingAbortedError``).  The
    preprocessing pipeline wraps every op in ``except Exception`` skip-and-continue
    guards (``ops._run_intensity_ops`` / ``ops.execute``); a normal ``Exception``
    raised on cancel would be SWALLOWED there — the op silently skipped and the
    pipeline continued on unprocessed data.  Because this derives from
    ``BaseException`` it sails through those guards up to the worker's top-level
    handler, which turns it into a clean "cancelled" outcome.

    The cancel check itself lives in the tile-pool management loop
    (``chunked.block_parallel_apply``), never inside an op fn — but the surviving
    reason is the ``BaseException`` derivation, so DO NOT re-home this under
    ``Exception`` on the belief the check location protects it (it does not; see
    the negative-control test that proves a plain ``Exception`` is swallowed).
    """
