"""Cancellation signal for the large-data (TIFF -> OME-Zarr) converter.

napari-free BY DESIGN (mirrors the rest of ``packages/epicycle_largedata``):
stdlib only, no Qt / napari.
"""

from __future__ import annotations


class ConvertCancelled(BaseException):
    """Raised when the user cancels a ``convert_to_ome_zarr`` run.

    Derives from :class:`BaseException` — **not** :class:`Exception` — on
    purpose (the project's cancel pattern, mirroring
    ``packages/epicycle_preprocessing/exceptions.py::PreprocessingCancelled``
    and ``packages/epicycle_training/exceptions.py::TrainingAbortedError``).
    The convert path is wrapped in broad ``except Exception`` guards on the way
    out — the Step-1 worker's own handler and dask's task machinery — so a
    normal ``Exception`` would be reported to the user as "Failed to prepare",
    a failure they did not cause.  Deriving from ``BaseException`` keeps the
    cancel distinguishable all the way up to the worker's terminal funnel,
    which turns it into a clean ``cancelled`` outcome.

    The converter's own ``finally`` still runs (``BaseException`` does not skip
    it), so the half-written store is removed and the ``epicycle_source``
    freshness marker — written only after the LAST level — is never present.
    """
