"""QThread worker for Step 3 Batch Refine.

Hosts the pure :func:`packages.epicycle_annotator.batch_refine.run_batch_recipe`
executor on a QThread so a batch (potentially hundreds–thousands of
labels) runs off the Qt main thread and keeps napari responsive.

The pure executor already parallelises per-label *compute* internally with
a ``ThreadPoolExecutor`` (scipy releases the GIL on the heavy morphology /
EDT loops).  This QObject merely hosts that single call on a worker
QThread and marshals progress / result / error back to the main thread
via signals.  Cancellation is a ``threading.Event`` the executor polls
between labels.

Thin by design — all batch logic lives in the pure module; this layer is
only the Qt thread/signal plumbing, so it carries no business logic to
test beyond "it forwards the call + emits the right signals".
"""
from __future__ import annotations

import threading
import traceback

from qtpy.QtCore import QObject, Signal

from packages.epicycle_annotator.batch_refine import run_batch_recipe


class BatchRefineWorker(QObject):
    """Run a batch refine recipe on a QThread.

    Signals
    -------
    stage(name, done, total)
        ``"preparing"`` (0, 0) before the working copy, ``"refining"``
        (0, n) before the first compute, ``"writing"`` (0, n) before the
        apply pass.
    progress(done, total)
        Emitted as each label's compute finishes (throttled).
    finished(result)
        Emitted once with the :class:`~packages.epicycle_annotator.batch_refine.BatchResult`.
    error(message)
        Emitted instead of ``finished`` if the executor raises.
    """

    stage = Signal(str, int, int)
    progress = Signal(int, int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        volume,
        label_ids,
        recipe,
        *,
        n_workers=None,
        cancel_event=None,
        spacing=None,
        channel=None,
        prob=None,
        orphans=None,
    ):
        super().__init__()
        self._volume = volume
        self._label_ids = list(label_ids)
        # A saved Refine chain (`refine_recipe.Recipe`) is a NON-iterable
        # dataclass: it is stored unchanged and takes the chain path;
        # a step list is copied as before.
        from packages.epicycle_annotator.refine_recipe import Recipe

        self._recipe = recipe if isinstance(recipe, Recipe) else list(recipe)
        self._n_workers = n_workers
        self._cancel_event = cancel_event or threading.Event()
        self._spacing = spacing
        self._channel = channel
        self._prob = prob
        self._orphans = orphans

    @property
    def cancel_event(self) -> threading.Event:
        return self._cancel_event

    def cancel(self) -> None:
        """Request cancellation; the executor stops between labels."""
        self._cancel_event.set()

    def run(self) -> None:
        """Execute the batch (call this as the QThread's ``started`` slot).

        Synchronous: returns after emitting either ``finished`` or
        ``error``.  Safe to call directly in tests.
        """
        try:
            result = run_batch_recipe(
                self._volume,
                self._label_ids,
                self._recipe,
                n_workers=self._n_workers,
                progress_cb=lambda done, total: self.progress.emit(
                    int(done), int(total)
                ),
                cancel_event=self._cancel_event,
                stage_cb=lambda name, done, total: self.stage.emit(
                    str(name), int(done), int(total)
                ),
                spacing=self._spacing,
                channel=self._channel,
                prob=self._prob,
                orphans=self._orphans,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.error.emit(f"{exc}\n{traceback.format_exc()}")
            return
        self.finished.emit(result)
