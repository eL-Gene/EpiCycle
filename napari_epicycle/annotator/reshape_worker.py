"""QThread worker for the Step 3 Reshape engine (guided Split).

Hosts the pure :func:`packages.epicycle_annotator.reshape.reshape_region` on a
QThread so the solve -- preprocessing chains, the watershed / random-walker,
the arbiter statistics and the refine stage -- runs off the Qt main thread and
napari stays responsive (heavy work goes to a worker; the UI thread only does
Qt and napari-layer calls).

Thin by design, modelled verbatim on :mod:`batch_refine_worker`: it carries no
business logic beyond "forward the call, emit the right signals".

**Cancel is honest**.  ``cancel_event`` is polled *between* stages;
three stages cannot be interrupted once entered (``random_walker``,
``diffuse_labels3D``, ``guidedfilter``).  So Cancel means **"abandon the
result"**, not "stop the computation": the UI returns immediately and the
worker's eventual output is discarded by the caller's stale-finish token.
"""
from __future__ import annotations

import threading
import time as _time
import traceback
from dataclasses import dataclass, field

from qtpy.QtCore import QObject, Signal

from packages.epicycle_annotator.reshape import find_seeds, reshape_region

#: Stage-name -> human sentence, for the busy bar's status line.  The engine
#: emits raw stage names (``reshape.engine.STAGES``); the UI shows these.
STAGE_TEXT = {
    "roi": "Finding the cell...",
    "walls": "Building the membrane surface...",
    "seeds": "Placing the seeds...",
    "guard": "Checking the job size...",
    "solve": "Cutting the cell...",
    "arbiter": "Weighing the membrane evidence...",
    "refine": "Tidying the pieces...",
    "stats": "Measuring the pieces...",
    "done": "Done.",
    # The Refine tab's own stages.  Same table, because
    # the one busy strip on that page shows whichever job is running and a
    # second table would be a second thing to keep in step.
    "prepare": "Preparing the channel...",
    "elevation": "Building the wall picture...",
    "band": "Measuring the band around the outline...",
    "contour": "Sliding the outline...",
    "rebuild": "Filling in the slices between...",
    "settle": "Settling the outline onto the wall...",
    "stopping": "Stopping...",
    # The Add tab's own stages (the region target
    # emits add_box / elevation / add_markers / add_cut as each STARTS).
    "add_box": "Reading the box...",
    # the same stage on a whole patch (the page picks it)
    "add_box_patch": "Reading the patch...",
    "add_markers": "Finding markers...",
    "add_cut": "Filling the empty space...",
    "add_find": "Finding bright objects...",
    "add_smooth": "Smoothing the new objects...",
    "add_info": "Measuring the box...",
    # Cells: the tissue outline, Spread from each marker (in tiles on a
    # box), and the growth engine's first set-up
    "add_outline": "Measuring the tissue outline...",
    "add_grow": "Spreading from each marker...",
    "add_piece": "Spreading from each marker...",
    "add_warm": "Preparing the growth engine...",
    # The Add page's own waits on the main thread (the page paints the line
    # before the work starts), and its readers on a worker
    "add_accept": "Writing the new objects...",
    "add_draw": "Drawing the new objects...",
    "add_clear": "Clearing the objects...",
    "add_undo_all": "Taking the objects back...",
    "add_discard": "Taking the result off the screen...",
    "add_walls": "Drawing the wall picture...",
    "add_balls": "Drawing the markers...",
    "add_info_patch": "Measuring the patch...",
    # Cells > Click markers: the click fill's first update of a box, and of
    # a whole patch filled as you click (the page picks it)
    "add_prepare": "Preparing the box...",
    "add_prepare_patch": "Preparing the patch...",
    "add_run": "Reading the run's objects...",
}


def stage_text(stage) -> str:
    """Plain-language sentence for a stage name (unknown -> the name itself)."""
    return STAGE_TEXT.get(str(stage), str(stage))


def touching_verdicts(labels, seed_layer, target_id, roi, *, margin=4,
                      seeds_in_region=None, cancel_event=None):
    """The touching rule's verdicts for markers taken from a label mask.

    ``{label id: MarkerContact}`` for every label of ``seed_layer`` inside
    the object ``target_id``.  ``roi`` is the search box (computed from
    ``margin`` when None, as the engine does); ``seeds_in_region`` is the
    guide already cropped and masked to the object, when the caller has it
    (a Find's report).  The label sizes are counted over the WHOLE
    ``seed_layer``.  Runs on the worker's thread; returns ``{}`` when there
    is nothing to measure, or when ``cancel_event`` is set mid-sweep.
    """
    import numpy as np

    from packages.epicycle_annotator.batch_refine import compute_label_roi
    from packages.epicycle_annotator.reshape.seeds import marker_contacts

    if seed_layer is None or target_id is None:
        return {}
    labels = np.asarray(labels)
    if roi is None:
        roi = compute_label_roi(labels, int(target_id), int(margin))
    if roi is None:
        return {}
    region = labels[roi] == int(target_id)
    guide = np.asarray(seed_layer)
    if seeds_in_region is None:
        crop = guide[roi] if guide.shape == labels.shape else guide
        if crop.shape != region.shape:
            return {}
        seeds_in_region = np.where(region, crop, 0)
    return marker_contacts(seeds_in_region, region, guide,
                           cancel_event=cancel_event)


def _touching_line(contacts, wall_ms) -> str:
    kept = sum(1 for c in contacts.values() if c.keep)
    return ("[RESHAPE-FIND] touching rule wall={:.0f}ms kept={} "
            "left_unticked={}".format(wall_ms, kept, len(contacts) - kept))


class ReshapeWorker(QObject):
    """Run one :func:`reshape_region` call on a QThread.

    Signals
    -------
    progress(stage, i, n)
        Emitted as each stage begins; ``stage`` is a member of
        ``reshape.engine.STAGES`` and ``i``/``n`` size a determinate bar.  Use
        :func:`stage_text` for the UI string.
    message(text)
        Emitted for every plain-language note the engine produces, live, so the
        panel can show a guard firing before the run ends.
    finished(result)
        Emitted once with the :class:`~packages.epicycle_annotator.reshape.ReshapeResult`.
        A refusal (no seeds, over budget) arrives here with ``status="refused"``
        -- a refusal is an *answer*, not an error.
    error(message)
        Emitted instead of ``finished`` if the engine raises.
    """

    progress = Signal(object, int, int)
    message = Signal(str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, labels, images, spec, *, target_id=None, roi=None,
                 seed_layer=None, spacing=(1, 1, 1), margin=4,
                 keep_seed_ids=None, cancel_event=None,
                 keep_intermediates=False, touching_rule=False):
        super().__init__()
        #: Markers from a label mask with no Find before the Preview: the
        #: touching labels are left out of the cut here, on this thread,
        #: exactly as a Find would have left them unticked.  Only applied
        #: when ``keep_seed_ids`` is None (no ticks to honour).
        self._touching_rule = bool(touching_rule)
        self._labels = labels
        self._images = dict(images or {})
        self._spec = spec
        self._target_id = target_id
        self._roi = roi
        self._seed_layer = seed_layer
        self._spacing = tuple(spacing)
        self._margin = margin
        self._keep_seed_ids = keep_seed_ids
        self._keep_intermediates = bool(keep_intermediates)
        self._cancel_event = cancel_event or threading.Event()

    @property
    def cancel_event(self) -> threading.Event:
        return self._cancel_event

    def cancel(self) -> None:
        """Request cancellation.

        The engine stops between stages.  A stage already entered runs to
        completion; the caller discards the result (see the module docstring).
        """
        self._cancel_event.set()

    def run(self) -> None:
        """Execute the reshape (the QThread's ``started`` slot).

        Synchronous: returns after emitting either ``finished`` or ``error``.
        Safe to call directly in tests.
        """
        try:
            keep_seed_ids = self._keep_seed_ids
            n_left_out = 0
            result = None
            if self._touching_rule and keep_seed_ids is None:
                t0 = _time.perf_counter()
                contacts = touching_verdicts(
                    self._labels, self._seed_layer, self._target_id,
                    self._roi, margin=self._margin,
                    cancel_event=self._cancel_event)
                keep_seed_ids = sorted(
                    int(k) for k, c in contacts.items() if c.keep)
                n_left_out = len(contacts) - len(keep_seed_ids)
                print(_touching_line(
                    contacts, (_time.perf_counter() - t0) * 1000.0))
                if (contacts and not keep_seed_ids
                        and not self._cancel_event.is_set()):
                    # Every label only touches the object.  Refused here,
                    # in words about label markers: the engine would build
                    # the walls first and then talk about prominence.
                    from napari_epicycle.annotator import (
                        _reshape_text as RT,
                    )
                    from packages.epicycle_annotator.reshape import (
                        ReshapeResult,
                    )

                    result = ReshapeResult(
                        status="refused", refusal_reason="no-seeds",
                        roi=self._roi, messages=[RT.MARKERS_ALL_TOUCHING])
                    self.message.emit(RT.MARKERS_ALL_TOUCHING)
                    print("[RESHAPE] refused before the engine - every "
                          "marker only touches the object")
            if result is None:
                result = self._run_engine(keep_seed_ids)
            if (n_left_out and keep_seed_ids
                    and getattr(result, "status", "") != "cancelled"):
                # Some were kept and no list was on screen to show the
                # rest, so the cut says so first.  In front: on a refusal
                # the LAST message is the reason, and this stays a remark
                # beside it.  (None kept is refused above, in one line.)
                from napari_epicycle.annotator import _reshape_text as RT

                result.messages.insert(0, RT.MARKERS_LEFT_OUT.format(
                    n=n_left_out, s=RT.plural(n_left_out)))
        except Exception as exc:  # noqa: BLE001 - never strand a locked surface
            self.error.emit("{}\n{}".format(exc, traceback.format_exc()))
            return
        self.finished.emit(result)

    def _run_engine(self, keep_seed_ids):
        """The one engine call, with this worker's inputs."""
        return reshape_region(
            self._labels,
            self._images,
            self._spec,
            target_id=self._target_id,
            roi=self._roi,
            seed_layer=self._seed_layer,
            spacing=self._spacing,
            margin=self._margin,
            keep_seed_ids=keep_seed_ids,
            cancel_event=self._cancel_event,
            log=lambda text: self.message.emit(str(text)),
            progress=lambda stage, i, n: self.progress.emit(
                stage, int(i), int(n)
            ),
            keep_intermediates=self._keep_intermediates,
        )


class SeedFindWorker(QObject):
    """Run one :func:`find_seeds` call on a QThread ("Find markers").

    The same shape as :class:`ReshapeWorker`, and thin for the same
    reason: the marker stage runs six detectors over a preprocessed
    crop, which is seconds of CLAHE and gaussian work, so it runs on a worker
    (heavy work never runs on the UI thread).

    Signals
    -------
    finished(report)
        The :class:`~packages.epicycle_annotator.reshape.SeedReport`.  A
        refusal (no markers) arrives HERE with ``status='refused'`` -- a
        refusal is an answer, not an error.
    message(text)
        Every plain-language note, live.
    error(message)
        Emitted instead of ``finished`` if the engine raises.

    **Cancel is the same honest thing it is for the solve**: the seed
    stage is one uninterruptible call, so ``cancel()`` means "abandon
    the result".  A cancel that lands before ``run()`` starts skips the
    work; one that lands during it is caught by the caller's launch
    token, which discards the answer.
    """

    message = Signal(str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, labels, images, spec, *, target_id=None, roi=None,
                 seed_layer=None, spacing=(1, 1, 1), margin=4,
                 cancel_event=None, touching_rule=False):
        super().__init__()
        #: Markers from a label mask: measure every marker against the
        #: object and fill ``report.contacts`` before ``finished``.
        self._touching_rule = bool(touching_rule)
        self._labels = labels
        self._images = dict(images or {})
        self._spec = spec
        self._target_id = target_id
        self._roi = roi
        self._seed_layer = seed_layer
        self._spacing = tuple(spacing)
        self._margin = margin
        self._cancel_event = cancel_event or threading.Event()

    @property
    def cancel_event(self) -> threading.Event:
        return self._cancel_event

    def cancel(self) -> None:
        """Abandon the result (see the class docstring)."""
        self._cancel_event.set()

    def run(self) -> None:
        """Execute the marker search (the QThread's ``started`` slot)."""
        if self._cancel_event.is_set():
            return
        try:
            report = find_seeds(
                self._labels,
                self._images,
                self._spec,
                target_id=self._target_id,
                roi=self._roi,
                seed_layer=self._seed_layer,
                spacing=self._spacing,
                margin=self._margin,
            )
            if self._touching_rule and getattr(report, "ok", False):
                t0 = _time.perf_counter()
                report.contacts = touching_verdicts(
                    self._labels, self._seed_layer, self._target_id,
                    report.roi, margin=self._margin,
                    seeds_in_region=report.seeds,
                    cancel_event=self._cancel_event)
                print(_touching_line(
                    report.contacts, (_time.perf_counter() - t0) * 1000.0))
        except Exception as exc:  # noqa: BLE001 - never strand a locked surface
            self.error.emit("{}\n{}".format(exc, traceback.format_exc()))
            return
        for text in list(getattr(report, "messages", []) or []):
            self.message.emit(str(text))
        self.finished.emit(report)


class GuideLoadWorker(QObject):
    """Read + validate a trusted guide mask on a QThread.

    A full-resolution guide is hundreds of MB; reading and majority-
    downsampling it on the main thread freezes napari for seconds, so it
    runs on a worker.  The array
    comes back through ``finished``; the caller adds the napari layer on the
    main thread, which is the only part that has to be there.

    A REFUSAL (wrong size, not a label mask, over the byte budget) arrives on
    ``refused`` with a plain-language sentence -- it is an answer, not a
    crash, and the caller shows it verbatim.
    """

    finished = Signal(object, object)   # (array, notes)
    refused = Signal(str)
    error = Signal(str)

    def __init__(self, path, volume_shape, *, downsample_factor=1):
        super().__init__()
        self._path = str(path)
        self._volume_shape = tuple(volume_shape)
        self._downsample_factor = int(downsample_factor)

    def run(self) -> None:
        from napari_epicycle.annotator import reshape_guide as RG

        try:
            guide = RG.load_guide_array(self._path)
            guide, notes = RG.validate_and_match_guide(
                guide, self._volume_shape,
                downsample_factor=self._downsample_factor,
            )
        except RG.GuideRefused as exc:
            self.refused.emit(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - never strand a busy bar
            self.error.emit(
                "{}".format(exc) + chr(10) + traceback.format_exc()
            )
            return
        self.finished.emit(guide, list(notes))


@dataclass
class RefineOutcome:
    """What a :class:`RefineWorker` hands back.

    ``result`` is whatever the pure function returned -- a
    :class:`~packages.epicycle_annotator.reshape.settle.SettleResult` for
    Settle, a ``(result_roi, n_modified, message)`` tuple for the contour --
    and ``kind`` says which, so the finish handler branches on the job it
    launched rather than on the shape of the answer.

    A **refusal arrives here**, not on ``error``: a refusal is an answer.
    """

    result: object = None
    kind: str = ""
    elapsed_s: float = 0.0
    messages: list = field(default_factory=list)


class RefineWorker(QObject):
    """Run one Refine method on a QThread.

    Shaped exactly like :class:`ReshapeWorker` and :class:`SeedFindWorker`,
    and thin for the same reason: ``fn`` is already bound to plain arrays, so
    this class imports nothing from the engine, is synchronous, and is
    directly callable in a test.

    **Used by the contour and by Settle only.**  The other five methods are
    measured under 100 ms; threading them would buy nothing and cost five
    more stale-guard surfaces.

    **Cancel is the same honest thing it is for the solve.**  ``random_walker``
    and ``diffuse_labels3D`` cannot be interrupted once entered, so
    :meth:`cancel` means "abandon the result": the panel enters its
    *Stopping...* state and stays there until the thread actually exits, and
    the caller's dispatch token throws the answer away when it lands.

    Signals
    -------
    progress(stage, i, n)
        ``(stage, i, n)``; the 2D sampler drives this in B2 and a 3D run
        leaves the bar in marquee.  Use :func:`stage_text` for the string.
    message(text)
        Every plain-language note, live.
    finished(outcome)
        One :class:`RefineOutcome`.  A refusal arrives HERE.
    error(message)
        Emitted instead of ``finished`` if the function raises.
    """

    progress = Signal(object, int, int)
    message = Signal(str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, fn, kwargs=None, *, kind="", cancel_event=None,
                 wants_progress=False, wants_log=False):
        super().__init__()
        self._fn = fn
        self._kwargs = dict(kwargs or {})
        self._kind = str(kind or "")
        self._wants_progress = bool(wants_progress)
        self._wants_log = bool(wants_log)
        self._cancel_event = cancel_event or threading.Event()

    @property
    def cancel_event(self) -> threading.Event:
        return self._cancel_event

    @property
    def kind(self) -> str:
        return self._kind

    def cancel(self) -> None:
        """Request cancellation (see the class docstring)."""
        self._cancel_event.set()

    def run(self) -> None:
        """Execute the method (the QThread's ``started`` slot).

        Synchronous: returns after emitting either ``finished`` or ``error``.
        A cancel that lands before this starts skips the work entirely; one
        that lands during it is caught by the caller's launch token.
        """
        if self._cancel_event.is_set():
            # Recipes audit H2: a cancel that lands BEFORE this starts used
            # to return without a word, and nothing quits the thread -- so
            # the owner's `_on_refine_thread_finished` backstop never ran and
            # the page held "Stopping..." with the lock on, for ever.  An
            # answer with no result is emitted instead; the finish handler's
            # stopping branch routes it to the Stop path.
            self.finished.emit(RefineOutcome(result=None, kind=self._kind))
            return
        call = dict(self._kwargs)
        if self._wants_progress:
            call["progress"] = lambda stage, i, n: self.progress.emit(
                stage, int(i), int(n)
            )
        if self._wants_log:
            call["log"] = lambda text: self.message.emit(str(text))
        t0 = _time.perf_counter()
        try:
            result = self._fn(**call)
        except Exception as exc:  # noqa: BLE001 - never strand a locked surface
            self.error.emit(
                "{}".format(exc) + chr(10) + traceback.format_exc())
            return
        notes = [str(t) for t in (getattr(result, "messages", None) or [])]
        for text in notes:
            self.message.emit(text)
        self.finished.emit(RefineOutcome(
            result=result, kind=self._kind,
            elapsed_s=float(_time.perf_counter() - t0),
            messages=notes,
        ))
