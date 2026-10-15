"""The Reshape QThread worker -- thin plumbing, tested as plumbing.

The worker carries no business logic (the engine is tested in
``test_reshape_engine.py``), so what matters here is that it forwards every
argument, emits the right signal for each outcome, and never lets an exception
escape into a locked Qt surface.

``run()`` is synchronous and directly callable, so none of this needs a
QThread -- only a QApplication-free ``Signal``, which qtpy provides on a plain
``QObject``.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from packages.epicycle_annotator.reshape import ReshapeSpec
from packages.epicycle_annotator.reshape.engine import STAGES

from .reshape_fixtures import two_columns

pytest.importorskip("qtpy")

from napari_epicycle.annotator import reshape_worker as RW  # noqa: E402


class _Spy:
    """Collects a signal's payloads without needing an event loop."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args if len(args) != 1 else args[0])


def _wire(worker):
    spies = {
        "progress": _Spy(),
        "message": _Spy(),
        "finished": _Spy(),
        "error": _Spy(),
    }
    worker.progress.connect(spies["progress"])
    worker.message.connect(spies["message"])
    worker.finished.connect(spies["finished"])
    worker.error.connect(spies["error"])
    return spies


def _guided_case(contrast=0.30):
    labels, guide, membrane, _region = two_columns(contrast)
    spec = ReshapeSpec(target="label", seeds="labels", walls="membrane",
                       params={"wall_channel": "membrane"})
    worker = RW.ReshapeWorker(
        labels, {"membrane": membrane}, spec,
        target_id=7, seed_layer=guide, spacing=(1.0, 1.0, 1.0),
    )
    return worker, labels, guide, membrane


# --------------------------------------------------------------------------- #
# the happy path
# --------------------------------------------------------------------------- #
def test_a_successful_run_emits_finished_with_the_result_and_no_error():
    worker, *_ = _guided_case()
    spies = _wire(worker)

    worker.run()

    assert spies["error"].calls == []
    assert len(spies["finished"].calls) == 1
    result = spies["finished"].calls[0]
    assert result.status == "ok"
    assert result.n_pieces >= 2, "a walled two-column cell must come apart"


def test_progress_reports_every_stage_in_order_with_a_total():
    worker, *_ = _guided_case()
    spies = _wire(worker)

    worker.run()

    stages = [call[0] for call in spies["progress"].calls]
    assert stages, "the worker must report progress"
    # Reported stages are a prefix-preserving subsequence of the fixed order.
    assert stages == [s for s in STAGES if s in stages]
    indices = [call[1] for call in spies["progress"].calls]
    totals = {call[2] for call in spies["progress"].calls}
    assert indices == sorted(indices), "progress must not go backwards"
    assert totals == {len(STAGES)}, "one fixed total, so a bar can size itself"


def test_engine_messages_arrive_live_and_also_land_in_the_result():
    """A guard that fires must be visible before the run ends."""
    worker, *_ = _guided_case()
    spies = _wire(worker)

    worker.run()

    result = spies["finished"].calls[0]
    assert list(spies["message"].calls) == list(result.messages)


# --------------------------------------------------------------------------- #
# a refusal is an ANSWER, not an error
# --------------------------------------------------------------------------- #
def test_zero_seeds_is_a_refused_RESULT_not_an_error_signal():
    labels, _guide, membrane, _region = two_columns(0.30)
    empty_guide = np.zeros_like(labels)
    spec = ReshapeSpec(target="label", seeds="labels", walls="membrane",
                       params={"wall_channel": "membrane"})
    worker = RW.ReshapeWorker(labels, {"membrane": membrane}, spec,
                              target_id=7, seed_layer=empty_guide)
    spies = _wire(worker)

    worker.run()

    assert spies["error"].calls == [], "a refusal must not look like a crash"
    result = spies["finished"].calls[0]
    assert result.status == "refused"
    assert result.messages, "a refusal must say why, in words"


# --------------------------------------------------------------------------- #
# cancellation
# --------------------------------------------------------------------------- #
def test_cancel_before_run_yields_a_cancelled_result_and_no_error():
    worker, *_ = _guided_case()
    spies = _wire(worker)

    worker.cancel()
    worker.run()

    assert spies["error"].calls == []
    assert spies["finished"].calls[0].status == "cancelled"


def test_cancel_event_is_shared_so_the_caller_can_hold_its_own():
    event = threading.Event()
    labels, guide, membrane, _ = two_columns(0.30)
    spec = ReshapeSpec(target="label", seeds="labels", walls="membrane",
                       params={"wall_channel": "membrane"})
    worker = RW.ReshapeWorker(labels, {"membrane": membrane}, spec,
                              target_id=7, seed_layer=guide,
                              cancel_event=event)

    assert worker.cancel_event is event
    worker.cancel()
    assert event.is_set()


# --------------------------------------------------------------------------- #
# never strand a locked surface
# --------------------------------------------------------------------------- #
def test_an_engine_exception_becomes_an_error_signal_not_a_raise(monkeypatch):
    worker, *_ = _guided_case()
    spies = _wire(worker)

    def _boom(*a, **k):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(RW, "reshape_region", _boom)

    worker.run()  # must NOT raise -- a raise here strands _reshape_running

    assert spies["finished"].calls == []
    assert len(spies["error"].calls) == 1
    assert "engine exploded" in spies["error"].calls[0]
    assert "Traceback" in spies["error"].calls[0], "keep the traceback"


def test_every_constructor_argument_reaches_the_engine(monkeypatch):
    seen = {}

    def _spy(labels, images, spec, **kwargs):
        seen["labels"] = labels
        seen["images"] = images
        seen["spec"] = spec
        seen.update(kwargs)

        class _R:
            status = "ok"
        return _R()

    monkeypatch.setattr(RW, "reshape_region", _spy)

    labels = np.zeros((4, 4, 4), np.int32)
    guide = np.zeros((4, 4, 4), np.int32)
    membrane = np.zeros((4, 4, 4), np.float32)
    spec = ReshapeSpec(target="label", seeds="labels", walls="membrane",
                       params={"wall_channel": "membrane"})
    roi = (slice(0, 2), slice(0, 2), slice(0, 2))
    worker = RW.ReshapeWorker(
        labels, {"membrane": membrane}, spec, target_id=7, roi=roi,
        seed_layer=guide, spacing=(5.0, 1.0, 1.0), margin=9,
        keep_seed_ids=[1, 2], keep_intermediates=True,
    )
    worker.run()

    assert seen["labels"] is labels
    assert seen["spec"] is spec
    assert seen["images"]["membrane"] is membrane
    assert seen["target_id"] == 7
    assert seen["roi"] == roi
    assert seen["seed_layer"] is guide
    assert seen["spacing"] == (5.0, 1.0, 1.0)
    assert seen["margin"] == 9
    assert seen["keep_seed_ids"] == [1, 2]
    assert seen["keep_intermediates"] is True
    assert seen["cancel_event"] is worker.cancel_event


# --------------------------------------------------------------------------- #
# the busy bar's words
# --------------------------------------------------------------------------- #
def test_every_engine_stage_has_a_plain_language_sentence():
    for stage in STAGES:
        text = RW.stage_text(stage)
        assert text != stage, "stage {!r} has no user-facing wording".format(stage)
        assert text[0].isupper() and text.endswith(("...", ".")), text


def test_an_unknown_stage_degrades_to_its_own_name_rather_than_raising():
    assert RW.stage_text("something-new") == "something-new"


def test_the_worker_module_stays_ascii_for_the_windows_console():
    """Diagnostics must survive a cp1252 console (no arrows, no box glyphs)."""
    import io
    import pathlib

    path = pathlib.Path(RW.__file__)
    text = io.open(path, encoding="utf-8").read()
    for lineno, line in enumerate(text.splitlines(), 1):
        try:
            line.encode("cp1252")
        except UnicodeEncodeError as exc:  # pragma: no cover - the failure msg
            pytest.fail(
                "reshape_worker.py:{} is not cp1252-safe: {}".format(lineno, exc)
            )


# --------------------------------------------------------------------------- #
# SeedFindWorker -- the marker stage on its own thread (Brick 2a)
# --------------------------------------------------------------------------- #
def _find_case():
    labels, guide, membrane, _region = two_columns(0.30)
    spec = ReshapeSpec(target="label", seeds="labels", walls="membrane",
                       params={"wall_channel": "membrane"})
    worker = RW.SeedFindWorker(
        labels, {"membrane": membrane}, spec,
        target_id=7, seed_layer=guide, spacing=(1.0, 1.0, 1.0),
    )
    return worker, labels, guide, membrane


def _wire_find(worker):
    spies = {"message": _Spy(), "finished": _Spy(), "error": _Spy()}
    worker.message.connect(spies["message"])
    worker.finished.connect(spies["finished"])
    worker.error.connect(spies["error"])
    return spies


def test_the_find_worker_emits_a_report_and_no_error():
    worker, *_ = _find_case()
    spies = _wire_find(worker)

    worker.run()

    assert spies["error"].calls == []
    assert len(spies["finished"].calls) == 1
    report = spies["finished"].calls[0]
    assert report.status == "ok"
    assert report.n_in_region == 2
    assert [m["id"] for m in report.markers] == [1, 2]


def test_zero_markers_is_a_refused_REPORT_not_an_error_signal():
    labels, _guide, membrane, _region = two_columns(0.30)
    spec = ReshapeSpec(target="label", seeds="labels", walls="membrane",
                       params={"wall_channel": "membrane"})
    worker = RW.SeedFindWorker(labels, {"membrane": membrane}, spec,
                               target_id=7, seed_layer=np.zeros_like(labels))
    spies = _wire_find(worker)

    worker.run()

    assert spies["error"].calls == []
    report = spies["finished"].calls[0]
    assert report.status == "refused"
    assert report.messages, "a refusal must say why, in words"


def test_the_find_workers_messages_match_the_report():
    labels, guide, membrane, _region = two_columns(0.30)
    guide = guide.copy()
    guide[0, 4, 6] = 44                      # a marker outside the cell
    spec = ReshapeSpec(target="label", seeds="labels", walls="membrane",
                       params={"wall_channel": "membrane"})
    worker = RW.SeedFindWorker(labels, {"membrane": membrane}, spec,
                               target_id=7, seed_layer=guide)
    spies = _wire_find(worker)

    worker.run()

    report = spies["finished"].calls[0]
    assert list(spies["message"].calls) == list(report.messages)
    assert spies["message"].calls, "the dropped marker must be said out loud"


def test_a_cancel_before_run_does_the_work_of_nobody():
    worker, *_ = _find_case()
    spies = _wire_find(worker)

    worker.cancel()
    worker.run()

    assert spies["finished"].calls == []
    assert spies["error"].calls == []


def test_the_find_workers_cancel_event_is_shareable():
    event = threading.Event()
    labels, guide, membrane, _ = two_columns(0.30)
    spec = ReshapeSpec(target="label", seeds="labels", walls="membrane",
                       params={"wall_channel": "membrane"})
    worker = RW.SeedFindWorker(labels, {"membrane": membrane}, spec,
                               target_id=7, seed_layer=guide,
                               cancel_event=event)
    assert worker.cancel_event is event
    worker.cancel()
    assert event.is_set()


def test_an_exception_in_the_find_becomes_an_error_signal(monkeypatch):
    worker, *_ = _find_case()
    spies = _wire_find(worker)

    def _boom(*a, **k):
        raise RuntimeError("marker search exploded")

    monkeypatch.setattr(RW, "find_seeds", _boom)

    worker.run()   # must NOT raise -- a raise strands _seed_find_running

    assert spies["finished"].calls == []
    assert len(spies["error"].calls) == 1
    assert "marker search exploded" in spies["error"].calls[0]
    assert "Traceback" in spies["error"].calls[0]


def test_every_find_constructor_argument_reaches_the_engine(monkeypatch):
    seen = {}

    def _spy(labels, images, spec, **kwargs):
        seen["labels"] = labels
        seen["images"] = images
        seen["spec"] = spec
        seen.update(kwargs)

        class _R:
            status = "ok"
            messages = []
        return _R()

    monkeypatch.setattr(RW, "find_seeds", _spy)

    labels = np.zeros((4, 4, 4), np.int32)
    guide = np.zeros((4, 4, 4), np.int32)
    membrane = np.zeros((4, 4, 4), np.float32)
    spec = ReshapeSpec(target="label", seeds="labels", walls="membrane",
                       params={"wall_channel": "membrane"})
    roi = (slice(0, 2), slice(0, 2), slice(0, 2))
    worker = RW.SeedFindWorker(
        labels, {"membrane": membrane}, spec, target_id=7, roi=roi,
        seed_layer=guide, spacing=(5.0, 1.0, 1.0), margin=9,
    )
    worker.run()

    assert seen["labels"] is labels
    assert seen["spec"] is spec
    assert seen["images"]["membrane"] is membrane
    assert seen["target_id"] == 7
    assert seen["roi"] == roi
    assert seen["seed_layer"] is guide
    assert seen["spacing"] == (5.0, 1.0, 1.0)
    assert seen["margin"] == 9


def test_the_find_worker_passes_the_PREVIEWS_roi_not_a_fresh_one():
    """The detectors normalise over the CROP and number by enumeration
    order, so a different crop gives different markers with different ids
    -- and the ticks would then select the wrong ones."""
    labels, guide, membrane, _ = two_columns(0.30)
    spec = ReshapeSpec(target="label", seeds="labels", walls="membrane",
                       params={"wall_channel": "membrane"})
    roi = (slice(0, 12), slice(5, 35), slice(5, 28))
    worker = RW.SeedFindWorker(labels, {"membrane": membrane}, spec,
                               target_id=7, roi=roi, seed_layer=guide)
    spies = _wire_find(worker)
    worker.run()
    assert spies["finished"].calls[0].roi == roi


def test_the_marker_worker_reports_no_stages_and_claims_none():
    """`find_seeds` has no progress callback, so there is no marker stage to
    word.  A STAGE_TEXT entry for one would be a sentence nothing shows."""
    assert "markers" not in RW.STAGE_TEXT
    # The table is SHARED with the Refine tab: one busy
    # strip shows whichever job is running, and a second table would be a
    # second thing to keep in step.  The claim being made here is about the
    # ENGINE's stages, so the Refine-only entries are named and excluded --
    # naming them is the point, so adding one silently still fails.
    _REFINE_ONLY = {"prepare", "elevation", "band", "contour", "rebuild",
                    "settle", "stopping"}
    # The Add rebuild's own stages, named the same way: its
    # worker shares the table too, and a stray key must still fail here.
    _ADD_ONLY = {"add_box", "add_markers", "add_cut", "add_find",
                 "add_smooth", "add_info"}
    assert (set(RW.STAGE_TEXT) - {"done"} - _REFINE_ONLY - _ADD_ONLY
            == set(STAGES))
    assert _ADD_ONLY <= set(RW.STAGE_TEXT)
