"""Model Refinement fix pass -- the controller-side corrections (F1-F11)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from napari_epicycle.fusion_reconstruction import _paths
from napari_epicycle.fusion_reconstruction import _step3_mr_text as M
from tests.fusion_reconstruction._refine_launch_harness import (
    finish, make_launch_controller, run_timers,
)


def _controller(tmp_path, monkeypatch, **kw):
    cards, status = [], []
    label = SimpleNamespace(text="", setText=lambda t: setattr(
        label, "text", t))
    ui = SimpleNamespace(
        set_refine_card=lambda title="", **k: cards.append((title, k)),
        show_refine_status=lambda text, kind="neutral": status.append(
            (text, kind)),
        refinement_output_label=label,
    )
    for p in ("patchA", "patchB"):
        (_paths.refinement_pairs_root(tmp_path) / p).mkdir(
            parents=True, exist_ok=True)
    ctrl, h = make_launch_controller(tmp_path, monkeypatch, ui=ui, **kw)
    ctrl._append_log = lambda _m: None
    return ctrl, h, cards, status


def test_f1_late_cancel_does_not_start_the_axial_pass(tmp_path, monkeypatch):
    ctrl, h, cards, status = _controller(
        tmp_path, monkeypatch, mode="Fine-tune Both", name="b")
    ctrl._launch_refinement()
    assert len(h.workers) == 1
    # Cancel lands after the worker's last epoch callback
    ctrl._refinement_cancel_requested = True
    finish(ctrl, h.workers[0])
    run_timers(h)
    assert len(h.workers) == 1                    # no axial worker
    assert ctrl._refinement_queue == []
    assert ctrl._refinement_running is False
    assert status[-1] == (M.MR_STATUS_CANCELLED, "info")


def test_f2_last_pass_without_pairs_still_shows_the_card(
        tmp_path, monkeypatch):
    ctrl, h, cards, status = _controller(
        tmp_path, monkeypatch, mode="Fine-tune Both", name="b",
        no_pairs=("Fine-tune Axial",))
    ctrl._launch_refinement()
    finish(ctrl, h.workers[0])
    run_timers(h)
    assert len(cards) == 1
    assert status[-1][0].startswith("Complete: ")
    assert ctrl._refinement_running is False


def test_f3_browse_is_refused_while_a_run_is_live(tmp_path, monkeypatch):
    from napari_epicycle.fusion_reconstruction import controller as cmod
    ctrl, h, cards, status = _controller(tmp_path, monkeypatch)
    called = []
    monkeypatch.setattr(
        cmod.QFileDialog, "getExistingDirectory",
        staticmethod(lambda *a, **k: called.append(1) or ""))
    for flag in ("_refinement_running", "_extraction_running"):
        ctrl._refinement_running = False
        ctrl._extraction_running = False
        setattr(ctrl, flag, True)
        ctrl._browse_workdir()
        assert status[-1] == (M.MR_STATUS_BUSY, "warn")
    assert called == []


def test_m6_bar_text_names_the_pass_then_done_then_blank():
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController as Ctl,
    )
    calls = []
    bar = SimpleNamespace(
        setFormat=lambda f: calls.append(("fmt", f)),
        setTextVisible=lambda v: calls.append(("vis", v)))
    c = Ctl.__new__(Ctl)
    c.ui = SimpleNamespace(refinement_progress=bar)
    c._set_refinement_bar_text("axial")
    assert ("fmt", "axial %v / %m") in calls
    c._set_refinement_bar_text(None, done=True)
    assert ("fmt", M.MR_PROGRESS_DONE) in calls
    c._set_refinement_bar_text(None)
    assert calls[-1] == ("vis", False)


def test_n2_error_collapses_the_loss_curves(tmp_path, monkeypatch):
    ctrl, h, cards, status = _controller(tmp_path, monkeypatch)
    closed = []
    ctrl.ui.refine_loss_collapsible = SimpleNamespace(
        collapse=lambda animate=False: closed.append(animate))
    ctrl._on_refinement_error("CUDA out of memory\nstack")
    assert closed == [False]
    assert status[-1][1] == "error"
