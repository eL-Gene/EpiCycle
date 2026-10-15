"""Model Refinement look, Brick B2 -- every refusal and failure is a sentence
on the panel's foot, from ``_step3_mr_text``, with the right kind.

The REAL ``_launch_refinement`` / ``_launch_next_refinement`` /
``_on_refinement_*`` run through ``_refine_launch_harness`` (only the worker,
its thread and the timer are fake); ``ui.show_refine_status`` is a recorder.
"""
from __future__ import annotations

import shutil
from types import SimpleNamespace

from napari_epicycle.fusion_reconstruction import _paths
from napari_epicycle.fusion_reconstruction import _step3_mr_text as M
from tests.fusion_reconstruction._refine_launch_harness import (
    make_launch_controller,
)


def _controller(tmp_path, monkeypatch, **kw):
    calls = []
    ui = SimpleNamespace(
        show_refine_status=lambda text, kind="neutral": calls.append(
            (text, kind)),
    )
    ctrl, h = make_launch_controller(tmp_path, monkeypatch, ui=ui, **kw)
    ctrl._append_log = lambda _m: None
    return ctrl, h, calls


def test_busy_is_a_warn_sentence(tmp_path, monkeypatch):
    ctrl, h, calls = _controller(tmp_path, monkeypatch)
    ctrl._prediction_running = True
    ctrl._launch_refinement()
    assert calls == [(M.MR_STATUS_BUSY, "warn")]
    assert h.workers == []


def test_no_pairs_folder_is_a_warn_sentence(tmp_path, monkeypatch):
    ctrl, h, calls = _controller(tmp_path, monkeypatch)
    shutil.rmtree(_paths.refinement_pairs_root(tmp_path))
    ctrl._launch_refinement()
    assert calls == [(M.MR_STATUS_NOTHING, "neutral")]


def test_every_pass_empty_is_a_warn_sentence(tmp_path, monkeypatch):
    ctrl, h, calls = _controller(
        tmp_path, monkeypatch, mode="Fine-tune Both",
        no_pairs=("Fine-tune Lateral", "Fine-tune Axial"))
    ctrl._launch_refinement()
    assert (M.MR_STATUS_NOTHING, "neutral") in calls
    assert h.workers == []
    assert not ctrl._refinement_running


def test_name_taken_is_a_warn_sentence(tmp_path, monkeypatch):
    ctrl, h, calls = _controller(tmp_path, monkeypatch, name="wound_v1")
    ctrl._check_refinement_collision = lambda mode: "cancel"
    ctrl._launch_refinement()
    assert calls == [(M.MR_STATUS_NAME_TAKEN.format(name="wound_v1"),
                      "warn")]
    assert h.workers == []


def test_a_bucket_from_scratch_is_a_warn_sentence(tmp_path, monkeypatch):
    ctrl, h, calls = _controller(tmp_path, monkeypatch)
    ctrl.ui.refine_init_combo = SimpleNamespace(
        currentText=lambda: "Import model")
    ctrl._resolve_init_mode_for_bucket = lambda b: (
        ("custom", "C:/m/lat") if b == "lateral" else ("scratch", None))
    ctrl._launch_refinement()
    assert calls == [(M.MR_STATUS_NO_SOURCE.format(bucket="axial"), "warn")]
    assert h.workers == []


def test_a_launch_failure_is_an_error_with_the_first_line(
        tmp_path, monkeypatch):
    ctrl, h, calls = _controller(tmp_path, monkeypatch)

    def _boom(_mode):
        raise RuntimeError("disk gone\nsecond line")

    ctrl._collect_training_pairs = _boom
    ctrl._launch_refinement()
    assert calls[-1] == (
        M.MR_STATUS_LAUNCH_FAILED.format(first="disk gone"), "error")
    assert not ctrl._refinement_running


def test_a_training_error_names_the_pass_the_epoch_and_the_first_line(
        tmp_path, monkeypatch):
    ctrl, h, calls = _controller(tmp_path, monkeypatch,
                                 mode="Fine-tune Lateral")
    ctrl._launch_refinement()
    assert len(h.workers) == 1
    ctrl._on_refinement_epoch(7, 100, 0.5, None)
    ctrl._on_refinement_error(
        "RuntimeError: out of memory\nTraceback (most recent call last):\n x")
    assert calls[-1] == (M.MR_STATUS_FAILED.format(
        bucket="lateral", e=7, first="RuntimeError: out of memory"),
        "error")
    assert not ctrl._refinement_running


def test_running_and_cancelled_lines_are_info(tmp_path, monkeypatch):
    ctrl, h, calls = _controller(tmp_path, monkeypatch,
                                 mode="Fine-tune Lateral")
    ctrl._launch_refinement()
    assert calls[-1][1] == "info"
    assert calls[-1][0].startswith("Training lateral… [")
    ctrl._on_refinement_epoch(3, 100, 0.25, None)
    text, kind = calls[-1]
    assert kind == "info"
    assert text.startswith("Training lateral… (loss 0.25)")
    ctrl._on_refinement_cancelled()
    assert calls[-1] == (M.MR_STATUS_CANCELLED, "info")


def test_the_gap_cancel_is_an_info_sentence(tmp_path, monkeypatch):
    ctrl, h, calls = _controller(tmp_path, monkeypatch)
    ctrl._refinement_running = True
    ctrl._refinement_worker = None
    ctrl._cancel_refinement()
    assert calls[-1] == (M.MR_STATUS_CANCELLED, "info")


def test_the_live_cancel_says_so_and_greys_cancel(tmp_path, monkeypatch):
    ctrl, h, calls = _controller(tmp_path, monkeypatch)
    cancelled = []
    ctrl._refinement_worker = SimpleNamespace(
        request_cancel=lambda: cancelled.append(1))
    ctrl._refinement_running = True
    ctrl._cancel_refinement()
    assert cancelled and ctrl._refinement_cancel_requested is True
    assert calls[-1] == (M.MR_STATUS_CANCELLING, "info")
    ctrl.ui.refinement_cancel_btn.setEnabled.assert_called_with(False)
