"""A1 follow-up: a QC Settings OK reaches the Add page at once.

The amber ADD_NOTE_DEFAULT_SPACING tells the user to set Spacing under QC
Features -> Settings...; doing exactly that must take the note down without
any other change on the page (the driver found it stayed up until the next
box or channel change).  The spacing also feeds the box info (the dark
threshold's blur is in microns), so a live box re-measures.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from .add_harness import FakeThread, Harness, install_fakes


@pytest.fixture
def h(monkeypatch):
    install_fakes(monkeypatch)
    hh = Harness(np.zeros((12, 20, 20), np.int32), images={},
                 spacing=(1.0, 1.0, 1.0))
    hh.adv.preset = "cells"
    return hh


def test_the_note_follows_the_spacing_with_no_other_change(h):
    h._add_push_context(sources=False)
    assert h.adv.last("default_spacing") is True
    h.ctx["spacing"] = (0.5, 0.517, 0.517)       # QC Settings -> OK
    n = len(h.adv.pushes)
    h._add_spacing_changed()
    assert len(h.adv.pushes) > n
    assert h.adv.last("default_spacing") is False
    h.ctx["spacing"] = (1.0, 1.0, 1.0)
    h._add_spacing_changed()
    assert h.adv.last("default_spacing") is True


def test_a_live_box_forgets_its_dark_threshold_and_re_measures(h):
    h.place_box((slice(2, 10), slice(2, 18), slice(2, 18)))
    h._add_dark_cache = {("P", 1, ()): {"t": 1}}
    n = len(FakeThread.instances)
    h.ctx["spacing"] = (0.5, 0.517, 0.517)
    h._add_spacing_changed()
    assert h._add_dark_cache == {}, "the blur is in microns"
    assert len(FakeThread.instances) == n + 1, "box info re-requested"


def test_no_box_requests_nothing(h):
    n = len(FakeThread.instances)
    h._add_spacing_changed()
    assert len(FakeThread.instances) == n


def test_the_public_door_never_raises():
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    fake = types.SimpleNamespace()

    def boom():
        raise RuntimeError("x")

    fake._add_spacing_changed = boom
    Curation3DAnnotator.notify_spacing_changed(fake)     # swallowed, logged


# --------------------------------------------------------------------------- #
# the controller: the QC-settings OK calls it
# --------------------------------------------------------------------------- #
def test_the_qc_settings_accept_notifies_the_annotator(monkeypatch):
    from napari_epicycle.fusion_reconstruction import controller as CTRL
    from napari_epicycle.fusion_reconstruction import qc_extraction_dialog

    new = {"erosion_iters": 1, "max_radius": 2, "composite_method": "zscore",
           "spacing": (0.5, 0.517, 0.517), "pca_features": []}

    class _Dlg:
        Accepted = 1

        def __init__(self, *a, **k):
            self.spacing_auto_btn = types.SimpleNamespace(
                clicked=types.SimpleNamespace(connect=lambda *_a: None))

        def exec_(self):
            return 1

        def get_settings(self):
            return dict(new)

    monkeypatch.setattr(qc_extraction_dialog, "QCExtractionDialog", _Dlg)
    told = []
    widget = types.SimpleNamespace(
        notify_spacing_changed=lambda: told.append(True))
    fake = types.SimpleNamespace(
        ui=types.SimpleNamespace(), _qc_settings={"spacing": None},
        viewer=None, _qc_features_df=None,
        _append_log=lambda *_a: None,
        _detect_available_channels=lambda: [],
        _check_qc_extraction_staleness=lambda: None,
        _maybe_show_qc_staleness_notification=lambda: None,
        _get_curation_annotator_widget=lambda: widget)
    K = CTRL.FusionReconstructionController
    fake._notify_spacing_changed = K._notify_spacing_changed.__get__(fake)
    K._open_qc_settings(fake)
    assert fake._qc_settings["spacing"] == (0.5, 0.517, 0.517)
    assert told == [True]


def test_the_controller_notify_never_raises():
    from napari_epicycle.fusion_reconstruction import controller as CTRL

    def boom():
        raise RuntimeError("no annotator")

    fake = types.SimpleNamespace(_get_curation_annotator_widget=boom)
    CTRL.FusionReconstructionController._notify_spacing_changed(fake)
    fake = types.SimpleNamespace(_get_curation_annotator_widget=lambda: None)
    CTRL.FusionReconstructionController._notify_spacing_changed(fake)
