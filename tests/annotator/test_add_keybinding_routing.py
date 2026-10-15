"""The labels key while Add runs.

A new file because ``test_keybinding_isolation_aware.py`` is known-stale.
While Add runs or a candidate exists, the labels key returns the default
targets plus the candidate, the ball layer and their side-view copies; an
image-channel key still falls through to the default.
"""

from __future__ import annotations

from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
from napari_epicycle.annotator.native_multiview_widget import (
    NativeMultiviewAnnotator,
)


class _L:
    def __init__(self, name):
        self.name = name


class _Sub:
    def __init__(self, copies):
        self.copies = copies

    def get_subviewer_layer(self, name, orient):
        return self.copies.get((name, orient))


def _ann(monkeypatch, *, running=True, cand=True, balls=True):
    a = Curation3DAnnotator.__new__(Curation3DAnnotator)
    a._active_labels_names = ["L"]
    a._add_running = running
    a._add_candidate_layer = _L("_epicycle_add_candidate") if cand else None
    a._add_marker_layer = _L("_epicycle_add_markers") if balls else None
    a._refine_layer = None
    copies = {}
    for layer in (a._add_candidate_layer, a._add_marker_layer):
        if layer is not None:
            for o in ("XY", "XZ", "YZ"):
                copies[(layer.name, o)] = _L(f"{layer.name}_{o}")
    a.subviewer_manager = _Sub(copies)
    default = [_L("L"), _L("L_XY")]
    monkeypatch.setattr(NativeMultiviewAnnotator,
                        "_get_toggle_targets_for_name",
                        lambda self, name: list(default))
    return a


def test_the_labels_key_adds_the_add_artefacts(monkeypatch):
    a = _ann(monkeypatch)
    got = [t.name for t in a._get_toggle_targets_for_name("L")]
    assert got[:2] == ["L", "L_XY"]
    for n in ("_epicycle_add_candidate", "_epicycle_add_markers"):
        assert n in got
        for o in ("XY", "XZ", "YZ"):
            assert f"{n}_{o}" in got


def test_an_image_key_falls_through(monkeypatch):
    a = _ann(monkeypatch)
    got = [t.name for t in a._get_toggle_targets_for_name("nuclei")]
    assert got == ["L", "L_XY"]


def test_a_candidate_alone_routes_after_the_session(monkeypatch):
    a = _ann(monkeypatch, running=False, balls=False)
    got = [t.name for t in a._get_toggle_targets_for_name("L")]
    assert "_epicycle_add_candidate" in got


def test_no_add_artefacts_no_extra_targets(monkeypatch):
    a = _ann(monkeypatch, running=False, cand=False, balls=False)
    assert a._add_toggle_targets() == []
