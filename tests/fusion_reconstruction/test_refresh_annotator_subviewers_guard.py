"""A 3D annotator whose dock-close cleanup has run has no side views left, so
the controller must not realign or refresh it (the hidden dock stays
registered until the next launch)."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

DOCK = "EpiCycle 3D Curation Annotator"


def _controller(widget=None):
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    ctrl = FusionReconstructionController.__new__(FusionReconstructionController)
    docks = {} if widget is None else {DOCK: widget}
    ctrl.viewer = types.SimpleNamespace(
        window=types.SimpleNamespace(_dock_widgets=docks))
    return ctrl


def _widget(calls, **flags):
    return types.SimpleNamespace(
        _active_labels_names=["a", "b"],
        _active_labels_name="a",
        _clear_draw_snapshot=lambda: calls.append("snap"),
        _ensure_shared_data_aligned=lambda n: calls.append(("align", n)),
        _refresh_label_data_in_subviewers=lambda: calls.append("refresh"),
        **flags)


def test_live_annotator_is_realigned_and_refreshed():
    calls = []
    _controller(_widget(calls))._refresh_annotator_subviewers()
    assert calls == ["snap", ("align", "a"), ("align", "b"), "refresh"]


@pytest.mark.parametrize("flag", ["_dock_close_hook_fired", "_close_cleanup_done"])
def test_cleaned_up_annotator_is_left_alone(flag, capsys):
    calls = []
    _controller(_widget(calls, **{flag: True}))._refresh_annotator_subviewers()
    assert calls == []
    assert "annotator realign skipped" in capsys.readouterr().out


def test_no_dock_returns_quietly():
    _controller(None)._refresh_annotator_subviewers()


def test_revert_from_disk_replaces_data_but_skips_realign(monkeypatch, tmp_path):
    calls = []
    widget = _widget(calls, _close_cleanup_done=True)
    ctrl = _controller(widget)
    path = tmp_path / "labels.tif"
    path.write_bytes(b"x")
    new = np.ones((2, 2, 2), dtype=np.uint16)
    stub = types.ModuleType("tifffile")
    stub.imread = lambda p: new
    monkeypatch.setitem(sys.modules, "tifffile", stub)
    layer = types.SimpleNamespace(data=None, refresh=lambda: calls.append("lr"))
    ctrl._get_active_labels_path = lambda: str(path)
    ctrl._find_active_labels_layer = lambda: layer
    ctrl._append_log = lambda m: None
    ctrl._reload_current_patch_features = lambda: calls.append("features")
    assert ctrl._revert_labels_from_disk() is True
    assert layer.data is new
    assert "features" in calls
    assert "snap" not in calls and "refresh" not in calls
    assert not any(isinstance(c, tuple) for c in calls)
