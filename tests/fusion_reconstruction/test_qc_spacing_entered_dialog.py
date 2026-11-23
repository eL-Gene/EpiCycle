"""The QC Settings spacing mark against the REAL dialog (offscreen Qt).

The Qt-free tests in ``test_qc_spacing_entered.py`` hand the controller a
fake dialog; these open ``QCExtractionDialog`` itself, so the round trip
through its spin boxes (3 decimals, filled from ``spacing_values``) and the
Auto button the controller wires are what is measured:

* an untouched dialog gives back numbers that are not an edit, for the
  controller's factory settings and for a value the boxes round;
* a typed box is an edit;
* ``_open_qc_settings`` with the real dialog: an untouched OK leaves the
  spacing unknown, Auto from a scaled layer enters it, Auto with only
  unscaled layers enters nothing.
"""
from __future__ import annotations

import os
import types

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from napari_epicycle.fusion_reconstruction import controller as ctl_mod  # noqa: E402
from napari_epicycle.fusion_reconstruction import qc_extraction_dialog  # noqa: E402
from napari_epicycle.fusion_reconstruction.controller import (  # noqa: E402
    FusionReconstructionController as K,
)
from napari_epicycle.fusion_reconstruction.qc_features import (  # noqa: E402
    PCA_FEATURES,
)

_ONE = (1.0, 1.0, 1.0)


@pytest.fixture(scope="module")
def app():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _factory_settings() -> dict:
    """The QC settings a fresh controller holds (its ``__init__`` values)."""
    return {
        "spacing": None,
        "spacing_values": (1.0, 1.0, 1.0),
        "erosion_iters": 2,
        "max_radius": 5,
        "composite_method": "zscore",
        "cluster_algorithm": "leiden",
        "min_cluster_size": 15,
        "n_clusters": 5,
        "leiden_resolution": 1.0,
        "leiden_n_neighbors": 15,
        "n_cluster_components": 5,
        "pca_features": list(PCA_FEATURES),
    }


def _round_trip(settings, edit=None):
    d = qc_extraction_dialog.QCExtractionDialog(None, settings=settings)
    try:
        if edit is not None:
            edit(d)
        return d.get_settings()
    finally:
        d.deleteLater()


def _edited(before, after) -> bool:
    return ctl_mod._qc_spacing_edited(ctl_mod._qc_spacing_numbers(before),
                                      ctl_mod._qc_spacing_numbers(after))


def test_the_factory_settings_come_back_untouched(app):
    before = _factory_settings()
    after = _round_trip(before)
    assert after["spacing"] is None
    assert tuple(after["spacing_values"]) == _ONE
    assert _edited(before, after) is False


def test_a_value_the_boxes_round_is_not_an_edit(app):
    """A file's 0.10833 shows as 0.108 in a 3-decimal box: same number."""
    before = dict(_factory_settings(), spacing=(0.5, 0.10833, 0.10833),
                  spacing_values=(0.5, 0.10833, 0.10833))
    after = _round_trip(before)
    assert tuple(after["spacing_values"]) == (0.5, 0.108, 0.108)
    assert _edited(before, after) is False


def test_a_typed_box_is_an_edit(app):
    before = _factory_settings()
    after = _round_trip(before, edit=lambda d: d.spacing_z_sb.setValue(0.3))
    assert tuple(after["spacing_values"]) == (0.3, 1.0, 1.0)
    assert _edited(before, after) is True


# ---------------------------------------------------------------------------
# _open_qc_settings with the real dialog
# ---------------------------------------------------------------------------

def _layer(scale):
    return types.SimpleNamespace(scale=tuple(scale), ndim=len(scale))


def _controller(tmp_path, layers=()):
    c = object.__new__(K)
    c.ui = None                       # the dialog's parent
    c.viewer = types.SimpleNamespace(layers=list(layers))
    c._qc_features_df = None
    c._qc_spacing_entered = False
    c._qc_settings = _factory_settings()
    c._qc_view_patch_name = "p01.tif"
    c.working_dir = str(tmp_path)
    c._current_downsample_factor = lambda *_a, **_k: 1
    c._append_log = lambda *_a: None
    c._detect_available_channels = lambda: []
    c._qc_channel_roles = lambda *_a: {}
    c._match_picker_state = lambda *_a: ([], None, False)
    c._extract_spacing_lines = lambda: []
    c._notify_spacing_changed = lambda: None
    c._check_qc_extraction_staleness = lambda: None
    c._maybe_show_qc_staleness_notification = lambda: None
    return c


def _ok_after(monkeypatch, action=None):
    """The real dialog, whose modal loop is replaced by *action* and OK."""

    class _Real(qc_extraction_dialog.QCExtractionDialog):
        def exec_(self):
            if action is not None:
                action(self)
            return self.Accepted

    monkeypatch.setattr(qc_extraction_dialog, "QCExtractionDialog", _Real)


def test_an_untouched_ok_of_the_real_dialog_stays_unknown(
        app, monkeypatch, tmp_path):
    c = _controller(tmp_path)
    _ok_after(monkeypatch)
    K._open_qc_settings(c)
    assert c._qc_settings["spacing"] is None
    assert c._qc_spacing_entered is False
    assert c._on_annotator_reshape_context()["spacing_source"] == "unknown"


def test_auto_from_a_scaled_layer_enters_the_spacing(
        app, monkeypatch, tmp_path):
    c = _controller(tmp_path, layers=[_layer((0.3, 0.1, 0.1))])
    _ok_after(monkeypatch, action=lambda d: d.spacing_auto_btn.click())
    K._open_qc_settings(c)
    assert tuple(c._qc_settings["spacing_values"]) == (0.3, 0.1, 0.1)
    assert c._qc_spacing_entered is True
    ctx = c._on_annotator_reshape_context()
    assert ctx["spacing_source"] == "qc_settings"
    assert ctx["spacing"] == (0.3, 0.1, 0.1)


def test_auto_with_only_unscaled_layers_enters_nothing(
        app, monkeypatch, tmp_path):
    c = _controller(tmp_path, layers=[_layer(_ONE)])
    _ok_after(monkeypatch, action=lambda d: d.spacing_auto_btn.click())
    K._open_qc_settings(c)
    assert c._qc_spacing_entered is False
    assert c._on_annotator_reshape_context()["spacing_source"] == "unknown"
