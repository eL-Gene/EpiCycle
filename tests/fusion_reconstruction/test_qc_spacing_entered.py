"""When a 1 / 1 / 1 spacing counts as one the user ENTERED.

QC Settings stores an entered 1 / 1 / 1 exactly like its factory default
(``spacing=None``), so the controller keeps one mark, ``_qc_spacing_entered``,
for "the user set this voxel size".  The Add page's micron settings refuse an
``"unknown"`` spacing, so the mark must never be set by an accept that left
the spacing boxes alone: QC Settings is also where features, PCA features and
the pairing run are chosen, and an OK there is not a voxel size.

Covered here:

* an accept with the spacing untouched leaves a never-set 1 / 1 / 1 unknown;
* a spacing edited in the dialog counts (also an edit back to 1 / 1 / 1);
* a later accept that does not touch it keeps an entered spacing;
* only ``spacing_values`` is what the boxes show; an unreadable value is
  never an edit;
* Session's "Set spacing..." is entered per patch, at 1 / 1 / 1 too, and
  tells an open Add page.

The workspace and Session-folder resets live with their own fakes in
``test_session_folder_guard.py``; the real dialog is driven in
``test_qc_spacing_entered_dialog.py``.

Qt-free: the controller is built with ``object.__new__``.
"""
from __future__ import annotations

import types

import pytest

from napari_epicycle.fusion_reconstruction import qc_extraction_dialog
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController as K,
)

_ONE = (1.0, 1.0, 1.0)


def _dialog_returning(spacing_values):
    """A QC Settings dialog that the user OKs with *spacing_values* in the
    spacing boxes (the real dialog stores 1 / 1 / 1 as ``spacing=None``)."""
    vals = tuple(float(v) for v in spacing_values)

    class _Dlg:
        Accepted = 1

        def __init__(self, *a, **k):
            self.spacing_auto_btn = types.SimpleNamespace(
                clicked=types.SimpleNamespace(connect=lambda *_a: None))

        def exec_(self):
            return 1

        def get_settings(self):
            return {"spacing": None if vals == _ONE else vals,
                    "spacing_values": vals,
                    "erosion_iters": 1, "max_radius": 2,
                    "composite_method": "zscore", "pca_features": []}

        def get_match_run(self):
            return None

    return _Dlg


def _controller(tmp_path):
    c = object.__new__(K)
    c.ui = types.SimpleNamespace()
    c.viewer = None
    c._qc_features_df = None
    c._qc_spacing_entered = False
    c._qc_settings = {"spacing": None, "spacing_values": _ONE}
    c._qc_view_patch_name = "p01.tif"
    c.working_dir = str(tmp_path)
    c._current_downsample_factor = lambda *_a, **_k: 1
    c.logged = []
    c._append_log = c.logged.append
    c._detect_available_channels = lambda: []
    c._qc_channel_roles = lambda *_a: {}
    c._match_picker_state = lambda *_a: ([], None, False)
    c._extract_spacing_lines = lambda: []
    c._notify_spacing_changed = lambda: None
    c._check_qc_extraction_staleness = lambda: None
    c._maybe_show_qc_staleness_notification = lambda: None
    return c


def _accept(monkeypatch, c, spacing_values):
    monkeypatch.setattr(qc_extraction_dialog, "QCExtractionDialog",
                        _dialog_returning(spacing_values))
    K._open_qc_settings(c)


def _source(c):
    return c._on_annotator_reshape_context()["spacing_source"]


# ---------------------------------------------------------------------------
# QC Settings
# ---------------------------------------------------------------------------

def test_an_accept_that_leaves_the_spacing_alone_is_not_a_voxel_size(
        monkeypatch, tmp_path):
    """The user opened QC Settings to tick features or a pairing run and
    pressed OK: the never-set 1 / 1 / 1 stays unknown, so the Add page's
    micron settings still refuse."""
    c = _controller(tmp_path)
    _accept(monkeypatch, c, _ONE)
    assert c._qc_spacing_entered is False
    ctx = c._on_annotator_reshape_context()
    assert ctx["spacing_source"] == "unknown"
    assert ctx["spacing"] == _ONE


def test_a_spacing_edited_in_the_dialog_counts(monkeypatch, tmp_path):
    c = _controller(tmp_path)
    _accept(monkeypatch, c, (0.3, 0.1, 0.1))
    assert c._qc_spacing_entered is True
    ctx = c._on_annotator_reshape_context()
    assert ctx["spacing_source"] == "qc_settings"
    assert ctx["spacing"] == (0.3, 0.1, 0.1)


def test_an_edit_back_to_one_micron_is_an_entered_one_micron_voxel(
        monkeypatch, tmp_path):
    """0.3 / 0.1 / 0.1 first, then the boxes typed back to 1 / 1 / 1: the
    user chose 1.0 um, which the dialog stores as ``spacing=None``.  The
    mark starts False here (as after a change of workspace, which keeps the
    boxes' last numbers), so only the edit can set it."""
    c = _controller(tmp_path)
    c._qc_settings = {"spacing": (0.3, 0.1, 0.1),
                      "spacing_values": (0.3, 0.1, 0.1)}
    assert c._qc_spacing_entered is False
    _accept(monkeypatch, c, _ONE)
    assert c._qc_settings["spacing"] is None
    assert c._qc_spacing_entered is True
    assert _source(c) == "qc_settings"


def test_a_later_untouched_accept_keeps_an_entered_spacing(
        monkeypatch, tmp_path):
    c = _controller(tmp_path)
    _accept(monkeypatch, c, (0.3, 0.1, 0.1))
    _accept(monkeypatch, c, _ONE)            # typed back to 1 um
    _accept(monkeypatch, c, _ONE)            # then OK for the features only
    assert c._qc_spacing_entered is True
    assert _source(c) == "qc_settings"


def test_a_spin_box_rounding_is_not_an_edit(monkeypatch, tmp_path):
    """The boxes hold 3 decimals: a stored 1.0000001 coming back as 1.0 is
    the same number, not an entry."""
    c = _controller(tmp_path)
    c._qc_settings = {"spacing": None,
                      "spacing_values": (1.0000001, 1.0, 0.9999999)}
    _accept(monkeypatch, c, _ONE)
    assert c._qc_spacing_entered is False
    assert _source(c) == "unknown"


def test_the_accept_log_says_whether_the_spacing_was_entered(
        monkeypatch, tmp_path):
    c = _controller(tmp_path)
    _accept(monkeypatch, c, _ONE)
    assert any("spacing entered=False" in line for line in c.logged)
    _accept(monkeypatch, c, (0.3, 0.1, 0.1))
    assert any("spacing entered=True" in line for line in c.logged)


def test_a_cancel_changes_nothing(monkeypatch, tmp_path):
    c = _controller(tmp_path)
    Dlg = _dialog_returning((0.3, 0.1, 0.1))
    Dlg.exec_ = lambda self: 0
    monkeypatch.setattr(qc_extraction_dialog, "QCExtractionDialog", Dlg)
    K._open_qc_settings(c)
    assert c._qc_spacing_entered is False
    assert _source(c) == "unknown"


def test_a_run_switch_keeps_an_entered_spacing(tmp_path):
    """A run switch shows the same patch's images, so the voxel size the
    user entered still holds: neither reset a run switch runs forgets it
    (only a change of workspace or Session folder does)."""
    from napari_epicycle.fusion_reconstruction import controller as ctl_mod

    c = _controller(tmp_path)
    c._qc_spacing_entered = True
    c._sync_loaded_patch_marker = lambda: None
    c._reset_qc_plot_area = lambda *a, **k: None
    ctl_mod._reset_per_patch_state(c)
    K._reset_workspace_scoped_caches(c)
    assert c._qc_spacing_entered is True


# ---------------------------------------------------------------------------
# what the boxes show
# ---------------------------------------------------------------------------

def test_the_boxes_are_filled_from_spacing_values_only():
    """The dialog fills its spacing boxes from ``spacing_values`` alone; a
    dict holding only ``spacing`` opens the boxes at their 1.0 default."""
    from napari_epicycle.fusion_reconstruction import controller as ctl_mod

    assert ctl_mod._qc_spacing_numbers({"spacing": (0.5, 0.5, 0.5)}) == _ONE
    assert ctl_mod._qc_spacing_numbers(
        {"spacing": None, "spacing_values": (0.3, 0.1, 0.1)}) == (
            0.3, 0.1, 0.1)
    assert ctl_mod._qc_spacing_numbers(None) == _ONE


def test_an_untouched_ok_over_a_spacing_only_dict_is_not_an_entry(
        monkeypatch, tmp_path):
    """Settings with ``spacing`` but no ``spacing_values``: the boxes open
    at 1 / 1 / 1 and come back at 1 / 1 / 1 -- nothing was typed."""
    c = _controller(tmp_path)
    c._qc_settings = {"spacing": (0.5, 0.5, 0.5)}
    _accept(monkeypatch, c, _ONE)
    assert c._qc_spacing_entered is False
    assert _source(c) == "unknown"


def test_an_unreadable_value_is_never_an_edit():
    from napari_epicycle.fusion_reconstruction import controller as ctl_mod

    assert ctl_mod._qc_spacing_edited(None, _ONE) is False
    assert ctl_mod._qc_spacing_edited(_ONE, None) is False
    assert ctl_mod._qc_spacing_edited(None, None) is False
    assert ctl_mod._qc_spacing_numbers(
        {"spacing_values": "not three numbers"}) is None
    assert ctl_mod._qc_spacing_edited(_ONE, (0.3, 0.1, 0.1)) is True


# ---------------------------------------------------------------------------
# Session > Set spacing...
# ---------------------------------------------------------------------------

class _SpacingStore:
    """The state manager's per-patch voxel sizes, in memory."""

    def __init__(self):
        self.entries = {}
        self.last_spacing_error = ""

    def get_patch_spacing(self, name):
        return self.entries.get(name)

    def set_patch_spacing(self, name, zyx, source):
        self.entries[name] = {"zyx": [float(v) for v in zyx],
                              "source": source}
        return True


class _SpacingDialog:
    def __init__(self, zyx):
        self._zyx = zyx

    def exec_(self):
        return 1

    def values(self):
        return {"zyx": list(self._zyx)}


def _set_spacing_controller(tmp_path, zyx, *, ok=True):
    c = _controller(tmp_path)
    c.state_manager = _SpacingStore()
    c._spacing_locked = lambda: False
    c._refresh_session_spacing_gate = lambda: None
    c._spacing_stale_names = lambda names, z: []
    dlg = _SpacingDialog(zyx)
    if not ok:
        dlg.exec_ = lambda: 0
    c._make_set_spacing_dialog = lambda names, record: dlg
    c._refresh_patch_list_colors = lambda: None
    c._update_cell_width_line = lambda: None
    # an open Add page: the real notify reaches it
    c.told = []
    page = types.SimpleNamespace(
        notify_spacing_changed=lambda: c.told.append(
            c._on_annotator_reshape_context()["spacing_source"]))
    c._get_curation_annotator_widget = lambda: page
    c._notify_spacing_changed = K._notify_spacing_changed.__get__(c)
    return c


def test_set_spacing_tells_an_open_add_page(tmp_path):
    """After Set spacing... the Add page drops its refusal and re-measures a
    placed box -- it is told, and the spacing it then reads is known."""
    c = _set_spacing_controller(tmp_path, (0.3, 0.1, 0.1))
    assert K._open_set_spacing(c, ["p01.tif"]) is True
    assert c.told == ["patch"]


def test_a_cancelled_set_spacing_tells_nothing(tmp_path):
    c = _set_spacing_controller(tmp_path, (0.3, 0.1, 0.1), ok=False)
    assert K._open_set_spacing(c, ["p01.tif"]) is False
    assert c.told == []


@pytest.mark.parametrize("zyx", [(1.0, 1.0, 1.0), (0.3, 0.1, 0.1)])
def test_set_spacing_is_an_entered_voxel_size_for_its_patch(tmp_path, zyx):
    c = _set_spacing_controller(tmp_path, zyx)
    assert _source(c) == "unknown"
    assert K._open_set_spacing(c, ["p01.tif"]) is True
    ctx = c._on_annotator_reshape_context()
    assert ctx["spacing_source"] == "patch"
    assert ctx["spacing"] == zyx
    # QC Settings' own mark is not what made it known
    assert c._qc_spacing_entered is False
