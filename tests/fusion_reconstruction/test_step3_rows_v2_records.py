"""A3: the metadata and
voxel-size lines on Predict + Reconstruct's rows and the tier line on its
protected rows.  Run one file at a time with ``-o addopts=""``.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from napari_epicycle.fusion_reconstruction import _step3_rows as R  # noqa: E402
from napari_epicycle.fusion_reconstruction import _step3_pr_text as T  # noqa: E402

INFO = {"labels_shape": [147, 240, 240], "num_cells_final": 497}
SP = {"zyx": [0.52, 0.52, 0.52], "source": "step1"}


def test_the_narrow_meta_line_has_shape_channels_labels():
    rec = R.build_row("a.tif", "reconstructed", "CZYX", INFO,
                      source_shape=(2, 147, 240, 240))
    assert rec["shape_text"] == "147×240×240 · 2 ch · 497 labels"
    # axes stay on the status line
    assert rec["axes"] == "CZYX"


def test_without_a_source_shape_the_line_is_what_it_was():
    rec = R.build_row("a.tif", "reconstructed", "CZYX", INFO)
    assert rec["shape_text"] == R.shape_line(INFO)


def test_a_predicted_row_without_a_manifest_still_shows_its_shape():
    rec = R.build_row("a.tif", "predicted", "CZYX", None,
                      source_shape=(2, 155, 240, 195))
    assert rec["shape_text"] == "155×240×195 · 2 ch"


def test_the_voxel_line_and_its_amber_none():
    rec = R.build_row("a.tif", "reconstructed", "CZYX", INFO,
                      source_shape=(2, 147, 240, 240), spacing=SP)
    assert rec["vox_text"] == T.spacing_row_text(SP)
    assert not rec["vox_warn"] and "row_action" not in rec
    none = R.build_row("a.tif", "predicted", "CZYX", None, spacing=None)
    assert none["vox_text"] == T.S3_ROW_SPACING_NONE and none["vox_warn"]
    assert "row_action" not in none           # no inline button on narrow rows
    old = R.build_row("a.tif", "predicted", "CZYX", None)
    assert "vox_text" not in old                 # nothing asked, nothing drawn


def test_a_missing_patch_has_no_voxel_line():
    rec = R.build_row("a.tif", "reconstructed", "CZYX", INFO, missing=True,
                      spacing=SP)
    assert "vox_text" not in rec


@pytest.fixture(scope="module")
def app():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def ui(app):
    from qtpy.QtWidgets import QWidget
    from napari_epicycle.fusion_reconstruction.ui import (
        FusionReconstructionCuration,
    )
    w = FusionReconstructionCuration.__new__(FusionReconstructionCuration)
    QWidget.__init__(w)
    w.viewer = None
    w._build_ui()
    w._apply_theme()
    yield w
    w.deleteLater()


ITEMS = (("good", 3, (1, 2, 3)), ("review", 4, (4, 5, 6)))


def test_the_list_feeds_shape_and_spacing_into_the_rows(ui):
    from napari_epicycle.ui_shell._stack_row_delegate import ROLE_RECORD
    ui.refresh_patch_state_list(
        {"a.tif": "reconstructed", "b.tif": "predicted"},
        axes_by_patch={"a.tif": "CZYX", "b.tif": "CZYX"},
        info_by_patch={"a.tif": INFO},
        shapes={"a.tif": (2, 147, 240, 240), "b.tif": (2, 155, 240, 195)},
        spacing_by_patch={"a.tif": SP})
    lst = ui._patch_state_list
    recs = {lst.item(i).data(0x100): lst.item(i).data(ROLE_RECORD)
            for i in range(lst.count())}
    assert recs["a.tif"]["shape_text"] == "147×240×240 · 2 ch · 497 labels"
    assert recs["a.tif"]["vox_text"] == T.spacing_row_text(SP)
    assert recs["b.tif"]["vox_warn"] is True


def test_tier_counts_reach_the_protected_rows_only(ui):
    from napari_epicycle.ui_shell._stack_row_delegate import ROLE_RECORD
    ui.refresh_patch_state_list({"a.tif": "accepted", "b.tif": "finalized",
                                 "c.tif": "curating",
                                 "d.tif": "reconstructed"})
    ui.set_tier_counts({f"{n}.tif": ITEMS for n in "abcd"}, replace=True)
    lst = ui._patch_state_list
    got = {lst.item(i).data(0x100): lst.item(i).data(ROLE_RECORD)
           for i in range(lst.count())}
    assert got["a.tif"]["tier_counts"] == ITEMS
    assert got["b.tif"]["tier_counts"] == ITEMS
    assert "tier_counts" not in got["c.tif"]
    assert "tier_counts" not in got["d.tif"]
    # a rebuild keeps them; a status change away from protected drops them
    ui.refresh_patch_state_list({"a.tif": "accepted", "b.tif": "curating"})
    got = {lst.item(i).data(0x100): lst.item(i).data(ROLE_RECORD)
           for i in range(lst.count())}
    assert got["a.tif"]["tier_counts"] == ITEMS
    assert "tier_counts" not in got["b.tif"]


# -- the fold: Curation rows get the metadata and the voxel line ------------

def test_a_curation_row_carries_the_metadata_line_and_the_voxel_line():
    rec = R.build_curation_row(
        "a.tif", "reconstructed", qc_extracted=True, edited=True,
        axes="CZYX", source_shape=(2, 147, 240, 240), manifest_info=INFO,
        spacing=SP)
    # (audit L6) the marks first, joined with the shared separator
    assert rec["line2"] == T.S3_LINE_SEP.join((
        T.S3_CURATION_TAG_QC, T.S3_CURATION_TAG_EDITED,
        "CZYX · 147×240×240 · 2 ch · 497 labels"))
    assert rec["line3"] == T.spacing_row_text(SP) and not rec["line3_warn"]
    assert "row_action" not in rec
    none = R.build_curation_row("a.tif", "reconstructed", spacing=None)
    assert none["line3"] == T.S3_ROW_SPACING_NONE and none["line3_warn"]
    assert "row_action" not in none


def test_a_curation_row_asked_for_nothing_is_what_it_was():
    rec = R.build_curation_row("a.tif", "reconstructed", qc_extracted=True)
    assert rec["line2"] == T.S3_CURATION_TAG_QC and "line3" not in rec


def test_a_missing_curation_row_has_no_voxel_line():
    rec = R.build_curation_row("a.tif", "reconstructed", missing=True,
                               spacing=SP)
    assert "line3" not in rec


def test_the_box_list_is_handed_shapes_and_spacing(ui):
    from napari_epicycle.ui_shell._stack_row_delegate import ROLE_RECORD
    ui.refresh_curation_bucket(
        ["a.tif"], {"a.tif": "reconstructed"}, axes_by_patch={"a.tif": "CZYX"},
        info_by_patch={"a.tif": INFO}, shapes={"a.tif": (2, 147, 240, 240)},
        spacing_by_patch={"a.tif": SP})
    rec = ui._curation_bucket.item(0).data(ROLE_RECORD)
    assert "2 ch · 497 labels" in rec["line2"]
    assert rec["line3"] == T.spacing_row_text(SP)


def _loaded(lst):
    from napari_epicycle.ui_shell._stack_row_delegate import ROLE_RECORD
    return [lst.item(i).data(0x100) for i in range(lst.count())
            if (lst.item(i).data(ROLE_RECORD) or {}).get("loaded")]


def test_exactly_one_curation_row_is_loaded_and_none_after_a_clear(ui):
    names = ["a.tif", "b.tif", "c.tif"]
    st = {n: "reconstructed" for n in names}
    ui.refresh_curation_bucket(names, st)
    assert _loaded(ui._curation_bucket) == []
    ui.set_loaded_patch("b.tif")
    assert _loaded(ui._curation_bucket) == ["b.tif"]
    ui.set_loaded_patch("c.tif")                      # a switch
    assert _loaded(ui._curation_bucket) == ["c.tif"]
    ui.refresh_curation_bucket(names, st)             # a rebuild keeps it
    assert _loaded(ui._curation_bucket) == ["c.tif"]
    ui.set_loaded_patch(None)                         # a clear
    assert _loaded(ui._curation_bucket) == []
    ui.refresh_curation_bucket(names, st)
    assert _loaded(ui._curation_bucket) == []


def test_the_controller_pushes_the_loaded_patch(ui):
    from types import SimpleNamespace
    from napari_epicycle.fusion_reconstruction import controller as C
    ui.refresh_curation_bucket(["a.tif", "b.tif"],
                               {"a.tif": "reconstructed",
                                "b.tif": "reconstructed"})
    ns = SimpleNamespace(ui=ui, _qc_view_patch_name="a.tif")
    C.FusionReconstructionController._sync_loaded_patch_marker(ns)
    assert _loaded(ui._curation_bucket) == ["a.tif"]
    ns._qc_view_patch_name = ""
    C.FusionReconstructionController._sync_loaded_patch_marker(ns)
    assert _loaded(ui._curation_bucket) == []
    C.FusionReconstructionController._sync_loaded_patch_marker(
        SimpleNamespace(ui=object(), _qc_view_patch_name="a.tif"))  # no-op


# -- the fold: Step 3's Set spacing dialog is amber -----------------------------

GREENS = ("6abf69", "106, 191, 105", "106,191,105")


def _sheets(dlg):
    from qtpy.QtWidgets import QWidget
    out = [dlg.styleSheet()]
    out += [w.styleSheet() for w in dlg.findChildren(QWidget)]
    return "\n".join(out).lower()


def test_step3s_set_spacing_dialog_carries_no_green(app):
    from types import SimpleNamespace
    from napari_epicycle.fusion_reconstruction import controller as C
    ns = SimpleNamespace(ui=None)
    dlg = C.FusionReconstructionController._make_set_spacing_dialog(
        ns, ["a.tif"], {})
    try:
        sheet = _sheets(dlg)
        assert "212, 168, 67" in dlg.ok_btn.styleSheet()
        assert not any(g in sheet for g in GREENS), sheet
    finally:
        dlg.deleteLater()


def test_the_dialog_defaults_to_step_4s_green(app):
    from napari_epicycle.ui_shell._set_spacing_dialog import SetSpacingDialog
    from napari_epicycle.segmentation import _set_spacing_dialog as T4
    dlg = SetSpacingDialog(None, names=["a.tif"], record={},
                           texts=T4.step4_spacing_texts())
    try:
        assert "106, 191, 105" in dlg.ok_btn.styleSheet()
    finally:
        dlg.deleteLater()


def test_a_failed_load_leaves_the_loaded_marker_on_the_loaded_patch(ui):
    """Audit L9: the marker moves when the load LANDS, not when it starts."""
    from types import SimpleNamespace
    from napari_epicycle.fusion_reconstruction import controller as C
    Ctl = C.FusionReconstructionController
    ui.refresh_curation_bucket(["a.tif", "b.tif"],
                               {"a.tif": "reconstructed",
                                "b.tif": "reconstructed"})
    ui.set_loaded_patch("a.tif")
    ns = SimpleNamespace(ui=ui, viewer=object(), working_dir="x",
                         _qc_view_patch_name="a.tif",
                         layer_lifecycle_manager=None,
                         _resolve_patch_payload=lambda n: None)
    ns._sync_loaded_patch_marker = lambda: Ctl._sync_loaded_patch_marker(ns)
    assert Ctl._load_patch_for_qc_view(ns, "b.tif") is False    # early return
    assert _loaded(ui._curation_bucket) == ["a.tif"]
    # ... and a load that lands moves it
    ns._qc_view_patch_name = "b.tif"
    ns._sync_loaded_patch_marker()
    assert _loaded(ui._curation_bucket) == ["b.tif"]
