"""A2 (board B): the Session panel -- the Workspace card's ``?`` and
words, the Patches card (title, ``?``, description, the header count line),
"Use another folder…" under the list with the dim sentence (the path field
hidden, kept), the rows' eye (the same View as the P+R foot; refused under
the lock), the inline "Set spacing…" (that row only), the tier counts on
both lists, and the Pipeline status block's words.
Run one file at a time with ``-o addopts=""``.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

ITEMS = (("good", 3, (80, 200, 120)), ("review", 1, (255, 200, 60)),
         ("bad", 0, (255, 120, 40)), ("garbage", 2, (240, 60, 60)))


@pytest.fixture(scope="module")
def app():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _P():
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    return P


def _D():
    from napari_epicycle.ui_shell import _stack_row_delegate as D
    return D


def _bare():
    from qtpy.QtWidgets import QWidget

    from napari_epicycle.fusion_reconstruction.ui import (
        FusionReconstructionCuration,
    )
    w = FusionReconstructionCuration.__new__(FusionReconstructionCuration)
    QWidget.__init__(w)
    w.viewer = None
    w._build_ui()
    w._apply_theme()
    return w


@pytest.fixture
def ui(app):
    w = _bare()
    yield w
    w.deleteLater()
    app.processEvents()


@pytest.fixture
def full(app):
    from napari_epicycle.fusion_reconstruction.ui import (
        FusionReconstructionCuration,
    )
    w = FusionReconstructionCuration(viewer=None)
    c = w.controller
    statuses = {}
    c.state_manager.get_all_patch_statuses = lambda: dict(statuses)
    yield w, c, statuses
    w.deleteLater()
    app.processEvents()


def _fill(ui, names):
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QListWidgetItem
    ui.loaded_patch_list.clear()
    for n in names:
        it = QListWidgetItem(n)
        it.setData(Qt.UserRole, n)
        ui.loaded_patch_list.addItem(it)


def _labels(widget):
    from qtpy.QtWidgets import QLabel
    return [lb.text() for lb in widget.findChildren(QLabel)]


# -- the two cards -----------------------------------------------------------------

def test_the_workspace_card_has_its_words_and_a_guide(ui):
    P = _P()
    texts = _labels(ui._session_panel)
    assert P.S3_SESSION_WORKSPACE_TITLE in texts
    assert P.S3_SESSION_WORKSPACE_DESC in texts
    assert ui.session_workspace_help_btn.property("guide") \
        .endswith("step3_session.md")


def test_the_patches_card_has_title_guide_and_description(ui):
    P = _P()
    texts = _labels(ui._session_panel)
    assert P.S3_SESSION_PATCHES_TITLE == "Patches"
    assert P.S3_SESSION_PATCHES_TITLE in texts
    assert P.S3_SESSION_PATCHES_DESC in texts
    assert "Patch / Stack Directory" not in texts


@pytest.mark.parametrize("counts,head,amber", [
    ((0, 0, 0, 0), "No patches yet.", ""),
    ((1, 0, 0, 0), "1 patch", ""),
    ((10, 3, 1, 0), "10 patches · 3 curating · 1 finalized", ""),
    ((10, 3, 1, 1), "10 patches · 3 curating · 1 finalized · ",
     "1 without a voxel size"),
    ((4, 0, 0, 2), "4 patches · ", "2 without a voxel size"),
])
def test_the_header_count_line_in_its_states(counts, head, amber):
    assert _P().session_header_counts(*counts) == (head, amber)


def test_the_header_line_shows_the_amber_clause_in_amber(ui):
    ui.set_session_header_counts(10, 3, 1, 1)
    t = ui.session_header_counts.text()
    assert "10 patches · 3 curating · 1 finalized" in t
    assert "1 without a voxel size" in t and "#" in t     # a colour span
    ui.set_session_header_counts(0, 0, 0, 0)
    assert ui.session_header_counts.text() == "No patches yet."


def test_the_folder_button_sits_under_the_list_and_the_field_is_hidden(ui):
    P = _P()
    lay = ui._session_panel.layout()
    row = ui._session_folder_row
    assert lay.indexOf(row) > lay.indexOf(ui.loaded_patch_list)
    assert ui.set_patch_directory_button.text() == P.S3_SESSION_USE_FOLDER_BTN
    assert ui.set_patch_directory_path.isHidden()       # kept, hidden
    assert ui.set_patch_directory_button.parent() is row


def test_the_dim_sentence_says_where_the_patches_come_from(ui):
    P = _P()
    ui.set_patch_dir_source(True, "C:/ws/artifacts/step1/patches")
    assert ui.patch_dir_qualifier.text() == P.S3_SESSION_USING_DEFAULT
    assert not ui.patch_dir_qualifier.isHidden()
    ui.set_patch_dir_source(False, "D:/data/my_patches")
    assert ui.patch_dir_qualifier.text() == P.S3_SESSION_USING_FOLDER.format(
        folder="my_patches")
    assert ui.patch_dir_qualifier.toolTip() == "D:/data/my_patches"


def test_the_old_spacing_row_is_hidden_not_deleted(ui):
    assert ui._session_spacing_row.isHidden()
    assert ui.session_set_spacing_btn is not None


def test_the_empty_list_says_how_to_get_patches(ui):
    assert ui.loaded_patch_list._empty_lines == list(
        _P().S3_SESSION_EMPTY_LINES)


def test_the_session_rows_have_the_eye(ui):
    d = ui.loaded_patch_list.itemDelegate()
    assert d._show_eye is True
    assert d._eye_lock_property == "step3SessionEyeLocked"
    ui.set_session_eye_locked(True)
    assert ui.loaded_patch_list.property("step3SessionEyeLocked") is True


# -- the tier counts on both lists ---------------------------------------------------

def test_tier_counts_reach_session_and_curation_rows_and_survive_rebuilds(ui):
    from napari_epicycle.ui_shell._stack_row_delegate import ROLE_RECORD
    _fill(ui, ["a.tif", "b.tif"])
    ui.update_patch_list_status({"a.tif": "curating", "b.tif": "predicted"})
    ui.refresh_curation_bucket(["a.tif"], {"a.tif": "curating"})
    ui.set_tier_counts({"a.tif": ITEMS, "b.tif": None}, replace=True)
    s = {ui.loaded_patch_list.item(i).text():
         ui.loaded_patch_list.item(i).data(ROLE_RECORD)
         for i in range(2)}
    assert s["a.tif"]["tier_counts"] == ITEMS
    assert "tier_counts" not in s["b.tif"]
    assert ui._curation_bucket.item(0).data(ROLE_RECORD)["tier_counts"] \
        == ITEMS
    # a rebuild keeps them
    ui.update_patch_list_status({"a.tif": "curating", "b.tif": "predicted"})
    ui.refresh_curation_bucket(["a.tif"], {"a.tif": "curating"})
    assert ui.loaded_patch_list.item(0).data(ROLE_RECORD)["tier_counts"] \
        == ITEMS
    assert ui._curation_bucket.item(0).data(ROLE_RECORD)["tier_counts"] \
        == ITEMS
    # a whole-list replace drops what it does not name
    ui.set_tier_counts({"b.tif": ITEMS}, replace=True)
    assert "tier_counts" not in ui.loaded_patch_list.item(0).data(ROLE_RECORD)


# -- the Pipeline status block --------------------------------------------------------

def test_the_pipeline_block_has_the_mockups_words(ui):
    P = _P()
    texts = _labels(ui._session_panel)
    assert P.S3_PIPELINE_TITLE in texts and P.S3_PIPELINE_DESC in texts
    assert "Pipeline Status" not in texts
    for word in P.S3_PIPELINE_COUNTERS:
        assert word in texts
    assert ui.summary_patches_value.isHidden()


def test_the_pipeline_block_sits_under_the_list(ui):
    lay = ui._session_panel.layout()
    assert lay.indexOf(ui._session_dashboard) > lay.indexOf(
        ui._session_folder_row)
    assert lay.indexOf(ui.session_model_pointer) > lay.indexOf(
        ui._session_dashboard)


# -- the controller: the eye, the inline button, the header line ----------------------

def test_the_eye_runs_the_same_view_as_the_foot(full, monkeypatch):
    w, c, statuses = full
    seen = []
    monkeypatch.setattr(c, "_view_patch", lambda name: seen.append(name))
    w.loaded_patch_list.itemDelegate().eye_clicked.emit("a.tif")
    assert seen == ["a.tif"]


def test_the_foot_view_goes_through_the_same_door(full, monkeypatch):
    from qtpy.QtCore import Qt
    w, c, statuses = full
    statuses.update({"a.tif": "reconstructed"})
    w.refresh_patch_state_list({"a.tif": "reconstructed"})
    w._patch_state_list.item(0).setSelected(True)
    seen = []
    monkeypatch.setattr(c, "_view_patch", lambda name: seen.append(name))
    c._view_reconstruction()
    assert seen == [w._patch_state_list.item(0).data(Qt.UserRole)]


def test_the_eye_is_refused_under_the_lock_with_the_lock_sentence(
        full, monkeypatch):
    w, c, statuses = full
    P = _P()
    seen, said = [], []
    monkeypatch.setattr(c, "_view_patch", lambda name: seen.append(name))
    monkeypatch.setattr(c, "_pr_status", lambda t, *a: said.append(t))
    c._save_in_flight = True
    try:
        c._refresh_step3_lock()
        assert w.loaded_patch_list.property("step3SessionEyeLocked") is True
        w.loaded_patch_list.itemDelegate().eye_clicked.emit("a.tif")
    finally:
        c._save_in_flight = False
    assert seen == []
    assert said == [dict(P.S3_LOCK_SENTENCES)["_save_in_flight"]]
    c._refresh_step3_lock()
    assert w.loaded_patch_list.property("step3SessionEyeLocked") is False


def test_the_inline_button_opens_set_spacing_for_that_row_only(
        full, monkeypatch):
    w, c, statuses = full
    opened = []
    monkeypatch.setattr(c, "_open_set_spacing",
                        lambda names: opened.append(list(names)))
    monkeypatch.setattr(c, "_session_highlighted",
                        lambda: ["a.tif", "b.tif"])
    w.loaded_patch_list.itemDelegate().action_clicked.emit(
        "b.tif", "set_spacing")
    assert opened == [["b.tif"]]


def test_the_inline_button_is_refused_while_predict_runs(full, monkeypatch):
    w, c, statuses = full
    P = _P()
    opened, said = [], []
    monkeypatch.setattr(c, "_open_set_spacing",
                        lambda names: opened.append(names))
    monkeypatch.setattr(c, "_pr_status", lambda t, *a: said.append(t))
    monkeypatch.setattr(c, "_spacing_locked", lambda: True)
    w.loaded_patch_list.itemDelegate().action_clicked.emit(
        "b.tif", "set_spacing")
    assert opened == [] and said == [P.S3_SET_SPACING_LOCKED]


def test_the_header_line_counts_the_listed_patches(full, monkeypatch):
    w, c, statuses = full
    _fill(w, ["a.tif", "b.tif", "c.tif", "d.tif"])
    statuses.update({"a.tif": "curating", "b.tif": "finalized",
                     "c.tif": "curating", "d.tif": "unprocessed"})
    monkeypatch.setattr(c, "_session_spacing", lambda names: {
        "a.tif": {"zyx": [1, 1, 1], "source": "user"},
        "b.tif": {"zyx": [1, 1, 1], "source": "user"},
        "c.tif": {"zyx": [1, 1, 1], "source": "user"}})
    c._refresh_patch_list_colors()
    t = w.session_header_counts.text()
    assert t.startswith("4 patches · 2 curating · 1 finalized · ")
    assert "1 without a voxel size" in t


def test_the_controller_tells_the_panel_where_the_folder_came_from(
        full, tmp_path):
    w, c, statuses = full
    P = _P()
    d = tmp_path / "chosen"
    d.mkdir()
    c._set_patch_dir_qualifier(False, str(d))
    assert w.patch_dir_qualifier.text() == P.S3_SESSION_USING_FOLDER.format(
        folder="chosen")
    c._set_patch_dir_qualifier(True, str(d))
    assert w.patch_dir_qualifier.text() == P.S3_SESSION_USING_DEFAULT
