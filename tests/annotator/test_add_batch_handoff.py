"""Add's REVIEW card hands every added cell to Batch Refine.

"Refine all ▸" in the existing piece row (0 px of height, fits the narrowest
page width, hidden -- not greyed -- while the list is empty), wired through
``set_add_callbacks(on_batch_refine=)`` to the explicit-target open, the
Selected radio reading "Added this session (N)", a refused open's sentence
in the Add status, and ``ADD_BATCH_BUSY`` on Add's four doors while a batch
runs or is staged.

Plain ``QApplication`` (pytest-qt is not installed).
"""
import os
import types

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from packages.epicycle_annotator import batch_refine as br  # noqa: E402
from tests.annotator.test_batch_chain_dialog import make_ann  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def widgets(qapp):
    made = []
    yield made
    for w in made:
        try:
            if hasattr(w, "mark_clean_close"):
                w.mark_clean_close()
            w.close()
            w.deleteLater()
        except RuntimeError:
            pass
    qapp.processEvents()


@pytest.fixture
def panel(qapp):
    from napari_epicycle.annotator.advanced_features_widget import (
        AdvancedFeaturesWidget,
    )

    w = AdvancedFeaturesWidget(mode="3d_curation")
    yield w
    w.hide()
    w.setParent(None)
    w.deleteLater()
    qapp.processEvents()


def review(panel, ids=(5, 6, 7)):
    panel.update_add_ui_state(
        review_active=True, piece_ids=list(ids),
        piece_sizes={i: 1000 + i for i in ids})


# ================================================================== panel


class TestButton:
    def test_words_and_glyph(self, panel):
        from napari_epicycle.annotator.advanced_features_widget import (
            ADD_REFINE_ALL_BTN,
            ADD_REFINE_ALL_TIP,
        )

        b = panel.add_batch_refine_btn
        assert b.text() == ADD_REFINE_ALL_BTN == "Refine all ▸"
        assert b.text()[-1] == panel.add_refine_piece_btn.text()[-1]
        assert b.toolTip() == ADD_REFINE_ALL_TIP

    def test_hidden_until_there_are_pieces(self, panel):
        b = panel.add_batch_refine_btn
        assert b.isHidden()
        review(panel)
        assert not b.isHidden() and b.isEnabled()
        panel.update_add_ui_state(piece_ids=[])
        assert b.isHidden()

    def test_the_explicit_keyword_wins(self, panel):
        review(panel)
        panel.update_add_ui_state(add_batch_refine_enabled=False)
        assert panel.add_batch_refine_btn.isHidden()
        panel.update_add_ui_state(add_batch_refine_enabled=True)
        assert not panel.add_batch_refine_btn.isHidden()

    def test_same_row_as_refine(self, panel):
        a = panel.add_refine_piece_btn
        b = panel.add_batch_refine_btn
        assert a.parentWidget() is b.parentWidget()
        lay = a.parentWidget().layout()

        def row_of(w):
            for i in range(lay.count()):
                item = lay.itemAt(i)
                sub = item.layout()
                if sub is not None and sub.indexOf(w) >= 0:
                    return sub
            return None

        assert row_of(a) is not None and row_of(a) is row_of(b)

    def test_click_calls_on_batch_refine(self, panel):
        calls = []
        panel.set_add_callbacks(on_batch_refine=lambda: calls.append(1))
        review(panel)
        panel.add_batch_refine_btn.click()
        assert calls == [1]


class TestLayout:
    def _slots(self, panel, qapp):
        panel.resize(340, 350)
        panel.show()
        panel.focus_tab("Add") if hasattr(panel, "focus_tab") else None
        qapp.processEvents()
        return panel._panel_slots["Add"]

    def test_both_buttons_fit_the_narrowest_width(self, panel, qapp):
        review(panel)
        slots = self._slots(panel, qapp)
        viewport = slots["scroll"].viewport().width()
        row = panel.add_refine_piece_btn.parentWidget().layout()
        spacing = max(0, row.spacing())
        need = (panel.add_refine_piece_btn.sizeHint().width()
                + panel.add_batch_refine_btn.sizeHint().width() + spacing)
        assert need <= viewport, (need, viewport)
        assert slots["content"].minimumSizeHint().width() <= viewport

    def test_the_page_does_not_get_taller(self, panel, qapp):
        review(panel)
        slots = self._slots(panel, qapp)
        content = slots["content"]
        with_btn = content.sizeHint().height()
        panel.update_add_ui_state(add_batch_refine_enabled=False)
        qapp.processEvents()
        without = content.sizeHint().height()
        assert with_btn == without


# ================================================================= dialog


def test_the_selected_radio_names_the_added_cells(widgets):
    from tests.annotator.test_batch_chain_dialog import make_dialog

    dlg = make_dialog(widgets, selected_count=3, added_targets=True)
    assert dlg._radio_selected.text() == br.BATCH_TARGET_ADDED.format(n=3) \
        == "Added this session (3)"
    dlg.set_target_counts(selected=2)
    assert dlg._radio_selected.text() == "Added this session (2)"
    plain = make_dialog(widgets, selected_count=3)
    assert plain._radio_selected.text() == br.BATCH_TARGET_SELECTED.format(
        n=3)


# ============================================================== annotator


def add_ann(monkeypatch, widgets):
    ann = make_ann(monkeypatch, widgets)
    ann.pushed = []
    ann._push_add_ui = lambda **k: ann.pushed.append(k)
    ann._add_piece_ids = [1, 2]
    return ann


def last_status(ann):
    for k in reversed(ann.pushed):
        if "status_text" in k:
            return k["status_text"]
    return None


class TestHandOff:
    def test_refine_all_opens_on_the_added_cells(self, monkeypatch, widgets):
        ann = add_ann(monkeypatch, widgets)
        assert ann._batch_refine_add_pieces() is None
        dlg = ann._batch_refine_dialog
        assert dlg._radio_selected.isChecked()
        assert dlg._radio_selected.text() == "Added this session (2)"
        assert dlg.chain_name() is None

    @pytest.mark.parametrize("setup,sentence", [
        ("lock", br.BATCH_BUSY),
        ("worker", br.BATCH_STILL_STOPPING),
        ("dirty", br.BATCH_NOT_CLEAN),
    ])
    def test_a_refused_open_says_why_in_the_add_status(
            self, monkeypatch, widgets, setup, sentence):
        ann = add_ann(monkeypatch, widgets)
        if setup == "lock":
            ann._load_locked = True
        elif setup == "worker":
            ann._batch_worker = object()
            ann._load_locked = True
        else:
            ann._refine_state = types.SimpleNamespace(
                can_start_batch_refine=lambda: False)
        assert ann._batch_refine_add_pieces() == sentence
        assert last_status(ann) == sentence
        assert "Can't refine right now" not in str(ann.pushed)


class TestAddBatchBusy:
    """ADD_BATCH_BUSY on all four doors, running or staged."""

    @pytest.mark.parametrize("flag", ["_batch_running", "_batch_staged"])
    def test_start(self, monkeypatch, widgets, flag):
        ann = add_ann(monkeypatch, widgets)
        setattr(ann, flag, True)
        ann._add_running = False
        ann._start_add()
        assert last_status(ann) == br.ADD_BATCH_BUSY
        assert not getattr(ann, "_add_running", False)

    def test_accept_while_running(self, monkeypatch, widgets):
        ann = add_ann(monkeypatch, widgets)
        ann._batch_running = True
        ann._add_running = True
        ann._add_preview_result = "must not be read"
        ann._accept_add()
        assert last_status(ann) == br.ADD_BATCH_BUSY
        assert ann._add_running is True

    def test_add_another_keeps_review(self, monkeypatch, widgets):
        ann = add_ann(monkeypatch, widgets)
        ann._batch_staged = True
        ann._add_review = True
        ann._add_another()
        assert ann._add_review is True
        assert last_status(ann) == br.ADD_BATCH_BUSY

    def test_undo_all_does_not_undo(self, monkeypatch, widgets):
        ann = add_ann(monkeypatch, widgets)
        ann._batch_staged = True
        undone = []
        ann._add_session = [{"ids": [1]}]
        ann._on_add_undo_session_cb = lambda n: undone.append(n)
        ann._undo_add_all()
        assert undone == []
        assert last_status(ann) == br.ADD_BATCH_BUSY

    def test_the_sentence(self):
        assert br.ADD_BATCH_BUSY == (
            "Batch Refine is working on these cells — Accept or Reject it "
            "first.")

    def test_every_door_checks_first(self):
        import inspect

        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator as C,
        )

        src = inspect.getsource(C._add_another)
        assert src.index("_add_refused_while_batch") < src.index(
            "self._add_review = False")
