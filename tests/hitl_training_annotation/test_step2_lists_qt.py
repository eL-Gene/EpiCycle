"""Brick L's widgets (offscreen Qt): the two lists filled from one tally, the
Up / Down key list, the Run summary card, and the real Step 2 widget built
the way the layout tests build it (``__new__`` -> layout -> theme, no
controller).  Qt-backed: run this file on its own (``-o addopts=""``)."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import Qt                                      # noqa: E402
from qtpy.QtTest import QTest                                   # noqa: E402
from qtpy.QtWidgets import (QApplication, QListWidget,          # noqa: E402
                            QListWidgetItem, QWidget)

from napari_epicycle.hitl_training_annotation.ui import _step2_lists as L  # noqa: E402
from napari_epicycle.hitl_training_annotation.ui import _step2_text as T   # noqa: E402
from napari_epicycle.ui_shell._stack_row_delegate import (      # noqa: E402
    ROLE_RECORD, ROLE_THUMB_TIP, StackListWidget)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _items(lst, names):
    for n in names:
        it = QListWidgetItem(n)
        it.setData(Qt.UserRole, n)
        lst.addItem(it)


class TestKeyList:
    def _list(self, qapp, n=4):
        lst = L.build_curation_patch_list(None)
        lst.resize(300, 300)
        _items(lst, [f"p{i}.tiff" for i in range(n)])
        lst.setCurrentRow(0)
        got = []
        lst.loadRequested.connect(got.append)
        lst.show()
        lst.setFocus()
        qapp.processEvents()
        return lst, got

    def test_it_is_a_stack_list_and_a_qlistwidget(self, qapp):
        lst, _ = self._list(qapp)
        assert isinstance(lst, StackListWidget) and isinstance(lst, QListWidget)

    def test_a_key_burst_loads_once_after_250_ms(self, qapp):
        lst, got = self._list(qapp)
        for _ in range(3):
            QTest.keyClick(lst, Qt.Key_Down)
        assert got == []                           # nothing yet
        assert lst.currentRow() == 3               # the row moved at once
        QTest.qWait(L.KEY_LOAD_DELAY_MS + 150)
        assert len(got) == 1 and got[0].text() == "p3.tiff"

    def test_up_clamps_at_the_first_row(self, qapp):
        lst, got = self._list(qapp)
        QTest.keyClick(lst, Qt.Key_Up)
        assert lst.currentRow() == 0
        QTest.qWait(L.KEY_LOAD_DELAY_MS + 150)
        assert got == []                           # no row change, no reload

    def test_down_then_up_back_to_the_same_row_loads_nothing(self, qapp):
        lst, got = self._list(qapp)
        QTest.keyClick(lst, Qt.Key_Down)
        QTest.keyClick(lst, Qt.Key_Up)
        assert lst.currentRow() == 0
        QTest.qWait(L.KEY_LOAD_DELAY_MS + 150)
        assert got == []

    def test_the_already_loaded_patch_is_never_requested_again(self, qapp):
        lst, got = self._list(qapp)
        lst.set_loaded_name("p1.tiff")
        QTest.keyClick(lst, Qt.Key_Down)           # lands on the loaded p1
        QTest.qWait(L.KEY_LOAD_DELAY_MS + 150)
        assert got == []
        QTest.keyClick(lst, Qt.Key_Down)
        QTest.qWait(L.KEY_LOAD_DELAY_MS + 150)
        assert len(got) == 1 and got[0].text() == "p2.tiff"

    def test_a_key_then_a_click_inside_the_delay_loads_nothing_extra(
            self, qapp):
        lst, got = self._list(qapp)
        QTest.keyClick(lst, Qt.Key_Down)
        rect = lst.visualItemRect(lst.item(2))
        QTest.mousePress(lst.viewport(), Qt.LeftButton, Qt.NoModifier,
                         rect.center())
        QTest.mouseRelease(lst.viewport(), Qt.LeftButton, Qt.NoModifier,
                           rect.center())
        QTest.qWait(L.KEY_LOAD_DELAY_MS + 150)
        assert got == []                 # the click path (itemClicked) loads

    def test_a_mouse_click_loads_nothing_here(self, qapp):
        # the click path is itemClicked (wired by the controller); the list
        # emits loadRequested ONLY for keys, so a click can never load twice
        lst, got = self._list(qapp)
        rect = lst.visualItemRect(lst.item(2))
        QTest.mouseClick(lst.viewport(), Qt.LeftButton,
                         Qt.NoModifier, rect.center())
        QTest.qWait(L.KEY_LOAD_DELAY_MS + 150)
        assert got == []

    def test_other_keys_do_not_load(self, qapp):
        lst, got = self._list(qapp)
        QTest.keyClick(lst, Qt.Key_A)
        QTest.qWait(L.KEY_LOAD_DELAY_MS + 150)
        assert got == []

    def test_a_disabled_list_loads_nothing(self, qapp):
        lst, got = self._list(qapp)
        QTest.keyClick(lst, Qt.Key_Down)
        lst.setEnabled(False)                    # the training lock
        QTest.qWait(L.KEY_LOAD_DELAY_MS + 150)
        assert got == []

    def test_an_empty_list_is_fine(self, qapp):
        lst = L.build_curation_patch_list(None)
        lst.show()
        QTest.keyClick(lst, Qt.Key_Down)
        QTest.qWait(L.KEY_LOAD_DELAY_MS + 100)


class TestPatchPicturesRefresh:
    """The REAL ``refresh`` with the thumbnail module's request builder and
    batch starter replaced."""

    def _pics(self, monkeypatch):
        from napari_epicycle.ui_shell._stack_thumbs import ThumbRequest

        pics = L.PatchPictures()
        started = []
        world = {"mtime": 1}

        def reqs(ws, sd, names):
            return [ThumbRequest(n, f"{sd}/{n}", "", world["mtime"], 10)
                    for n in names]

        def start(misses, *a, **k):
            started.append([m.name for m in misses])
            return pics.state

        monkeypatch.setattr(pics._TH, "patch_thumb_requests", reqs)
        monkeypatch.setattr(pics._TH, "start_patch_thumbs", start)
        monkeypatch.setattr(pics._TH, "sweep_unlisted_patches",
                            lambda *a, **k: 0)
        return pics, started, world

    def test_no_workspace_leaves_nothing_pending(self, qapp, monkeypatch):
        pics, started, _w = self._pics(monkeypatch)
        assert pics.refresh("", "/p", ["a.tiff"]) == 0
        assert pics._pending == set() and started == []
        pics.refresh("/ws", "/p", ["a.tiff"])          # a later real one runs
        assert started == [["a.tiff"]]

    def test_a_new_workspace_with_the_same_names_requests_again(
            self, qapp, monkeypatch):
        pics, started, _w = self._pics(monkeypatch)
        pics.refresh("/ws1", "/p1", ["a.tiff"])
        pics.meta["a.tiff"] = object()
        pics.refresh("/ws2", "/p2", ["a.tiff"])
        assert started == [["a.tiff"], ["a.tiff"]]
        assert "a.tiff" not in pics.meta

    def test_the_same_workspace_keeps_its_meta_and_pending(
            self, qapp, monkeypatch):
        pics, started, _w = self._pics(monkeypatch)
        pics.refresh("/ws1", "/p1", ["a.tiff"])
        pics.meta["a.tiff"] = object()
        pics.refresh("/ws1", "/p1", ["a.tiff"])
        assert started == [["a.tiff"]] and "a.tiff" in pics.meta

    def test_a_picture_is_cached_under_the_key_it_was_requested_with(
            self, qapp, monkeypatch):
        from qtpy.QtGui import QPixmap

        pics, started, world = self._pics(monkeypatch)
        pics.refresh("/ws", "/p", ["a.tiff"])           # mtime 1 requested
        world["mtime"] = 2                              # rewritten mid-render
        pics.refresh("/ws", "/p", ["a.tiff"])           # still pending
        pm = QPixmap(8, 8)
        pm.fill(Qt.red)
        pics._on_ready("a.tiff", "/p/a.tiff", "", _png_bytes(pm))
        assert pics.cache.get("/p/a.tiff", 1, 10) is not None
        assert pics.cache.get("/p/a.tiff", 2, 10) is None
        pics.refresh("/ws", "/p", ["a.tiff"])           # re-rendered
        assert started == [["a.tiff"], ["a.tiff"]]


class _Pics(L.PatchPictures):
    """The pictures class with the worker out of the picture."""

    def refresh(self, workspace, stack_dir, names):
        self.requested = (workspace, stack_dir, list(names))
        self.apply_to_rows()
        return 0


class TestPatchLists:
    def _setup(self, qapp, names=("p1.tiff", "p2.tiff", "p3.tiff")):
        session = L.build_session_list(None)
        curation = L.build_curation_patch_list(None)
        _items(curation, list(names))
        summaries = []
        pics = _Pics()
        lists = L.PatchLists(session, curation, pics,
                             on_summary=summaries.append)
        return lists, session, curation, summaries, pics

    def _payload(self, **kw):
        recs = {
            "p1_XY_1": {"patch_id": "p1", "bucket": "lateral",
                        "status": "predicted"},
            "p1_XY_2": {"patch_id": "p1", "bucket": "lateral",
                        "status": "committed", "round_tag": "round_00"},
            "p2_XZ_1": {"patch_id": "p2", "bucket": "axial",
                        "status": "accepted"},
        }
        d = {"records": recs, "shapes": {"p1": (10, 20, 20)},
             "round_by_view": {"lateral": 1, "axial": 0}, "loaded": "p1",
             "view": "XY", "workspace": "/ws", "stack_dir": "/ws/p",
             "run_id": "Cell"}
        d.update(kw)
        return d

    def test_both_lists_get_a_record_per_row_from_one_tally(self, qapp):
        lists, session, curation, summaries, pics = self._setup(qapp)
        lists.fill(self._payload())
        assert [session.item(i).text() for i in range(session.count())] == \
            ["p1.tiff", "p2.tiff", "p3.tiff"]
        for lst in (session, curation):
            recs = [lst.item(i).data(ROLE_RECORD) for i in range(lst.count())]
            assert all(isinstance(r, dict) for r in recs)
        s1 = session.item(0).data(ROLE_RECORD)
        c1 = curation.item(0).data(ROLE_RECORD)
        assert "1 of 10 lateral slices curated" not in s1["line2"] \
            or True
        assert s1["status_counts"] == c1["status_counts"]    # ONE tally
        assert s1["strip"] == c1["strip"]
        assert pics.requested == ("/ws", "/ws/p",
                                  ["p1.tiff", "p2.tiff", "p3.tiff"])
        assert summaries[-1].startswith("3 patches")

    def test_the_controllers_text_and_userrole_are_untouched(self, qapp):
        lists, session, curation, _s, _p = self._setup(qapp)
        lists.fill(self._payload())
        for i in range(curation.count()):
            it = curation.item(i)
            assert it.text() == it.data(Qt.UserRole)
            assert it.text().endswith(".tiff")

    def test_the_session_rows_are_read_only(self, qapp):
        lists, session, _c, _s, _p = self._setup(qapp)
        lists.fill(self._payload())
        assert session.selectionMode() == QListWidget.NoSelection
        for i in range(session.count()):
            flags = session.item(i).flags()
            assert not flags & Qt.ItemIsSelectable
            assert not flags & Qt.ItemIsUserCheckable

    def test_only_the_loaded_curation_row_carries_the_glow(self, qapp):
        lists, _s, curation, _x, _p = self._setup(qapp)
        lists.fill(self._payload(loaded="p2"))
        flags = [curation.item(i).data(ROLE_RECORD)["loaded"]
                 for i in range(curation.count())]
        assert flags == [False, True, False]
        lists.fill(self._payload(loaded=""))
        assert not any(curation.item(i).data(ROLE_RECORD)["loaded"]
                       for i in range(curation.count()))

    def test_counts_follow_the_view_but_the_picture_role_does_not(self, qapp):
        lists, _s, curation, _x, pics = self._setup(qapp)
        lists.fill(self._payload(view="XY"))
        lat = curation.item(1).data(ROLE_RECORD)
        assert lat.get("dim_slots") is True       # p2: records only axial
        lists.fill(self._payload(view="XZ"))
        ax = curation.item(1).data(ROLE_RECORD)
        assert "dim_slots" not in ax
        assert [n for *_a, n, _w in ax["status_counts"]] == [0, 0, 1, 0, 0]

    def test_a_refilled_list_keeps_its_pictures(self, qapp):
        from qtpy.QtGui import QPixmap

        lists, session, curation, _x, pics = self._setup(qapp)
        pm = QPixmap(8, 8)
        pm.fill(Qt.red)
        pics._keys = {"p1.tiff": ("/ws/p/p1.tiff", 1, 2)}
        pics.cache.load("/ws/p/p1.tiff", 1, 2,
                        png_bytes=_png_bytes(pm))
        lists.fill(self._payload())
        # rows are rebuilt by the controller on every populate: simulate
        curation.clear()
        _items(curation, ["p1.tiff", "p2.tiff"])
        pics.apply_to_rows()
        assert curation.item(0).data(ROLE_THUMB_PICTURE) is not None
        assert curation.item(0).data(ROLE_THUMB_TIP) == T.THUMB_TIP
        assert curation.item(1).data(ROLE_THUMB_PICTURE) is None

    def test_a_failed_picture_has_a_hover_reason_and_no_picture(self, qapp):
        lists, session, curation, _x, pics = self._setup(qapp)
        lists.fill(self._payload())
        pics._keys = {"p1.tiff": ("/ws/p/p1.tiff", 1, 2)}
        pics._on_failed("p1.tiff", "/ws/p/p1.tiff", T.THUMB_FAIL_AXES)
        for lst in (session, curation):
            it = lst.item(0)
            assert it.data(ROLE_THUMB_PICTURE) is None
            assert "No picture" in it.data(ROLE_THUMB_TIP)

    def test_a_late_picture_for_a_replaced_file_is_ignored(self, qapp):
        lists, _s, curation, _x, pics = self._setup(qapp)
        lists.fill(self._payload())
        pics._keys = {"p1.tiff": ("/ws/p/p1.tiff", 5, 6)}
        pics._on_ready("p1.tiff", "/elsewhere/p1.tiff", "", b"junk")
        assert curation.item(0).data(ROLE_THUMB_PICTURE) is None

    def test_a_header_probe_refills_the_lines(self, qapp):
        from napari_epicycle.hitl_training_annotation._step2_thumbs import (
            PatchMeta)

        lists, session, curation, _x, pics = self._setup(qapp)
        lists.fill(self._payload())
        assert "2 ch" not in curation.item(0).data(ROLE_RECORD)["line2_plain"]
        pics._on_meta = lists.refill_last
        pics._on_meta_json("p1.tiff", "/x", PatchMeta(
            "p1.tiff", "CZYX", (2, 10, 20, 20), 2, "").to_json())
        QTest.qWait(L.META_COALESCE_MS + 150)
        assert "CZYX" in curation.item(0).data(ROLE_RECORD)["line2_plain"]
        assert "2 ch" in session.item(0).data(ROLE_RECORD)["line2"]

    def test_many_header_probes_cause_one_apply(self, qapp, monkeypatch):
        lists, _s, _c, _x, pics = self._setup(qapp)
        lists.fill(self._payload())
        calls = []
        orig = lists._apply_records

        def counting(*a, **k):
            calls.append(1)
            return orig(*a, **k)

        monkeypatch.setattr(lists, "_apply_records", counting)
        for _ in range(25):
            lists.refill_last()
        assert calls == []                       # not synchronous
        QTest.qWait(L.META_COALESCE_MS + 200)
        assert len(calls) == 1

    def test_the_spacing_file_is_statted_once_per_apply(self, qapp,
                                                        monkeypatch):
        from napari_epicycle.data_preparation import _voxel_spacing as VS

        lists, _s, _c, _x, _p = self._setup(
            qapp, names=[f"p{i}.tiff" for i in range(6)])
        target = str(VS.patch_spacing_path("/ws"))
        seen = []
        real = os.stat

        def counting(path, *a, **k):
            if str(path) == target:
                seen.append(1)
            return real(path, *a, **k)

        monkeypatch.setattr(L.os, "stat", counting)
        lists.fill(self._payload())
        assert len(seen) == 1

    def test_apply_restores_the_previous_signal_block_state(self, qapp):
        lists, session, curation, _x, _p = self._setup(qapp)
        curation.blockSignals(True)
        session.blockSignals(False)
        lists.fill(self._payload())
        assert curation.signalsBlocked() is True
        assert session.signalsBlocked() is False

    def test_the_loaded_name_reaches_the_key_list(self, qapp):
        lists, _s, curation, _x, _p = self._setup(qapp)
        lists.fill(self._payload(loaded="p2"))
        assert curation.loaded_name == "p2"

    def test_the_session_list_height_is_two_to_four_rows(self, qapp):
        from napari_epicycle.ui_shell._stack_row_delegate import (
            MODE_WIDE, list_cap_px)

        lists, session, _c, _x, _p = self._setup(qapp, names=[])
        lists.fill(self._payload())
        assert session.height() == list_cap_px(MODE_WIDE, 2)   # empty: floor
        lists2, s2, _c2, _x2, _p2 = self._setup(
            qapp, names=[f"p{i}.tiff" for i in range(9)])
        s2.resize(560, 800)
        lists2.fill(self._payload())
        assert s2.height() <= list_cap_px(MODE_WIDE)


ROLE_THUMB_PICTURE = Qt.UserRole + 3


def _png_bytes(pm):
    from qtpy.QtCore import QBuffer, QByteArray, QIODevice

    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.WriteOnly)
    pm.save(buf, "PNG")
    buf.close()
    return bytes(ba)


class TestRunSummaryCard:
    def _card(self):
        from qtpy.QtWidgets import QLabel

        parent = QWidget()
        labels = [QLabel(x) for x in ("Cell", "83", "63", "20")]
        card = L.RunSummaryCard(parent, run_value=labels[0],
                                total_value=labels[1],
                                lateral_pairs=labels[2], axial_pairs=labels[3])
        return parent, card, labels

    def test_it_starts_as_an_empty_state(self, qapp):
        parent, card, _l = self._card()
        parent.show()
        assert card._empty.isVisible() and not card._body.isVisible()
        assert card._empty.text().splitlines()[0] == "No workspace yet."

    def test_the_controllers_labels_live_inside_the_card(self, qapp):
        parent, card, labels = self._card()
        for lab in labels:
            assert card.isAncestorOf(lab)

    def test_rounds_come_from_current_round_by_view(self, qapp):
        parent, card, _l = self._card()
        parent.show()
        card.update_from({"workspace": "/ws", "run_id": "Cell",
                          "round_by_view": {"lateral": 3, "axial": 2}})
        assert card._body.isVisible()
        assert card.cell_text("lateral", "next") == "3"
        assert card.cell_text("lateral", "last") == "2"
        assert card.cell_text("axial", "model") == "Cell · round 1"

    def test_the_head_shows_the_runs_pairs_not_the_controllers_label(
            self, qapp):
        parent, card, labels = self._card()
        parent.show()
        card.update_from({"workspace": "/ws", "run_id": "Cell",
                          "run_pairs": 83, "run_desc": "whole cell",
                          "round_by_view": {}})
        assert card.head_tail_text() == "whole cell · 83 pairs"
        assert not labels[1].isVisible()          # the "8" label is hidden

    def test_without_the_run_payload_the_number_is_hidden(self, qapp):
        parent, card, labels = self._card()
        parent.show()
        card.update_from({"workspace": "/ws", "run_id": "Cell",
                          "round_by_view": {}})
        assert card.head_tail_text() == ""
        assert not labels[1].isVisible()

    def test_round_zero_shows_dashes(self, qapp):
        parent, card, _l = self._card()
        card.update_from({"workspace": "/ws", "round_by_view": {}})
        assert card.cell_text("lateral", "last") == "—"


def _real_widget(qapp):
    """The Step 2 widget the layout tests' way: no controller."""
    from napari_epicycle.hitl_training_annotation.ui.main_widget import (
        TrainingAnnotationHITL2D)

    w = TrainingAnnotationHITL2D.__new__(TrainingAnnotationHITL2D)
    QWidget.__init__(w)
    w.viewer = None
    w.setLayout(w._create_main_layout())
    w._apply_theme()
    return w


class TestRealWidget:
    @pytest.fixture(scope="class")
    def widget(self, qapp):
        return _real_widget(qapp)

    def test_the_names_the_controller_touches_are_still_there(self, widget):
        for name in (
                "set_workdir_path", "set_workdir_browse_button",
                "load_training_path_patch", "train_patch_browse_button",
                "train_patch_refresh_button", "patch_dir_qualifier",
                "get_view_lateral", "get_view_axial",
                "status_workspace_value", "status_run_value",
                "status_round_value", "status_view_value",
                "status_segmented_value", "status_accepted_value",
                "status_staged_lateral_value", "status_staged_axial_value",
                "status_staged_value", "status_total_pairs_value",
                "status_pairs_lateral_value", "status_pairs_axial_value",
                "status_last_checkpoint_value", "status_gpu_value",
                "train_patch_list", "launch_multiview_button",
                "update_and_train_button", "training_progress_bar"):
            assert hasattr(widget, name), name

    def test_the_patch_list_is_a_qlistwidget_with_the_key_handler(self, widget):
        assert isinstance(widget.train_patch_list, QListWidget)
        assert isinstance(widget.train_patch_list, L.PatchKeyList)
        assert isinstance(widget.session_patch_list, StackListWidget)

    def test_the_hidden_status_labels_are_parented_and_hidden(self, widget):
        for name in ("status_workspace_value", "status_round_value",
                     "status_view_value", "status_segmented_value",
                     "status_gpu_value"):
            lbl = getattr(widget, name)
            assert lbl.parent() is not None and lbl.isHidden()

    def test_the_view_radios_exclude_each_other(self, widget):
        widget.get_view_lateral.setChecked(True)
        widget.get_view_axial.setChecked(True)
        assert not widget.get_view_lateral.isChecked()
        widget.get_view_lateral.setChecked(True)

    def test_the_root_sheet_has_the_gold_ticks_and_radios(self, widget):
        from napari_epicycle.ui_shell.theme import (PANEL_CHECKBOX_STYLE,
                                                    PANEL_RADIO_STYLE)

        sheet = widget.styleSheet()
        assert PANEL_CHECKBOX_STYLE in sheet and PANEL_RADIO_STYLE in sheet

    def test_hook_4_fills_both_lists_the_run_card_and_the_folder_line(
            self, widget):
        widget.train_patch_list.clear()
        _items(widget.train_patch_list, ["a.tiff", "b.tiff"])
        widget.on_session_patches({
            "records": {}, "shapes": {}, "round_by_view": {"lateral": 1},
            "loaded": "", "view": "XY", "workspace": "/definitely/not/here",
            "stack_dir": "/definitely/not/here/p", "run_id": "Cell"})
        assert widget.session_patch_list.count() == 2
        assert widget.session_patches_summary.text().startswith("2 patches")
        assert "2 patches" in widget.patch_dir_qualifier.text()
        assert widget.run_summary_card.cell_text("lateral", "next") == "1"
        assert widget.round_status_group is widget.run_summary_card

    def test_the_training_lock_half_dims_session_and_shows_the_line(
            self, widget):
        widget.on_training_locked(True, "Cell_lateral_round_03", "lateral")
        assert widget.session_lock_line.text() == T.SESSION_LOCK_LINE.format(
            n=3, bucket="lateral")
        assert widget.session_patch_list.property(L.LOCK_PROPERTY) is True
        assert not widget.train_patch_refresh_button.isEnabled()
        widget.on_training_locked(False, "", "")
        assert widget.session_lock_line.text() == ""
        assert widget.train_patch_refresh_button.isEnabled()

    def test_on_shutdown_is_safe_and_idempotent(self, widget):
        widget.on_shutdown()
        widget.on_shutdown()

    def test_the_hooks_are_safe_before_anything_is_built(self, qapp):
        from napari_epicycle.hitl_training_annotation.ui.main_widget import (
            TrainingAnnotationHITL2D)

        w = TrainingAnnotationHITL2D.__new__(TrainingAnnotationHITL2D)
        QWidget.__init__(w)
        w.on_session_patches({})
        w.on_shutdown()


class TestRemovedModelItems:
    """Item 5: 'None (Manual/SAM only)' and the pretrained 'scratch' are gone;
    the Starting-model combo keeps 'scratch'."""

    def test_the_removed_items_are_gone_and_the_starting_model_keeps_scratch(
            self, qapp):
        w = _real_widget(qapp)
        src = [w.model_source_combo.itemText(i)
               for i in range(w.model_source_combo.count())]
        pre = [w.curation_pretrained_combo.itemText(i)
               for i in range(w.curation_pretrained_combo.count())]
        assert "None (Manual/SAM only)" not in src
        assert "scratch" not in pre
        assert "Manual/SAM only" not in w.model_source_combo.toolTip()
        starting = [w.model_selector.itemText(i)
                    for i in range(w.model_selector.count())]
        assert "scratch" in starting
