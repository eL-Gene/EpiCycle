"""Brick A: Step 2's Slices rows and Training's staged rows -- the delegate
records, the kept ``text()`` / ``UserRole``, the status words and colours,
the chips, the sort order with ``curating``, the filter-chip counts, and the
``SlicePictures`` manager's rules (misses only, nothing duplicated in flight,
the debounce, the old picture kept).  The builders are Qt-free; the list /
manager classes use the offscreen ``qapp`` (run this file on its own:
``-o addopts=""``)."""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from qtpy.QtCore import Qt

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from napari_epicycle.hitl_training_annotation import _step2_thumbs as TH
from napari_epicycle.hitl_training_annotation.ui import _step2_lists as L
from napari_epicycle.hitl_training_annotation.ui import _step2_text as T
from napari_epicycle.ui_shell import _stack_thumbs as S
from napari_epicycle.ui_shell._stack_row_delegate import (
    ROLE_RECORD, ROLE_THUMB, ROLE_THUMB_TIP)


def _rec(status="predicted", view="XY", idx=31, patch="WholeScale_patch_4",
         bucket=None, **kw):
    r = {"view": view, "slice_index": idx, "patch_id": patch,
         "status": status,
         "bucket": bucket or ("lateral" if view == "XY" else "axial"),
         "updated_at": f"2026-10-12T10:{idx:02d}:00"}
    r.update(kw)
    return r


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


# ---------------------------------------------------------------------------
# the Slices row
# ---------------------------------------------------------------------------

class TestSliceRow:
    def test_the_keys(self):
        row = L.build_slice_row("WholeScale_patch_4_XY_31", _rec("accepted"))
        assert row["name"] == "XY slice 31"
        assert row["thumb_px"] == 56 == L.SLICE_ROW_THUMB_PX
        assert row["word"] == "accepted" and row["dots"] == ""
        assert row["colour"] == "#6abf69" == row["icon_colour"]
        assert row["icon"] == "mdi6.check-circle-outline"
        assert row["axes"] == "" and "chips" not in row

    @pytest.mark.parametrize("status,word,colour,icon", [
        ("predicted", "predicted", "#d4a843", "mdi6.layers-outline"),
        ("curating", "curating", "#a78bca", "mdi6.pencil-outline"),
        ("accepted", "accepted", "#6abf69", "mdi6.check-circle-outline"),
        ("staged", "finalized", "#80deea", "mdi6.diamond-stone"),
        ("committed", "trained", "#8c9eff", "mdi6.brain"),
    ])
    def test_each_status_word_colour_and_icon(self, status, word, colour,
                                              icon):
        row = L.build_slice_row("a_XY_1", _rec(status, idx=1, round_tag="3"))
        assert (row["word"], row["colour"], row["icon"]) == (word, colour,
                                                              icon)

    def test_a_status_outside_the_table_is_grey(self):
        row = L.build_slice_row("a_XY_1", _rec("rejected", idx=1))
        assert row["word"] == "rejected" and row["colour"] == "#777777"
        assert row["icon"] == "mdi6.circle-outline"

    def test_names_for_the_three_views(self):
        for sid, view, idx, want in (("p_XZ_12", "XZ", 12, "XZ slice 12"),
                                     ("p_YZ_7", "YZ", 7, "YZ slice 7"),
                                     ("p_XY_0", "XY", 0, "XY slice 0")):
            assert L.slice_name(sid, _rec(view=view, idx=idx)) == want
        # a record without view / index: parsed from the id
        assert L.slice_name("my_patch_XZ_40", {"status": "x"}) == "XZ slice 40"
        assert L.slice_name("garbage", {}) == "garbage"

    def test_the_round_chip_is_the_disks_number_and_trained_only(self):
        row = L.build_slice_row("a_XY_1", _rec("committed", idx=1,
                                               round_tag="round_00"))
        assert row["chips"] == (("round_00", False),)
        for tag, want in (("round_07", "round_07"), ("2", "round_02"),
                          ("lateral_3", "round_03")):
            r = L.build_slice_row("a_XY_1", _rec("committed", idx=1,
                                                 round_tag=tag))
            assert r["chips"][0][0] == want
        for status in ("predicted", "curating", "accepted", "staged"):
            r = L.build_slice_row("a_XY_1", _rec(status, idx=1,
                                                 round_tag="round_01"))
            assert "chips" not in r
        # a trained slice with no readable tag: no chip, no crash
        assert "chips" not in L.build_slice_row(
            "a_XY_1", _rec("committed", idx=1, round_tag="-"))

    def test_the_imported_chip(self):
        for key in ("inherited_from_pair", "inherited_from_external"):
            r = L.build_slice_row("a_XY_1", _rec("accepted", idx=1,
                                                 **{key: {"x": 1}}))
            assert r["chips"] == (("imported", False),)
        r = L.build_slice_row("a_XY_1", _rec(
            "committed", idx=1, round_tag="round_01",
            inherited_from_pair={"x": 1}))
        assert [c[0] for c in r["chips"]] == ["round_01", "imported"]

    def test_the_text_of_a_row_is_todays(self):
        from napari_epicycle.hitl_training_annotation.controller import (
            training_annotation_controller as C)

        cls = C.TrainingAnnotationController
        fake = SimpleNamespace(_STATUS_DOTS=cls._STATUS_DOTS)
        for rec in (_rec("predicted"), _rec("committed", round_tag="round_01"),
                    _rec("accepted", round_tag="3",
                         inherited_from_pair={"a": 1}),
                    {"sample_id": "x_y_XZ_5", "view": "XZ",
                     "slice_index": 5, "status": "curating"}):
            assert L.format_queue_row(rec) == cls._format_queue_row(fake, rec)


# ---------------------------------------------------------------------------
# the staged row
# ---------------------------------------------------------------------------

class TestStagedRow:
    META = TH.SliceMeta("p_XY_26", 57, 2, (240, 240), True, "pair")

    def test_the_keys(self):
        row = L.build_staged_row("WholeScale_patch_4_XY_26",
                                 _rec("staged", idx=26), meta=self.META)
        assert row["name"] == "XY slice 26"
        assert row["line2_plain"] == "WholeScale_patch_4"
        assert row["shape_text"] == "240 × 240 · 2 ch · 57 labels"
        assert row["thumb_px"] == 56
        assert row["icon"] == "mdi6.diamond-stone"
        assert row["colour"] == "#80deea" and row["word"] == ""

    def test_line_three_waits_for_the_worker(self):
        row = L.build_staged_row("p_XY_26", _rec("staged", idx=26, patch="p"))
        assert "shape_text" not in row

    def test_one_label_and_no_mask(self):
        one = TH.SliceMeta("x", 1, 1, (10, 20), True, "pair")
        assert L.staged_shape_line(one) == "10 × 20 · 1 ch · 1 label"
        none = TH.SliceMeta("x", 0, 3, (10, 20), False, "pair")
        assert L.staged_shape_line(none) == "10 × 20 · 3 ch"
        unknown = TH.SliceMeta("x", -1, 2, (), True, "pair")
        assert L.staged_shape_line(unknown) == ""
        assert L.staged_shape_line(None) == ""

    def test_the_patch_is_parsed_when_the_record_has_none(self):
        row = L.build_staged_row("some_patch_XZ_9", {"status": "staged"})
        assert row["line2_plain"] == "some_patch" and \
            row["name"] == "XZ slice 9"


# ---------------------------------------------------------------------------
# sort order and chip counts
# ---------------------------------------------------------------------------

class TestOrderAndCounts:
    def test_the_order_is_trained_finalized_accepted_curating_predicted(self):
        recs = {f"p_XY_{i}": _rec(s, idx=i)
                for i, s in enumerate(("predicted", "curating", "accepted",
                                       "staged", "committed", "predicted"),
                                      start=1)}
        order = L.sorted_slice_ids(recs)
        assert [recs[s]["status"] for s in order] == [
            "committed", "staged", "accepted", "curating", "predicted",
            "predicted"]
        # ties by updated_at, oldest first (today's key)
        assert order[-2:] == ["p_XY_1", "p_XY_6"]

    def test_curating_ranks_above_predicted_and_below_accepted(self):
        r = L.SLICE_RANK
        assert r["committed"] > r["staged"] > r["accepted"] > r["curating"] \
            > r["predicted"] > 0

    def test_ids_subset_and_unknown_status_last(self):
        recs = {"a_XY_1": _rec("rejected", idx=1),
                "a_XY_2": _rec("predicted", idx=2),
                "a_XY_3": _rec("committed", idx=3)}
        assert L.sorted_slice_ids(recs) == ["a_XY_3", "a_XY_2", "a_XY_1"]
        assert L.sorted_slice_ids(recs, ["a_XY_2", "nope", "a_XY_3"]) == [
            "a_XY_3", "a_XY_2"]

    def test_filter_chip_counts(self):
        recs = {}
        for i, s in enumerate(["predicted"] * 3 + ["curating", "accepted"]
                              + ["staged", "committed"], start=1):
            recs[f"P_XY_{i}"] = _rec(s, idx=i, patch="P")
        recs["P_XZ_1"] = _rec("predicted", view="XZ", idx=1, patch="P")
        recs["Q_XY_1"] = _rec("accepted", idx=1, patch="Q")
        c = L.filter_chip_counts(recs, bucket="lateral", patch="P")
        assert c == {"all": 7, "predicted": 3, "curating": 1, "accepted": 1,
                     "staged": 2}
        ax = L.filter_chip_counts(recs, bucket="axial", patch="P")
        assert ax["all"] == 1 and ax["predicted"] == 1
        every = L.filter_chip_counts(recs, bucket="lateral")
        assert every["all"] == 8 and every["accepted"] == 2
        assert L.filter_chip_counts({}, bucket="lateral") == {
            "all": 0, "predicted": 0, "curating": 0, "accepted": 0,
            "staged": 0}

    def test_the_chip_counts_skip_an_unknown_status(self):
        recs = {"P_XY_1": _rec("rejected", idx=1, patch="P"),
                "P_XY_2": _rec("accepted", idx=2, patch="P")}
        assert L.filter_chip_counts(recs, bucket="lateral",
                                    patch="P")["all"] == 1

    def test_the_chip_texts(self):
        texts = L.filter_chip_texts({"all": 7, "predicted": 3, "curating": 1,
                                     "accepted": 1, "staged": 2})
        assert texts == {"all": "All 7", "predicted": "Predicted 3",
                         "curating": "Curating 1", "accepted": "Accepted 1",
                         "staged": "Finalized 2"}


# ---------------------------------------------------------------------------
# the two lists (Qt)
# ---------------------------------------------------------------------------

class TestLists:
    def _fill(self, lst, ids, records):
        from qtpy.QtCore import Qt

        for sid in ids:
            lst.addItem(L.format_queue_row(records[sid]))
            lst.item(lst.count() - 1).setData(Qt.UserRole, sid)

    def test_text_and_userrole_are_kept(self, qapp):
        from qtpy.QtCore import Qt

        recs = {"p_XY_1": _rec("accepted", idx=1, patch="p"),
                "p_XY_2": _rec("committed", idx=2, patch="p",
                               round_tag="round_00")}
        lst = L.build_slice_list(None)
        self._fill(lst, list(recs), recs)
        before = [(lst.item(i).text(), lst.item(i).data(Qt.UserRole))
                  for i in range(lst.count())]
        assert L.apply_slice_records(lst, recs) == 2
        after = [(lst.item(i).text(), lst.item(i).data(Qt.UserRole))
                 for i in range(lst.count())]
        assert before == after
        rec0 = lst.item(0).data(ROLE_RECORD)
        assert rec0["name"] == "XY slice 1" and rec0["word"] == "accepted"
        assert lst.item(1).data(ROLE_RECORD)["chips"][0][0] == "round_00"

    def test_the_staged_list_keeps_the_sample_id_text(self, qapp):
        recs = {"p_XY_26": _rec("staged", idx=26, patch="p")}
        lst = L.build_staged_list(None)
        lst.addItem("p_XY_26")
        from qtpy.QtCore import Qt

        lst.item(0).setData(Qt.UserRole, "p_XY_26")
        meta = TH.SliceMeta("p_XY_26", 57, 2, (240, 240), True, "pair")
        L.apply_slice_records(lst, recs, staged=True,
                              metas={"p_XY_26": meta})
        assert lst.item(0).text() == "p_XY_26"          # `_train_model` reads this
        assert lst.item(0).data(ROLE_RECORD)["shape_text"].endswith(
            "57 labels")

    def test_the_slice_list_selects_many_like_today(self, qapp):
        from qtpy.QtWidgets import QListWidget

        assert L.build_slice_list(None).selectionMode() == \
            QListWidget.ExtendedSelection

    def test_a_row_without_a_record_is_left_alone(self, qapp):
        lst = L.build_slice_list(None)
        lst.addItem("orphan")
        assert L.apply_slice_records(lst, {}) == 0
        assert lst.item(0).data(ROLE_RECORD) is None


# ---------------------------------------------------------------------------
# SlicePictures
# ---------------------------------------------------------------------------

class _FakeSignal:
    def __init__(self):
        self.fns = []

    def connect(self, fn):
        self.fns.append(fn)


class _FakeTimer:
    def __init__(self):
        self.timeout = _FakeSignal()
        self.running = False

    def setInterval(self, ms):
        self.interval = ms

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def fire(self):
        self.running = False
        for fn in self.timeout.fns:
            fn()


def _write_pair(d, sid):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    ip, mp = d / f"{sid}_img.tif", d / f"{sid}_mask.tif"
    tifffile.imwrite(str(ip), np.ones((1, 20, 30), np.float32))
    tifffile.imwrite(str(mp), np.eye(20, 30, dtype=np.int32))
    return ip, mp


def _png_bytes():
    return S.encode_png(np.full((8, 12, 3), 90, np.uint8), 56)


@pytest.fixture
def pics(qapp, tmp_path, monkeypatch):
    """A SlicePictures whose worker is replaced by a recorder (no thread)."""
    timer = _FakeTimer()
    asked = []

    def fake_submit(requests, cache_dir, on_ready, on_failed, on_meta, *,
                    state, on_finished=None, on_error=None):
        asked.append([r.name for r in requests])
        return len(requests)

    monkeypatch.setattr(TH, "submit_slice_thumbs", fake_submit)
    p = L.SlicePictures(timer_factory=lambda: timer)
    p.timer, p.asked, p.tmp = timer, asked, tmp_path
    return p


def _setup(pics, qapp, status="accepted", sid="P_XY_3"):
    ws = pics.tmp / "ws"
    ip, mp = _write_pair(ws / "acc", sid)
    recs = {sid: _rec(status, idx=3, patch="P", img_uri=str(ip),
                      mask_uri=str(mp))}
    lst = L.build_slice_list(None)
    lst.addItem(sid)
    from qtpy.QtCore import Qt

    lst.item(0).setData(Qt.UserRole, sid)
    pics.lists = [lst]
    return str(ws), recs, lst, ip, mp


def _answer(pics, sid):
    """The worker's answer for *sid* (ready), as the main thread gets it."""
    req = pics._reqs[sid]
    pics._on_ready(sid, req.path, "", _png_bytes())


class TestSlicePictures:
    def test_misses_are_requested_once_and_never_duplicated(self, pics, qapp):
        ws, recs, lst, ip, mp = _setup(pics, qapp)
        assert pics.refresh(ws, recs, ["P_XY_3"], loaded_patch="P") == 1
        assert pics.asked == [["P_XY_3"]]
        # a second refresh while it is in flight queues nothing
        assert pics.refresh(ws, recs, ["P_XY_3"], loaded_patch="P") == 0
        assert pics.asked == [["P_XY_3"]]
        assert len(pics.inflight) == 1
        tip = lst.item(0).data(ROLE_THUMB_TIP)
        assert tip == T.SLICE_THUMB_PENDING
        assert lst.item(0).data(ROLE_THUMB) is None

    def test_a_ready_picture_is_set_and_a_rebuilt_row_keeps_it(self, pics,
                                                               qapp):
        from qtpy.QtCore import Qt

        ws, recs, lst, ip, mp = _setup(pics, qapp)
        pics.refresh(ws, recs, ["P_XY_3"], loaded_patch="P")
        _answer(pics, "P_XY_3")
        assert lst.item(0).data(ROLE_THUMB) is not None
        assert lst.item(0).data(ROLE_THUMB_TIP) == T.SLICE_THUMB_TIP
        assert len(pics.inflight) == 0
        # the controller's `clear()` + re-add: pictures return SYNCHRONOUSLY
        lst.clear()
        lst.addItem("P_XY_3")
        lst.item(0).setData(Qt.UserRole, "P_XY_3")
        assert lst.item(0).data(ROLE_THUMB) is None
        assert pics.refresh(ws, recs, ["P_XY_3"], loaded_patch="P") == 0
        assert lst.item(0).data(ROLE_THUMB) is not None
        assert pics.asked == [["P_XY_3"]]                  # no new request

    def test_a_stale_answer_is_ignored(self, pics, qapp):
        ws, recs, lst, ip, mp = _setup(pics, qapp)
        pics.refresh(ws, recs, ["P_XY_3"], loaded_patch="P")
        pics._on_ready("P_XY_3", "some/other/path", "", _png_bytes())
        assert lst.item(0).data(ROLE_THUMB) is None
        pics._on_ready("gone_XY_1", "x", "", _png_bytes())     # no crash

    def test_a_recurated_slice_of_the_loaded_patch_waits_for_silence(
            self, pics, qapp):
        ws, recs, lst, ip, mp = _setup(pics, qapp, status="curating")
        pics.refresh(ws, recs, ["P_XY_3"], loaded_patch="P")
        _answer(pics, "P_XY_3")
        old = lst.item(0).data(ROLE_THUMB).cacheKey()
        # the user draws: the mask file changes (a new mtime) ...
        st = os.stat(mp)
        os.utime(mp, ns=(st.st_atime_ns, st.st_mtime_ns + 7_000_000_000))
        # ... and the list is refilled three times (a stroke each)
        for _ in range(3):
            assert pics.refresh(ws, recs, ["P_XY_3"], loaded_patch="P") == 0
        assert pics.asked == [["P_XY_3"]]                   # nothing new yet
        assert lst.item(0).data(ROLE_THUMB).cacheKey() == old  # old one stays
        assert pics.timer.running and pics.debounce.pending == {"P_XY_3"}
        pics.timer.fire()                                    # 1.5 s of silence
        assert pics.asked == [["P_XY_3"], ["P_XY_3"]]        # ONE request
        assert lst.item(0).data(ROLE_THUMB).cacheKey() == old  # until it lands

    def test_a_curating_slice_with_no_picture_yet_is_requested_at_once(
            self, pics, qapp):
        ws, recs, lst, ip, mp = _setup(pics, qapp, status="curating")
        assert pics.refresh(ws, recs, ["P_XY_3"], loaded_patch="P") == 1
        assert not pics.timer.running

    def test_other_patches_curating_rows_are_not_debounced(self, pics, qapp):
        ws, recs, lst, ip, mp = _setup(pics, qapp, status="curating")
        pics.refresh(ws, recs, ["P_XY_3"], loaded_patch="OTHER")
        _answer(pics, "P_XY_3")
        st = os.stat(mp)
        os.utime(mp, ns=(st.st_atime_ns, st.st_mtime_ns + 7_000_000_000))
        assert pics.refresh(ws, recs, ["P_XY_3"], loaded_patch="OTHER") == 1

    def test_note_mask_written_pokes_the_debounce(self, pics, qapp):
        ws, recs, lst, ip, mp = _setup(pics, qapp, status="curating")
        pics.refresh(ws, recs, ["P_XY_3"], loaded_patch="P")
        _answer(pics, "P_XY_3")
        st = os.stat(mp)
        os.utime(mp, ns=(st.st_atime_ns, st.st_mtime_ns + 7_000_000_000))
        pics.note_mask_written("P_XY_3")
        pics.note_mask_written("P_XY_3")
        assert pics.asked == [["P_XY_3"]]
        pics.timer.fire()
        assert pics.asked[-1] == ["P_XY_3"] and len(pics.asked) == 2

    def test_a_busy_answer_drops_the_in_flight_mark_and_keeps_the_picture(
            self, pics, qapp):
        ws, recs, lst, ip, mp = _setup(pics, qapp)
        pics.refresh(ws, recs, ["P_XY_3"], loaded_patch="P")
        _answer(pics, "P_XY_3")
        st = os.stat(mp)
        os.utime(mp, ns=(st.st_atime_ns, st.st_mtime_ns + 7_000_000_000))
        assert pics.refresh(ws, recs, ["P_XY_3"], loaded_patch="P") == 1
        assert len(pics.inflight) == 1
        busy = TH.SliceMeta("P_XY_3", -1, 1, (), True, "pair", True)
        req = pics._reqs["P_XY_3"]
        pics._on_meta_json("P_XY_3", req.path, busy.to_json())
        assert len(pics.inflight) == 0
        assert "P_XY_3" not in pics.meta
        assert lst.item(0).data(ROLE_THUMB) is not None      # the old one
        assert "P_XY_3" not in pics._failed
        assert pics.refresh(ws, recs, ["P_XY_3"], loaded_patch="P") == 1

    def test_a_failure_is_a_blank_tile_with_the_slices_words(self, pics,
                                                            qapp):
        ws, recs, lst, ip, mp = _setup(pics, qapp)
        pics.refresh(ws, recs, ["P_XY_3"], loaded_patch="P")
        req = pics._reqs["P_XY_3"]
        pics._on_failed("P_XY_3", req.path, S.THUMB_FAIL_UNREADABLE)
        assert lst.item(0).data(ROLE_THUMB) is None
        tip = lst.item(0).data(ROLE_THUMB_TIP)
        assert tip == "No picture: The slice's image file could not be read."
        # the same files are not asked for again
        assert pics.refresh(ws, recs, ["P_XY_3"], loaded_patch="P") == 0

    def test_the_meta_is_kept_for_the_staged_rows(self, pics, qapp):
        got = []

        class Owner:
            def changed(self):
                got.append(1)

        o = Owner()
        pics._on_meta = o.changed
        m = TH.SliceMeta("P_XY_3", 5, 2, (20, 30), True, "pair")
        pics._on_meta_json("P_XY_3", "x", m.to_json())
        assert pics.meta["P_XY_3"].n_labels == 5 and got == [1]

    def test_a_record_with_nothing_to_draw_is_skipped(self, pics, qapp):
        ws, recs, lst, ip, mp = _setup(pics, qapp)
        recs["P_XY_3"]["img_uri"] = ""
        assert pics.refresh(ws, recs, ["P_XY_3"], loaded_patch="P") == 0
        assert pics.asked == []

    def test_no_workspace_requests_nothing(self, pics, qapp):
        ws, recs, lst, ip, mp = _setup(pics, qapp)
        assert pics.refresh("", recs, ["P_XY_3"]) == 0

    def test_forget_patch_drops_its_pictures_and_pngs(self, pics, qapp):
        ws, recs, lst, ip, mp = _setup(pics, qapp)
        pics.refresh(ws, recs, ["P_XY_3"], loaded_patch="P")
        _answer(pics, "P_XY_3")
        d = TH.slice_thumbs_dir(ws)
        d.mkdir(parents=True, exist_ok=True)
        png = d / "P_XY_3.0badf00d.1.2.c2i.s56.png"
        png.write_bytes(b"x")
        pics.forget_patch(ws, "P")
        assert not png.exists() and "P_XY_3" not in pics._reqs
        assert "P_XY_3" not in pics._latest

    def test_stop_cancels_the_debounce(self, pics, qapp):
        ws, recs, lst, ip, mp = _setup(pics, qapp)
        pics.note_mask_written("P_XY_3")
        assert pics.stop(wait_ms=0)
        assert pics.debounce.pending == set()


# ---------------------------------------------------------------------------
# Brick A hookup: the real widget (no controller), the chips, the header, the
# Train gate, the tooltip, Forget and the workspace-open sweep
# ---------------------------------------------------------------------------

def _real_widget(qapp):
    from qtpy.QtWidgets import QWidget

    from napari_epicycle.hitl_training_annotation.ui.main_widget import (
        TrainingAnnotationHITL2D)

    w = TrainingAnnotationHITL2D.__new__(TrainingAnnotationHITL2D)
    QWidget.__init__(w)
    w.viewer = None
    w.setLayout(w._create_main_layout())
    w._apply_theme()
    return w


def _records():
    out = {}
    spec = [("predicted", 3), ("curating", 9), ("accepted", 7),
            ("staged", 5), ("committed", 2), ("predicted", 11)]
    for status, idx in spec:
        out[f"P_XY_{idx}"] = _rec(status=status, idx=idx, patch="P")
    out["P_XZ_4"] = _rec(status="predicted", view="XZ", idx=4, patch="P")
    out["Q_XY_1"] = _rec(status="predicted", idx=1, patch="Q")
    return out


class TestHookup:
    @pytest.fixture()
    def widget(self, qapp):
        return _real_widget(qapp)

    def test_both_lists_are_stack_lists_and_still_qlistwidgets(self, widget):
        from qtpy.QtWidgets import QListWidget
        from napari_epicycle.ui_shell._stack_row_delegate import (
            StackListWidget)

        for lst in (widget.segment_slice_data_list, widget.data_to_add_list):
            assert isinstance(lst, StackListWidget)
            assert isinstance(lst, QListWidget)

    def test_the_aliases_are_the_one_slice_list(self, widget):
        lst = widget.segment_slice_data_list
        assert widget.accepted_queue_list is lst
        assert widget.staged_queue_list is lst
        assert widget.sample_queue_list is lst

    def test_the_slice_list_keeps_extended_selection(self, widget):
        from qtpy.QtWidgets import QAbstractItemView

        assert widget.segment_slice_data_list.selectionMode() == \
            QAbstractItemView.ExtendedSelection

    def test_the_two_assignments_are_literal_in_the_source(self):
        import inspect
        import re
        from napari_epicycle.hitl_training_annotation.ui import main_widget

        src = inspect.getsource(main_widget)
        assert re.search(
            r"self\.segment_slice_data_list\s*=\s*L\.build_slice_list\(", src)
        assert re.search(
            r"self\.data_to_add_list\s*=\s*L\.build_staged_list\(", src)

    def test_the_pictures_manager_serves_both_lists(self, widget):
        assert widget.slice_pictures.lists == [
            widget.segment_slice_data_list, widget.data_to_add_list]

    def _fill(self, widget, ids, staged=False, **extra):
        lst = widget.data_to_add_list if staged else \
            widget.segment_slice_data_list
        lst.clear()
        for sid in ids:
            lst.addItem(sid)
            lst.item(lst.count() - 1).setData(Qt.UserRole, sid)
        payload = {"records": _records(), "ids": list(ids), "staged": staged,
                   "bucket": "lateral", "patch": "P", "loaded_patch": "P",
                   "workspace": "", "stack_paths": {}}
        payload.update(extra)
        widget.on_slice_rows(payload)

    def test_staged_rows_keep_the_sample_id_as_text_and_are_painted(
            self, widget):
        ids = ["P_XY_5", "P_XY_2"]
        self._fill(widget, ids, staged=True)
        lst = widget.data_to_add_list
        assert [lst.item(i).text() for i in range(lst.count())] == ids
        rec = lst.item(0).data(ROLE_RECORD)
        assert rec["name"] == "XY slice 5" and rec["thumb_px"] == 56
        assert rec["line2_plain"] == "P"

    def test_slice_rows_are_painted_and_keep_text_and_userrole(self, widget):
        ids = ["P_XY_5", "P_XY_9"]
        self._fill(widget, ids)
        lst = widget.segment_slice_data_list
        for i, sid in enumerate(ids):
            assert lst.item(i).data(Qt.UserRole) == sid
            assert lst.item(i).data(ROLE_RECORD)["name"].startswith("XY slice")

    def test_the_imported_row_tooltip_survives(self, widget):
        lst = widget.segment_slice_data_list
        lst.addItem("P_XY_5")
        it = lst.item(0)
        it.setData(Qt.UserRole, "P_XY_5")
        it.setToolTip("Imported from run Cell, round 2")
        widget.on_slice_rows({"records": _records(), "ids": ["P_XY_5"],
                              "staged": False, "bucket": "lateral",
                              "patch": "P", "workspace": ""})
        assert it.toolTip() == "Imported from run Cell, round 2"
        assert it.data(ROLE_RECORD)["row_tip"] == \
            "Imported from run Cell, round 2"
        # and the delegate shows it for a point on the row (not the picture)
        from qtpy.QtCore import QPoint
        from qtpy.QtWidgets import QStyleOptionViewItem

        lst.resize(300, 200)
        lst.show()
        deleg = lst.itemDelegate()
        opt = QStyleOptionViewItem()
        opt.rect = lst.visualItemRect(it)
        idx = lst.indexFromItem(it)
        tips = {deleg.tip_text(opt, idx, QPoint(x, opt.rect.top() + 4))
                for x in range(opt.rect.left() + 70, opt.rect.right(), 15)}
        assert "Imported from run Cell, round 2" in tips
        lst.hide()

    def test_the_chip_texts_are_icon_free_words_and_counts(self, widget):
        widget._apply_chip_mode(False)          # tall mode keeps the words
        self._fill(widget, ["P_XY_5"])
        assert widget.filter_all_btn.text() == "All 6"
        assert widget.filter_predicted_btn.text() == "Predicted 2"
        assert widget.filter_curating_btn.text() == "Curating 1"
        assert widget.filter_accepted_btn.text() == "Accepted 1"
        assert widget.filter_staged_btn.text() == "Finalized 2"

    def test_the_chips_follow_the_bucket_and_the_patch(self, widget):
        widget._apply_chip_mode(False)
        self._fill(widget, [], bucket="axial")
        assert widget.filter_all_btn.text() == "All 1"
        self._fill(widget, [], patch="Q")
        assert widget.filter_all_btn.text() == "All 1"
        assert widget.filter_predicted_btn.text() == "Predicted 1"

    def test_a_zero_chip_dims_and_stays_checkable(self, widget):
        self._fill(widget, [], bucket="axial")
        assert widget.filter_curating_btn.property("zero") is True
        assert widget.filter_predicted_btn.property("zero") is False
        assert widget.filter_curating_btn.isCheckable()

    def test_the_chips_stay_clickable_buttons_the_controller_reads(
            self, widget):
        for name in ("filter_all_btn", "filter_predicted_btn",
                     "filter_curating_btn", "filter_accepted_btn",
                     "filter_staged_btn", "filter_committed_btn"):
            assert hasattr(getattr(widget, name), "isChecked")
        assert widget.filter_committed_btn.isHidden()

    def test_the_slices_header_names_the_patch_and_the_view(self, widget):
        self._fill(widget, [])
        assert widget.slices_header_patch.text() == "P · lat"
        self._fill(widget, [], bucket="axial")
        assert widget.slices_header_patch.text() == "P · ax"
        self._fill(widget, [], loaded_patch="")
        assert widget.slices_header_patch.text() == ""

    def test_the_empty_state_words_follow_the_mockup(self, widget):
        assert widget.segment_slice_data_list._empty_lines == [
            "No slices in this view yet.",
            "Open the 2D annotator and Predict, or draw one."]

    def test_a_workspace_change_forgets_the_old_pictures(self, widget, tmp_path):
        pics = widget.slice_pictures
        self._fill(widget, ["P_XY_5"], workspace=str(tmp_path / "a"))
        pics._latest["x"] = object()
        self._fill(widget, ["P_XY_5"], workspace=str(tmp_path / "b"))
        assert "x" not in pics._latest

    def test_on_slice_written_feeds_the_debounce(self, widget):
        widget.on_slice_written("P_XY_5")
        assert widget.slice_pictures.debounce.pending == {"P_XY_5"} \
            if hasattr(widget.slice_pictures.debounce, "pending") else True
        widget.slice_pictures.debounce.cancel()

    def test_on_shutdown_stops_both_managers(self, widget):
        calls = []
        widget._patch_pictures.stop = lambda *a, **k: calls.append("patch")
        widget.slice_pictures.stop = lambda *a, **k: calls.append("slice")
        widget.on_shutdown()
        assert calls == ["patch", "slice"]

    def test_the_hooks_are_safe_with_nothing_built(self, qapp):
        from qtpy.QtWidgets import QWidget

        from napari_epicycle.hitl_training_annotation.ui.main_widget import (
            TrainingAnnotationHITL2D)

        w = TrainingAnnotationHITL2D.__new__(TrainingAnnotationHITL2D)
        QWidget.__init__(w)
        w.on_slice_rows({})
        w.on_slice_written("x")
        w.on_shutdown()


class TestTrainGateWithPaintedRows:
    """``_train_model`` reads ``data_to_add_list.item(i).text()``: with the
    rows painted it is still the sample id, so the gate still works."""

    def _ctrl(self, widget, tmp_path, records):
        from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
            TrainingAnnotationController as C)

        c = C.__new__(C)
        c.ui = widget
        c.sample_records = records
        c.working_dir = str(tmp_path)
        c.selected_view = "XY"
        c.view_target = "lateral"
        c.loaded_patch_name = "P"
        c._effective_stack_path = lambda: ""
        c._collect_training_config = lambda: {"model_name": "m"}
        c._flush_pending_state_timers = lambda: None
        c._save_internal_state = lambda: None
        c._training_thread = None
        c.logs = []
        c._append_ui_log = c.logs.append
        return c

    def test_no_staged_slices_still_stops_training(self, qapp, tmp_path):
        w = _real_widget(qapp)
        c = self._ctrl(w, tmp_path, {
            "P_XY_1": _rec(status="accepted", idx=1, patch="P")})
        c._train_model()
        assert any("No staged slices" in m for m in c.logs)
        assert w.data_to_add_list.count() == 0

    def test_staged_slices_read_back_as_ids_the_way_the_gate_reads_them(
            self, qapp, tmp_path):
        w = _real_widget(qapp)
        recs = {"P_XY_1": _rec(status="staged", idx=1, patch="P"),
                "P_XY_2": _rec(status="staged", idx=2, patch="P")}
        c = self._ctrl(w, tmp_path, recs)
        c._sync_data_to_add_list()      # the first lines of the gate
        lst = w.data_to_add_list
        labels = [lst.item(i).text() for i in range(lst.count())]
        assert sorted(labels) == ["P_XY_1", "P_XY_2"] and labels
        assert lst.item(0).data(ROLE_RECORD)["name"].startswith("XY slice")


def _png(folder, name, h="deadbeef"):
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{name}.{h}.1.2.png"
    p.write_bytes(b"x")
    return p


class TestForgetAndSweep:
    def test_remove_patch_pngs_takes_only_that_patch(self, tmp_path):
        d = TH.patch_thumbs_dir(tmp_path)
        a = _png(d, "patch_4.tif")
        b = _png(d, "patch_5.tif")
        c = _png(d, "patch_4_b.tif")
        s = _png(TH.slice_thumbs_dir(tmp_path), "patch_4_XY_3")
        assert TH.remove_patch_pngs(tmp_path, "patch_4") == 1
        assert not a.exists() and b.exists() and c.exists() and s.exists()

    def test_remove_patch_pngs_takes_the_file_name_form_too(self, tmp_path):
        d = TH.patch_thumbs_dir(tmp_path)
        a = _png(d, "patch_4.tif")
        assert TH.remove_patch_pngs(tmp_path, "patch_4.tif") == 1
        assert not a.exists()

    def test_remove_patch_pngs_ignores_non_worker_files(self, tmp_path):
        d = TH.patch_thumbs_dir(tmp_path)
        d.mkdir(parents=True)
        (d / "patch_4.tif.txt").write_text("keep")
        assert TH.remove_patch_pngs(tmp_path, "patch_4") == 0
        assert TH.remove_patch_pngs("", "patch_4") == 0
        assert TH.remove_patch_pngs(tmp_path, "") == 0

    def test_forget_patch_clears_both_folders_and_memory(self, qapp, tmp_path):
        w = _real_widget(qapp)
        a = _png(TH.patch_thumbs_dir(tmp_path), "patch_4.tif")
        s = _png(TH.slice_thumbs_dir(tmp_path), "patch_4_XY_3")
        keep = _png(TH.slice_thumbs_dir(tmp_path), "patch_5_XY_3")
        w.controller = SimpleNamespace(working_dir=str(tmp_path))
        w.forget_patch("patch_4.tif")
        assert not a.exists() and not s.exists() and keep.exists()

    def test_the_workspace_sweep_keeps_every_runs_samples(self, tmp_path):
        import json

        from napari_epicycle.hitl_training_annotation import _paths

        state = _paths.state_json_path(tmp_path)
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps({"schema_version": 3, "runs": {
            "A": {"sample_records": {"p_XY_1": {}}},
            "B": {"sample_records": {"p_XY_2": {}}}}}), encoding="utf-8")
        d = TH.slice_thumbs_dir(tmp_path)
        k1 = _png(d, "p_XY_1")
        k2 = _png(d, "p_XY_2")
        k3 = _png(d, "p_XY_3")        # the open run's, not on disk yet
        gone = _png(d, "p_XY_9")
        # an orphan is a PNG of an earlier session: old (a picture newer
        # than the sweep's start is kept, audit L3)
        os.utime(gone, (1, 1))
        t = TH.sweep_slices_async(str(tmp_path), ["p_XY_3"])
        t.join(5)
        assert k1.exists() and k2.exists() and k3.exists()
        assert not gone.exists()

    def test_an_unreadable_state_sweeps_nothing(self, tmp_path):
        from napari_epicycle.hitl_training_annotation import _paths

        state = _paths.state_json_path(tmp_path)
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text("{not json", encoding="utf-8")
        k = _png(TH.slice_thumbs_dir(tmp_path), "p_XY_1")
        t = TH.sweep_slices_async(str(tmp_path), [])
        t.join(5)
        assert k.exists()

    def test_the_sweep_never_touches_the_patches_folder(self, tmp_path):
        import json

        from napari_epicycle.hitl_training_annotation import _paths

        state = _paths.state_json_path(tmp_path)
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps({"schema_version": 3, "runs": {
            "A": {"sample_records": {"p_XY_1": {}}}}}), encoding="utf-8")
        pp = _png(TH.patch_thumbs_dir(tmp_path), "patch_4.tif")
        _png(TH.slice_thumbs_dir(tmp_path), "p_XY_1")
        TH.sweep_slices_async(str(tmp_path), []).join(5)
        assert pp.exists()


class TestParkedConstantsMoved:
    def test_nothing_parked_is_left_in_the_lists_or_thumbs_modules(self):
        import inspect

        for mod in (L, TH):
            assert "moved to _step2_text.py at integration" not in \
                inspect.getsource(mod)

    def test_the_words_live_in_the_text_module(self):
        for name in ("ROW_SLICE_NAME", "ROW_CHIP_IMPORTED", "ROW_LABELS",
                     "ROW_LABELS_ONE", "ROW_PLANE_DIMS", "FILTER_ALL_WORD",
                     "FILTER_CHIP_FORMAT", "SLICES_EMPTY_LINES",
                     "STAGED_EMPTY_LINES", "SLICE_THUMB_TIP",
                     "SLICE_THUMB_PENDING", "SLICE_THUMB_FAILED",
                     "SLICE_THUMB_REASONS", "SLICE_THUMB_BUSY",
                     "SLICE_THUMB_NO_SOURCE", "SLICES_HEADER_PATCH"):
            assert hasattr(T, name), name
            assert not hasattr(L, name), name
