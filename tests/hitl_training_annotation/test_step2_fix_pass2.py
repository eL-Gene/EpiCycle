"""Step 2 look, fix pass 2 (the parity audit + the Opus brick A / T audit):
narrow rows whole, the compact right column, the sublabels, the `?` on
Diagnostics, the lock-line words, the seam hardening.  Qt-backed: run on its
own with ``-o addopts=""``."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QRect                                    # noqa: E402
from qtpy.QtGui import QFontMetrics                              # noqa: E402
from qtpy.QtWidgets import QApplication, QListWidgetItem        # noqa: E402

from napari_epicycle.hitl_training_annotation.ui import _step2_lists as L  # noqa: E402
from napari_epicycle.hitl_training_annotation.ui import _step2_text as T   # noqa: E402
from napari_epicycle.ui_shell import _stack_row_delegate as D            # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _recs(*rows):
    out = {}
    for i, (patch, bucket, status) in enumerate(rows):
        sid = f"{patch}_{'XY' if bucket == 'lateral' else 'XZ'}_{i}"
        out[sid] = {"sample_id": sid, "patch_id": patch, "bucket": bucket,
                    "status": status, "round_tag": "-", "slice_index": i}
    return out


class _Meta:
    axes = "CZYX"
    n_channels = 2
    reason = ""


def _narrow_layout(qapp, rec, width=250):
    lst = L.build_curation_patch_list(None)
    deleg = lst.itemDelegate()
    return deleg, deleg.layout(QRect(0, 0, width, 200), rec, lst.font())


# 1 -- the patch name: no extension, the tail survives
def test_tail_keep_chars_names_patch_n():
    assert L.tail_keep_chars("WholeScale_patch_8") == len("patch_8")
    assert L.tail_keep_chars("WholeScale_patch_12") == len("patch_12")
    assert L.tail_keep_chars("abcdefgh") == 6


def test_the_curation_row_name_has_no_extension_and_keeps_its_tail():
    rec = L.curation_patch_record("WholeScale_patch_8.tiff", tally={})
    assert rec["name"] == "WholeScale_patch_8"
    assert rec["name_keep_tail"] == len("patch_8")


def test_elide_keep_tail_keeps_patch_n(qapp):
    lst = L.build_curation_patch_list(None)
    fm = QFontMetrics(lst.font())
    full = "WholeScale_patch_8"
    w = fm.horizontalAdvance("Whole…patch_8") + 4
    shown = D._elide_keep_tail(fm, full, w, 7)
    assert shown.endswith("patch_8") and "…" in shown
    assert shown.startswith("Whole") or shown.startswith("Wh")
    assert fm.horizontalAdvance(shown) <= w
    # a width that fits the whole name leaves it alone
    assert D._elide_keep_tail(fm, full, 10_000, 7) == full
    # too narrow for even the tail: never an exception
    D._elide_keep_tail(fm, full, 5, 7)


def test_a_painted_narrow_row_never_clips_the_patch_number(qapp):
    from qtpy.QtGui import QImage, QPainter
    from qtpy.QtWidgets import QStyleOptionViewItem
    lst = L.build_curation_patch_list(None)
    it = QListWidgetItem("WholeScale_patch_8.tiff")
    lst.addItem(it)
    rec = L.curation_patch_record("WholeScale_patch_8.tiff", tally={},
                                  shape=(124, 240, 240), meta=_Meta())
    it.setData(D.ROLE_RECORD, rec)
    seen = []
    orig = D._elide_keep_tail
    D._elide_keep_tail = lambda *a, **k: (seen.append(orig(*a, **k))
                                          or seen[-1])
    try:
        deleg = lst.itemDelegate()
        img = QImage(250, 200, QImage.Format_ARGB32)
        p = QPainter(img)
        opt = QStyleOptionViewItem()
        opt.rect = QRect(0, 0, 250, 120)
        opt.font = lst.font()
        deleg.paint(p, opt, lst.model().index(0, 0))
        p.end()
    finally:
        D._elide_keep_tail = orig
    assert seen and seen[0].endswith("patch_8")


# 2 -- line 2 fits at 250 px
def test_line2_has_shorter_spellings():
    v = L.line2_variants("CZYX · 140 × 240 × 240 · 2 ch")
    assert v[0].startswith("CZYX · 140 ×")
    assert v[1] == "CZYX · 140×240×240 · 2 ch"
    assert v[-1] == "CZYX·140×240×240·2 ch"
    assert len(set(v)) == len(v)


def test_fit_variant_walks_down_to_the_first_that_fits(qapp):
    rec = L.curation_patch_record("WholeScale_patch_8.tiff", tally={},
                                  shape=(140, 240, 240), meta=_Meta())
    lst = L.build_curation_patch_list(None)
    fm = QFontMetrics(lst.itemDelegate()._fonts(lst.font())["dim"])
    v = rec["line2_variants"]
    assert [fm.horizontalAdvance(x) for x in v] == sorted(
        (fm.horizontalAdvance(x) for x in v), reverse=True)
    for want in v:
        assert D._fit_variant(fm, v, fm.horizontalAdvance(want)) == want
    # nothing fits: the tightest one, elided
    assert D._fit_variant(fm, v, 20).endswith("…") or         D._fit_variant(fm, v, 20) == ""


def test_the_tightest_line2_is_short_enough_for_250_px():
    """At the real dock a 250 px row leaves ~154 px for line 2 (the 10 px
    mono-less dim font, ~6 px a character): 21 characters fit, the 33 of
    the spaced spelling did not (the shot clipped ``2 ch``)."""
    rec = L.curation_patch_record("WholeScale_patch_8.tiff", tally={},
                                  shape=(140, 240, 240), meta=_Meta())
    assert len(rec["line2_variants"][-1]) <= 22
    assert rec["line2_variants"][-1].endswith("2 ch")


# 3 -- the counts line is one line at 250 px
def test_narrow_row_shows_nonzero_statuses_only(qapp):
    t = L.tally_records(_recs(("p1", "lateral", "predicted"),
                              ("p1", "lateral", "predicted"),
                              ("p1", "lateral", "staged")))
    rec = L.curation_patch_record("p1.tiff", tally=t, shape=(50, 9, 9),
                                  meta=_Meta())
    assert rec["counts_compact"] is True
    deleg, g = _narrow_layout(qapp, rec, 250)
    assert len(g["counts"]) == 2                       # predicted + finalized
    assert len({r[0].y() for r in g["counts"]}) == 1   # one line


def test_a_wide_enough_row_keeps_its_dimmed_slots(qapp):
    t = L.tally_records(_recs(("p1", "lateral", "predicted")))
    rec = L.curation_patch_record("p1.tiff", tally=t, shape=(50, 9, 9),
                                  meta=_Meta())
    deleg, g = _narrow_layout(qapp, rec, 700)
    assert len(g["counts"]) == 5


def test_dim_only_row_is_icons_in_one_line_when_narrow(qapp):
    t = L.tally_records(_recs(("p1", "axial", "predicted")))
    rec = L.curation_patch_record("p1.tiff", tally=t, bucket="lateral",
                                  shape=(50, 9, 9), meta=_Meta())
    assert rec.get("dim_slots") is True
    deleg, g = _narrow_layout(qapp, rec, 250)
    assert len(g["counts"]) == 5
    assert len({r[0].y() for r in g["counts"]}) == 1


def test_records_without_the_new_keys_lay_out_as_before(qapp):
    rec = {"name": "x", "status_counts": ((("i", "#fff", 0, "w"),) * 5),
           "line2_plain": "abc", "no_icon": True}
    deleg, g = _narrow_layout(qapp, rec, 250)
    assert len(g["counts"]) == 5


# 5 -- the empty Slices message is never clipped to dots
def _grab_text_rows(lst):
    from qtpy.QtGui import QImage
    img = QImage(lst.size(), QImage.Format_ARGB32)
    img.fill(0)
    lst.render(img)
    return img


def test_short_empty_slices_list_shows_one_line_and_the_whole_on_hover(qapp):
    lst = L.build_slice_list(None)
    assert lst.empty_one_line_when_short is True
    lst.resize(250, 40)           # (polish: a taller box shows it whole)
    lst.show()
    qapp.processEvents()
    _grab_text_rows(lst)
    assert lst.viewport().toolTip() == "\n".join(T.SLICES_EMPTY_LINES)
    lst.close()


def test_a_tall_enough_empty_list_keeps_both_lines(qapp):
    lst = L.build_slice_list(None)
    lst.resize(250, 220)
    lst.show()
    qapp.processEvents()
    _grab_text_rows(lst)
    assert lst.viewport().toolTip() == ""
    lst.close()


def test_the_flag_is_opt_in_for_other_steps(qapp):
    from napari_epicycle.ui_shell._stack_row_delegate import StackListWidget
    assert StackListWidget().empty_one_line_when_short is False


# 4 -- the compact right column in short mode, the full one in tall mode
from test_step2_curation import _real_widget                      # noqa: E402


def test_short_mode_shows_the_one_line_summary_and_use_active_only(qapp):
    w = _real_widget(qapp)
    table = w.curation_progress_table
    block = w.curation_model_block
    table.refresh({"lateral": {"predicted": 62, "curating": 2,
                               "staged": 5, "committed": 4},
                   "axial": {"accepted": 9}}, "lateral", "")
    w.curation_right_column.set_short_mode(True)
    assert table.table_body.isHidden() and not table.summary_row.isHidden()
    assert table.summary_text() == (
        "Predicted 62 | Curating 2 | Accepted 0 | Finalized 5 | Trained 4")
    assert block.source_combo.isHidden() is False       # Use stays
    assert block.active_label.isHidden() is False       # Active stays
    for part in (block.pretrained_combo, block.pretrained_note,
                 block.inference_collapsible):
        assert not part.isVisibleTo(block), part
    assert block.path_row.isHidden()


def test_the_summary_follows_the_active_bucket(qapp):
    w = _real_widget(qapp)
    table = w.curation_progress_table
    counts = {"lateral": {"predicted": 1}, "axial": {"accepted": 7}}
    table.refresh(counts, "axial", "")
    assert "Accepted 7" in table.summary_text()
    assert "Predicted 0" in table.summary_text()


def test_tall_mode_brings_the_table_and_the_full_block_back(qapp):
    w = _real_widget(qapp)
    right = w.curation_right_column
    right.set_short_mode(True)
    right.set_short_mode(False)
    table, block = w.curation_progress_table, w.curation_model_block
    assert not table.table_body.isHidden() and table.summary_row.isHidden()
    assert block.pretrained_combo.isVisibleTo(block)
    assert block.inference_collapsible.isVisibleTo(block)


def test_external_path_row_returns_in_tall_mode_only(qapp):
    w = _real_widget(qapp)
    block = w.curation_model_block
    block.source_combo.setCurrentText("External model file")
    w.curation_right_column.set_short_mode(True)
    assert block.path_row.isHidden()
    w.curation_right_column.set_short_mode(False)
    assert not block.path_row.isHidden()


def test_short_mode_favours_slices_over_patches(qapp):
    w = _real_widget(qapp)
    w.resize(578, 560)
    w.show()
    qapp.processEvents()
    w._curation_short = None
    w._apply_curation_room()
    assert w._curation_short is True
    sizes = w.curation_left_splitter.sizes()
    assert sizes[1] > sizes[0]
    w.close()


# 6 -- the filter chips: tight enough for one row at a 1000 px dock.  The
# real width needs the real font (the offscreen fallback is ~1.5x wider), so
# the test pins the numbers the shot was fixed with: 5 px padding, 3 px gaps,
# and the left column's 406 px; the row itself is judged in the shots.
def test_filter_chip_padding_is_tight():
    style = L.filter_chip_style("#fff")
    assert "padding: 2px 5px" in style and "font-size: 10px" in style


def test_the_chip_row_uses_three_px_gaps():
    import inspect
    from napari_epicycle.hitl_training_annotation.ui.main_widget import (
        TrainingAnnotationHITL2D as W)
    assert "FlowLayout(h_spacing=3" in inspect.getsource(
        W._create_curation_patches_column)


# 8 -- the idle bold loss line follows the Writes bucket
def _plot(qapp):
    from napari_epicycle.hitl_training_annotation.ui import loss_plot_widget as lpw
    w = lpw.LossPlotWidget()
    w.set_palette({"accent": "#d9a441", "accent_rgb": (217, 164, 65),
                   "ok_text": "#cccccc"})
    w.set_grade_by_round(True)
    w.load_round_history({"lateral_0": [1.0, 0.8], "lateral_1": [0.9, 0.6],
                          "axial_0": [1.0, 0.7], "axial_1": [0.8, 0.45]})
    return w


def test_idle_bold_family_follows_set_idle_family(qapp):
    w = _plot(qapp)
    assert w._bold_family() == "axial"          # the newest stored round's
    assert w.grading_caption().startswith("last: ax r1")
    w.set_idle_family("lateral")
    assert w._bold_family() == "lateral"
    assert w.grading_caption().startswith("last: lat r1")
    assert "4 rounds" in w.grading_caption()
    w.set_idle_family(None)
    assert w._bold_family() == "axial"
    w.deleteLater()


def test_a_live_round_still_wins_over_the_idle_family(qapp):
    w = _plot(qapp)
    w.set_idle_family("lateral")
    w.set_live_family("axial")
    w.begin_live_round("round_02")
    assert w._bold_family() == "axial"
    w.deleteLater()


def test_the_writes_refresh_tells_the_plot_its_bucket():
    from test_step2_training_seams import _ctrl as tctrl, _Label
    got = []
    plot = type("P", (), {"set_idle_family":
                          lambda s, b: got.append(b)})()
    c = tctrl(model_name_lateral_label=_Label(),
              model_name_axial_label=_Label(), loss_plot=plot)
    c._refresh_model_name_display()
    assert got == ["lateral"]
    c.view_target = "axial"
    c._refresh_model_name_display()
    assert got == ["lateral", "axial"]


def test_a_plot_without_the_method_is_fine():
    from test_step2_training_seams import _ctrl as tctrl, _Label
    c = tctrl(model_name_lateral_label=_Label(),
              model_name_axial_label=_Label(), loss_plot=object())
    c._refresh_model_name_display()          # no AttributeError


# 9 -- the engine names beside the plain labels
def test_setup_labels_carry_the_engine_names(qapp):
    w = _real_widget(qapp)
    col = w.training_setup_column
    assert T.SETUP_ENGINE_NAMES == {"starting_model": "pretrained_model",
                                    "epochs": "n_epochs",
                                    "batch": "batch_size"}
    assert col.starting_model_label.plain_label.text() == "Starting model"
    assert col.starting_model_label.engine_label.text() == "pretrained_model"
    texts = [(b.plain_label.text(), b.engine_label.text())
             for b in col.engine_labels]
    assert texts == [("Epochs", "n_epochs"), ("Batch size", "batch_size")]


# M1 -- the Starting model is greyed with the rest of the setup
def test_model_selector_is_greyed_under_the_training_lock(qapp):
    w = _real_widget(qapp)
    col = w.training_setup_column
    assert w.model_selector in col.lock_extra
    w.model_selector.setEnabled(True)
    w.on_training_locked(True, "Cell_lateral_round_03", "lateral")
    assert not w.model_selector.isEnabled()
    w.on_training_locked(False, "", "")
    assert w.model_selector.isEnabled()


def test_the_override_grey_survives_the_lock_cycle(qapp):
    w = _real_widget(qapp)
    col = w.training_setup_column
    col.set_hint("override", "init locked")
    assert not w.model_selector.isEnabled()
    w.on_training_locked(True, "Cell_lateral_round_00", "lateral")
    w.on_training_locked(False, "", "")
    assert not w.model_selector.isEnabled()


# 10 -- Diagnostics has its two `?` buttons
def test_diagnostics_headers_have_help_buttons(qapp):
    from qtpy.QtWidgets import QPushButton
    w = _real_widget(qapp)
    panel = w.gpu_status_label.parentWidget()
    tips = sorted(b.toolTip() for b in panel.findChildren(QPushButton)
                  if b.toolTip().startswith("Help:"))
    assert tips == [f"Help: {T.DIAG_SYSTEM_HEADER}",
                    f"Help: {T.DIAG_LOG_HEADER}"] or \
        tips == sorted([f"Help: {T.DIAG_SYSTEM_HEADER}",
                        f"Help: {T.DIAG_LOG_HEADER}"])
    assert w.diag_system_header.count() >= 3 and w.diag_log_header.count() >= 3


# wording -- the lock lines never read "Round  (lateral)"; "no slices" is
# about the records, not a filter that hides them all
def test_lock_lines_say_a_question_mark_for_an_unknown_round(qapp):
    w = _real_widget(qapp)
    w.on_training_locked(True, "", "lateral")
    assert "Round ? (lateral)" in w.session_lock_line.text()
    assert "Round ? (lateral)" in w.curation_lock_line.text()
    assert "Round ? (lateral)" in w.training_panel.lock_line.text()


def test_lock_lines_use_the_disk_round_number_like_writes(qapp):
    w = _real_widget(qapp)
    w.on_training_locked(True, "Cell_lateral_round_03", "lateral")
    assert "Round 3 (lateral) is training" in w.session_lock_line.text()
    assert "Round 3 (lateral) is training" in w.curation_lock_line.text()
    assert T.writes_text("Cell_lateral_round_03", 3).endswith("round 3")


def test_every_chip_off_is_not_no_slices(qapp):
    from types import SimpleNamespace
    w = _real_widget(qapp)
    w.controller = SimpleNamespace(loaded_patch_name="p1", selected_view="XY")
    recs = _recs(("p1", "lateral", "predicted"), ("p1", "lateral", "accepted"))
    w._refresh_slice_chips(recs, {"bucket": "lateral", "patch": "p1"})
    assert w._slice_total == 2
    assert w.segment_slice_data_list.count() == 0        # the filter hides all
    w.refresh_stage_line()
    assert w.curation_foot.stage_label.text() == \
        "Ready · p1 · lateral · no slice selected"


def test_a_view_with_no_records_still_says_no_slices(qapp):
    from types import SimpleNamespace
    w = _real_widget(qapp)
    w.controller = SimpleNamespace(loaded_patch_name="p1", selected_view="XY")
    w._refresh_slice_chips({}, {"bucket": "lateral", "patch": "p1"})
    w.refresh_stage_line()
    assert w.curation_foot.stage_label.text() == T.CUR_STAGE_NO_SLICES


# M2 -- the Run summary is re-sent when the picker rewrites its cache
class _FakeCombo:
    def __init__(self):
        self.items = []
        self._idx = -1

    def blockSignals(self, b):
        pass

    def clear(self):
        self.items = []

    def addItem(self, text, data=None):
        self.items.append((text, data))

    def insertSeparator(self, i):
        pass

    def count(self):
        return len(self.items)

    def setEnabled(self, b):
        pass

    def setCurrentIndex(self, i):
        self._idx = i


def test_the_picker_refresh_resends_the_run_summary_after_the_cache():
    from test_step2_hooks import _UIWithHooks, _ctrl as hctrl
    ui = _UIWithHooks()
    ui.run_picker_combo = _FakeCombo()
    c = hctrl(ui)
    c._run_summaries_cache = [          # the PREVIOUS workspace's cache
        {"run_id": "Old", "is_active": True, "accepted_pair_count": 5,
         "description": "old"}]
    c._layout_allows_new_run = lambda: True
    c._format_run_picker_label = lambda s: s["run_id"]
    c._list_runs_for_ui = lambda: [
        {"run_id": "Cell", "is_active": True, "accepted_pair_count": 83,
         "description": "whole cell"}]
    # the workspace-switch order: the badges first (stale cache) ...
    c._refresh_patch_list_badges()
    assert ui.calls[-1][1]["run_pairs"] == 5
    # ... then the picker, which must re-send with the new cache
    c._refresh_run_picker()
    last = ui.calls[-1][1]
    assert last["run_pairs"] == 83 and last["run_desc"] == "whole cell"


def test_an_empty_picker_also_resends():
    from test_step2_hooks import _UIWithHooks, _ctrl as hctrl
    ui = _UIWithHooks()
    ui.run_picker_combo = _FakeCombo()
    c = hctrl(ui)
    c._run_summaries_cache = [{"run_id": "Old", "is_active": True,
                               "accepted_pair_count": 5}]
    c._layout_allows_new_run = lambda: True
    c._list_runs_for_ui = lambda: []
    c._refresh_run_picker()
    assert ui.calls[-1][1]["run_pairs"] is None


# M3 -- the staged count is read from the records, never from a stale cache
def test_the_count_ignores_another_workspaces_cache():
    from test_step2_training_seams import _ctrl as tctrl, _Label
    lab = _Label()
    col = type("Col", (), {"set_desc": lambda s, b: None})()
    foot = type("F", (), {"set_stage_line": lambda s, t: setattr(s, "t", t)})()
    c = tctrl(dataset_staging=lab, training_staged_column=col,
              training_foot=foot)
    c._train_ui_locked = False
    c._staged_new_count = 12                  # workspace A's leftover
    c.sample_records = {}                     # workspace B: nothing
    c._staged_prior_pairs = ("lateral", 0)
    c._write_staged_count_text("lateral")
    assert lab.text.split("\n")[0].startswith("0 new")
    c._refresh_training_stage_line()
    assert foot.t == T.STAGE_IDLE_BARE


def test_staged_count_for_counts_the_bucket_only():
    from test_step2_training_seams import _ctrl as tctrl
    c = tctrl()
    c.sample_records = {
        "a": {"status": "staged", "bucket": "lateral"},
        "b": {"status": "staged", "bucket": "axial"},
        "c": {"status": "accepted", "bucket": "lateral"},
        "d": {"status": "committed", "bucket": "lateral"}}
    assert c._staged_count_for("lateral") == 1
    assert c._staged_count_for("axial") == 1


# L1 / L2 / L3 -- the slice picture manager and the open-time sweep
from test_step2_slice_rows import (_FakeTimer, _png_bytes, _setup,   # noqa: E402
                                   pics)                              # noqa: F401,E402
from napari_epicycle.hitl_training_annotation import _step2_thumbs as TH   # noqa: E402
from napari_epicycle.ui_shell._stack_row_delegate import ROLE_THUMB       # noqa: E402


def test_an_older_requests_answer_is_not_cached_under_the_new_key(pics, qapp):
    ws, recs, lst, ip, mp = _setup(pics, qapp, status="accepted")
    pics.refresh(ws, recs, ["P_XY_3"], loaded_patch="P")
    old_req = pics._reqs["P_XY_3"]
    old_png = TH.slice_png_path(TH.slice_thumbs_dir(ws), old_req)
    # the mask changes again: a NEWER request (a new key, a new png name)
    import os
    st = os.stat(mp)
    os.utime(mp, ns=(st.st_atime_ns, st.st_mtime_ns + 7_000_000_000))
    assert pics.refresh(ws, recs, ["P_XY_3"], loaded_patch="P") == 1
    new_req = pics._reqs["P_XY_3"]
    new_png = TH.slice_png_path(TH.slice_thumbs_dir(ws), new_req)
    assert new_png.name != old_png.name
    # the OLDER request lands last
    pics._on_ready("P_XY_3", old_req.path, str(old_png), _png_bytes())
    assert lst.item(0).data(ROLE_THUMB) is None          # not cached / shown
    assert pics.cache.get(*TH.cache_key_for(new_req)) is None
    assert new_png.name in pics.inflight                 # the newer one flies
    assert old_png.name not in pics.inflight             # the older one is done
    assert pics._png_of["P_XY_3"] == new_png.name
    # and the newer one is accepted when it lands
    pics._on_ready("P_XY_3", new_req.path, str(new_png), _png_bytes())
    assert lst.item(0).data(ROLE_THUMB) is not None
    assert new_png.name not in pics.inflight


def test_an_answer_with_no_png_path_still_lands(pics, qapp):
    ws, recs, lst, ip, mp = _setup(pics, qapp)
    pics.refresh(ws, recs, ["P_XY_3"], loaded_patch="P")
    pics._on_ready("P_XY_3", pics._reqs["P_XY_3"].path, "", _png_bytes())
    assert lst.item(0).data(ROLE_THUMB) is not None


class _W:
    """A stand-in worker (the identity ``state.worker`` carries)."""


def test_a_batchs_finished_clears_only_its_own_names(pics, qapp, monkeypatch):
    ws, recs, lst, ip, mp = _setup(pics, qapp)
    batches = []

    def fake_submit(requests, cache_dir, on_ready, on_failed, on_meta, *,
                    state, on_finished=None, on_error=None):
        # the first call starts worker 1, the second finds it closed and
        # starts worker 2
        state.worker = _W()
        batches.append(on_finished)
        return len(requests)

    monkeypatch.setattr(TH, "submit_slice_thumbs", fake_submit)
    pics.inflight = TH.InFlight()
    # two requests, two distinct png names, two workers
    a = TH.slice_png_path(TH.slice_thumbs_dir(ws),
                          TH.slice_thumb_request("P_XY_3", recs["P_XY_3"],
                                                 stack_paths={},
                                                 workspace=ws)).name
    pics.refresh(ws, recs, ["P_XY_3"], loaded_patch="P")
    first = batches[0]
    assert a in pics.inflight
    # a second sample, a second worker
    recs2 = dict(recs)
    ip2, mp2 = __import__("test_step2_slice_rows")._write_pair(
        pics.tmp / "ws" / "acc", "P_XY_4")
    recs2["P_XY_4"] = dict(recs["P_XY_3"], img_uri=str(ip2), mask_uri=str(mp2),
                           slice_index=4)
    lst.addItem("P_XY_4")
    from qtpy.QtCore import Qt
    lst.item(1).setData(Qt.UserRole, "P_XY_4")
    pics.refresh(ws, recs2, ["P_XY_3", "P_XY_4"], loaded_patch="P")
    assert len(batches) == 2
    b_name = pics._png_of["P_XY_4"]
    assert b_name in pics.inflight
    # the FIRST batch ends: only its name leaves
    first()
    assert a not in pics.inflight
    assert b_name in pics.inflight and "P_XY_4" in pics._png_of
    batches[1]()
    assert b_name not in pics.inflight and "P_XY_4" not in pics._png_of


def test_the_sweep_keeps_a_picture_rendered_after_the_snapshot(tmp_path):
    ws = tmp_path
    d = TH.slice_thumbs_dir(str(ws))
    d.mkdir(parents=True)
    import os
    import time
    old = d / "ghost_XY_1.deadbeef.1.2.png"
    new = d / "late_XY_9.deadbeef.1.2.png"
    flying = d / "fly_XY_2.deadbeef.1.2.png"
    for p in (old, new, flying):
        p.write_bytes(b"x")
    os.utime(old, (1, 1))
    os.utime(flying, (1, 1))
    started = time.time() - 1.0
    n = TH.sweep_unlisted_slices(str(ws), [], newer_than=started,
                                 in_flight={flying.name})
    names = sorted(p.name for p in d.iterdir())
    assert not old.exists()                       # a real orphan goes
    assert new.exists() and flying.exists()       # newer / in flight stay
    assert n == 1 and names == sorted([new.name, flying.name])


def test_the_async_sweep_passes_its_start_time_and_the_in_flight_set(
        tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(TH, "all_runs_sample_ids", lambda ws, extra: set())
    monkeypatch.setattr(
        TH, "sweep_unlisted_slices",
        lambda ws, ids, **kw: seen.update(kw) or 0)
    (TH.slice_thumbs_dir(str(tmp_path))).mkdir(parents=True)
    flying = TH.InFlight()
    t = TH.sweep_slices_async(str(tmp_path), [], in_flight=flying)
    t.join(5)
    assert seen["in_flight"] is flying
    import time
    assert 0 < time.time() - seen["newer_than"] < 30


# 4b -- the short viewport ends on a block boundary (no half-cut header)
def test_short_mode_snaps_the_scroll_viewport_to_a_block_boundary(qapp):
    from napari_epicycle.hitl_training_annotation.ui import _step2_curation as C
    w = _real_widget(qapp)
    right = w.curation_right_column
    right.resize(302, 224)
    right.show()
    right.set_short_mode(True)
    qapp.processEvents()
    qapp.processEvents()
    cap = right.scroll.maximumHeight()
    assert C.SCROLL_MIN_H_SHORT <= cap <= (
        right.height() - right.foot.height()
        + C.snap_cap(right.scroll, 0))
    from qtpy.QtCore import QPoint
    bottoms = {b.mapTo(right.content, QPoint(0, b.height())).y()
               for b in right._blocks if not b.isHidden()}
    assert any(C.snap_cap(right.scroll, b) == cap for b in bottoms), (
        cap, sorted(bottoms))
    # the Curation progress header is dropped; its summary line is not
    assert right._descs[1][0].isHidden()
    assert not right.progress_table.summary_row.isHidden()
    # tall mode: no cap, the header is back
    right.set_short_mode(False)
    qapp.processEvents()
    assert right.scroll.maximumHeight() >= 16777215
    assert not right._descs[1][0].isHidden()
    right.close()


def test_the_summary_hover_is_the_table_as_text(qapp):
    w = _real_widget(qapp)
    table = w.curation_progress_table
    table.refresh({"lateral": {"predicted": 63, "committed": 8},
                   "axial": {"predicted": 2}}, "lateral", "Next round: 3.")
    tip = table.summary_row.toolTip().split("\n")
    assert tip[0] == "Predicted: Lateral (XY) 63 · Axial (XZ/YZ) 2"
    assert "Trained: Lateral (XY) 8 · Axial (XZ/YZ) 0" in tip
    assert tip[-1] == "Next round: 3."


# the Curation section fits a 578 px dock with the thin vertical bar showing
# (the lock state overflows the 218 px room): no horizontal scrollbar
def test_the_curation_panel_min_width_leaves_room_for_the_section_bar(qapp):
    w = _real_widget(qapp)
    panel = w.curation_splitter.parentWidget()
    assert panel.minimumSizeHint().width() <= 566      # 574 - an 8 px bar
