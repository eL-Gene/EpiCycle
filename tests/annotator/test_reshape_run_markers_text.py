"""The Reshape sentences about markers taken from labels -- each one shown.

Every sentence below is a constant of ``napari_epicycle/annotator/
_reshape_text.py`` (the wall line: of the engine), and each test drives the
state that shows it and reads the label back.  A sentence that exists in
the module but is never shown, or is shown in other words, fails here.

The offscreen panel for the panel half, Qt-free fakes for the orchestrator
half -- the split every Reshape suite uses.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qtpy")

from napari_epicycle.annotator import _reshape_text as T  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        try:
            app = QApplication([])
        except Exception as exc:  # pragma: no cover - CI without Qt
            pytest.skip("cannot create a QApplication: {}".format(exc))
    return app


@pytest.fixture
def panel(qapp):
    from napari_epicycle.annotator.advanced_features_widget import (
        AdvancedFeaturesWidget,
    )

    w = AdvancedFeaturesWidget(mode="3d_curation")
    yield w
    try:
        w.setParent(None)
    except Exception:
        pass


def _K():
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    return Curation3DAnnotator


class _GuideAdv:
    """Records every guide-note push with its kind."""

    def __init__(self, seeds_from=2, source=None):
        self.guided = []
        self.ui = []
        self._seeds_from = seeds_from
        self._source = source

        class _Combo:
            def __init__(inner, idx):
                inner._idx = idx

            def currentIndex(inner):
                return inner._idx

        self.split_seeds_from_combo = _Combo(seeds_from)

    def update_split_guided_ui(self, **kw):
        self.guided.append(kw)

    def update_split_ui_state(self, **kw):
        self.ui.append(kw)

    def selected_guide_source(self):
        return self._source

    def guided_needs_guide_mask(self):
        return True

    def note(self):
        for kw in reversed(self.guided):
            if "guide_note" in kw:
                return kw["guide_note"], kw.get("guide_note_kind")
        return None, None


class _Orch:
    """The orchestrator seams the guide flow touches."""

    def __init__(self, adv):
        self.adv = adv
        self.ui_calls = []
        self._guide_thread = None
        self._guide_worker = None
        self._guide_pending = None
        self._reshape_guide = None
        self._reshape_guide_path = ""
        self._split_label_id = 42
        self._split_running = True
        self.added = []
        for name in ("_on_guide_loaded", "_on_guide_refused",
                     "_on_reshape_guide_load", "_guide_thread_cleanup",
                     "_reshape_ready_text", "_reshape_review_status"):
            setattr(self, name, getattr(_K(), name).__get__(self, _Orch))

    def _get_advanced_features_widget(self):
        return self.adv

    def _update_split_ui(self, **kw):
        self.ui_calls.append(kw)

    def _reshape_context(self):
        return {"workspace": "ws", "patch_name": "P", "run_id": None,
                "run_epoch": 0, "downsample_factor": 1,
                "spacing": (1.0, 1.0, 1.0)}

    def _remove_guide_layer(self):
        pass

    def _add_guide_layer(self, guide, name):
        self.added.append(name)

    def status(self):
        for kw in reversed(self.ui_calls):
            if "status_text" in kw:
                return kw["status_text"]
        return None

    def land(self, notes):
        import time

        self._guide_pending = {"display": "Nuclei", "run_id": "nuclei",
                               "path": "x.tif", "src": None, "patch": "P",
                               "t0": time.perf_counter()}
        guide = np.zeros((2, 3, 3), np.int16)
        guide[0, 0, 0] = 1
        guide[1, 1, 1] = 2
        self._on_guide_loaded(guide, notes)


# --------------------------------------------------------------------------- #
# W1 -- the status after Load mask
# --------------------------------------------------------------------------- #
class TestW1TheStatusAfterLoadMask:
    def test_a_loaded_mask_says_what_to_press_next(self):
        o = _Orch(_GuideAdv())
        o.land([])
        assert o.status() == T.STATUS_MASK_LOADED.format(id=42)

    def test_no_session_no_status(self):
        o = _Orch(_GuideAdv())
        o._split_running = False
        o.land([])
        assert o.status() is None

    def test_the_two_ready_lines_are_the_modules(self):
        o = _Orch(_GuideAdv())
        assert o._reshape_ready_text(42) == T.STATUS_READY_LOAD_MASK.format(
            id=42)
        o._reshape_guide = np.zeros((1, 1, 1), np.int16)
        assert o._reshape_ready_text(42) == T.STATUS_READY_PREVIEW.format(
            id=42)


# --------------------------------------------------------------------------- #
# W2 -- "Markers: N" counts the ticked rows
# --------------------------------------------------------------------------- #
class TestW2TheMarkerCount:
    def test_it_counts_the_ticked_rows(self, panel):
        from qtpy.QtCore import Qt

        panel.update_split_ui_state(split_active=True, marker_ids=[1, 2, 3])
        assert panel.split_seed_label.text() == T.MARKERS_COUNT.format(n=3)
        panel.split_marker_list.item(0).setCheckState(Qt.Unchecked)
        assert panel.split_seed_label.text() == T.MARKERS_COUNT.format(n=2)

    def test_an_empty_list_is_zero(self, panel):
        assert panel.split_seed_label.text() == T.MARKERS_COUNT.format(n=0)
        panel.update_split_ui_state(split_active=True, marker_ids=[1, 2])
        panel.update_split_ui_state(marker_ids=[])
        assert panel.split_seed_label.text() == T.MARKERS_COUNT.format(n=0)

    def test_a_pushed_count_with_no_list_is_shown(self, panel):
        panel.update_split_ui_state(seed_count=3)
        assert panel.split_seed_label.text() == T.MARKERS_COUNT.format(n=3)


# --------------------------------------------------------------------------- #
# W3 -- "N pieces from M markers" counts the markers the cut USED
# --------------------------------------------------------------------------- #
class TestW3TheReviewCount:
    def test_it_is_the_engines_seed_count(self):
        o = _Orch(_GuideAdv())
        o._split_painted_marker_ids = [5, 6, 7, 8, 9]

        class _R:
            n_seeds = 2
            coverage = 1.0

        line = o._reshape_review_status(_R(), 2)
        assert line == T.REVIEW_STATUS.format(
            id=42, n=2,
            from_markers=T.REVIEW_FROM_MARKERS.format(n=2, s="s"),
            share=" 100 % assigned.")


# --------------------------------------------------------------------------- #
# W4 -- "Wall contrast" over most of the object
# --------------------------------------------------------------------------- #
class TestW4TheWallContrast:
    def test_the_two_numbers_are_the_5th_and_95th_percentile(self):
        from packages.epicycle_annotator.reshape import engine as E

        elev = np.linspace(0.0, 1.0, 1000).reshape(10, 10, 10)
        region = np.ones_like(elev, dtype=bool)
        lo, hi = np.percentile(elev[region], [5, 95])
        assert E._wall_contrast_line(elev, region) == (
            E.WALL_CONTRAST_LINE.format(lo=lo, hi=hi))
        assert E.WALL_CONTRAST_LINE == "Wall contrast {lo:.2f}-{hi:.2f}."

    def test_one_extreme_voxel_no_longer_sets_the_range(self):
        from packages.epicycle_annotator.reshape import engine as E

        elev = np.full((10, 10, 10), 0.5)
        elev[0, 0, 0] = 0.0
        elev[9, 9, 9] = 1.0
        region = np.ones_like(elev, dtype=bool)
        assert E._wall_contrast_line(elev, region) == "Wall contrast 0.50-0.50."

    def test_nothing_measurable_still_says_one_sentence(self):
        from packages.epicycle_annotator.reshape import engine as E

        line = E._wall_contrast_line(np.zeros((2, 2, 2)),
                                     np.zeros((2, 2, 2), bool))
        assert line == E.WALL_CONTRAST_UNMEASURED


# --------------------------------------------------------------------------- #
# W5 -- the guide note: information grey, refusals and cautions amber
# --------------------------------------------------------------------------- #
class TestW5TheGuideNoteKinds:
    def test_the_panel_draws_info_grey_and_warn_amber(self, panel):
        from napari_epicycle.ui_shell.theme import (
            PANEL_HELP_STYLE,
            PANEL_WARN_STYLE,
        )

        lab = panel.split_guide_note_label
        panel.update_split_guided_ui(guide_note="x", guide_note_kind="info")
        assert lab.styleSheet() == PANEL_HELP_STYLE
        panel.update_split_guided_ui(guide_note="y", guide_note_kind="warn")
        assert lab.styleSheet() == PANEL_WARN_STYLE
        panel.update_split_guided_ui(guide_note="z", guide_note_kind="info")
        panel.update_split_guided_ui(guide_note="w")
        assert lab.styleSheet() == PANEL_WARN_STYLE, (
            "a note with no kind keeps the amber it always had")

    def test_a_plain_load_is_information(self):
        o = _Orch(_GuideAdv())
        o.land(["The mask was stored using smaller whole numbers to save "
                "memory; the cell ids are unchanged.",
                "Guide mask held as int16 whole numbers, using 0.0 MB."])
        note, kind = o.adv.note()
        assert note.startswith("Nuclei - 2 labels in the mask.")
        assert kind == T.GUIDE_NOTE_INFO

    def test_the_downsample_match_is_a_caution(self):
        from napari_epicycle.annotator import reshape_guide as RG

        o = _Orch(_GuideAdv())
        o.land([RG.GUIDE_DOWNSAMPLE_NOTE.format("half size")])
        note, kind = o.adv.note()
        assert "Small labels may merge" in note
        assert kind == T.GUIDE_NOTE_WARN

    def test_a_refusal_is_amber(self):
        o = _Orch(_GuideAdv())
        o._guide_pending = None
        o._on_guide_refused("That mask is a different size from this patch.")
        assert o.adv.note()[1] == T.GUIDE_NOTE_WARN

    def test_reading_is_information(self, monkeypatch):
        import qtpy.QtCore as QC

        from napari_epicycle.annotator import reshape_worker as RW

        class _Sig:
            def connect(self, *_a):
                pass

        class _W:
            def __init__(self, *a, **kw):
                self.finished = self.refused = self.error = _Sig()

            def moveToThread(self, _t):
                pass

            def run(self):
                pass

        class _Th:
            def __init__(self):
                self.started = _Sig()

            def start(self):
                pass

        monkeypatch.setattr(RW, "GuideLoadWorker", _W)
        monkeypatch.setattr(QC, "QThread", _Th)

        class _Src:
            path = "x.tif"
            run_id = "nuclei"
            display_name = "Nuclei"

        o = _Orch(_GuideAdv(source=_Src()))

        class _Main:
            data = np.zeros((2, 3, 3), np.int32)

        o._active_labels_name = "x"
        o.viewer = type("V", (), {"layers": []})()
        o._layer_by_name = lambda layers, name: _Main()
        o._on_reshape_guide_load()
        assert o.adv.note() == (T.GUIDE_NOTE_READING, T.GUIDE_NOTE_INFO)


# --------------------------------------------------------------------------- #
# W6 -- no "Use Browse" where there is no Browse button
# --------------------------------------------------------------------------- #
class TestW6TheNoRunNote:
    def test_load_with_no_other_run_names_the_mask_file_choice(self):
        o = _Orch(_GuideAdv(seeds_from=2, source=None))
        o._on_reshape_guide_load()
        assert o.adv.note() == (T.NO_RUN_NOTE, T.GUIDE_NOTE_WARN)
        assert "Browse" not in T.NO_RUN_NOTE

    def test_the_empty_picker_item_is_the_modules(self, panel):
        panel.set_guide_sources([])
        assert panel.split_guide_source_combo.itemText(0) == T.NO_RUN_ITEM


# --------------------------------------------------------------------------- #
# W7 -- Load mask grey while the run picker is empty
# --------------------------------------------------------------------------- #
class TestW7LoadMaskWithNothingToLoad:
    def _src(self):
        class _Src:
            label = "Nuclei - you are trusting these labels."
            path = "x.tif"
            run_id = "nuclei"

        return _Src()

    def test_grey_on_another_run_with_no_run(self, panel):
        from napari_epicycle.annotator.advanced_features_widget import (
            SEEDS_FROM_RUN_LABELS,
        )

        panel.split_seeds_from_combo.setCurrentIndex(SEEDS_FROM_RUN_LABELS)
        panel.set_guide_sources([])
        assert not panel.split_guide_load_btn.isEnabled()

    def test_live_again_on_the_mask_file_choice(self, panel):
        from napari_epicycle.annotator.advanced_features_widget import (
            SEEDS_FROM_EXTERNAL_MASK,
            SEEDS_FROM_RUN_LABELS,
        )

        panel.split_seeds_from_combo.setCurrentIndex(SEEDS_FROM_RUN_LABELS)
        panel.set_guide_sources([])
        panel.split_seeds_from_combo.setCurrentIndex(SEEDS_FROM_EXTERNAL_MASK)
        assert panel.split_guide_load_btn.isEnabled()
        panel.split_seeds_from_combo.setCurrentIndex(SEEDS_FROM_RUN_LABELS)
        assert not panel.split_guide_load_btn.isEnabled()

    def test_live_when_there_is_a_run(self, panel):
        from napari_epicycle.annotator.advanced_features_widget import (
            SEEDS_FROM_RUN_LABELS,
        )

        panel.split_seeds_from_combo.setCurrentIndex(SEEDS_FROM_RUN_LABELS)
        panel.set_guide_sources([])
        panel.set_guide_sources([self._src()])
        assert panel.split_guide_load_btn.isEnabled()


# --------------------------------------------------------------------------- #
# D1 -- the row and the count line, on the panel alone
# --------------------------------------------------------------------------- #
class TestD1ThePanelHalf:
    def _push(self, panel):
        panel.update_split_ui_state(
            split_active=True, marker_ids=[11, 12, 13],
            marker_stats={11: {"voxels": 900}, 12: {"voxels": 800},
                          13: {"voxels": 3}},
            marker_start_off={13: {"inside": 3, "object_voxels": 1800}})

    def test_the_row_carries_the_word_and_the_tooltip(self, panel):
        from qtpy.QtCore import Qt

        self._push(panel)
        item = panel.split_marker_list.item(2)
        assert item.checkState() != Qt.Checked
        assert item.text().endswith("  ·  " + T.MARKER_ROW_TOUCHING)
        assert T.MARKER_TOUCHING_TIP.format(n=3, s="s", pct="0.2") in (
            item.toolTip())

    def test_the_count_line(self, panel):
        self._push(panel)
        assert panel.split_marker_count_label.text() == (
            T.MARKERS_KEPT_LINE.format(kept=2, s="s", n=1, s2=""))

    def test_the_line_goes_back_to_todays_once_the_user_changes_a_tick(
            self, panel):
        from qtpy.QtCore import Qt

        self._push(panel)
        panel.split_marker_list.item(0).setCheckState(Qt.Unchecked)
        assert panel.split_marker_count_label.text() == (
            T.MARKERS_TICKED_LINE.format(kept=1, total=3))

    def test_a_reconcile_keeps_the_word(self, panel):
        self._push(panel)
        panel.update_split_ui_state(seed_ids=[11, 12, 13, 14])
        item = panel.split_marker_list.item(2)
        assert item.text().endswith(T.MARKER_ROW_TOUCHING)
        assert not panel.split_marker_list.item(3).text().endswith(
            T.MARKER_ROW_TOUCHING)

    def test_a_find_without_verdicts_forgets_them(self, panel):
        self._push(panel)
        panel.update_split_ui_state(marker_ids=[21, 22])
        for i in range(panel.split_marker_list.count()):
            assert T.MARKER_ROW_TOUCHING not in (
                panel.split_marker_list.item(i).text())
        assert panel.get_unticked_marker_ids() == []

    def test_one_and_many(self):
        assert T.plural(1) == "" and T.plural(2) == "s" and T.plural(0) == "s"

    def test_none_kept_reads_right_for_one_and_for_many(self):
        assert T.none_kept_line(1) == T.MARKER_NONE_KEPT_ONE
        assert "1 markers" not in T.none_kept_line(1)
        assert T.none_kept_line(3) == T.MARKERS_NONE_KEPT.format(n=3)

    def test_a_share_under_a_tenth_of_a_percent_is_not_zero(self, panel):
        panel.update_split_ui_state(
            split_active=True, marker_ids=[11, 12],
            marker_stats={11: {"voxels": 7000}, 12: {"voxels": 9}},
            marker_start_off={12: {"inside": 1, "object_voxels": 7215}})
        tip = panel.split_marker_list.item(1).toolTip()
        assert T.MARKER_TOUCHING_TIP.format(n=1, s="", pct="under 0.1") in tip
        assert "0.0 %" not in tip

    def test_the_tip_stays_true_once_the_row_is_ticked(self, panel):
        """The tooltip says how much the label touches -- true ticked or
        not -- and never "left unticked" on a ticked row."""
        from qtpy.QtCore import Qt

        self._push(panel)
        item = panel.split_marker_list.item(2)
        item.setCheckState(Qt.Checked)
        item = panel.split_marker_list.item(2)
        assert item.text().endswith(T.MARKER_ROW_TOUCHING)
        tip = item.toolTip()
        assert T.MARKER_TOUCHING_TIP.format(n=3, s="s", pct="0.2") in tip
        assert "unticked" not in tip.lower()
        assert "Tick it" not in tip


class TestTheNamesTheSentencesPointAt:
    def test_the_no_run_note_names_the_combos_own_item(self, panel):
        from napari_epicycle.annotator.advanced_features_widget import (
            SEEDS_FROM_EXTERNAL_MASK,
        )

        item = panel.split_seeds_from_combo.itemText(SEEDS_FROM_EXTERNAL_MASK)
        assert item == "Markers from a mask file"
        assert '"{}"'.format(item) in T.NO_RUN_NOTE

    def test_mask_loaded_holds_off_while_a_preview_is_on_screen(self):
        """The review line is about the cut; a mask arriving under it
        must not replace it."""
        o = _Orch(_GuideAdv())
        o._review_is_on_screen = lambda: True
        o.land([])
        assert o.status() is None
        o2 = _Orch(_GuideAdv())
        o2._review_is_on_screen = lambda: False
        o2.land([])
        assert o2.status() == T.STATUS_MASK_LOADED.format(id=42)
