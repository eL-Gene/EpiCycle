"""Brick U1 -- ONE marker list for every source.

Before this brick there were two lists.  A painted `split_seed_list`,
which was a colour legend with no ticks, and a tickable
`split_marker_list` that only the three finding sources ever filled --
so two of the five sources could not untick anything, and the two lists
sat three slots apart looking identical while meaning opposite things (a
marker decides what the cut is MADE FROM; a piece decides what Accept
WRITES).

`split_seed_list` is retired.  `seed_ids=` and `active_seed_id=` point
at the marker list, so no orchestrator call site moved.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qtpy")


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
def widget(qapp):
    from napari_epicycle.annotator.advanced_features_widget import (
        AdvancedFeaturesWidget,
    )

    w = AdvancedFeaturesWidget(mode="3d_curation")
    yield w
    try:
        w.setParent(None)
    except Exception:
        pass


def _mod():
    from napari_epicycle.annotator import advanced_features_widget as M

    return M


class TestThereIsOnlyOneList:
    def test_the_seed_list_is_gone(self, widget):
        assert not hasattr(widget, "split_seed_list")

    def test_seed_ids_fills_the_marker_list(self, widget):
        widget.update_split_ui_state(split_active=True, seed_ids=[7, 8])
        assert widget.split_marker_list.count() == 2
        assert widget.get_checked_marker_ids() == [7, 8]

    def test_marker_ids_fills_the_same_list(self, widget):
        widget.update_split_ui_state(split_active=True, marker_ids=[1, 2, 3])
        assert widget.split_marker_list.count() == 3

    @pytest.mark.parametrize("source", [0, 1, 2, 3, 4])
    def test_every_source_can_show_a_tickable_list(self, widget, source):
        widget.split_seeds_from_combo.setCurrentIndex(source)
        widget.update_split_ui_state(split_active=True, seed_ids=[11, 12])
        assert widget.split_marker_list.count() == 2
        assert widget.get_checked_marker_ids() == [11, 12]

    def test_the_seed_count_label_says_markers(self, widget):
        widget.update_split_ui_state(seed_count=3)
        assert "Markers" in widget.split_seed_label.text()
        assert "Seeds" not in widget.split_seed_label.text()


class TestCheckStatePreservation:
    def test_a_RECONCILE_keeps_the_ticks(self, widget):
        """`seed_ids=` is the orchestrator re-listing what is painted."""
        from qtpy.QtCore import Qt

        widget.update_split_ui_state(split_active=True, seed_ids=[7, 8])
        widget.split_marker_list.item(1).setCheckState(Qt.Unchecked)
        assert widget.get_checked_marker_ids() == [7]
        widget.update_split_ui_state(seed_ids=[7, 8, 9])
        assert widget.get_checked_marker_ids() == [7, 9], (
            "a reconcile must keep 'id 8 is not an object'")

    def test_a_FIND_re_ticks_everything(self, widget):
        """A-M6: the detectors renumber from 1 on every run, so keeping
        "id 2 was unticked" would untick a DIFFERENT marker."""
        from qtpy.QtCore import Qt

        widget.update_split_ui_state(split_active=True, marker_ids=[1, 2, 3])
        widget.split_marker_list.item(1).setCheckState(Qt.Unchecked)
        assert widget.get_checked_marker_ids() == [1, 3]
        widget.update_split_ui_state(marker_ids=[1, 2, 3])
        assert widget.get_checked_marker_ids() == [1, 2, 3]


class TestRowContents:
    def test_a_painted_row_carries_no_count(self, widget):
        """Counts drift between reconciles and say nothing."""
        widget.update_split_ui_state(split_active=True, seed_ids=[7])
        assert widget.split_marker_list.item(0).text() == "Marker 1"

    def test_a_found_row_keeps_its_size_and_place(self, widget):
        widget.update_split_ui_state(
            split_active=True, marker_ids=[1],
            marker_stats={1: {"voxels": 900, "centroid_zyx": (5, 6, 7)}})
        assert "900" in widget.split_marker_list.item(0).text()

    def test_an_unticked_row_is_dimmed_and_struck_through(self, widget):
        """M1: unticking used to look exactly like deleting."""
        from qtpy.QtCore import Qt

        widget.update_split_ui_state(split_active=True, marker_ids=[1, 2])
        item = widget.split_marker_list.item(1)
        assert item.font().strikeOut() is False
        item.setCheckState(Qt.Unchecked)
        assert item.font().strikeOut() is True
        item.setCheckState(Qt.Checked)
        assert item.font().strikeOut() is False

    def test_the_hint_says_an_unticked_marker_is_lifted_off(self, widget):
        """U2b inverted this line.  "Unticked markers stay drawn"
        was the shipped rule and the owner found it misleading on real
        data: the marker you have just said is NOT an object went on
        sitting on it, in the same palette as the ones about to cut it.
        Unticking now LIFTS it off, and the line says so."""
        widget.update_split_ui_state(split_active=True, marker_ids=[1, 2])
        # The grey LINE that used to carry this was retired in Brick U4
        # (audit H2) -- 14 px the tallest state could not spare, for a
        # question the object has answered by itself since U2b.  The
        # sentence is on the list's own tooltip now.
        assert not hasattr(widget, "split_marker_hint")
        text = widget.split_marker_list.toolTip().lower()
        assert "lifts it off" in text
        assert "stay drawn" not in text


class TestTheTwoCaptions:
    def test_the_marker_card_says_what_a_marker_decides(self, widget):
        text = widget.split_marker_caption.text().lower()
        assert "markers" in text and "before the cut" in text

    def test_the_result_card_says_what_a_piece_decides(self, widget):
        text = widget.split_piece_card_caption.text().lower()
        assert "pieces" in text and "after the cut" in text

    def test_the_result_caption_is_permanent(self, widget):
        """The per-cut sentence (`split_piece_caption`) is overwritten;
        this one never is."""
        before = widget.split_piece_card_caption.text()
        widget.update_split_guided_ui(
            pieces=[1, 2], piece_sizes={1: 10, 2: 20},
            caption="unticked pieces stay part of object 149")
        assert widget.split_piece_card_caption.text() == before

    def test_the_rows_are_visibly_different(self, widget):
        assert (widget.split_marker_list.styleSheet()
                != widget.split_piece_list.styleSheet())
        assert "border-left" in widget.split_marker_list.styleSheet()


class TestATickTouchesNothingOutsideThePanel:
    def test_the_marker_list_is_not_wired_to_currentRowChanged(self, widget):
        """A-M5: Qt moves the current row on a CHECKBOX click too, so
        that connection ran a four-viewer `sync_selected_label` on every
        tick."""
        import inspect

        from napari_epicycle.annotator.advanced_features_widget import (
            AdvancedFeaturesWidget,
        )

        src = inspect.getsource(
            AdvancedFeaturesWidget._build_assisted_split_section)
        assert "split_marker_list.currentRowChanged" not in src
        # ...and nothing CONNECTS it anywhere else either.  Checked as a
        # connection, not as the word: both sites carry a comment saying
        # why the connection is gone, and a comment is not a wire.
        src2 = inspect.getsource(AdvancedFeaturesWidget.set_split_callbacks)
        assert "currentRowChanged.connect" not in src2
        assert "currentRowChanged.connect" not in src

    def test_active_seed_id_never_emits(self, widget):
        fired = []
        widget.update_split_ui_state(split_active=True, seed_ids=[7, 8])
        widget.split_marker_list.currentRowChanged.connect(
            lambda r: fired.append(r))
        widget.update_split_ui_state(active_seed_id=8)
        assert fired == [], (
            "`setCurrentRow` must run inside blockSignals")

    def test_active_seed_id_still_highlights_the_row(self, widget):
        widget.update_split_ui_state(split_active=True, seed_ids=[7, 8])
        widget.update_split_ui_state(active_seed_id=8)
        assert widget.split_marker_list.currentRow() == 1

    def test_the_click_that_toggled_a_tick_does_not_select(self, widget):
        """`itemClicked` fires for the checkbox click as well, so the
        one that produced a tick is suppressed."""
        from qtpy.QtCore import Qt

        picked = []
        widget.set_split_callbacks(on_select_seed=lambda i: picked.append(i))
        widget.update_split_ui_state(split_active=True, seed_ids=[7, 8])
        item = widget.split_marker_list.item(1)
        item.setCheckState(Qt.Unchecked)      # -> itemChanged
        widget._on_marker_row_clicked(item)   # -> itemClicked, same click
        assert picked == []

    def test_a_plain_row_click_DOES_select(self, widget):
        picked = []
        widget.set_split_callbacks(on_select_seed=lambda i: picked.append(i))
        widget.update_split_ui_state(split_active=True, seed_ids=[7, 8])
        widget._on_marker_row_clicked(widget.split_marker_list.item(1))
        assert picked == [8]


class TestAllAndNone:
    def test_they_ride_the_list_not_the_find_row(self, widget):
        """Source 0 has no Find row at all, and it needs them."""
        widget.split_seeds_from_combo.setCurrentIndex(0)
        widget.update_split_ui_state(split_active=True, seed_ids=[7, 8])
        assert widget.split_marker_all_btn.isVisibleTo(widget) is True
        assert widget.split_marker_none_btn.isVisibleTo(widget) is True

    def test_None_dims_and_strikes_every_row(self, widget):
        """Fix pass M2: `_set_all_marker_checks` ran under blockSignals
        and never restyled, so after pressing None the rows kept their
        bright, unstruck look -- with nothing on the row to say the
        marker had been unticked at all.  (Since U2b the object
        itself answers that too: the markers come off it.  The row
        styling is still what tells you WHICH ones.)"""
        widget.update_split_ui_state(split_active=True, marker_ids=[1, 2, 3])
        widget.split_marker_none_btn.click()
        for i in range(3):
            item = widget.split_marker_list.item(i)
            assert item.font().strikeOut() is True, i
            assert item.foreground().color().red() == 0x77, i

    def test_All_puts_the_rows_back(self, widget):
        widget.update_split_ui_state(split_active=True, marker_ids=[1, 2, 3])
        widget.split_marker_none_btn.click()
        widget.split_marker_all_btn.click()
        for i in range(3):
            item = widget.split_marker_list.item(i)
            assert item.font().strikeOut() is False, i

    def test_none_then_all_round_trips(self, widget):
        widget.update_split_ui_state(split_active=True, marker_ids=[1, 2, 3])
        widget.split_marker_none_btn.click()
        assert widget.get_checked_marker_ids() == []
        widget.split_marker_all_btn.click()
        assert widget.get_checked_marker_ids() == [1, 2, 3]


class TestWhichSourceGetsAMarkerLayer:
    def test_no_source_gets_a_second_layer_any_more(self):
        """The markers ARE the coloured dots on the Refine layer,
        for every source.  That layer is mirrored into the XY / XZ / YZ
        side views; `guide_markers` was not, which is why the owner
        could not see the markers of sources 2-4 in the very views the
        markers are painted in."""
        M = _mod()
        # The LITERAL set, not `== FIND_MARKER_SOURCES` (audit M4).  The
        # module defines `MATERIALISED_MARKER_SOURCES =
        # FIND_MARKER_SOURCES`, so comparing the two asserts that a name
        # equals itself and would survive both of them changing together
        # -- which is exactly the edit that would silently stop a source
        # painting its markers where the side views can see them.
        assert set(M.MATERIALISED_MARKER_SOURCES) == {1, 2, 3, 4}
        assert M.SEEDS_FROM_PAINTED not in M.MATERIALISED_MARKER_SOURCES
        for idx in sorted(M.IMAGE_DETECTOR_SOURCES):
            assert idx in M.MATERIALISED_MARKER_SOURCES, idx


# --------------------------------------------------------------------------- #
# Start must not list a marker that has no voxels (fix pass, H1)
# --------------------------------------------------------------------------- #
class _StartLayer:
    """The Refine layer `_start_split` creates, with nothing painted."""

    def __init__(self, data):
        self.data = data
        self.selected_label = 0
        self.mode = "pan_zoom"
        self.editable = True

    def refresh(self):
        pass


class _Selection:
    def __init__(self):
        self.active = None


class _Layers(list):
    def __init__(self):
        super().__init__()
        self.selection = _Selection()


class _Viewer:
    def __init__(self):
        self.layers = _Layers()


class StartHarness:
    """The real `_start_split` over the real panel, driven by its button.

    Deliberately NOT a direct method call: the Brick 2a routing bug hid
    behind a driver that reached past the UI, so the only honest way to
    ask "what does the panel hold after Start" is to press Start.
    """

    def __init__(self, panel, target=7):
        import numpy as np

        self._panel = panel
        self.viewer = _Viewer()
        self._active_labels_name = "labels"
        self._shared = np.zeros((6, 6, 6), np.int32)
        self._shared[1:4, 1:4, 1:4] = target
        self._target = target
        self._split_running = False
        self._smooth_running = False
        self._batch_staged = False
        self._batch_running = False
        self._isolation_active = False
        self._reshape_active = False
        self._reshape_guide = None
        self._refine_layer = None
        self._split_roi = None
        self._split_snapshot = None
        self._split_label_id = None
        self._split_seed_ids = []
        self._split_painted_marker_ids = []
        self._split_isolate_was_on = False
        self._split_main_visible = True
        self._detected_peaks = None
        self._auto_detect_active = False
        self._reshape_running = False
        self._seed_find_running = False
        self._reshape_promoted = False
        self._split_previewed = False
        self._refine_pre_preview = None
        self._reshape_found_seeds = None
        self._reshape_found_marker_ids = []
        self._reshape_found_marker_stats = {}
        self.locks = []
        self.worker_kwargs = []

    # --- the bits `_start_split` leans on, stubbed ------------------
    def _get_advanced_features_widget(self):
        return self._panel

    def _ensure_shared_data_aligned(self, name):
        return self._shared

    def _get_selected_label(self):
        return self._target

    def _fire_reshape_lock(self, on):
        self.locks.append(bool(on))

    def _refresh_reshape_inputs(self):
        pass

    def _reshape_search_padding(self):
        return 1

    def _snapshot_label_roi(self, data, label, _precomputed_mask=None,
                            margin=0):
        roi = (slice(0, 6), slice(0, 6), slice(0, 6))
        return roi, data[roi].copy()

    def _refresh_merge_button_state(self):
        pass

    def _refresh_label_action_buttons_state(self):
        pass

    def _clear_draw_snapshot(self):
        pass

    def _create_refine_layer(self, data, label, _precomputed_mask=None):
        import numpy as np

        vol = np.where(data == label, label, 0).astype(np.int32)
        self._refine_layer = _StartLayer(vol)
        return self._refine_layer

    def _layer_by_name(self, layers, name):
        return None

    def _sync_all_views_after_operation(self, roi=None, label_id=None):
        pass

    def _next_label_id_after_accept(self):
        return 300

    def _auto_enable_edt_for_split(self):
        pass

    def _refresh_refine_in_subviewers(self):
        pass

    def _lock_subviewers_for_preview(self):
        pass

    def _restore_ready_to_draw_state(self):
        pass

    def _reshape_labels_for_engine(self):
        return self._shared

    def _reshape_context(self):
        return {"patch_name": "p", "run_epoch": 0, "spacing": (1, 1, 1)}

    def _reshape_recipe(self):
        from packages.epicycle_annotator.reshape.spec import ReshapeSpec

        return (ReshapeSpec(target="label", seeds="distance",
                            walls="distance"), {}, None)


def _bind_start(harness, *names):
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    for n in names:
        setattr(harness, n,
                getattr(Curation3DAnnotator, n).__get__(harness, StartHarness))
    return harness


@pytest.fixture
def started(widget):
    """A harness whose Start button is wired to the real `_start_split`."""
    h = StartHarness(widget)
    _bind_start(h, "_start_split", "_update_split_ui", "_reshape_ready_text",
                "_marker_rows_from_refine", "_painted_marker_ids_on_layer",
                "_checked_marker_ids", "_preview_reshape",
                "_marker_finder_name")
    widget.set_split_callbacks(on_start=h._start_split,
                               on_preview=h._preview_reshape)
    return h


class TestStartListsNoPhantomMarker:
    """H1: `_start_split` allocated the first id and LISTED it.

    Once `seed_ids=` points at the tickable list, that id is a row --
    a marker with no voxels anywhere.  `get_checked_marker_ids()` then
    answers `[id]` instead of `None`, the guided Preview hands it over
    as `keep_seed_ids`, and `np.isin(seeds, keep)` inside the engine
    filters away every marker the detector actually found: "No markers
    found in this object" on a perfectly good object.
    """

    def test_a_guided_start_leaves_the_list_empty(self, widget, started):
        widget.split_seeds_from_combo.setCurrentIndex(4)  # image channel
        widget.split_start_btn.click()
        assert started._split_running is True, "Start did not run"
        assert widget.split_marker_list.count() == 0
        assert widget.get_checked_marker_ids() is None

    def test_a_painted_start_leaves_the_list_empty_too(self, widget, started):
        """The rule is "never list a marker with no voxels", and it is
        not conditional: routing is decided LIVE, so a session started
        on source 0 can be previewed on source 4 without a restart."""
        widget.split_seeds_from_combo.setCurrentIndex(0)
        widget.split_start_btn.click()
        assert widget.split_marker_list.count() == 0
        assert widget.get_checked_marker_ids() is None

    def test_the_guided_preview_sends_no_tick_list(self, widget, started,
                                                   monkeypatch):
        """The whole point: `keep_seed_ids` reaches the engine as None."""
        import napari_epicycle.annotator.reshape_worker as RW

        seen = {}

        class _Recorder:
            def __init__(self, *a, **kw):
                seen.update(kw)
                raise RuntimeError("recorded")

        monkeypatch.setattr(RW, "ReshapeWorker", _Recorder)
        widget.split_seeds_from_combo.setCurrentIndex(4)
        widget.split_start_btn.click()
        widget.split_preview_btn.setEnabled(True)
        widget.split_preview_btn.click()
        assert seen, "the worker was never constructed"
        assert seen["keep_seed_ids"] is None, (
            "a phantom Start marker reached the engine as keep_seed_ids")

    def test_the_start_line_names_the_button_that_is_ON_SCREEN(
            self, widget, started):
        """Fix pass M5: the Start line offered "Detect Seeds" to every
        source.  Source 1 is the legacy path when the walls are
        shape-derived, so it reached that line -- and index 1 lost
        Detect Seeds to Find markers in Brick U1, so the one instruction
        the user was given named a button that is not on the page."""
        widget.split_seeds_from_combo.setCurrentIndex(1)
        widget.split_start_btn.click()
        line = widget.split_status_label.text()
        assert "Find markers" in line, line
        assert widget.split_detect_seeds_btn.isVisibleTo(widget) is False

    def test_source_zero_is_pointed_at_its_own_finder(self, widget, started):
        widget.split_seeds_from_combo.setCurrentIndex(0)
        widget.split_start_btn.click()
        line = widget.split_status_label.text()
        assert "Find from shape" in line, line
        assert "Detect Seeds" not in line

    def test_the_count_line_does_not_claim_a_marker(self, widget, started):
        widget.split_start_btn.click()
        assert widget.split_seed_label.text() == "Markers: 0"


# --------------------------------------------------------------------------- #
# A second Find must replace the first, not stack on it (fix pass, H2)
# --------------------------------------------------------------------------- #
class _MatLayer:
    def __init__(self, data):
        self.data = data
        self.selected_label = 0

    def refresh(self):
        pass


class _Sync:
    def __init__(self, vol):
        self.shared_labels = {"labels": vol}


class _MatReport:
    """A `SeedReport` with the markers written into an ROI-shaped crop."""

    def __init__(self, places, shape=(4, 4, 4)):
        import numpy as np

        self.status = "ok"
        self.seeds = np.zeros(shape, np.int32)
        self.markers = []
        self.messages = []
        self.roi = tuple(slice(0, s) for s in shape)
        for k, place in enumerate(places, start=1):
            self.seeds[place] = k
            self.markers.append(
                {"id": k, "voxels": 1, "centroid_zyx": (0.0, 0.0, 0.0)})
        self.n_found = len(self.markers)
        self.n_in_region = len(self.markers)


class MatHarness:
    """The real materialiser + the real finish handler + the real panel."""

    def __init__(self, panel, painted=()):
        import numpy as np

        self._panel = panel
        vol = np.zeros((4, 4, 4), np.int32)
        self._shared = vol
        self.sync_manager = _Sync(vol)
        self._active_labels_name = "labels"
        roi_data = np.zeros((4, 4, 4), np.int32)
        for pid, place in painted:
            roi_data[place] = pid
        self._refine_layer = _MatLayer(roi_data)
        self._split_roi = (slice(0, 4), slice(0, 4), slice(0, 4))
        self._split_label_id = 7
        self._split_running = True
        self._split_previewed = False
        self._split_painted_marker_ids = [pid for pid, _ in painted]
        self._split_seed_ids = list(self._split_painted_marker_ids)
        self._materialised_ids = []
        self._reshape_found_seeds = None
        self._reshape_found_marker_ids = []
        self._reshape_found_marker_stats = {}
        self._seed_find_running = True
        self._seed_find_token = 1
        self._seed_find_launch_labels_name = "labels"
        self._seed_find_launch_patch = "p"
        self._seed_find_launch_run_epoch = 0
        self._seed_find_t0 = 0.0
        self._split_busy_kind = "find"
        self.subviewer_refreshes = 0

    def _get_advanced_features_widget(self):
        return self._panel

    def _reshape_context(self):
        return {"patch_name": "p", "run_epoch": 0, "spacing": (1, 1, 1)}

    def _result_token(self):
        return 1

    def _seed_find_thread_cleanup(self):
        pass

    def _remove_temp_layer(self, key):
        pass

    def _add_temp_layer(self, key, array, **kw):
        raise AssertionError(
            "no source draws its markers in a second layer any more")

    def _refresh_refine_in_subviewers(self):
        self.subviewer_refreshes += 1

    def _roi(self):
        return self._refine_layer.data[self._split_roi]


def _bind_mat(h, *names):
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    for n in names:
        setattr(h, n,
                getattr(Curation3DAnnotator, n).__get__(h, MatHarness))
    return h


#: What a `MatHarness` binds.  ONE list, shared by both fixtures --
#: they had drifted into two copies, and U2b's parking was added to one
#: of them, so half the class failed on a missing attribute.
_MAT_BOUND = (
    "_materialise_found_markers", "_on_seed_find_finished",
    "_drop_materialised_markers", "_seed_find_is_stale",
    "_marker_source_materialises", "_marker_source_index",
    "_update_split_ui", "_painted_marker_ids_on_layer",
    # U2b audit H2: the finish handler reconciles parking now, and this
    # harness drives the REAL handler against the REAL panel -- so it
    # gets the real parking too, not a stub.
    "_sync_marker_parking", "_park_marker", "_unpark_marker",
    "_unpark_all_markers", "_parked_marker_ids", "_unticked_marker_ids",
    "_checked_marker_ids",
)


def _find(h, report):
    """Press Find.  The handler clears `_seed_find_running` on its way
    out, so a second press has to re-arm it exactly as a real launch
    does -- without this the second call is discarded as a late result
    and the test passes on a bug."""
    h._seed_find_running = True
    h._split_busy_kind = "find"
    h._on_seed_find_finished(report)


@pytest.fixture
def mat(widget):
    widget.split_seeds_from_combo.setCurrentIndex(1)  # shape thickness peaks
    h = MatHarness(widget, painted=[(101, (3, 3, 3))])
    _bind_mat(h, *_MAT_BOUND)
    return h


class TestAMarkerWithNoVoxelsIsNotAMarker:
    """`_start_split` allocates the id of the FIRST brush stroke before
    anything is painted.  `b6d811e9` stopped START from listing that id;
    a MATERIALISING Find put it straight back, because the materialiser
    returned the whole bookkeeping list rather than the ids that have
    voxels.

    Live driver U1-08d, 2026-09-19: after a Find on "Shape thickness
    peaks" the panel read "Found 2 markers" while the engine's own line
    said `in_region=1`.  The extra row was a "Marker 1" with no voxels
    anywhere, no size and no position -- and it was counted in "3
    ticked" for a cut that could only ever produce 2 pieces.
    """

    def _phantom(self, widget, painted=()):
        widget.split_seeds_from_combo.setCurrentIndex(1)
        h = MatHarness(widget, painted=painted)
        if not painted:
            # exactly Start's state: an id allocated, nothing painted,
            # and the brush aimed at it
            h._split_painted_marker_ids = [101]
            h._split_seed_ids = [101]
            h._refine_layer.selected_label = 101
        _bind_mat(h, *_MAT_BOUND)
        return h

    def test_the_never_painted_id_is_not_listed(self, widget):
        h = self._phantom(widget)
        _find(h, _MatReport([(0, 0, 0), (0, 0, 1)]))
        ids = [widget.split_marker_list.item(i).data(_Qt().UserRole)
               for i in range(widget.split_marker_list.count())]
        assert 101 not in ids, (
            "the id Start allocated is listed as a marker although nothing "
            "was ever painted with it: {}".format(ids))
        assert len(ids) == 2, ids

    def test_it_is_not_in_the_bookkeeping_either(self, widget):
        """Otherwise `get_checked_marker_ids()` and
        `_split_painted_marker_ids` disagree about what the cut is made
        from, which is the one thing section 2.2 exists to prevent."""
        h = self._phantom(widget)
        _find(h, _MatReport([(0, 0, 0), (0, 0, 1)]))
        assert 101 not in h._split_painted_marker_ids
        assert list(h._split_seed_ids) == list(h._split_painted_marker_ids)

    def test_every_listed_marker_carries_its_size_and_place(self, widget):
        h = self._phantom(widget)
        _find(h, _MatReport([(0, 0, 0), (0, 0, 1)]))
        for i in range(widget.split_marker_list.count()):
            tip = widget.split_marker_list.item(i).toolTip()
            assert "voxel" in tip and " at " in tip, tip

    def test_a_marker_that_WAS_painted_survives_the_find(self, widget):
        """The guard: only the never-painted ids go.  A hand-painted
        marker is the one thing a Find may not touch."""
        h = self._phantom(widget, painted=[(101, (3, 3, 3))])
        _find(h, _MatReport([(0, 0, 0), (0, 0, 1)]))
        ids = [widget.split_marker_list.item(i).data(_Qt().UserRole)
               for i in range(widget.split_marker_list.count())]
        assert 101 in ids, ids
        assert len(ids) == 3, ids

    def test_the_brush_is_left_on_a_marker_that_exists(self, widget):
        """Dropping the id also drops what the next brush stroke would
        have painted into, so the layer's `selected_label` has to move to
        a marker that is really there."""
        h = self._phantom(widget)
        _find(h, _MatReport([(0, 0, 0), (0, 0, 1)]))
        assert h._refine_layer.selected_label in h._split_painted_marker_ids


class TestASecondFindReplacesTheFirst:
    """H2: `_materialise_found_markers` was purely ADDITIVE.

    Change *Prominence* and press Find again and the first Find's
    voxels were still sitting in the Refine ROI under their old ids,
    still in `_split_painted_marker_ids`, and still in the list -- so
    the object was cut from both sets of markers at once, and the count
    the user had just tuned was silently doubled.
    """

    def test_the_first_finds_voxels_are_gone_from_the_roi(self, mat):
        """Asserted by PLACE, not by id: taking the old generation out
        frees its ids, so the second Find legitimately mints the same
        numbers again.  What must not survive is the old geometry."""
        _find(mat, _MatReport([(0, 0, 0), (0, 0, 1)]))
        assert list(mat._materialised_ids) == [
            i for i in mat._split_painted_marker_ids if i != 101]
        _find(mat, _MatReport([(1, 1, 1), (1, 1, 2)]))
        roi = mat._roi()
        assert roi[0, 0, 0] == 0 and roi[0, 0, 1] == 0, (
            "the previous Find's markers are still painted on the object")
        assert roi[1, 1, 1] > 0 and roi[1, 1, 2] > 0, "the new ones are missing"
        assert roi[3, 3, 3] == 101, "the hand-painted marker was erased"

    def test_the_id_list_holds_one_generation_only(self, mat):
        _find(mat, _MatReport([(0, 0, 0), (0, 0, 1)]))
        _find(mat, _MatReport([(1, 1, 1), (1, 1, 2)]))
        ids = mat._split_painted_marker_ids
        assert 101 in ids, "the HAND-PAINTED marker must survive a re-Find"
        assert len([i for i in ids if i != 101]) == 2, (
            "the previous Find's ids are still in the marker list: "
            "{}".format(ids))
        assert len(mat._materialised_ids) == 2

    def test_the_panel_shows_one_generation_too(self, mat, widget):
        _find(mat, _MatReport([(0, 0, 0), (0, 0, 1)]))
        _find(mat, _MatReport([(1, 1, 1), (1, 1, 2)]))
        assert widget.split_marker_list.count() == 3

    def test_the_painted_marker_keeps_its_untick_across_a_find(
            self, mat, widget):
        """A Find re-ticks what IT found -- the ids restart every run.
        It has no business re-ticking a marker the user painted and
        then deliberately unticked."""
        _find(mat, _MatReport([(0, 0, 0), (0, 0, 1)]))
        for i in range(widget.split_marker_list.count()):
            item = widget.split_marker_list.item(i)
            if item.data(_Qt().UserRole) == 101:
                item.setCheckState(_Qt().Unchecked)
        assert 101 not in (widget.get_checked_marker_ids() or [])
        _find(mat, _MatReport([(1, 1, 1), (1, 1, 2)]))
        assert 101 not in (widget.get_checked_marker_ids() or []), (
            "the Find re-ticked a marker the user had unticked by hand")
        assert len(widget.get_checked_marker_ids() or []) == 2, (
            "the newly FOUND markers must all be ticked")

    def test_the_slate_is_wiped_by_the_session_reset(self):
        """`_materialised_ids` is session state, so it goes with the
        session -- otherwise the next object's first Find would try to
        erase ids that belong to somebody else."""
        import inspect

        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        src = inspect.getsource(Curation3DAnnotator._reset_split_state)
        assert "_materialised_ids" in src


def _Qt():
    from qtpy.QtCore import Qt

    return Qt


class TestPreserveIdsIsNarrow:
    def test_only_the_named_ids_may_keep_an_untick(self, widget):
        Qt = _Qt()
        widget.update_split_ui_state(split_active=True,
                                     marker_ids=[101, 1, 2])
        for i in range(3):
            widget.split_marker_list.item(i).setCheckState(Qt.Unchecked)
        widget.update_split_ui_state(marker_ids=[101, 1, 2],
                                     marker_preserve_ids=[101])
        assert widget.get_checked_marker_ids() == [1, 2]
