"""Brick U1b -- every found source draws its markers on the Refine layer.

Owner, 2026-09-20, on the live panel: markers found in an image channel
are **not visible in the XY / XZ / YZ side views**.  "Shape thickness
peaks" (source 1) looks right, and the reason is an accident of Bug B's
fix -- that source MATERIALISES its peaks onto the Refine layer, and
the Refine layer is mirrored into the three subviewers.  Sources 2, 3
and 4 put theirs in the unmanaged ``guide_markers`` temp layer, which
only the main viewer holds, so the side views -- the very place the
markers are painted and checked -- showed nothing.

So the rule is now the same for every found source: **the
markers land on the Refine layer, and ``guide_markers`` is not created
for markers at all.**  The temp-layer registry stays (Brick U3's
``guide_walls`` uses it) and the ``key="mask"`` guide layer is
untouched -- that one deliberately survives Accept and Discard.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

pytest.importorskip("qtpy")


def _K():
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    return Curation3DAnnotator


def _M():
    from napari_epicycle.annotator import advanced_features_widget as M

    return M


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


# --------------------------------------------------------------------------- #
# the harness -- the real materialiser + the real finish handler
# --------------------------------------------------------------------------- #
class _Layer:
    def __init__(self, data):
        self.data = data
        self.selected_label = 0
        self.refreshed = 0

    def refresh(self):
        self.refreshed += 1


class _Sync:
    def __init__(self, vol):
        self.shared_labels = {"labels": vol}


class _Report:
    """A ``SeedReport`` whose markers are written into an ROI-shaped crop."""

    def __init__(self, places, shape=(4, 4, 4)):
        self.status = "ok"
        self.seeds = np.zeros(shape, np.int32)
        self.markers = []
        self.messages = []
        self.roi = tuple(slice(0, s) for s in shape)
        for k, place in enumerate(places, start=1):
            self.seeds[place] = k
            self.markers.append(
                {"id": k, "voxels": int(np.sum(self.seeds == k)),
                 "centroid_zyx": tuple(float(v) for v in place)})
        self.n_found = len(self.markers)
        self.n_in_region = len(self.markers)


class Harness:
    def __init__(self, panel, painted=()):
        self._panel = panel
        vol = np.zeros((4, 4, 4), np.int32)
        self.sync_manager = _Sync(vol)
        self._active_labels_name = "labels"
        roi_data = np.zeros((4, 4, 4), np.int32)
        roi_data[2:, :, :] = 7                # the object being reshaped
        for pid, place in painted:
            roi_data[place] = pid
        self._refine_layer = _Layer(roi_data)
        self._refine_pre_preview = None
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
        self._reshape_result = None
        self._seed_find_running = True
        self._seed_find_token = 1
        self._seed_find_launch_labels_name = "labels"
        self._seed_find_launch_patch = "p"
        self._seed_find_launch_run_epoch = 0
        self._seed_find_t0 = 0.0
        self._split_busy_kind = "find"
        self.subviewer_refreshes = 0
        self.temp_layers_added = []

    # -- seams the handler needs ---------------------------------------
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
        self.temp_layers_added.append(key)
        return None

    def _refresh_refine_in_subviewers(self):
        self.subviewer_refreshes += 1

    def _roi(self):
        return self._refine_layer.data[self._split_roi]


#: U2b audit H2: the finish handler and the invalidation handler both
#: reconcile parking now, and this file drives the REAL handlers against
#: the REAL panel -- so they get the real parking too, not a stub.
_PARKING = (
    "_sync_marker_parking", "_park_marker", "_unpark_marker",
    "_unpark_all_markers", "_parked_marker_ids", "_unticked_marker_ids",
    "_checked_marker_ids",
    # Brick U3: the snap-back drops the wall picture too, so a harness
    # that drives the REAL `_leave_reshape_review` needs the REAL
    # method -- a stub would pass this file by not having the wire.
    "_drop_walls_picture", "_remove_temp_layer",
)


def _bind(h, *names):
    K = _K()
    for n in tuple(names) + _PARKING:
        setattr(h, n, getattr(K, n).__get__(h, Harness))
    return h


def _harness(widget, source, painted=()):
    widget.split_seeds_from_combo.setCurrentIndex(source)
    h = Harness(widget, painted=painted)
    _bind(h, "_materialise_found_markers", "_on_seed_find_finished",
          "_drop_materialised_markers",
          "_seed_find_is_stale", "_marker_source_materialises",
          "_marker_source_index", "_update_split_ui",
          "_painted_marker_ids_on_layer", "_marker_rows_from_refine",
          "_leave_reshape_review")
    return h


def _find(h, report):
    h._seed_find_running = True
    h._split_busy_kind = "find"
    h._on_seed_find_finished(report)


#: Every source that has a "Find markers" button, i.e. 1, 2, 3 and 4.
FOUND_SOURCES = (1, 2, 3, 4)


# --------------------------------------------------------------------------- #
# the rule
# --------------------------------------------------------------------------- #
class TestEveryFoundSourceMaterialises:
    def test_the_set_is_exactly_the_find_sources(self):
        M = _M()
        # The LITERAL set (audit M4): the module defines one of these
        # names AS the other, so comparing them asserts nothing.
        assert set(M.MATERIALISED_MARKER_SOURCES) == {1, 2, 3, 4}
        assert set(M.MATERIALISED_MARKER_SOURCES) == set(FOUND_SOURCES)
        assert M.SEEDS_FROM_PAINTED not in M.MATERIALISED_MARKER_SOURCES

    @pytest.mark.parametrize("source", FOUND_SOURCES)
    def test_the_predicate_says_yes(self, widget, source):
        h = _harness(widget, source)
        assert h._marker_source_materialises() is True

    def test_the_hand_painted_source_has_nothing_to_materialise(self, widget):
        h = _harness(widget, 0)
        assert h._marker_source_materialises() is False


class TestGuideMarkersIsNotCreatedAnyMore:
    def test_the_helper_is_gone(self):
        assert not hasattr(_K(), "_show_marker_layer"), (
            "a second colormap drawing the same markers is exactly what "
            "U1b removed"
        )

    def test_the_finish_handler_does_not_reach_for_it(self):
        src = inspect.getsource(_K()._on_seed_find_finished)
        assert "_show_marker_layer" not in src

    @pytest.mark.parametrize("source", FOUND_SOURCES)
    def test_a_find_adds_no_temporary_layer(self, widget, source):
        h = _harness(widget, source)
        _find(h, _Report([(2, 0, 0), (3, 3, 3)]))
        assert h.temp_layers_added == [], h.temp_layers_added


# --------------------------------------------------------------------------- #
# what the object looks like afterwards
# --------------------------------------------------------------------------- #
class TestTheMarkersLandOnTheRefineLayer:
    @pytest.mark.parametrize("source", FOUND_SOURCES)
    def test_the_roi_holds_exactly_n_marker_ids(self, widget, source):
        h = _harness(widget, source)
        _find(h, _Report([(2, 0, 0), (3, 3, 3)]))
        roi = h._roi()
        new = set(int(v) for v in np.unique(roi)) - {0, 7}
        assert len(new) == 2, sorted(new)
        assert int(roi[2, 0, 0]) in new
        assert int(roi[3, 3, 3]) in new

    @pytest.mark.parametrize("source", FOUND_SOURCES)
    def test_the_panel_lists_exactly_those_ids(self, widget, source):
        from qtpy.QtCore import Qt

        h = _harness(widget, source)
        _find(h, _Report([(2, 0, 0), (3, 3, 3)]))
        listed = [widget.split_marker_list.item(i).data(Qt.UserRole)
                  for i in range(widget.split_marker_list.count())]
        roi = h._roi()
        on_layer = sorted(set(int(v) for v in np.unique(roi)) - {0, 7})
        assert sorted(listed) == on_layer

    @pytest.mark.parametrize("source", FOUND_SOURCES)
    def test_every_row_still_carries_its_size_and_place(self, widget, source):
        """(e) -- the stats follow the new numbering for EVERY source,
        not only for the one that already materialised."""
        h = _harness(widget, source)
        _find(h, _Report([(2, 0, 0), (3, 3, 3)]))
        for i in range(widget.split_marker_list.count()):
            tip = widget.split_marker_list.item(i).toolTip()
            assert "voxel" in tip and " at " in tip, tip

    @pytest.mark.parametrize("source", FOUND_SOURCES)
    def test_the_side_views_are_refreshed(self, widget, source):
        """The whole point: the Refine layer is mirrored, so the three
        subviewers have to be told it changed."""
        h = _harness(widget, source)
        _find(h, _Report([(2, 0, 0), (3, 3, 3)]))
        assert h.subviewer_refreshes >= 1

    @pytest.mark.parametrize("source", FOUND_SOURCES)
    def test_a_second_find_replaces_the_first(self, widget, source):
        h = _harness(widget, source)
        _find(h, _Report([(2, 0, 0), (3, 3, 3)]))
        _find(h, _Report([(2, 1, 1)]))
        roi = h._roi()
        new = set(int(v) for v in np.unique(roi)) - {0, 7}
        assert len(new) == 1, sorted(new)

    @pytest.mark.parametrize("source", FOUND_SOURCES)
    def test_a_hand_painted_marker_survives_the_find(self, widget, source):
        h = _harness(widget, source, painted=[(101, (0, 0, 0))])
        _find(h, _Report([(2, 0, 0)]))
        assert int(h._roi()[0, 0, 0]) == 101
        assert 101 in h._split_painted_marker_ids

    @pytest.mark.parametrize("source", FOUND_SOURCES)
    def test_a_marker_with_no_voxels_is_not_listed(self, widget, source):
        """The report says the marker is there; the crop says it is not.
        Only what is really on the layer is a marker."""
        h = _harness(widget, source)
        report = _Report([(2, 0, 0)])
        report.markers.append(
            {"id": 99, "voxels": 0, "centroid_zyx": (0.0, 0.0, 0.0)})
        _find(h, report)
        roi = h._roi()
        new = set(int(v) for v in np.unique(roi)) - {0, 7}
        assert len(new) == 1


class TestTheSnapBackPutsThemBack:
    """(c)+(d) -- the Preview snapshot is taken AFTER the Find, so it
    holds the materialised markers and a snap-back shows them again."""

    @pytest.mark.parametrize("source", FOUND_SOURCES)
    def test_the_markers_come_back_after_a_preview(self, widget, source):
        h = _harness(widget, source)
        _find(h, _Report([(2, 0, 0), (3, 3, 3)]))
        # what `_preview_reshape` snapshots, at the moment it snapshots it
        h._refine_pre_preview = h._roi().copy()
        markers = sorted(set(int(v) for v in np.unique(h._roi())) - {0, 7})
        # the cut writes piece ids over the whole ROI
        h._refine_layer.data[h._split_roi] = 55
        h._split_previewed = True
        h._reshape_result = object()
        h._leave_reshape_review("Markers changed - press Preview to cut again.")
        back = sorted(set(int(v) for v in np.unique(h._roi())) - {0, 7})
        assert back == markers


# --------------------------------------------------------------------------- #
# what the recipe sends afterwards
# --------------------------------------------------------------------------- #
class TestTheRecipeReadsThemBackAsPaint:
    def test_the_remap_covers_every_find_source(self):
        src = inspect.getsource(_K()._reshape_recipe)
        assert "FIND_MARKER_SOURCES" in src, (
            "the index-1 re-map has to widen to every source that "
            "materialises, or a found marker is silently re-found"
        )

    def test_the_materialise_receipt_gates_it(self):
        """`_reshape_found_seeds is None` is the receipt: it is nulled
        exactly when the voxels landed on the layer, so a refused
        materialise still reaches the detector."""
        src = inspect.getsource(_K()._reshape_recipe)
        assert "_reshape_found_seeds" in src

    @pytest.mark.parametrize("source", FOUND_SOURCES)
    def test_the_found_seeds_are_handed_over_to_the_layer(
            self, widget, source):
        h = _harness(widget, source)
        _find(h, _Report([(2, 0, 0), (3, 3, 3)]))
        assert h._reshape_found_seeds is None, (
            "the markers are on the layer now; keeping the crop as well "
            "would list every marker twice"
        )
        assert h._reshape_found_marker_ids == []


# --------------------------------------------------------------------------- #
# H1 (audit, 2026-09-20) -- the receipt has to be the VOXELS, not the crop
# --------------------------------------------------------------------------- #
from .test_reshape_split_orchestration import (  # noqa: E402
    FakeAdv,
    FakeAnnotator,
    FakeLayer,
    bind as _obind,
)


def _recipe_fake(source, painted=(), materialised=(), guide=True):
    """A fake that can run the REAL `_reshape_recipe`."""
    vol = np.zeros((6, 6, 6), np.int32)
    vol[2:4, 2:4, 2:4] = 7
    roi = (slice(0, 6),) * 3
    refine = FakeLayer(np.zeros((6, 6, 6), np.int32), "Refine")
    fake = FakeAnnotator(vol, roi, 7, adv=FakeAdv({"seeds_from": source}),
                         refine=refine)
    fake.viewer = type("V", (), {"layers": []})()
    _obind(fake, "_reshape_recipe")
    for pid, place in painted:
        fake._refine_layer.data[place] = pid
    fake._split_painted_marker_ids = [pid for pid, _ in painted]
    fake._split_seed_ids = list(fake._split_painted_marker_ids)
    fake._materialised_ids = list(materialised)
    if guide:
        g = np.zeros((6, 6, 6), np.int32)
        g[0, 0, 0] = 1
        g[5, 5, 5] = 2
        fake._reshape_guide = g
    return fake


class TestTheReceiptIsTheVoxelsNotTheCrop:
    """`_reshape_found_seeds is None` was the receipt, and it is None in
    three different situations, only one of which means "materialised":

    * before any Find at all;
    * after a marker knob moved (the handler nulls it);
    * after a Find that really did paint markers.

    Only the third may re-point the recipe at ``paint``.  The other two
    made the tool cut from something other than what the user asked for
    -- silently.  ``_materialised_ids`` is non-empty only in the third,
    so it is the receipt.
    """

    def test_a_mask_source_with_NO_find_still_sends_labels(self):
        """(a) Load a mask, paint ONE marker by hand, press Preview.

        The guide mask is the marker set the user chose; promoting to
        ``paint`` here throws it away and cuts the object from the one
        hand-painted dot instead."""
        f = _recipe_fake(2, painted=[(101, (0, 0, 0))])
        spec, _images, _seeds = f._reshape_recipe()
        assert spec is not None
        assert spec.seeds == "labels", (
            "the loaded mask was dropped in favour of one painted marker"
        )

    def test_the_external_mask_source_with_NO_find_does_the_same(self):
        """Source 3 takes the identical gate; it is here because the two
        mask sources are the ones a user reaches with a marker already
        painted, having loaded the mask and then added one by hand."""
        f = _recipe_fake(3, painted=[(101, (0, 0, 0))])
        spec, _images, _seeds = f._reshape_recipe()
        assert spec is not None
        assert spec.seeds == "labels", spec.seeds

    def test_a_find_that_DID_materialise_re_points_at_paint(self):
        """...and the case the re-point exists for still works."""
        f = _recipe_fake(2, painted=[(101, (0, 0, 0))],
                         materialised=[101])
        spec, _images, _seeds = f._reshape_recipe()
        assert spec is not None
        assert spec.seeds == "paint"

    def test_a_REFUSED_materialise_falls_through_to_the_guide(self):
        """`_materialised_ids` stays empty when nothing landed, which is
        the same answer as "no Find yet" -- and it is the right one."""
        f = _recipe_fake(2, painted=[(101, (0, 0, 0))], materialised=[])
        spec, _images, _seeds = f._reshape_recipe()
        assert spec is not None
        assert spec.seeds == "labels"

    def test_the_gate_reads_the_ids_and_not_the_crop(self):
        src = inspect.getsource(_K()._reshape_recipe)
        i = src.index("_materialised =")
        line = src[i:src.index("\n", i)]
        assert "_materialised_ids" in line, line
        assert "_reshape_found_seeds" not in line, line


class TestAMarkerKnobTakesTheMarkersWithIt:
    """(b) `_on_reshape_markers_invalidated` nulled the CROP and left the
    voxels painted on the object.

    The panel empties its list at the same moment, so
    `get_checked_marker_ids()` answers None -- there is nothing left to
    filter with -- while `_split_painted_marker_ids` still names the old
    generation.  The next Preview therefore cut from the markers the
    user had just invalidated, and the knob they turned was never used.
    The count line says "Find again, or Preview to find them fresh",
    which is only true once the old ones are really gone.
    """

    def _h(self, widget):
        h = _harness(widget, 4, painted=[(101, (0, 0, 0)),
                                         (102, (1, 1, 1))])
        h._materialised_ids = [101, 102]
        # (`_bind` carries `_PARKING` for every fake: with the list
        # emptied no tick can un-park a marker, so the handler is the
        # one that puts them back.)
        _bind(h, "_on_reshape_markers_invalidated",
              "_drop_materialised_markers")
        return h

    def test_the_voxels_are_taken_off_the_object(self, widget):
        h = self._h(widget)
        assert 101 in np.unique(h._roi())
        h._on_reshape_markers_invalidated()
        left = set(int(v) for v in np.unique(h._roi()))
        assert 101 not in left and 102 not in left, left

    def test_the_bookkeeping_list_loses_them_too(self, widget):
        h = self._h(widget)
        h._on_reshape_markers_invalidated()
        assert h._split_painted_marker_ids == []
        assert h._materialised_ids == []

    def test_a_HAND_painted_marker_is_never_touched(self, widget):
        """Only what a Find wrote comes out.  A marker the user painted
        is theirs."""
        h = self._h(widget)
        h._refine_layer.data[h._split_roi][3, 3, 3] = 555
        h._split_painted_marker_ids.append(555)
        h._on_reshape_markers_invalidated()
        assert 555 in set(int(v) for v in np.unique(h._roi()))
        assert 555 in h._split_painted_marker_ids

    def test_the_side_views_are_told(self, widget):
        h = self._h(widget)
        before = h.subviewer_refreshes
        h._on_reshape_markers_invalidated()
        assert h.subviewer_refreshes > before

    def test_the_object_itself_survives(self, widget):
        h = self._h(widget)
        h._on_reshape_markers_invalidated()
        assert (h._roi() == 7).any(), "the object was erased with the markers"

    def test_the_next_preview_is_not_promoted(self, widget):
        """The whole point: with the old generation gone, the recipe
        falls back to the detector and the new knob value is used."""
        h = self._h(widget)
        h._on_reshape_markers_invalidated()
        f = _recipe_fake(2, painted=(), materialised=h._materialised_ids)
        spec, _i, _s = f._reshape_recipe()
        assert spec is not None and spec.seeds == "labels"

    def test_the_handler_delegates_to_the_shared_helper(self):
        src = inspect.getsource(_K()._on_reshape_markers_invalidated)
        assert "_drop_materialised_markers" in src

    def test_the_count_line_promise_is_now_true(self, widget):
        """The panel says Preview will find them fresh; that sentence is
        only honest because the old ones are gone."""
        from napari_epicycle.annotator.advanced_features_widget import (
            AdvancedFeaturesWidget,
        )

        assert "Preview to find them fresh" in (
            AdvancedFeaturesWidget._MARKERS_STALE_TEXT)


class TestTheRecipeRoutingIsMeasuredNotAsserted:
    """M4: the promotion was pinned by reading the recipe's SOURCE.

    Source order tells you a branch exists, not which one a given state
    takes -- and the H1 bug lived entirely in which one it took.  These
    run the real recipe.
    """

    @pytest.mark.parametrize("source", [2, 3])
    def test_before_a_find_the_mask_is_what_gets_sent(self, source):
        f = _recipe_fake(source, painted=[(101, (0, 0, 0))])
        spec, _i, _s = f._reshape_recipe()
        assert spec.seeds == "labels"

    @pytest.mark.parametrize("source", [2, 3])
    def test_after_a_find_the_painted_markers_are(self, source):
        f = _recipe_fake(source, painted=[(101, (0, 0, 0))],
                         materialised=[101])
        spec, _i, seeds = f._reshape_recipe()
        assert spec.seeds == "paint"
        assert seeds is not None
        assert 101 in set(int(v) for v in np.unique(seeds))

    def test_the_seed_array_holds_the_materialised_ids(self):
        f = _recipe_fake(2, painted=[(101, (0, 0, 0)), (102, (5, 5, 5))],
                         materialised=[101, 102])
        _spec, _i, seeds = f._reshape_recipe()
        got = set(int(v) for v in np.unique(seeds)) - {0}
        assert got == {101, 102}, got


class TestARefusedMaterialiseKeepsTheCrop:
    """M4: the `_materialise_ok is False` branch had no test at all.

    It is the branch that makes a shape mismatch survivable: the crop is
    kept, so the engine finds its own markers again on the next Preview
    and the promotion rule still merges in a hand-painted addition.
    """

    def _refused(self, widget):
        """A report whose seed crop is NOT the ROI's shape."""
        h = _harness(widget, 4)
        rep = _Report([(2, 0, 0), (3, 3, 3)])
        rep.seeds = np.zeros((2, 2, 2), np.int32)   # wrong shape on purpose
        return h, rep

    def test_the_materialise_reports_failure(self, widget):
        h, rep = self._refused(widget)
        _find(h, rep)
        assert h._materialise_ok is False

    def test_the_crop_is_KEPT(self, widget):
        h, rep = self._refused(widget)
        _find(h, rep)
        assert h._reshape_found_seeds is not None, (
            "the crop is the fallback: without it the next Preview has "
            "neither painted markers nor a seed array"
        )

    def test_it_says_so_on_the_console(self, widget, capsys):
        h, rep = self._refused(widget)
        _find(h, rep)
        out = capsys.readouterr().out
        assert "nothing was materialised" in out, out[-400:]

    def test_nothing_is_recorded_as_materialised(self, widget):
        h, rep = self._refused(widget)
        _find(h, rep)
        assert h._materialised_ids == []

    def test_the_recipe_is_NOT_promoted(self, widget):
        """The receipt is the ids, and there are none."""
        h, rep = self._refused(widget)
        _find(h, rep)
        f = _recipe_fake(2, painted=[(101, (0, 0, 0))],
                         materialised=h._materialised_ids)
        spec, _i, _s = f._reshape_recipe()
        assert spec.seeds == "labels"

    def test_a_previous_generation_is_left_alone(self, widget):
        """M1: the shape check runs BEFORE anything is erased, so a
        refusal does not strip the markers that are already drawn."""
        h = _harness(widget, 4, painted=[(101, (0, 0, 0))])
        h._materialised_ids = [101]
        rep = _Report([(2, 0, 0)])
        rep.seeds = np.zeros((2, 2, 2), np.int32)
        _find(h, rep)
        assert 101 in set(int(v) for v in np.unique(h._roi())), (
            "the refused materialise took the previous markers off the "
            "object and left the panel listing markers drawn nowhere"
        )
