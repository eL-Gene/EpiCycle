"""REFINE Brick B3 audit fix pass -- the code bugs behind help promises.

Each class is one finding of the B3 help-accuracy / lifecycle audits, and
each was written FAILING before its fix:

* Flood Fill (HIGH): a stray click on background inside the object's box
  filled a separate 1,584-voxel slab and staged it for Accept;
* the chain's region growth compared margin NEEDS, not the object's real
  clearance, so Dilate +10 twice clipped silently at the region wall;
* Settle's own "nothing moved" sentence was thrown away;
* the shape wall kinds under Settle grow to the band limit -- an amber note;
* the probability-overlay tick was a silent no-op without an object or map;
* the lifecycle LOWs and the small LOWs.
"""

from __future__ import annotations

import numpy as np
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


# --------------------------------------------------------------------------- #
# 1. Flood Fill: the fill must touch the object and be enclosed
# --------------------------------------------------------------------------- #
def _padded_cube():
    """The auditor's probe: a 4-voxel cube in a bbox+15 crop."""
    v = np.zeros((10, 10, 10), np.uint16)
    v[3:7, 3:7, 3:7] = 1
    return np.pad(v, 15)


class TestFloodFillIsForAGapInThisObject:
    def test_a_stray_click_on_open_background_is_refused(self):
        """The measured bug: XY at (18, 2, 2) filled the whole slice's
        background (1,584 voxels).  Every face of this crop is an interior
        wall, and the region runs out through them."""
        from packages.epicycle_annotator.refine import (
            FLOOD_NOT_ENCLOSED,
            flood_fill_label,
        )

        roi = _padded_cube()
        _r, n, msg = flood_fill_label(roi, 1, (18, 2, 2), axis="XY",
                                      walls=(True,) * 6)
        assert n == 0
        assert msg == FLOOD_NOT_ENCLOSED

    def test_empty_space_that_does_not_touch_the_object_is_refused(self):
        from packages.epicycle_annotator.refine import (
            FLOOD_NO_CONTACT,
            flood_fill_label,
        )

        v = np.zeros((20, 20, 20), np.int32)
        v[2:6, 2:6, 2:6] = 1                     # this object
        v[10:18, 10:18, 10:18] = 9               # a hollow neighbour...
        v[12:16, 12:16, 12:16] = 0               # ...with a cavity in it
        _r, n, msg = flood_fill_label(v, 1, (13, 13, 13), axis="3D")
        assert n == 0
        assert msg == FLOOD_NO_CONTACT
        assert "does not touch this object" in msg

    def test_a_gap_that_touches_the_object_diagonally_still_fills(self):
        """26-connectivity: a corner contact counts."""
        from packages.epicycle_annotator.refine import flood_fill_label

        v = np.zeros((9, 9, 9), np.int32)
        v[:, :, :] = 9                           # everything a neighbour
        v[0:4, 0:4, 0:4] = 1                     # this object in a corner
        v[4, 4, 4] = 0                           # a 1-voxel gap, diagonal
        _r, n, _m = flood_fill_label(v, 1, (4, 4, 4), axis="3D",
                                     walls=(True,) * 6)
        assert n == 1

    def test_an_enclosed_pocket_inside_the_crop_still_fills(self):
        from packages.epicycle_annotator.refine import flood_fill_label

        v = np.zeros((20, 20, 20), np.int32)
        v[4:16, 4:16, 4:16] = 1
        v[8:12, 8:12, 8:12] = 0                  # an enclosed void
        _r, n, _m = flood_fill_label(v, 1, (10, 10, 10), axis="3D",
                                     walls=(True,) * 6)
        assert n == 64

    def test_a_seed_outside_the_region_says_where_to_put_it(self):
        from packages.epicycle_annotator.refine import (
            FLOOD_SEED_OUTSIDE,
            flood_fill_label,
        )

        _r, n, msg = flood_fill_label(_padded_cube(), 1, (99, 99, 99))
        assert n == 0
        assert msg == FLOOD_SEED_OUTSIDE
        assert "Put the crosshair on or next to this object." in msg

    def test_max_size_is_still_the_second_guard(self):
        from packages.epicycle_annotator.refine import flood_fill_label

        v = np.zeros((20, 20, 20), np.int32)
        v[4:16, 4:16, 4:16] = 1
        v[6:14, 6:14, 6:14] = 0                  # a 512-voxel void
        _r, n, msg = flood_fill_label(v, 1, (10, 10, 10), axis="3D",
                                      max_fill_size=100, walls=(True,) * 6)
        assert n == 0 and "exceeds" in msg


class TestTheOrchestratorShowsTheFloodReason:
    """The refusal reaches S1 as ITS sentence, not the generic one."""

    def _fake(self, vol, crosshair):
        from tests.annotator.test_refine_cross_method_chain import Fake

        f = Fake(vol)
        f.adv.method = "flood_fill"
        f.adv.refine_ff_axis_combo = _Val("XY")
        f.adv.refine_ff_max_size_sb = _Val(50000)
        z, y, x = crosshair
        f.viewer_model_xy = _Dims(z)
        f.viewer_model_xz = _Dims(y)
        f.viewer_model_yz = _Dims(x)
        return f

    def test_a_stray_click_is_refused_in_S1_and_nothing_is_staged(
            self, qapp, monkeypatch):
        import napari_epicycle.annotator.curation_3d_widget as mod
        from packages.epicycle_annotator.refine import FLOOD_NOT_ENCLOSED
        from tests.annotator.test_refine_cross_method_chain import _Layer

        monkeypatch.setattr(mod, "Labels", _Layer)
        vol = np.zeros((60, 60, 60), np.int32)
        vol[25:29, 25:29, 25:29] = 7
        before = vol.copy()
        # background, inside the object's bbox+15 box, same z as the object
        f = self._fake(vol, (26, 12, 12))
        f._preview_smooth()
        assert np.array_equal(vol, before)
        assert f._smooth_running is False
        assert f.last("status_text") == FLOOD_NOT_ENCLOSED


class _Val:
    def __init__(self, v):
        self._v = v

    def value(self):
        return self._v

    def currentText(self):
        return self._v

    def currentIndex(self):
        return self._v


class _Dims:
    def __init__(self, v):
        self.dims = type("D", (), {"current_step": (v, 0, 0)})()


# --------------------------------------------------------------------------- #
# 2. The chain grows on the object's REAL clearance, not on margin needs
# --------------------------------------------------------------------------- #
def _chain_fake(vol, qapp, monkeypatch):
    import napari_epicycle.annotator.curation_3d_widget as mod
    from tests.annotator.test_refine_cross_method_chain import Fake, _Layer

    monkeypatch.setattr(mod, "Labels", _Layer)
    f = Fake(vol)
    f.adv.method = "dilate_erode"
    f.adv.refine_de_radius_sb = _Val(10)
    return f


class TestTheChainGrowsOnRealClearance:
    def _cube(self):
        v = np.zeros((90, 90, 90), np.int32)
        v[40:50, 40:50, 40:50] = 7
        return v

    def test_dilate_10_twice_equals_a_fresh_dilate_20(self, qapp, monkeypatch):
        """The measured bug: 39,280 voxels chained vs 44,920 unclipped --
        both steps need the shared 15, so the old rule never grew, and the
        second +10 hit the region wall 5 voxels out."""
        from packages.epicycle_annotator.refine import dilate_erode_label

        vol = self._cube()
        want, _n, _m = dilate_erode_label(vol.copy(), 7, radius=20)
        f = _chain_fake(vol, qapp, monkeypatch)
        f._preview_smooth()
        f._preview_smooth()
        assert f._smooth_chain_count == 2
        assert int((vol == 7).sum()) == int((want == 7).sum()) == 44920
        assert np.array_equal(vol == 7, want == 7)

    def test_discard_is_byte_identical_over_the_grown_region(
            self, qapp, monkeypatch):
        vol = self._cube()
        before = vol.copy()
        f = _chain_fake(vol, qapp, monkeypatch)
        f._preview_smooth()
        roi1 = f._smooth_roi
        f._preview_smooth()
        assert f._smooth_roi != roi1, "the region did not grow"
        f._reject_smooth()
        assert np.array_equal(vol, before)

    def test_an_object_at_the_volume_edge_does_not_grow_forever(
            self, qapp, monkeypatch):
        """Where the ROI IS the volume edge the clearance is infinite, so an
        object that spans the whole volume never asks for more room, however
        many steps are chained -- and one that touches five faces grows only
        on the sixth, where it really did get closer to a wall."""
        v = np.zeros((30, 30, 30), np.int32)
        v[:, :, :] = 7
        v[10:20, 10:20, 10:20] = 0              # a hole, so erode has work
        f = _chain_fake(v, qapp, monkeypatch)
        f.adv.refine_de_radius_sb = _Val(-1)
        f._preview_smooth()
        roi1 = f._smooth_roi
        assert f._refine_room("dilate_erode", {})[1] == float("inf")
        for _ in range(3):
            f._preview_smooth()
        assert f._smooth_roi == roi1

        w = np.zeros((40, 40, 60), np.int32)
        w[0:40, 0:40, 0:6] = 7                  # touches 5 of 6 faces
        g = _chain_fake(w, qapp, monkeypatch)
        g.adv.refine_de_radius_sb = _Val(2)
        g._preview_smooth()
        r1 = g._smooth_roi
        g._preview_smooth()
        r2 = g._smooth_roi
        assert r2[:2] == r1[:2] == (slice(0, 40), slice(0, 40))
        assert r2[2].start == 0 and r2[2].stop > r1[2].stop

    def test_the_clearance_is_measured_off_the_object(self, qapp,
                                                       monkeypatch):
        vol = self._cube()
        f = _chain_fake(vol, qapp, monkeypatch)
        f._preview_smooth()                     # object now 30 wide, ROI +15
        need, have = f._refine_room("dilate_erode", {})
        assert need == 15 and have == 5


# --------------------------------------------------------------------------- #
# 3. Settle's own "nothing moved" sentence reaches the user
# --------------------------------------------------------------------------- #
class TestSettleSaysWhyNothingMoved:
    def test_the_engine_s_reason_is_the_state_9_sentence(self, qapp):
        """The auditor's case: random walker on a ball with the Pinch-points
        wall moves nothing, and `settle_label` says why -- "the wall signal
        already runs along this object's outline" -- but the orchestrator
        showed its generic "touches its neighbours everywhere" instead,
        which is false for a ball in open space."""
        from packages.epicycle_annotator.reshape.settle import (
            settle_with_walls,
        )
        from tests.annotator.test_refine_lock_lifecycle import Fake, _K

        Z = np.indices((40, 60, 60))
        ball = ((Z[0] - 20) ** 2 + (Z[1] - 30) ** 2 + (Z[2] - 30) ** 2) <= 64
        vol = np.zeros((40, 60, 60), np.uint16)
        vol[ball] = 1
        res = settle_with_walls(vol, 1, wall_kind="distance", band_um=3.0,
                                solver="random_walker")
        assert res.status == "unchanged"
        engine_says = res.messages[-1]
        assert "wall signal already runs along" in engine_says

        f = Fake(vol.astype(np.int32))
        f._smooth_label_id = 1
        K = _K()
        for name in ("_finish_settle_result", "_refine_band_um"):
            setattr(f, name, getattr(K, name).__get__(f, type(f)))
        f._finish_settle_result(res, 1, f._smooth_roi)
        assert f.last_ui("status_text") == engine_says


# --------------------------------------------------------------------------- #
# 6. The two lifecycle LOWs
# --------------------------------------------------------------------------- #
class TestTheLifecycleLows:
    def test_low1_the_channel_crop_is_never_a_view_of_the_layer(self, qapp):
        from tests.annotator.test_refine_lock_lifecycle import Fake, _K

        layer_data = np.random.default_rng(0).random((10, 12, 12))  # float64

        class _Layer:
            data = layer_data
            multiscale = False

        class _Adv:
            def refine_channel_name(self):
                return "membrane"

        f = Fake(np.zeros((10, 12, 12), np.int32))
        f._get_advanced_features_widget = lambda: _Adv()
        f._layer_by_name = lambda layers, name: _Layer()
        f.viewer = type("V", (), {"layers": []})()
        K = _K()
        crop = K._get_refine_channel_roi.__get__(f, type(f))(
            (slice(1, 5), slice(2, 6), slice(0, 4)))
        assert not np.shares_memory(crop, layer_data)

    def test_low2_a_refused_settle_drops_its_pending_picture(self, qapp):
        from packages.epicycle_annotator.reshape.settle import SettleResult
        from tests.annotator.test_refine_lock_lifecycle import Fake, _K

        vol = np.zeros((10, 12, 12), np.int32)
        vol[2:8, 3:9, 3:9] = 7
        f = Fake(vol)
        K = _K()
        f._finish_settle_result = K._finish_settle_result.__get__(f, Fake)
        f._refine_pending_elevation = (np.zeros(1), (slice(0, 1),))
        res = SettleResult(result_roi=np.zeros_like(vol), status="refused",
                           messages=["no."])
        f._finish_settle_result(res, 7, f._smooth_roi)
        assert f._refine_pending_elevation is None


# --------------------------------------------------------------------------- #
# 4. A shape wall under Settle: an amber note, not a refusal
# --------------------------------------------------------------------------- #
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


class TestTheShapeWallNoteOnSettle:
    def _set(self, w, key, kind):
        from napari_epicycle.annotator.advanced_features_widget import (
            REFINE_METHOD_INDEX,
            REFINE_WALL_KINDS,
        )

        w.refine_method_combo.setCurrentIndex(REFINE_METHOD_INDEX[key])
        w.refine_wall_kind_combo.setCurrentIndex(REFINE_WALL_KINDS.index(kind))

    @pytest.mark.parametrize("kind", ["distance", "skeleton"])
    def test_a_shape_wall_on_settle_shows_the_note(self, panel, kind):
        from napari_epicycle.annotator.advanced_features_widget import (
            REFINE_SHAPE_WALL_NOTE,
        )

        self._set(panel, "settle", kind)
        note = panel.refine_shape_wall_note
        assert note.isHidden() is False
        assert note.text() == REFINE_SHAPE_WALL_NOTE
        assert "Shape-based walls are for splitting" in note.text()

    @pytest.mark.parametrize("kind", ["membrane", "probability", "gradient"])
    def test_any_other_wall_hides_it(self, panel, kind):
        self._set(panel, "settle", kind)
        assert panel.refine_shape_wall_note.isHidden() is True

    def test_the_contour_never_shows_it(self, panel):
        self._set(panel, "contour", "distance")
        assert panel.refine_shape_wall_note.isHidden() is True

    def test_it_toggles_with_the_combo(self, panel):
        self._set(panel, "settle", "distance")
        assert panel.refine_shape_wall_note.isHidden() is False
        self._set(panel, "settle", "membrane")
        assert panel.refine_shape_wall_note.isHidden() is True
        self._set(panel, "settle", "skeleton")
        assert panel.refine_shape_wall_note.isHidden() is False

    def test_the_item_help_says_so_too(self):
        from napari_epicycle.annotator.advanced_features_widget import (
            ITEM_HELP,
            REFINE_WALL_KINDS,
        )

        for kind in ("distance", "skeleton"):
            text = ITEM_HELP["refine_wall_kind_combo"][
                REFINE_WALL_KINDS.index(kind)]
            assert "splitting" in text, text

    def test_the_measured_growth_behind_it(self):
        """The reason the note exists: norm(-EDT) is maximal everywhere
        outside the object, so the default solver runs to the band."""
        from packages.epicycle_annotator.reshape.settle import (
            settle_with_walls,
        )

        Z = np.indices((40, 60, 60))
        ball = ((Z[0] - 20) ** 2 + (Z[1] - 30) ** 2 + (Z[2] - 30) ** 2) <= 64
        vol = np.zeros((40, 60, 60), np.uint16)
        vol[ball] = 1
        res = settle_with_walls(vol, 1, wall_kind="distance", band_um=3.0)
        assert res.n_grown > 0.5 * int(ball.sum()), res.n_grown   # +75 %

    def test_settle_with_the_note_on_stays_under_its_ceiling(self, panel):
        """The note adds three lines, but a shape wall needs no channel, so
        the channel card and its prepare block leave: measured 546 px, far
        under Settle's 2.9 screens (710 px)."""
        from qtpy.QtWidgets import QApplication

        from tests.annotator.test_panel_skeleton import (
            TestTheRefinePageAfterBrickB1 as B1,
            _slots,
        )

        b1 = B1()
        b1._shown(panel)
        try:
            panel.set_refine_guided_grow_available(False)
            self._set(panel, "settle", "distance")
            panel.refine_prep_btn.setChecked(True)
            QApplication.processEvents()
            content = _slots(panel, "Refine")["content"].sizeHint().height()
            screen = b1._screen_h(panel)
            assert panel.refine_shape_wall_note.isHidden() is False
            assert content <= b1.REFINE_SETTLE_PREP_SCREENS * screen
        finally:
            panel.hide()


# --------------------------------------------------------------------------- #
# 5. The probability-overlay tick is never a silent no-op
# --------------------------------------------------------------------------- #
class TestTheProbOverlayTickSaysWhy:
    def _fake(self, panel, vol, selected, prob):
        from tests.annotator.test_refine_lock_lifecycle import Fake, _K

        f = Fake(vol)
        f._get_advanced_features_widget = lambda: panel
        f._get_selected_label = lambda: selected
        f._get_prob_roi = lambda roi: prob
        removed = []
        f._remove_prob_overlay = lambda: removed.append(1)
        K = _K()
        f._toggle_prob_overlay = K._toggle_prob_overlay.__get__(f, type(f))
        f._snapshot_label_roi = K._snapshot_label_roi
        panel.refine_prob_overlay_cb.toggled.connect(f._toggle_prob_overlay)
        return f

    def test_no_object_selected_unticks_and_says_so(self, panel):
        from napari_epicycle.annotator.advanced_features_widget import (
            REFINE_PROB_OVERLAY_NO_OBJECT,
        )

        vol = np.zeros((8, 8, 8), np.int32)
        self._fake(panel, vol, 0, None)
        panel.refine_prob_overlay_cb.setChecked(True)
        assert panel.refine_prob_overlay_cb.isChecked() is False
        assert panel.refine_prob_overlay_msg.isHidden() is False
        assert panel.refine_prob_overlay_msg.text() == (
            REFINE_PROB_OVERLAY_NO_OBJECT)

    def test_no_map_unticks_and_says_so(self, panel):
        from napari_epicycle.annotator.advanced_features_widget import (
            REFINE_PROB_OVERLAY_NO_MAP,
        )

        vol = np.zeros((8, 8, 8), np.int32)
        vol[2:5, 2:5, 2:5] = 7
        self._fake(panel, vol, 7, None)
        panel.refine_prob_overlay_cb.setChecked(True)
        assert panel.refine_prob_overlay_cb.isChecked() is False
        assert panel.refine_prob_overlay_msg.text() == REFINE_PROB_OVERLAY_NO_MAP

    def test_the_reason_goes_when_the_tick_is_taken_off(self, panel):
        panel.set_refine_prob_overlay_message("x.")
        assert panel.refine_prob_overlay_msg.isHidden() is False
        panel.set_refine_prob_overlay_message(None)
        assert panel.refine_prob_overlay_msg.isHidden() is True


# --------------------------------------------------------------------------- #
# 7. The small LOWs
# --------------------------------------------------------------------------- #
class TestTheSmallLows:
    def test_no_object_says_select_an_object(self, qapp, monkeypatch):
        import napari_epicycle.annotator.curation_3d_widget as mod
        from tests.annotator.test_refine_cross_method_chain import Fake

        class _NoSel:
            selected_label = 0

        monkeypatch.setattr(mod, "Labels", _NoSel)
        f = Fake(np.zeros((8, 8, 8), np.int32))
        f._layer_by_name = lambda layers, name: _NoSel()
        f._preview_smooth()
        assert f.last("status_text") == mod.REFINE_SELECT_OBJECT
        assert mod.REFINE_SELECT_OBJECT == "Select an object first."

    def test_the_gate_greys_preview_with_no_object(self, qapp):
        import napari_epicycle.annotator.curation_3d_widget as mod
        from tests.annotator.test_refine_cross_method_chain import Fake, _K

        f = Fake(np.zeros((8, 8, 8), np.int32))
        f._get_selected_label = lambda: 0
        K = _K()
        f._refresh_refine_object_row = K._refresh_refine_object_row.__get__(
            f, Fake)
        f._refine_s1_text = lambda adv: ""
        assert f._refresh_refine_preview_gate() is False
        assert f.last("preview_enabled") is False
        assert f.last("preview_reason") == mod.REFINE_SELECT_OBJECT
        assert f.last("status_text") == mod.REFINE_SELECT_OBJECT
        # ...and an object puts it back
        f._get_selected_label = lambda: 7
        f.sync_manager.shared_labels["labels"][2:4, 2:4, 2:4] = 7
        assert f._refresh_refine_preview_gate() is True
        assert f.last("preview_enabled") is True

    def test_the_mid_chain_refusal_names_reshape(self):
        import inspect

        from tests.annotator.test_refine_cross_method_chain import _K

        src = inspect.getsource(_K()._preview_smooth)
        assert "Split is in progress." not in src
        assert "Reshape is in progress." in src

    def test_the_traced_sentence_adds_up(self):
        import re

        from tests.annotator.test_refine_serial_2d import _cylinder, _flat
        from packages.epicycle_annotator.refine import contour_refine

        vol = _cylinder()
        vol[2] = 0
        vol[2, 20, 20] = 4
        res = contour_refine(vol, 4, mode="snap", elevation=_flat(vol),
                             balloon=-1.0, iterations=3, limit_um=5.0,
                             orientation="XY")
        text = res.details["traced_text"]
        d = res.details["dropped"]
        assert d >= 1
        nums = [int(x) for x in re.findall(r"\d+", text)]
        traced, total = nums[0], nums[1]
        kept, filled = nums[2], nums[3]
        assert (traced, kept) == (res.details["traced"], d)
        assert traced + kept + filled == total, text

    def test_the_dilate_readout_in_xy_has_no_depth_number(self, panel):
        panel.set_refine_spacing((3.0, 1.0, 1.0))
        panel.refine_de_radius_sb.setValue(2)
        panel.refine_de_axis_combo.setCurrentIndex(0)             # 3D
        assert "through depth" in panel.refine_de_um_label.text()
        panel.refine_de_axis_combo.setCurrentIndex(1)             # XY
        text = panel.refine_de_um_label.text()
        assert "through depth" not in text and "2" in text, text

    def test_a_selection_change_re_runs_the_gate_only_on_refine(
            self, qapp, panel):
        """With Preview greyed for "no object", a selection nothing re-read
        would leave it greyed for good.  Hooked ONCE per layer, deferred,
        and a no-op unless the Refine tab is in front."""
        import time

        from qtpy.QtWidgets import QApplication

        from tests.annotator.test_refine_cross_method_chain import Fake, _K

        class _Sig:
            def __init__(self):
                self.slots = []

            def connect(self, fn):
                self.slots.append(fn)

        class _L:
            def __init__(self):
                self.events = type("E", (), {})()
                self.events.selected_label = _Sig()

        layer = _L()
        f = Fake(np.zeros((6, 6, 6), np.int32))
        f._layer_by_name = lambda layers, name: layer
        f._get_advanced_features_widget = lambda: panel
        ran = []
        f._refresh_refine_preview_gate = lambda: ran.append(1)
        K = _K()
        for name in ("_hook_refine_selection", "_on_refine_selection_changed"):
            setattr(f, name, getattr(K, name).__get__(f, Fake))
        f._hook_refine_selection()
        f._hook_refine_selection()
        assert len(layer.events.selected_label.slots) == 1

        def _fire():
            layer.events.selected_label.slots[0]()
            t0 = time.time()
            while time.time() - t0 < 0.3:
                QApplication.processEvents()

        panel.focus_tab("Reshape")
        _fire()
        assert ran == []
        panel.focus_tab("Refine")
        _fire()
        assert ran == [1]


# --------------------------------------------------------------------------- #
# 8. The walls legend belongs to the STAGED picture, and speaks Refine
# --------------------------------------------------------------------------- #
class TestTheLegendFollowsTheStagedWallKind:
    def test_turning_the_combo_does_not_relabel_the_picture(self, panel):
        from napari_epicycle.annotator.advanced_features_widget import (
            REFINE_WALL_KINDS,
            REFINE_WALLS_LEGEND_PINCH,
        )

        panel.set_refine_walls_kind("distance")
        panel.refine_wall_kind_combo.setCurrentIndex(
            REFINE_WALL_KINDS.index("membrane"))
        panel.set_refine_walls_toggle_enabled(True)
        panel.set_refine_walls_toggle_checked(True)
        assert panel.refine_walls_legend.text() == REFINE_WALLS_LEGEND_PINCH
        assert "cut" not in panel.refine_walls_legend.text()
        assert "cut" not in panel.refine_show_walls_cb.toolTip()

    def test_the_orchestrator_records_the_kind_it_launched_with(self, qapp):
        from tests.annotator.test_refine_lock_lifecycle import Fake

        vol = np.zeros((10, 12, 12), np.int32)
        vol[2:8, 3:9, 3:9] = 7
        f = Fake(vol)
        seen = []

        class _Adv:
            def set_refine_walls_kind(self, kind=None):
                seen.append(kind)

            def __getattr__(self, name):
                raise AttributeError(name)

        f._get_advanced_features_widget = lambda: _Adv()
        f._refine_launch_wall_kind = "skeleton"
        f._refine_pending_elevation = (np.zeros(vol.shape, np.float32),
                                       tuple(f._smooth_roi))
        new = np.where(vol == 7, 7, 0)
        new[2:8, 2, 3:9] = 7
        f._finish_refine_preview("settle", new, 36, "moved.", 7,
                                 f._smooth_roi)
        assert seen and seen[-1] == "skeleton"
