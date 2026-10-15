"""The merged Active contour's PAGE, Brick B2.

What the page must do, state by state:

* the MODE decides the rows: *Snap to edges* shows the push (with its LIVE
  line), the wall-type row and the Advanced edge knobs; *Inside vs outside
  brightness* shows the two weights and takes the wall rows away (trim 2);
* the ORIENTATION decides the sampler: a plane shows the slice sampler and
  the rebuild's blur, 3D hides both;
* the mode's grey line describes the CHOSEN mode only (trim 1), Orientation
  explains itself in a tooltip (trim 3), and the wall-type line is not
  repeated on the contour page (trim 4);
* the push line is computed from the push, the gate and the sharpness,
  and says "No effect at this setting" BEFORE any Preview;
* every value reaches the orchestrator through ``get_refine_params()``,
  combos BY INDEX through module constants.

Then the orchestrator side of the page: the determinate "Plane i of n"
stage line, the reason-keyed "nothing moved" sentences, and the walls tick
per mode.
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
        app = QApplication([])
    return app


@pytest.fixture
def widget(qapp):
    from napari_epicycle.annotator.advanced_features_widget import (
        REFINE_METHOD_INDEX,
        AdvancedFeaturesWidget,
    )

    w = AdvancedFeaturesWidget(mode="3d_curation")
    w.refine_method_combo.setCurrentIndex(REFINE_METHOD_INDEX["contour"])
    yield w
    try:
        w.setParent(None)
    except Exception:
        pass


def _M():
    from napari_epicycle.annotator import advanced_features_widget as M

    return M


# --------------------------------------------------------------------------- #
# the wire format
# --------------------------------------------------------------------------- #
class TestTheWireFormat:
    def test_the_three_combos_line_up_with_the_engine(self):
        from packages.epicycle_annotator import refine as R

        M = _M()
        assert len(M.REFINE_CONTOUR_MODE_LABELS) == len(R.CONTOUR_MODES)
        assert M.REFINE_CONTOUR_AXES == R.CONTOUR_ORIENTATIONS
        assert len(M.REFINE_CONTOUR_SLICE_LABELS) == len(R.SLICE_SAMPLINGS)

    def test_the_combos_are_filled_from_the_constants(self, widget):
        M = _M()
        for attr, labels in (
                ("refine_contour_mode_combo", M.REFINE_CONTOUR_MODE_LABELS),
                ("refine_contour_axis_combo", M.REFINE_CONTOUR_AXES),
                ("refine_contour_slices_combo",
                 M.REFINE_CONTOUR_SLICE_LABELS)):
            combo = getattr(widget, attr)
            assert [combo.itemText(i) for i in range(combo.count())] == list(
                labels), attr

    def test_the_defaults_are_the_spec_s(self, widget):
        p = widget.get_refine_params()
        assert p["contour_mode"] == 0
        assert p["contour_axis"] == "3D"
        assert p["contour_sampling"] == 0
        assert p["contour_share_pct"] == 25.0
        assert p["contour_n_slices"] == 12
        assert p["contour_limit_um"] == 3.0
        assert p["contour_iterations"] == 100
        assert p["contour_balloon"] == 0.0
        assert p["contour_gate"] == pytest.approx(0.30)
        assert p["contour_blur_um"] == 2.0
        assert p["contour_sharpness"] == 100.0
        assert p["contour_rebuild_um"] == 1.5

    def test_the_ranges_are_the_spec_s(self, widget):
        w = widget
        assert (w.refine_contour_limit_sb.minimum(),
                w.refine_contour_limit_sb.maximum()) == (0.5, 30.0)
        assert (w.refine_contour_iterations_sb.minimum(),
                w.refine_contour_iterations_sb.maximum()) == (10, 500)
        assert (w.refine_contour_balloon_sb.minimum(),
                w.refine_contour_balloon_sb.maximum()) == (-1.0, 1.0)
        assert (w.refine_contour_gate_sb.minimum(),
                w.refine_contour_gate_sb.maximum()) == (0.0, 1.0)
        assert (w.refine_contour_sharpness_sb.minimum(),
                w.refine_contour_sharpness_sb.maximum()) == (1, 1000)
        assert (w.refine_contour_share_sb.minimum(),
                w.refine_contour_share_sb.maximum()) == (5, 100)
        assert (w.refine_contour_nslices_sb.minimum(),
                w.refine_contour_nslices_sb.maximum()) == (3, 60)

    def test_combos_are_read_by_index(self, widget):
        widget.refine_contour_axis_combo.setCurrentIndex(2)
        widget.refine_contour_slices_combo.setCurrentIndex(1)
        widget.refine_contour_mode_combo.setCurrentIndex(1)
        p = widget.get_refine_params()
        assert p["contour_axis"] == "XZ"
        assert p["contour_sampling"] == 1
        assert p["contour_mode"] == 1

    def test_the_retired_boxes_are_gone(self, widget):
        for attr in ("refine_iterations_sb", "refine_smoothing_sb",
                     "refine_balloon_sb", "refine_acwe_iterations_sb",
                     "refine_acwe_smoothing_sb", "_refine_contour_edges",
                     "_refine_contour_brightness"):
            assert not hasattr(widget, attr), attr


# --------------------------------------------------------------------------- #
# the mode decides the rows
# --------------------------------------------------------------------------- #
class TestTheModeDecidesTheRows:
    def test_snap_to_edges(self, widget):
        widget.refine_contour_mode_combo.setCurrentIndex(0)
        assert widget._refine_contour_push_box.isHidden() is False
        assert widget._refine_contour_weights_box.isHidden() is True
        assert widget._refine_wall_container.isHidden() is False
        assert widget._refine_contour_snap_adv.isHidden() is False

    def test_brightness_takes_the_wall_rows_away(self, widget):
        """Trim 2, and the B2 driver's check 3."""
        widget.refine_contour_mode_combo.setCurrentIndex(1)
        assert widget._refine_wall_container.isHidden() is True
        assert widget._refine_contour_push_box.isHidden() is True
        assert widget._refine_contour_weights_box.isHidden() is False
        assert widget._refine_contour_snap_adv.isHidden() is True
        # ...but the CHANNEL stays -- that mode reads it directly
        assert widget._refine_channel_container.isHidden() is False
        assert widget.refine_prep_clahe_cb.isHidden() is False

    def test_snap_hides_local_contrast(self, widget):
        widget.refine_contour_mode_combo.setCurrentIndex(0)
        assert widget.refine_prep_clahe_cb.isHidden() is True

    def test_the_mode_line_describes_the_chosen_mode_only(self, widget):
        """Trim 1."""
        M = _M()
        label = widget._item_help_labels["refine_contour_mode_combo"]
        for idx in (0, 1, 0):
            widget.refine_contour_mode_combo.setCurrentIndex(idx)
            assert label.text() == M.ITEM_HELP[
                "refine_contour_mode_combo"][idx]
        other = M.ITEM_HELP["refine_contour_mode_combo"][1]
        widget.refine_contour_mode_combo.setCurrentIndex(0)
        assert other not in label.text()

    def test_the_wall_line_is_not_repeated_on_the_contour_page(self, widget):
        """Trim 4: Settle keeps it; the contour page does not."""
        from napari_epicycle.annotator.advanced_features_widget import (
            REFINE_METHOD_INDEX,
        )

        widget.refine_contour_mode_combo.setCurrentIndex(0)
        assert widget._refine_wall_help.isHidden() is True
        widget.refine_method_combo.setCurrentIndex(
            REFINE_METHOD_INDEX["settle"])
        assert widget._refine_wall_help.isHidden() is False
        assert widget._refine_wall_container.isHidden() is False

    def test_orientation_explains_itself_in_a_tooltip(self, widget):
        """Trim 3: no grey line, a tooltip."""
        assert widget.refine_contour_axis_combo.toolTip().strip()
        assert "refine_contour_axis_combo" not in _M().HELP_LINES

    def test_the_description_does_not_describe_both_modes(self, widget):
        from napari_epicycle.annotator.advanced_features_widget import (
            REFINE_METHOD_INDEX,
        )

        text = widget._REFINE_METHOD_DESCRIPTIONS[
            REFINE_METHOD_INDEX["contour"]]
        assert "Snap to edges" not in text
        assert "brightness" not in text


# --------------------------------------------------------------------------- #
# the orientation decides the sampler
# --------------------------------------------------------------------------- #
class TestTheOrientationDecidesTheSampler:
    def test_3D_hides_it(self, widget):
        widget.refine_contour_axis_combo.setCurrentIndex(0)
        assert widget._refine_contour_sampling.isHidden() is True
        assert widget._refine_contour_rebuild_adv.isHidden() is True

    @pytest.mark.parametrize("idx", [1, 2, 3])
    def test_a_plane_shows_it(self, widget, idx):
        widget.refine_contour_axis_combo.setCurrentIndex(idx)
        assert widget._refine_contour_sampling.isHidden() is False
        assert widget._refine_contour_rebuild_adv.isHidden() is False

    def test_the_sub_knob_follows_the_sampling(self, widget):
        widget.refine_contour_axis_combo.setCurrentIndex(1)
        sc = widget.refine_contour_slices_combo
        sc.setCurrentIndex(0)
        assert widget._refine_contour_share_row.isHidden() is False
        assert widget._refine_contour_nslices_row.isHidden() is True
        sc.setCurrentIndex(1)
        assert widget._refine_contour_share_row.isHidden() is True
        assert widget._refine_contour_nslices_row.isHidden() is False
        sc.setCurrentIndex(2)
        assert widget._refine_contour_share_row.isHidden() is True
        assert widget._refine_contour_nslices_row.isHidden() is True

    def test_the_contour_advanced_card_is_the_contour_s_only(self, widget):
        from napari_epicycle.annotator.advanced_features_widget import (
            REFINE_METHOD_INDEX,
        )

        assert widget._refine_contour_advanced.isHidden() is False
        widget.refine_method_combo.setCurrentIndex(REFINE_METHOD_INDEX["sdf"])
        assert widget._refine_contour_advanced.isHidden() is True


# --------------------------------------------------------------------------- #
# the live push line
# --------------------------------------------------------------------------- #
class TestTheLivePushLine:
    def test_below_the_gate_it_says_no_effect_BEFORE_any_preview(self, widget):
        from napari_epicycle.annotator.advanced_features_widget import (
            PANEL_WARN_STYLE,
        )

        widget.refine_contour_balloon_sb.setValue(0.2)
        text = widget.refine_contour_push_line.text()
        assert text.startswith("No effect at this setting")
        assert "(0.30)" in text
        assert widget.refine_contour_push_line.styleSheet() == PANEL_WARN_STYLE

    def test_above_it_it_says_where_it_acts(self, widget):
        widget.refine_contour_balloon_sb.setValue(1.0)
        assert widget.refine_contour_push_line.text() == (
            "Acts where the wall picture is fainter than 0.15.")

    def test_lowering_the_gate_turns_the_same_push_on(self, widget):
        widget.refine_contour_balloon_sb.setValue(0.2)
        widget.refine_contour_gate_sb.setValue(0.10)
        assert widget.refine_contour_push_line.text().startswith("Acts where")

    def test_the_sharpness_moves_the_number(self, widget):
        widget.refine_contour_balloon_sb.setValue(1.0)
        a = widget.refine_contour_push_line.text()
        widget.refine_contour_sharpness_sb.setValue(400)
        assert widget.refine_contour_push_line.text() != a

    def test_a_negative_push_is_judged_by_its_size(self, widget):
        widget.refine_contour_balloon_sb.setValue(-0.2)
        assert widget.refine_contour_push_line.text().startswith("No effect")
        widget.refine_contour_balloon_sb.setValue(-1.0)
        assert widget.refine_contour_push_line.text().startswith("Acts where")

    def test_zero_is_neutral_not_a_warning(self, widget):
        widget.refine_contour_balloon_sb.setValue(0.0)
        assert not widget.refine_contour_push_line.text().startswith(
            "No effect")

    def test_a_gate_of_zero(self, widget):
        widget.refine_contour_balloon_sb.setValue(0.2)
        widget.refine_contour_gate_sb.setValue(0.0)
        assert "everywhere" in widget.refine_contour_push_line.text()


# --------------------------------------------------------------------------- #
# the orchestrator side
# --------------------------------------------------------------------------- #
def _fake():
    from tests.annotator.test_refine_lock_lifecycle import Fake, _K

    vol = np.zeros((10, 30, 30), np.int32)
    vol[2:8, 8:22, 8:22] = 7
    f = Fake(vol)
    K = _K()
    for name in ("_on_refine_progress", "_on_refine_finished",
                 "_launch_refine_method", "_refine_contour_call",
                 "_refuse_refine_step"):
        setattr(f, name, getattr(K, name).__get__(f, type(f)))
    f._refine_running = True
    f._result_token = lambda: 1
    return f, vol


class TestThePlaneCounter:
    def test_plane_one_has_no_estimate_yet(self):
        f, _v = _fake()
        f._on_refine_progress("contour", 1, 11)
        text = f.last_ui("stage_text")
        assert text.startswith("Plane 1 of 11")
        assert "left" not in text.split("  ")[0]
        assert f.last_ui("busy_progress") == (1, 11)

    def test_later_planes_carry_a_time_estimate(self):
        import time

        f, _v = _fake()
        f._on_refine_progress("contour", 1, 11)
        time.sleep(0.02)
        f._on_refine_progress("contour", 4, 11)
        text = f.last_ui("stage_text")
        assert text.startswith("Plane 4 of 11")
        assert "s left" in text

    def test_3D_is_a_marquee_with_a_sentence(self):
        f, _v = _fake()
        f._on_refine_progress("contour", 0, 0)
        assert f.last_ui("busy_progress") == (0, 0)
        assert f.last_ui("stage_text").startswith("Sliding the outline")

    def test_the_rebuild_goes_back_to_a_marquee(self):
        f, _v = _fake()
        f._on_refine_progress("rebuild", 0, 0)
        assert f.last_ui("busy_progress") == (0, 0)
        assert f.last_ui("stage_text").startswith("Filling in")


class _Outcome:
    def __init__(self, result, kind="contour"):
        self.result = result
        self.kind = kind


class TestTheReasonKeyedSentences:
    def _finish(self, details, n_mod=0):
        from packages.epicycle_annotator.refine import ContourResult

        f, vol = _fake()
        f._refine_launch_roi = f._smooth_roi
        f._refine_launch_label = 7
        res = ContourResult(np.where(vol == 7, 7, 0), n_mod, "Nothing moved.",
                            **details)
        f._on_refine_finished(_Outcome(res))
        return f

    def test_push_below_the_gate_names_both_numbers(self):
        f = self._finish(dict(reason="push_below_gate", balloon=0.2,
                              gate=0.30))
        s1 = f.last_ui("status_text")
        assert s1.startswith("Nothing moved. 'Push outward' is 0.2 and the "
                             "Edge gate is 0.30")
        assert "Lower the gate on the Advanced card" in s1

    def test_at_the_limit_names_the_knob(self):
        f = self._finish(dict(reason="limit", limit_um=3.0))
        s1 = f.last_ui("status_text")
        assert "further than 3.0 um" in s1
        assert "Edge may move" in s1

    def test_otherwise_the_method_s_own_sentence(self):
        f = self._finish(dict(reason=None))
        assert f.last_ui("status_text") == f._REFINE_NOTHING_MOVED["contour"]

    def test_a_moved_result_says_what_was_traced(self):
        from packages.epicycle_annotator.refine import ContourResult

        f, vol = _fake()
        f._refine_launch_roi = f._smooth_roi
        f._refine_launch_label = 7
        new = np.where(vol == 7, 7, 0)
        new[2:8, 7, 8:22] = 7
        res = ContourResult(new, int((new != np.where(vol == 7, 7, 0)).sum()),
                            "the outline moved on 84 voxels.",
                            reason=None, traced=11, total=44,
                            traced_text="Traced 11 of 44 slices and filled in "
                                        "the other 33.",
                            hit_limit=True, limit_um=3.0)
        f._on_refine_finished(_Outcome(res))
        note = f.last_ui("result_note")
        assert "Traced 11 of 44 slices and filled in the other 33." in note
        assert "From Active contour" in note
        assert "3.0 um limit" in f.last_ui("result_warning")


class TestTheWallPicturePerMode:
    """Audit C-M3: the contour hands the worker CROPS and
    decisions; the wall picture is built THERE and comes back with the
    answer, where it becomes the pending picture the walls tick reads."""

    def _launch(self, mode, kind="membrane"):
        f, vol = _fake()
        launched = []

        f._launch_refine_worker = lambda *a, **k: launched.append((a, k))
        f._get_refine_channel_roi = lambda roi: np.ones(vol.shape)
        f._get_prob_roi = lambda roi: np.ones(vol.shape)
        f._launch_refine_contour = _K_bind(f, "_launch_refine_contour")
        params = {"method": "contour", "contour_mode": mode,
                  "prep_chain": "none", "wall_kind": kind}
        f._launch_refine_method("contour", params, f._smooth_roi, vol, 7)
        return f, launched

    def test_snap_hands_over_the_channel_crop_not_a_picture(self):
        from packages.epicycle_annotator.refine import contour_with_walls

        f, launched = self._launch(0)
        args, opts = launched[0]
        fn, kw = args[0], args[1]
        assert fn is contour_with_walls
        assert kw["channel_roi"] is not None and kw["wall_kind"] == "membrane"
        assert f._refine_pending_elevation is None
        assert opts["wants_progress"] is True
        assert opts["cancel_event"] is kw["cancel_event"]

    def test_brightness_hands_over_the_channel_too(self):
        f, launched = self._launch(1)
        kw = launched[0][0][1]
        assert kw["channel_roi"] is not None and kw["mode"] == 1

    def test_a_shape_wall_needs_no_channel(self):
        f, launched = self._launch(0, kind="distance")
        assert launched[0][0][1]["channel_roi"] is None

    def test_the_probability_wall_hands_over_the_probability_crop(self):
        f, launched = self._launch(0, kind="probability")
        kw = launched[0][0][1]
        assert kw["prob_roi"] is not None and kw["channel_roi"] is None

    def test_the_picture_arrives_with_the_answer(self):
        from packages.epicycle_annotator.refine import ContourResult

        f, vol = _fake()
        f._refine_launch_roi = f._smooth_roi
        f._refine_launch_label = 7
        seen = {}
        real = f._finish_refine_preview

        def _spy(*a, **k):
            seen["pending"] = f._refine_pending_elevation
            return real(*a, **k)

        f._finish_refine_preview = _spy
        new = np.where(vol == 7, 7, 0)
        new[2:8, 7, 8:22] = 7
        elev = np.zeros(vol.shape, np.float32)
        res = ContourResult(new, int((new != np.where(vol == 7, 7, 0)).sum()),
                            "moved.", reason=None, elevation=elev)
        f._on_refine_finished(_Outcome(res))
        assert seen["pending"][0] is elev
        assert seen["pending"][1] == tuple(f._smooth_roi)


def _K_bind(f, name):
    from tests.annotator.test_refine_lock_lifecycle import _K

    return getattr(_K(), name).__get__(f, type(f))


class TestTheAuditOrchestratorFixes:
    def test_C_M1_the_contour_region_clears_its_own_limit(self):
        f, _v = _fake()
        for name in ("_refine_step_margin", "_settle_roi_margin",
                     "_contour_roi_margin"):
            setattr(f, name, _K_bind(f, name))
        f._reshape_context = lambda: {"patch_name": "p", "run_epoch": 0,
                                      "spacing": (1.0, 1.0, 1.0)}
        assert f._refine_step_margin(
            "contour", {"contour_limit_um": 25.0}) >= 27
        assert f._refine_step_margin(
            "contour", {"contour_limit_um": 3.0}) == 15

    def test_C_L1_a_queued_progress_does_not_overwrite_stopping(self):
        f, _v = _fake()
        f._refine_stopping = True
        n = len(f.ui_calls)
        f._on_refine_progress("contour", 3, 11)
        assert len(f.ui_calls) == n

    def test_C_L4_the_pending_picture_goes_with_the_session(self):
        f, _v = _fake()
        f._refine_pending_elevation = (np.zeros(1), (slice(0, 1),))
        f._reset_smooth_state("x")
        assert f._refine_pending_elevation is None
        f._refine_pending_elevation = (np.zeros(1), (slice(0, 1),))
        f._on_refine_stopped()
        assert f._refine_pending_elevation is None

    def test_the_breadcrumb_word_is_the_result_word(self):
        from tests.annotator.test_refine_lock_lifecycle import _K

        K = _K()
        assert K._REFINE_STEP_WORD["contour"] == "Active contour"
        assert K._REFINE_METHOD_WORD["contour"] == "Active contour"


class TestTheAuditSentences:
    def _finish(self, details, n_mod=0, msg="Nothing moved."):
        from packages.epicycle_annotator.refine import ContourResult

        f, vol = _fake()
        f._refine_launch_roi = f._smooth_roi
        f._refine_launch_label = 7
        res = ContourResult(np.where(vol == 7, 7, 0), n_mod, msg, **details)
        f._on_refine_finished(_Outcome(res))
        return f

    def test_the_limit_sentence_says_what_happened(self):
        s1 = self._finish(dict(reason="limit", limit_um=3.0)).last_ui(
            "status_text")
        assert s1 == ("Nothing moved: every move the outline tried went "
                      "further than 3.0 um, so all were undone. Raise 'Edge "
                      "may move'.")

    def test_the_fence_has_its_own_sentence(self):
        s1 = self._finish(dict(reason="fence")).last_ui("status_text")
        assert "into a neighbouring object" in s1

    def test_the_generic_sentence_no_longer_claims_the_limit(self):
        s1 = self._finish(dict(reason=None)).last_ui("status_text")
        assert "limit" not in s1

    def test_a_negative_push_is_named_with_its_sign(self):
        s1 = self._finish(dict(reason="push_below_gate", balloon=-0.2,
                               gate=0.30)).last_ui("status_text")
        assert "'Push outward' is -0.2 and" in s1

    def test_too_large_is_state_8_amber_not_state_9(self):
        f = self._finish(dict(reason="too_large"),
                         msg="This object's region is too large to trace.")
        assert f.last_ui("status_text") == (
            "Nothing to show - the object is unchanged.")
        assert f.last_ui("status_kind") == "warn"
        assert "too large" in f.last_ui("result_warning")


class _Lazy:
    """Array-like that refuses to be converted WHOLE."""

    ndim = 3
    shape = (50, 50, 50)

    def __init__(self):
        self.asked = []

    def __array__(self, *a, **k):
        raise AssertionError("the WHOLE layer was converted")

    def __getitem__(self, key):
        self.asked.append(key)
        return np.ones((2, 3, 4))


class TestTheChannelIsCroppedBeforeItIsCopied:
    """C-M3: `np.asarray(layer.data)` on the WHOLE layer materialised the
    entire volume to use one object's box."""

    def test_only_the_box_is_ever_read(self):
        f, _v = _fake()
        lazy = _Lazy()

        class _Layer:
            data = lazy
            multiscale = False

        class _Adv:
            def refine_channel_name(self):
                return "membrane"

        f._get_advanced_features_widget = lambda: _Adv()
        f._layer_by_name = lambda layers, name: _Layer()
        f.viewer = type("V", (), {"layers": []})()
        f._get_refine_channel_roi = _K_bind(f, "_get_refine_channel_roi")
        roi = (slice(1, 3), slice(2, 5), slice(0, 4))
        out = f._get_refine_channel_roi(roi)
        assert lazy.asked == [roi]
        assert out.dtype == np.float64 and out.shape == (2, 3, 4)


class TestSmoothMeasuresThisStepsInput:
    def test_C_L5_the_thickness_is_the_live_input_not_the_snapshot(self):
        from tests.annotator.test_refine_lock_lifecycle import Fake

        vol = np.zeros((10, 12, 12), np.int32)
        vol[3:6, 3:9, 3:9] = 7                       # 3 slices NOW
        f = Fake(vol)
        snap = np.zeros_like(vol)
        snap[2:8, 3:9, 3:9] = 7                      # 6 slices before
        f._smooth_snapshot = snap
        f._finish_refine_preview("sdf", np.zeros_like(vol), 108, "changed.",
                                 7, f._smooth_roi)
        assert "The object is 3 slices thick." in f.last_ui("result_warning")


class TestOnARealThread:
    """The contour through a REAL QThread, driven through the annotator's
    own launch / finish / stop paths: the picture arrives in the answer,
    and a real ``worker.cancel()`` mid-plane stops it between planes."""

    def _annotator(self, qapp, vol):
        from tests.annotator.test_refine_worker import (
            TestOnARealThread as _W,
        )

        a = _W()._annotator(qapp, vol)
        for name in ("_launch_refine_method", "_launch_refine_contour",
                     "_refine_contour_call", "_refuse_refine_step",
                     "_push_still_staged", "_refine_still_staged_text"):
            setattr(a, name, _K_bind(a, name))
        a._get_refine_channel_roi = lambda roi: np.asarray(
            vol[roi] > 0, np.float64)
        a._get_prob_roi = lambda roi: None
        a._refine_pending_elevation = None
        a._refine_staged_s5 = None
        return a

    def _pump(self, qapp, pred, seconds=30.0):
        import time

        t0 = time.time()
        while time.time() - t0 < seconds:
            qapp.processEvents()
            if pred():
                return True
            time.sleep(0.005)
        return False

    def _cyl(self):
        zz, yy, xx = np.mgrid[:40, :30, :30]
        r = np.sqrt((yy - 15) ** 2 + (xx - 15) ** 2)
        vol = np.zeros((40, 30, 30), np.int32)
        vol[(r <= 6) & (zz >= 2) & (zz < 38)] = 7
        return vol

    def test_the_picture_arrives_and_the_result_stages(self, qapp):
        vol = self._cyl()
        a = self._annotator(qapp, vol)
        params = {"method": "contour", "contour_mode": 0,
                  "wall_kind": "membrane", "prep_chain": "none",
                  "contour_axis": "XY", "contour_balloon": 1.0,
                  "contour_iterations": 5}
        a._launch_refine_method("contour", params, a._smooth_roi,
                                vol.copy(), 7)
        assert self._pump(qapp, lambda: not a._refine_running)
        assert a._refine_thread is None
        assert a._refine_elevation is not None, "the walls tick has nothing"
        assert a._refine_elevation.shape == vol.shape
        assert a._refine_staged_method == "contour"

    def test_a_real_cancel_mid_plane_stops_between_planes(self, qapp):
        import threading

        vol = self._cyl()
        before = vol.copy()
        a = self._annotator(qapp, vol)
        planes = []
        gate = threading.Event()
        real_call = a._refine_contour_call

        def _slow_call(*args, **kw):
            fn, kwargs = real_call(*args, **kw)

            def _wrapped(**k):
                user_progress = k.pop("progress")

                def _progress(stage, i, n):
                    user_progress(stage, i, n)
                    if stage == "contour" and i >= 2:
                        planes.append(i)
                        gate.wait(5.0)      # hold plane 2 until Stop lands
                return fn(progress=_progress, **k)

            return _wrapped, kwargs

        a._refine_contour_call = _slow_call
        params = {"method": "contour", "contour_mode": 0,
                  "wall_kind": "membrane", "prep_chain": "none",
                  "contour_axis": "XY", "contour_sampling": 2,
                  "contour_balloon": 1.0, "contour_iterations": 5}
        a._launch_refine_method("contour", params, a._smooth_roi,
                                vol.copy(), 7)
        assert self._pump(qapp, lambda: bool(planes))
        worker = a._refine_worker
        a._cancel_refine_preview()          # the real Stop: worker.cancel()
        assert worker.cancel_event.is_set()
        gate.set()
        assert self._pump(qapp, lambda: not a._refine_running)
        # it stopped BETWEEN planes: plane 3 never started
        assert max(planes) == 2, planes
        assert np.array_equal(vol, before)
        assert a._smooth_running is False
        texts = [kw.get("status_text") for kw in a.ui if "status_text" in kw]
        assert texts[-1] == "Stopped. The object is unchanged."


# --------------------------------------------------------------------------- #
# Guided Grow: greyed BEFORE any click when there is no probability map
# --------------------------------------------------------------------------- #
class TestGuidedGrowIsGreyedWithItsReason:
    """And the B2 driver's check 4: no Source picker, no image
    combo, and on a patch with no probability map the ITEM is greyed with
    its reason ON SCREEN before any click (Qt shows no tooltip on a
    disabled item)."""

    def _gg(self):
        from napari_epicycle.annotator.advanced_features_widget import (
            REFINE_METHOD_INDEX,
        )

        return REFINE_METHOD_INDEX["guided_grow"]

    def test_no_source_picker_and_no_image_combo(self, widget):
        for attr in ("refine_gg_source_combo", "refine_gg_image_combo"):
            assert not hasattr(widget, attr), attr

    def test_max_radius_is_capped_at_the_roi_pad(self, widget):
        sb = widget.refine_gg_max_radius_sb
        assert (sb.minimum(), sb.maximum()) == (1, 15)

    def test_without_a_map_the_item_is_greyed_and_the_reason_is_on_screen(
            self, widget):
        M = _M()
        widget.set_refine_guided_grow_available(False)
        item = widget.refine_method_combo.model().item(self._gg())
        assert item.isEnabled() is False
        assert item.text() == M.REFINE_GUIDED_GROW_ITEM_NO_PROB
        # the item's OWN text is the on-screen reason (the line under the
        # combo was dropped, B2 audit UX-M4); the full sentence is its tip
        assert item.toolTip() == M.REFINE_GUIDED_GROW_NO_PROB
        assert widget.refine_guided_grow_available() is False

    def test_it_comes_back_when_a_map_does(self, widget):
        widget.set_refine_guided_grow_available(False)
        widget.set_refine_guided_grow_available(True)
        item = widget.refine_method_combo.model().item(self._gg())
        assert item.isEnabled() is True
        assert item.text() == "Guided Grow"
        # B3: the item's own help comes back, not an empty tooltip
        assert item.toolTip() == _M().ITEM_HELP["refine_method_combo"][
            self._gg()]

    def test_a_greyed_method_that_is_showing_steps_off_it(self, widget):
        widget.refine_method_combo.setCurrentIndex(self._gg())
        widget.set_refine_guided_grow_available(False)
        assert widget.refine_method_combo.currentIndex() == 0

    def test_the_greying_changes_no_index(self, widget):
        """The index is the wire format; only the item's text moves."""
        from napari_epicycle.annotator.advanced_features_widget import (
            REFINE_METHOD_KEYS,
        )

        widget.set_refine_guided_grow_available(False)
        assert widget.refine_method_combo.count() == len(REFINE_METHOD_KEYS)
        widget.refine_method_combo.setCurrentIndex(6)
        assert widget.refine_method_key() == "settle"


class TestTheAnnotatorFollowsTheMap:
    def _bare(self, tmp_path=None):
        from tests.annotator.test_refine_lock_lifecycle import Fake, _K

        f = Fake(np.zeros((4, 4, 4), np.int32))
        K = _K()
        for name in ("set_prob_volume_path", "_has_prob_volume",
                     "_refresh_refine_guided_grow"):
            setattr(f, name, getattr(K, name).__get__(f, type(f)))
        calls = []

        class _Adv:
            def set_refine_guided_grow_available(self, avail, reason=""):
                calls.append((avail, reason))

        f._get_advanced_features_widget = lambda: _Adv()
        f._prob_volume = None
        f._prob_volume_path = None
        return f, calls

    def test_no_path_greys_it(self):
        f, calls = self._bare()
        f.set_prob_volume_path("")
        assert calls[-1][0] is False
        assert "probability map" in calls[-1][1]

    def test_a_missing_file_greys_it(self, tmp_path):
        f, calls = self._bare()
        f.set_prob_volume_path(str(tmp_path / "nope.npy"))
        assert calls[-1][0] is False

    def test_a_real_file_offers_it_without_loading_it(self, tmp_path):
        p = tmp_path / "prob.npy"
        np.save(p, np.zeros((2, 2, 2), np.float32))
        f, calls = self._bare()
        f.set_prob_volume_path(str(p))
        assert calls[-1][0] is True
        assert f._prob_volume is None


# --------------------------------------------------------------------------- #
# B2 audit -- the panel fixes
# --------------------------------------------------------------------------- #
class TestTheMockupOrder:
    """UX-M5: mockup frame e puts Channel -> Prepare -> Wall type ABOVE
    Orientation; the contour's rows below Mode now sit under the shared
    block, in their own container."""

    def test_channel_then_wall_then_the_contour_rows(self, widget):
        lay = widget._refine_method_card.layout()
        i_ch = lay.indexOf(widget._refine_channel_container)
        i_wall = lay.indexOf(widget._refine_wall_container)
        i_low = lay.indexOf(widget._refine_contour_lower)
        assert 0 <= i_ch < i_wall < i_low

    def test_the_orientation_and_the_push_are_in_the_lower_block(self, widget):
        low = widget._refine_contour_lower
        for w in (widget.refine_contour_axis_combo,
                  widget.refine_contour_limit_sb,
                  widget.refine_contour_balloon_sb):
            assert low.isAncestorOf(w)
        assert not low.isAncestorOf(widget.refine_contour_mode_combo)

    def test_the_lower_block_is_the_contour_s_only(self, widget):
        from napari_epicycle.annotator.advanced_features_widget import (
            REFINE_METHOD_INDEX,
        )

        assert widget._refine_contour_lower.isHidden() is False
        widget.refine_method_combo.setCurrentIndex(
            REFINE_METHOD_INDEX["settle"])
        assert widget._refine_contour_lower.isHidden() is True

    def test_the_card_says_why_local_contrast_is_missing(self, widget):
        widget.refine_contour_mode_combo.setCurrentIndex(0)
        line = widget._refine_prep_no_clahe_line
        assert line.text() == ("Local contrast is not used when the tool is "
                               "looking for walls.")
        assert line.isHidden() is False
        widget.refine_contour_mode_combo.setCurrentIndex(1)
        assert line.isHidden() is True


class TestTheAuditPanelWords:
    def test_the_guided_grow_note_is_dropped_the_item_carries_it(self, widget):
        """UX-M4 (orchestrator decision, pending the owner): the note under
        the combo cost every page 26 px on a patch with no map."""
        M = _M()
        widget.set_refine_guided_grow_available(False)
        assert not hasattr(widget, "_refine_method_note")
        item = widget.refine_method_combo.model().item(
            M.REFINE_METHOD_INDEX["guided_grow"])
        assert item.text() == M.REFINE_GUIDED_GROW_ITEM_NO_PROB
        assert item.toolTip() == M.REFINE_GUIDED_GROW_NO_PROB

    def test_a_push_threshold_at_or_above_1_acts_everywhere(self, widget):
        widget.refine_contour_balloon_sb.setValue(1.0)
        widget.refine_contour_gate_sb.setValue(0.05)
        widget.refine_contour_sharpness_sb.setValue(1)
        assert widget.refine_contour_push_line.text().startswith(
            "Acts everywhere")

    def test_the_slices_line_follows_the_sampling(self, widget):
        M = _M()
        widget.refine_contour_axis_combo.setCurrentIndex(1)
        label = widget._item_help_labels["refine_contour_slices_combo"]
        for idx in range(3):
            widget.refine_contour_slices_combo.setCurrentIndex(idx)
            assert label.text() == M.ITEM_HELP[
                "refine_contour_slices_combo"][idx]
        assert "60" in M.ITEM_HELP["refine_contour_slices_combo"][2]

    def test_the_orientation_tooltip(self, widget):
        assert widget.refine_contour_axis_combo.toolTip() == (
            "XY/XZ/YZ trace some slices in that plane and fill in the rest.")

    def test_the_join_is_smoothed_in_microns_with_a_voxel_readout(
            self, widget):
        """C-L3: every contour number is physical; the voxel equivalent is
        a read-out beside it, per axis of the chosen plane."""
        sb = widget.refine_contour_rebuild_um_sb
        assert sb.suffix() == " um" and sb.value() == 1.5
        assert not hasattr(widget, "refine_contour_rebuild_sigma_sb")
        widget.set_refine_spacing((3.0, 1.0, 1.0))
        widget.refine_contour_axis_combo.setCurrentIndex(1)      # XY -> z
        ro = widget.refine_contour_rebuild_readout.text()
        assert "0.5 voxels deep" in ro, ro
        widget.refine_contour_axis_combo.setCurrentIndex(2)      # XZ -> y
        assert "1.5 voxels deep" in widget.refine_contour_rebuild_readout.text()
        assert widget.get_refine_params()["contour_rebuild_um"] == 1.5
        assert "contour_rebuild_sigma" not in widget.get_refine_params()

    def test_its_label_is_smooth_the_join(self, widget):
        from qtpy.QtWidgets import QLabel

        texts = [lb.text() for lb in
                 widget._refine_contour_rebuild_adv.findChildren(QLabel)]
        assert any(t.startswith("Smooth the join") for t in texts), texts
