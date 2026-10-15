"""The safety rules that apply to EVERY Refine method.

Three things shipped wrong before Brick B1, and all three were invisible to
a green suite:

* **an empty result was committed.**  The only zero-guard was ``n_mod == 0``,
  and a result that removes every voxel has ``n_mod = 367``, so it was
  written, Accept was enabled, and the status said "refined";
* **the limit was one-sided.**  ``evolved & (edt(~original) <= clamp)`` is
  trivially true everywhere *inside* the original footprint, so it capped
  growth and never restrained collapse -- the one failure mode that actually
  happened;
* **nothing was spacing-aware**, and a micron number computed from an
  unknown voxel size is a lie the page has no business printing.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

pytest.importorskip("qtpy")

from packages.epicycle_annotator.refine import two_sided_limit


@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        try:
            app = QApplication([])
        except Exception as exc:  # pragma: no cover
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


def _ann():
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    return Curation3DAnnotator


ALL_METHODS = ("sdf", "contour", "dilate_erode", "guided_grow",
               "fill_holes", "flood_fill", "settle")


class TestTheEmptyResultIsNamedBeforeItIsAccepted:
    """ONE gate, in the shared tail, for all seven methods --
    not seven copies in seven branches."""

    def test_the_gate_lives_in_the_shared_tail(self):
        src = inspect.getsource(_ann()._finish_refine_preview)
        assert "n_left" in src
        assert "(result_roi == label_id).sum()" in src

    def test_it_says_the_whole_object_would_go(self):
        src = inspect.getsource(_ann()._finish_refine_preview)
        assert "This removes the whole object" in src

    def test_the_primary_button_changes_its_words(self):
        """"Accept" on a result that deletes the object is a lie about
        what the button does."""
        src = inspect.getsource(_ann()._finish_refine_preview)
        assert 'Delete object {}' in src

    def test_accept_stays_enabled_because_deletion_is_sometimes_intended(self):
        """By design.  The result IS written, so the user sees the
        truth on screen, and the amber says what Accept would mean."""
        src = inspect.getsource(_ann()._finish_refine_preview)
        head, _sep, tail = src.partition("if n_left == 0:")
        assert tail, "the empty branch is gone"
        assert "accept_enabled=False" not in tail.split("self._update_smooth_ui")[0]

    def test_the_object_is_held_in_the_list_until_a_decision(self):
        src = inspect.getsource(_ann()._finish_refine_preview)
        assert "is held until you" in src

    def test_settle_REFUSES_instead(self):
        """The one exception: Settle's write-back
        returns -1 and writes nothing."""
        src = inspect.getsource(_ann()._finish_refine_preview)
        assert "apply_settle_to_volume" in src
        assert "written < 0" in src

    def test_the_n_mod_zero_guard_still_means_nothing_happened(self):
        src = inspect.getsource(_ann()._finish_refine_preview)
        assert "if n_mod == 0:" in src
        assert "_REFINE_NOTHING_MOVED" in src

    def test_a_chained_zero_step_decrements_the_chain_and_keeps_the_rest(
            self, qapp, monkeypatch):
        """The behaviour was always there; the new sentences make it
        visible for the first time.  DRIVEN on the bare fake."""
        import napari_epicycle.annotator.curation_3d_widget as mod
        from tests.annotator.test_refine_cross_method_chain import (
            Fake, _Layer, _vol,
        )

        monkeypatch.setattr(mod, "Labels", _Layer)
        f = Fake(_vol())
        f.adv.method = "sdf"
        f._preview_smooth()
        f.adv.method = "fill_holes"          # the object has no cavity
        f._preview_smooth()
        assert f._smooth_chain_count == 1
        assert f._smooth_running is True
        assert f.last("accept_enabled") is True
        assert f.last("reject_enabled") is True
        assert f.last("status_text").startswith(
            _ann()._REFINE_NOTHING_MOVED["fill_holes"])

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_every_method_has_its_own_nothing_moved_sentence(self, method):
        """A "nothing moved" that does not say WHY is the
        failure the audit named."""
        table = _ann()._REFINE_NOTHING_MOVED
        assert method in table
        assert table[method].startswith("Nothing moved")
        assert len(table[method]) > 40, "that is not a reason"

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_every_method_has_a_word_for_the_breadcrumb_and_the_result(
            self, method):
        assert method in _ann()._REFINE_STEP_WORD
        assert method in _ann()._REFINE_METHOD_WORD


class TestTheLimitIsTwoSided:
    """The pure helper ships in B1; B2 wires the knob."""

    def _shell(self):
        original = np.zeros((10, 18, 18), bool)
        original[3:7, 6:12, 6:12] = True
        return original

    def test_a_collapse_is_restrained(self):
        original = self._shell()
        collapsed = np.zeros_like(original)
        collapsed[4:6, 8:10, 8:10] = True
        out, hit, _mean, mx = two_sided_limit(
            original, collapsed, limit_um=1.0)
        assert hit is True and mx <= 1.0 + 1e-6
        assert out.sum() > collapsed.sum()

    def test_the_old_one_sided_guard_could_not_see_it(self):
        from scipy.ndimage import distance_transform_edt

        original = self._shell()
        collapsed = np.zeros_like(original)
        collapsed[4:6, 8:10, 8:10] = True
        # the shipped expression, verbatim
        old = collapsed & (distance_transform_edt(~original) <= 1)
        assert np.array_equal(old, collapsed), (
            "the one-sided clamp changed something -- rewrite the story, "
            "not the test"
        )

    def test_growth_is_still_capped(self):
        original = self._shell()
        grown = np.zeros_like(original)
        grown[1:9, 4:14, 4:14] = True
        out, hit, _mean, mx = two_sided_limit(original, grown, limit_um=1.0)
        assert hit is True and mx <= 1.0 + 1e-6
        assert out.sum() < grown.sum()

    def test_the_two_distances_come_off_the_same_transforms_as_the_clip(self):
        """The panel's read-out and the clip cannot disagree, because they
        are the same two arrays."""
        src = inspect.getsource(two_sided_limit)
        # ONE pair of transforms, computed once and used by both the clip
        # and the two read-outs.
        assert src.count("distance_transform_edt(") == 2
        assert src.count("d_out = distance_transform_edt(") == 1
        assert src.count("d_in = distance_transform_edt(") == 1
        assert "d_out[grew]" in src and "d_in[shrank]" in src


class TestSpacing:
    """Where it comes from, and what happens when it does not."""

    def test_it_comes_from_the_reshape_context(self):
        src = inspect.getsource(_ann()._refine_spacing)
        assert "_reshape_context" in src
        assert '"spacing"' in src

    def test_the_fallback_is_one_one_one(self):
        src = inspect.getsource(_ann()._refine_spacing)
        assert "(1.0, 1.0, 1.0)" in src

    def test_the_fallback_is_recognisable_as_unknown(self):
        src = inspect.getsource(_ann()._refine_spacing_known)
        assert "(1.0, 1.0, 1.0)" in src

    def test_an_unknown_spacing_raises_the_amber_note(self):
        src = inspect.getsource(_ann()._finish_refine_preview)
        assert "_refine_spacing_known" in src
        assert "Voxel size unknown" in src

    def test_the_settle_roi_margin_is_a_SCALAR(self):
        """`_snapshot_label_roi` takes one number for all three axes and
        is shared with Reshape's Start, so it is NOT given a per-axis
        keyword (audit C-H2)."""
        src = inspect.getsource(_ann()._preview_smooth)
        assert "roi_margin_voxels" in src
        assert "margin=margin" in src
        snap = inspect.getsource(_ann()._snapshot_label_roi)
        assert "margin: int = 15" in snap, (
            "`_snapshot_label_roi` changed -- this work must not change it"
        )


class TestTheResultCard:
    def test_the_result_line_names_its_method(self, widget):
        """The line is self-describing even before the user
        looks at the combo."""
        src = inspect.getsource(_ann()._finish_refine_preview)
        assert '"From {}: {}"' in src

    def test_the_breadcrumb_appears_only_from_two_steps_on(self):
        obj = type("O", (), {"_refine_steps": ["Smooth"]})()
        fn = _ann()._refine_breadcrumb_text
        assert fn(obj) == ""
        obj._refine_steps = ["Smooth", "Dilate/Erode", "Fill holes"]
        assert fn(obj) == "Smooth -> Dilate/Erode -> Fill holes"

    def test_the_word_is_steps_everywhere_and_never_ops(self):
        src = inspect.getsource(_ann()._finish_refine_preview)
        assert "steps staged" in src
        assert "ops)" not in src, 'the old "(3 ops)" string survived'

    def test_the_footer_summary_says_what_discard_undoes(self, widget):
        widget.update_smooth_ui_state(reject_enabled=True)
        assert "before the first step" in (
            widget._refine_footer_summary.text().lower()
        )

    def test_the_card_is_hidden_until_something_is_staged(self, widget):
        assert widget._refine_result_container.isHidden() is True

    def test_it_appears_when_a_note_is_pushed(self, widget):
        widget.update_smooth_ui_state(result_note="From Settle: it moved.")
        assert widget._refine_result_container.isHidden() is False
        assert widget.refine_result_note.text().startswith("From Settle")

    def test_and_goes_away_again_when_it_is_cleared(self, widget):
        widget.update_smooth_ui_state(result_note="something")
        widget.update_smooth_ui_state(
            result_note="", result_warning="", breadcrumb="")
        assert widget._refine_result_container.isHidden() is True


class TestTheMethodChangeRule:
    """Decided 2026-09-23: a method change under a staged
    result keeps Preview live (the old state 4m refusal is gone).  DRIVEN on
    the bare fake of `test_refine_cross_method_chain.py` -- the source-
    inspection versions this class held passed whatever the handler did."""

    @pytest.fixture
    def staged(self, qapp, monkeypatch):
        import napari_epicycle.annotator.curation_3d_widget as mod
        from tests.annotator.test_refine_cross_method_chain import (
            Fake, _Layer, _vol,
        )

        monkeypatch.setattr(mod, "Labels", _Layer)
        f = Fake(_vol())
        f.adv.method = "sdf"
        f._preview_smooth()
        assert f._smooth_running and f._smooth_chain_count == 1
        return f

    def test_smooth_then_dilate_runs_and_chains(self, staged):
        f = staged
        f.adv.method = "dilate_erode"
        f._on_refine_method_combo_changed(2)
        assert f.last("preview_enabled") is True
        f._preview_smooth()
        assert f._smooth_chain_count == 2
        assert f._refine_steps == ["Smooth", "Dilate/Erode"]
        assert f._refine_staged_method == "dilate_erode"

    def test_accept_and_S5_are_untouched_by_a_method_change(self, staged):
        f = staged
        n = len(f.ui_calls)
        f.adv.method = "dilate_erode"
        f._on_refine_method_combo_changed(2)
        pushed = f.ui_calls[n:]
        assert pushed, "the handler pushed nothing"
        for kw in pushed:
            assert "accept_enabled" not in kw, (
                "the method-change handler must not touch Accept")
            assert "result_note" not in kw, "S5 must be left alone"
            assert "result_warning" not in kw, "S5 must be left alone"

    def test_settle_chains_onto_another_method(self, staged):
        """B1c (decided 2026-09-23): the Settle crossing is gone --
        Settle moves voxels only between background and the object, and a
        step that needs more room grows the region."""
        from tests.annotator.test_refine_cross_method_chain import (
            _sync_settle,
        )

        f = _sync_settle(staged)
        f.adv.method = "settle"
        f.adv.channel = "membrane"
        f._on_refine_method_combo_changed(6)
        assert f.last("preview_enabled") is True
        f._preview_smooth()
        assert f._smooth_chain_count == 2
        assert f._refine_steps == ["Smooth", "Settle"]
        assert f._refine_staged_method == "settle"


# --------------------------------------------------------------------------- #
# Brick B2 / -- Smooth's honest wording on a thin object
# --------------------------------------------------------------------------- #
class TestSmoothSaysWhatItDidToAThinObject:
    """The measured case: a 3-slice object under the
    default 3.0-voxel blur through depth -- 9.0 um on (3, 1, 1) data -- goes
    to nothing.  The behaviour is kept by design; the page must SAY it:
    the micron read-out beside the voxel knob, and the red block naming the
    knob that did it and how thick the object was."""

    def test_the_readout_prints_9_point_0_um(self, widget):
        widget.set_refine_spacing((3.0, 1.0, 1.0))
        text = widget.refine_sdf_um_label.text()
        assert "counted in voxels, not microns" in text
        assert "one voxel through depth is 3 um" in text
        assert "3.0 blurs over about 9.0 um" in text
        assert widget.refine_sdf_um_label.isHidden() is False

    def test_an_unknown_spacing_prints_no_micron_number(self, widget):
        widget.set_refine_spacing(None)
        assert widget.refine_sdf_um_label.text() == ""

    def _empty_smooth(self):
        from tests.annotator.test_refine_lock_lifecycle import Fake

        vol = np.zeros((8, 12, 12), np.int32)
        vol[3:6, 3:9, 3:9] = 7                       # three slices thick
        f = Fake(vol)
        f._finish_refine_preview("sdf", np.zeros_like(vol), 108, "changed.",
                                 7, f._smooth_roi)
        return f

    def test_the_red_block_names_the_knob_and_the_thickness(self):
        f = self._empty_smooth()
        warn = f.last_ui("result_warning")
        assert warn.startswith("This removes the whole object.")
        assert "The object is 3 slices thick." in warn
        assert "lower 'Blur through depth', or press Discard" in warn
        assert f.last_ui("accept_text") == "Delete object 7"

    def test_another_method_is_not_told_to_lower_a_blur_it_has_not_got(self):
        from tests.annotator.test_refine_lock_lifecycle import Fake

        vol = np.zeros((8, 12, 12), np.int32)
        vol[3:6, 3:9, 3:9] = 7
        f = Fake(vol)
        f._finish_refine_preview("dilate_erode", np.zeros_like(vol), 108,
                                 "changed.", 7, f._smooth_roi)
        warn = f.last_ui("result_warning")
        assert "Blur through depth" not in warn
        assert "press Discard" in warn
