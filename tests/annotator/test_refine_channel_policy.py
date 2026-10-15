"""ONE channel picker for the whole Refine tab, and NO fallback.

What shipped before Brick B1: three per-method image combos, and
``_get_refine_image_roi`` falling through to the hidden ``image_combo`` --
i.e. to the FIRST image layer in the viewer -- whenever its combo was blank.
That defeated the blank-stays-blank policy the Reshape combos ship, made
every "that layer is gone" refusal unreachable, and the guide said the
opposite was true.

So this file asserts three things a green suite could not tell you before:

* the restore MISS survives two refreshes -- the known hazard
  where reading the choice back off ``currentText()`` heals a
  deliberately blank combo into whatever layer is now first;
* a blank combo disables Preview BEFORE the click, with the reason in its
  tooltip, and the refusal on click remains behind it;
* ``Curation3DAnnotator._get_image_roi`` **no longer exists** -- the symbol
  is gone, not merely unused, because a fallback nobody calls is a fallback
  somebody will call.
"""

from __future__ import annotations

import inspect

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


class TestTheFallbackIsGone:
    def test_get_image_roi_no_longer_exists(self):
        """Audit C-M3: its only two callers were the two fallback lines,
        so the METHOD goes with them."""
        assert not hasattr(_ann(), "_get_image_roi"), (
            "the first-image-layer fallback is still reachable"
        )

    def test_get_refine_image_roi_no_longer_exists(self):
        assert not hasattr(_ann(), "_get_refine_image_roi")

    def test_its_replacement_has_no_fallback_branch(self):
        """The docstring names the deleted method, so the assertion is
        made against the BODY -- the source below the docstring."""
        src = inspect.getsource(_ann()._get_refine_channel_roi)
        body = src.split('"""', 2)[-1]
        assert "_get_image_roi" not in body
        assert "image_combo" not in body
        assert "return None" in body

    def test_the_hidden_widget_the_parent_needs_is_KEPT(self):
        """Only the annotator's ROI-cropping method goes: `image_combo`
        itself is still read at two sites and the parent builds its own."""
        src = inspect.getsource(_ann())
        assert "self.image_combo" in src

    def test_the_old_refresher_is_gone_from_both_sides(self, widget):
        assert not hasattr(widget, "refresh_refine_image_combos")
        assert not hasattr(_ann(), "_refresh_refine_image_combos")


class TestARestoreMissStaysBlank:
    """The hazard in full: a FIRST refresh after a rename blanks the combo
    (correctly); reading the choice back off `currentText()` then makes the
    SECOND refresh take "" for "nothing was ever chosen", and Qt's addItems
    auto-selection at index 0 stands.  Two refreshes undo the whole policy.
    """

    def test_a_miss_blanks_it(self, widget):
        widget.set_refine_channels(["membrane", "dapi"])
        widget.refine_channel_combo.setCurrentIndex(1)
        widget.set_refine_channels(["membrane", "brightfield"])
        assert widget.refine_channel_combo.currentIndex() == -1

    def test_and_it_STAYS_blank_across_a_second_refresh(self, widget):
        widget.set_refine_channels(["membrane", "dapi"])
        widget.refine_channel_combo.setCurrentIndex(1)
        widget.set_refine_channels(["membrane", "brightfield"])
        widget.set_refine_channels(["membrane", "brightfield"])
        assert widget.refine_channel_combo.currentIndex() == -1, (
            "the blank state healed into whatever layer is now first"
        )
        assert widget.refine_channel_name() == ""

    def test_renaming_the_layer_BACK_restores_the_selection(self, widget):
        widget.set_refine_channels(["membrane", "dapi"])
        widget.refine_channel_combo.setCurrentIndex(1)
        widget.set_refine_channels(["membrane", "brightfield"])
        widget.set_refine_channels(["membrane", "dapi"])
        assert widget.refine_channel_name() == "dapi"

    def test_the_choice_lives_in_a_qt_property_not_in_currentText(self, widget):
        from napari_epicycle.annotator.advanced_features_widget import (
            AdvancedFeaturesWidget,
        )

        widget.set_refine_channels(["membrane", "dapi"])
        widget.refine_channel_combo.setCurrentIndex(1)
        widget.set_refine_channels(["membrane", "brightfield"])
        assert widget.refine_channel_combo.property(
            AdvancedFeaturesWidget._CHANNEL_CHOICE_PROP) == "dapi"

    def test_a_first_fill_leaves_index_zero_selected(self, widget):
        """Nothing ever chosen, nothing remembered -- there is nothing to
        restore, so Qt's own selection stands."""
        widget.set_refine_channels(["membrane", "dapi"])
        assert widget.refine_channel_combo.currentIndex() == 0


class TestPreviewIsDisabledBeforeTheClick:
    def test_the_gate_names_the_reason(self, monkeypatch):
        """DRIVEN on the bare fake: a channel method with no channel."""
        import napari_epicycle.annotator.curation_3d_widget as mod
        from tests.annotator.test_refine_cross_method_chain import (
            Fake, _Layer, _vol,
        )

        monkeypatch.setattr(mod, "Labels", _Layer)
        f = Fake(_vol())
        f.adv.method = "contour"
        assert f._refresh_refine_preview_gate() is False
        assert f.last("preview_enabled") is False
        assert f.last("preview_reason") == "Pick a channel first."
        assert f.last("status_text") == "Pick a channel first."

    def test_it_abstains_while_a_worker_runs_or_a_result_is_staged(self):
        """Those states own the button; a refresh must not re-enable it."""
        src = inspect.getsource(_ann()._refresh_refine_preview_gate)
        assert "_refine_running" in src
        assert "_refine_staged_method" in src

    def test_the_refusal_on_click_remains_behind_it(self):
        """Belt and braces: the button is already off with the reason in
        its tooltip, and this is what answers a keyboard activation."""
        src = inspect.getsource(_ann()._preview_smooth)
        assert 'params.get("needs_channel")' in src
        assert "Pick a channel first" in src

    def test_the_tooltip_is_where_the_reason_goes(self, widget):
        widget.update_smooth_ui_state(
            preview_enabled=False, preview_reason="Pick a channel first.")
        assert widget.smooth_preview_btn.isEnabled() is False
        assert "Pick a channel first." in widget.smooth_preview_btn.toolTip()

    def test_clearing_the_reason_puts_the_default_tooltip_back(self, widget):
        widget.update_smooth_ui_state(preview_reason="")
        assert "Nothing is final until you Accept" in (
            widget.smooth_preview_btn.toolTip())


class TestWhoRefreshesIt:
    def test_it_is_in_the_shared_refresh_set(self):
        src = inspect.getsource(_ann()._refresh_all_image_combos)
        assert "_refresh_refine_inputs" in src

    def test_the_refresher_skips_temporary_layers(self):
        """`guide_walls` is `_epicycle_temporary`, so it must never be
        offered as a channel."""
        src = inspect.getsource(_ann()._refresh_refine_inputs)
        assert "_epicycle_temporary" in src

    def test_it_also_hands_the_page_its_spacing(self):
        src = inspect.getsource(_ann()._refresh_refine_inputs)
        assert "set_refine_spacing" in src
        assert "_refine_spacing_known" in src


class TestTheMicronReadoutsAreSuppressedWhenSpacingIsUnknown:
    """When the controller could not resolve a voxel size, every
    micron number on the page would be a lie, so they are not printed."""

    def test_unknown_spacing_prints_nothing(self, widget):
        widget.set_refine_spacing(None)
        assert widget.refine_sdf_um_label.text() == ""
        assert widget.refine_de_um_label.text() == ""

    def test_a_known_spacing_prints_the_equivalent(self, widget):
        widget.set_refine_spacing((3.0, 1.0, 1.0))
        text = widget.refine_sdf_um_label.text()
        assert "3" in text and "um" in text
        assert "voxels, not microns" in text

    def test_the_readout_follows_the_knob(self, widget):
        widget.set_refine_spacing((2.0, 0.5, 0.5))
        widget.smooth_standalone_sigma_z_sb.setValue(4.0)
        assert "8" in widget.refine_sdf_um_label.text()

    def test_the_annotator_only_hands_over_a_KNOWN_spacing(self):
        src = inspect.getsource(_ann()._refresh_refine_inputs)
        assert "if self._refine_spacing_known() else None" in src
