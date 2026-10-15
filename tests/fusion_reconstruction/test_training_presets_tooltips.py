"""Tests for the 2026-05-22 training-presets + tooltip refinement work.

Covers:
* Each preset in `_apply_training_preset` matches a description in
  `_PRESET_DESCRIPTIONS` (so the dropdown shows the right text).
* All 6 preset names from the UI dropdown have wiring + descriptions.
* Spinbox / checkbox tooltips on training params exist and are
  non-empty (regression guard against future refactors that drop
  them silently).
* `_on_refine_preset_changed` updates the description widget when
  called (pure-Python, no Qt event loop).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


# The presets the UI dropdown exposes — keep in sync with
# `refine_preset_combo.addItems` in ui.py.
_EXPECTED_PRESETS = (
    "default",
    "quick",
    "thorough",
    "finetune",
    "dense_tissue",
    "stratified_epithelia",
)


class TestPresetDescriptionCoverage:
    """Every preset name in the UI dropdown must have a matching
    entry in `_PRESET_DESCRIPTIONS` so the description label isn't
    empty when the user selects it."""

    def test_all_preset_names_have_descriptions(self):
        from napari_epicycle.fusion_reconstruction.ui import (
            PRESET_DESCRIPTIONS,
        )
        for name in _EXPECTED_PRESETS:
            assert name in PRESET_DESCRIPTIONS, (
                f"Preset {name!r} appears in UI dropdown but has no "
                f"entry in PRESET_DESCRIPTIONS"
            )
            desc = PRESET_DESCRIPTIONS[name]
            assert isinstance(desc, str) and desc.strip(), (
                f"Description for {name!r} is empty or non-string"
            )

    def test_backcompat_underscore_alias_exists(self):
        """Older imports may still use `_PRESET_DESCRIPTIONS` —
        keep the alias working until all callers are updated."""
        from napari_epicycle.fusion_reconstruction.ui import (
            _PRESET_DESCRIPTIONS, PRESET_DESCRIPTIONS,
        )
        assert _PRESET_DESCRIPTIONS is PRESET_DESCRIPTIONS

    def test_all_preset_names_have_controller_wiring(self):
        """The controller's `_apply_training_preset` must define each
        of the UI-exposed preset names.  Without this, clicking Apply
        produces 'Unknown preset' warning + no UI updates."""
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        src = inspect.getsource(
            FusionReconstructionController._apply_training_preset
        )
        for name in _EXPECTED_PRESETS:
            assert f'"{name}":' in src or f"'{name}':" in src, (
                f"Preset {name!r} in UI dropdown but no entry in "
                f"_apply_training_preset"
            )

    def test_default_preset_matches_ui_initial_values(self):
        """2026-05-22 audit fix: the `default` preset's n_epochs must
        match the UI spinbox's initial value (100).  Pre-fix the
        preset said 500 — clicking Apply on the default-selected
        combo jumped epochs 100 → 500, which was surprising.
        """
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        src = inspect.getsource(
            FusionReconstructionController._apply_training_preset
        )
        # Find the default preset dict and confirm n_epochs=100.
        # Look for the specific line.
        assert '"default": dict(\n' in src or "'default': dict(\n" in src
        # Find the n_epochs value within the default preset.
        # The first occurrence of n_epochs in the default block:
        default_idx = src.find('"default"')
        assert default_idx >= 0
        # Within the next 200 chars after "default", n_epochs must be 100.
        nearby = src[default_idx : default_idx + 300]
        assert "n_epochs=100" in nearby, (
            f"default preset's n_epochs is no longer 100; this should "
            f"match the UI spinbox's initial value to avoid surprising "
            f"the user with a 5× jump when they click Apply on the "
            f"default-selected dropdown.  Source around default:\n"
            f"{nearby}"
        )


class TestPresetDescriptionUpdater:
    """The signal handler that updates the description label when the
    user changes the combo selection."""

    def _make_controller(self):
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        class _FakeLabel:
            def __init__(self):
                self._text = ""
            def setText(self, t):
                self._text = t
            def text(self):
                return self._text

        ui = SimpleNamespace()
        ui.refine_preset_description = _FakeLabel()

        ctrl = SimpleNamespace(ui=ui)
        method = FusionReconstructionController._on_refine_preset_changed
        ctrl._on_refine_preset_changed = method.__get__(ctrl)
        return ctrl

    def test_handler_sets_description_for_known_preset(self):
        ctrl = self._make_controller()
        ctrl._on_refine_preset_changed("quick")
        text = ctrl.ui.refine_preset_description.text()
        # Should be the "quick" preset description (mentions 100 epochs).
        assert "100 epochs" in text or "quick" in text.lower()

    def test_handler_sets_each_preset_description(self):
        ctrl = self._make_controller()
        for name in _EXPECTED_PRESETS:
            ctrl._on_refine_preset_changed(name)
            text = ctrl.ui.refine_preset_description.text()
            assert text.strip(), f"Empty description for {name}"

    def test_handler_handles_unknown_preset(self):
        """If somehow the combo emits an unrecognized name, the
        handler should show a graceful 'no description' fallback,
        not crash."""
        ctrl = self._make_controller()
        ctrl._on_refine_preset_changed("not_a_real_preset")
        text = ctrl.ui.refine_preset_description.text()
        assert "no description" in text.lower()

    def test_handler_no_op_when_widget_missing(self):
        """Older test fakes / contexts where the description widget
        isn't built — handler must not crash."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        ui = SimpleNamespace()  # no refine_preset_description
        ctrl = SimpleNamespace(ui=ui)
        method = FusionReconstructionController._on_refine_preset_changed
        ctrl._on_refine_preset_changed = method.__get__(ctrl)
        # Must not raise.
        ctrl._on_refine_preset_changed("default")


class TestTooltipCoverage:
    """Regression guard against future refactors silently dropping
    tooltips from key Model Refinement widgets.  Uses inspect on
    `ui.py` source — pure-Python, no Qt required."""

    def _source(self):
        import inspect
        from napari_epicycle.fusion_reconstruction import ui
        return inspect.getsource(ui)

    def test_main_action_tooltips_present(self):
        """The two most important buttons must have setToolTip calls."""
        src = self._source()
        # Find the line where each is constructed, then verify
        # setToolTip appears within 30 lines after.
        for widget_attr in (
            "launch_refinement_btn",
            "extract_training_pairs_btn",
            "refinement_mode_combo",
        ):
            idx = src.find(f"self.{widget_attr}")
            assert idx >= 0, f"Widget {widget_attr} not constructed"
            nearby = src[idx : idx + 1500]
            assert ".setToolTip(" in nearby, (
                f"{widget_attr} has no setToolTip call within 1500 "
                f"chars of its construction"
            )

    def test_training_param_spinbox_tooltips_present(self):
        """The 5 training spinboxes added in 2026-05-22."""
        src = self._source()
        for attr in (
            "refine_epochs_sb",
            "refine_lr_sb",
            "refine_batch_sb",
            "refine_wd_sb",
            "refine_momentum_sb",
        ):
            idx = src.find(f"self.{attr} = QSpinBox()") + src.find(
                f"self.{attr} = QDoubleSpinBox()"
            ) + 1
            # At least one construction site exists.
            assert (
                f"self.{attr} = QSpinBox()" in src
                or f"self.{attr} = QDoubleSpinBox()" in src
            ), f"{attr} not constructed"
            # Tooltip appears.
            assert f"self.{attr}.setToolTip(" in src, (
                f"{attr} has no setToolTip — regression"
            )

    def test_channel_combo_tooltips_present(self):
        """The unified channel labels (from the 2026-05-21 work)."""
        src = self._source()
        assert "self.refine_chan1_combo.setToolTip(" in src
        assert "self.refine_chan2_combo.setToolTip(" in src

    def test_device_radio_tooltips_present(self):
        """GPU / CPU radio buttons added 2026-05-22."""
        src = self._source()
        assert "self.refine_device_gpu.setToolTip(" in src
        assert "self.refine_device_cpu.setToolTip(" in src

    def test_config_buttons_tooltips_present(self):
        """Save Config / Load Config buttons."""
        src = self._source()
        assert "self.refine_save_config_btn.setToolTip(" in src
        assert "self.refine_load_config_btn.setToolTip(" in src
