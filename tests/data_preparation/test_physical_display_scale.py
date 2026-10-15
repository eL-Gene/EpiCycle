"""Physical-µm display: the voxel-edit re-apply + preview-crop geometry.

Covers the two Brick-A audit follow-ups:

* ``_reapply_display_physical_scale`` — the HIGH fix: a post-load voxel edit must
  push the new physical ``scale``/``units`` onto the live raw display layers, so
  the display can't desync from the LIVE-read placement voxel (a desync silently
  misplaces the extraction region).  Tested Qt-free via the unbound method on a
  ``SimpleNamespace`` stand-in (no napari viewer needed).
* ``_preview_layer_geometry`` — the previously-untested element-wise
  ``translate``/``scale`` × voxel and the bounds-missing ``(None, voxel)`` branch.
"""

import types

from napari_epicycle.data_preparation.controller import DataPreparationController


# ---------------------------------------------------------------------------
# _preview_layer_geometry — element-wise multiply + bounds-missing branch
# ---------------------------------------------------------------------------

def test_preview_geometry_multiplies_translate_and_scale_per_axis():
    geom = DataPreparationController._preview_layer_geometry
    bounds = {"z0": 2, "y0": 4, "x0": 8, "z1": 6, "y1": 12, "x1": 24}
    # crop = (4, 8, 16); proc_shape == crop → ratio (1, 1, 1); anisotropic voxel.
    translate, scale = geom(object(), bounds, (4, 8, 16), (3.0, 2.0, 0.5))
    assert translate == (2 * 3.0, 4 * 2.0, 8 * 0.5)     # origin ⊙ voxel
    assert scale == (1.0 * 3.0, 1.0 * 2.0, 1.0 * 0.5)   # ratio ⊙ voxel


def test_preview_geometry_resample_ratio_times_voxel():
    geom = DataPreparationController._preview_layer_geometry
    bounds = {"z0": 0, "y0": 0, "x0": 0, "z1": 8, "y1": 8, "x1": 8}
    # crop (8, 8, 8) vs proc_shape (4, 8, 8) → ratio (2, 1, 1); unit voxel.
    translate, scale = geom(object(), bounds, (4, 8, 8), (1.0, 1.0, 1.0))
    assert translate == (0, 0, 0)
    assert scale == (2.0, 1.0, 1.0)


def test_preview_geometry_bounds_missing_returns_voxel_not_unit():
    geom = DataPreparationController._preview_layer_geometry
    # No bounds → origin translate, but STILL the physical voxel scale (not None
    # / unit — else the layer would render at voxel-index scale 1, voxel× off).
    translate, scale = geom(object(), None, (4, 8, 8), (3.0, 2.0, 0.5))
    assert translate is None
    assert scale == (3.0, 2.0, 0.5)


def test_preview_geometry_bad_voxel_falls_back_to_unit():
    geom = DataPreparationController._preview_layer_geometry
    translate, scale = geom(object(), None, (4, 8, 8), "not-a-voxel")
    assert translate is None
    assert scale == (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# _reapply_display_physical_scale — the voxel-edit re-apply (HIGH fix)
# ---------------------------------------------------------------------------

class _FakeLayer:
    def __init__(self, name):
        self.name = name
        self.scale = (1.0, 1.0, 1.0)
        self.units = ("pixel", "pixel", "pixel")


def _fake_ctl(voxel, source_names):
    layers = [_FakeLayer(n) for n in source_names]
    ctl = types.SimpleNamespace(
        viewer=types.SimpleNamespace(layers=layers),
        _loaded_source_layers=list(source_names),
        _box_tool=None,
        _visible_patch_locations=set(),
        _preview_box=None,
    )
    ctl._preview_spacing = lambda: voxel
    return ctl, layers


def test_reapply_pushes_physical_scale_and_um_units_to_raw_layers():
    ctl, layers = _fake_ctl((3.0, 2.0, 0.5), ["ch0", "ch1"])
    DataPreparationController._reapply_display_physical_scale(ctl)
    for layer in layers:
        assert tuple(layer.scale) == (3.0, 2.0, 0.5)
        assert tuple(layer.units) == ("um", "um", "um")


def test_reapply_unknown_voxel_is_unit_scale_and_pixel_units():
    ctl, layers = _fake_ctl((1.0, 1.0, 1.0), ["ch0"])
    DataPreparationController._reapply_display_physical_scale(ctl)
    assert tuple(layers[0].scale) == (1.0, 1.0, 1.0)
    # "pixel" (napari's default token), NOT "px" (→ pint css_pixel → mismatch).
    assert tuple(layers[0].units) == ("pixel", "pixel", "pixel")


def test_reapply_leaves_non_source_layers_untouched():
    ctl, layers = _fake_ctl((3.0, 2.0, 0.5), ["ch0"])
    foreign = _FakeLayer("some-other-layer")
    ctl.viewer.layers.append(foreign)
    DataPreparationController._reapply_display_physical_scale(ctl)
    # The foreign layer is NOT in _loaded_source_layers → never re-scaled.
    assert tuple(foreign.scale) == (1.0, 1.0, 1.0)
    assert tuple(foreign.units) == ("pixel", "pixel", "pixel")


def test_reapply_is_guarded_against_no_viewer():
    ctl = types.SimpleNamespace(
        viewer=None, _loaded_source_layers=["ch0"], _box_tool=None,
        _visible_patch_locations=set(), _preview_box=None)
    ctl._preview_spacing = lambda: (2.0, 2.0, 2.0)
    # Must not raise (best-effort out of the voxel-edit slot).
    DataPreparationController._reapply_display_physical_scale(ctl)


# ---------------------------------------------------------------------------
# _apply_live_scale_bar — the Brick B live scale-bar wiring
# ---------------------------------------------------------------------------

class _FakeScaleBar:
    def __init__(self):
        for a in ("visible", "colored", "color", "box", "box_color", "ticks",
                  "font_size", "length", "position", "unit"):
            setattr(self, a, None)


class _FakeCombo:
    def __init__(self, text):
        self._t = text

    def currentText(self):
        return self._t


class _FakeCheck:
    def __init__(self, v):
        self._v = v

    def isChecked(self):
        return self._v


class _FakeSpin:
    def __init__(self, v):
        self._v = v

    def value(self):
        return self._v


def _fake_scalebar_ctl(show, auto=True, length=50.0, font=10, color="light",
                       box=False, ticks=True, pos="bottom right", unit="auto"):
    sb = _FakeScaleBar()
    w = types.SimpleNamespace(
        scalebar_toggle_button=_FakeCheck(show),
        scalebar_auto_length_check=_FakeCheck(auto),
        scalebar_length_spin=_FakeSpin(length),
        scalebar_font_spin=_FakeSpin(font),
        scalebar_unit_combo=_FakeCombo(unit),
        scalebar_color_combo=_FakeCombo(color),
        scalebar_box_check=_FakeCheck(box),
        scalebar_ticks_check=_FakeCheck(ticks),
        scalebar_position_combo=_FakeCombo(pos),
    )
    ctl = types.SimpleNamespace(
        viewer=types.SimpleNamespace(scale_bar=sb), widget=w)
    # The bar's label is driven through layer.units — record what the picker
    # forces onto the display layers, and give the auto branch a voxel.
    ctl._units_applied = []
    ctl._set_source_layer_units = lambda u: ctl._units_applied.append(tuple(u))
    ctl._preview_spacing = lambda: (1.0, 1.0, 1.0)
    return ctl, sb


def test_live_scale_bar_off_sets_invisible_and_skips_styling():
    ctl, sb = _fake_scalebar_ctl(show=False)
    DataPreparationController._apply_live_scale_bar(ctl)
    assert sb.visible is False
    assert sb.color is None   # nothing else applied when off


def test_live_scale_bar_on_applies_full_style():
    ctl, sb = _fake_scalebar_ctl(
        show=True, auto=True, font=12, color="dark", box=True, ticks=False,
        pos="top left", unit="auto")
    DataPreparationController._apply_live_scale_bar(ctl)
    assert sb.visible is True
    assert sb.colored is True
    assert sb.color == (0.0, 0.0, 0.0, 1.0)   # dark = black bar/text
    assert sb.box is True
    assert sb.ticks is False
    assert sb.font_size == 12.0
    assert sb.length is None                  # auto
    assert sb.position == "top_left"
    # 'auto' units → scale_bar.unit stays None (reads layer.units), and the
    # voxel-derived unit ((1,1,1) → pixel) is pushed onto the display layers.
    assert sb.unit is None
    assert ctl._units_applied[-1] == ("pixel", "pixel", "pixel")


def test_live_scale_bar_fixed_length_used_when_not_auto():
    ctl, sb = _fake_scalebar_ctl(show=True, auto=False, length=25.5)
    DataPreparationController._apply_live_scale_bar(ctl)
    assert sb.length == 25.5


def test_live_scale_bar_forced_unit_pushed_onto_layers():
    # Forcing a unit drives it through layer.units (reliable), NOT scale_bar.unit.
    ctl, sb = _fake_scalebar_ctl(show=True, unit="µm")
    DataPreparationController._apply_live_scale_bar(ctl)
    assert ctl._units_applied[-1] == ("um", "um", "um")   # µm → pint "um"
    assert sb.unit is None                                # bar reads layer.units
    ctl, sb = _fake_scalebar_ctl(show=True, unit="nm")
    DataPreparationController._apply_live_scale_bar(ctl)
    assert ctl._units_applied[-1] == ("nm", "nm", "nm")


def test_live_scale_bar_guarded_against_no_viewer():
    ctl = types.SimpleNamespace(viewer=None, widget=None)
    DataPreparationController._apply_live_scale_bar(ctl)  # must not raise
