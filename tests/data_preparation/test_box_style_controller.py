"""④ per-patch box styling — controller handlers (P2 Brick 2, slice 3).

The persistence math is tested in ``test_state.py`` and the overlay rendering in
``test_box_overlay.py``; here we verify the thin controller handlers wire disk +
resolve correctly, via the unbound method on a ``SimpleNamespace`` stand-in
(Qt-free — no widget/viewer needed).
"""

import types

from napari_epicycle.data_preparation.controller import DataPreparationController
from napari_epicycle.data_preparation._state import (
    empty_state, upsert_source, register_patch, save_state, load_state,
    resolve_box_style,
)


def _ws_with_two_patches(tmp_path):
    ws = str(tmp_path)
    st = empty_state()
    sid = upsert_source(st, path="C:/data/img.tif")
    for i in (1, 2):
        register_patch(
            st, patch_name=f"img_patch_{i}",
            relative_path=f"patches/img_patch_{i}.tiff",
            sidecar_path=f"patches/img_patch_{i}.json", source_id=sid)
    save_state(ws, st)
    return ws, sid


def _fake_ctl(ws, sid, patch_names):
    ctl = types.SimpleNamespace(
        widget=types.SimpleNamespace(
            output_file_path=types.SimpleNamespace(text=lambda: ws),
            library_box_list=types.SimpleNamespace(refresh=lambda: None)),
        _visible_patch_locations=set(),
        _loaded_source_file="C:/data/img.tif",
    )
    ctl._log = lambda *a, **k: None
    ctl._rebuild_loc_boxes = lambda: None
    ctl._current_source_id = lambda: sid
    ctl._current_source_patch_names = lambda: list(patch_names)
    ctl._step1_state_cache = None
    ctl._cached_step1_state = types.MethodType(
        DataPreparationController._cached_step1_state, ctl)
    ctl._refresh_after_style = types.MethodType(
        DataPreparationController._refresh_after_style, ctl)
    return ctl


def test_on_patch_style_changed_persists(tmp_path):
    ws, sid = _ws_with_two_patches(tmp_path)
    ctl = _fake_ctl(ws, sid, ["img_patch_1", "img_patch_2"])
    DataPreparationController._on_patch_style_changed(
        ctl, "img_patch_1",
        {"color": [0.1, 0.2, 0.3, 1.0], "shape": "wireframe"})
    got = resolve_box_style(load_state(ws), "img_patch_1")
    assert got["color"] == [0.1, 0.2, 0.3, 1.0] and got["shape"] == "wireframe"
    # The sibling patch is untouched.
    assert resolve_box_style(load_state(ws), "img_patch_2") is None


def test_apply_style_to_source_writes_default_and_every_patch(tmp_path):
    ws, sid = _ws_with_two_patches(tmp_path)
    ctl = _fake_ctl(ws, sid, ["img_patch_1", "img_patch_2"])
    DataPreparationController._on_apply_style_to_source(
        ctl, "img_patch_1", {"color": [0.9, 0.1, 0.1, 1.0]})
    st = load_state(ws)
    # Both patches carry the style now...
    assert resolve_box_style(st, "img_patch_1")["color"] == [0.9, 0.1, 0.1, 1.0]
    assert resolve_box_style(st, "img_patch_2")["color"] == [0.9, 0.1, 0.1, 1.0]
    # ...and it's the source default, so a NEW patch inherits it.
    register_patch(st, patch_name="img_patch_3",
                   relative_path="patches/img_patch_3.tiff",
                   sidecar_path="patches/img_patch_3.json", source_id=sid)
    assert resolve_box_style(st, "img_patch_3")["color"] == [0.9, 0.1, 0.1, 1.0]


def test_swatch_color_for_falls_back_to_default(tmp_path):
    from napari_epicycle.data_preparation.controller import _DEFAULT_LOC_RGB
    ws, sid = _ws_with_two_patches(tmp_path)
    ctl = _fake_ctl(ws, sid, ["img_patch_1"])
    # No style set → the built-in default magenta.
    assert DataPreparationController._swatch_color_for(ctl, "img_patch_1") == _DEFAULT_LOC_RGB
    # A set colour is returned as (r, g, b).
    DataPreparationController._on_patch_style_changed(
        ctl, "img_patch_1", {"color": [0.2, 0.4, 0.6, 1.0]})
    assert DataPreparationController._swatch_color_for(ctl, "img_patch_1") == (0.2, 0.4, 0.6)


def test_handlers_guarded_against_missing_workspace(tmp_path):
    ctl = _fake_ctl("C:/does/not/exist", "src_0", ["img_patch_1"])
    # Must not raise when the workspace path is bad.
    DataPreparationController._on_patch_style_changed(
        ctl, "img_patch_1", {"color": [0.1, 0.1, 0.1, 1.0]})
    DataPreparationController._on_apply_style_to_source(
        ctl, "img_patch_1", {"color": [0.1, 0.1, 0.1, 1.0]})
