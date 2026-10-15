"""Brick B — ⟨processed⟩ store persistence + reopen detection + swap busy cue.

Brick B.  Streaming (prepared/lazy-source) path only — the
in-memory/small-source store writer is OUT (deferred).

Covered:
  * the ``processed_store`` pointer round-trips through ``_state``
    (`set_processed_store` / `get_processed_store`) + is written on a streaming
    Process finish / fresh-skip (`_persist_processed_pointer`).
  * reopen detection (`_maybe_light_processed_toggle_on_load`) LIGHTS the toggle
    (`_processed_available` + `_processed_redisplay`) when a FRESH on-disk store
    exists — resolved POINTER-FIRST (survives rename) or via the deterministic
    clean name — and leaves it GREYED when absent / stale (recipe edited or raw
    file changed).  Freshness is store-attrs authoritative (T3).
  * default view stays RAW on reopen (processed NOT auto-shown).
  * the busy "Loading…" cue is shown + hidden around a synchronous view swap
    (and skipped while a Process owns ②'s bar).

Qt-free: fake widget / viewer, real zarr `epicycle_processed` stores on disk.
"""

from __future__ import annotations

import os

import pytest

from napari_epicycle.data_preparation import controller
from napari_epicycle.data_preparation import _state
from napari_epicycle.data_preparation._state import (
    load_state, get_processed_store, set_processed_store, upsert_source,
)


DPC = controller.DataPreparationController


# --------------------------------------------------------------------------- #
# Qt-free fakes
# --------------------------------------------------------------------------- #

class _FakeText:
    def __init__(self, val):
        self._val = val

    def text(self):
        return self._val


class _FakeWidget:
    def __init__(self, workspace, recipe=None):
        self.output_file_path = _FakeText(workspace)
        self._recipe = recipe
        self.busy_calls = []
        self.view_toggle_calls = []

    def get_recipe_json(self):
        return self._recipe

    def _busy_show(self, label):
        self.busy_calls.append(("show", label))

    def _busy_hide(self, idle="— idle"):
        self.busy_calls.append(("hide", idle))

    def _set_view_toggle_state(self, has_processed, showing_processed):
        self.view_toggle_calls.append((has_processed, showing_processed))


def _bare_ctl(**attrs):
    ctl = DPC.__new__(DPC)
    ctl._log = lambda *a, **k: None
    # defaults the toggle/reopen paths read
    ctl.viewer = None
    ctl._processed_store_layers = []
    ctl._processed_layers = []
    ctl._processed_available = False
    ctl._processed_redisplay = None
    ctl._view_toggle_key_bound = False
    ctl._processing = False
    for k, v in attrs.items():
        setattr(ctl, k, v)
    return ctl


# a simple 2-channel intensity recipe (streamable — no geometry/CLAHE)
_RECIPE_2CH = {
    "channels": [
        [{"key": "gaussian", "params": {"sigma": 1.0}}],
        [{"key": "gaussian", "params": {"sigma": 1.0}}],
    ],
    "geometry": None,
}

_CHANNELS_2 = [("kidney_ch0", [None]), ("kidney_ch1", [None])]


def _make_proc_store(path, recipe_hash, src_path):
    """A minimal valid zarr group carrying an ``epicycle_processed`` block whose
    ``source`` fingerprint is captured from *src_path*'s CURRENT mtime/size."""
    zarr = pytest.importorskip("zarr")
    fp = {"path": str(src_path)}
    if src_path and os.path.exists(src_path):
        fp["source_mtime"] = os.path.getmtime(src_path)
        fp["source_size"] = os.path.getsize(src_path)
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    group = zarr.open_group(str(path), mode="w")
    group.attrs["epicycle_processed"] = {
        "recipe_hash": recipe_hash, "source": fp}


# --------------------------------------------------------------------------- #
# 1 — pointer round-trips through _state
# --------------------------------------------------------------------------- #

def test_set_get_processed_store_roundtrip(tmp_path):
    ws = str(tmp_path)
    state = _state.empty_state()
    sid = upsert_source(state, path="/data/src.tif", display_name="kidney")
    rel = "artifacts/step1/largedata/kidney_processed.zarr"
    set_processed_store(state, source_id=sid, path="/data/src.tif", rel_path=rel)
    _state.save_state(ws, state)

    reloaded = load_state(ws)
    row = next(s for s in reloaded["sources"] if s["source_id"] == sid)
    ptr = get_processed_store(row)
    assert ptr is not None
    assert ptr["path"] == rel
    assert ptr["format"] == "ome-zarr"
    assert isinstance(ptr.get("materialized_at"), str) and ptr["materialized_at"]
    # origin unioned "prepared"; no recipe_hash in the pointer (attrs-authoritative)
    assert "prepared" in row.get("origin", [])
    assert "recipe_hash" not in ptr


def test_get_processed_store_none_when_absent():
    assert get_processed_store(None) is None
    assert get_processed_store({}) is None
    assert get_processed_store({"processed_store": "notadict"}) is None


# --------------------------------------------------------------------------- #
# 2 — _persist_processed_pointer writes a workspace-relative pointer
# --------------------------------------------------------------------------- #

def test_persist_processed_pointer_writes_relative(tmp_path):
    ws = str(tmp_path)
    src = tmp_path / "src.tif"
    src.write_bytes(b"x" * 64)
    store = os.path.join(ws, "artifacts", "step1", "largedata",
                         "kidney_processed.zarr")
    os.makedirs(store)
    ctl = _bare_ctl(widget=_FakeWidget(ws), _loaded_source_file=str(src))

    ctl._persist_processed_pointer(store)

    state = load_state(ws)
    row = ctl._find_source_row(state, str(src))
    ptr = get_processed_store(row)
    assert ptr is not None
    assert ptr["path"] == "artifacts/step1/largedata/kidney_processed.zarr"
    assert ptr["format"] == "ome-zarr"


def test_persist_processed_pointer_best_effort_no_workspace(tmp_path):
    # No workspace → best-effort no-op, never raises.
    ctl = _bare_ctl(widget=_FakeWidget(""), _loaded_source_file="/x/src.tif")
    ctl._persist_processed_pointer("/x/store.zarr")   # must not raise


# --------------------------------------------------------------------------- #
# 3 — reopen detection: light the toggle when a FRESH store exists
# --------------------------------------------------------------------------- #

def _wire_reopen(ws, tmp_path, store_path=None, display_name="kidney"):
    """A controller ready for `_maybe_light_processed_toggle_on_load`, with a
    real 2-channel-recipe src file loaded.  Returns (ctl, src_path, expect_hash)."""
    src = tmp_path / "src.tif"
    src.write_bytes(b"y" * 128)
    ctl = _bare_ctl(
        widget=_FakeWidget(ws, recipe=_RECIPE_2CH),
        _loaded_source_file=str(src),
        _source_display_channels=list(_CHANNELS_2),
    )
    expect_hash = ctl._processed_recipe_hash(_RECIPE_2CH, 2)
    return ctl, str(src), expect_hash


def test_reopen_lights_toggle_via_deterministic_name(tmp_path):
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    ctl, src, expect_hash = _wire_reopen(ws, tmp_path)
    # store at the deterministic clean name (no pointer)
    store = os.path.join(ctl._largedata_dir(ws), "kidney_processed.zarr")
    _make_proc_store(store, expect_hash, src)

    ctl._maybe_light_processed_toggle_on_load("kidney")

    assert ctl._processed_available is True
    assert ctl._processed_redisplay["kind"] == "store"
    assert os.path.normpath(ctl._processed_redisplay["store_path"]) == \
        os.path.normpath(store)
    assert ctl._processed_redisplay["names"] == ["kidney_ch0", "kidney_ch1"]
    # the toggle was refreshed: enabled, NOT showing processed (default raw)
    assert ctl.widget.view_toggle_calls[-1] == (True, False)


def test_reopen_lights_toggle_via_pointer_after_rename(tmp_path):
    """A display-name RENAME orphans the deterministic name — the persisted
    pointer resolves the store first (pointer-first resolver)."""
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    ctl, src, expect_hash = _wire_reopen(ws, tmp_path)
    # store lives at a RENAMED path recorded only in the pointer
    rel = "artifacts/step1/largedata/oldname_processed.zarr"
    store = os.path.join(ws, *rel.split("/"))
    _make_proc_store(store, expect_hash, src)
    state = load_state(ws)
    sid = upsert_source(state, path=src, display_name="whatever")
    set_processed_store(state, source_id=sid, path=src, rel_path=rel)
    _state.save_state(ws, state)

    # pass a DIFFERENT display_name → deterministic name would miss; pointer wins
    ctl._maybe_light_processed_toggle_on_load("newname")

    assert ctl._processed_available is True
    assert os.path.normpath(ctl._processed_redisplay["store_path"]) == \
        os.path.normpath(store)


def test_reopen_greyed_when_no_store(tmp_path):
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    ctl, src, _ = _wire_reopen(ws, tmp_path)
    ctl._maybe_light_processed_toggle_on_load("kidney")   # nothing on disk
    assert ctl._processed_available is False
    assert ctl._processed_redisplay is None


def test_reopen_greyed_when_recipe_edited(tmp_path):
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    ctl, src, expect_hash = _wire_reopen(ws, tmp_path)
    store = os.path.join(ctl._largedata_dir(ws), "kidney_processed.zarr")
    _make_proc_store(store, expect_hash + "EDITED", src)   # stale hash
    ctl._maybe_light_processed_toggle_on_load("kidney")
    assert ctl._processed_available is False
    assert ctl._processed_redisplay is None


def test_reopen_greyed_when_source_changed(tmp_path):
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    ctl, src, expect_hash = _wire_reopen(ws, tmp_path)
    store = os.path.join(ctl._largedata_dir(ws), "kidney_processed.zarr")
    _make_proc_store(store, expect_hash, src)
    # the raw file grows AFTER the store was fingerprinted → size drift → stale
    with open(src, "ab") as fh:
        fh.write(b"z" * 256)
    ctl._maybe_light_processed_toggle_on_load("kidney")
    assert ctl._processed_available is False


def test_reopen_default_raw_does_not_show_processed(tmp_path):
    """T6 / default: reopen LIGHTS the toggle but never AUTO-SHOWS processed
    (no `_show_processed_view`, no processed layers on the canvas)."""
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    ctl, src, expect_hash = _wire_reopen(ws, tmp_path)
    store = os.path.join(ctl._largedata_dir(ws), "kidney_processed.zarr")
    _make_proc_store(store, expect_hash, src)
    shown = []
    ctl._show_processed_view = lambda: shown.append(True)

    ctl._maybe_light_processed_toggle_on_load("kidney")

    assert ctl._processed_available is True   # lit …
    assert shown == []                        # … but NOT shown
    assert ctl._processed_store_layers == []  # no processed layers on the canvas
    assert ctl._is_processed_view_showing() is False


def test_reopen_never_raises_on_bad_state(tmp_path):
    # a widget whose get_recipe_json blows up must not raise out of the load hook
    class _Boom(_FakeWidget):
        def get_recipe_json(self):
            raise RuntimeError("boom")

    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    src = tmp_path / "src.tif"
    src.write_bytes(b"y" * 8)
    ctl = _bare_ctl(
        widget=_Boom(ws),
        _loaded_source_file=str(src),
        _source_display_channels=list(_CHANNELS_2),
    )
    ctl._maybe_light_processed_toggle_on_load("kidney")   # must not raise
    assert ctl._processed_available is False


# --------------------------------------------------------------------------- #
# 4 — busy "Loading…" cue around a synchronous view swap
# --------------------------------------------------------------------------- #

def _bare_swap_ctl(widget, processing=False):
    ctl = _bare_ctl(
        widget=widget,
        viewer=object(),                 # not None → swap proceeds
        _processing=processing,
        _processed_available=True,
        _processed_store_layers=[object()],  # non-empty → skip re-create
        _processed_layers=[],
        _source_layers_on_canvas=False,      # skip the raw removal
        _swapping_source_display=False,
    )
    ctl._refresh_view_toggle_state = lambda: None
    return ctl


def test_view_swap_shows_and_hides_busy():
    widget = _FakeWidget("")
    ctl = _bare_swap_ctl(widget)
    ctl._show_processed_view()
    kinds = [k for (k, _) in widget.busy_calls]
    assert "show" in kinds and "hide" in kinds
    assert kinds.index("show") < kinds.index("hide")   # hide AFTER the churn


def test_view_swap_busy_skipped_while_processing():
    widget = _FakeWidget("")
    ctl = _bare_swap_ctl(widget, processing=True)
    ctl._show_processed_view()
    assert widget.busy_calls == []   # a Process owns ②'s bar → no view-swap cue


def test_view_swap_busy_noop_without_widget_api():
    # a Qt-free fake widget lacking _busy_show must not raise (best-effort)
    class _NoBusy:
        pass
    ctl = _bare_swap_ctl(_NoBusy())
    ctl._show_processed_view()   # must not raise
