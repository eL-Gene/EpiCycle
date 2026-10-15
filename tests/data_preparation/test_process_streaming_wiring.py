"""Controller-level wiring tests for the STREAMING (out-of-core) whole-volume
process (Phase 2a).

These exercise the Qt-free decision + keying + display + worker-cancel surface the
regime router wires into ``DataPreparationController`` — WITHOUT a full napari
viewer (mirrors ``test_process_whole_volume.py`` / ``test_process_cancel_runtime``):

  * ``_is_lazy_handle`` / ``_should_stream_process`` — lazy (dask/zarr) → stream,
    resident numpy → in-memory.
  * ``_recipe_has_stream_remainder`` — a CLAHE / bleach / geometry op → refuse
    (Phase 2a scope), local/pointwise/normalize → stream.
  * store keying (``_resolve_processed_store_path``) + fresh-skip
    (``_fresh_processed_store_for``) + superseded GC
    (``_delete_superseded_processed_stores``).
  * ``_display_processed_store`` — ADDITIVE lazy-multiscale layers that NEVER touch
    the raw-source extract state (the double-processing fix by construction).
  * ``_PreprocProcessToStoreWorker`` cancel on a real QThread — emits a clean
    "cancelled" + leaves NO store (mirrors ``test_process_cancel_runtime.py``).
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

from napari_epicycle.data_preparation import controller
from packages.epicycle_preprocessing.exceptions import PreprocessingCancelled


DPC = controller.DataPreparationController


def _bare_ctl(**attrs):
    """A controller instance with NO __init__ (real bound methods, minimal state).
    `_log` is stubbed; extra state is injected via kwargs."""
    ctl = DPC.__new__(DPC)
    ctl._log = lambda *a, **k: None
    for k, v in attrs.items():
        setattr(ctl, k, v)
    return ctl


# recipe fixtures -------------------------------------------------------------

_GAUSS = {"channels": [[{"key": "gaussian", "params": {"sigma": 1.5}}]],
          "geometry": None}
_FUSED = {"channels": [[{"key": "gaussian", "params": {"sigma": 1.5}},
                        {"key": "background", "params": {"radius": 3}}]],
          "geometry": None}
_CLAHE = {"channels": [[{"key": "clahe", "params": {}}]], "geometry": None}
_BLEACH = {"channels": [[{"key": "bleach", "params": {}}]], "geometry": None}
_GEOM = {"channels": [[]], "geometry": {
    "key": "isotropic", "boundary_index": 0,
    "config": {"mode": "iso", "target_um": 1.0, "method": "linear"}}}
_EMPTY = {"channels": [[]], "geometry": None}


# --------------------------------------------------------------------------- #
# A — regime oracle: lazy → stream, resident → in-memory
# --------------------------------------------------------------------------- #

def test_is_lazy_handle_numpy_is_resident():
    assert DPC._is_lazy_handle(np.zeros((3, 3, 3))) is False


def test_is_lazy_handle_memmap_is_resident(tmp_path):
    # A memmap is an ndarray subclass → resident (a raw >RAM memmap Prepares first).
    mm = np.memmap(str(tmp_path / "m.dat"), dtype=np.uint16,
                   mode="w+", shape=(2, 2, 2))
    assert DPC._is_lazy_handle(mm) is False


def test_is_lazy_handle_dask_and_zarr_are_lazy():
    da = pytest.importorskip("dask.array")
    zarr = pytest.importorskip("zarr")
    assert DPC._is_lazy_handle(da.zeros((4, 4, 4), chunks=2)) is True
    z = zarr.zeros((4, 4, 4), chunks=(2, 2, 2), dtype="uint16")
    assert DPC._is_lazy_handle(z) is True


def test_should_stream_resident_false():
    ctl = _bare_ctl()
    chans = [np.zeros((4, 4, 4)), np.zeros((4, 4, 4))]
    assert ctl._should_stream_process(chans) is False


def test_should_stream_lazy_true():
    da = pytest.importorskip("dask.array")
    ctl = _bare_ctl()
    chans = [da.zeros((4, 4, 4), chunks=2)]
    assert ctl._should_stream_process(chans) is True


def test_should_stream_empty_false():
    assert _bare_ctl()._should_stream_process([]) is False


def test_should_stream_any_lazy_true():
    """A mixed set with ANY lazy channel streams (the engine reads numpy blocks
    too, so a mixed set is safe)."""
    da = pytest.importorskip("dask.array")
    ctl = _bare_ctl()
    chans = [np.zeros((4, 4, 4)), da.zeros((4, 4, 4), chunks=2)]
    assert ctl._should_stream_process(chans) is True


# --------------------------------------------------------------------------- #
# B — remainder gate: CLAHE / bleach / geometry refuse; local/pointwise stream
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("recipe", [_GAUSS, _FUSED, _EMPTY])
def test_no_remainder_for_streamable_recipes(recipe):
    assert _bare_ctl()._recipe_has_stream_remainder(recipe, 1) is False


@pytest.mark.parametrize("recipe", [_CLAHE, _BLEACH, _GEOM])
def test_remainder_for_nonstreaming_recipes(recipe):
    assert _bare_ctl()._recipe_has_stream_remainder(recipe, 1) is True


def test_remainder_true_if_any_channel_has_one():
    # ch0 streamable, ch1 has CLAHE → refuse (per-channel scan).
    recipe = {"channels": [[{"key": "gaussian", "params": {"sigma": 1.0}}],
                           [{"key": "clahe", "params": {}}]],
              "geometry": None}
    assert _bare_ctl()._recipe_has_stream_remainder(recipe, 2) is True


# --------------------------------------------------------------------------- #
# C — store keying + fresh-skip + superseded GC
# --------------------------------------------------------------------------- #

def test_resolve_processed_store_path_keys_on_source_id(tmp_path):
    """B2 collision guard (2026-08-20): the store name carries the source_id, so
    two sources that share a filename STEM get DISTINCT store paths.  Pre-B2 both
    mapped to `<stem>_processed.zarr` and the 2nd Process's streaming commit
    (rmtree+rename) DESTROYED the 1st source's store — silent data loss."""
    from napari_epicycle.data_preparation._state import (
        empty_state, upsert_source, save_state)
    ctl = _bare_ctl()
    ws = str(tmp_path)
    store, sid = ctl._resolve_processed_store_path(ws, "kidney", "/data/src.tif")
    base = os.path.basename(store)
    # source_id IS in the name now (the B2 guard); still no `__proc__`
    assert base == f"kidney__{sid}_processed.zarr"
    assert "__proc__" not in base
    assert sid
    assert os.path.normpath(store).startswith(
        os.path.normpath(ctl._largedata_dir(ws)))
    # two SAME-STEM sources (distinct ids in state) → DISTINCT store paths.
    state = empty_state()
    id_a = upsert_source(state, path="/data/a/kidney.tif", display_name="kidney")
    id_b = upsert_source(state, path="/data/b/kidney.tif", display_name="kidney")
    save_state(ws, state)
    assert id_a != id_b
    sa, _ = ctl._resolve_processed_store_path(ws, "kidney", "/data/a/kidney.tif")
    sb, _ = ctl._resolve_processed_store_path(ws, "kidney", "/data/b/kidney.tif")
    assert sa != sb                              # collision fixed
    assert os.path.basename(sa) == f"kidney__{id_a}_processed.zarr"
    assert os.path.basename(sb) == f"kidney__{id_b}_processed.zarr"


def test_delete_superseded_gc_recovers_display_name_from_b2_name(tmp_path):
    """A2: with a B2 per-source keep name `<name>__<sid>_processed.zarr`, the GC
    must recover the TRUE display_name (strip the FULL `__<sid>_processed.zarr`
    suffix) so its legacy `__proc__` sweep matches — a bare `_processed.zarr` strip
    would leave `<name>__<sid>` and double the sid in the glob (orphaning caches)."""
    from napari_epicycle.data_preparation._state import (
        empty_state, upsert_source, save_state)
    ws = tmp_path / "ws"
    ld = ws / "artifacts" / "step1" / "largedata"
    os.makedirs(ld)
    src = tmp_path / "kidney.tif"
    src.write_bytes(b"x" * 10)
    state = empty_state()
    sid = upsert_source(state, path=str(src), display_name="kidney")
    save_state(str(ws), state)
    ctl = _bare_ctl(
        widget=SimpleNamespace(
            output_file_path=SimpleNamespace(text=lambda: str(ws))),
        _loaded_source_file=str(src))
    keep = ld / f"kidney__{sid}_processed.zarr"         # B2 clean name
    legacy1 = ld / f"kidney__{sid}__proc__HASH1.zarr"   # legacy caches for THIS src
    legacy2 = ld / f"kidney__{sid}__proc__HASH2.zarr"
    other = ld / "liver__src_9__proc__HASHX.zarr"       # different source
    for p in (keep, legacy1, legacy2, other):
        os.makedirs(p)
    ctl._delete_superseded_processed_stores(str(keep))
    assert os.path.isdir(keep)          # current clean store kept
    assert not os.path.isdir(legacy1)   # this source's legacy caches → swept
    assert not os.path.isdir(legacy2)
    assert os.path.isdir(other)         # a different source is untouched


def test_reclaim_superseded_store(tmp_path):
    """B2-B: the pointer-driven reclaim deletes EXACTLY the old store when it
    differs from the new one — never the new (live) store, a missing path, or on
    None.  (The highest-consequence line in the B2 diff — audit flagged it.)"""
    ctl = _bare_ctl()
    old = tmp_path / "old_processed.zarr"
    new = tmp_path / "new_processed.zarr"
    os.makedirs(old)
    os.makedirs(new)
    ctl._reclaim_superseded_store(str(old), str(new))   # old != new, exists → gone
    assert not os.path.isdir(old)
    assert os.path.isdir(new)
    ctl._reclaim_superseded_store(str(new), str(new))   # old == new → NEVER deletes
    assert os.path.isdir(new)
    ctl._reclaim_superseded_store(str(tmp_path / "gone.zarr"), str(new))  # no-op
    ctl._reclaim_superseded_store(None, str(new))                        # no-op
    assert os.path.isdir(new)


def test_store_id_fallback_stable_per_path_on_unreadable_state(
        tmp_path, monkeypatch):
    """B2 (audit LOW): if state is UNREADABLE at name-derivation, the fallback id
    is a STABLE per-source-PATH hash (not a shared "src"), so two distinct
    same-stem sources still get DISTINCT store names."""
    ctl = _bare_ctl()
    ws = str(tmp_path)

    def _boom(*a, **k):
        raise OSError("corrupt state")
    monkeypatch.setattr(controller, "load_state", _boom)

    sa, ida = ctl._resolve_processed_store_path(ws, "kidney", "/data/a/kidney.tif")
    sb, idb = ctl._resolve_processed_store_path(ws, "kidney", "/data/b/kidney.tif")
    assert ida.startswith("src_") and idb.startswith("src_")
    assert ida != idb                    # distinct per path → distinct names
    assert sa != sb
    sa2, ida2 = ctl._resolve_processed_store_path(   # same path → stable
        ws, "kidney", "/data/a/kidney.tif")
    assert ida2 == ida and sa2 == sa


def test_current_processed_pointer_path(tmp_path):
    """`_current_processed_pointer_path` returns the abspath of the loaded
    source's persisted pointer, or None when absent."""
    from napari_epicycle.data_preparation._state import (
        empty_state, upsert_source, set_processed_store, save_state)
    ws = tmp_path / "ws"
    os.makedirs(ws)
    src = tmp_path / "src.tif"
    src.write_bytes(b"x" * 8)
    ctl = _bare_ctl(
        widget=SimpleNamespace(
            output_file_path=SimpleNamespace(text=lambda: str(ws))),
        _loaded_source_file=str(src))
    state = empty_state()
    sid = upsert_source(state, path=str(src), display_name="kidney")
    save_state(str(ws), state)
    assert ctl._current_processed_pointer_path() is None      # no pointer yet
    rel = f"artifacts/step1/largedata/kidney__{sid}_processed.zarr"
    set_processed_store(state, source_id=sid, path=str(src), rel_path=rel)
    save_state(str(ws), state)
    got = ctl._current_processed_pointer_path()
    assert got is not None
    assert os.path.normpath(got) == os.path.normpath(
        os.path.join(str(ws), *rel.split("/")))


def test_resolve_processed_store_path_overwrites_on_recipe_change(tmp_path):
    """A DIFFERENT recipe resolves to the SAME clean path (the streaming writer
    overwrites in place) — no per-recipe store proliferation."""
    ctl = _bare_ctl()
    ws = str(tmp_path)
    s1, _ = ctl._resolve_processed_store_path(ws, "kidney", "/data/src.tif")
    s2, _ = ctl._resolve_processed_store_path(ws, "kidney", "/data/src.tif")
    assert s1 == s2


def _make_proc_store(path, recipe_hash, src_path):
    """Write a minimal valid zarr group carrying an `epicycle_processed` block."""
    zarr = pytest.importorskip("zarr")
    fp = {"path": src_path}
    if src_path and os.path.exists(src_path):
        fp["source_mtime"] = os.path.getmtime(src_path)
        fp["source_size"] = os.path.getsize(src_path)
    group = zarr.open_group(str(path), mode="w")
    group.attrs["epicycle_processed"] = {
        "recipe_hash": recipe_hash, "source": fp}


def test_fresh_processed_store_hit(tmp_path):
    zarr = pytest.importorskip("zarr")
    ctl = _bare_ctl()
    src = tmp_path / "src.tif"
    src.write_bytes(b"x" * 100)
    store = tmp_path / "s__src_0__proc__abc.zarr"
    _make_proc_store(store, "abc123", str(src))
    assert ctl._fresh_processed_store_for(str(store), str(src), "abc123") == str(store)


def test_fresh_processed_store_miss_on_recipe_change(tmp_path):
    ctl = _bare_ctl()
    src = tmp_path / "src.tif"
    src.write_bytes(b"x" * 100)
    store = tmp_path / "s__src_0__proc__abc.zarr"
    _make_proc_store(store, "abc123", str(src))
    # different recipe hash → not fresh
    assert ctl._fresh_processed_store_for(str(store), str(src), "OTHER") is None


def test_fresh_processed_store_miss_on_source_edit(tmp_path):
    ctl = _bare_ctl()
    src = tmp_path / "src.tif"
    src.write_bytes(b"x" * 100)
    store = tmp_path / "s__src_0__proc__abc.zarr"
    _make_proc_store(store, "abc123", str(src))
    # the source grows (size drift an mtime-preserving copy would hide) → not fresh
    src.write_bytes(b"x" * 200)
    os.utime(str(src), None)
    assert ctl._fresh_processed_store_for(str(store), str(src), "abc123") is None


def test_fresh_processed_store_missing_dir_is_none(tmp_path):
    ctl = _bare_ctl()
    assert ctl._fresh_processed_store_for(
        str(tmp_path / "nope.zarr"), None, "abc") is None


def test_delete_superseded_processed_stores(tmp_path):
    """Under the clean-name scheme the GC sweeps only LEGACY per-recipe caches
    (`<name>__<sid>__proc__<hash>.zarr`) for THIS source — never the kept clean
    `<name>_processed.zarr`, never a DIFFERENT display-name's stores."""
    ctl = _bare_ctl()
    keep = tmp_path / "kidney_processed.zarr"              # current clean store
    old1 = tmp_path / "kidney__src_0__proc__HASHOLD1.zarr"  # legacy caches
    old2 = tmp_path / "kidney__src_0__proc__HASHOLD2.zarr"
    other = tmp_path / "liver__src_1__proc__HASHX.zarr"    # different source
    for p in (keep, old1, old2, other):
        os.makedirs(p)
    # No widget/workspace on the bare ctl → source_id unresolved → the
    # display-name-scoped fallback pattern (still same-name-only).
    ctl._delete_superseded_processed_stores(str(keep))
    assert os.path.isdir(keep)         # current clean store kept
    assert not os.path.isdir(old1)     # legacy → gone
    assert not os.path.isdir(old2)
    assert os.path.isdir(other)        # a different display-name is untouched


def test_delete_superseded_noop_when_no_legacy(tmp_path):
    """A clean-name store with no legacy siblings is a safe no-op (never removes
    the current store)."""
    ctl = _bare_ctl()
    keep = tmp_path / "kidney_processed.zarr"
    os.makedirs(keep)
    ctl._delete_superseded_processed_stores(str(keep))
    assert os.path.isdir(keep)


# --------------------------------------------------------------------------- #
# D — additive display NEVER touches raw-source extract state
# --------------------------------------------------------------------------- #

class _FakeLayer:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.visible = True


class _FakeLayerList(list):
    def remove(self, layer):
        list.remove(self, layer)


class _FakeViewer:
    def __init__(self):
        self.layers = _FakeLayerList()
        self.added = []

    def add_image(self, data, **kwargs):
        layer = _FakeLayer(data=data, **kwargs)
        self.layers.append(layer)
        self.added.append(kwargs)
        return layer


def _stream_tiny_store(tmp_path):
    """Stream a tiny 1-channel gaussian recipe into a real __proc__ store; return
    its path.  Uses the REAL engine (zarr/dask) so the store loads back through
    `open_multiscale`."""
    zarr = pytest.importorskip("zarr")
    pytest.importorskip("dask")
    from packages.epicycle_preprocessing import ops, streaming

    vol = np.random.default_rng(0).integers(0, 4000, (40, 48, 48), dtype=np.uint16)
    src_path = tmp_path / "src.zarr"
    z = zarr.open(str(src_path), mode="w", shape=vol.shape,
                  chunks=(32, 32, 32), dtype="uint16")
    z[:] = vol
    out = tmp_path / "disp__src_0__proc__hh.zarr"
    streaming.stream_process_to_store(
        [zarr.open(str(src_path), mode="r")], _GAUSS, (1.0, 1.0, 1.0), out,
        bounds_per_channel=[ops.percentile_bounds(vol.astype(np.float32))],
        block_workers=2, chunk=32)
    return str(out)


def test_display_processed_store_additive_preserves_source_state(tmp_path):
    store = _stream_tiny_store(tmp_path)
    viewer = _FakeViewer()
    sentinel_lazy = object()
    ctl = _bare_ctl(
        viewer=viewer,
        _processed_store_layers=[],
        _adding_processed_layers=False,
        _source_display_channels=None,      # → colormap "gray" for single channel
        # RAW-source extract state — must be UNTOUCHED after display
        _source_lazy=sentinel_lazy,
        _loaded_source_file="/raw/src.tif",
        _channel_index_by_name={"disp": 0},
        _source_axes_resolved="ZYX",
        _current_multiscale_store="/raw/prepared.zarr",
    )

    n = ctl._display_processed_store(store, ["disp"])

    assert n == 1
    assert len(ctl._processed_store_layers) == 1
    kw = viewer.added[0]
    assert kw["name"].startswith(controller.PROCESSED_LAYER_MARKER)
    assert kw["contrast_limits"] == (0, 65535)
    assert kw["blending"] == "additive"
    assert kw.get("multiscale") is True
    # the re-entrancy guard is reset after the insert
    assert ctl._adding_processed_layers is False
    # CRITICAL: the raw-source extract state is byte-for-byte untouched — the next
    # Process re-processes the RAW source, not the processed output.
    assert ctl._source_lazy is sentinel_lazy
    assert ctl._loaded_source_file == "/raw/src.tif"
    assert ctl._channel_index_by_name == {"disp": 0}
    assert ctl._source_axes_resolved == "ZYX"
    assert ctl._current_multiscale_store == "/raw/prepared.zarr"


def test_clear_processed_store_layers_removes_and_empties(tmp_path):
    viewer = _FakeViewer()
    l0 = viewer.add_image(np.zeros((2, 2, 2)), name="a")
    l1 = viewer.add_image(np.zeros((2, 2, 2)), name="b")
    ctl = _bare_ctl(viewer=viewer, _processed_store_layers=[l0, l1])
    ctl._clear_processed_store_layers()
    assert ctl._processed_store_layers == []
    assert l0 not in viewer.layers and l1 not in viewer.layers


# --------------------------------------------------------------------------- #
# E — worker cancel on a real QThread: clean "cancelled" + NO store left
# --------------------------------------------------------------------------- #

pytest.importorskip("qtpy")
from qtpy.QtCore import QCoreApplication, QThread  # noqa: E402
from qtpy.QtTest import QTest  # noqa: E402


def _slow_stream(channels, recipe, spacing, out_store, *, block_workers=None,
                 should_cancel=None, on_progress=None, source_fingerprint=None):
    """Stand-in engine that polls should_cancel so a mid-run Cancel reliably
    raises PreprocessingCancelled (and writes NO store)."""
    for _ in range(1000):
        if should_cancel is not None and should_cancel():
            raise PreprocessingCancelled("cancelled between waves")
        time.sleep(0.005)
    return {"out_store_path": str(out_store), "whole_array_remainder": False}


@pytest.fixture
def app():
    a = QCoreApplication.instance() or QCoreApplication([])
    yield a


def test_store_worker_cancel_mid_run_emits_cancelled_no_store(app, monkeypatch,
                                                              tmp_path):
    zarr = pytest.importorskip("zarr")
    monkeypatch.setattr(controller, "_pp_stream_to_store", _slow_stream)

    # a small in-memory zarr source (the slow stub ignores it, mirrors realism)
    src = zarr.zeros((4, 4, 4), chunks=(2, 2, 2), dtype="uint16")
    out_store = tmp_path / "out__src_0__proc__hh.zarr"

    worker = controller._PreprocProcessToStoreWorker(
        _GAUSS, [src], (1.0, 1.0, 1.0), str(out_store), block_workers=2)
    thread = QThread()
    worker.moveToThread(thread)

    got = []
    worker.error.connect(lambda m: got.append(("error", m)))
    worker.finished.connect(lambda *a: got.append(("finished", None)))
    worker.error.connect(thread.quit)
    worker.finished.connect(thread.quit)
    thread.started.connect(worker.run)

    thread.start()
    QTest.qWait(60)               # let the worker enter the polling loop
    worker.request_cancel()       # Cancel from the MAIN thread

    waited = 0
    while thread.isRunning() and waited < 4000:
        QTest.qWait(20)
        waited += 20

    assert not thread.isRunning(), "worker did not stop after Cancel"
    thread.wait(2000)
    assert ("error", "cancelled") in got
    assert all(kind != "finished" for kind, _ in got)
    # a cancelled streaming run leaves NO store at the final path
    assert not os.path.exists(out_store)

    thread.deleteLater()
    worker.deleteLater()
    QTest.qWait(10)


def test_process_store_worker_forwards_chunk_params(app, monkeypatch):
    """Chunk-parity: the process-to-store worker forwards the source's Prepare
    chunk/downsample/min_size to the streaming engine so the processed store
    matches the raw prepared store's structure; a None is OMITTED so the engine
    keeps its default (== the raw convert default)."""
    captured = {}

    def _capture(channels, recipe, spacing, out, **kw):
        captured.clear()
        captured.update(kw)
        return {"levels": 1}
    monkeypatch.setattr(controller, "_pp_stream_to_store", _capture)

    # explicit Prepare settings → forwarded verbatim
    controller._PreprocProcessToStoreWorker(
        _GAUSS, [], (1.0, 1.0, 1.0), "/tmp/x.zarr",
        chunk=128, downsample=4, min_size=512).run()
    assert captured.get("chunk") == 128
    assert captured.get("downsample") == 4
    assert captured.get("min_size") == 512

    # unset (None) → omitted, so the engine default (== raw default) applies
    controller._PreprocProcessToStoreWorker(
        _GAUSS, [], (1.0, 1.0, 1.0), "/tmp/x.zarr").run()
    assert "chunk" not in captured
    assert "downsample" not in captured
    assert "min_size" not in captured


# --------------------------------------------------------------------------- #
# B2 — the CONTROLLER gate honours `spacing` (Brick 1b #6)
# --------------------------------------------------------------------------- #

class TestDefaultRecipeGateSpacing:
    """`_recipe_has_stream_remainder` is the pre-launch gate for "Apply recipe →
    whole volume".  The default recipe seeds a shared ``isotropic`` geometry
    step, which is a NO-OP at unit spacing — the fix for that lives in the
    streaming engine, but only the controller call site decides which spacing
    the gate is given.  These pin the controller side: without a spacing the
    gate must still REFUSE (fail-safe — an unresolved `iso` factor could be a
    real resize whose remainder would be silently dropped)."""

    def _default_json(self, n_channels=1):
        from packages.epicycle_preprocessing import recipe as _r

        return _r.to_json(_r.default_recipe(n_channels))

    def test_default_recipe_streams_at_unit_spacing(self):
        rec = self._default_json(1)
        assert _bare_ctl()._recipe_has_stream_remainder(
            rec, 1, spacing=(1, 1, 1)) is False

    def test_default_recipe_refuses_without_a_spacing(self):
        rec = self._default_json(1)
        assert _bare_ctl()._recipe_has_stream_remainder(
            rec, 1, spacing=None) is True

    def test_spacing_defaults_to_none_so_the_gate_fails_safe(self):
        """The kwarg's default must stay the refusing one: a caller that forgets
        to pass a spacing gets a refusal, never a silent partial process."""
        rec = self._default_json(1)
        assert _bare_ctl()._recipe_has_stream_remainder(rec, 1) is True

    def test_a_real_resize_refuses_even_with_a_spacing(self):
        rec = self._default_json(1)
        assert _bare_ctl()._recipe_has_stream_remainder(
            rec, 1, spacing=(2.0, 0.5, 0.5)) is True

    def test_multichannel_default_streams_at_unit_spacing(self):
        rec = self._default_json(3)
        assert _bare_ctl()._recipe_has_stream_remainder(
            rec, 3, spacing=(1, 1, 1)) is False
