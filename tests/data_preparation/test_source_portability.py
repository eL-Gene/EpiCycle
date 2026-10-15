"""Brick 4a Part 1 — portability of a Step 1 workspace.

Covers the identity + relink machinery that lets a copied workspace (or a
renamed / moved raw file) keep its ④ Library rows attached to the source they
were cut from:

* the ``_loaded_source_id`` session identity (set + cleared exactly where
  ``_loaded_source_file`` is),
* the "every registry row has an id" invariant (legacy back-fill),
* :func:`relink_source` and the controller's automatic / manual relink,
* the prepared-store fallback when the raw file is gone.

Qt-free wherever possible; the controller-level cases reuse the same fake
viewer/widget pattern as ``test_library_registry_view.py``.
"""
from __future__ import annotations

import inspect
import os
import re

import pytest

from napari_epicycle.data_preparation import _state as st_mod
from napari_epicycle.data_preparation._state import (
    _next_source_id,
    _normalize,
    empty_state,
    load_state,
    register_source,
    save_state,
    upsert_source,
)


# ---------------------------------------------------------------------------
# Invariant: every row in sources[] has a source_id
# ---------------------------------------------------------------------------
class TestSourceIdInvariant:
    def test_normalize_backfills_a_legacy_row(self):
        raw = {
            "schema_version": 1,
            "sources": [
                {"path": "C:/data/a.tif", "display_name": "a"},
                {"source_id": "src_3", "path": "C:/data/b.tif"},
                {"path": "C:/data/c.tif"},
            ],
        }
        out = _normalize(raw)
        ids = [s.get("source_id") for s in out["sources"]]
        assert all(ids), f"every row must carry an id, got {ids}"
        assert len(set(ids)) == 3, f"ids must be unique, got {ids}"
        # The pre-existing id is preserved verbatim.
        assert out["sources"][1]["source_id"] == "src_3"
        # New ids never collide with it.
        for sid in ids:
            assert re.match(r"^src_\d+$", sid), sid

    def test_backfill_is_idempotent(self):
        raw = {"sources": [{"path": "p"}]}
        first = _normalize(raw)["sources"][0]["source_id"]
        second = _normalize({"sources": [{"path": "p", "source_id": first}]})
        assert second["sources"][0]["source_id"] == first

    def test_load_state_backfills_on_disk_rows(self, tmp_path):
        ws = str(tmp_path)
        import json
        p = os.path.join(ws, "step1_state.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"schema_version": 1,
                       "sources": [{"path": "x.tif"}],
                       "patches": []}, f)
        state = load_state(ws)
        assert state["sources"][0]["source_id"]

    def test_resolve_source_id_never_mints_a_second_id(self, tmp_path):
        """After the back-fill a path-matched legacy row keeps ONE id."""
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController,
        )
        state = _normalize({"sources": [{"path": "C:/data/a.tif"}]})
        sid = state["sources"][0]["source_id"]
        got = DataPreparationController._resolve_source_id_for(
            None, state, "C:/data/a.tif")
        assert got == sid


# ---------------------------------------------------------------------------
# relink_source — rewrites path + fingerprint, preserves source_id
# ---------------------------------------------------------------------------
class TestRelinkSource:
    def test_rewrites_path_and_fingerprint_keeps_id(self, tmp_path):
        new = tmp_path / "moved.tif"
        new.write_bytes(b"0123456789")
        state = empty_state()
        sid = register_source(state, path="C:/old/orig.tif", display_name="orig",
                              axes="ZYX")
        state["sources"][0]["fingerprint"] = {"size": 1, "mtime": 1.0}
        state["sources"][0]["prepared_store"] = "artifacts/step1/largedata/o.zarr"

        assert st_mod.relink_source(state, sid, str(new)) is True
        row = state["sources"][0]
        assert row["source_id"] == sid
        assert row["path"] == str(new)
        assert row["fingerprint"]["size"] == 10
        # Everything else survives.
        assert row["display_name"] == "orig"
        assert row["prepared_store"] == "artifacts/step1/largedata/o.zarr"
        assert len(state["sources"]) == 1

    def test_unknown_id_is_a_no_op(self):
        state = empty_state()
        assert st_mod.relink_source(state, "src_9", "whatever.tif") is False

    def test_upsert_source_still_does_not_rewrite_a_path(self, tmp_path):
        """Regression lock: `upsert_source` must stay path-additive only."""
        state = empty_state()
        sid = upsert_source(state, path="C:/old/orig.tif", display_name="orig")
        upsert_source(state, source_id=sid, path="C:/new/orig.tif")
        assert state["sources"][0]["path"] == "C:/old/orig.tif"


# ---------------------------------------------------------------------------
# Session identity: `_loaded_source_id` moves with `_loaded_source_file`
# ---------------------------------------------------------------------------
def _assigning_functions(src, attr):
    """Names of the `def`s that contain a `self.<attr> = ` assignment."""
    out = set()
    current = None
    for line in src.splitlines():
        m = re.match(r"\s*def (\w+)\(", line)
        if m:
            current = m.group(1)
        if re.search(rf"self\.{attr}\s*=\s*", line):
            out.add(current)
    return out


class TestLoadedSourceIdSetEquality:
    def test_id_is_set_and_cleared_exactly_where_the_path_is(self):
        from napari_epicycle.data_preparation import controller as ctrl_mod

        src = inspect.getsource(ctrl_mod)
        paths = _assigning_functions(src, "_loaded_source_file")
        ids = _assigning_functions(src, "_loaded_source_id")
        assert paths == ids, (
            "`_loaded_source_id` must be set/cleared in exactly the same "
            f"functions as `_loaded_source_file` — path-only: {paths - ids}, "
            f"id-only: {ids - paths}"
        )
        # Sanity: the scan found SOMETHING (a silently-empty scan would make
        # the set-equality above vacuous).  The exact count is deliberately not
        # asserted -- a new legitimate load path would only have to set both
        # attributes, which is precisely what the equality above checks.
        assert paths, "no assignment sites found - the scan is broken"


# ---------------------------------------------------------------------------
# Journey (b): the raw file is gone, the prepared store carries the source
# ---------------------------------------------------------------------------
class _Text:
    def __init__(self, v=""):
        self._v = v

    def text(self):
        return self._v


def _stub_controller(ws, display=""):
    """The minimal `self` `_fresh_prepared_store_for` actually touches."""
    import types

    from napari_epicycle.data_preparation.controller import (
        DataPreparationController as C,
    )

    w = types.SimpleNamespace(
        output_file_path=_Text(ws), display_name_edit=_Text(display))
    stub = types.SimpleNamespace(widget=w, _log=lambda *a, **k: None)
    # The handful of bound helpers the probed methods reach for.
    stub._find_source_row = C._find_source_row
    stub._tiff_level0_shape = C._tiff_level0_shape
    stub._tiff_level0_shape_axes = C._tiff_level0_shape_axes
    stub._zyx_from_shape = C._zyx_from_shape
    stub._zyx_for_file = C._zyx_for_file
    stub._row_is_bare = C._row_is_bare
    stub._carry_stub_fields = C._carry_stub_fields
    stub._row_zyx = lambda w_, r: C._row_zyx(stub, w_, r)
    stub._shapes_agree = C._shapes_agree
    stub._store_level0_shape = lambda w_, r: C._store_level0_shape(stub, w_, r)
    stub._row_claims_store = lambda r, w_, sa: C._row_claims_store(stub, r, w_, sa)
    stub._largedata_dir = lambda w_: os.path.join(
        w_, "artifacts", "step1", "largedata")
    return stub


def _make_store(ws, name, prov):
    """A directory that stands in for a prepared store; `read_store_source` is
    monkeypatched to return *prov* for it."""
    d = os.path.join(ws, "artifacts", "step1", "largedata", name)
    os.makedirs(d, exist_ok=True)
    return d


class TestStoreFallbackWhenRawIsMissing:
    def _setup(self, tmp_path, monkeypatch, *, pointer):
        from napari_epicycle.data_preparation import controller as ctrl_mod

        ws = str(tmp_path / "ws")
        os.makedirs(ws, exist_ok=True)
        raw = str(tmp_path / "gone.tif")          # deliberately never created
        store = _make_store(ws, "gone.zarr", None)
        prov = {"path": raw, "source_mtime": 123.0, "source_size": 4096}
        monkeypatch.setattr(ctrl_mod, "read_store_source", lambda p: dict(prov))

        state = empty_state()
        sid = register_source(state, path=raw, display_name="gone", axes="ZYX")
        if pointer:
            state["sources"][0]["prepared_store"] = (
                "artifacts/step1/largedata/gone.zarr")
        save_state(ws, state)
        return ctrl_mod, ws, raw, store, sid

    def test_pointer_bound_store_is_served_when_the_raw_is_gone(
            self, tmp_path, monkeypatch):
        ctrl_mod, ws, raw, store, _sid = self._setup(
            tmp_path, monkeypatch, pointer=True)
        fn = ctrl_mod.DataPreparationController._fresh_prepared_store_for
        got = fn(_stub_controller(ws, "gone"), raw, allow_missing_source=True)
        assert got == store

    def test_name_reconstructed_store_is_REFUSED_when_the_raw_is_gone(
            self, tmp_path, monkeypatch):
        """"Same name, different data" must never be served as this source."""
        ctrl_mod, ws, raw, store, _sid = self._setup(
            tmp_path, monkeypatch, pointer=False)
        fn = ctrl_mod.DataPreparationController._fresh_prepared_store_for
        got = fn(_stub_controller(ws, "gone"), raw, allow_missing_source=True)
        assert got is None

    def test_default_still_refuses_a_missing_raw(self, tmp_path, monkeypatch):
        ctrl_mod, ws, raw, store, _sid = self._setup(
            tmp_path, monkeypatch, pointer=True)
        fn = ctrl_mod.DataPreparationController._fresh_prepared_store_for
        assert fn(_stub_controller(ws, "gone"), raw) is None

    def test_reasons_out_param_reports_the_missing_raw(
            self, tmp_path, monkeypatch):
        ctrl_mod, ws, raw, store, _sid = self._setup(
            tmp_path, monkeypatch, pointer=True)
        fn = ctrl_mod.DataPreparationController._fresh_prepared_store_for
        reasons = []
        fn(_stub_controller(ws, "gone"), raw,
           allow_missing_source=True, reasons=reasons)
        assert "raw_missing" in reasons


class TestRawMissingUiGates:
    """Prepare + Process are off while the raw is missing; Extract/Trim are not."""

    def test_gates_flip_with_the_flag(self, tmp_path):
        pytest.importorskip("qtpy")
        from qtpy.QtWidgets import QApplication
        from napari_epicycle.data_preparation.ui import DataPreparationWidget

        app = QApplication.instance() or QApplication([])
        assert app is not None

        class _FakeEvents:
            def __init__(self):
                class _E:
                    def connect(self, *a, **k):
                        pass
                self.inserted = _E()
                self.removed = _E()

        class _FakeLayers(list):
            def __init__(self):
                super().__init__()
                self.events = _FakeEvents()

        class _FakeViewer:
            def __init__(self):
                self.layers = _FakeLayers()
                self.mouse_drag_callbacks = []

        w = DataPreparationWidget(viewer=_FakeViewer())
        try:
            import types
            w.controller = types.SimpleNamespace(_raw_missing=False,
                                                 _step1_busy=False)
            w._apply_process_button_gate(True)
            assert w.apply_preprocess_button.isEnabled() is True

            w.controller._raw_missing = True
            w.refresh_raw_missing_gates()
            assert w.apply_preprocess_button.isEnabled() is False
            assert w.prepare_largedata_button.isEnabled() is False
            # Extract + Trim are deliberately untouched by this gate.
            assert not hasattr(w, "_extract_disabled_by_raw_missing")

            w.controller._raw_missing = False
            w.refresh_raw_missing_gates()
            assert w.apply_preprocess_button.isEnabled() is True
        finally:
            w.setParent(None)


class TestSidecarSourceFrom:
    def test_source_from_is_written_only_when_given(self, tmp_path):
        import json

        import numpy as np

        from napari_epicycle.data_preparation._export import persist_patch

        ws = str(tmp_path)
        data = np.zeros((1, 2, 3, 4), dtype="uint16")
        common = dict(
            channel_names=["c0"], original_shape=data.shape,
            cropped_shape=data.shape, selected_z_range=[0, 2],
            bounding_box="bb", source_path=str(tmp_path / "s.tif"),
        )
        plain = persist_patch(ws, "P", data, **common)
        with open(plain["sidecar_path"], encoding="utf-8") as f:
            assert "source_from" not in json.load(f)

        stored = persist_patch(ws, "Q", data,
                               source_from="prepared_store", **common)
        with open(stored["sidecar_path"], encoding="utf-8") as f:
            assert json.load(f)["source_from"] == "prepared_store"


# ---------------------------------------------------------------------------
# Journey (a): workspace AND data copied — the row follows the file
# ---------------------------------------------------------------------------
def _write_tiff(path, shape=(4, 8, 8)):
    import numpy as np
    import tifffile

    tifffile.imwrite(str(path), np.zeros(shape, dtype="uint16"))
    return str(path)


def _seed_prepared_workspace(tmp_path, raw_name="orig.tif",
                             store_shape=(1, 4, 8, 8)):
    """A workspace whose single source row points at a raw file + a prepared
    store (a stub directory carrying a real ``0/.zarray``)."""
    import json

    ws = str(tmp_path / "ws")
    os.makedirs(ws, exist_ok=True)
    raw = _write_tiff(tmp_path / raw_name)
    store = os.path.join(ws, "artifacts", "step1", "largedata", "orig.zarr")
    os.makedirs(os.path.join(store, "0"), exist_ok=True)
    with open(os.path.join(store, "0", ".zarray"), "w", encoding="utf-8") as f:
        json.dump({"shape": list(store_shape)}, f)

    state = empty_state()
    sid = register_source(state, path=raw, display_name="orig", axes="ZYX")
    row = state["sources"][0]
    row["prepared_store"] = "artifacts/step1/largedata/orig.zarr"
    row["fingerprint"] = {"size": os.path.getsize(raw),
                          "mtime": os.path.getmtime(raw)}
    save_state(ws, state)
    return ws, raw, store, sid


class TestAutoRelinkAtLoad:
    def _ctrl(self, ws):
        return _stub_controller(ws, "orig")

    def test_relinks_a_moved_file_and_keeps_the_id(self, tmp_path):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        ws, raw, store, sid = _seed_prepared_workspace(tmp_path)
        # Simulate the copy: the raw moves to a NEW folder, the old one is gone.
        newdir = tmp_path / "elsewhere"
        newdir.mkdir()
        moved = str(newdir / "orig.tif")
        os.replace(raw, moved)

        state = load_state(ws)
        got = C._relink_row_for(self._ctrl(ws), state, moved, ws)
        assert got == sid
        assert len(state["sources"]) == 1
        assert state["sources"][0]["path"] == moved
        assert state["sources"][0]["source_id"] == sid

    def test_a_live_row_is_never_a_candidate(self, tmp_path):
        """Two files that both exist must never be merged into one row."""
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        ws, raw, store, sid = _seed_prepared_workspace(tmp_path)
        twin = _write_tiff(tmp_path / "twin.tif")     # same size + shape
        state = load_state(ws)
        assert C._relink_row_for(self._ctrl(ws), state, twin, ws) is None
        assert state["sources"][0]["path"] == raw

    def test_a_different_shape_is_refused(self, tmp_path):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        ws, raw, store, sid = _seed_prepared_workspace(
            tmp_path, store_shape=(1, 9, 9, 9))
        moved = str(tmp_path / "moved.tif")
        os.replace(raw, moved)
        state = load_state(ws)
        assert C._relink_row_for(self._ctrl(ws), state, moved, ws) is None

    def test_an_already_known_path_is_a_no_op(self, tmp_path):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        ws, raw, store, sid = _seed_prepared_workspace(tmp_path)
        state = load_state(ws)
        assert C._relink_row_for(self._ctrl(ws), state, raw, ws) is None

    def test_shapes_agree_is_axis_order_agnostic(self):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        # ZCYX vs CZYX, and a 3D single-channel TIFF vs its 4D store level 0.
        assert C._shapes_agree((4, 1, 8, 8), (1, 4, 8, 8)) is True
        assert C._shapes_agree((4, 8, 8), (1, 4, 8, 8)) is True
        assert C._shapes_agree((2, 4, 8, 8), (1, 4, 8, 8)) is False
        assert C._shapes_agree(None, (1, 2)) is False


class TestReconcileSkipsClaimedStores:
    def test_no_duplicate_row_after_a_relink(self, tmp_path, monkeypatch):
        """The store's `.zattrs` still names the OLD path; reconcile must NOT
        synthesize a second row for it."""
        from napari_epicycle.data_preparation import controller as ctrl_mod
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        import types

        ws, raw, store, sid = _seed_prepared_workspace(tmp_path)
        # A move KEEPS the filename -- that is what the silent path requires
        # (Brick 4c item 1); a rename goes through "Find the original file...".
        newdir = tmp_path / "moved_here"
        newdir.mkdir()
        moved = str(newdir / "orig.tif")
        os.replace(raw, moved)
        state = load_state(ws)
        assert C._relink_row_for(_stub_controller(ws), state, moved, ws) == sid

        # The store keeps recording the ORIGINAL (now absent) path.
        monkeypatch.setattr(
            ctrl_mod, "read_store_source",
            lambda p: {"path": raw, "source_mtime": 1.0, "source_size": 1})

        C._reconcile_sources_with_stores(_stub_controller(ws), ws, state)
        assert len(state["sources"]) == 1, state["sources"]
        assert state["sources"][0]["source_id"] == sid
        assert state["sources"][0]["path"] == moved


# ---------------------------------------------------------------------------
# "Find the original file…" — the manual relink
# ---------------------------------------------------------------------------
class TestLocateDecisionMatrix:
    """size+shape / shape only / size only / neither -> relink, relink,
    confirm, refuse.

    Both shapes are the axis-resolved ``(Z, Y, X)`` since Brick 4c item 4, so
    the compare is EXACT -- a transpose is not a match."""

    def _d(self, *a):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        return C._locate_decision(*a)

    def test_size_and_shape_match(self):
        assert self._d(100, (4, 8, 8), 100, (4, 8, 8)) == "relink"

    def test_shape_matches_size_differs(self):
        """A re-export of the same stack — silent relink."""
        assert self._d(120, (4, 8, 8), 100, (4, 8, 8)) == "relink"

    def test_size_matches_shape_differs(self):
        assert self._d(100, (9, 9, 9), 100, (4, 8, 8)) == "confirm"

    def test_a_transpose_is_not_a_match(self):
        assert self._d(100, (8, 4, 8), 100, (4, 8, 8)) == "confirm"

    def test_neither_matches(self):
        assert self._d(120, (9, 9, 9), 100, (4, 8, 8)) == "refuse"

    def test_unknown_shape_falls_back_to_the_size(self):
        assert self._d(100, None, 100, (4, 8, 8)) == "confirm"
        assert self._d(120, None, 100, (4, 8, 8)) == "refuse"


class TestLocateSourceAction:
    def _widget_stub(self, ws, tmp_path):
        import types
        return types.SimpleNamespace(
            output_file_path=_Text(ws), display_name_edit=_Text("orig"),
            source_status_label=types.SimpleNamespace(
                setText=lambda *a: None, setStyleSheet=lambda *a: None),
        )

    def _ctrl_for(self, ws, tmp_path, picked, answer=True):
        """A controller stub wired to the pure locate helper."""
        import types
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        c = _stub_controller(ws, "orig")
        c.widget = self._widget_stub(ws, tmp_path)
        c._set_source_status_amber = lambda t: setattr(c, "status", t)
        c._set_raw_missing = lambda m: setattr(c, "raw_missing", m)
        c._loaded_source_id = None
        c._loaded_source_file = None
        c._pick_source_file = lambda: picked
        c._confirm_shape_mismatch = lambda *a, **k: answer
        c._refresh_after_relink = lambda *a, **k: None
        c._locate_decision = C._locate_decision
        c._loaded_source_row = lambda st: C._loaded_source_row(c, st)
        return C.locate_source_for_row.__get__(c, C)

    def test_a_matching_file_relinks_silently(self, tmp_path):
        ws, raw, store, sid = _seed_prepared_workspace(tmp_path)
        moved = str(tmp_path / "moved.tif")
        os.replace(raw, moved)
        fn = self._ctrl_for(ws, tmp_path, moved)
        assert fn(sid) is True
        assert load_state(ws)["sources"][0]["path"] == moved

    def test_cancel_leaves_the_row_completely_unchanged(self, tmp_path):
        ws, raw, store, sid = _seed_prepared_workspace(tmp_path)
        os.replace(raw, str(tmp_path / "gone.tif"))
        other = _write_tiff(tmp_path / "other.tif", shape=(9, 9, 9))
        # Force the "size matches, shape differs" branch, then decline.
        st = load_state(ws)
        st["sources"][0]["fingerprint"]["size"] = os.path.getsize(other)
        save_state(ws, st)
        before = load_state(ws)["sources"][0]["path"]

        fn = self._ctrl_for(ws, tmp_path, other, answer=False)
        assert fn(sid) is False
        assert load_state(ws)["sources"][0]["path"] == before

    def test_a_totally_different_file_is_refused(self, tmp_path):
        ws, raw, store, sid = _seed_prepared_workspace(tmp_path)
        os.replace(raw, str(tmp_path / "gone.tif"))
        other = _write_tiff(tmp_path / "other.tif", shape=(9, 9, 9))
        before = load_state(ws)["sources"][0]["path"]
        fn = self._ctrl_for(ws, tmp_path, other)
        assert fn(sid) is False
        assert load_state(ws)["sources"][0]["path"] == before


# ---------------------------------------------------------------------------
# a3 — freshness drift is an amber LINE, never a modal
# ---------------------------------------------------------------------------
class TestFreshnessAmber:
    def _setup(self, tmp_path, monkeypatch):
        from napari_epicycle.data_preparation import controller as ctrl_mod

        ws = str(tmp_path / "ws")
        os.makedirs(ws, exist_ok=True)
        raw = _write_tiff(tmp_path / "v.tif")
        store = _make_store(ws, "v.zarr", None)
        from packages.epicycle_largedata.multiscale import sampled_file_sha

        prov = {"path": raw,
                "source_mtime": os.path.getmtime(raw),
                "source_size": os.path.getsize(raw),
                # Brick 4c item 2: Prepare records a sampled content digest, so
                # "the date moved but the bytes did not" is decidable.
                "source_head_sha": sampled_file_sha(raw)}
        monkeypatch.setattr(ctrl_mod, "read_store_source", lambda p: dict(prov))
        state = empty_state()
        register_source(state, path=raw, display_name="v", axes="ZYX")
        state["sources"][0]["prepared_store"] = "artifacts/step1/largedata/v.zarr"
        save_state(ws, state)
        return ctrl_mod, ws, raw, store

    def test_mtime_only_drift_keeps_serving_the_store(
            self, tmp_path, monkeypatch):
        """A copy without -p / a cloud sync moves the mtime but not a byte."""
        ctrl_mod, ws, raw, store = self._setup(tmp_path, monkeypatch)
        fn = ctrl_mod.DataPreparationController._fresh_prepared_store_for
        now = os.path.getmtime(raw)
        os.utime(raw, (now + 5000, now + 5000))

        reasons = []
        got = fn(_stub_controller(ws, "v"), raw, reasons=reasons)
        assert got == store
        assert reasons == ["mtime_drift"]

    def test_size_drift_falls_back_to_the_raw(self, tmp_path, monkeypatch):
        ctrl_mod, ws, raw, store = self._setup(tmp_path, monkeypatch)
        fn = ctrl_mod.DataPreparationController._fresh_prepared_store_for
        with open(raw, "ab") as f:
            f.write(b"\x00" * 16)

        reasons = []
        assert fn(_stub_controller(ws, "v"), raw, reasons=reasons) is None
        assert "size_drift" in reasons

    def test_notes_are_plain_language_and_never_a_dialog(self):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        note = C._freshness_note(["size_drift"])
        assert note and "changed" in note.lower()
        assert C._freshness_note(["mtime_drift"])
        assert C._freshness_note([]) is None
        for r in (["size_drift"], ["mtime_drift"]):
            txt = C._freshness_note(r).lower()
            for banned in ("fingerprint", "pointer", "mtime", "provenance",
                           "invalidate"):
                assert banned not in txt, (banned, txt)


# ---------------------------------------------------------------------------
# Brick 4c item 1 — the SILENT auto-relink must not bind the wrong file
# ---------------------------------------------------------------------------
class TestAutoRelinkIdentityIsStrict:
    """Size + dimensions alone are NOT identity.

    Two acquisitions off the same microscope share both, so the silent path
    additionally requires the FILENAME to match (a move or a copy keeps it; a
    rename goes through "Find the original file…", which asks) and refuses when
    more than one row could claim the file."""

    def _ctrl(self, ws):
        return _stub_controller(ws, "orig")

    def test_same_dimensions_different_file_is_not_relinked(self, tmp_path):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        ws, raw, store, sid = _seed_prepared_workspace(tmp_path)
        os.replace(raw, str(tmp_path / "archived.tif"))
        # A DIFFERENT acquisition: identical size + dimensions, other name.
        other = _write_tiff(tmp_path / "tuesday.tif")
        state = load_state(ws)
        assert C._relink_row_for(self._ctrl(ws), state, other, ws) is None
        assert state["sources"][0]["path"] == raw

    def test_two_candidate_rows_relink_neither(self, tmp_path):
        import json

        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        ws, raw, store, sid = _seed_prepared_workspace(tmp_path)
        # A second row for a same-named file in another (now missing) folder.
        second_dir = tmp_path / "other"
        second_dir.mkdir()
        second = _write_tiff(second_dir / "orig.tif")
        state = load_state(ws)
        sid2 = register_source(state, path=second, display_name="orig2",
                               axes="ZYX")
        store2 = os.path.join(ws, "artifacts", "step1", "largedata", "o2.zarr")
        os.makedirs(os.path.join(store2, "0"), exist_ok=True)
        with open(os.path.join(store2, "0", ".zarray"), "w",
                  encoding="utf-8") as f:
            json.dump({"shape": [1, 4, 8, 8]}, f)
        row2 = state["sources"][-1]
        row2["prepared_store"] = "artifacts/step1/largedata/o2.zarr"
        row2["fingerprint"] = {"size": os.path.getsize(second), "mtime": 1.0}
        save_state(ws, state)

        moved_dir = tmp_path / "moved"
        moved_dir.mkdir()
        picked = str(moved_dir / "orig.tif")
        os.replace(raw, picked)
        os.remove(second)                    # both rows now name missing files

        state = load_state(ws)
        assert C._relink_row_for(self._ctrl(ws), state, picked, ws) is None
        assert {r["path"] for r in state["sources"]} == {raw, second}

    def test_a_move_that_keeps_the_name_still_relinks(self, tmp_path):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        ws, raw, store, sid = _seed_prepared_workspace(tmp_path)
        newdir = tmp_path / "drive_e"
        newdir.mkdir()
        moved = str(newdir / "orig.tif")
        os.replace(raw, moved)
        state = load_state(ws)
        assert C._relink_row_for(self._ctrl(ws), state, moved, ws) == sid
        assert state["sources"][0]["path"] == moved

    def test_zyx_is_compared_axis_resolved_not_as_a_multiset(self):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        # CZYX store level 0 vs a 3D ZYX raw -> the same (Z, Y, X).
        assert C._zyx_from_shape((1, 4, 8, 8), "CZYX") == (4, 8, 8)
        assert C._zyx_from_shape((4, 8, 8), "ZYX") == (4, 8, 8)
        # ZCYX raw: the channel axis is 1, so Z stays 4.
        assert C._zyx_from_shape((4, 2, 8, 8), "ZCYX") == (4, 8, 8)
        # A transposed stack is NOT the same stack, even though the multiset is.
        assert C._zyx_from_shape((8, 4, 8), "ZYX") != (4, 8, 8)
        assert C._zyx_from_shape(None, "ZYX") is None


# ---------------------------------------------------------------------------
# Brick 4c item 2 — an in-place pixel edit keeps the byte size
# ---------------------------------------------------------------------------
class TestSampledContentDigest:
    """`{size, mtime}` cannot see an ImageJ "Save" back over the original.

    Prepare records a sampled digest; a size match with a moved mtime is decided
    on that, so "new date, same contents" is a true statement rather than a
    guess."""

    def _setup(self, tmp_path, monkeypatch, *, with_hash=True):
        from napari_epicycle.data_preparation import controller as ctrl_mod
        from packages.epicycle_largedata.multiscale import sampled_file_sha

        ws = str(tmp_path / "ws")
        os.makedirs(ws, exist_ok=True)
        raw = _write_tiff(tmp_path / "v.tif", shape=(6, 32, 32))
        store = _make_store(ws, "v.zarr", None)
        prov = {"path": raw,
                "source_mtime": os.path.getmtime(raw),
                "source_size": os.path.getsize(raw)}
        if with_hash:
            prov["source_head_sha"] = sampled_file_sha(raw)
        monkeypatch.setattr(ctrl_mod, "read_store_source", lambda p: dict(prov))
        state = empty_state()
        register_source(state, path=raw, display_name="v", axes="ZYX")
        state["sources"][0]["prepared_store"] = "artifacts/step1/largedata/v.zarr"
        save_state(ws, state)
        return ctrl_mod, ws, raw, store

    @staticmethod
    def _edit_in_place(raw):
        """Rewrite pixel bytes WITHOUT changing the file's length."""
        size = os.path.getsize(raw)
        with open(raw, "r+b") as f:
            f.seek(size - 64)
            f.write(b"\xff" * 64)
        assert os.path.getsize(raw) == size
        now = os.path.getmtime(raw)
        os.utime(raw, (now + 5000, now + 5000))

    def test_an_in_place_edit_falls_back_to_the_raw(self, tmp_path,
                                                    monkeypatch):
        ctrl_mod, ws, raw, store = self._setup(tmp_path, monkeypatch)
        fn = ctrl_mod.DataPreparationController._fresh_prepared_store_for
        self._edit_in_place(raw)
        reasons = []
        assert fn(_stub_controller(ws, "v"), raw, reasons=reasons) is None
        assert reasons == ["size_drift"], reasons

    def test_a_touch_only_change_still_serves_the_store(self, tmp_path,
                                                        monkeypatch):
        ctrl_mod, ws, raw, store = self._setup(tmp_path, monkeypatch)
        fn = ctrl_mod.DataPreparationController._fresh_prepared_store_for
        now = os.path.getmtime(raw)
        os.utime(raw, (now + 5000, now + 5000))
        reasons = []
        assert fn(_stub_controller(ws, "v"), raw, reasons=reasons) == store
        assert reasons == ["mtime_drift"]

    def test_a_legacy_store_without_a_digest_is_conservative(self, tmp_path,
                                                             monkeypatch):
        """No recorded digest -> we cannot tell a copy from an edit, so fall
        back to the raw (the pre-4a behaviour)."""
        ctrl_mod, ws, raw, store = self._setup(tmp_path, monkeypatch,
                                               with_hash=False)
        fn = ctrl_mod.DataPreparationController._fresh_prepared_store_for
        now = os.path.getmtime(raw)
        os.utime(raw, (now + 5000, now + 5000))
        reasons = []
        assert fn(_stub_controller(ws, "v"), raw, reasons=reasons) is None
        # Reported as UNVERIFIABLE, not as a change: we fall back to the raw,
        # but we never observed the file change (Brick 4e item 7).
        assert reasons == ["unverifiable"]

    def test_sampled_sha_is_stable_and_size_sensitive(self, tmp_path):
        from packages.epicycle_largedata.multiscale import sampled_file_sha

        a = tmp_path / "a.bin"
        a.write_bytes(b"A" * 4096)
        first = sampled_file_sha(str(a))
        assert first and first == sampled_file_sha(str(a))
        b = tmp_path / "b.bin"
        b.write_bytes(b"A" * 4097)
        assert sampled_file_sha(str(b)) != first
        assert sampled_file_sha(str(tmp_path / "nope.bin")) is None


# ---------------------------------------------------------------------------
# Brick 4c item 4 — a source that was never Prepared
# ---------------------------------------------------------------------------
def _seed_unprepared_workspace(tmp_path, raw_name="orig.tif"):
    """A workspace with a source row + patches but NO prepared store."""
    ws = str(tmp_path / "ws")
    os.makedirs(ws, exist_ok=True)
    raw = _write_tiff(tmp_path / raw_name)
    state = empty_state()
    sid = register_source(state, path=raw, display_name="orig", axes="ZYX")
    row = state["sources"][0]
    row["fingerprint"] = {"size": os.path.getsize(raw),
                          "mtime": os.path.getmtime(raw)}
    row["source_shape"] = [4, 8, 8]
    save_state(ws, state)
    return ws, raw, sid


class TestNeverPreparedSource:
    def test_row_zyx_falls_back_to_the_recorded_shape(self, tmp_path):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        ws, raw, sid = _seed_unprepared_workspace(tmp_path)
        row = load_state(ws)["sources"][0]
        assert C._row_zyx(_stub_controller(ws), ws, row) == (4, 8, 8)

    def test_auto_relink_works_without_a_store(self, tmp_path):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        ws, raw, sid = _seed_unprepared_workspace(tmp_path)
        newdir = tmp_path / "moved"
        newdir.mkdir()
        moved = str(newdir / "orig.tif")
        os.replace(raw, moved)
        state = load_state(ws)
        assert C._relink_row_for(_stub_controller(ws), state, moved, ws) == sid
        assert state["sources"][0]["path"] == moved

    def test_a_different_shape_is_still_refused_without_a_store(self, tmp_path):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        ws, raw, sid = _seed_unprepared_workspace(tmp_path)
        state = load_state(ws)
        state["sources"][0]["source_shape"] = [9, 9, 9]
        save_state(ws, state)
        newdir = tmp_path / "moved"
        newdir.mkdir()
        moved = str(newdir / "orig.tif")
        os.replace(raw, moved)
        state = load_state(ws)
        assert C._relink_row_for(_stub_controller(ws), state, moved, ws) is None

    def test_locate_can_relink_a_never_prepared_row(self, tmp_path):
        import types

        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        ws, raw, sid = _seed_unprepared_workspace(tmp_path)
        renamed = str(tmp_path / "monday_run.tif")
        os.replace(raw, renamed)

        c = _stub_controller(ws, "orig")
        c.widget = types.SimpleNamespace(
            output_file_path=_Text(ws), display_name_edit=_Text("orig"),
            source_status_label=types.SimpleNamespace(
                setText=lambda *a: None, setStyleSheet=lambda *a: None))
        c._set_source_status_amber = lambda t: None
        c._set_raw_missing = lambda m: None
        c._refresh_after_relink = lambda *a, **k: None
        c._pick_source_file = lambda: renamed
        c._confirm_shape_mismatch = lambda *a, **k: False
        c._locate_decision = C._locate_decision
        c._loaded_source_row = lambda st: C._loaded_source_row(c, st)
        c._loaded_source_id = None
        c._loaded_source_file = None

        assert C.locate_source_for_row.__get__(c, C)(sid) is True
        assert load_state(ws)["sources"][0]["path"] == renamed

    def test_the_modal_never_prints_the_word_None(self):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        assert C._shape_label(None) == "unknown"
        assert C._shape_label(()) == "unknown"
        assert C._shape_label((4, 8, 8)) == "4 × 8 × 8"
        assert C._shape_label(("x",)) == "unknown"


class TestUnloadClearsRawMissing:
    """Brick 4c item 5: Prepare / Process stayed greyed after unloading a
    source that was being served from its prepared copy."""

    def test_unload_resets_the_flag_and_the_gates(self):
        import inspect

        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        src = inspect.getsource(C._unload_source_inner)
        assert "_set_raw_missing(False)" in src, (
            "the unload path must clear the raw-missing state")

    def test_the_gates_come_back(self):
        pytest.importorskip("qtpy")
        import types

        from qtpy.QtWidgets import QApplication

        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        from napari_epicycle.data_preparation.ui import DataPreparationWidget

        app = QApplication.instance() or QApplication([])
        assert app is not None

        class _FakeEvents:
            def __init__(self):
                class _E:
                    def connect(self, *a, **k):
                        pass
                self.inserted = _E()
                self.removed = _E()

        class _FakeLayers(list):
            def __init__(self):
                super().__init__()
                self.events = _FakeEvents()

        class _FakeViewer:
            def __init__(self):
                self.layers = _FakeLayers()
                self.mouse_drag_callbacks = []

        w = DataPreparationWidget(viewer=_FakeViewer())
        try:
            ctl = types.SimpleNamespace(_raw_missing=True, _step1_busy=False,
                                        widget=w)
            w.controller = ctl
            w._apply_process_button_gate(True)
            assert w.apply_preprocess_button.isEnabled() is False

            C._set_raw_missing(ctl, False)          # what unload now calls
            assert ctl._raw_missing is False
            assert w.apply_preprocess_button.isEnabled() is True
        finally:
            w.setParent(None)


class TestRelinkSurvivesAnEditBeforeTheFirstLoad:
    """Brick 4c item 6 — journey (a) ordering.

    On the new machine the user often types an alias (or a voxel size) BEFORE
    pressing Load.  That edit is saved with `upsert_source(path=…)`, which
    creates a row for the NEW path, and the relink's "this path already has a
    row" bail then left the real row — store, patches and all — orphaned."""

    def test_a_bare_stub_row_is_folded_into_the_relink(self, tmp_path):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        ws, raw, store, sid = _seed_prepared_workspace(tmp_path)
        newdir = tmp_path / "pc2"
        newdir.mkdir()
        moved = str(newdir / "orig.tif")
        os.replace(raw, moved)

        # The alias edit lands first, creating a stub for the NEW path.
        state = load_state(ws)
        upsert_source(state, path=moved, display_name="wound_d3")
        save_state(ws, state)
        assert len(load_state(ws)["sources"]) == 2

        state = load_state(ws)
        assert C._relink_row_for(_stub_controller(ws), state, moved, ws) == sid
        assert len(state["sources"]) == 1, state["sources"]
        row = state["sources"][0]
        assert row["source_id"] == sid          # patches stay attached
        assert row["path"] == moved
        assert row["display_name"] == "wound_d3"   # the edit is carried over
        assert row["prepared_store"] == "artifacts/step1/largedata/orig.zarr"

    def test_a_row_with_real_work_behind_it_still_bails(self, tmp_path):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        from napari_epicycle.data_preparation._state import register_patch

        ws, raw, store, sid = _seed_prepared_workspace(tmp_path)
        state = load_state(ws)
        # `raw` still exists and its row has a store -> not a stub.
        assert C._row_is_bare(state, state["sources"][0]) is False
        assert C._relink_row_for(_stub_controller(ws), state, raw, ws) is None

        # A row with no store but a patch is not bare either.
        sid2 = upsert_source(state, path=str(tmp_path / "b.tif"),
                             display_name="b")
        register_patch(state, patch_name="B_1",
                       relative_path="artifacts/step1/patches/B_1.tiff",
                       sidecar_path="artifacts/step1/patches/B_1_metadata.json",
                       source_id=sid2)
        row2 = next(r for r in state["sources"] if r["source_id"] == sid2)
        assert C._row_is_bare(state, row2) is False


class TestRelinkWritesARelativePointer:
    """Brick 4c item 7 — the reconcile pass NULLs an absolute pointer whose
    store records a different source path, which is precisely what a relink
    creates.  A workspace-relative pointer never enters that branch."""

    def test_an_absolute_pointer_is_rewritten_relative(self, tmp_path):
        ws = str(tmp_path / "ws")
        os.makedirs(ws, exist_ok=True)
        store = os.path.join(ws, "artifacts", "step1", "largedata", "o.zarr")
        os.makedirs(store, exist_ok=True)
        new = tmp_path / "moved.tif"
        new.write_bytes(b"0123456789")

        state = empty_state()
        sid = register_source(state, path="C:/old/o.tif", display_name="o",
                              axes="ZYX")
        state["sources"][0]["prepared_store"] = store          # ABSOLUTE

        assert st_mod.relink_source(state, sid, str(new), ws) is True
        assert state["sources"][0]["prepared_store"] == (
            "artifacts/step1/largedata/o.zarr")

    def test_a_pointer_outside_the_workspace_is_left_alone(self, tmp_path):
        ws = str(tmp_path / "ws")
        os.makedirs(ws, exist_ok=True)
        outside = str(tmp_path / "elsewhere" / "o.zarr")
        new = tmp_path / "moved.tif"
        new.write_bytes(b"0123456789")

        state = empty_state()
        sid = register_source(state, path="C:/old/o.tif", display_name="o",
                              axes="ZYX")
        state["sources"][0]["prepared_store"] = outside

        assert st_mod.relink_source(state, sid, str(new), ws) is True
        assert state["sources"][0]["prepared_store"] == outside

    def test_a_relative_pointer_and_no_workspace_are_untouched(self, tmp_path):
        new = tmp_path / "moved.tif"
        new.write_bytes(b"0123456789")
        state = empty_state()
        sid = register_source(state, path="C:/old/o.tif", display_name="o",
                              axes="ZYX")
        state["sources"][0]["prepared_store"] = "artifacts/step1/largedata/o.zarr"
        assert st_mod.relink_source(state, sid, str(new)) is True
        assert state["sources"][0]["prepared_store"] == (
            "artifacts/step1/largedata/o.zarr")


class TestLocateButtonIsGuarded:
    def test_the_button_and_the_menu_share_one_guarded_route(self):
        import inspect

        from napari_epicycle.data_preparation import ui as ui_mod

        src = inspect.getsource(ui_mod.DataPreparationWidget)
        # The raw, unguarded call is gone.
        assert "self.controller.locate_source_for_row()" not in src
        assert "_call_locate_source" in src

    def test_a_controller_without_the_handler_does_not_raise(self):
        import types

        from napari_epicycle.data_preparation.ui import DataPreparationWidget

        stub = types.SimpleNamespace(controller=types.SimpleNamespace())
        DataPreparationWidget._call_locate_source(stub)          # must not raise
        stub.controller = None
        DataPreparationWidget._call_locate_source(stub)


class TestRelinkStatusCopy:
    def test_it_says_what_it_is_now_using(self):
        import inspect

        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        src = inspect.getsource(C._refresh_after_relink)
        assert "Now using " in src
        assert "Re-linked" not in src


# ---------------------------------------------------------------------------
# Brick 4e item 1 — folding a stub must not delete authored work
# ---------------------------------------------------------------------------
class TestStubFoldPreservesWork:
    def _move(self, tmp_path, raw, sub="pc2"):
        d = tmp_path / sub
        d.mkdir()
        moved = str(d / os.path.basename(raw))
        os.replace(raw, moved)
        return moved

    def test_a_stub_holding_a_recipe_is_not_bare(self, tmp_path):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        ws, raw, store, sid = _seed_prepared_workspace(tmp_path)
        moved = self._move(tmp_path, raw)

        state = load_state(ws)
        upsert_source(state, path=moved, display_name="wound",
                      recipe={"channels": [[{"key": "gaussian"}]],
                              "geometry": None})
        stub = state["sources"][-1]
        assert C._row_is_bare(state, stub) is False
        # ...so the relink bails and the recipe survives untouched.
        assert C._relink_row_for(_stub_controller(ws), state, moved, ws) is None
        assert len(state["sources"]) == 2
        assert state["sources"][-1]["recipe"]["channels"] == [
            [{"key": "gaussian"}]]

    def test_an_empty_recipe_still_counts_as_bare(self, tmp_path):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        ws, raw, store, sid = _seed_prepared_workspace(tmp_path)
        moved = self._move(tmp_path, raw)
        state = load_state(ws)
        upsert_source(state, path=moved,
                      recipe={"channels": [[], []], "geometry": None})
        assert C._row_is_bare(state, state["sources"][-1]) is True
        assert C._relink_row_for(_stub_controller(ws), state, moved, ws) == sid

    def test_a_confirmed_axis_order_is_carried_across(self, tmp_path):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        ws, raw, store, sid = _seed_prepared_workspace(tmp_path)
        moved = self._move(tmp_path, raw)

        state = load_state(ws)
        state["sources"][0]["source_axes"] = "CZYX"
        upsert_source(state, path=moved, display_name="wound_d3",
                      source_axes="ZCYX", prefix="wd3")
        assert C._relink_row_for(_stub_controller(ws), state, moved, ws) == sid

        row = state["sources"][0]
        assert len(state["sources"]) == 1
        # The order the user CONFIRMED against the new path wins.
        assert row["source_axes"] == "ZCYX"
        assert row["display_name"] == "wound_d3"
        assert row["prefix"] == "wd3"


# ---------------------------------------------------------------------------
# Brick 4e item 2 — Locate must not leave two rows claiming one file
# ---------------------------------------------------------------------------
class TestLocateHandlesAnExistingRowForThePickedPath:
    def _ctrl(self, ws, picked):
        import types

        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        c = _stub_controller(ws, "orig")
        c.widget = types.SimpleNamespace(
            output_file_path=_Text(ws), display_name_edit=_Text("orig"),
            source_status_label=types.SimpleNamespace(
                setText=lambda *a: None, setStyleSheet=lambda *a: None))
        c.status = None
        c._set_source_status_amber = lambda t: setattr(c, "status", t)
        c._set_raw_missing = lambda m: None
        c._refresh_after_relink = lambda *a, **k: None
        c._pick_source_file = lambda: picked
        c._confirm_shape_mismatch = lambda *a, **k: True
        c._locate_decision = C._locate_decision
        c._loaded_source_row = lambda st: C._loaded_source_row(c, st)
        c._loaded_source_id = None
        c._loaded_source_file = None
        return c, C.locate_source_for_row.__get__(c, C)

    def test_a_bare_stub_for_the_picked_path_is_folded(self, tmp_path):
        ws, raw, store, sid = _seed_prepared_workspace(tmp_path)
        newdir = tmp_path / "pc2"
        newdir.mkdir()
        picked = str(newdir / "orig.tif")
        os.replace(raw, picked)

        # The ambiguous-candidate fall-through left a stub for the new path.
        state = load_state(ws)
        upsert_source(state, path=picked, display_name="wound_d3")
        save_state(ws, state)
        assert len(load_state(ws)["sources"]) == 2

        c, fn = self._ctrl(ws, picked)
        assert fn(sid) is True
        after = load_state(ws)["sources"]
        assert len(after) == 1, after
        assert after[0]["source_id"] == sid
        assert after[0]["path"] == picked
        assert after[0]["display_name"] == "wound_d3"   # carried, not lost

    def test_a_real_row_for_the_picked_path_is_refused(self, tmp_path):
        from napari_epicycle.data_preparation._state import register_patch

        ws, raw, store, sid = _seed_prepared_workspace(tmp_path)
        os.replace(raw, str(tmp_path / "archived.tif"))
        # A DIFFERENT, already-registered source with patches of its own.
        picked = _write_tiff(tmp_path / "other.tif")
        state = load_state(ws)
        sid2 = upsert_source(state, path=picked, display_name="tuesday")
        register_patch(state, patch_name="T_1",
                       relative_path="artifacts/step1/patches/T_1.tiff",
                       sidecar_path="artifacts/step1/patches/T_1_metadata.json",
                       source_id=sid2)
        save_state(ws, state)
        before = [dict(r) for r in load_state(ws)["sources"]]

        c, fn = self._ctrl(ws, picked)
        assert fn(sid) is False
        assert c.status and "already registered as tuesday" in c.status
        assert "choose it in the list" in c.status
        # Nothing moved.
        assert [dict(r) for r in load_state(ws)["sources"]] == before


class TestUnreadableHeaderIsNotAMatch:
    """Brick 4e item 3: a row that KNOWS its dimensions must not be bound to a
    file whose header we could not read.  Silence is not agreement."""

    def test_an_unreadable_file_is_refused_when_the_row_has_a_shape(
            self, tmp_path, monkeypatch):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        ws, raw, store, sid = _seed_prepared_workspace(tmp_path)
        newdir = tmp_path / "pc2"
        newdir.mkdir()
        moved = str(newdir / "orig.tif")
        os.replace(raw, moved)

        ctl = _stub_controller(ws)
        ctl._tiff_level0_shape_axes = staticmethod(lambda _p: (None, None))
        state = load_state(ws)
        assert C._relink_row_for(ctl, state, moved, ws) is None
        assert state["sources"][0]["path"] == raw

    def test_a_dimension_less_row_still_accepts_on_name_and_size(
            self, tmp_path):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        ws, raw, sid = _seed_unprepared_workspace(tmp_path)
        state = load_state(ws)
        del state["sources"][0]["source_shape"]      # a pre-4c legacy row
        save_state(ws, state)

        newdir = tmp_path / "pc2"
        newdir.mkdir()
        moved = str(newdir / "orig.tif")
        os.replace(raw, moved)

        ctl = _stub_controller(ws)
        ctl._tiff_level0_shape_axes = staticmethod(lambda _p: (None, None))
        state = load_state(ws)
        assert C._relink_row_for(ctl, state, moved, ws) == sid


# ---------------------------------------------------------------------------
# Brick 4e items 4-8 — the audit notes
# ---------------------------------------------------------------------------
class TestDigestCoversTheInterior:
    """v1 sampled only head + tail, so a targeted write into ONE interior Z
    slice — an in-place deconvolution, a scripted per-slice fix — read back as
    unchanged and the stale store kept being served."""

    def _big(self, tmp_path, name="big.bin", mb=6):
        p = tmp_path / name
        p.write_bytes(b"A" * (mb << 20))
        return str(p)

    def test_an_interior_edit_is_detected(self, tmp_path):
        from packages.epicycle_largedata.multiscale import sampled_file_sha

        f = self._big(tmp_path)
        before = sampled_file_sha(f)
        size = os.path.getsize(f)
        # One "Z slice" of a 6-slice stack, rewritten in place in the middle --
        # what an in-place deconvolution does to every slice in turn.  Bigger
        # than the window spacing (size / 9), so it cannot fall in a gap.
        slice_bytes = size // 6
        with open(f, "r+b") as fh:
            fh.seek(size // 2 - slice_bytes // 2)
            fh.write(b"Z" * slice_bytes)
        assert os.path.getsize(f) == size          # the size is unchanged
        assert sampled_file_sha(f) != before

    def test_an_unchanged_file_hashes_the_same(self, tmp_path):
        from packages.epicycle_largedata.multiscale import sampled_file_sha

        f = self._big(tmp_path)
        assert sampled_file_sha(f) == sampled_file_sha(f)

    def test_the_digest_records_its_pattern_version(self, tmp_path):
        from packages.epicycle_largedata.multiscale import (
            SAMPLED_SHA_VERSION, sampled_file_sha,
        )

        got = sampled_file_sha(self._big(tmp_path, mb=1))
        assert got.startswith(f"v{SAMPLED_SHA_VERSION}:")

    def test_a_v1_digest_is_not_trusted(self, tmp_path, monkeypatch):
        """An older pattern compares unequal -> the caller falls back to the
        raw.  Conservative, which is the only safe direction here."""
        from napari_epicycle.data_preparation import controller as ctrl_mod

        ws = str(tmp_path / "ws")
        os.makedirs(ws, exist_ok=True)
        raw = _write_tiff(tmp_path / "v.tif", shape=(6, 32, 32))
        store = _make_store(ws, "v.zarr", None)
        prov = {"path": raw, "source_mtime": os.path.getmtime(raw),
                "source_size": os.path.getsize(raw),
                "source_head_sha": "deadbeef"}      # an un-versioned v1 value
        monkeypatch.setattr(ctrl_mod, "read_store_source", lambda p: dict(prov))
        state = empty_state()
        register_source(state, path=raw, display_name="v", axes="ZYX")
        state["sources"][0]["prepared_store"] = "artifacts/step1/largedata/v.zarr"
        save_state(ws, state)

        fn = ctrl_mod.DataPreparationController._fresh_prepared_store_for
        now = os.path.getmtime(raw)
        os.utime(raw, (now + 5000, now + 5000))
        assert fn(_stub_controller(ws, "v"), raw) is None


class TestAxesHintFromTheRow:
    """Brick 4e item 5: a user who confirmed ZCYX has already answered the
    channel-axis question; a fresh guess must not override it."""

    def test_a_confirmed_zcyx_row_does_not_false_mismatch(self):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        # (5, 6, Y, X): the guess picks axis 0 as channel -> (6, 64, 64).
        assert C._zyx_from_shape((5, 6, 64, 64), None) == (6, 64, 64)
        # The row says ZCYX, so Z is 5.
        row = {"source_axes": "ZCYX"}
        assert C._zyx_for_file((5, 6, 64, 64), None, row) == (5, 64, 64)

    def test_a_usable_header_still_wins(self):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        row = {"source_axes": "ZCYX"}
        assert C._zyx_for_file((5, 6, 64, 64), "CZYX", row) == (6, 64, 64)

    def test_a_time_axis_is_refused_not_guessed(self):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        assert C._zyx_from_shape((10, 6, 64, 64), "TZYX") is None
        assert C._zyx_from_shape((10, 6, 64, 64), "TCYX") is None
        assert C._zyx_from_shape((6, 64, 64), "TYX") is None


class TestRelinkDropsStaleDimensions:
    def test_source_shape_does_not_survive_a_relink(self, tmp_path):
        new = tmp_path / "moved.tif"
        new.write_bytes(b"0123456789")
        state = empty_state()
        sid = register_source(state, path="C:/old/o.tif", display_name="o",
                              axes="ZYX")
        state["sources"][0]["source_shape"] = [4, 8, 8]
        assert st_mod.relink_source(state, sid, str(new)) is True
        assert "source_shape" not in state["sources"][0]


class TestLegacyStoreWording:
    def test_it_says_MAY_have_changed(self):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        note = C._freshness_note(["unverifiable"])
        assert "may have changed" in note
        assert "to be safe" in note
        assert "Re-Prepare" in note
        # It must NOT assert a change we did not observe.
        assert "has changed" not in note
        # The genuine content change keeps its own, definite wording.
        assert "has changed" in C._freshness_note(["size_drift"])
