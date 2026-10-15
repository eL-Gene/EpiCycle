"""Brick A (clean-swap rework) — Raw ⇄ Processed mutually-exclusive VIEW toggle.


DISPLAY ONLY — the toggle swaps which display layers are on the canvas; it never
touches the raw-source extract state.

The rework replaces the old hide/show (``.visible`` swap, both sets resident) with
a CLEAN SWAP: only the viewed set is on the canvas — raw is REMOVED while processed
shows, and the processed set is REMOVED + the raw source re-added on the way back
(2 layers for a 2-channel image, never 4).  Covered gates:

  * T1 clean swap — only ONE set is present in the layer list at a time; a flip
    REMOVES the other set (not merely hides it) and restores the viewed set.
  * greying / availability — ``_has_processed_view`` reads ``_processed_available``
    (set at each Process finish, cleared on teardown), so it stays True across a
    swap to raw even though the processed layers were removed.
  * guard — the layer churn runs under ``_swapping_source_display`` so the ②
    recipe-panel rebuild (`_on_layers_changed`) is suppressed mid-swap.
  * re-create — a flip back to processed re-creates the set from
    ``_processed_redisplay`` (never a recompute).
  * T8 ``.visible`` authoritative — ``_is_processed_view_showing`` reads the
    processed layers' ``.visible`` (present only while processed is the view).

Qt-free, mirroring the fake-viewer / fake-layer convention in
``test_process_streaming_wiring.py``.  The two Qt-heavy re-create helpers
(``_reshow_source_layers`` — re-adds raw fresh + re-engages progressive; and
``_readd_processed_layers`` — re-opens the store / re-adds the display pyramid)
are STUBBED with fake-viewer equivalents so the swap's remove/re-add orchestration
+ guard is exercised without their real machinery.
"""

from __future__ import annotations

from napari_epicycle.data_preparation import controller


DPC = controller.DataPreparationController

_PROC = "⟨processed⟩ "


# --------------------------------------------------------------------------- #
# Qt-free fakes
# --------------------------------------------------------------------------- #

class _FakeLayer:
    def __init__(self, name, visible=True, metadata=None):
        self.name = name
        self.visible = visible
        self.metadata = metadata or {}


class _FakeLayerList(list):
    """A layer list that resolves membership + indexing by NAME (str) as well
    as by identity (layer object) — enough of napari's `viewer.layers`."""

    def __contains__(self, item):
        if isinstance(item, str):
            return any(getattr(l, "name", None) == item for l in self)
        return list.__contains__(self, item)

    def __getitem__(self, key):
        if isinstance(key, str):
            for l in self:
                if getattr(l, "name", None) == key:
                    return l
            raise KeyError(key)
        return list.__getitem__(self, key)


class _FakeDims:
    def __init__(self, ndisplay=2):
        self.ndisplay = ndisplay


class _FakeViewer:
    def __init__(self, ndisplay=2):
        self.layers = _FakeLayerList()
        self.dims = _FakeDims(ndisplay)


def _bare_ctl(**attrs):
    ctl = DPC.__new__(DPC)
    ctl._log = lambda *a, **k: None
    for k, v in attrs.items():
        setattr(ctl, k, v)
    return ctl


def _names_in(viewer):
    return [getattr(l, "name", None) for l in viewer.layers]


def _wire(ndisplay=2, n_raw=2, n_proc=2, kind="store"):
    """A controller showing the PROCESSED view initially is not assumed — this
    wires the RAW view as the starting canvas (raw present, processed present in
    the tracking lists but NOT yet swapped in), plus the two RE-CREATE helpers
    stubbed with fake-viewer equivalents.

    Returns (ctl, viewer, raw_layers, proc_layers).  `raw_layers` / `proc_layers`
    are the INITIAL objects; a clean swap re-creates fresh objects, so assert on
    the viewer's layer NAMES + `ctl._reshown_raw` / the tracking lists, not on the
    original objects, after a swap.
    """
    viewer = _FakeViewer(ndisplay=ndisplay)
    raw_names = [f"ch{i}" for i in range(n_raw)]
    proc_names = [f"ch{i}" for i in range(n_proc)]
    raw_layers = [_FakeLayer(n, visible=True) for n in raw_names]
    proc_layers = [_FakeLayer(f"{_PROC}ch{i}", visible=True)
                   for i in range(n_proc)]
    for l in raw_layers:
        viewer.layers.append(l)
    for l in proc_layers:
        viewer.layers.append(l)
    store = list(proc_layers) if kind == "store" else []
    mem = list(proc_layers) if kind == "mem" else []
    ctl = _bare_ctl(
        viewer=viewer,
        _loaded_source_layers=list(raw_names),
        _processed_store_layers=store,
        _processed_layers=mem,
        _processed_available=True,
        _processed_redisplay={"kind": kind, "names": proc_names},
        _source_layers_on_canvas=True,
        _source_display_channels=[(n, [None]) for n in raw_names],
        _saved_channel_selection=None,
        _reshown_raw=None,
    )

    # Stub `_reshow_source_layers`: re-add the raw source layers FRESH (idempotent),
    # mirroring the real helper's contract (fresh objects + `_source_layers_on_canvas
    # = True`) without the Qt-heavy `_add_source_display_layers` machinery.
    def _reshow():
        if any(n in viewer.layers for n in ctl._loaded_source_layers):
            return
        new = [_FakeLayer(n, visible=True) for n in ctl._loaded_source_layers]
        for l in new:
            viewer.layers.append(l)
        ctl._source_layers_on_canvas = True
        ctl._reshown_raw = new

    # Stub `_readd_processed_layers`: re-create the processed set from the stored
    # spec (fresh objects, visible) + set the matching tracking list.
    def _readd():
        spec = getattr(ctl, "_processed_redisplay", None) or {}
        new = [_FakeLayer(f"{_PROC}ch{i}", visible=True) for i in range(n_proc)]
        for l in new:
            viewer.layers.append(l)
        if (spec.get("kind") or kind) == "store":
            ctl._processed_store_layers = new
        else:
            ctl._processed_layers = new

    ctl._reshow_source_layers = _reshow
    ctl._readd_processed_layers = _readd
    return ctl, viewer, raw_layers, proc_layers


# --------------------------------------------------------------------------- #
# T1 — clean swap: only the viewed set is on the canvas (both `store` and `mem`)
# --------------------------------------------------------------------------- #

def test_show_processed_removes_raw_keeps_processed_store():
    ctl, viewer, raw, proc = _wire(kind="store")
    ctl._show_processed_view()
    # raw REMOVED from the canvas (not merely hidden)
    assert all(l not in viewer.layers for l in raw)
    assert all(n not in viewer.layers for n in ctl._loaded_source_layers)
    # processed present + visible
    assert all(l in viewer.layers for l in proc)
    assert all(l.visible is True for l in proc)
    # only the processed set is resident (2 layers, not 4)
    assert len(viewer.layers) == len(proc)


def test_show_processed_removes_raw_keeps_processed_mem():
    ctl, viewer, raw, proc = _wire(kind="mem")
    ctl._show_processed_view()
    assert all(l not in viewer.layers for l in raw)
    assert all(l in viewer.layers for l in proc)
    assert len(viewer.layers) == len(proc)


def test_show_raw_removes_processed_readds_raw():
    ctl, viewer, raw, proc = _wire(kind="store")
    ctl._show_processed_view()           # raw removed, processed shown
    ctl._show_raw_view()                 # processed removed, raw re-added
    # processed REMOVED (both tracking lists emptied by the clears)
    assert all(l not in viewer.layers for l in proc)
    assert ctl._processed_store_layers == []
    assert ctl._processed_layers == []
    # raw re-added (fresh objects) + visible, and it is the ONLY resident set
    assert ctl._reshown_raw is not None
    assert all(n in viewer.layers for n in ctl._loaded_source_layers)
    assert all(l.visible is True for l in ctl._reshown_raw)
    assert len(viewer.layers) == len(ctl._loaded_source_layers)


def test_only_one_set_resident_across_a_double_flip():
    ctl, viewer, raw, proc = _wire(kind="store")
    for _ in range(3):
        ctl._show_processed_view()
        # processed resident, no raw names present
        assert not any(n in viewer.layers for n in ctl._loaded_source_layers)
        assert any(getattr(l, "name", "").startswith(_PROC) for l in viewer.layers)
        ctl._show_raw_view()
        # raw resident, no processed layers present
        assert all(n in viewer.layers for n in ctl._loaded_source_layers)
        assert not any(getattr(l, "name", "").startswith(_PROC)
                       for l in viewer.layers)


# --------------------------------------------------------------------------- #
# re-create — a flip back to processed rebuilds the set from the stored spec
# --------------------------------------------------------------------------- #

def test_flip_back_to_processed_recreates_from_spec():
    ctl, viewer, raw, proc = _wire(kind="store")
    ctl._show_processed_view()           # processed shown
    ctl._show_raw_view()                 # processed removed (lists emptied)
    assert ctl._processed_store_layers == []
    ctl._show_processed_view()           # must re-create processed via the spec
    # processed present again (fresh objects tracked) + raw removed
    assert ctl._processed_store_layers  # non-empty
    assert all(getattr(l, "name", "").startswith(_PROC) for l in viewer.layers)
    assert not any(n in viewer.layers for n in ctl._loaded_source_layers)


# --------------------------------------------------------------------------- #
# availability — `_has_processed_view` reads `_processed_available`
# --------------------------------------------------------------------------- #

def test_has_processed_view_false_when_flag_unset():
    ctl = _bare_ctl(
        viewer=_FakeViewer(),
        _loaded_source_layers=["ch0"],
        _processed_store_layers=[],
        _processed_layers=[],
        _processed_available=False,
    )
    assert ctl._has_processed_view() is False


def test_has_processed_view_true_when_flag_set():
    ctl, *_ = _wire(kind="store")
    assert ctl._has_processed_view() is True


def test_has_processed_view_survives_swap_to_raw():
    """The clean swap to raw REMOVES the processed layers, but the processed view
    still EXISTS this session — availability must not depend on the layer lists."""
    ctl, viewer, raw, proc = _wire(kind="store")
    ctl._show_processed_view()
    ctl._show_raw_view()
    assert ctl._processed_store_layers == []      # layers gone
    assert ctl._has_processed_view() is True       # …but still available


def test_show_processed_noop_when_not_available():
    """A stray internal call with no processed view available must not blank the
    canvas (never remove raw with nothing to show)."""
    viewer = _FakeViewer()
    raw = _FakeLayer("ch0", visible=True)
    viewer.layers.append(raw)
    ctl = _bare_ctl(
        viewer=viewer,
        _loaded_source_layers=["ch0"],
        _processed_store_layers=[],
        _processed_layers=[],
        _processed_available=False,
        _source_layers_on_canvas=True,
        _processed_redisplay=None,
    )
    ctl._show_processed_view()
    assert raw in viewer.layers and raw.visible is True


# --------------------------------------------------------------------------- #
# guard — the swap runs under `_swapping_source_display` (recipe rebuild suppressed)
# --------------------------------------------------------------------------- #

def test_swap_sets_swapping_guard_during_churn():
    ctl, viewer, raw, proc = _wire(kind="store")
    seen = {}

    # Probe the guard AT THE MOMENT the raw-removal runs inside `_show_processed_view`.
    real_remove = ctl._remove_source_layers_from_viewer

    def _probe_remove():
        seen["during_processed"] = ctl._swapping_source_display
        return real_remove()

    ctl._remove_source_layers_from_viewer = _probe_remove
    ctl._show_processed_view()
    assert seen["during_processed"] is True
    assert ctl._swapping_source_display is False

    # …and the RAW direction: the re-add runs inside the guarded block too.
    def _probe_reshow():
        seen["during_raw"] = ctl._swapping_source_display

    ctl._reshow_source_layers = _probe_reshow
    ctl._show_raw_view()
    assert seen["during_raw"] is True
    assert ctl._swapping_source_display is False


# --------------------------------------------------------------------------- #
# T8 — `_is_processed_view_showing` reads the processed layers' `.visible`
# --------------------------------------------------------------------------- #

def test_is_processed_view_showing_tracks_the_swap():
    ctl, viewer, raw, proc = _wire(kind="store")
    ctl._show_processed_view()
    assert ctl._is_processed_view_showing() is True
    ctl._show_raw_view()
    # processed layers removed → nothing visible → not showing processed
    assert ctl._is_processed_view_showing() is False


def test_is_processed_view_showing_reads_visible_not_presence():
    """Even with processed layers tracked, an externally-hidden set reads as not
    showing — the reader keys off `.visible`, not the list length."""
    ctl, viewer, raw, proc = _wire(kind="store")
    ctl._show_processed_view()
    for l in ctl._processed_store_layers:
        l.visible = False
    assert ctl._is_processed_view_showing() is False


# --------------------------------------------------------------------------- #
# `_show_raw_view` re-adds raw via `_reshow_source_layers` (progressive re-engaged
# by construction — the old `_kick_raw_loaders_if_3d` is no longer needed)
# --------------------------------------------------------------------------- #

def test_show_raw_view_invokes_reshow_source_layers():
    ctl, viewer, raw, proc = _wire(kind="store")
    calls = []
    ctl._reshow_source_layers = lambda: calls.append(True)
    ctl._show_processed_view()
    ctl._show_raw_view()
    assert calls == [True]
