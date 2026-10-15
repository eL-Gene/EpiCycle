"""Regression test for the loss-plot filter+load round-trip bug
(fix/step2-training-history follow-up, 2026-05-26).

Scenario reproduced by user:
1. Open workspace with run "nuclei" (3 lateral rounds of training).
2. Switch to a different run, then switch back.
3. Loss plot is blank — even though `losses_by_round` is restored.

Root cause: `_switch_active_run` calls `set_view_filter("lateral")` at
the end. The filter matches keys with prefix `"lateral_"`. But
`load_round_history` was prepending `"round_"` to keys, so restored
curves were stored as `"round_lateral_0"`, never matching the filter.

Live-trained curves (via `add_round_curve("lateral_N", ...)`) match
the filter correctly — that's why initial workspace-load worked but
post-switch reload did not.

These tests stub the Qt-dependent `_sync_round_selector` /
`_show_empty` / `_redraw` so they can run without a live QApplication.
The label-formatting logic in `load_round_history` is pure Python and
that's the part under test.
"""
from __future__ import annotations

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("pyqtgraph")


@pytest.fixture
def widget(monkeypatch):
    """Return a `LossPlotWidget` instance with all Qt-touching methods
    stubbed.  The pure-data attributes (`_round_curves`, `_round_order`,
    `_view_filter`) and the `_filtered_rounds()` algorithm are real."""
    from napari_epicycle.hitl_training_annotation.ui import loss_plot_widget as lpw

    # Stub everything that needs a live Qt object.
    monkeypatch.setattr(lpw.LossPlotWidget, "_sync_round_selector", lambda self: None)
    monkeypatch.setattr(lpw.LossPlotWidget, "_show_empty", lambda self, empty: None)
    monkeypatch.setattr(lpw.LossPlotWidget, "_redraw", lambda self: None)

    # Build a bare instance — bypass QWidget.__init__ entirely.
    w = lpw.LossPlotWidget.__new__(lpw.LossPlotWidget)
    w._round_curves = {}
    w._round_order = []
    w._live_x = []
    w._live_y = []
    w._live_round_id = None
    w._view_filter = None
    return w


def test_load_round_history_keys_match_filter_lateral(widget):
    """Loaded curves with bucket-keyed `losses_by_round` (the on-disk
    format) must be visible after `set_view_filter("lateral")`."""
    widget.load_round_history({
        "lateral_0": [1.0, 0.8, 0.6],
        "lateral_1": [0.5, 0.4, 0.3],
        "lateral_2": [0.3, 0.2, 0.1],
    })
    widget._view_filter = "lateral"  # bypass set_view_filter's redraw

    filtered = widget._filtered_rounds()
    assert len(filtered) == 3, (
        f"Expected 3 lateral curves after load+filter, got {len(filtered)}: "
        f"{filtered!r}.  (Likely cause: `load_round_history` is prepending "
        f"`round_` to keys, breaking the prefix filter.)"
    )
    assert all(r.startswith("lateral_") for r in filtered), filtered


def test_load_round_history_keys_match_filter_axial(widget):
    """Same regression check for axial bucket."""
    widget.load_round_history({
        "axial_0": [1.0, 0.8],
        "axial_1": [0.5, 0.4],
    })
    widget._view_filter = "axial"

    filtered = widget._filtered_rounds()
    assert len(filtered) == 2, filtered
    assert all(r.startswith("axial_") for r in filtered), filtered


def test_load_then_filter_then_switch_back_round_trip(widget):
    """Simulate the run-switch sequence end-to-end:
    1. Load lateral curves (initial workspace open).
    2. set_view_filter("lateral") (active view is XY).
    3. clear_all() + load_round_history again (run switch back).
    4. set_view_filter("lateral") again.
    Curves should still be visible after the round-trip.
    """
    snap = {"lateral_0": [1.0, 0.5], "lateral_1": [0.4, 0.3]}

    widget.load_round_history(snap)
    widget._view_filter = "lateral"
    assert len(widget._filtered_rounds()) == 2

    # Simulate the post-switch sequence in `_switch_active_run`:
    # cancel_live_round + clear_all happen BEFORE load (the 11351da fix).
    widget.cancel_live_round()
    widget.clear_all()
    widget.load_round_history(snap)
    widget._view_filter = "lateral"

    filtered = widget._filtered_rounds()
    assert len(filtered) == 2, (
        f"After switch-back round-trip, expected 2 lateral curves, "
        f"got {filtered!r}"
    )


def test_load_round_history_does_not_double_prefix_legacy_keys(widget):
    """Defensive: if old state contained keys like `"round_lateral_0"`
    (from any legacy path), don't pile on another `round_` prefix.
    They still won't match the bucket filter, but they shouldn't break
    or duplicate either."""
    widget.load_round_history({"round_lateral_0": [1.0, 0.5]})

    # Single key in, single key out — no double-prefix.
    assert list(widget._round_curves.keys()) == ["round_lateral_0"]


def test_filter_none_shows_all_curves(widget):
    """`set_view_filter(None)` must show every loaded curve regardless
    of key prefix (sanity check that we didn't break the no-filter
    path)."""
    widget.load_round_history({
        "lateral_0": [1.0, 0.5],
        "axial_0": [0.8, 0.4],
    })
    widget._view_filter = None

    filtered = widget._filtered_rounds()
    assert len(filtered) == 2, filtered


def test_multi_view_dict_load_keeps_bucket_prefix(widget):
    """Multi-view `Mode=Both` rounds get saved as
    `{round_key: {view_type: [losses]}}` where `round_key` is already
    `"lateral_N"`.  After load+filter, the multi-view child labels
    must STILL start with the bucket prefix so they survive the filter.
    """
    widget.load_round_history({
        "lateral_0": {"spatial": [1.0, 0.5]},
    })
    widget._view_filter = "lateral"

    filtered = widget._filtered_rounds()
    assert len(filtered) == 1, filtered
    assert filtered[0].startswith("lateral_"), filtered[0]
