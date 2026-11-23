"""One Preview draws the candidate layer once; a tick changes its colours.

The result handler draws the whole result -- every object, ticked or not --
with the colours of the rows as they start.  Filling the candidate list then
fires the overlay sync with those same rows ticked, which changes nothing.
A tick, All or None changes the colours only: the layer's data is never
drawn again for the same result.
"""
from __future__ import annotations

import numpy as np

from packages.epicycle_annotator import add_preview as AP

from .add_harness import Harness, install_fakes
from .test_add_audit_fixes_b import _PA, _PB, _PC, _three_with_the_middle_covered  # noqa: F401


def _harness(monkeypatch):
    install_fakes(monkeypatch)
    h = Harness(np.zeros((12, 36, 36), np.int32), images={})
    h.adv.preset = "cells"
    h.place_box((slice(0, 12), slice(0, 12), slice(0, 30)))
    h._add_click_markers = []
    h._add_markers_base = 500
    h._get_advanced_features_widget = lambda: h.adv
    return h


def _started(out):
    off = set(out.edge_ids) | set(out.small_ids)
    return [int(i) for i in out.ids if int(i) not in off]


def test_the_list_fill_does_not_draw_the_same_pieces_again(monkeypatch):
    h = _harness(monkeypatch)
    out = _three_with_the_middle_covered()
    h._apply_add_result(out)
    assert len(h.rendered) == 1
    assert h.rendered[0][0] is out.result
    h._add_candidate_layer = object()
    h.adv.checked = _started(out)           # the fill: the rows as they start
    h._sync_add_candidate_overlay()
    assert len(h.rendered) == 1             # not drawn again
    assert h.coloured == []                 # nor recoloured
    assert h._add_user_emptied is False


def test_a_tick_changes_colours_not_data(monkeypatch):
    h = _harness(monkeypatch)
    out = _three_with_the_middle_covered()
    h._apply_add_result(out)
    h._add_candidate_layer = object()
    ids = sorted(out.ids)
    h.adv.checked = ids[:1]
    h._sync_add_candidate_overlay()
    assert len(h.rendered) == 1, "a tick drew the data again"
    assert h.coloured[-1] == frozenset(ids[:1])
    h.adv.checked = ids                      # tick it back: a real change
    h._sync_add_candidate_overlay()
    assert len(h.rendered) == 1
    assert h.coloured[-1] == frozenset(ids)
    n = len(h.coloured)
    h._sync_add_candidate_overlay()          # the same ticks again: nothing
    assert len(h.coloured) == n
    h.adv.checked = []                       # None
    h._sync_add_candidate_overlay()
    assert len(h.rendered) == 1
    assert h.coloured[-1] == frozenset()
    assert h.adv.last("status_text") == AP.ADD_S1_TICK_ONE


def test_the_first_draw_carries_the_starting_colours(monkeypatch):
    h = _harness(monkeypatch)
    out = _three_with_the_middle_covered()
    h._apply_add_result(out)
    result, _roi, colours = h.rendered[0]
    main, side = colours
    started = set(_started(out))
    for i in out.ids:
        assert int(i) in side.color_dict
        assert (int(i) in main.color_dict) == (int(i) in started)


def test_a_new_result_is_never_taken_for_the_drawn_one(monkeypatch):
    """A layer drawn for another result is never recoloured for this one
    (every Preview draws its own result before the list fires)."""
    h = _harness(monkeypatch)
    first = _three_with_the_middle_covered()
    h._apply_add_result(first)
    h._add_candidate_layer = object()
    second = _three_with_the_middle_covered()
    h._add_preview_result = second.result
    h.adv.checked = list(second.ids)[:1]
    h._sync_add_candidate_overlay()
    assert len(h.rendered) == 1
    assert h.coloured == []
