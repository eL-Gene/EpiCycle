"""The owner's live test (2026-09-29): "can't really set the channels to
segment".  A user pick in the membrane / nuclei combo must stick.

Root cause: the combos remembered the chosen channel only on ``activated``,
but Qt emits ``currentIndexChanged`` FIRST, and the controller's
``_on_channels_changed`` (with no trained channels known) rebuilds both
combos from the remembered choice on that signal -- so every pick snapped
back before it was remembered.
"""
from __future__ import annotations

import pytest

from ._step4_panel import close, make_workspace, panel, pump, tick


def _pick(combo, idx):
    """A user pick, in Qt's own order: the index moves (currentIndexChanged)
    and only then ``activated`` fires."""
    combo.setCurrentIndex(idx)
    combo.activated.emit(idx)


@pytest.fixture
def w(qapp, tmp_path):
    ws = make_workspace(tmp_path, patches=("P1.tif", "P2.tif"), channels=3)
    w = panel(qapp, ws)
    w.controller._trained_codes = None          # a pretrained model
    tick(w, qapp, "P1.tif")
    pump(qapp)
    yield w
    close(w)


def test_a_membrane_pick_sticks(qapp, w):
    assert w.chan1_combo.count() == 3
    _pick(w.chan1_combo, 2)
    pump(qapp)
    assert w.chan1_combo.currentIndex() == 2
    assert w.chosen_channels()[0] == 2


def test_a_nuclei_pick_sticks_and_none_stays_pickable(qapp, w):
    _pick(w.chan2_combo, 2)              # item 2 = channel 1 ("none" first)
    pump(qapp)
    assert w.chosen_channels()[1] == 1
    _pick(w.chan2_combo, 0)              # back to "none"
    pump(qapp)
    assert w.chosen_channels()[1] is None


def test_the_pick_survives_a_later_rebuild(qapp, w):
    _pick(w.chan1_combo, 1)
    w.controller._refresh_stacks()
    pump(qapp)
    assert w.chan1_combo.currentIndex() == 1


def test_through_the_controller_handler(qapp, w):
    """The controller path itself: _on_channels_changed with no trained
    channels rebuilds the combos, and the pick is kept."""
    c = w.controller
    w.chan1_combo.setCurrentIndex(2)        # emits currentIndexChanged
    c._on_channels_changed()
    pump(qapp)
    assert w.chan1_combo.currentIndex() == 2
