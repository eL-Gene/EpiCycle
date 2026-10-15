"""Final fix pass U1: Add re-points the shared volume at the main layer
before it reads it, as split / smooth / merge do.

A controller path (Commit Triage's zero, Discard's revert, an undo) may
reassign ``layer.data``; ``sync_manager.shared_labels`` must follow before
Add snapshots or writes, or it works on an abandoned copy.
"""

from __future__ import annotations

import numpy as np
import pytest

from .add_harness import Harness, install_fakes, nuclei_volume
from .test_add_nuclei_golden import box_of


class _Labels:
    def __init__(self, name, data):
        self.name = name
        self.data = data


@pytest.fixture
def h(monkeypatch):
    install_fakes(monkeypatch)
    img, labels = nuclei_volume()
    hh = Harness(labels, images={"nuc": img})
    # the main labels layer now holds a NEW array; shared still the old one
    hh.main = _Labels("L", labels.copy())
    hh.viewer.layers.append(hh.main)
    return hh


def test_start_add_realigns_shared(h):
    assert h.sync_manager.shared_labels["L"] is not h.main.data
    h._start_add()
    assert h.sync_manager.shared_labels["L"] is h.main.data


def test_preview_add_realigns_shared(h):
    h.place_box(box_of(h.main.data.shape))
    h._add_threshold_value = 0.5
    assert h.sync_manager.shared_labels["L"] is not h.main.data
    h._preview_add()
    assert h.sync_manager.shared_labels["L"] is h.main.data
