"""Extract (QC features) never runs over unsaved curation edits, says why,
and a finished extract starts both undo stacks over."""
from __future__ import annotations

import pandas as pd
import pytest

from napari_epicycle.fusion_reconstruction import _qc_plot_text as T
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController as C,
)

from .test_qc_tier_preservation import (
    _fresh_features_df,
    _make_bare_controller,
)


class _Area:
    def __init__(self):
        self.cards = []

    def set_card_message(self, text, kind="info"):
        self.cards.append((text, kind))


def test_extract_is_refused_while_curation_edits_are_unsaved(tmp_path):
    c = C.__new__(C)
    c._curation_dirty = True
    lbl = tmp_path / "labels_3d.tif"
    lbl.write_bytes(b"x")
    c._get_active_labels_path = lambda: str(lbl)
    c._append_log = lambda t: None
    area = _Area()
    c._set_plot_status = lambda text, kind="info": area.set_card_message(text, kind)
    c._launch_qc_extract_subprocess = lambda kw: pytest.fail("launched")
    c._launch_qc_extract_inprocess = lambda kw: pytest.fail("launched")
    c._extract_features_current()
    assert area.cards == [(T.CARD_WARN_EXTRACT_UNSAVED, "warn")]


class _Stack:
    def __init__(self):
        self.items = [1, 2]
        self.cleared = False

    def clear(self):
        self.items = []
        self.cleared = True

    def __bool__(self):
        return bool(self.items)


def test_a_finished_extract_empties_both_undo_stacks(tmp_path):
    prev = pd.DataFrame({"label": [1, 2, 3], "qc_tier": ["good"] * 3,
                         "label_status": ["active"] * 3})
    ctrl = _make_bare_controller(prev_df=prev, working_dir=tmp_path)
    ctrl._curation_undo_stack = _Stack()
    ctrl._tier_undo_stack = [pd.Series(["good"])]
    ctrl._drop_tier_undo = lambda: ctrl._tier_undo_stack.clear()
    C._on_qc_extract_finished(ctrl, _fresh_features_df([1, 2, 3]), "p.tif")
    assert ctrl._curation_undo_stack.cleared is True
    assert ctrl._tier_undo_stack == []
