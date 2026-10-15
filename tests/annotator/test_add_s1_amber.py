"""A2: every blocking reason in S1 is drawn AMBER.

The owner missed the voxel-size refusal because it was grey: the dot was
amber, the sentence was not.  So the S1 TEXT follows ``s1_kind`` -- amber
for "warn", grey otherwise -- and every orchestrator push that writes S1
directly says which kind it is, so it cannot inherit the previous colour.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP

from .add_harness import Harness, install_fakes
from .add_widget_fixture import qapp, widget  # noqa: F401
from .test_add_s1_priority import ROWS, _ready

REPO = pathlib.Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# the table
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("row,kw,want", [r for r in ROWS if 8 <= r[0] <= 12])
def test_every_blocking_row_is_warn(row, kw, want):
    s = _ready(**kw)
    assert AP.s1_sentence(s) == want, row
    assert AP.s1_kind(s) == "warn", row


@pytest.mark.parametrize("transient", [
    AP.ADD_S1_STOPPED, AP.ADD_S1_FAILED, AP.ADD_S1_INPUTS_CHANGED,
    AP.ADD_S1_NOTHING_NUCLEI, AP.ADD_S1_ALL_FILTERED, AP.ADD_S1_TAKEN,
    AP.ADD_S1_BUSY_REFUSED, AP.ADD_S1_CLICK_OUTSIDE,
    AP.ADD_S1_CLICK_ON_KEPT.format(id=7), AP.ADD_S1_COMMIT_FAILED])
def test_every_transient_is_warn(transient):
    assert AP.s1_kind(_ready(transient=transient)) == "warn"


def test_tick_one_is_warn():
    s = _ready(candidates=(4, 1, 1, 0, 0), user_emptied=True)
    assert AP.s1_kind(s) == "warn"


def test_the_opening_instruction_and_measuring_are_not_amber():
    """Row 6 is the page's first instruction beside its own button; row 13
    resolves by itself (deliberately not amber)."""
    assert AP.s1_kind(_ready(has_box=False)) == "idle"
    assert AP.s1_kind(_ready(info_pending=True)) == "busy"


# --------------------------------------------------------------------------- #
# the panel draws the TEXT amber
# --------------------------------------------------------------------------- #
def test_a_warn_push_draws_s1_amber_and_a_ready_one_grey(widget):
    from napari_epicycle.annotator.advanced_features_widget import (
        ADD_S1_STYLE,
        ADD_S1_WARN_STYLE,
    )

    assert "#e6c26a" in ADD_S1_WARN_STYLE
    lbl = widget.add_status_label
    h0 = lbl.sizeHint().height()
    widget.update_add_ui_state(status_text=AP.ADD_S1_PICK_WALL_CHANNEL,
                               status_kind="warn")
    assert lbl.styleSheet() == ADD_S1_WARN_STYLE
    assert lbl.sizeHint().height() == h0, "colour only, no height"
    for kind in ("ready", "idle", "busy", "review"):
        widget.update_add_ui_state(status_kind=kind)
        assert lbl.styleSheet() == ADD_S1_STYLE, kind
    widget.update_add_ui_state(status_kind="warn")
    widget.update_add_ui_state(box_active=True)      # a dot-only push
    assert lbl.styleSheet() == ADD_S1_STYLE, (
        "the box push paints the dot green; the text must follow it")


# --------------------------------------------------------------------------- #
# the orchestrator: no direct S1 push without its kind
# --------------------------------------------------------------------------- #
def test_every_direct_s1_push_names_its_kind():
    src = (REPO / "napari_epicycle" / "annotator"
           / "curation_3d_widget.py").read_text(encoding="utf-8")
    bad = []
    n = 0
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("_push_add_ui", "update_add_ui_state")):
            kws = {k.arg for k in node.keywords}
            if "status_text" in kws:
                n += 1
                if "status_kind" not in kws:
                    bad.append(node.lineno)
    assert n >= 10, "the walk found the pushes"
    assert not bad, "status_text without status_kind at lines {}".format(bad)


@pytest.fixture
def h(monkeypatch):
    install_fakes(monkeypatch)
    hh = Harness(np.zeros((12, 20, 20), np.int32), images={})
    hh.place_box((slice(2, 10), slice(2, 18), slice(2, 18)))
    return hh


def test_the_batch_refusal_is_amber(h):
    from packages.epicycle_annotator.batch_refine import ADD_BATCH_BUSY

    h._batch_running = True
    assert h._add_refused_while_batch("accept") is True
    assert h.adv.last("status_text") == ADD_BATCH_BUSY
    assert h.adv.last("status_kind") == "warn"


def test_nothing_yet_to_refine_is_amber(h):
    from packages.epicycle_annotator import add_preview as P

    h._batch_running = False
    h._batch_staged = False
    h._add_piece_ids = []
    h._batch_refine_add_pieces()
    assert h.adv.last("status_text") == P.ADD_NO_ADDED_TO_REFINE
    assert h.adv.last("status_kind") == "warn"
