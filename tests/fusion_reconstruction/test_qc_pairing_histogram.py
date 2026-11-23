"""The pairing check in the triage histogram: Colour bars by "Overlap with ...".

With the five match columns in the table, "Colour bars by" offers the QC
plot's own item ("Overlap with {other}", or "Overlap with other run" when
the name is not known).  Picked, the bars stack by the six match words in
the plot's colours, the words read from the columns the way the plot reads
them; deleted and superseded rows stay in the grey ghost stack.  Without
the columns the item is not offered.  "Triage by" never offers
``match_id``, a label number.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from napari_epicycle.fusion_reconstruction import _qc_pairing_text as T
from napari_epicycle.fusion_reconstruction import qc_pairing as P
from tests.fusion_reconstruction import _qc_plot_snap as S

COLS = ("match_count", "match_id", "match_overlap", "match_shared",
        "match_minor_ratio")
UNKNOWN = T.COLOUR_BY_MATCH_UNKNOWN


def _df() -> pd.DataFrame:
    """Fourteen rows: two of each of the six words on active rows, then a
    deleted and a superseded row (the ghosts), qc_score spread over 0-1."""
    nan = np.nan
    rows = [
        # count, id, overlap, shared, minor, status, expected
        (0, 0, 0.0, 0, nan, "active", P.STATUS_NONE),
        (0, 0, 0.0, 0, nan, "active", P.STATUS_NONE),
        (1, 7, 0.9, 1, nan, "active", P.STATUS_ONE),
        (1, 8, 0.8, 1, nan, "active", P.STATUS_ONE),
        (2, 9, 0.7, 1, 0.6, "active", P.STATUS_SEVERAL),
        (3, 9, 0.6, 2, 0.2, "active", P.STATUS_SEVERAL),
        (1, 9, 0.5, 2, nan, "active", P.STATUS_SHARED),
        (1, 9, 0.4, 3, nan, "active", P.STATUS_SHARED),
        (0, 0, nan, 0, nan, "active", P.STATUS_TOO_SMALL),
        (0, 0, nan, 0, nan, "active", P.STATUS_TOO_SMALL),
        (nan, 0, nan, nan, nan, "active", P.STATUS_NOT_MEASURED),
        (nan, 0, nan, nan, nan, "active", P.STATUS_NOT_MEASURED),
        (1, 7, 0.9, 1, nan, "deleted", None),
        (1, 8, 0.8, 1, nan, "superseded", None),
    ]
    n = len(rows)
    df = pd.DataFrame({
        "label": np.arange(1, n + 1),
        "qc_score": np.linspace(0.0, 1.0, n),
        "area": np.linspace(100.0, 900.0, n),
        "qc_tier": ["good"] * n,
        "label_status": [r[5] for r in rows],
    })
    for i, c in enumerate(COLS):
        df[c] = [r[i] for r in rows]
    df.attrs["expected"] = [r[6] for r in rows]
    return df


@pytest.fixture
def panel():
    S.app()
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        TriageHistogramPanel,
    )

    p = TriageHistogramPanel()
    yield p
    p.deleteLater()


def _items(combo) -> list:
    return [combo.itemText(i) for i in range(combo.count())]


def _stacks(panel) -> dict:
    """``{(r, g, b): total bar height}`` of the bars drawn now."""
    out = {}
    for bar in panel._bar_items:
        c = bar.opts["brush"].color()
        key = (c.red(), c.green(), c.blue())
        out[key] = out.get(key, 0.0) + float(
            np.sum(np.asarray(bar.opts["height"], dtype=float)))
    return out


def _pick(panel, text):
    combo = panel._color_combo
    i = combo.findText(text)
    assert i >= 0, (text, _items(combo))
    combo.setCurrentIndex(i)


# --------------------------------------------------------------------------
# the item and its gate
# --------------------------------------------------------------------------
def test_the_item_is_offered_only_with_the_five_columns(panel):
    panel.set_data(_df().drop(columns=list(COLS)))
    assert not any(T.is_match_colour_by(t) for t in _items(panel._color_combo))
    panel.set_data(_df().drop(columns=["match_shared"]))
    assert not any(T.is_match_colour_by(t) for t in _items(panel._color_combo))
    panel.set_data(_df())
    items = _items(panel._color_combo)
    assert UNKNOWN in items
    # after the categorical columns, before the one gradient entry
    assert items.index(UNKNOWN) == items.index("value") - 1


def test_the_item_has_its_own_tooltip_with_the_bar_colours(panel):
    from qtpy.QtCore import Qt

    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        HIST_COLOUR_BARS_BY,
        HIST_HELP,
    )

    panel.set_data(_df())
    combo = panel._color_combo
    tip = combo.itemData(combo.findText(UNKNOWN), Qt.ItemDataRole.ToolTipRole)
    assert tip == T.HIST_COLOUR_BY_MATCH_TOOLTIP == (
        "Stacks the bars by what the other run says about each object: pink "
        "no match, blue one match, gold several matches, cyan shared match; "
        "dark grey too small to judge, light grey not measured.")
    # the control's own tooltip says the entry exists
    assert HIST_HELP[HIST_COLOUR_BARS_BY].endswith(
        ' After a comparison with another run, "Overlap with …" stacks '
        "them by the match words.")
    assert panel._color_combo.toolTip() == HIST_HELP[HIST_COLOUR_BARS_BY]


def test_the_item_names_the_other_run_and_keeps_the_choice(panel):
    panel.set_match_other("Nuclei")
    panel.set_data(_df())
    assert "Overlap with Nuclei" in _items(panel._color_combo)
    _pick(panel, "Overlap with Nuclei")
    # a new name renames the item in place: still the choice, same bars
    before = _stacks(panel)
    panel.set_match_other("DAPI nuclei")
    assert panel._color_combo.currentText() == "Overlap with DAPI nuclei"
    assert _stacks(panel) == before
    # a fresh table keeps the choice under the new name
    panel.set_data(_df())
    assert panel._color_combo.currentText() == "Overlap with DAPI nuclei"
    # no name: "other run", still the choice
    panel.set_match_other("")
    assert panel._color_combo.currentText() == UNKNOWN
    panel.set_data(_df())
    assert panel._color_combo.currentText() == UNKNOWN


def test_the_item_goes_with_the_columns(panel):
    panel.set_data(_df())
    _pick(panel, UNKNOWN)
    panel.set_data(_df().drop(columns=list(COLS)))
    items = _items(panel._color_combo)
    assert not any(T.is_match_colour_by(t) for t in items)
    assert panel._color_combo.currentText() == "qc_tier"
    # the bars are the tier stacks again: no match colour left
    match_rgb = set(P.MATCH_COLOURS.values())
    assert not set(_stacks(panel)) & match_rgb


# --------------------------------------------------------------------------
# the stacks
# --------------------------------------------------------------------------
def test_six_stacks_in_the_plot_colours_and_the_ghost_stack(panel):
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        _MATCH_COLORS,
    )
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        GHOST_GREY,
    )

    # one colour table for both widgets
    assert _MATCH_COLORS is P.MATCH_COLOURS
    assert tuple(P.MATCH_COLOURS) == P.STATUS_WORDS
    panel.set_data(_df())
    _pick(panel, UNKNOWN)
    stacks = _stacks(panel)
    for word in P.STATUS_WORDS:
        assert stacks.get(P.MATCH_COLOURS[word]) == 2.0, (word, stacks)
    assert stacks.get(GHOST_GREY) == 2.0
    assert sum(stacks.values()) == len(_df())
    assert len(stacks) == 7


def test_the_stacks_follow_the_words_of_the_plot(panel):
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        _match_words,
    )

    df = _df()
    words = _match_words(df)
    active = (df["label_status"] == "active").to_numpy()
    assert list(words[active]) == [w for w in df.attrs["expected"] if w]
    panel.set_data(df)
    _pick(panel, UNKNOWN)
    stacks = _stacks(panel)
    for word in P.STATUS_WORDS:
        assert stacks[P.MATCH_COLOURS[word]] == float(
            np.sum(words[active] == word))


def test_hidden_ghosts_leave_the_bars(panel):
    from napari_epicycle.fusion_reconstruction.triage_histogram_panel import (
        GHOST_GREY,
    )

    panel.set_data(_df())
    _pick(panel, UNKNOWN)
    panel.set_ghosts_visible(False)
    stacks = _stacks(panel)
    assert GHOST_GREY not in stacks
    assert sum(stacks.values()) == 12.0


def test_a_blanked_column_reads_not_measured_and_never_raises(panel):
    df = _df()
    df["match_count"] = pd.array(
        [None if pd.isna(v) else int(v) for v in df["match_count"]],
        dtype="Int64")
    df["match_shared"] = np.nan
    panel.set_data(df)
    _pick(panel, UNKNOWN)
    stacks = _stacks(panel)
    grey = P.MATCH_COLOURS[P.STATUS_NOT_MEASURED]
    # every active row lacks its shared count: not measured, except the
    # two under the size floor, which keep their own word
    assert stacks[grey] == 10.0
    assert stacks[P.MATCH_COLOURS[P.STATUS_TOO_SMALL]] == 2.0


def test_triage_by_never_offers_match_id(panel):
    panel.set_data(_df())
    feats = _items(panel._feature_combo)
    assert "match_id" not in feats
    for c in ("match_count", "match_overlap", "match_shared",
              "match_minor_ratio"):
        assert c in feats, c


def test_the_tier_lines_and_the_gate_are_unchanged(panel):
    panel.set_data(_df())
    _pick(panel, UNKNOWN)
    assert panel._feature_combo.currentText() == "qc_score"
    assert all(line.isVisible() for line in panel._tier_lines)
    assert not panel._reset_btn.isHidden() and panel._gate_cb.isHidden()
    panel._feature_combo.setCurrentText("area")
    assert not any(line.isVisible() for line in panel._tier_lines)
    assert panel._reset_btn.isHidden() and not panel._gate_cb.isHidden()


# --------------------------------------------------------------------------
# the controller hands the panel the other run's name
# --------------------------------------------------------------------------
def test_the_controller_provider_names_the_run_to_the_histogram(tmp_path):
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController as FRC,
    )

    names = []
    ui = SimpleNamespace(_triage_histogram_panel=SimpleNamespace(
        set_match_other=names.append))
    ctxs = {"patch_1.tif": {"match_run": "nuclei",
                            "match_run_display_name": "Nuclei"}}
    fake = SimpleNamespace(
        ui=ui, _qc_table_patch="patch_1.tif", working_dir=str(tmp_path),
        _match_values_predate_edits=False, _curation_dirty=False,
        _qc_features_df=_df(), _get_active_labels_path=lambda: "",
        _patch_extraction_context=lambda patch: dict(ctxs.get(patch, {})))
    fake._match_values_predate = (
        lambda c: FRC._match_values_predate(fake, c))
    assert FRC._match_plot_notes(fake)["other"] == "Nuclei"
    assert names == ["Nuclei"]
    # the table of a patch with no comparison: no name
    fake._qc_table_patch = "patch_2.tif"
    FRC._match_plot_notes(fake)
    assert names == ["Nuclei", ""]


def test_the_controller_provider_without_a_histogram(tmp_path):
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController as FRC,
    )

    fake = SimpleNamespace(
        ui=SimpleNamespace(), _qc_table_patch="", working_dir=str(tmp_path),
        _match_values_predate_edits=False, _curation_dirty=False,
        _qc_features_df=None, _get_active_labels_path=lambda: "",
        _patch_extraction_context=lambda patch: {})
    fake._match_values_predate = (
        lambda c: FRC._match_values_predate(fake, c))
    assert FRC._match_plot_notes(fake)["other"] == ""
