"""The pairing check in the QC plot: Colour by "Overlap with ...".

The plot offers "Overlap with {other}" (the other run's name, or "Overlap
with other run" when it is not known) only when the table carries the five
match columns.  Picked, it reads every row's match word from those columns (the
check's own precedence; a row that is not active reads "not measured"),
colours it in a fixed order, writes the words into the table's
``match_status`` column so a legend pick finds its rows (the features CSV
leaves that column out), and shows under the legend the two notes the host
supplies through its provider: the other run's file changed, and the values
predate an edit.  The match columns' tooltips name the other run.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from napari_epicycle.fusion_reconstruction import _qc_pairing_text as T
from napari_epicycle.fusion_reconstruction import qc_pairing as P
from tests.fusion_reconstruction import _qc_plot_snap as S

ORDER = (P.STATUS_NONE, P.STATUS_ONE, P.STATUS_SEVERAL, P.STATUS_SHARED,
         P.STATUS_TOO_SMALL, P.STATUS_NOT_MEASURED)
#: the Colour by item with no provider (the name is not known)
MATCH = T.COLOUR_BY_MATCH_UNKNOWN
COLS = ("match_count", "match_id", "match_overlap", "match_shared",
        "match_minor_ratio")


def _df() -> pd.DataFrame:
    """Twelve rows, two of each match word (rows 11-12: a measured row
    that is deleted, a measured row that is superseded)."""
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
        (nan, 0, nan, nan, nan, "active", P.STATUS_NOT_MEASURED),
        (1, 7, 0.9, 1, nan, "deleted", P.STATUS_NOT_MEASURED),
        (1, 8, 0.8, 1, nan, "superseded", P.STATUS_NOT_MEASURED),
    ]
    n = len(rows)
    rng = np.random.RandomState(3)
    df = pd.DataFrame({
        "label": np.arange(1, n + 1),
        "area": np.round(rng.rand(n) * 1000 + 100, 2),
        "solidity": np.round(rng.rand(n), 3),
        "qc_score": np.round(rng.rand(n), 3),
        "qc_tier": ["good"] * n,
        "label_status": [r[5] for r in rows],
        "euler_number": np.ones(n, dtype=np.int64),
    })
    for i, c in enumerate(COLS):
        df[c] = [r[i] for r in rows]
    df.attrs["expected"] = [r[6] for r in rows]
    return df


def _wb():
    S.app()
    from napari_epicycle.fusion_reconstruction._qc_plot_text import (
        STEP3_PLOT_TEXTS,
    )
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        STEP3_SCHEME,
        QCPlotWidget,
    )
    from napari_epicycle.ui_shell.theme import STEP3_PLOT_PALETTE

    w = QCPlotWidget(layout="workbench", texts=STEP3_PLOT_TEXTS,
                     palette=STEP3_PLOT_PALETTE, tier_scheme=STEP3_SCHEME)
    w.resize(562, 600)
    return w


@pytest.fixture
def wb():
    w = _wb()
    yield w
    w.deleteLater()


@pytest.fixture
def classic():
    S.app()
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        QCPlotWidget,
    )

    w = QCPlotWidget()
    yield w
    w.deleteLater()


def _items(combo) -> list:
    return [combo.itemText(i) for i in range(combo.count())]


def _pick_match(w):
    w._color_combo.setCurrentIndex(w._match_colour_index())
    w._redraw()


# --------------------------------------------------------------------------
# the option and its gate
# --------------------------------------------------------------------------
def test_the_option_word_and_tooltip():
    assert T.colour_by_match("Nuclei") == "Overlap with Nuclei"
    assert T.colour_by_match("") == "Overlap with other run"
    assert T.colour_by_match("  ") == "Overlap with other run"
    # the stock first run reads as a run, not as "by default"
    assert T.colour_by_match("Default") == "Overlap with the Default run"
    assert T.COLOUR_BY_MATCH_DEFAULT == "Overlap with the Default run"
    assert T.is_match_colour_by(T.COLOUR_BY_MATCH_DEFAULT)
    assert T.colour_by_match("Default 2") == "Overlap with Default 2"
    assert T.colour_by_match("default") == "Overlap with default"
    assert T.is_match_colour_by("Overlap with Nuclei")
    assert T.is_match_colour_by(MATCH)
    assert not T.is_match_colour_by("Topology")
    assert not T.is_match_colour_by(None)
    assert T.COLOUR_BY_MATCH_TOOLTIP == (
        "Colour by what the other run says about each object: no match, "
        "one, several, or a match shared with another object.")
    assert P.STATUS_WORDS == (
        "no match", "one match", "several matches", "shared match",
        "too small to judge", "not measured")
    # the legend words are the rule's, not a second copy
    assert (T.STATUS_NONE, T.STATUS_ONE, T.STATUS_SEVERAL, T.STATUS_SHARED,
            T.STATUS_TOO_SMALL, T.STATUS_NOT_MEASURED) == P.STATUS_WORDS


@pytest.mark.parametrize("layout", ["wb", "classic"])
def test_match_is_offered_only_with_the_five_columns(layout, request):
    w = request.getfixturevalue(layout)
    df = _df()
    w.set_features(df.drop(columns=list(COLS)))
    assert w._match_colour_index() < 0
    w.set_features(_df().drop(columns=["match_minor_ratio"]))
    assert w._match_colour_index() < 0
    w.set_features(_df())
    items = _items(w._color_combo)
    assert MATCH in items
    # after the other categorical options, before the features
    assert items.index(MATCH) == items.index("Topology") + 1
    assert w.set_colour_by(MATCH) if layout == "wb" else True


@pytest.mark.parametrize("layout", ["wb", "classic"])
def test_the_item_names_the_other_run_and_keeps_the_choice(layout,
                                                            request):
    w = request.getfixturevalue(layout)
    facts = {"other": "Nuclei", "stale": None, "predate_edits": False}
    w.set_match_notes_provider(lambda: dict(facts))
    w.set_features(_df())
    items = _items(w._color_combo)
    assert "Overlap with Nuclei" in items and MATCH not in items
    _pick_match(w)
    assert w._color_combo.currentText() == "Overlap with Nuclei"
    # the run is renamed: the item follows, and stays the choice
    facts["other"] = "DAPI nuclei"
    w.set_features(_df())
    assert w._color_combo.currentText() == "Overlap with DAPI nuclei"
    assert _items(w._color_combo).count("Overlap with DAPI nuclei") == 1
    # the name goes: the item reads "other run", still the choice
    facts["other"] = ""
    w.set_features(_df())
    assert w._color_combo.currentText() == MATCH
    # the match columns go: the item goes with them
    w.set_features(_df().drop(columns=list(COLS)))
    assert w._match_colour_index() < 0
    assert not any(T.is_match_colour_by(t)
                   for t in _items(w._color_combo))


def test_any_overlap_item_picks_the_one_item(wb):
    wb.set_match_notes_provider(
        lambda: {"other": "Nuclei", "stale": None, "predate_edits": False})
    wb.set_features(_df())
    assert wb.set_colour_by("Overlap with an old name")
    assert wb._color_combo.currentText() == "Overlap with Nuclei"
    col, _ids, _values = wb.colour_by_column_values()
    assert col == "match_status"
    assert "Overlap with Nuclei:" in wb._legend_label.text()


# --------------------------------------------------------------------------
# the words, their order and their colours
# --------------------------------------------------------------------------
def test_the_words_follow_the_columns_and_the_label_status(wb):
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        _match_words,
    )

    df = _df()
    assert list(_match_words(df)) == df.attrs["expected"]


def test_the_legend_order_and_colours(wb):
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        _MATCH_COLORS,
    )

    df = _df()
    wb.set_features(df)
    assert wb.set_colour_by(MATCH)
    col, cats = wb._get_categorical_info(MATCH)
    assert col == "match_status"
    assert tuple(cats) == ORDER
    assert tuple(_MATCH_COLORS) == ORDER
    # Topology's colours for the measured words, a grey under the size
    # floor, a light blue-grey for the rows that were not measured
    assert cats[P.STATUS_ONE] == (120, 150, 175)
    assert cats[P.STATUS_NONE] == (236, 96, 140)
    assert cats[P.STATUS_SEVERAL] == (246, 190, 70)
    assert cats[P.STATUS_SHARED] == (80, 200, 215)
    assert cats[P.STATUS_TOO_SMALL] == (105, 105, 110)
    assert cats[P.STATUS_NOT_MEASURED] == (190, 190, 190)
    # every drawn point in its word's colour
    rgba = wb._get_point_colors_for(df)
    for i, word in enumerate(df.attrs["expected"]):
        r, g, b = _MATCH_COLORS[word]
        assert np.allclose(rgba[i, :3], [r / 255, g / 255, b / 255]), i


def test_the_greys_are_listed_only_when_an_active_row_reads_them(wb):
    df = _df().iloc[:8].copy()          # measured words only
    wb.set_features(df)
    wb.set_colour_by(MATCH)
    _col, cats = wb._get_categorical_info(MATCH)
    assert tuple(cats) == ORDER[:4]
    df = _df()
    df = df[df["label_status"] != "active"].copy()  # only ghosts unmeasured
    df = pd.concat([_df().iloc[:2], df])
    wb.set_features(df.reset_index(drop=True))
    wb.set_colour_by(MATCH)
    _col, cats = wb._get_categorical_info(MATCH)
    assert P.STATUS_NOT_MEASURED not in cats


def test_the_legend_counts_active_rows(wb):
    wb.set_features(_df())
    wb.set_colour_by(MATCH)
    html = wb._legend_label.text()
    assert MATCH + ":" in html
    for word in ORDER:
        assert _nb(word) in html
    # the two ghosts count nowhere: one "not measured" row is active
    assert ("&nbsp;not&nbsp;measured</a>&nbsp;"
            '<span style="color:#888888;">1</span>') in html
    assert ("&nbsp;one&nbsp;match</a>&nbsp;"
            '<span style="color:#888888;">2</span>') in html
    pos = [html.index(f"&nbsp;{_nb(w)}</a>") for w in ORDER]
    assert pos == sorted(pos)


def _nb(word):
    return word.replace(" ", "&nbsp;")


def _entries(html):
    """The legend's entries, as split where a line may break (the only
    plain spaces left between them)."""
    import re

    body = re.sub(r"<[^>]*>", "", html.split("</span>", 1)[1])
    return [e for e in body.split(" &nbsp; ") if e.strip()]


def test_a_match_entry_never_breaks_inside(wb):
    wb.set_features(_df())
    wb.set_colour_by(MATCH)
    html = wb._legend_label.text()
    entries = _entries(html)[:len(ORDER)]    # the ghosts' count may follow
    assert len(entries) == len(ORDER)
    for entry in entries:
        # no plain space inside an entry: a wrap falls between entries
        assert " " not in entry.strip(), entry


def test_a_narrow_legend_wraps_between_match_entries_only(wb):
    """Laid out at a narrow width, every line of the legend starts with
    an entry's dot, never with the tail of a word."""
    from qtpy.QtGui import QTextDocument

    wb.set_features(_df())
    wb.set_colour_by(MATCH)
    doc = QTextDocument()
    doc.setDefaultFont(wb._legend_label.font())
    doc.setHtml(wb._legend_label.text())
    doc.setTextWidth(300)
    doc.documentLayout().documentSize()     # lays the lines out
    lines = []
    block = doc.begin()
    while block.isValid():
        lay = block.layout()
        text = block.text()
        for i in range(lay.lineCount()):
            ln = lay.lineAt(i)
            lines.append(text[ln.textStart():ln.textStart() + ln.textLength()])
        block = block.next()
    assert len(lines) >= 2, lines
    for line in lines[1:]:
        assert line.lstrip().startswith("●"), lines


def test_the_topology_legend_keeps_its_plain_spaces(wb):
    df = _df()
    df["euler_number"] = [1, 0, 2, -1] * (len(df) // 4) + [1] * (len(df) % 4)
    wb.set_features(df)
    assert wb.set_colour_by("Topology")
    html = wb._legend_label.text()
    assert "&nbsp;Has cavity</a>" in html
    assert "&nbsp;Has tunnel</a>" in html


def test_blanked_and_inactive_rows_read_not_measured(wb):
    df = _df()
    wb.set_features(df)
    wb.set_colour_by(MATCH)
    words = list(df["match_status"])
    assert words[9] == P.STATUS_NOT_MEASURED      # blanked (NaN count)
    assert words[10] == P.STATUS_NOT_MEASURED     # deleted
    assert words[11] == P.STATUS_NOT_MEASURED     # superseded


# --------------------------------------------------------------------------
# the legend pick
# --------------------------------------------------------------------------
@pytest.mark.parametrize("layout", ["wb", "classic"])
def test_a_legend_click_selects_the_rows_of_its_word(layout, request):
    w = request.getfixturevalue(layout)
    df = _df()
    w.set_features(df)
    _pick_match(w)
    w._on_legend_clicked("cat://match_status/several matches")
    assert sorted(w.get_selected_df_indices()) == [4, 5]
    assert w.selected_label_ids() == [5, 6]
    w._on_legend_clicked("cat://match_status/shared match")
    assert w.selected_label_ids() == [7, 8]


# --------------------------------------------------------------------------
# the display-only column never reaches the CSV
# --------------------------------------------------------------------------
def test_match_status_is_in_the_table_and_not_in_the_csv(wb, tmp_path):
    from napari_epicycle.fusion_reconstruction.qc_features import (
        save_patch_csv,
    )

    df = _df()
    df.attrs = {}
    wb.set_features(df)
    wb.set_colour_by(MATCH)
    assert "match_status" in df.columns        # written by reference
    out = tmp_path / "p_features.csv"
    save_patch_csv(df, out)
    back = pd.read_csv(out)
    assert "match_status" not in back.columns
    assert "match_status" in df.columns         # the table keeps it
    for c in COLS:
        assert c in back.columns


# --------------------------------------------------------------------------
# the two notes under the legend
# --------------------------------------------------------------------------
STALE = T.STALE_MATCH.format(other="Nuclei")


def _provider(facts):
    calls = []

    def cb():
        calls.append(1)
        return dict(facts)
    cb.calls = calls
    return cb


@pytest.mark.parametrize("layout", ["wb", "classic"])
def test_the_stale_line_follows_the_provider(layout, request):
    w = request.getfixturevalue(layout)
    facts = {"other": "Nuclei", "stale": STALE, "predate_edits": False}
    w.set_match_notes_provider(lambda: dict(facts))
    w.set_features(_df())
    assert w.match_notes_shown() == []          # not colouring by the match item
    _pick_match(w)
    assert w.match_notes_shown() == [STALE]
    facts["stale"] = None
    w._redraw()
    assert w.match_notes_shown() == []
    assert w._match_note_label.isHidden()


def test_the_predate_edits_line_follows_the_flag(wb):
    facts = {"other": "Nuclei", "stale": None, "predate_edits": True}
    wb.set_match_notes_provider(lambda: dict(facts))
    wb.set_features(_df())
    wb.set_colour_by(MATCH)
    assert wb.match_notes_shown() == [T.PREDATE_EDITS]
    assert T.PREDATE_EDITS == (
        "Match values predate your edits. Extract again to refresh them.")
    facts["stale"] = STALE
    wb._redraw()
    assert wb.match_notes_shown() == [STALE, T.PREDATE_EDITS]
    facts["predate_edits"] = False
    facts["stale"] = None
    wb._redraw()
    assert wb.match_notes_shown() == []


def test_the_notes_go_with_another_colour_and_without_a_table(wb):
    wb.set_match_notes_provider(
        lambda: {"other": "Nuclei", "stale": STALE, "predate_edits": True})
    wb.set_features(_df())
    wb.set_colour_by(MATCH)
    assert len(wb.match_notes_shown()) == 2
    wb.set_colour_by("QC tier")
    assert wb.match_notes_shown() == []
    wb.set_colour_by(MATCH)
    assert len(wb.match_notes_shown()) == 2
    wb.clear()
    assert wb.match_notes_shown() == []


def test_no_provider_call_and_no_note_label_without_match_columns(wb):
    cb = _provider({"other": "Nuclei", "stale": STALE})
    wb.set_match_notes_provider(cb)
    wb.set_features(_df().drop(columns=list(COLS)))
    wb.set_colour_by("Topology")
    assert cb.calls == []
    assert wb._match_note_label is None


def test_a_failing_provider_shows_nothing_and_never_raises(wb):
    def boom():
        raise RuntimeError("no sidecar")
    wb.set_match_notes_provider(boom)
    wb.set_features(_df())
    assert wb.set_colour_by(MATCH)
    assert wb.match_notes_shown() == []
    with pytest.raises(TypeError):
        wb.set_match_notes_provider("not callable")


# --------------------------------------------------------------------------
# the tooltips
# --------------------------------------------------------------------------
def test_the_match_tooltips_name_the_other_run(wb):
    from qtpy.QtCore import Qt

    from napari_epicycle.segmentation.analysis_widget import (
        _FEATURE_DESCRIPTIONS,
    )

    # the host supplies the descriptions: the plot never imports them
    descriptions = {c: _FEATURE_DESCRIPTIONS[c] for c in COLS}
    wb.set_match_notes_provider(
        lambda: {"other": "Nuclei", "stale": STALE, "predate_edits": False,
                 "descriptions": descriptions})
    wb.set_features(_df())
    role = Qt.ItemDataRole.ToolTipRole
    combo = wb._feature_x_combo
    tip = combo.itemData(combo.findText("match_count"), role)
    desc = _FEATURE_DESCRIPTIONS["match_count"]
    assert "the other run" in desc
    assert tip.startswith("match_count: ")
    assert "the other run (Nuclei)" in tip
    assert tip.endswith("\n" + STALE)
    for c in ("match_overlap", "match_shared", "match_minor_ratio"):
        t = wb._feature_y_combo.itemData(
            wb._feature_y_combo.findText(c), role)
        assert t.startswith(c + ": ") and (
            "\nCompared with run 'Nuclei'.\n") in t, (c, t)
    cc = wb._color_combo
    assert cc.itemData(cc.findText("Overlap with Nuclei"), role) == (
        T.COLOUR_BY_MATCH_TOOLTIP)


def test_the_column_tooltip_without_a_name_is_the_description():
    assert T.column_tooltip("match_count", "Counted against the other run.") \
        == "match_count: Counted against the other run."
    assert T.column_tooltip(
        "match_id", "Counted against the other run.", "Cells",
        [None, T.PREDATE_EDITS]) == (
        "match_id: Counted against the other run (Cells).\n"
        + T.PREDATE_EDITS)
    # a description that does not name the other run gets its own line
    assert T.column_tooltip("match_overlap", "Shared over the smaller.",
                            "Cells") == (
        "match_overlap: Shared over the smaller.\n"
        "Compared with run 'Cells'.")


def test_the_match_numbers_are_axes_but_not_colour_options(wb):
    wb.set_features(_df())
    colours = _items(wb._color_combo)
    assert MATCH in colours
    for c in ("match_count", "match_overlap", "match_shared",
              "match_minor_ratio"):
        assert c in _items(wb._feature_x_combo), c
        assert c in _items(wb._feature_y_combo), c
        assert c not in colours, c


# --------------------------------------------------------------------------
# the viewer's hover, and robustness
# --------------------------------------------------------------------------
def test_the_hover_map_reads_the_match_words(wb):
    df = _df()
    wb.set_features(df)
    wb.set_colour_by(MATCH)
    col, ids, values = wb.colour_by_column_values()
    assert col == "match_status"
    assert list(ids) == list(range(1, 11))        # the ghosts left out
    assert list(values) == df.attrs["expected"][:10]
    ids2, rgba = wb.viewer_colours()
    assert list(ids2) == list(ids) and rgba.shape == (10, 4)


@pytest.mark.parametrize("layout", ["wb", "classic"])
def test_nan_and_nullable_match_counts_do_not_raise(layout, request):
    w = request.getfixturevalue(layout)
    df = _df()
    df["match_count"] = np.nan
    w.set_features(df)
    _pick_match(w)
    assert set(df["match_status"]) <= {P.STATUS_NOT_MEASURED}
    df2 = _df()
    df2["match_count"] = pd.array(
        [None if pd.isna(v) else int(v) for v in df2["match_count"]],
        dtype="Int64")
    df2["match_shared"] = pd.array(
        [None if pd.isna(v) else int(v) for v in df2["match_shared"]],
        dtype="Int64")
    w.set_features(df2)
    _pick_match(w)
    assert list(df2["match_status"]) == df2.attrs["expected"]
    for pt in ("Scatter", "Violin"):
        w.plot_type_combo.setCurrentText(pt)
        w._redraw()


# --------------------------------------------------------------------------
# the controller's provider
# --------------------------------------------------------------------------
def _ctrl(tmp_path, ctx, flag=False, *, df=None, dirty=False,
          labels_path=""):
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController as FRC,
    )

    fake = SimpleNamespace(
        _qc_table_patch="patch_1.tif", working_dir=str(tmp_path),
        _match_values_predate_edits=flag, _curation_dirty=dirty,
        _qc_features_df=_df() if df is None else df,
        _get_active_labels_path=lambda: labels_path,
        _patch_extraction_context=lambda patch: dict(ctx))
    fake._match_values_predate = (
        lambda c: FRC._match_values_predate(fake, c))
    return lambda: FRC._match_plot_notes(fake)


def test_the_controller_provider_says_stale_only_when_the_file_changed(
        tmp_path):
    f = tmp_path / "runs" / "nuclei" / "labels_3d.tif"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x" * 10)
    st = os.stat(f)
    ctx = {"match_run": "nuclei", "match_run_display_name": "Nuclei",
           "match_file": "runs/nuclei/labels_3d.tif",
           "match_file_mtime": st.st_mtime, "match_file_size": 10}
    notes = _ctrl(tmp_path, ctx)()
    descriptions = notes.pop("descriptions")
    assert notes == {"other": "Nuclei", "stale": None,
                     "predate_edits": False}
    assert set(descriptions) == set(COLS)
    assert all(descriptions.values())
    f.write_bytes(b"y" * 12)
    notes = _ctrl(tmp_path, ctx, flag=True)()
    assert notes["stale"] == STALE
    assert notes["predate_edits"] is True
    f.unlink()
    assert _ctrl(tmp_path, ctx)()["stale"] == STALE


def test_the_controller_provider_with_no_match_recorded(tmp_path):
    notes = _ctrl(tmp_path, {})()
    notes.pop("descriptions")
    assert notes == {"other": "", "stale": None, "predate_edits": False}


def test_the_predate_line_needs_match_columns(tmp_path):
    plain = _df().drop(columns=list(COLS))
    assert _ctrl(tmp_path, {}, flag=True, df=plain)()[
        "predate_edits"] is False
    assert _ctrl(tmp_path, {}, dirty=True, df=plain)()[
        "predate_edits"] is False
    assert _ctrl(tmp_path, {}, flag=True)()["predate_edits"] is True


def test_the_predate_line_follows_unsaved_edits(tmp_path):
    assert _ctrl(tmp_path, {}, dirty=True)()["predate_edits"] is True


def test_the_predate_line_follows_the_labels_file_after_a_restart(
        tmp_path):
    lab = tmp_path / "labels_3d.tif"
    lab.write_bytes(b"x" * 10)
    st = os.stat(lab)
    ctx = {"patch_file_size": 10, "patch_file_mtime": st.st_mtime}
    # the file the extraction measured: no line
    assert _ctrl(tmp_path, ctx, labels_path=str(lab))()[
        "predate_edits"] is False
    # saved again since (a Save rewrites it): the line, with no flag
    lab.write_bytes(b"y" * 11)
    assert _ctrl(tmp_path, ctx, labels_path=str(lab))()[
        "predate_edits"] is True
    # a sidecar from before the record: nothing to compare
    assert _ctrl(tmp_path, {}, labels_path=str(lab))()[
        "predate_edits"] is False


def test_the_controller_hands_the_plot_its_provider():
    import inspect

    from napari_epicycle.fusion_reconstruction import controller

    src = inspect.getsource(controller)
    assert "set_match_notes_provider" in src
    assert "_set_notes(self._match_plot_notes)" in src


# --------------------------------------------------------------------------
# the other run's name: a rename, markup, an empty answer
# --------------------------------------------------------------------------
def _descriptions():
    from napari_epicycle.segmentation.analysis_widget import (
        _FEATURE_DESCRIPTIONS,
    )
    return {c: _FEATURE_DESCRIPTIONS[c] for c in COLS}


def _tip(combo, text):
    from qtpy.QtCore import Qt
    return combo.itemData(combo.findText(text), Qt.ItemDataRole.ToolTipRole)


def test_the_match_id_tooltip_names_the_run_once():
    tip_id = T.column_tooltip("match_id", _descriptions()["match_id"],
                              "Nuclei")
    assert tip_id == (
        "match_id: The label, in the other run (Nuclei), of the object it "
        "shares the most voxels with (0 = none, too small, or not "
        "measured).")
    assert "((" not in tip_id and ") (" not in tip_id


def test_an_empty_answer_keeps_the_tooltips_it_had(wb):
    facts = {"other": "Nuclei", "descriptions": _descriptions()}
    wb.set_match_notes_provider(lambda: dict(facts))
    wb.set_features(_df())
    before = _tip(wb._feature_x_combo, "match_count")
    assert before.startswith("match_count: How many objects")
    facts.clear()            # the provider failed or said nothing
    wb._redraw()
    assert _tip(wb._feature_x_combo, "match_count") == before


def test_a_rename_reaches_the_legend_and_the_violin_note_in_one_redraw(wb):
    facts = {"other": "Nuclei", "descriptions": _descriptions()}
    wb.set_match_notes_provider(lambda: dict(facts))
    wb.set_features(_df())
    wb.plot_type_combo.setCurrentText("Violin")
    _pick_match(wb)
    assert wb._color_combo.currentText() == "Overlap with Nuclei"
    assert "Overlap with Nuclei" in wb._legend_label.text()
    assert wb._wb_axes_note.text().endswith("by Overlap with Nuclei")
    facts["other"] = "Nuclei v2"
    wb._redraw()             # one redraw, no second one
    assert wb._color_combo.currentText() == "Overlap with Nuclei v2"
    assert "Overlap with Nuclei v2" in wb._legend_label.text()
    assert wb._wb_axes_note.text().endswith("by Overlap with Nuclei v2")


def test_markup_in_a_run_name_is_shown_as_typed(wb):
    from qtpy.QtCore import Qt

    name = "<b>x</b>"
    wb.set_match_notes_provider(
        lambda: {"other": name, "stale": T.STALE_MATCH.format(other=name),
                 "descriptions": _descriptions()})
    wb.set_features(_df())
    wb.plot_type_combo.setCurrentText("Violin")
    _pick_match(wb)
    assert wb._color_combo.currentText() == "Overlap with <b>x</b>"
    # the axes note is plain text: the name is not read as markup
    assert wb._wb_axes_note.textFormat() == Qt.TextFormat.PlainText
    assert wb._wb_axes_note.text().endswith("by Overlap with <b>x</b>")
    # the legend escapes it
    assert "Overlap with &lt;b&gt;x&lt;/b&gt;" in wb._legend_label.text()
    # a tooltip with a name like this is rich text with the name escaped,
    # one line each
    seen = 0
    for col in COLS:
        tip = _tip(wb._feature_x_combo, col)
        if tip is None:
            continue
        seen += 1
        assert "<b>x</b>" not in tip, (col, tip)
        assert "&lt;b&gt;x&lt;/b&gt;" in tip, (col, tip)
        assert tip.startswith("<qt>") and "<br>" in tip, (col, tip)
    assert seen >= 4
    assert T.column_tooltip("match_count", "Against the other run.",
                            name) == (
        "<qt>match_count: Against the other run (&lt;b&gt;x&lt;/b&gt;)."
        "</qt>")
    # a plain name keeps the plain form
    assert T.column_tooltip("match_count", "Against the other run.",
                            "Nuclei") == (
        "match_count: Against the other run (Nuclei).")


def test_the_categorical_words_know_every_overlap_item():
    from napari_epicycle.fusion_reconstruction import qc_plot_widget as M

    s = M.QCPlotWidget._CATEGORICAL_COLOR_BY
    assert isinstance(s, M._ColourByWords)
    assert "Overlap with X" in s
    assert T.COLOUR_BY_MATCH_DEFAULT in s
    assert "QC tier" in s and "area" not in s


# --------------------------------------------------------------------------
# "QC status" is offered only with its column (Step 3)
# --------------------------------------------------------------------------
def test_qc_status_is_offered_only_when_the_table_has_it(wb, classic):
    from napari_epicycle.fusion_reconstruction._qc_plot_area import (
        _PLOT_TEXTS,
    )
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        STEP3_SCHEME,
        QCPlotWidget,
    )

    assert _PLOT_TEXTS["QC_STATUS_NEEDS_COLUMN"] is True
    df = _df()
    step3 = QCPlotWidget(layout="workbench", texts=_PLOT_TEXTS,
                         tier_scheme=STEP3_SCHEME)
    try:
        step3.set_features(df)
        assert "QC status" not in _items(step3._color_combo)
        assert "QC tier" in _items(step3._color_combo)
        step3.set_features(df.assign(qc_status="pass"))
        assert _items(step3._color_combo)[0] == "QC status"
    finally:
        step3.deleteLater()
    # a host without the rule (the bare words, the classic layout) keeps it
    for w in (wb, classic):
        w.set_features(df)
        assert _items(w._color_combo)[0] == "QC status"


# --------------------------------------------------------------------------
# the controller's provider: the run's current name, read once
# --------------------------------------------------------------------------
def _named_run_workspace(tmp_path, display_name):
    import json

    from napari_epicycle.fusion_reconstruction import _paths
    from packages.epicycle_io.step3_run_manifest import RUN_MANIFEST_FILENAME

    manifest = (_paths.step3_runs_root(tmp_path) / "nuclei"
                / RUN_MANIFEST_FILENAME)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"run_id": "nuclei",
                                    "display_name": display_name}),
                        encoding="utf-8")
    return manifest


def _provider_fake(tmp_path, read_ctx):
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController as FRC,
    )
    fake = SimpleNamespace(
        _qc_table_patch="patch_1.tif", working_dir=str(tmp_path),
        _match_values_predate_edits=False, _curation_dirty=False,
        _qc_features_df=_df(), _get_active_labels_path=lambda: "",
        _patch_extraction_context=read_ctx)
    fake._match_values_predate = (
        lambda c: FRC._match_values_predate(fake, c))
    return fake, (lambda: FRC._match_plot_notes(fake))


def test_the_provider_follows_a_rename_and_reads_once(tmp_path,
                                                      monkeypatch):
    import builtins
    import io

    from napari_epicycle.fusion_reconstruction import qc_pairing_group as G

    _named_run_workspace(tmp_path, "Nuclei")
    labels = tmp_path / "other_labels.tif"
    labels.write_bytes(b"x" * 10)
    ctx = {"match_run": "nuclei", "match_run_display_name": "Nuclei",
           "match_file": "other_labels.tif",
           "match_file_mtime": os.stat(labels).st_mtime,
           "match_file_size": 99}          # not the file now: stale
    reads = []

    def read_ctx(patch):
        reads.append(patch)
        return dict(ctx)

    _fake, notes_now = _provider_fake(tmp_path, read_ctx)
    notes = notes_now()
    assert notes["other"] == "Nuclei"
    assert notes["stale"] == T.STALE_MATCH.format(other="Nuclei")
    assert reads == ["patch_1.tif"]

    # a drag tick: the same answer, and not one file opened
    opened = []
    real_open = builtins.open

    def counting(*a, **k):
        opened.append(a[0] if a else k.get("file"))
        return real_open(*a, **k)

    listings = []
    real_listing = G.read_run_listing
    monkeypatch.setattr(G, "read_run_listing",
                        lambda ws: listings.append(ws) or real_listing(ws))
    with monkeypatch.context() as m:
        m.setattr(builtins, "open", counting)
        m.setattr(io, "open", counting)
        again = notes_now()
    assert again == notes
    assert opened == [] and listings == [] and reads == ["patch_1.tif"]

    # the other run renamed: the name and the stale line follow (the
    # plot's entry and tooltips take the provider's name)
    _named_run_workspace(tmp_path, "Nuclei (DAPI)")
    renamed = notes_now()
    assert renamed["other"] == "Nuclei (DAPI)"
    assert renamed["stale"] == T.STALE_MATCH.format(other="Nuclei (DAPI)")
    assert len(listings) == 1


def test_a_renamed_run_renames_the_plot_entry_and_its_tooltips(tmp_path, wb):
    ctx = {"match_run": "nuclei", "match_run_display_name": "Nuclei"}
    _named_run_workspace(tmp_path, "Nuclei")
    _fake, notes_now = _provider_fake(tmp_path, lambda patch: dict(ctx))
    wb.set_match_notes_provider(notes_now)
    wb.set_features(_df())
    assert "Overlap with Nuclei" in _items(wb._color_combo)
    _named_run_workspace(tmp_path, "Nuclei (DAPI)")
    wb._redraw()
    assert "Overlap with Nuclei (DAPI)" in _items(wb._color_combo)
    assert "the other run (Nuclei (DAPI))" in _tip(wb._feature_x_combo,
                                                   "match_count")


def test_the_provider_falls_back_to_the_recorded_name(tmp_path):
    # "nuclei" is not listed (deleted since): the name the sidecar recorded
    ctx = {"match_run": "nuclei", "match_run_display_name": "Nuclei"}
    fake, notes_now = _provider_fake(tmp_path, lambda patch: dict(ctx))
    assert notes_now()["other"] == "Nuclei"
    # the stock Default run, listed under its own name
    ctx2 = {"match_run": "default", "match_run_display_name": "Old name"}
    fake._patch_extraction_context = lambda patch: dict(ctx2)
    fake._qc_table_patch = "patch_2.tif"
    other = notes_now()["other"]
    assert other == "Default"
    assert T.colour_by_match(other) == "Overlap with the Default run"
