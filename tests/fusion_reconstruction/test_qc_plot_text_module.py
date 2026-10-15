"""Step 3's QC plot words.

``_qc_plot_text.py`` is Qt-free, every constant is a string, the help
tables carry the keys the plot and the area need, and the plot module
imports neither it nor Step 4's text (a Step 3 module never depends on a
host's sentence table; the host hands the words in through ``texts=``).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TEXT = ROOT / "napari_epicycle" / "fusion_reconstruction" / "_qc_plot_text.py"
PLOT = ROOT / "napari_epicycle" / "fusion_reconstruction" / "qc_plot_widget.py"

AREA_HELP_KEYS = ("Histogram tab", "Selection tab", "Divider", "Triage by",
                  "Colour bars by", "Reset lines", "Gate", "Assign",
                  "Undo assign", "Clear", "Colours in viewer", "Status",
                  "Ghosts legend")


def _imports(path: Path) -> list:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.append((node.module or "") + ":" + ",".join(
                a.name for a in node.names))
        elif isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
    return names


def test_the_text_module_is_qt_free():
    names = _imports(TEXT)
    assert not [n for n in names if any(q in n for q in (
        "qtpy", "PyQt", "PySide", "napari", "pyqtgraph"))], names
    # loading the FILE in a fresh interpreter loads no Qt binding (the
    # packages' own __init__ files are not this module's to answer for)
    code = ("import sys, importlib.util as u; "
            "s = u.spec_from_file_location('t', sys.argv[1]); "
            "m = u.module_from_spec(s); s.loader.exec_module(m); "
            "print(any(k.split('.')[0] in ('PyQt5','PyQt6','PySide2',"
            "'PySide6','qtpy') for k in sys.modules))")
    out = subprocess.run([sys.executable, "-I", "-c", code, str(TEXT)],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", out.stdout


def test_every_constant_is_a_string_or_a_table_of_strings():
    from napari_epicycle.fusion_reconstruction import _qc_plot_text as T

    seen = 0
    for name in dir(T):
        if not name.isupper():
            continue
        value = getattr(T, name)
        seen += 1
        if isinstance(value, dict):
            for k, v in value.items():
                assert isinstance(k, str), (name, k)
                if isinstance(v, dict):
                    assert all(isinstance(a, str) and isinstance(b, str)
                               for a, b in v.items()), (name, k)
                else:
                    assert isinstance(v, str), (name, k)
        elif isinstance(value, tuple):
            assert all(isinstance(v, str) for v in value), name
        else:
            assert isinstance(value, str), name
    assert seen >= 40


def test_one_noun_one_spelling():
    """Labels, never cells; colours, never colors."""
    from napari_epicycle.fusion_reconstruction import _qc_plot_text as T

    for name in dir(T):
        if not name.isupper():
            continue
        value = getattr(T, name)
        texts = []
        if isinstance(value, dict):
            for v in value.values():
                texts.extend(v.values() if isinstance(v, dict) else [v])
        elif isinstance(value, tuple):
            texts.extend(value)
        else:
            texts.append(value)
        for t in texts:
            assert "color" not in t.lower().replace("colour", ""), (name, t)
            assert "cell" not in t.lower(), (name, t)


def test_spec_words_are_verbatim():
    from napari_epicycle.fusion_reconstruction import _qc_plot_text as T

    assert T.UMAP_EMPTY == ("No UMAP yet.|Press UMAP under QC Features. It "
                            "takes about 45 s for 3,000 labels.")
    assert T.TSNE_EMPTY == ("No t-SNE yet.|Press tSNE under QC Features. It "
                            "takes about 20 s for 3,000 labels.")
    assert T.VIOLIN_GROUPS == "{feat} by {by}"
    assert T.GHOSTS_WORD == "Ghosts"
    assert T.GHOSTS_LEGEND == "ghosts"
    assert T.UNMARKED_WORD == "unassigned"
    assert T.STATUS_IDLE_HINT == "Lasso or click points to select."
    assert T.STATUS_ISOLATED.format(n=97) == "Viewer: 97 labels shown alone"
    assert T.STATUS_ALL_SHOWN == "Viewer: showing every label"
    assert T.STATUS_ISOLATION_FAILED == (
        "Isolation failed — toggle Isolate off and retry.")
    assert T.STATUS_ASSIGNED.format(n=15, tier="review") == (
        "Assigned 15 labels to review.")
    assert T.STATUS_UNDONE == "Undid the last assignment."
    assert T.PLOT_NO_PATCH == ("Load a patch to see its QC plot.|Click a "
                               "patch in the list on the left.")
    assert T.PLOT_NO_FEATURES == ("No QC features for this patch yet.|Press "
                                  "Extract under QC Features.")
    assert T.SEE_NO_PATCH == "No patch loaded."
    assert T.SEE_NO_FEATURES == "No QC features yet."
    assert T.SEE_LOADING.format(patch="p3") == "Loading p3…"
    assert T.HIST_NO_DATA == "No QC data yet."
    assert T.HIST_NO_VALUES.format(feature="area") == (
        "No values for area on this patch.")
    assert T.HIST_DRAG_NOTE == ("Dragging a line re-sorts every label that "
                                "is not curated, hand assignments included.")
    assert T.HIST_ALL_CURATED.format(n=842) == (
        "All 842 labels are curated; the lines change nothing.")
    assert T.CARD_MSG_LASSO.format(n=15) == "Lasso: 15 labels."
    assert T.CARD_MSG_GATE.format(n=402, feature="area", lo="1,020",
                                  hi="3,400") == (
        "Gate: 402 labels with area between 1,020 and 3,400.")
    assert T.CARD_MSG_LEGEND.format(tier="review", n=12) == (
        "Legend: review, 12 labels.")
    assert T.CARD_NONE == (
        "No labels selected. Drag a lasso, click a point, click a colour in "
        "the legend, or tick Gate under Histogram (any feature but "
        "qc_score).")
    assert T.ACTION_ASSIGN_LABEL.format(n=15) == "Assign 15 selected to"
    assert (T.ACTION_ASSIGN, T.ACTION_UNDO, T.ACTION_CLEAR) == (
        "Assign", "Undo assign", "✕ Clear")
    assert (T.HIST_TRIAGE_BY, T.HIST_COLOUR_BARS_BY, T.HIST_RESET_LINES,
            T.HIST_GATE) == ("Triage by", "Colour bars by", "Reset lines",
                             "Gate")
    assert (T.TAB_HISTOGRAM, T.TAB_SELECTION) == ("Histogram", "Selection")
    assert T.COLOURS_IN_VIEWER == "Colours in viewer"
    parts = T.SEE_SEP.join(T.CARD_TIER_COUNT.format(tier=t, n=n) for t, n in (
        ("curated", 842), ("good", 0), ("review", 0), ("bad", 0),
        ("garbage", 0)))
    assert T.count_text(837, T.CARD_TOTALS, T.CARD_TOTALS_ONE_GHOST,
                        parts=parts) == (
        "All labels: curated 842 · good 0 · review 0 · bad 0 · garbage 0, "
        "plus 837 ghosts")
    assert T.count_text(1, T.CARD_TOTALS, T.CARD_TOTALS_ONE_GHOST,
                        parts="good 3") == "All labels: good 3, plus 1 ghost"
    assert T.CARD_MSG_CLICK.format(id=17) == "Click: label 17."
    assert T.count_text(4, T.CARD_MSG_ECHO, T.CARD_MSG_ECHO_ONE) == (
        "From the Label Navigator or the viewer: 4 labels.")
    assert T.count_text(1, T.CARD_MSG_ECHO, T.CARD_MSG_ECHO_ONE) == (
        "From the Label Navigator or the viewer: 1 label.")
    assert T.count_text(9, T.CARD_MSG_KEPT, T.CARD_MSG_KEPT_ONE) == (
        "Kept from the last plot: 9 labels.")
    assert T.count_text(1, T.CARD_MSG_KEPT, T.CARD_MSG_KEPT_ONE) == (
        "Kept from the last plot: 1 label.")
    assert T.STEP3_PLOT_HELP["Undo assign"] == (
        "Take back the last tier change: an assignment, a line drag or "
        "Reset lines.")
    assert T.STEP3_PLOT_TEXTS["GROUP_WORD"] == "C{id}"
    assert T.STEP3_PLOT_TEXTS["NOISE_WORD"] == "noise"


def test_count_text_singular_and_thousands():
    from napari_epicycle.fusion_reconstruction import _qc_plot_text as T

    assert T.count_text(1, T.SEE_LABELS, T.SEE_LABELS_ONE) == "1 label"
    assert T.count_text(3000, T.SEE_LABELS, T.SEE_LABELS_ONE) == "3,000 labels"
    assert T.count_text(1, T.STATUS_ASSIGNED, T.STATUS_ASSIGNED_ONE,
                        tier="bad") == "Assigned 1 label to bad."


def test_the_help_tables_have_their_keys():
    from napari_epicycle.fusion_reconstruction import _qc_plot_text as T
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        WB_HELP_KEYS,
        WB_TEXT_KEYS,
    )

    assert set(T.STEP3_PLOT_TEXTS["help"]) == set(WB_HELP_KEYS)
    for k in WB_TEXT_KEYS:
        assert k in T.STEP3_PLOT_TEXTS, k
    assert tuple(T.STEP3_PLOT_HELP) == AREA_HELP_KEYS
    for table in (T.STEP3_PLOT_TEXTS["help"], T.STEP3_PLOT_HELP):
        for k, line in table.items():
            assert line.endswith("."), (k, line)
            assert len(line) <= 160, (k, len(line))
    # the optional keys stay optional: Step 4's table still validates
    for k in ("GHOSTS_WORD", "GHOSTS_LEGEND", "UNMARKED_WORD"):
        assert k not in WB_TEXT_KEYS


def test_step3_texts_validate_in_the_constructor():
    from tests.fusion_reconstruction import _qc_plot_snap as S

    S.app()
    from napari_epicycle.fusion_reconstruction._qc_plot_text import (
        STEP3_PLOT_TEXTS,
    )
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        QCPlotWidget,
    )

    pw = QCPlotWidget(layout="workbench", texts=STEP3_PLOT_TEXTS)
    try:
        assert pw._history_cb.text() == "Ghosts"
        assert pw._select_btn.toolTip() == STEP3_PLOT_TEXTS["help"]["Lasso"]
    finally:
        pw.deleteLater()
    broken = dict(STEP3_PLOT_TEXTS)
    broken.pop("MAP_AXES")
    with pytest.raises(ValueError):
        QCPlotWidget(layout="workbench", texts=broken)


def test_the_plot_module_imports_neither_host_text():
    """W1-10 extended: neither Step 4's table nor Step 3's."""
    names = _imports(PLOT)
    assert not [n for n in names if "segmentation" in n
                or "_workbench_text" in n or "_qc_plot_text" in n
                or "ui_shell" in n], names
