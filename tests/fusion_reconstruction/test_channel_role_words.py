"""The two Cellpose channel boxes are named by what they do for the model.

A nuclei model segments the nuclear stain, so a first box called
"Membrane channel" told the user to put the nuclei in the wrong box.  The
boxes are "Channel to segment:" and "Helper channel:" in Step 3's settings,
Step 3 Model Refinement's Advanced block and Step 4's Segment section, and
a model's trained channels are named "segment" / "helper".

Rules held here:

* No string on screen in those panels names the boxes by biology
  ("Membrane channel", "Nuclei channel", "Auxiliary channel") or a role in
  the old "membrane (" / "nuclei (" form.  The scan reads every string
  literal that is not a docstring in the text modules and the two panels.
  Three other families rightly keep the biological words and are NOT
  scanned: the EDA Workbench's per-channel feature categories
  (``segmentation/analysis_widget.py``), the Add tab's Markers / Walls rows
  and sentences (``annotator/advanced_features_widget.py``,
  ``packages/epicycle_annotator/add_preview.py``), and Refine's nuclei row.
* One channel, one name: each Model card names a trained channel by the
  item its own box shows for it, never by a bare 0-based number.
* A one-channel (ZYX) patch is listed with the new labels in place and the
  Step 3 items unchanged.
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "napari_epicycle"

#: the modules whose strings reach the two Steps' channel surfaces
SCANNED = (
    "ui_shell/_recon_settings_text.py",
    "fusion_reconstruction/_step3_pr_text.py",
    "fusion_reconstruction/_step3_mr_text.py",
    "fusion_reconstruction/ui.py",
    "segmentation/_step4_text.py",
    "segmentation/ui.py",
)
#: the retired words (lower case; matched case-insensitively)
FORBIDDEN = ("membrane channel", "nuclei channel", "auxiliary channel",
             "membrane (", "nuclei (")


def _docstring_nodes(tree) -> set:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None) or []
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                ids.add(id(body[0].value))
    return ids


def _strings(source: str):
    """Every non-docstring string literal of *source*, f-string parts
    included."""
    tree = ast.parse(source)
    docs = _docstring_nodes(tree)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docs):
            yield node.lineno, node.value


def hits(source: str) -> list:
    out = []
    for line, text in _strings(source):
        low = text.lower()
        for word in FORBIDDEN:
            if word in low:
                out.append((line, word, text[:80]))
    return out


@pytest.mark.parametrize("rel", SCANNED)
def test_no_panel_string_names_the_boxes_by_biology(rel):
    path = PKG / rel
    assert path.is_file(), rel
    found = hits(path.read_text(encoding="utf-8"))
    assert not found, (rel, found)


def test_the_scan_has_teeth():
    src = (PKG / SCANNED[0]).read_text(encoding="utf-8")
    planted = src + '\nPLANTED = "Membrane channel:"\n'
    assert hits(planted)
    planted = src + '\nPLANTED = ("Optional " "auxiliary channel")\n'
    assert hits(planted)
    planted = src + '\nPLANTED = "{i}: nuclei (model)"\n'
    assert hits(planted)
    # a docstring is not on screen
    assert not hits('"""Membrane channel, once."""\nX = 1\n')


def test_the_labels_are_the_role_words():
    from napari_epicycle.fusion_reconstruction import _step3_mr_text as MR
    from napari_epicycle.segmentation import _step4_text as S4
    from napari_epicycle.ui_shell import _recon_settings_text as R
    assert R.SEG_MEMBRANE_LABEL == "Channel to segment:"
    assert R.SEG_NUCLEI_LABEL == "Helper channel:"
    # one shared constant: Step 4 says the same words
    assert S4.SEG_MEMBRANE_LABEL is R.SEG_MEMBRANE_LABEL
    assert S4.SEG_NUCLEI_LABEL is R.SEG_NUCLEI_LABEL
    # Model Refinement says them too
    assert (MR.MR_CHAN1_LABEL, MR.MR_CHAN2_LABEL) == (
        R.SEG_MEMBRANE_LABEL, R.SEG_NUCLEI_LABEL)
    assert S4.SEG_PICK_CHANNEL == "Pick the channel to segment."


# ------------------------------------------ one channel, one name (Step 3)
def _step3_parts(text: str) -> list:
    """The channel parts of a Step 3 trained line ("Trained on {names}.",
    no Step 2 run tail)."""
    prefix = "Trained on "
    assert text.startswith(prefix), text
    body = text[len(prefix):].rstrip(".")
    return [p.strip() for p in body.split(", ")]


def _bare_index(text: str) -> bool:
    return bool(re.search(r"\(\d\)", text))


@pytest.mark.parametrize("c1", range(4))
@pytest.mark.parametrize("c2", range(4))
def test_the_step3_card_names_each_channel_by_its_box_item(c1, c2):
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as PT
    from napari_epicycle.fusion_reconstruction import _step3_rows as RW
    rec = {"kind": "refined", "trained_channels": [c1, c2], "run_id": "m"}
    text, known = RW.trained_line(rec, dict(rec))
    assert known
    if c1 == 0:
        # gray: every channel averaged, the helper ignored -- neither named
        assert text == PT.S3_MODEL_TRAINED_GRAY.format(run="")
        assert text == ("Trained on one grayscale image (all channels "
                        "averaged).")
        return
    seg = PT.S3_MODEL_CHANNEL.format(role=PT.S3_MODEL_ROLE_MEMBRANE,
                                     item=PT.S3_CHAN1_ITEMS[c1])
    want = [seg]
    if c2 > 0:
        want.append(PT.S3_MODEL_CHANNEL.format(
            role=PT.S3_MODEL_ROLE_NUCLEI, item=PT.S3_CHAN2_ITEMS[c2]))
    assert _step3_parts(text) == want
    assert not _bare_index(text), text


def test_the_step3_card_check_has_teeth():
    """The retired form -- the role and a 0-based number, "membrane (1)" for
    code 2 -- names a different channel than the box's "2: green"."""
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as PT
    old = "Trained on membrane ({index}).".format(index=2 - 1)
    want = PT.S3_MODEL_CHANNEL.format(role=PT.S3_MODEL_ROLE_MEMBRANE,
                                      item=PT.S3_CHAN1_ITEMS[2])
    assert _step3_parts(old) != [want]
    assert _bare_index(old)


def test_two_different_models_name_both_sets_in_the_box_words():
    from napari_epicycle.fusion_reconstruction import _step3_rows as RW
    lat = {"kind": "refined", "trained_channels": [2, 1], "run_id": "m"}
    ax = {"kind": "refined", "trained_channels": [1, 0], "run_id": "m"}
    text, _ = RW.trained_line(lat, ax)
    assert text == ("Trained on lateral segment 2: green, helper 1: red · "
                    "axial segment 1: red.")
    # a gray model among the two: named as gray, with no helper
    gray = {"kind": "refined", "trained_channels": [0, 2], "run_id": "m"}
    text, _ = RW.trained_line(gray, ax)
    assert text == ("Trained on lateral gray (all channels averaged) · "
                    "axial segment 1: red.")


# ------------------------------------------ one channel, one name (Step 4)
@pytest.mark.parametrize("names", [None, ["DAPI", "Membrane", None]])
@pytest.mark.parametrize("c1", range(4))
@pytest.mark.parametrize("c2", range(4))
def test_the_step4_card_names_each_channel_by_its_box_item(c1, c2, names):
    """Step 4's boxes count by Cellpose code, as Step 3's do, and show the
    shown stack's layer name after the code ("2: Membrane"), else Step 3's
    word for the code; the card names each channel by that same item."""
    from napari_epicycle.segmentation import _step4_models as M
    from napari_epicycle.segmentation import _step4_text as T
    from napari_epicycle.ui_shell import _recon_settings_text as RT

    def item(code, helper=False):
        name = (names[code - 1] if names and code - 1 < len(names)
                else None)
        if name:
            return T.SEG_CHANNEL_ITEM.format(code=code, name=name)
        words = RT.SEG_CODE_ITEMS_HELPER if helper else (
            RT.SEG_CODE_ITEMS_SEGMENT)
        return words[code]

    text = M.trained_channels_text([c1, c2], names)
    assert "channels:" not in text and "1 channels" not in text
    if c1 == 0:
        # gray: every channel averaged, the helper ignored -- no channel
        # (and no position 0) is named
        assert text == ("Trained on one grayscale image (all channels "
                        "averaged).")
        return
    want = [T.SEG_MODEL_CHANNEL.format(role=T.SEG_MODEL_ROLE_MEMBRANE,
                                       item=item(c1))]
    if c2 > 0:
        want.append(T.SEG_MODEL_CHANNEL.format(
            role=T.SEG_MODEL_ROLE_NUCLEI, item=item(c2, helper=True)))
    assert text == "Trained on " + ", ".join(want) + "."
    assert not _bare_index(text), text


def test_the_step4_card_example():
    from napari_epicycle.segmentation import _step4_models as M
    # Step 3's card's words ("segment 2: green, helper 1: red")
    assert M.trained_channels_text([2, 1]) == (
        "Trained on segment 2: green, helper 1: red.")
    assert M.trained_channels_text([2, 1], ["DAPI", "Membrane"]) == (
        "Trained on segment 2: Membrane, helper 1: DAPI.")
    assert M.trained_channels_text([1, 0]) == "Trained on segment 1: red."


# ------------------------------------------- a one-channel patch is listed
@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def ui(app):
    from qtpy.QtWidgets import QWidget

    from napari_epicycle.fusion_reconstruction.ui import (
        FusionReconstructionCuration,
    )
    w = FusionReconstructionCuration.__new__(FusionReconstructionCuration)
    QWidget.__init__(w)
    w.viewer = None
    w._build_ui()
    yield w
    w.deleteLater()


def test_a_one_channel_patch_is_listed_beside_the_new_labels(ui, tmp_path):
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QLabel

    from napari_epicycle.fusion_reconstruction import _step3_pr_text as PT
    from napari_epicycle.ui_shell import _recon_settings_text as R

    from ._one_channel import ZYX_SHAPE, write_zyx_patch

    path = write_zyx_patch(tmp_path)
    import tifffile
    assert tifffile.imread(str(path)).shape == ZYX_SHAPE
    name = path.name
    ui.refresh_patch_state_list(
        {name: "unprocessed", "two_ch.tif": "unprocessed"},
        axes_by_patch={name: "ZYX", "two_ch.tif": "ZCYX"},
        shapes={name: ZYX_SHAPE, "two_ch.tif": (4, 2, 8, 8)})
    lst = ui._patch_state_list
    listed = {lst.item(i).data(Qt.UserRole) for i in range(lst.count())}
    assert name in listed
    texts = {lb.text() for lb in ui._pr_settings.findChildren(QLabel)}
    assert R.SEG_MEMBRANE_LABEL in texts and R.SEG_NUCLEI_LABEL in texts
    # the items are Step 3's Cellpose codes, unchanged
    assert tuple(ui.chan1_combo.itemText(i)
                 for i in range(ui.chan1_combo.count())) == PT.S3_CHAN1_ITEMS
    assert tuple(ui.chan2_combo.itemText(i)
                 for i in range(ui.chan2_combo.count())) == PT.S3_CHAN2_ITEMS
    assert (ui.chan1_combo.currentIndex(), ui.chan2_combo.currentIndex()) == (
        2, 1)
