"""The Set spacing… dialog lifted into ``ui_shell`` (R7a).

* the words arrive through ``texts=`` -- none is Step 4's unless Step 4
  passes it;
* ``show_z_orientation=False`` / ``show_axes=False`` (Step 3's call) build
  neither row, and ``values()`` leaves both keys out;
* Step 4's ``segmentation._set_spacing_dialog.SetSpacingDialog`` is a thin
  subclass whose tree equals the golden captured BEFORE the lift
  (``tests/segmentation/golden/step4_spacing_dialog.json``);
* the names Step 4's module exported are the shared module's objects.

Run one file at a time with ``-o addopts=""``.
"""
from __future__ import annotations

import json
import os

import pytest

import napari_epicycle  # noqa: F401  (the app's import order)


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


#: words of no step, so a leak of Step 4's would show
TEXTS = {
    "title": "T-title", "desc": "T-desc {n}", "desc_one": "T-one {name}",
    "z_label": "T-z", "y_label": "T-y", "x_label": "T-x",
    "zorient_label": "T-zo", "zorient_items": ("T-down", "T-up"),
    "axes_label": "T-axes", "axes_auto": "T-auto",
    "cancel": "T-cancel", "ok": "T-ok",
}
#: Step 3's call needs no row words
STEP3_KEYS = ("title", "desc", "desc_one", "z_label", "y_label", "x_label",
              "cancel", "ok")


def _texts_on(dlg):
    from qtpy.QtWidgets import QAbstractButton, QLabel
    out = [dlg.windowTitle()]
    for w in dlg.findChildren((QLabel, QAbstractButton)):
        out.append(w.text())
    return out


def _form_rows(dlg):
    from qtpy.QtWidgets import QFormLayout
    forms = dlg.findChildren(QFormLayout)
    assert len(forms) == 1
    return forms[0].rowCount()


def test_the_words_come_in_through_texts(qapp):
    from napari_epicycle.ui_shell._set_spacing_dialog import SetSpacingDialog
    dlg = SetSpacingDialog(None, names=["a.tif"], record=None, texts=TEXTS)
    try:
        shown = _texts_on(dlg)
        for key in ("title", "z_label", "y_label", "x_label", "zorient_label",
                    "axes_label", "cancel", "ok"):
            assert TEXTS[key] in shown, key
        assert "T-one a.tif" in shown
        assert [dlg.z_orientation_combo.itemText(i) for i in range(2)] == \
            ["T-down", "T-up"]
        assert dlg.axes_combo.itemText(0) == "T-auto"
        assert not any("µm" in s or "Set spacing" in s for s in shown)
        assert _form_rows(dlg) == 5
    finally:
        dlg.deleteLater()
    dlg = SetSpacingDialog(None, names=["a", "b"], texts=TEXTS)
    try:
        assert "T-desc 2" in _texts_on(dlg)
    finally:
        dlg.deleteLater()


def test_step3s_call_hides_both_rows(qapp):
    from napari_epicycle.ui_shell._set_spacing_dialog import SetSpacingDialog
    texts = {k: TEXTS[k] for k in STEP3_KEYS}
    dlg = SetSpacingDialog(
        None, names=["p.tif"],
        record={"spacing": {"zyx": [3.0, 0.4, 0.4], "source": "user"},
                "z_orientation": "basal_up", "axes_override": "ZCYX"},
        texts=texts, show_z_orientation=False, show_axes=False)
    try:
        assert dlg.z_orientation_combo is None
        assert dlg.axes_combo is None
        from qtpy.QtWidgets import QComboBox
        assert not dlg.findChildren(QComboBox)
        assert _form_rows(dlg) == 3
        assert dlg.values() == {"zyx": [3.0, 0.4, 0.4]}
    finally:
        dlg.deleteLater()


@pytest.mark.parametrize("hide", ["show_z_orientation", "show_axes"])
def test_each_keyword_hides_its_own_row(qapp, hide):
    from napari_epicycle.ui_shell._set_spacing_dialog import SetSpacingDialog
    dlg = SetSpacingDialog(None, names=["p"], texts=TEXTS, **{hide: False})
    try:
        assert _form_rows(dlg) == 4
        vals = dlg.values()
        if hide == "show_z_orientation":
            assert dlg.z_orientation_combo is None
            assert "z_orientation" not in vals and "axes_override" in vals
        else:
            assert dlg.axes_combo is None
            assert "axes_override" not in vals and "z_orientation" in vals
    finally:
        dlg.deleteLater()


def test_a_missing_word_is_refused(qapp):
    from napari_epicycle.ui_shell._set_spacing_dialog import SetSpacingDialog
    texts = {k: TEXTS[k] for k in STEP3_KEYS if k != "ok"}
    with pytest.raises(ValueError, match="ok"):
        SetSpacingDialog(None, names=["p"], texts=texts,
                         show_z_orientation=False, show_axes=False)
    # Step 3's subset is complete when both rows are hidden ...
    SetSpacingDialog(None, names=["p"], texts={k: TEXTS[k] for k in
                                               STEP3_KEYS},
                     show_z_orientation=False,
                     show_axes=False).deleteLater()
    # ... and not when a row is shown
    with pytest.raises(ValueError, match="axes"):
        SetSpacingDialog(None, names=["p"],
                         texts={k: TEXTS[k] for k in STEP3_KEYS},
                         show_z_orientation=False)


def test_step4s_subclass_is_the_golden(qapp):
    from tests.segmentation import _step4_goldens as G
    from tests.segmentation.test_step4_lift_goldens import _first_difference
    want = json.loads(G.SPACING_DIALOG_GOLDEN.read_text(encoding="utf-8"))
    got = G.build_spacing_dialog_states(qapp)
    assert want == got, _first_difference(want, got)


def test_step4s_module_is_a_thin_subclass():
    from napari_epicycle.segmentation import _set_spacing_dialog as S4
    from napari_epicycle.ui_shell import _set_spacing_dialog as SH
    assert issubclass(S4.SetSpacingDialog, SH.SetSpacingDialog)
    assert S4.SetSpacingDialog is not SH.SetSpacingDialog
    assert S4.Z_ORIENTATIONS is SH.Z_ORIENTATIONS
    assert S4.AXES_CHOICES is SH.AXES_CHOICES
    assert set(S4.step4_spacing_texts()) == set(SH.SPACING_TEXT_KEYS)
    # the subclass adds no method of its own beyond __init__
    own = {k for k, v in vars(S4.SetSpacingDialog).items()
           if callable(v) and not k.startswith("__")}
    assert not own, own
