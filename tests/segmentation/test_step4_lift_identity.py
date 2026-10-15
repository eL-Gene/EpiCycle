"""The R1 lift keeps every Step 4 name answering, and with the SAME object.

* ``_step4_text`` re-exports the settings words of
  ``ui_shell/_recon_settings_text``, and ``STEP4_HELP_LINES`` holds
  ``RECON_HELP_LINES`` (same keys, values and place);
* ``_step4_models`` re-exports the presets of ``ui_shell/_recon_presets``,
  the voxel-size resolver of
  ``data_preparation/_voxel_spacing``, and the model-file
  readers of ``ui_shell/_model_files``;
* ``segmentation/ui.py`` imports the form helpers of
  ``ui_shell/_recon_settings_widgets`` (its two private classes subclass
  the shared ones, so the widget tree keeps Step 4's class names);
* ``segmentation._stack_row_delegate`` / ``._stack_thumbs`` ARE the
  ``ui_shell`` modules (a ``sys.modules`` alias, not a star re-export), so
  underscore names and monkeypatches on the old path keep working.
"""
from __future__ import annotations

import sys

import pytest

import napari_epicycle  # noqa: F401  (the app's import order)
from napari_epicycle.segmentation import _step4_models as M
from napari_epicycle.segmentation import _step4_text as T
from napari_epicycle.ui_shell import _recon_presets as P
from napari_epicycle.ui_shell import _recon_settings_text as RT

TEXT_NAMES = [
    "SEG_SETTINGS_TITLE", "SEG_START_FROM_LABEL", "SEG_PRESET_EPICYCLE",
    "SEG_PRESET_USEG3D", "SEG_PRESET_CUSTOM", "SEG_INFERENCE_TITLE",
    "SEG_MEMBRANE_LABEL", "SEG_NUCLEI_LABEL", "SEG_CELL_WIDTH_LABEL",
    "SEG_CELL_WIDTH_AUTO", "SEG_CELL_WIDTH_SUFFIX", "SEG_CELL_WIDTH_PX",
    "SEG_NORMALIZE_LABEL", "SEG_RESCALE_LABEL", "SEG_ADVANCED_TITLE",
    "SEG_METHOD_DIRECT", "SEG_JOIN_TITLE", "SEG_JOIN_DESC", "SEG_SIZE_DESC",
    "SEG_SIZE_OFF_NOTE", "SEG_MAX_SIZE_NONE", "SEG_PROB_THRESH_AUTO",
    "PS_FLOWS_OFF_NOTE", "STEP4_ADVANCED_LABELS", "STEP4_ADVANCED_GROUPS",
    "STEP4_HIDDEN_KNOBS", "RECON_HELP_LINES",
]
PRESET_NAMES = [
    "STEP4_PRESET_EPICYCLE", "USEG3D_WIDGET_KEYS", "_NONE_AS_MINUS_ONE",
    "_useg3d_params", "_key_present", "_lookup", "_preset_from_groups",
    "useg3d_preset", "_STEP3_TOP_KEYS", "step3_recon_preset",
]
WIDGET_NAMES = ["ZWSP", "FORM_FIELD_W", "ENGINE_NAME_STYLE",
                "adv_label_html"]
# R2b: the voxel-size resolver moved into ``data_preparation/_voxel_spacing``
# ("The resolver").
# R2b part 2: ``read_step1_state`` joined it, so Step 3 never imports a
# Step 4 module for it.
# R7a: ``SPACING_SOURCES`` joined them (Step 3's ``patch_spacing`` map
# validates its source against it).
SPACING_NAMES = ["_real", "_positive", "same_spacing", "_read_voxel_spacing",
                 "_rel_to", "resolve_stack_spacing", "read_step1_state"]


def test_step4_models_re_exports_spacing_sources():
    """R7a lifted ``SPACING_SOURCES`` VERBATIM into
    ``data_preparation/_voxel_spacing`` (a tuple has no ``__module__``, so
    identity is the whole check)."""
    from napari_epicycle.data_preparation import _voxel_spacing as VS
    assert M.SPACING_SOURCES is VS.SPACING_SOURCES
    assert VS.SPACING_SOURCES == ("step1", "file", "user")
# R6a: the model-file readers moved into ``ui_shell/_model_files``,
# so Step 3's provenance reads them
# without importing a Step 4 module.
MODEL_FILE_NAMES = ["file_fingerprint", "_training_config_files",
                    "read_trained_channels"]


@pytest.mark.parametrize("name", TEXT_NAMES)
def test_step4_text_re_exports_the_shared_word(name):
    assert getattr(T, name) is getattr(RT, name)


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_step4_models_re_exports_the_shared_preset(name):
    assert getattr(M, name) is getattr(P, name)


@pytest.mark.parametrize("name", SPACING_NAMES)
def test_step4_models_re_exports_the_spacing_resolver(name):
    from napari_epicycle.data_preparation import _voxel_spacing as VS
    assert getattr(M, name) is getattr(VS, name)
    assert getattr(VS, name).__module__ == \
        "napari_epicycle.data_preparation._voxel_spacing"


@pytest.mark.parametrize("name", MODEL_FILE_NAMES)
def test_step4_models_re_exports_the_model_file_readers(name):
    from napari_epicycle.ui_shell import _model_files as MF
    assert getattr(M, name) is getattr(MF, name)
    assert getattr(MF, name).__module__ ==         "napari_epicycle.ui_shell._model_files"


def test_step4s_own_callers_reach_the_lifted_readers():
    """``finalized_fingerprint`` / ``model_fingerprints`` call
    ``file_fingerprint`` from ``_step4_models``' globals: the same object."""
    from napari_epicycle.ui_shell import _model_files as MF
    assert M.finalized_fingerprint.__globals__["file_fingerprint"] is         MF.file_fingerprint
    assert MF.read_trained_channels.__globals__["_training_config_files"]         is MF._training_config_files


def test_step4_keeps_spacing_sources_and_its_own_caller():
    """``SPACING_SOURCES`` still answers on ``_step4_models`` with the same
    value, and ``model_snapshot`` (which calls ``_rel_to`` from
    ``_step4_models``' globals) still resolves it.

    Until R7a this test PINNED ``not hasattr(VS, "SPACING_SOURCES")`` (R2b
    left the tuple Step 4's).  R7a flipped it DELIBERATELY: Step 3's
    per-patch ``patch_spacing`` map validates its ``source`` against the
    same tuple and may not import a Step 4 module, so it moved beside the
    resolver."""
    from napari_epicycle.data_preparation import _voxel_spacing as VS
    assert M.SPACING_SOURCES == ("step1", "file", "user")
    assert hasattr(VS, "SPACING_SOURCES")
    assert M.SPACING_SOURCES is VS.SPACING_SOURCES
    assert M.model_snapshot.__globals__["_rel_to"] is VS._rel_to


def test_a_monkeypatch_on_the_new_module_reaches_the_resolver(monkeypatch,
                                                              tmp_path):
    """The resolver reads its OWN module's ``_read_voxel_spacing``: a patch
    there changes what ``M.resolve_stack_spacing`` returns (and one on
    ``M`` would not -- the test files patch the new module)."""
    from napari_epicycle.data_preparation import _voxel_spacing as VS
    p = str(tmp_path / "none.tif")          # no file: the real read is None
    assert M.resolve_stack_spacing({}, None, p, str(tmp_path)) is None
    monkeypatch.setattr(VS, "_read_voxel_spacing", lambda _p: (3.0, .5, .5))
    assert M.resolve_stack_spacing({}, None, p, str(tmp_path)) == \
        {"zyx": [3.0, 0.5, 0.5], "source": "file"}


def test_step4_keeps_its_own_words():
    for name in ("SEG_SETTINGS_DESC", "SEG_PRESET_STEP3", "ROW_WIDE_LINE2",
                 "STEP4_HELP_LINES"):
        assert not hasattr(RT, name), name
    assert "newest_step3_manifest" in vars(M)
    assert not hasattr(P, "newest_step3_manifest")


def test_the_help_lines_hold_the_shared_run_in_its_place():
    keys = list(T.STEP4_HELP_LINES)
    shared = list(RT.RECON_HELP_LINES)
    i = keys.index(shared[0])
    assert keys[i:i + len(shared)] == shared
    assert keys[i - 1] == "chan2_combo" and keys[i + len(shared)] == \
        "predict_btn"
    for k, v in RT.RECON_HELP_LINES.items():
        assert T.STEP4_HELP_LINES[k] is v
    assert len(shared) == 46
    assert set(RT.STEP4_ADVANCED_LABELS) <= set(shared)
    assert {"cell_width_um_sb", "normalize_cb", "rescale_cb"} <= set(shared)
    for own in ("start_from_combo", "chan1_combo", "chan2_combo"):
        assert own in T.STEP4_HELP_LINES and own not in shared


def test_the_panel_imports_the_shared_form_helpers():
    from napari_epicycle.segmentation import ui as U
    from napari_epicycle.ui_shell import _recon_settings_widgets as W
    for name in WIDGET_NAMES:
        assert getattr(U, name) is getattr(W, name), name
    assert U.LOCK_LINE_STYLE == W.lock_line_style()
    assert issubclass(U._TickLabel, W.TickLabel)
    assert issubclass(U._LockLine, W.LockLine)


def test_the_lock_line_and_engine_names_are_parameters(qapp):
    from napari_epicycle.ui_shell import _recon_settings_widgets as W
    a = W.LockLine()
    b = W.LockLine(object_name="step3LockLine")
    try:
        assert a.objectName() == "step4LockLine"
        assert b.objectName() == "step3LockLine"
        assert "QFrame#step3LockLine" in b.styleSheet()
        assert "step4" not in b.styleSheet()
        c = W.collapsible("Advanced", "batch", engine_object_name="x")
        assert c._epicycle_engine_label.objectName() == "x"
        assert W.collapsible("Advanced", "batch") \
            ._epicycle_engine_label.objectName() == "step4EngineName"
    finally:
        a.deleteLater()
        b.deleteLater()


def test_adv_row_writes_the_label_into_the_output_dict(qapp):
    from qtpy.QtWidgets import QCheckBox, QSpinBox, QWidget
    from napari_epicycle.ui_shell import _recon_settings_widgets as W
    host = QWidget()
    try:
        g = W.frm4()
        host.setLayout(g)
        out = {}
        sb, cb = QSpinBox(), QCheckBox("x")
        lbl = W.adv_row(g, "gd_iters_sb", sb, out)
        tick = W.adv_row(g, "do_mp_cb", cb, out)
        assert out == {"gd_iters_sb": lbl, "do_mp_cb": tick}
        assert sb.width() == W.FORM_FIELD_W or sb.maximumWidth() == \
            W.FORM_FIELD_W
        assert isinstance(tick, W.TickLabel) and cb.text() == ""
        assert "Steps" in lbl.text()
        assert "n_iter" in lbl.text().replace(W.ZWSP, "")   # wraps at "_"
        other = W.adv_row(g, "k", QSpinBox(), None,
                          table={"k": ("Plain", "eng_name")})
        assert "Plain" in other.text()
    finally:
        host.deleteLater()


@pytest.mark.parametrize("name", ["_stack_row_delegate", "_stack_thumbs"])
def test_the_old_path_is_the_same_module_object(name):
    import importlib
    old = importlib.import_module(f"napari_epicycle.segmentation.{name}")
    new = importlib.import_module(f"napari_epicycle.ui_shell.{name}")
    assert old is new
    assert sys.modules[f"napari_epicycle.segmentation.{name}"] is new
    from napari_epicycle import segmentation
    assert getattr(segmentation, name) is new
    assert new.__file__.replace("\\", "/").endswith(f"ui_shell/{name}.py")


def test_a_monkeypatch_on_the_old_path_reaches_the_code(monkeypatch):
    from napari_epicycle.segmentation import _stack_thumbs as old
    from napari_epicycle.ui_shell import _stack_thumbs as new
    monkeypatch.setattr(old, "THUMB_SIZE", 7)
    assert new.THUMB_SIZE == 7
    from napari_epicycle.segmentation._stack_thumbs import (  # noqa: F401
        _CALLER_SLOTS, _LIVE_THUMB_WORKERS, _StackThumbWorker, _ThumbThread,
    )
    from napari_epicycle.segmentation._stack_row_delegate import (  # noqa
        _icon_pixmap, CHECK_DISABLED_STYLE,
    )


# ------------------------------------------- the panel's delegates (audit)

class _RecordingPainter:
    """Stands in for ``QPainter`` in ``StackRowDelegate.paint``: every call
    is accepted and every ``drawText`` text is kept, in order."""

    def __init__(self):
        self.texts = []

    def drawText(self, *args):
        if args and isinstance(args[-1], str):
            self.texts.append(args[-1])

    def __getattr__(self, name):
        def _any(*_a, **_k):
            return None
        return _any


def _row_option(lw, it):
    from qtpy.QtWidgets import QStyle, QStyleOptionViewItem
    opt = QStyleOptionViewItem()
    lw.initViewItemOption(opt)
    opt.rect = lw.visualItemRect(it)
    opt.widget = lw
    opt.state |= QStyle.StateFlag.State_Enabled
    return opt


def test_the_panel_builds_its_delegates_with_step4s_words(qapp, tmp_path,
                                                          monkeypatch):
    """The three construction sites in ``segmentation/ui.py`` pass
    ``help_lines=`` and ``wide_line2=``; the shared delegate's defaults
    (``{}`` / ``""``) would blank the wide row's line 2 and every
    painted-part tooltip without a single error.  Pinned on the REAL
    panel: the attributes, a tooltip per delegate, and the painted line 2
    of a wide Session row."""
    from qtpy.QtCore import QEvent, QPoint
    from qtpy.QtGui import QHelpEvent

    from napari_epicycle.ui_shell import _stack_row_delegate as D

    from ._step4_panel import close, make_workspace, panel, pump

    ws = make_workspace(tmp_path)
    w = panel(qapp, ws)
    try:
        delegates = {"session": w._session_delegate, "seg": w._seg_delegate,
                     "qc": w._qc_delegate}
        for name, dlg in delegates.items():
            assert dlg._help_lines is T.STEP4_HELP_LINES, name
            assert dlg._wide_line2 == T.ROW_WIDE_LINE2, name

        w.accordion.setCurrentIndex(0)
        pump(qapp, 10)
        lw = w.session_stacks_list
        assert lw.count() >= 1
        it = lw.item(0)
        rec = it.data(D.ROLE_RECORD)
        opt = _row_option(lw, it)
        idx = lw.model().index(0, 0)

        shown = []

        class _Tip:
            @staticmethod
            def showText(pos, text, view=None):
                shown.append(text)

            @staticmethod
            def hideText():
                shown.append(None)

        monkeypatch.setattr(D, "QToolTip", _Tip)
        g = w._session_delegate.layout(opt.rect, rec, opt.font)
        cases = [(w._session_delegate, g["tick"], "row_tick"),
                 (w._session_delegate, g["eye"], "row_eye"),
                 (w._qc_delegate, g["lines"]["l1"], "mt_row_click")]
        seg_g = w._seg_delegate.layout(opt.rect, rec, opt.font)
        cases.append((w._seg_delegate, seg_g["tick"], "row_tick"))
        for dlg, part, key in cases:
            shown.clear()
            ev = QHelpEvent(QEvent.Type.ToolTip, part.center(), QPoint(0, 0))
            assert dlg.helpEvent(ev, lw, opt, idx)
            assert shown and shown[-1] == T.STEP4_HELP_LINES[key], key

        painter = _RecordingPainter()
        w._session_delegate.paint(painter, opt, idx)
        # A3 fold: ①'s record carries its own line 2 (axes, size, channels,
        # cells -- Session's pattern); the template is Step 4's old form
        head = rec["line2"]
        assert head.strip()
        assert any(t.startswith(head) for t in painter.texts), painter.texts
    finally:
        close(w)
        pump(qapp, 5)
