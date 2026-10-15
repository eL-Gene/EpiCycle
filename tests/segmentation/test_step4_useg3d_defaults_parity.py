"""Step 4's u-Segment3D defaults equal Step 3's.

Both widgets' UIs are built offscreen WITHOUT their controllers (``__new__`` +
``QWidget.__init__`` + ``_build_ui``), then every ``ps_* / gd_* / ccp_* /
post_* / ccg_* / cc_* / sf_*`` spin box and check box, plus ``do_mp_cb`` and
``skip_postprocess_cb``, present on BOTH sides is compared by attribute name
(value, range, single step, decimals).  Names present on one side only are
printed and FAIL (``test_no_name_is_on_one_side_only``; Step 3 R2a audit A
LOW-5 -- the docstring said so before the test did).
``qapp`` comes from ``tests/segmentation/conftest.py``.
"""

from __future__ import annotations

import os
import re

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

_PREFIX = re.compile(r"^((ps|gd|ccp|post|ccg|cc|sf)_[a-z0-9_]+"
                     r"|do_mp_cb|skip_postprocess_cb)$")


def _build(cls):
    from qtpy.QtWidgets import QWidget

    w = cls.__new__(cls)
    QWidget.__init__(w)
    w.viewer = None
    w._build_ui()
    return w


def _defaults(widget) -> dict:
    from qtpy.QtWidgets import QAbstractButton, QAbstractSpinBox, QComboBox

    out = {}
    for name, obj in vars(widget).items():
        if not _PREFIX.match(name):
            continue
        if isinstance(obj, QAbstractSpinBox) and hasattr(obj, "value"):
            out[name] = ("spin", obj.value(), obj.minimum(), obj.maximum(),
                         obj.singleStep(),
                         obj.decimals() if hasattr(obj, "decimals") else None)
        elif isinstance(obj, QAbstractButton) and obj.isCheckable():
            out[name] = ("check", obj.isChecked())
        elif isinstance(obj, QComboBox):
            out[name] = ("combo", obj.currentText())
    return out


@pytest.fixture(scope="module")
def tables(qapp):
    from napari_epicycle.fusion_reconstruction.ui import FusionReconstructionCuration
    from napari_epicycle.segmentation.ui import SegmentStack

    s3 = _build(FusionReconstructionCuration)
    s4 = _build(SegmentStack)
    try:
        yield _defaults(s3), _defaults(s4)
    finally:
        s3.deleteLater()
        s4.deleteLater()


def test_both_sides_have_the_groups(tables):
    s3, s4 = tables
    for prefix in ("ps_", "gd_", "ccp_", "post_", "ccg_", "cc_", "sf_"):
        assert any(k.startswith(prefix) for k in s3), f"Step 3 has no {prefix}*"
        assert any(k.startswith(prefix) for k in s4), f"Step 4 has no {prefix}*"


def test_pre_smoothing_defaults_are_step3s(tables):
    s3, s4 = tables
    ps = sorted(k for k in s3 if k.startswith("ps_") and k in s4)
    assert ps, "no shared ps_* controls"
    assert s4["ps_sigma_perp_sb"][1] == pytest.approx(3.0)
    assert s4["ps_sigma_inplane_sb"][1] == pytest.approx(1.0)
    for k in ps:
        assert s4[k] == s3[k], f"{k}: Step 4 {s4[k]} != Step 3 {s3[k]}"


def test_every_shared_default_matches(tables):
    s3, s4 = tables
    shared = sorted(set(s3) & set(s4))
    only3 = sorted(set(s3) - set(s4))
    only4 = sorted(set(s4) - set(s3))
    print(f"\n[PARITY] shared={len(shared)} only_step3={only3} only_step4={only4}")
    diffs = {k: (s3[k], s4[k]) for k in shared if s3[k] != s4[k]}
    assert not diffs, f"Step 3 vs Step 4 defaults differ: {diffs}"


def test_the_two_named_ticks_are_compared(tables):
    s3, s4 = tables
    for name in ("do_mp_cb", "skip_postprocess_cb"):
        assert name in s3 and name in s4, name


def test_no_name_is_on_one_side_only(tables):
    s3, s4 = tables
    only3 = sorted(set(s3) - set(s4))
    only4 = sorted(set(s4) - set(s3))
    assert not only3 and not only4, (
        f"one-sided names: only Step 3 {only3}, only Step 4 {only4}")
