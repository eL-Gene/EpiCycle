"""The Predict + Reconstruct list gets pictures (STEP3_PR_INCOMPLETE_BRICKS_
SPEC item 4, brick A1): the additive ``show_thumb`` keyword of
``StackRowDelegate`` (MODE_NARROW without it is what it was), the list's
picture through the one ``set_patch_thumbs`` feed, the picture surviving a
rebuild of the rows, and the controller's ``labels_for``.
Run one file at a time with ``-o addopts=""``.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")


@pytest.fixture(scope="module")
def app():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _D():
    from napari_epicycle.ui_shell import _stack_row_delegate as D
    return D


def _layout(delegate, rec, width=300):
    from qtpy.QtCore import QRect
    from qtpy.QtGui import QFont
    return delegate.layout(QRect(0, 0, width, 0), rec, QFont())


REC = {"name": "p.tif", "word": "curating", "axes": "CZYX"}


def test_a_narrow_row_without_show_thumb_is_what_it_was(app):
    D = _D()
    g = _layout(D.StackRowDelegate(D.MODE_NARROW), REC)
    assert "thumb" not in g
    same = _layout(D.StackRowDelegate(D.MODE_NARROW, show_thumb=False), REC)
    assert same["height"] == g["height"] and same["icon"] == g["icon"]
    assert same["mid"] == g["mid"]


def test_show_thumb_draws_a_picture_after_the_tick_and_makes_the_row_tall_enough(
        app):
    D = _D()
    plain = _layout(D.StackRowDelegate(D.MODE_NARROW), REC)
    pic = _layout(D.StackRowDelegate(D.MODE_NARROW, show_thumb=True), REC)
    assert pic["thumb"].width() == D.thumb_px(D.MODE_NARROW) == pic["thumb"].height()
    assert pic["thumb"].left() > pic["tick"].right()
    assert pic["icon"].left() > pic["thumb"].right()
    assert pic["height"] >= D.thumb_px(D.MODE_NARROW) + 2 * D.PAD_Y
    assert pic["height"] >= plain["height"]
    assert pic["mid"].left() > plain["mid"].left()


def test_a_wide_row_ignores_the_keyword(app):
    D = _D()
    a = _layout(D.StackRowDelegate(D.MODE_WIDE), REC)
    b = _layout(D.StackRowDelegate(D.MODE_WIDE, show_thumb=True), REC)
    assert a == b


def test_a_measure_row_draws_the_picture_when_asked(app):
    """A3: Step 4's ③ takes the 80 px picture; without the keyword it is
    what it was."""
    D = _D()
    a = _layout(D.StackRowDelegate(D.MODE_MEASURE), REC)
    b = _layout(D.StackRowDelegate(D.MODE_MEASURE, show_thumb=True), REC)
    assert "thumb" not in a and b["thumb"].width() == D.thumb_px(
        D.MODE_NARROW)


def test_step3s_pr_list_delegate_shows_the_picture(app):
    from napari_epicycle.fusion_reconstruction.ui import (
        FusionReconstructionCuration,
    )
    from qtpy.QtWidgets import QWidget
    w = FusionReconstructionCuration.__new__(FusionReconstructionCuration)
    QWidget.__init__(w)
    w.viewer = None
    w._build_ui()
    w._apply_theme()
    try:
        assert w._pr_row_delegate._show_thumb is True
        from qtpy.QtGui import QPixmap
        from qtpy.QtGui import QColor
        pm = QPixmap(8, 8)
        pm.fill(QColor("red"))
        w.refresh_patch_state_list({"a.tif": "reconstructed",
                                    "b.tif": "unprocessed"})
        w.set_patch_thumbs({"a.tif": (pm, "tip"), "b.tif": (None, "why")})
        lst = w._patch_state_list
        D = _D()
        got = {lst.item(i).data(D.Qt.UserRole): lst.item(i)
               for i in range(lst.count())}
        assert got["a.tif"].data(D.ROLE_THUMB).cacheKey() == pm.cacheKey()
        assert got["a.tif"].data(D.ROLE_THUMB_TIP) == "tip"
        assert got["b.tif"].data(D.ROLE_THUMB) is None
        # one picture arriving later reaches this list too
        w.set_patch_thumb("b.tif", pm, "t2")
        assert got["b.tif"].data(D.ROLE_THUMB).cacheKey() == pm.cacheKey()
        # a REBUILD of the rows keeps the pictures
        w.refresh_patch_state_list({"a.tif": "curating",
                                    "b.tif": "unprocessed"})
        again = {lst.item(i).data(D.Qt.UserRole): lst.item(i)
                 for i in range(lst.count())}
        assert again["a.tif"].data(D.ROLE_THUMB).cacheKey() == pm.cacheKey()
        assert again["b.tif"].data(D.ROLE_THUMB).cacheKey() == pm.cacheKey()
    finally:
        w.deleteLater()


# -- the controller's labels_for and stamp ---------------------------------

def _ctl(tmp_path, statuses, run_id=None):
    from napari_epicycle.fusion_reconstruction import controller as C

    class _SM:
        def get_patch_status(self, name):
            return statuses[name]

    ns = SimpleNamespace(working_dir=str(tmp_path), state_manager=_SM(),
                         _active_run_path_id=lambda: run_id)
    ns._S3_THUMB_LABEL_STATUSES = (
        C.FusionReconstructionController._S3_THUMB_LABEL_STATUSES)
    return C, ns


def _put_labels(tmp_path, name, run_id=None, ext=".tif"):
    from napari_epicycle.fusion_reconstruction import _paths
    stem = _paths.patch_labels_stem(str(tmp_path), name, run_id=run_id)
    stem.parent.mkdir(parents=True, exist_ok=True)
    f = stem.with_name(stem.name + ext)
    if ext == ".zarr":
        f.mkdir()
    else:
        f.write_bytes(b"x")
    return f


def test_labels_for_names_the_run_labels_of_a_reconstructed_patch(tmp_path):
    C, ns = _ctl(tmp_path, {"a.tif": "reconstructed", "b.tif": "predicted",
                            "c.tif": "finalized", "d.tif": "curating"})
    fa = _put_labels(tmp_path, "a.tif")
    _put_labels(tmp_path, "b.tif")              # a file, but not this run's
    fc = _put_labels(tmp_path, "c.tif", ext=".zarr")
    f = C.FusionReconstructionController._s3_thumb_labels_for
    assert Path(f(ns, "a.tif")) == fa
    assert f(ns, "b.tif") is None               # status says not yet
    assert Path(f(ns, "c.tif")) == fc
    assert f(ns, "d.tif") is None               # curating but no file


def test_labels_for_is_per_run(tmp_path):
    C, ns = _ctl(tmp_path, {"a.tif": "reconstructed"}, run_id=None)
    _put_labels(tmp_path, "a.tif", run_id="other")      # only another run's
    f = C.FusionReconstructionController._s3_thumb_labels_for
    assert f(ns, "a.tif") is None


def test_the_stamp_includes_the_labels_mtime():
    from napari_epicycle.fusion_reconstruction import controller as C
    from napari_epicycle.ui_shell._stack_thumbs import ThumbRequest
    a = ThumbRequest("p", "x/p.tif", "ZYX", 1, 2)
    b = a._replace(labels_path="l", labels_mtime_ns=77)
    st = C.FusionReconstructionController._s3_thumb_stamp
    assert st(a) != st(b) and st(b)[3] == 77 and st(a)[3] == 0


def test_the_pixmap_cache_keys_on_the_labels_mtime(app):
    from napari_epicycle.ui_shell._stack_thumbs import ThumbCache
    k = ThumbCache.key
    assert k("p", 1, 2) == k("p", 1, 2, 0) and len(k("p", 1, 2)) == 3
    assert k("p", 1, 2, 5) != k("p", 1, 2)
