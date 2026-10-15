"""Release hide list 2026-10-12: the 2D annotator hides Advanced Features."""
from __future__ import annotations

import inspect

from napari_epicycle.annotator.curation_2d_widget import Curation2DAnnotator
from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator


def test_2d_build_draw_group_hides_advanced_features_btn():
    src = inspect.getsource(Curation2DAnnotator._build_draw_group)
    assert "open_advanced_features_btn" in src
    assert "open_advanced_features_btn.setVisible(False)" in src


def test_3d_build_draw_group_does_not_hide_advanced_features_btn():
    src = inspect.getsource(Curation3DAnnotator._build_draw_group)
    assert "open_advanced_features_btn.setVisible(False)" not in src


def test_2d_guide_no_longer_mentions_advanced_features():
    from pathlib import Path
    p = (Path(__file__).resolve().parents[2] / "docs" / "guides"
         / "annotator_2d_quick_annotate.md")
    assert "Advanced Features" not in p.read_text(encoding="utf-8")


def test_multiviewer_settings_btn_hidden_in_base_and_3d():
    from napari_epicycle.annotator.native_multiview_widget import (
        NativeMultiviewAnnotator)
    for cls in (NativeMultiviewAnnotator, Curation3DAnnotator):
        src = inspect.getsource(cls._build_draw_group)
        assert "_open_multiviewer_settings_btn.setVisible(False)" in src, cls


def test_guides_no_longer_mention_multiviewer_settings():
    from pathlib import Path
    g = Path(__file__).resolve().parents[2] / "docs" / "guides"
    for p in g.glob("annotator_*.md"):
        assert "Multiviewer Settings" not in p.read_text(encoding="utf-8"), p
