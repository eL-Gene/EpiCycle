"""Model Refinement look, Brick B2 -- the lines around the list: "Will train
on" counts the SAME disk scan as ``_collect_training_pairs`` (I11), the run
line, the short warn chip with the full sentence as the tooltip."""
from __future__ import annotations

from types import SimpleNamespace

from napari_epicycle.fusion_reconstruction import _paths
from napari_epicycle.fusion_reconstruction import _step3_mr_text as M
from napari_epicycle.fusion_reconstruction import _step3_rows as rows
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController as Ctl, count_pairs_on_disk,
)


def _pairs(d, n, orphan=0):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"s_{i:04d}_img.tif").touch()
        (d / f"s_{i:04d}_mask.tif").touch()
    for i in range(orphan):
        (d / f"o_{i:04d}_img.tif").touch()          # no mask: not a pair


def _tree(root):
    _pairs(root / "patchA" / "spatial", 5, orphan=2)
    _pairs(root / "patchA" / "axial", 3)
    _pairs(root / "patchB" / "spatial", 4)
    (root / "patchC" / "spatial").mkdir(parents=True)        # empty patch
    _pairs(root / ".hidden" / "spatial", 9)                  # ignored
    _pairs(root / "spatial", 2)                              # legacy flat
    _pairs(root / "axial", 1)


def test_count_matches_collect_training_pairs(tmp_path):
    root = _paths.refinement_pairs_root(tmp_path)
    _tree(root)
    ctrl = Ctl.__new__(Ctl)
    ctrl.working_dir = str(tmp_path)
    n_lat, n_ax, k = count_pairs_on_disk(root)
    assert n_lat == len(ctrl._collect_training_pairs("Fine-tune Lateral")[0])
    assert n_ax == len(ctrl._collect_training_pairs("Fine-tune Axial")[0])
    assert (n_lat, n_ax) == (11, 4)
    assert k == 3            # patchA, patchB, and the legacy flat folders as one
    assert ctrl._count_pairs_on_disk() == (n_lat, n_ax, k)


def test_empty_or_missing_root(tmp_path):
    assert count_pairs_on_disk(tmp_path / "nope") == (0, 0, 0)
    root = _paths.refinement_pairs_root(tmp_path)
    root.mkdir(parents=True)
    assert count_pairs_on_disk(root) == (0, 0, 0)


def test_the_will_train_refresh_feeds_the_ui(tmp_path):
    root = _paths.refinement_pairs_root(tmp_path)
    _tree(root)
    got = []
    ctrl = Ctl.__new__(Ctl)
    ctrl.working_dir = str(tmp_path)
    ctrl.ui = SimpleNamespace(
        set_refine_will_train=lambda a, b, c: got.append((a, b, c)))
    ctrl._refresh_refine_will_train()
    assert got == [(11, 4, 3)]


def test_the_run_line(tmp_path):
    got = []
    ctrl = Ctl.__new__(Ctl)
    ctrl.ui = SimpleNamespace(set_refine_run_line=got.append)
    ctrl.state_manager = SimpleNamespace(get_active_run_id=lambda: "run_a")
    ctrl._run_display_name_for = lambda slug: "Wound v2"
    ctrl._refresh_refine_run_line()
    assert got == ["Wound v2"]


def test_the_warn_chip_is_short_and_the_sentence_is_the_tooltip():
    rec = rows.build_refine_row("p.tiff", "finalized", None,
                                n_lateral=12, n_axial=900)
    # two stacked chips; the lateral one is drawn amber (the warn flag)
    assert list(rec["chips"]) == [
        (M.MR_ROW_PAIRS_LATERAL.format(n="12"), True),
        (M.MR_ROW_PAIRS_AXIAL.format(m="900"), False)]
    assert M.MR_ROW_FEW_PAIRS_TIP.format(n=12) in rec["row_tip"]
    ok = rows.build_refine_row("p.tiff", "finalized", None,
                               n_lateral=50, n_axial=1)
    assert list(ok["chips"]) == [
        (M.MR_ROW_PAIRS_LATERAL.format(n="50"), False),
        (M.MR_ROW_PAIRS_AXIAL.format(m="1"), False)]
