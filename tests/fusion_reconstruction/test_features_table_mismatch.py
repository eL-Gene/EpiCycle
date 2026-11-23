"""A features table that no longer matches the labels is reported on load.

The real ``_reload_current_patch_features`` runs on the harness of the save
integrity tests, with the real ``_table_volume_mismatch``.  The notice is a
non-modal napari warning scheduled on the event loop; no dialog is opened.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from napari_epicycle.fusion_reconstruction import controller as _ctl
from napari_epicycle.fusion_reconstruction.qc_features import save_patch_csv

from tests.fusion_reconstruction.test_curation_save_integrity import (
    _ControllerHarness,
    _bind_real_methods,
)

PATCH = "test_patch.tif"


class _Timer:
    queue: list = []

    @staticmethod
    def singleShot(ms, fn):
        _Timer.queue.append(fn)


@pytest.fixture
def env(tmp_path, monkeypatch):
    h = _ControllerHarness(tmp_path)
    _bind_real_methods(h, [
        "_reload_current_patch_features",
        "_current_downsample_factor",
        "_patch_layer_key",
        "_check_csv_volume_consistency",
        "_scan_qc_df_for_duplicates",
        # 2026-10-31 (cross-run pairing): the reload reads the
        # extraction-context sidecar.  Never raises on a fake.
        "_patch_extraction_context",
    ])
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController as _C,
    )
    for _name in ("_table_volume_mismatch", "_check_table_against_volume"):
        if hasattr(_C, _name):
            _bind_real_methods(h, [_name])
    h._DEBUG_CSV_VOLUME_CONSISTENCY = False
    h._check_qc_extraction_staleness = lambda pn: None
    h._maybe_show_qc_staleness_notification = lambda pn: None
    h._qc_staleness_notified = set()
    h._qc_table_mismatch = False
    h._qc_view_patch_name = PATCH
    h._cur_patch = PATCH
    h._patch_name = PATCH
    (tmp_path / "artifacts" / "step3" / "qc").mkdir(parents=True, exist_ok=True)

    _Timer.queue = []
    monkeypatch.setattr(_ctl, "QTimer", _Timer)
    warnings = []
    import napari.utils.notifications as notif
    monkeypatch.setattr(notif, "show_warning", lambda msg: warnings.append(msg))

    class _NoBoxMeta(type):
        def __getattr__(cls, name):
            raise AssertionError(f"QMessageBox.{name} was used")

    class _NoBox(metaclass=_NoBoxMeta):
        """Any use of QMessageBox, static or instance, fails the test."""

        def __init__(self, *a, **k):
            raise AssertionError("a modal dialog was opened")

    monkeypatch.setattr(_ctl, "QMessageBox", _NoBox)
    return SimpleNamespace(h=h, tmp=tmp_path, warnings=warnings)


def _volume(n=100):
    """n 3x3x3 cells on a grid in a (6, 60, 60) volume; label i at grid i."""
    vol = np.zeros((6, 60, 60), dtype=np.uint32)
    pos = {}
    for i in range(n):
        gy, gx = divmod(i, 10)
        y0, x0 = 6 * gy, 6 * gx
        vol[1:4, y0:y0 + 3, x0:x0 + 3] = i + 1
        pos[i + 1] = (2, y0 + 1, x0 + 1)
    return vol, pos


def _setup(env, n=100, n_miss=0, factor=1, vol_shape_override=None,
           layer_name="curation_3d_test_patch", dup=False, rows=None):
    vol, pos = _volume(n)
    if rows is not None:
        n = rows
    labels = list(range(1, n + 1))
    cent = []
    for k, lab in enumerate(labels):
        z, y, x = pos[lab]
        if k < n_miss:
            z = 5  # nowhere near the cell (cell spans z 1..3)
        cent.append((z * factor, y * factor, x * factor))
    df = pd.DataFrame({
        "label": labels,
        "centroid-0": [c[0] for c in cent],
        "centroid-1": [c[1] for c in cent],
        "centroid-2": [c[2] for c in cent],
        "qc_tier": ["good"] * n,
        "label_status": ["active"] * n,
        "feature_state": ["fresh"] * n,
    })
    if dup:
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    save_patch_csv(df, str(env.tmp / "artifacts" / "step3" / "qc"
                           / "test_patch_features.csv"))
    h = env.h
    if vol_shape_override is not None:
        vol = np.zeros(vol_shape_override, dtype=np.uint32)
    h._labels_layer = SimpleNamespace(
        name=layer_name, data=vol, refresh=lambda: None
    )
    h.state_manager.get_patch_downsample_factor = MagicMock(return_value=factor)
    h.state_manager.get_patch_original_labels_shape = MagicMock(
        return_value=(6 * factor, 60 * factor, 60 * factor) if factor > 1 else None
    )
    return h


def _run(env):
    env.h._reload_current_patch_features()
    pending, _Timer.queue[:] = list(_Timer.queue), []
    for fn in pending:
        fn()
    assert not any("check failed" in m for m in env.h._log), env.h._log
    assert not any(m.startswith("[WARN]") for m in env.h._log), env.h._log


def _stale_lines(env):
    return [m for m in env.h._log if m.startswith("[QC-STALE]")]


def test_healthy_table_is_silent(env, capsys):
    _setup(env)
    _run(env)
    assert env.h._qc_table_mismatch is False
    assert _stale_lines(env) == []
    assert env.warnings == [] and _Timer.queue == []
    assert "[QC-TABLE] patch=test_patch.tif rows=100 misses=0 dropped=0 factor=1" in (
        capsys.readouterr().out
    )


def test_most_positions_missing_flags_and_warns_once(env):
    _setup(env, n_miss=80)
    env.h._reload_current_patch_features()
    # the warning is deferred to the event loop, not shown during the load
    assert env.warnings == []
    assert len(_Timer.queue) == 1
    pending, _Timer.queue[:] = list(_Timer.queue), []
    for fn in pending:
        fn()
    assert env.h._qc_table_mismatch is True
    lines = _stale_lines(env)
    assert len(lines) == 1
    assert ("[QC-STALE] Patch 'test_patch.tif': the features table no longer "
            "matches the labels (80 of 100 saved cell positions miss). "
            "Re-extract to refresh.") == lines[0]
    assert env.warnings == [
        "Patch 'test_patch.tif': the features table no longer matches the "
        "labels (edits were discarded or not saved). "
        "Re-extract the features to refresh it."
    ]


def test_few_misses_do_not_flag(env):
    _setup(env, n_miss=8)
    _run(env)
    assert env.h._qc_table_mismatch is False
    assert _stale_lines(env) == []


def test_small_table_floor_of_ten(env):
    _setup(env, n=8, n_miss=3, rows=8)
    _run(env)
    assert env.h._qc_table_mismatch is False


def test_dropped_duplicate_rows_flag(env):
    _setup(env, dup=True)
    _run(env)
    assert env.h._qc_table_mismatch is True
    assert len(_stale_lines(env)) == 1


def test_duplicates_only_wording(env):
    _setup(env, dup=True)
    _run(env)
    assert _stale_lines(env) == [
        "[QC-STALE] Patch 'test_patch.tif': the features table had 1 "
        "duplicate rows (removed on load). Re-extract to refresh."
    ]
    assert env.warnings == [
        "Patch 'test_patch.tif': the features table had 1 duplicate rows "
        "(removed on load). Re-extract to refresh."
    ]


def test_downsampled_patch_with_full_res_centroids_is_healthy(env, capsys):
    _setup(env, factor=2)
    _run(env)
    assert env.h._qc_table_mismatch is False
    assert "misses=0 dropped=0 factor=2" in capsys.readouterr().out


def test_wrong_volume_shape_for_the_factor_is_skipped(env, capsys):
    _setup(env, factor=2, vol_shape_override=(12, 120, 120), n_miss=0)
    _run(env)
    assert env.h._qc_table_mismatch is False
    out = capsys.readouterr().out
    assert "[QC-TABLE] skipped:" in out


def test_other_patchs_layer_is_skipped(env, capsys):
    _setup(env, n_miss=100, layer_name="curation_3d_other_patch")
    _run(env)
    assert env.h._qc_table_mismatch is False
    assert "[QC-TABLE] skipped:" in capsys.readouterr().out


def test_second_reload_schedules_the_warning_once(env):
    _setup(env, n_miss=80)
    _run(env)
    _run(env)
    assert len(env.warnings) == 1
    assert env.h._qc_staleness_notified == {(PATCH, "table-mismatch")}


def test_no_modal_dialog_is_opened(env):
    _setup(env, n_miss=80)
    _run(env)  # the patched exec_/exec raise if reached
    assert len(env.warnings) == 1


def _csv(env):
    return env.tmp / "artifacts" / "step3" / "qc" / "test_patch_features.csv"


def test_negative_centroid_counts_as_a_miss(env, capsys):
    _setup(env)
    df = pd.read_csv(_csv(env))
    df.loc[0, "centroid-0"] = -1
    save_patch_csv(df, str(_csv(env)))
    _run(env)
    assert "rows=100 misses=1 dropped=0" in capsys.readouterr().out


def test_one_voxel_off_the_cell_edge_is_a_hit(env, capsys):
    _setup(env)
    df = pd.read_csv(_csv(env))
    df.loc[0, "centroid-0"] = 4  # the cell spans z 1..3
    save_patch_csv(df, str(_csv(env)))
    _run(env)
    assert "rows=100 misses=0 dropped=0" in capsys.readouterr().out


def test_no_active_layer_skips_and_sets_no_flag(env, capsys):
    h = _setup(env, n_miss=80)
    h._labels_layer = None
    _run(env)
    assert "[QC-TABLE] skipped: no active labels layer" in capsys.readouterr().out
    assert h._qc_table_mismatch is False
    assert _stale_lines(env) == [] and env.warnings == []


def test_no_centroid_columns_skips_and_sets_no_flag(env, capsys):
    h = _setup(env, n_miss=80)
    df = pd.read_csv(_csv(env)).drop(columns=["centroid-0"])
    save_patch_csv(df, str(_csv(env)))
    _run(env)
    assert "[QC-TABLE] skipped: no centroid columns" in capsys.readouterr().out
    assert h._qc_table_mismatch is False
    assert _stale_lines(env) == [] and env.warnings == []


def test_extract_all_patch_done_clears_flag_and_key(env):
    h = _setup(env, n_miss=80)
    _run(env)
    assert h._qc_table_mismatch is True
    assert (PATCH, "table-mismatch") in h._qc_staleness_notified
    _bind_real_methods(h, ["_on_qc_batch_patch_done"])
    h._qc_batch_launch_epoch = None
    h._record_extraction_spacing = lambda pn: None
    h._write_qc_extraction_context = lambda pn: None
    h._on_qc_batch_patch_done(PATCH, 100)
    assert h._qc_table_mismatch is False
    assert (PATCH, "table-mismatch") not in h._qc_staleness_notified


def test_patch_without_a_csv_resets_the_flag(env):
    h = _setup(env, n_miss=80)
    _run(env)
    assert h._qc_table_mismatch is True
    _csv(env).unlink()
    h._reload_current_patch_features()
    assert h._qc_table_mismatch is False


def test_warning_is_dropped_when_the_patch_in_view_changed(env):
    h = _setup(env, n_miss=80)
    h._reload_current_patch_features()
    assert len(_Timer.queue) == 1
    h._qc_view_patch_name = "other_patch.tif"
    pending, _Timer.queue[:] = list(_Timer.queue), []
    for fn in pending:
        fn()
    assert env.warnings == []
    assert (PATCH, "table-mismatch") not in h._qc_staleness_notified
