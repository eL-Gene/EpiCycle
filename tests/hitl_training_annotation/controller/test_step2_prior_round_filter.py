"""B-lite: cumulative training only inherits slices the user accepted.

Journey items #1 + #7: a round's ``accepted/`` directory doubles as
prediction scratch AND training truth (every prediction and every
per-stroke debounce is persisted there), and ``_collect_prior_round_pairs``
globbed it with no status filter -- so abandoned raw predictions, edits in
progress, and discarded-then-resurrected slices all trained as ground truth
in the NEXT cumulative round.

B-lite is a READER-side filter only: no storage change, no directory
change, no reconciliation (that is Part B of
deferred).

All fakes are Qt-free.
"""

from __future__ import annotations

from pathlib import Path

from napari_epicycle.hitl_training_annotation import _paths as step2_paths
from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    TrainingAnnotationController,
)

TRAINABLE = ("accepted", "staged", "committed")


class _LogOutput:
    def __init__(self):
        self.lines = []

    def append(self, text):
        self.lines.append(text)


class _UI:
    def __init__(self):
        self.training_log_output = _LogOutput()


def _write_pair(out_dir: Path, stack: str, view: str, idx: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{stack}_{view}_z{idx:04d}"
    img = out_dir / f"{base}_img.tif"
    img.write_bytes(b"img")
    (out_dir / f"{base}_mask.tif").write_bytes(b"mask")
    return img


def _write_legacy_pair(out_dir: Path, stack: str, view: str, idx: int) -> Path:
    """VERY-LEGACY naming: ``{stack}_{VIEW}_{idx}_img.tif`` (no z prefix)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{stack}_{view}_{idx}"
    img = out_dir / f"{base}_img.tif"
    img.write_bytes(b"img")
    (out_dir / f"{base}_mask.tif").write_bytes(b"mask")
    return img


def _make_ctrl(tmp_path, records, rounds=2):
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.ui = _UI()
    ctrl.working_dir = str(tmp_path)
    ctrl.active_run_id = step2_paths.DEFAULT_RUN_ID
    ctrl.current_round_by_view = {"lateral": rounds, "axial": rounds}
    ctrl.selected_view = "XY"
    ctrl.sample_records = records
    ctrl._append_ui_log = TrainingAnnotationController._append_ui_log.__get__(ctrl)
    ctrl._current_round_key = lambda: "lateral"
    ctrl._current_round = lambda: rounds
    ctrl._infer_view_from_label = (
        TrainingAnnotationController._infer_view_from_label.__get__(ctrl)
    )
    ctrl._infer_slice_index = (
        TrainingAnnotationController._infer_slice_index.__get__(ctrl)
    )
    ctrl._slice_key_for_pair_filename = (
        TrainingAnnotationController._slice_key_for_pair_filename.__get__(ctrl)
    )
    ctrl._collect_prior_round_pairs = (
        TrainingAnnotationController._collect_prior_round_pairs.__get__(ctrl)
    )
    return ctrl


def _accepted_dir(tmp_path, round_idx):
    return step2_paths.round_accepted_dir(
        tmp_path, step2_paths.DEFAULT_RUN_ID, "lateral", round_idx
    )


def _basenames(pairs):
    return sorted(Path(p["img_uri"]).name for p in pairs)


# ---------------------------------------------------------------------------
# The filter
# ---------------------------------------------------------------------------

def test_only_accepted_staged_committed_slices_feed_cumulative_training(tmp_path):
    records = {
        "P1_XY_z0001": {"status": "accepted"},
        "P1_XY_z0002": {"status": "staged"},
        "P1_XY_z0003": {"status": "committed"},
        "P1_XY_z0004": {"status": "predicted"},   # raw prediction, never touched
        "P1_XY_z0005": {"status": "curating"},    # edits in progress
        # z0006 has NO record at all: discarded, then resurrected from the
        # prior round's surviving copy (journey #7).
    }
    out = _accepted_dir(tmp_path, 0)
    for idx in range(1, 7):
        _write_pair(out, "P1", "XY", idx)

    ctrl = _make_ctrl(tmp_path, records)
    pairs = ctrl._collect_prior_round_pairs(bucket="lateral")

    assert _basenames(pairs) == [
        "P1_XY_z0001_img.tif",
        "P1_XY_z0002_img.tif",
        "P1_XY_z0003_img.tif",
    ]


def _skip_lines(ctrl):
    return [l for l in ctrl.ui.training_log_output.lines if "left out" in l]


def _mixed_round(tmp_path):
    records = {
        "P1_XY_z0001": {"status": "accepted"},
        "P1_XY_z0002": {"status": "predicted"},
        "P1_XY_z0003": {"status": "curating"},
    }
    out = _accepted_dir(tmp_path, 1)
    for idx in range(1, 5):  # z0004 has no record at all
        _write_pair(out, "P1", "XY", idx)
    return records


def test_training_logs_one_line_per_kind_per_bucket_and_round(tmp_path):
    ctrl = _make_ctrl(tmp_path, _mixed_round(tmp_path))
    ctrl._collect_prior_round_pairs(bucket="lateral", log_skips=True)

    lines = _skip_lines(ctrl)
    assert len(lines) == 2, ctrl.ui.training_log_output.lines
    unaccepted, unknown = lines
    assert "round_01 (lateral)" in unaccepted
    assert "2 slice(s) you had not accepted yet" in unaccepted
    assert "round_01 (lateral)" in unknown
    assert "1 slice(s) EpiCycle has no curation record for" in unknown


def test_status_refresh_logs_nothing(tmp_path):
    """`_refresh_round_status` calls this on every accept / predict / load /
    status tick — the summary must not flood the training log."""
    ctrl = _make_ctrl(tmp_path, _mixed_round(tmp_path))

    ctrl._collect_prior_round_pairs(bucket="lateral")  # default log_skips=False

    assert _skip_lines(ctrl) == []
    assert ctrl.ui.training_log_output.lines == []


def test_no_unknown_record_line_when_every_skip_is_a_known_status(tmp_path):
    records = {
        "P1_XY_z0001": {"status": "accepted"},
        "P1_XY_z0002": {"status": "predicted"},
    }
    out = _accepted_dir(tmp_path, 0)
    _write_pair(out, "P1", "XY", 1)
    _write_pair(out, "P1", "XY", 2)

    ctrl = _make_ctrl(tmp_path, records)
    ctrl._collect_prior_round_pairs(bucket="lateral", log_skips=True)

    lines = _skip_lines(ctrl)
    assert len(lines) == 1
    assert "no curation record" not in lines[0]


def test_no_log_line_when_every_prior_pair_is_accepted(tmp_path):
    records = {"P1_XY_z0001": {"status": "accepted"}}
    _write_pair(_accepted_dir(tmp_path, 0), "P1", "XY", 1)

    ctrl = _make_ctrl(tmp_path, records)
    pairs = ctrl._collect_prior_round_pairs(bucket="lateral")

    assert len(pairs) == 1
    assert _skip_lines(ctrl) == []


def test_legacy_slice_label_format_still_resolves(tmp_path):
    """Pre-2026-05 records are keyed ``{stack}_{VIEW}_{idx}`` (no z-pad)."""
    records = {"P1_XY_7": {"status": "accepted"}}
    _write_pair(_accepted_dir(tmp_path, 0), "P1", "XY", 7)

    ctrl = _make_ctrl(tmp_path, records)
    pairs = ctrl._collect_prior_round_pairs(bucket="lateral")

    assert _basenames(pairs) == ["P1_XY_z0007_img.tif"]


def test_underscored_stack_names_are_not_confused(tmp_path):
    records = {
        "my_patch_v2_XY_z0001": {"status": "accepted"},
        "my_patch_v2_XY_z0010": {"status": "predicted"},
    }
    out = _accepted_dir(tmp_path, 0)
    _write_pair(out, "my_patch_v2", "XY", 1)
    _write_pair(out, "my_patch_v2", "XY", 10)

    ctrl = _make_ctrl(tmp_path, records)
    pairs = ctrl._collect_prior_round_pairs(bucket="lateral")

    assert _basenames(pairs) == ["my_patch_v2_XY_z0001_img.tif"]


# ---------------------------------------------------------------------------
# Regression locks -- the scan itself is unchanged
# ---------------------------------------------------------------------------

def test_legacy_tiers_are_still_scanned(tmp_path):
    """Very-legacy `training_data/round_NN/accepted_pairs/spatial` + the
    legacy-split tier must still resolve for accepted slices."""
    records = {
        "P1_XY_z0001": {"status": "accepted"},
        "P1_XY_z0002": {"status": "accepted"},
        "P1_XY_z0003": {"status": "predicted"},
    }
    very_legacy = tmp_path / "training_data" / "round_00" / "accepted_pairs" / "spatial"
    _write_pair(very_legacy, "P1", "XY", 1)
    _write_pair(very_legacy, "P1", "XY", 3)
    legacy_split = (
        step2_paths.legacy_split_round_dir(
            tmp_path, step2_paths.DEFAULT_RUN_ID, "lateral", 1
        )
        / "accepted"
    )
    _write_pair(legacy_split, "P1", "XY", 2)

    ctrl = _make_ctrl(tmp_path, records)
    pairs = ctrl._collect_prior_round_pairs(bucket="lateral")

    assert _basenames(pairs) == ["P1_XY_z0001_img.tif", "P1_XY_z0002_img.tif"]


def test_latest_round_wins_is_preserved(tmp_path):
    records = {"P1_XY_z0001": {"status": "accepted"}}
    _write_pair(_accepted_dir(tmp_path, 0), "P1", "XY", 1)
    newer = _write_pair(_accepted_dir(tmp_path, 1), "P1", "XY", 1)

    ctrl = _make_ctrl(tmp_path, records)
    pairs = ctrl._collect_prior_round_pairs(bucket="lateral")

    assert len(pairs) == 1
    assert Path(pairs[0]["img_uri"]) == newer


def test_unparsable_filename_is_skipped_without_raising(tmp_path):
    records = {"P1_XY_z0001": {"status": "accepted"}}
    out = _accepted_dir(tmp_path, 0)
    _write_pair(out, "P1", "XY", 1)
    (out / "junk_img.tif").write_bytes(b"img")
    (out / "junk_mask.tif").write_bytes(b"mask")

    ctrl = _make_ctrl(tmp_path, records)
    pairs = ctrl._collect_prior_round_pairs(bucket="lateral")

    assert _basenames(pairs) == ["P1_XY_z0001_img.tif"]


def test_very_legacy_filenames_are_matched_to_their_records(tmp_path):
    """`{stack}_{VIEW}_{idx}_img.tif` (pre-Phase-6c) has no `z` prefix.

    It fails `parse_slice_label`'s regex, so without the tolerant fallback a
    legacy workspace silently loses EVERY prior pair.
    """
    records = {
        "P1_XY_5": {"status": "accepted"},
        "P1_XY_6": {"status": "predicted"},
    }
    legacy = tmp_path / "training_data" / "round_00" / "accepted_pairs" / "spatial"
    _write_legacy_pair(legacy, "P1", "XY", 5)
    _write_legacy_pair(legacy, "P1", "XY", 6)

    ctrl = _make_ctrl(tmp_path, records)
    pairs = ctrl._collect_prior_round_pairs(bucket="lateral")

    assert _basenames(pairs) == ["P1_XY_5_img.tif"]


def test_very_legacy_filename_matches_a_z_padded_record(tmp_path):
    """The two formats must meet in the middle: same (stack, view, idx)."""
    records = {"P1_XY_z0005": {"status": "accepted"}}
    legacy = tmp_path / "training_data" / "round_00" / "accepted_pairs" / "spatial"
    _write_legacy_pair(legacy, "P1", "XY", 5)

    ctrl = _make_ctrl(tmp_path, records)
    pairs = ctrl._collect_prior_round_pairs(bucket="lateral")

    assert _basenames(pairs) == ["P1_XY_5_img.tif"]
