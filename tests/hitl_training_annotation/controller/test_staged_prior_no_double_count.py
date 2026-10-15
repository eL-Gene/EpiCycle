"""Q7: a slice trained in a prior round, reverted and re-finalized is staged
now AND has an old pair file; the "pairs in all" line must count it once."""
from __future__ import annotations

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    TrainingAnnotationController as C)
from napari_epicycle.hitl_training_annotation.ui import _step2_text as T
# Package import (tests/hitl_training_annotation/controller/ is a package):
# a sys.path.insert would stay for the rest of the session and import the
# helper module a second time under a top-level name.
from tests.hitl_training_annotation.controller.test_step2_prior_round_filter import (
    _accepted_dir, _make_ctrl, _write_pair)


class _Label:
    def __init__(self):
        self.text = ""

    def setText(self, t):
        self.text = t


def _setup(tmp_path):
    records = {
        "P1_XY_z0001": {"status": "staged", "bucket": "lateral"},
        "P1_XY_z0002": {"status": "committed", "bucket": "lateral"},
    }
    ctrl = _make_ctrl(tmp_path, records, rounds=2)
    d = _accepted_dir(tmp_path, 0)
    _write_pair(d, "P1", "XY", 1)
    _write_pair(d, "P1", "XY", 2)
    for name in ("_prior_pair_total_excluding_staged", "_staged_count_for",
                 "_staged_prior_for", "_write_staged_count_text"):
        setattr(ctrl, name, getattr(C, name).__get__(ctrl))
    return ctrl


def test_prior_total_excludes_slices_staged_now(tmp_path):
    ctrl = _setup(tmp_path)
    assert len(ctrl._collect_prior_round_pairs("lateral")) == 2  # unfiltered
    assert ctrl._prior_pair_total_excluding_staged("lateral") == 1


def test_staged_line_counts_the_reverted_slice_once(tmp_path):
    ctrl = _setup(tmp_path)
    lab = _Label()
    ctrl.ui.dataset_staging = lab
    ctrl._staged_prior_pairs = (
        "lateral", ctrl._prior_pair_total_excluding_staged("lateral"))
    ctrl._write_staged_count_text("lateral")
    assert lab.text.split("\n")[0] == T.staged_count_text(1, 2, 2, "lateral")
    assert "2 lateral pairs in all" in lab.text.split("\n")[0]


def test_no_staged_records_keeps_the_full_total(tmp_path):
    ctrl = _setup(tmp_path)
    ctrl.sample_records["P1_XY_z0001"]["status"] = "committed"
    assert ctrl._prior_pair_total_excluding_staged("lateral") == 2


def test_refresh_round_status_caches_the_filtered_total_and_shows_the_full_one(
        tmp_path):
    """The wiring: ``_refresh_round_status`` caches the staged-excluded total
    for the staged line while the Status ``total pairs`` stays unfiltered."""
    ctrl = _setup(tmp_path)
    shown = {}
    ctrl._ensure_runtime_defaults = lambda: None
    ctrl._iter_list_widget_items = lambda _w: []
    ctrl._safe_ui_set_text = lambda attr, value: shown.__setitem__(attr, value)
    ctrl._format_target_label = lambda: "Lateral (XY)"
    ctrl._last_checkpoint_text = lambda: "None"
    for name in ("_refresh_curation_model_status", "_refresh_training_stage_line",
                 "_refresh_init_hint"):
        setattr(ctrl, name, lambda: None)
    ctrl._refresh_model_name_display = lambda **_k: None
    C._refresh_round_status(ctrl)
    assert ctrl._staged_prior_pairs == ("lateral", 1)
    assert shown["status_total_pairs_value"] == 2
