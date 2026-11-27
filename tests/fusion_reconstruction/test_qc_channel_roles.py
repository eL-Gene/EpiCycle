"""The QC Extraction window's Intensity channels rows say each channel's
role in the active run.

The patch files carry no channel names, so a row names its channel by
number and by what the run did with it: "Channel 0 (channel to segment)",
"Channel 1 (helper channel)", or "Channel 2" alone.  The column prefix
("ch0_") moves to the row's tooltip.  The roles come from the two Cellpose
codes the run's records last asked for; a Cellpose code is not an array
index (code 0 is gray / none, code k >= 1 is the image's channel k - 1).
The checkboxes, their keys and ``get_channel_configs()`` do not change.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from napari_epicycle.fusion_reconstruction import _step3_pr_text as T  # noqa: E402
from napari_epicycle.fusion_reconstruction._step3_channels import (  # noqa: E402
    ROLE_HELPER,
    ROLE_SEGMENT,
    channel_roles,
)


# --- channel_roles: the pure rule ------------------------------------------

def test_no_codes_names_no_role():
    assert channel_roles(None, 3) == {}
    assert channel_roles("bad", 3) == {}
    assert channel_roles((1,), 3) == {}


def test_gray_names_no_role():
    # the channel to segment on 0: every channel averaged, helper unused
    assert channel_roles((0, 0), 3) == {}
    assert channel_roles((0, 2), 3) == {}


def test_a_code_is_the_channel_after_it():
    assert channel_roles((1, 0), 3) == {0: ROLE_SEGMENT}
    assert channel_roles((2, 1), 3) == {1: ROLE_SEGMENT, 0: ROLE_HELPER}
    assert channel_roles([3, 2], 3) == {2: ROLE_SEGMENT, 1: ROLE_HELPER}


def test_a_code_past_the_last_channel_is_left_out():
    assert channel_roles((3, 1), 2) == {0: ROLE_HELPER}
    assert channel_roles((1, 3), 2) == {0: ROLE_SEGMENT}
    # not a Cellpose code at all: no role anywhere
    assert channel_roles((4, 1), 5) == {}
    assert channel_roles((-1, 1), 3) == {}
    assert channel_roles((True, 0), 3) == {}


def test_one_channel_names_no_role():
    # a one-channel patch ran as gray whatever it asked for
    assert channel_roles((1, 0), 1) == {}
    assert channel_roles((2, 1), 0) == {}
    assert channel_roles((2, 1), None) == {}


def test_a_helper_on_the_channel_to_segment_is_not_named_twice():
    assert channel_roles((2, 2), 3) == {1: ROLE_SEGMENT}


# --- the controller's reader ------------------------------------------------

def _roles_via_controller(record, channels):
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController as C,
    )
    fake = SimpleNamespace(_run_record_codes=lambda: record)
    return C._qc_channel_roles(fake, channels)


def _channels(n):
    return [{"index": i, "prefix": f"ch{i}_"} for i in range(n)]


def test_the_controller_reads_the_runs_recorded_codes():
    assert _roles_via_controller(((2, 1), "reconstruction"), _channels(2)) \
        == {1: ROLE_SEGMENT, 0: ROLE_HELPER}
    assert _roles_via_controller(None, _channels(2)) == {}
    assert _roles_via_controller(((2, 1), "prediction"), []) == {}
    assert _roles_via_controller(((1, 0), "prediction"), _channels(1)) == {}


def test_a_failing_record_read_gives_plain_rows():
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController as C,
    )

    def boom():
        raise RuntimeError("unreadable")

    fake = SimpleNamespace(_run_record_codes=boom)
    assert C._qc_channel_roles(fake, _channels(3)) == {}


# --- the dialog's rows --------------------------------------------------------

@pytest.fixture(scope="module")
def app():
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _dialog(roles=None, n=3):
    from napari_epicycle.fusion_reconstruction.qc_extraction_dialog import (
        QCExtractionDialog,
    )
    kw = {} if roles is None else {"channel_roles": roles}
    return QCExtractionDialog(None, available_channels=_channels(n), **kw)


def test_rows_with_roles(app):
    dlg = _dialog({1: ROLE_SEGMENT, 0: ROLE_HELPER})
    try:
        cbs = dlg._channel_checkboxes
        assert sorted(cbs) == [0, 1, 2]
        assert cbs[0].text().strip() == "Channel 0 (helper channel)"
        assert cbs[1].text().strip() == "Channel 1 (channel to segment)"
        assert cbs[2].text().strip() == "Channel 2"
        assert cbs[1].toolTip() == (
            "Its columns start with ch1_.\n"
            "The active run segmented this channel.")
        assert cbs[0].toolTip() == (
            "Its columns start with ch0_.\n"
            "The active run used this channel as its helper.")
        assert cbs[2].toolTip() == "Its columns start with ch2_."
    finally:
        dlg.deleteLater()


def test_rows_without_roles(app):
    for dlg in (_dialog(), _dialog({})):
        try:
            for i, cb in dlg._channel_checkboxes.items():
                assert cb.text().strip() == T.QC_CHANNEL_ROW.format(i=i)
                assert cb.toolTip() == T.QC_CHANNEL_TIP_PREFIX.format(
                    prefix=f"ch{i}_")
                # the prefix left the row text
                assert "ch" not in cb.text()
                assert "—" not in cb.text()
        finally:
            dlg.deleteLater()


def test_the_channel_configs_do_not_change(app):
    plain, roled = _dialog(), _dialog({1: ROLE_SEGMENT, 0: ROLE_HELPER})
    try:
        for dlg in (plain, roled):
            assert dlg.get_channel_configs() == []
            dlg._channel_checkboxes[0].setChecked(True)
            dlg._channel_checkboxes[2].setChecked(True)
        assert plain.get_channel_configs() == roled.get_channel_configs() == [
            {"index": 0, "prefix": "ch0_"}, {"index": 2, "prefix": "ch2_"}]
        assert roled.get_settings()["channel_configs"] == (
            plain.get_settings()["channel_configs"])
    finally:
        plain.deleteLater()
        roled.deleteLater()


def test_an_unknown_role_reads_as_none(app):
    dlg = _dialog({0: "nonsense", 7: ROLE_SEGMENT})
    try:
        assert dlg._channel_checkboxes[0].text().strip() == "Channel 0"
        assert sorted(dlg._channel_checkboxes) == [0, 1, 2]
    finally:
        dlg.deleteLater()


def test_the_row_words_are_the_channel_boxes_role_words():
    assert T.QC_CHANNEL_ROLE_SEGMENT == "channel to segment"
    assert T.QC_CHANNEL_ROLE_HELPER == "helper channel"
    for s in (T.QC_CHANNEL_ROW, T.QC_CHANNEL_ROW_ROLE,
              T.QC_CHANNEL_TIP_PREFIX, T.QC_CHANNEL_TIP_SEGMENT,
              T.QC_CHANNEL_TIP_HELPER):
        assert "—" not in s


# --- the roles of the patch in view -------------------------------------------

def _records_fake(tmp_path):
    """A controller-like object over a real workspace folder (the default
    run), with the record readers bound."""
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController as C,
    )
    fake = SimpleNamespace(working_dir=str(tmp_path))
    for name in ("_prediction_cache_dir", "_pr_manifest_paths",
                 "_run_record_entries", "_run_record_codes"):
        setattr(fake, name, getattr(C, name).__get__(fake))
    roles = (lambda channels, patch="":
             C._qc_channel_roles(fake, channels, patch))
    return fake, roles


def _write_record(fake, patch, codes, when, kind="prediction"):
    import json
    from pathlib import Path

    pm, rm = fake._pr_manifest_paths(patch)
    path = Path(pm if kind == "prediction" else rm)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = "predicted_at" if kind == "prediction" else "generated_at"
    path.write_text(json.dumps({key: when,
                                "seg_config": {"channels": list(codes)}}),
                    encoding="utf-8")


def test_the_dialog_shows_the_roles_of_the_patch_in_view(tmp_path):
    fake, roles = _records_fake(tmp_path)
    _write_record(fake, "patch_A", (2, 1), "2026-10-01T10:00:00")
    _write_record(fake, "patch_B", (1, 2), "2026-10-02T10:00:00")
    # the run's newest record is patch B's ...
    assert fake._run_record_codes() == ((1, 2), "prediction")
    # ... but patch A's window shows patch A's roles
    assert roles(_channels(2), "patch_A") == {
        1: ROLE_SEGMENT, 0: ROLE_HELPER}
    assert roles(_channels(2), "patch_B") == {
        0: ROLE_SEGMENT, 1: ROLE_HELPER}


def test_a_patch_with_no_record_reads_the_run(tmp_path):
    fake, roles = _records_fake(tmp_path)
    _write_record(fake, "patch_A", (2, 1), "2026-10-01T10:00:00")
    _write_record(fake, "patch_B", (1, 2), "2026-10-02T10:00:00")
    assert roles(_channels(2), "patch_C") == {
        0: ROLE_SEGMENT, 1: ROLE_HELPER}
    assert roles(_channels(2)) == {0: ROLE_SEGMENT, 1: ROLE_HELPER}


def test_a_patch_whose_records_disagree_shows_no_role(tmp_path):
    fake, roles = _records_fake(tmp_path)
    _write_record(fake, "patch_A", (2, 1), "2026-10-01T10:00:00")
    _write_record(fake, "patch_A", (1, 2), "2026-10-01T11:00:00",
                  kind="reconstruction")
    assert roles(_channels(2), "patch_A") == {}
    # its two records agreeing: the roles
    _write_record(fake, "patch_A", (2, 1), "2026-10-01T11:00:00",
                  kind="reconstruction")
    assert roles(_channels(2), "patch_A") == {
        1: ROLE_SEGMENT, 0: ROLE_HELPER}


# --- a fresh run seeded from its model --------------------------------------

def test_a_fresh_default_runs_roles_follow_its_model(tmp_path):
    """The workspace's first run, seeded from a model trained on [1, 0] and
    predicted with it: QC names channel 0 the channel to segment, and no
    helper (a hard-coded [2, 1] used to name channel 1)."""
    fake, roles = _records_fake(tmp_path)
    _write_record(fake, "P1.tif", (1, 0), "2026-11-25T10:00:00")
    assert roles(_channels(2), "P1.tif") == {0: ROLE_SEGMENT}
    assert roles(_channels(2)) == {0: ROLE_SEGMENT}


def test_the_seg_channel_code_still_changes_no_feature():
    """The run's code for the channel to segment reaches the extraction as
    ``seg_channel``, but no feature group reads it: the intensity rows
    measure the channels ticked in the dialog, by index.  Codes 1 and 2
    give the same table."""
    import numpy as np

    from napari_epicycle.segmentation.feature_registry import (
        build_registry, run_extraction,
    )
    labels = np.zeros((6, 12, 12), np.uint32)
    labels[1:4, 1:5, 1:5] = 1
    labels[2:5, 6:11, 6:11] = 2
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 255, size=(2, 6, 12, 12)).astype(np.uint8)
    reg = build_registry(checked_channels=_channels(2))
    assert {g.name for g in reg} >= {"intensity_ch0_", "intensity_ch1_"}
    a = run_extraction(reg, {"labels": labels, "raw_stack": raw,
                             "seg_channel": 1})
    b = run_extraction(reg, {"labels": labels, "raw_stack": raw,
                             "seg_channel": 2})
    assert any(col.startswith("ch1_") for col in a.columns)
    assert a.equals(b)
