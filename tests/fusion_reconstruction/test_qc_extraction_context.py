"""Tests for QC extraction-context sidecar (2026-05-31, fix/qc-stale-notice).

Background: QC features' centroid and physical-unit measurements depend on the
spacing + downsample factor at extraction time.  If the user changes spacing or
downsample after extraction and reloads, the navigator's auto-pan + the area /
volume features in the df are out of date.  These helpers record the extraction
context next to the features CSV and let the controller detect mismatch and
notify the user.
"""
from __future__ import annotations

import json

from napari_epicycle.fusion_reconstruction._qc_extraction_context import (
    EXTRACTION_CONTEXT_FILENAME,
    read_extraction_context,
    stale_reason,
    write_extraction_context,
)


# ---------------------------------------------------------------------------
# write_extraction_context
# ---------------------------------------------------------------------------


def test_write_creates_sidecar_next_to_csv(tmp_path):
    csv = tmp_path / "features.csv"
    csv.write_text("label,centroid-0\n")
    write_extraction_context(csv, spacing=(1.0, 0.25, 0.25), downsample_factor=4)
    sidecar = tmp_path / EXTRACTION_CONTEXT_FILENAME
    assert sidecar.exists()
    data = json.loads(sidecar.read_text())
    assert data["spacing"] == [1.0, 0.25, 0.25]
    assert data["downsample_factor"] == 4
    assert "extracted_at" in data


def test_write_overwrites_existing(tmp_path):
    csv = tmp_path / "features.csv"
    csv.write_text("")
    write_extraction_context(csv, spacing=(1.0, 1.0, 1.0), downsample_factor=1)
    write_extraction_context(csv, spacing=(4.0, 0.25, 0.25), downsample_factor=2)
    data = read_extraction_context(csv)
    assert data["spacing"] == [4.0, 0.25, 0.25]
    assert data["downsample_factor"] == 2


def test_write_handles_tuple_or_list_spacing(tmp_path):
    csv = tmp_path / "features.csv"
    csv.write_text("")
    write_extraction_context(csv, spacing=[2.0, 0.5, 0.5], downsample_factor=1)
    data = read_extraction_context(csv)
    assert data["spacing"] == [2.0, 0.5, 0.5]


# ---------------------------------------------------------------------------
# read_extraction_context
# ---------------------------------------------------------------------------


def test_read_returns_none_when_no_sidecar(tmp_path):
    csv = tmp_path / "features.csv"
    csv.write_text("")  # CSV exists, sidecar doesn't
    assert read_extraction_context(csv) is None


def test_read_returns_none_when_csv_missing(tmp_path):
    csv = tmp_path / "missing.csv"
    assert read_extraction_context(csv) is None


def test_read_returns_none_on_malformed_json(tmp_path):
    csv = tmp_path / "features.csv"
    csv.write_text("")
    (tmp_path / EXTRACTION_CONTEXT_FILENAME).write_text("{not json")
    assert read_extraction_context(csv) is None


# ---------------------------------------------------------------------------
# stale_reason — the comparison helper
# ---------------------------------------------------------------------------


def test_stale_returns_none_when_context_matches():
    stored = {"spacing": [1.0, 1.0, 1.0], "downsample_factor": 1}
    assert stale_reason(stored, current_spacing=(1.0, 1.0, 1.0), current_downsample=1) is None


def test_stale_detects_spacing_change():
    stored = {"spacing": [1.0, 1.0, 1.0], "downsample_factor": 1}
    reason = stale_reason(stored, current_spacing=(4.0, 0.25, 0.25), current_downsample=1)
    assert reason is not None
    assert "spacing" in reason.lower()


def test_stale_detects_downsample_change():
    stored = {"spacing": [1.0, 1.0, 1.0], "downsample_factor": 1}
    reason = stale_reason(stored, current_spacing=(1.0, 1.0, 1.0), current_downsample=4)
    assert reason is not None
    assert "downsample" in reason.lower()


def test_stale_detects_both_changed():
    stored = {"spacing": [1.0, 1.0, 1.0], "downsample_factor": 1}
    reason = stale_reason(stored, current_spacing=(4.0, 0.25, 0.25), current_downsample=4)
    assert reason is not None
    # Either is fine — implementation may report first-change-found or combined.
    assert "spacing" in reason.lower() or "downsample" in reason.lower()


def test_stale_returns_none_when_stored_is_none():
    """Backward-compat: workspace with features but no sidecar (extracted
    before this feature shipped).  No comparison possible → not flagged."""
    assert stale_reason(None, current_spacing=(4.0, 0.25, 0.25), current_downsample=4) is None


def test_stale_tolerates_small_float_drift_on_spacing():
    """Floating-point round-trip through JSON shouldn't trigger a false
    staleness signal."""
    stored = {"spacing": [1.0000001, 0.25, 0.25], "downsample_factor": 1}
    assert stale_reason(stored, current_spacing=(1.0, 0.25, 0.25), current_downsample=1) is None


def test_stale_handles_missing_keys_gracefully():
    """Defensive: an incomplete sidecar (e.g. older schema or hand-edited)
    should not crash."""
    assert stale_reason({}, current_spacing=(1.0, 1.0, 1.0), current_downsample=1) is None
    assert stale_reason({"spacing": [1.0, 1.0, 1.0]}, current_spacing=(1.0, 1.0, 1.0), current_downsample=1) is None


# ---------------------------------------------------------------------------
# Controller wiring (source-guard tests — verify the hooks are in place)
# ---------------------------------------------------------------------------


def test_controller_has_staleness_methods():
    """The 3 staleness methods must be defined on the controller class —
    `_write_qc_extraction_context` (sidecar writer after extraction),
    `_check_qc_extraction_staleness` (the comparison helper), and
    `_maybe_show_qc_staleness_notification` (the modal UX)."""
    import inspect
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    assert hasattr(FusionReconstructionController, "_write_qc_extraction_context")
    assert hasattr(FusionReconstructionController, "_check_qc_extraction_staleness")
    assert hasattr(FusionReconstructionController, "_maybe_show_qc_staleness_notification")
    # Each method's source must reference the helper module so a future
    # refactor that breaks the wiring fires this test.
    for name in (
        "_write_qc_extraction_context",
        "_check_qc_extraction_staleness",
    ):
        src = inspect.getsource(getattr(FusionReconstructionController, name))
        assert "_qc_extraction_context" in src, (
            f"{name} must import from _qc_extraction_context"
        )


def test_controller_init_creates_stale_state_fields():
    """Source-guard: the controller's __init__ must initialize the
    staleness tracking fields (`_qc_extraction_stale` and
    `_qc_staleness_notified`).  Without these, the patch-load + settings-
    update hooks would AttributeError."""
    import inspect
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    src = inspect.getsource(FusionReconstructionController.__init__)
    assert "_qc_extraction_stale" in src
    assert "_qc_staleness_notified" in src


def test_extract_finished_writes_sidecar():
    """Source-guard: `_on_qc_extract_finished` must call
    `_write_qc_extraction_context` after a successful extraction so the
    sidecar travels with the new features CSV.  Without this, every
    re-extraction would leave the sidecar stale and the comparison
    would keep flagging the same drift."""
    import inspect
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    src = inspect.getsource(FusionReconstructionController._on_qc_extract_finished)
    assert "_write_qc_extraction_context" in src


def test_features_reload_runs_staleness_check():
    """Source-guard: `_reload_current_patch_features` must call the
    staleness check at its END, AFTER `_qc_features_df` has been
    populated for the new patch.  Audit-found bug (2026-05-31): putting
    the check inside `_load_patch_for_qc_view` ran it against the prior
    patch's df because `_reload_current_patch_features` happens after.
    """
    import inspect
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    src = inspect.getsource(FusionReconstructionController._reload_current_patch_features)
    assert "_check_qc_extraction_staleness" in src
    assert "_maybe_show_qc_staleness_notification" in src


def test_modal_marks_notified_only_after_successful_exec():
    """Source-guard: `_maybe_show_qc_staleness_notification` must add the
    dedup key AFTER `box.exec_()` returns successfully, not before.
    Audit-found bug (2026-05-31): if QMessageBox raises (no parent, etc.)
    and the flag was set before, the user permanently misses the modal
    AND won't be re-prompted on subsequent loads.
    """
    import inspect
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    src = inspect.getsource(
        FusionReconstructionController._maybe_show_qc_staleness_notification
    )
    # The `.add(key)` call must appear AFTER `box.exec_()`, not before.
    add_idx = src.find("_qc_staleness_notified.add(key)")
    exec_idx = src.find("box.exec_()")
    assert add_idx > 0 and exec_idx > 0, "both calls must be present"
    assert add_idx > exec_idx, (
        "_qc_staleness_notified.add(key) must run AFTER box.exec_() succeeds, "
        "otherwise a dialog-display failure permanently silences the warning."
    )


def test_qc_settings_accept_runs_staleness_check():
    """Source-guard: the QC settings dialog accept path must re-check
    staleness so a user who changes spacing mid-session is told
    immediately, not on next patch load."""
    import inspect
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    src = inspect.getsource(FusionReconstructionController._open_qc_settings)
    assert "_check_qc_extraction_staleness" in src


def test_write_is_atomic_through_the_shared_writer(tmp_path, monkeypatch):
    """B LOW-6: the sidecar goes through ``_step3_manifests.write_json_atomic``
    (mkstemp + fsync + ``os.replace``), so an interrupted write never leaves
    half a file a later staleness check reads as "no context"; the output
    is the same shape."""
    from napari_epicycle.fusion_reconstruction import _step3_manifests as SM
    calls = []
    real = SM.write_json_atomic

    def spy(path, data):
        calls.append(path)
        real(path, data)

    monkeypatch.setattr(SM, "write_json_atomic", spy)
    csv = tmp_path / "p01_features.csv"
    write_extraction_context(csv, spacing=(2, 0.5, 0.5), downsample_factor=1)
    side = tmp_path / "p01_extraction_context.json"
    assert [str(p) for p in calls] == [str(side)]
    data = json.loads(side.read_text(encoding="utf-8"))
    assert set(data) == {"spacing", "downsample_factor", "extracted_at"}
    assert data["spacing"] == [2.0, 0.5, 0.5]
    assert data["downsample_factor"] == 1
    assert read_extraction_context(csv)["spacing"] == [2.0, 0.5, 0.5]
    assert not [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]


# ---------------------------------------------------------------------------
# An unknown voxel size is written as unknown: ``"spacing": null`` plus the
# word saying where the spacing came from, and every reader takes ``null`` as
# "measured in voxel units".
# ---------------------------------------------------------------------------


def test_unknown_spacing_is_written_as_null_with_its_word(tmp_path):
    csv = tmp_path / "p01_features.csv"
    write_extraction_context(csv, spacing=None, spacing_source="unknown",
                             downsample_factor=1)
    data = json.loads((tmp_path / "p01_extraction_context.json")
                      .read_text(encoding="utf-8"))
    assert "spacing" in data and data["spacing"] is None
    assert data["spacing_source"] == "unknown"
    assert data["downsample_factor"] == 1
    assert "extracted_at" in data


def test_no_source_given_writes_no_source_key(tmp_path):
    """A direct write without the word keeps the three keys of an
    extraction and nothing else (the atomic-write test pins the same)."""
    csv = tmp_path / "p01_features.csv"
    write_extraction_context(csv, spacing=(2, 0.5, 0.5), downsample_factor=1)
    data = read_extraction_context(csv)
    assert set(data) == {"spacing", "downsample_factor", "extracted_at"}
    write_extraction_context(csv, spacing=None, downsample_factor=1)
    data = read_extraction_context(csv)
    assert set(data) == {"spacing", "downsample_factor", "extracted_at"}
    assert data["spacing"] is None


def test_a_known_spacing_writes_its_word(tmp_path):
    csv = tmp_path / "p01_features.csv"
    write_extraction_context(csv, spacing=(2, 0.5, 0.5),
                             spacing_source="step1", downsample_factor=1)
    data = read_extraction_context(csv)
    assert data["spacing"] == [2.0, 0.5, 0.5]
    assert data["spacing_source"] == "step1"


def test_a_pairing_key_cannot_overwrite_the_source(tmp_path):
    csv = tmp_path / "p01_features.csv"
    write_extraction_context(
        csv, spacing=None, spacing_source="unknown", downsample_factor=1,
        pairing={"spacing_source": "x", "spacing": [9, 9, 9],
                 "match_run": "nuclei"})
    data = read_extraction_context(csv)
    assert data["spacing_source"] == "unknown"
    assert data["spacing"] is None
    assert data["match_run"] == "nuclei"


def test_a_null_spacing_compares_as_voxel_units():
    stored = {"spacing": None, "spacing_source": "unknown",
              "downsample_factor": 1}
    assert stale_reason(stored, current_spacing=(1, 1, 1),
                        current_downsample=1) is None
    assert stale_reason(stored, current_spacing=(2, 0.5, 0.5),
                        current_downsample=1) == \
        "spacing changed from voxel units to (2.0, 0.5, 0.5)"


def test_a_null_spacing_still_flags_a_downsample_change():
    stored = {"spacing": None, "spacing_source": "unknown",
              "downsample_factor": 2}
    assert stale_reason(stored, current_spacing=(1, 1, 1),
                        current_downsample=1) == \
        "downsample factor changed from 2 to 1"


def test_a_known_spacing_keeps_its_reason_words():
    stored = {"spacing": [2.0, 0.5, 0.5], "spacing_source": "step1",
              "downsample_factor": 1}
    assert stale_reason(stored, current_spacing=(4, 1, 1),
                        current_downsample=1) == \
        "spacing changed from (2.0, 0.5, 0.5) to (4.0, 1.0, 1.0)"


def test_spacing_source_cannot_be_set_by_update(tmp_path):
    from napari_epicycle.fusion_reconstruction._qc_extraction_context import (
        update_extraction_context,
    )
    csv = tmp_path / "p01_features.csv"
    write_extraction_context(csv, spacing=None, spacing_source="unknown",
                             downsample_factor=1)
    assert update_extraction_context(csv, spacing_source="user",
                                     match_values_predate_edits=True)
    data = read_extraction_context(csv)
    assert data["spacing_source"] == "unknown"
    assert data["spacing"] is None
    assert data["match_values_predate_edits"] is True


def test_an_old_1_1_1_sidecar_reads_as_before():
    """A sidecar written before the word existed: ``[1, 1, 1]`` and no
    ``spacing_source`` -- read back as that list, compared as before."""
    from napari_epicycle.fusion_reconstruction._qc_extraction_context import (
        VOXEL_UNITS, measured_zyx,
    )
    old = {"spacing": [1.0, 1.0, 1.0], "downsample_factor": 1}
    assert measured_zyx(old) == [1.0, 1.0, 1.0]
    assert stale_reason(old, current_spacing=(2, 0.5, 0.5),
                        current_downsample=1) == \
        "spacing changed from (1.0, 1.0, 1.0) to (2.0, 0.5, 0.5)"
    assert tuple(VOXEL_UNITS) == (1.0, 1.0, 1.0)


def test_measured_zyx_reads_null_as_voxel_units_and_absent_as_unknown():
    from napari_epicycle.fusion_reconstruction._qc_extraction_context import (
        measured_zyx,
    )
    assert measured_zyx({"spacing": [2, 0.5, 0.5]}) == [2.0, 0.5, 0.5]
    assert measured_zyx({"spacing": None}) == [1.0, 1.0, 1.0]
    assert measured_zyx({}) is None
    assert measured_zyx({"spacing": "1,1,1"}) is None
    assert measured_zyx({"spacing": [1, 2]}) is None
    assert measured_zyx(None) is None
    # a fresh list every call: a caller may change it
    a = measured_zyx({"spacing": None})
    a[0] = 9.0
    assert measured_zyx({"spacing": None}) == [1.0, 1.0, 1.0]


def test_an_unknown_current_size_is_named_voxel_units():
    """The CURRENT side unknown (``None``) compares as voxel units and is
    named so: never "(1.0, 1.0, 1.0)", a size nobody entered."""
    stored = {"spacing": [3.0, 1.0, 1.0], "spacing_source": "qc_settings",
              "downsample_factor": 1}
    assert stale_reason(stored, current_spacing=None,
                        current_downsample=1) == \
        "spacing changed from (3.0, 1.0, 1.0) to voxel units"
    # both unknown: nothing changed
    unknown = {"spacing": None, "spacing_source": "unknown",
               "downsample_factor": 1}
    assert stale_reason(unknown, current_spacing=None,
                        current_downsample=1) is None
    # an old [1, 1, 1] sidecar was voxel units too: nothing changed
    old = {"spacing": [1.0, 1.0, 1.0], "downsample_factor": 1}
    assert stale_reason(old, current_spacing=None,
                        current_downsample=1) is None
    # the downsample factor is still compared
    assert stale_reason(dict(unknown, downsample_factor=2),
                        current_spacing=None, current_downsample=1) == \
        "downsample factor changed from 2 to 1"


def test_pairing_fields_never_carry_an_extraction_key_into_the_file(
        tmp_path):
    """With NO source word given, a pairing field named like an extraction
    key (``spacing_source``, ``downsample_factor``, ``extracted_at``) must
    not reach the file: the source key stays absent and the others are the
    extraction's own."""
    csv = tmp_path / "p01_features.csv"
    write_extraction_context(
        csv, spacing=[2.0, 0.5, 0.5], downsample_factor=1,
        pairing={"spacing_source": "x", "downsample_factor": 9,
                 "extracted_at": "never", "match_run": "nuclei"})
    data = read_extraction_context(csv)
    assert "spacing_source" not in data
    assert data["downsample_factor"] == 1
    assert data["extracted_at"] != "never"
    assert data["spacing"] == [2.0, 0.5, 0.5]
    assert data["match_run"] == "nuclei"
