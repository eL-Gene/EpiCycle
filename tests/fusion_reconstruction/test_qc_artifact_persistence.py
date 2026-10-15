"""Tests for the QC-artifact persistence helpers added 2026-05-20.

Covers ``qc_features.save_adjacency_csv``,
``qc_features.save_features_raw_backup``, and the top-level
``qc_features.persist_qc_artifacts``.  These helpers were added to
fill the gap between the Step 3 directory restructure
(which scoped the net-new files) and the implementation (which had
only added the path helpers, not the writers).

Pure-Python, no Qt / napari dependency.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from napari_epicycle.fusion_reconstruction.qc_features import (
    persist_qc_artifacts,
    save_adjacency_csv,
    save_features_raw_backup,
    save_patch_csv,
)


# ---------------------------------------------------------------------------
# save_adjacency_csv
# ---------------------------------------------------------------------------


class TestSaveAdjacencyCsv:

    def test_writes_csv_when_adjacency_list_present(self, tmp_path):
        df = pd.DataFrame({"label": [1, 2, 3]})
        df.attrs["_adjacency_list"] = [
            {"label_a": 1, "label_b": 2, "contact_voxels": 42},
            {"label_a": 2, "label_b": 3, "contact_voxels": 17},
        ]
        path = tmp_path / "out_adjacency.csv"
        wrote = save_adjacency_csv(df, path)
        assert wrote is True
        assert path.is_file()
        loaded = pd.read_csv(path)
        assert list(loaded.columns) == ["label_a", "label_b", "contact_voxels"]
        assert len(loaded) == 2
        assert loaded.iloc[0]["label_a"] == 1

    def test_no_op_when_adjacency_list_missing(self, tmp_path):
        df = pd.DataFrame({"label": [1]})
        # No attrs set.
        path = tmp_path / "out_adjacency.csv"
        wrote = save_adjacency_csv(df, path)
        assert wrote is False
        assert not path.exists()

    def test_no_op_when_adjacency_list_empty(self, tmp_path):
        df = pd.DataFrame({"label": [1]})
        df.attrs["_adjacency_list"] = []
        path = tmp_path / "out_adjacency.csv"
        wrote = save_adjacency_csv(df, path)
        assert wrote is False
        assert not path.exists()

    def test_creates_parent_dir_if_missing(self, tmp_path):
        df = pd.DataFrame({"label": [1]})
        df.attrs["_adjacency_list"] = [{"label_a": 1, "label_b": 2}]
        nested = tmp_path / "deep" / "nested" / "out.csv"
        wrote = save_adjacency_csv(df, nested)
        assert wrote is True
        assert nested.is_file()

    def test_overwrites_existing_file(self, tmp_path):
        path = tmp_path / "out.csv"
        path.write_text("OLD CONTENT", encoding="utf-8")
        df = pd.DataFrame({"label": [1]})
        df.attrs["_adjacency_list"] = [{"label_a": 1, "label_b": 2}]
        wrote = save_adjacency_csv(df, path)
        assert wrote is True
        loaded = pd.read_csv(path)
        assert "label_a" in loaded.columns


# ---------------------------------------------------------------------------
# save_features_raw_backup
# ---------------------------------------------------------------------------


class TestSaveFeaturesRawBackup:

    def _make_features_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"label": [1, 2], "volume": [100, 200]}).to_csv(
            path, index=False,
        )

    def test_writes_backup_when_absent(self, tmp_path):
        src = tmp_path / "patch_features.csv"
        self._make_features_csv(src)
        dst = tmp_path / "patch_features_raw.csv"
        wrote = save_features_raw_backup(src, dst)
        assert wrote is True
        assert dst.is_file()
        # Content identical.
        assert pd.read_csv(src).equals(pd.read_csv(dst))

    def test_does_not_overwrite_existing_backup(self, tmp_path):
        src = tmp_path / "patch_features.csv"
        self._make_features_csv(src)
        dst = tmp_path / "patch_features_raw.csv"
        # Pre-existing raw backup with different content.
        dst.write_text("PRE-EXISTING RAW", encoding="utf-8")
        wrote = save_features_raw_backup(src, dst)
        assert wrote is False
        assert dst.read_text(encoding="utf-8") == "PRE-EXISTING RAW"

    def test_no_op_when_source_missing(self, tmp_path):
        src = tmp_path / "doesnt_exist.csv"
        dst = tmp_path / "raw_backup.csv"
        wrote = save_features_raw_backup(src, dst)
        assert wrote is False
        assert not dst.exists()

    def test_idempotent(self, tmp_path):
        src = tmp_path / "patch_features.csv"
        self._make_features_csv(src)
        dst = tmp_path / "patch_features_raw.csv"
        wrote_first = save_features_raw_backup(src, dst)
        wrote_second = save_features_raw_backup(src, dst)
        assert wrote_first is True
        assert wrote_second is False  # already backed up


# ---------------------------------------------------------------------------
# persist_qc_artifacts (top-level)
# ---------------------------------------------------------------------------


class TestPersistQcArtifacts:

    def _make_df_with_adjacency(self):
        df = pd.DataFrame({"label": [1, 2, 3]})
        df.attrs["_adjacency_list"] = [
            {"label_a": 1, "label_b": 2, "contact_voxels": 10},
            {"label_a": 2, "label_b": 3, "contact_voxels": 5},
        ]
        return df

    def _write_features_csv(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"label": [1, 2, 3], "volume": [100, 50, 200]}).to_csv(
            path, index=False,
        )

    def test_initial_extraction_writes_raw_backup_and_adjacency(self, tmp_path):
        """First extraction (no prior raw backup) → writes both."""
        features_csv = tmp_path / "patch_001_features.csv"
        self._write_features_csv(features_csv)
        df = self._make_df_with_adjacency()

        result = persist_qc_artifacts(df, features_csv)

        assert result["is_initial"] is True
        assert result["raw_backup_written"] is True
        assert result["adjacency_written"] is True
        raw_backup = tmp_path / "patch_001_features_raw.csv"
        raw_adj = tmp_path / "patch_001_adjacency_raw.csv"
        assert raw_backup.is_file()
        assert raw_adj.is_file()
        assert Path(result["adjacency_path"]) == raw_adj

    def test_re_extraction_post_curation_writes_curated_adjacency_not_raw(self, tmp_path):
        """Re-extraction with patch_status post-curation → curated path.

        Post-2026-05-20 audit: the raw-vs-curated decision is now
        gated on patch_status (not just on raw-backup existence).
        Re-extracting POST-curation goes to adjacency_curated.csv;
        the raw backup is preserved.
        """
        features_csv = tmp_path / "patch_001_features.csv"
        self._write_features_csv(features_csv)
        # Pre-existing raw backup (from initial pre-curation extract).
        raw_backup = tmp_path / "patch_001_features_raw.csv"
        raw_backup.write_text("ORIGINAL_BACKUP", encoding="utf-8")
        df = self._make_df_with_adjacency()

        result = persist_qc_artifacts(
            df, features_csv, patch_status="curating",
        )

        assert result["is_initial"] is False
        assert result["is_pre_curation"] is False
        assert result["raw_backup_written"] is False
        # Raw backup contents preserved.
        assert raw_backup.read_text(encoding="utf-8") == "ORIGINAL_BACKUP"
        # Adjacency went to the curated sibling.
        curated_adj = tmp_path / "patch_001_adjacency_curated.csv"
        raw_adj = tmp_path / "patch_001_adjacency_raw.csv"
        assert curated_adj.is_file()
        assert not raw_adj.exists()
        assert Path(result["adjacency_path"]) == curated_adj

    def test_re_extraction_pre_curation_does_not_overwrite_raw_adjacency(self, tmp_path):
        """User-reported 2026-05-20: re-extracting features pre-curation
        was overwriting the FIRST adjacency_raw.csv.  Wrong — the
        raw pair (features_raw.csv + adjacency_raw.csv) is meant to
        be the immutable first-extraction snapshot.  Both files must
        survive subsequent pre-curation re-extractions unchanged.
        """
        features_csv = tmp_path / "patch_001_features.csv"
        self._write_features_csv(features_csv)
        # Pre-existing raw backup + raw adjacency (from a prior
        # initial extraction).
        raw_backup = tmp_path / "patch_001_features_raw.csv"
        raw_backup.write_text("ORIGINAL_BACKUP", encoding="utf-8")
        raw_adj = tmp_path / "patch_001_adjacency_raw.csv"
        raw_adj.write_text("ORIGINAL_RAW_ADJ", encoding="utf-8")
        df = self._make_df_with_adjacency()

        result = persist_qc_artifacts(
            df, features_csv, patch_status="reconstructed",
        )

        assert result["is_initial"] is False  # raw backup already exists
        assert result["is_pre_curation"] is True
        assert result["raw_backup_written"] is False
        # Both raw files preserved unchanged.
        assert raw_backup.read_text(encoding="utf-8") == "ORIGINAL_BACKUP"
        assert raw_adj.read_text(encoding="utf-8") == "ORIGINAL_RAW_ADJ"
        # No curated file written either (still pre-curation).
        curated_adj = tmp_path / "patch_001_adjacency_curated.csv"
        assert not curated_adj.exists()
        # No adjacency was written this call.
        assert result["adjacency_written"] is False

    def test_post_curation_extraction_without_prior_raw_skips_backup(self, tmp_path):
        """User curated BEFORE first QC extract.  Worker reads
        post-curation labels — must NOT snapshot them as "raw" — that
        would silently violate the design's "raw = pre-curation"
        contract.  Instead: skip raw backup, write curated adjacency."""
        features_csv = tmp_path / "patch_001_features.csv"
        self._write_features_csv(features_csv)
        df = self._make_df_with_adjacency()

        result = persist_qc_artifacts(
            df, features_csv, patch_status="accepted",
        )

        assert result["is_initial"] is False
        assert result["is_pre_curation"] is False
        assert result["raw_backup_written"] is False
        assert result["raw_skipped_post_curation"] is True
        # No raw files at all — pre-curation snapshot is unavailable
        # for this patch (the labels were already curated before first
        # extraction).
        assert not (tmp_path / "patch_001_features_raw.csv").exists()
        assert not (tmp_path / "patch_001_adjacency_raw.csv").exists()
        # Adjacency went to curated.
        curated_adj = tmp_path / "patch_001_adjacency_curated.csv"
        assert curated_adj.is_file()

    def test_re_extraction_overwrites_existing_curated_adjacency(self, tmp_path):
        features_csv = tmp_path / "patch_001_features.csv"
        self._write_features_csv(features_csv)
        raw_backup = tmp_path / "patch_001_features_raw.csv"
        raw_backup.write_text("ORIGINAL", encoding="utf-8")
        # Pre-existing curated adjacency.
        curated_adj = tmp_path / "patch_001_adjacency_curated.csv"
        curated_adj.write_text("STALE_CURATED", encoding="utf-8")
        df = self._make_df_with_adjacency()

        result = persist_qc_artifacts(
            df, features_csv, patch_status="finalized",
        )

        assert result["adjacency_written"] is True
        # Was overwritten with real adjacency data.
        loaded = pd.read_csv(curated_adj)
        assert "label_a" in loaded.columns

    def test_re_initial_drops_stale_curated_adjacency(self, tmp_path):
        """Audit finding 2026-05-20 (C1): if raw backup gets manually
        deleted and we re-enter INITIAL state, any pre-existing
        curated adjacency from the prior cycle would otherwise stay
        on disk alongside the fresh raw adjacency — internally
        inconsistent.  The helper should clean that up."""
        features_csv = tmp_path / "patch_001_features.csv"
        self._write_features_csv(features_csv)
        # Pre-existing stale curated adjacency (from a prior cycle
        # before raw was deleted).
        stale_curated = tmp_path / "patch_001_adjacency_curated.csv"
        stale_curated.write_text("STALE", encoding="utf-8")
        df = self._make_df_with_adjacency()

        result = persist_qc_artifacts(
            df, features_csv, patch_status="reconstructed",
        )

        assert result["is_initial"] is True
        assert not stale_curated.exists()  # cleaned up

    def test_zero_byte_raw_backup_treated_as_missing(self, tmp_path):
        """Audit finding 2026-05-20 (W2): a 0-byte features_raw.csv
        from an aborted prior copy must not block a fresh backup —
        otherwise the user permanently loses the raw snapshot."""
        features_csv = tmp_path / "patch_001_features.csv"
        self._write_features_csv(features_csv)
        # 0-byte stub raw backup.
        raw_stub = tmp_path / "patch_001_features_raw.csv"
        raw_stub.write_bytes(b"")
        df = self._make_df_with_adjacency()

        result = persist_qc_artifacts(
            df, features_csv, patch_status="reconstructed",
        )

        assert result["is_initial"] is True
        assert result["raw_backup_written"] is True
        # Stub got replaced with the real CSV.
        assert raw_stub.stat().st_size > 0

    def test_no_adjacency_attr_still_writes_raw_backup(self, tmp_path):
        """If extraction didn't compute adjacency, raw backup still runs."""
        features_csv = tmp_path / "patch_001_features.csv"
        self._write_features_csv(features_csv)
        df = pd.DataFrame({"label": [1]})  # no _adjacency_list attr

        result = persist_qc_artifacts(df, features_csv)

        assert result["is_initial"] is True
        assert result["raw_backup_written"] is True
        assert result["adjacency_written"] is False
        assert not Path(result["adjacency_path"]).exists()

    def test_non_standard_csv_name_still_works(self, tmp_path):
        """Caller passed a non-standard CSV name (doesn't end with
        _features.csv) — falls back to stem-based prefix."""
        odd = tmp_path / "weird_name.csv"
        odd.write_text("label,volume\n1,100\n", encoding="utf-8")
        df = self._make_df_with_adjacency()
        result = persist_qc_artifacts(df, odd)
        # Stem fallback uses Path.stem = "weird_name".
        assert result["raw_backup_written"] is True
        assert (tmp_path / "weird_name_features_raw.csv").is_file()
        assert (tmp_path / "weird_name_adjacency_raw.csv").is_file()

    def test_idempotent_first_call_then_immediate_recall(self, tmp_path):
        """Calling persist twice: first initial (pre-curation), second
        re-extract POST-curation (status flipped after user committed
        edits between calls — typical workflow)."""
        features_csv = tmp_path / "patch_001_features.csv"
        self._write_features_csv(features_csv)
        df = self._make_df_with_adjacency()

        r1 = persist_qc_artifacts(
            df, features_csv, patch_status="reconstructed",
        )
        r2 = persist_qc_artifacts(
            df, features_csv, patch_status="accepted",
        )

        assert r1["is_initial"] is True
        assert r2["is_initial"] is False
        # Both adjacency files now exist.
        assert (tmp_path / "patch_001_adjacency_raw.csv").is_file()
        assert (tmp_path / "patch_001_adjacency_curated.csv").is_file()

    def test_uses_new_layout_paths_via_paths_module(self, tmp_path):
        """End-to-end with the _paths module to confirm path stems
        align with how the path helpers would produce them.
        """
        from napari_epicycle.fusion_reconstruction import _paths

        ws = tmp_path / "ws"
        ws.mkdir()
        # Build the canonical features.csv path via _paths helper.
        features_csv = _paths.qc_features_csv(ws, "patch_001")
        self._write_features_csv(features_csv)
        df = self._make_df_with_adjacency()

        result = persist_qc_artifacts(df, features_csv)

        # Compare against the canonical paths.
        assert Path(result["adjacency_path"]) == (
            _paths.qc_adjacency_raw_csv(ws, "patch_001")
        )
        # Raw features backup matches the helper path too.
        assert (
            _paths.qc_features_raw_csv(ws, "patch_001")
        ).is_file()
