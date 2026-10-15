"""Tests for the curated-adjacency derivation in re-extract flow.

The 2026-05-20 follow-up to user-reported gap (clarified): every
re-extract is past the initial-snapshot state.  Every re-extract
produces a fresh ``adjacency_curated.csv`` representing the
current active-label neighbor graph.  The controller's
``_on_qc_extract_finished`` derives it from the worker's raw
adjacency edge list by filtering edges to active labels only —
which drops any edge referencing a label that didn't survive
(ghost-dropped, retroactively garbage-tiered, etc.).  When the
active set equals the worker's set, the curated file is content-
identical to the raw file but still gets written (parity with
features.csv being re-written on every re-extract).

This module tests the derivation LOGIC in isolation — the
controller wire-up itself is Qt-heavy and tested manually.  The
pure-Python edge filter is what produces the curated file content.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _filter_adjacency_to_active(worker_adjacency, active_labels):
    """Mirror of the inline filter logic in controller.py
    `_on_qc_extract_finished` after the merge + ghost-drop.
    """
    active = set(int(v) for v in active_labels)
    out = []
    for edge in worker_adjacency:
        if not isinstance(edge, dict):
            continue
        a = edge.get("label_a")
        b = edge.get("label_b")
        if a is None or b is None:
            continue
        try:
            if int(a) in active and int(b) in active:
                out.append(edge)
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# Filter correctness
# ---------------------------------------------------------------------------


class TestAdjacencyFilter:

    def test_all_edges_survive_when_no_drops(self):
        worker_adj = [
            {"label_a": 1, "label_b": 2, "contact_voxels": 10},
            {"label_a": 2, "label_b": 3, "contact_voxels": 5},
            {"label_a": 1, "label_b": 3, "contact_voxels": 2},
        ]
        active = [1, 2, 3]
        out = _filter_adjacency_to_active(worker_adj, active)
        assert len(out) == 3

    def test_drops_edges_touching_removed_labels(self):
        """Edges where either endpoint was removed get dropped."""
        worker_adj = [
            {"label_a": 1, "label_b": 2, "contact_voxels": 10},  # keep
            {"label_a": 2, "label_b": 3, "contact_voxels": 5},   # drop (3 removed)
            {"label_a": 1, "label_b": 3, "contact_voxels": 2},   # drop (3 removed)
            {"label_a": 4, "label_b": 5, "contact_voxels": 8},   # drop (both removed)
        ]
        active = [1, 2]
        out = _filter_adjacency_to_active(worker_adj, active)
        assert len(out) == 1
        assert out[0]["label_a"] == 1
        assert out[0]["label_b"] == 2

    def test_empty_active_set_returns_empty(self):
        worker_adj = [
            {"label_a": 1, "label_b": 2},
        ]
        out = _filter_adjacency_to_active(worker_adj, [])
        assert out == []

    def test_empty_worker_adjacency(self):
        out = _filter_adjacency_to_active([], [1, 2, 3])
        assert out == []

    def test_non_dict_edges_skipped(self):
        """Defensive: gracefully skip malformed entries."""
        worker_adj = [
            {"label_a": 1, "label_b": 2},
            "not a dict",
            None,
            42,
            {"label_a": 2, "label_b": 3},
        ]
        out = _filter_adjacency_to_active(worker_adj, [1, 2, 3])
        assert len(out) == 2

    def test_missing_label_keys_skipped(self):
        worker_adj = [
            {"label_a": 1, "label_b": 2},
            {"label_a": 1},               # missing label_b
            {"label_b": 2},               # missing label_a
            {"other_field": 42},
            {"label_a": 2, "label_b": 3},
        ]
        out = _filter_adjacency_to_active(worker_adj, [1, 2, 3])
        assert len(out) == 2

    def test_non_integer_labels_skipped(self):
        """Defensive: TypeError on int conversion → skip edge."""
        worker_adj = [
            {"label_a": 1, "label_b": 2},
            {"label_a": "foo", "label_b": "bar"},
            {"label_a": None, "label_b": 3},
        ]
        out = _filter_adjacency_to_active(worker_adj, [1, 2, 3])
        # First edge survives; second is dropped by TypeError;
        # third has None which is caught by the explicit None check.
        assert len(out) == 1

    def test_preserves_edge_metadata(self):
        """Beyond label_a/label_b, edge dict fields are passed through."""
        worker_adj = [
            {"label_a": 1, "label_b": 2, "contact_voxels": 42,
             "z_overlap": 5, "custom": "extra"},
        ]
        out = _filter_adjacency_to_active(worker_adj, [1, 2])
        assert out[0]["contact_voxels"] == 42
        assert out[0]["z_overlap"] == 5
        assert out[0]["custom"] == "extra"


# ---------------------------------------------------------------------------
# Round-trip — write filtered adjacency to disk, read back
# ---------------------------------------------------------------------------


class TestCuratedAdjacencyOnDisk:

    def test_filtered_adjacency_writes_to_curated_path(self, tmp_path):
        """End-to-end: derive filtered adjacency + write to the
        canonical curated path produced by _paths.qc_adjacency_curated_csv."""
        from napari_epicycle.fusion_reconstruction import _paths

        worker_adj = [
            {"label_a": 1, "label_b": 2, "contact_voxels": 10},
            {"label_a": 3, "label_b": 4, "contact_voxels": 5},  # drop
        ]
        active = [1, 2]
        filtered = _filter_adjacency_to_active(worker_adj, active)
        assert len(filtered) == 1

        ws = tmp_path / "ws"
        ws.mkdir()
        out_path = _paths.qc_adjacency_curated_csv(ws, "patch_001")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(filtered).to_csv(str(out_path), index=False)

        # Reads back as a 1-edge CSV.
        loaded = pd.read_csv(out_path)
        assert len(loaded) == 1
        assert loaded.iloc[0]["label_a"] == 1
        assert loaded.iloc[0]["label_b"] == 2

    def test_curated_path_is_distinct_from_raw_path(self, tmp_path):
        from napari_epicycle.fusion_reconstruction import _paths
        raw = _paths.qc_adjacency_raw_csv(tmp_path, "patch_001")
        curated = _paths.qc_adjacency_curated_csv(tmp_path, "patch_001")
        assert raw != curated
        assert raw.name.endswith("_adjacency_raw.csv")
        assert curated.name.endswith("_adjacency_curated.csv")


# ---------------------------------------------------------------------------
# Realistic scenario — user's reported workflow
# ---------------------------------------------------------------------------


class TestUserReportedScenario:
    """Reproduces the 2026-05-20 user-reported flow:
    1. Initial extract of 268 cells (worker writes raw)
    2. User marks 50 as ghosts via tier=garbage
    3. Re-extract: drops 50 → 218 active rows
    4. Expected: adjacency_curated.csv with edges filtered to 218 labels
    """

    def test_user_flow_filter_correctness(self):
        # 268 cells; 50 marked as ghosts; 218 surviving.
        # Synthetic adjacency: cells 1-268, each connected to neighbors
        # (a simple chain plus some random crosses).
        worker_adj = []
        for i in range(1, 268):
            worker_adj.append({
                "label_a": i, "label_b": i + 1, "contact_voxels": 10,
            })
        # Add a few "long-range" edges that span the dropped range.
        worker_adj.append({"label_a": 10, "label_b": 200, "contact_voxels": 3})
        worker_adj.append({"label_a": 50, "label_b": 250, "contact_voxels": 2})

        # Drop cells 100-149 (the "garbage" tier).
        dropped = set(range(100, 150))
        active = [i for i in range(1, 269) if i not in dropped]
        assert len(active) == 218

        filtered = _filter_adjacency_to_active(worker_adj, active)

        # All chain edges that DO NOT touch dropped cells survive.
        # Chain edges (i, i+1) for i in [1..267]: dropped when
        # i in dropped OR i+1 in dropped.  That's edges where
        # i in {99} ∪ dropped ∪ {149}.  Specifically:
        #   i=99 → edge (99, 100) dropped (100 is ghost)
        #   i in [100..148] → edge (i, i+1) dropped (both endpoints ghosts)
        #   i=149 → edge (149, 150) dropped (149 is ghost)
        # Total chain edges dropped: 50 (i in 99..149, inclusive of bounds)
        # Note: actually 99,100,..148,149 = 51 numbers if both endpoints
        # checked; but (149,150) is dropped because 149 in dropped.
        # Let's be precise:
        chain_dropped = 0
        for i in range(1, 268):
            if i in dropped or (i + 1) in dropped:
                chain_dropped += 1
        chain_kept = 267 - chain_dropped

        # Long-range edges: (10, 200) survives (neither in dropped);
        # (50, 250) survives (neither in dropped).
        long_range_kept = 2

        expected = chain_kept + long_range_kept
        assert len(filtered) == expected

        # Spot check: no surviving edge references a dropped label.
        for edge in filtered:
            assert edge["label_a"] not in dropped
            assert edge["label_b"] not in dropped
