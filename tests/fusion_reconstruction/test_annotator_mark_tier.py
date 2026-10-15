"""Tests for FusionReconstructionController._on_annotator_mark_tier.

Post-2026-05-10: replaces the controller's single-tier
_on_annotator_label_accepted with a unified flag-only dispatcher that
handles all 5 tiers (curated / good / review / bad / garbage).

Whitelisting + df update behavior verified here via a minimal mock
controller (avoids the heavy controller initialization).
"""
from __future__ import annotations

import pandas as pd
import pytest


# Import the controller class for the method-under-test.  We bind the
# method to a minimal mock controller instance (a SimpleNamespace
# subclass) so we don't have to construct a full FusionReconstruction-
# Controller for unit tests.
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController,
)


class _MockController:
    """Minimal stand-in for FusionReconstructionController providing
    just the fields _on_annotator_mark_tier touches."""

    _VALID_TIERS = FusionReconstructionController._VALID_TIERS

    def __init__(self, df):
        self._qc_features_df = df
        self._commit_log = []
        self._append_log_messages = []

    def _commit_curation_action(self, action, label_id, msg, *, df_updater=None, **_):
        if df_updater is not None:
            df_updater()
        self._commit_log.append((action, label_id, msg))

    def _append_log(self, msg):
        self._append_log_messages.append(msg)


_mark_tier = FusionReconstructionController._on_annotator_mark_tier


def _df(rows):
    return pd.DataFrame(rows)


def test_mark_tier_good_updates_qc_tier():
    df = _df([{"label": 42, "qc_tier": "review", "label_status": "active"}])
    mc = _MockController(df)
    _mark_tier(mc, 42, "good")
    assert df.loc[df["label"] == 42, "qc_tier"].iloc[0] == "good"


def test_mark_tier_curated_updates_qc_tier():
    df = _df([{"label": 1, "qc_tier": "good", "label_status": "active"}])
    mc = _MockController(df)
    _mark_tier(mc, 1, "curated")
    assert df.loc[df["label"] == 1, "qc_tier"].iloc[0] == "curated"


def test_mark_tier_review_updates_qc_tier():
    df = _df([{"label": 5, "qc_tier": "good", "label_status": "active"}])
    mc = _MockController(df)
    _mark_tier(mc, 5, "review")
    assert df.loc[df["label"] == 5, "qc_tier"].iloc[0] == "review"


def test_mark_tier_bad_updates_qc_tier():
    df = _df([{"label": 9, "qc_tier": "review", "label_status": "active"}])
    mc = _MockController(df)
    _mark_tier(mc, 9, "bad")
    assert df.loc[df["label"] == 9, "qc_tier"].iloc[0] == "bad"


def test_mark_tier_garbage_updates_qc_tier_flag_only():
    """Garbage = flag-only.  The destructive zeroing is in
    _on_annotator_discard_garbage; mark_tier MUST NOT mutate the
    label_status column."""
    df = _df([{"label": 13, "qc_tier": "bad", "label_status": "active"}])
    mc = _MockController(df)
    _mark_tier(mc, 13, "garbage")
    assert df.loc[df["label"] == 13, "qc_tier"].iloc[0] == "garbage"
    # label_status stays "active" — garbage flag is set without
    # zeroing the volume / marking deleted.
    assert df.loc[df["label"] == 13, "label_status"].iloc[0] == "active"


def test_mark_tier_skips_invalid_tier():
    df = _df([{"label": 1, "qc_tier": "good", "label_status": "active"}])
    mc = _MockController(df)
    _mark_tier(mc, 1, "not_a_real_tier")
    # Unchanged — invalid tier rejected silently.
    assert df.loc[df["label"] == 1, "qc_tier"].iloc[0] == "good"
    assert mc._commit_log == []


def test_mark_tier_skips_deleted_rows():
    """Ghost rows (label_status=deleted) must not be re-flagged."""
    df = _df([
        {"label": 7, "qc_tier": "good", "label_status": "deleted"},
        {"label": 7, "qc_tier": "review", "label_status": "active"},
    ])
    mc = _MockController(df)
    _mark_tier(mc, 7, "bad")
    # Active row updated.
    active = df[df["label_status"] == "active"]
    assert active["qc_tier"].iloc[0] == "bad"
    # Ghost row unchanged.
    ghost = df[df["label_status"] == "deleted"]
    assert ghost["qc_tier"].iloc[0] == "good"


def test_mark_tier_handles_none_df():
    mc = _MockController(None)
    # Should silently no-op when df is None (e.g., before any patch
    # loaded).  Must not raise.
    _mark_tier(mc, 1, "good")
    assert mc._commit_log == []


def test_mark_tier_logs_action_with_tier_name():
    df = _df([{"label": 1, "qc_tier": "review", "label_status": "active"}])
    mc = _MockController(df)
    _mark_tier(mc, 1, "curated")
    assert mc._commit_log[0][0] == "mark_curated"
    assert "Annotator: marked label 1 as curated" in mc._append_log_messages


def test_legacy_alias_label_accepted_delegates_to_mark_tier_good():
    """_on_annotator_label_accepted is a back-compat wrapper kept for
    refinement-accept paths that still call it with just label_id.
    Must route to mark_tier(lid, "good")."""
    df = _df([{"label": 99, "qc_tier": "review", "label_status": "active"}])
    mc = _MockController(df)
    # Bind the real instance method so the legacy alias can self-
    # dispatch through self._on_annotator_mark_tier.
    mc._on_annotator_mark_tier = (
        FusionReconstructionController._on_annotator_mark_tier.__get__(mc)
    )
    FusionReconstructionController._on_annotator_label_accepted(mc, 99)
    assert df.loc[df["label"] == 99, "qc_tier"].iloc[0] == "good"
