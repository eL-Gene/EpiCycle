"""A2 fold (audit 1): one rule names the patch an action works on -- the
patch whose QC table is in memory (``_qc_view_patch_name``) FIRST, then the
Curation selection, then the Session highlight -- the rule
``_save_tiers_csv`` / ``_save_curation_prepare`` already used.  Since a
Session click no longer reloads the table, every door that wrote,
reloaded or moved a status by "Curation or Session" could name the Session
highlight while the table, the layers and the viewer showed another patch
(``refresh_curation_bucket`` drops the Curation selection on every rebuild).

Bare controllers via ``__new__``: A is in view, the Curation selection is
empty, Session highlights B.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from napari_epicycle.fusion_reconstruction import _paths
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController as C,
    _active_run_path_id,
)


class _SM:
    def __init__(self):
        self.status = {}
        self._state = {"patches": {"A.tif": {}, "B.tif": {}}}

    def get_patch_status(self, name):
        return self.status.get(name, "curating")

    def set_patch_status(self, name, status, **kw):
        self.status[name] = status

    def schedule_save(self):
        pass

    def get_active_run_id(self):
        return "default"


def _ctrl(tmp_path):
    c = C.__new__(C)
    c.working_dir = str(tmp_path)
    c.state_manager = _SM()
    c._qc_view_patch_name = "A.tif"
    c._selected_curation_patch = lambda: ""
    c._selected_patch_name = lambda: "B.tif"
    c.logged = []
    c._append_log = lambda t: c.logged.append(t)
    return c


def _qc_csv(c, name, tiers):
    d = _paths.qc_dir(c.working_dir, run_id=_active_run_path_id(c))
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{Path(name).stem}_features.csv"
    pd.DataFrame({"label": range(1, len(tiers) + 1),
                  "qc_tier": tiers}).to_csv(p, index=False)
    return p


def test_the_rule_itself(tmp_path):
    c = _ctrl(tmp_path)
    assert c._active_qc_patch() == "A.tif"
    c._qc_view_patch_name = ""
    assert c._active_qc_patch() == "B.tif"
    c._selected_curation_patch = lambda: "C.tif"
    assert c._active_qc_patch() == "C.tif"


def test_the_reload_reads_the_table_in_views_csv(tmp_path, monkeypatch):
    """H1: the reload (the Curation switch's finish, Discard's revert)
    reads A's CSV, never the Session highlight's."""
    c = _ctrl(tmp_path)
    _qc_csv(c, "A.tif", ["good", "good"])
    _qc_csv(c, "B.tif", ["bad"])
    seen = []
    import napari_epicycle.fusion_reconstruction.qc_features as Q
    real = Q.load_patch_csv
    monkeypatch.setattr(Q, "load_patch_csv",
                        lambda p: seen.append(Path(p).name) or real(p))

    class Stop(Exception):
        pass

    def stop(*a, **k):
        raise Stop
    monkeypatch.setattr(c, "_scan_qc_df_for_duplicates", stop, raising=False)
    with pytest.raises(Stop):
        c._reload_current_patch_features()
    assert seen == ["A_features.csv"]


def _gated_status_door(c, monkeypatch):
    monkeypatch.setattr(c, "_save_curation_checked", lambda: True,
                        raising=False)
    for name in ("_refresh_patch_list_colors", "_refresh_summary_dashboard",
                 "_refresh_patch_state_list", "_refresh_save_button_state",
                 "_refresh_step3_lock", "_set_curation_status",
                 "_close_3d_curation_annotator", "_update_curation_buttons"):
        monkeypatch.setattr(c, name, lambda *a, **k: None, raising=False)


def test_accept_moves_the_table_in_views_status(tmp_path, monkeypatch):
    """H3: Accept names A."""
    c = _ctrl(tmp_path)
    names = []
    monkeypatch.setattr(c, "_save_curation_checked",
                        lambda: names.append(c._active_qc_patch()) or False,
                        raising=False)
    c._on_3d_curation_accept()
    assert names == ["A.tif"]
    assert any("A.tif" in t for t in c.logged)
    assert not any("B.tif" in t for t in c.logged)


def test_finalize_names_the_table_in_view(tmp_path, monkeypatch):
    """H3: Finalize sets A's status, never B's."""
    c = _ctrl(tmp_path)
    monkeypatch.setattr(c, "_pr_any_block_flag", lambda: False,
                        raising=False)
    for flag in ("_load_locked", "_step3_locked"):
        setattr(c, flag, False)
    monkeypatch.setattr(c, "_save_curation_checked", lambda: False,
                        raising=False)
    c._finalize_curation()
    assert any("A.tif" in t for t in c.logged), c.logged
    assert "B.tif" not in c.state_manager.status


def test_extract_names_the_table_in_views_csv(tmp_path, monkeypatch):
    """H4: the CSV the extraction writes is A's (the same patch as the
    labels path)."""
    c = _ctrl(tmp_path)
    labels = tmp_path / "labels_3d.tif"
    labels.write_bytes(b"x")
    monkeypatch.setattr(c, "_get_active_labels_path", lambda: str(labels))
    seen = {}

    class Stop(Exception):
        pass

    def confirm(name):
        seen["name"] = name
        raise Stop
    monkeypatch.setattr(c, "_confirm_reextract_themed", confirm)
    _qc_csv(c, "A.tif", ["good"])
    _qc_csv(c, "B.tif", ["good"])
    with pytest.raises(Stop):
        c._extract_features_current()
    assert seen["name"] == "A.tif"


def test_the_seg_channel_reads_the_table_in_views_manifest(tmp_path,
                                                           monkeypatch):
    c = _ctrl(tmp_path)
    asked = []
    monkeypatch.setattr(c, "_resolve_patch_payload",
                        lambda name: asked.append(name) or {
                            "seg_config": {"channels": [2, 0]}})
    c._get_active_seg_channel()
    assert asked == ["A.tif"]


def test_the_labels_layer_lookups_use_the_table_in_view(tmp_path,
                                                        monkeypatch):
    """M2."""
    c = _ctrl(tmp_path)
    c.viewer = type("V", (), {"layers": []})()
    keys = []
    monkeypatch.setattr(c, "_patch_layer_key",
                        lambda name: keys.append(name) or "")
    c._find_active_labels_layer()
    c._find_all_patch_labels_layers()
    assert keys == ["A.tif", "A.tif"]


def test_the_session_eye_asks_about_unsaved_work_first(tmp_path,
                                                      monkeypatch):
    """H2: View closes the annotator and clears the curation layers, so
    the eye (and the foot's View) run the unsaved-changes guard first; a
    cancel views nothing."""
    c = _ctrl(tmp_path)
    c.viewer = object()
    c._curation_dirty = True
    asked = []
    monkeypatch.setattr(c, "_guard_unsaved_changes",
                        lambda: asked.append(1) or False)
    monkeypatch.setattr(c, "_pr_lock_flag", lambda: None)
    monkeypatch.setattr(c, "_pr_ticks_locked", lambda: False)
    monkeypatch.setattr(c, "_resolve_patch_payload",
                        lambda name: pytest.fail("viewed after a cancel"))
    c._on_session_eye("B.tif")
    assert asked == [1]


def test_commit_triage_prep_names_the_table_in_view(tmp_path, monkeypatch):
    """M3 (replaces the source-string test): the prep's patch and its CSV
    are A's."""
    c = _ctrl(tmp_path)
    c._qc_features_df = pd.DataFrame({"label": [1, 2],
                                      "qc_tier": ["good", "garbage"],
                                      "label_status": ["active", "active"]})
    c.ui = type("U", (), {})()
    monkeypatch.setattr(c, "_zero_labels_in_viewer", lambda ids: None)
    monkeypatch.setattr(c, "_get_active_labels_path", lambda: "")
    monkeypatch.setattr(c, "_find_active_labels_layer", lambda: None)
    prep = c._commit_triage_prepare()
    assert prep["patch_name"] == "A.tif"
    flat = repr(prep["persist_kwargs"])
    assert "A_features.csv" in flat and "B_features" not in flat
