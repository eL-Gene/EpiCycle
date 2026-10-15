"""Tests for the 2026-06-08 Step 3 trained-model reader unification.

Covers the shared scanner + the three readers that converge on it:

* ``_list_all_model_sources`` — unified, mtime-sorted list wrapping the
  canonical Step 2 listers (``list_run_model_sources`` +
  ``list_step3_refined_sources``).
* ``_populate_refinement_source_combo`` — the refine-init source combo
  fix: now includes new-layout Step 2 models + Step 3 refined bundles
  (was the stale-combo bug — only legacy ``{ws}/models/round_*/``).
* ``_discover_workspace_models`` — the PREDICT-path auto-fill, converged
  onto the same scanner.  This module exercises the PRODUCTION method
  against the same fixtures the dualpath contract test mirrors, proving
  the convergence preserves the behavior contract.

All Qt-free: a fake controller binds the real methods onto a
``SimpleNamespace`` (mirrors ``test_refinement_registry.py``'s harness).

Step 2 / packages are READ-ONLY here — the fixtures plant on-disk
layouts and the tests only consume them via the canonical listers.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Fake controller harness (Qt-free)
# ---------------------------------------------------------------------------


class _Combo:
    """Fake QComboBox supporting text + userData (Qt text/data split).

    Mirrors the real QComboBox contract the production code now relies
    on: ``addItem(text, userData=None)``, ``itemData(i)`` /
    ``currentData()`` for the path, ``setItemData(i, value, role)`` for
    the tooltip, and ``setCurrentIndex``.
    """

    def __init__(self):
        self._items = []          # list[str] (display text)
        self._data = []           # list[Any] (userData per item)
        self._tooltips = []       # list[str | None] (ToolTipRole per item)
        self._index = -1

    def clear(self):
        self._items = []
        self._data = []
        self._tooltips = []
        self._index = -1

    def addItem(self, text, userData=None):
        self._items.append(text)
        self._data.append(userData)
        self._tooltips.append(None)
        if self._index < 0:
            self._index = 0

    def itemText(self, i):
        return self._items[i] if 0 <= i < len(self._items) else ""

    def itemData(self, i):
        return self._data[i] if 0 <= i < len(self._data) else None

    def setItemData(self, i, value, role=None):
        # Distinguish ToolTipRole (a Qt enum) from userData.  We only
        # need the tooltip bucket for the production tooltip set; any
        # role that isn't None lands in the tooltip slot here.
        if 0 <= i < len(self._items):
            if role is None:
                self._data[i] = value
            else:
                self._tooltips[i] = value

    def itemToolTip(self, i):
        return self._tooltips[i] if 0 <= i < len(self._tooltips) else None

    def count(self):
        return len(self._items)

    def setCurrentText(self, t):
        # Lenient fake (editable-combo semantics): if the text isn't an
        # existing item, add it so mode-selector combos that the test
        # doesn't pre-populate (e.g. refine_init_combo) still report the
        # set text via currentText().
        if t in self._items:
            self._index = self._items.index(t)
        else:
            self.addItem(t)
            self._index = len(self._items) - 1

    def setCurrentIndex(self, i):
        if 0 <= i < len(self._items):
            self._index = i

    def currentText(self):
        if 0 <= self._index < len(self._items):
            return self._items[self._index]
        return ""

    def currentData(self):
        if 0 <= self._index < len(self._data):
            return self._data[self._index]
        return None

    def currentIndex(self):
        return self._index


class _LineEdit:
    def __init__(self):
        self._text = ""

    def text(self):
        return self._text

    def setText(self, t):
        self._text = t


def _make_controller(working_dir: Path):
    """Minimal fake controller with reader methods bound from the real class."""
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    logs: list = []

    ui = SimpleNamespace()
    ui.lateral_model_path = _LineEdit()
    ui.axial_model_path = _LineEdit()
    ui.refinement_source_combo = _Combo()
    ui.refinement_source_combo_axial = _Combo()

    ctrl = SimpleNamespace(
        working_dir=str(working_dir),
        ui=ui,
        _append_log=logs.append,
        _logs=logs,
        _refresh_model_status=lambda: None,
    )
    for name in (
        "_list_all_model_sources",
        "_populate_refinement_source_combo",
        "_discover_workspace_models",
    ):
        method = getattr(FusionReconstructionController, name)
        setattr(ctrl, name, method.__get__(ctrl))
    return ctrl


# ---------------------------------------------------------------------------
# On-disk fixture builders (mirror the dualpath test's layouts)
# ---------------------------------------------------------------------------


def _plant_new_flat(ws: Path, run_id: str, bucket: str, round_id: str) -> Path:
    d = ws / "artifacts" / "step2" / "runs" / run_id / bucket / round_id / "model"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{run_id}_{bucket}_{round_id}"
    f.write_bytes(f"{run_id} {bucket} {round_id}".encode())
    (d / f"{run_id}_{bucket}_{round_id}_train_loss.npy").write_bytes(b"loss")
    return f


def _plant_legacy_split(ws: Path, run_id: str, bucket: str, round_id: str) -> Path:
    d = ws / "artifacts" / "step2" / "runs" / run_id / bucket / "models" / round_id
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{run_id}_{bucket}"
    f.write_bytes(f"split {bucket}".encode())
    (d / f"{run_id}_{bucket}_train_loss.npy").write_bytes(b"loss")
    return f


def _plant_legacy_workspace_models(ws: Path, round_id: str, bucket: str) -> Path:
    d = ws / "models" / round_id
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"model_{bucket}_round_001.ckpt"
    f.write_bytes(f"legacy {bucket}".encode())
    return f


def _plant_step3_refined(ws: Path, bundle: str, bucket: str) -> Path:
    d = ws / "artifacts" / "step3" / "refinement" / "models" / bundle
    d.mkdir(parents=True, exist_ok=True)
    f = d / bucket
    f.write_bytes(f"refined {bucket}".encode())
    return f


def _plant_legacy_no_keyword(ws: Path, round_id: str, filename: str) -> Path:
    """A legacy {ws}/models/round_NN/<file> WITHOUT an orientation keyword.

    Baseline's suffix-only combo walk made these selectable; the
    reader-unify's keyword-scanner skips them ("Unknown bucket").  The
    consumer-side union in _list_all_model_sources must restore them.
    """
    d = ws / "models" / round_id
    d.mkdir(parents=True, exist_ok=True)
    f = d / filename
    f.write_bytes(f"legacy {filename}".encode())
    return f


# ---------------------------------------------------------------------------
# _list_all_model_sources
# ---------------------------------------------------------------------------


class TestListAllModelSources:

    def test_empty_working_dir_returns_empty(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        ctrl.working_dir = ""
        assert ctrl._list_all_model_sources() == []

    def test_empty_workspace_returns_empty(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        assert ctrl._list_all_model_sources() == []

    def test_new_flat_step2_included(self, tmp_path):
        _plant_new_flat(tmp_path, "default", "lateral", "round_00")
        _plant_new_flat(tmp_path, "default", "axial", "round_00")
        ctrl = _make_controller(tmp_path)
        out = ctrl._list_all_model_sources()
        buckets = {e["bucket"] for e in out}
        assert buckets == {"lateral", "axial"}
        assert all(e["source_type"] == "step2_run" for e in out)
        assert all(e["run_id"] == "default" for e in out)
        # weights paths point into the new-flat layout
        for e in out:
            assert "/round_00/model/" in e["weights_path"].replace("\\", "/")

    def test_legacy_split_step2_included(self, tmp_path):
        _plant_legacy_split(tmp_path, "default", "lateral", "round_00")
        _plant_legacy_split(tmp_path, "default", "axial", "round_00")
        ctrl = _make_controller(tmp_path)
        out = ctrl._list_all_model_sources()
        assert {e["bucket"] for e in out} == {"lateral", "axial"}
        for e in out:
            assert "/models/round_00/" in e["weights_path"].replace("\\", "/")

    def test_legacy_workspace_models_tagged_legacy(self, tmp_path):
        _plant_legacy_workspace_models(tmp_path, "round_00", "lateral")
        _plant_legacy_workspace_models(tmp_path, "round_00", "axial")
        ctrl = _make_controller(tmp_path)
        out = ctrl._list_all_model_sources()
        assert {e["bucket"] for e in out} == {"lateral", "axial"}
        assert all(e["source_type"] == "step2_legacy" for e in out)

    def test_step3_refined_included(self, tmp_path):
        _plant_step3_refined(tmp_path, "refined_v1", "lateral")
        _plant_step3_refined(tmp_path, "refined_v1", "axial")
        ctrl = _make_controller(tmp_path)
        out = ctrl._list_all_model_sources()
        refined = [e for e in out if e["source_type"] == "step3_refined"]
        assert {e["bucket"] for e in refined} == {"lateral", "axial"}
        assert all(e["run_id"] == "refined_v1" for e in refined)

    def test_combined_step2_and_step3(self, tmp_path):
        _plant_new_flat(tmp_path, "default", "lateral", "round_00")
        _plant_step3_refined(tmp_path, "refined_v1", "lateral")
        ctrl = _make_controller(tmp_path)
        out = ctrl._list_all_model_sources()
        types = {e["source_type"] for e in out}
        assert "step2_run" in types
        assert "step3_refined" in types

    def test_sorted_newest_first(self, tmp_path):
        old = _plant_new_flat(tmp_path, "old_run", "lateral", "round_00")
        new = _plant_new_flat(tmp_path, "new_run", "lateral", "round_00")
        # Force a clear mtime gap (second-precision in mtime_iso).
        import os
        os.utime(old, (time.time() - 5000, time.time() - 5000))
        os.utime(new, (time.time() + 5000, time.time() + 5000))
        ctrl = _make_controller(tmp_path)
        out = ctrl._list_all_model_sources()
        lateral = [e for e in out if e["bucket"] == "lateral"]
        assert lateral[0]["weights_path"] == str(new)

    def test_human_label_is_readable(self, tmp_path):
        _plant_new_flat(tmp_path, "default", "lateral", "round_00")
        ctrl = _make_controller(tmp_path)
        out = ctrl._list_all_model_sources()
        label = out[0]["human_label"]
        assert "lateral" in label
        assert "round_00" in label

    def test_legacy_model_without_keyword_included(self, tmp_path):
        """FIX 3: a legacy model with no orientation keyword in its name

        must STILL appear in the init sources (the keyword-scanner skips
        it; the consumer-side union restores baseline coverage).
        """
        f = _plant_legacy_no_keyword(tmp_path, "round_03", "cellpose_model")
        ctrl = _make_controller(tmp_path)
        out = ctrl._list_all_model_sources()
        paths = {e["weights_path"] for e in out}
        assert str(f) in paths
        entry = next(e for e in out if e["weights_path"] == str(f))
        assert entry["source_type"] == "step2_legacy"
        assert "round_03" in entry["human_label"]

    def test_legacy_keyword_and_non_keyword_not_double_listed(self, tmp_path):
        """The union dedups: a keyworded legacy file (emitted by the

        scanner) is NOT re-emitted by the fallback walk.
        """
        kw = _plant_legacy_workspace_models(tmp_path, "round_00", "lateral")
        nokw = _plant_legacy_no_keyword(tmp_path, "round_00", "plain_model")
        ctrl = _make_controller(tmp_path)
        out = ctrl._list_all_model_sources()
        paths = [e["weights_path"] for e in out]
        # No duplicates overall.
        assert len(paths) == len(set(paths))
        # Both files present exactly once.
        assert paths.count(str(kw)) == 1
        assert paths.count(str(nokw)) == 1

    def test_run_display_name_present_for_step2(self, tmp_path):
        """FIX 2 support: step2 entries carry a friendly run_display_name."""
        _plant_new_flat(tmp_path, "default", "lateral", "round_00")
        ctrl = _make_controller(tmp_path)
        out = ctrl._list_all_model_sources()
        step2 = [e for e in out if e["source_type"] == "step2_run"]
        assert step2
        assert all(e.get("run_display_name") for e in step2)


# ---------------------------------------------------------------------------
# _populate_refinement_source_combo (the stale-combo bug fix)
# ---------------------------------------------------------------------------


class TestRefineInitComboPopulation:

    def test_new_layout_models_appear_in_combo(self, tmp_path):
        """The bug: new-layout Step 2 models never showed as init sources.

        2026-06-24 display fix: the combo now stores the friendly label
        as TEXT and the absolute path as DATA — so assert on item-data.
        """
        _plant_new_flat(tmp_path, "default", "lateral", "round_00")
        _plant_new_flat(tmp_path, "default", "axial", "round_00")
        ctrl = _make_controller(tmp_path)
        ctrl._populate_refinement_source_combo()
        combo = ctrl.ui.refinement_source_combo
        paths = [combo.itemData(i) for i in range(combo.count())]
        assert len(paths) == 2
        assert all("/round_00/model/" in p.replace("\\", "/") for p in paths)
        # Both combos populated identically (Mode=Both twin routing).
        axial = ctrl.ui.refinement_source_combo_axial
        axial_paths = [axial.itemData(i) for i in range(axial.count())]
        assert axial_paths == paths

    def test_step3_refined_appears_in_combo(self, tmp_path):
        _plant_step3_refined(tmp_path, "refined_v1", "lateral")
        ctrl = _make_controller(tmp_path)
        ctrl._populate_refinement_source_combo()
        combo = ctrl.ui.refinement_source_combo
        # The friendly label carries the bundle name; the path carries
        # the bundle dir.
        texts = [combo.itemText(i) for i in range(combo.count())]
        paths = [combo.itemData(i) for i in range(combo.count())]
        assert any("refined_v1" in t for t in texts)
        assert any("refined_v1" in p for p in paths)

    def test_item_text_is_friendly_label_data_is_path(self, tmp_path):
        """Display text != path; item-DATA == the absolute weights path.

        The 2026-06-24 display fix: the combo TEXT is the friendly
        human_label (readable) while the value the resolver consumes
        (currentData) is the absolute path.  This is the regression that
        was hidden by the old text==path coverage.
        """
        f = _plant_new_flat(tmp_path, "default", "lateral", "round_00")
        ctrl = _make_controller(tmp_path)
        ctrl._populate_refinement_source_combo()
        combo = ctrl.ui.refinement_source_combo
        # Exactly one item; its text is a friendly label, its data the path.
        assert combo.count() == 1
        text = combo.itemText(0)
        data = combo.itemData(0)
        assert data == str(f)            # value IS the path
        assert text != str(f)            # display is NOT the path
        assert "lateral" in text         # friendly label is readable
        assert "round_00" in text
        # And the path is reachable as a tooltip for hover-discovery.
        assert combo.itemToolTip(0) == str(f)

    def test_resolver_reads_path_from_current_data(self, tmp_path):
        """_resolve_init_mode_for_bucket returns the PATH (currentData)."""
        f = _plant_new_flat(tmp_path, "default", "lateral", "round_00")
        ctrl = _make_controller(tmp_path)
        # Bind the resolver + its dependencies onto the fake controller.
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        m = FusionReconstructionController._resolve_init_mode_for_bucket
        ctrl._resolve_init_mode_for_bucket = m.__get__(ctrl)
        ctrl.ui.refine_init_combo = _Combo()
        ctrl.ui.refine_init_combo.addItem("From HITL model")
        ctrl.ui.refine_init_combo.setCurrentText("From HITL model")
        ctrl._populate_refinement_source_combo()
        # Select the only source item.
        ctrl.ui.refinement_source_combo.setCurrentIndex(0)
        mode, path = ctrl._resolve_init_mode_for_bucket("lateral")
        assert mode == "custom"
        assert path == str(f)            # the resolved path is the DATA, not text

    def test_paths_deduped(self, tmp_path):
        _plant_new_flat(tmp_path, "default", "lateral", "round_00")
        ctrl = _make_controller(tmp_path)
        ctrl._populate_refinement_source_combo()
        combo = ctrl.ui.refinement_source_combo
        paths = [combo.itemData(i) for i in range(combo.count())]
        assert len(paths) == len(set(paths))

    def test_selection_preserved_across_repopulate(self, tmp_path):
        f = _plant_new_flat(tmp_path, "default", "lateral", "round_00")
        ctrl = _make_controller(tmp_path)
        ctrl._populate_refinement_source_combo()
        # Select by index (the item whose DATA is the path).
        ctrl.ui.refinement_source_combo.setCurrentIndex(0)
        assert ctrl.ui.refinement_source_combo.currentData() == str(f)
        # Repopulate (e.g. after a training run finishes) — selection is
        # restored by matching item-DATA (the path), not visible text.
        ctrl._populate_refinement_source_combo()
        assert ctrl.ui.refinement_source_combo.currentData() == str(f)

    def test_empty_working_dir_clears_combo(self, tmp_path):
        _plant_new_flat(tmp_path, "default", "lateral", "round_00")
        ctrl = _make_controller(tmp_path)
        ctrl._populate_refinement_source_combo()
        assert ctrl.ui.refinement_source_combo.count() > 0
        ctrl.working_dir = ""
        ctrl._populate_refinement_source_combo()
        assert ctrl.ui.refinement_source_combo.count() == 0

    def test_legacy_no_keyword_model_selectable_in_combo(self, tmp_path):
        """FIX 3: a non-keyworded legacy model is a selectable combo item

        (its path reachable as item-DATA) — the regression surface.
        """
        f = _plant_legacy_no_keyword(tmp_path, "round_02", "my_model")
        ctrl = _make_controller(tmp_path)
        ctrl._populate_refinement_source_combo()
        combo = ctrl.ui.refinement_source_combo
        paths = [combo.itemData(i) for i in range(combo.count())]
        assert str(f) in paths


# ---------------------------------------------------------------------------
# _discover_workspace_models (PRODUCTION method against dualpath fixtures)
#
# Auto-discovery convergence onto the shared scanner was REVERTED (the
# canonical list_step3_refined_sources lister doesn't recognize Step 3
# refined bundles named `*_model`, which this method's broader rglob
# does — see test_persistent_model_selection.py).  These tests exercise
# the PRODUCTION method (unchanged) against the Step 2 layouts that BOTH
# the shared scanner and this method cover identically — proving the
# Step 2 predict-path discovery still works alongside the combo fix.
# ---------------------------------------------------------------------------


class TestDiscoverWorkspaceModelsStep2Layouts:

    def test_new_only_layout_fills_both_paths(self, tmp_path):
        _plant_new_flat(tmp_path, "default", "lateral", "round_00")
        _plant_new_flat(tmp_path, "default", "axial", "round_00")
        ctrl = _make_controller(tmp_path)
        ctrl._discover_workspace_models()
        lat = ctrl.ui.lateral_model_path.text()
        ax = ctrl.ui.axial_model_path.text()
        assert "/lateral/" in lat.replace("\\", "/")
        assert "/model/" in lat.replace("\\", "/")
        assert "/axial/" in ax.replace("\\", "/")
        assert "/model/" in ax.replace("\\", "/")

    def test_legacy_split_only_fills_both_paths(self, tmp_path):
        _plant_legacy_split(tmp_path, "default", "lateral", "round_00")
        _plant_legacy_split(tmp_path, "default", "axial", "round_00")
        ctrl = _make_controller(tmp_path)
        ctrl._discover_workspace_models()
        assert "/lateral/models/" in ctrl.ui.lateral_model_path.text().replace("\\", "/")
        assert "/axial/models/" in ctrl.ui.axial_model_path.text().replace("\\", "/")

    def test_empty_workspace_clears_both(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        ctrl.ui.lateral_model_path.setText("stale_lateral")
        ctrl.ui.axial_model_path.setText("stale_axial")
        ctrl._discover_workspace_models()
        assert ctrl.ui.lateral_model_path.text() == ""
        assert ctrl.ui.axial_model_path.text() == ""

    def test_one_bucket_only_other_cleared(self, tmp_path):
        _plant_new_flat(tmp_path, "default", "lateral", "round_00")
        ctrl = _make_controller(tmp_path)
        ctrl.ui.axial_model_path.setText("stale_axial")
        ctrl._discover_workspace_models()
        assert ctrl.ui.lateral_model_path.text() != ""
        assert ctrl.ui.axial_model_path.text() == ""

    def test_picks_newer_across_layouts(self, tmp_path):
        # Legacy split round_00 + flat round_01; make flat newer.
        _plant_legacy_split(tmp_path, "default", "lateral", "round_00")
        flat = _plant_new_flat(tmp_path, "default", "lateral", "round_01")
        import os
        future = time.time() + 5000
        os.utime(flat, (future, future))
        ctrl = _make_controller(tmp_path)
        ctrl._discover_workspace_models()
        assert "/round_01/model/" in ctrl.ui.lateral_model_path.text().replace("\\", "/")


# ---------------------------------------------------------------------------
# Registry action routing for step2_run rows (Qt-free)
# ---------------------------------------------------------------------------


def _make_registry_controller(working_dir: Path):
    """Fake controller with the registry-action methods + a richer UI."""
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    logs: list = []

    ui = SimpleNamespace()
    ui.lateral_model_path = _LineEdit()
    ui.axial_model_path = _LineEdit()
    ui.refine_init_combo = _Combo()
    ui.refinement_source_combo = _Combo()
    ui.refinement_source_combo_axial = _Combo()

    ctrl = SimpleNamespace(
        working_dir=str(working_dir),
        ui=ui,
        _append_log=logs.append,
        _logs=logs,
        _refresh_model_status=lambda: None,
        _persist_current_model_selection=lambda mode="": None,
    )
    for name in (
        "_list_all_model_sources",
        "_registry_action_use",
        "_registry_action_open_folder",
        "_registry_action_set_init",
    ):
        method = getattr(FusionReconstructionController, name)
        setattr(ctrl, name, method.__get__(ctrl))
    return ctrl


class TestRegistryActionStep2Run:

    def test_use_sets_lateral_predict_path_only(self, tmp_path):
        f = _plant_new_flat(tmp_path, "default", "lateral", "round_00")
        ctrl = _make_registry_controller(tmp_path)
        ctrl._registry_action_use({
            "_kind": "step2_run",
            "bucket": "lateral",
            "weights_path": str(f),
            "run_id": "default",
        })
        assert ctrl.ui.lateral_model_path.text() == str(f)
        # Axial untouched (Step 2 rows are single-bucket).
        assert ctrl.ui.axial_model_path.text() == ""

    def test_use_sets_axial_predict_path_only(self, tmp_path):
        f = _plant_new_flat(tmp_path, "default", "axial", "round_00")
        ctrl = _make_registry_controller(tmp_path)
        ctrl._registry_action_use({
            "_kind": "step2_run",
            "bucket": "axial",
            "weights_path": str(f),
            "run_id": "default",
        })
        assert ctrl.ui.axial_model_path.text() == str(f)
        assert ctrl.ui.lateral_model_path.text() == ""

    def test_set_init_flips_to_hitl_and_sets_lateral_combo(self, tmp_path):
        f = _plant_new_flat(tmp_path, "default", "lateral", "round_00")
        ctrl = _make_registry_controller(tmp_path)
        ctrl._registry_action_set_init({
            "_kind": "step2_run",
            "bucket": "lateral",
            "weights_path": str(f),
            "run_id": "default",
            "human_label": "default · lateral · round_00",
        })
        assert ctrl.ui.refine_init_combo.currentText() == "From HITL model"
        # 2026-06-24 display fix: the source combo stores the path as
        # item-DATA (text is the friendly label).
        assert ctrl.ui.refinement_source_combo.currentData() == str(f)
        assert (
            ctrl.ui.refinement_source_combo.currentText()
            == "default · lateral · round_00"
        )
        # Axial combo untouched.
        assert ctrl.ui.refinement_source_combo_axial.currentData() is None

    def test_set_init_axial_sets_axial_combo(self, tmp_path):
        f = _plant_new_flat(tmp_path, "default", "axial", "round_00")
        ctrl = _make_registry_controller(tmp_path)
        ctrl._registry_action_set_init({
            "_kind": "step2_run",
            "bucket": "axial",
            "weights_path": str(f),
            "run_id": "default",
        })
        assert ctrl.ui.refine_init_combo.currentText() == "From HITL model"
        assert ctrl.ui.refinement_source_combo_axial.currentData() == str(f)
        assert ctrl.ui.refinement_source_combo.currentData() is None

    def test_open_folder_reveals_weights_parent(self, tmp_path, monkeypatch):
        f = _plant_new_flat(tmp_path, "default", "lateral", "round_00")
        ctrl = _make_registry_controller(tmp_path)
        called = []
        import os
        import subprocess
        monkeypatch.setattr(
            os, "startfile",
            lambda p: called.append(("startfile", p)),
            raising=False,
        )
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: called.append(("run", a)),
        )
        ctrl._registry_action_open_folder({
            "_kind": "step2_run",
            "weights_path": str(f),
            "run_id": "default",
        })
        assert len(called) == 1
        revealed = str(called[0][1]) if called[0][0] == "startfile" else str(called[0][1])
        assert "round_00" in revealed and "model" in revealed

    def test_view_models_click_passes_step2_models(self, tmp_path, monkeypatch):
        """_on_view_refined_models_clicked threads step2_models into the dialog."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        _plant_new_flat(tmp_path, "default", "lateral", "round_00")

        captured = {}

        # Patch the dialog's show_dialog to capture kwargs (no Qt).
        import napari_epicycle.fusion_reconstruction.refinement_registry_dialog as rrd

        def _fake_show_dialog(*args, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(
            rrd.RefinementRegistryDialog, "show_dialog",
            staticmethod(_fake_show_dialog),
        )

        ctrl = _make_registry_controller(tmp_path)
        ctrl._scan_refinement_model_registry = (lambda: [])
        ctrl.ui.parent = lambda: None
        method = getattr(FusionReconstructionController, "_on_view_refined_models_clicked")
        setattr(ctrl, "_on_view_refined_models_clicked", method.__get__(ctrl))
        ctrl._on_view_refined_models_clicked()

        assert "step2_models" in captured
        s2 = captured["step2_models"]
        assert len(s2) >= 1
        assert all(
            e["source_type"] in ("step2_run", "step2_legacy") for e in s2
        )

    def test_dialog_feed_excludes_bucketless_legacy(self, tmp_path, monkeypatch):
        """The registry dialog feed drops bucket='' (non-keyworded legacy)
        entries — the dialog's bucket-routed Use / Set-as-init actions can't
        route an empty bucket — while keeping bucketed run + keyworded-legacy
        entries.  The bucketless entry stays selectable in the flat init combo
        (covered by test_legacy_no_keyword_model_selectable_in_combo)."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        import napari_epicycle.fusion_reconstruction.refinement_registry_dialog as rrd

        captured = {}
        monkeypatch.setattr(
            rrd.RefinementRegistryDialog, "show_dialog",
            staticmethod(lambda *a, **k: captured.update(k)),
        )
        ctrl = _make_registry_controller(tmp_path)
        ctrl._scan_refinement_model_registry = (lambda: [])
        ctrl.ui.parent = lambda: None
        # A run model (real bucket), a keyworded legacy (real bucket), and a
        # bucketless non-keyworded legacy entry (bucket="").
        ctrl._list_all_model_sources = lambda: [
            {"weights_path": "/ws/runs/default/lateral/round_00/model/w",
             "bucket": "lateral", "source_type": "step2_run"},
            {"weights_path": "/ws/models/round_00/lateral_model",
             "bucket": "lateral", "source_type": "step2_legacy"},
            {"weights_path": "/ws/models/round_00/some_model",
             "bucket": "", "source_type": "step2_legacy"},
        ]
        method = getattr(
            FusionReconstructionController, "_on_view_refined_models_clicked"
        )
        setattr(ctrl, "_on_view_refined_models_clicked", method.__get__(ctrl))
        ctrl._on_view_refined_models_clicked()

        s2 = captured["step2_models"]
        paths = [e["weights_path"] for e in s2]
        # Bucketless entry excluded from the dialog feed...
        assert "/ws/models/round_00/some_model" not in paths
        # ...but bucketed run + keyworded-legacy entries are kept.
        assert "/ws/runs/default/lateral/round_00/model/w" in paths
        assert "/ws/models/round_00/lateral_model" in paths
        assert all(e.get("bucket") for e in s2)


# ---------------------------------------------------------------------------
# Dialog rendering (requires QApplication)
# ---------------------------------------------------------------------------


@pytest.fixture
def _qapp():
    try:
        from qtpy.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app
    except Exception:
        pytest.skip("No QApplication available")


class TestDialogStep2Section:

    def _step2_entry(self, bucket="lateral", run_display_name="My Run"):
        return {
            "weights_path": f"/ws/runs/default/{bucket}/round_00/model/w",
            "bucket": bucket,
            "mtime": 1.0,
            "source_type": "step2_run",
            "run_id": "default",
            "run_display_name": run_display_name,
            "round_id": "round_00",
            "human_label": f"{run_display_name} · {bucket} · round_00",
        }

    def test_step2_row_cells_align_to_headers(self, _qapp):
        """FIX 2: each step2_run value lands under its correct header.

        Headers: Name | Mode | Lateral | Axial | Created | Init.
        The Name cell shows the FRIENDLY run name (not the raw slug);
        the absolute weights path appears in NO visible cell, only in
        the Init cell's tooltip.
        """
        from napari_epicycle.fusion_reconstruction.refinement_registry_dialog import (
            RefinementRegistryDialog,
        )
        entry = self._step2_entry("lateral", run_display_name="My Run")
        dlg = RefinementRegistryDialog(rows=[], step2_models=[entry])
        # Locate the table row holding the step2_run payload.
        row_idx = next(
            i for i, (k, _p) in enumerate(dlg._row_payloads) if k == "step2_run"
        )

        def _cell(col):
            it = dlg._table.item(row_idx, col)
            return it.text() if it is not None else None

        name, mode, lateral, axial, created, init = (
            _cell(c) for c in range(6)
        )
        # Name: friendly run name (round suffix is fine), NOT the slug-only
        # "default" and NOT the absolute path.
        assert "My Run" in name
        assert name != entry["weights_path"]
        # Mode: the bucket.
        assert mode == "Lateral"
        # Lateral / Axial: pair-count columns — N/A for a step2 row.
        assert lateral == "—"
        assert axial == "—"
        # Init: a short source label, NOT the absolute path.
        assert init in ("Step 2 run", "Step 2 (legacy)")
        assert init != entry["weights_path"]
        # The absolute path is in NO visible cell.
        all_cell_texts = [_cell(c) for c in range(6)]
        assert entry["weights_path"] not in all_cell_texts
        # ...but it IS the Init cell's tooltip (hover-discoverable).
        init_item = dlg._table.item(row_idx, 5)
        assert init_item.toolTip() == entry["weights_path"]
        # Selection→value payload is untouched: UserRole on col 0 maps
        # back to the original step2 dict with weights_path intact.
        from qtpy.QtCore import Qt as _Qt
        idx_payload = dlg._table.item(row_idx, 0).data(_Qt.UserRole)
        kind, payload = dlg._row_payloads[idx_payload]
        assert kind == "step2_run"
        assert payload["weights_path"] == entry["weights_path"]
        dlg.deleteLater()

    def test_section_renders_when_step2_models_present(self, _qapp):
        from napari_epicycle.fusion_reconstruction.refinement_registry_dialog import (
            RefinementRegistryDialog,
        )
        dlg = RefinementRegistryDialog(
            rows=[],
            step2_models=[self._step2_entry("lateral"), self._step2_entry("axial")],
        )
        kinds = [k for (k, _p) in dlg._row_payloads]
        assert "step2_run" in kinds
        # Header label present.
        labels = [
            p.get("label") for (k, p) in dlg._row_payloads if k == "header"
        ]
        assert "Step 2 Run Models" in labels
        dlg.deleteLater()

    def test_section_absent_when_no_step2_models(self, _qapp):
        from napari_epicycle.fusion_reconstruction.refinement_registry_dialog import (
            RefinementRegistryDialog,
        )
        dlg = RefinementRegistryDialog(rows=[], step2_models=[])
        labels = [
            p.get("label") for (k, p) in dlg._row_payloads if k == "header"
        ]
        assert "Step 2 Run Models" not in labels
        dlg.deleteLater()

    def test_step2_row_selection_dispatches_init(self, _qapp):
        from napari_epicycle.fusion_reconstruction.refinement_registry_dialog import (
            RefinementRegistryDialog,
        )
        init_calls = []
        dlg = RefinementRegistryDialog(
            rows=[],
            step2_models=[self._step2_entry("lateral")],
            on_init=lambda payload: init_calls.append(payload),
        )
        # Find the table row index of the step2_run payload.
        target_idx = next(
            i for i, (k, _p) in enumerate(dlg._row_payloads) if k == "step2_run"
        )
        dlg._table.selectRow(target_idx)
        dlg._handle_init_clicked()
        assert len(init_calls) == 1
        assert init_calls[0]["_kind"] == "step2_run"
        assert init_calls[0]["bucket"] == "lateral"
        dlg.deleteLater()

    def test_step2_row_init_button_enabled(self, _qapp):
        from napari_epicycle.fusion_reconstruction.refinement_registry_dialog import (
            RefinementRegistryDialog,
        )
        dlg = RefinementRegistryDialog(
            rows=[],
            step2_models=[self._step2_entry("lateral")],
        )
        target_idx = next(
            i for i, (k, _p) in enumerate(dlg._row_payloads) if k == "step2_run"
        )
        dlg._table.selectRow(target_idx)
        dlg._refresh_button_enabled_state()
        assert dlg._use_btn.isEnabled()
        assert dlg._init_btn.isEnabled()
        assert dlg._open_btn.isEnabled()
        dlg.deleteLater()

    def test_step2_row_use_dispatches(self, _qapp):
        from napari_epicycle.fusion_reconstruction.refinement_registry_dialog import (
            RefinementRegistryDialog,
        )
        use_calls = []
        dlg = RefinementRegistryDialog(
            rows=[],
            step2_models=[self._step2_entry("axial")],
            on_use=lambda payload: use_calls.append(payload),
        )
        target_idx = next(
            i for i, (k, _p) in enumerate(dlg._row_payloads) if k == "step2_run"
        )
        dlg._table.selectRow(target_idx)
        dlg._handle_use_clicked()
        assert len(use_calls) == 1
        assert use_calls[0]["_kind"] == "step2_run"
        assert use_calls[0]["bucket"] == "axial"
        dlg.deleteLater()


class TestRefineInitComboBucketDefault:
    def test_axial_combo_defaults_to_an_axial_source(self, tmp_path):
        """F4: no prior selection -> each combo starts on the newest source
        of its OWN bucket, the full list stays in both."""
        import os
        import time
        ax = _plant_new_flat(tmp_path, "default", "axial", "round_00")
        time.sleep(0.05)
        lat = _plant_new_flat(tmp_path, "default", "lateral", "round_00")
        os.utime(lat, (time.time() + 10, time.time() + 10))   # lateral newest
        ctrl = _make_controller(tmp_path)
        ctrl._populate_refinement_source_combo()
        p = ctrl.ui.refinement_source_combo
        a = ctrl.ui.refinement_source_combo_axial
        assert p.count() == a.count() == 2
        assert p.currentData() == str(lat)
        assert a.currentData() == str(ax)
