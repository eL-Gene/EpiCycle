"""Phase 4b — Step 3 run-management UI handlers (controller side).

These tests build bare controllers via ``__new__`` (no Qt / napari / real
state-manager) and exercise the run-picker + Manage-panel glue:

  * ``_list_runs_for_ui``         — single summaries data source (guarded)
  * ``_format_run_picker_label``  — combo label formatting
  * ``_on_run_combo_changed``     — itemData slug → ``_switch_active_run``
  * source providers              — pretrained / step2 (stub) / step3 refined
  * ``_do_delete_run``            — disk delete + state delete both fire

The Qt dialogs themselves (NewRunDialog / ManageRunsPanel) are NOT
constructed here (Qt teardown crashes the process — known repo test notes).
"""

from __future__ import annotations

import pytest

from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController,
)
from napari_epicycle.fusion_reconstruction import _run_management


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeStateManager:
    def __init__(self, active="default"):
        self._active = active
        self.deleted = []
        self.created = []

    def get_active_run_id(self):
        return self._active

    def list_run_ids(self):
        return ["default", "cells"]

    def delete_run(self, slug, workspace=None):
        self.deleted.append(slug)

    def create_run(self, slug, **kwargs):
        self.created.append((slug, kwargs))


class _FakeCombo:
    def __init__(self, data_by_index):
        self._data = data_by_index

    def itemData(self, index):
        return self._data.get(index)


class _FakeUI:
    pass


def _bare_controller():
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    c.working_dir = ""
    c._log_messages = []
    c._append_log = lambda m: c._log_messages.append(m)
    return c


# ---------------------------------------------------------------------------
# _list_runs_for_ui
# ---------------------------------------------------------------------------

def test_list_runs_for_ui_empty_without_workspace():
    c = _bare_controller()
    c.working_dir = ""
    assert c._list_runs_for_ui() == []


def test_list_runs_for_ui_delegates_to_run_management(monkeypatch):
    c = _bare_controller()
    c.working_dir = "/ws"
    sentinel = [{"run_id": "default", "patch_counts": {"total": 3}}]
    monkeypatch.setattr(
        _run_management, "list_run_summaries", lambda ws: sentinel
    )
    assert c._list_runs_for_ui() == sentinel


def test_list_runs_for_ui_never_raises(monkeypatch):
    c = _bare_controller()
    c.working_dir = "/ws"

    def _boom(ws):
        raise RuntimeError("scan failed")

    monkeypatch.setattr(_run_management, "list_run_summaries", _boom)
    assert c._list_runs_for_ui() == []


# ---------------------------------------------------------------------------
# _format_run_picker_label
# ---------------------------------------------------------------------------

def test_format_run_picker_label_active_and_total():
    c = _bare_controller()
    label = c._format_run_picker_label({
        "run_id": "cells",
        "display_name": "Cell membranes",
        "is_active": True,
        "patch_counts": {"total": 4},
    })
    assert "●" in label
    assert "Cell membranes" in label
    assert "4 patches" in label


def test_format_run_picker_label_singular_and_inactive_and_slug_fallback():
    c = _bare_controller()
    label = c._format_run_picker_label({
        "run_id": "nuclei_v1",
        "display_name": "",
        "is_active": False,
        "patch_counts": {"total": 1},
    })
    assert "●" not in label
    assert "nuclei_v1" in label
    assert "1 patch)" in label  # singular


# ---------------------------------------------------------------------------
# _on_run_combo_changed
# ---------------------------------------------------------------------------

def test_on_run_combo_changed_switches_to_itemdata_slug():
    c = _bare_controller()
    c.state_manager = _FakeStateManager(active="default")
    c.ui = _FakeUI()
    c.ui.run_picker_combo = _FakeCombo({2: "cells"})
    switched = []
    c._switch_active_run = lambda s: switched.append(s)
    c._refresh_run_picker = lambda: None

    c._on_run_combo_changed(2)
    assert switched == ["cells"]


def test_on_run_combo_changed_noop_for_active_run():
    c = _bare_controller()
    c.state_manager = _FakeStateManager(active="cells")
    c.ui = _FakeUI()
    c.ui.run_picker_combo = _FakeCombo({0: "cells"})
    switched = []
    c._switch_active_run = lambda s: switched.append(s)
    c._refresh_run_picker = lambda: None

    c._on_run_combo_changed(0)
    assert switched == []  # already active — no switch


def test_on_run_combo_changed_noop_for_separator_none_data():
    c = _bare_controller()
    c.state_manager = _FakeStateManager(active="default")
    c.ui = _FakeUI()
    c.ui.run_picker_combo = _FakeCombo({1: None})  # separator row
    switched = []
    c._switch_active_run = lambda s: switched.append(s)
    c._refresh_run_picker = lambda: None

    c._on_run_combo_changed(1)
    assert switched == []


# ---------------------------------------------------------------------------
# Source providers
# ---------------------------------------------------------------------------

def test_list_pretrained_names_nonempty():
    c = _bare_controller()
    names = c._list_pretrained_names()
    assert isinstance(names, list) and names
    assert "cyto3" in names


def test_list_step2_run_sources_empty_without_workspace():
    c = _bare_controller()
    c.working_dir = ""
    assert c._list_step2_run_sources() == []


def test_list_step2_run_sources_aggregates_per_run(monkeypatch):
    c = _bare_controller()
    c.working_dir = "/ws"
    from napari_epicycle.hitl_training_annotation import (
        _run_management as s2_rm,
    )
    monkeypatch.setattr(
        s2_rm,
        "list_run_model_sources",
        lambda ws: [
            {"run_id": "rA", "run_display_name": "Run A", "bucket": "lateral",
             "mtime_iso": "2026-06-01T00:00:00Z", "exists": True},
            {"run_id": "rA", "run_display_name": "Run A", "bucket": "axial",
             "mtime_iso": "2026-06-02T00:00:00Z", "exists": True},
            # rB has only a lateral model → not offered (needs the pair).
            {"run_id": "rB", "run_display_name": "Run B", "bucket": "lateral",
             "mtime_iso": "2026-06-03T00:00:00Z", "exists": True},
        ],
    )
    out = c._list_step2_run_sources()
    assert [e["run_id"] for e in out] == ["rA"]
    assert out[0]["display_name"] == "Run A"
    assert out[0]["mtime_iso"] == "2026-06-02T00:00:00Z"


def test_list_step3_refined_sources_empty_without_workspace():
    c = _bare_controller()
    c.working_dir = ""
    assert c._list_step3_refined_sources() == []


def test_list_step3_refined_sources_passes_trained_only(monkeypatch):
    # Audit HIGH-1A: the reuse SOURCE list must exclude copy-on-reuse seeds.
    c = _bare_controller()
    c.working_dir = "/ws"
    captured = {}

    def _fake(ws, *, trained_only=False, default_nested=False):
        captured["ws"] = ws
        captured["trained_only"] = trained_only
        captured["default_nested"] = default_nested
        return [{"bundle_name": "trainedA", "run_id": None, "bucket": "lateral"}]

    monkeypatch.setattr(
        _run_management, "list_cross_run_refined_models", _fake
    )
    out = c._list_step3_refined_sources()
    assert captured["trained_only"] is True
    # Phase 7: default_nested is threaded from the workspace flag (False here).
    assert captured["default_nested"] is False
    assert out == [{"bundle_name": "trainedA", "run_id": None,
                    "bucket": "lateral"}]


# ---------------------------------------------------------------------------
# _on_new_run_clicked — model_source_mode mapping at create boundary (MED-1)
# ---------------------------------------------------------------------------

class _FakeNewRunDialog:
    """Stand-in for NewRunDialog: returns a preset result, no Qt."""
    _preset_result = None

    def __init__(self, *args, **kwargs):
        pass

    def exec_(self):
        return True

    def get_result(self):
        return type(self)._preset_result


def _make_new_run_controller(state_active="default"):
    c = _bare_controller()
    c.working_dir = "/ws"
    c.ui = _FakeUI()
    c.state_manager = _FakeStateManager(active=state_active)
    c._append_log = lambda m: None
    c._show_run_error = lambda title, body: None
    c._refresh_run_picker = lambda: None
    c._switch_active_run = lambda s: None
    c._list_pretrained_names = lambda: ["cyto3"]
    c._list_step2_run_sources = lambda: []
    c._list_step3_refined_sources = lambda: []
    return c


@pytest.mark.parametrize("rich_mode,expected_state_mode", [
    ("pretrained", "pretrained"),
    ("browse", "import"),
    ("step2_run", "import"),
    ("step3_refined", "import"),
    (None, "auto"),
])
def test_new_run_maps_model_source_mode_to_state_vocab(
    monkeypatch, rich_mode, expected_state_mode
):
    c = _make_new_run_controller()

    # Patch the dialog the controller imports + the run-management calls.
    from napari_epicycle.fusion_reconstruction import _new_run_dialog
    _FakeNewRunDialog._preset_result = {
        "slug": "cells_v2",
        "display_name": "Cells V2",
        "target": "cells",
        "description": "",
        "backend": "cellpose",
        "init_spec": {"mode": rich_mode} if rich_mode else None,
        "notes": "",
        "activate": False,
    }
    monkeypatch.setattr(_new_run_dialog, "NewRunDialog", _FakeNewRunDialog)
    monkeypatch.setattr(
        _run_management, "create_run_on_disk",
        lambda ws, slug, **kw: {"run_id": slug},
    )
    monkeypatch.setattr(
        _run_management, "materialize_run_model",
        lambda ws, slug, spec, target="": {
            "lateral_model_path": "/m/lat",
            "axial_model_path": "/m/ax",
            "model_source_mode": rich_mode,
        },
    )

    c._on_new_run_clicked()

    assert len(c.state_manager.created) == 1
    slug, kwargs = c.state_manager.created[0]
    assert slug == "cells_v2"
    # The STATE entry's model_source_mode is the MAPPED consumer value...
    assert kwargs["model_source_mode"] == expected_state_mode
    # ...and never a rich value outside the consumer vocabulary.
    assert kwargs["model_source_mode"] in {"auto", "import", "pretrained"}
    # Rich provenance is still carried by created_from (the init_spec).
    assert kwargs["created_from"] == (
        {"mode": rich_mode} if rich_mode else None
    )


# ---------------------------------------------------------------------------
# _do_delete_run
# ---------------------------------------------------------------------------

def test_do_delete_run_calls_both_delete_fns(monkeypatch):
    c = _bare_controller()
    c.working_dir = "/ws"
    c.state_manager = _FakeStateManager(active="default")
    c._refresh_run_picker = lambda: None
    disk_calls = []
    monkeypatch.setattr(
        _run_management,
        "delete_run_from_disk",
        lambda ws, slug, confirm=False: disk_calls.append((ws, slug, confirm)),
    )

    ok = c._do_delete_run("cells")
    assert ok is True
    assert disk_calls == [("/ws", "cells", True)]
    assert c.state_manager.deleted == ["cells"]


def test_do_delete_run_switches_off_active_first(monkeypatch):
    c = _bare_controller()
    c.working_dir = "/ws"
    c.state_manager = _FakeStateManager(active="cells")
    c._refresh_run_picker = lambda: None
    switched = []
    c._switch_active_run = lambda s: switched.append(s)
    monkeypatch.setattr(
        _run_management,
        "delete_run_from_disk",
        lambda ws, slug, confirm=False: None,
    )

    c._do_delete_run("cells")
    # Switched to the default run BEFORE deleting the active one.
    assert switched == ["default"]


def test_do_delete_run_no_workspace_returns_false():
    c = _bare_controller()
    c.working_dir = ""
    assert c._do_delete_run("cells") is False


# ---------------------------------------------------------------------------
# _do_clean_intermediates (Clean intermediates feature)
# ---------------------------------------------------------------------------

class _FakeStateManagerNested(_FakeStateManager):
    def __init__(self, active="default", nested=False):
        super().__init__(active=active)
        self._nested = nested

    def get_default_run_nested(self):
        return self._nested


def _make_clean_controller(active="default", nested=False):
    c = _bare_controller()
    c.working_dir = "/ws"
    c.ui = _FakeUI()
    c.state_manager = _FakeStateManagerNested(active=active, nested=nested)
    c._refresh_run_picker = lambda: None
    c._run_display_name_for = lambda s: s
    c._info_calls = []
    c._show_run_info = lambda t, b: c._info_calls.append((t, b))
    c._confirm_result = True
    c._confirm_calls = []
    c._confirm_clean_intermediates = (
        lambda slug, info: (c._confirm_calls.append((slug, info))
                            or c._confirm_result)
    )
    return c


def test_do_clean_no_workspace_returns_false():
    c = _make_clean_controller()
    c.working_dir = ""
    assert c._do_clean_intermediates("cells") is False


def test_do_clean_noop_with_info_modal_when_no_files(monkeypatch):
    c = _make_clean_controller()
    monkeypatch.setattr(
        _run_management, "summarize_run_intermediates",
        lambda ws, slug, run_path_id=None: {
            "n_patches": 0, "bytes": 0, "n_files": 0},
    )
    cleaned = []
    monkeypatch.setattr(
        _run_management, "clean_run_intermediates",
        lambda *a, **k: cleaned.append(True),
    )
    ok = c._do_clean_intermediates("cells")
    assert ok is False
    assert len(c._info_calls) == 1            # info modal shown
    assert cleaned == []                       # clean NOT invoked
    assert c._confirm_calls == []              # confirm NOT shown


def test_do_clean_calls_clean_and_logs_on_confirm(monkeypatch):
    c = _make_clean_controller()
    monkeypatch.setattr(
        _run_management, "summarize_run_intermediates",
        lambda ws, slug, run_path_id=None: {
            "n_patches": 4, "bytes": 234, "n_files": 8},
    )
    clean_calls = []
    monkeypatch.setattr(
        _run_management, "clean_run_intermediates",
        lambda ws, slug, run_path_id=None, **k: (
            clean_calls.append((ws, slug, run_path_id))
            or {"n_cleared": 8, "bytes_freed": 234}),
    )
    ok = c._do_clean_intermediates("cells")
    assert ok is True
    assert clean_calls == [("/ws", "cells", "cells")]
    assert c._confirm_calls and c._confirm_calls[0][0] == "cells"
    assert any("Cleaned intermediates" in m for m in c._log_messages)


def test_do_clean_aborts_when_declined(monkeypatch):
    c = _make_clean_controller()
    c._confirm_result = False
    monkeypatch.setattr(
        _run_management, "summarize_run_intermediates",
        lambda ws, slug, run_path_id=None: {
            "n_patches": 4, "bytes": 234, "n_files": 8},
    )
    cleaned = []
    monkeypatch.setattr(
        _run_management, "clean_run_intermediates",
        lambda *a, **k: cleaned.append(True),
    )
    ok = c._do_clean_intermediates("cells")
    assert ok is False
    assert cleaned == []


def test_do_clean_resolves_path_id_default_flat(monkeypatch):
    c = _make_clean_controller(active="default", nested=False)
    captured = {}
    monkeypatch.setattr(
        _run_management, "summarize_run_intermediates",
        lambda ws, slug, run_path_id=None: (
            captured.update(path_id=run_path_id)
            or {"n_patches": 0, "bytes": 0, "n_files": 0}),
    )
    c._do_clean_intermediates("default")
    assert captured["path_id"] is None      # flat default


def test_do_clean_resolves_path_id_default_nested(monkeypatch):
    c = _make_clean_controller(active="default", nested=True)
    captured = {}
    monkeypatch.setattr(
        _run_management, "summarize_run_intermediates",
        lambda ws, slug, run_path_id=None: (
            captured.update(path_id=run_path_id)
            or {"n_patches": 0, "bytes": 0, "n_files": 0}),
    )
    c._do_clean_intermediates("default")
    assert captured["path_id"] == "default"  # nested default tree


def test_do_clean_drops_cached_views_when_active(monkeypatch):
    c = _make_clean_controller(active="cells")
    c._cached_source_path = "/ws/patch_a.tif"
    monkeypatch.setattr(
        _run_management, "summarize_run_intermediates",
        lambda ws, slug, run_path_id=None: {
            "n_patches": 1, "bytes": 10, "n_files": 2},
    )
    monkeypatch.setattr(
        _run_management, "clean_run_intermediates",
        lambda ws, slug, run_path_id=None, **k: {
            "n_cleared": 2, "bytes_freed": 10},
    )
    c._do_clean_intermediates("cells")   # cells IS active
    assert c._cached_source_path == ""


def test_do_clean_keeps_cached_views_when_not_active(monkeypatch):
    c = _make_clean_controller(active="default")
    c._cached_source_path = "/ws/patch_a.tif"
    monkeypatch.setattr(
        _run_management, "summarize_run_intermediates",
        lambda ws, slug, run_path_id=None: {
            "n_patches": 1, "bytes": 10, "n_files": 2},
    )
    monkeypatch.setattr(
        _run_management, "clean_run_intermediates",
        lambda ws, slug, run_path_id=None, **k: {
            "n_cleared": 2, "bytes_freed": 10},
    )
    c._do_clean_intermediates("cells")   # cleaning a NON-active run
    assert c._cached_source_path == "/ws/patch_a.tif"


# ---------------------------------------------------------------------------
# _human_bytes
# ---------------------------------------------------------------------------

def test_human_bytes_scales():
    c = _bare_controller()
    assert c._human_bytes(0) == "0 B"
    assert c._human_bytes(512).endswith("B")
    assert "KB" in c._human_bytes(2048)
    assert "MB" in c._human_bytes(5 * 1024 * 1024)
    assert "GB" in c._human_bytes(3 * 1024 ** 3)
