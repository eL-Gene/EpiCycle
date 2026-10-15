"""Step 4 `adopt_workspace` — the shared-workspace hand-over.

Adopt is the QUIET half of Browse: it fills the path box, loads the state and
lists the stacks — and it is READ-ONLY.  Nothing is written to the workspace
(no `stack_segmentation_state.json`) and nothing is registered into the state
manager, because the user only *walked past* this step; they have not chosen
to work in it yet.  Browse still does everything it did before.

Qt-free: the controller methods under test are exercised on a bare instance
with stub `ui` / `state_manager` objects.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from napari_epicycle.segmentation.controller import SegmentStackController
from napari_epicycle.segmentation.state_manager import StackStateManager


class _Line:
    def __init__(self, text=""):
        self._text = text
        self.sets = []

    def text(self):
        return self._text

    def setText(self, value):
        self._text = value
        self.sets.append(value)

    def strip(self):          # pragma: no cover - convenience
        return self._text.strip()


class _Log:
    def __init__(self):
        self.lines = []

    def append(self, line):
        self.lines.append(line)


class _Combo:
    def __init__(self):
        self.index = 0

    def setCurrentIndex(self, i):
        self.index = i

    def currentIndex(self):
        return self.index

    def findText(self, _t):
        return -1


class _RunStore:
    def list_manifests(self, _ws):
        return []


def _controller(tmp_path):
    """A bare controller on the B1a fake panel (B1a:
    the fake used to build ``input_stacks_dir_line`` / ``axes_combo`` /
    ``refresh_stacks_list``, which the revamp deleted)."""
    from ._step4_fakes import UI

    ctrl = SegmentStackController.__new__(SegmentStackController)
    ctrl.ui = UI()
    ctrl.working_dir = ""
    ctrl.state_manager = StackStateManager()
    ctrl._suppress_publish = False
    ctrl._workflow_shell = None
    ctrl.run_store = _RunStore()
    ctrl._init_runtime_state()
    return ctrl


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "Proj"
    stacks = ws / "stacks"
    stacks.mkdir(parents=True)
    (stacks / "a.tif").write_bytes(b"not a real tiff")
    (stacks / "b.tif").write_bytes(b"not a real tiff")
    (ws / "stack_segmentation_state.json").write_text(json.dumps({
        "schema_version": 1,
        "stacks": {},
        "session": {"input_stacks_dir": str(stacks)},
    }), encoding="utf-8")
    return ws


class TestAdoptWorkspace:
    def test_sets_the_path_box_and_working_dir(self, tmp_path, workspace):
        ctrl = _controller(tmp_path)
        ctrl.adopt_workspace(str(workspace))
        assert ctrl.working_dir == str(workspace)
        assert ctrl.ui.workdir_line.text() == str(workspace)

    def test_no_op_when_the_step_already_has_a_workspace(self, tmp_path,
                                                         workspace):
        ctrl = _controller(tmp_path)
        ctrl.working_dir = "D:/chosen-by-hand"
        ctrl.adopt_workspace(str(workspace))
        assert ctrl.working_dir == "D:/chosen-by-hand"
        assert ctrl.ui.workdir_line.sets == []

    def test_writes_nothing_to_the_workspace(self, tmp_path, workspace):
        state = workspace / "stack_segmentation_state.json"
        before = state.read_bytes(), state.stat().st_mtime_ns
        listing_before = sorted(p.name for p in workspace.iterdir())
        ctrl = _controller(tmp_path)
        ctrl.adopt_workspace(str(workspace))
        assert (state.read_bytes(), state.stat().st_mtime_ns) == before
        assert sorted(p.name for p in workspace.iterdir()) == listing_before

    def test_registers_no_stacks_in_state(self, tmp_path, workspace):
        ctrl = _controller(tmp_path)
        ctrl.adopt_workspace(str(workspace))
        assert ctrl.state_manager.get_all_stack_statuses() == {}

    def test_still_lists_the_stacks_for_the_user(self, tmp_path, workspace):
        ctrl = _controller(tmp_path)
        ctrl.adopt_workspace(str(workspace))
        assert ctrl.ui.listed and ctrl.ui.listed[-1] == ["a.tif", "b.tif"]

    def test_does_not_publish_back_to_the_shell(self, tmp_path, workspace):
        published = []

        class _Shell:
            def set_workspace(self, path, source_step=""):
                published.append((path, source_step))

        ctrl = _controller(tmp_path)
        ctrl._workflow_shell = _Shell()
        ctrl.adopt_workspace(str(workspace))
        assert published == []

    def test_missing_stacks_dir_is_harmless(self, tmp_path):
        ws = tmp_path / "Empty"
        ws.mkdir()
        ctrl = _controller(tmp_path)
        ctrl.adopt_workspace(str(ws))
        assert ctrl.working_dir == str(ws)


class TestReAdopt:
    """The symmetric rule: the shared workspace can move under this step."""

    def test_re_adopts_when_still_holding_the_inherited_path(self, tmp_path,
                                                             workspace):
        other = tmp_path / "Other"
        other.mkdir()
        ctrl = _controller(tmp_path)
        ctrl.adopt_workspace(str(workspace))
        ctrl.adopt_workspace(str(other), previous=str(workspace))
        assert ctrl.working_dir == str(other)
        assert ctrl.ui.workdir_line.text() == str(other)

    def test_re_adopt_clears_the_previous_workspace_listing(self, tmp_path,
                                                            workspace):
        other = tmp_path / "Other"
        other.mkdir()
        ctrl = _controller(tmp_path)
        ctrl.adopt_workspace(str(workspace))
        assert ctrl.ui.listed[-1] == ["a.tif", "b.tif"]
        ctrl.adopt_workspace(str(other), previous=str(workspace))
        # The new workspace has no saved session, so the list must be empty
        # rather than still showing the old workspace's stacks.
        assert ctrl.ui.listed[-1] == []

    def test_does_not_re_adopt_a_path_the_user_chose_here(self, tmp_path,
                                                          workspace):
        ctrl = _controller(tmp_path)
        ctrl.working_dir = "D:/chosen-by-hand"
        ctrl.adopt_workspace(str(workspace), previous=str(tmp_path / "X"))
        assert ctrl.working_dir == "D:/chosen-by-hand"

    def test_re_adopt_is_still_read_only(self, tmp_path, workspace):
        other = tmp_path / "Other"
        other.mkdir()
        ctrl = _controller(tmp_path)
        ctrl.adopt_workspace(str(workspace))
        before = sorted(p.name for p in other.iterdir())
        ctrl.adopt_workspace(str(other), previous=str(workspace))
        assert sorted(p.name for p in other.iterdir()) == before
        assert ctrl.state_manager.get_all_stack_statuses() == {}


class TestScanPersistFlag:
    def test_persist_true_registers_stacks(self, tmp_path, workspace):
        ctrl = _controller(tmp_path)
        ctrl.working_dir = str(workspace)
        ctrl.state_manager.load(str(workspace))
        ctrl._scan_stacks_directory(str(workspace / "stacks"), persist=True)
        assert set(ctrl.state_manager.get_all_stack_statuses()) == {
            "a.tif", "b.tif"}

    def test_the_read_only_assertion_has_teeth(self, tmp_path, workspace):
        """Force the persisting path through the same flow: it DOES register.

        Without this, `test_registers_no_stacks_in_state` could pass for the
        wrong reason (e.g. the scan silently not running at all).
        """
        ctrl = _controller(tmp_path)
        ctrl.working_dir = str(workspace)
        ctrl.state_manager.load(str(workspace))
        ctrl._restore_session_from_state(persist=True)
        assert set(ctrl.state_manager.get_all_stack_statuses()) == {
            "a.tif", "b.tif"}

    def test_persist_false_registers_nothing(self, tmp_path, workspace):
        ctrl = _controller(tmp_path)
        ctrl.working_dir = str(workspace)
        ctrl.state_manager.load(str(workspace))
        ctrl._scan_stacks_directory(str(workspace / "stacks"), persist=False)
        assert ctrl.state_manager.get_all_stack_statuses() == {}
        assert ctrl.ui.listed[-1] == ["a.tif", "b.tif"]


class TestPublish:
    def test_publish_goes_to_the_shell_with_the_step_id(self, tmp_path):
        published = []

        class _Shell:
            def set_workspace(self, path, source_step=""):
                published.append((path, source_step))

        ctrl = _controller(tmp_path)
        ctrl._workflow_shell = _Shell()
        ctrl._publish_workspace_to_shell("D:/ws")
        assert published == [("D:/ws", "apply_3d")]

    def test_publish_is_a_no_op_without_a_shell(self, tmp_path):
        ctrl = _controller(tmp_path)
        ctrl._publish_workspace_to_shell("D:/ws")   # must not raise

    def test_publish_is_suppressed_during_adopt(self, tmp_path):
        published = []

        class _Shell:
            def set_workspace(self, path, source_step=""):
                published.append(path)

        ctrl = _controller(tmp_path)
        ctrl._workflow_shell = _Shell()
        ctrl._suppress_publish = True
        ctrl._publish_workspace_to_shell("D:/ws")
        assert published == []


class TestAdoptCorruptStateFile:
    """Adopt reports a bad file, renames none."""

    def _corrupt(self, ws):
        # the OLD place on purpose: an unmoved state is still read there
        path = ws / "artifacts" / "step4" / "step4_state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ not json", encoding="utf-8")
        return path

    def test_adopt_renames_and_writes_nothing(self, tmp_path, workspace):
        path = self._corrupt(workspace)
        listing = sorted(p.name for p in path.parent.iterdir())
        ctrl = _controller(tmp_path)
        ctrl.adopt_workspace(str(workspace))
        assert sorted(p.name for p in path.parent.iterdir()) == listing
        assert path.read_text(encoding="utf-8") == "{ not json"
        assert ctrl.state_manager.save_blocked is True
        assert any("could not be read" in line
                   for line in ctrl.ui.log_output.lines)

    def test_the_first_user_action_sets_it_aside(self, tmp_path, workspace):
        path = self._corrupt(workspace)
        ctrl = _controller(tmp_path)
        ctrl.adopt_workspace(str(workspace))
        ctrl._ensure_writable_state()
        assert not path.exists() or path.read_text(
            encoding="utf-8") != "{ not json"
        assert list(path.parent.glob("step4_state.json.corrupt-*"))
        assert ctrl.state_manager.save_blocked is False


class TestBrowseWritesTheNewStateFile:
    """A Browse writes step4_state.json; the legacy file is untouched."""

    def test_browse_migrates_and_leaves_the_legacy_file(self, tmp_path,
                                                        workspace,
                                                        monkeypatch):
        import napari_epicycle.segmentation.controller as C

        legacy = workspace / "stack_segmentation_state.json"
        before = legacy.read_bytes()
        monkeypatch.setattr(C.QFileDialog, "getExistingDirectory",
                            staticmethod(lambda *a, **k: str(workspace)))
        ctrl = _controller(tmp_path)
        ctrl._init_runtime_state()
        ctrl._browse_workdir()
        new = workspace / "step4_state.json"
        assert new.exists()
        data = json.loads(new.read_text(encoding="utf-8"))
        assert data["schema_version"] == 2
        assert data["migrated_from_legacy_at"]
        assert set(data["stacks"]) == {"a.tif", "b.tif"}
        assert legacy.read_bytes() == before

    def test_adopt_still_writes_no_state_file(self, tmp_path, workspace):
        ctrl = _controller(tmp_path)
        ctrl.adopt_workspace(str(workspace))
        assert not (workspace / "artifacts").exists()


class TestExitAfterAdopt:
    """Audit L12: leaving Step 4 after a read-only adopt writes nothing."""

    def test_exit_after_adopt_writes_no_state_file(self, tmp_path, workspace):
        ctrl = _controller(tmp_path)
        ctrl._init_runtime_state()
        ctrl.adopt_workspace(str(workspace))
        ctrl.on_step_exit()
        assert not (workspace / "artifacts").exists()

    def test_exit_after_a_user_action_saves(self, tmp_path, workspace):
        ctrl = _controller(tmp_path)
        ctrl._init_runtime_state()
        ctrl.adopt_workspace(str(workspace))
        ctrl._ensure_writable_state()               # the user acted here
        ctrl.on_step_exit()
        assert (workspace / "step4_state.json").exists()
