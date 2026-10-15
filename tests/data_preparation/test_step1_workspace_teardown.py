"""Step 1 workspace-change teardown - the shared-workspace change.

A workspace change is a PROJECT change.  Before this teardown existed, Step 1's
`output_file_path.textChanged` chain flushed the source-state save, dropped
the pending recipe, cleared the contrast cache and restored the new
workspace's source list - but it left the PREVIOUS project's pixels on the
shared canvas: the loaded source layers, the preview render, the box
overlay, the patch-location boxes and the processed-store layers, with
`_loaded_source_id` still pointing at a source that belongs to a folder the
user has left.

B5 adds `_teardown_for_workspace_change(reason)`, called from BOTH doors
that can change the workspace - Browse (`on_workspace_browsed`) and a shell
re-adopt (`adopt_workspace(path, previous=...)`) - and REFUSES the change
outright while any Step 1 job is running, mirroring Step 2's training guard.

Qt-free: bare controller instances with stub widgets, the same style as
`test_step1_adopt_workspace.py`.
"""
from __future__ import annotations

import os

import pytest

from napari_epicycle.data_preparation.controller import (
    DataPreparationController,
)


class _Line:
    def __init__(self, text=""):
        self._text = text
        self.sets = []

    def text(self):
        return self._text

    def setText(self, value):
        self._text = value
        self.sets.append(value)


class _Widget:
    def __init__(self):
        self.output_file_path = _Line()
        self.logged = []

    def _log(self, prefix, msg):
        self.logged.append((prefix, msg))

    def __getattr__(self, name):
        """Absorb the widget calls the real unload path makes (all no-ops here).

        Only reached for names the stub does not define, so `_log` and the
        path box keep their real behaviour.
        """
        if name.startswith("__"):
            raise AttributeError(name)
        return _Anything()


class _Anything:
    """Callable, attribute-permissive no-op — a stand-in for any sub-widget."""

    def __call__(self, *a, **k):
        return None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Anything()

    def __iter__(self):
        return iter(())

    def __bool__(self):
        return False


class _Timer:
    def __init__(self):
        self.stopped = 0

    def stop(self):
        self.stopped += 1


#: Every helper `_teardown_for_workspace_change` delegates to, in the order
#: the spec fixes (flush first, canvas last).
TEARDOWN_ORDER = [
    "_flush_source_state_save",
    "_detach_intensity_worker",
    "_close_capture_dialogs",
    "box_tool.reset",
    "_teardown_preview",
    "_clear_patch_locations",
    "_clear_processed_store_layers",
    "_unload_source_inner",
    "_clear_source_list_for_workspace_change",
]

#: (attribute, truthy value) for every Step 1 job that must REFUSE a change.
JOB_FLAGS = [
    ("_source_loading", True),
    ("_largedata_preparing", True),
    ("_processing", True),
    ("_extract_running", True),
    ("_trim_running", True),
]


class _BoxTool:
    def __init__(self, calls):
        self._calls = calls

    def reset(self):
        self._calls.append("box_tool.reset")


def _controller(with_spies=True):
    ctrl = DataPreparationController.__new__(DataPreparationController)
    ctrl.widget = _Widget()
    ctrl._step1_migration_offered = set()
    ctrl._step1_sources_restored = set()
    ctrl._migration_offer_timer = _Timer()
    ctrl._suppress_publish = False
    ctrl._workflow_shell = None
    ctrl.calls = []
    ctrl.status_amber = []
    ctrl._set_workspace_status_amber = ctrl.status_amber.append
    for attr, _ in JOB_FLAGS:
        setattr(ctrl, attr, False)
    if with_spies:
        for name in TEARDOWN_ORDER:
            if name.startswith("box_tool."):
                continue
            setattr(ctrl, name,
                    (lambda n: (lambda *a, **k: ctrl.calls.append(n)))(name))
        ctrl.box_tool = _BoxTool(ctrl.calls)
    return ctrl


def _key(path):
    return os.path.normcase(os.path.normpath(path))


class TestTeardownHelper:
    """`_teardown_for_workspace_change` delegates; it re-implements nothing."""

    def test_calls_every_helper_in_the_spec_order(self):
        ctrl = _controller()
        ctrl._teardown_for_workspace_change("browse")
        assert ctrl.calls == TEARDOWN_ORDER

    def test_a_failing_helper_does_not_stop_the_rest(self):
        """Teardown is best-effort: one broken piece must not strand the others."""
        ctrl = _controller()

        def _boom(*_a, **_k):
            raise RuntimeError("nope")

        ctrl._teardown_preview = _boom
        ctrl._teardown_for_workspace_change("browse")
        assert "_clear_processed_store_layers" in ctrl.calls
        assert "_unload_source_inner" in ctrl.calls

    def test_missing_box_tool_is_tolerated(self):
        ctrl = _controller()
        ctrl.box_tool = None
        ctrl._teardown_for_workspace_change("browse")
        assert "_unload_source_inner" in ctrl.calls

    def test_it_releases_the_restore_one_shot(self, tmp_path):
        """The arriving workspace's source list MUST restore (2.3).

        `_step1_sources_restored` keeps the reconcile off every keystroke by
        claiming "this workspace's rows are already listed".  The teardown
        empties the list, so that claim is false for every key in it — and a
        user browsing A -> B -> A would land on A with an EMPTY (2) list.
        Verified live before this was fixed.
        """
        ctrl = _controller(with_spies=False)
        ctrl.widget.source_file_list = None
        ctrl._step1_sources_restored.add(_key(str(tmp_path / "old")))
        ctrl._clear_source_list_for_workspace_change()
        assert ctrl._step1_sources_restored == set()

    def test_it_drops_the_source_reload_hint(self):
        """Note 1: otherwise (2) paints "click Load to view it again" for a
        project the user has left -- the re-adopt teardown runs while Step 1
        is off screen, and `on_step_enter` reads this flag on the way back."""
        ctrl = _controller()
        ctrl._source_needs_reload = True
        ctrl._teardown_for_workspace_change("browse")
        assert ctrl._source_needs_reload is False

    def test_it_logs_one_plain_line_naming_the_reason(self):
        ctrl = _controller()
        ctrl._teardown_for_workspace_change("browse")
        assert any("browse" in msg for _p, msg in ctrl.widget.logged)


class TestSourceListClear:
    """`_populate_source_list_from_registry` only ADDS, so the list must be
    emptied on a workspace change or the previous project's rows stay listed
    under the new path — and clicking one would load a foreign file."""

    def test_the_list_is_cleared_and_the_selection_dropped(self):
        cleared = []

        class _List:
            def clear(self):
                cleared.append(True)

        ctrl = _controller(with_spies=False)
        ctrl.widget.source_file_list = _List()
        ctrl._suspend_source_restore = False
        ctrl._selected_source_path = "D:/old/stack.tif"
        ctrl._refresh_source_file_list = lambda: None
        ctrl._clear_source_list_for_workspace_change()
        assert cleared == [True]
        assert ctrl._selected_source_path is None

    def test_the_selection_funnel_is_released_even_if_clear_raises(self):
        class _List:
            def clear(self):
                raise RuntimeError("nope")

        ctrl = _controller(with_spies=False)
        ctrl.widget.source_file_list = _List()
        ctrl._suspend_source_restore = False
        ctrl._selected_source_path = "D:/old/stack.tif"
        ctrl._refresh_source_file_list = lambda: None
        with pytest.raises(RuntimeError):
            ctrl._clear_source_list_for_workspace_change()
        assert ctrl._suspend_source_restore is False

    def test_the_combo_mirror_is_refreshed_after_the_clear(self):
        """Note 5: the (2) mirror must not keep the old workspace's rows when
        the new path is not a directory (so nothing repopulates)."""
        calls = []

        class _List:
            def clear(self):
                calls.append("clear")

        ctrl = _controller(with_spies=False)
        ctrl.widget.source_file_list = _List()
        ctrl._suspend_source_restore = False
        ctrl._selected_source_path = None
        ctrl._refresh_source_file_list = lambda: calls.append("refresh")
        ctrl._clear_source_list_for_workspace_change()
        assert calls == ["clear", "refresh"]

    def test_no_list_widget_is_tolerated(self):
        ctrl = _controller(with_spies=False)
        ctrl.widget.source_file_list = None
        ctrl._clear_source_list_for_workspace_change()


class TestBrowseWritesTheBoxItself:
    """FIX (lifecycle audit): the box must move BEFORE the shell is told.

    The publish marks `project_setup` as `browsed` and the shell then
    recomputes divergence by READING this step's path box.  Publishing while
    the box still held the OLD folder entered Step 1 into `_diverged_steps`
    against itself and logged a spurious "diverged" line; the banner healed
    on the next repaint but the sidebar tooltip kept claiming
    "Browsed elsewhere: Project Setup -> OLD" until the next step switch.
    """

    def test_the_box_holds_the_new_path_when_the_shell_is_told(self, tmp_path):
        seen = []

        class _Shell:
            def __init__(self, line):
                self._line = line

            def set_workspace(self, path, source_step=""):
                # what `_refresh_divergence` would read at this instant
                seen.append((path, self._line.text()))

        ctrl = _controller()
        ctrl._workflow_shell = _Shell(ctrl.widget.output_file_path)
        ctrl.widget.output_file_path.setText(str(tmp_path / "A"))
        ctrl.on_workspace_browsed(str(tmp_path / "B"))
        assert seen == [(str(tmp_path / "B"), str(tmp_path / "B"))]

    def test_the_step_is_not_diverged_against_itself(self, tmp_path):
        """The real shell rule, exercised end to end."""
        diverged = {}

        class _Shell:
            """A faithful miniature of `set_workspace` + `_refresh_divergence`."""

            def __init__(self, line):
                self._line = line
                self.shared = ""
                self.origin = {}
                self.logged = []

            def set_workspace(self, path, source_step=""):
                if source_step:
                    self.origin[source_step] = "browsed"
                self.shared = path
                diverged.clear()
                for step, origin in self.origin.items():
                    if origin != "browsed":
                        continue
                    step_path = self._line.text()
                    if step_path and os.path.normcase(step_path) !=                             os.path.normcase(self.shared):
                        diverged[step] = step_path
                        self.logged.append(step)

        ctrl = _controller()
        shell = _Shell(ctrl.widget.output_file_path)
        ctrl._workflow_shell = shell
        ctrl.widget.output_file_path.setText(str(tmp_path / "A"))
        ctrl.on_workspace_browsed(str(tmp_path / "B"))
        assert "project_setup" not in diverged
        assert shell.logged == []

    def test_a_refused_browse_leaves_the_box_alone(self, tmp_path):
        ctrl = _controller()
        ctrl.widget.output_file_path.setText(str(tmp_path / "A"))
        ctrl._trim_running = True
        assert ctrl.on_workspace_browsed(str(tmp_path / "B")) is False
        assert ctrl.widget.output_file_path.text() == str(tmp_path / "A")

    def test_a_refusal_is_shown_to_the_user_not_just_logged(self, tmp_path):
        """Note 4: the user picked a folder and the box did not move."""
        ctrl = _controller()
        ctrl.widget.output_file_path.setText(str(tmp_path / "A"))
        ctrl._extract_running = True
        ctrl.on_workspace_browsed(str(tmp_path / "B"))
        assert ctrl.status_amber, "nothing was said in the UI"
        said = " ".join(ctrl.status_amber).lower()
        assert "job" in said or "running" in said


class TestJobGuard:
    def test_no_job_running_is_free_to_change(self):
        ctrl = _controller()
        assert ctrl._step1_job_in_flight() == ""

    @pytest.mark.parametrize("attr,value", JOB_FLAGS)
    def test_each_running_job_is_named(self, attr, value):
        ctrl = _controller()
        setattr(ctrl, attr, value)
        assert ctrl._step1_job_in_flight() != ""

    @pytest.mark.parametrize("attr,value", JOB_FLAGS)
    def test_the_label_reads_as_a_sentence(self, attr, value):
        """Each label is the SUBJECT of "<label> is running, so ..." — the
        first draft said "A source is still loading is running"."""
        ctrl = _controller()
        setattr(ctrl, attr, value)
        sentence = f"{ctrl._step1_job_in_flight()} is running"
        assert " is running is running" not in sentence
        assert "loading is running" not in sentence

    def test_a_missing_flag_attribute_is_not_a_job(self):
        ctrl = _controller()
        for attr, _ in JOB_FLAGS:
            delattr(ctrl, attr)
        assert ctrl._step1_job_in_flight() == ""


class TestBrowse:
    def test_browse_elsewhere_tears_down(self, tmp_path):
        ctrl = _controller()
        ctrl.widget.output_file_path.setText(str(tmp_path / "A"))
        assert ctrl.on_workspace_browsed(str(tmp_path / "B")) is True
        assert ctrl.calls == TEARDOWN_ORDER

    def test_browse_from_an_empty_box_tears_down_nothing(self, tmp_path):
        """First-ever Browse: there is no previous project to unload."""
        ctrl = _controller()
        assert ctrl.on_workspace_browsed(str(tmp_path / "B")) is True
        assert ctrl.calls == []

    def test_browse_to_the_same_workspace_tears_down_nothing(self, tmp_path):
        ctrl = _controller()
        ctrl.widget.output_file_path.setText(str(tmp_path / "A"))
        assert ctrl.on_workspace_browsed(str(tmp_path / "A")) is True
        assert ctrl.calls == []

    def test_same_workspace_across_case_and_slashes_tears_down_nothing(
            self, tmp_path):
        a = str(tmp_path / "A")
        ctrl = _controller()
        ctrl.widget.output_file_path.setText(a)
        ctrl.on_workspace_browsed(a.replace("\\", "/"))
        assert ctrl.calls == []

    @pytest.mark.parametrize("attr,value", JOB_FLAGS)
    def test_a_running_job_refuses_the_browse(self, attr, value, tmp_path):
        ctrl = _controller()
        ctrl.widget.output_file_path.setText(str(tmp_path / "A"))
        setattr(ctrl, attr, value)
        assert ctrl.on_workspace_browsed(str(tmp_path / "B")) is False
        assert ctrl.calls == []
        assert ctrl.widget.output_file_path.text() == str(tmp_path / "A")

    @pytest.mark.parametrize("attr,value", JOB_FLAGS)
    def test_the_refusal_is_logged_in_plain_words(self, attr, value, tmp_path):
        ctrl = _controller()
        ctrl.widget.output_file_path.setText(str(tmp_path / "A"))
        setattr(ctrl, attr, value)
        ctrl.on_workspace_browsed(str(tmp_path / "B"))
        prefixes = " ".join(p for p, _m in ctrl.widget.logged)
        joined = " ".join(msg for _p, msg in ctrl.widget.logged).lower()
        assert "WORKSPACE" in prefixes
        assert "finish" in joined or "running" in joined

    def test_a_refused_browse_does_not_publish(self, tmp_path):
        published = []

        class _Shell:
            def set_workspace(self, path, source_step=""):
                published.append(path)

        ctrl = _controller()
        ctrl._workflow_shell = _Shell()
        ctrl.widget.output_file_path.setText(str(tmp_path / "A"))
        ctrl._extract_running = True
        ctrl.on_workspace_browsed(str(tmp_path / "B"))
        assert published == []

    def test_an_allowed_browse_still_publishes(self, tmp_path):
        published = []

        class _Shell:
            def set_workspace(self, path, source_step=""):
                published.append((path, source_step))

        ctrl = _controller()
        ctrl._workflow_shell = _Shell()
        ctrl.widget.output_file_path.setText(str(tmp_path / "A"))
        ctrl.on_workspace_browsed(str(tmp_path / "B"))
        assert published == [(str(tmp_path / "B"), "project_setup")]

    def test_an_empty_path_is_ignored_and_refused(self):
        ctrl = _controller()
        assert ctrl.on_workspace_browsed("") is False
        assert ctrl.calls == []


class TestReAdopt:
    def test_a_re_adopt_tears_down(self, tmp_path):
        a, b = str(tmp_path / "A"), str(tmp_path / "B")
        ctrl = _controller()
        ctrl.adopt_workspace(a)
        ctrl.calls.clear()
        ctrl.adopt_workspace(b, previous=a)
        assert ctrl.calls == TEARDOWN_ORDER
        assert ctrl.widget.output_file_path.text() == b

    def test_a_virgin_adopt_tears_down_nothing(self, tmp_path):
        """Startup prefill / first entry: nothing of ours is on the canvas."""
        ctrl = _controller()
        ctrl.adopt_workspace(str(tmp_path / "A"))
        assert ctrl.calls == []

    def test_an_unchanged_re_adopt_tears_down_nothing(self, tmp_path):
        a = str(tmp_path / "A")
        ctrl = _controller()
        ctrl.adopt_workspace(a)
        ctrl.calls.clear()
        ctrl.adopt_workspace(a, previous=a)
        assert ctrl.calls == []

    def test_a_path_the_user_chose_here_is_never_torn_down(self, tmp_path):
        a, b = str(tmp_path / "A"), str(tmp_path / "B")
        ctrl = _controller()
        ctrl.widget.output_file_path.setText("D:/chosen-by-hand")
        ctrl.adopt_workspace(b, previous=a)
        assert ctrl.calls == []
        assert ctrl.widget.output_file_path.text() == "D:/chosen-by-hand"

    @pytest.mark.parametrize("attr,value", JOB_FLAGS)
    def test_a_running_job_refuses_the_re_adopt(self, attr, value, tmp_path):
        a, b = str(tmp_path / "A"), str(tmp_path / "B")
        ctrl = _controller()
        ctrl.adopt_workspace(a)
        ctrl.calls.clear()
        setattr(ctrl, attr, value)
        ctrl.adopt_workspace(b, previous=a)
        assert ctrl.calls == []
        assert ctrl.widget.output_file_path.text() == a

    def test_a_running_job_does_not_block_a_virgin_adopt(self, tmp_path):
        """Nothing to lose: an empty box has no project to protect."""
        ctrl = _controller()
        ctrl._extract_running = True
        ctrl.adopt_workspace(str(tmp_path / "A"))
        assert ctrl.widget.output_file_path.text() == str(tmp_path / "A")

    def test_a_re_adopt_still_seeds_the_migration_one_shot(self, tmp_path):
        a, b = str(tmp_path / "A"), str(tmp_path / "B")
        ctrl = _controller()
        ctrl.adopt_workspace(a)
        ctrl.adopt_workspace(b, previous=a)
        assert _key(b) in ctrl._step1_migration_offered


class TestUnloadResetsIdentity:
    """The real `_unload_source_inner` - not a spy - clears the identity."""

    def test_loaded_source_id_and_layers_are_reset(self):
        ctrl = _controller(with_spies=False)

        class _Layer:
            def __init__(self, name):
                self.name = name
                self.visible = True

        class _Layers(dict):
            def remove(self, layer):
                self.pop(layer.name)

        class _Viewer:
            def __init__(self):
                self.layers = _Layers()

        viewer = _Viewer()
        viewer.layers["ch0"] = _Layer("ch0")
        ctrl.viewer = viewer
        ctrl.box_tool = None
        ctrl._loaded_source_layers = ["ch0"]
        ctrl._loaded_source_file = "D:/old/stack.tif"
        ctrl._loaded_source_id = "src_old"
        for name in ("_clear_patch_locations", "_teardown_preview",
                     "_set_raw_missing"):
            setattr(ctrl, name, lambda *a, **k: None)
        ctrl._unload_source_inner()
        assert ctrl._loaded_source_id is None
        assert ctrl._loaded_source_file is None
        assert list(ctrl._loaded_source_layers) == []
        assert "ch0" not in viewer.layers


class TestThroughTheBrowseHandler:
    """Driven through the REAL `browse_output_file`, because that is where the
    ask-before-write order actually lives.

    A mutation that restored the old "setText first, then ask" order survived
    every controller-level test, since they all call `on_workspace_browsed`
    directly and never exercise the UI method that used to write the box.
    """

    @staticmethod
    def _browse(controller, chosen):
        from napari_epicycle.data_preparation import ui as step1_ui

        class _Widget:
            pass

        w = _Widget()
        w.controller = controller
        w.output_file_path = controller.widget.output_file_path
        real = step1_ui.QFileDialog.getExistingDirectory
        step1_ui.QFileDialog.getExistingDirectory = staticmethod(
            lambda *a, **k: chosen)
        try:
            step1_ui.DataPreparationWidget.browse_output_file(w)
        finally:
            step1_ui.QFileDialog.getExistingDirectory = real

    @pytest.mark.parametrize("attr,value", JOB_FLAGS)
    def test_a_refused_browse_leaves_the_box_untouched(self, attr, value,
                                                       tmp_path):
        ctrl = _controller()
        ctrl.widget.output_file_path.setText(str(tmp_path / "A"))
        setattr(ctrl, attr, value)
        self._browse(ctrl, str(tmp_path / "B"))
        assert ctrl.widget.output_file_path.text() == str(tmp_path / "A")
        assert ctrl.calls == []

    def test_an_allowed_browse_moves_the_box(self, tmp_path):
        ctrl = _controller()
        ctrl.widget.output_file_path.setText(str(tmp_path / "A"))
        self._browse(ctrl, str(tmp_path / "B"))
        assert ctrl.widget.output_file_path.text() == str(tmp_path / "B")
        assert ctrl.calls == TEARDOWN_ORDER

    def test_cancelling_the_dialog_changes_nothing(self, tmp_path):
        ctrl = _controller()
        ctrl.widget.output_file_path.setText(str(tmp_path / "A"))
        self._browse(ctrl, "")
        assert ctrl.widget.output_file_path.text() == str(tmp_path / "A")
        assert ctrl.calls == []

    def test_the_box_moves_before_the_shell_is_told(self, tmp_path):
        """The false-divergence fix, through the real entry point."""
        seen = []

        class _Shell:
            def __init__(self, line):
                self._line = line

            def set_workspace(self, path, source_step=""):
                seen.append(self._line.text())

        ctrl = _controller()
        ctrl._workflow_shell = _Shell(ctrl.widget.output_file_path)
        ctrl.widget.output_file_path.setText(str(tmp_path / "A"))
        self._browse(ctrl, str(tmp_path / "B"))
        assert seen == [str(tmp_path / "B")]
