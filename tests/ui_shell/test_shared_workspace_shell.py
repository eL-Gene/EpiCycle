"""Shell-level tests for the shared workspace — Brick B1 (symmetric rule).


The rule under test: **the most recent Browse in ANY step sets the shared
workspace.**  Each step remembers how it got its path — `browsed` (the user
picked it there) or `adopted` (inherited) — and only a `browsed` step whose
path differs from the shared one is amber.  Adopted steps re-adopt.

Constructing the whole `EpiCycleWorkflowShell` boots napari and all five step
widgets, which is too heavy for most unit tests, so these drive the REAL shell
methods on a real instance built with `__new__` plus the attributes those
methods touch, and REAL widgets for the banner and the sidebar.  `_select_step`
itself IS exercised for real in `TestRealSelectStep` below, on a shell whose Qt
scaffolding is built and whose step widgets are stubs.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

qtpy = pytest.importorskip("qtpy")
from qtpy.QtCore import Qt
from qtpy.QtGui import QFontMetrics
from qtpy.QtWidgets import (
    QApplication, QLabel, QLineEdit, QListWidget, QStackedWidget, QVBoxLayout,
    QWidget,
)

from napari_epicycle.ui_shell import _workspace_facts as wf


@pytest.fixture(scope="module")
def app():
    existing = QApplication.instance()
    if existing is not None:
        yield existing
        return
    yield QApplication([])


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """No test may touch the developer's real ~/.epicycle."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(wf.Path, "home", staticmethod(lambda: home))
    wf.clear_cache()
    yield
    wf.clear_cache()


# ---------------------------------------------------------------------------
# a minimal-but-real shell
# ---------------------------------------------------------------------------

class _FakeStepWidget(QWidget):
    """A step widget with just the line-edit the shell reads."""

    def __init__(self, attr: str, controller=None):
        super().__init__()
        setattr(self, attr, QLineEdit(self))
        self.controller = controller


class _RecordingController:
    """A controller on the real `adopt_workspace(path, previous=…)` contract."""

    def __init__(self, line: QLineEdit):
        self._line = line
        self.adopted: list = []
        self.entered = 0
        self.order: list = []

    def adopt_workspace(self, path, previous=""):
        current = self._line.text().strip()
        if current and not (previous and os.path.normcase(current)
                            == os.path.normcase(previous)):
            return                     # a folder the user chose here
        self.adopted.append((path, previous))
        self.order.append("adopt")
        self._line.setText(path)

    def on_step_enter(self):
        self.entered += 1
        self.order.append("enter")


class _NoAdoptController:
    def __init__(self):
        self.entered = 0

    def on_step_enter(self):
        self.entered += 1


def _make_shell(app, with_controllers=()):
    """A real shell object with only the pieces B1's methods touch."""
    from napari_epicycle.ui_shell.workflow_shell import (
        EpiCycleWorkflowShell, _BannerLabel, _StatusCard,
    )
    shell = EpiCycleWorkflowShell.__new__(EpiCycleWorkflowShell)
    shell._workspace_path = ""
    shell._workspace_source_step = ""
    shell._suggested_workspace = ""
    shell._step_path_origin = {}
    shell._diverged_steps = {}
    shell._diverged_logged = set()
    shell._shell_ready = True
    shell._active_step_id = "project_setup"
    shell._step_specs = EpiCycleWorkflowShell._build_step_specs(shell)
    shell._step_index_by_id = {
        s.step_id: i for i, s in enumerate(shell._step_specs)}
    shell._step_widgets = {
        "project_setup": _FakeStepWidget("output_file_path"),
        "roi_hitl_2d": _FakeStepWidget("set_workdir_path"),
        "reconstruct_curate_3d": _FakeStepWidget("set_workdir_path"),
        "apply_3d": _FakeStepWidget("workdir_line"),
        "whole_tissue": _FakeStepWidget("workdir_line"),
    }
    shell._step_widgets["apply_3d"].stacks_list = QListWidget(
        shell._step_widgets["apply_3d"])
    for step_id in with_controllers:
        attr = EpiCycleWorkflowShell._STEP_WORKSPACE_FIELDS[step_id]
        widget = shell._step_widgets[step_id]
        widget.controller = _RecordingController(getattr(widget, attr))
    shell.guidance_label = _BannerLabel()
    shell.advisory_label = _BannerLabel()
    shell.step_title = QLabel("Project Setup & ROI Extraction")
    shell.state_panel = _StatusCard()
    return shell


def _browse(shell, step_id, path):
    """What a step's Browse handler does: set its box, then publish."""
    attr = shell._STEP_WORKSPACE_FIELDS[step_id]
    getattr(shell._step_widgets[step_id], attr).setText(str(path))
    shell.set_workspace(str(path), source_step=step_id)


def _sheet(label):
    return label.styleSheet()


# ---------------------------------------------------------------------------
# the symmetric rule
# ---------------------------------------------------------------------------

class TestSymmetricBrowse:
    def test_any_step_can_set_the_shared_workspace(self, app, tmp_path):
        shell = _make_shell(app)
        ws = tmp_path / "Proj"
        ws.mkdir()
        _browse(shell, "reconstruct_curate_3d", ws)
        assert shell._read_workspace_path() == os.path.abspath(str(ws))
        assert shell._workspace_source_step == "reconstruct_curate_3d"
        assert wf.load_last_workspace() == os.path.abspath(str(ws))

    def test_the_most_recent_browse_wins_whichever_step(self, app, tmp_path):
        shell = _make_shell(app)
        a, b, c = tmp_path / "A", tmp_path / "B", tmp_path / "C"
        for p in (a, b, c):
            p.mkdir()
        _browse(shell, "project_setup", a)
        _browse(shell, "apply_3d", b)
        assert shell._read_workspace_path() == os.path.abspath(str(b))
        _browse(shell, "reconstruct_curate_3d", c)
        assert shell._read_workspace_path() == os.path.abspath(str(c))

    def test_the_browsing_step_is_never_diverged(self, app, tmp_path):
        shell = _make_shell(app)
        a, b = tmp_path / "A", tmp_path / "B"
        a.mkdir(), b.mkdir()
        _browse(shell, "project_setup", a)
        _browse(shell, "apply_3d", b)
        assert "apply_3d" not in shell._diverged_steps
        shell._active_step_id = "apply_3d"
        shell._refresh_step_guidance()
        assert shell.guidance_label._color_key == "green"

    def test_a_step_browsed_earlier_elsewhere_becomes_diverged(self, app,
                                                               tmp_path):
        shell = _make_shell(app)
        a, b = tmp_path / "A", tmp_path / "B"
        a.mkdir(), b.mkdir()
        _browse(shell, "reconstruct_curate_3d", a)
        _browse(shell, "project_setup", b)
        assert shell._diverged_steps == {
            "reconstruct_curate_3d": os.path.abspath(str(a))}
        shell._active_step_id = "reconstruct_curate_3d"
        shell._refresh_step_guidance()
        assert shell.guidance_label._color_key == "amber"

    def test_divergence_logs_once_per_change(self, app, tmp_path, capsys):
        shell = _make_shell(app)
        a, b, c = tmp_path / "A", tmp_path / "B", tmp_path / "C"
        for p in (a, b, c):
            p.mkdir()
        _browse(shell, "apply_3d", a)
        _browse(shell, "project_setup", b)
        first = [ln for ln in capsys.readouterr().out.splitlines()
                 if "diverged" in ln]
        assert len(first) == 1 and "apply_3d" in first[0]
        shell.refresh_global_state()          # a poll must not re-log
        assert "diverged" not in capsys.readouterr().out
        _browse(shell, "project_setup", c)    # a NEW divergence pairing
        second = [ln for ln in capsys.readouterr().out.splitlines()
                  if "diverged" in ln]
        assert len(second) == 1

    def test_divergence_ends_when_the_shared_comes_back(self, app, tmp_path):
        shell = _make_shell(app)
        a, b = tmp_path / "A", tmp_path / "B"
        a.mkdir(), b.mkdir()
        _browse(shell, "apply_3d", a)
        _browse(shell, "project_setup", b)
        assert "apply_3d" in shell._diverged_steps
        _browse(shell, "project_setup", a)    # back to where Step 4 is
        assert shell._diverged_steps == {}

    def test_divergence_ends_when_the_step_browses_to_the_shared_path(
            self, app, tmp_path):
        shell = _make_shell(app)
        a, b = tmp_path / "A", tmp_path / "B"
        a.mkdir(), b.mkdir()
        _browse(shell, "apply_3d", a)
        _browse(shell, "project_setup", b)
        _browse(shell, "apply_3d", b)         # unchanged shared path
        assert shell._diverged_steps == {}
        assert shell._read_workspace_path() == os.path.abspath(str(b))

    def test_unchanged_path_does_not_re_announce(self, app, tmp_path, capsys):
        shell = _make_shell(app)
        ws = tmp_path / "Proj"
        ws.mkdir()
        _browse(shell, "project_setup", ws)
        capsys.readouterr()
        _browse(shell, "project_setup", ws)
        assert "shared workspace set" not in capsys.readouterr().out

    def test_case_and_slash_variants_count_as_unchanged(self, app, tmp_path,
                                                        capsys):
        shell = _make_shell(app)
        ws = tmp_path / "Proj"
        ws.mkdir()
        _browse(shell, "project_setup", ws)
        capsys.readouterr()
        shell.set_workspace(str(ws).replace("\\", "/"), "project_setup")
        assert "shared workspace set" not in capsys.readouterr().out

    def test_empty_path_is_ignored(self, app):
        shell = _make_shell(app)
        shell.set_workspace("", "project_setup")
        assert shell._read_workspace_path() == ""


# ---------------------------------------------------------------------------
# re-adopt
# ---------------------------------------------------------------------------

class TestReAdopt:
    def test_the_visible_step_re_adopts_immediately(self, app, tmp_path):
        shell = _make_shell(app, with_controllers=("apply_3d",))
        a, b = tmp_path / "A", tmp_path / "B"
        a.mkdir(), b.mkdir()
        _browse(shell, "project_setup", a)
        shell._active_step_id = "apply_3d"
        shell._adopt_into_step("apply_3d")           # first entry
        assert shell._step_path_origin["apply_3d"] == "adopted"
        _browse(shell, "reconstruct_curate_3d", b)   # shared moves
        ctrl = shell._step_widgets["apply_3d"].controller
        assert ctrl.adopted[-1] == (os.path.abspath(str(b)),
                                    os.path.abspath(str(a)))
        assert shell._read_step_workspace_path("apply_3d") == \
            os.path.abspath(str(b))

    def test_an_off_screen_step_re_adopts_on_entry(self, app, tmp_path):
        shell = _make_shell(app, with_controllers=("apply_3d",))
        a, b = tmp_path / "A", tmp_path / "B"
        a.mkdir(), b.mkdir()
        _browse(shell, "project_setup", a)
        shell._adopt_into_step("apply_3d")
        _browse(shell, "project_setup", b)           # Step 4 not on screen
        ctrl = shell._step_widgets["apply_3d"].controller
        assert shell._adopt_into_step("apply_3d") is True
        assert ctrl.adopted[-1][0] == os.path.abspath(str(b))

    def test_a_browsed_step_is_never_re_adopted(self, app, tmp_path):
        shell = _make_shell(app, with_controllers=("apply_3d",))
        a, b = tmp_path / "A", tmp_path / "B"
        a.mkdir(), b.mkdir()
        _browse(shell, "apply_3d", a)
        _browse(shell, "project_setup", b)
        assert shell._adopt_into_step("apply_3d") is False
        assert shell._read_step_workspace_path("apply_3d") == \
            os.path.abspath(str(a))

    def test_a_step_already_on_the_shared_path_is_left_alone(self, app,
                                                             tmp_path):
        shell = _make_shell(app, with_controllers=("apply_3d",))
        ws = tmp_path / "Proj"
        ws.mkdir()
        _browse(shell, "project_setup", ws)
        shell._adopt_into_step("apply_3d")
        ctrl = shell._step_widgets["apply_3d"].controller
        n = len(ctrl.adopted)
        assert shell._adopt_into_step("apply_3d") is False
        assert len(ctrl.adopted) == n

    def test_no_adopt_without_a_shared_workspace(self, app):
        shell = _make_shell(app, with_controllers=("apply_3d",))
        assert shell._adopt_into_step("apply_3d") is False

    def test_no_adopt_before_the_shell_is_ready(self, app, tmp_path):
        shell = _make_shell(app, with_controllers=("apply_3d",))
        ws = tmp_path / "Proj"
        ws.mkdir()
        _browse(shell, "project_setup", ws)
        shell._shell_ready = False
        assert shell._adopt_into_step("apply_3d") is False

    def test_a_controller_without_adopt_is_skipped(self, app, tmp_path):
        shell = _make_shell(app)
        shell._step_widgets["apply_3d"].controller = _NoAdoptController()
        ws = tmp_path / "Proj"
        ws.mkdir()
        _browse(shell, "project_setup", ws)
        assert shell._adopt_into_step("apply_3d") is False

    def test_a_raising_adopt_is_contained(self, app, tmp_path, capsys):
        class _Boom:
            def adopt_workspace(self, path, previous=""):
                raise RuntimeError("boom")

        shell = _make_shell(app)
        shell._step_widgets["apply_3d"].controller = _Boom()
        ws = tmp_path / "Proj"
        ws.mkdir()
        _browse(shell, "project_setup", ws)
        assert shell._adopt_into_step("apply_3d") is False
        assert "adopt failed for apply_3d" in capsys.readouterr().out

    def test_an_old_single_argument_controller_still_works(self, app,
                                                           tmp_path):
        calls = []

        class _Old:
            def adopt_workspace(self, path):
                calls.append(path)

        shell = _make_shell(app)
        shell._step_widgets["apply_3d"].controller = _Old()
        ws = tmp_path / "Proj"
        ws.mkdir()
        _browse(shell, "project_setup", ws)
        shell._adopt_into_step("apply_3d")
        assert calls == [os.path.abspath(str(ws))]


# ---------------------------------------------------------------------------
# the real _select_step
# ---------------------------------------------------------------------------

class TestRealSelectStep:
    """Exercises `_select_step` itself, not a transcription of it."""

    def _shell(self, app):
        from napari_epicycle.ui_shell.workflow_shell import _HistoryDrawer
        shell = _make_shell(app, with_controllers=("apply_3d",))
        shell._self_describing_steps = {"project_setup"}
        shell._step_nav_locked = False
        shell._subviewer_steps = set()
        shell._nav_items = {}
        shell._nav_sections = {}
        shell._step_section_color = {
            s.step_id: "#888888" for s in shell._step_specs}
        shell.step_subtitle = QLabel("")
        shell._header_accent = QWidget()
        shell.step_stack = QStackedWidget()
        for _ in shell._step_specs:
            shell.step_stack.addWidget(QWidget())
        shell.history_drawer = _HistoryDrawer()
        shell._viewer_canvas_snapshot = None
        # neutralise the bits that need a live napari viewer
        shell._sync_viewer_canvas_mode = lambda **kw: None
        shell._apply_nav_highlight = lambda *a, **k: None
        shell._update_step_intent = lambda *a, **k: None
        return shell

    def test_select_step_adopts_then_enters_then_paints(self, app, tmp_path):
        shell = self._shell(app)
        ws = tmp_path / "Proj"
        ws.mkdir()
        _browse(shell, "project_setup", ws)
        shell._select_step("apply_3d")
        ctrl = shell._step_widgets["apply_3d"].controller
        assert ctrl.order == ["adopt", "enter"]
        assert ctrl.adopted[0][0] == os.path.abspath(str(ws))
        assert shell._active_step_id == "apply_3d"
        assert shell.guidance_label._color_key == "green"
        assert shell._step_path_origin["apply_3d"] == "adopted"

    def test_select_step_does_not_adopt_before_the_shell_is_ready(
            self, app, tmp_path):
        shell = self._shell(app)
        ws = tmp_path / "Proj"
        ws.mkdir()
        _browse(shell, "project_setup", ws)
        shell._shell_ready = False
        shell._select_step("apply_3d")
        ctrl = shell._step_widgets["apply_3d"].controller
        assert ctrl.adopted == []
        assert ctrl.entered == 1          # the step still opens

    def test_select_step_re_adopts_a_stale_inherited_path(self, app, tmp_path):
        shell = self._shell(app)
        a, b = tmp_path / "A", tmp_path / "B"
        a.mkdir(), b.mkdir()
        _browse(shell, "project_setup", a)
        shell._select_step("apply_3d")            # adopts A
        shell._select_step("project_setup")
        _browse(shell, "project_setup", b)        # shared moves to B
        shell._select_step("apply_3d")
        ctrl = shell._step_widgets["apply_3d"].controller
        assert ctrl.adopted[-1] == (os.path.abspath(str(b)),
                                    os.path.abspath(str(a)))


# ---------------------------------------------------------------------------
# banner
# ---------------------------------------------------------------------------

class TestBanner:
    def test_state_1_red_on_step_1(self, app):
        shell = _make_shell(app)
        shell._refresh_step_guidance("project_setup")
        assert shell.guidance_label._text == \
            "Choose a workspace folder to begin."
        assert shell.guidance_label._color_key == "red"
        assert "#e07070" in _sheet(shell.guidance_label)
        assert shell.advisory_label._text == ""

    def test_state_1_red_on_other_steps(self, app):
        shell = _make_shell(app)
        shell._active_step_id = "apply_3d"
        shell._refresh_step_guidance()
        assert "Set a workspace in Project Setup" in shell.guidance_label._text
        assert "#e07070" in _sheet(shell.guidance_label)

    def test_the_remembered_suggestion_stays_red(self, app, tmp_path):
        """F1: a pre-filled box is not a shared workspace."""
        shell = _make_shell(app)
        ws = tmp_path / "Remembered"
        ws.mkdir()
        shell._suggested_workspace = str(ws)
        shell._prefill_step1_from_suggestion()
        shell._refresh_step_guidance("project_setup")
        assert shell.guidance_label._color_key == "red"
        assert shell._read_workspace_path() == ""

    def test_a_step_path_without_a_shared_workspace_is_green_unqualified(
            self, app, tmp_path):
        shell = _make_shell(app)
        ws = tmp_path / "Own"
        ws.mkdir()
        shell._step_widgets["apply_3d"].workdir_line.setText(str(ws))
        shell._active_step_id = "apply_3d"
        shell._refresh_step_guidance()
        assert shell.guidance_label._color_key == "green"
        assert "shared across steps" not in shell.guidance_label._text

    def test_state_2_green(self, app, tmp_path):
        shell = _make_shell(app)
        ws = tmp_path / "Proj"
        ws.mkdir()
        _browse(shell, "project_setup", ws)
        shell._refresh_step_guidance("project_setup")
        assert shell.guidance_label._color_key == "green"
        from napari_epicycle.ui_shell import _workspace_facts as facts
        assert facts.SHARED_ACCENT in _sheet(shell.guidance_label)
        assert "(shared across steps)" in shell.guidance_label._text
        assert shell.guidance_label.toolTip() == os.path.abspath(str(ws))

    def test_state_3_amber_with_both_paths_in_the_tooltip(self, app, tmp_path):
        shell = _make_shell(app)
        a, b = tmp_path / "A", tmp_path / "B"
        a.mkdir(), b.mkdir()
        _browse(shell, "apply_3d", a)
        _browse(shell, "project_setup", b)
        shell._active_step_id = "apply_3d"
        shell._refresh_step_guidance()
        assert shell.guidance_label._color_key == "amber"
        assert "#d4a843" in _sheet(shell.guidance_label)
        assert "(different from the shared workspace)" in \
            shell.guidance_label._text
        tip = shell.guidance_label.toolTip()
        assert os.path.abspath(str(a)) in tip
        assert os.path.abspath(str(b)) in tip

    def test_state_3_stacks_the_advisory_below(self, app, tmp_path):
        shell = _make_shell(app)
        a, b = tmp_path / "A", tmp_path / "B"
        a.mkdir(), b.mkdir()
        _browse(shell, "reconstruct_curate_3d", a)
        _browse(shell, "project_setup", b)
        shell._active_step_id = "reconstruct_curate_3d"
        shell._refresh_step_guidance()
        assert shell.guidance_label._color_key == "amber"
        assert shell.advisory_label._color_key == "amber"
        assert shell.advisory_label._text.startswith("No trained 2D model")

    def test_missing_folder_is_red(self, app, tmp_path):
        shell = _make_shell(app)
        ws = tmp_path / "Proj"
        ws.mkdir()
        _browse(shell, "project_setup", ws)
        ws.rmdir()
        shell._refresh_step_guidance("project_setup")
        if sys.platform.startswith("win"):
            assert shell.guidance_label._color_key == "red"
            assert "no longer where it was" in shell.guidance_label._text

    def test_two_strips_workspace_first_then_advisory(self, app, tmp_path):
        shell = _make_shell(app)
        ws = tmp_path / "Proj"
        ws.mkdir()
        _browse(shell, "project_setup", ws)
        shell._active_step_id = "reconstruct_curate_3d"
        shell._refresh_step_guidance()
        assert shell.guidance_label._color_key == "green"
        assert shell.advisory_label._color_key == "amber"
        assert shell.advisory_label._text.startswith("No trained 2D model")

    def test_advisory_is_gone_when_rounds_exist_on_disk(self, app, tmp_path):
        shell = _make_shell(app)
        ws = tmp_path / "Proj"
        rd = ws / "artifacts" / "step2" / "runs" / "cells" / "lateral" / "round_00"
        (rd / "model").mkdir(parents=True)
        _browse(shell, "project_setup", ws)
        shell._active_step_id = "reconstruct_curate_3d"
        shell._refresh_step_guidance()
        assert shell.guidance_label._color_key == "green"
        assert shell.advisory_label._text == ""
        assert shell.advisory_label.isVisible() is False

    def test_state_1_suppresses_the_advisory(self, app):
        shell = _make_shell(app)
        shell._active_step_id = "reconstruct_curate_3d"
        shell._refresh_step_guidance()
        assert shell.guidance_label._color_key == "red"
        assert shell.advisory_label._text == ""

    def test_banner_is_selectable_and_elides(self, app, tmp_path):
        shell = _make_shell(app)
        ws = tmp_path / ("Very" + "Long" * 30)
        ws.mkdir()
        _browse(shell, "project_setup", ws)
        shell._refresh_step_guidance("project_setup")
        label = shell.guidance_label
        assert label.textInteractionFlags() & Qt.TextSelectableByMouse
        label.resize(420, 24)
        assert "\u2026" in label.text()          # middle-elided
        assert label.toolTip() == os.path.abspath(str(ws))

    def test_a_long_path_does_not_widen_the_panel(self, app):
        """F2: a non-wrapping label used to force the whole dock wider."""
        from napari_epicycle.ui_shell.workflow_shell import _BannerLabel
        parent = QWidget()
        layout = QVBoxLayout(parent)
        label = _BannerLabel()
        layout.addWidget(label)
        before = parent.minimumSizeHint().width()
        long_path = "C:/very/long/path" * 20
        label.set_strip("Workspace: " + long_path, "green", long_path)
        assert label.minimumSizeHint().width() <= 60
        assert parent.minimumSizeHint().width() <= before + 5

    @pytest.mark.parametrize("width", [520, 400, 300, 220, 140])
    def test_amber_line_fits_the_width_budget(self, app, tmp_path, width):
        """The diverged line must not run off the panel, at any width.

        The path is painted in the monospace face, so it is measured in that
        face too — measuring it in the UI font under-counts and the tail is
        clipped (seen in the live screenshots).
        """
        import re
        from qtpy.QtGui import QFont
        shell = _make_shell(app)
        a = tmp_path / ("Diverged" + "Long" * 25)
        b = tmp_path / ("Shared" + "Long" * 25)
        a.mkdir(), b.mkdir()
        _browse(shell, "apply_3d", a)
        _browse(shell, "project_setup", b)
        shell._active_step_id = "apply_3d"
        shell._refresh_step_guidance()
        label = shell.guidance_label
        label.resize(width, 24)
        label._render()
        html_text = label.text()
        mono_part = re.search(r"monospace\">([^<]*)</span>", html_text)
        assert mono_part, html_text
        plain = re.sub(r"<[^>]+>", "", html_text)
        rest = plain.replace(mono_part.group(1), "")
        ui = QFontMetrics(label.font())
        mono_font = QFont(label.font())
        mono_font.setFamilies(
            ["Consolas", "Cascadia Mono", "Courier New", "monospace"])
        total = (ui.horizontalAdvance(rest)
                 + QFontMetrics(mono_font).horizontalAdvance(
                     mono_part.group(1)))
        assert total <= label.width(), (total, label.width(), plain)

    def test_repaint_with_identical_content_is_a_no_op(self, app, tmp_path):
        """F3: the 3 s poll must not wipe a text selection."""
        shell = _make_shell(app)
        ws = tmp_path / "Proj"
        ws.mkdir()
        _browse(shell, "project_setup", ws)
        label = shell.guidance_label
        calls = []
        original = label.setText

        def _spy(text):
            calls.append(text)
            original(text)

        label.setText = _spy
        try:
            shell._refresh_step_guidance("project_setup")
            shell.refresh_global_state()
        finally:
            label.setText = original
        assert calls == []


# ---------------------------------------------------------------------------
# sidebar card
# ---------------------------------------------------------------------------

class TestSidebarCard:
    def test_no_workspace_row_is_red(self, app):
        shell = _make_shell(app)
        shell.refresh_global_state()
        card = shell.state_panel
        assert card._labels["workspace"].text() == "No workspace"
        assert "#e07070" in card._labels["workspace"].styleSheet()
        assert card._labels["status"].text() == "Step 1  \u00b7  nothing loaded"

    def test_no_workspace_line_names_the_current_step(self, app):
        shell = _make_shell(app)
        shell._active_step_id = "apply_3d"
        shell.refresh_global_state()
        assert shell.state_panel._labels["status"].text() == \
            "Step 4  \u00b7  nothing loaded"

    def test_nothing_is_read_from_disk_before_the_shell_is_ready(self, app):
        shell = _make_shell(app)
        shell._shell_ready = False
        assert shell._read_active_step_status("project_setup") == ""

    def test_models_row_is_gone(self, app):
        shell = _make_shell(app)
        assert "models" not in shell.state_panel._labels

    def test_folder_name_and_full_path_tooltip(self, app, tmp_path):
        shell = _make_shell(app)
        ws = tmp_path / "WoundReconstruciton-Test"
        ws.mkdir()
        _browse(shell, "project_setup", ws)
        shell.refresh_global_state()
        card = shell.state_panel
        assert card._labels["workspace"].text() == "WoundReconstruciton-Test"
        assert card._labels["workspace"].toolTip() == os.path.abspath(str(ws))
        assert "bold" in card._labels["workspace"].styleSheet()

    def test_tooltip_lists_divergences(self, app, tmp_path):
        shell = _make_shell(app)
        a, b = tmp_path / "A", tmp_path / "B"
        a.mkdir(), b.mkdir()
        _browse(shell, "apply_3d", a)
        _browse(shell, "project_setup", b)
        shell.refresh_global_state()
        tip = shell.state_panel._labels["workspace"].toolTip()
        assert "Browsed elsewhere" in tip and os.path.abspath(str(a)) in tip
        assert shell.state_panel._labels["workspace"].text() == "B"

    def test_status_line_step1_patch_count(self, app, tmp_path):
        shell = _make_shell(app)
        ws = tmp_path / "Proj"
        patches = ws / "artifacts" / "step1" / "patches"
        patches.mkdir(parents=True)
        (patches / "p1.tif").write_bytes(b"x")
        (patches / "p2.tif").write_bytes(b"x")
        _browse(shell, "project_setup", ws)
        shell.refresh_global_state()
        assert shell.state_panel._labels["status"].text() == \
            "Step 1  \u00b7  2 patches"

    def test_status_line_step2_rounds_from_disk(self, app, tmp_path):
        shell = _make_shell(app)
        ws = tmp_path / "Proj"
        for bucket, n in (("lateral", 3), ("axial", 1)):
            for i in range(n):
                (ws / "artifacts" / "step2" / "runs" / "cells" / bucket
                 / f"round_{i:02d}" / "model").mkdir(parents=True)
        _browse(shell, "project_setup", ws)

        class _C:
            active_run_id = "cells"
        shell._step_widgets["roi_hitl_2d"].controller = _C()
        shell._active_step_id = "roi_hitl_2d"
        shell.refresh_global_state()
        assert shell.state_panel._labels["status"].text() == \
            "Step 2  \u00b7  run cells  \u00b7  lateral 3  \u00b7  axial 1"

    def test_status_line_step3_without_a_patch(self, app, tmp_path):
        shell = _make_shell(app)
        ws = tmp_path / "Proj"
        ws.mkdir()
        _browse(shell, "project_setup", ws)
        shell._active_step_id = "reconstruct_curate_3d"
        shell.refresh_global_state()
        assert shell.state_panel._labels["status"].text() == \
            "Step 3  \u00b7  no patch selected"

    def test_status_line_step3_with_a_patch(self, app, tmp_path):
        shell = _make_shell(app)
        ws = tmp_path / "Proj"
        ws.mkdir()
        _browse(shell, "project_setup", ws)

        class _SM:
            def get_patch_status(self, name):
                return "reconstructed"

        class _C:
            _selected_curation_patch = "patch_6"
            state_manager = _SM()
        shell._step_widgets["reconstruct_curate_3d"].controller = _C()
        shell._active_step_id = "reconstruct_curate_3d"
        shell.refresh_global_state()
        assert shell.state_panel._labels["status"].text() == \
            "Step 3  \u00b7  patch_6  \u00b7  reconstructed"

    def test_status_line_step4_stack_count(self, app, tmp_path):
        shell = _make_shell(app)
        ws = tmp_path / "Proj"
        ws.mkdir()
        _browse(shell, "project_setup", ws)
        listw = shell._step_widgets["apply_3d"].stacks_list
        listw.addItems(["a.tif", "b.tif"])
        shell._active_step_id = "apply_3d"
        shell.refresh_global_state()
        assert shell.state_panel._labels["status"].text() == \
            "Step 4  \u00b7  2 stacks"


# ---------------------------------------------------------------------------
# start-up pre-fill
# ---------------------------------------------------------------------------

class TestPrefill:
    def test_prefill_fills_step1_without_firing_signals(self, app, tmp_path):
        shell = _make_shell(app)
        ws = tmp_path / "Proj"
        ws.mkdir()
        shell._suggested_workspace = str(ws)
        line = shell._step_widgets["project_setup"].output_file_path
        seen = []
        line.textChanged.connect(lambda t: seen.append(t))
        shell._prefill_step1_from_suggestion()
        assert line.text() == str(ws)
        assert seen == []          # no textChanged side effects at start-up

    def test_prefill_does_not_claim_a_shared_workspace(self, app, tmp_path):
        shell = _make_shell(app)
        shell._suggested_workspace = str(tmp_path / "Remembered")
        shell._prefill_step1_from_suggestion()
        assert shell._read_workspace_path() == ""
        assert shell._step_path_origin == {}

    def test_prefill_does_not_overwrite_a_path_already_there(self, app,
                                                            tmp_path):
        shell = _make_shell(app)
        shell._suggested_workspace = str(tmp_path / "Suggested")
        line = shell._step_widgets["project_setup"].output_file_path
        line.setText("D:/chosen")
        shell._prefill_step1_from_suggestion()
        assert line.text() == "D:/chosen"

    def test_prefill_is_a_no_op_without_a_suggestion(self, app):
        shell = _make_shell(app)
        shell._prefill_step1_from_suggestion()
        assert shell._step_widgets["project_setup"].output_file_path.text() == ""


# ---------------------------------------------------------------------------
# _read_step_workspace_path
# ---------------------------------------------------------------------------

def test_read_step_workspace_path(app):
    shell = _make_shell(app)
    shell._step_widgets["reconstruct_curate_3d"].set_workdir_path.setText(
        "  D:/ws  ")
    assert shell._read_step_workspace_path("reconstruct_curate_3d") == "D:/ws"
    assert shell._read_step_workspace_path("nope") == ""


# ---------------------------------------------------------------------------
# B2 — the advisory describes the workspace THIS STEP is on
# ---------------------------------------------------------------------------

class TestB2Advisory:
    """`_build_step_guidance` must read the step's OWN folder.

    Found by the B2 live run: a Step 3 sitting on an old-layout folder was
    told about the SHARED workspace's state file instead of its own, so the
    "press Browse" line never appeared where the user needed it.
    """

    def _ws(self, root, name, step3_schema=None, with_run=False):
        ws = root / name
        ws.mkdir()
        if step3_schema is not None:
            (ws / "step3_state.json").write_text(
                json.dumps({"schema_version": step3_schema, "patches": {}}),
                encoding="utf-8")
        if with_run:
            (ws / "artifacts" / "step2" / "runs" / "cells").mkdir(parents=True)
        return ws

    def test_step3_advisory_follows_the_diverged_step(self, app, tmp_path):
        shared = self._ws(tmp_path, "shared", step3_schema=3, with_run=True)
        mine = self._ws(tmp_path, "mine", step3_schema=2, with_run=True)
        shell = _make_shell(app)
        _browse(shell, "reconstruct_curate_3d", shared)
        shell._step_widgets["reconstruct_curate_3d"].set_workdir_path.setText(
            str(mine))
        advisory = shell._build_step_guidance("reconstruct_curate_3d")
        assert "old folder layout" in advisory, advisory

    def test_step3_on_the_shared_v3_workspace_gets_no_layout_line(
            self, app, tmp_path):
        shared = self._ws(tmp_path, "shared", step3_schema=3, with_run=True)
        shell = _make_shell(app)
        _browse(shell, "reconstruct_curate_3d", shared)
        assert "old folder layout" not in shell._build_step_guidance(
            "reconstruct_curate_3d")

    def test_step2_no_run_line_follows_the_diverged_step(self, app, tmp_path):
        shared = self._ws(tmp_path, "shared", with_run=True)
        empty = self._ws(tmp_path, "empty")
        shell = _make_shell(app)
        _browse(shell, "roi_hitl_2d", shared)
        assert "No run yet" not in shell._build_step_guidance("roi_hitl_2d")
        shell._step_widgets["roi_hitl_2d"].set_workdir_path.setText(str(empty))
        assert "No run yet" in shell._build_step_guidance("roi_hitl_2d")

    def test_step2_legacy_layout_beats_the_no_run_line(self, app, tmp_path):
        legacy = self._ws(tmp_path, "legacy")
        (legacy / "training_data").mkdir()
        shell = _make_shell(app)
        _browse(shell, "roi_hitl_2d", legacy)
        advisory = shell._build_step_guidance("roi_hitl_2d")
        assert "old folder layout" in advisory
        assert "No run yet" not in advisory

    def test_no_advisory_and_no_disk_read_before_anything_is_shared(
            self, app, tmp_path):
        """Invariant I5 / audit N1: a step's pre-filled box must not make the
        shell read a workspace that nobody has opened."""
        shell = _make_shell(app)
        shell._step_widgets["roi_hitl_2d"].set_workdir_path.setText(
            str(tmp_path))
        assert shell._build_step_guidance("roi_hitl_2d") == ""


# ---------------------------------------------------------------------------
# User feedback on the B2 build: bring back the per-row chips
# ---------------------------------------------------------------------------

class TestSidebarRowChips:
    """Each text row sits in its own small rounded bordered box again.

    Pre-B1 that look came for free, because `_CARD_STYLE` was set on the card
    with no selector and Qt cascaded it into every child QLabel.  B1 scoped it
    to `QWidget#epicycleStatusCard` to stop the border landing on the RAM /
    CPU / GPU labels too — and took the chips with it.  They are now drawn
    deliberately, on the two row labels only, addressed by their own object
    names so nothing cascades anywhere.
    """

    _CHIP_BITS = ("border-radius", "border: 1px solid", "background:",
                  "padding")

    def _rows(self, shell):
        card = shell.state_panel
        return card._labels["workspace"], card._labels["status"]

    def test_both_rows_carry_the_chip(self, app):
        shell = _make_shell(app)
        for label in self._rows(shell):
            sheet = label.styleSheet()
            for bit in self._CHIP_BITS:
                assert bit in sheet, f"{bit!r} missing from {sheet!r}"

    def test_each_row_is_addressed_by_its_own_object_name(self, app):
        shell = _make_shell(app)
        ws_row, status_row = self._rows(shell)
        assert ws_row.objectName() == "epicycleWorkspaceRow"
        assert status_row.objectName() == "epicycleStatusRow"
        assert f"QLabel#{ws_row.objectName()}" in ws_row.styleSheet()
        assert f"QLabel#{status_row.objectName()}" in status_row.styleSheet()

    def test_the_chip_survives_the_red_no_workspace_state(self, app):
        shell = _make_shell(app)
        shell.refresh_global_state()
        sheet = shell.state_panel._labels["workspace"].styleSheet()
        assert "#e07070" in sheet
        for bit in self._CHIP_BITS:
            assert bit in sheet, f"the red state dropped {bit!r}"

    def test_the_chip_survives_a_real_workspace(self, app, tmp_path):
        shell = _make_shell(app)
        ws = tmp_path / "Proj"
        ws.mkdir()
        _browse(shell, "project_setup", ws)
        shell.refresh_global_state()
        sheet = shell.state_panel._labels["workspace"].styleSheet()
        assert "bold" in sheet
        for bit in self._CHIP_BITS:
            assert bit in sheet

    def test_the_card_does_not_double_box(self, app):
        """The card paints ONE outer box, via its own id selector — if it
        cascaded again the rows would sit in a box inside a box inside a box,
        which is what B1 removed."""
        shell = _make_shell(app)
        from napari_epicycle.ui_shell.workflow_shell import _StatusCard
        assert "QWidget#epicycleStatusCard {" in _StatusCard._CARD_STYLE
        assert not _StatusCard._CARD_STYLE.strip().startswith("background")

    def test_the_resource_labels_get_no_chip(self, app):
        """RAM / CPU / GPU are captions next to a bar, not status rows —
        boxing them is the cascade bug B1 fixed."""
        from napari_epicycle.ui_shell.workflow_shell import _StatusCard
        assert "border" not in _StatusCard._LABEL_DIM


class TestSharedAccentIsOneConstant:
    """The shared-workspace colour is a single named value.

    The user is weighing Apply-green against the teal already in the palette.
    Whatever they pick, the switch must be one edit — not a hunt through the
    banner, the advisory colour and the sidebar.
    """

    def test_the_banner_palette_reads_the_constant(self, app):
        from napari_epicycle.ui_shell import workflow_shell as ws_mod
        from napari_epicycle.ui_shell import _workspace_facts as facts
        assert ws_mod._BANNER_COLORS["green"] == facts.SHARED_ACCENT
        assert ws_mod._BANNER_COLORS["shared"] == facts.SHARED_ACCENT

    def test_todays_value_is_the_cpu_bar_teal(self):
        """Decided 2026-09-08: the shared line is the sidebar CPU-bar
        teal, not the Apply green (which collided with the Step 4 stripe)."""
        from napari_epicycle.ui_shell import _workspace_facts as facts
        assert facts.SHARED_ACCENT == "#00d4aa"

    def test_switching_the_constant_repaints_the_banner(self, app, monkeypatch,
                                                        tmp_path):
        """The one-line change, exercised: set the constant back to the Apply
        green and the shared strip is painted green, with nothing else edited."""
        from napari_epicycle.ui_shell import _workspace_facts as facts
        teal = "#6abf69"
        monkeypatch.setattr(facts, "SHARED_ACCENT", teal)
        shell = _make_shell(app)
        ws = tmp_path / "Proj"
        ws.mkdir()
        _browse(shell, "project_setup", ws)
        shell._refresh_step_guidance()
        sheet = _sheet(shell.guidance_label)
        assert teal in sheet, sheet
        assert "#00d4aa" not in sheet

    def test_the_other_two_colours_are_untouched_by_the_swap(
            self, app, monkeypatch):
        from napari_epicycle.ui_shell import workflow_shell as ws_mod
        from napari_epicycle.ui_shell import _workspace_facts as facts
        monkeypatch.setattr(facts, "SHARED_ACCENT", "#00d4aa")
        assert ws_mod._BANNER_COLORS["red"] == "#e07070"
        assert ws_mod._BANNER_COLORS["amber"] == "#d4a843"

    def test_the_neutral_advisory_follows_the_same_key(self):
        from napari_epicycle.ui_shell import _workspace_facts as facts
        assert facts.advisory_color(facts.MSG_STEP2_NO_RUN) == \
            facts.SHARED_ACCENT_KEY
        assert facts.advisory_color(facts.MSG_NO_TRAINED_MODEL) == "amber"

    def test_the_named_teal_alternatives_still_exist_in_the_codebase(self):
        """The comment on SHARED_ACCENT points at four real hexes; if one is
        renamed away, this catches the doc rot before the user picks it."""
        import pathlib
        root = pathlib.Path(__file__).resolve().parents[2] / "napari_epicycle"
        haystack = "\n".join(
            p.read_text(encoding="utf-8", errors="ignore")
            for p in root.rglob("*.py"))
        for hex_value in ("#00d4aa", "#3fb6b6", "#4ecdc4", "#5ba4cf"):
            assert hex_value in haystack, f"{hex_value} no longer exists"
