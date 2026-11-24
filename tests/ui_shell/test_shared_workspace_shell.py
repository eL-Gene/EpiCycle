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
# the amber line's width, measured with real fonts in a child process
# ---------------------------------------------------------------------------
#
# The headless Qt platform finds no fonts on Windows unless QT_QPA_FONTDIR
# points at them, and then every letter is the same wide box, so a pixel
# sum measures nothing real.  Setting that variable in this process would
# give real fonts to every other test collected with this file, and some of
# those (the row pictures) are pinned to the box letters.  So the widths are
# measured in a child process whose environment alone names the fonts.

_AMBER_WIDTHS = [520, 400, 300, 220, 140]
_FONT_DIR = "C:/Windows/Fonts"
_NO_FONTS_REASON = ("no fonts loaded on this Qt platform; text widths "
                    "cannot be measured")

_AMBER_CHILD = """
import importlib.util, json, os, sys
from pathlib import Path
test_file, root, widths = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
spec = importlib.util.spec_from_file_location("_amber_width_child", test_file)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from qtpy.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
# Keep the child away from the developer's real ~/.epicycle.
home = root / "home"
home.mkdir()
mod.wf.Path.home = staticmethod(lambda: home)
out = mod._measure_amber_widths(app, root, [int(w) for w in widths])
print("AMBERWIDTHS" + json.dumps(out), flush=True)
os._exit(0)
"""


def _measure_amber_widths(app, root, widths):
    """Render the diverged line at each width and add up what it takes.

    Runs in the child process only.  Returns ``{"no_fonts": True}`` when even
    the child's Qt finds no font families.
    """
    import re
    from qtpy.QtGui import QFont, QFontDatabase
    try:
        families = QFontDatabase.families()
    except TypeError:              # Qt5: families() is not static
        families = QFontDatabase().families()
    if not families:
        return {"no_fonts": True}
    out = {}
    for width in widths:
        wf.clear_cache()
        base = root / f"w{width}"
        base.mkdir()
        shell = _make_shell(app)
        a = base / ("Diverged" + "Long" * 25)
        b = base / ("Shared" + "Long" * 25)
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
        row = {"mono_found": bool(mono_part), "html": html_text,
               "label_width": label.width(), "total": 0, "plain": ""}
        if mono_part:
            plain = re.sub(r"<[^>]+>", "", html_text)
            rest = plain.replace(mono_part.group(1), "")
            ui = QFontMetrics(label.font())
            mono_font = QFont(label.font())
            mono_font.setFamilies(
                ["Consolas", "Cascadia Mono", "Courier New", "monospace"])
            row["total"] = (ui.horizontalAdvance(rest)
                            + QFontMetrics(mono_font).horizontalAdvance(
                                mono_part.group(1)))
            row["plain"] = plain
        out[str(width)] = row
    return out


@pytest.fixture(scope="module")
def amber_widths(tmp_path_factory):
    import subprocess
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    root = tmp_path_factory.mktemp("amber_widths")
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    if os.path.isdir(_FONT_DIR):
        env["QT_QPA_FONTDIR"] = _FONT_DIR
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _AMBER_CHILD, str(Path(__file__).resolve()),
         str(root)] + [str(w) for w in _AMBER_WIDTHS],
        cwd=str(repo), env=env, capture_output=True, text=True, timeout=300)
    lines = [ln for ln in proc.stdout.splitlines()
             if ln.startswith("AMBERWIDTHS")]
    assert lines, (proc.returncode, proc.stdout[-1500:], proc.stderr[-3000:])
    return json.loads(lines[-1][len("AMBERWIDTHS"):])


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

    def _crumbs(self, monkeypatch):
        from napari_epicycle.utils import crash_logger
        seen = []
        monkeypatch.setattr(crash_logger, "breadcrumb", seen.append)
        return seen

    def test_accepted_switch_logs_one_step_line(self, app, monkeypatch):
        shell = self._shell(app)
        shell._active_step_id = "project_setup"
        seen = self._crumbs(monkeypatch)
        shell._select_step("apply_3d")
        assert [m for m in seen if m.startswith("step ")] == [
            "step project_setup -> apply_3d"]

    def test_reselecting_the_active_step_logs_no_step_line(
            self, app, monkeypatch):
        shell = self._shell(app)
        shell._select_step("apply_3d")
        seen = self._crumbs(monkeypatch)
        shell._select_step("apply_3d")
        assert [m for m in seen if m.startswith("step ")] == []

    def test_locked_nav_logs_no_step_line(self, app, monkeypatch):
        shell = self._shell(app)
        shell._active_step_id = "project_setup"
        shell._step_nav_locked = True
        seen = self._crumbs(monkeypatch)
        shell._select_step("apply_3d")
        assert [m for m in seen if m.startswith("step ")] == []

    def test_vetoed_exit_logs_no_step_line(self, app, monkeypatch):
        shell = self._shell(app)
        shell._select_step("apply_3d")
        shell._step_widgets["apply_3d"].controller.on_step_exit = lambda: False
        seen = self._crumbs(monkeypatch)
        shell._select_step("project_setup")
        assert shell._active_step_id == "apply_3d"
        assert [m for m in seen if m.startswith("step ")] == []

    def test_set_workspace_logs_a_line(self, app, tmp_path, monkeypatch):
        shell = self._shell(app)
        seen = self._crumbs(monkeypatch)
        ws = tmp_path / "Proj"
        ws.mkdir()
        _browse(shell, "project_setup", ws)
        assert any(m.startswith("workspace set from project_setup")
                   for m in seen)


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

    @pytest.mark.parametrize("kind", ["missing", "plain folder"])
    def test_a_remembered_folder_missing_or_not_a_workspace_stays_red(
            self, app, tmp_path, capsys, kind):
        """Only a folder that holds a workspace is adopted at start-up; a
        folder that is gone, or holds no workspace, stays a suggestion."""
        shell = _make_shell(app)
        QWidget.__init__(shell)        # live Qt object: no "closing" skip
        ws = tmp_path / "Remembered"
        if kind == "plain folder":
            ws.mkdir()
            (ws / "notes.txt").write_text("not a workspace")
        shell._suggested_workspace = str(ws)
        shell._prefill_step1_from_suggestion()
        shell._adopt_remembered_workspace_on_start()
        shell._refresh_step_guidance("project_setup")
        assert shell.guidance_label._color_key == "red"
        assert shell.guidance_label._text == \
            "Choose a workspace folder to begin."
        assert shell._read_workspace_path() == ""
        assert ("[WORKSPACE] remembered workspace not adopted: the folder"
                in capsys.readouterr().out)

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

    @pytest.mark.parametrize("width", _AMBER_WIDTHS)
    def test_amber_line_fits_the_width_budget(self, amber_widths, width):
        """The diverged line must not run off the panel, at any width.

        The path is painted in the monospace face, so it is measured in that
        face too — measuring it in the UI font under-counts and the tail is
        clipped (seen in the live screenshots).

        The widths are measured in a child process with real fonts (see
        ``amber_widths``); this side only checks them.
        """
        if amber_widths.get("no_fonts"):
            pytest.skip(_NO_FONTS_REASON)
        got = amber_widths[str(width)]
        assert got["mono_found"], got["html"]
        assert got["total"] <= got["label_width"], (
            got["total"], got["label_width"], got["plain"])

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
# start-up adoption of the remembered workspace
# ---------------------------------------------------------------------------

def _pump(app, rounds: int = 10) -> None:
    """Let the event loop run pending zero-length timers."""
    for _ in range(rounds):
        app.processEvents()


def _pump_until(app, condition, timeout: float = 10.0) -> bool:
    """Run the event loop until `condition()` holds (or the timeout)."""
    import time as _time
    deadline = _time.perf_counter() + timeout
    while _time.perf_counter() < deadline:
        app.processEvents()
        if condition():
            return True
        _time.sleep(0.005)
    return condition()


class _Step1Controller(_RecordingController):
    """Step 1's side of the start-up handshake: the read-only refusal and
    the migration release, both recorded."""

    def __init__(self, line, refusal=""):
        super().__init__(line)
        self.refusal = refusal
        self.asked: list = []
        self.released: list = []

    def start_up_refusal(self, path):
        self.asked.append(path)
        return self.refusal

    def release_migration_offer(self, path):
        self.released.append(path)


class TestStartupAdoption:
    """The start-up rule: a remembered folder that holds a workspace is
    OPENED once the window has painted -- Step 1 adopts it as it adopts any
    shared workspace -- unless opening it would need a migration or a repair
    (Browse does those, with the user)."""

    def _shell(self, app, monkeypatch, suggested, refusal="",
               warm_done=True):
        import threading
        from napari_epicycle.ui_shell.workflow_shell import (
            EpiCycleWorkflowShell)
        monkeypatch.setattr(EpiCycleWorkflowShell,
                            "_START_ADOPT_DELAY_MS", 0)
        shell = _make_shell(app)
        line = shell._step_widgets["project_setup"].output_file_path
        shell._step_widgets["project_setup"].controller = _Step1Controller(
            line, refusal)
        # A real Qt object underneath, so the timers and the deletion check
        # behave as they do on the live shell.
        QWidget.__init__(shell)
        shell._suggested_workspace = str(suggested)
        warm = threading.Event()
        if warm_done:
            warm.set()

        def _no_thread_warmup():
            shell._warm_imports_done = warm
        shell._start_import_warmup = _no_thread_warmup
        shell._warm = warm
        saves: list = []
        real_save = wf.save_last_workspace
        monkeypatch.setattr(
            wf, "save_last_workspace",
            lambda path: (saves.append(path), real_save(path)))
        emitted: list = []
        line.textChanged.connect(lambda text: emitted.append(text))
        strips: list = []
        real_set = shell.guidance_label.set_strip

        def _recording_set(text, color_key, *a, **k):
            strips.append((text, color_key))
            return real_set(text, color_key, *a, **k)
        shell.guidance_label.set_strip = _recording_set
        # What `_build_ui` does, in its order.
        shell._schedule_remembered_workspace_adoption()
        shell._prefill_step1_from_suggestion()
        return shell, saves, emitted, strips

    @staticmethod
    def _workspace(tmp_path, marker="step1_state.json", name="Proj"):
        ws = tmp_path / name
        ws.mkdir()
        if marker.endswith(".json"):
            (ws / marker).write_text("{}", encoding="utf-8")
        else:
            (ws / marker).mkdir()
        return ws

    @staticmethod
    def _ctrl(shell):
        return shell._step_widgets["project_setup"].controller

    @pytest.mark.parametrize(
        "marker", ["step1_state.json", "step4_state.json", "artifacts"])
    def test_a_remembered_workspace_is_adopted_after_the_first_paint(
            self, app, tmp_path, monkeypatch, capsys, marker):
        monkeypatch.setenv("EPICYCLE_SYNC_DEBUG", "1")
        ws = self._workspace(tmp_path, marker)
        shell, saves, emitted, strips = self._shell(app, monkeypatch, ws)
        # Deferred: nothing is shared until the event loop has turned.
        assert shell._workspace_path == ""
        assert _pump_until(app, lambda: bool(shell._workspace_path))
        expected = os.path.abspath(str(ws))
        assert shell._workspace_path == expected
        # Step 1 ADOPTED it -- it was not browsed there -- so it follows a
        # later Browse in any step.
        assert shell._step_path_origin == {"project_setup": "adopted"}
        ctrl = self._ctrl(shell)
        assert ctrl.adopted == [(expected, "")]
        assert emitted == [expected]           # the box's ordinary chain
        assert ctrl.asked == [str(ws)]
        assert ctrl.released == [str(ws)]
        assert shell.guidance_label._color_key == "green"
        assert expected in shell.guidance_label._text
        assert "(shared across steps)" in shell.guidance_label._text
        out = capsys.readouterr().out
        assert ("[WORKSPACE] remembered workspace adopted at start-up: "
                f"{expected}") in out
        assert "[WORKSPACE-PERF]" in out
        assert ("remembered workspace shown; opening it after the first "
                "paint") in out
        assert len(saves) <= 1

    def test_the_opening_lists_sources_but_never_loads_an_image(
            self, app, tmp_path, monkeypatch):
        """The start-up rule: opening the remembered workspace lists Step 1's
        sources and loads none of them -- no Image layer, no source marked
        loaded, no load worker.  Step 1 here is the REAL controller (adopt,
        refusal check, migration release and the box's source restore)."""
        from napari.components import ViewerModel
        from napari.layers import Image
        from qtpy.QtWidgets import QListWidget
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController)

        # A clean workspace with one source whose prepared store is there.
        ws = tmp_path / "Clean_S"
        (ws / "artifacts" / "step1" / "largedata" / "Wound.zarr").mkdir(
            parents=True)
        (ws / "step1_state.json").write_text(json.dumps({
            "schema_version": 2,
            "sources": [{
                "source_id": "src_1",
                "path": str(tmp_path / "data" / "Wound.tif"),
                "display_name": "Wound",
                "prepared_store": "artifacts/step1/largedata/Wound.zarr",
            }],
            "patches": [],
        }), encoding="utf-8")

        loads: list = []
        monkeypatch.setattr(DataPreparationController, "load_source",
                            lambda self, *a, **k: loads.append(a))
        shell, saves, emitted, strips = self._shell(app, monkeypatch, ws)
        step1 = shell._step_widgets["project_setup"]
        step1.source_file_list = QListWidget(step1)
        ctrl = DataPreparationController.__new__(DataPreparationController)
        ctrl.widget = step1
        ctrl.viewer = ViewerModel()
        ctrl._step1_migration_offered = set()
        ctrl._step1_sources_restored = set()
        ctrl._suppress_publish = False
        ctrl._workflow_shell = shell
        ctrl._scanned_files = []
        ctrl._suspend_source_restore = False
        ctrl._loaded_source_file = None
        ctrl._loaded_source_layers = []
        ctrl._source_load_worker = None
        ctrl._source_loading = False
        step1.controller = ctrl
        # The box's source restore, as Step 1 wires it.
        step1.output_file_path.textChanged.connect(
            lambda _t: ctrl._maybe_restore_step1_sources())

        assert _pump_until(app, lambda: bool(shell._workspace_path))
        assert shell._step_path_origin == {"project_setup": "adopted"}
        listed = [step1.source_file_list.item(i).data(Qt.UserRole)
                  for i in range(step1.source_file_list.count())]
        assert listed == [str(tmp_path / "data" / "Wound.tif")]
        assert loads == []
        assert ctrl._loaded_source_file is None
        assert ctrl._loaded_source_layers == []
        assert ctrl._source_load_worker is None
        assert ctrl._source_loading is False
        assert [layer for layer in ctrl.viewer.layers
                if isinstance(layer, Image)] == []

    def test_the_perf_line_is_a_debug_line(
            self, app, tmp_path, monkeypatch, capsys):
        monkeypatch.delenv("EPICYCLE_SYNC_DEBUG", raising=False)
        ws = self._workspace(tmp_path)
        shell, *_ = self._shell(app, monkeypatch, ws)
        assert _pump_until(app, lambda: bool(shell._workspace_path))
        out = capsys.readouterr().out
        assert "adopted at start-up" in out
        assert "[WORKSPACE-PERF]" not in out

    def test_the_first_frames_say_opening_never_choose_a_workspace(
            self, app, tmp_path, monkeypatch):
        ws = self._workspace(tmp_path)
        shell, saves, emitted, strips = self._shell(app, monkeypatch, ws)
        # Before the deferred work: a calm, neutral line.
        assert shell.guidance_label._text == wf.MSG_OPENING.format(
            name="Proj")
        assert shell.guidance_label._color_key == "grey"
        assert shell._read_active_step_status("project_setup") == \
            "Step 1  ·  opening…"
        assert _pump_until(app, lambda: bool(shell._workspace_path))
        assert strips, "the banner was never painted"
        assert all(text != "Choose a workspace folder to begin."
                   for text, _key in strips), strips
        assert all(key != "red" for _text, key in strips), strips
        assert strips[-1][1] == "green"

    def test_step1_follows_a_later_browse_in_another_step(
            self, app, tmp_path, monkeypatch):
        """Adopt A at start-up, Browse B in Step 2, come back to Step 1:
        Step 1 is on B and the banner is green, not amber on A."""
        a = self._workspace(tmp_path, name="A")
        b = self._workspace(tmp_path, name="B")
        shell, *_ = self._shell(app, monkeypatch, a)
        assert _pump_until(app, lambda: bool(shell._workspace_path))
        shell._active_step_id = "roi_hitl_2d"
        _browse(shell, "roi_hitl_2d", b)
        shell._active_step_id = "project_setup"
        shell._adopt_into_step("project_setup")     # what entering Step 1 does
        shell._refresh_divergence()
        shell._refresh_step_guidance()
        assert shell._step_widgets["project_setup"].output_file_path.text() \
            == os.path.abspath(str(b))
        assert shell._diverged_steps == {}
        assert shell._step_path_origin["project_setup"] == "adopted"
        assert shell.guidance_label._color_key == "green"

    @pytest.mark.parametrize("refusal,kind", [
        ("it needs the migration Browse offers", "migration"),
        ("its step1_state.json cannot be read; Browse moves it aside and "
         "starts afresh", "repair"),
        ("its prepared-store records are out of date; Browse repairs them",
         "repair"),
    ])
    def test_a_folder_that_needs_browse_is_named_not_prefilled(
            self, app, tmp_path, monkeypatch, capsys, refusal, kind):
        """Refused: the box is emptied (so no Step 1 entry restores -- and
        writes into -- a folder nobody opened), the red line stays, and the
        line under it names the folder and what Browse will do."""
        ws = self._workspace(tmp_path)
        shell, saves, emitted, strips = self._shell(
            app, monkeypatch, ws, refusal=refusal)
        ctrl = self._ctrl(shell)
        assert _pump_until(app, lambda: bool(ctrl.asked))
        _pump(app)
        assert shell._workspace_path == ""
        assert shell._step_path_origin == {}
        assert ctrl.adopted == [] and ctrl.released == []
        assert emitted == [] and saves == []          # emptied silently
        step1 = shell._step_widgets["project_setup"]
        assert step1.output_file_path.text() == ""
        # Step 1's Browse dialog starts at the remembered folder.
        assert step1.remembered_workspace == str(ws)
        # Refused: today's red sentence, and the folder named under it.
        assert shell.guidance_label._text == \
            "Choose a workspace folder to begin."
        assert shell.guidance_label._color_key == "red"
        expected = (wf.MSG_REMEMBERED_MIGRATION if kind == "migration"
                    else wf.MSG_REMEMBERED_REPAIR).format(name="Proj")
        assert shell.advisory_label._text == expected
        assert shell._read_active_step_status("project_setup") == \
            "Step 1  ·  nothing loaded"
        out = capsys.readouterr().out
        assert f"[WORKSPACE] remembered workspace not adopted: {refusal}" \
            in out

    def test_the_named_folder_goes_once_a_workspace_is_shared(
            self, app, tmp_path, monkeypatch):
        ws = self._workspace(tmp_path)
        other = self._workspace(tmp_path, name="Other")
        shell, *_ = self._shell(
            app, monkeypatch, ws,
            refusal="it needs the migration Browse offers")
        assert _pump_until(app, lambda: bool(self._ctrl(shell).asked))
        _pump(app)
        _browse(shell, "project_setup", other)
        assert shell.guidance_label._color_key == "green"
        assert shell.advisory_label._text == ""

    def test_a_plain_folder_stays_a_suggestion(
            self, app, tmp_path, monkeypatch, capsys):
        ws = tmp_path / "Plain"
        ws.mkdir()
        shell, saves, emitted, strips = self._shell(app, monkeypatch, ws)
        _pump(app)
        assert shell._workspace_path == ""
        assert self._ctrl(shell).asked == []
        assert emitted == [] and saves == []
        assert shell._step_widgets["project_setup"].output_file_path.text() \
            == str(ws)
        assert shell.guidance_label._color_key == "red"
        out = capsys.readouterr().out
        assert "remembered workspace not adopted" in out
        assert "remembered workspace shown (nothing loaded)" in out
        assert "opening it after the first paint" not in out

    def test_no_remembered_workspace_says_so_once(
            self, app, monkeypatch, capsys):
        shell = _make_shell(app)
        QWidget.__init__(shell)
        shell._schedule_remembered_workspace_adoption()
        _pump(app)
        out = capsys.readouterr().out
        assert out.count("[WORKSPACE] no remembered workspace (none saved, "
                         "or its folder is gone)") == 1
        assert shell._workspace_path == ""

    def test_it_waits_for_the_import_warm_up(
            self, app, tmp_path, monkeypatch):
        ws = self._workspace(tmp_path)
        shell, *_ = self._shell(app, monkeypatch, ws, warm_done=False)
        _pump(app, 30)
        assert shell._workspace_path == ""        # still warming up
        shell._warm.set()
        assert _pump_until(app, lambda: bool(shell._workspace_path))

    def test_a_warm_up_that_never_ends_is_capped(
            self, app, tmp_path, monkeypatch):
        from napari_epicycle.ui_shell.workflow_shell import (
            EpiCycleWorkflowShell)
        monkeypatch.setattr(EpiCycleWorkflowShell, "_WARM_UP_CAP_S", 0.2)
        ws = self._workspace(tmp_path)
        shell, *_ = self._shell(app, monkeypatch, ws, warm_done=False)
        assert _pump_until(app, lambda: bool(shell._workspace_path),
                           timeout=5.0)

    def test_the_timer_does_nothing_once_the_window_is_closing(
            self, app, tmp_path, monkeypatch):
        ws = self._workspace(tmp_path)
        shell, saves, emitted, strips = self._shell(app, monkeypatch, ws)
        shell._closing = True
        _pump(app, 30)
        assert shell._workspace_path == ""
        assert emitted == [] and saves == []
        assert self._ctrl(shell).asked == []

    def test_the_timer_does_nothing_once_the_shell_is_deleted(
            self, app, tmp_path, monkeypatch):
        from qtpy import sip
        ws = self._workspace(tmp_path)
        shell, saves, emitted, strips = self._shell(app, monkeypatch, ws)
        ctrl = self._ctrl(shell)
        sip.delete(shell)
        _pump(app, 30)
        assert shell.__dict__["_workspace_path"] == ""
        assert emitted == [] and saves == []
        assert ctrl.asked == []

    def test_a_browse_to_the_same_folder_first_is_not_adopted_twice(
            self, app, tmp_path, monkeypatch, capsys):
        ws = self._workspace(tmp_path)
        shell, saves, emitted, strips = self._shell(app, monkeypatch, ws)
        _browse(shell, "project_setup", os.path.abspath(str(ws)))
        _pump(app, 30)
        assert shell._workspace_path == os.path.abspath(str(ws))
        assert len(saves) == 1                    # the Browse's own
        assert self._ctrl(shell).adopted == []
        assert "adopted at start-up" not in capsys.readouterr().out

    def test_a_browse_elsewhere_first_wins(self, app, tmp_path, monkeypatch):
        ws = self._workspace(tmp_path, name="Remembered")
        other = self._workspace(tmp_path, name="Chosen")
        shell, saves, emitted, strips = self._shell(app, monkeypatch, ws)
        _browse(shell, "project_setup", other)
        _pump(app, 30)
        assert shell._workspace_path == os.path.abspath(str(other))
        assert saves == [os.path.abspath(str(other))]

    def test_it_runs_once(self, app, tmp_path, monkeypatch):
        ws = self._workspace(tmp_path)
        shell, saves, emitted, strips = self._shell(app, monkeypatch, ws)
        shell._adopt_remembered_workspace_on_start()
        shell._adopt_remembered_workspace_on_start()
        _pump(app, 30)
        assert len(saves) == 1
        assert emitted == [os.path.abspath(str(ws))]

    def test_a_box_the_user_already_changed_is_left_alone(
            self, app, tmp_path, monkeypatch):
        ws = self._workspace(tmp_path)
        shell, saves, emitted, strips = self._shell(app, monkeypatch, ws)
        line = shell._step_widgets["project_setup"].output_file_path
        line.setText(str(tmp_path / "Typed"))
        emitted.clear()
        shell._adopt_remembered_workspace_on_start()
        assert shell._workspace_path == ""
        assert line.text() == str(tmp_path / "Typed")
        assert emitted == [] and saves == []

    def test_a_step1_without_its_box_says_why(
            self, app, tmp_path, monkeypatch, capsys):
        ws = self._workspace(tmp_path)
        shell, *_ = self._shell(app, monkeypatch, ws)
        shell._step_widgets["project_setup"] = QWidget()
        shell._adopt_remembered_workspace_on_start()
        assert shell._workspace_path == ""
        assert ("remembered workspace not adopted: Step 1 has no "
                "workspace box") in capsys.readouterr().out

    @pytest.mark.skipif(os.name == "nt",
                        reason="an absolute POSIX path is probed off Windows")
    def test_a_missing_folder_is_missing_off_windows(self, app, tmp_path):
        shell = _make_shell(app)
        assert shell._workspace_is_missing(str(tmp_path / "gone")) is True


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

    def test_the_log_link_opens_the_session_log_folder(
            self, app, tmp_path, monkeypatch):
        from napari_epicycle.ui_shell import workflow_shell as ws
        from napari_epicycle.utils import crash_logger
        log = tmp_path / "logs" / "epicycle_session_x.log"
        monkeypatch.setattr(crash_logger, "log_path", lambda: log)
        opened = []
        monkeypatch.setattr(ws, "_open_folder", opened.append)
        card = ws._StatusCard()
        assert card._log_link.text() == "Open log folder"
        card._refresh_log_tooltip()
        assert card._log_link.toolTip() == str(log)
        card._open_log_folder()
        assert opened == [str(log.parent)]

    def test_the_log_link_with_no_log_shows_a_hint_and_opens_nothing(
            self, app, monkeypatch):
        from napari_epicycle.ui_shell import workflow_shell as ws
        from napari_epicycle.utils import crash_logger
        monkeypatch.setattr(crash_logger, "log_path", lambda: None)
        opened = []
        monkeypatch.setattr(ws, "_open_folder", opened.append)
        card = ws._StatusCard()
        card._refresh_log_tooltip()
        assert card._log_link.toolTip() == "no session log"
        card._open_log_folder()
        assert opened == []

    def test_a_failing_opener_does_not_raise(self, app, tmp_path, monkeypatch):
        from napari_epicycle.ui_shell import workflow_shell as ws
        from napari_epicycle.utils import crash_logger
        monkeypatch.setattr(crash_logger, "log_path",
                            lambda: tmp_path / "a.log")

        def boom(_):
            raise OSError("no such folder")

        monkeypatch.setattr(ws, "_open_folder", boom)
        ws._StatusCard()._open_log_folder()


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
