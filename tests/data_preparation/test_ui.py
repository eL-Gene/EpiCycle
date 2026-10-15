"""Tests for the Step 1 (Data Preparation) UI rebuild.

**Every assertion here checks BEHAVIOUR, not existence.**  That rule is the
direct result of a failure: an earlier attempt at this rebuild shipped a suite
whose central test was `assert getattr(widget, attr) is not None` over 76
attribute names, and reported "52/52 elements" from it.  Two independent audits
then found ~15 defects — unwired buttons, a permanently-hidden progress bar,
labels nothing ever writes, a control in the wrong parent, a plain QListWidget
standing in for the Box List — and **every one of them passed that suite**.

So an element is only "present" here if the test proves something about how it
behaves: it is connected, or it is disabled with a stated reason, or writing to
it changes what is displayed, or it sits inside the section it belongs to.

Two kinds of test:

**(A) Contract guards** — green against today's code and must stay green: the
frozen public-name-and-type contract, the connections,
the `_init_ui` ordering trap, and the phase-1 invariant that `ui.py`
touches no viewer artefact.

**(B) Feature tests** for whatever the current step has built.

Source-level guards go through `_code_only`: a plain substring search over
source is the wrong tool for "does the code reference X", because `ui.py`
legitimately documents in prose the very things it no longer does.  Four tests
in the earlier attempt were defeated by exactly that.
"""
from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

import pytest

# Pure, napari-free preprocessing model (Brick 1a) the ② column UI binds to.
from packages.epicycle_preprocessing.recipe import (
    GeometryStep,
    flatten_for_channel,
)
from packages.epicycle_preprocessing.registry import REGISTRY

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STEP1_DIR = _REPO_ROOT / "napari_epicycle" / "data_preparation"
_UI_PY = _STEP1_DIR / "ui.py"
_CONTROLLER_PY = _STEP1_DIR / "controller.py"
_SHELL_PY = _REPO_ROOT / "napari_epicycle" / "ui_shell" / "workflow_shell.py"
_GUIDES_DIR = _REPO_ROOT / "docs" / "guides"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _norm(text: str) -> str:
    """Collapse whitespace so a multi-line call matches a one-line pattern."""
    return re.sub(r"\s+", " ", text)


def _code_only(path: Path) -> str:
    """Source with comments and string literals blanked, line structure kept."""
    src = path.read_text(encoding="utf-8")
    out = list(src)
    offsets = [0]
    for line in src.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))

    def _blank(start, end):
        (srow, scol), (erow, ecol) = start, end
        begin, finish = offsets[srow - 1] + scol, offsets[erow - 1] + ecol
        for i in range(begin, min(finish, len(out))):
            if out[i] != "\n":
                out[i] = " "

    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                _blank(tok.start, tok.end)
    except (tokenize.TokenError, IndentationError):  # pragma: no cover
        pytest.fail(f"could not tokenize {path}")
    return "".join(out)


def _is_connected(button) -> bool:
    """True if anything is connected to `button.clicked`.

    Qt has no portable connection count, but `disconnect()` raises TypeError on
    PyQt (and returns False on PySide) when there is nothing connected — so
    probing and immediately restoring is the reliable cross-binding check.
    Callers pass a throwaway widget, or re-connect after.
    """
    try:
        return bool(button.receivers(button.clicked))
    except (AttributeError, TypeError):
        try:
            button.clicked.disconnect()
            return True
        except (TypeError, RuntimeError):
            return False


# ---------------------------------------------------------------------------
# (A) Contract guards — source level, no Qt
# ---------------------------------------------------------------------------


class TestUiTouchesNoViewerArtefacts:
    """Phase 1's defining invariant.

    `ui.py` builds widgets; it must not create napari layers, write
    `layer.scale`, or reach for the canvas.  Scanned over CODE
    only, so prose may name what the code must not do.
    """

    FORBIDDEN = (
        ("add_image", r"add_image"),
        ("add_shapes", r"add_shapes"),
        ("add_labels", r"add_labels"),
        ("layer.scale", r"layer\.scale"),
        (".canvas", r"\.canvas\b"),
    )

    @pytest.mark.parametrize("label,pattern", FORBIDDEN,
                             ids=[f[0] for f in FORBIDDEN])
    def test_forbidden_token_absent(self, label, pattern):
        hits = [
            (i, line.strip())
            for i, line in enumerate(_code_only(_UI_PY).splitlines(), start=1)
            if re.search(pattern, line)
        ]
        assert not hits, (
            f"ui.py must not reference {label!r}: {hits}"
        )


class TestUiImportsStayStep1Local:
    """`ui.py` imports nothing cross-step."""

    # Level-1 imports are same-package and fine; the concern is
    # cross-STEP imports, which the absolute-import test below also covers.
    # `_paths` (B3): the inline `_step1_patches_dir` duplicate was folded into
    # the module that owns the layout.  Same-package, which this test's own
    # comment above says is fine — the rule is about CROSS-STEP imports.
    # `ui_shell._guides` (2026-10-13): the shared help-guide resolver, the same
    # shared layer as `ui_shell.theme` -- not a step.
    ALLOWED_RELATIVE = {(2, "ui_shell.theme"), (2, "ui_shell._guides"), (1, "controller"),
                        (1, "logic"), (1, "_paths")}

    def test_no_unexpected_relative_imports(self):
        tree = ast.parse(_read(_UI_PY))
        unexpected = [
            (n.level, n.module or "", n.lineno)
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and n.level
            and (n.level, n.module or "") not in self.ALLOWED_RELATIVE
        ]
        assert not unexpected, f"cross-step relative import: {unexpected}"

    def test_no_absolute_cross_step_imports(self):
        tree = ast.parse(_read(_UI_PY))
        bad = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module or ""]
            else:
                continue
            bad += [
                (n, node.lineno) for n in names
                if n.startswith("napari_epicycle")
                and not n.startswith("napari_epicycle.data_preparation")
                and n not in ("napari_epicycle.ui_shell.theme", "napari_epicycle.ui_shell._guides")
            ]
        assert not bad, f"cross-step absolute import: {bad}"


class TestInitUiOrdering:
    """The ordering trap.

    `_apply_theme` dereferences widget attributes **unguarded**, so every
    section must be built first; the controller is constructed after it (its
    `__init__` connects to widgets); `_connect_signals` dereferences
    `self.controller` and so comes last.
    """

    def _init_ui_source(self) -> str:
        src = _read(_UI_PY)
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.FunctionDef) and node.name == "_init_ui":
                return ast.get_source_segment(src, node) or ""
        pytest.fail("_init_ui not found")

    def test_apply_theme_precedes_controller(self):
        src = self._init_ui_source()
        assert -1 < src.find("self._apply_theme()") < src.find(
            "DataPreparationController("), (
            "_apply_theme() must run before the controller"
        )

    def test_controller_assigned_before_connect_signals(self):
        src = self._init_ui_source()
        assert -1 < src.find("self.controller =") < src.find(
            "self._connect_signals()"), (
            "self.controller must be assigned before _connect_signals()"
        )

    def test_all_panels_built_before_apply_theme(self):
        src = self._init_ui_source()
        theme_at = src.find("self._apply_theme()")
        late = [m.group(0) for m in re.finditer(r"self\._create_\w+_panel\(", src)
                if m.start() > theme_at]
        assert not late, f"panel built after _apply_theme(): {late}"

    def test_help_guides_initialised_before_panels(self):
        """`_make_subsection_header` appends to `_help_guides` during panel
        construction, so the list must exist first or construction raises."""
        src = self._init_ui_source()
        assert -1 < src.find("self._help_guides") < src.find(
            "self._create_session_panel("), (
            "_help_guides must be initialised before the first panel"
        )


class TestConnectionsAreReachable:
    """The connections, and that they live somewhere that runs.

    An earlier edit severed `_connect_signals` and stranded six connections
    inside an unrelated method; they were only wired if the user happened to
    tick a checkbox.  Every source-presence test still passed, because the
    calls WERE in the file — just unreachable.  Presence is not connectedness.
    """

    UI_CONNECTIONS = (
        # trap: these two are wired in the panel ctor, NOT _connect_signals
        "self.output_file_path.textChanged.connect(self._update_workspace_status)",
        "self.browse_button.clicked.connect(self.browse_output_file)",
        "self.free_bbox_button.clicked.connect(self.start_free_drawing)",
        "self.fixed_bbox_button.clicked.connect(self.start_fixed_drawing)",
        # F1: Extract runs on a worker + the button doubles as Cancel, so the
        # click now dispatches through `on_extract_or_cancel_clicked`.
        "self.extract_roi.clicked.connect(\n"
        "            self.controller.on_extract_or_cancel_clicked)",
        "self.clear_shapes_button.clicked.connect(self.controller.clear_shapes)",
        "self.source_browse_button.clicked.connect("
        "self.controller.browse_source_folder)",
        "self.source_file_list.itemClicked.connect("
        "self.controller.on_source_file_selected)",
        "self.load_source_button.clicked.connect(\n"
        "            self.controller.on_load_or_cancel_clicked)",
        "self.unload_source_button.clicked.connect("
        "self.controller.unload_source)",
        "self.viewer.mouse_drag_callbacks.append(self.mouse_callback)",
    )

    CONTROLLER_CONNECTIONS = (
        "self.viewer.layers.events.inserted.connect(self._on_layers_changed)",
        "self.viewer.layers.events.removed.connect(self._on_layers_changed)",
        "self.viewer.layers.events.inserted.connect(self._on_layer_inserted)",
        "self.widget.set_select_zrange_button.clicked.connect("
        "self._on_set_zrange_clicked)",
        "self.widget.auto_select_zrange_button.clicked.connect("
        "self._on_autoselect_zrange_clicked)",
        "self.widget.save_reviewed_roi.clicked.connect(self._on_export_clicked)",
        "self.widget.export_all_ready_roi.clicked.connect("
        "self._on_export_all_ready_clicked)",
        "self.widget.extracted_roi_list.itemSelectionChanged.connect("
        "self._on_roi_selection_changed)",
        "self._migration_offer_timer.timeout.connect("
        "self._maybe_offer_step1_migration)",
    )

    @pytest.mark.parametrize("expr", UI_CONNECTIONS)
    def test_ui_connection_present(self, expr):
        assert _norm(expr) in _norm(_read(_UI_PY)), f"ui.py lost: {expr}"

    @pytest.mark.parametrize("expr", CONTROLLER_CONNECTIONS)
    def test_controller_connection_present(self, expr):
        assert _norm(expr) in _norm(_read(_CONTROLLER_PY)), (
            f"controller.py lost: {expr}"
        )

    def test_connect_calls_live_in_a_method_that_runs(self):
        """Every `.connect(` in `ui.py` must sit in `_connect_signals` or a
        constructor — something `_init_ui` actually reaches."""
        # Constructors and the explicitly-invoked subscribe helpers.  Kept
        # narrow on purpose: the bug this test exists for was six connections
        # stranded inside `_toggle_all_boxes`, which is none of these.
        allowed = ("_connect_signals", "_init_ui", "__init__", "_create_",
                   "_make_", "_rebuild_", "_refresh_", "_show_", "_watch_",
                   "browse_output_file", "start_",
                   # Per-invocation dialogs, the "3 per-dialog" category:
                   # their widgets are built and destroyed per open.
                   "confirm_")
        offenders = []
        for func in ast.walk(ast.parse(_read(_UI_PY))):
            if not isinstance(func, ast.FunctionDef) or func.name.startswith(allowed):
                continue
            offenders += [
                (func.name, n.lineno) for n in ast.walk(func)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "connect"
            ]
        assert not offenders, (
            f".connect( in a method that may never run: {offenders}"
        )


class TestShellReachesStep1ByName:
    """The four names the SHELL resolves by `getattr`, so a rename
    fails silently: the UI renders and the feature just stops."""

    REGISTRATIONS = (
        '("project_setup", "output_file_path")',
        '("project_setup", "extracted_roi_list")',
        '("project_setup", "extract_roi", "ROI extraction triggered")',
        '("project_setup", "save_reviewed_roi", "ROI export triggered")',
        'getattr(widget, "extracted_roi_list", None)',
    )

    @pytest.mark.parametrize("expr", REGISTRATIONS)
    def test_registration_present(self, expr):
        assert _norm(expr) in _norm(_read(_SHELL_PY)), (
            f"the shell no longer reaches Step 1 via {expr}"
        )


# ---------------------------------------------------------------------------
# Qt fixtures
# ---------------------------------------------------------------------------


class _FakeEvent:
    def __init__(self):
        self.callbacks = []

    def connect(self, cb):
        self.callbacks.append(cb)
        return cb

    def disconnect(self, cb=None):
        self.callbacks.clear() if cb is None else self.callbacks.remove(cb)

    def __call__(self, *a, **k):
        for cb in list(self.callbacks):
            cb(*a, **k)


class _FakeLayerEvents:
    def __init__(self):
        self.inserted = _FakeEvent()
        self.removed = _FakeEvent()


class _FakeLayerList(list):
    def __init__(self, items=()):
        super().__init__(items)
        self.events = _FakeLayerEvents()


class _FakeViewer:
    """Enough napari Viewer for `DataPreparationController.__init__`, which
    connects to `layers.events` and iterates `layers` — `viewer=None` crashes.
    Follows the suite's `_Fake*` convention."""

    def __init__(self):
        self.layers = _FakeLayerList()
        self.mouse_drag_callbacks = []


@pytest.fixture(scope="module")
def app():
    """QApplication, mirroring `tests/ui_shell/test_help_dialog.py:20-30`.
    Deliberately here and not in the shared `tests/conftest.py`."""
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication

    existing = QApplication.instance()
    if existing is not None:
        yield existing
        return
    yield QApplication([])


def _stage(widget, *names):
    """Add rows that read as `staged` — in memory, not yet on disk.

    ④'s row status comes from `controller.extracted_patches`; a row absent from
    it is `captured`, which v4.1 makes untickable (already on disk, so "export"
    would be a no-op or an overwrite).  Tests that want a tickable row must
    therefore stage it, exactly as `extract_patches` does in production.
    """
    widget.controller.extracted_patches = {n: {"data": None} for n in names}
    for n in names:
        widget.extracted_roi_list.addItem(n)
    widget.library_box_list.refresh()


@pytest.fixture
def widget(app):
    from napari_epicycle.data_preparation.ui import DataPreparationWidget

    w = DataPreparationWidget(viewer=_FakeViewer())
    yield w
    w.setParent(None)


@pytest.fixture(autouse=True)
def _qt_names(app, request):
    """Expose the handful of Qt names the test helpers use.

    Bound here rather than imported at module scope so the Qt-free source
    guards still run without a Qt binding.
    """
    import qtpy.QtCore as qtc
    import qtpy.QtWidgets as qtw

    module = request.node.module
    module.Qt = qtc.Qt
    module.QLabel = qtw.QLabel
    module.QListWidgetItem = qtw.QListWidgetItem
    yield


# ---------------------------------------------------------------------------
# (A cont.) Frozen contract, against a constructed widget
# ---------------------------------------------------------------------------


class TestFrozenNamesAndTypes:
    """Names + TYPES.

    is the load-bearing amendment: the shell calls type-specific APIs
    through `getattr`, so keeping the name is not enough.
    """

    FROZEN_TYPES = {
        "output_file_path": "QLineEdit",      # .textChanged, .text().strip()
        "extracted_roi_list": "QListWidget",  # .model(), .currentItem(), .count()
        "extract_roi": "QPushButton",
        "save_reviewed_roi": "QPushButton",
    }

    CONTROLLER_REFERENCED = (
        "auto_select_zrange_button", "bbox_layer_combo", "display_name_edit",
        "export_all_ready_roi", "extracted_roi_list", "image_layer_combo",
        "output_file_path", "plot_area",
        "progress_bar", "save_reviewed_roi",
        "set_select_zrange_button", "source_file_list", "source_folder_path",
        "source_progress_bar", "source_status_label", "transpose_button",
    )

    WIDGET_METHODS = (
        "update_image_layer_list", "update_shape_layer_list",
        "set_roi_queue_summary", "set_export_status", "set_selected_roi_status",
        "browse_output_file", "start_free_drawing", "start_fixed_drawing",
        "_set_draw_mode",
    )

    @pytest.mark.parametrize("name,typename", sorted(FROZEN_TYPES.items()))
    def test_shell_critical_name_keeps_its_type(self, widget, name, typename):
        import qtpy.QtWidgets as qtw

        found = getattr(widget, name, None)
        assert found is not None, f"{name!r} is FROZEN"
        assert isinstance(found, getattr(qtw, typename)), (
            f"{name!r} must stay a {typename}, got "
            f"{type(found).__name__}"
        )

    @pytest.mark.parametrize("name", CONTROLLER_REFERENCED)
    def test_controller_referenced_name_exists(self, widget, name):
        assert getattr(widget, name, None) is not None, (
            f"controller.py dereferences self.widget.{name}"
        )

    @pytest.mark.parametrize("name", WIDGET_METHODS)
    def test_widget_method_callable(self, widget, name):
        assert callable(getattr(widget, name, None))

    def test_widget_exposes_controller(self, widget):
        assert getattr(widget, "controller", None) is not None


class TestBusyLockList:
    """Trap 1: `_set_step1_busy` locks buttons by STRING LITERAL under a
    `getattr` guard.  Rename one and it silently stops being disabled."""

    @staticmethod
    def _literals():
        for node in ast.walk(ast.parse(_read(_CONTROLLER_PY))):
            if isinstance(node, ast.FunctionDef) and node.name == "_set_step1_busy":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.For) and isinstance(
                            sub.iter, (ast.Tuple, ast.List)):
                        return [e.value for e in sub.iter.elts
                                if isinstance(e, ast.Constant)]
        pytest.fail("could not find the _set_step1_busy name list")

    def test_every_locked_name_exists_and_can_be_disabled(self, widget):
        broken = [n for n in self._literals()
                  if not hasattr(getattr(widget, n, None), "setEnabled")]
        assert not broken, (
            f"_set_step1_busy names widgets that cannot be disabled: {broken}"
        )

    def test_locking_actually_disables_them(self, widget):
        """Behavioural, not structural: run the lock and read the widgets.

        Restoration is checked only for controls that were enabled BEFORE the
        lock — some are deliberately disabled (Apply has no handler yet), and
        forcing them on at unlock is its own defect.
        """
        names = self._literals()
        was_on = [n for n in names if getattr(widget, n).isEnabled()]
        widget.controller._set_step1_busy(True)
        still_on = [n for n in names if getattr(widget, n).isEnabled()]
        widget.controller._set_step1_busy(False)
        assert not still_on, f"still enabled while busy: {still_on}"
        not_back = [n for n in was_on if not getattr(widget, n).isEnabled()]
        assert not not_back, f"not restored after busy: {not_back}"


# ---------------------------------------------------------------------------
# (B) the family shell
# ---------------------------------------------------------------------------


class TestAccordionShell:
    """The 5-section skeleton."""

    EXPECTED = ("Session", "Load && Adjust", "Draw && Extract",
                "Review && Library", "Diagnostics")

    def test_five_sections_in_order(self, widget):
        titles = tuple(s._title for s in widget.accordion._sections)
        assert titles == self.EXPECTED, f"order drifted: {titles}"

    def test_ampersand_titles_are_escaped(self, widget):
        """`Accordion.addItem` puts the title into a QPushButton, where a lone
        `&` becomes a mnemonic and vanishes."""
        offenders = [s._title for s in widget.accordion._sections
                     if re.search(r"(?<!&)&(?!&)", s._title)]
        assert not offenders, f"unescaped '&' in {offenders}"

    def test_accordion_is_exclusive(self, widget):
        """Exactly one section open at a time."""
        open_now = [s._title for s in widget.accordion._sections if s._expanded]
        assert len(open_now) <= 1, f"{open_now} open together"

    def test_header_strip_is_unchanged(self, widget):
        """Decided 2026-07-26: the flow/header strip stays as it was.

        Guards against a well-meaning future edit renaming its stages to match
        the design mockup while the header is still frozen.
        """
        from qtpy.QtWidgets import QLabel

        texts = [c.text() for c in widget.findChildren(QLabel)]
        assert any("Workspace" in t and "ROI Capture" in t for t in texts), (
            "the original flow strip text is gone — the header is frozen"
        )


class TestDiagnosticsPane:
    """The log sink."""

    def test_matches_the_family_shape(self, widget):
        """The Console section is held to **parity with S2/S3/S4**.  The operative
        clause is parity, and Steps 3 and 4 are both `System` → separator →
        `Log Output` (`fusion_reconstruction/ui.py:3191,3204`;
        `segmentation/ui.py:1293,1307`), with Step 2 using the same `Log Output`
        heading.  An earlier pass renamed this "Console" from the design bench —
        but the bench is a mockup and the shipped steps are the family, which
        principle 1 makes the authority.
        """
        from qtpy.QtWidgets import QLabel

        section = widget.accordion._sections[4]
        headings = [c.text() for c in section.findChildren(QLabel)
                    if c.text() in ("System", "Log Output", "Console")]
        assert headings == ["System", "Log Output"], headings

    def test_system_line_exists_like_steps_3_and_4(self, widget):
        """Steps 3/4 show GPU here.  Step 1 has no GPU op, so it shows the
        number its own load guard decides on — available memory."""
        assert widget.system_status_label.text()
        assert "memory" in widget.system_status_label.text().lower()

    def test_log_format_matches_step_3(self, widget):
        """`[HH:MM:SS] [PREFIX] message`, Step 3's exact shape
        (`fusion_reconstruction/controller.py:18621-18623`), so the three logs
        read the same when a user pastes one into a bug report."""
        import re

        widget.log_output.clear()
        widget._log("STEP1-TEST", "hello")
        line = widget.log_output.toPlainText().strip()
        assert re.match(r"^\[\d{2}:\d{2}:\d{2}\] \[STEP1-TEST\] hello$", line), line

    def test_pane_matches_the_family_styling(self, widget):
        from napari_epicycle.ui_shell.theme import LOG_OUTPUT_STYLE

        assert widget.log_output.styleSheet() == LOG_OUTPUT_STYLE
        assert widget.log_output.minimumHeight() == 220, (
            "Steps 2/3/4 all use 220px"
        )

    def test_pane_is_a_read_only_text_edit(self, widget):
        from qtpy.QtWidgets import QTextEdit

        assert isinstance(widget.log_output, QTextEdit)
        assert widget.log_output.isReadOnly(), "the pane is a sink, not an input"

    def test_pane_lives_in_the_diagnostics_section(self, widget):
        """Parent-chain membership, not mere existence — an earlier build put a
        whole-session readout in the wrong section."""
        diagnostics = widget.accordion._sections[-1]
        assert widget.log_output in diagnostics.findChildren(
            type(widget.log_output))

    def test_log_writes_prefix_and_message(self, widget):
        widget.log_output.clear()
        widget._log("STEP1-TEST", "hello")
        text = widget.log_output.toPlainText()
        assert "STEP1-TEST" in text and "hello" in text

    def test_log_accumulates(self, widget):
        widget.log_output.clear()
        widget._log("A", "first")
        widget._log("B", "second")
        text = widget.log_output.toPlainText()
        assert "first" in text and "second" in text

    def test_log_survives_a_missing_pane(self, widget):
        """Callable during construction and after teardown, so no caller needs
        a guard."""
        saved = widget.log_output
        del widget.log_output
        widget._log("STEP1-TEST", "must not raise")
        widget.log_output = saved

    def test_controller_log_reaches_the_pane(self, widget):
        """The controller's 22 rewritten print sites must land in the pane, not
        only on stdout."""
        widget.log_output.clear()
        widget.controller._log("STEP1-ZRANGE", "routed from the controller")
        assert "routed from the controller" in widget.log_output.toPlainText()

    def test_no_print_bypasses_the_sink(self):
        """`_log` mirrors to stdout on purpose — that is the sink, not a bypass.
        Every other `print(` in either file is a message the user never sees."""
        for path in (_UI_PY, _CONTROLLER_PY):
            src = _read(path)
            exempt = set()
            for node in ast.walk(ast.parse(src)):
                if isinstance(node, ast.FunctionDef) and node.name == "_log":
                    exempt |= set(range(node.lineno,
                                        (node.end_lineno or node.lineno) + 1))
            offenders = [
                (path.name, i, line.strip())
                for i, line in enumerate(_code_only(path).splitlines(), start=1)
                if re.search(r"(?<![\w.])print\s*\(", line) and i not in exempt
            ]
            assert not offenders, f"print() outside _log(): {offenders}"


class TestSubsectionHelp:
    """`?` buttons and their guides."""

    def test_guides_are_registered(self, widget):
        """v1 only builds a `?` whose guide is really on disk, so this is the
        number of guides that have been WRITTEN, not the number planned."""
        assert len(widget._help_guides) >= 2

    def test_every_guide_is_a_step1_markdown(self, widget):
        bad = [g for g in widget._help_guides
               if not (g.startswith("step1_") and g.endswith(".md"))]
        assert not bad, f"guide names must be step1_*.md: {bad}"

    def test_session_guide_is_written(self):
        """Phase 1 names this as the one guide to author first; the `?`
        buttons whose guide is not written yet are not built at all (v1)."""
        assert (_GUIDES_DIR / "step1_session.md").is_file()

    def test_help_buttons_are_connected(self, widget):
        """A `?` that opens nothing is worse than no `?`."""
        from qtpy.QtWidgets import QPushButton

        helps = [b for b in widget.findChildren(QPushButton)
                 if b.toolTip().startswith("Help:")]
        assert len(helps) >= 2, f"only {len(helps)} help buttons found"
        unwired = [b.toolTip() for b in helps if not _is_connected(b)]
        assert not unwired, f"help buttons connected to nothing: {unwired}"

    def test_no_action_button_in_a_header_row(self):
        """A header row carries only title + stretch + `?`."""
        src = _code_only(_UI_PY)
        for node in ast.walk(ast.parse(_read(_UI_PY))):
            if (isinstance(node, ast.FunctionDef)
                    and node.name == "_make_subsection_header"):
                body = src[
                    sum(len(l) + 1 for l in src.splitlines()[:node.lineno - 1]):
                ][:2000]
                assert "insertWidget" not in body, (
                    "the header helper must not accept extra widgets"
                )
                return
        pytest.fail("_make_subsection_header not found")


class TestSummaryDashboard:
    """`new · staged · captured`, and no fourth counter."""

    def test_three_counters_and_no_ready(self, widget):
        for name in ("summary_new_label", "summary_staged_label",
                     "summary_captured_label"):
            assert getattr(widget, name, None) is not None
        assert getattr(widget, "summary_ready_label", None) is None, (
            "the `ready` counter is retired — it must not come back"
        )

    def test_counters_start_at_zero(self, widget):
        assert widget.summary_new_label.text() == "0 new"

    def test_refresh_updates_all_three(self, widget):
        widget.refresh_summary_dashboard(
            {"new": 3, "staged": 12, "captured": 7})
        assert widget.summary_new_label.text() == "3 new"
        assert widget.summary_staged_label.text() == "12 staged"
        assert widget.summary_captured_label.text() == "7 captured"

    @pytest.mark.parametrize("counts", [{}, None, {"staged": 2}])
    def test_partial_input_is_safe(self, widget, counts):
        widget.refresh_summary_dashboard(counts)
        assert widget.summary_new_label.text() == "0 new"

    def test_counters_live_in_the_session_section(self, widget):
        """Deliberate deviation from the mockup, which puts them in the header band:
        the header is frozen by user decision, so they live in ① instead."""
        session = widget.accordion._sections[0]
        assert widget.summary_new_label in session.findChildren(
            type(widget.summary_new_label))


class TestTheming:
    """One accent, and a toggle that shows its ON state."""

    def test_accent_is_a_named_constant(self):
        code = _code_only(_UI_PY)
        assert "_STEP1_ACCENT" in code
        assert 'Accordion(accent="#5ba4cf")' not in _norm(code), (
            "use Accordion(accent=_STEP1_ACCENT)"
        )

    def test_off_palette_colour_retired(self):
        assert "4a9ede" not in _code_only(_UI_PY), (
            "#4a9ede is an off-palette near-miss on _STEP1_ACCENT"
        )

    def test_checkable_button_has_a_visible_on_state(self, widget):
        """`transpose_button` is checkable and had SECONDARY_BTN_STYLE, which
        has no `:checked` rule — so it had no ON state at all."""
        assert widget.transpose_button.isCheckable()
        assert ":checked" in widget.transpose_button.styleSheet()

    def test_both_progress_bars_share_the_accent(self, widget):
        assert (widget.progress_bar.styleSheet()
                == widget.source_progress_bar.styleSheet())

    def test_lists_use_the_teal_variant(self, widget):
        from napari_epicycle.ui_shell.theme import LIST_STYLE_TEAL

        for name in ("source_file_list", "extracted_roi_list",
                     "image_layer_combo"):
            assert getattr(widget, name).styleSheet() == LIST_STYLE_TEAL, (
                f"{name} is not on LIST_STYLE_TEAL"
            )


class TestSharedBusyBar:
    """②'s bar is a deliberate exception to the repo's
    per-op-bar convention, granted on ONE condition: because it serves two
    buttons it MUST name what is running.

    The previous attempt shipped `setVisible(False)` with no label mechanism at
    all, so during a load the user saw a nameless marquee. Every test here
    checks the label, not the widget's presence.
    """

    def test_bar_and_status_are_distinct_widgets(self, widget):
        """`source_status_label` is the load RESULT line the
        controller already writes in four colours. The shared bar needs its OWN
        status line, or ①'s "Found N TIFF file(s)" browse counter ends up
        rendering as ②'s busy text."""
        assert widget.busy_status_label is not widget.source_status_label

    def test_bar_lives_at_the_section_foot_not_in_a_column(self, widget):
        """"full width at the section bottom" — not inside the left column."""
        panel = widget.accordion._sections[1]
        assert widget.source_progress_bar in panel.findChildren(
            type(widget.source_progress_bar))
        left_col = widget.source_status_label.parentWidget()
        assert widget.source_progress_bar.parentWidget() is not left_col, (
            "the shared bar must not sit inside the Source column"
        )

    def test_busy_show_names_the_operation(self, widget):
        widget._busy_show("Reading wound_d3.tif…")
        assert "wound_d3" in widget.source_progress_bar.format()
        assert "wound_d3" in widget.busy_status_label.text()

    def test_busy_show_uses_scan_mode(self, widget):
        """No fake percentage where there is no denominator — Step 3's
        documented rationale."""
        widget._busy_show("Loading…")
        assert widget.source_progress_bar.minimum() == 0
        assert widget.source_progress_bar.maximum() == 0

    def test_busy_progress_uses_a_real_percentage(self, widget):
        widget._busy_progress("step 2 of 4 · isotropic", 2, 4)
        assert widget.source_progress_bar.maximum() == 4
        assert widget.source_progress_bar.value() == 2
        assert "isotropic" in widget.source_progress_bar.format()

    def test_busy_progress_falls_back_to_scan_without_a_total(self, widget):
        widget._busy_progress("Loading…", 0, 0)
        assert widget.source_progress_bar.maximum() == 0

    def test_busy_hide_returns_to_rest(self, widget):
        widget._busy_show("Loading…")
        widget._busy_hide()
        assert widget.source_progress_bar.maximum() == 1
        assert widget.source_progress_bar.format() == ""
        assert widget.busy_status_label.text() == "— idle"


class TestLoadColumn:
    """Left column: Source → Voxel size → Extract from."""

    def test_result_line_is_the_existing_wired_label(self, widget):
        """'Extend its content; do NOT build a new row and orphan
        the existing wiring.' The previous attempt added `load_result_label`,
        which nothing ever wrote, so the row read '— not loaded' forever."""
        assert getattr(widget, "load_result_label", None) is None, (
            "a second result row orphans source_status_label's four wired states"
        )
        assert widget.source_status_label is not None

    def test_result_line_sits_beside_load_not_at_the_panel_foot(self, widget):
        assert (widget.source_status_label.parentWidget()
                is widget.load_source_button.parentWidget())

    def test_axis_order_does_not_claim_a_detection_it_never_made(self, widget):
        """Names asserting an unverified axis order as the exact failure to
        avoid. No detection code exists yet, so the hint must not say
        'detected'."""
        assert "not detected" in widget.axis_order_hint.text().lower()

    def test_order_combo_is_the_single_visible_editor(self, widget):
        """One editor per field. `transpose_button` is frozen and read
        by controller.py:242, so it survives — hidden, and driven by the combo
        rather than sitting beside it."""
        # `isHidden()` rather than `isVisibleTo()`: ② is a COLLAPSED accordion
        # section, so ancestor visibility is False for everything inside it.
        # `isHidden` reports the explicit flag, which is what the code sets.
        assert widget.transpose_button.isHidden()
        assert not widget.axis_order_combo.isHidden()

    def test_order_combo_actually_drives_the_transpose_flag(self, widget):
        """Behavioural: the combo must change what `load_source` reads."""
        widget.axis_order_combo.setCurrentText("ZCYX")
        assert widget.transpose_button.isChecked() is True
        widget.axis_order_combo.setCurrentText("CZYX")
        assert widget.transpose_button.isChecked() is False

    def test_voxel_spacing_does_not_fabricate_a_reading(self, widget):
        """The mockup's 2/1/1 are illustrative. Shipping them as defaults with a
        'detected' label makes a panel that has never opened a file claim it
        read a 2 µm Z spacing."""
        assert "load a source" in widget.voxel_source_label.text().lower()
        assert widget.voxel_z_spin.value() == 1.0

    def test_reset_agrees_with_the_initial_state(self, widget):
        """`_detected_voxel_size` is the single source of truth for both, so ↺
        on a fresh panel cannot change the numbers."""
        before = [s.value() for s in (widget.voxel_z_spin, widget.voxel_y_spin,
                                      widget.voxel_x_spin)]
        widget.voxel_reset_button.click()
        after = [s.value() for s in (widget.voxel_z_spin, widget.voxel_y_spin,
                                     widget.voxel_x_spin)]
        assert before == after

    def test_editing_spacing_marks_it_user_supplied(self, widget):
        widget.voxel_z_spin.setValue(3.5)
        assert "hand" in widget.voxel_source_label.text()

    def test_aspect_note_reports_anisotropy(self, widget):
        widget.voxel_z_spin.setValue(2.0)
        assert "coarser" in widget.voxel_aspect_label.text()

    def test_aspect_note_says_isotropic_when_equal(self, widget):
        for s in (widget.voxel_z_spin, widget.voxel_y_spin, widget.voxel_x_spin):
            s.setValue(1.0)
        assert widget.voxel_aspect_label.text() == "isotropic"

    def test_view_toggle_mirrors_into_the_draw_section(self, widget):
        """The accordion is exclusive, so the Raw / Processed toggle in ②
        and the Extract button in ③ are never co-visible — the state must travel.
        Brick A: the toggle governs the view; selecting Processed mirrors
        'processed' into ③'s hint."""
        assert "from raw" in widget.extract_source_hint.text()
        # a processed version now exists → Processed lights up + is shown; the
        # controller reflects that via `_set_view_toggle_state`.
        widget._set_view_toggle_state(has_processed=True, showing_processed=True)
        assert "from processed" in widget.extract_source_hint.text()

    def test_processed_and_compare_greyed_until_processed_exists(self, widget):
        """Brick A greying: Processed + Compare are disabled until a processed
        version exists this session; Processed lights up once it does (Compare
        stays present-but-disabled until Brick C)."""
        assert not widget.view_processed_button.isEnabled()
        assert not widget.compare_raw_processed_button.isEnabled()
        assert widget.view_processed_button.toolTip()
        widget._set_view_toggle_state(has_processed=True, showing_processed=False)
        assert widget.view_processed_button.isEnabled()
        # raw stays the selected segment while showing_processed is False
        assert widget.view_raw_button.isChecked()

    def test_view_toggle_routes_to_controller_view_swap(self, widget):
        """Clicking a segment routes to the controller's mutually-exclusive VIEW
        swap (behaviour, not existence)."""
        calls = []
        widget.controller._show_processed_view = lambda: calls.append("processed")
        widget.controller._show_raw_view = lambda: calls.append("raw")
        widget.view_processed_button.setEnabled(True)
        widget.view_processed_button.click()
        assert calls and calls[-1] == "processed"
        assert "from processed" in widget.extract_source_hint.text()


def _columns(widget):
    """The per-channel card widgets currently rendered in ② (QFrame, objectName
    `recipeCard` — the box-list rebuild 2026-08-07)."""
    from qtpy.QtWidgets import QWidget as _QW
    container = widget.recipe_columns_container
    return [w for w in container.findChildren(_QW)
            if w.objectName() == "recipeCard"]


def _geo_chips(widget):
    """The shared-geometry chips rendered across ②'s columns (one per column
    when a geometry step is present)."""
    from qtpy.QtWidgets import QWidget as _QW
    container = widget.recipe_columns_container
    return [w for w in container.findChildren(_QW)
            if w.objectName() == "geoChip"]


def _channel_checkbox(widget, row: int):
    """The include QCheckBox in ③'s channel row `row` (the first — and only —
    checkbox in the row overlay built by `_make_channel_row`)."""
    from qtpy.QtWidgets import QCheckBox
    lst = widget.image_layer_combo
    row_widget = lst.itemWidget(lst.item(row))
    boxes = row_widget.findChildren(QCheckBox)
    return boxes[0] if boxes else None


class TestChannelColorsMatchCanvas:
    """COSMETIC fix (2026-08): the ② recipe cards, the ③ channel-picker swatches,
    and the before/after histogram curves must take each channel's colour from the
    SAME source as the canvas (`_SOURCE_DISPLAY_CMAPS`) — so channel 0's card no
    longer reads green while its canvas layer is magenta ("channels look
    switched").  All three now key off the ONE `_channel_display_hex(ci)` helper.
    """

    @staticmethod
    def _all_stylesheets(w):
        """Every stylesheet in the widget subtree, concatenated — so a swatch /
        header colour can be asserted regardless of which child carries it."""
        from qtpy.QtWidgets import QWidget as _QW
        sheets = [w.styleSheet()] + [c.styleSheet() for c in w.findChildren(_QW)]
        return " ".join(sheets)

    def test_helper_keys_off_source_display_cmaps(self):
        """`_channel_display_hex(ci)` is the canvas palette in canvas ORDER — ch0 =
        magenta, ch1 = green (the pre-fix chrome had ch0 green, ch1 magenta)."""
        import napari_epicycle.data_preparation.controller as ctrl_mod
        assert ctrl_mod._channel_display_hex(0) == ctrl_mod.source_cmap_hex(
            "magenta")
        assert ctrl_mod._channel_display_hex(1) == ctrl_mod.source_cmap_hex(
            "green")
        # Cycles past the 6-entry palette (never IndexErrors on >6 channels).
        assert ctrl_mod._channel_display_hex(6) == ctrl_mod._channel_display_hex(0)

    def test_recipe_card_header_uses_canvas_colour(self, widget):
        import napari_epicycle.data_preparation.controller as ctrl_mod
        for names in (["cad", "nuc"], ["cad", "nuc", "mem"]):
            widget.update_image_layer_list(names)
            cols = _columns(widget)
            assert len(cols) == len(names)
            for ci, card in enumerate(cols):
                expected = ctrl_mod._channel_display_hex(ci)
                assert expected in self._all_stylesheets(card), (
                    f"card {ci} must use canvas colour {expected}")

    def test_channel_picker_swatch_uses_canvas_colour(self, widget):
        import napari_epicycle.data_preparation.controller as ctrl_mod
        for names in (["cad", "nuc"], ["cad", "nuc", "mem"]):
            widget.update_image_layer_list(names)
            lst = widget.image_layer_combo
            assert lst.count() == len(names)
            for ci in range(len(names)):
                row = lst.itemWidget(lst.item(ci))
                expected = ctrl_mod._channel_display_hex(ci)
                assert expected in self._all_stylesheets(row), (
                    f"picker swatch {ci} must use canvas colour {expected}")

    def test_histogram_entry_lut_uses_canvas_colour(self, widget):
        import numpy as np
        import napari_epicycle.data_preparation.controller as ctrl_mod
        names = ["cad", "nuc", "mem"]
        # The Preview path resolves per-channel luts keyed off the canvas palette;
        # build the entries with those exact luts and confirm each is stored.
        luts = tuple(ctrl_mod._channel_display_hex(ci) for ci in range(len(names)))
        befores = [(np.random.rand(2, 4, 4) * 500).astype(np.float32)
                   for _ in names]
        afters = [np.ones((2, 4, 4), np.float32) for _ in names]
        entries = ctrl_mod.build_preview_histogram_entries(
            names, luts, befores, afters)
        assert len(entries) == len(names)
        for ci, entry in enumerate(entries):
            assert entry.lut == ctrl_mod._channel_display_hex(ci)

    def test_controller_hands_worker_canvas_luts(self, widget, monkeypatch):
        """Wiring: `on_preview_recipe` must hand the worker per-channel luts keyed
        off `_SOURCE_DISPLAY_CMAPS`, not a parallel palette."""
        import numpy as np
        import napari_epicycle.data_preparation.controller as ctrl_mod
        ctrl = widget.controller
        cad = (np.random.rand(6, 300, 300) * 500).astype(np.float32)
        nuc = (np.random.rand(6, 300, 300) * 500).astype(np.float32)
        ctrl._source_display_channels = [("cad", cad), ("nuc", nuc)]
        widget._recipe_channel_names = ["cad", "nuc"]

        captured = {}

        def _fake_worker(recipe_json, channels, bounds_list, spacing, nm, luts,
                         norm_bounds=None):
            captured["luts"] = luts
            raise RuntimeError("stop before the QThread starts")

        monkeypatch.setattr(ctrl_mod, "_PreprocPreviewWorker", _fake_worker)
        ctrl.on_preview_recipe()

        assert captured.get("luts") == (
            ctrl_mod._channel_display_hex(0), ctrl_mod._channel_display_hex(1))


class TestPreviewLayersNotCountedAsChannels:
    """The ② Preview drops ⟨preview⟩ processed-crop layers on the shared canvas
    (`_add_preview_layers`).  They are REAL napari Image layers, so before the
    fix a Preview of an N-channel source transiently inflated BOTH ③'s channel
    picker AND ②'s per-channel recipe columns to 2N — `_on_layers_changed` fed
    every Image layer straight through the one `update_image_layer_list`
    chokepoint that drives both.  The controller now excludes anything tracked in
    `controller._preview_layers` (identity-based, not a name-prefix match)."""

    def test_preview_layer_does_not_add_a_channel_or_recipe_column(self, widget):
        from napari.layers import Image as _Image
        import numpy as np

        ctl = widget.controller
        viewer = ctl.viewer
        # Two REAL source channels on the canvas.
        viewer.layers.append(_Image(np.zeros((3, 4, 5), dtype=np.uint16), name="cad"))
        viewer.layers.append(_Image(np.ones((3, 4, 5), dtype=np.uint16), name="nuc"))
        ctl._on_layers_changed()
        assert len(widget._recipe.channels) == 2
        assert len(_columns(widget)) == 2

        # A Preview render adds a ⟨preview⟩-marked Image layer, tracked in
        # controller._preview_layers exactly as `_add_preview_layers` does.
        prev = _Image(np.zeros((3, 4, 5), dtype=np.float32), name="⟨preview⟩ cad")
        viewer.layers.append(prev)
        ctl._preview_layers = [prev]
        ctl._on_layers_changed()

        # Both surfaces stay at 2: the preview is not a source channel.  (One
        # chokepoint feeds both, so a single controller-side exclusion fixes
        # picker + recipe columns together.)
        assert len(widget._recipe.channels) == 2, (
            "a ⟨preview⟩ layer must not inflate the recipe channel count")
        assert len(_columns(widget)) == 2, (
            "a ⟨preview⟩ layer must not add a recipe column")
        combo = widget.image_layer_combo
        picker = [combo.item(i).text() for i in range(combo.count())]
        assert picker == ["cad", "nuc"], (
            f"③ picker must not list the preview layer: {picker}")

    def test_preview_marker_is_a_single_source_constant(self):
        """The ⟨preview⟩ name literal lives in ONE place (the module constant),
        and `_add_preview_layers` builds its layer name from it."""
        from napari_epicycle.data_preparation import controller as _ctl

        assert getattr(_ctl, "PREVIEW_LAYER_MARKER", None) == "⟨preview⟩ "
        # The builder routes through the constant rather than an inline literal.
        assert "PREVIEW_LAYER_MARKER" in _code_only(_CONTROLLER_PY)

    def test_layers_changed_guarded_during_preview_add(self, widget):
        """quirk 1: during `_add_preview_layers`' inserts, each ⟨preview⟩ layer
        exists on the canvas a beat BEFORE it's tracked in `_preview_layers`, so
        an unguarded rebuild counts it as a channel (it flashed as an extra column
        on the FIRST preview until the next redraw dropped it).  The add now sets
        `_adding_preview_layers`, which makes `_on_layers_changed` a no-op for
        that window."""
        from napari.layers import Image as _Image
        import numpy as np

        ctl = widget.controller
        viewer = ctl.viewer
        viewer.layers.append(_Image(np.zeros((3, 4, 5), dtype=np.uint16), name="cad"))
        viewer.layers.append(_Image(np.ones((3, 4, 5), dtype=np.uint16), name="nuc"))
        ctl._on_layers_changed()
        assert len(_columns(widget)) == 2

        # Mid-add window: flag set, preview layer inserted BEFORE it's tracked.
        ctl._adding_preview_layers = True
        prev = _Image(np.zeros((3, 4, 5), dtype=np.float32), name="⟨preview⟩ cad")
        viewer.layers.append(prev)      # fires the inserted → _on_layers_changed
        ctl._on_layers_changed()        # + an explicit call — both must be no-ops
        assert len(_columns(widget)) == 2, (
            "a mid-add ⟨preview⟩ insert must not inflate the recipe columns")

        # Add finished: flag cleared + layer tracked → identity exclusion holds.
        ctl._adding_preview_layers = False
        ctl._preview_layers = [prev]
        ctl._on_layers_changed()
        assert len(_columns(widget)) == 2


class TestRecipeBuilder:
    """The per-channel COLUMN recipe builder (Brick 1b), live over a real
    `Recipe`.  Replaces the flat `_recipe_steps` / `_OP_SUITE` op-chain: the menu
    reads the real `packages/epicycle_preprocessing` registry, the columns are a
    projection of the Recipe, and geometry is ONE shared linked step."""

    def test_coming_soon_overlay_is_gone(self):
        """Deletes it; replaces it with live widgets + an empty state."""
        tree = ast.parse(_read(_UI_PY))
        assert not [
            a.name for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) for a in n.names
            if a.name == "ComingSoonOverlay"
        ], "ui.py still imports ComingSoonOverlay"

    # -- column build (0 / 1 / 2 / 3 channels) --

    def test_starts_empty_with_an_empty_state(self, widget):
        assert widget._recipe.channels == []
        assert widget._recipe.geometry is None
        assert not widget.recipe_empty_label.isHidden()
        assert not _columns(widget)

    @pytest.mark.parametrize("names", [
        ["cad"], ["cad", "nuc"], ["cad", "nuc", "third"],
    ])
    def test_one_column_per_detected_channel(self, widget, names):
        widget.update_image_layer_list(names)
        assert len(_columns(widget)) == len(names)
        assert widget.recipe_empty_label.isHidden()
        assert len(widget._recipe.channels) == len(names)

    def test_before_after_histogram_area_present_and_toggles(self, widget):
        """The before→after histogram slot must EXIST (restored 2026-08-07 — do
        not silently drop it again).  With the real pyqtgraph plot now always
        mounted in the vertical splitter it no longer hides on the empty state:
        it is present in BOTH states, so the guard is presence, not toggling."""
        assert hasattr(widget, "recipe_histogram_area")
        # The real pyqtgraph plot must be constructed (not a dead placeholder).
        import pyqtgraph as pg
        assert hasattr(widget, "_recipe_plot_widget")
        assert isinstance(widget._recipe_plot_widget, pg.PlotWidget)
        assert not widget.recipe_histogram_area.isHidden()      # always mounted
        widget.update_image_layer_list(["cad", "nuc"])
        assert not widget.recipe_histogram_area.isHidden()      # still present

    def test_body_is_a_vertical_splitter_with_two_panes(self, widget):
        """②'s body stacks the card strip over the plot in ONE vertical
        QSplitter (2026-08-07): box-list frame on top, plot frame below, neither
        collapsible."""
        from qtpy.QtWidgets import QSplitter
        from qtpy.QtCore import Qt as _Qt

        verticals = [s for s in widget.accordion._sections[1].findChildren(
            QSplitter) if s.orientation() == _Qt.Vertical]
        assert len(verticals) == 1, (
            "② must stack the card strip over the plot in exactly one vertical "
            f"splitter, found {len(verticals)}"
        )
        split = verticals[0]
        assert split.count() == 2, "the vertical splitter has exactly two panes"
        assert not split.childrenCollapsible(), (
            "neither pane may collapse to nothing"
        )

    def test_footer_is_pinned_outside_splitter_and_card(self, widget):
        """Button-hierarchy pass: Preview and Apply have DIFFERENT scopes and must
        stop reading as peers.  Apply (recipe-scoped, terminal bake) stays PINNED
        — a sibling of body_split, never a descendant of the vertical splitter OR
        the region-editor card.  Preview (region-scoped, iterative) now lives
        INSIDE the region card, so it must be a DESCENDANT of that card."""
        from qtpy.QtWidgets import QSplitter
        from qtpy.QtCore import Qt as _Qt

        section = widget.accordion._sections[1]
        vsplit = [s for s in section.findChildren(QSplitter)
                  if s.orientation() == _Qt.Vertical][0]
        card = widget.preview_region_card
        group = widget.preview_region_group

        # Apply — pinned OUTSIDE the splitter and the card.
        apply_chain, anc = [], widget.apply_preprocess_button.parent()
        while anc is not None:
            apply_chain.append(anc)
            anc = anc.parent()
        assert vsplit not in apply_chain, "Apply is inside the vertical splitter"
        assert card not in apply_chain, "Apply is inside the region-editor card"

        # Preview — now a DESCENDANT of the region card (and its editor body).
        preview_chain, anc = [], widget.test_region_button.parent()
        while anc is not None:
            preview_chain.append(anc)
            anc = anc.parent()
        assert card in preview_chain, "Preview must live inside the region card"
        assert group in preview_chain, (
            "Preview must live inside the region-editor body (preview_region_group)"
        )

    def test_preview_button_label_and_lives_inside_card(self, widget):
        """Button-hierarchy pass (a): Preview is relabelled "↻ Preview this
        region" and now lives INSIDE the region-editor card as its prominent
        bottom action."""
        assert widget.test_region_button.text() == "↻ Preview this region"
        card = widget.preview_region_card
        chain, anc = [], widget.test_region_button.parent()
        while anc is not None:
            chain.append(anc)
            anc = anc.parent()
        assert card in chain, "Preview must live inside the region-editor card"

    def test_apply_button_label_and_is_the_pinned_footer_button(self, widget):
        """Button-hierarchy pass (b): the footer action is labelled "Process
        whole volume" (Brick 1b-1 renamed it from the old "Apply recipe → whole
        volume (uint16)" stub) and stays PINNED — outside the splitter + card."""
        from qtpy.QtWidgets import QSplitter
        from qtpy.QtCore import Qt as _Qt

        assert widget.apply_preprocess_button.text() == "Process whole volume"
        section = widget.accordion._sections[1]
        vsplit = [s for s in section.findChildren(QSplitter)
                  if s.orientation() == _Qt.Vertical][0]
        card = widget.preview_region_card
        chain, anc = [], widget.apply_preprocess_button.parent()
        while anc is not None:
            chain.append(anc)
            anc = anc.parent()
        assert vsplit not in chain, "Apply is inside the vertical splitter"
        assert card not in chain, "Apply is inside the region-editor card"

    def test_collapsed_cta_relabelled(self, widget):
        """Button-hierarchy pass (c): the collapsed CTA reads "Place a region to
        preview"."""
        assert widget.place_preview_button.text() == (
            "▭ Place a region to preview")

    def test_preview_button_reveals_and_collapses_with_group(self, widget):
        """Button-hierarchy pass (d): Preview lives inside `preview_region_group`,
        so it reveals when a region is placed and collapses on teardown — driven
        entirely by the group's visibility (reuse of the reveal-test pattern).
        Uses isVisibleTo(card) because the button is never *itself* hidden; its
        visibility rides the group between it and the always-mounted card."""
        import numpy as np

        ctrl = widget.controller
        card = widget.preview_region_card
        ctrl._source_display_channels = [
            ("cad", np.random.rand(4, 32, 32).astype(np.float32))]
        widget._recipe_channel_names = ["cad"]
        widget._refresh_recipe_view()
        assert widget.preview_region_group.isHidden()           # collapsed start
        assert not widget.test_region_button.isVisibleTo(card)  # Preview hidden
        ctrl.place_preview_region()
        assert not widget.preview_region_group.isHidden()       # revealed on place
        assert widget.test_region_button.isVisibleTo(card)      # Preview shown
        ctrl._teardown_preview()
        assert widget.preview_region_group.isHidden()           # collapsed again
        assert not widget.test_region_button.isVisibleTo(card)  # Preview hidden

    def test_region_editor_card_always_visible_body_toggles(self, widget):
        """spec non-negotiable #2: the card FRAME is always mounted; only its
        body (`preview_region_group`) toggles on place / teardown."""
        import numpy as np

        card = widget.preview_region_card
        assert card is not None
        assert not card.isHidden()                       # frame always mounted
        assert widget.preview_region_group.isHidden()    # body collapsed at start

        ctrl = widget.controller
        ctrl._source_display_channels = [
            ("cad", np.random.rand(4, 32, 32).astype(np.float32))]
        widget._recipe_channel_names = ["cad"]
        widget._refresh_recipe_view()
        ctrl.place_preview_region()
        assert not card.isHidden()                       # frame STILL visible
        assert not widget.preview_region_group.isHidden()  # body revealed
        ctrl._teardown_preview()
        assert not card.isHidden()                       # frame survives collapse
        assert widget.preview_region_group.isHidden()    # body hidden again

    def test_slider_grid_is_two_by_two(self, widget):
        """spec non-negotiable #3: the four sliders sit in a 2×2 grid — X/Y in the
        LEFT column (rows 0/1), Z-position/Z-depth in the RIGHT column."""
        grid = widget._preview_slider_grid

        def pos(slider):
            r, c, _rs, _cs = grid.getItemPosition(grid.indexOf(slider))
            return r, c

        xr, xc = pos(widget.preview_x_slider)
        yr, yc = pos(widget.preview_y_slider)
        zr, zc = pos(widget.preview_z_slider)
        dr, dc = pos(widget.preview_zdepth_slider)
        assert (xr, yr, zr, dr) == (0, 1, 0, 1), "rows must be X0 Y1 · Zpos0 Zdep1"
        assert xc == yc and zc == dc, "each half shares a slider column"
        assert xc < zc, "X/Y column is left of the Z column"

    def test_cards_are_top_aligned(self, widget):
        """The load-bearing invariant: every card is added with Qt.AlignTop, so a
        shorter card cannot drop to vertical-centre when a sibling grows.  Build
        two columns of UNEQUAL op count and read the layout items' alignment."""
        from qtpy.QtCore import Qt as _Qt

        widget.update_image_layer_list(["cad", "nuc"])
        widget._add_op_from_menu(0, "gaussian")
        widget._add_op_from_menu(0, "clahe")
        widget._add_op_from_menu(0, "background")   # column 0 is taller
        widget._add_op_from_menu(1, "gaussian")     # column 1 is shorter
        assert len(_columns(widget)) == 2

        lay = widget._recipe_columns_layout
        card_aligns = [
            lay.itemAt(i).alignment()
            for i in range(lay.count())
            if lay.itemAt(i).widget() is not None
            and lay.itemAt(i).widget().objectName() == "recipeCard"
        ]
        assert len(card_aligns) == 2, "both cards must be in the layout"
        assert all(a & _Qt.AlignTop for a in card_aligns), (
            f"cards must be added with Qt.AlignTop, got {card_aligns}"
        )

    def test_no_channels_disables_preview_and_apply_with_a_reason(self, widget):
        for btn in (widget.test_region_button, widget.apply_preprocess_button):
            assert not btn.isEnabled()
            assert btn.toolTip(), "a disabled stub must state a reason"

    def test_channels_enable_apply_but_preview_needs_a_region(self, widget):
        """Preview v2 Piece 2: Apply enables on channels alone, but Preview stays
        disabled (with a hint) until a preview REGION is placed."""
        import numpy as np
        widget.update_image_layer_list(["cad"])
        assert widget.apply_preprocess_button.isEnabled()
        assert not widget.test_region_button.isEnabled()      # no region yet
        assert "region" in widget.test_region_button.toolTip().lower()
        # Placing a region flips Preview on.
        ctrl = widget.controller
        ctrl._source_display_channels = [
            ("cad", np.random.rand(4, 32, 32).astype(np.float32))]
        ctrl.place_preview_region()
        assert widget.test_region_button.isEnabled()

    # -- add intensity op → appears in THAT column's recipe only --

    def test_add_intensity_op_appends_to_that_channel_only(self, widget):
        widget.update_image_layer_list(["cad", "nuc"])
        widget._add_op_from_menu(0, "gaussian")
        assert [s.key for s in widget._recipe.channels[0]] == ["gaussian"]
        assert widget._recipe.channels[1] == []

    def test_add_op_fills_registry_defaults(self, widget):
        widget.update_image_layer_list(["cad"])
        widget._add_op_from_menu(0, "gaussian")
        assert widget._recipe.channels[0][0].params == {"sigma": 1.5}

    def test_intensity_ops_are_independent_per_channel(self, widget):
        widget.update_image_layer_list(["cad", "nuc"])
        widget._add_op_from_menu(0, "gaussian")
        widget._add_op_from_menu(1, "clahe")
        assert [s.key for s in widget._recipe.channels[0]] == ["gaussian"]
        assert [s.key for s in widget._recipe.channels[1]] == ["clahe"]

    # -- inline param knob is editable + writes back to the step --

    def test_param_knob_is_inline_editable_and_updates_the_step(self, widget):
        from qtpy.QtWidgets import QAbstractSpinBox

        widget.update_image_layer_list(["cad"])
        widget._add_op_from_menu(0, "gaussian")
        col = _columns(widget)[0]
        spins = col.findChildren(QAbstractSpinBox)
        assert spins and not spins[0].isReadOnly()
        spins[0].setValue(3.0)
        assert widget._recipe.channels[0][0].params["sigma"] == 3.0

    def test_multi_param_op_renders_all_its_knobs(self, widget):
        from qtpy.QtWidgets import QAbstractSpinBox

        widget.update_image_layer_list(["cad"])
        widget._add_op_from_menu(0, "clahe")   # clip + kernel
        col = _columns(widget)[0]
        assert len(col.findChildren(QAbstractSpinBox)) >= 2

    # -- add geometry op → ONE shared step across ALL columns --

    def test_add_geometry_is_shared_across_all_columns(self, widget):
        widget.update_image_layer_list(["cad", "nuc", "third"])
        widget._add_op_from_menu(0, "isotropic")
        assert isinstance(widget._recipe.geometry, GeometryStep)
        # one blue geometry chip rendered inside EVERY column
        assert len(_geo_chips(widget)) == 3

    def test_geometry_re_pick_stays_a_single_shared_step(self, widget):
        widget.update_image_layer_list(["cad", "nuc"])
        widget._add_op_from_menu(0, "isotropic")
        widget._add_op_from_menu(1, "isotropic")   # picked again from column 1
        assert isinstance(widget._recipe.geometry, GeometryStep)
        assert len(_geo_chips(widget)) == 2         # still one per column

    # -- move geometry boundary → re-segments ALL channels identically --

    def test_move_geometry_boundary_resegments_all_channels(self, widget):
        widget.update_image_layer_list(["cad", "nuc"])
        for ci in (0, 1):
            widget._add_op_from_menu(ci, "gaussian")
            widget._add_op_from_menu(ci, "background")
        widget._add_op_from_menu(0, "isotropic")   # boundary defaults to end (2)
        widget._move_geometry(-1)                  # → boundary 1
        flat0 = [s.key for s in flatten_for_channel(widget._recipe, 0)]
        flat1 = [s.key for s in flatten_for_channel(widget._recipe, 1)]
        assert flat0 == flat1 == ["gaussian", "isotropic", "background"]

    # -- remove syncs across all columns --

    def test_remove_geometry_syncs_across_all_columns(self, widget):
        widget.update_image_layer_list(["cad", "nuc"])
        widget._add_op_from_menu(0, "isotropic")
        assert _geo_chips(widget)
        widget._remove_geometry()
        assert widget._recipe.geometry is None
        assert not _geo_chips(widget)

    def test_remove_intensity_step_rebuilds_without_stale_index(self, widget):
        widget.update_image_layer_list(["cad"])
        for key in ("gaussian", "clahe", "background"):
            widget._add_op_from_menu(0, key)
        widget._remove_intensity_step(0, 0)   # drop gaussian (index 0)
        assert [s.key for s in widget._recipe.channels[0]] == [
            "clahe", "background"]

    def test_reorder_intensity_step(self, widget):
        widget.update_image_layer_list(["cad"])
        for key in ("gaussian", "background"):
            widget._add_op_from_menu(0, key)
        widget._move_intensity_step(0, 1, -1)   # background earlier
        assert [s.key for s in widget._recipe.channels[0]] == [
            "background", "gaussian"]

    @pytest.mark.parametrize("i,delta", [(0, -1), (0, 1), (9, 1)])
    def test_move_out_of_range_is_a_no_op(self, widget, i, delta):
        widget.update_image_layer_list(["cad"])
        widget._add_op_from_menu(0, "gaussian")
        widget._move_intensity_step(0, i, delta)
        assert [s.key for s in widget._recipe.channels[0]] == ["gaussian"]

    def test_clear_channel_removes_all_its_steps(self, widget):
        widget.update_image_layer_list(["cad", "nuc"])
        widget._add_op_from_menu(0, "gaussian")
        widget._add_op_from_menu(1, "clahe")
        widget._clear_channel_steps(0)
        assert widget._recipe.channels[0] == []
        assert [s.key for s in widget._recipe.channels[1]] == ["clahe"]

    # -- the "+ Add step" menu: every implemented op, enabled, plain label --
    # (v1 no longer lists the declared-but-not-built ops at all — see
    #  TestV1HideList.test_the_add_step_menu_offers_only_ops_that_run.)

    def test_menu_offers_every_implemented_op_enabled(self, widget):
        widget.update_image_layer_list(["cad"])
        menu = widget._make_add_op_menu(0)
        by_text = {a.text(): a for a in menu.actions()
                   if not a.isSeparator() and a.text()}
        assert by_text["Gaussian"].isEnabled()
        assert by_text["CLAHE"].isEnabled()
        for op in REGISTRY:
            if op.implemented:
                assert any(op.label in t for t in by_text), op.label

    def test_menu_geometry_group_is_all_channels(self, widget):
        widget.update_image_layer_list(["cad"])
        menu = widget._make_add_op_menu(0)
        disabled_headings = [a.text() for a in menu.actions()
                             if not a.isSeparator() and not a.isEnabled()]
        assert any("Geometry" in h and "all channels" in h
                   for h in disabled_headings)

    def test_menu_is_styled_and_uses_disabled_headings(self):
        """The only QMenu in the repo, and `theme.py` has no QMenu rule — an
        unstyled menu renders in the OS light palette inside a dark dock.
        `addSection` is avoided because the Windows Aero style does not render
        section text, which would make the group headings vanish."""
        code = _code_only(_UI_PY)
        assert "_MENU_STYLE" in code
        assert "setStyleSheet" in code
        assert "addSection" not in code, (
            "addSection text does not render under the Windows style"
        )

    def test_menu_reads_the_real_registry_not_a_placeholder(self):
        """The menu builder points at the real `packages/epicycle_preprocessing`
        registry, not the retired `_OP_SUITE` placeholder."""
        code = _code_only(_UI_PY)
        assert "_OP_SUITE" not in code, "the _OP_SUITE placeholder is gone"
        assert "by_bucket" in code, "the menu must read registry.by_bucket()"

    def test_percentile_normalize_is_not_offered(self):
        """Hoists it as the implicit, non-removable first stage — it must not
        appear as an addable registry op."""
        labels = " ".join(op.label.lower() for op in REGISTRY)
        keys = " ".join(op.key.lower() for op in REGISTRY)
        assert "percentile" not in labels and "percentile" not in keys

    # -- the Fiji-default order hint --

    def test_fiji_order_hint(self, widget):
        widget.update_image_layer_list(["cad"])
        for key in ("gaussian", "background"):
            widget._add_op_from_menu(0, key)
        widget._add_op_from_menu(0, "isotropic")   # boundary → end (2)
        widget._move_geometry(-1)                  # gaussian · isotropic · background
        assert "Fiji" in widget.recipe_order_hint.text()
        widget._move_geometry(-1)                  # isotropic first → not Fiji
        assert widget.recipe_order_hint.text() == ""

    # -- "Process whole volume" is WIRED (Brick 1b-1) — no longer an inert stub --

    def test_process_button_routes_to_controller(self, widget):
        """"Process whole volume" is wired to the controller (Brick 1b-1).  With
        no source loaded, clicking routes to `on_process_whole_volume`, which
        surfaces a themed "load a source" status and starts NO worker — it no
        longer logs the old inert "PREPROC-STUB" recipe dump."""
        widget.update_image_layer_list(["cad"])
        widget._add_op_from_menu(0, "gaussian")
        widget._add_op_from_menu(0, "isotropic")
        widget._on_apply_clicked()
        assert "source" in widget.preprocess_status_label.text().lower()
        # Nothing to process without a source → no in-flight process thread.
        assert getattr(widget.controller, "_process_thread", None) is None
        assert getattr(widget.controller, "_processing", False) is False

    def test_preview_with_no_source_surfaces_status_not_crash(self, widget):
        """Preview no longer LOGS the recipe (Brick 2b computes) — with no source
        loaded it must surface a themed status and NOT raise / launch a thread."""
        widget.update_image_layer_list(["cad"])   # recipe columns, but no source
        widget._on_preview_clicked()               # delegates to the controller
        assert "source" in widget.preprocess_status_label.text().lower()
        # No in-flight preview thread was started (nothing to compute).
        assert getattr(widget.controller, "_preview_thread", None) is None

    # -- the recipe survives a JSON round-trip (persistence contract) --

    def test_recipe_json_round_trip_through_the_widget(self, widget):
        widget.update_image_layer_list(["cad", "nuc"])
        widget._add_op_from_menu(0, "gaussian")
        widget._add_op_from_menu(1, "clahe")
        widget._add_op_from_menu(0, "isotropic")
        payload = widget.get_recipe_json()
        assert payload is not None            # never None → upsert stores it
        # restoring the payload reproduces the recipe (geometry is a SEPARATE
        # shared step, not a per-channel op).
        widget.set_recipe_from_json(payload)
        assert [s.key for s in widget._recipe.channels[0]] == ["gaussian"]
        assert [s.key for s in widget._recipe.channels[1]] == ["clahe"]
        assert widget._recipe.geometry is not None

    def test_clearing_a_recipe_serializes_present_not_none(self, widget):
        """Hygiene: a cleared recipe must serialize to an empty-but-PRESENT
        dict (not None), so the upsert's skip-None can't leave a stale recipe."""
        widget.update_image_layer_list(["cad"])
        payload = widget.get_recipe_json()
        assert isinstance(payload, dict)
        assert "channels" in payload


class TestTwoColumnSplit:
    """'a 50/50 split of the whole section is wrong; the chain
    needs the wider half.'"""

    def test_section_is_two_columns(self, widget):
        from qtpy.QtWidgets import QSplitter

        assert widget.accordion._sections[1].findChildren(QSplitter), (
            "② is a 2-column panel"
        )

    def test_preprocess_column_is_the_wider_half(self):
        """Qt exposes no `stretchFactor` getter, and an un-shown splitter
        distributes by size hint — so neither `sizes()` nor a `resize()` can
        report the intent here.  Read the arguments actually passed, which are
        the mechanism that governs every real resize."""
        src = _read(_UI_PY)
        for node in ast.walk(ast.parse(src)):
            if not (isinstance(node, ast.FunctionDef)
                    and node.name == "_create_load_adjust_panel"):
                continue
            factors, sizes = {}, None
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                attr = getattr(call.func, "attr", None)
                if attr == "setStretchFactor" and len(call.args) == 2:
                    idx, val = call.args
                    if isinstance(idx, ast.Constant) and isinstance(val, ast.Constant):
                        factors[idx.value] = val.value
                elif attr == "setSizes" and call.args:
                    arg = call.args[0]
                    if isinstance(arg, ast.List):
                        sizes = [e.value for e in arg.elts
                                 if isinstance(e, ast.Constant)]
            assert factors.get(1, 0) > factors.get(0, 0), (
                f"the recipe column needs the wider half, got {factors} "
                "(a 50/50 split is not allowed)"
            )
            assert sizes and sizes[1] > sizes[0], (
                f"initial sizes must favour the recipe column, got {sizes}"
            )
            return
        pytest.fail("_create_load_adjust_panel not found")


class TestBoxListComponent:
    """③ and ④ share ONE Box List component.

    The previous attempt shipped two disjoint plain `QListWidget`s, and nothing
    populated ③'s — so the widget called "the linchpin" was permanently empty
    while real patches went to ④'s separate list.  Row grammar scored 1 of 12.
    """

    def test_both_sections_use_the_same_component(self, widget):
        from napari_epicycle.data_preparation.ui import BoxListPanel

        assert isinstance(widget.draw_box_list, BoxListPanel)
        assert isinstance(widget.library_box_list, BoxListPanel)
        assert widget.draw_box_list is not widget.library_box_list

    def test_library_list_is_the_frozen_extracted_roi_list(self, widget):
        """The shell calls `.model()`, `.currentItem()`, `.count()` and
        `.item(0).text()` on `extracted_roi_list`, so ④'s instance must BE it."""
        from qtpy.QtWidgets import QListWidget

        assert widget.extracted_roi_list is widget.library_box_list.list
        assert isinstance(widget.extracted_roi_list, QListWidget)

    def test_draw_list_is_box_list(self, widget):
        assert widget.box_list is widget.draw_box_list.list

    def test_row_text_is_preserved_for_the_controller(self, widget):
        """THE trap: seven controller sites and the shell key into
        `extracted_patches` off `item.text()`.  A `setItemWidget` row that
        leaves the text empty breaks plot, snap, Trim and export
        SILENTLY."""
        widget.extracted_roi_list.addItem("wound_d3_patch_1")
        item = widget.extracted_roi_list.item(0)
        assert widget.extracted_roi_list.itemWidget(item) is not None, (
            "the row should be an overlaid card"
        )
        assert item.text() == "wound_d3_patch_1", (
            "item text must survive setItemWidget — the controller keys off it"
        )

    def test_status_is_encoded_three_times(self, widget):
        """M4 colourblind-safety: colour, line style AND row text."""
        from napari_epicycle.data_preparation.ui import _BOX_STATUS

        for status, (colour, glyph, text) in _BOX_STATUS.items():
            assert colour.startswith("#") and glyph and text == status

    def test_row_has_no_status_badge(self, widget):
        """Status badges (swatch / line glyph / new-staged-captured text) were
        DROPPED 2026-08-18: ④ patches are always on disk, so the badge was noise
        and even flipped staged↔captured for the same patch across a reopen.  A
        row must render NONE of the retired status glyphs/text, but MUST keep the
        name label (seven controller sites key off `item.text()`)."""
        from qtpy.QtWidgets import QLabel

        widget.extracted_roi_list.addItem("p1")
        row = widget.extracted_roi_list.itemWidget(
            widget.extracted_roi_list.item(0))
        texts = [c.text() for c in row.findChildren(QLabel)]
        assert not any(t in ("─", "═", "┈") for t in texts), texts
        assert not any(t in ("new", "staged", "captured") for t in texts), texts
        assert "p1" in texts, texts

    def test_both_instances_share_one_legend(self, widget):
        assert (widget.draw_box_list.legend_label.text()
                == widget.library_box_list.legend_label.text())

    def test_count_is_live(self, widget):
        """④ has no tick column any more (F1 writes Extract straight to disk),
        so its count reports the TOTAL number of patches on disk — captured rows
        included."""
        assert widget.library_box_list.count_label.text() == "0"
        _stage(widget, "p1", "p2")
        assert widget.library_box_list.count_label.text() == "2 patch(es)"
        widget.extracted_roi_list.addItem("already_on_disk")   # captured
        widget.library_box_list.refresh()
        assert widget.library_box_list.count_label.text() == "3 patch(es)", (
            "④ counts every patch row, staged or captured"
        )

    def test_empty_state_toggles(self, widget):
        assert not widget.library_box_list.empty_label.isHidden()
        widget.extracted_roi_list.addItem("p1")
        assert widget.library_box_list.empty_label.isHidden()
        widget.extracted_roi_list.clear()
        assert not widget.library_box_list.empty_label.isHidden()

    def test_the_two_empty_states_say_different_things(self, widget):
        """③ tells you to draw; ④ tells you to extract."""
        assert (widget.draw_box_list.empty_label.text()
                != widget.library_box_list.empty_label.text())

    def test_select_all_ticks_every_row(self, widget):
        for i in range(3):
            item = QListWidgetItemFactory(f"p{i}")
            widget.box_list.addItem(item)
        widget.draw_box_list.select_all_check.setChecked(True)
        assert len(widget.draw_box_list.checked_names()) == 3
        widget.draw_box_list.select_all_check.setChecked(False)
        assert widget.draw_box_list.checked_names() == []

    def test_draw_list_names_match_the_extract_naming_rule(self, widget):
        """Rows preview the name the patch will actually get, using the same
        rule as `controller.extract_patches:643-650`."""
        widget.display_name_edit.setText("wound_d3")

        class _Layer:
            data = [object(), object()]

        widget._current_shapes_layer = lambda: _Layer()
        widget.refresh_draw_box_list()
        names = [widget.box_list.item(i).text()
                 for i in range(widget.box_list.count())]
        assert names == ["wound_d3_patch_1", "wound_d3_patch_2"], names

    def test_draw_list_falls_back_to_bare_patch(self, widget):
        widget.display_name_edit.setText("")

        class _Layer:
            data = [object()]

        widget._current_shapes_layer = lambda: _Layer()
        widget.refresh_draw_box_list()
        assert widget.box_list.item(0).text() == "patch_1"

    def test_shapes_layer_is_watched_only_once(self, widget):
        """Reconnecting on every draw-mode switch would fire the rebuild N
        times per edit — the handler-accumulation bug class."""
        calls = []

        class _Events:
            class data:
                @staticmethod
                def connect(cb):
                    calls.append(cb)

        class _Layer:
            events = _Events()
            data = []

        layer = _Layer()
        widget._watch_shapes_layer(layer)
        widget._watch_shapes_layer(layer)
        assert len(calls) == 1

    def test_retired_ready_label_is_gone(self, widget):
        """Retires the word; the frozen METHOD survives with a new body."""
        assert getattr(widget, "roi_queue_summary_label", None) is None
        assert callable(widget.set_roi_queue_summary)

    def test_queue_summary_feeds_the_dashboard(self, widget):
        widget.set_roi_queue_summary(4, 2)
        assert widget.summary_staged_label.text() == "4 staged"


class TestSessionSources:
    """① merges workspace + source browser into one panel."""

    @staticmethod
    def _add_source(widget, stem, display=None):
        """Mimic `browse_source_folder:162-168`: the item carries the stem in
        `UserRole + 1` and has NO text; the text arrives in a second pass."""
        item = QListWidgetItem()
        item.setData(Qt.UserRole, f"C:/data/{stem}.tif")
        item.setData(Qt.UserRole + 1, stem)
        widget.source_file_list.addItem(item)
        if display is not None:
            item.setText(display)
        return item

    def test_browser_lives_in_session_not_load(self, widget):
        """Merges the fragmented Workspace/Source-browser split."""
        session = widget.accordion._sections[0]
        for name in ("source_file_list", "display_name_edit",
                     "source_folder_path", "add_source_button"):
            w = getattr(widget, name)
            assert w in session.findChildren(type(w)), f"{name} is not in ①"

    def test_no_duplicate_browser_in_load_section(self, widget):
        load = widget.accordion._sections[1]
        assert widget.source_file_list not in load.findChildren(
            type(widget.source_file_list))

    def test_add_button_is_not_in_a_header_row(self, widget):
        """A header row carries only title + stretch + `?`."""
        from qtpy.QtWidgets import QPushButton

        assert isinstance(widget.add_source_button, QPushButton)
        siblings = widget.add_source_button.parentWidget().findChildren(QLabel)
        assert siblings, "the Add button should sit in the body, not a header"

    def test_rows_render_as_cards(self, widget):
        self._add_source(widget, "wound_d3")
        item = widget.source_file_list.item(0)
        assert widget.source_file_list.itemWidget(item) is not None

    def test_card_reads_the_stem_not_the_not_yet_written_text(self, widget):
        from qtpy.QtWidgets import QLabel as _QLabel

        self._add_source(widget, "wound_d3")
        card = widget.source_file_list.itemWidget(
            widget.source_file_list.item(0))
        texts = [c.text() for c in card.findChildren(_QLabel)]
        assert "wound_d3" in texts, texts

    def test_card_updates_when_the_controller_fills_the_text(self, widget):
        """`_refresh_source_file_list` writes the text in a SECOND pass, so the
        card must re-render on `dataChanged` or it shows blank details."""
        from qtpy.QtWidgets import QLabel as _QLabel

        item = self._add_source(widget, "wound_d3")
        # The real format from `_refresh_source_file_list:465` — a LEADING
        # space, then a newline before the shape/badge line.  Splitting on " "
        # gave "" and made the whole string the detail, so the alias rendered
        # twice on one card.
        item.setText(" wound_d3\n   64×512×512   • 4 patch(es)")
        card = widget.source_file_list.itemWidget(item)
        texts = [c.text() for c in card.findChildren(_QLabel)]
        assert any("4 patch(es)" in t for t in texts), texts
        assert texts.count("wound_d3") == 1, f"alias rendered twice: {texts}"

    def test_moved_source_is_flagged(self, widget):
        """The path is the immutable key — if it is gone, say so."""
        from qtpy.QtWidgets import QLabel as _QLabel

        self._add_source(widget, "gone")
        card = widget.source_file_list.itemWidget(
            widget.source_file_list.item(0))
        texts = " ".join(c.text() for c in card.findChildren(_QLabel))
        assert "moved" in texts.lower(), texts

    def test_empty_state(self, widget):
        assert not widget.source_empty_label.isHidden()
        self._add_source(widget, "a")
        assert widget.source_empty_label.isHidden()

    def test_alias_hint_matches_the_naming_rule(self, widget):
        """Mirrors `controller.extract_patches:643-650`, both branches."""
        widget.display_name_edit.setText("wound_d3")
        assert "wound_d3_patch_1" in widget.alias_hint_label.text()
        widget.display_name_edit.setText("   ")
        text = widget.alias_hint_label.text()
        assert "patch_1" in text and "_patch_1" not in text

    def test_channels_are_read_only_here(self, widget):
        """① shows channels read-only and points at ③, which owns
        identity — ① cannot own them because the count is only knowable after
        the load that happens in ②."""
        assert "Draw & Extract" in widget.session_channels_label.text()


class TestSourcePickerProxy:
    """②'s combo is the single editor of "which source"."""

    @staticmethod
    def _add(widget, stem):
        item = QListWidgetItem()
        item.setData(Qt.UserRole, f"C:/data/{stem}.tif")
        item.setData(Qt.UserRole + 1, stem)
        widget.source_file_list.addItem(item)
        return item

    def test_combo_mirrors_the_list(self, widget):
        self._add(widget, "a")
        self._add(widget, "b")
        assert [widget.source_combo.itemText(i) for i in range(2)] == ["a", "b"]

    def test_combo_reads_stem_not_empty_text(self, widget):
        """Items are inserted with no text; reading `item.text()` on
        `rowsInserted` would give blank entries."""
        self._add(widget, "a")
        assert widget.source_combo.itemText(0) == "a"

    def test_combo_repopulates_after_clear(self, widget):
        self._add(widget, "old")
        widget.source_file_list.clear()
        assert widget.source_combo.count() == 0

    def test_combo_pick_drives_the_row_load_source_reads(self, widget):
        """`load_source:188` reads `currentItem()` — the combo must write the
        row, or ② can show one source and load another."""
        self._add(widget, "a")
        self._add(widget, "b")
        widget._on_source_combo_picked(1)
        assert widget.source_file_list.currentRow() == 1

    def test_round_trip_terminates(self, widget):
        self._add(widget, "a")
        self._add(widget, "b")
        widget.source_file_list.setCurrentRow(1)
        widget._on_source_combo_picked(0)
        assert widget._source_sync_active is False


class TestChannelPicker:
    """③ owns channel identity, include and order."""

    def test_one_row_per_channel(self, widget):
        widget.update_image_layer_list(["cad", "nuc"])
        assert widget.image_layer_combo.count() == 2
        for i in range(2):
            item = widget.image_layer_combo.item(i)
            assert widget.image_layer_combo.itemWidget(item) is not None

    def test_row_carries_swatch_name_and_index(self, widget):
        from qtpy.QtWidgets import QLabel

        widget.update_image_layer_list(["cad"])
        row = widget.image_layer_combo.itemWidget(
            widget.image_layer_combo.item(0))
        texts = [c.text() for c in row.findChildren(QLabel)]
        assert "cad" in texts and "ch0" in texts, texts

    def test_empty_state_when_nothing_loaded(self, widget):
        assert not widget.channel_empty_label.isHidden()
        widget.update_image_layer_list(["cad"])
        assert widget.channel_empty_label.isHidden()

    def test_order_readout_matches_the_checked_channels(self, widget):
        """The readout is derived from `checked_channel_names()` — the SAME
        inclusion+order the controller extracts (`_selected_channel_names` /
        `get_combined_image_stack`)."""
        widget.update_image_layer_list(["cad", "nuc"])
        expected = widget.checked_channel_names()
        assert expected == ["cad", "nuc"]      # all included by default
        shown = widget.channel_order_label.text().split(":", 1)[1]
        assert [n.strip() for n in shown.split(",")] == expected

    def test_selection_order_follows_row_order_not_click_order(self, widget):
        """`_normalize_channel_selection` (retained, decorative) keeps the row
        SELECTION in row order.  Inclusion no longer rides on selection, but the
        visual-focus normalisation stays."""
        widget.update_image_layer_list(["cad", "nuc"])
        widget.image_layer_combo.item(1).setSelected(True)   # nuc first
        widget.image_layer_combo.item(0).setSelected(True)   # then cad
        assert [i.text() for i in widget.image_layer_combo.selectedItems()] == [
            "cad", "nuc"], "row order must win over click order"

    def test_moving_a_row_changes_the_extraction_order(self, widget):
        """The ↑↓ buttons must change the DATA order — the order
        `checked_channel_names()` returns, not just the display."""
        widget.update_image_layer_list(["cad", "nuc"])
        assert widget.checked_channel_names() == ["cad", "nuc"]
        widget._move_channel(1, -1)
        assert widget.checked_channel_names() == ["nuc", "cad"], (
            "reordering rows must reorder the stack")

    @pytest.mark.parametrize("index,delta", [(0, -1), (1, 1), (5, 1)])
    def test_move_out_of_range_is_a_no_op(self, widget, index, delta):
        widget.update_image_layer_list(["cad", "nuc"])
        widget._move_channel(index, delta)
        assert [widget.image_layer_combo.item(i).text() for i in range(2)] == [
            "cad", "nuc"]

    def test_channel_pairs_is_one_ordered_list(self, widget):
        """Rule 4: `channel_names` and the stacked array must come from
        ONE ordered pair-list, or they drift.  Reads the INCLUDED channels in
        row order (`checked_channel_names()`)."""
        widget.update_image_layer_list(["cad", "nuc"])
        pairs = widget.channel_pairs()
        assert [p[0] for p in pairs] == ["cad", "nuc"]
        assert [p[0] for p in pairs] == widget.checked_channel_names()

    def test_stale_click_order_claim_is_gone(self, widget):
        assert "order matters" not in widget.image_layer_label.text().lower()


class TestChannelIncludeCheckboxes:
    """A1 — real per-row include checkboxes (all checked by default), the
    name-keyed `_channel_included` dict as the ONE inclusion source, and the
    Select all / Deselect all bulk toggles.  `checked_channel_names()` is the
    single inclusion+order reader every extract path now flows through."""

    def test_all_included_on_fresh_load_in_row_order(self, widget):
        widget.update_image_layer_list(["cad", "nuc", "mem"])
        assert widget.checked_channel_names() == ["cad", "nuc", "mem"]

    def test_row_has_a_real_checkbox_checked_by_default(self, widget):
        widget.update_image_layer_list(["cad"])
        box = _channel_checkbox(widget, 0)
        assert box is not None and box.isChecked()

    def test_unticking_excludes_that_channel(self, widget):
        widget.update_image_layer_list(["cad", "nuc"])
        _channel_checkbox(widget, 0).setChecked(False)       # untick cad
        assert widget.checked_channel_names() == ["nuc"]

    def test_untick_persists_across_a_rebuild_with_the_same_names(self, widget):
        """The dict persists the untick; a rebuild with the SAME names must NOT
        silently re-check it."""
        widget.update_image_layer_list(["cad", "nuc"])
        _channel_checkbox(widget, 0).setChecked(False)       # untick cad
        widget.update_image_layer_list(["cad", "nuc"])       # simulated rebuild
        assert widget.checked_channel_names() == ["nuc"]
        assert _channel_checkbox(widget, 0).isChecked() is False

    def test_new_name_defaults_included_and_removed_name_pruned(self, widget):
        widget.update_image_layer_list(["cad", "nuc"])
        _channel_checkbox(widget, 1).setChecked(False)       # untick nuc
        # nuc leaves, mem arrives on the next detected set
        widget.update_image_layer_list(["cad", "mem"])
        assert widget.checked_channel_names() == ["cad", "mem"]  # mem defaults in
        assert "nuc" not in widget._channel_included             # pruned

    def test_reorder_preserves_checked_state_and_changes_order(self, widget):
        widget.update_image_layer_list(["cad", "nuc"])
        _channel_checkbox(widget, 0).setChecked(False)       # exclude cad
        widget._move_channel(1, -1)                           # nuc → row 0
        assert widget.checked_channel_names() == ["nuc"]      # cad still excluded
        assert [widget.image_layer_combo.item(i).text() for i in range(2)] == [
            "nuc", "cad"]
        assert _channel_checkbox(widget, 0).isChecked() is True   # nuc row
        assert _channel_checkbox(widget, 1).isChecked() is False  # cad row

    def test_select_all_and_deselect_all_set_the_whole_set(self, widget):
        widget.update_image_layer_list(["cad", "nuc", "mem"])
        widget._deselect_all_channels()
        assert widget.checked_channel_names() == []
        assert all(not _channel_checkbox(widget, i).isChecked()
                   for i in range(3))
        widget._select_all_channels()
        assert widget.checked_channel_names() == ["cad", "nuc", "mem"]
        assert all(_channel_checkbox(widget, i).isChecked() for i in range(3))

    def test_readout_prompts_only_when_the_included_set_is_empty(self, widget):
        widget.update_image_layer_list(["cad", "nuc"])
        assert "no channels" not in widget.channel_order_label.text().lower()
        widget._deselect_all_channels()
        assert "no channels" in widget.channel_order_label.text().lower()

    def test_select_all_buttons_are_wired(self, widget):
        assert _is_connected(widget.channel_select_all_button)
        assert _is_connected(widget.channel_deselect_all_button)


class TestStepExitPreservesInclusion:
    """H1 (audit) — step-exit removes the source layers UNDER the
    `_swapping_source_display` guard, so `_on_layers_changed` cannot reconcile
    `_channel_included` down to {} mid-removal → a deliberate channel untick
    silently reverting to included on step re-enter (Extract-a-channel-you-
    excluded, silent wrong-data)."""

    def _fake_layers(self, ctl=None, probe=None):
        class _FakeLayers:
            def __contains__(self, n):
                return True
            def __getitem__(self, n):
                return type("L", (), {"visible": True})()
            def remove(self, layer):
                if probe is not None:
                    probe.append(ctl._swapping_source_display)
        return _FakeLayers()

    def test_removal_runs_under_the_swap_guard(self):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController,
        )
        ctl = DataPreparationController.__new__(DataPreparationController)
        ctl.widget = None
        ctl._loaded_source_layers = ["s_ch0", "s_ch1"]
        ctl._swapping_source_display = False
        probe = []
        ctl.viewer = type("V", (), {"layers": self._fake_layers(ctl, probe)})()
        ctl._remove_source_layers_from_viewer()
        assert probe == [True, True]                   # guard set during each removal
        assert ctl._swapping_source_display is False   # restored (nested-safe)

    def test_nested_guard_preserved_when_caller_holds_it(self):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController,
        )
        ctl = DataPreparationController.__new__(DataPreparationController)
        ctl.widget = None
        ctl._loaded_source_layers = ["a"]
        ctl._swapping_source_display = True            # e.g. the processed swap holds it
        ctl.viewer = type("V", (), {"layers": self._fake_layers()})()
        ctl._remove_source_layers_from_viewer()
        assert ctl._swapping_source_display is True    # outer guard NOT cleared


class TestReviewLibrary:
    """④'s order, and the export roles."""

    def test_no_visible_export_control(self, widget):
        """F1: ③ Extract now writes each patch to disk immediately, so ④ has NO
        export control any more — the old bulk "Export checked → Library" commit
        is hidden (retained-but-retired, see below).  It previously had ONE
        visible control scoped by the Box List tick column."""
        from qtpy.QtWidgets import QPushButton

        review = widget.accordion._sections[3]
        visible = [b.text() for b in review.findChildren(QPushButton)
                   if not b.isHidden() and "export" in b.text().lower()]
        assert visible == [], visible

    def test_bulk_commit_name_retained_but_hidden(self, widget):
        """F1 retires ④'s export button but KEEPS the `save_reviewed_roi` name:
        `workflow_shell.py:1418` logs "ROI export triggered" on it and
        `_set_step1_busy`'s literal list references it, so the widget must stay
        alive — just hidden (③ Extract now performs the write)."""
        assert widget.save_reviewed_roi.text() == "Export checked → Library"
        assert widget.save_reviewed_roi.isHidden()

    def test_retired_button_is_hidden_but_alive(self, widget):
        """`export_all_ready_roi` stays for `_set_step1_busy`'s literal list and
        the `hasattr` connect at `controller.py:88`."""
        assert widget.export_all_ready_roi is not None
        assert widget.export_all_ready_roi.isHidden()

    def test_no_per_patch_export_button(self, widget):
        assert getattr(widget, "export_this_button", None) is None
        assert widget.discard_patch_button.isHidden(), (
            "delete is a ROW action, not a detail-pane button beside a commit"
        )

    @staticmethod
    def _stage(widget, *names):
        """Make rows read as `staged` — i.e. in memory, not yet on disk.

        ④'s status comes from `controller.extracted_patches`; a row absent from
        it is `captured` (already exported), which v4.1 makes untickable.
        """
        widget.controller.extracted_patches = {
            n: {"data": None} for n in names}
        for n in names:
            widget.extracted_roi_list.addItem(n)
        widget.library_box_list.refresh()

    # (removed) The ④ "export scope is the tick column" tests
    # (test_export_scope_is_the_tick_column / test_unticking_everything_means_nothing
    # / test_hidden_rows_are_out_of_scope): `checked_patch_names` + the per-row tick
    # drove the retired "Export checked → Library" flow, which F1 killed (Extract
    # writes to disk immediately) and this cleanup removed from the UI.  ④ rows are
    # no longer checkable — see test_library_rows_are_not_checkable.

    def test_library_rows_status_distinct_but_untickable(self, widget):
        """④ has NO tick UI (F1 writes Extract to disk): NO ④ row is checkable,
        and none renders a native check indicator.  The staged-vs-captured
        CONCEPT still survives via `_status_of` (reading `extracted_patches`),
        which is what drives the intensity-plot memory-vs-disk path — NOT the
        item flags.  ④'s count reports the total patch count."""
        self._stage(widget, "staged_1")
        widget.extracted_roi_list.addItem("captured_1")   # absent → captured
        widget.library_box_list.refresh()
        bl = widget.library_box_list
        # The status concept survives (drives the plot path)…
        assert bl._status_of("staged_1") == "staged"
        assert bl._status_of("captured_1") == "captured"
        # …but NEITHER row is user-checkable, and no native indicator renders.
        for i in (0, 1):
            item = widget.extracted_roi_list.item(i)
            assert not (item.flags() & Qt.ItemIsUserCheckable)
            assert item.data(Qt.CheckStateRole) is None
        assert "2 patch(es)" in bl.count_label.text()

    def test_row_actions_are_panel_scoped(self, widget):
        """2026-08 rework: ④'s per-patch cluster is location-eye + capture +
        delete (every patch shares it now the staged/captured badge is gone);
        ③'s box actions (Duplicate etc.) never appear in ④.  One component, two
        panels, so the cluster is still panel-scoped."""
        from qtpy.QtWidgets import QPushButton

        self._stage(widget, "p1")
        widget.extracted_roi_list.addItem("cap")
        widget.library_box_list.refresh()

        def tips(panel, name):
            for i in range(panel.list.count()):
                it = panel.list.item(i)
                if it.text() == name:
                    row = panel.list.itemWidget(it)
                    return [b.toolTip() for b in row.findChildren(QPushButton)]
            return []

        p1_tips = tips(widget.library_box_list, "p1")
        assert any("sits in the source" in t for t in p1_tips), p1_tips
        assert any("capture view" in t for t in p1_tips), p1_tips
        assert any("Delete this patch" in t for t in p1_tips), p1_tips
        assert not any("Duplicate" in t for t in p1_tips), (
            "④ acts on patches — Duplicate is a ③ box action"
        )

    def test_sort_combo_offers_only_orders_the_data_supports(self, widget):
        """"newest first" was unimplementable (nothing stores a date) AND the
        two name orders were inverted: `descending = index == 1` against
        `["newest first", "name A–Z", "name Z–A"]` put A–Z on the descending
        branch."""
        items = [widget.patch_sort_combo.itemText(i)
                 for i in range(widget.patch_sort_combo.count())]
        assert items == ["name A–Z", "name Z–A"], items
        self._stage(widget, "p_1", "p_2", "p_9")
        widget.patch_sort_combo.setCurrentIndex(1)          # Z–A
        lst = widget.extracted_roi_list
        assert [lst.item(i).text() for i in range(lst.count())] == \
            ["p_9", "p_2", "p_1"]
        widget.patch_sort_combo.setCurrentIndex(0)          # A–Z
        assert [lst.item(i).text() for i in range(lst.count())] == \
            ["p_1", "p_2", "p_9"]

    def test_library_rows_are_not_checkable(self, widget):
        """④ rows carry NO tick (F1 writes Extract to disk): `make_checkable`
        strips ItemIsUserCheckable AND the CheckStateRole so no native indicator
        renders — the row card would otherwise sit ON TOP of an unclickable box
        (the bug this fixes)."""
        self._stage(widget, "p1")
        item = widget.extracted_roi_list.item(0)
        assert not (item.flags() & Qt.ItemIsUserCheckable)
        assert item.data(Qt.CheckStateRole) is None

    def test_retired_ready_vocabulary_is_gone(self, widget):
        """Retires `ready` — check only what the user can read."""
        from qtpy.QtWidgets import QPushButton

        review = widget.accordion._sections[3]
        for b in review.findChildren(QPushButton):
            if not b.isHidden():
                assert "ready" not in b.text().lower(), b.text()

    def test_panel_feedback_lives_in_the_detail_column(self):
        """The ④ action feedback — the result text (green/red) above a slim
        progress bar — now lives at the BOTTOM of the RIGHT (detail) column, not
        as a full-width strip below the splitter (which left an orphaned separator
        + status line).  Asserted on construction order + the `det` parent."""
        src = _read(_UI_PY)
        body_start = src.find("def _create_review_library_panel")
        body = src[body_start:src.find("def _filter_patch_list")]
        order = [
            body.find("self.plot_area = QWidget()"),
            body.find("self.export_all_ready_roi = QPushButton"),
            body.find("self.export_status_label = QLabel"),
            body.find("self.export_progress_bar = QProgressBar"),
        ]
        assert -1 not in order, order
        assert order == sorted(order), (
            "④ feedback: status text is constructed above the slim progress bar"
        )
        # Both go into the right (detail) column now — not the full-width outer.
        assert "det.addWidget(self.export_status_label)" in body
        assert "det.addWidget(self.export_progress_bar)" in body

    def test_search_filters_a_populated_list(self, widget):
        """'s 'live, empty and honest' licence covers widgets with NO
        backend.  ④'s list is populated today, so an inert search box is a
        broken control, not an honest empty one."""
        for n in ("wound_d3_patch_1", "wound_d5_patch_2"):
            widget.extracted_roi_list.addItem(n)
        widget.patch_search_edit.setText("d5")
        hidden = [widget.extracted_roi_list.item(i).isHidden()
                  for i in range(2)]
        assert hidden == [True, False], hidden
        widget.patch_search_edit.setText("")
        assert not any(widget.extracted_roi_list.item(i).isHidden()
                       for i in range(2))

    def test_sort_reorders(self, widget):
        """Each label must produce the order it NAMES.

        This test used to certify the bug: it set index 1 and asserted
        `b_patch_1` first — i.e. that the item reading "name A–Z" sorted
        DESCENDING — and its own comment showed the author knew index 0
        ("newest first") never fired at all.  A green test locking in an
        inverted control.  The combo now offers only the two orders the data
        supports, and the assertion follows the label.
        """
        for n in ("b_patch_1", "a_patch_1"):
            widget.extracted_roi_list.addItem(n)
        # Index 1 first: the combo starts at 0, so setting it to 0 again emits
        # nothing and would never exercise the handler.
        widget.patch_sort_combo.setCurrentIndex(1)               # name Z–A
        assert widget.extracted_roi_list.item(0).text() == "b_patch_1"
        widget.patch_sort_combo.setCurrentIndex(0)               # name A–Z
        assert widget.extracted_roi_list.item(0).text() == "a_patch_1"

    def test_ztrunc_spins_write_the_range_set_range_commits(self, widget):
        """The enabled-dead-editor bug: `_on_set_zrange_clicked` commits
        `controller.selected_z_range`, so typing must write it or Trim
        stores whatever the span last held."""
        widget.ztrunc_start_spin.setValue(12)
        widget.ztrunc_end_spin.setValue(62)
        assert widget.controller.selected_z_range == (12, 62)

    def test_ztrunc_rejects_an_inverted_range(self, widget):
        widget.ztrunc_start_spin.setValue(50)
        widget.ztrunc_end_spin.setValue(60)
        widget.controller.selected_z_range = None
        widget.ztrunc_start_spin.setValue(90)   # start beyond end
        assert widget.controller.selected_z_range is None

    def test_details_block_is_written_on_selection(self, widget):
        """`patch_details_label` was never written by anything before.

        The status note is the CURRENT vocabulary "saved on disk" — the old
        "captured"/"locked" framing was retired 2026-08-18 (every ④ patch is on
        disk after Extract; see `_refresh_patch_details`).  This assertion was
        stale (failing on the branch tip before Brick 1) and is retargeted at the
        note the code actually writes."""
        assert "select a patch" in widget.patch_details_label.text()
        widget.extracted_roi_list.addItem("p1")
        widget.extracted_roi_list.setCurrentRow(0)
        assert "select a patch" not in widget.patch_details_label.text()
        assert "saved on disk" in widget.patch_details_label.text().lower()

    def test_export_progress_bar_is_styled_and_placed(self, widget):
        from napari_epicycle.ui_shell.theme import PROGRESS_BAR_STYLE_TEAL

        assert widget.export_progress_bar.styleSheet() == PROGRESS_BAR_STYLE_TEAL
        review = widget.accordion._sections[3]
        assert widget.export_progress_bar in review.findChildren(
            type(widget.export_progress_bar))

    def test_box_appearance_section_is_axed(self, widget):
        """Brick 1 removes the whole "Box appearance" off-ramp: the global
        colour / thickness / opacity controls, the hidden provenance-figure
        container, and its widget lists must no longer exist on the widget."""
        for name in ("box_style_color_combo", "box_thickness_slider",
                     "box_opacity_slider", "box_style_line_combo",
                     "style_scope_combo", "box_show_label_check",
                     "box_show_cuboid_check", "box_preview_style_check",
                     "provenance_button",
                     "_box_style_figure_container", "_box_style_live_widgets",
                     "_box_style_figure_widgets", "_box_style_widgets"):
            assert not hasattr(widget, name), (
                f"{name} should be removed by the ④ P2 Brick-1 axing"
            )

    def test_no_double_iconography(self, widget):
        """A text glyph plus `apply_button_icon` renders two arrows."""
        for btn in (widget.export_all_ready_roi, widget.save_reviewed_roi,
                    widget.extract_roi):
            assert not re.match(r"^[^\w\s]", btn.text().strip()), (
                f"{btn.text()!r} leads with a glyph and also carries an icon"
            )


def QListWidgetItemFactory(text):
    """A checkable list item, as the Box List's own rows are."""
    from qtpy.QtCore import Qt as _Qt
    from qtpy.QtWidgets import QListWidgetItem as _Item

    item = _Item(text)
    item.setFlags(item.flags() | _Qt.ItemIsUserCheckable)
    item.setCheckState(_Qt.Unchecked)
    return item


class TestAuditFixes:
    """Regressions for defects the five-agent audit found in this rebuild.

    Each of these shipped green under the previous suite, which is why they are
    written as behaviour rather than existence.
    """

    def test_load_source_uses_the_busy_contract(self):
        """'s shared-bar exception is granted on ONE condition — the bar names
        what is running.  `_busy_show` was defined, tested directly, and called
        by NOTHING; `load_source` drove the bar raw, so it still spun
        anonymous.  A helper with only test callers is not shipped."""
        code = _code_only(_CONTROLLER_PY)
        assert "_busy_show" in code, "load_source must go through _busy_show"
        assert "pbar.setRange" not in code, (
            "the bar must not be driven around its own contract"
        )

    def test_busy_helpers_have_production_callers(self):
        for helper in ("_busy_show", "_busy_hide"):
            assert helper in _code_only(_CONTROLLER_PY), helper

    def test_lock_list_covers_what_section_5_enumerates(self, widget):
        """Names the source combo, the order combo, the voxel fields, Apply,
        and ③'s draw + Extract.  The list had 5 names and omitted 8 of them."""
        names = TestBusyLockList._literals()
        for required in ("source_combo", "axis_order_combo", "voxel_z_spin",
                         "voxel_y_spin", "voxel_x_spin",
                         "apply_preprocess_button", "free_bbox_button",
                         "fixed_bbox_button", "extract_roi",
                         "save_reviewed_roi", "export_all_ready_roi"):
            assert required in names, f"{required} must lock"

    def test_load_source_locks_the_surface(self):
        code = _code_only(_CONTROLLER_PY)
        i = code.find("def load_source")
        body = code[i:code.find("def ", i + 10)]
        assert "_set_step1_busy" in body, (
            "load_source locked nothing — the whole surface stayed live"
        )

    def test_export_all_drives_its_bar(self):
        # Raw source, not `_code_only`: these three names reach the widget
        # through `getattr(..., "name")`, so they live in STRING LITERALS by
        # nature and `_code_only` correctly blanks them.
        #
        # Brick 5 fix 3 moved the loop into the shared `_run_patch_export_loop`
        # (the export-all copy set the bar but never pumped Qt, so it rendered
        # at rest for the whole run), so assert on the helper AND on export-all
        # delegating to it — the previous slice would now pass vacuously.
        code = _read(_CONTROLLER_PY)
        i = code.find("def _run_patch_export_loop")
        assert i != -1, "the shared export loop is gone"
        body = code[i:code.find("def _export_named_patches", i)]
        assert "export_progress_bar" in body, (
            "④'s bar existed and nothing ever touched it"
        )
        assert "QApplication.processEvents()" in body, (
            "a determinate bar that never repaints is decorative"
        )
        j = code.find("def _on_export_all_ready_clicked")
        all_body = code[j:code.find("def _persist_patch_to_disk", j)]
        assert "_run_patch_export_loop" in all_body, (
            "export-all must go through the one pumped loop"
        )

    def test_progressive_default_open_has_a_production_caller(self):
        """Built, tested, and never called — `on_step_enter` only offered the
        migration.  A three-part mechanism that can never run."""
        assert "apply_progressive_default_open" in _read(_CONTROLLER_PY)

    def test_extract_honours_the_include_ticks(self):
        """Specifies "Extract checked".  `checked_names()` had no
        production caller, so unticking a box excluded nothing and the tick
        column was decorative."""
        assert "checked_box_indices" in _read(_CONTROLLER_PY)

    def test_combo_clears_when_the_list_has_no_current_row(self, widget):
        """MEASURED: QComboBox auto-selects index 0 on populate while the list's
        currentRow is -1, so ② named a source `load_source` then reported as
        "No file selected"."""
        item = QListWidgetItem()
        item.setData(Qt.UserRole, "C:/data/a.tif")
        item.setData(Qt.UserRole + 1, "a")
        widget.source_file_list.addItem(item)
        assert widget.source_file_list.currentRow() == -1
        assert widget.source_combo.currentIndex() == -1, (
            "the combo must not name a source the list has not selected"
        )

    def test_switching_source_updates_an_auto_alias(self, widget):
        """The alias drives the patch prefix.  Switching A→B while
        the alias still read "A" exported B's patches as `A_patch_N` with
        `source_display_name="A"` against B's path."""
        for stem in ("alpha", "beta"):
            it = QListWidgetItem()
            it.setData(Qt.UserRole, f"C:/data/{stem}.tif")
            it.setData(Qt.UserRole + 1, stem)
            widget.source_file_list.addItem(it)
        widget.display_name_edit.setText("alpha")
        widget._auto_alias = "alpha"
        widget._on_source_combo_picked(1)
        assert widget.display_name_edit.text() == "beta"

    def test_switching_source_shows_the_new_sources_alias(self, widget):
        """Brick-2 FIX 1 supersedes the old PARTIAL "hand-typed alias is sticky"
        behavior: switching now restores the TARGET source's own alias on EVERY
        route (mouse / keyboard / combo), FULLY closing the A-on-B provenance gap
        the old rule left open for hand-typed names (a hand-typed alias for the
        prior source is saved to ITS registry row via `editingFinished` and
        restored when you switch back — not carried onto the new source).  Here
        there is no workspace/registry, so switching to `beta` shows `beta`."""
        for stem in ("alpha", "beta"):
            it = QListWidgetItem()
            it.setData(Qt.UserRole, f"C:/data/{stem}.tif")
            it.setData(Qt.UserRole + 1, stem)
            widget.source_file_list.addItem(it)
        widget.display_name_edit.setText("my_own_name")
        widget._auto_alias = "alpha"
        widget._on_source_combo_picked(1)
        assert widget.display_name_edit.text() == "beta"

    def test_no_duplicate_browse_affordance(self, widget):
        """`+ Add` and `Browse` were wired to the same slot, six lines apart."""
        assert widget.source_browse_button.isHidden()
        assert not widget.add_source_button.isHidden()

    def test_unbacked_controls_are_disabled_with_reasons(self, widget):
        """Brick 1b: Preview + Apply are inert stubs — disabled WITH a stated
        reason until channels are loaded (honest, not enabled-and-inert)."""
        for name in ("test_region_button", "apply_preprocess_button"):
            w = getattr(widget, name)
            assert not w.isEnabled(), f"{name} is enabled with no channels"
            assert w.toolTip(), f"{name} gives no reason"

    def test_no_conditional_promises_that_can_expire(self, widget):
        """A label saying "…once a source is loaded" becomes a stale claim the
        moment one is.  "Honest" means unconditional scope."""
        from qtpy.QtWidgets import QLabel

        load = widget.accordion._sections[1]
        for c in load.findChildren(QLabel):
            assert "once a source is loaded" not in c.text(), c.text()

    def test_voxel_description_does_not_promise_persistence(self, widget):
        """`persist_patch` has no `voxel_size` parameter; the field is dormant
        with no caller."""
        from qtpy.QtWidgets import QLabel

        load = widget.accordion._sections[1]
        texts = " ".join(c.text() for c in load.findChildren(QLabel))
        assert "Recorded with each patch and never" not in texts

    def test_axis_combo_offers_only_states_the_loader_has(self, widget):
        """`transpose_button` is a boolean, so a third option resolves
        identically to one of the two and promises a mode that does not
        exist."""
        options = [widget.axis_order_combo.itemText(i)
                   for i in range(widget.axis_order_combo.count())]
        assert options == ["CZYX", "ZCYX"], options


class TestAuditRound2:
    """Regressions for defects the five-agent audit found, including two that
    my own earlier fixes introduced."""

    def test_load_source_clears_the_lock_on_every_async_path(self):
        """A raise in transpose/split/add_image left `_step1_busy` True forever
        — and since the lock now covers 20 widgets including the Load button
        and the source picker, the user could not even retry. A cosmetic bar
        defect became an unrecoverable deadlock.

        The disk read now runs on a QThread, so the old synchronous
        `try/finally` terminal funnel in `load_source` is gone.  The "lock
        always clears" guarantee is RELOCATED to (a) `load_source`'s launch
        guard (clears if worker/thread construction raises), (b)
        `_on_source_loaded`'s `finally` (clears if the display continuation
        raises), and (c) the `_on_source_load_error` / `cancel_source_load`
        paths.  Every one must clear `_step1_busy`."""
        code = _code_only(_CONTROLLER_PY)

        # (a) launch guard: try/except around the worker launch that clears.
        i = code.find("def load_source")
        load_body = code[i:code.find("def _preflight_source_route", i)]
        assert "try:" in load_body and "except" in load_body, (
            "load_source must guard the worker launch"
        )
        assert "_set_step1_busy(False)" in load_body, (
            "a failed launch must clear the busy lock"
        )

        # (b) the main-thread continuation clears in a finally (display raise).
        loaded_body = code[
            code.find("def _on_source_loaded"):
            code.find("def _on_source_load_error")]
        assert "finally:" in loaded_body and (
            "_set_step1_busy(False)" in loaded_body), (
            "_on_source_loaded must clear the lock even if the display raises"
        )

        # (c) the error + cancel paths clear it too.
        for meth in ("def _on_source_load_error", "def cancel_source_load"):
            k = code.find(meth)
            body = code[k:code.find("\n    def ", k + 1)]
            assert "_set_step1_busy(False)" in body, (
                f"{meth} must clear the busy lock"
            )

    def test_busy_unlock_restores_rather_than_forces(self, widget):
        """`setEnabled(True)` on unlock switched on every deliberately-disabled
        control. After one busy cycle, Apply — which has no handler — became
        clickable and inert, the exact state forbids."""
        assert not widget.apply_preprocess_button.isEnabled()
        widget.controller._set_step1_busy(True)
        widget.controller._set_step1_busy(False)
        assert not widget.apply_preprocess_button.isEnabled(), (
            "a busy cycle must not enable a control that was disabled"
        )

    def test_busy_cycle_restores_normally_enabled_controls(self, widget):
        assert widget.load_source_button.isEnabled()
        widget.controller._set_step1_busy(True)
        assert not widget.load_source_button.isEnabled()
        widget.controller._set_step1_busy(False)
        assert widget.load_source_button.isEnabled()

    def test_tick_filter_bails_out_when_the_list_is_stale(self):
        """DATA LOSS, introduced by my own tick filter: the None fallback
        covered an EMPTY row list but not a STALE one. With 2 rows and 5 shapes
        the positional filter dropped 3 drawn boxes silently."""
        code = _code_only(_CONTROLLER_PY)
        assert "box_row_count" in _read(_CONTROLLER_PY), (
            "the filter must compare row count to box count before trusting "
            "positional indices"
        )
        i = code.find("def extract_patches")
        # Scan the WHOLE extract_patches body (to the next method def), not a
        # fixed char window — the guard grows over time (e.g. the processed-view
        # extract block) and a hardcoded slice would miss the check downstream.
        j = code.find("\n    def ", i + 1)
        body = code[i:j if j != -1 else len(code)]
        assert "n_rows != len(bounding_boxes)" in body

    def test_box_row_count_reports_the_list(self, widget):
        assert widget.box_row_count() == 0
        widget.box_list.addItem(QListWidgetItemFactory("p1"))
        assert widget.box_row_count() == 1

    def test_library_rows_have_no_user_facing_checkbox(self, widget):
        """④ no longer draws a per-row tick (F1 made Extract write straight to
        disk, so the old "export checked → Library" tick is dead): EVERY ④ row
        renders the location-eye on the left, never a checkbox — for both a
        staged (in-memory) row and a captured (on-disk) one."""
        from qtpy.QtWidgets import QCheckBox

        _stage(widget, "p1")
        row = widget.extracted_roi_list.itemWidget(
            widget.extracted_roi_list.item(0))
        assert not row.findChildren(QCheckBox), (
            "④ rows must render no user-facing checkbox — Extract writes to "
            "disk directly, so there is nothing to commit"
        )
        # A captured (on-disk) row must likewise carry no checkbox.
        widget.extracted_roi_list.addItem("already_on_disk")
        widget.library_box_list.refresh()
        cap_row = widget.extracted_roi_list.itemWidget(
            widget.extracted_roi_list.item(1))
        assert not cap_row.findChildren(QCheckBox)

    def test_row_checkbox_drives_the_item_state(self, widget):
        from qtpy.QtWidgets import QCheckBox

        widget.box_list.addItem(QListWidgetItemFactory("p1"))
        widget.draw_box_list.refresh()
        row = widget.box_list.itemWidget(widget.box_list.item(0))
        box = row.findChildren(QCheckBox)[0]
        box.setChecked(True)
        assert widget.box_list.item(0).checkState() == Qt.Checked

    def test_rows_are_opaque_so_item_text_cannot_show_through(self, widget):
        """MEASURED: the delegate paints the item's own text UNDER the overlay
        — 4027px of ink vs 764px with it blanked. The text cannot be dropped
        (seven controller sites key off it), so the row must cover it."""
        widget.extracted_roi_list.addItem("p1")
        row = widget.extracted_roi_list.itemWidget(
            widget.extracted_roi_list.item(0))
        assert row.autoFillBackground()

    def test_rows_rerender_when_content_changes(self, widget):
        """The rebuild key was the NAME, which never changes — so Trim
        never updated the row and a staged box said "full Z" forever."""
        widget.extracted_roi_list.addItem("p1")
        widget.controller.extracted_patches = {
            "p1": {"data": None, "selected_z_range": None}}
        widget.library_box_list.refresh()
        first = widget.extracted_roi_list.item(0).data(Qt.UserRole + 9)
        widget.controller.extracted_patches["p1"]["selected_z_range"] = (12, 62)
        widget.library_box_list.refresh()
        assert widget.extracted_roi_list.item(0).data(Qt.UserRole + 9) != first

    def test_count_reflects_the_filter(self, widget):
        """No hand-cranked `refresh()` — the SEARCH must drive the count.

        This test used to call `library_box_list.refresh()` itself between the
        search and the assertion, which `_filter_patch_list` never did: a
        vacuous pass over a count that went stale the moment a user typed.
        """
        _stage(widget, "alpha", "beta")
        assert widget.library_box_list.count_label.text() == "2 patch(es)"
        widget.patch_search_edit.setText("alpha")
        assert "1 of 2 shown" in widget.library_box_list.count_label.text()

    def test_alias_hint_starts_where_extraction_will(self, widget):
        """The hint, ③'s row labels and `find_next_patch_index` must agree."""
        widget.display_name_edit.setText("wound_d3")
        assert "wound_d3_patch_1" in widget.alias_hint_label.text()

    def test_source_card_does_not_print_the_alias_twice(self, widget):
        from qtpy.QtWidgets import QLabel as _QLabel

        item = QListWidgetItem()
        item.setData(Qt.UserRole, "C:/data/wound.tif")
        item.setData(Qt.UserRole + 1, "wound")
        widget.source_file_list.addItem(item)
        item.setText(" wound\n   64×512×512   • 2 patch(es)")
        card = widget.source_file_list.itemWidget(item)
        texts = [c.text() for c in card.findChildren(_QLabel)]
        assert texts.count("wound") == 1, texts


class TestProgressiveDefaultOpen:
    """Phase 1: a constant default-open is wrong."""

    def test_opens_session_when_nothing_is_set_up(self, widget):
        widget.apply_progressive_default_open()
        assert widget.accordion._current == 0

    def test_opens_draw_once_workspace_and_pixels_exist(self, widget):
        widget.output_file_path.setText("C:/ws")
        widget.update_image_layer_list(["src_ch0"])
        widget.apply_progressive_default_open()
        assert widget.accordion._current == 2, (
            "③ is primary once there is something to draw on"
        )

    def test_workspace_without_pixels_stays_on_session(self, widget):
        widget.output_file_path.setText("C:/ws")
        widget.apply_progressive_default_open()
        assert widget.accordion._current == 0

    def test_a_manual_click_wins_for_the_visit(self, widget):
        widget._mark_section_chosen()
        widget.output_file_path.setText("C:/ws")
        widget.update_image_layer_list(["src_ch0"])
        widget.apply_progressive_default_open()
        assert widget.accordion._current == 0, (
            "must not yank the user out of a section they chose"
        )

    def test_clicking_a_header_marks_the_choice(self, widget):
        """Behavioural: the flag is set by the real signal, not only by hand."""
        assert widget._section_chosen_by_user is False
        widget.accordion._sections[1]._header.click()
        assert widget._section_chosen_by_user is True


class TestConsistencySweepFixes:
    """The panel-wide consistency audit (2026-07-26) — one test per finding.

    Every one of these was MEASURED on the rendered widget before being fixed,
    so each test states the observed wrong behaviour rather than just the new
    invariant.  The audit's own summary of the failure mode: "every layer of it
    is individually well-reasoned and the layers were never reconciled with each
    other" — wherever two widgets, two methods, or a comment and its code
    described the same fact, they disagreed.
    """

    def _boxes(self, widget, n):
        class _Layer:
            data = [object() for _ in range(n)]
        widget._current_shapes_layer = lambda: _Layer()
        widget.refresh_draw_box_list()

    # -- H1: two mechanisms wrote (3)'s tick column and the wrong one won --
    def test_untick_survives_drawing_another_box(self, widget):
        """MEASURED: untick row 1 -> `{0, 2}`; draw a 4th box -> `{0,1,2,3}`.

        `refresh_draw_box_list` set `Qt.Unchecked` and then `addItem` fired
        `rowsInserted` -> `BoxListPanel.refresh` -> `make_checkable`, which saw
        an uninitialised item and forced it back to Checked.  The box the user
        excluded was silently re-included — and `extract_patches` filters on
        exactly this set, so it would have been extracted.
        """
        widget.display_name_edit.setText("w")
        self._boxes(widget, 3)
        widget.box_list.item(1).setCheckState(Qt.Unchecked)
        assert widget.checked_box_indices() == {0, 2}
        self._boxes(widget, 4)
        assert widget.checked_box_indices() == {0, 2, 3}, (
            "the untick must survive a rebuild; index 3 is new so it defaults in"
        )

    def test_untick_survives_an_alias_change(self, widget):
        """MEASURED: renaming the alias renumbered every row, so a by-NAME
        untick matched nothing.  Preservation is positional — which is also
        what `checked_box_indices` and the extract itself are."""
        widget.display_name_edit.setText("wound")
        self._boxes(widget, 3)
        widget.box_list.item(0).setCheckState(Qt.Unchecked)
        widget.display_name_edit.setText("wounds")
        self._boxes(widget, 3)
        assert widget.checked_box_indices() == {1, 2}

    # -- H2: the Z-truncate spins were write-only --
    def test_ztrunc_spins_follow_the_selected_range(self, widget):
        """MEASURED: the span selected (12, 62), the label read "12 to 62", and
        both spins sat at 0 — so one nudge wrote `(0, 1)` into
        `selected_z_range`, which `Trim` commits and the export slices the
        TIFF with.  A 1-slice crop from a stray click on a control that LOOKED
        like it held the real range."""
        widget.set_selected_roi_status("p1", (12, 62))
        assert widget.ztrunc_start_spin.value() == 12
        assert widget.ztrunc_end_spin.value() == 62

    def test_seeding_does_not_write_back(self, widget):
        """The mirror-image bug: if seeding re-entered
        `_on_ztrunc_spin_changed` it would overwrite the very range it came
        from, clamped to the spins' own limits."""
        widget.controller.selected_z_range = (12, 62)
        widget.seed_ztrunc_spins((12, 62))
        assert widget.controller.selected_z_range == (12, 62)

    def test_controller_routes_both_writers_through_the_seed(self, widget):
        """Two live writers existed (`onselect`, autoselect) and neither told
        the editor.  One chokepoint now does."""
        widget.controller._set_selected_z_range((5, 9))
        assert widget.controller.selected_z_range == (5, 9)
        assert widget.ztrunc_start_spin.value() == 5
        assert widget.ztrunc_end_spin.value() == 9

    # -- H4: the fixed-box size was unreachable --
    def test_preset_reaches_the_fixed_draw_size(self, widget):
        """MEASURED: presets wrote `box_size_*_spin` while drawing read the
        HIDDEN `bbox_size_spinbox`, which had no writer outside `__init__` — so
        there was NO path to change the fixed box size, while the button's
        tooltip said "change the size with Geometry's presets"."""
        assert "presets" in widget.fixed_bbox_button.toolTip()
        widget._apply_box_preset(512)
        assert widget.bbox_size_spinbox.value() == 512
        assert "512" in widget.fixed_bbox_button.text(), (
            "the label is built once at construction — a stale label would "
            "show 256 while drawing a 512 box"
        )

    def test_presets_are_reachable(self, widget):
        """They were disabled by the blanket box-editor lock, which is why the
        size was unreachable even after the write was fixed."""
        assert widget.box_preset_buttons
        for btn in widget.box_preset_buttons:
            assert btn.isEnabled()

    # -- M1: one reason string pasted onto two different kinds of blocker --
    def test_disabled_reasons_name_the_actual_blocker(self, widget):
        """Geometry spins are BoxRec projections and must say so.  (The former
        Box-appearance block that used to share this test was AXED in the ④ P2
        Brick-1 rework — see `test_box_appearance_section_is_axed`.)"""
        for w in widget._box_geometry_widgets:
            assert "BoxRec" in w.toolTip(), w.toolTip()

    def test_disable_reason_does_not_clobber_a_real_tooltip(self, widget):
        """The blanket disable loop must not overwrite tooltips set moments
        earlier — the presets keep their own size explanation and lock keeps its
        (neither is in the disable loop).  v2: the 128³ preset now describes the
        absolute voxel size it sets, not the old W/H/D writer."""
        assert "128 voxels" in widget.box_preset_buttons[0].toolTip()
        assert "Freeze" in widget.box_lock_size_check.toolTip()

    # -- H7: two views of one value with different domains --
    def test_geometry_slider_and_spin_share_one_domain(self, widget):
        """MEASURED lossy: slider 0..2048 against spin 0..100000 meant typing
        5000 came back as 2048 — which defeats the stated reason for having
        both controls."""
        for spin in (widget.box_size_w_spin, widget.box_center_x_spin):
            spin.setValue(4096)
            assert spin.value() == 4096
        assert (widget.bbox_size_spinbox.maximum()
                == widget.box_size_w_spin.maximum()), (
            "the two size models must share a ceiling"
        )



class TestBoxToolCorrectivePass:
    """Corrective pass on the box-tool refinement batch: dynamic geometry
    maxima, retired cube presets, busy-lock coverage of the new size controls,
    and the size-preset no-op-when-off contract."""

    # -- Fix 2: geometry spin/slider maxima follow the source shape --
    def test_geometry_maxima_lift_off_a_large_source(self, widget):
        """A source deeper/wider than the 4096 build-time ceiling must raise the
        spin (and its paired slider) ceilings, else editing a spin clamps + the
        Z/Y/X slab silently crops."""
        Z, Y, X = 5000, 4600, 4200
        widget.set_box_geometry_maxima((Z, Y, X))
        assert widget.box_size_d_spin.maximum() == Z      # D <- Z
        assert widget.box_center_z_spin.maximum() == Z
        assert widget.box_size_h_spin.maximum() == Y      # H <- Y
        assert widget.box_center_y_spin.maximum() == Y
        assert widget.box_size_w_spin.maximum() == X      # W <- X
        assert widget.box_center_x_spin.maximum() == X
        # The paired slider ceilings move too, so the spin's cross-connected
        # setValue is not clamped back to 4096.
        widget.box_size_d_spin.setValue(4800)
        assert widget.box_size_d_spin.value() == 4800

    def test_geometry_maxima_ignore_bad_shape(self, widget):
        before = widget.box_size_d_spin.maximum()
        widget.set_box_geometry_maxima(None)
        widget.set_box_geometry_maxima((10, 20))          # not 3-tuple
        assert widget.box_size_d_spin.maximum() == before

    def test_box_mode_start_lifts_maxima(self, widget):
        """Wiring: the box tool asks the widget to lift the ceilings when box
        mode starts (driven here through its resolved-shape helper, since a real
        `start()` needs a napari camera)."""
        bt = widget.controller.box_tool
        bt._full_shape = (3333, 400, 600)
        bt._apply_geometry_spin_maxima()
        assert widget.box_size_d_spin.maximum() == 3333

    # -- v2 (feedback #6): the cube presets are RESTORED with safe semantics --
    def test_cube_presets_restored_and_wired_to_voxel_sizing(self, widget):
        """The 128³/256³/512³ presets are back — visible (not explicitly
        hidden) and wired to `set_box_size_voxels` (the sticky absolute-size
        model), NOT the old retroactive W/H/D writer."""
        assert widget.box_preset_buttons                  # still built
        for btn in widget.box_preset_buttons:
            assert not btn.isHidden()                     # restored (unhidden)
            assert btn.isEnabled()
            assert hasattr(btn, "setEnabled")             # attr/contract intact
        # `_apply_box_preset` (the retired W/H/D writer) stays reachable for its
        # own tests + the fixed-draw-size contract, but is no longer wired here.
        widget._apply_box_preset(512)
        assert widget.bbox_size_spinbox.value() == 512

    def test_custom_size_field_present_and_wired(self, widget):
        """v2 custom absolute size: a voxel spin + Set button feed the same
        sticky size model."""
        assert widget.box_custom_size_spin.maximum() == widget.box_size_w_spin.maximum()
        assert not widget.box_custom_apply_button.isHidden()

    # -- Fix 4: busy-lock covers the new size presets + place toggle --
    def test_busy_lock_disables_size_presets_and_place_check(self, widget):
        for b in widget.box_size_preset_buttons:
            assert b.isEnabled()
        assert widget.box_place_check.isEnabled()
        widget.controller._set_step1_busy(True)
        assert all(not b.isEnabled() for b in widget.box_size_preset_buttons)
        assert not widget.box_place_check.isEnabled()
        widget.controller._set_step1_busy(False)
        assert all(b.isEnabled() for b in widget.box_size_preset_buttons)
        assert widget.box_place_check.isEnabled()

    # -- Fix 5: a size click must not start box mode / flip to 3D --
    def test_size_fraction_is_a_no_op_when_box_mode_off(self, widget):
        class _Stub:
            def __init__(self):
                self.started = False
                self.resized = None
            def is_active(self):
                return False
            def start(self):
                self.started = True
                return True
            def set_size_fraction(self, f):
                self.resized = f
        stub = _Stub()
        widget.controller.box_tool = stub
        widget.controller.set_box_size_fraction(0.5)
        assert not stub.started, "must NOT flip the canvas to 3D on a size click"
        assert stub.resized is None, "nothing to resize when box mode is off"

    def test_size_fraction_resizes_when_box_mode_active(self, widget):
        class _Stub:
            def __init__(self):
                self.started = False
                self.resized = None
            def is_active(self):
                return True
            def start(self):
                self.started = True
                return True
            def set_size_fraction(self, f):
                self.resized = f
        stub = _Stub()
        widget.controller.box_tool = stub
        widget.controller.set_box_size_fraction(0.5)
        assert not stub.started
        assert stub.resized == 0.5

    def test_size_presets_are_themed_like_secondary_buttons(self, widget):
        # S/M/L ride SECONDARY_BTN_STYLE but override only the padding so the
        # single glyph is not clipped (2026-07-27).  They must still be styled,
        # uniform across the three, and share SECONDARY's visual base with Add.
        base = widget.box_add_button.styleSheet()
        assert base                                        # Add is themed
        for b in widget.box_size_preset_buttons:
            assert b.styleSheet()                          # not unstyled
            assert b.styleSheet().startswith(base)         # SECONDARY base + override
        sheets = {b.styleSheet() for b in widget.box_size_preset_buttons}
        assert len(sheets) == 1                            # uniform across S/M/L


class TestBoxSizingIndependence:
    """Part A (2026-07-27): each ROI box has its OWN size.  Editing one box must
    not touch another; a size preset is sticky for the NEXT box only, never
    retroactive.  Driven through the REAL geometry spins (the on-canvas edit
    path) so the fix is proven against Qt's live slider<->spin signal wiring."""

    SHAPE = (30, 400, 600)

    def _tool(self, widget):
        t = widget.controller.box_tool
        t._full_shape = self.SHAPE
        t.widget = widget
        t._connect_geometry()
        return t

    def test_editing_one_box_leaves_others_unchanged(self, widget):
        t = self._tool(widget)
        t.add_box(center_yx=(100, 100), size_yx=(40, 40))
        t.add_box(center_yx=(200, 200), size_yx=(80, 80))
        box1_before = (t.boxes[1].sy, t.boxes[1].sx)
        t.select(0)                                   # box 0 is the edit target
        widget.box_size_w_spin.setValue(120)          # type/drag box 0's width
        assert t.boxes[0].sx == 120                    # box 0 changed
        assert (t.boxes[1].sy, t.boxes[1].sx) == box1_before   # box 1 untouched
        assert t.boxes[0] is not t.boxes[1]            # distinct BoxRecs

    def test_selecting_a_box_loads_its_own_size(self, widget):
        t = self._tool(widget)
        t.add_box(center_yx=(100, 100), size_yx=(40, 44))
        t.add_box(center_yx=(200, 200), size_yx=(80, 88))
        t.select(0)
        assert (widget.box_size_h_spin.value(),
                widget.box_size_w_spin.value()) == (40, 44)
        t.select(1)
        assert (widget.box_size_h_spin.value(),
                widget.box_size_w_spin.value()) == (80, 88)

    def test_size_preset_is_sticky_for_next_box_not_retroactive(self, widget):
        """Fail-first on the pre-fix code: clicking S/M/L resized the SELECTED
        box.  It must now leave placed boxes alone and only size the next one."""
        t = self._tool(widget)
        t.add_box(center_yx=(100, 100), size_yx=(40, 40))
        before = (t.boxes[0].sy, t.boxes[0].sx)
        t.set_size_fraction(0.5)                       # L — sticky, not retroactive
        assert (t.boxes[0].sy, t.boxes[0].sx) == before   # placed box UNCHANGED
        t.add_box(center_yx=(300, 300))                # next box gets sticky L
        _, Y, X = self.SHAPE
        assert (t.boxes[1].sy, t.boxes[1].sx) == (int(Y * 0.5), int(X * 0.5))


class TestBoxToolLayoutAndStyle:
    """Part B (splitter) + Part C (the global ④ style block AXED in Brick 1;
    the count row now carries the location toggle + Snapshot split button)."""

    # -- Part B: the ③ hint that pinned the splitter must word-wrap --
    def test_draw_hint_label_wraps(self, widget):
        from qtpy.QtWidgets import QLabel
        draw = widget.accordion._sections[2]
        hints = [l for l in draw.findChildren(QLabel)
                 if "narrow Z" in l.text()]
        assert hints, "the ③ hint label was not found"
        assert all(l.wordWrap() for l in hints), (
            "a long single line pins the left column and freezes the splitter"
        )

    # -- Part C (Brick 1): the global style block is AXED; the count row now
    # holds the renamed location toggle + the Snapshot split button --
    def test_show_locations_and_snapshot_in_count_row(self, widget):
        from qtpy.QtWidgets import QCheckBox, QToolButton
        review = widget.accordion._sections[3]
        # The location master toggle: renamed, frozen attr name kept, in ④.
        chk = widget.show_locations_check
        assert isinstance(chk, QCheckBox)
        assert chk.text() == "Show locations"
        assert chk in review.findChildren(QCheckBox)
        assert not chk.isHidden()
        # Its tooltip must NOT mention the axed "Box appearance" section.
        assert "Box appearance" not in chk.toolTip()
        assert _is_connected(chk)
        # The Snapshot split button sits in the same review section.
        snap = widget.snapshot_button
        assert isinstance(snap, QToolButton)
        assert "Snapshot" in snap.text()
        assert snap in review.findChildren(QToolButton)
        assert snap.menu() is not None          # the ▾ popover exists
        assert _is_connected(snap)              # main-half click is wired

    def test_snapshot_popover_controls_exist(self, widget):
        from qtpy.QtWidgets import QSpinBox
        # The scale bar moved to its own LIVE "Scale bar" control (captured
        # WYSIWYG); the snapshot popover now carries only the export resolution.
        assert isinstance(widget.snapshot_scale_spin, QSpinBox)
        assert widget.snapshot_scale_spin.minimum() == 1
        assert widget.snapshot_scale_spin.maximum() == 8
        assert not hasattr(widget, "snapshot_scalebar_check")
        assert not hasattr(widget, "snapshot_color_combo")

    def test_live_scalebar_controls_exist(self, widget):
        from qtpy.QtWidgets import (
            QToolButton, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox)
        assert isinstance(widget.scalebar_toggle_button, QToolButton)
        assert widget.scalebar_toggle_button.isCheckable()
        assert widget.scalebar_toggle_button.menu() is not None
        assert isinstance(widget.scalebar_auto_length_check, QCheckBox)
        assert isinstance(widget.scalebar_length_spin, QDoubleSpinBox)
        assert isinstance(widget.scalebar_font_spin, QSpinBox)
        assert isinstance(widget.scalebar_unit_combo, QComboBox)
        assert [widget.scalebar_unit_combo.itemText(i)
                for i in range(widget.scalebar_unit_combo.count())] == [
                    "auto", "µm", "nm", "mm", "pixel"]
        assert isinstance(widget.scalebar_color_combo, QComboBox)
        assert isinstance(widget.scalebar_box_check, QCheckBox)
        assert isinstance(widget.scalebar_ticks_check, QCheckBox)
        assert isinstance(widget.scalebar_position_combo, QComboBox)
        # Auto length starts on → the fixed-length spin is greyed out.
        assert widget.scalebar_auto_length_check.isChecked()
        assert not widget.scalebar_length_spin.isEnabled()
        widget.scalebar_auto_length_check.setChecked(False)
        assert widget.scalebar_length_spin.isEnabled()

    def test_box_tool_set_box_style_retained(self, widget):
        """The ③ box tool's own `set_box_style` is RETAINED (Brick 1 only axes
        the ④ GLOBAL style widgets, not the box tool's default-style API)."""
        assert hasattr(widget.controller.box_tool, "set_box_style")


class TestSnapshotExport:
    """④ Snapshot: `_on_snapshot_clicked` saves a WYSIWYG TIFF of the CURRENT
    canvas via `viewer.screenshot(canvas_only, scale)` — it does NOT force 2D,
    auto-show boxes, or touch the scale bar (the LIVE control owns it, captured
    verbatim).  The ONLY thing it mutates is `camera.zoom` for the N× super-sample,
    restored on every path (success + cancel)."""

    def _install_fake_viewer(self, widget, monkeypatch, patch_names=("p1", "p2")):
        controller = widget.controller
        captured = {}

        class _SB:
            def __init__(self):
                self.visible = False
                self.colored = False
                self.unit = None
                self.position = "bottom_right"
                self.color = (1.0, 0.0, 1.0, 1.0)
                self.font_size = 10.0

        class _Dims:
            def __init__(self):
                self.ndisplay = 3

        class _Cam:
            def __init__(self):
                self.zoom = 1.0

        class _SV:
            def __init__(self):
                self.scale_bar = _SB()
                self.dims = _Dims()
                self.camera = _Cam()

            def screenshot(self, path=None, size=None, scale=None,
                           canvas_only=True, flash=False):
                # Snapshot the view state AT capture time — proves the live scale
                # bar is captured verbatim, that ndisplay + locations are left
                # AS-IS, and the zoom was pre-bumped by `scale` (super-sample).
                captured.update(
                    path=path, scale=scale, canvas_only=canvas_only,
                    unit=self.scale_bar.unit, visible=self.scale_bar.visible,
                    position=self.scale_bar.position,
                    font_size=self.scale_bar.font_size,
                    ndisplay=self.dims.ndisplay, zoom=self.camera.zoom,
                    locations=set(controller._visible_patch_locations),
                )
                return None

        sv = _SV()
        controller.viewer = sv
        monkeypatch.setattr(controller, "_current_source_patch_names",
                            lambda: list(patch_names))
        # Deterministic factor: skip the (headless-flaky) canvas-size clamp.
        monkeypatch.setattr(controller, "_snapshot_canvas_size", lambda: None)
        return controller, sv, captured

    def test_wysiwyg_capture_never_touches_scale_bar(self, widget, monkeypatch):
        """MAIN half captures the CURRENT view via screenshot(canvas_only, scale).
        The scale bar is the LIVE control's — the snapshot captures it VERBATIM and
        reconfigures NOTHING; ndisplay + the location set are left AS-IS."""
        controller, sv, captured = self._install_fake_viewer(widget, monkeypatch)
        controller._visible_patch_locations = {"prior"}
        widget.snapshot_scale_spin.setValue(3)
        # Whatever the live bar is showing must be captured as-is.
        sv.scale_bar.visible = True
        sv.scale_bar.unit = "um"
        sv.scale_bar.position = "top_left"
        monkeypatch.setattr(controller, "_ask_snapshot_path",
                            lambda: "C:/tmp/snap.tiff")

        controller._on_snapshot_clicked()

        assert captured["scale"] == 3
        assert captured["canvas_only"] is True
        assert captured["path"] == "C:/tmp/snap.tiff"
        # The live scale bar was captured verbatim — NOT reconfigured to px.
        assert captured["unit"] == "um"
        assert captured["visible"] is True
        assert captured["position"] == "top_left"
        # WYSIWYG: ndisplay + the location set were NOT changed for the shot.
        assert captured["ndisplay"] == 3
        assert captured["locations"] == {"prior"}
        # The zoom was pre-bumped by `scale` (super-sample the SAME framing, not
        # zoom out), then restored to the live value.
        assert captured["zoom"] == 3.0
        assert sv.camera.zoom == 1.0
        # The bar is left EXACTLY as it was — nothing to restore.
        assert sv.scale_bar.visible is True
        assert sv.scale_bar.unit == "um"
        assert sv.scale_bar.position == "top_left"

    def test_export_compensates_scale_bar_font_for_supersampling(
            self, widget, monkeypatch):
        """A scale=N shot renders the fixed-pixel bar at 1/N, so the export bumps
        the bar's font by N for the capture and restores it after."""
        controller, sv, captured = self._install_fake_viewer(widget, monkeypatch)
        sv.scale_bar.visible = True
        sv.scale_bar.font_size = 10.0
        widget.snapshot_scale_spin.setValue(2)
        monkeypatch.setattr(controller, "_ask_snapshot_path",
                            lambda: "C:/tmp/snap.tiff")

        controller._on_snapshot_clicked()

        # Font was ×2 AT capture time (so the exported bar keeps its size)…
        assert captured["font_size"] == 20.0
        # …and restored to the live value afterwards.
        assert sv.scale_bar.font_size == 10.0

    def test_export_no_font_bump_at_scale_one(self, widget, monkeypatch):
        controller, sv, captured = self._install_fake_viewer(widget, monkeypatch)
        sv.scale_bar.visible = True
        sv.scale_bar.font_size = 10.0
        widget.snapshot_scale_spin.setValue(1)
        monkeypatch.setattr(controller, "_ask_snapshot_path",
                            lambda: "C:/tmp/snap.tiff")

        controller._on_snapshot_clicked()

        assert captured["font_size"] == 10.0     # no super-sample → no bump
        assert sv.scale_bar.font_size == 10.0

    def test_snapshot_leaves_a_bar_that_is_off_alone(self, widget, monkeypatch):
        """A live bar that's OFF stays off — the snapshot never toggles it on."""
        controller, sv, captured = self._install_fake_viewer(widget, monkeypatch)
        sv.scale_bar.visible = False
        monkeypatch.setattr(controller, "_ask_snapshot_path",
                            lambda: "C:/tmp/snap.tiff")

        controller._on_snapshot_clicked()

        assert captured["path"] == "C:/tmp/snap.tiff"
        assert captured["visible"] is False
        assert sv.scale_bar.visible is False                 # unchanged

    def test_cancelled_dialog_leaves_bar_untouched(self, widget, monkeypatch):
        controller, sv, captured = self._install_fake_viewer(widget, monkeypatch)
        sv.scale_bar.visible = True
        monkeypatch.setattr(controller, "_ask_snapshot_path", lambda: "")

        controller._on_snapshot_clicked()

        assert captured == {}                                # capture never ran
        assert sv.scale_bar.visible is True                  # untouched on cancel

    def test_snapshot_scale_clamped_to_canvas(self, widget, monkeypatch):
        """The scale-factor clamp keeps canvas·factor ≤ the GL limit —
        `screenshot(canvas_only, scale)` renders the canvas at scale×."""
        controller = widget.controller
        # large canvas (4096) → 4096·8 = 32768 > 8192 → clamp to 2
        monkeypatch.setattr(
            controller, "_snapshot_canvas_size", lambda: (4096, 4096))
        assert controller._clamp_snapshot_scale(8) == (2, True)   # 8192 // 4096
        assert controller._clamp_snapshot_scale(2) == (2, False)  # 4096·2 = 8192
        # small canvas (256) → 256·8 = 2048 → no clamp
        monkeypatch.setattr(
            controller, "_snapshot_canvas_size", lambda: (256, 256))
        assert controller._clamp_snapshot_scale(8) == (8, False)
        # unresolvable canvas → no clamp
        monkeypatch.setattr(controller, "_snapshot_canvas_size", lambda: None)
        assert controller._clamp_snapshot_scale(8) == (8, False)


class TestBoxLockFreezesSelectedBox:
    """2026-07-27: the 'lock' checkbox freezes the SELECTED box — its six
    geometry spins are disabled and `on_props_change` no-ops — so a locked box
    cannot be moved or resized until unlocked.  Per-box: the flag rides the box
    through reorder and a NEW box is born unlocked.  Driven through the REAL
    geometry spins (the on-canvas edit path) so the freeze is proven against
    Qt's live slider<->spin signal wiring."""

    SHAPE = (30, 400, 600)

    def _tool(self, widget):
        t = widget.controller.box_tool
        t._full_shape = self.SHAPE
        t.widget = widget
        t._connect_geometry()
        return t

    def test_locking_disables_spins_and_freezes_edits(self, widget):
        t = self._tool(widget)
        t.add_box(center_yx=(100, 100), size_yx=(40, 40))
        t.select(0)
        assert widget.box_size_w_spin.isEnabled()          # editable before lock
        t.set_active_locked(True)
        assert not widget.box_size_w_spin.isEnabled()       # locked -> disabled
        widget.box_size_w_spin.setValue(111)                # a would-be resize
        assert t.boxes[0].sx == 40                           # FROZEN, unchanged

    def test_unlock_re_enables_edits_and_resyncs_spins(self, widget):
        t = self._tool(widget)
        t.add_box(center_yx=(100, 100), size_yx=(40, 40))
        t.select(0)
        t.set_active_locked(True)
        widget.box_size_w_spin.setValue(111)                # ignored while locked
        t.set_active_locked(False)
        assert widget.box_size_w_spin.isEnabled()
        assert widget.box_size_w_spin.value() == 40          # spin re-synced to box
        widget.box_size_w_spin.setValue(55)                  # now it takes
        assert t.boxes[0].sx == 55

    def test_lock_is_per_box_and_follows_reorder(self, widget):
        t = self._tool(widget)
        t.add_box(center_yx=(100, 100), size_yx=(40, 40))    # box A (index 0)
        t.add_box(center_yx=(200, 200), size_yx=(80, 80))    # box B (index 1)
        t.select(0)
        t.set_active_locked(True)                            # lock A
        assert t._reorder_boxes([1, 0])                      # A moves to index 1
        assert t._is_locked(1)                                # lock followed A
        assert not t._is_locked(0)                            # B still unlocked

    def test_new_box_is_born_unlocked(self, widget):
        t = self._tool(widget)
        t.add_box(center_yx=(100, 100), size_yx=(40, 40))
        t.select(0)
        t.set_active_locked(True)
        t.add_box(center_yx=(300, 300), size_yx=(40, 40))    # a fresh box
        assert not t._is_locked(1)                            # born unlocked
        assert widget.box_size_w_spin.isEnabled()             # editable

    def test_checkbox_reflects_selected_box_lock(self, widget):
        t = self._tool(widget)
        t.add_box(center_yx=(100, 100), size_yx=(40, 40))
        t.add_box(center_yx=(200, 200), size_yx=(80, 80))
        t.select(0)
        t.set_active_locked(True)
        t.select(1)
        assert not widget.box_lock_size_check.isChecked()     # B unlocked
        assert widget.box_lock_size_check.isEnabled()
        t.select(0)
        assert widget.box_lock_size_check.isChecked()         # A locked


class TestBusyDepth:
    """H6 — `_set_step1_busy` was a depth-1 save/restore with the re-entry
    guard on only one of its four callers."""

    def _ctl(self):
        from qtpy.QtWidgets import QPushButton
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController,
        )

        class _Widget:
            def __init__(self):
                self.extract_roi = QPushButton("x")
                self.save_reviewed_roi = QPushButton("y")

        ctl = DataPreparationController.__new__(DataPreparationController)
        ctl.widget = _Widget()
        ctl._step1_busy = False
        return ctl

    def test_nested_lock_is_survivable(self, app):
        """MEASURED: lock, lock, unlock -> every listed widget disabled,
        `_pre_busy_enabled` empty, `_step1_busy` False.  The surface unusable
        while the flag said idle, and the only code that re-enables it lives in
        `finally` blocks of ops that can no longer be launched."""
        ctl = self._ctl()
        ctl._set_step1_busy(True)
        ctl._set_step1_busy(True)
        ctl._set_step1_busy(False)
        assert ctl._step1_busy, "still one op in flight"
        assert not ctl.widget.extract_roi.isEnabled()
        ctl._set_step1_busy(False)
        assert not ctl._step1_busy
        assert ctl.widget.extract_roi.isEnabled()
        assert ctl.widget.save_reviewed_roi.isEnabled()

    def test_unlock_restores_prior_enablement_not_true(self, app):
        """A deliberately-disabled control must stay disabled after a busy
        cycle — one cycle used to switch on everything in the list."""
        ctl = self._ctl()
        ctl.widget.save_reviewed_roi.setEnabled(False)
        ctl._set_step1_busy(True)
        ctl._set_step1_busy(False)
        assert ctl.widget.extract_roi.isEnabled()
        assert not ctl.widget.save_reviewed_roi.isEnabled()

    def test_browse_locks_during_a_step1_op(self, widget):
        """`load_source` drains the event queue BEFORE it launches the read
        worker, so a click queued on Browse lands mid-load, changes
        `output_file_path`, and `_restore_bounding_boxes(display_name)` (run in
        `_on_source_loaded`) then restores boxes from a DIFFERENT workspace than
        the load was started against.

        Asserted on the real widget, not by grepping for the name: the first
        version of this test searched `_code_only(_CONTROLLER_PY)` for
        `"browse_button"` — which that helper blanks, being a string literal —
        so it could never pass, and had it been written against a raw read it
        would have passed on the name's mere presence while proving nothing
        about the lock.
        """
        assert widget.browse_button.isEnabled()
        widget.controller._set_step1_busy(True)
        assert not widget.browse_button.isEnabled()
        widget.controller._set_step1_busy(False)
        assert widget.browse_button.isEnabled()


class TestBoxToolListOwnership:
    """③'s box tool OWNS the Box List (spec item C / feedback #7/#8): exactly
    ONE (native) checkbox per row, unticks preserved across rebuilds, and the
    list is drag-reorderable with list order == extraction order.

    Driven on the REAL widget + list (the box tool's `widget` IS
    `controller.widget`), the geometry resolved directly so no napari camera /
    Shapes layer is needed — the checkbox + tick-state logic is Qt, not viewer.
    """

    def _prep(self, widget):
        bt = widget.controller.box_tool
        bt._full_shape = (30, 400, 600)
        bt._own_list()        # suppress the row-card overlay + enable InternalMove
        bt._connect_list()    # wire itemChanged (tick sync) + reorder signals
        return bt

    def test_exactly_one_native_checkbox_no_row_widget(self, widget):
        bt = self._prep(widget)
        bt.add_box(center_yx=(100, 100))
        lst = widget.draw_box_list.list
        assert lst.count() == 1
        item = lst.item(0)
        assert bool(item.flags() & Qt.ItemIsUserCheckable)
        assert item.checkState() == Qt.Checked
        # No overlaid row-card widget -> the native checkbox is the ONLY tick.
        assert lst.itemWidget(item) is None

    def test_list_is_internal_move_reorderable(self, widget):
        from qtpy.QtWidgets import QAbstractItemView
        bt = self._prep(widget)
        bt.add_box(center_yx=(100, 100))
        assert (widget.draw_box_list.list.dragDropMode()
                == QAbstractItemView.InternalMove)

    def test_untick_survives_rebuild(self, widget):
        bt = self._prep(widget)
        bt.add_box(center_yx=(100, 100))
        lst = widget.draw_box_list.list
        lst.item(0).setCheckState(Qt.Unchecked)   # user untick -> box_checked[0]=False
        assert bt.box_checked[0] is False
        bt.add_box(center_yx=(200, 200))           # forces a full rebuild
        assert lst.count() == 2
        # Phase-1 F1 fixed: the untick is NOT re-ticked by the rebuild.
        assert lst.item(0).checkState() == Qt.Unchecked
        assert lst.item(1).checkState() == Qt.Checked

    def test_build_index_stored_for_reorder_mapping(self, widget):
        bt = self._prep(widget)
        for cy in (10, 20, 30):
            bt.add_box(center_yx=(cy, cy))
        lst = widget.draw_box_list.list
        # Each row carries its 0..n-1 build-index (UserRole+5) so the drop
        # handler can map the new visual order back onto `boxes`.
        assert [lst.item(r).data(Qt.UserRole + 5) for r in range(3)] == [0, 1, 2]


# ---------------------------------------------------------------------------
# (B) Brick 2b — the preprocessing PREVIEW: test-crop compute + before/after
#     histogram.  Apply stays a stub (Brick 3), covered by TestRecipeBuilder.
# ---------------------------------------------------------------------------


class TestPreprocessPreviewCompute:
    """The PURE compute path (Qt-free): crops -> execute -> before/after
    histograms.  execute is referenced through the controller module global so
    it can be monkeypatched (the convert_to_ome_zarr seam)."""

    def test_compute_preview_before_is_raw_not_normalized(self, monkeypatch):
        import numpy as np
        import napari_epicycle.data_preparation.controller as ctrl_mod

        calls = []

        def fake_execute(recipe, crops, spacing, bounds_per_channel=None):
            calls.append((recipe, spacing, [np.asarray(c).shape for c in crops]))
            return [np.asarray(c, dtype=np.float32) for c in crops], spacing, {}

        monkeypatch.setattr(ctrl_mod, "_pp_execute", fake_execute)

        crops = [
            (np.random.rand(4, 8, 8) * 500).astype(np.float32),
            (np.random.rand(4, 8, 8) * 3).astype(np.float32),
        ]
        recipe = {"channels": [[], []], "geometry": None}
        befores, afters, sp = ctrl_mod._compute_preview(
            recipe, crops, (2.0, 0.5, 0.5))

        assert len(calls) == 1, "execute must be called exactly once"
        assert calls[0][1] == (2.0, 0.5, 0.5)
        assert len(befores) == 2 and len(afters) == 2
        # BEFORE is now the RAW crop (honest reference), NOT normalize(crop): it
        # keeps the raw dynamic range (>1 here), so the histogram can plot it
        # through the UNCLIPPED u-map and show the clip normalize applies.
        assert float(befores[0].max()) > 1.0
        np.testing.assert_allclose(befores[0], crops[0])

    def test_build_histogram_entries_shape_and_stats(self):
        import numpy as np
        import napari_epicycle.data_preparation.controller as ctrl_mod

        befores = [(np.random.rand(2, 4, 4) * 500).astype(np.float32),
                   (np.random.rand(2, 4, 4) * 500).astype(np.float32)]
        afters = [np.ones((2, 4, 4), np.float32),
                  np.zeros((2, 4, 4), np.float32)]
        entries = ctrl_mod.build_preview_histogram_entries(
            ["cad", "nuc"], ("#111111", "#222222"), befores, afters)

        assert len(entries) == 2
        for ci, entry in enumerate(entries):
            idx, name, lut, bc, ac, edges, stats = entry
            assert idx == ci
            # 256 fixed bins over the [-0.1, 1.1] frame (edges = bins + 1).
            assert len(bc) == 256 and len(ac) == 256
            assert len(edges) == 257
            assert abs(float(edges[0]) - (-0.1)) < 1e-9
            assert abs(float(edges[-1]) - 1.1) < 1e-9
            assert name in ("cad", "nuc")
            # The stat dict carries the factual read-out keys.
            for key in ("p_lo", "p_hi", "pct_clipped_low", "pct_clipped_high",
                        "median_before_u", "median_after_u"):
                assert key in stats
        # Attribute access works too (namedtuple).
        assert entries[0].stats["p_hi"] > entries[0].stats["p_lo"]

    def test_normalize_only_before_and_after_differ(self):
        """The old-bug guard: for a normalize-only recipe, BEFORE (raw through the
        UNCLIPPED u) and AFTER (clip(u,0,1)) must NOT be identical — before must
        carry mass OUTSIDE [0,1] (the clipped tails) while after does not."""
        import numpy as np
        import napari_epicycle.data_preparation.controller as ctrl_mod

        rng = np.random.default_rng(0)
        raw = (rng.random((4, 16, 16)) * 1000).astype(np.float32)
        # AFTER = what percentile_normalize produces (the real normalize-only op).
        from packages.epicycle_preprocessing.ops import percentile_normalize
        after = percentile_normalize(raw)

        entries = ctrl_mod.build_preview_histogram_entries(
            ["c"], ("#111111",), [raw], [after])
        idx, name, lut, bc, ac, edges, stats = entries[0]
        bc = np.asarray(bc)
        ac = np.asarray(ac)
        edges = np.asarray(edges)
        # The two curves are NOT identical (the pre-2026-08 bug drew them equal).
        assert not np.array_equal(bc, ac)
        # BEFORE has mass in the overflow (clipped) bins; AFTER (clip(u,0,1)) does
        # not spill past 0 / 1, so bins FULLY below 0 / above 1 are empty for it.
        # (Use fully-outside bins so the boundary bin straddling 0.0 / 1.0 — where
        # after's clipped pile-up legitimately lives — is excluded.)
        below = edges[1:] <= 0.0        # bins with right edge at/below 0
        above = edges[:-1] >= 1.0       # bins with left edge at/above 1
        assert int(bc[below].sum()) > 0 or int(bc[above].sum()) > 0
        assert int(ac[below].sum()) == 0 and int(ac[above].sum()) == 0
        # And the p0.1/p99.5 clip fractions are reported.
        assert stats["pct_clipped_low"] >= 0.0
        assert stats["pct_clipped_high"] >= 0.0

    def test_clip_fractions_on_crafted_crop(self):
        """`pct_clipped_low/high` == the KNOWN tail fractions of a crafted crop.

        raw = 0..999 (1000 distinct voxels).  np.percentile linear: p0.1 ≈ 0.999
        (only value 0 < it → 0.1%); p99.5 ≈ 994.005 (values 995..999 > it →
        0.5%).  These are exactly the 0.1 / 0.5 clip fractions the readout shows.
        """
        import numpy as np
        import napari_epicycle.data_preparation.controller as ctrl_mod

        raw = np.arange(1000, dtype=np.float32).reshape(1, 25, 40)
        after = np.zeros((1, 25, 40), np.float32)
        entries = ctrl_mod.build_preview_histogram_entries(
            ["c"], ("#111111",), [raw], [after])
        stats = entries[0].stats
        assert abs(stats["pct_clipped_low"] - 0.1) < 1e-6      # 1/1000
        assert abs(stats["pct_clipped_high"] - 0.5) < 1e-6     # 5/1000

    def test_clip_fractions_match_percentile_definition(self):
        """The controller's clip fractions match the definitional formula exactly
        (raw < p0.1 → u<0; raw > p99.5 → u>1) for a large smooth distribution."""
        import numpy as np
        import napari_epicycle.data_preparation.controller as ctrl_mod

        rng = np.random.default_rng(1)
        raw = (rng.random((1, 200, 200)) * 4000).astype(np.float32)
        after = np.zeros_like(raw)
        entries = ctrl_mod.build_preview_histogram_entries(
            ["c"], ("#111111",), [raw], [after])
        stats = entries[0].stats
        a = raw.ravel()
        p_lo = float(np.percentile(a, 0.1))
        p_hi = float(np.percentile(a, 99.5))
        exp_low = float(np.mean(a < p_lo) * 100.0)
        exp_high = float(np.mean(a > p_hi) * 100.0)
        assert abs(stats["pct_clipped_low"] - exp_low) < 1e-4
        assert abs(stats["pct_clipped_high"] - exp_high) < 1e-4

    def test_degenerate_flat_crop_does_not_raise(self):
        """A flat crop (p_hi == p_lo) must not divide by zero; shared edges hold."""
        import numpy as np
        import napari_epicycle.data_preparation.controller as ctrl_mod

        flat = np.full((2, 4, 4), 7.0, np.float32)
        entries = ctrl_mod.build_preview_histogram_entries(
            ["c"], ("#111111",), [flat], [np.zeros((2, 4, 4), np.float32)])
        idx, name, lut, bc, ac, edges, stats = entries[0]
        assert len(bc) == 256 and len(edges) == 257
        assert np.all(np.isfinite(np.asarray(bc, dtype=float)))
        assert stats["pct_clipped_low"] == 0.0
        assert stats["pct_clipped_high"] == 0.0

    def test_degenerate_flag_marks_stats_honestly(self):
        """A degenerate crop (percentile_normalize collapses it to 0) is flagged
        ``stats["degenerate"] = True``; a normal crop is ``False`` — the readout
        keys off this so it never mislabels the min-max fallback as p0.1/p99.5."""
        import numpy as np
        import napari_epicycle.data_preparation.controller as ctrl_mod

        flat = np.full((2, 4, 4), 7.0, np.float32)
        zeros = np.zeros((2, 4, 4), np.float32)
        deg = ctrl_mod.build_preview_histogram_entries(
            ["c"], ("#111111",), [flat], [zeros])
        assert deg[0].stats["degenerate"] is True

        normal = (np.random.default_rng(2).random((2, 8, 8)) * 500).astype(np.float32)
        ok = ctrl_mod.build_preview_histogram_entries(
            ["c"], ("#111111",), [normal], [np.zeros_like(normal)])
        assert ok[0].stats["degenerate"] is False

    def test_raw_percentile_bounds_lockstep_with_normalize(self):
        """The controller's preview bounds == the ``(lo, hi)`` that
        ``percentile_normalize`` actually applies (single source of truth), on a
        normal crop AND with injected NaN — so 'before' can never drift off
        'after' if ops's percentiles/interpolation/finite-filter ever change."""
        import numpy as np
        import napari_epicycle.data_preparation.controller as ctrl_mod
        from packages.epicycle_preprocessing import ops

        rng = np.random.default_rng(3)
        normal = (rng.random((3, 12, 12)) * 4000).astype(np.float32)
        with_nan = normal.copy()
        with_nan[0, 0, 0] = np.nan
        with_nan[1, 2, 3] = np.nan

        for crop in (normal, with_nan):
            got = ctrl_mod._raw_percentile_bounds(crop)
            exp = ops.percentile_bounds(crop, 0.1, 99.5)
            assert got is not None and got == exp
            lo, hi = got
            # These bounds are EXACTLY what percentile_normalize applies: its
            # output equals clip((a - lo)/(hi - lo)) elementwise (NaN stays NaN).
            recomputed = np.clip(
                (crop.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0
            ).astype(np.float32)
            np.testing.assert_array_equal(
                ops.percentile_normalize(crop, 0.1, 99.5), recomputed)

    def test_shared_bin_edges_between_curves(self):
        import numpy as np
        import napari_epicycle.data_preparation.controller as ctrl_mod

        befores = [(np.random.rand(2, 8, 8) * 300).astype(np.float32)]
        afters = [np.clip(np.random.rand(2, 8, 8), 0, 1).astype(np.float32)]
        entries = ctrl_mod.build_preview_histogram_entries(
            ["c"], ("#111111",), befores, afters)
        idx, name, lut, bc, ac, edges, stats = entries[0]
        # Both curves are binned over the SAME edges (len matches the one array).
        assert len(bc) == len(ac) == len(edges) - 1

    def test_center_bounds_caps_xy_and_keeps_small_z(self):
        import napari_epicycle.data_preparation.controller as ctrl_mod
        b = ctrl_mod._preview_center_bounds((10, 300, 300))
        assert (b["z1"] - b["z0"]) == 10       # Z <= 64 kept whole
        assert (b["y1"] - b["y0"]) == 256      # XY capped at 256
        assert (b["x1"] - b["x0"]) == 256
        b2 = ctrl_mod._preview_center_bounds((200, 100, 100))
        assert (b2["z1"] - b2["z0"]) == 64     # a large Z is capped too


class TestPreprocessPreviewController:
    """The controller orchestration: gather inputs, hand results to the widget,
    the stale-finish guard.  Uses the real widget + controller."""

    def test_gather_inputs_center_crops_by_name(self, widget):
        import numpy as np
        ctrl = widget.controller
        cad = (np.random.rand(6, 300, 300) * 500).astype(np.float32)
        nuc = (np.random.rand(6, 300, 300) * 500).astype(np.float32)
        ctrl._source_display_channels = [("cad", cad), ("nuc", nuc)]
        widget._recipe_channel_names = ["cad", "nuc"]

        got = ctrl._gather_preview_inputs()
        assert got is not None
        names, channels, bounds_list, spacing, recipe_json = got
        assert names == ["cad", "nuc"]
        assert len(channels) == 2 and len(bounds_list) == 2
        assert (bounds_list[0]["y1"] - bounds_list[0]["y0"]) == 256
        assert isinstance(recipe_json, dict)

    def test_gather_inputs_none_without_source(self, widget):
        ctrl = widget.controller
        ctrl._source_display_channels = None
        assert ctrl._gather_preview_inputs() is None

    def test_on_preview_computed_consumes_precomputed_entries(self, widget):
        """The histograms are now built OFF the main thread (in the worker); the
        finish handler CONSUMES the pre-computed entries and hands them straight
        to the widget WITHOUT recomputing (the main-thread freeze fix)."""
        import numpy as np
        import napari_epicycle.data_preparation.controller as ctrl_mod
        ctrl = widget.controller
        captured = []
        widget.set_recipe_histograms = lambda entries: captured.append(entries)

        befores = [(np.random.rand(2, 4, 4) * 500).astype(np.float32),
                   (np.random.rand(2, 4, 4) * 500).astype(np.float32)]
        afters = [np.ones((2, 4, 4), np.float32),
                  np.ones((2, 4, 4), np.float32)]
        # The worker would have built these; the handler must pass them through
        # UNCHANGED (identity), never rebuild.
        entries = ctrl_mod.build_preview_histogram_entries(
            ["cad", "nuc"], ("#6abf69", "#c774c7"), befores, afters)
        ctrl._preview_token = 7
        ctrl._on_preview_computed(entries, afters, ["cad", "nuc"], token=7)

        assert len(captured) == 1
        assert captured[0] is entries, "handler must pass entries through, not rebuild"
        assert len(captured[0]) == 2
        for entry in captured[0]:
            idx, name, lut, bc, ac, edges, stats = entry
            assert len(bc) == 256 and len(ac) == 256
            assert isinstance(stats, dict) and "p_lo" in stats
        # section 6 honesty: the status still says the region is not the whole
        # volume.  The word used to be "approximate", which named nothing the
        # user could act on; it is now the one caveat the per-op advisories do
        # NOT cover (edges + resize).
        assert "still differ on the whole volume" in             widget.preprocess_status_label.text()

    def test_stale_finish_is_discarded(self, widget):
        import numpy as np
        ctrl = widget.controller
        captured = []
        widget.set_recipe_histograms = lambda entries: captured.append(entries)

        ctrl._preview_token = 9                 # the live run is #9
        ctrl._on_preview_computed(
            [np.zeros((1, 2, 2), np.float32)],
            [np.zeros((1, 2, 2), np.float32)],
            ["cad"], token=8)                   # a superseded run
        assert captured == [], "a stale result must not touch the plot"

    def test_recipe_edit_marks_preview_stale_without_recompute(self, widget):
        """A recipe-card edit (on_recipe_edited) bumps the stale-token, clears any
        ⟨preview⟩ layer, and greys the histogram — but NEVER recomputes live."""
        import numpy as np
        from napari_epicycle.data_preparation.controller import (
            build_preview_histogram_entries)
        ctrl = widget.controller

        raw = (np.random.rand(2, 8, 8) * 500).astype(np.float32)
        after = np.clip(np.random.rand(2, 8, 8), 0, 1).astype(np.float32)
        entries = build_preview_histogram_entries(
            ["cad"], ("#6abf69",), [raw], [after])
        widget.set_recipe_histograms(entries)
        assert widget._recipe_plot_widget.listDataItems()   # a preview is shown

        # Spy — set_recipe_histograms must NOT fire again (no live recompute).
        recompute = []
        orig = widget.set_recipe_histograms
        widget.set_recipe_histograms = lambda e: recompute.append(e)

        tok0 = getattr(ctrl, "_preview_token", 0)
        ctrl._selected_source_path = None       # skip persistence; invalidate only
        ctrl.on_recipe_edited()

        assert ctrl._preview_token == tok0 + 1          # stale-token bumped
        assert widget._recipe_hist_stale is True        # histogram greyed
        assert "re-preview" in widget.recipe_histogram_hint.text().lower()
        assert recompute == [], "recipe edit must NOT recompute the preview"
        widget.set_recipe_histograms = orig

    def test_end_to_end_worker_run_to_widget(self, widget, monkeypatch):
        """crop -> execute -> histogram -> set_recipe_histograms, driven through
        the real worker.run() synchronously + the finished handler."""
        import numpy as np
        import napari_epicycle.data_preparation.controller as ctrl_mod

        execute_calls = []

        def fake_execute(recipe, crops, spacing, bounds_per_channel=None):
            execute_calls.append(recipe)
            return [np.asarray(c, dtype=np.float32) for c in crops], spacing, {}

        monkeypatch.setattr(ctrl_mod, "_pp_execute", fake_execute)

        ctrl = widget.controller
        cad = np.random.rand(4, 300, 300).astype(np.float32)
        nuc = np.random.rand(4, 300, 300).astype(np.float32)
        ctrl._source_display_channels = [("cad", cad), ("nuc", nuc)]
        widget._recipe_channel_names = ["cad", "nuc"]

        names, channels, bounds_list, spacing, recipe_json = \
            ctrl._gather_preview_inputs()
        worker = ctrl_mod._PreprocPreviewWorker(
            recipe_json, channels, bounds_list, spacing, names,
            ("#6abf69", "#c774c7"))

        # The WORKER emits pre-built entries (histograms built off the main
        # thread); the handler must NOT recompute — spy on the pure builder so a
        # regression that moves the compute back onto the finish handler is caught.
        main_thread_builds = []
        real_build = ctrl_mod.build_preview_histogram_entries

        emitted = {}

        def _spy_finished(entries, afters, hist_ms):
            emitted["entries"] = entries
            emitted["hist_ms"] = hist_ms
            # From here on we are on the "main thread" — no build may happen.
            monkeypatch.setattr(
                ctrl_mod, "build_preview_histogram_entries",
                lambda *a, **k: main_thread_builds.append(a) or real_build(*a, **k))
            ctrl._on_preview_computed(entries, afters, names, 1)

        captured = {}
        widget.set_recipe_histograms = lambda entries: captured.__setitem__(
            "entries", entries)
        ctrl._preview_token = 1
        worker.finished.connect(_spy_finished)
        worker.run()                            # synchronous (same thread)

        assert execute_calls, "execute must run"
        # The worker emitted ready-made entries (256-bin counts) — the worker path.
        assert "entries" in emitted and len(emitted["entries"]) == 2
        for entry in emitted["entries"]:
            assert len(entry.before_counts) == 256 and len(entry.after_counts) == 256
        assert isinstance(emitted["hist_ms"], float)
        # The handler passed them straight through — no main-thread histogram build.
        assert captured.get("entries") is emitted["entries"]
        assert main_thread_builds == [], \
            "histograms must be built in the worker, not on the finish handler"


class TestPreprocessPreviewPlot:
    """set_recipe_histograms draws the SELECTED channel's raw→processed curves,
    the channel selector switches channels without recompute, and the Log-Y /
    Raw-DN toggles + stale-marking + empty state all behave."""

    def _entries(self, n):
        import numpy as np
        from napari_epicycle.data_preparation.controller import PreviewHistEntry
        edges = np.linspace(-0.1, 1.1, 257)
        luts = ("#6abf69", "#c774c7", "#5ba4cf", "#d4a843")
        out = []
        for i in range(n):
            bc = np.arange(256) + i          # distinguishable per channel
            ac = np.arange(256)[::-1] + i
            stats = {
                "p_lo": 10.0 + 90.0 * i, "p_hi": 20.0 + 180.0 * i,
                "pct_clipped_low": 0.5 * (i + 1),
                "pct_clipped_high": 0.7 * (i + 1),
                "median_before_u": 0.30 + 0.05 * i,
                "median_after_u": 0.40 + 0.05 * i,
            }
            out.append(PreviewHistEntry(i, f"ch{i}", luts[i], bc, ac, edges, stats))
        return out

    def test_draws_selected_channel_two_curves_and_hides_hint(self, widget):
        widget.set_recipe_histograms(self._entries(2))
        assert widget.recipe_histogram_hint.isHidden()
        # SELECTED channel only: 2 curves (raw + processed), NOT all channels.
        items = widget._recipe_plot_widget.listDataItems()
        assert len(items) == 2

    def test_x_range_framed_to_fixed_window_after_set_data(self, widget):
        """The plot must show the WHOLE fixed intensity frame [-0.1, 1.1] right
        after set-data — no leftover zoom (the "click A to see the plot" fix).
        X auto-range is off (frame is fixed); Y auto-range stays on."""
        widget.set_recipe_histograms(self._entries(2))
        vb = widget._recipe_plot_widget.getViewBox()
        (x0, x1), _y = vb.viewRange()
        # ~[-0.1, 1.1] with a hair of padding (padding=0.02 → about [-0.12, 1.12]).
        assert -0.16 <= x0 <= -0.04, x0
        assert 1.04 <= x1 <= 1.16, x1
        # X fixed, Y auto (state values: False for X, truthy for Y).
        autox, autoy = vb.state["autoRange"]
        assert not autox
        assert autoy
        # Holds across the Log-Y toggle (linear counts are much taller).
        widget.recipe_hist_logy_toggle.setChecked(False)
        (x0b, x1b), _y2 = vb.viewRange()
        assert -0.16 <= x0b <= -0.04 and 1.04 <= x1b <= 1.16

    def test_empty_reshows_hint_and_clears(self, widget):
        widget.set_recipe_histograms(self._entries(1))
        assert widget.recipe_histogram_hint.isHidden()
        widget.set_recipe_histograms([])
        assert not widget.recipe_histogram_hint.isHidden()   # hint back
        assert widget._recipe_plot_widget.listDataItems() == []
        assert not widget.recipe_hist_channel_bar.isVisible()
        assert not widget.recipe_hist_stat_label.isVisible()

    def test_redraw_replaces_not_accumulates(self, widget):
        widget.set_recipe_histograms(self._entries(2))
        widget.set_recipe_histograms(self._entries(1))       # 1 channel now
        # Still the selected channel × 2 curves (the first draw was cleared).
        assert len(widget._recipe_plot_widget.listDataItems()) == 2

    def test_channel_selector_switches_without_recompute(self, widget):
        """Clicking a channel tab shows THAT channel's cached before/after — no
        recompute (set_recipe_histograms is not re-called), reflected in the
        factual stat read-out switching to the new channel's raw range."""
        widget.set_recipe_histograms(self._entries(2))
        assert len(widget._recipe_hist_channel_buttons) == 2
        # Channel 0 is active by default: its raw range (10–20) is in the readout.
        assert "10" in widget.recipe_hist_stat_label.text()
        # Switch to channel 1 (raw range 100–200) with a real click (INSTANT — the
        # cached entries are re-read; set_recipe_histograms is NOT called again).
        widget._recipe_hist_channel_buttons[1].click()
        assert widget._recipe_hist_selected_channel == 1
        assert widget._recipe_hist_channel_buttons[1].isChecked()
        assert not widget._recipe_hist_channel_buttons[0].isChecked()
        txt = widget.recipe_hist_stat_label.text()
        assert "100" in txt and "200" in txt
        # Still exactly 2 curves (the selected channel), not an overlay of both.
        assert len(widget._recipe_plot_widget.listDataItems()) == 2

    def test_log_y_default_on_and_toggle_to_linear(self, widget):
        widget.set_recipe_histograms(self._entries(1))
        assert widget.recipe_hist_logy_toggle.isChecked()    # default ON
        left = widget._recipe_plot_widget.getAxis("left").labelText.lower()
        assert "log" in left
        widget.recipe_hist_logy_toggle.setChecked(False)     # -> linear, redraws
        left2 = widget._recipe_plot_widget.getAxis("left").labelText.lower()
        assert "log" not in left2
        assert len(widget._recipe_plot_widget.listDataItems()) == 2

    def test_x_axis_label_toggle_normalized_vs_raw(self, widget):
        widget.set_recipe_histograms(self._entries(1))
        bottom = widget._recipe_plot_widget.getAxis("bottom").labelText.lower()
        assert "normalized" in bottom                        # default
        widget.recipe_hist_rawaxis_toggle.setChecked(True)   # -> raw DN, redraws
        bottom2 = widget._recipe_plot_widget.getAxis("bottom").labelText.lower()
        assert "raw" in bottom2
        assert len(widget._recipe_plot_widget.listDataItems()) == 2

    def test_stat_readout_is_factual_only(self, widget):
        widget.set_recipe_histograms(self._entries(1))
        txt = widget.recipe_hist_stat_label.text().lower()
        assert "clipped" in txt and "median" in txt
        assert "preview region only" in txt
        # No mean / SD / SNR / composite score wording.
        for banned in ("snr", "std", "mean", "score", "improved"):
            assert banned not in txt

    def test_mark_stale_greys_without_recompute(self, widget):
        widget.set_recipe_histograms(self._entries(1))
        widget.mark_recipe_preview_stale()
        assert widget._recipe_hist_stale is True
        assert "re-preview" in widget.recipe_histogram_hint.text().lower()
        assert not widget.recipe_histogram_hint.isHidden()
        for item in widget._recipe_plot_widget.listDataItems():
            assert item.opacity() < 1.0

    def test_stale_survives_view_toggles_and_channel_switch(self, widget):
        """A Log-Y / Raw-DN toggle or channel switch AFTER a recipe edit must
        stay VISIBLY stale (greyed + 're-Preview'); only a fresh Preview clears
        it.  Guards the regression where `_redraw_recipe_histogram` cleared the
        stale flag at its top and silently re-plotted at full opacity."""
        widget.set_recipe_histograms(self._entries(2))
        widget.mark_recipe_preview_stale()
        assert widget._recipe_hist_stale is True

        # Log-Y toggle → redraw from cache; stale must persist.
        widget.recipe_hist_logy_toggle.setChecked(False)
        assert widget._recipe_hist_stale is True
        assert "re-preview" in widget.recipe_histogram_hint.text().lower()
        assert not widget.recipe_histogram_hint.isHidden()
        assert all(it.opacity() < 1.0
                   for it in widget._recipe_plot_widget.listDataItems())

        # Raw-DN toggle → redraw; still stale.
        widget.recipe_hist_rawaxis_toggle.setChecked(True)
        assert widget._recipe_hist_stale is True
        assert all(it.opacity() < 1.0
                   for it in widget._recipe_plot_widget.listDataItems())

        # Channel switch → redraw the other channel; still stale + greyed.
        widget._recipe_hist_channel_buttons[1].click()
        assert widget._recipe_hist_stale is True
        assert "re-preview" in widget.recipe_hist_stat_label.text().lower()
        assert all(it.opacity() < 1.0
                   for it in widget._recipe_plot_widget.listDataItems())

        # A FRESH Preview result clears stale + restores full opacity.
        widget.set_recipe_histograms(self._entries(2))
        assert widget._recipe_hist_stale is False
        assert all(it.opacity() > 0.9
                   for it in widget._recipe_plot_widget.listDataItems())

    def test_degenerate_readout_is_honest_not_mislabeled(self, widget):
        """A degenerate crop shows an HONEST note ('normalize collapses this
        region to 0'), NOT the mislabeled p0.1/p99.5 + '0% clipped' line."""
        import numpy as np
        from napari_epicycle.data_preparation.controller import (
            build_preview_histogram_entries)
        flat = np.full((2, 4, 4), 7.0, np.float32)
        entries = build_preview_histogram_entries(
            ["c"], ("#6abf69",), [flat], [np.zeros((2, 4, 4), np.float32)])
        assert entries[0].stats["degenerate"] is True

        widget.set_recipe_histograms(entries)
        txt = widget.recipe_hist_stat_label.text().lower()
        assert "degenerate" in txt and "collapse" in txt
        assert "clipped" not in txt          # the mislabeled clip numbers are gone
        assert "p0.1" not in txt and "p99.5" not in txt

    def test_legend_survives_redraw(self, widget):
        """The legend must stay in the scene on EVERY redraw, not vanish after the
        first (pyqtgraph's addLegend hands back the existing, detached legend
        unless the plotItem back-ref is reset each draw)."""
        widget.set_recipe_histograms(self._entries(2))
        widget.set_recipe_histograms(self._entries(2))       # 2nd draw
        legend = widget._recipe_plot_legend
        assert legend is not None
        assert legend.scene() is not None                    # still in the scene

    def test_plot_title_retitled_and_cleared_when_empty(self, widget):
        """The title reads 'raw ... processed' (no misleading 'before → after');
        the empty state hides the title."""
        widget.set_recipe_histograms(self._entries(2))
        tl = widget._recipe_plot_widget.plotItem.titleLabel
        assert tl.isVisible()
        low = tl.text.lower()
        assert "raw" in low and "processed" in low
        assert "before" not in low
        widget.set_recipe_histograms([])
        assert not tl.isVisible()          # title hidden in the empty state


# ---------------------------------------------------------------------------
# (B2c) The WYSIWYG PREVIEW render — the processed test crop as toggleable
#       ⟨preview⟩ napari layer(s), positioned/scaled where the crop was taken.
# ---------------------------------------------------------------------------


class _PreviewLayer:
    """A minimal stand-in for a napari Image layer that records the kwargs the
    controller passes to `add_image` (name / colormap / translate / scale / …)."""

    def __init__(self, data, log=None, **kw):
        self.data = data
        self._log = log
        self.name = kw.get("name")
        self.colormap = kw.get("colormap")
        self.blending = kw.get("blending")
        self.contrast_limits = kw.get("contrast_limits")
        self.translate = kw.get("translate")
        self.scale = kw.get("scale")
        self._visible = kw.get("visible", True)

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, value):
        self._visible = value
        if self._log is not None:
            self._log.append(("visible", self.name, value))


class _RemoveLoggingLayerList(_FakeLayerList):
    """A `_FakeLayerList` that records `remove(...)` so a test can assert the
    hide-before-remove ordering."""

    def __init__(self, log):
        super().__init__()
        self._log = log

    def remove(self, layer):
        self._log.append(("remove", getattr(layer, "name", None)))
        super().remove(layer)


class _PreviewViewer:
    """A viewer whose `add_image` makes `_PreviewLayer`s and appends them to a
    (list-based) layer list — enough for the ⟨preview⟩ render/cleanup paths."""

    def __init__(self, layers=None):
        self.layers = layers if layers is not None else _FakeLayerList()
        self.mouse_drag_callbacks = []

    def add_image(self, data, **kw):
        layer = _PreviewLayer(data, **kw)
        self.layers.append(layer)
        return layer


class TestPreprocessPreviewRender:
    """Brick 2c — the processed crop is added as ⟨preview⟩ layer(s), replaced on
    re-preview, cleared on source-change / step-exit, and skipped for a stale
    finish; `_clear_preview_layers` is a tolerant hide-then-remove."""

    def _afters(self, shapes):
        import numpy as np
        return [np.linspace(0, 1, int(np.prod(s)), dtype=np.float32).reshape(s)
                for s in shapes]

    def _preview_names(self, ctrl, names):
        chans = [(n, None) for n in names]
        ctrl._source_display_channels = chans

    def test_adds_one_preview_layer_per_channel(self, widget):
        import napari_epicycle.data_preparation.controller as ctrl_mod
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        widget.set_recipe_histograms = lambda entries: None
        self._preview_names(ctrl, ["cad", "nuc"])
        ctrl._preview_bounds = [
            ctrl_mod._preview_center_bounds((6, 300, 300)),
            ctrl_mod._preview_center_bounds((6, 300, 300)),
        ]
        afters = self._afters([(6, 256, 256), (6, 256, 256)])
        ctrl._preview_token = 3
        ctrl._on_preview_computed(afters, afters, ["cad", "nuc"], token=3)

        prev = [l for l in ctrl.viewer.layers
                if str(getattr(l, "name", "")).startswith("⟨preview⟩")]
        assert len(prev) == 2
        assert {l.name for l in prev} == {"⟨preview⟩ cad", "⟨preview⟩ nuc"}
        # Contrast pinned to [0,1] + additive so it overlays the source.
        assert all(tuple(l.contrast_limits) == (0.0, 1.0) for l in prev)
        assert all(l.blending == "additive" for l in prev)
        assert ctrl._preview_layers == prev
        # Status advertises the rendered layers.
        assert "preview" in widget.preprocess_status_label.text().lower()
        assert "layer" in widget.preprocess_status_label.text().lower()

    def test_colormap_matches_source_channel(self, widget):
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        widget.set_recipe_histograms = lambda entries: None
        self._preview_names(ctrl, ["cad", "nuc"])
        ctrl._preview_bounds = [None, None]
        afters = self._afters([(2, 4, 4), (2, 4, 4)])
        ctrl._preview_token = 1
        ctrl._on_preview_computed(afters, afters, ["cad", "nuc"], token=1)
        by_name = {l.name: l for l in ctrl.viewer.layers}
        # Mirrors `_add_source_display_layers`: ch0->magenta, ch1->green.
        assert by_name["⟨preview⟩ cad"].colormap == "magenta"
        assert by_name["⟨preview⟩ nuc"].colormap == "green"

    def test_single_channel_source_is_gray(self, widget):
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        widget.set_recipe_histograms = lambda entries: None
        self._preview_names(ctrl, ["only"])
        ctrl._preview_bounds = [None]
        afters = self._afters([(2, 4, 4)])
        ctrl._preview_token = 1
        ctrl._on_preview_computed(afters, afters, ["only"], token=1)
        layer = [l for l in ctrl.viewer.layers
                 if str(l.name).startswith("⟨preview⟩")][0]
        assert layer.colormap == "gray"

    def test_no_geometry_translate_only_scale_is_unit(self, widget):
        import napari_epicycle.data_preparation.controller as ctrl_mod
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        widget.set_recipe_histograms = lambda entries: None
        self._preview_names(ctrl, ["cad"])
        bounds = ctrl_mod._preview_center_bounds((6, 300, 300))
        ctrl._preview_bounds = [bounds]
        # processed shape == crop shape -> a same-shape (no-geometry) recipe.
        afters = self._afters([(6, 256, 256)])
        ctrl._preview_token = 1
        ctrl._on_preview_computed(afters, afters, ["cad"], token=1)
        layer = ctrl._preview_layers[0]
        assert tuple(layer.translate) == (bounds["z0"], bounds["y0"], bounds["x0"])
        assert tuple(layer.scale) == (1.0, 1.0, 1.0)

    def test_geometry_rescale_sets_scale_factor(self, widget):
        import napari_epicycle.data_preparation.controller as ctrl_mod
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        widget.set_recipe_histograms = lambda entries: None
        self._preview_names(ctrl, ["cad"])
        bounds = ctrl_mod._preview_center_bounds((6, 300, 300))  # crop 6x256x256
        ctrl._preview_bounds = [bounds]
        # A geometry step halved every axis -> the layer must span the SAME world
        # box, i.e. scale = crop/processed = 2 on each axis, translate unchanged.
        afters = self._afters([(3, 128, 128)])
        ctrl._preview_token = 1
        ctrl._on_preview_computed(afters, afters, ["cad"], token=1)
        layer = ctrl._preview_layers[0]
        assert tuple(layer.translate) == (bounds["z0"], bounds["y0"], bounds["x0"])
        assert tuple(layer.scale) == (2.0, 2.0, 2.0)

    def test_second_preview_replaces_not_accumulates(self, widget):
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        widget.set_recipe_histograms = lambda entries: None
        self._preview_names(ctrl, ["cad", "nuc"])
        ctrl._preview_bounds = [None, None]
        afters = self._afters([(2, 4, 4), (2, 4, 4)])
        ctrl._preview_token = 1
        ctrl._on_preview_computed(afters, afters, ["cad", "nuc"], token=1)
        first = list(ctrl._preview_layers)
        ctrl._on_preview_computed(afters, afters, ["cad", "nuc"], token=1)
        prev = [l for l in ctrl.viewer.layers
                if str(getattr(l, "name", "")).startswith("⟨preview⟩")]
        assert len(prev) == 2                       # no accumulation
        assert all(l not in prev for l in first)    # the first set was removed

    def test_stale_finish_adds_no_layers(self, widget):
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        widget.set_recipe_histograms = lambda entries: None
        self._preview_names(ctrl, ["cad"])
        ctrl._preview_bounds = [None]
        afters = self._afters([(2, 4, 4)])
        ctrl._preview_token = 9                       # the live run is #9
        ctrl._on_preview_computed(afters, afters, ["cad"], token=8)  # superseded
        assert ctrl._preview_layers == []
        assert not any(str(getattr(l, "name", "")).startswith("⟨preview⟩")
                       for l in ctrl.viewer.layers)

    def test_unload_source_clears_preview_layers(self, widget):
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        ctrl.box_tool = None                          # skip the box-tool teardown
        widget.set_recipe_histograms = lambda entries: None
        self._preview_names(ctrl, ["cad"])
        ctrl._preview_bounds = [None]
        ctrl._preview_token = 1
        ctrl._on_preview_computed(
            self._afters([(2, 4, 4)]), self._afters([(2, 4, 4)]),
            ["cad"], token=1)
        assert ctrl._preview_layers                    # a layer is present
        ctrl.unload_source()
        assert ctrl._preview_layers == []
        assert not any(str(getattr(l, "name", "")).startswith("⟨preview⟩")
                       for l in ctrl.viewer.layers)

    def test_step_exit_clears_preview_layers(self, widget):
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        ctrl.box_tool = None                          # skip the box-tool teardown
        widget.set_recipe_histograms = lambda entries: None
        self._preview_names(ctrl, ["cad"])
        ctrl._preview_bounds = [None]
        ctrl._preview_token = 1
        ctrl._on_preview_computed(
            self._afters([(2, 4, 4)]), self._afters([(2, 4, 4)]),
            ["cad"], token=1)
        assert ctrl._preview_layers
        ctrl.on_step_exit()
        assert ctrl._preview_layers == []

    def test_source_switch_wired_to_clear(self, widget):
        """The new-source display path (`_on_source_loaded`) must drop the prior
        ⟨preview⟩ render so a stale crop never sits over new data.  It now routes
        through the single `_teardown_preview` entry point (which clears the
        render + destroys the box + collapses the ② controls)."""
        import inspect
        ctrl = widget.controller
        src = inspect.getsource(ctrl._on_source_loaded)
        assert "_teardown_preview" in src

    def test_clear_preview_layers_hides_before_remove(self, widget):
        ctrl = widget.controller
        log = []
        layers = _RemoveLoggingLayerList(log)
        ctrl.viewer = _PreviewViewer(layers=layers)
        layer = _PreviewLayer(None, log=log, name="⟨preview⟩ cad")
        layers.append(layer)
        ctrl._preview_layers = [layer]
        ctrl._clear_preview_layers()
        # visible=False must be logged BEFORE the remove (freed-texture safety).
        assert log == [("visible", "⟨preview⟩ cad", False),
                       ("remove", "⟨preview⟩ cad")]
        assert ctrl._preview_layers == []

    def test_clear_preview_layers_idempotent_and_tolerant(self, widget):
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        # A layer that was ALREADY removed from the viewer (user deleted it):
        orphan = _PreviewLayer(None, name="⟨preview⟩ gone")
        ctrl._preview_layers = [orphan]               # tracked but not in viewer
        ctrl._clear_preview_layers()                  # must not raise
        assert ctrl._preview_layers == []
        ctrl._clear_preview_layers()                  # idempotent second call
        assert ctrl._preview_layers == []

    # --- C#1 (release hardening): a STALE terminal emit must still release the
    #     busy refcount.  Before the fix both handlers early-returned ABOVE the
    #     only `_set_step1_busy(False)`, so leaving Step 1 (or editing the
    #     recipe) mid-Preview left `_busy_depth` at 1 for the rest of the
    #     session — ②/③/④ dead with `_step1_busy` reading "idle".

    def test_stale_finish_releases_the_busy_lock(self, widget):
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        widget.set_recipe_histograms = lambda entries: None
        self._preview_names(ctrl, ["cad"])
        ctrl._preview_bounds = [None]
        afters = self._afters([(2, 4, 4)])
        ctrl._preview_token = 9
        ctrl._set_step1_busy(True)                    # as `on_preview_recipe` did
        ctrl._on_preview_computed(afters, afters, ["cad"], token=8)  # superseded
        assert ctrl._busy_depth == 0
        assert ctrl._step1_busy is False
        assert widget.load_source_button.isEnabled()

    def test_stale_error_releases_the_busy_lock(self, widget):
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        ctrl._preview_token = 4
        ctrl._set_step1_busy(True)
        ctrl._on_preview_error("boom", token=3)       # superseded
        assert ctrl._busy_depth == 0
        assert ctrl._step1_busy is False
        assert widget.load_source_button.isEnabled()

    def test_a_stale_finish_under_a_live_run_leaves_the_bar_spinning(
            self, widget):
        """C#1 follow-up: `_release_preview_busy` dropped the refcount correctly
        but hid the bar unconditionally.  A superseded run finishing while a
        SECOND Preview is still computing then rested the bar under the live
        run — the surface stayed (correctly) locked with no visible reason.
        The bar belongs to the outermost op: rest it only at depth 0."""
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        widget.set_recipe_histograms = lambda entries: None
        self._preview_names(ctrl, ["cad"])
        ctrl._preview_bounds = [None]
        afters = self._afters([(2, 4, 4)])

        ctrl._set_step1_busy(True)                 # Preview A launches
        ctrl._preview_token = 1
        ctrl._set_step1_busy(True)                 # Preview B launches
        ctrl._preview_token = 2
        widget._busy_show("Preview — test region…")

        ctrl._on_preview_computed(afters, afters, ["cad"], token=1)   # A: stale
        assert ctrl._busy_depth == 1
        assert widget.source_progress_bar.maximum() == 0,             "the bar rested while Preview B was still running"

        ctrl._on_preview_computed(afters, afters, ["cad"], token=2)   # B: live
        assert ctrl._busy_depth == 0
        assert widget.source_progress_bar.maximum() == 1

    def test_a_stale_error_under_a_live_run_leaves_the_bar_spinning(
            self, widget):
        ctrl = widget.controller
        ctrl._set_step1_busy(True)
        ctrl._set_step1_busy(True)
        ctrl._preview_token = 5
        widget._busy_show("Preview — test region…")
        ctrl._on_preview_error("boom", token=4)                       # stale
        assert ctrl._busy_depth == 1
        assert widget.source_progress_bar.maximum() == 0

    def test_live_finish_releases_exactly_once(self, widget):
        """The normal path is unchanged: one release, never a double-decrement
        (which would unlock a legitimately-nested outer op)."""
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        widget.set_recipe_histograms = lambda entries: None
        self._preview_names(ctrl, ["cad"])
        ctrl._preview_bounds = [None]
        afters = self._afters([(2, 4, 4)])
        ctrl._preview_token = 1
        ctrl._set_step1_busy(True)                    # outer op
        ctrl._set_step1_busy(True)                    # the Preview itself
        ctrl._on_preview_computed(afters, afters, ["cad"], token=1)
        assert ctrl._busy_depth == 1                  # outer op still holds it
        assert ctrl._step1_busy is True
        ctrl._set_step1_busy(False)
        assert ctrl._busy_depth == 0

    def test_live_error_releases_exactly_once(self, widget):
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        ctrl._preview_token = 2
        ctrl._set_step1_busy(True)
        ctrl._set_step1_busy(True)
        ctrl._on_preview_error("boom", token=2)
        assert ctrl._busy_depth == 1
        assert "boom" in widget.preprocess_status_label.text()

    # --- C#10: `self._preview_layers = added` used to run only AFTER the add
    #     loop, so anything raising mid-loop (here the per-layer units apply)
    #     stranded the layers already on the canvas — untracked, so
    #     `_clear_preview_layers` could not reach them and they leaked into
    #     Steps 2-5 (feeds the "annotator takes ALL Image layers" hazard).

    def test_midloop_failure_still_tracks_the_added_layers(self, widget,
                                                           monkeypatch):
        import napari_epicycle.data_preparation.controller as ctrl_mod
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        self._preview_names(ctrl, ["cad", "nuc"])
        ctrl._preview_bounds = [None, None]
        calls = {"n": 0}

        def _boom(layer, units):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("units setter blew up")

        monkeypatch.setattr(ctrl_mod, "_apply_layer_units", _boom)
        afters = self._afters([(2, 4, 4), (2, 4, 4)])
        with pytest.raises(RuntimeError):
            ctrl._add_preview_layers(afters, ["cad", "nuc"])
        # Both reached the canvas; both must be tracked so cleanup can reach them.
        on_canvas = [l for l in ctrl.viewer.layers
                     if str(getattr(l, "name", "")).startswith("⟨preview⟩")]
        assert len(on_canvas) == 2
        assert ctrl._preview_layers == on_canvas
        ctrl._clear_preview_layers()
        assert not any(str(getattr(l, "name", "")).startswith("⟨preview⟩")
                       for l in ctrl.viewer.layers)

    # --- C#12: bumping the stale token discards the RESULT but used to let the
    #     (possibly minutes-long) compute run to completion.  Ask the worker to
    #     stop as well — it checks the flag before it emits.

    def test_bumping_the_token_asks_the_worker_to_cancel(self, widget):
        ctrl = widget.controller

        class _W:
            def __init__(self):
                self.cancelled = False

            def request_cancel(self):
                self.cancelled = True

        worker = _W()
        ctrl._preview_worker = worker
        ctrl._bump_preview_token()
        assert worker.cancelled is True

    def test_reclicking_preview_cancels_the_in_flight_worker(self, widget,
                                                             monkeypatch):
        """The most common supersede of all — clicking Preview again — went
        through an INLINE `self._preview_token += 1`, bypassing
        `_bump_preview_token` and so never asking the first worker to stop.  It
        ran its minutes-long compute to completion on a result the token would
        then discard."""
        import numpy as np
        import napari_epicycle.data_preparation.controller as ctrl_mod
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        cad = (np.random.rand(6, 300, 300) * 500).astype(np.float32)
        ctrl._source_display_channels = [("cad", cad)]
        widget._recipe_channel_names = ["cad"]

        class _W:
            def __init__(self):
                self.cancelled = False

            def request_cancel(self):
                self.cancelled = True

        first = _W()
        ctrl._preview_worker = first
        # Stop before a real QThread starts; the bump has already happened.
        monkeypatch.setattr(
            ctrl_mod, "_PreprocPreviewWorker",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no thread")))
        ctrl.on_preview_recipe()
        assert first.cancelled is True,             "re-clicking Preview did not ask the running worker to stop"

    def test_reclicking_preview_still_advances_the_token(self, widget,
                                                         monkeypatch):
        import numpy as np
        import napari_epicycle.data_preparation.controller as ctrl_mod
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        cad = (np.random.rand(6, 300, 300) * 500).astype(np.float32)
        ctrl._source_display_channels = [("cad", cad)]
        widget._recipe_channel_names = ["cad"]
        monkeypatch.setattr(
            ctrl_mod, "_PreprocPreviewWorker",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no thread")))
        before = ctrl._preview_token
        ctrl.on_preview_recipe()
        assert ctrl._preview_token == before + 1

    def test_bumping_the_token_is_tolerant_of_no_worker(self, widget):
        ctrl = widget.controller
        ctrl._preview_worker = None
        before = ctrl._preview_token
        ctrl._bump_preview_token()                    # must not raise
        assert ctrl._preview_token == before + 1

    def test_teardown_preview_cancels_the_in_flight_compute(self, widget):
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        ctrl.box_tool = None

        class _W:
            def __init__(self):
                self.cancelled = False

            def request_cancel(self):
                self.cancelled = True

        worker = _W()
        ctrl._preview_worker = worker
        ctrl._teardown_preview()
        assert worker.cancelled is True

    # --- C#6: the ⟨preview⟩ layer is additive and pinned to contrast (0,1), so
    #     it sits OVER the still-visible source and a no-op recipe "looks
    #     processed".  The status used to ask the user to hide the source by
    #     hand.  Hide the matching source (or ⟨processed⟩) layer for them, and
    #     put it back when the preview goes away.

    def test_preview_hides_the_matching_source_layer(self, widget):
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        widget.set_recipe_histograms = lambda entries: None
        self._preview_names(ctrl, ["cad", "nuc"])
        ctrl._preview_bounds = [None, None]
        src = [_PreviewLayer(None, name="cad"), _PreviewLayer(None, name="nuc")]
        for l in src:
            ctrl.viewer.layers.append(l)
        afters = self._afters([(2, 4, 4), (2, 4, 4)])
        ctrl._preview_token = 1
        ctrl._on_preview_computed(afters, afters, ["cad", "nuc"], token=1)
        assert [l.visible for l in src] == [False, False]
        ctrl._clear_preview_layers()
        assert [l.visible for l in src] == [True, True]

    def test_preview_hides_the_processed_layer_when_that_is_what_is_shown(
            self, widget):
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        widget.set_recipe_histograms = lambda entries: None
        self._preview_names(ctrl, ["cad"])
        ctrl._preview_bounds = [None]
        proc = _PreviewLayer(None, name="⟨processed⟩ cad")
        ctrl.viewer.layers.append(proc)
        ctrl._preview_token = 1
        ctrl._on_preview_computed(self._afters([(2, 4, 4)]),
                                  self._afters([(2, 4, 4)]), ["cad"], token=1)
        assert proc.visible is False
        ctrl._clear_preview_layers()
        assert proc.visible is True

    def test_a_failed_render_puts_the_hidden_source_back(self, widget,
                                                         monkeypatch):
        """`_on_preview_computed` deliberately swallows a render failure so a
        drawn histogram survives it — but the auto-hide (C#6) had already run,
        so the user was left with their source channel invisible, no ⟨preview⟩
        layer, and nothing saying why.  Restore on that path too."""
        import napari_epicycle.data_preparation.controller as ctrl_mod
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        widget.set_recipe_histograms = lambda entries: None
        self._preview_names(ctrl, ["cad", "nuc"])
        ctrl._preview_bounds = [None, None]
        src = [_PreviewLayer(None, name="cad"), _PreviewLayer(None, name="nuc")]
        for l in src:
            ctrl.viewer.layers.append(l)

        # Channel 0 renders + hides its source; channel 1 blows up mid-loop.
        calls = {"n": 0}

        def _boom(layer, units):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("units setter blew up")

        monkeypatch.setattr(ctrl_mod, "_apply_layer_units", _boom)
        afters = self._afters([(2, 4, 4), (2, 4, 4)])
        ctrl._preview_token = 1
        ctrl._on_preview_computed(afters, afters, ["cad", "nuc"], token=1)
        assert calls["n"] == 2                     # the render really did fail
        assert [l.visible for l in src] == [True, True],             "a source layer stayed hidden after a failed render"
        assert ctrl._preview_hidden_layers == []

    def test_a_layer_the_user_already_hid_stays_hidden(self, widget):
        """Restore means "put back what WE changed", not "turn everything on"."""
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        widget.set_recipe_histograms = lambda entries: None
        self._preview_names(ctrl, ["cad"])
        ctrl._preview_bounds = [None]
        src = _PreviewLayer(None, name="cad")
        src.visible = False                       # the user hid it themselves
        ctrl.viewer.layers.append(src)
        ctrl._preview_token = 1
        ctrl._on_preview_computed(self._afters([(2, 4, 4)]),
                                  self._afters([(2, 4, 4)]), ["cad"], token=1)
        ctrl._clear_preview_layers()
        assert src.visible is False

    def test_preview_status_no_longer_asks_the_user_to_hide_the_source(
            self, widget):
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        widget.set_recipe_histograms = lambda entries: None
        self._preview_names(ctrl, ["cad"])
        ctrl._preview_bounds = [None]
        ctrl.viewer.layers.append(_PreviewLayer(None, name="cad"))
        ctrl._preview_token = 1
        ctrl._on_preview_computed(self._afters([(2, 4, 4)]),
                                  self._afters([(2, 4, 4)]), ["cad"], token=1)
        assert "toggle the SOURCE layer off" not in             widget.preprocess_status_label.text()

    # --- C#8: the streaming (prepared-source) apply REFUSES a CLAHE / bleach /
    #     resize recipe, but Preview happily rendered it and said nothing — the
    #     user only learns at Apply.  Warn at Preview, using the SAME predicate
    #     the apply gate uses so the two can never disagree.

    def _lazy_channels(self, ctrl, n=1):
        class _Lazy:
            shape = (4, 8, 8)
            ndim = 3

            def __getitem__(self, k):
                import numpy as np
                return np.zeros((4, 8, 8), dtype=np.float32)[k]

        ctrl._source_display_channels = [(f"c{i}", _Lazy()) for i in range(n)]

    def test_preview_warns_when_the_recipe_cannot_stream(self, widget,
                                                         monkeypatch):
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        widget.set_recipe_histograms = lambda entries: None
        self._lazy_channels(ctrl)
        widget._recipe_channel_names = ["c0"]
        monkeypatch.setattr(type(ctrl), "_recipe_has_stream_remainder",
                            lambda self, *a, **k: True)
        monkeypatch.setattr(type(ctrl), "_should_stream_process",
                            lambda self, chans: True)
        ctrl._resolve_preview_stream_warning({"channels": [[]]}, [object()],
                                             (1.0, 1.0, 1.0))
        assert ctrl._preview_stream_warning
        # And it reaches the user on the finish status.
        self._preview_names(ctrl, ["c0"])
        ctrl._preview_bounds = [None]
        ctrl._preview_token = 1
        ctrl._on_preview_computed(self._afters([(2, 4, 4)]),
                                  self._afters([(2, 4, 4)]), ["c0"], token=1)
        assert ctrl._preview_stream_warning in             widget.preprocess_status_label.text()

    def test_no_warning_for_an_in_memory_source(self, widget, monkeypatch):
        """A resident numpy source takes the in-memory apply path, which has no
        remainder restriction — warning there would be a lie."""
        ctrl = widget.controller
        monkeypatch.setattr(type(ctrl), "_recipe_has_stream_remainder",
                            lambda self, *a, **k: True)
        monkeypatch.setattr(type(ctrl), "_should_stream_process",
                            lambda self, chans: False)
        ctrl._resolve_preview_stream_warning({"channels": [[]]}, [object()],
                                             (1.0, 1.0, 1.0))
        assert ctrl._preview_stream_warning is None

    def test_no_warning_when_the_recipe_streams_fine(self, widget, monkeypatch):
        ctrl = widget.controller
        monkeypatch.setattr(type(ctrl), "_recipe_has_stream_remainder",
                            lambda self, *a, **k: False)
        monkeypatch.setattr(type(ctrl), "_should_stream_process",
                            lambda self, chans: True)
        ctrl._resolve_preview_stream_warning({"channels": [[]]}, [object()],
                                             (1.0, 1.0, 1.0))
        assert ctrl._preview_stream_warning is None

    def test_launch_resolves_the_warning(self, widget, monkeypatch):
        """Wiring: `on_preview_recipe` must actually ASK, with the recipe and
        the channels it is about to hand the worker.  (A source-text search for
        the method name proved nothing — it stays green for a call that is
        commented out or unreachable.)"""
        import numpy as np
        import napari_epicycle.data_preparation.controller as ctrl_mod
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        cad = (np.random.rand(6, 300, 300) * 500).astype(np.float32)
        ctrl._source_display_channels = [("cad", cad)]
        widget._recipe_channel_names = ["cad"]

        seen = {}

        def _spy(self, recipe_json, channels, spacing):
            seen["recipe"] = recipe_json
            seen["n_channels"] = len(channels)
            seen["spacing"] = tuple(spacing)
            self._preview_stream_warning = None

        monkeypatch.setattr(type(ctrl), "_resolve_preview_stream_warning", _spy)
        monkeypatch.setattr(
            ctrl_mod, "_PreprocPreviewWorker",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no thread")))
        ctrl.on_preview_recipe()

        assert seen["n_channels"] == 1
        assert seen["spacing"] == tuple(ctrl._preview_spacing())
        assert "channels" in (seen["recipe"] or {})

    # --- C#3 / C#4 wiring: the fidelity notes must reach the user's status.

    def test_fidelity_notes_reach_the_status(self, widget):
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        widget.set_recipe_histograms = lambda entries: None
        self._preview_names(ctrl, ["cad"])
        ctrl._preview_bounds = [None]
        ctrl._preview_fidelity = ["Depth-brightness (bleach) does nothing here."]
        ctrl._preview_token = 1
        ctrl._on_preview_computed(self._afters([(2, 4, 4)]),
                                  self._afters([(2, 4, 4)]), ["cad"], token=1)
        assert "Depth-brightness (bleach) does nothing here." in             widget.preprocess_status_label.text()

    def test_a_faithful_recipe_keeps_the_plain_status(self, widget):
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        widget.set_recipe_histograms = lambda entries: None
        self._preview_names(ctrl, ["cad"])
        ctrl._preview_bounds = [None]
        ctrl._preview_fidelity = []
        ctrl._preview_token = 1
        ctrl._on_preview_computed(self._afters([(2, 4, 4)]),
                                  self._afters([(2, 4, 4)]), ["cad"], token=1)
        assert "not previewable at this region size" not in             widget.preprocess_status_label.text()

    def test_launch_resolves_the_fidelity_notes(self, widget, monkeypatch):
        """Wiring: the notes are judged on the SAME crop the worker is handed,
        against the full source shape."""
        import numpy as np
        import napari_epicycle.data_preparation.controller as ctrl_mod
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        cad = (np.random.rand(6, 300, 300) * 500).astype(np.float32)
        ctrl._source_display_channels = [("cad", cad)]
        widget._recipe_channel_names = ["cad"]

        seen = {}

        def _spy(recipe_json, n_channels, crop_shape, full_shape):
            seen["crop"] = crop_shape
            seen["full"] = full_shape
            seen["n"] = n_channels
            return ["a note"]

        monkeypatch.setattr(ctrl_mod, "_preview_fidelity_notes", _spy)
        monkeypatch.setattr(
            ctrl_mod, "_PreprocPreviewWorker",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no thread")))
        ctrl.on_preview_recipe()

        assert seen["n"] == 1
        bounds = ctrl._preview_bounds[0]
        assert seen["crop"] == (
            int(bounds["z1"]) - int(bounds["z0"]),
            int(bounds["y1"]) - int(bounds["y0"]),
            int(bounds["x1"]) - int(bounds["x0"]))
        assert tuple(seen["full"]) == (6, 300, 300)
        assert ctrl._preview_fidelity == ["a note"]

    # --- Brick 2b item 5: a preview that SUCCEEDED must read as a success.
    #     The advisories (C#3/C#4/C#8) were appended by re-calling
    #     `_set_preview_status(..., ok=False)`, which repainted the whole line
    #     amber — a working preview looked like a failure.  Success stays green;
    #     advisories ride along on their own amber "⚠ " lines.

    def _advisory_setup(self, widget):
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        widget.set_recipe_histograms = lambda entries: None
        self._preview_names(ctrl, ["cad"])
        ctrl._preview_bounds = [None]
        ctrl._preview_token = 1
        return ctrl

    def test_a_successful_preview_with_advisories_stays_green(self, widget):
        ctrl = self._advisory_setup(widget)
        ctrl._preview_fidelity = ["Region too thin."]
        ctrl._on_preview_computed(self._afters([(2, 4, 4)]),
                                  self._afters([(2, 4, 4)]), ["cad"], token=1)
        style = widget.preprocess_status_label.styleSheet()
        assert "#6abf69" in style, f"success line was not green: {style}"
        text = widget.preprocess_status_label.text()
        assert "Region too thin." in text
        assert "⚠" in text, "the advisory is not marked as an advisory"

    def test_advisories_are_a_separate_line_from_the_success(self, widget):
        ctrl = self._advisory_setup(widget)
        ctrl._preview_fidelity = ["Note one.", "Note two."]
        ctrl._preview_stream_warning = "Note three."
        ctrl._on_preview_computed(self._afters([(2, 4, 4)]),
                                  self._afters([(2, 4, 4)]), ["cad"], token=1)
        text = widget.preprocess_status_label.text()
        for note in ("Note one.", "Note two.", "Note three."):
            assert note in text
        # Each advisory gets its own marked line, not one run-on sentence.
        assert text.count("⚠") == 3

    def test_a_real_failure_is_still_amber(self, widget):
        widget.set_preview_status("Preview failed: boom", ok=False)
        assert "#d4a843" in widget.preprocess_status_label.styleSheet()

    def test_the_success_line_keeps_one_honest_caveat(self, widget):
        """The blanket "CLAHE / bleach / geometry differ" copy was replaced by
        the specific C#3/C#4 notes, which dropped the two classes those notes do
        NOT cover: the region's EDGES (halo, C#5) and a resize step.  One
        sentence, always shown, covers them."""
        ctrl = self._advisory_setup(widget)
        ctrl._preview_fidelity = []
        ctrl._on_preview_computed(self._afters([(2, 4, 4)]),
                                  self._afters([(2, 4, 4)]), ["cad"], token=1)
        text = widget.preprocess_status_label.text()
        assert "Edges of the region and any resize step still differ on the "                "whole volume." in text

    def test_process_with_no_source_touches_no_preview_layers(self, widget):
        """"Process whole volume" (Brick 1b-1) must never touch the ⟨preview⟩
        render.  With no source loaded it early-returns (a themed status) — no
        exception, no ⟨preview⟩ layers added or cleared."""
        ctrl = widget.controller
        ctrl.viewer = _PreviewViewer()
        widget._on_apply_clicked()                    # no exception, no layers
        assert not any(str(getattr(l, "name", "")).startswith("⟨preview⟩")
                       for l in ctrl.viewer.layers)
        assert ctrl._preview_layers == []


# ---------------------------------------------------------------------------
# (Preview v2 Piece 2, slice 2a-i) draw-a-box → Preview computes THAT region.
#   The preview REGION box: its OWN overlay (separate from the ③ box tool),
#   region-compute (axes-lie fix), the budget guard, mode arbitration, and
#   Preview-disabled-until-a-region.
# ---------------------------------------------------------------------------


class _FakeBigArr:
    """A 3D array-like that reports a huge shape but is never sliced.

    `_gather_preview_inputs` only reads `.shape` / `.ndim` (the crop happens
    later in the worker), so this stands in for a deep-stack channel to exercise
    the compute-budget guard without allocating gigabytes."""

    ndim = 3

    def __init__(self, shape):
        self.shape = tuple(shape)


class TestPreviewComputeStride:
    """The pure compute-budget guard: stride > 1 only when the region would blow
    the float32 budget; 1 for a small region (byte-identical read)."""

    def test_small_region_needs_no_stride(self):
        import napari_epicycle.data_preparation.controller as ctrl_mod
        b = {"z0": 0, "z1": 32, "y0": 0, "y1": 32, "x0": 0, "x1": 32}
        assert ctrl_mod._preview_compute_stride(b, 1) == 1
        assert ctrl_mod._preview_compute_stride(b, 4) == 1

    def test_over_budget_region_is_strided_down(self):
        import napari_epicycle.data_preparation.controller as ctrl_mod
        # 1024^3 voxels * 1ch * 4 bytes ≈ 4.3 GB ≫ the 512 MB budget → stride > 1.
        b = {"z0": 0, "z1": 1024, "y0": 0, "y1": 1024, "x0": 0, "x1": 1024}
        s = ctrl_mod._preview_compute_stride(b, 1)
        assert s > 1
        # Striding by s cuts voxels by ~s^3 → the strided crop must fit budget.
        dz = (1024 + s - 1) // s
        assert (dz ** 3) * 4 <= ctrl_mod._PREVIEW_COMPUTE_BUDGET_BYTES

    def test_more_channels_push_over_budget(self):
        import napari_epicycle.data_preparation.controller as ctrl_mod
        # ~192 MB per channel: fits alone, but 4 channels ≈ 768 MB → strided.
        b = {"z0": 0, "z1": 384, "y0": 0, "y1": 512, "x0": 0, "x1": 256}
        assert ctrl_mod._preview_compute_stride(b, 1) == 1
        assert ctrl_mod._preview_compute_stride(b, 4) > 1

    def test_crop_center_honors_stride_keys(self):
        import numpy as np
        import napari_epicycle.data_preparation.controller as ctrl_mod
        arr = np.arange(8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
        strided = ctrl_mod._crop_center(
            arr, {"z0": 0, "z1": 8, "y0": 0, "y1": 8, "x0": 0, "x1": 8,
                  "stride_z": 2, "stride_y": 2, "stride_x": 2})
        assert strided.shape == (4, 4, 4)
        # No stride keys → byte-identical to the plain slice.
        plain = ctrl_mod._crop_center(
            arr, {"z0": 0, "z1": 8, "y0": 0, "y1": 8, "x0": 0, "x1": 8})
        assert plain.shape == (8, 8, 8)


class TestPreviewRegionBox:
    """The preview REGION box drives Preview compute + Preview-enable, its own
    overlay is created on Place + destroyed on unload/step-exit, and placing it
    arbitrates against the ③ extract-box tool."""

    def test_gather_inputs_uses_box_bounds_not_center_crop(self, widget):
        import numpy as np
        import napari_epicycle.data_preparation.controller as ctrl_mod
        from packages.epicycle_annotator.roi_box import BoxRec

        ctrl = widget.controller
        cad = np.random.rand(8, 200, 200).astype(np.float32)
        nuc = np.random.rand(8, 200, 200).astype(np.float32)
        ctrl._source_display_channels = [("cad", cad), ("nuc", nuc)]
        widget._recipe_channel_names = ["cad", "nuc"]
        # An off-centre box so its origin can't coincide with the centre crop.
        box = BoxRec(cz=4, cy=50, cx=60, sz=8, sy=40, sx=30)
        ctrl._preview_box = box

        got = ctrl._gather_preview_inputs()
        assert got is not None
        names, channels, bounds_list, spacing, recipe_json = got
        expected = box.bounds_dict((8, 200, 200))
        for b in bounds_list:
            for k in ("z0", "z1", "y0", "y1", "x0", "x1"):
                assert b[k] == expected[k]
        # Distinct from the legacy centre crop (which starts at y0=0 here).
        center = ctrl_mod._preview_center_bounds((8, 200, 200))
        assert (bounds_list[0]["y0"], bounds_list[0]["x0"]) != \
            (center["y0"], center["x0"])
        # The actual crop off the per-channel handle matches the box extent.
        crop = ctrl_mod._crop_center(cad, bounds_list[0])
        assert crop.shape == box.shape_in((8, 200, 200))

    def test_gather_inputs_center_crop_when_no_box(self, widget):
        import numpy as np
        import napari_epicycle.data_preparation.controller as ctrl_mod
        ctrl = widget.controller
        cad = np.random.rand(8, 300, 300).astype(np.float32)
        ctrl._source_display_channels = [("cad", cad)]
        widget._recipe_channel_names = ["cad"]
        ctrl._preview_box = None
        _n, _c, bounds_list, _s, _r = ctrl._gather_preview_inputs()
        expected = ctrl_mod._preview_center_bounds((8, 300, 300))
        assert bounds_list[0] == expected     # unchanged legacy behaviour

    def test_over_budget_box_injects_compute_stride(self, widget):
        import napari_epicycle.data_preparation.controller as ctrl_mod
        from packages.epicycle_annotator.roi_box import BoxRec
        ctrl = widget.controller
        shape = (512, 2048, 2048)
        ctrl._source_display_channels = [("cad", _FakeBigArr(shape))]
        widget._recipe_channel_names = ["cad"]
        # A FULL-volume box (full Z, full XY) → far over the 512 MB budget.
        ctrl._preview_box = BoxRec(cz=256, cy=1024, cx=1024,
                                   sz=512, sy=2048, sx=2048)
        _n, _c, bounds_list, _s, _r = ctrl._gather_preview_inputs()
        assert bounds_list[0].get("stride_z", 1) > 1
        assert bounds_list[0].get("stride_y", 1) > 1
        assert bounds_list[0].get("stride_x", 1) > 1

    def test_over_budget_box_scales_spacing_by_stride(self, widget):
        """MED-1: a strided (over-budget) preview MUST scale the spacing handed to
        the recipe by the stride — else a geometry (isotropic) op previews the
        wrong resample ratio (factor = spacing/target)."""
        import napari_epicycle.data_preparation.controller as ctrl_mod
        from packages.epicycle_annotator.roi_box import BoxRec
        ctrl = widget.controller
        shape = (512, 2048, 2048)
        ctrl._source_display_channels = [("cad", _FakeBigArr(shape))]
        widget._recipe_channel_names = ["cad"]
        box = BoxRec(cz=256, cy=1024, cx=1024, sz=512, sy=2048, sx=2048)
        ctrl._preview_box = box
        base = tuple(float(v) for v in ctrl._preview_spacing())
        stride = ctrl_mod._preview_compute_stride(dict(box.bounds_dict(shape)), 1)
        assert stride > 1
        _n, _c, _b, spacing, _r = ctrl._gather_preview_inputs()
        assert tuple(float(v) for v in spacing) == \
            tuple(v * stride for v in base)

    def test_preview_button_disabled_until_region_placed(self, widget):
        import numpy as np
        ctrl = widget.controller
        cad = np.random.rand(6, 64, 64).astype(np.float32)
        ctrl._source_display_channels = [("cad", cad)]
        widget._recipe_channel_names = ["cad"]
        widget._refresh_recipe_view()                 # channels, no region yet
        assert not widget.test_region_button.isEnabled()
        assert "region" in widget.test_region_button.toolTip().lower()
        ctrl.place_preview_region()
        assert ctrl._preview_box is not None
        assert widget.test_region_button.isEnabled()

    def test_overlay_created_on_place_and_destroyed_on_unload(self, widget):
        import numpy as np
        ctrl = widget.controller
        cad = np.random.rand(6, 64, 64).astype(np.float32)
        ctrl._source_display_channels = [("cad", cad)]
        assert ctrl._preview_box_overlay is None
        ctrl.place_preview_region()
        assert ctrl._preview_box_overlay is not None   # its OWN overlay instance
        assert ctrl._preview_box is not None
        # Its overlay is SEPARATE from the ③ box tool's overlay.
        assert ctrl._preview_box_overlay is not getattr(ctrl.box_tool, "_overlay")
        ctrl.unload_source()
        assert ctrl._preview_box_overlay is None
        assert ctrl._preview_box is None

    def test_overlay_destroyed_on_step_exit(self, widget):
        import numpy as np
        ctrl = widget.controller
        cad = np.random.rand(6, 64, 64).astype(np.float32)
        ctrl._source_display_channels = [("cad", cad)]
        ctrl.place_preview_region()
        assert ctrl._preview_box is not None
        ctrl.on_step_exit()
        assert ctrl._preview_box is None
        assert ctrl._preview_box_overlay is None

    def test_placing_region_stops_active_box_tool(self, widget, monkeypatch):
        import numpy as np
        ctrl = widget.controller
        cad = np.random.rand(6, 64, 64).astype(np.float32)
        ctrl._source_display_channels = [("cad", cad)]
        bt = ctrl.box_tool
        bt._active = True                              # pretend extract-box mode
        stopped = []
        orig_stop = bt.stop
        monkeypatch.setattr(
            bt, "stop",
            lambda *a, **k: (stopped.append(True), orig_stop(*a, **k))[-1])
        ctrl.place_preview_region()
        assert stopped, "placing a preview region must stop the active box tool"
        assert not bt.is_active()

    def test_move_resize_bump_the_preview_token(self, widget):
        import numpy as np
        ctrl = widget.controller
        cad = np.random.rand(6, 200, 200).astype(np.float32)
        ctrl._source_display_channels = [("cad", cad)]
        ctrl.place_preview_region()
        t0 = ctrl._preview_token
        ctrl.recenter_preview_region()
        assert ctrl._preview_token > t0                # a move discards stale runs
        t1 = ctrl._preview_token
        ctrl.set_preview_size_fraction(0.5)
        assert ctrl._preview_token > t1                # a resize does too

    def test_place_without_source_is_a_guarded_noop(self, widget):
        ctrl = widget.controller
        ctrl._source_display_channels = None
        ctrl.place_preview_region()                    # must not raise
        assert ctrl._preview_box is None
        assert not widget.test_region_button.isEnabled()


# ---------------------------------------------------------------------------
# (Preview v2 Piece 2, next slice) X/Y/Z + Z-depth position sliders (reveal
#   after Place), the single `_teardown_preview` exit, and the lifecycle sites
#   routing through it.
# ---------------------------------------------------------------------------


class TestPreviewRegionSliders:
    """Dragging a slider MOVES / RESIZES the box live (bumps the stale-token +
    clears the shown preview); Z-position greys at full depth; and
    `_teardown_preview` is the single idempotent exit the lifecycle sites use."""

    def _place(self, widget, shape):
        import numpy as np
        ctrl = widget.controller
        arr = np.random.rand(*shape).astype(np.float32)
        ctrl._source_display_channels = [("cad", arr)]
        # Brick 5b: the off-canvas gate reads this pair.  A preview region can
        # only be PLACED with the source displayed, so a fixture that sets the
        # channel refs without the on-canvas flag describes a state production
        # never reaches -- and would trip the gate in the box-mode test below.
        ctrl._source_layers_on_canvas = True
        widget._recipe_channel_names = ["cad"]
        widget._refresh_recipe_view()
        ctrl.place_preview_region()
        return ctrl

    # -- controller-side: the box moves + the preview is invalidated ----------

    def test_set_preview_center_moves_box_bumps_token_clears(self, widget):
        ctrl = self._place(widget, (8, 400, 400))     # M box (sy=sx=100), full Z
        ctrl._preview_layers = [_PreviewLayer(None, name="⟨preview⟩ cad")]
        t0 = ctrl._preview_token
        ctrl.set_preview_center(cx=100, cy=120)
        # Interior centre (sx=sy=100 → no clamp shift) reflects the slider values.
        assert ctrl._preview_box.cx == 100
        assert ctrl._preview_box.cy == 120
        b = ctrl._preview_box.bounds_dict((8, 400, 400))
        assert (b["x0"] + b["x1"]) // 2 == 100
        assert (b["y0"] + b["y1"]) // 2 == 120
        assert ctrl._preview_token > t0               # stale-guard bumped
        assert ctrl._preview_layers == []             # shown preview cleared

    def test_set_preview_z_depth_resizes_box_bumps_token_clears(self, widget):
        ctrl = self._place(widget, (16, 128, 128))
        assert ctrl._preview_box.sz == 16             # full Z on place
        ctrl._preview_layers = [_PreviewLayer(None, name="⟨preview⟩ cad")]
        t0 = ctrl._preview_token
        ctrl.set_preview_z_depth(6)
        assert ctrl._preview_box.sz == 6
        b = ctrl._preview_box.bounds_dict((16, 128, 128))
        assert (b["z1"] - b["z0"]) == 6
        assert ctrl._preview_token > t0
        assert ctrl._preview_layers == []

    def test_slider_setters_guarded_without_box(self, widget):
        ctrl = widget.controller
        ctrl._preview_box = None
        ctrl.set_preview_center(cx=5)                  # must not raise
        ctrl.set_preview_z_depth(3)                    # must not raise
        assert ctrl._preview_box is None

    def test_slider_setters_noop_while_busy(self, widget):
        ctrl = self._place(widget, (8, 200, 200))
        before = ctrl._preview_box
        ctrl._step1_busy = True
        ctrl.set_preview_center(cx=10)
        ctrl.set_preview_z_depth(2)
        assert ctrl._preview_box is before            # unchanged while busy

    # -- UI-side: the sliders are wired + Z-position greys off the depth ------

    def test_x_slider_move_invokes_controller_set_center(self, widget):
        ctrl = self._place(widget, (8, 200, 200))
        calls = []
        ctrl.set_preview_center = lambda **kw: calls.append(kw)
        widget.preview_x_slider.setValue(40)
        assert calls and calls[-1] == {"cx": 40}

    def test_y_slider_move_invokes_controller_set_center(self, widget):
        ctrl = self._place(widget, (8, 200, 200))
        calls = []
        ctrl.set_preview_center = lambda **kw: calls.append(kw)
        widget.preview_y_slider.setValue(33)
        assert calls and calls[-1] == {"cy": 33}

    def test_zdepth_slider_move_invokes_controller_and_enables_zpos(self, widget):
        ctrl = self._place(widget, (16, 128, 128))
        assert not widget.preview_z_slider.isEnabled()  # full depth → greyed
        calls = []
        ctrl.set_preview_z_depth = lambda v: calls.append(v)
        widget.preview_zdepth_slider.setValue(6)
        assert calls and calls[-1] == 6
        assert widget.preview_z_slider.isEnabled()      # depth < full → enabled

    def test_zpos_slider_greyed_at_full_depth_enabled_below(self, widget):
        self._place(widget, (16, 128, 128))
        assert widget.preview_zdepth_slider.value() == 16
        assert not widget.preview_z_slider.isEnabled()  # spans all Z → inert
        widget.preview_zdepth_slider.setValue(4)        # sub-slab
        assert widget.preview_z_slider.isEnabled()
        widget.preview_zdepth_slider.setValue(16)       # back to full
        assert not widget.preview_z_slider.isEnabled()

    def test_programmatic_seed_does_not_fire_controller(self, widget):
        """`set_preview_region_active` seeds the sliders WITHOUT firing the
        live-move callbacks (else a feedback loop / spurious token bumps)."""
        ctrl = self._place(widget, (8, 200, 200))
        calls = []
        ctrl.set_preview_center = lambda **kw: calls.append(kw)
        ctrl.set_preview_z_depth = lambda v: calls.append(("depth", v))
        # Re-seed from the box (as place does) — must not call the setters.
        widget.set_preview_region_active(True, (8, 200, 200))
        assert calls == []

    # -- reveal / collapse + the single teardown entry point ------------------

    def test_place_reveals_group_teardown_collapses(self, widget):
        ctrl = self._place(widget, (8, 64, 64))
        assert not widget.preview_region_group.isHidden()   # revealed on place
        assert widget.test_region_button.isEnabled()        # Preview enabled
        ctrl._teardown_preview()
        assert ctrl._preview_box is None
        assert ctrl._preview_box_overlay is None
        assert widget.preview_region_group.isHidden()       # collapsed again
        assert not widget.test_region_button.isEnabled()    # Preview disabled

    def test_teardown_clears_layers_and_is_idempotent(self, widget):
        ctrl = self._place(widget, (6, 64, 64))
        ctrl._preview_layers = [_PreviewLayer(None, name="⟨preview⟩ cad")]
        ctrl._teardown_preview()
        assert ctrl._preview_layers == []
        assert ctrl._preview_box is None
        assert widget.preview_region_group.isHidden()
        ctrl._teardown_preview()                        # idempotent, no region
        assert ctrl._preview_box is None
        assert ctrl._preview_layers == []

    def test_teardown_safe_with_no_region(self, widget):
        ctrl = widget.controller
        assert ctrl._preview_box is None
        ctrl._teardown_preview()                        # must not raise
        assert ctrl._preview_box is None
        assert widget.preview_region_group.isHidden()

    def test_lifecycle_sites_route_through_teardown(self, widget):
        import inspect
        ctrl = widget.controller
        # `unload_source` is now a thin wrapper (busy bar + [STEP1-PERF] probe)
        # around `_unload_source_inner`, which holds the actual teardown.
        for meth in (ctrl.on_step_exit, ctrl._unload_source_inner,
                     ctrl._on_source_loaded, ctrl._apply_multiscale_store):
            src = inspect.getsource(meth)
            assert "_teardown_preview" in src, meth.__name__
        # ...and the wrapper really does reach it.
        assert "_unload_source_inner" in inspect.getsource(ctrl.unload_source)

    def test_step_exit_collapses_group(self, widget):
        ctrl = self._place(widget, (6, 64, 64))
        assert not widget.preview_region_group.isHidden()
        ctrl.on_step_exit()
        assert ctrl._preview_box is None
        assert widget.preview_region_group.isHidden()

    # -- post-audit: any box MUTATION re-seeds the sliders + clears the preview -

    def test_recenter_reseeds_sliders_and_clears_preview(self, widget):
        """audit finding 1: Recenter moves the box, so the sliders must re-sync to
        it + the shown preview must clear (else the next nudge jumps)."""
        ctrl = self._place(widget, (8, 200, 200))
        ctrl.set_preview_center(cx=30, cy=40)          # move the handles off-centre
        ctrl._preview_layers = [_PreviewLayer(None, name="⟨preview⟩ cad")]
        ctrl.recenter_preview_region()
        assert ctrl._preview_layers == []              # stale preview cleared
        assert widget.preview_x_slider.value() == ctrl._preview_box.cx
        assert widget.preview_y_slider.value() == ctrl._preview_box.cy

    def test_size_preset_reseeds_and_clears_preview(self, widget):
        """audit finding 1: an S/M/L resize must also clear the stale preview."""
        ctrl = self._place(widget, (8, 200, 200))
        ctrl._preview_layers = [_PreviewLayer(None, name="⟨preview⟩ cad")]
        ctrl.set_preview_size_fraction(0.5)
        assert ctrl._preview_layers == []

    def test_edge_clamp_reseeds_slider_to_clamped_center(self, widget):
        """audit finding 2: an edge-clamped drag must snap the slider handle to the
        clamped box centre, not leave it at the out-of-range value set."""
        ctrl = self._place(widget, (8, 200, 200))       # M box, sx=sy=50
        ctrl.set_preview_center(cx=199)                 # past the right edge
        assert ctrl._preview_box.cx < 199               # clamp pulled it in
        assert widget.preview_x_slider.value() == ctrl._preview_box.cx

    def test_entering_box_mode_tears_down_preview_region(self, widget):
        """mode arbitration: opening the ③ extract-box tool must FULLY tear the
        preview region down (box + overlay + sliders + layers), not just the box."""
        ctrl = self._place(widget, (6, 64, 64))
        ctrl._preview_layers = [_PreviewLayer(None, name="⟨preview⟩ cad")]
        ctrl.box_tool.start = lambda: True              # avoid the real Qt start
        ctrl.on_box_mode_toggled(True)
        assert ctrl._preview_box is None
        assert ctrl._preview_layers == []
        assert widget.preview_region_group.isHidden()

    def test_box_mode_sites_route_through_teardown(self, widget):
        import inspect
        ctrl = widget.controller
        # The three box-mode ENTRY sites route through the arbitration helper,
        # which snaps to raw + tears down the mutually-exclusive preview REGION
        # WITHOUT dropping a re-flippable processed view (the fix for the "box
        # mode hides everything" empty-canvas bug).
        for meth in (ctrl.on_box_mode_toggled, ctrl.add_box, ctrl.on_place_toggled):
            assert "_arbitrate_for_box_mode" in inspect.getsource(meth), meth.__name__
        # …and the arbitration helper is where the region teardown still happens.
        assert "_teardown_preview" in inspect.getsource(ctrl._arbitrate_for_box_mode)


class TestSecondSectionSizing:
    """Brick 3b item 2 — ②'s width must not be hostage to the longest source
    name.  Qt's default `sizeAdjustPolicy` makes the longest item the combo's
    MINIMUM width, which propagates through the ② layout to the splitter and
    steals width from everything to its right (same fix already in Step 2's
    `main_widget.py` and Step 3's `curation_3d_widget.py`)."""

    LONG = "a_very_long_microscopy_source_file_name_" * 5   # 200 chars

    def test_combos_cannot_be_widened_by_their_items(self, widget):
        """The load-bearing contract: both ② combos are pinned to a minimum
        contents length, so a 200-char source name elides into the popup
        instead of forcing the whole section wider."""
        from qtpy.QtWidgets import QComboBox

        for name in ("source_combo", "axis_order_combo"):
            combo = getattr(widget, name)
            assert (combo.sizeAdjustPolicy()
                    == QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon), name
            assert combo.minimumContentsLength() > 0, name

    def test_a_long_source_name_keeps_the_combo_narrow(self, widget):
        combo = widget.source_combo
        combo.addItem(self.LONG)
        combo.setCurrentIndex(combo.count() - 1)
        for w in (combo.sizeHint().width(), combo.minimumSizeHint().width()):
            assert w < 300, f"source_combo asks for {w}px on a 200-char item"

    def test_voxel_spins_keep_a_usable_width(self, widget):
        for spin in (widget.voxel_z_spin, widget.voxel_y_spin,
                     widget.voxel_x_spin):
            assert spin.minimumWidth() >= 72

    def test_long_text_labels_wrap_instead_of_stretching(self, widget):
        for name in ("axis_order_hint", "selected_roi_label"):
            assert getattr(widget, name).wordWrap(), f"{name} must wrap"


class TestRowNameElision:
    """Brick 3b item 3 — a long name in a list row must not push the row's
    buttons (or the ③ dimensions) off the edge."""

    def test_reserved_dim_width_leaves_room_for_the_name(self):
        from napari_epicycle.data_preparation.ui import reserved_dim_width

        # A normal row: the dims get their own column at the right.
        assert reserved_dim_width(dim_text_width=60, row_width=300) > 60
        # A row too narrow to carry both: the dims stand down, they never
        # get painted over the name.
        assert reserved_dim_width(dim_text_width=60, row_width=70) == 0
        assert reserved_dim_width(dim_text_width=0, row_width=300) == 0

    def test_elide_label_shortens_and_keeps_the_full_text_in_the_tooltip(
            self, app):
        from qtpy.QtWidgets import QLabel
        from napari_epicycle.ui_shell.theme import elide_label

        long_name = "patch_from_a_very_long_source_name_0123456789" * 3
        label = QLabel()
        shown = elide_label(label, long_name, 80)
        assert shown != long_name and len(shown) < len(long_name)
        assert label.text() == shown
        assert label.toolTip() == long_name

    def test_elide_label_leaves_a_short_name_alone(self, app):
        from qtpy.QtWidgets import QLabel
        from napari_epicycle.ui_shell.theme import elide_label

        label = QLabel()
        assert elide_label(label, "patch_01", 400) == "patch_01"
        assert label.text() == "patch_01"
        assert label.toolTip() == "patch_01"

    def test_a_long_patch_name_does_not_widen_the_row(self, widget):
        long_name = "patch_" + "x" * 200
        _stage(widget, long_name)
        row = widget.library_box_list.list.itemWidget(
            widget.library_box_list.list.item(0))
        assert row is not None
        assert row.sizeHint().width() < 500, (
            f"row asks for {row.sizeHint().width()}px on a 200-char name")


class TestV1HideList:
    """Brick 3b item 4 — controls that promise something Step 1 cannot do yet
    are hidden for v1 (hidden, never removed: every attribute name survives,
    because the controller binds widgets by name)."""

    def test_the_add_step_menu_offers_only_ops_that_run(self, widget):
        widget.update_image_layer_list(["cad"])
        menu = widget._make_add_op_menu(0)
        texts = [a.text() for a in menu.actions()
                 if not a.isSeparator() and a.text()]
        assert not [t for t in texts if "soon" in t], (
            f"unimplemented ops still offered: {texts}")
        # the implemented v1 ops are all still there, enabled
        by_text = {a.text(): a for a in menu.actions()
                   if not a.isSeparator() and a.text()}
        for label in ("Gaussian", "CLAHE"):
            assert by_text[label].isEnabled()
        for op in REGISTRY:
            if op.implemented:
                assert any(op.label in t for t in texts), op.label

    def test_dead_controls_are_hidden(self, widget):
        # Present (the controller binds these by name) but not shown.
        assert widget.compare_raw_processed_button.isHidden()
        assert widget.draw_mode_label.isHidden()

    def test_writing_to_a_hidden_label_still_works(self, widget):
        """Hidden, not removed: the controller's writes must not raise."""
        widget._set_draw_mode("Placing")
        assert "Placing" in widget.draw_mode_label.text()

    def test_box_row_placeholder_glyphs_are_hidden(self, widget):
        from qtpy.QtWidgets import QPushButton

        widget.draw_box_list.list.addItem("box_1")
        widget.draw_box_list.refresh()
        row = widget.draw_box_list.list.itemWidget(
            widget.draw_box_list.list.item(0))
        assert row is not None
        for btn in row.findChildren(QPushButton):
            if btn.text() in ("\u270e", "\u29c9", "\U0001f441", "\U0001f5d1"):
                assert btn.isHidden(), f"{btn.text()} placeholder still shown"
            assert "spec " not in btn.toolTip(), btn.toolTip()

    def test_every_visible_help_button_has_a_guide_on_disk(self, widget):
        """A `?` that opens a 'not written yet' page is worse than no `?`."""
        assert widget._help_guides, "no help buttons wired at all"
        missing = [g for g in widget._help_guides
                   if not (_GUIDES_DIR / g).is_file()]
        assert not missing, f"`?` buttons with no guide file: {missing}"

    def test_the_help_buttons_that_do_resolve_are_still_wired(self, widget):
        from qtpy.QtWidgets import QPushButton

        helps = [b for b in widget.findChildren(QPushButton)
                 if b.toolTip().startswith("Help:")]
        assert len(helps) == len(widget._help_guides)
        assert all(_is_connected(b) for b in helps)


class TestSourceComboTooltip:
    """Brick 3d note 3 — the ② combo is pinned narrow, so a long source name is
    elided in the CLOSED box.  Hovering it must show the full name."""

    def test_the_closed_box_shows_the_current_item_tooltip(self, widget):
        widget.source_combo.clear()
        widget.source_combo.addItem("short")
        widget.source_combo.setItemData(0, "short\nC:/data/short.tif",
                                        Qt.ToolTipRole)
        widget.source_combo.addItem("elided_name")
        widget.source_combo.setItemData(
            1, "a_very_long_source_name\nC:/data/a_very_long_source_name.tif",
            Qt.ToolTipRole)

        widget.source_combo.setCurrentIndex(1)
        assert "a_very_long_source_name" in widget.source_combo.toolTip()
        widget.source_combo.setCurrentIndex(0)
        assert "C:/data/short.tif" in widget.source_combo.toolTip()

    def test_no_selection_falls_back_to_the_standing_hint(self, widget):
        widget.source_combo.clear()
        assert "Register sources" in widget.source_combo.toolTip()


class TestBoxDimDelegatePaint:
    """Brick 3d note 4 — the dims get their own column, but the row's
    highlight / hover strip must still span the FULL row width."""

    def _paint(self, app, name, dim):
        from qtpy.QtCore import QRect
        from qtpy.QtGui import QPainter, QPixmap
        from qtpy.QtWidgets import (QListWidget, QListWidgetItem,
                                    QStyleOptionViewItem)
        from napari_epicycle.data_preparation.ui import _BoxDimDelegate

        lst = QListWidget()
        item = QListWidgetItem(name)
        item.setData(Qt.UserRole + 8, dim)
        lst.addItem(item)
        delegate = _BoxDimDelegate(lst)

        pm = QPixmap(240, 20)
        painter = QPainter(pm)
        opt = QStyleOptionViewItem()
        opt.rect = QRect(0, 0, 240, 20)
        try:
            delegate.paint(painter, opt, lst.model().index(0, 0))
        finally:
            painter.end()
            lst.setParent(None)
        return opt

    def test_paint_does_not_shrink_the_row_rect(self, app):
        """The delegate must not mutate the caller's option rect -- Qt reuses
        it, and a shrunk rect is a half-width selection bar."""
        opt = self._paint(app, "box_1", "12x256x256")
        assert opt.rect.width() == 240

    def test_the_base_paint_still_gets_the_full_row_width(self, app):
        """The highlight / hover strip is drawn by the BASE paint, so handing
        it a rect narrowed by the dims column would leave a short selection
        bar with a bare gap under the dimensions.  Only the TEXT is elided."""
        from unittest.mock import patch
        from qtpy.QtWidgets import QStyledItemDelegate

        seen = {}

        def _spy(self, painter, option, index):
            seen["width"] = option.rect.width()
            seen["text"] = option.text

        with patch.object(QStyledItemDelegate, "paint", _spy):
            self._paint(app, "box_" + "y" * 300, "12x256x256")
        assert seen["width"] == 240, "base paint got a narrowed rect"
        assert seen["text"].endswith("…") or "…" in seen["text"], (
            f"the name was not elided: {seen['text']!r}")

    def test_paint_survives_a_very_long_name(self, app):
        opt = self._paint(app, "box_" + "y" * 300, "12x256x256")
        assert opt.rect.width() == 240

    def test_paint_without_dims_is_the_plain_row(self, app):
        opt = self._paint(app, "box_1", None)
        assert opt.rect.width() == 240


# --------------------------------------------------------------------------- #
# Final fix pass U4 -- no closure slot on a QAction nothing keeps
# --------------------------------------------------------------------------- #
def _free_names(lam):
    import ast

    bound = {a.arg for a in lam.args.args + lam.args.kwonlyargs}
    if lam.args.vararg:
        bound.add(lam.args.vararg.arg)
    if lam.args.kwarg:
        bound.add(lam.args.kwarg.arg)
    return {n.id for n in ast.walk(lam.body)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            and n.id not in bound}


def test_no_closure_lambda_is_connected_on_an_add_action_result():
    """U4: in ui.py, no ``.connect(lambda ...)`` whose lambda has a free
    variable (``self`` included) sits on a name assigned from
    ``X.addAction(...)`` -- the QAction's lambda is the stale-closure GC
    crash class (the navigator's `selectionChanged`, 2026-09-27)."""
    import ast
    import builtins
    from pathlib import Path

    import napari_epicycle.data_preparation.ui as ui_mod

    tree = ast.parse(Path(ui_mod.__file__).read_text(encoding="utf-8"))
    offenders = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        actions = set()
        for node in ast.walk(fn):
            if (isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "addAction"):
                actions |= {t.id for t in node.targets
                            if isinstance(t, ast.Name)}
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "connect" and node.args
                    and isinstance(node.args[0], ast.Lambda)):
                continue
            base = node.func.value
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id in actions:
                free = _free_names(node.args[0]) - set(dir(builtins))
                if free:
                    offenders.append((fn.name, node.lineno, sorted(free)))
    assert not offenders, offenders


def test_the_add_step_menu_routes_through_the_bound_slot(widget):
    widget.update_image_layer_list(["cad", "nuc"])
    seen = []
    widget._add_op_from_menu = lambda ci, key: seen.append((ci, key))
    menu = widget._make_add_op_menu(1)
    acts = [a for a in menu.actions() if a.isEnabled() and a.data()]
    assert acts, "no pickable op in the menu"
    acts[0].trigger()
    assert seen == [tuple(acts[0].data())]
    assert seen[0][0] == 1

