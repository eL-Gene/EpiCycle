"""Step 2: the patch-load busy bar, the UI pacing, and the lazy work labels.

Covers dev-PC checklist item 27 -- "click a patch: a thin bar under the patch
list sweeps and the window stays draggable, with the annotator open and
closed".  All fakes are Qt-free.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    TrainingAnnotationController,
)


class _LogOutput:
    def __init__(self):
        self.lines = []

    def append(self, text):
        self.lines.append(text)


class _Bar:
    def __init__(self):
        self.range = (0, 100)
        self.value = 0
        self.visible = True          # permanent, like the real widget
        self.history = []

    def setRange(self, lo, hi):
        self.range = (lo, hi)
        self.history.append(("range", lo, hi))

    def setValue(self, v):
        self.value = v

    def setVisible(self, flag):
        self.visible = bool(flag)
        self.history.append(("visible", bool(flag)))


class _UI:
    def __init__(self):
        self.training_log_output = _LogOutput()
        self.training_progress_bar = _Bar()
        self.patch_busy_bar = _Bar()


class _Layer:
    def __init__(self, name, data=None):
        self.name = name
        self.data = data if data is not None else np.zeros((2, 4, 4), np.int32)
        self.visible = True
        self.editable = True
        self.opacity = 0.8
        self.colormap = None
        self.selected_label = 1

    def refresh(self):
        pass


class _Labels(_Layer):
    """Named `Labels` on purpose: `_iter_work_label_layers` filters on it."""


_Labels.__name__ = "Labels"


class _Viewer:
    def __init__(self):
        self.layers = []

    def add_labels(self, data, name=None, **kw):
        layer = _Labels(name, data)
        self.layers.append(layer)
        return layer


class _RealLayers(list):
    def remove(self, layer):
        list.remove(self, layer)


class _LoadViewer(_Viewer):
    def __init__(self):
        self.layers = _RealLayers()

    def add_image(self, data, name=None, **kw):
        layer = _Layer(name, data)
        self.layers.append(layer)
        return layer


def _make_load_ctrl(tmp_path):
    """Controller wired for the REAL `_load_patch` prelude/read/apply split."""
    import json

    import tifffile

    stack_dir = tmp_path / "patches"
    stack_dir.mkdir()
    tifffile.imwrite(
        str(stack_dir / "p1.tif"), np.zeros((2, 4, 8, 8), dtype=np.uint16),
    )
    (stack_dir / "p1_metadata.json").write_text(
        json.dumps({"patch_name": "p1", "channel_names": ["c0", "c1"]})
    )

    from napari_epicycle.hitl_training_annotation.controller.image_manager import (
        ImageManager,
    )

    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.ui = _UI()
    ctrl.viewer = _LoadViewer()
    ctrl.image_manager = ImageManager()
    ctrl.loaded_stack_path = str(stack_dir)
    ctrl.loaded_patch_name = "old_patch"
    ctrl.working_dir = str(tmp_path)
    ctrl.selected_view = "XY"
    ctrl.view_target = "lateral"
    ctrl._curating_persist_pending = {}
    ctrl._flush_curating_persist_pending = lambda: None
    ctrl._exit_refine_mode = lambda: None

    class _IM:
        def cleanup_prompt_layers(self, viewer):
            pass

    ctrl.interactive_manager = _IM()
    ctrl.layer_lifecycle_manager = None
    ctrl._read_view_selection_from_ui = lambda: ("lateral", "XY")
    ctrl._set_axial_combo_enabled = lambda: None
    ctrl._clean_patch_filename = lambda item: "p1.tif"
    ctrl._refresh_round_status = lambda: None
    ctrl.apply_active_view_to_viewer = lambda **k: None
    ctrl._set_work_labels_visibility_for_context = lambda **k: None
    ctrl._refresh_inference_channel_options = lambda: None
    ctrl._layer_by_name = lambda name: None
    ctrl._schedule_state_save = lambda: None
    ctrl.active_run_id = "default"

    class _List:
        def selectedItems(self):
            return ["p1.tif"]

    ctrl.ui.train_patch_list = _List()
    return ctrl


def _bare(ui=None):
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.ui = ui if ui is not None else _UI()
    return ctrl


# ---------------------------------------------------------------------------
# The bar itself
# ---------------------------------------------------------------------------

def test_the_bar_lives_in_the_right_columns_pinned_foot_and_is_permanent():
    """Brick P: the bar moved from the bottom of the
    curation box into the right column's pinned foot (words + stage line
    under it); still permanent, still never hidden."""
    from napari_epicycle.hitl_training_annotation.ui.main_widget import (
        TrainingAnnotationHITL2D,
    )

    col = inspect.getsource(
        TrainingAnnotationHITL2D._create_curation_config_column
    )
    assert "self.patch_busy_bar" in col, (
        "the bar is built with the right column, for its pinned foot"
    )
    assert col.index("self.patch_busy_bar = QProgressBar") < col.index(
        "build_pinned_foot"
    ), "built before the foot that receives it"
    assert "self.patch_busy_bar," in col[col.index("build_pinned_foot"):], (
        "handed to the pinned foot"
    )
    assert "setTextVisible(False)" in col
    assert "setVisible(True)" in col, "permanent — never hidden"
    # polish: the height is the foot's (14 px, 8 px in a short dock), not a
    # literal 12 here -- the sheet's own min-height used to fight it
    assert "setFixedHeight(12)" not in col

    panel = inspect.getsource(TrainingAnnotationHITL2D._create_curation_panel)
    assert "patch_busy_bar" not in panel, (
        "no longer a strip under the whole box"
    )
    left = inspect.getsource(
        TrainingAnnotationHITL2D._create_curation_patches_column
    )
    assert "patch_busy_bar" not in left, (
        "it must not be built inside the left patch column"
    )


def test_begin_busy_sweeps_the_patch_bar_and_end_busy_returns_it_to_idle():
    ctrl = _bare()
    bar = ctrl.ui.patch_busy_bar
    ctrl._begin_busy("[PATCH] Loading 'p1' ...", cursor=False)
    assert bar.range == (0, 0), "scan mode (marquee) while busy"
    ctrl._end_busy()
    assert bar.range == (0, 100) and bar.value == 0, "idle static strip"


def test_the_busy_helpers_never_show_or_hide_the_bar():
    """It is PERMANENT — appearing/disappearing reads as a glitch."""
    ctrl = _bare()
    bar = ctrl.ui.patch_busy_bar
    ctrl._begin_busy("[PATCH] Loading 'p1' ...", cursor=False)
    ctrl._end_busy()
    assert bar.visible is True, "the bar is never hidden"
    assert not [h for h in bar.history if h[0] == "visible"], (
        f"_begin_busy/_end_busy must not touch setVisible (saw {bar.history})"
    )


def test_a_missing_bar_never_raises():
    """Old/partial UIs (and the shell before the column is built) are safe."""

    class _NoBarUI:
        def __init__(self):
            self.training_log_output = _LogOutput()

    ctrl = _bare(_NoBarUI())
    ctrl._begin_busy("x", cursor=False)
    ctrl._end_busy()


# ---------------------------------------------------------------------------
# The synchronous (annotator-open) path is wrapped too
# ---------------------------------------------------------------------------

def _arm_switch_ctrl():
    ctrl = _bare()
    ctrl.loaded_patch_name = "old"
    ctrl.view_target = "lateral"
    ctrl.selected_view = "XY"
    ctrl._patch_switch_loading = False
    ctrl.viewer = _Viewer()
    ctrl._select_patch_item = lambda p: True
    ctrl._load_patch = lambda force_axis=None: None
    ctrl.apply_active_view_to_viewer = lambda *a, **k: None
    ctrl._set_work_labels_visibility_for_context = lambda **k: None
    ctrl._refresh_sample_queue = lambda: None
    ctrl._refresh_patch_list_badges = lambda: None
    ctrl._refresh_round_status = lambda: None
    ctrl._sync_data_to_add_list = lambda: None
    ctrl._append_ui_log = lambda text: None
    return ctrl


def test_the_annotator_open_switch_sweeps_the_bar_and_always_clears_it():
    ctrl = _arm_switch_ctrl()
    seen = []
    ctrl._load_patch = lambda force_axis=None: seen.append(
        ctrl.ui.patch_busy_bar.range
    )

    ctrl._switch_patch_context_from_native("p2")

    assert seen == [(0, 0)], (
        "the synchronous ~6 s patch switch must sweep WHILE it runs, not after"
    )
    assert ctrl.ui.patch_busy_bar.range == (0, 100), "released in the finally"


def test_the_annotator_open_switch_clears_the_bar_when_the_load_explodes():
    ctrl = _arm_switch_ctrl()

    def _boom(force_axis=None):
        raise RuntimeError("disk on fire")

    ctrl._load_patch = _boom
    ctrl._switch_patch_context_from_native("p2")

    assert ctrl.ui.patch_busy_bar.range == (0, 100)
    assert ctrl.ui.patch_busy_bar.visible is True, "still permanent"
    assert ctrl._patch_switch_loading is False


def test_the_bar_is_held_until_the_annotator_has_rebuilt(monkeypatch):
    """Releasing at `return` would hide the bar and THEN freeze the window."""
    ctrl = _arm_switch_ctrl()
    queued = []

    import qtpy.QtCore as QC
    import qtpy.QtWidgets as QW

    class _App:
        pass

    monkeypatch.setattr(QW.QApplication, "instance", staticmethod(lambda: _App()))
    monkeypatch.setattr(
        QC.QTimer, "singleShot",
        staticmethod(lambda ms, fn: queued.append((ms, fn))),
    )

    ctrl._switch_patch_context_from_native("p2")

    assert ctrl.ui.patch_busy_bar.range == (0, 0), (
        "the bar keeps sweeping while the annotator rebuilds its subviewers"
    )
    assert len(queued) == 1
    queued[0][1]()                       # the queued _end_busy runs
    assert ctrl.ui.patch_busy_bar.range == (0, 100)
    assert getattr(ctrl, "_busy_depth", 0) == 0, "still exactly one end"


def test_a_queued_release_fires_exactly_once(monkeypatch):
    """A double delivery must not pop a scope it does not own."""
    import qtpy.QtCore as QC
    import qtpy.QtWidgets as QW
    monkeypatch.setattr(QW.QApplication, "instance", staticmethod(lambda: object()))
    monkeypatch.setattr(QC.QTimer, "singleShot", staticmethod(lambda ms, fn: None))

    ctrl = _bare()
    ctrl._begin_busy("[PATCH] one ...", cursor=False)
    ctrl._end_busy_after_annotator_rebuild()
    pending = list(getattr(ctrl, "_pending_rebuild_busy", ()) or ())
    assert len(pending) == 1, "the queued release is tracked by epoch"

    ctrl._release_annotator_rebuild_busy(pending[0])
    assert getattr(ctrl, "_busy_depth", 0) == 0
    assert ctrl.ui.patch_busy_bar.range == (0, 100)

    ctrl._begin_busy("[PATCH] two ...", cursor=False)
    ctrl._release_annotator_rebuild_busy(pending[0])   # duplicate delivery
    assert getattr(ctrl, "_busy_depth", 0) == 1, (
        "a stale release must not end the NEXT load's scope"
    )
    assert ctrl.ui.patch_busy_bar.range == (0, 0), "still sweeping"


def test_a_release_queued_by_load_n_cannot_end_load_n_plus_1(monkeypatch):
    ctrl = _arm_switch_ctrl()
    queued = []
    import qtpy.QtCore as QC
    import qtpy.QtWidgets as QW
    monkeypatch.setattr(QW.QApplication, "instance", staticmethod(lambda: object()))
    monkeypatch.setattr(
        QC.QTimer, "singleShot",
        staticmethod(lambda ms, fn: queued.append(fn)),
    )

    ctrl._switch_patch_context_from_native("p2")      # load N, release queued
    ctrl.loaded_patch_name = "p2"
    ctrl._switch_patch_context_from_native("p3")      # load N+1, release queued
    assert len(queued) == 2

    queued[0]()          # N's release lands while N+1 still holds a scope
    assert ctrl.ui.patch_busy_bar.range == (0, 0), (
        "the bar must keep sweeping for the load that is still running"
    )
    queued[0]()          # and a duplicate changes nothing
    assert ctrl.ui.patch_busy_bar.range == (0, 0)
    queued[1]()
    assert ctrl.ui.patch_busy_bar.range == (0, 100)


def test_the_same_patch_and_reentrancy_guards_do_not_open_a_busy_scope():
    ctrl = _arm_switch_ctrl()
    ctrl._switch_patch_context_from_native("old")   # same patch
    assert ctrl.ui.patch_busy_bar.history == []
    assert getattr(ctrl, "_busy_depth", 0) == 0

    ctrl._patch_switch_loading = True
    ctrl._switch_patch_context_from_native("p2")    # reentrant
    assert ctrl.ui.patch_busy_bar.history == []
    assert getattr(ctrl, "_busy_depth", 0) == 0


# ---------------------------------------------------------------------------
# Pacing
# ---------------------------------------------------------------------------

def test_pump_is_inert_outside_a_busy_scope(monkeypatch):
    ctrl = _bare()
    calls = []
    import qtpy.QtWidgets as QW
    monkeypatch.setattr(
        QW.QApplication, "instance",
        staticmethod(lambda: calls.append("instance") or None),
    )
    ctrl._busy_depth = 0
    ctrl._pump_busy_ui()
    assert calls == [], "no Qt is touched when nothing is loading"


def test_pump_excludes_user_input_so_a_second_click_cannot_re_enter(monkeypatch):
    ctrl = _bare()
    seen = []

    class _App:
        def processEvents(self, flags=None):
            seen.append(flags)

    import qtpy.QtWidgets as QW
    app = _App()
    monkeypatch.setattr(QW.QApplication, "instance", staticmethod(lambda: app))
    ctrl._busy_depth = 1
    ctrl._pump_busy_ui()

    from qtpy.QtCore import QEventLoop
    assert seen == [QEventLoop.ExcludeUserInputEvents]


def test_the_removal_loop_pumps_once_per_layer_it_removes(tmp_path):
    """~250 ms per removal x 4-11 removals is Step 2's longest single freeze."""
    ctrl = _make_load_ctrl(tmp_path)
    for name in ("old_c0_XY_view", "old_c1_XY_view",
                 "old_work_labels_lateral_3d", "old_XY_native_rgb_stack"):
        ctrl.viewer.layers.append(_Layer(name))
    pumps = []
    ctrl._pump_busy_ui = lambda: pumps.append(len(ctrl.viewer.layers))

    ctrl._load_patch_prelude(force_axis="XY")

    assert len(pumps) == 4, (
        "one pump per layer removed, so the bar keeps sweeping through the "
        f"loop (saw {pumps})"
    )
    assert pumps == [3, 2, 1, 0], "the pump happens AFTER each removal"


def test_the_add_sequence_pumps_once_per_layer_it_adds(tmp_path):
    ctrl = _make_load_ctrl(tmp_path)
    ctx = ctrl._load_patch_prelude(force_axis="XY")
    payload = ctrl._read_patch_payload(ctx["file_path"])
    pumps = []
    ctrl._pump_busy_ui = lambda: pumps.append(1)

    ctrl._apply_loaded_patch(ctx, payload)

    assert len(pumps) == 3, (
        "one pump per channel added, plus one after apply_active_view_to_viewer "
        f"(saw {len(pumps)})"
    )


# ---------------------------------------------------------------------------
# The pump must not open a re-entrancy hole in the load
# ---------------------------------------------------------------------------

def test_a_stale_finish_delivered_mid_prelude_is_discarded(tmp_path):
    """The pump can run a queued `finished` slot from an older async load."""
    ctrl = _make_load_ctrl(tmp_path)
    ctx_a = ctrl._load_patch_prelude(force_axis="XY")
    payload = ctrl._read_patch_payload(ctx_a["file_path"])

    # A second click starts a new load; its removal loop pumps, and Qt
    # delivers load A's queued finished slot right in the middle of it.
    for name in ("old_c0_XY_view", "old_c1_XY_view"):
        ctrl.viewer.layers.append(_Layer(name))
    delivered = []

    def _pump():
        if not delivered:
            delivered.append(1)
            ctrl._on_patch_load_finished(payload, ctx_a)

    ctrl._pump_busy_ui = _pump
    ctrl._load_patch_prelude(force_axis="XY")

    assert delivered == [1], "the test must actually have delivered the slot"
    assert ctrl.loaded_patch_name == "", (
        "load A is superseded by the load whose teardown is still running -- "
        "installing it would put layers into a half-torn-down viewer"
    )
    assert [layer.name for layer in ctrl.viewer.layers] == [], (
        "no layers from the stale load may survive the teardown"
    )


def test_the_token_is_claimed_before_the_first_pump(tmp_path):
    ctrl = _make_load_ctrl(tmp_path)
    ctrl.viewer.layers.append(_Layer("old_c0_XY_view"))
    before = int(getattr(ctrl, "_patch_load_token", 0))
    seen = []
    ctrl._pump_busy_ui = lambda: seen.append(
        int(getattr(ctrl, "_patch_load_token", 0))
    )

    ctrl._load_patch_prelude(force_axis="XY")

    assert seen and seen[0] > before, (
        "the token bump must be hoisted above the removal loop, or a stale "
        "finish delivered by the pump reads the OLD token and is installed"
    )


# ---------------------------------------------------------------------------
# Lazy work labels
# ---------------------------------------------------------------------------

def _arm_labels_ctrl():
    ctrl = _bare()
    ctrl.viewer = _Viewer()
    ctrl.loaded_patch_name = "p1"
    ctrl.selected_view = "XY"
    ctrl.view_target = "lateral"
    ctrl.sample_records = {}
    ctrl._saved_work_labels_colormap = None
    ctrl.layer_lifecycle_manager = None
    ctrl._connect_label_safety_hooks = lambda layer: None
    ctrl._sync_selected_label_to_volume = lambda layer: None
    ctrl._get_patch_view_image_layers = lambda view_name=None: []
    ctrl._get_any_patch_view_image_layers = lambda: [
        _Layer("p1_c0_XY_view", np.zeros((3, 4, 4), np.uint16))
    ]
    ctrl._append_ui_log = lambda text: None
    return ctrl


def _names(ctrl):
    return sorted(str(layer.name) for layer in ctrl.viewer.layers)


def test_a_lateral_context_builds_only_the_lateral_work_labels():
    ctrl = _arm_labels_ctrl()
    ctrl._set_work_labels_visibility_for_context(
        patch_name="p1", target_view="lateral",
    )
    assert _names(ctrl) == ["p1_work_labels_lateral_3d"], (
        "the two axial volumes nobody is looking at cost 650-780 ms per load"
    )


def test_switching_to_axial_builds_the_two_axial_work_labels_on_demand():
    ctrl = _arm_labels_ctrl()
    ctrl._set_work_labels_visibility_for_context(
        patch_name="p1", target_view="lateral",
    )
    ctrl._set_work_labels_visibility_for_context(
        patch_name="p1", target_view="axial",
    )
    assert _names(ctrl) == [
        "p1_work_labels_axial_xz_3d",
        "p1_work_labels_axial_yz_3d",
        "p1_work_labels_lateral_3d",
    ]


def test_the_first_switch_to_a_view_sweeps_the_bar_while_it_builds():
    ctrl = _arm_labels_ctrl()
    seen = []
    real = ctrl._ensure_work_labels_layer

    def _spy(**kw):
        seen.append(ctrl.ui.patch_busy_bar.range)
        return real(**kw)

    ctrl._ensure_work_labels_layer = _spy
    ctrl._set_work_labels_visibility_for_context(
        patch_name="p1", target_view="axial",
    )
    assert seen == [(0, 0), (0, 0)], "~230 ms per view is not a silent freeze"
    assert ctrl.ui.patch_busy_bar.range == (0, 100), "back to idle afterwards"


def test_a_switch_with_nothing_to_build_does_not_flash_the_bar():
    ctrl = _arm_labels_ctrl()
    ctrl._set_work_labels_visibility_for_context(
        patch_name="p1", target_view="lateral",
    )
    ctrl.ui.patch_busy_bar.history.clear()
    ctrl._set_work_labels_visibility_for_context(
        patch_name="p1", target_view="lateral",
    )
    assert ctrl.ui.patch_busy_bar.history == []


def test_the_lazy_create_is_idempotent_across_repeat_calls():
    ctrl = _arm_labels_ctrl()
    for _ in range(3):
        ctrl._set_work_labels_visibility_for_context(
            patch_name="p1", target_view="axial",
        )
    assert len(ctrl.viewer.layers) == 2, "no duplicate layers on re-entry"


def test_a_view_never_shown_still_gets_its_layer_on_a_direct_ensure():
    """Segmenting on a view that was never displayed must still find a layer."""
    ctrl = _arm_labels_ctrl()
    ctrl._set_work_labels_visibility_for_context(
        patch_name="p1", target_view="lateral",
    )
    name = ctrl._ensure_work_labels_layer(view_name="YZ")
    assert name == "p1_work_labels_axial_yz_3d"
    assert "p1_work_labels_axial_yz_3d" in _names(ctrl)


def test_only_the_shown_layers_are_visible():
    ctrl = _arm_labels_ctrl()
    ctrl._set_work_labels_visibility_for_context(
        patch_name="p1", target_view="axial",
    )
    ctrl._set_work_labels_visibility_for_context(
        patch_name="p1", target_view="lateral",
    )
    by_name = {str(layer.name): layer for layer in ctrl.viewer.layers}
    assert by_name["p1_work_labels_lateral_3d"].visible is True
    assert by_name["p1_work_labels_axial_xz_3d"].visible is False


def test_the_three_eager_ensure_triples_are_gone_from_the_load_paths():
    """The load paths must not rebuild all three views on every patch click."""
    for meth in (
        TrainingAnnotationController._on_patch_list_clicked,
        TrainingAnnotationController._switch_patch_context_from_native,
        TrainingAnnotationController._launch_multiview_annotator,
    ):
        src = inspect.getsource(meth)
        assert 'view_name="XZ"' not in src, (
            f"{meth.__name__} still eagerly builds the axial volumes"
        )


def test_the_import_reload_hooks_do_not_eagerly_create_all_three_views():
    """A creation with napari's default visible=True would leave strays."""
    for meth in (
        TrainingAnnotationController._on_import_pairs_clicked,
        TrainingAnnotationController._on_import_external_clicked,
    ):
        src = inspect.getsource(meth)
        assert 'for v in ("XY", "XZ", "YZ")' not in src, (
            f"{meth.__name__}'s post-import reload still builds all three "
            "views; with lazy creation that CREATES visible XZ/YZ layers"
        )
        assert "_set_work_labels_visibility_for_context" in src, (
            f"{meth.__name__} must route through the visibility choke point"
        )


def test_a_post_import_refresh_in_lateral_mode_leaves_only_xy_visible():
    """The body the two import hooks now run, on a fresh (lazy) viewer."""
    ctrl = _arm_labels_ctrl()
    ctrl._set_work_labels_visibility_for_context(
        patch_name=ctrl.loaded_patch_name, target_view=ctrl.view_target,
    )
    visible = sorted(
        str(layer.name) for layer in ctrl.viewer.layers if layer.visible
    )
    assert visible == ["p1_work_labels_lateral_3d"], (
        "an import while in lateral mode must not strand visible XZ/YZ layers"
    )


def test_a_failed_view_build_does_not_skip_the_next_one():
    ctrl = _arm_labels_ctrl()
    real = ctrl._ensure_work_labels_layer
    tried = []

    def _flaky(**kw):
        tried.append(kw.get("view_name"))
        if kw.get("view_name") == "XZ":
            raise RuntimeError("XZ volume is the wrong shape")
        return real(**kw)

    ctrl._ensure_work_labels_layer = _flaky
    ctrl._set_work_labels_visibility_for_context(
        patch_name="p1", target_view="axial",
    )

    assert tried == ["XZ", "YZ"], "a failed XZ must not cost you YZ"
    assert "p1_work_labels_axial_yz_3d" in _names(ctrl)
    assert ctrl.ui.patch_busy_bar.range == (0, 100), "the scope still closes"


# ---------------------------------------------------------------------------
# The view radios fire the handler ONCE per click
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    """A real QApplication, so the radios deliver REAL `toggled` signals."""
    QW = pytest.importorskip("qtpy.QtWidgets")
    app = QW.QApplication.instance()
    if app is None:
        try:
            app = QW.QApplication([])
        except Exception as exc:            # pragma: no cover - headless CI
            pytest.skip(f"no Qt display: {exc}")
    return app


def test_one_click_on_a_real_view_radio_runs_the_body_once(qapp):
    """The guard is worth nothing if the surviving signal stops arriving."""
    from qtpy.QtWidgets import QButtonGroup, QRadioButton, QWidget

    host = QWidget()
    lateral = QRadioButton("Lateral", parent=host)
    axial = QRadioButton("Axial", parent=host)
    group = QButtonGroup(host)
    group.addButton(lateral)
    group.addButton(axial)
    group.setExclusive(True)
    lateral.setChecked(True)

    ctrl = _bare()
    ctrl.ui = None
    runs = []
    ctrl._ensure_runtime_defaults = lambda: runs.append(1)
    lateral.toggled.connect(ctrl._on_view_changed)
    axial.toggled.connect(ctrl._on_view_changed)

    axial.setChecked(True)               # ONE user click: lateral off, axial on
    assert runs == [1], (
        "an exclusive pair delivers two `toggled` signals per click; the body "
        f"must run for the one coming ON only (ran {len(runs)}x)"
    )

    lateral.setChecked(True)             # and back again
    assert runs == [1, 1]

    host.deleteLater()


def test_a_radio_going_off_does_not_run_the_view_handler():
    ctrl = _bare()
    ran = []
    ctrl._ensure_runtime_defaults = lambda: ran.append(1)
    ctrl._on_view_changed(False)
    assert ran == [], (
        "both radios are wired to `toggled`, so one click delivered two "
        "signals and ran the ~100 ms handler twice"
    )


def test_a_radio_coming_on_still_runs_the_view_handler():
    ctrl = _bare()
    ran = []
    ctrl._ensure_runtime_defaults = lambda: ran.append(1)
    ctrl.ui = None
    ctrl._on_view_changed(True)
    assert ran == [1]


def test_a_combo_text_signal_still_runs_the_view_handler():
    """`axial_plane_combo.currentTextChanged` passes a str, never a bool."""
    ctrl = _bare()
    ran = []
    ctrl._ensure_runtime_defaults = lambda: ran.append(1)
    ctrl.ui = None
    ctrl._on_view_changed("XZ")
    assert ran == [1]


def test_a_bare_call_still_runs_the_view_handler():
    ctrl = _bare()
    ran = []
    ctrl._ensure_runtime_defaults = lambda: ran.append(1)
    ctrl.ui = None
    ctrl._on_view_changed()
    assert ran == [1]
