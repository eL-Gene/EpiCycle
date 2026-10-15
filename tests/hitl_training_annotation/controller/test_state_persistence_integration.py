import json

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    TrainingAnnotationController,
)
from napari_epicycle.io_registry import HITLPersistenceService


def _controller_stub(tmp_path):
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.persistence = HITLPersistenceService()
    ctrl.working_dir = str(tmp_path)
    ctrl.current_round = 1
    ctrl.training_data_by_round = {
        1: [
            {
                "dataset_id": "dataset_a",
                "stack_id": "stack01",
                "view": "xy",
                "slice_index": 12,
                "channel_map": {"membrane": 0},
                "img_uri": "training_data/round_01/spatial/stack01_XY_z0012_img.tif",
                "mask_uri": "training_data/round_01/spatial/stack01_XY_z0012_mask.tif",
            }
        ]
    }
    ctrl.losses_by_round = {}
    ctrl.model_checkpoint_paths = {}
    ctrl.current_config = {"epochs": 200, "learning_rate": 0.01}
    ctrl.selected_view = "XY"
    return ctrl


def test_controller_state_persistence_roundtrip(tmp_path):
    ctrl = _controller_stub(tmp_path)
    ctrl._save_internal_state()

    # Clear in-memory values then load from disk.
    ctrl.current_round = 0
    ctrl.training_data_by_round = {}
    ctrl.losses_by_round = {}
    ctrl.model_checkpoint_paths = {}
    ctrl.current_config = {}
    ctrl.selected_view = "YZ"

    ctrl._load_internal_state()
    assert ctrl.current_round == 1
    # JSON serialization will coerce int dict keys to strings.
    assert "1" in ctrl.training_data_by_round or 1 in ctrl.training_data_by_round
    assert ctrl.selected_view == "XY"
    assert ctrl.current_config.get("epochs") == 200


def test_controller_load_uses_round_metadata_as_config_fallback(tmp_path):
    ctrl = _controller_stub(tmp_path)
    ctrl._save_internal_state()

    # Simulate legacy state file with empty current_config.
    state_path = tmp_path / "hitl_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["current_config"] = {}
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    ctrl.current_config = {}
    ctrl._load_internal_state()
    assert ctrl.current_config.get("epochs") == 200


# ============================================================================
# 2026-05-27 widget-state-persistence tests.  Added AFTER user surfaced two
# bugs that simple round-trip tests would have caught:
#
# (1) `_save_internal_state` writes self.current_config to disk, but
#     widget changes never reach self.current_config unless the user
#     clicks Train (training widgets) or Apply Inference Params
#     (inference widgets).  So between clicks, the user's UI changes
#     were silently discarded on save.  Fix: refresh current_config
#     from widgets at top of _save_internal_state via
#     `_refresh_current_config_from_widgets`.
#
# (2) First draft of the fix REPLACED current_config wholesale with
#     widget collector output, dropping non-widget-managed keys like
#     `pretrained_model_path` (set during training-complete promotion).
#     Fix: merge widget values into current_config instead of replacing.
#
# Both bugs would have been caught by tests below.  Write the test first next time.
# ============================================================================


class _StubSpinBox:
    """Minimal QSpinBox / QDoubleSpinBox stub for unit tests."""
    def __init__(self, initial=0):
        self._v = initial
    def value(self):
        return self._v
    def setValue(self, v):
        self._v = v


class _StubCheckBox:
    def __init__(self, initial=False):
        self._c = bool(initial)
        self.toggled = _StubSignal()
    def isChecked(self):
        return self._c
    def setChecked(self, c):
        new_c = bool(c)
        if self._c != new_c:
            self._c = new_c
            self.toggled.emit(self._c)


class _StubLineEdit:
    def __init__(self, initial=""):
        self._t = str(initial)
    def text(self):
        return self._t
    def setText(self, t):
        self._t = str(t)


class _StubSignal:
    """Stub Qt signal that just notifies registered callbacks."""
    def __init__(self):
        self._cbs = []
    def connect(self, cb):
        self._cbs.append(cb)
    def emit(self, *args):
        for cb in self._cbs:
            cb(*args)


class _StubComboBox:
    def __init__(self, items=(), initial=""):
        self._items = list(items)
        self._t = str(initial) if initial else (self._items[0] if self._items else "")
        self.currentTextChanged = _StubSignal()
        self._signals_blocked = False
    def currentText(self):
        return self._t
    def setCurrentText(self, t):
        if str(t) in self._items and self._t != str(t):
            self._t = str(t)
            if not self._signals_blocked:
                self.currentTextChanged.emit(self._t)
    def findText(self, t):
        try:
            return self._items.index(str(t))
        except ValueError:
            return -1
    def addItems(self, items):
        self._items.extend(str(i) for i in items)
    def clear(self):
        self._items = []
        self._t = ""
    def count(self):
        return len(self._items)
    def itemText(self, i):
        return self._items[i] if 0 <= i < len(self._items) else ""
    def currentIndex(self):
        try:
            return self._items.index(self._t)
        except ValueError:
            return -1
    def setCurrentIndex(self, i):
        if 0 <= i < len(self._items):
            self.setCurrentText(self._items[i])
    def blockSignals(self, blocked):
        prev = self._signals_blocked
        self._signals_blocked = bool(blocked)
        return prev


def _controller_with_ui(tmp_path):
    """Stub controller with a SimpleNamespace UI carrying the widgets that
    `_collect_training_config` and `_apply_inference_params_from_ui` read.
    """
    from types import SimpleNamespace
    ctrl = _controller_stub(tmp_path)
    ctrl.ui = SimpleNamespace(
        # Training Setup (Basic)
        epochs_sb=_StubSpinBox(500),
        batch_sb=_StubSpinBox(8),
        device_gpu=_StubCheckBox(True),
        device_cpu=_StubCheckBox(False),
        model_selector=_StubComboBox(["scratch", "cyto3", "nuclei"], "scratch"),
        starting_checkpoint_path=_StubLineEdit(""),
        # Advanced Parameters (training)
        lr_db=_StubSpinBox(0.01),
        wd_db=_StubSpinBox(0.0001),
        minmask_sb=_StubSpinBox(5),
        chan1_combo=_StubComboBox(["0: gray", "1: red", "2: green", "3: blue"], "2: green"),
        chan2_combo=_StubComboBox(["0: none", "1: red", "2: green", "3: blue"], "1: red"),
        norm_cb=_StubCheckBox(True),
        rescale_cb=_StubCheckBox(True),
        flow_cb=_StubCheckBox(True),
        # Inference Parameters
        inference_channel1_combo=_StubComboBox(["0: gray", "1: red", "2: green", "3: blue"], "2: green"),
        inference_channel2_combo=_StubComboBox(["0: none", "1: red", "2: green", "3: blue"], "1: red"),
        inference_flow_thresh=_StubSpinBox(0.4),
        inference_cellprob_thresh=_StubSpinBox(0.0),
        inference_diameter_sb=_StubSpinBox(0),
        inference_niter_dynamics=_StubSpinBox(0),
        inference_normalize_cb=_StubCheckBox(True),
        inference_rescale_cb=_StubCheckBox(True),
        inference_invert_cb=_StubCheckBox(False),
    )
    # Disable status-log writer (no real UI).
    ctrl._append_ui_log = lambda *a, **kw: None
    # Stub viewer attribute — _refresh_inference_channel_options checks
    # self.viewer / self.loaded_patch_name and short-circuits when both
    # are falsy.  This lets _sync_inference_controls_from_config run
    # cleanly without a real napari viewer.
    ctrl.viewer = None
    ctrl.loaded_patch_name = ""
    ctrl.selected_view = "XY"
    # Run the channel-pair sync wiring that __init__ would normally do
    # via _connect_signals.  Bypass-init stubs skip that, so wire it
    # manually here.  Without this, training-side widget changes don't
    # propagate to the inference combo (and vice versa).
    ctrl._channel_sync_in_progress = False
    for src in ("chan1_combo", "chan2_combo",
                "inference_channel1_combo", "inference_channel2_combo"):
        combo = getattr(ctrl.ui, src, None)
        if combo is not None and hasattr(combo, "currentTextChanged"):
            combo.currentTextChanged.connect(
                lambda t, s=src: ctrl._sync_channel_combos(s, t)
            )
    for src in ("norm_cb", "rescale_cb",
                "inference_normalize_cb", "inference_rescale_cb"):
        cb = getattr(ctrl.ui, src, None)
        if cb is not None and hasattr(cb, "toggled"):
            cb.toggled.connect(
                lambda c, s=src: ctrl._sync_bool_checkboxes(s, c)
            )
    return ctrl


def test_save_captures_training_widget_changes_without_train_click(tmp_path):
    """The original bug (2026-05-27).  User changes epochs from 500 to 100
    in the widget without clicking Train.  Pre-fix:
    `_save_internal_state` wrote self.current_config (still 500) to disk
    and the widget change was lost on reload.
    Post-fix: save refreshes current_config from widgets first.
    """
    ctrl = _controller_with_ui(tmp_path)
    ctrl.ui.epochs_sb.setValue(100)
    ctrl.ui.lr_db.setValue(0.005)
    ctrl.ui.chan1_combo.setCurrentText("1: red")
    ctrl.ui.chan2_combo.setCurrentText("0: none")
    ctrl.ui.norm_cb.setChecked(False)

    ctrl._save_internal_state()

    assert ctrl.current_config.get("epochs") == 100
    assert ctrl.current_config.get("learning_rate") == 0.005
    assert ctrl.current_config.get("channel1") == "1: red"
    assert ctrl.current_config.get("channel2") == "0: none"
    assert ctrl.current_config.get("normalize") is False


def test_save_captures_inference_widget_changes_without_apply_click(tmp_path):
    """Same bug, inference side.  User changes Dynamics iterations from 0
    to 200 in Inference Parameters without clicking Apply.  Pre-fix the
    change never reached self.current_config.
    """
    ctrl = _controller_with_ui(tmp_path)
    ctrl.ui.inference_niter_dynamics.setValue(200)
    ctrl.ui.inference_diameter_sb.setValue(30)
    ctrl.ui.inference_flow_thresh.setValue(0.7)
    ctrl.ui.inference_invert_cb.setChecked(True)

    ctrl._save_internal_state()

    assert ctrl.current_config.get("n_iterations") == 200
    assert ctrl.current_config.get("diameter") == 30
    assert ctrl.current_config.get("flow_threshold") == 0.7
    assert ctrl.current_config.get("invert") is True


def test_save_preserves_non_widget_managed_keys(tmp_path):
    """The regression I caused (first draft of save-side refresh).
    `_refresh_current_config_from_widgets` originally REPLACED
    self.current_config wholesale with widget collector output,
    dropping `pretrained_model_path` (set elsewhere during training).
    Fix: merge widget values into current_config, not replace.
    """
    ctrl = _controller_with_ui(tmp_path)
    ctrl.current_config["pretrained_model_path"] = "/some/checkpoint/path"
    ctrl.current_config["custom_legacy_key"] = "should-survive"

    ctrl._save_internal_state()

    assert ctrl.current_config.get("pretrained_model_path") == "/some/checkpoint/path"
    assert ctrl.current_config.get("custom_legacy_key") == "should-survive"
    # And widget-managed keys still got captured.
    assert ctrl.current_config.get("epochs") == 500


def test_load_with_dynamic_channel_name_no_patch_yet(tmp_path):
    """The bug user surfaced 2026-05-27.  Saved cfg holds the
    viewer-channel-aware text 'channel1=1: cdh1', 'channel2=0: none'.
    On workspace reload, no patch is selected yet so channel combos
    still have GENERIC items ['0: gray', '1: red', '2: green', '3: blue'].

    Pre-fix: _set_combo_text_if_present tried to set the combo to
    '1: cdh1' via findText, which fails because the item is named
    '1: red' at this point.  Silent no-op → combo stays at default
    '2: green' → integer index lost.

    Post-fix: sync sets combo by INTEGER INDEX extracted from the
    cfg value.  Even though the label doesn't match (no patch
    loaded), the integer index is preserved.  Later when patch
    loads and labels become viewer-aware ('1: cdh1', '2: h2b'),
    the saved integer index resolves to the correct channel.
    """
    ctrl = _controller_with_ui(tmp_path)
    # Simulate prior session: write state.json directly with the
    # viewer-aware channel labels that would have been saved when a
    # patch was loaded (cdh1 + h2b).  Bypass _save_internal_state
    # because that re-snapshots from widgets which currently show the
    # generic defaults.
    saved_state = {
        "schema_version": 2,
        "current_round": 0,
        "current_round_by_view": {"lateral": 0, "axial": 0},
        "training_data_by_round": {},
        "losses_by_round": {},
        "model_checkpoint_paths": {},
        "round_backend_info": {},
        "current_config": {
            "channel1": "1: cdh1",
            "channel2": "0: none",
            "epochs": 100,
        },
        "selected_view": "XY",
        "view_target": "lateral",
        "axial_plane": "XZ",
        "sample_records": {},
        "active_curation_checkpoints": {"lateral": "", "axial": ""},
        "curation_model_source_mode": "auto",
        "imported_curation_model_path": "",
        "curation_pretrained_model_type": "cyto3",
        "training_mode": "cumulative",
        "auto_update_model_after_training": True,
        "loaded_stack_path": "",
        "loaded_patch_name": "",
    }
    state_path = tmp_path / "hitl_state.json"
    state_path.write_text(json.dumps(saved_state, indent=2), encoding="utf-8")

    # Sanity: widgets are at hardcoded defaults BEFORE load.
    assert ctrl.ui.inference_channel1_combo.currentText() == "2: green"
    assert ctrl.ui.inference_channel1_combo.currentIndex() == 2

    ctrl._load_internal_state()

    # After load: combo's current TEXT may be generic ('1: red'), but
    # its current INTEGER INDEX should match the saved value (1).
    # When a patch later loads + items refresh to '1: cdh1, 2: h2b',
    # the existing integer index of 1 will resolve to '1: cdh1'.
    assert ctrl.ui.inference_channel1_combo.currentIndex() == 1, \
        f"Expected combo at index 1, got {ctrl.ui.inference_channel1_combo.currentIndex()}"
    assert ctrl.ui.inference_channel2_combo.currentIndex() == 0, \
        f"Expected combo at index 0, got {ctrl.ui.inference_channel2_combo.currentIndex()}"
    # Training combos should also be in sync (set to same integer index).
    assert ctrl.ui.chan1_combo.currentIndex() == 1
    assert ctrl.ui.chan2_combo.currentIndex() == 0


def test_round_trip_widget_changes_persist(tmp_path):
    """End-to-end: change widgets, save, simulate fresh session by
    clearing in-memory state, load, verify widgets get the saved values."""
    ctrl = _controller_with_ui(tmp_path)
    ctrl.ui.epochs_sb.setValue(120)
    ctrl.ui.lr_db.setValue(0.002)
    ctrl.ui.chan1_combo.setCurrentText("1: red")
    ctrl.ui.chan2_combo.setCurrentText("0: none")
    ctrl.ui.inference_diameter_sb.setValue(45)
    ctrl.ui.inference_niter_dynamics.setValue(300)
    ctrl.ui.inference_invert_cb.setChecked(True)
    ctrl._save_internal_state()

    # Simulate fresh session: reset widgets to defaults + clear cfg.
    ctrl.ui.epochs_sb.setValue(500)
    ctrl.ui.lr_db.setValue(0.01)
    ctrl.ui.chan1_combo.setCurrentText("2: green")
    ctrl.ui.chan2_combo.setCurrentText("1: red")
    ctrl.ui.inference_diameter_sb.setValue(0)
    ctrl.ui.inference_niter_dynamics.setValue(0)
    ctrl.ui.inference_invert_cb.setChecked(False)
    ctrl.current_config = {}

    ctrl._load_internal_state()

    # Widgets should now reflect the saved values.
    assert ctrl.ui.epochs_sb.value() == 120
    assert ctrl.ui.lr_db.value() == 0.002
    assert ctrl.ui.chan1_combo.currentText() == "1: red"
    assert ctrl.ui.chan2_combo.currentText() == "0: none"
    assert ctrl.ui.inference_diameter_sb.value() == 45
    assert ctrl.ui.inference_niter_dynamics.value() == 300
    assert ctrl.ui.inference_invert_cb.isChecked() is True
