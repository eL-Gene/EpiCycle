"""Qt-free stand-ins for the Step 4 controller's ``ui`` (B0 tests).

The controller methods under test run on a bare
``SegmentStackController.__new__`` instance with these fakes, the pattern of
``test_step4_adopt_workspace.py``.  Only the attributes the methods read are
provided; a fake that answers reads as well as writes, so a test cannot pass
by the fake swallowing what the code did.
"""
from __future__ import annotations

from napari_epicycle.segmentation.controller import SegmentStackController
from napari_epicycle.segmentation.state_manager import StackStateManager


class Line:
    def __init__(self, text=""):
        self._text = text
        self.sets = []
        self.styles = []

    def text(self):
        return self._text

    def setText(self, value):
        self._text = value
        self.sets.append(value)

    def setStyleSheet(self, s):
        self.styles.append(s)


class Log:
    def __init__(self):
        self.lines = []

    def append(self, line):
        self.lines.append(str(line))

    def text(self):
        return "\n".join(self.lines)


class Button:
    def __init__(self):
        self.enabled = True

    def setEnabled(self, v):
        self.enabled = bool(v)

    def isEnabled(self):
        return self.enabled


class Bar:
    def __init__(self):
        self.range = (0, 100)
        self.value = 0

    def setRange(self, a, b):
        self.range = (a, b)

    def setValue(self, v):
        self.value = v

    def maximum(self):
        return self.range[1]

    def setFormat(self, f):
        pass


class Combo:
    def __init__(self, index=0, text=""):
        self.index = index
        self._text = text

    def currentIndex(self):
        return self.index

    def setCurrentIndex(self, i):
        self.index = i

    def currentText(self):
        return self._text

    def findText(self, t):
        return -1


class DataCombo:
    """A combo with item data (Start from)."""

    def __init__(self):
        self.items = []
        self.index = -1

    def blockSignals(self, v):
        return False

    def clear(self):
        self.items = []
        self.index = -1

    def addItem(self, text, data=None):
        self.items.append((text, data))
        if self.index < 0:
            self.index = 0

    def removeItem(self, i):
        del self.items[i]
        self.index = min(self.index, len(self.items) - 1)

    def findData(self, data):
        for i, (_t, d) in enumerate(self.items):
            if d == data:
                return i
        return -1

    def setCurrentIndex(self, i):
        self.index = i

    def currentIndex(self):
        return self.index

    def currentData(self):
        return self.items[self.index][1] if 0 <= self.index < len(
            self.items) else None

    def currentText(self):
        return self.items[self.index][0] if 0 <= self.index < len(
            self.items) else ""

    def itemData(self, i):
        return self.items[i][1]

    def count(self):
        return len(self.items)


class Spin(Button):
    def __init__(self, v):
        super().__init__()
        self.v = v

    def value(self):
        return self.v


class Check(Button):
    def __init__(self, v=False):
        super().__init__()
        self.v = v

    def isChecked(self):
        return self.v


class UI:
    def __init__(self):
        self.workdir_line = Line()
        self.input_stacks_dir_line = Line()
        self.model_pack_line = Line()
        self.lateral_model_path = Line()
        self.axial_model_path = Line()
        self.run_status_label = Line()
        self.qc_status_line = Line()
        self.log_output = Log()
        self.model_source_combo = Combo(1)
        self.pretrained_combo = Combo(0, "cyto3")
        self.axes_combo = Combo(0, "ZCYX")
        self.chan1_combo = Combo(0)
        self.extract_features_btn = Button()
        self.qc_progress = Bar()
        self.smooth_all_btn = Button()
        self.finalize_btn = Button()
        self.save_tiers_btn = Button()
        self.delete_bad_btn = Button()
        self.undo_btn = Button()
        self.open_analysis_btn = Button()
        self.feature_reference_btn = Button()
        self.smooth_strength_combo = Button()
        self.fix_by_hand_btn = Button()
        self.reset_original_btn = Button()
        self.export_btn = Button()
        self.predict_btn = Button()
        self.reconstruct_btn = Button()
        self.cancel_btn = Button()
        self.run_progress = Bar()
        self.smooth_sigma_z_sb = Spin(3.0)
        self.smooth_sigma_xy_sb = Spin(1.5)
        self.smooth_preserve_topology_cb = Check(True)
        self._channel_rows = []
        self.listed = []
        self.rows = []
        self.ticked = []
        self.view_btn = Button()
        self.start_from_combo = DataCombo()
        self.chan2_combo = Combo(0)

    # B1a panel API (the controller computes, the panel shows)
    def refresh_stack_lists(self, rows, *, ticked=()):
        self.listed.append([r["name"] for r in rows])
        self.rows = list(rows)
        self.ticked = list(ticked)

    def update_row(self, name, record):
        pass

    def set_row_thumbnail(self, path, pixmap):
        return False

    def show_session_summary(self, main, amber, counts, n_ticked,
                             spacing_note=""):
        self.session_summary = (main, amber, dict(counts), n_ticked,
                                spacing_note)

    def show_run_row(self, n):
        self.run_row_n = n

    def show_model_card(self, text, colour, trained, missing=False):
        self.model_card = (text, colour, trained, missing)

    def show_ticked_line(self, dim, amber):
        self.ticked_line = (dim, amber)

    def show_seg_status(self, text, *, active=False):
        self.run_status_label.setText(text)

    def show_cell_width_px(self, text):
        self.cell_width_px = text

    def set_lock_lines(self, text=""):
        self.lock_line = text

    def ticked_names(self):
        return list(getattr(self, "ticked", []))

    def chosen_channels(self):
        return self.chan1_combo.currentIndex(), None

    def get_checked_channels(self):
        return []

    def set_channel_rows(self, n, axes_label=""):
        self.channel_rows = (n, axes_label)

    # ③ panel API (B1b; the controller computes, the panel shows)
    def show_mt_status(self, text, tip=None):
        self.qc_status_line.setText(text)

    def show_stack_card(self, name="", pill="", *, pill_grey=False,
                        line="", line_amber="", note=""):
        self.stack_card = (name, pill, pill_grey, line, line_amber, note)

    def show_mt_chips(self, *, measure="", measure_warn=False, review=None,
                      tidy=""):
        self.chips = (measure, measure_warn, review, tidy)

    def show_not_reconstructed(self, text):
        self.not_reconstructed = text

    def show_finalize_note(self, text):
        self.finalize_note = text

    def show_qc_bar(self, *, value=None, text=None, busy=None,
                    percent=False):
        if busy is True:
            self.qc_progress.setRange(0, 0)
        elif busy is False and self.qc_progress.maximum() == 0:
            self.qc_progress.setRange(0, 100)
        if value is not None:
            self.qc_progress.setValue(value)
        if text is not None:
            self.qc_bar_text = text

    def set_measure_button_text(self, text):
        self.measure_btn_text = text

    def set_mt_unsaved(self, name):
        self.mt_unsaved = name

    def select_mt_row(self, name):
        self.mt_selected = name

    def set_sigma_spacing(self, zyx=None):
        self.sigma_spacing = zyx

    def smooth_sigmas_voxels(self):
        return (self.smooth_sigma_z_sb.value(),
                self.smooth_sigma_xy_sb.value())


class RunStore:
    def list_manifests(self, _ws):
        return []


def controller(workspace=None):
    """A bare controller with fakes; the flags ``__init__`` would set."""
    ctrl = SegmentStackController.__new__(SegmentStackController)
    ctrl.ui = UI()
    ctrl.viewer = None
    ctrl.working_dir = str(workspace) if workspace else ""
    ctrl.state_manager = StackStateManager()
    if workspace:
        ctrl.state_manager.load(str(workspace))
    ctrl._suppress_publish = False
    ctrl._workflow_shell = None
    ctrl.run_store = RunStore()
    ctrl._init_runtime_state()
    ctrl._step4_active = True        # K6: tests stand in Step 4
    return ctrl
