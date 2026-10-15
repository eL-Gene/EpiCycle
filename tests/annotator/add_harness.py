"""A Qt-free harness for the Add orchestrator (Add rebuild B2.3+).

The real ``Curation3DAnnotator`` methods whose names start with ``_add`` /
``_on_add`` / ``_launch_add`` / ``_stop_add`` / ``_apply_add`` /
``_request_add`` / ``_preview_add`` (plus a few helpers they call) are
borrowed onto a plain Python class, so a test drives the REAL code without a
QWidget.  ``FakeThread`` stands in for ``QThread`` (the test emits its
``started`` / ``finished`` by hand, and ``finished`` only after something
called ``quit()`` -- as a real thread) and ``FakeWorker`` is the real
``RefineWorker`` with ``moveToThread`` made a no-op.
"""

from __future__ import annotations

import numpy as np

from napari_epicycle.annotator import curation_3d_widget as CW
from napari_epicycle.annotator import reshape_worker as RW

C = CW.Curation3DAnnotator

_PREFIXES = ("_add", "_on_add", "_launch_add", "_stop_add", "_apply_add",
             "_request_add", "_preview_add", "_accept_add", "_reject_add",
             "_reset_add", "_teardown_add", "_start_add", "_invalidate_add",
             "_recenter_add", "_set_add_box", "_do_add", "_schedule_add",
             "_sync_add", "_undo_add", "_render_add", "_remove_add",
             "_refine_add", "_batch_refine_add", "_push_add",
             "_refresh_add", "_get_add", "_discard_add", "_restore_side",
             "_render_add_balls", "_build_add", "_close_add",
             "_install_add", "_drop_add", "_show_add", "_on_add_show")
_EXTRA = ("_signal_sender", "_result_token", "_refine_channel_source",
          "_layer_by_name", "_on_add_box_changed", "_compute_add_histogram",
          "_on_add_otsu_reset", "_on_add_clahe_toggled",
          "_on_add_threshold_changed", "_on_add_advanced_changed",
          "_ensure_shared_data_aligned")


class FakeSignal:
    def __init__(self):
        self.cbs = []

    def connect(self, cb):
        self.cbs.append(cb)

    def disconnect(self, *a):
        self.cbs.clear()

    def emit(self, *a):
        for cb in list(self.cbs):
            cb(*a)


class FakeThread:
    instances: list = []
    fail_start = False
    stuck = False

    def __init__(self, *a, **k):
        self.started = FakeSignal()
        self.finished = FakeSignal()
        self.quit_called = False
        FakeThread.instances.append(self)

    def start(self):
        if FakeThread.fail_start:
            raise RuntimeError("start failed")

    def quit(self):
        self.quit_called = True

    def wait(self, _ms):
        return not FakeThread.stuck

    # test helpers
    def run(self):
        self.started.emit()

    def end(self):
        # A real QThread's ``finished`` fires only once its event loop has
        # been told to quit (L-M3, the B2 fix pass): a harness that fired it
        # regardless hid a launch that never connected ``quit`` -- busy stuck
        # on for ever against a real thread while every test here was green.
        assert self.quit_called, (
            "FakeThread.end(): nothing called quit() -- a real QThread "
            "would never emit finished")
        self.finished.emit()


class FakeWorker(RW.RefineWorker):
    def moveToThread(self, _th):      # noqa: N802 - Qt name
        return None


class FakeAdv:
    """The panel, as far as the orchestrator reads it."""

    def __init__(self):
        self.pushes = []
        self.preset = "nuclei"
        self.source = ("paint", None)
        self.wall = {"channel": "mem", "chain": "none",
                     "wall_kind": "membrane"}
        self.nuclei_channel = "nuc"
        self.params = {"channel": "nuc", "clahe": False, "clahe_clip": 0.01,
                       "min_size": 0, "separate": True, "min_distance": 3,
                       "clear_border": True, "edge_margin": 3,
                       "auto_smooth": False, "smooth_sigma": 1.5,
                       "dark_outside": True}
        self.checked = None
        adv = self

        class _Combo:
            def currentText(self):          # noqa: N802 - Qt name
                return adv.params.get("channel", "")

        self.add_channel_combo = _Combo()

    def update_add_ui_state(self, **kw):
        self.pushes.append(kw)

    def get_add_params(self):
        return dict(self.params)

    def get_add_preset(self):
        return self.preset

    def get_add_marker_source(self):
        return self.source

    def get_add_wall_params(self):
        return dict(self.wall)

    def add_nuclei_channel_name(self):
        return self.nuclei_channel

    def get_checked_add_pieces(self):
        return list(self.checked or [])

    def get_selected_add_pieces(self):
        return []

    def last(self, key):
        for kw in reversed(self.pushes):
            if key in kw:
                return kw[key]
        return None


class _SM:
    def __init__(self, vol):
        self.shared_labels = {"L": vol}

    def register_shared_data(self, name, data):
        self.shared_labels[name] = data


class _Viewer:
    def __init__(self, layers):
        self.layers = layers


class Harness:
    """The borrowed orchestrator with stubs for everything visual."""

    def __init__(self, vol, images=None, spacing=(1.0, 1.0, 1.0)):
        from napari.layers import Image

        self.adv = FakeAdv()
        self.sync_manager = _SM(vol)
        self._active_labels_name = "L"
        self.viewer = _Viewer([Image(a, name=n)
                               for n, a in (images or {}).items()])
        self.ctx = {"patch_name": "P", "run_epoch": 1, "spacing": spacing,
                    "workspace": ""}
        self.rendered = []
        self.sender = None
        self.locks = []
        self._refine_parked_threads = []
        self._add_threshold_value = None
        self._add_running = False
        self._add_review = False
        self._add_box = None
        self._add_roi = None
        self._add_preview_result = None
        self._add_session = []
        self._add_piece_ids = []
        self._add_box_timer = None
        self._add_candidate_layer = None
        self._add_box_overlay = None
        self._add_box_shapes = None
        for k in ("_add_busy", "_add_stopping", "_add_info_dirty"):
            setattr(self, k, False)
        for k in ("_add_thread", "_add_worker", "_add_info_thread",
                  "_add_info_worker", "_add_info", "_add_transient",
                  "_add_preview_roi", "_add_preview_res", "_add_run_guide",
                  "_add_run_state", "_add_nuclei_threshold_value"):
            setattr(self, k, None)
        self._add_token = 0
        self._add_info_token = 0
        self._add_dark_cache = {}
        self._add_click_markers = []
        self._add_marker_layer = None
        self._add_place_markers_on = False

    # --- stubs ---------------------------------------------------------- #
    def _get_advanced_features_widget(self):
        return self.adv

    def _reshape_context(self):
        return dict(self.ctx)

    def _refine_spacing(self):
        return tuple(self.ctx["spacing"])

    def _refine_spacing_known(self):
        return tuple(self.ctx["spacing"]) != (1.0, 1.0, 1.0)

    def _signal_sender(self):
        return self.sender

    def _render_add_candidate(self, result, roi=None):
        self.rendered.append((result, roi))

    def _remove_add_candidate(self):
        self._add_candidate_layer = None

    def _remove_add_box_shapes(self):
        self._add_box_overlay = None

    def _render_add_box(self):
        pass

    def _remove_temp_layer(self, key):
        pass

    def _add_temp_layer(self, key, array, **kw):
        self.temp_layers = getattr(self, "temp_layers", {})
        self.temp_layers[key] = (array, kw)
        return object()

    def _sync_all_views_after_operation(self, **_k):
        pass

    def _restore_ready_to_draw_state(self):
        pass

    def _lock_subviewers_for_preview(self):
        pass

    def _fire_add_mode_lock(self, active):
        self.locks.append(bool(active))

    def _add_main_label_layer(self):
        class _L:
            data = self.sync_manager.shared_labels["L"]
            scale = (1.0, 1.0, 1.0)
            translate = (0.0, 0.0, 0.0)
        return _L()

    def _schedule_add_histogram(self):
        pass

    def _compute_add_histogram(self, **_k):
        pass

    def _add_crosshair_center(self, shape):
        return tuple(int(s) // 2 for s in shape)

    def _layer_by_name(self, layers, name):
        for layer in layers:
            if layer.name == name:
                return layer
        return None

    # a box for the tests
    def place_box(self, box_slices):
        from packages.epicycle_annotator.roi_box import BoxRec

        self._add_running = True
        self._add_roi = tuple(box_slices)
        c = [(s.start + s.stop) // 2 for s in box_slices]
        z = [s.stop - s.start for s in box_slices]
        self._add_box = BoxRec(*c, *z)


Harness._ADD_SETTINGS = C._ADD_SETTINGS
Harness._ADD_BALL_LAYER = C._ADD_BALL_LAYER
Harness._ADD_SIDE_VIEWS = C._ADD_SIDE_VIEWS
Harness._squeeze_image_channel = staticmethod(C._squeeze_image_channel)
for _name in dir(C):
    if _name.startswith(_PREFIXES) or _name in _EXTRA:
        if _name in Harness.__dict__:
            continue
        _fn = getattr(C, _name)
        if callable(_fn):
            setattr(Harness, _name, _fn)


def install_fakes(monkeypatch):
    """QThread -> FakeThread and RefineWorker -> FakeWorker, at the points
    the orchestrator imports them from."""
    import qtpy.QtCore

    FakeThread.instances = []
    FakeThread.fail_start = False
    FakeThread.stuck = False
    monkeypatch.setattr(qtpy.QtCore, "QThread", FakeThread)
    monkeypatch.setattr(RW, "RefineWorker", FakeWorker)


def nuclei_volume():
    from .test_add_nuclei_golden import nuclei_fixture

    img, labels = nuclei_fixture(0)
    return img, labels


def marker_zyx(h):
    """The click markers' positions, in placement order; the
    markers are records ``{"zyx", "id", "ticked"}``; a bare tuple a test
    assigned directly is read as its own position)."""
    return [tuple(m["zyx"]) if isinstance(m, dict) else tuple(m)
            for m in h._add_click_markers]


def blank(shape=(12, 20, 20)):
    return np.zeros(shape, np.int32)
