"""3D Curation annotator — specialized multiview widget for Reconstruct + Curate 3D.

Inherits all subviewer infrastructure from NativeMultiviewAnnotator (viewer models,
Qt viewers, managers, sliders, crosshairs, sync, drawing, layer lifecycle).  Replaces
the HITL-specific Quick Annotate controls (Segment, Accept, Overlay) with 3D
curation tools: contour interpolation, focused refinement navigation, and an
Advanced Features button.

Phase K5.3.
"""

from __future__ import annotations

# [XCRASH-FAULT] Arm faulthandler so a native segfault (Qt/vispy/GL)
# prints a Python traceback before the process dies, instead of a
# silent exit.  Idempotent — multiple calls are fine.  Output goes to
# stderr by default which the napari terminal captures.
import faulthandler as _fh
if not _fh.is_enabled():
    _fh.enable()
    print("[XCRASH-FAULT] faulthandler armed (segfault traceback enabled)", flush=True)

# Crash-logger breadcrumb helpers — persistent, flushed-to-disk diagnostics.
# The module-level faulthandler above writes to stderr (lost on a hard crash);
# install_crash_logging() (called at annotator launch) re-points faulthandler
# at a file and enables these breadcrumbs.  Guarded so a missing/failed import
# can never break this module's load; no-op until install runs.
try:
    from napari_epicycle.utils.crash_logger import (
        breadcrumb as _crumb,
        liveness as _liveness,
    )
except Exception:  # pragma: no cover
    def _crumb(*_a, **_k):
        return None

    def _liveness(**_k):
        return ""

from dataclasses import dataclass, field
import time as _time
from pathlib import Path as _Path
from typing import Callable

def _defer_to_event_loop(fn):
    """Run ``fn`` on the next Qt event-loop turn.

    A single named seam so the deferral is (a) obvious at the call site and
    (b) monkeypatchable in Qt-free tests.  See the 2026-09-07 right-click
    crash fix in ``_make_subviewer_right_click_handler``.
    """
    from qtpy.QtCore import QTimer

    QTimer.singleShot(0, fn)



import numpy as np
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from qtpy.QtCore import QEvent, QObject, Qt

from napari.layers import Image, Labels

from napari_epicycle.annotator.native_multiview_widget import NativeMultiviewAnnotator
from napari_epicycle.annotator.refinement_state import RefinementOpState
from napari_epicycle.ui_shell.theme import (
    DESTRUCTIVE_BTN_STYLE,
    DIM_LABEL_STYLE,
    INPUT_STYLE,
    PRIMARY_BTN_STYLE,
    PRIMARY_BTN_STYLE_CYAN,
    PRIMARY_BTN_STYLE_GREEN,
    PRIMARY_BTN_STYLE_TEAL,
    SECONDARY_BTN_STYLE,
    SECTION_HEADER_STYLE,
    SEPARATOR_LINE_STYLE,
    STATUS_LABEL_STYLE,
    help_button as _help_button,
)


# 2026-06-01: per-section help buttons (?) for the 3D annotator.  Guides
# live under docs/guides/ (an installed wheel: napari_epicycle/guides/);
# ui_shell._guides.guides_dir() picks the folder.
from napari_epicycle.ui_shell._guides import guides_dir as _guides_dir
_ANNOTATOR_GUIDES_DIR = _guides_dir()

#: Brick U3: the ceiling on the wall picture, in BYTES.
#:
#: Deliberately NOT ``reshape.spec.DEFAULT_SIZE_BUDGET``, which is a
#: voxels x seeds PRODUCT -- reusing it would be a unit error hiding as
#: a constant.  64 MB rather than something roomier because a 3-D Image
#: layer is also a GPU TEXTURE UPLOAD, and the dev workstation's 8 GB of
#: VRAM is already carrying the labels volume, the isolation proxy and
#: every open channel.
#:
#: **It is measured against the ROI CROP, which is why 64 MB is the
#: right number again.**  The first build put
#: the elevation into a volume-shaped array; the live driver then
#: measured the owner's patch at **(317, 480, 480)** -- 292 MB -- and
#: the tick was refused on every Preview of the run, i.e. the feature
#: was dead on the only data it exists for.  The layer is the crop now,
#: placed with ``translate`` + ``scale``, which is what a
#: full-size array would be FOR (alignment) at 1/110th the memory: the same
#: object's crop was ``((34, 103), (45, 144), (59, 156))`` = 69x99x97 =
#: **2.7 MB**.  So the cap is a guard on the pathological case again --
#: 64 MB of float32 is 16 M voxels, about 24x the crop measured above --
#: rather than a ceiling the ordinary case walks into.
WALLS_DISPLAY_BUDGET_BYTES = 64 * 1024 * 1024


@dataclass
class OperationSlot:
    """Describes an operation that can have uncommitted work."""

    name: str                          # e.g., "smooth", "split", "interpolation"
    is_active: Callable[[], bool]      # Returns True if this op has uncommitted work
    description: Callable[[], str]     # Human-readable reason for dialog
    reject: Callable[[], None]         # Clean rejection/discard
    priority: int = 0                  # Higher priority = checked first


def _safe_curation_debug(obj, msg: str) -> None:
    """Best-effort debug call that tolerates lightweight test doubles."""
    dbg = getattr(obj, "_curation_debug", None)
    if callable(dbg):
        try:
            dbg(msg)
        except Exception:
            pass


_ARROW_NAMES = {Qt.Key_Up: "Up", Qt.Key_Down: "Down",
                Qt.Key_Left: "Left", Qt.Key_Right: "Right"}


def _add_shift_nudge(event, orient, nudge_cb) -> bool:
    """The Shift branch both side-view key filters run FIRST (grill E-M10): Shift + an arrow goes to Add's box nudge, which
    answers True when it took the key.  Anything else -- no callback, no
    Shift, not an arrow, Add not active -- is False, and the filter falls
    through to its own code unchanged."""
    if nudge_cb is None or orient is None:
        return False
    try:
        if event.type() != QEvent.KeyPress:
            return False
        name = _ARROW_NAMES.get(event.key())
        if name is None or not (event.modifiers() & Qt.ShiftModifier):
            return False
        return bool(nudge_cb(orient, name))
    except Exception as exc:
        print(f"[ADD] Shift+arrow nudge failed: {exc!r}")
        return False


class _UpDownBlocker(QObject):
    """Event filter that routes Up/Down arrows to label navigation.

    Installed on ALL focusable children of subviewer panels (sliders,
    canvas overlays, etc.).  Up/Down triggers prev/next label navigation
    in the Label Navigator instead of scrolling subviewers.  With a
    *nudge_callback* and an *orient*, Shift + an arrow goes to Add's box
    nudge first (see :func:`_add_shift_nudge`).
    """

    def __init__(self, navigate_callback=None, parent=None,
                 nudge_callback=None, orient=None):
        super().__init__(parent)
        self._navigate_cb = navigate_callback
        self._nudge_cb = nudge_callback
        self._orient = orient

    def eventFilter(self, obj, event):
        if _add_shift_nudge(event, self._orient, self._nudge_cb):
            return True  # consume: Add moved (or refused) the box
        if event.type() == QEvent.KeyPress and event.key() in (Qt.Key_Up, Qt.Key_Down):
            if self._navigate_cb is not None:
                direction = "prev" if event.key() == Qt.Key_Up else "next"
                try:
                    self._navigate_cb(direction)
                except Exception:
                    pass
            return True  # consume
        return False


class _SliceStepFilter(QObject):
    """Event filter on subviewer canvases for arrow-key slice scrolling.

    Left/Right scroll by the step size from the spinbox, clamping to
    [0, max_slice].  Up/Down navigate to prev/next label in the
    Label Navigator.
    """

    def __init__(self, viewer_model, step_spinbox, navigate_callback=None,
                 parent=None, nudge_callback=None, orient=None):
        super().__init__(parent)
        self._vm = viewer_model
        self._step_sb = step_spinbox
        self._navigate_cb = navigate_callback
        self._nudge_cb = nudge_callback
        self._orient = orient

    def eventFilter(self, obj, event):
        if event.type() != QEvent.KeyPress:
            return False
        if _add_shift_nudge(event, self._orient, self._nudge_cb):
            return True  # consume: Add moved (or refused) the box
        key = event.key()

        # Up/Down → navigate to prev/next label in Navigator
        if key in (Qt.Key_Up, Qt.Key_Down):
            if self._navigate_cb is not None:
                direction = "prev" if key == Qt.Key_Up else "next"
                try:
                    self._navigate_cb(direction)
                except Exception:
                    pass
            return True  # consume

        if key not in (Qt.Key_Left, Qt.Key_Right):
            return False

        step = max(1, int(self._step_sb.value()))
        if step <= 1:
            return False  # default napari handling is fine at step=1

        try:
            current = int(self._vm.dims.current_step[0])
            max_slice = int(self._vm.dims.range[0][1]) - 1
            if max_slice <= 0:
                return False

            if key == Qt.Key_Right:
                new = current + step
                if new > max_slice:
                    new = max_slice
            else:  # Left
                new = current - step
                if new < 0:
                    new = 0

            if new != current:
                self._vm.dims.set_current_step(0, new)
        except Exception:
            return False
        return True  # consume the event so napari doesn't also step by 1



def _heal_render(widget, layer, *, tag: str = "") -> int:
    """Call ``widget._heal_label_render`` when present (test fakes may lack
    it); never raises.  See ``Curation3DAnnotator._heal_label_render``."""
    heal = getattr(widget, "_heal_label_render", None)
    if not callable(heal):
        return 0
    try:
        return int(heal(layer, tag=tag))
    except Exception:
        return 0


#: napari's own words when it refuses to draw a contour, which it does
#: whenever the sliced labels have more than two dimensions -- i.e. in
#: the main curation viewer, which is 3D (audit H2, 2026-09-20).  The
#: write is still correct and still reaches the three side views; only
#: the toast is filtered, and only this one message.  Module level, not
#: a class attribute, because a test fake binds the method without
#: inheriting anything else from the class.
_CONTOUR_3D_WARNING = "Contours are not displayed"

#: Add's outline (A-M1 of the A1-A4 audit): napari re-raises the
#: 3D-contour warning on EVERY refresh of a contoured 3D layer -- a slider
#: move, a dims step, an in-place candidate refill -- not only on the
#: ``contour`` write, so a filter around the write alone let it through.
#: While the Add outline is on a live candidate layer, this ONE message is
#: ignored process-wide (napari's warning names no layer, so it cannot be
#: filtered per layer); the entry is removed when the outline goes off or
#: the candidate layer is removed.  ``None`` = not installed.
_ADD_OUTLINE_FILTER = None


def _add_outline_filter(on: bool) -> None:
    """Install / remove the one-message filter (idempotent both ways).

    Residual: the filter is process-wide on that one
    message while it is installed, so the same napari warning from another
    3D contoured layer is also silent for that time; and a
    ``warnings.catch_warnings()`` block that was ENTERED before the install
    and exits after it restores the older filter list, dropping the entry --
    the next outline write installs it again."""
    import warnings

    global _ADD_OUTLINE_FILTER
    entry = _ADD_OUTLINE_FILTER
    installed = entry is not None and entry in warnings.filters
    if on:
        if installed:
            return
        warnings.filterwarnings("ignore", message=_CONTOUR_3D_WARNING + ".*")
        _ADD_OUTLINE_FILTER = warnings.filters[0]
        return
    if installed:
        try:
            warnings.filters.remove(entry)
            mutated = getattr(warnings, "_filters_mutated", None)
            if mutated is not None:
                mutated()
        except Exception:  # noqa: BLE001
            pass
    _ADD_OUTLINE_FILTER = None


#: The contour's two "nothing moved" sentences that are keyed on a reason
#: the pure function reports.  The push one names BOTH
#: numbers and says which to change; the limit one names the knob.
REFINE_CONTOUR_PUSH_BELOW_GATE = (
    "Nothing moved. 'Push outward' is {push:g} and the Edge gate is "
    "{gate:.2f} \u2014 below \u00b1{gate:.2f} the push does nothing "
    "anywhere. Lower the gate on the Advanced card, or push harder."
)
REFINE_CONTOUR_AT_LIMIT = (
    "Nothing moved: every move the outline tried went further than "
    "{limit:.1f} um, so all were undone. Raise 'Edge may move'."
)
#: B2 audit UX-M3: the neighbour fence took every move.
REFINE_CONTOUR_FENCED = (
    "Nothing moved: every move the outline tried was into a neighbouring "
    "object, and no other object ever loses a voxel."
)
#: State 8's S1 for a step refused before it ran.
REFINE_NOTHING_TO_SHOW = "Nothing to show - the object is unchanged."

#: The gate's channel refusal, in S1 and in Preview's tooltip.
REFINE_PICK_CHANNEL = "Pick a channel first."

#: ...and its no-object refusal (B3 audit LOW): Preview is
#: greyed until an object is selected, the reason in S1 and the tooltip.

# ---------------------------------------------------------------------------
# Refine recipes -- recording the chain.
#
# Module-level on purpose: the Refine tests drive the real `_preview_smooth`
# on BARE fakes that bind only the methods they list, and a recorder they
# would each have to learn to bind is a recorder that breaks two dozen test
# files for no behaviour.  `ann` is the annotator (or such a fake).
# ---------------------------------------------------------------------------


def _refine_clear_records(ann) -> None:
    """Beside every ``_refine_steps = []`` (all seven sites)."""
    ann._refine_step_records = []
    ann._refine_pending_record = None
    ann._refine_chain_recipe_names = []


def _refine_capture_record(ann, method, params, prep_ticks):
    """The record of the step about to run -- or ``None``, never raises.

    ``sdf`` / ``contour`` / ``settle`` are built from the ``params`` dict
    their branches run on; the other four from their widgets, which their
    branches read from the same moment on.
    """
    from packages.epicycle_annotator import refine_recipe as rr

    try:
        if method in ("sdf", "contour", "settle"):
            return rr.record_from_params(method, params or {},
                                         prep_ticks=prep_ticks)
        adv = ann._get_advanced_features_widget()
        if adv is None or not hasattr(adv, "get_refine_step_params"):
            return None
        return rr.RecipeStep(method, dict(adv.get_refine_step_params(method)))
    except Exception as exc:  # noqa: BLE001 - a record is a nicety
        print("[RECIPE] step not recorded: {!r}".format(exc))
        return None


def _refine_provenance(ann):
    """``(steps, recipes)`` for the Accept callback's debug trail."""
    steps = []
    for rec in list(getattr(ann, "_refine_step_records", []) or []):
        try:
            steps.append(rec.to_json() if rec is not None else None)
        except Exception:
            steps.append(None)
    return steps, list(getattr(ann, "_refine_chain_recipe_names", []) or [])


def _recipe_filter_ui(ann, run, kw):
    """While a recipe runs: remember what the three buttons were ASKED to
    be, compose the stage line, keep Stop up."""
    shadow = run.setdefault("shadow", {})
    for key in ("preview_enabled", "accept_enabled", "reject_enabled",
                "preview_reason"):
        if key in kw:
            shadow[key] = kw[key]
    if "stage_text" in kw or "busy_text" in kw:
        stage = kw.pop("stage_text", None)
        if stage is None:
            stage = kw.pop("busy_text", None)
        kw.pop("busy_text", None)
        kw["stage_text"] = ann._recipe_stage_text(stage or "")
    if any(k in kw for k in ("busy", "stop_visible", "stop_enabled")):
        kw["stop_visible"] = True
        kw["stop_enabled"] = not _recipe_stopping(ann, run)
    return kw


def _recipe_stopping(ann, run) -> bool:
    return bool(getattr(ann, "_refine_stopping", False)
                or (run or {}).get("stop_requested"))


def _recipe_hold_buttons(ann, adv) -> None:
    """Preview / Accept / Discard are DISABLED for the whole run; Stop is
    visible and live (unless it is already stopping)."""
    from packages.epicycle_annotator.refine_recipe import S1_RUNNING_BLOCK

    for attr in ("smooth_preview_btn", "smooth_accept_btn",
                 "smooth_reject_btn"):
        w = getattr(adv, attr, None)
        if w is not None:
            w.setEnabled(False)
    pb = getattr(adv, "smooth_preview_btn", None)
    if pb is not None:
        pb.setToolTip(S1_RUNNING_BLOCK)
    stop = getattr(adv, "refine_cancel_preview_btn", None)
    if stop is not None:
        stop.setVisible(True)
        stop.setEnabled(not _recipe_stopping(
            ann, getattr(ann, "_refine_recipe_run", None)))

REFINE_SELECT_OBJECT = "Select an object first."

#: S1 on an idle Refine page -- the panel's own starting sentence.
REFINE_IDLE_STATUS = "Select an object, then pick a method."

#: The margin every method but Settle cuts its ROI with -- the default of
#: `_snapshot_label_roi`, named so the region growth and
#: the first snapshot agree.  Only Settle's need can exceed it
#: (`_refine_step_margin`).
REFINE_SHARED_ROI_MARGIN = 15


def _batch_recount(dlg) -> None:
    """Re-count the Batch dialog's targets after a refusal put it back to
    idle, so the count line and Run agree with the refusal next to them
    (wiring audit M1).  Tolerates a test fake without the method."""
    fn = getattr(dlg, "refresh_count", None)
    if fn is None:
        return
    try:
        fn(force=True)
    except Exception as exc:  # noqa: BLE001 -- never raise into a refusal
        print(f"[BATCH] recount after a refusal failed: {exc!r}")


def _batch_perf(what, t0) -> None:
    """A ``[BATCH-PERF]`` line for the count path -- env-gated like the
    ``label_bboxes`` roi print (``EPICYCLE_SYNC_DEBUG=1``), silent by
    default; a dev probe for the public-cut strip list."""
    import os

    if os.environ.get("EPICYCLE_SYNC_DEBUG") == "1":
        print(f"[BATCH-PERF] {what} ms={(_time.perf_counter() - t0) * 1000:.1f}")


def _raise_batch_dialog(dlg) -> None:
    """Bring the open Batch dialog to the front (a door that found it open)."""
    for fn in ("show", "raise_", "activateWindow"):
        try:
            getattr(dlg, fn)()
        except Exception:  # noqa: BLE001 -- a test fake, a closing window
            pass


def _safe_attr(obj, name, default=None):
    """``getattr`` that never raises -- a bare ``__new__`` Qt instance (the
    test fakes) raises RuntimeError, not AttributeError, for a missing
    attribute."""
    try:
        return getattr(obj, name, default)
    except Exception:
        return default

#: The sentences S1 holds only because the GATE put them there; once the
#: gate's condition clears it may replace them (B1b audit C-L2).
_REFINE_GATE_SENTENCES = frozenset((REFINE_PICK_CHANNEL,
                                    REFINE_SELECT_OBJECT))


def _qobject_is_deleted(obj) -> bool:
    """True when ``obj`` is a sip wrapper whose C++ side is gone.

    A plain Python object (a test fake) is never "deleted".
    """
    try:
        from qtpy import sip as _sip
        return bool(_sip.isdeleted(obj))
    except Exception:
        return False


def _afw_pair_alive(dock, widget) -> bool:
    """Is a cached Advanced Features (dock, widget) pair still the live one?

    Cheap checks only (no ``findChildren``): neither side deleted, the dock
    still parented (napari's ``remove_dock_widget`` unparents it) and still
    holding the same widget.
    """
    try:
        if _qobject_is_deleted(dock) or _qobject_is_deleted(widget):
            return False
        if hasattr(dock, "parent") and dock.parent() is None:
            return False
        if hasattr(dock, "widget") and dock.widget() is not widget:
            return False
        return True
    except Exception:
        return False


class Curation3DAnnotator(NativeMultiviewAnnotator):
    """Multiview annotator specialized for 3D label curation.

    Replaces the HITL Quick Annotate group with 3D curation controls:
    draw + contour interpolation + focused refinement navigation.
    """

    # --- Reshape R1: CLASS-level defaults for the guided-split flags ------
    # These must resolve without touching instance state.  `_refinement_op_active`
    # is read by the navigator's isolation gate, and PyQt6 raises RuntimeError
    # ("super-class __init__() was never called"), NOT AttributeError, for a
    # missing attribute on a QObject whose C++ side is not constructed -- so
    # `getattr(self, "_reshape_running", False)` does NOT fall back to its
    # default there.  A class attribute resolves on the type, never reaching
    # that machinery.  Caught by `tests/annotator/test_isolation_op_guard.py`,
    # whose fixture builds the widget with `__new__` and no `__init__`.
    _reshape_running = False
    _reshape_active = False

    # The de-novo Add ROI box is drawn as vispy Line OVERLAYS (a 3D wireframe on
    # the main canvas + a 2D rectangle per subviewer) via ``box_overlay`` — NOT a
    # napari Shapes layer.  Rationale: a Shapes-in-3D layer's mesh VBO could be
    # freed while a draw was still queued on teardown/patch-switch, faulting the
    # 3D canvas with a glDrawArrays access violation (dev-PC crash pid75780,
    # 2026-07-03).  An overlay never enters viewer.layers, so it cannot orphan.
    # This flag remains as a kill-switch (set False to hide the box entirely).
    _ADD_BOX_RENDER_ENABLED = True

    def __init__(self, viewer):
        # Drawing undo state — snapshot before first stroke on a label.
        self._draw_snapshot = None     # ROI data before first draw stroke
        self._draw_roi = None          # tuple of slices for the snapshot region
        self._draw_label_id = None     # label ID that was drawn on

        # Interpolation state — must exist before super().__init__ calls _wire_signals.
        self._interp_snapshot = None   # ROI crop snapshot before Preview
        self._interp_roi = None        # tuple of slices for the snapshot region
        self._interp_label_id = None   # label ID that was interpolated
        self._interp_running = False   # guard against concurrent Preview clicks
        self._interp_refine_mode = False  # True when refining after interpolation

        # Split state — mirrors interpolation state pattern
        self._split_snapshot = None      # ROI crop snapshot before seed painting
        self._split_roi = None           # tuple of slices for the snapshot region
        # --- Reshape R1: guided Split ----------------
        # `_reshape_active` is THE gate.  It is set at Start from the two
        # pickers and is False for every legacy split, so every guarded
        # branch below is inert on the path users already know.
        self._reshape_active = False     # this split session is guided
        self._reshape_running = False    # the engine worker is in flight
        self._reshape_thread = None
        self._reshape_worker = None
        self._reshape_result = None      # the full ReshapeResult
        self._reshape_final_ids = {}     # interim piece id -> final label id
        self._reshape_flags = {}         # final label id -> arbiter flag
        self._reshape_guide = None       # the matched guide array
        self._reshape_guide_layer = None # the UNMANAGED read-only Labels layer
        # Brick 2a: the temporary-layer REGISTRY.  `_add_guide_layer` was
        # a single slot, so a markers layer would have evicted the mask.
        # key -> layer; every one is `guide_`-prefixed, unmanaged and
        # read-only.  The mask keeps key 'mask' and its own two removal
        # sites (it deliberately survives Accept/Discard); 'markers' is
        # removed at five.
        self._temp_guide_layers = {}
        # Brick U3: does the user want the wall picture?
        # The INTENT, not the checkbox -- a re-Preview puts every
        # result-shaped control back to idle, the tick included.
        self._walls_wanted = False
        # --- Brick B1: the Refine tab's worker, its stale guards, its
        # staged-step bookkeeping and its own wall picture.
        # `_refine_running` is true while a RefineWorker is in flight OR a
        # result is staged, because a staged preview is already written
        # into `shared_data`.
        self._refine_running = False
        self._refine_thread = None
        self._refine_worker = None
        self._refine_token = 0
        self._refine_busy_kind = None       # None | "contour" | "settle"
        self._refine_launch_labels_name = None
        self._refine_launch_patch = None
        self._refine_launch_run_epoch = None
        self._refine_launch_roi = None
        self._refine_launch_label = None
        self._refine_lock_fired = False
        self._refine_staged_method = None   # which method made what is staged
        self._refine_steps = []             # the S5 breadcrumb, in order
        # what each staged step RAN with, one record per
        # breadcrumb word (invariant R1: the two lists are the same length
        # after every Refine event), the record of the step in flight, and
        # every recipe that contributed to the staged chain.
        self._refine_step_records = []
        self._refine_pending_record = None
        self._refine_chain_recipe_names = []
        # the one run in flight (None when idle) and the
        # token bumped at every start AND every end, so a late judge from a
        # finished run can never act on the next one.
        self._refine_recipe_run = None
        self._refine_recipe_token = 0
        self._recipe_stage_left = False
        self._refine_elevation = None       # the wall picture, if any
        self._refine_elevation_roi = None
        self._refine_pending_elevation = None   # a worker step's, until it writes
        self._refine_walls_wanted = False   # the INTENT, not the checkbox
        self._refine_stopping = False       # a Stop was asked for; the worker is still ending
        # L5: declared here rather than created lazily inside
        # `_refine_thread_cleanup`, so a reader (a teardown, a test, a
        # driver) never has to guess whether the attribute exists yet.
        self._refine_parked_threads = []
        self._refine_live_messages = []
        # The marker-find worker, its own stale guards, and which of the
        # two jobs the one Stop button is currently for.
        self._seed_find_running = False
        self._seed_find_thread = None
        self._seed_find_worker = None
        self._seed_find_token = 0
        self._seed_find_launch_token = None
        self._seed_find_launch_labels_name = None
        self._seed_find_launch_patch = None
        self._seed_find_launch_run_epoch = None
        self._split_busy_kind = None     # None | "find" | "solve"
        self._reshape_guide_path = ""    # an external mask picked via Browse
        self._reshape_guide_run = ""     # provenance: which run it came from
        self._reshape_override = False   # the user pressed "Split anyway"
        self._reshape_arbiter_merged = False
        # Stale-finish guards: a lock stops a SECOND op
        # starting; it does nothing about a worker finishing after the
        # user has moved on.  All three are captured at launch and
        # re-checked in the finish handler, which early-returns on any
        # mismatch.
        self._reshape_token = 0
        self._reshape_launch_labels_name = None
        self._reshape_launch_patch = None
        self._reshape_launch_run_epoch = None
        self._reshape_lock_fired = False
        self._guide_thread = None    # the guide READ worker
        self._guide_worker = None
        self._guide_pending = None
        #: The last Find's seed crop (ROI-shaped), kept so a painted
        #: addition merges WITH the found markers instead of replacing
        #: them on the next Preview, plus the rows the
        #: Find put in the list.
        self._reshape_found_seeds = None
        self._reshape_found_marker_ids: list[int] = []
        self._reshape_found_marker_stats: dict = {}
        # Threads that would not stop in time.  Held alive until they finish
        # on their own -- dropping a running QThread aborts the process
        # (see _reshape_thread_cleanup).
        self._reshape_parked_threads = []
        self._split_label_id = None      # label ID being split
        self._split_running = False      # guard against concurrent clicks
        self._split_seed_ids = []        # list of seed label IDs assigned
        # Brick U1: the SOURCE OF TRUTH for the markers
        # that are painted on the Refine layer.  `_split_seed_ids` is
        # overwritten with PIECE ids by a preview, so it cannot be it:
        # after one Preview the "marker" list would have been listing
        # pieces, the recipe would have handed the whole previous
        # partition back as seeds, and Undo would have erased a piece.
        self._split_painted_marker_ids: list[int] = []
        #: What the last Find materialised into the Refine layer (H2).
        self._materialised_ids: list[int] = []
        # Brick U2b: the voxels of every marker that is
        # currently UNTICKED, keyed by id and held ROI-relative.  An
        # unticked marker comes OFF the object (owner, 2026-09-20), so
        # its shape has to live somewhere until the user ticks it again
        # -- "roughly back" would be a silent edit of the thing being
        # cut.  A parked marker keeps its row, its id and its place in
        # `_split_painted_marker_ids`; only its voxels are lifted.
        self._parked_marker_voxels: dict[int, tuple] = {}
        # ...and the materialise's own verdict, read by the finish
        # handler to decide whether to keep the crop.  Initialised here
        # so it is never a missing attribute (audit M6).
        self._materialise_ok = False
        self._split_previewed = False    # True after Preview runs watershed
        self._split_isolate_was_on = False  # whether isolate was on before split
        self._refine_layer = None        # temporary "Refine" Labels layer
        self._refine_baseline = None     # ROI snapshot of refine layer (before seeds)

        # De-novo "Add" mode state (2026-06-30) — mirrors the split pattern.
        self._add_running = False        # True while Add mode holds the surface
        self._add_box = None             # roi_box.BoxRec (center+size, ZYX)
        self._add_roi = None             # slices-tuple for the box ROI
        self._add_candidate_layer = None  # full-volume temp Labels (candidate)
        self._add_box_shapes = None      # legacy ref (kept null; see box_overlay)
        self._add_box_overlay = None     # vispy Line overlay (3D wireframe + subviewer rects)
        self._add_last_size = None        # sticky box size (sz,sy,sx) across Place Box
        self._add_preview_result = None  # last watershed result (the candidate)
        self._add_piece_ids = []         # committed piece label IDs (REVIEW)
        self._add_review = False         # True after Accept (REVIEW state)
        # Per-Accept session log — one entry {ids} per box, so REVIEW lists every
        # piece this session and "Undo all" knows how many controller-side
        # add_denovo undo entries to pop.
        self._add_session = []
        self._add_mode_lock_cb = None    # controller lock callback
        self._on_add_committed_cb = None  # controller commit callback
        self._on_add_undo_session_cb = None  # controller undo-N-commits callback
        self._add_threshold_value = None  # current threshold (channel value space)
        self._add_box_timer = None       # debounce timer for box recompute
        # Add rebuild B2.3: the preview worker, the one
        # box-info worker, Stop as a state, the transient S1 sentence.
        self._add_busy = False
        self._add_stopping = False
        self._add_token = 0
        self._add_thread = None
        self._add_worker = None
        self._add_info_token = 0
        self._add_info_thread = None
        self._add_info_worker = None
        self._add_info_dirty = False
        self._add_info = None
        # B4.2: the Brightness inputs (preset, channel, Local contrast) the
        # landed info's histogram and Auto were measured with
        self._add_info_brightness_key = None
        # final fix pass B1: the user set the Brightness threshold (typed or
        # dragged) -- an untouched one follows every new Auto landing; the
        # value the last Auto wrote, so a value set any other way is kept
        self._add_threshold_touched = False
        self._add_threshold_auto_written = None
        self._add_brightness_channel_seen = None
        self._add_dark_cache = {}
        self._add_transient = None
        self._add_preview_roi = None
        self._add_preview_res = None
        # records {"zyx", "id", "ticked"}; the next_label
        # the ids were reserved against; the markers epoch (stale guard)
        self._add_click_markers = []
        self._add_markers_base = None
        self._add_absorbed_ids = set()     # B-M1: the last Preview's
        self._add_rebase_note = None       # B-M2: a pending re-base
        self._add_volume_max = None        # B-L3: (id(shared), max)
        self._add_info_spacing = None      # A-L3
        self._add_markers_epoch = 0
        self._add_launch_markers_epoch = 0
        self._add_run_guide = None
        self._add_run_state = None
        self._add_nuclei_threshold_value = None
        self._add_nuclei_threshold_typed = False
        self._add_place_markers_on = False
        # "Show candidates as outlines"
        self._add_contour_on = False
        self._add_marker_layer = None
        self._add_side_callbacks_installed = False
        self._split_main_visible = True  # main labels visibility before split
        self._refine_pre_preview = None  # seed painting state before watershed (for re-seeding)
        self._split_fragment_refine = False   # True when in fragment refine mode
        self._split_fragment_ids: list[int] = []  # label IDs of split fragments
        self._fragment_refine_roi = None  # cached ROI for fast proxy bbox during fragment refine
        # U4.  Fragment entry WRITES the pieces, and the
        # panel's piece list is cleared the moment it does -- so the two
        # things the commit record needs and nothing else still holds are
        # captured here: which pieces were ticked, and how many voxels
        # the entry wrote.  None means "this session never entered
        # fragment mode from an engine result", which is what routes the
        # fragment Accept back to the legacy split callback.
        self._reshape_fragment_ticked = None
        self._reshape_fragment_written = 0

        # Auto-detect seed state
        self._detected_peaks: list[dict] | None = None
        self._auto_detect_active: bool = False
        self._seed_id_base: int = 0  # cached max label ID for seed allocation

        # Track last navigated label for deferred delete detection.
        self._last_navigated_label_id = None
        # Debounce timer for real-time fill-delete detection.
        self._alive_check_timer = None
        # Lifecycle guard: prevents deferred callbacks from firing during teardown.
        self._is_closing = False
        # Re-entrancy guard for _close_cleanup (prevents double-execution).
        self._close_cleanup_done = False
        # Brick U2: "Show this object as an outline".  On the
        # ANNOTATOR, not on the panel, so it survives an Accept, the next
        # object, and a panel rebuilt mid-session -- `_create_refine_layer`
        # reads it, so the annotator's value is the one that wins.
        self._refine_contour_on = False
        # Brick U2: what the LAST cut recipe asked of the
        # expected-count stage, so the result line can report the target
        # that was actually sent rather than whatever the spin box says
        # by the time the worker comes back.  Diagnostic only; nothing
        # branches on it.
        self._reshape_count_diag = {}
        # Session token for deferred callback gating across close/relaunch.
        self._session_token = 1

        # Smooth state — mirrors split state pattern
        self._smooth_running = False
        self._smooth_snapshot = None
        self._smooth_roi = None
        self._smooth_label_id = None
        self._smooth_chain_count = 0

        # Phase 1 of the refinement-state centralization. The dataclass
        # mirrors the per-op flags above into a single typed object so
        # external callers (e.g. the merge dialog gate) can use named
        # predicates instead of inline boolean expressions. Refresh via
        # `self._sync_refine_state()` before reading.
        self._refine_state = RefinementOpState()

        # Floating MergeDialog instance (None when no merge in progress).
        self._merge_dialog = None

        # Probability volume cache — lazy-loaded for MorphGAC edge stopping
        self._prob_volume_path: str | None = None
        self._prob_volume: np.ndarray | None = None

        # Multi-label isolation state
        self._isolation_active = False
        self._pre_isolation_colormap = None
        self._colormap_authority = None  # Injected by controller (WP-3)

        # Phase 2: 3D isolation proxy display layer
        self._isolation_proxy_layer = None
        self._isolation_proxy_active = False
        self._isolation_proxy_label_id = None
        # Tracks the frozen label set the proxy currently displays when
        # used in multi-label mode (e.g. merge dialog tick-to-isolate).
        # None when the proxy is in single-label mode (uses
        # show_selected_label) or inactive. Set by _apply_multi_label_proxy.
        self._multi_label_proxy_set = None
        # Multi-label proxy uses N small single-label crops (one per
        # selected label) — each rendered via the proven show_selected_label
        # 3D path so the visual matches refine isolation. dict[label_id, Labels].
        # Empty when not in multi-label mode.
        self._multi_label_proxy_layers = {}

        # Triage callbacks — set by controller via set_triage_callbacks()
        # _on_mark_tier_cb fires the unified flag-only path for the
        # navigator's 5 tier buttons (Curated/Good/Review/Bad/Garbage).
        # _on_delete_label_cb fires the destructive ghost-row path when
        # the volume is manually erased (e.g. paint 0 in a subviewer).
        self._on_mark_tier_cb = None
        self._on_delete_label_cb = None
        self._on_discard_garbage_cb = None
        self._on_merge_cb = None
        self._on_isolation_cleared_cb = None

        # Navigator state — floating Label Navigator widget
        self._navigator_widget = None
        self._navigator_dock = None  # QDockWidget ref for robust close
        self._cached_flagged_items = []

        # Paint-isolation state (companion, 2026-05-16) — opt-in
        # toggle that auto-isolates whatever label the brush is
        # currently set to.  Tracks the navigator's
        # "Isolate while painting" checkbox.  Independent from the
        # main isolation system (last write wins on the underlying
        # _on_isolation_changed dispatch).
        # - _paint_isolation_enabled: the checkbox state (persists for
        #   the session, survives patch switches)
        # - _paint_isolation_active: is paint-isolation currently
        #   driving the isolated label?  Reset on close / patch switch
        #   / toggle OFF / navigator-driven isolation override.
        # - _paint_isolation_label_id: last label_id we isolated via
        #   paint events; used to deduplicate event firings (skipping
        #   the ~245ms isolation cost when the same label re-fires).
        self._paint_isolation_enabled = False
        self._paint_isolation_active = False
        self._paint_isolation_label_id = None

        # super().__init__ sets _3d_curation_mode=False (line 57 in parent),
        # then calls _build_ui() → _build_draw_group() (our override),
        # then _wire_signals() → our override.
        super().__init__(viewer)
        # Override the parent's defaults now that init is complete.
        self._3d_curation_mode = True
        self._active_target_mode = "3d_curation"
        self._auto_increment_labels = False
        self._apply_draw_settings()
        self._apply_edit_constraints()
        # Auto-load: parent __init__ already called _refresh_layer_options()
        # which populated image_combo/labels_combo. Load if both have a selection.
        # Use a short timer so the dock is fully attached before heavy subviewer work.
        # NOTE: set_context(auto_load=True) also schedules _auto_load_if_ready
        # at 150ms.  Use a cancellable timer so set_context can cancel this one,
        # preventing a double load (SLOW PATH + redundant FAST PATH).
        from qtpy.QtCore import QTimer
        self._init_auto_load_timer = QTimer()
        self._init_auto_load_timer.setSingleShot(True)
        self._init_auto_load_timer.setInterval(100)
        self._init_auto_load_timer.timeout.connect(self._auto_load_if_ready)
        self._init_auto_load_timer.start()
        self._deferred_auto_load_timer = QTimer()
        self._deferred_auto_load_timer.setSingleShot(True)
        self._deferred_auto_load_timer.timeout.connect(self._auto_load_if_ready)

        # Operation slot registry — must be last, after all state attrs exist.
        self._register_operation_slots()

    def _sync_refine_state(self) -> None:
        """Mirror per-op flags into the centralized RefinementOpState.

        Phase 1 of the refinement-state centralization. Callers that
        need fresh state (e.g. cross-dock predicates like the merge
        dialog gate) should call this before reading
        ``self._refine_state``. Cheap (~15 attribute reads).

        Phase 2 will migrate the per-op flags into the dataclass as
        the source of truth, at which point this adapter goes away.

        """
        s = self._refine_state
        # Interpolation family
        s.interp_active = bool(getattr(self, "_interp_running", False))
        s.interp_refine_mode = bool(getattr(self, "_interp_refine_mode", False))
        s.interp_snapshot = getattr(self, "_interp_snapshot", None)
        s.interp_label_id = getattr(self, "_interp_label_id", None)
        s.interp_roi = getattr(self, "_interp_roi", None)
        # Split family
        s.split_active = bool(getattr(self, "_split_running", False))
        s.split_previewed = bool(getattr(self, "_split_previewed", False))
        s.split_fragment_refine = bool(getattr(self, "_split_fragment_refine", False))
        s.split_snapshot = getattr(self, "_split_snapshot", None)
        s.split_label_id = getattr(self, "_split_label_id", None)
        s.split_roi = getattr(self, "_split_roi", None)
        s.split_fragment_ids = list(getattr(self, "_split_fragment_ids", []))
        # Smooth family
        s.smooth_active = bool(getattr(self, "_smooth_running", False))
        s.smooth_chain_count = int(getattr(self, "_smooth_chain_count", 0))
        s.smooth_snapshot = getattr(self, "_smooth_snapshot", None)
        s.smooth_label_id = getattr(self, "_smooth_label_id", None)
        s.smooth_roi = getattr(self, "_smooth_roi", None)
        # Drawing / merge (mirror annotator-side flags that gate the
        # destructive label-action buttons — see
        # ``can_start_destructive_label_op``).
        s.draw_snapshot_present = getattr(self, "_draw_snapshot", None) is not None
        s.merge_dialog_open = getattr(self, "_merge_dialog", None) is not None
        # Batch Refine — active whenever its dialog is present (open),
        # whether idle / running / staged.  Mutual-exclusion with other
        # ops is enforced via any_op_active.
        s.batch_refine_active = getattr(self, "_batch_refine_dialog", None) is not None

    def _refinement_op_active(self) -> bool:
        """Return True iff any refinement op (interp / split / smooth) is
        currently staged or in refine mode.  Used to gate cross-op
        actions like "Merge Into" that must not run concurrently with
        a staged op.
        """
        return bool(
            getattr(self, "_split_running", False)
            # Reshape sits INSIDE the navigator isolation gate from the
            # start, so a mid-preview isolation click does not
            # fight the tool.  `_split_running` already covers the whole
            # guided session (it rides `_start_split`); the explicit flag
            # is belt-and-braces for the worker window.
            or getattr(self, "_reshape_running", False)
            or getattr(self, "_smooth_running", False)
            or getattr(self, "_interp_snapshot", None) is not None
            or getattr(self, "_interp_refine_mode", False)
        )

    def _is_op_active_for_navigator(self) -> bool:
        """Predicate handed to ``CurationLabelNavigator.set_callbacks``
        so the navigator's three user-initiated isolation entry points
        (`_on_isolate_toggled`, `_fire_isolation_if_active`,
        `_on_render_selected_clicked`) can self-gate while a refinement
        op is staged.

        Toggling isolation while a split or interpolation preview is staged
        must not leave the Refine layer hidden or contradictory, so the
        navigator asks this predicate (is any refinement op staged?) before
        acting.

        Returns ``_refinement_op_active()`` — the most conservative
        predicate available, covering interp snapshot held, interp
        refine-further mode, split running, and smooth running.  More
        inclusive than ``_refine_state.any_op_active`` (which doesn't
        cover refine-mode without a fresh ``_sync_refine_state()``).
        """
        active = self._refinement_op_active()
        if active:
            # Diagnostic — when the gate trips, surface which flag
            # caused it so the user can see in the console exactly
            # what op the navigator thinks is in flight.
            print(
                f"[ISO-GATE] _is_op_active_for_navigator -> True "
                f"split={getattr(self, '_split_running', False)} "
                f"smooth={getattr(self, '_smooth_running', False)} "
                f"interp_snapshot="
                f"{getattr(self, '_interp_snapshot', None) is not None} "
                f"interp_refine_mode="
                f"{getattr(self, '_interp_refine_mode', False)}"
            )
        return active

    def _get_toggle_targets_for_name(self, name: str) -> list:
        """FM8 override (2026-05-16): when isolation is active, the
        channel-aware visibility key mapped to the LABELS layer must
        toggle the *visible* artifact (proxy / Refine layer), not the
        main labels layer (which the isolation system intentionally
        hid).  The a/s/d/f channel toggles must act on the artifact that is
        actually visible, not on a layer isolation hides on purpose.

        Adaptive keybinding context: ``_setup_channel_keybindings``
        maps the four keys (``a/s/d/f``) positionally — image channels
        first, then labels.  With 1 image channel, ``a`` = image and
        ``s`` = labels.  With 2 image channels, ``a/s`` = images and
        ``d`` = labels.  This override applies the isolation-aware
        routing ONLY when ``name`` is a labels layer.  Image-channel
        key presses always fall through to the default behavior,
        regardless of which isolation mode is active.

        Priority order (first match wins, applied ONLY when `name`
        is in `_active_labels_names`):
          1. ``_refine_layer`` — split active (or fragment refine).
             Toggle the Refine layer in main + Refine_<orient> in
             each subviewer.
          2. ``_multi_label_proxy_layers`` non-empty — multi-label
             isolation (lasso / merge dialog).  Toggle all N sub-
             proxies in main + subviewer label layers (which use
             DirectLabelColormap for multi-iso).
          3. ``_isolation_proxy_layer`` non-None AND ndisplay == 3 —
             single-label 3D isolation.  Toggle proxy in main +
             subviewer label layers.
          4. Fallback to ``super()._get_toggle_targets_for_name`` —
             no isolation, 2D mode, OR ``name`` is an image channel.
        """
        # Image-channel gate: any name that is NOT a labels layer
        # falls straight through to the default behavior, regardless
        # of which isolation mode is active.  Critical for 2+ channel
        # workflows where 'a' (image ch1) must not route to the
        # Refine layer just because a split happens to be running.
        labels_names = list(
            getattr(self, "_active_labels_names", []) or []
        )
        if name not in labels_names:
            print(
                f"[CHKEY] FM8 routing: name={name!r} is NOT a labels "
                f"layer (labels={labels_names}) — falling through to "
                f"default image-channel toggle"
            )
            return super()._get_toggle_targets_for_name(name)

        # while Add runs or a candidate exists, the
        # labels key also toggles the candidate, the click balls and their
        # side-view copies.
        if (getattr(self, "_add_running", False)
                or getattr(self, "_add_candidate_layer", None) is not None):
            extra = self._add_toggle_targets()
            if extra:
                targets = list(super()._get_toggle_targets_for_name(name))
                targets += [t for t in extra if t not in targets]
                print(f"[CHKEY] FM8 routing: mode=ADD name={name!r} "
                      f"n_targets={len(targets)}")
                return targets

        # Priority 1: Refine layer (split / fragment refine)
        if getattr(self, "_refine_layer", None) is not None:
            targets = [self._refine_layer]
            # Subviewers in split mode have their own Refine_<orient>
            # layers; route through the subviewer manager if available.
            sub_mgr = getattr(self, "subviewer_manager", None)
            if sub_mgr is not None:
                for orient in ("XY", "XZ", "YZ"):
                    sub = sub_mgr.get_subviewer_layer("Refine", orient)
                    if sub is not None:
                        targets.append(sub)
            print(
                f"[CHKEY] FM8 routing: mode=SPLIT/REFINE name={name!r} "
                f"n_targets={len(targets)} "
                f"refine_layer={getattr(self._refine_layer, 'name', '?')}"
            )
            return targets

        # Priority 2: Multi-label isolation (lasso / merge)
        multi_proxies = getattr(self, "_multi_label_proxy_layers", None)
        if multi_proxies:
            targets = list(multi_proxies.values())
            # Subviewers still use the main-labels-name pattern for
            # multi-label mode (DirectLabelColormap on main labels
            # subviewer copies).
            for vm in [self.viewer_model_xy, self.viewer_model_xz, self.viewer_model_yz]:
                orient = self._orientation_for_vm(vm)
                sub_layer = (
                    self._layer_by_name(vm.layers, f"{name}_{orient}")
                    or self._layer_by_name(vm.layers, name)
                )
                if sub_layer is not None:
                    targets.append(sub_layer)
            print(
                f"[CHKEY] FM8 routing: mode=MULTI-ISO name={name!r} "
                f"n_targets={len(targets)} "
                f"multi_proxy_ids={sorted(multi_proxies.keys())}"
            )
            return targets

        # Priority 3: Single-label isolation in 3D (proxy active)
        proxy = getattr(self, "_isolation_proxy_layer", None)
        ndisplay = getattr(self.viewer.dims, "ndisplay", 2)
        if proxy is not None and ndisplay == 3:
            targets = [proxy]
            for vm in [self.viewer_model_xy, self.viewer_model_xz, self.viewer_model_yz]:
                orient = self._orientation_for_vm(vm)
                sub_layer = (
                    self._layer_by_name(vm.layers, f"{name}_{orient}")
                    or self._layer_by_name(vm.layers, name)
                )
                if sub_layer is not None:
                    targets.append(sub_layer)
            print(
                f"[CHKEY] FM8 routing: mode=SINGLE-ISO-3D name={name!r} "
                f"n_targets={len(targets)} "
                f"proxy_label={getattr(self, '_isolation_proxy_label_id', None)}"
            )
            return targets

        # Priority 4: Fallback to base — no isolation, or 2D mode.
        targets = super()._get_toggle_targets_for_name(name)
        print(
            f"[CHKEY] FM8 routing: mode=DEFAULT name={name!r} "
            f"n_targets={len(targets)} ndisplay={ndisplay}"
        )
        return targets

    def _prewarm_colormap(self) -> None:
        """Pre-warm napari's GPU resources for ``DirectLabelColormap``.

        Hook from the base class's ``_load_selected_layers`` after all
        label layers are populated + their original colormap applied.

        On the first ``DirectLabelColormap`` apply per layer per session,
        napari runs a ~250ms GPU init + LUT-texture upload.  Without
        pre-warming, the user pays this cost on their first ctrl+click
        multi-isolation or first merge-dialog checkbox toggle in a
        session — measured at ~760ms across the 3 subviewer label
        layers (the parked perf branch's ``[ISO-PERF]`` data, measured
        2026-05-03).

        Pre-warm pattern: on each label layer, save the current
        colormap, apply a tiny dummy ``DirectLabelColormap``, restore
        the original.  napari caches the GPU resources on the first
        apply so subsequent applies (the real ones from
        ``_apply_multi_isolation_colormap``) are fast.

        Wrapped defensively — any failure logs and continues; pre-warm
        is a perf nice-to-have, not load-critical.
        """
        try:
            from napari.utils.colormaps import DirectLabelColormap
        except ImportError:
            return
        import numpy as np
        from napari.layers import Labels

        hidden = np.array([0, 0, 0, 0], dtype=np.float32)
        dummy = DirectLabelColormap(color_dict={None: hidden, 0: hidden})

        targets = []

        # Main viewer label layer.
        labels_name = getattr(self, "_active_labels_name", None)
        if labels_name:
            main = self._layer_by_name(self.viewer.layers, labels_name)
            if isinstance(main, Labels):
                targets.append(main)

        # Subviewer label layers (XY/XZ/YZ).
        sm = getattr(self, "subviewer_manager", None)
        if sm is not None:
            for vm in (sm.viewer_model_xy, sm.viewer_model_xz, sm.viewer_model_yz):
                for layer in vm.layers:
                    if isinstance(layer, Labels):
                        targets.append(layer)

        # Apply + revert dummy on each target.  Suppress the property-
        # sync cascade so the dummy colormap doesn't propagate through
        # LabelSyncManager (which would also run the GPU init on
        # subviewers a second time, redundantly).
        from contextlib import nullcontext
        sync = getattr(self, "sync_manager", None)
        ctx = (sync.suppress_property_sync()
               if sync is not None and hasattr(sync, "suppress_property_sync")
               else nullcontext())

        with ctx:
            for layer in targets:
                try:
                    original = layer.colormap
                    layer.colormap = dummy
                    layer.colormap = original
                except Exception:
                    pass

        _safe_curation_debug(
            self, f"colormap pre-warm: {len(targets)} label layers primed"
        )

    def _refresh_label_action_buttons_state(self) -> None:
        """Refresh enable-state of the navigator's label-action
        buttons (Delete Label, + New Label) per the canonical
        destructive-op gate.  Call this at every site where the
        gating state could change.

        Delegates the predicate to the canonical
        ``RefinementOpState.can_start_destructive_label_op`` — the
        single source of truth for "is it safe to mutate the volume
        via a destructive label-action right now?".  Sites that
        change any gating flag (refinement op start/end, draw
        snapshot set/clear, merge dialog open/close) must call this
        AND ``_refresh_merge_button_state`` so both UI areas stay
        in sync.

        Pairs with ``_refresh_merge_button_state`` which gates Merge
        Into separately.  Two single-responsibility refreshers
        intentionally — sharing one function for two unrelated UI
        areas was a side-channel.
        """
        nav = self._navigator_widget
        if nav is None or not hasattr(nav, "set_label_action_buttons_enabled"):
            return
        self._sync_refine_state()
        try:
            nav.set_label_action_buttons_enabled(
                self._refine_state.can_start_destructive_label_op()
            )
        except Exception:
            pass

    def _refresh_merge_button_state(self) -> None:
        """Refresh the enable-state of the navigator's Merge Into
        button based on the current refinement-op state. The button
        is enabled only when no refinement op is active.

        Called at every refinement op transition (preview/accept/reject)
        so the button visually disables while interp/split/smooth is
        staged, instead of failing silently when clicked.  Tooltip
        updates too.

        Single responsibility: only the Merge Into button.  The
        label-action buttons (+ New Label / Delete Label) are refreshed
        by the sibling ``_refresh_label_action_buttons_state`` because
        their gate is WIDER (also covers draw snapshot + merge dialog
        open) — sharing one refresh function for two unrelated UI
        areas was a side-channel.
        """
        nav = getattr(self, "_navigator_widget", None)
        if nav is None or not hasattr(nav, "set_merge_enabled"):
            return
        try:
            nav.set_merge_enabled(not self._refinement_op_active())
        except Exception:
            pass

    def _label_action_buttons_active(self) -> bool:
        """Predicate: are the + New Label / Delete Label buttons safe
        to enable right now?

        Thin delegate to ``RefinementOpState.can_start_destructive_label_op``
        — the canonical predicate on the centralized refinement-state
        dataclass.  Wider than ``_refinement_op_active`` because the
        canonical predicate also covers an uncommitted paint stroke
        (``draw_snapshot_present``) and an open merge dialog
        (``merge_dialog_open``).

        Method retained (rather than callers reading the predicate
        directly off ``self._refine_state``) because the
        ``_on_delete_label_from_navigator`` handler uses this as its
        belt-and-braces in-handler gate and it's clearer to call a
        named method than to remember the sync step.
        """
        self._sync_refine_state()
        return self._refine_state.can_start_destructive_label_op()

    def _curation_debug_enabled(self) -> bool:
        """Return True when curation debug prints are enabled."""
        import os
        return os.environ.get("EPICYCLE_CURATION_DEBUG") == "1"

    def _curation_debug(self, msg: str) -> None:
        """Emit a lifecycle debug line for close/relaunch diagnostics."""
        if self._curation_debug_enabled():
            print(f"[CURATION-DEBUG] {msg}")

    def _invalidate_session_callbacks(self) -> None:
        """Invalidate deferred callbacks captured by old session token."""
        try:
            self._session_token = int(getattr(self, "_session_token", 0)) + 1
        except Exception:
            self._session_token = 1

    def _run_in_active_session(self, token: int, callback) -> None:
        """Run callback only when this widget is still in the active session."""
        if token != getattr(self, "_session_token", None):
            return
        if getattr(self, "_is_closing", False):
            return
        try:
            callback()
        except Exception:
            pass

    def _single_shot_in_session(self, delay_ms: int, callback) -> None:
        """Schedule callback guarded by current session token."""
        from qtpy.QtCore import QTimer

        token = int(getattr(self, "_session_token", 0))
        QTimer.singleShot(
            max(0, int(delay_ms)),
            lambda t=token, cb=callback: self._run_in_active_session(t, cb),
        )

    def _auto_load_if_ready(self) -> None:
        """Deferred auto-load: refresh combos and load layers into subviewers."""
        if getattr(self, "_is_closing", False):
            return
        try:
            if hasattr(self, "isVisible") and not self.isVisible():
                return
        except Exception:
            return
        self._refresh_layer_options()
        img = self.image_combo.currentText()
        lbl = self.labels_combo.currentText()
        if img and lbl:
            try:
                self._load_selected_layers()
                self._set_label_to_next_unused()
            except Exception as exc:
                self.status_label.setText(f"Auto-load failed: {exc}")
        elif not img:
            self.status_label.setText("No image layers found — add images to the viewer first.")
        elif not lbl:
            self.status_label.setText("No labels layer found — add a labels layer first.")

    def _schedule_deferred_auto_load(self, delay_ms: int) -> None:
        """Schedule deferred auto-load with cancellation support."""
        timer = getattr(self, "_deferred_auto_load_timer", None)
        if timer is None:
            from qtpy.QtCore import QTimer
            timer = QTimer()
            timer.setSingleShot(True)
            timer.timeout.connect(self._auto_load_if_ready)
            self._deferred_auto_load_timer = timer
        try:
            timer.stop()
        except Exception:
            pass
        timer.setInterval(max(0, int(delay_ms)))
        timer.start()

    def _set_label_to_next_unused(self) -> None:
        """Set selected label to max(existing) + 1 so new draws don't overwrite."""
        try:
            lbl_name = self.labels_combo.currentText().strip()
            if not lbl_name:
                lbl_name = self._active_labels_name
            layer = self._layer_by_name(self.viewer.layers, lbl_name)
            if layer is None:
                return
            data = np.asarray(layer.data)
            max_label = int(data.max()) if data.size > 0 else 0
            next_label = max(max_label + 1, 1)
            self._set_selected_label_all(next_label)
            self._interp_status_label.setText(f"Label pinned to {next_label}.")
        except Exception:
            pass

    # ---------------------------------------------------------------
    # UI override — replaces Quick Annotate with 3D Curation Tools
    # ---------------------------------------------------------------

    def _build_draw_group(self) -> QGroupBox:
        """Build slim 3D curation tool panel.

        Contains only editing tools (patch, interpolation, scroll step) and
        launcher buttons.  Navigation/triage lives in the floating Label
        Navigator, launched via the "Label Navigator..." button.
        """
        group = QGroupBox("3D Curation Tools")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(4, 10, 4, 4)
        layout.setSpacing(6)

        # Crosshair toggle — kept as attribute for wiring, not shown in panel.
        # All stub widgets use parent=group to prevent top-level orphaning.
        # Without an explicit parent, Qt creates them as top-level windows;
        # if anything later calls setVisible(True), they appear on the
        # Windows taskbar as "magicgui" (the QApplication name fallback).
        self.crosshairs_check = QCheckBox("Crosshair", parent=group)
        self.crosshairs_check.setChecked(True)
        self.crosshairs_check.setVisible(False)

        # --- Hidden data combos (parent internals need them) ---
        self.image_combo = QComboBox(parent=group)
        self.image_combo.setVisible(False)
        self.labels_combo = QComboBox(parent=group)
        self.labels_combo.setVisible(False)
        self.sync_to_subviewers_btn = QPushButton(parent=group)
        self.sync_to_subviewers_btn.setVisible(False)
        self.refresh_layers_btn = QPushButton(parent=group)
        self.refresh_layers_btn.setVisible(False)
        self.create_labels_btn = QPushButton(parent=group)
        self.create_labels_btn.setVisible(False)

        # --- Patch context row ---
        # Patch names can be long ("Timelapse_E12.5_tp_dataset_1-1_Blur_..."),
        # so the combo gets a maximum width to keep the layout tidy, plus
        # a tooltip that always shows the full current selection text.
        # The dropdown list still shows full names when opened.
        self.patch_context_combo = QComboBox()
        self.patch_context_combo.setMaximumWidth(180)
        self.patch_context_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.patch_context_combo.setMinimumContentsLength(12)
        self.patch_context_combo.setStyleSheet(INPUT_STYLE)

        def _refresh_patch_combo_tooltip(_idx=None):
            full = self.patch_context_combo.currentText() or ""
            self.patch_context_combo.setToolTip(full or "Select a patch")

        self.patch_context_combo.currentIndexChanged.connect(
            _refresh_patch_combo_tooltip
        )
        _refresh_patch_combo_tooltip()  # initialize

        self.switch_patch_btn = QPushButton("Switch")
        self.switch_patch_btn.setToolTip("Switch to selected patch context")
        self.switch_patch_btn.setStyleSheet(SECONDARY_BTN_STYLE)
        patch_row = QHBoxLayout()
        patch_row.setSpacing(4)
        patch_row.addWidget(QLabel("Patch"))
        patch_row.addWidget(self.patch_context_combo, 1)
        patch_row.addWidget(self.switch_patch_btn)
        # 2026-06-01: top-section `?` help button (mirrors 2D HITL pattern).
        # Covers panel-level basics: hotkeys, patch switching, scroll step,
        # tools section overview.  Per-section deep-dive lives on the
        # "Refinement" header below; Label Navigator + Advanced Features
        # docks have their own help in their own widgets (follow-up).
        patch_row.addWidget(_help_button(
            self,
            guide_path=str(
                _ANNOTATOR_GUIDES_DIR / "annotator_3d_main.md"
            ),
            title="3D Curation Tools — Annotator",
            tooltip="Help: 3D Curation Tools",
        ))
        layout.addLayout(patch_row)

        # --- Invisible stub attributes for HITL controls ---
        self.segment_mode_btn = QPushButton(parent=group)
        self.segment_mode_btn.setVisible(False)
        self.accept_slice_btn = QPushButton(parent=group)
        self.accept_slice_btn.setVisible(False)
        self.overlay_mode_combo = QComboBox(parent=group)
        self.overlay_mode_combo.addItem("Off")
        self.overlay_mode_combo.setVisible(False)
        self.overlay_opacity_sb = QSpinBox(parent=group)
        self.overlay_opacity_sb.setValue(55)
        self.overlay_opacity_sb.setVisible(False)

        # --- Invisible stubs for parent-expected navigation widgets ---
        # Parent's _wire_signals() connects these; keep as hidden stubs.
        # All use parent=group to prevent top-level orphaning.
        self._focused_toggle_btn = QPushButton("Focused Mode: OFF", parent=group)
        self._focused_toggle_btn.setCheckable(True)
        self._focused_toggle_btn.setVisible(False)
        self._prev_flagged_btn = QPushButton("\u25c0 Prev", parent=group)
        self._prev_flagged_btn.setVisible(False)
        self._next_flagged_btn = QPushButton("Next \u25b6", parent=group)
        self._next_flagged_btn.setVisible(False)
        self._focused_mode_group = QGroupBox(parent=group)
        self._focused_mode_group.setVisible(False)
        # Force zero size so even if set_focused_refinement() calls
        # setVisible(True), it never occupies space or occludes content.
        # The 3D curation annotator uses the Label Navigator for triage,
        # not the inline focused mode group.
        self._focused_mode_group.setMaximumSize(0, 0)
        self._focused_nav_label = QLabel("", parent=group)
        self._focused_nav_label.setVisible(False)
        # Hidden list widget stub (parent compat)
        self._flagged_labels_list = QListWidget(parent=group)
        self._flagged_labels_list.setVisible(False)
        # Hidden triage button stubs (parent compat for _wire_signals)
        self._mark_good_btn = QPushButton("Mark Good", parent=group)
        self._mark_good_btn.setVisible(False)
        self._delete_label_btn = QPushButton("Delete Label", parent=group)
        self._delete_label_btn.setVisible(False)
        self._discard_garbage_btn = QPushButton("Discard All Garbage", parent=group)
        self._discard_garbage_btn.setVisible(False)
        # Hidden tier filter stubs
        self._tier_filter_good = QCheckBox("Good", parent=group)
        self._tier_filter_good.setVisible(False)
        self._tier_filter_review = QCheckBox("Review", parent=group)
        self._tier_filter_review.setVisible(False)
        self._tier_filter_bad = QCheckBox("Bad", parent=group)
        self._tier_filter_bad.setVisible(False)
        self._tier_filter_garbage = QCheckBox("Garbage", parent=group)
        self._tier_filter_garbage.setVisible(False)
        # Hidden status label stub
        self._nav_status_label = QLabel("", parent=group)
        self._nav_status_label.setVisible(False)

        # --- Scroll step ---
        step_row = QHBoxLayout()
        step_row.setSpacing(4)
        step_row.addWidget(QLabel("Scroll Step"))
        self._slice_step_sb = QSpinBox()
        self._slice_step_sb.setRange(1, 100)
        self._slice_step_sb.setValue(1)
        self._slice_step_sb.setMaximumWidth(55)
        self._slice_step_sb.setStyleSheet(INPUT_STYLE)
        self._slice_step_sb.setToolTip(
            "Arrow key scroll step (slices per key press) for all subviewers"
        )
        step_row.addWidget(self._slice_step_sb)
        step_row.addStretch(1)
        layout.addLayout(step_row)

        # --- Section divider ---
        # Visual separation between the patch/context section above
        # and the refinement workflow below.  Matches the Label
        # Navigator's section-divider styling.
        _divider1 = QFrame()
        _divider1.setFrameShape(QFrame.HLine)
        _divider1.setStyleSheet(SEPARATOR_LINE_STYLE)
        _divider1.setMaximumHeight(8)
        layout.addWidget(_divider1)

        # --- Interpolation / Refinement section ---
        # 2026-06-01: header wrapped in QHBoxLayout so the `?` help
        # button sits inline-right with the section title.  Mirrors the
        # main panel `?` pattern.
        _interp_header_row = QHBoxLayout()
        _interp_header_row.setContentsMargins(0, 0, 0, 0)
        _interp_header_row.setSpacing(4)
        _interp_header = QLabel("Refinement")
        _interp_header.setStyleSheet(SECTION_HEADER_STYLE)
        _interp_header_row.addWidget(_interp_header)
        _interp_header_row.addStretch(1)
        _interp_header_row.addWidget(_help_button(
            self,
            guide_path=str(
                _ANNOTATOR_GUIDES_DIR / "annotator_3d_refinement.md"
            ),
            title="Refinement (SDF Interpolation) — 3D Annotator",
            tooltip="Help: Refinement",
        ))
        layout.addLayout(_interp_header_row)

        interp_desc = QLabel(
            "Draw on 2+ slices with the same label, then Preview to fill "
            "gaps via SDF interpolation. Accept to keep, Reject to undo, "
            "or Refine Further to fine-tune with the Refine tab."
        )
        interp_desc.setStyleSheet(DIM_LABEL_STYLE)
        interp_desc.setWordWrap(True)
        layout.addWidget(interp_desc)

        # Refinement workflow buttons — themed for visual hierarchy:
        # Preview = teal (primary investigatory action)
        # Accept  = green (commit, irreversible-ish)
        # Reject  = red destructive (undo back to snapshot)
        # Refine Further = cyan (distinct from Preview; "chain into refine")
        # See ui_shell/theme.py for the palette; mirrors the merge dialog
        # styling for consistency across the refinement workflow.
        self._interpolate_btn = QPushButton("Preview")
        self._interpolate_btn.setToolTip(
            "Fill intermediate slices using SDF contour interpolation"
        )
        self._interpolate_btn.setStyleSheet(PRIMARY_BTN_STYLE_TEAL)
        layout.addWidget(self._interpolate_btn)

        self._accept_interp_btn = QPushButton("Accept")
        self._accept_interp_btn.setToolTip(
            "Keep the interpolated result and advance to the next label"
        )
        self._accept_interp_btn.setStyleSheet(PRIMARY_BTN_STYLE_GREEN)
        self._reject_interp_btn = QPushButton("Reject")
        self._reject_interp_btn.setToolTip(
            "Undo interpolation and restore the original volume"
        )
        self._reject_interp_btn.setStyleSheet(DESTRUCTIVE_BTN_STYLE)
        self._accept_interp_btn.setEnabled(False)
        self._reject_interp_btn.setEnabled(False)
        ar_row = QHBoxLayout()
        ar_row.setSpacing(4)
        ar_row.addWidget(self._accept_interp_btn)
        ar_row.addWidget(self._reject_interp_btn)
        layout.addLayout(ar_row)

        self._interp_refine_btn = QPushButton("Refine Further...")
        self._interp_refine_btn.setToolTip(
            "Accept interpolation and refine this label immediately"
        )
        self._interp_refine_btn.setStyleSheet(PRIMARY_BTN_STYLE_CYAN)
        self._interp_refine_btn.setEnabled(False)
        layout.addWidget(self._interp_refine_btn)

        self._interp_status_label = QLabel("")
        self._interp_status_label.setStyleSheet(STATUS_LABEL_STYLE)
        self._interp_status_label.setWordWrap(True)
        layout.addWidget(self._interp_status_label)

        self._clear_edits_btn = QPushButton("Clear Edits")
        self._clear_edits_btn.setToolTip(
            "Undo all drawn corrections and restore the label to its "
            "original shape before you started drawing"
        )
        self._clear_edits_btn.setStyleSheet(SECONDARY_BTN_STYLE)
        self._clear_edits_btn.setEnabled(False)
        layout.addWidget(self._clear_edits_btn)

        # --- Section divider before Tools row ---
        _divider2 = QFrame()
        _divider2.setFrameShape(QFrame.HLine)
        _divider2.setStyleSheet(SEPARATOR_LINE_STYLE)
        _divider2.setMaximumHeight(8)
        layout.addWidget(_divider2)

        # --- Tools section header ---
        _tools_header = QLabel("Tools")
        _tools_header.setStyleSheet(SECTION_HEADER_STYLE)
        layout.addWidget(_tools_header)

        # --- Label Navigator launcher ---
        # Amber primary styling for the dock-launcher buttons —
        # they're the "main entry points" into the broader curation
        # workflow and should pop visually.  Matches the Step 3
        # accordion's primary action accent (Build/Step 3 amber).
        self._open_navigator_btn = QPushButton("Label Navigator...")
        self._open_navigator_btn.setToolTip(
            "Open floating Label Navigator for triage and navigation"
        )
        self._open_navigator_btn.setStyleSheet(PRIMARY_BTN_STYLE)
        layout.addWidget(self._open_navigator_btn)

        # --- Advanced Features launcher ---
        self.open_advanced_features_btn = QPushButton("Advanced Features...")
        self.open_advanced_features_btn.setToolTip(
            "Open advanced segmentation workflows"
        )
        self.open_advanced_features_btn.setStyleSheet(PRIMARY_BTN_STYLE)
        layout.addWidget(self.open_advanced_features_btn)

        # --- Multiviewer Settings launcher ---
        self._open_multiviewer_settings_btn = QPushButton("Multiviewer Settings...")
        self._open_multiviewer_settings_btn.setToolTip(
            "Open multiviewer display settings (crosshairs, sync)"
        )
        self._open_multiviewer_settings_btn.setStyleSheet(SECONDARY_BTN_STYLE)
        layout.addWidget(self._open_multiviewer_settings_btn)
        # release hide list 2026-10-13 (dock controls are unwired)
        self._open_multiviewer_settings_btn.setVisible(False)

        layout.addStretch(1)

        return group

    def _build_utility_panel(self):
        """Override parent to prevent draw group from stretching vertically.

        The parent uses stretch=1 on the draw group which forces it to fill
        all available space, squishing the viewers. We use stretch=0 and a
        scroll area so the panel stays compact and scrolls if needed.
        """
        from qtpy.QtWidgets import QFrame, QScrollArea

        panel = QFrame()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        draw_group = self._build_draw_group()
        self.data_group = self._build_data_group()
        self.data_group.setVisible(False)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(draw_group)
        outer.addWidget(scroll, 1)
        # Parent data_group to panel so it's not a top-level orphan.
        outer.addWidget(self.data_group)
        return panel

    def _build_data_group(self) -> QGroupBox:
        """No-op — data combos are already in _build_draw_group."""
        group = QGroupBox()
        group.setVisible(False)
        return group

    # ---------------------------------------------------------------
    # Multi-channel image resolution override
    # ---------------------------------------------------------------

    def _resolve_source_image_layers(self, image_layer, labels_name: str):
        """Return ALL Image layers in the viewer for 3D curation.

        The parent implementation uses HITL naming conventions
        ({patch_prefix}_..._XY_view) which don't match user-loaded data.
        In 3D curation, load every Image layer so all channels appear
        in the subviewers.
        """
        # First try the parent's patch-based resolution.
        result = super()._resolve_source_image_layers(image_layer, labels_name)
        if len(result) > 1:
            return result  # patch naming matched multiple channels — use them

        # Fallback: collect all Image layers (2D/3D/4D-RGB).
        all_images = []
        for layer in self.viewer.layers:
            if not isinstance(layer, Image):
                continue
            # ...but never an UNMANAGED one (Brick U3 fix pass, H1).  The
            # parent's own loop skips `_epicycle_temporary`, and this one
            # is reached exactly when the parent found ONE layer or none
            # -- i.e. on a single-channel patch, which is the ordinary
            # case.  Without this, `guide_walls` becomes a real channel
            # there, is mirrored into the three side views, and shifts
            # a/s/d/f on the next `_load_selected_layers` (the next
            # time the layers are loaded).
            if getattr(layer, "_epicycle_temporary", False):
                continue  # hide transient images (guide_walls, etc.)
            ndim = layer.data.ndim
            if ndim in {2, 3}:
                all_images.append(layer)
            elif ndim == 4 and int(layer.data.shape[-1]) in {3, 4}:
                all_images.append(layer)
        return all_images if all_images else [image_layer]

    # ---------------------------------------------------------------
    # set_context override — deferred auto-load for 3D curation
    # ---------------------------------------------------------------

    def set_context(
        self,
        *,
        image_layer_name: str = "",
        labels_layer_name: str = "",
        auto_load: bool = True,
        target_view: str = "",
        selected_view: str = "",
    ) -> None:
        """Override to defer auto-load until dock is attached."""
        # Set combo selections without triggering immediate load.
        if image_layer_name:
            self._refresh_layer_options()
            if self.image_combo.findText(image_layer_name) >= 0:
                self.image_combo.setCurrentText(image_layer_name)
        if labels_layer_name:
            self._refresh_layer_options()
            if self.labels_combo.findText(labels_layer_name) >= 0:
                self.labels_combo.setCurrentText(labels_layer_name)
        if auto_load:
            # Cancel the __init__ auto-load timer to prevent double load.
            init_timer = getattr(self, "_init_auto_load_timer", None)
            if init_timer is not None:
                init_timer.stop()
            self._schedule_deferred_auto_load(150)

    # ---------------------------------------------------------------
    # Advanced Features override — 3D curation workflows
    # ---------------------------------------------------------------

    def _open_advanced_features(self) -> None:
        """Open Advanced Features with split callbacks wired."""
        try:
            from napari_epicycle.annotator.launcher import launch_advanced_features_dock

            widget, _ok, _msg = launch_advanced_features_dock(
                self.viewer,
                mode="3d_curation",
                split_callbacks={
                    "on_start": self._start_split,
                    "on_next_seed": self._next_seed,
                    "on_undo_seed": self._undo_last_seed,
                    "on_refine_seeds": self._refine_seeds,
                    "on_select_seed": self._on_select_seed,
                    "on_preview": self._preview_split,
                    "on_accept": self._accept_split,
                    "on_reject": self._reject_split,
                    "on_edt_toggle": self._toggle_edt_overlay,
                    "on_refine_further": self._enter_fragment_refine_mode,
                    "on_select_fragment": self._on_fragment_selected,
                    "on_detect_seeds": self._detect_seeds,
                    "on_threshold_changed": self._on_seed_threshold_changed,
                    # --- Reshape R1: guided Split ---
                    "on_seeds_from_changed": self._on_reshape_source_changed,
                    "on_walls_from_changed": self._on_reshape_source_changed,
                    "on_guide_browse": self._on_reshape_guide_browse,
                    "on_guide_load": self._on_reshape_guide_load,
                    "on_guide_show_toggled": self._on_reshape_guide_show,
                    "on_piece_check_changed": self._on_reshape_pieces_changed,
                    "on_cancel_preview": self._on_reshape_cancel,
                    "on_split_anyway": self._on_reshape_split_anyway,
                    # --- Brick 2a: the marker picker ---
                    "on_find_markers": self._find_reshape_markers,
                    "on_markers_invalidated":
                        self._on_reshape_markers_invalidated,
                    # --- Brick U1: the loop ---
                    "on_cut_knob_changed":
                        self._on_reshape_cut_knob_changed,
                    "on_marker_check_changed":
                        self._on_reshape_marker_check_changed,
                    # --- Brick U2: the outline tick ---
                    "on_contour_toggled": self._on_refine_contour_toggled,
                    # --- Brick U3: the wall picture ---
                    "on_show_walls_toggled":
                        self._on_reshape_show_walls_toggled,
                },
            )
            if widget is not None and hasattr(widget, "set_smooth_callbacks"):
                widget.set_smooth_callbacks(
                    # three thin slots, so a click that
                    # lands while a recipe runs is answered, not obeyed.
                    on_preview=self._on_refine_preview_clicked,
                    on_accept=self._on_refine_accept_clicked,
                    on_reject=self._on_refine_discard_clicked,
                    on_prob_toggle=self._toggle_prob_overlay,
                    # --- Brick B1, additive ---
                    on_cancel_preview=self._cancel_refine_preview,
                    on_show_walls_toggled=self._on_refine_show_walls_toggled,
                    on_method_changed=self._on_refine_method_combo_changed,
                    # --- Refine recipes, additive ---
                    on_recipes_menu=self._on_recipes_menu_about_to_show,
                    on_recipe_run=self._on_recipe_run,
                    on_recipe_load=self._on_recipe_load,
                    on_recipe_save=self._on_recipe_save,
                    on_recipe_delete=self._on_recipe_delete,
                )
                try:
                    self._refresh_refine_inputs()
                except Exception as exc:  # noqa: BLE001
                    print("[REFINE] input refresh at launch failed: "
                          "{!r}".format(exc))
            # Wire the de-novo Add tab (2026-06-30).  Mirrors the smooth block.
            if widget is not None and hasattr(widget, "set_add_callbacks"):
                widget.set_add_callbacks(
                    on_start=self._start_add,
                    on_box_changed=self._on_add_box_changed,
                    on_recenter=self._recenter_add_box_on_crosshair,
                    on_size_preset=self._set_add_box_size_fraction,
                    on_clahe_toggled=self._on_add_clahe_toggled,
                    on_threshold_changed=self._on_add_threshold_changed,
                    on_otsu_reset=self._on_add_otsu_reset,
                    on_preview=self._preview_add,
                    on_accept=self._accept_add,
                    # Add rebuild 4.7: Done / Discard / Cancel
                    on_reject=self._add_secondary,
                    on_advanced_changed=self._on_add_advanced_changed,
                    on_select_piece=self._on_add_piece_selected,
                    on_refine_piece=self._refine_add_piece,
                    on_batch_refine=self._batch_refine_add_pieces,
                    on_candidate_changed=self._sync_add_candidate_overlay,
                    # Merge is deferred to Phase 1.1 (left unwired; the button is
                    # hidden in the widget) — see _merge_add_pieces.
                    on_add_another=self._add_another,
                    on_undo_all=self._undo_add_all,
                    # ---  additive ---
                    on_preset_changed=self._on_add_preset_changed,
                    on_marker_source_changed=self._on_add_source_changed,
                    on_stop=self._stop_add_preview,
                    on_numbers=self._on_add_numbers,
                    on_nuclei_auto=self._on_add_nuclei_auto,
                    on_nuclei_threshold=self._on_add_nuclei_threshold_typed,
                    on_channels_changed=self._on_add_channels_changed,
                    on_dark_outside_toggled=self._on_add_dark_outside_toggled,
                    on_place_markers_toggled=self._on_add_place_markers_toggled,
                    on_click_undo=self._add_click_undo,
                    on_click_clear=self._add_click_clear,
                    on_click_tick=self._on_add_click_tick,
                    on_show_walls=self._on_add_show_walls,
                    on_contour_toggled=self._on_add_contour_toggled,
                    on_accept_refine=self._accept_add_and_refine,
                )
            # Populate image combos immediately so user can
            # select channels before hitting Start.
            self._refresh_refine_inputs()
            self._refresh_add_image_combo()
            # ...  and again whenever the user switches tool, because each
            # tool reads its own combo.  DISCONNECT-BEFORE-CONNECT: this
            # method is user-clickable and the launcher REUSES an open dock,
            # so this block re-runs on every click (policy 1).
            if widget is not None and hasattr(widget, "_tool_tabs"):
                try:
                    widget._tool_tabs.currentChanged.disconnect(
                        self._on_tool_tab_changed
                    )
                except (TypeError, RuntimeError):
                    pass
                widget._tool_tabs.currentChanged.connect(
                    self._on_tool_tab_changed
                )
        except Exception:
            pass

    def _on_tool_tab_changed(self, _index=None) -> None:
        """The user switched tool -- re-read the channel lists for it.  A
        tab change also unchecks Add's Place markers; the Add
        session itself stays."""
        self._refresh_all_image_combos()
        if getattr(self, "_add_place_markers_on", False):
            try:
                self._add_place_markers_off()
            except Exception:
                pass
        try:                 # the move cursor never outlives the Add tab
            self._add_clear_side_cursors()
        except Exception:
            pass

    # ---------------------------------------------------------------
    # Signal wiring override
    # ---------------------------------------------------------------

    def _wire_signals(self) -> None:
        """Wire signals for 3D curation context."""
        super()._wire_signals()
        # Interpolation
        self._interpolate_btn.clicked.connect(self._run_interpolation)
        self._accept_interp_btn.clicked.connect(self._accept_interpolation)
        self._reject_interp_btn.clicked.connect(self._reject_interpolation)
        self._interp_refine_btn.clicked.connect(self._enter_interp_refine_mode)
        self._clear_edits_btn.clicked.connect(self._clear_draw_edits)
        self._slice_step_sb.valueChanged.connect(self._on_step_size_changed)
        # Label Navigator launcher
        self._open_navigator_btn.clicked.connect(self._open_label_navigator)
        # Multiviewer Settings launcher
        self._open_multiviewer_settings_btn.clicked.connect(
            self._open_multiviewer_settings
        )
        # Auto-load when new layers are added and subviewers are still empty.
        self.viewer.layers.events.inserted.connect(self._on_layer_inserted)
        # The four channel combos are filled with LAYER NAMES, and nothing
        # used to tell them a layer had come or gone (policy 1).
        # Paired with the disconnect in _close_cleanup, beside the
        # inserted one -- an unpaired connect accumulates a handler per
        # annotator relaunch.
        self.viewer.layers.events.removed.connect(self._on_layer_removed)
        # Pre-sync: snapshot shared_data before the stroke is synced
        # (enables Clear Edits to undo raw draw corrections).
        self.drawing_handler.pre_sync_callback = self._on_subviewer_pre_sync
        # Post-sync: refresh isolation proxy so user sees drawn corrections.
        self.drawing_handler.post_sync_callback = self._on_subviewer_draw_complete
        # New-label hook: fires when enclosed-fill + auto-increment
        # detects the user just painted a brand-new cell.  Forwards to
        # the controller (via _on_new_label_painted_cb) so a navigator
        # row gets UPSERTed with qc_tier="" (untriaged).
        self.drawing_handler.new_label_callback = (
            self._on_subviewer_new_label_painted
        )

    # ---------------------------------------------------------------
    # Auto-load on layer insertion
    # ---------------------------------------------------------------

    def _refresh_wall_channels(self) -> None:
        """Re-fill ONLY the membrane-channel combo.

        `set_wall_channels` is otherwise reached through
        `_refresh_reshape_inputs`, which also scans other runs ON DISK for
        guide masks -- far too much for a layer insert.  This is the cheap
        half, and it is fed the SAME unfiltered Image-layer list that
        `_refresh_reshape_inputs` uses: the ndim filter the split/refine/add
        combos apply would turn a working membrane split on a 4-D stack into
        a hard error.
        """
        adv = self._get_advanced_features_widget()
        if adv is None or not hasattr(adv, "set_wall_channels"):
            return
        names = []
        for layer in self.viewer.layers:
            if getattr(layer, "_epicycle_temporary", False):
                continue
            if isinstance(layer, Image):
                names.append(layer.name)
        adv.set_wall_channels(names)

    def _refresh_seed_channels(self) -> None:
        """Re-fill ONLY the MARKER-channel combo -- the fifth one.

        Fed the same UNFILTERED Image-layer list as
        `_refresh_wall_channels`: the ndim filter the split/refine/add
        combos apply would turn a working marker search on a 4-D stack
        into a hard error.
        """
        adv = self._get_advanced_features_widget()
        if adv is None or not hasattr(adv, "set_seed_channels"):
            return
        names = []
        for layer in self.viewer.layers:
            if getattr(layer, "_epicycle_temporary", False):
                continue
            if isinstance(layer, Image):
                names.append(layer.name)
        adv.set_seed_channels(names)

    def _refresh_all_image_combos(self) -> None:
        """Re-read the viewer's image layers into all four channel combos.

        Cheap (four list rebuilds) and guarded on visibility, so it adds no
        main-thread work while the dock is closed.

        Every call is individually guarded because this runs on
        `layers.events.removed`: `_close_cleanup` destroys the Advanced dock
        at step 4b but only disconnects that handler at step 6, so a layer
        removed in between reaches a half-destroyed widget and the
        RuntimeError would land in napari's emitter during window close.
        """
        try:
            if not self.isVisible():
                return
        except Exception:
            return
        for _fn in (
            self._refresh_refine_inputs,
            self._refresh_add_image_combo,
            self._refresh_wall_channels,
            self._refresh_seed_channels,
            self._refresh_batch_chain_channels,
        ):
            try:
                _fn()
            except Exception as exc:  # noqa: BLE001 - see docstring
                print("[CHANNELS] refresh skipped ({}): {!r}".format(
                    getattr(_fn, "__name__", "?"), exc))

    def _on_layer_removed(self, _event=None) -> None:
        """A layer went away -- the channel pickers must stop offering it."""
        self._refresh_all_image_combos()

    def _on_layer_inserted(self, _event=None) -> None:
        """When layers are added to the viewer, auto-load if subviewers are empty."""
        # Skip if the dock is hidden/closed — avoids expensive subviewer
        # loading when the user switches patches without the annotator open.
        if not self.isVisible():
            return
        # A new image layer must reach the channel pickers even on the
        # (usual) path where the subviewers are already populated, so this
        # runs BEFORE the early return below.
        self._refresh_all_image_combos()
        # Only auto-load if subviewers haven't been populated yet.
        if self._active_labels_name:
            return
        # Defer slightly so the layer is fully registered in napari.
        self._schedule_deferred_auto_load(200)

    # ---------------------------------------------------------------
    # Triage actions (Mark <tier> for 5 tiers; Merge into…)
    # ---------------------------------------------------------------

    def set_triage_callbacks(
        self, on_mark_tier=None, on_label_deleted=None, on_merge=None
    ):
        """Store callbacks for triage actions.

        Parameters
        ----------
        on_mark_tier : callable(int, str) or None
            Called with ``(label_id, tier_str)`` when user clicks one
            of the five tier buttons in the Label Navigator (Mark
            Curated / Good / Review / Bad / Garbage).  ``tier_str`` is
            one of ``"curated"``, ``"good"``, ``"review"``, ``"bad"``,
            ``"garbage"``.  Flag-only — receiver should update
            ``qc_tier`` in the QC features df and push refreshed rows
            back via ``set_flagged_labels``.  No volume mutation.
        on_label_deleted : callable(int, *, roi_slices, old_roi_data) or None
            Called when a label is destructively removed from the
            volume (e.g., user paints 0 in a subviewer, reconciled by
            ``_handle_manually_deleted_label``).  Receiver should mark
            the row as ``label_status="deleted"`` (ghost) so it
            persists as a QC history trail.  NOT wired to a navigator
            button anymore (post-2026-05-10 redesign removed the
            "Delete Label" button in favor of flag-only "Mark Garbage"
            + bulk ``Discard All Garbage``).
        on_merge : callable(int, int) or None
            Called with (keep_id, absorb_id) when user merges labels.
        """
        self._on_mark_tier_cb = on_mark_tier
        self._on_delete_label_cb = on_label_deleted
        self._on_merge_cb = on_merge

    def set_discard_garbage_callback(self, callback):
        """Store callback for Discard All Garbage action.

        Parameters
        ----------
        callback : callable(list[int]) or None
            Called with list of garbage label IDs after zeroing.
        """
        self._on_discard_garbage_cb = callback

    def set_curation_callbacks(
        self,
        *,
        on_interpolation_accepted=None,
        on_label_split=None,
        on_label_smoothed=None,
        on_recalculate=None,
        on_new_label_painted=None,
        on_batch_refined=None,
        on_batch_refine_lock=None,
        on_add_mode_lock=None,
        on_add_committed=None,
        on_add_undo_session=None,
        on_save_batch_recipe=None,
        get_saved_batch_recipes=None,
        # --- Reshape R1 ---
        on_reshape_lock=None,
        on_refine_lock=None,
        on_reshape_committed=None,
        on_reshape_context=None,
        # --- Refine recipes ---
        on_save_refine_recipe=None,
        get_saved_refine_recipes=None,
        on_delete_refine_recipe=None,
        # --- Batch "Plot selection (N)" ---
        get_plot_selection=None,
        # ---  the df id floor (ghosts included) ---
        get_label_id_floor=None,
    ):
        """Store callbacks for curation operations that set tier to 'curated'.

        Parameters
        ----------
        on_interpolation_accepted : callable(int) or None
            Called with label_id after an interpolation is accepted.
        on_label_split : callable(int, list[int]) or None
            Future — called with (original_id, [new_ids]) after a split.
        on_label_smoothed : callable(int) or None
            Future — called with label_id after surface smoothing.
        on_recalculate : callable(list[int]) or None
            Called with list of curated label IDs when user clicks
            Recalculate QC in the Navigator.
        on_new_label_painted : callable(int) or None
            Called with ``label_id`` when the user paints a new cell
            via enclosed-fill + auto-increment in a subviewer
            (2026-05-16).  Receiver UPSERTs a navigator row for the
            label with ``qc_tier=""`` (untriaged).  ``label_id`` is
            the PRE-BUMP ``selected_label`` (the label that was just
            painted, not the new max+1).
        on_batch_refine_lock : callable(bool) or None
            Called with ``True`` when a Batch Refine STARTS and ``False`` when
            it FULLY RESOLVES (accept / reject / worker error / cancel / dialog
            close).  Lets the controller OR a ``_batch_refine_running`` flag into
            its unified Step-3 lock so the WHOLE surface (predict / reconstruct /
            extract / commit / save / step-nav / patch-switch + the navigator
            action buttons via ``set_load_locked``) locks for the duration of
            the worker compute AND the staged-preview window.  A standalone
            annotator (no controller) leaves this None → no-op.
        on_save_batch_recipe : callable(str, list[dict]) or None
            Called with ``(name, recipe_json)`` to persist a named
            Batch-Refine recipe to the workspace store.  ``recipe_json`` is
            the round-trippable list-of-dicts (steps + params; no target).
            A standalone annotator (no controller) leaves this None → the
            dialog's Save button is disabled.
        get_saved_batch_recipes : callable() -> dict[str, list[dict]] or None
            Called to read the persisted named recipes for the "Saved
            recipes" dropdown.  None → no Saved dropdown population.
        """
        # EVERY assignment below is guarded (2026-09-19).  A keyword that
        # was not passed means "leave it alone", never "set it to None":
        # `_post_switch_annotator_sync` re-wires this widget after every
        # patch switch with only the three Reshape kwargs, and these seven
        # used to be assigned unconditionally from their None defaults --
        # so one patch switch silently unwired Smooth, Split, Interpolate,
        # Recalculate, new-label and Batch Refine.  The volume still
        # changed on Accept, but `_commit_curation_action` never ran, so
        # there was no undo entry, no dirty flag and no Save prompt, and
        # the next patch load threw the edit away without a word.
        if on_interpolation_accepted is not None:
            self._on_interpolation_accepted_cb = on_interpolation_accepted
        if on_label_split is not None:
            self._on_label_split_cb = on_label_split
        if on_label_smoothed is not None:
            self._on_label_smoothed_cb = on_label_smoothed
        if on_recalculate is not None:
            self._on_recalculate_cb = on_recalculate
        if on_new_label_painted is not None:
            self._on_new_label_painted_cb = on_new_label_painted
        if on_batch_refined is not None:
            self._on_batch_refined_cb = on_batch_refined
        if on_batch_refine_lock is not None:
            self._batch_refine_lock_cb = on_batch_refine_lock
        # De-novo Add mode (2026-06-30): lock callback (mirrors batch-refine)
        # + commit callback (fires after the background-only safe-write so the
        # controller registers the new labels in the QC DataFrame + undo).
        if on_add_mode_lock is not None:
            self._add_mode_lock_cb = on_add_mode_lock
        if on_add_committed is not None:
            self._on_add_committed_cb = on_add_committed
        if on_add_undo_session is not None:
            self._on_add_undo_session_cb = on_add_undo_session
        if on_save_batch_recipe is not None:
            self._save_batch_recipe_cb = on_save_batch_recipe
        if get_saved_batch_recipes is not None:
            self._get_saved_batch_recipes_cb = get_saved_batch_recipes
        # Reshape R1: the lock mirrors on_add_mode_lock; the
        # commit callback carries the df rows + the arbiter flags; the
        # context provider is how this widget learns the workspace, the
        # patch, the run and the voxel spacing, none of which it owns.
        if on_reshape_lock is not None:
            self._reshape_lock_cb = on_reshape_lock
        # Brick B1: the Refine twin.  A staged Refine
        # preview is ALREADY written into `shared_data`, so Save Curation,
        # Commit Triage and a run switch must be unavailable for the whole
        # staged window -- not only while a worker runs.
        if on_refine_lock is not None:
            self._refine_lock_cb = on_refine_lock
        if on_reshape_committed is not None:
            self._on_reshape_committed_cb = on_reshape_committed
        if on_reshape_context is not None:
            self._reshape_context_cb = on_reshape_context
        # Refine recipes: the Batch pair's shape.  Save and
        # delete return None or a plain reason; the read returns
        # ``(store, problem)``.  Guarded like every assignment above.
        if on_save_refine_recipe is not None:
            self._save_refine_recipe_cb = on_save_refine_recipe
        if get_saved_refine_recipes is not None:
            self._get_saved_refine_recipes_cb = get_saved_refine_recipes
        if on_delete_refine_recipe is not None:
            self._delete_refine_recipe_cb = on_delete_refine_recipe
        # Batch Refine's "Plot selection" target: the controller's live
        # getter of the ids the QC plot highlights right now (``None`` when
        # there is no plot, ``[]`` when another patch's plot or nothing).
        if get_plot_selection is not None:
            self._get_plot_selection_cb = get_plot_selection
        # Add's id floor: new ids start above every df id,
        # ghosts included, so an added object never hides behind one.
        if get_label_id_floor is not None:
            self._get_label_id_floor_cb = get_label_id_floor

    def _on_subviewer_new_label_painted(self, label_id: int) -> None:
        """Forwarder from DrawingHandler.new_label_callback to the
        controller callback wired via ``set_curation_callbacks``.

        Defensive: swallows callback errors so a buggy controller
        callback can't propagate into napari's mouse-drag dispatch.
        """
        cb = getattr(self, "_on_new_label_painted_cb", None)
        if cb is None:
            return
        try:
            cb(int(label_id))
        except Exception:
            pass

    def set_navigator_selection_callback(self, callback):
        """Store callback for navigator label selection events.

        Parameters
        ----------
        callback : callable(int) or None
            Called with label_id when user selects a row in the navigator.
            Used by the controller to dispatch selection through the
            coordinator (R4-WP4).
        """
        self._on_navigator_label_selected_cb = callback

    def set_colormap_authority(self, authority):
        """Inject the shared ColormapAuthority from the controller (WP-3)."""
        self._colormap_authority = authority

    # ---------------------------------------------------------------
    # Navigation label list
    # ---------------------------------------------------------------

    def set_flagged_labels(self, items):
        """Cache flagged items and forward to navigator if open.

        Parameters
        ----------
        items : list of dict
            Each dict has keys: label_id, tier, centroid, area.
        """
        self._cached_flagged_items = list(items)
        if self._navigator_widget is not None:
            try:
                # Keep the navigator's swatch-source layer current so
                # the Color column samples the right colormap.
                if hasattr(self._navigator_widget, "set_label_layer"):
                    layer = self._layer_by_name(
                        self.viewer.layers, self._active_labels_name
                    )
                    self._navigator_widget.set_label_layer(layer)
                self._navigator_widget.set_items(items)
            except Exception:
                pass

    # ---------------------------------------------------------------
    # Label Navigator — creation and wiring
    # ---------------------------------------------------------------

    def _open_multiviewer_settings(self) -> None:
        """Open (or raise) the floating Multiviewer Settings dock."""
        try:
            from napari_epicycle.annotator.launcher import (
                launch_multiviewer_settings_dialog,
            )

            launch_multiviewer_settings_dialog(self.viewer)
        except Exception as exc:
            self._interp_status_label.setText(f"Settings error: {exc}")

    def _open_label_navigator(self) -> None:
        """Open (or raise) the floating Label Navigator dock."""
        try:
            from napari_epicycle.annotator.launcher import (
                launch_curation_label_navigator,
            )

            _safe_curation_debug(self, "navigator open requested")
            widget, ok, msg = launch_curation_label_navigator(
                self.viewer, self
            )
            if ok and widget is not None:
                self._navigator_widget = widget
                _safe_curation_debug(self, 
                    "navigator open/reuse ok "
                    f"type={type(widget).__name__} "
                    f"id={id(widget)}"
                )
            else:
                _safe_curation_debug(self, 
                    f"navigator open/reuse failed ok={ok} msg={msg}"
                )
        except Exception as exc:
            self._interp_status_label.setText(f"Navigator error: {exc}")
            _safe_curation_debug(self, f"navigator open exception: {exc!r}")

    def _wire_navigator_callbacks(self, nav):
        """Wire (or re-wire) navigator callbacks to this annotator.

        Called by ``_create_navigator`` on first launch and by the
        launcher's dock-reuse path on relaunch to rebind stale callbacks.
        """
        _safe_curation_debug(self, 
            "wire navigator callbacks "
            f"nav_type={type(nav).__name__} id={id(nav)}"
        )
        nav.set_callbacks(
            on_navigate=self._navigator_navigate,
            on_select_label=self._on_navigator_select_label,
            on_mark_tier=self._on_mark_tier_from_navigator,
            on_discard_garbage=self._on_discard_garbage_from_navigator,
            on_merge=self._on_merge_from_navigator,
            on_open_merge_dialog=self._open_merge_dialog,
            on_focused_mode_changed=self._toggle_focused_mode,
            on_isolation_changed=self._on_isolation_changed,
            on_scroll_subviewer=self._on_scroll_subviewer,
            on_recalculate=getattr(self, "_on_recalculate_cb", None),
            on_new_label=self._on_new_label_from_navigator,
            on_delete_label=self._on_delete_label_from_navigator,
            on_paint_isolation_toggled=self._on_paint_isolation_toggled,
            on_paint_label_changed=self._on_paint_label_changed,
            on_batch_refine=self._open_batch_refine_from_navigator,
            on_selection_changed=self._on_navigator_selection_changed,
            is_op_active=self._is_op_active_for_navigator,
        )
        _safe_curation_debug(self, 
            "callbacks set: "
            f"navigate={bool(getattr(nav, '_on_navigate_cb', None))} "
            f"select={bool(getattr(nav, '_on_select_label_cb', None))} "
            f"isolate={bool(getattr(nav, '_on_isolation_changed_cb', None))}"
        )

    def _create_navigator(self):
        """Create a new CurationLabelNavigator and wire callbacks.

        Called by the launcher to build the widget with proper callback
        bindings.
        """
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )

        nav = CurationLabelNavigator()
        self._wire_navigator_callbacks(nav)
        return nav

    def _on_scroll_subviewer(self, direction: str) -> None:
        """Scroll the subviewer that the mouse is hovering over.

        Called by the Label Navigator when Left/Right is pressed on its table.
        Detects which subviewer the cursor is over and scrolls that subviewer's
        dims by the current step size.
        """
        orient = self._detect_hovered_subviewer()
        if orient is None:
            return

        vm_map = {
            "XY": self.viewer_model_xy,
            "XZ": self.viewer_model_xz,
            "YZ": self.viewer_model_yz,
        }
        vm = vm_map.get(orient)
        if vm is None:
            return

        step = max(1, int(self._slice_step_sb.value()))
        try:
            current = int(vm.dims.current_step[0])
            max_slice = int(vm.dims.range[0][1]) - 1
            if max_slice <= 0:
                return

            if direction == "right":
                new = min(current + step, max_slice)
            else:
                new = max(current - step, 0)

            if new != current:
                vm.dims.set_current_step(0, new)
        except Exception:
            pass

    def _on_subviewer_navigate(self, direction: str) -> None:
        """Navigate to prev/next label when Up/Down pressed on subviewer.

        Called by _SliceStepFilter and _UpDownBlocker event filters.
        Routes to the Label Navigator's prev/next methods.
        """
        nav = getattr(self, "_navigator_widget", None)
        if nav is None:
            return
        try:
            if direction == "prev":
                nav._navigate_prev()
            elif direction == "next":
                nav._navigate_next()
        except Exception:
            pass

    def _detect_hovered_subviewer(self) -> str | None:
        """Return the orientation of the subviewer under the mouse, or None."""
        try:
            from qtpy.QtGui import QCursor
            pos = QCursor.pos()
            for orient, qt_viewer in [
                ("XY", self.qt_viewer_xy),
                ("XZ", self.qt_viewer_xz),
                ("YZ", self.qt_viewer_yz),
            ]:
                local_pos = qt_viewer.mapFromGlobal(pos)
                if qt_viewer.rect().contains(local_pos):
                    return orient
        except Exception:
            pass
        return None

    def _check_label_exists_in_volume(self, label_id: int) -> bool:
        """Check if a label still has any voxels in the volume.

        Uses the centroid from the Navigator items to check a local ROI
        first (fast path), falling back to a full volume scan if needed.
        Returns False if the label has been manually erased.
        """
        import numpy as np

        lbl_name = self.labels_combo.currentText()
        if not lbl_name or lbl_name not in self.viewer.layers:
            return True  # can't check, assume exists
        data = self.viewer.layers[lbl_name].data

        # Fast path: check a small cube around the cached centroid.
        # Look up centroid from Navigator items list (not _items_by_label).
        centroid = None
        nav = getattr(self, "_navigator_widget", None)
        if nav is not None:
            for item in getattr(nav, "_items", []):
                if item.get("label_id") == label_id:
                    centroid = item.get("centroid")
                    break

        if centroid is not None and len(centroid) >= 3:
            z, y, x = int(centroid[0]), int(centroid[1]), int(centroid[2])
            margin = 50
            roi = data[
                max(0, z - margin):min(data.shape[0], z + margin),
                max(0, y - margin):min(data.shape[1], y + margin),
                max(0, x - margin):min(data.shape[2], x + margin),
            ]
            if np.any(roi == label_id):
                print(f"[XCRASH-9] exists_check label={label_id} FAST-PATH hit", flush=True)
                return True

        # Slow fallback: full volume check
        import time as _time
        _t0 = _time.perf_counter()
        result = bool(np.any(data == label_id))
        _dt = _time.perf_counter() - _t0
        print(
            f"[XCRASH-9] exists_check label={label_id} SLOW-PATH "
            f"result={result} dt={_dt:.3f}s data.shape={data.shape}",
            flush=True,
        )
        return result

    def _zero_label_in_volume(self, label_id: int) -> None:
        """Zero out all remaining voxels of a label in the main volume.

        Used by intent-based deletion to clean up partial erasures.
        """
        lbl_name = getattr(self, "_active_labels_name", None)
        if not lbl_name:
            lbl_name = self.labels_combo.currentText()
        if not lbl_name or lbl_name not in self.viewer.layers:
            return
        data = self.viewer.layers[lbl_name].data
        data[data == label_id] = 0

    def _handle_manually_deleted_label(self, label_id: int) -> None:
        """Reconcile state for a label that was manually erased from volume.

        Fires the delete callback (same as "Delete Label" button) and removes
        the entry from the Navigator.
        """
        if self._on_delete_label_cb:
            try:
                self._on_delete_label_cb(label_id, roi_slices=None, old_roi_data=None)
            except Exception:
                pass
        nav = getattr(self, "_navigator_widget", None)
        if nav is not None:
            try:
                nav.remove_item(label_id)
            except Exception:
                pass

    def _on_navigator_select_label(self, label_id) -> None:
        """Handle label selection from the Navigator.

        Sets the selected label across all viewers.  EDT overlay is NOT
        recomputed on every label switch — it is expensive (full-volume
        np.argwhere + distance_transform_edt) and should only update when
        explicitly toggled or during split mode.

        Blocked during active split or smooth to prevent changing the target
        label while an operation is in progress.  Fragment refine mode is
        exempted — the user needs to select different fragments.
        """
        # [XCRASH-2] select_label entry
        print(
            f"[XCRASH-2] select_label ENTER target={label_id} "
            f"split_running={self._split_running} "
            f"split_fragment_refine={self._split_fragment_refine} "
            f"smooth_running={self._smooth_running} "
            f"interp_snapshot_set={self._interp_snapshot is not None} "
            f"interp_refine_mode={self._interp_refine_mode} "
            f"last_nav={self._last_navigated_label_id}",
            flush=True,
        )
        if self._split_running and not self._split_fragment_refine:
            print("[XCRASH-2]   GUARD: split_running, returning", flush=True)
            return
        if self._smooth_running:
            print("[XCRASH-2]   GUARD: smooth_running, returning", flush=True)
            return
        if self._interp_snapshot is not None or self._interp_refine_mode:
            print("[XCRASH-2]   GUARD: interp staged, returning", flush=True)
            return

        # WP-B: Check if the previously navigated label was manually deleted
        # since we last visited it (e.g. user fill-deleted it with label 0).
        prev = self._last_navigated_label_id
        if prev is not None and prev != label_id:
            print(f"[XCRASH-2]   checking prev label={prev} exists in volume", flush=True)
            if not self._check_label_exists_in_volume(prev):
                print(f"[XCRASH-2]   prev={prev} GONE, ghost-row reconciliation", flush=True)
                self._handle_manually_deleted_label(prev)

        # WP-B: Check if the target label itself still exists.
        print(f"[XCRASH-2]   checking target={label_id} exists in volume", flush=True)
        if not self._check_label_exists_in_volume(label_id):
            print(f"[XCRASH-2]   target={label_id} GONE, ghost-row + early return", flush=True)
            self._handle_manually_deleted_label(label_id)
            self._last_navigated_label_id = None
            return

        print(f"[XCRASH-2]   target={label_id} exists, proceeding to _set_selected_label_all", flush=True)
        self._last_navigated_label_id = label_id
        # Discard draw snapshot from previous label (user navigated away).
        self._clear_draw_snapshot()

        self._set_selected_label_all(label_id)

        # R4-WP4: Notify controller for coordinator dispatch.
        # WP-4 / Bug H: Pass all selected IDs and isolate state so the
        # bridge can dispatch mode="isolate" when appropriate.
        cb = getattr(self, "_on_navigator_label_selected_cb", None)
        if cb is not None:
            try:
                nav = getattr(self, "_navigator_widget", None)
                all_sel = list(nav.selected_label_ids()) if nav else [label_id]
                isolate = (
                    nav._isolate_cb.isChecked()
                    if nav and hasattr(nav, "_isolate_cb")
                    else False
                )
                # Autopan fix: when the controller already owns isolation
                # (from lasso), dispatch as focus to avoid re-running
                # _isolate_labels_in_viewer (which includes processEvents
                # that interferes with subviewer autopan).  The annotator's
                # _fire_isolation_if_active handles the lightweight
                # transition to ANNOTATOR_ISOLATION ownership.
                auth = getattr(self, '_colormap_authority', None)
                if isolate and auth is not None:
                    from napari_epicycle.fusion_reconstruction.colormap_authority import ColormapOwner
                    if auth.current_owner in (
                        ColormapOwner.CONTROLLER_QC,
                        ColormapOwner.CONTROLLER_ISOLATION,
                    ):
                        isolate = False
                # Phase 1: 3D single-label fast lane — skip controller
                # isolation when ndisplay=3 and single label.  The
                # annotator handles isolation locally via
                # _fire_isolation_if_active → show_selected_label=True,
                # avoiding the expensive controller _isolate_labels_in_viewer
                # which triggers a full iso_categorical re-render.
                # Multi-label 3D isolation still goes through the controller.
                if isolate and len(all_sel) == 1:
                    _dims = getattr(self.viewer, "dims", None)
                    ndisplay = getattr(_dims, "ndisplay", 2) if _dims else 2
                    if ndisplay == 3:
                        isolate = False
                # F3: Set guard flag so _on_isolation_changed (called next
                # by _fire_isolation_if_active) doesn't double-dispatch.
                self._isolation_bridge_dispatched = True
                cb(label_id, all_selected=all_sel, isolate=isolate)
            except Exception:
                pass

    def _refresh_edt_if_checked(self) -> None:
        """Re-compute EDT overlay if the checkbox is currently checked."""
        adv_widget = self._get_advanced_features_widget()
        if adv_widget is None:
            return
        cb = getattr(adv_widget, "split_edt_cb", None)
        if cb is not None and cb.isChecked():
            self._toggle_edt_overlay(True)

    def _auto_enable_edt_for_split(self) -> None:
        """Auto-check the EDT overlay checkbox when starting a split.

        When isolate mode was on, it must be disabled during split (because
        selected_label switches to the seed ID). The EDT overlay compensates
        by visually highlighting the target label with a turbo heatmap.
        """
        adv_widget = self._get_advanced_features_widget()
        if adv_widget is None:
            return
        cb = getattr(adv_widget, "split_edt_cb", None)
        if cb is not None:
            cb.setEnabled(True)  # enable first (disabled by default)
            if not cb.isChecked():
                cb.setChecked(True)  # triggers toggled → _toggle_edt_overlay

    def _ensure_shared_data_aligned(self, labels_name: str):
        """Defensive: ensure sync_manager.shared_labels[labels_name] points at
        the CURRENT main layer.data. Re-registers if divergent.

        Background: napari's Labels.data setter can create a new internal
        array (dtype normalization) and controller paths
        (`_zero_label_ids`, `_commit_triage`, layer reload from disk, etc.)
        do `data = layer.data.copy(); ...; layer.data = data` which
        replaces layer.data with a fresh array. After such reassignments,
        the registered shared_data still points at the OLD array. Refinement
        ops that read shared_data then see stale state.

        Symptom of the bug: snapshots / refine baselines / EDT overlays
        cover the pre-reassignment bbox of the label. After cancel-restore
        the volume looks like it was clipped at a clean horizontal slice.

        This helper is called at the entry point of every refinement op
        that reads shared_data (split, smooth, interp, merge). Cost:
        one identity comparison + an O(1) dict update if divergent.

        Returns the aligned array (= main layer.data) or None if the layer
        does not exist / is not registered.
        """
        if not labels_name:
            return None
        sm = getattr(self, "sync_manager", None)
        if sm is None:
            return None
        main_layer = self._layer_by_name(self.viewer.layers, labels_name)
        if main_layer is None:
            return sm.shared_labels.get(labels_name)
        registered = sm.shared_labels.get(labels_name)
        if registered is not main_layer.data:
            sm.register_shared_data(labels_name, main_layer.data)
        return main_layer.data

    def _navigator_navigate(self, label_id, centroid) -> None:
        """Navigate subviewers to centroid when navigator selects a label."""
        if centroid is not None:
            z, y, x = centroid
            self.navigate_to_slice("XY", z)
            self.navigate_to_slice("XZ", y)
            self.navigate_to_slice("YZ", x)

    def _on_isolation_changed(self, label_ids) -> None:
        """Handle multi-label isolation from the Navigator.

        When *label_ids* is non-empty, builds a ``DirectLabelColormap``
        showing only those labels (rest transparent).  When empty,
        restores the previous colormap.  Uses ``DirectLabelColormap``
        instead of ``show_selected_label`` because the latter only
        supports a single label.

        WP-3: Uses ColormapAuthority to coordinate with controller.
        """
        import time as _time
        from napari.layers import Labels

        _t0 = _time.perf_counter()
        # [XCRASH-3] isolation change entry — full state snapshot.
        try:
            _ndim = getattr(self.viewer.dims, "ndisplay", 2)
            print(
                f"[XCRASH-3] iso_changed ENTER ids={sorted(int(x) for x in (label_ids or []))} "
                f"ndisplay={_ndim} "
                f"current_cache={set(self._isolation_label_set) if getattr(self, '_isolation_label_set', None) else None} "
                f"iso_active={getattr(self, '_isolation_active', False)} "
                f"proxy_active={getattr(self, '_isolation_proxy_active', False)} "
                f"proxy_lid={getattr(self, '_isolation_proxy_label_id', None)} "
                f"active_labels_name={self._active_labels_name!r}",
                flush=True,
            )
        except Exception as _exc:
            print(f"[XCRASH-3] probe error: {_exc!r}", flush=True)

        labels_name = self._active_labels_name
        if not labels_name:
            print("[XCRASH-3]   EARLY-RETURN: no active labels name", flush=True)
            _safe_curation_debug(self, "isolation ignored: no active labels name")
            return
        main_layer = self._layer_by_name(self.viewer.layers, labels_name)
        if not isinstance(main_layer, Labels):
            print(f"[XCRASH-3]   EARLY-RETURN: layer not Labels ({labels_name})", flush=True)
            _safe_curation_debug(self,
                f"isolation ignored: active layer is not Labels ({labels_name})"
            )
            return
        print(
            f"[XCRASH-3]   main_layer found name={main_layer.name!r} "
            f"selected_label={getattr(main_layer, 'selected_label', None)} "
            f"visible={getattr(main_layer, 'visible', None)} "
            f"data.shape={getattr(main_layer.data, 'shape', '?')} ",
            flush=True,
        )

        auth = self._colormap_authority
        owner_before = getattr(auth, "current_owner", None) if auth is not None else None
        _safe_curation_debug(self, 
            "isolation changed "
            f"ids={sorted(int(x) for x in set(label_ids or []))} "
            f"owner_before={owner_before} "
            f"isolation_active={self._isolation_active}"
        )

        if label_ids:
            label_set = set(label_ids)
            if len(label_set) == 1:
                # --- Fast path: single label → use napari's built-in
                # show_selected_label (no 3D re-render, just shader uniform). ---
                target_id = next(iter(label_set))

                # Check ndisplay for proxy vs show_selected_label decision
                _dims = getattr(self.viewer, "dims", None)
                ndisplay = getattr(_dims, "ndisplay", 2) if _dims else 2

                # True no-op: proxy already showing this label
                if (
                    getattr(self, "_isolation_proxy_active", False)
                    and getattr(self, "_isolation_proxy_label_id", None) == target_id
                ):
                    return

                # True no-op: if isolation is already active via
                # show_selected_label and the label was already set by
                # _set_selected_label_all, there is nothing to do.
                # Exception: in 3D mode, we need to create the proxy even
                # if show_selected_label is True — the proxy replaces the
                # expensive full-volume iso_categorical render.
                _need_proxy = (
                    ndisplay == 3
                    and not getattr(self, "_isolation_proxy_active", False)
                )
                if (
                    self._isolation_active
                    and self._pre_isolation_colormap is None
                    and main_layer.selected_label == target_id
                    and getattr(main_layer, "show_selected_label", False)
                    and not _need_proxy
                ):
                    return

                # WP-3: Claim annotator isolation ownership.
                if auth is not None:
                    # Release any controller ownership first so we can claim.
                    from napari_epicycle.fusion_reconstruction.colormap_authority import ColormapOwner
                    if auth.current_owner in (
                        ColormapOwner.CONTROLLER_QC,
                        ColormapOwner.CONTROLLER_ISOLATION,
                    ):
                        auth.release(auth.current_owner)
                    if not auth.claim(ColormapOwner.ANNOTATOR_ISOLATION):
                        _safe_curation_debug(self,
                            "isolation claim failed in single-label path "
                            f"owner_now={auth.current_owner}"
                        )
                        return

                # If we were previously using a DirectLabelColormap for
                # multi-select, restore the original colormap first.
                if self._pre_isolation_colormap is not None:
                    main_layer.colormap = self._pre_isolation_colormap
                    self._pre_isolation_colormap = None
                self._isolation_label_set = None
                self._isolation_active = True
                if main_layer.selected_label != target_id:
                    main_layer.selected_label = target_id

                # Phase 2: 3D proxy display layer for fast isolation
                if ndisplay == 3:
                    # Set suppress flag FIRST — before any property changes
                    # that fire sync events.  main_layer.show_selected_label=False
                    # fires _sync_show_selected_label which would propagate
                    # False to subviewers if the flag isn't set yet.
                    sync = getattr(self, "sync_manager", None)
                    if sync is not None:
                        sync._suppress_show_selected_sync = True
                    # Use proxy layer — small ROI for fast isosurface
                    if hasattr(main_layer, "show_selected_label"):
                        main_layer.show_selected_label = False
                    self._update_isolation_proxy(target_id, main_layer)
                    # Subviewers stay on show_selected_label path (2D)
                    self._propagate_show_selected_label(
                        True, label_id=target_id
                    )
                    _safe_curation_debug(self,
                        "3D proxy isolation applied "
                        f"target={target_id} ndisplay={ndisplay}"
                    )
                else:
                    # 2D path: use show_selected_label (fast shader)
                    if getattr(self, "_isolation_proxy_active", False):
                        self._remove_isolation_proxy()
                    if hasattr(main_layer, "show_selected_label"):
                        main_layer.show_selected_label = True
                    # WP-A: Propagate show_selected_label to subviewers.
                    matched = self._propagate_show_selected_label(
                        True, label_id=target_id
                    )
                    _safe_curation_debug(self,
                        "single-label isolation applied "
                        f"target={target_id} "
                        f"main_selected={getattr(main_layer, 'selected_label', None)} "
                        f"matched_subviewer_layers={matched}"
                    )
            else:
                # --- Slow path: multiple labels → DirectLabelColormap
                # (only way to show N labels simultaneously). ---

                # WP-3: Claim annotator isolation ownership.
                if auth is not None:
                    from napari_epicycle.fusion_reconstruction.colormap_authority import ColormapOwner
                    if auth.current_owner in (
                        ColormapOwner.CONTROLLER_QC,
                        ColormapOwner.CONTROLLER_ISOLATION,
                    ):
                        auth.release(auth.current_owner)
                    if not auth.claim(ColormapOwner.ANNOTATOR_ISOLATION):
                        _safe_curation_debug(self, 
                            "isolation claim failed in multi-label path "
                            f"owner_now={auth.current_owner}"
                        )
                        return

                # Decide whether to use the multi-label PROXY (3D mode) or
                # the legacy DirectLabelColormap-on-main path (2D mode).
                # In 3D, applying a colormap to the full main layer forces
                # napari to re-run iso_categorical extraction over the
                # whole 1024×1024×Z volume — the dominant cost when the
                # user ticks merge-dialog candidates one by one. In 2D
                # there is no iso re-render, so the legacy path is fine.
                _dims = getattr(self.viewer, "dims", None)
                _ndisplay = getattr(_dims, "ndisplay", 2) if _dims else 2

                if _ndisplay == 3:
                    # 3D: build N small single-label proxy crops (one per
                    # selected label, each in show_selected_label mode).
                    # Subviewers stay on the legacy DirectLabelColormap path
                    # because it's cheap in 2D and gives consistent multi-color
                    # rendering for the 2D slice views.
                    if not self._isolation_active or self._pre_isolation_colormap is None:
                        self._pre_isolation_colormap = main_layer.colormap
                        if auth is not None:
                            auth.set_restore_point(main_layer.colormap)
                    self._isolation_active = True
                    # Suppress show_selected_label propagation — main is
                    # hidden and the multi sub-proxies handle the 3D render.
                    if hasattr(main_layer, "show_selected_label"):
                        main_layer.show_selected_label = False
                    self._propagate_show_selected_label(False)
                    # Build / diff the per-label sub-proxies (each one
                    # is the same lightweight create as a refine isolation
                    # crop — proven render path).
                    self._apply_multi_label_proxy(main_layer, label_set)
                    self._isolation_label_set = frozenset(int(x) for x in label_set)
                    # Build the multi-label DirectLabelColormap fresh and
                    # propagate to subviewers. The 3D render uses the
                    # sub-proxies (above); subviewers use this colormap.
                    self._propagate_multi_label_colormap_to_subviewers(label_set)
                    _safe_curation_debug(self,
                        f"multi-label proxy isolation applied count={len(label_set)}"
                    )
                else:
                    # 2D: legacy path — colormap on main is cheap because
                    # there is no iso_categorical re-render.
                    if getattr(self, "_isolation_proxy_active", False):
                        self._remove_isolation_proxy(restore_visibility=False)

                    if not self._isolation_active or self._pre_isolation_colormap is None:
                        self._pre_isolation_colormap = main_layer.colormap
                        if auth is not None:
                            auth.set_restore_point(main_layer.colormap)
                    self._isolation_active = True
                    if hasattr(main_layer, "show_selected_label"):
                        main_layer.show_selected_label = False
                    self._propagate_show_selected_label(False)
                    self._apply_multi_isolation_colormap(main_layer, label_set)
                    _safe_curation_debug(self,
                        f"multi-label isolation applied count={len(label_set)}"
                    )

            # F3: Dispatch through coordinator for plot highlight sync.
            # Only dispatches when the user manually toggled the Isolate
            # checkbox.  When called from row navigation, the bridge already
            # dispatched (guard flag prevents double dispatch).
            if not getattr(self, '_isolation_bridge_dispatched', False):
                cb = getattr(self, "_on_navigator_label_selected_cb", None)
                if cb is not None:
                    nav = getattr(self, "_navigator_widget", None)
                    all_sel = (
                        list(nav.selected_label_ids()) if nav else list(label_ids)
                    )
                    primary = all_sel[0] if all_sel else None
                    try:
                        cb(primary, all_selected=all_sel, isolate=True)
                    except Exception:
                        pass
            self._isolation_bridge_dispatched = False
        else:
            # Phase 2: Remove proxy on isolation clear.
            # Defer main_layer.visible=True until AFTER colormap and
            # property changes to avoid multiple expensive 3D re-renders.
            _had_proxy = getattr(self, "_isolation_proxy_active", False)
            if _had_proxy:
                self._remove_isolation_proxy(restore_visibility=False)
            # Restore previous colormap
            if self._isolation_active:
                self._isolation_active = False
                self._isolation_label_set = None  # Clear cache for next activation

                # Performance: When the controller callback exists, skip
                # the annotator's colormap restore — the controller will
                # immediately apply the correct QC feature colormap via
                # _on_annotator_isolation_cleared → _update_viewer_feature_colors.
                # This eliminates 4 redundant colormap assignments (1 main
                # layer + 3 subviewers) that would be overwritten anyway.
                _controller_will_handle_colormap = bool(self._on_isolation_cleared_cb)

                if not _controller_will_handle_colormap:
                    # No controller callback — annotator must restore colormap.
                    # WP-3: Restore from authority if available, else fallback.
                    restored = None
                    if auth is not None:
                        restored = auth.restore()
                    from contextlib import nullcontext
                    sync = getattr(self, "sync_manager", None)
                    _suppress_ctx = (sync.suppress_property_sync()
                                     if sync is not None and hasattr(sync, "suppress_property_sync")
                                     else nullcontext())
                    with _suppress_ctx:
                        if restored is not None:
                            main_layer.colormap = restored
                        elif self._pre_isolation_colormap is not None:
                            main_layer.colormap = self._pre_isolation_colormap
                        self._pre_isolation_colormap = None
                        self._propagate_colormap_to_subviewers(main_layer.colormap)
                else:
                    # Controller handles colormap — just clear saved state.
                    if auth is not None:
                        auth.restore()  # Pop restore point (discard)
                    self._pre_isolation_colormap = None

                # WP-3: Release annotator ownership.
                if auth is not None:
                    from napari_epicycle.fusion_reconstruction.colormap_authority import ColormapOwner
                    auth.release(ColormapOwner.ANNOTATOR_ISOLATION)

                if hasattr(main_layer, "show_selected_label"):
                    main_layer.show_selected_label = False
                # WP-A: Clear show_selected_label on subviewers too.
                self._propagate_show_selected_label(False)

            # B1 fix: Always notify controller when Isolate is unchecked,
            # even if _isolation_active was False.  This handles the case
            # where the controller set isolation via lasso (CONTROLLER_ISOLATION)
            # and the checkbox was silently checked — the annotator never set
            # _isolation_active, but the controller still needs to clean up
            # _isolated_label_ids, plot highlight, etc.
            if self._on_isolation_cleared_cb:
                try:
                    self._on_isolation_cleared_cb(reason="toggle_off")
                except TypeError:
                    self._on_isolation_cleared_cb()
                except Exception:
                    pass

            # Deferred visibility: restore main_layer.visible AFTER all
            # property changes (colormap, show_selected_label, controller
            # callback) are done.  Use QTimer.singleShot(0, ...) to push
            # the expensive 3D isosurface re-render to the next event loop
            # tick — keeps the UI responsive while the re-render runs.
            if _had_proxy:
                _ml = main_layer  # capture for closure
                try:
                    shot = getattr(self, "_single_shot_in_session", None)

                    def _reshow_main(_l=_ml):
                        _l.visible = True
                        _heal_render(self, _l, tag="isolation-off")

                    if callable(shot):
                        shot(0, _reshow_main)
                    else:
                        from qtpy.QtCore import QTimer
                        QTimer.singleShot(0, _reshow_main)
                except Exception:
                    # Fallback: set synchronously if Qt unavailable
                    try:
                        main_layer.visible = True
                        _heal_render(self, main_layer, tag="isolation-off-sync")
                    except Exception:
                        pass
        owner_after = getattr(auth, "current_owner", None) if auth is not None else None
        _dt = _time.perf_counter() - _t0
        if _dt > 0.03:
            _safe_curation_debug(self, 
                "isolation callback perf "
                f"dt={_dt:.3f}s owner_after={owner_after} "
                f"main_show_sel={getattr(main_layer, 'show_selected_label', None)} "
                f"main_selected={getattr(main_layer, 'selected_label', None)}"
            )

    # ------------------------------------------------------------------
    # Phase 2: 3D isolation proxy display layer
    # ------------------------------------------------------------------

    def _update_isolation_proxy(self, label_id: int, main_layer, *, force=False, roi_hint=None) -> None:
        """Create or update a small proxy layer for fast 3D isolation.

        Extracts the bounding-box ROI of *label_id* from the main layer
        and creates a small Labels layer positioned at the ROI origin.
        The proxy uses ``show_selected_label=True`` which triggers
        isosurface extraction only on the small ROI volume, avoiding
        the expensive full-volume ``iso_categorical`` re-render.

        The main layer is hidden (``visible=False``) while the proxy
        is active.  Subviewers are unaffected — they have their own
        layers and remain on the main data.

        Parameters
        ----------
        force : bool
            If True, rebuild the proxy even if it already shows *label_id*.
            Used after operations that modify the label's geometry in-place.
        roi_hint : tuple of slices, optional
            If provided, restricts the bounding-box scan to this ROI of the
            main volume instead of scanning the entire array.  Used during
            fragment refine where all fragments are known to reside within
            the split ROI — avoids expensive full-volume comparisons.
        """
        import numpy as np
        from napari.layers import Labels

        # [XCRASH-5] proxy update entry
        print(
            f"[XCRASH-5] update_proxy ENTER target={label_id} "
            f"force={force} roi_hint={roi_hint is not None} "
            f"current_proxy_active={self._isolation_proxy_active} "
            f"current_proxy_lid={self._isolation_proxy_label_id}",
            flush=True,
        )

        # No-op: already showing this label (unless forced)
        if (
            not force
            and self._isolation_proxy_active
            and self._isolation_proxy_label_id == label_id
        ):
            print("[XCRASH-5]   NO-OP: same label, returning", flush=True)
            return

        # Find bounding box via axis-reduced np.any (avoids materializing
        # a dense (N_voxels, 3) coordinate array from np.argwhere).
        # When roi_hint is provided, scan only that sub-volume and offset
        # the resulting indices by the ROI origin.
        if roi_hint is not None:
            scan_data = main_layer.data[roi_hint]
            offset = tuple(s.start for s in roi_hint)
        else:
            scan_data = main_layer.data
            offset = (0, 0, 0)
        print(f"[XCRASH-5]   bbox scan shape={scan_data.shape} target={label_id}", flush=True)

        mask = scan_data == label_id
        z_any = np.any(mask, axis=(1, 2))
        if not z_any.any():
            print(
                f"[XCRASH-5]   LABEL NOT IN VOLUME target={label_id} — "
                f"this means stale isolation request, returning",
                flush=True,
            )
            return
        y_any = np.any(mask, axis=(0, 2))
        x_any = np.any(mask, axis=(0, 1))
        z_idx = np.where(z_any)[0]
        y_idx = np.where(y_any)[0]
        x_idx = np.where(x_any)[0]

        margin = 5
        shape = main_layer.data.shape
        slices = (
            slice(max(0, offset[0] + int(z_idx[0]) - margin), min(shape[0], offset[0] + int(z_idx[-1]) + margin + 1)),
            slice(max(0, offset[1] + int(y_idx[0]) - margin), min(shape[1], offset[1] + int(y_idx[-1]) + margin + 1)),
            slice(max(0, offset[2] + int(x_idx[0]) - margin), min(shape[2], offset[2] + int(x_idx[-1]) + margin + 1)),
        )
        roi_data = main_layer.data[slices].copy()
        translate = [float(s.start) for s in slices]
        scale = list(main_layer.scale)

        if self._isolation_proxy_layer is not None:
            # Update existing proxy in-place
            self._isolation_proxy_layer.data = roi_data
            self._isolation_proxy_layer.translate = translate
            self._isolation_proxy_layer.selected_label = label_id
        else:
            # Create new proxy layer
            proxy = Labels(
                roi_data,
                name="_epicycle_isolation_proxy",
                translate=translate,
                scale=scale,
            )
            proxy.editable = False
            proxy.selected_label = label_id
            proxy.show_selected_label = True
            # brush_size locked at 1.  This is the
            # layer the user sees in the main viewer when isolation is
            # active; locking it ensures the brush slider on the main
            # viewer agrees with subviewers.
            from packages.epicycle_annotator.sync import lock_brush_size_to_1
            lock_brush_size_to_1(proxy)
            # Inherit rendering properties from main layer
            proxy.opacity = main_layer.opacity
            try:
                proxy.rendering = main_layer.rendering
            except Exception:
                pass
            self.viewer.layers.append(proxy)
            self._isolation_proxy_layer = proxy

        # Do NOT copy DirectLabelColormap from main layer to proxy —
        # DirectLabelColormap has explicit colors for ALL labels, which
        # defeats show_selected_label isolation (all labels render).
        # The proxy uses napari's default colormap; with
        # show_selected_label=True only the target label renders.
        # Copy rendering properties only (iso_gradient_mode).
        try:
            self._isolation_proxy_layer.iso_gradient_mode = (
                main_layer.iso_gradient_mode
            )
        except Exception:
            pass
        # Re-assert show_selected_label after all property setup to
        # ensure it's the final state the renderer sees.
        self._isolation_proxy_layer.show_selected_label = True

        # Hide main layer — proxy handles 3D rendering.
        # Only set visible=False if not already hidden.  Redundant
        # assignments fire events.visible which cascades to subviewers
        # via sync_layer_visibility, making subviewer labels invisible
        # on every stroke (post_sync_callback → _update_isolation_proxy).
        # A perf re-show scheduled by _set_selected_label_all in this same
        # tick must not un-hide main underneath the proxy (2026-09-02).
        # Inline token bump (== _cancel_main_perf_reshow) so lightweight
        # test doubles without the helper still work.
        self._perf_reshow_token = getattr(self, "_perf_reshow_token", 0) + 1
        if main_layer.visible:
            main_layer.visible = False
        self._isolation_proxy_active = True
        self._isolation_proxy_label_id = label_id
        # Single-label takeover: tear down any multi-label sub-proxies
        # (otherwise both would render) and clear the bookkeeping field.
        if getattr(self, "_multi_label_proxy_layers", None):
            for lid, layer in list(self._multi_label_proxy_layers.items()):
                try:
                    layer.visible = False
                    self.viewer.layers.remove(layer)
                except (ValueError, KeyError):
                    pass
            self._multi_label_proxy_layers = {}
        self._multi_label_proxy_set = None

        # Ensure suppress flag is set (redundant with _on_isolation_changed
        # but needed when _update_isolation_proxy is called from other paths).
        sync = getattr(self, "sync_manager", None)
        if sync is not None:
            sync._suppress_show_selected_sync = True

    def _remove_isolation_proxy(self, *, restore_visibility: bool = True) -> None:
        """Remove the proxy display layer and optionally restore main layer visibility.

        Parameters
        ----------
        restore_visibility : bool
            If True (default), set ``main_layer.visible = True`` immediately.
            Pass False to defer visibility restoration — useful when the
            caller will set colormap/properties first to avoid multiple
            expensive 3D re-renders.
        """
        # [XCRASH-6] proxy remove entry
        print(
            f"[XCRASH-6] remove_proxy ENTER restore_vis={restore_visibility} "
            f"proxy_layer_set={self._isolation_proxy_layer is not None} "
            f"proxy_active={self._isolation_proxy_active}",
            flush=True,
        )
        if self._isolation_proxy_layer is not None:
            try:
                # Hide before removing — disconnects the vispy visual from
                # the scene graph so no stale texture references remain when
                # the layer is destroyed.  Prevents access violations from
                # the rendering loop accessing freed GPU textures.
                self._isolation_proxy_layer.visible = False
                self.viewer.layers.remove(self._isolation_proxy_layer)
                print("[XCRASH-6]   proxy layer removed cleanly", flush=True)
            except (ValueError, KeyError) as _exc:
                print(f"[XCRASH-6]   proxy removal raised: {_exc!r}", flush=True)
            except Exception as _exc:
                print(f"[XCRASH-6]   proxy removal UNEXPECTED: {_exc!r}", flush=True)
            self._isolation_proxy_layer = None

        # Tear down all multi-label sub-proxies (one per selected label).
        # Same discipline (hide first, then remove) to avoid vispy access violations.
        if getattr(self, "_multi_label_proxy_layers", None):
            for lid, layer in list(self._multi_label_proxy_layers.items()):
                try:
                    layer.visible = False
                    self.viewer.layers.remove(layer)
                except (ValueError, KeyError):
                    pass
            self._multi_label_proxy_layers = {}

        self._isolation_proxy_active = False
        self._isolation_proxy_label_id = None
        # Clear multi-label proxy bookkeeping (no-op if it was a
        # single-label proxy).
        self._multi_label_proxy_set = None

        # Clear suppress flag so sync resumes normal show_selected_label
        # propagation now that proxy is gone.
        sync = getattr(self, "sync_manager", None)
        if sync is not None:
            sync._suppress_show_selected_sync = False

        # Restore main layer visibility (unless caller defers)
        if restore_visibility:
            try:
                main_layer = self.viewer.layers[self._active_labels_name]
                main_layer.visible = True
                _heal_render(self, main_layer, tag="proxy-removed")
            except (KeyError, Exception):
                pass

    def _apply_multi_isolation_colormap(self, main_layer, label_ids) -> None:
        """Build and apply DirectLabelColormap showing only *label_ids*.

        Skips rebuild if the isolated label set hasn't changed (e.g.,
        arrow-key navigation with single-select fires this on every
        keypress with the same single label).

        Applies colormap to both main layer and subviewers in all modes.
        """
        # Fast path: skip if the isolated set is unchanged.
        frozen = frozenset(label_ids)
        prev_cache = getattr(self, "_isolation_label_set", None)
        # [XCRASH-4] multi-isolation entry + cache fast-skip probe.
        # User repro post-merge: cache might hold dead label IDs that
        # were absorbed.  Print BOTH cache state and incoming set so
        # we can tell if the fast-skip is hiding a stale render.
        print(
            f"[XCRASH-4] apply_multi ENTER incoming={set(label_ids)} "
            f"cache={set(prev_cache) if prev_cache else None} "
            f"cache_hit={frozen == prev_cache} "
            f"main_layer={main_layer.name!r} "
            f"main_visible={main_layer.visible}",
            flush=True,
        )
        if frozen == prev_cache:
            print("[XCRASH-4]   FAST-SKIP cache hit, returning", flush=True)
            return
        self._isolation_label_set = frozen

        import numpy as np

        try:
            from napari.utils.colormaps import DirectLabelColormap, label_colormap
        except ImportError:
            return

        hidden = np.array([0, 0, 0, 0], dtype=np.float32)
        # Cache the default colormap — no need to recreate on every call.
        if not hasattr(self, "_cached_label_colormap"):
            self._cached_label_colormap = label_colormap(49, seed=0.5)
        default_cmap = self._cached_label_colormap

        color_dict = {None: hidden, 0: hidden}
        if label_ids:
            all_ids = np.array(sorted(label_ids), dtype=np.intp)
            all_colors = default_cmap.map(all_ids).astype(np.float32)
            for i, lid in enumerate(all_ids):
                color_dict[int(lid)] = all_colors[i]

        # Suppress sync cascade during colormap assignment — the explicit
        # _propagate_colormap_to_subviewers call below handles subviewer
        # updates, making the sync cascade redundant double work.
        from contextlib import nullcontext
        sync = getattr(self, "sync_manager", None)
        ctx = (sync.suppress_property_sync()
               if sync is not None and hasattr(sync, "suppress_property_sync")
               else nullcontext())

        # Hide main BEFORE colormap assignment to prevent synchronous
        # iso_categorical 3D re-render from blocking the UI thread.
        # Visibility is restored on the next event loop tick via QTimer
        # so the re-render happens asynchronously.  events.visible
        # cascades to subviewers via sync_layer_visibility — cheap 2D
        # flip that gets restored automatically when main is shown.
        #
        # Always-restore: multi-label needs main visible in steady state.
        # Schedule a QTimer regardless of was_visible — handles both fresh
        # apply (was_visible=True) and proxy→multi transition where main
        # came in invisible (was_visible=False, proxy was hiding it).
        # Mirrors the lasso path in controller._isolate_labels_in_viewer.
        if main_layer.visible:
            main_layer.visible = False

        with ctx:
            main_layer.colormap = DirectLabelColormap(color_dict=color_dict)
            try:
                from napari_epicycle.fusion_reconstruction._render_diag import render_state as _rs
                _rs('ann-multi-after-set', main_layer, viewer=self.viewer)
            except Exception:
                pass
            matched = self._propagate_colormap_to_subviewers(main_layer.colormap)

        # Always schedule visibility restore — multi-label steady state
        # needs main visible.  If main is already visible, this is a no-op
        # on next tick.
        from qtpy.QtCore import QTimer
        _l = main_layer
        QTimer.singleShot(0, lambda l=_l: setattr(l, "visible", True))

        _safe_curation_debug(self,
            "propagate multi-isolation colormap "
            f"labels={len(label_ids)} "
            f"matched_subviewer_layers={matched}"
        )

    def _apply_multi_label_proxy(self, main_layer, label_set) -> None:
        """Multi-label isolation via N small single-label proxy crops (3D mode).

        For each label in *label_set*, create or keep a small Labels
        layer cropped to that label's bounding box, rendered via
        ``show_selected_label=True``.  This is the SAME render path the
        single-label refine isolation uses (proven, fast,
        clean-looking) — just instantiated once per selected label.

        Why N proxies and not one multi-color proxy:
        napari's 3D ``iso_categorical`` mode reliably renders the
        ``show_selected_label`` shader path; with a
        ``DirectLabelColormap`` it does not always render multi-label
        sets correctly in 3D, even though 2D subviewers handle the
        colormap fine. Using N proven single-label proxies guarantees
        the visual matches refine isolation exactly — bounding-box
        crop, isosurface render, one cell at a time, just N of them.

        The win vs ``_apply_multi_isolation_colormap`` (which set a
        DirectLabelColormap on the FULL main layer): each per-toggle
        cost is the same as a single-label refine isolation switch
        (one bbox scan + one small proxy create), not a full-volume
        colormap re-render of all 1024×1024×Z voxels.

        Subviewers continue to use the propagated DirectLabelColormap
        path (cheap in 2D, no isosurface extraction).

        Lifecycle:
        - Per-label proxies live in ``self._multi_label_proxy_layers``
          (dict keyed by label_id).
        - Adding a label = compute its bbox + create one new proxy.
        - Removing a label = pop + remove that one proxy.
        - Clearing = ``_remove_isolation_proxy`` walks the dict and
          tears down all entries.
        """
        import numpy as np
        from napari.layers import Labels

        if not label_set:
            return

        frozen = frozenset(int(x) for x in label_set)
        # No-op: identical set already displayed.
        if (
            getattr(self, "_multi_label_proxy_set", None) == frozen
            and self._multi_label_proxy_layers
        ):
            return

        # Tear down any single-label proxy from a prior single→multi
        # transition. (The single-label proxy uses _isolation_proxy_layer;
        # multi uses the dict. Don't keep both — tear single down so the
        # one keep_id label gets recreated cleanly as a multi-mode proxy.)
        if self._isolation_proxy_layer is not None:
            try:
                self._isolation_proxy_layer.visible = False
                self.viewer.layers.remove(self._isolation_proxy_layer)
            except (ValueError, KeyError):
                pass
            self._isolation_proxy_layer = None
            self._isolation_proxy_label_id = None

        # Diff against current state — only touch labels that changed.
        current = set(self._multi_label_proxy_layers.keys())
        target = set(frozen)
        to_remove = current - target
        to_add = target - current

        # Remove proxies for labels no longer in the set.
        for lid in to_remove:
            layer = self._multi_label_proxy_layers.pop(lid, None)
            if layer is None:
                continue
            try:
                layer.visible = False
                self.viewer.layers.remove(layer)
            except (ValueError, KeyError):
                pass

        # Add a single-label proxy for each newly selected label.
        for lid in to_add:
            self._build_single_label_subproxy(main_layer, int(lid))

        # Hide main layer — the per-label proxies handle 3D rendering.
        # Cancel any pending perf re-show first (2026-09-02, see
        # _schedule_main_perf_reshow) so it cannot un-hide main under the
        # proxies on the next tick.  Inline token bump (== the helper) so
        # lightweight test doubles keep working.
        self._perf_reshow_token = getattr(self, "_perf_reshow_token", 0) + 1
        if main_layer.visible:
            main_layer.visible = False

        self._isolation_proxy_active = True
        self._isolation_proxy_label_id = None  # multi mode: no single ID
        self._multi_label_proxy_set = frozen

        # Suppress show_selected_label sync — subviewer label layers
        # use the propagated colormap, not show_selected_label, in
        # multi-label mode.
        sync = getattr(self, "sync_manager", None)
        if sync is not None:
            sync._suppress_show_selected_sync = True

    def _propagate_multi_label_colormap_to_subviewers(self, label_set) -> int:
        """Build a DirectLabelColormap for *label_set* and apply it to
        subviewer label layers only (NOT to main).

        The 3D main viewer uses N single-label sub-proxies for rendering
        (see ``_apply_multi_label_proxy``); subviewers continue to use a
        DirectLabelColormap because 2D rendering does not pay the
        ``iso_categorical`` cost and DirectLabelColormap correctly
        multi-colors the labels in the 2D slice.
        """
        import numpy as np
        try:
            from napari.utils.colormaps import DirectLabelColormap, label_colormap
        except ImportError:
            return 0

        hidden = np.array([0, 0, 0, 0], dtype=np.float32)
        if not hasattr(self, "_cached_label_colormap"):
            self._cached_label_colormap = label_colormap(49, seed=0.5)
        default_cmap = self._cached_label_colormap

        color_dict = {None: hidden, 0: hidden}
        if label_set:
            ids = np.array(sorted(int(x) for x in label_set), dtype=np.intp)
            colors = default_cmap.map(ids).astype(np.float32)
            for i, lid in enumerate(ids):
                color_dict[int(lid)] = colors[i]
        cmap = DirectLabelColormap(color_dict=color_dict)
        return self._propagate_colormap_to_subviewers(cmap)

    def _build_single_label_subproxy(self, main_layer, label_id: int) -> None:
        """Create one small proxy Labels layer for *label_id*.

        Mirrors the bbox-and-crop pattern of ``_update_isolation_proxy``
        but stores the result in ``self._multi_label_proxy_layers[label_id]``
        instead of the single ``_isolation_proxy_layer`` slot. Each
        sub-proxy uses ``show_selected_label=True`` for the proven 3D
        render path.
        """
        import numpy as np
        from napari.layers import Labels

        mask = main_layer.data == label_id
        z_any = np.any(mask, axis=(1, 2))
        if not z_any.any():
            return
        y_any = np.any(mask, axis=(0, 2))
        x_any = np.any(mask, axis=(0, 1))
        z_idx = np.where(z_any)[0]
        y_idx = np.where(y_any)[0]
        x_idx = np.where(x_any)[0]

        margin = 5
        shape = main_layer.data.shape
        slices = (
            slice(max(0, int(z_idx[0]) - margin),
                  min(shape[0], int(z_idx[-1]) + margin + 1)),
            slice(max(0, int(y_idx[0]) - margin),
                  min(shape[1], int(y_idx[-1]) + margin + 1)),
            slice(max(0, int(x_idx[0]) - margin),
                  min(shape[2], int(x_idx[-1]) + margin + 1)),
        )
        roi_data = main_layer.data[slices].copy()
        translate = [float(s.start) for s in slices]
        scale = list(main_layer.scale)

        proxy = Labels(
            roi_data,
            name=f"_epicycle_isolation_proxy_{label_id}",
            translate=translate,
            scale=scale,
        )
        proxy.editable = False
        proxy.selected_label = label_id
        proxy.show_selected_label = True
        # brush_size locked at 1.  Same rationale
        # as the single-label proxy in _update_isolation_proxy.
        from packages.epicycle_annotator.sync import lock_brush_size_to_1
        lock_brush_size_to_1(proxy)
        proxy.opacity = main_layer.opacity
        try:
            proxy.rendering = main_layer.rendering
        except Exception:
            pass
        try:
            proxy.iso_gradient_mode = main_layer.iso_gradient_mode
        except Exception:
            pass
        self.viewer.layers.append(proxy)
        self._multi_label_proxy_layers[label_id] = proxy

    def _propagate_colormap_to_subviewers(self, colormap) -> int:
        """Explicitly push a colormap to all subviewer label layers.

        Called by the annotator after setting isolation colormaps.
        Ensures subviewers get the colormap immediately without
        waiting for the sync.py event cycle.
        """
        from napari.layers import Labels

        labels_name = self._active_labels_name
        if not labels_name:
            return 0
        sync = getattr(self, "sync_manager", None)
        if sync is None:
            return 0
        base = sync._base_label_name(labels_name)
        matched = 0
        for vm_name in ('viewer_model_xy', 'viewer_model_xz', 'viewer_model_yz'):
            vm = getattr(sync, vm_name)
            for layer in vm.layers:
                if isinstance(layer, Labels) and sync._base_label_name(layer.name) == base:
                    layer.colormap = colormap
                    matched += 1
                    try:
                        from napari_epicycle.fusion_reconstruction._render_diag import render_state as _rs
                        _qtv = {'viewer_model_xy': 'qt_viewer_xy', 'viewer_model_xz': 'qt_viewer_xz', 'viewer_model_yz': 'qt_viewer_yz'}
                        _rs('sub-after-set-cmap', layer, qt_viewer=getattr(self, _qtv.get(vm_name, ''), None))
                    except Exception:
                        pass
        return matched

    def _propagate_show_selected_label(
        self, show: bool, *, label_id: int | None = None
    ) -> int:
        """Push show_selected_label (and optionally selected_label) to subviewers.

        Used for single-label isolation fast path where the annotator
        uses napari's built-in show_selected_label instead of a
        DirectLabelColormap.

        When *label_id* is provided, ``selected_label`` is also set on
        each subviewer layer so the correct label is displayed when
        ``show_selected_label=True``.
        """
        from napari.layers import Labels

        labels_name = self._active_labels_name
        if not labels_name:
            return 0
        sync = getattr(self, "sync_manager", None)
        if sync is None:
            return 0
        base = sync._base_label_name(labels_name)
        matched = 0
        for vm in [sync.viewer_model_xy, sync.viewer_model_xz, sync.viewer_model_yz]:
            for layer in vm.layers:
                if isinstance(layer, Labels) and sync._base_label_name(layer.name) == base:
                    # Set show_selected_label BEFORE selected_label.
                    # Setting selected_label fires sync events that can
                    # cascade; the show_selected_label must already be
                    # correct by that point so sync reads the intended value.
                    # Always assign (no redundancy skip) to ensure napari
                    # fires the property-change event and re-renders.
                    if hasattr(layer, "show_selected_label"):
                        layer.show_selected_label = show
                        matched += 1
                    if label_id is not None:
                        layer.selected_label = label_id
                    # When enabling isolation (show=True), ensure the
                    # subviewer label layer is visible.  sync_layer_visibility
                    # propagates main_layer.visible=False to subviewers when
                    # the 3D proxy hides the main layer — restore here.
                    if show:
                        layer.visible = True
                        _heal_render(self, None, tag="show-selected-sub")
                    # Force visual update only when enabling isolation —
                    # embedded ViewerModel canvases don't always repaint
                    # from evented property changes.  Skip refresh when
                    # clearing (show=False) to avoid 9 unnecessary canvas
                    # repaints; the subsequent colormap restore handles it.
                    if show:
                        try:
                            layer.refresh()
                        except Exception:
                            pass
        return matched

    # ---------------------------------------------------------------
    # Right-click → navigator-focus (subviewer-only feature)
    # ---------------------------------------------------------------

    def _right_click_navigate_allowed(self, layer) -> bool:
        """Gate function for the subviewer right-click navigator-focus.

        Returns True iff a right-click on *layer* should fire
        ``nav.select_label``.  Silently no-ops if any guard fails — no
        status hints, no exceptions, no surprise behavior.

        Guards enumerated 2026-05-10 (see commit message of this feature
        for the full hydra trace).  In order:

        Lifecycle:
          - ``self._is_closing`` False
          - ``self._navigator_widget`` present
          - User-facing toggle on the navigator is checked

        Refinement-op (centralized via RefinementOpState):
          - ``self._refine_state.any_op_active`` False
            (covers interp_active, interp_refine_mode, split_active,
             split_previewed, split_fragment_refine, smooth_active —
             which is the umbrella for all 7 smooth-family ops:
             SDF Smooth, MorphGAC, MorphACWE, Dilate/Erode, Guided
             Grow, Fill Holes, Flood Fill)

        Op-running (separate from staged; in-flight compute):
          - ``self._interp_running`` False
          - ``self._split_running`` False
          - ``self._smooth_running`` False

        Dialog:
          - ``self._merge_dialog`` is None or hidden

        Layer mode:
          - ``layer.mode`` not in {"paint", "erase"}

        Position/data validity is checked by the caller (the click
        handler) since it needs the actual click coordinate.
        """
        if getattr(self, "_is_closing", False):
            return False
        nav = getattr(self, "_navigator_widget", None)
        if nav is None:
            return False
        if not getattr(nav, "is_right_click_select_enabled", lambda: True)():
            return False
        refine_state = getattr(self, "_refine_state", None)
        if refine_state is not None and refine_state.any_op_active:
            return False
        if (
            getattr(self, "_interp_running", False)
            or getattr(self, "_split_running", False)
            or getattr(self, "_smooth_running", False)
        ):
            return False
        merge_dialog = getattr(self, "_merge_dialog", None)
        if merge_dialog is not None:
            try:
                if merge_dialog.isVisible():
                    return False
            except Exception:
                # Treat unknown state defensively as "open".
                return False
        # NOTE 2026-05-10: do NOT gate on layer.mode == "paint".
        # Subviewer label layers are ALWAYS in paint mode in EpiCycle's
        # normal usage (subviewers.py:139 — mode="paint" set at layer
        # creation so drawing is always available).  Gating on paint
        # would block right-click 100% of the time on subviewers, which
        # is what we observed 2026-05-10 in the first attempt.
        #
        # Right-button click is SAFE in paint mode: napari's paint
        # behavior only triggers on LEFT-drag.  The event.button == 2
        # short-circuit at the top of the callback is the only check
        # needed for paint coexistence.
        return True

    def _make_subviewer_right_click_handler(self):
        """Create a closure-bound mouse callback for subviewer right-click.

        Used by ``SubviewerManager.add_labels_to_subviewers`` when wiring
        each new subviewer label layer.  The handler:

          1. Short-circuits on anything other than ``event.button == 2``
             (right button).
          2. Defers to ``_right_click_navigate_allowed`` for all guards.
          3. Reads ``label_id`` at the click position from the layer's
             data array (with try/except for out-of-bounds).
          4. Skips background (label_id == 0).
          5. Calls ``nav.select_label(label_id)`` — which scrolls the
             navigator to the row, sets the table selection, and
             dispatches the existing on_row_clicked path (so Isolate-
             if-on triggers naturally).

        Note: napari's paint callback fires on right-button press
        regardless of button (see
        ``external-pkgs/napari/src/napari/layers/labels/_labels_mouse_bindings.py``).
        Paint suppression is handled separately by
        ``packages.epicycle_annotator.subviewers.install_button_aware_paint``
        which patches ``_drag_modes[Mode.PAINT]`` per-layer at layer-
        creation time.  Removing paint_cb from
        ``mouse_drag_callbacks`` inside THIS handler would be too late
        — paint already runs before we get a chance to remove it.
        """
        def callback(layer, event):
            """Generator callback — click-vs-drag aware.

            napari's mouse_drag_callbacks use the generator pattern:
              - code BEFORE the first ``yield`` runs on press
              - each ``yield`` returns control to napari; the function
                resumes on the next mouse_move or mouse_release event
              - code AFTER the loop runs on release

            Right-click + DRAG in napari is the built-in viewer zoom
            binding.  We yield through the drag; if the cursor moved
            more than ``drag_threshold`` voxels between press and
            release, treat as zoom and silently no-op.  Otherwise
            fire navigator focus on release.

            [RC-WHY] diagnostic prints at every short-circuit path —
            user reported intermittent right-click failures 2026-05-10;
            these prints expose which guard fired so we can see
            whether the failure is label_id=0, position out of bounds,
            label not in navigator filter, drag-vs-click misclass, or
            something else.
            """
            btn = getattr(event, "button", None)
            if btn != 2:
                return
            start_pos = getattr(event, "position", None)
            if start_pos is None:
                print("[RC-WHY] PRESS skip: event.position is None", flush=True)
                return
            try:
                start_px = tuple(float(p) for p in start_pos)
            except (TypeError, ValueError) as _exc:
                print(
                    f"[RC-WHY] PRESS skip: start_pos float-cast failed "
                    f"start_pos={start_pos!r} exc={_exc!r}",
                    flush=True,
                )
                return

            print(
                f"[RC-WHY] PRESS layer={layer.name!r} mode={layer.mode!r} "
                f"start_px={tuple(round(p, 2) for p in start_px)} "
                f"layer.data.shape={layer.data.shape}",
                flush=True,
            )

            yield  # press handled, wait for next event

            while event.type != "mouse_release":
                yield

            end_pos = getattr(event, "position", None)
            if end_pos is None:
                print("[RC-WHY] RELEASE skip: event.position is None", flush=True)
                return
            try:
                end_px = tuple(float(p) for p in end_pos)
            except (TypeError, ValueError) as _exc:
                print(
                    f"[RC-WHY] RELEASE skip: end_pos float-cast failed "
                    f"end_pos={end_pos!r} exc={_exc!r}",
                    flush=True,
                )
                return
            drag_threshold = 3.0
            displacement_sq = sum(
                (a - b) ** 2 for a, b in zip(end_px, start_px)
            )
            displacement = displacement_sq ** 0.5
            print(
                f"[RC-WHY] RELEASE end_px={tuple(round(p, 2) for p in end_px)} "
                f"|d|={displacement:.2f}vox (threshold={drag_threshold}vox)",
                flush=True,
            )
            if displacement_sq > drag_threshold * drag_threshold:
                print(
                    f"[RC-WHY]   -> DRAG path, skipping navigator focus",
                    flush=True,
                )
                return

            if not self._right_click_navigate_allowed(layer):
                # Print which gate condition failed for diagnostic
                # purposes.
                reasons = []
                if getattr(self, "_is_closing", False):
                    reasons.append("annotator closing")
                nav_w = getattr(self, "_navigator_widget", None)
                if nav_w is None:
                    reasons.append("no navigator widget")
                elif not getattr(nav_w, "is_right_click_select_enabled",
                                  lambda: True)():
                    reasons.append("toggle off")
                rs = getattr(self, "_refine_state", None)
                if rs is not None and rs.any_op_active:
                    reasons.append(f"refine op active ({rs!r})")
                if getattr(self, "_interp_running", False):
                    reasons.append("interp running")
                if getattr(self, "_split_running", False):
                    reasons.append("split running")
                if getattr(self, "_smooth_running", False):
                    reasons.append("smooth running")
                merge_dialog = getattr(self, "_merge_dialog", None)
                if merge_dialog is not None:
                    try:
                        if merge_dialog.isVisible():
                            reasons.append("merge dialog visible")
                    except Exception:
                        reasons.append("merge dialog probe failed")
                print(
                    f"[RC-WHY]   -> GATE BLOCKED — reasons={reasons or ['unknown']}",
                    flush=True,
                )
                return

            try:
                rs = getattr(self, "_refine_state", None)
                ils = getattr(self, "_isolation_label_set", None)
                print(
                    f"[XCRASH-1] right-click ENTER layer={layer.name!r} "
                    f"refine_any={rs.any_op_active if rs else '?'} "
                    f"iso_label_set={set(ils) if ils else None}",
                    flush=True,
                )
                # Durable breadcrumb + liveness probe.  If the next thing in the
                # session log is a faulthandler native dump, one of THESE objects
                # marked DELETED is the use-after-free the crash dereferenced.
                _nav_probe = getattr(self, "_navigator_widget", None)
                _crumb(
                    f"RCLICK-ENTER layer={layer.name!r} "
                    f"refine_any={rs.any_op_active if rs else '?'} "
                    + _liveness(annotator=self, navigator=_nav_probe)
                )
            except Exception as _exc:
                print(f"[XCRASH-1] probe error: {_exc!r}", flush=True)

            try:
                idx = tuple(int(round(p)) for p in end_px)
                if len(idx) != layer.data.ndim:
                    print(
                        f"[RC-WHY]   -> NDIM MISMATCH idx_len={len(idx)} "
                        f"data_ndim={layer.data.ndim}",
                        flush=True,
                    )
                    return
                if any(i < 0 for i in idx):
                    print(
                        f"[RC-WHY]   -> NEGATIVE INDEX idx={idx}",
                        flush=True,
                    )
                    return
                if any(i >= s for i, s in zip(idx, layer.data.shape)):
                    print(
                        f"[RC-WHY]   -> OUT OF BOUNDS idx={idx} "
                        f"shape={layer.data.shape}",
                        flush=True,
                    )
                    return
                label_id = int(layer.data[idx])
            except (IndexError, ValueError, TypeError) as _exc:
                print(
                    f"[RC-WHY]   -> INDEX/CAST ERROR end_px={end_px!r} "
                    f"exc={_exc!r}",
                    flush=True,
                )
                return
            if label_id == 0:
                print(
                    f"[RC-WHY]   -> BACKGROUND CLICK at idx={idx} label_id=0 "
                    f"(user likely clicked between labels or on empty space)",
                    flush=True,
                )
                return
            nav = getattr(self, "_navigator_widget", None)
            if nav is None:
                print("[RC-WHY]   -> NO NAVIGATOR (late check)", flush=True)
                return
            # Check if the label is in the navigator's filter view.
            # If not, select_label silently no-ops and the user thinks
            # "right-click did nothing".  Predict it here.
            items = getattr(nav, "_items", None)
            if items is not None:
                in_items = any(it.get("label_id") == label_id for it in items)
                if not in_items:
                    print(
                        f"[RC-WHY]   -> LABEL_ID={label_id} NOT IN NAVIGATOR "
                        f"_items — likely no QC features for it, OR "
                        f"controller hasn't pushed it yet.  select_label "
                        f"will silently no-op.",
                        flush=True,
                    )
                else:
                    # Also check if it's in the FILTERED view (table
                    # might have fewer rows than _items due to tier
                    # filters).
                    table = getattr(nav, "_table", None)
                    if table is not None:
                        try:
                            from qtpy.QtCore import Qt
                            in_table = False
                            for row_idx in range(table.rowCount()):
                                item = table.item(row_idx, 1)  # _COL_ID=1
                                if item is None:
                                    continue
                                d = item.data(Qt.UserRole)
                                if d and d.get("label_id") == label_id:
                                    in_table = True
                                    break
                            if not in_table:
                                print(
                                    f"[RC-WHY]   -> LABEL_ID={label_id} in "
                                    f"_items BUT NOT IN FILTERED TABLE "
                                    f"— user has the tier filter off for "
                                    f"this label's tier.  select_label "
                                    f"will silently no-op.",
                                    flush=True,
                                )
                        except Exception as _exc:
                            print(
                                f"[RC-WHY]   table-membership probe failed: "
                                f"{_exc!r}",
                                flush=True,
                            )
            print(
                f"[RC-WHY]   -> CALLING nav.select_label({label_id})",
                flush=True,
            )
            # 2026-09-07 crash fix — DEFER out of vispy's mouse-release
            # dispatch.  ``nav.select_label`` runs the whole navigator
            # focus path synchronously: table selectRow (which re-enters
            # the selection-model lambda), then _on_row_clicked ->
            # navigate + isolation -> _update_isolation_proxy (proxy
            # texture re-upload, main_layer.visible flip, a deferred
            # re-show timer).  Doing that INSIDE
            # vispy ``mouseReleaseEvent`` -> napari
            # ``mouse_release_callbacks`` mutates Qt/GL state while the
            # canvas is still inside its own event dispatch, and the
            # 2026-09-07 session crashed there with a Windows access
            # violation (crash log: <lambda> <- select_label:1433 <-
            # this callback <- mouse_release_callbacks).
            #
            # ``QTimer.singleShot(0, ...)`` runs the identical work one
            # event-loop turn later, with the vispy/Qt dispatch fully
            # unwound.  Nothing after this point depends on select_label
            # having completed (the callback returns immediately after),
            # so the deferral is behaviour-preserving.
            #
            # A monotonically increasing token coalesces a burst of rapid
            # right-clicks to the LAST label — without it a queue of
            # stale selects would each run the full isolate/render path.
            self._rclick_select_token = (
                getattr(self, "_rclick_select_token", 0) + 1
            )
            _token = self._rclick_select_token
            _crumb(
                f"select_label DEFERRED label={label_id} token={_token} "
                + _liveness(annotator=self, navigator=nav)
            )

            def _fire_select(label_id=label_id, token=_token, layer=layer):
                if not self._rclick_select_alive(token, label_id):
                    return
                nav_now = getattr(self, "_navigator_widget", None)
                # Re-check the CLICK-TIME gate at FIRE time.  A patch switch
                # does NOT delete the annotator (`_post_switch_annotator_sync`
                # keeps it) — it resets the navigator and rebuilds it with the
                # NEW patch's rows, so a deferred select could match the same
                # label_id in a different patch and navigate + isolate the
                # wrong cell.  Refinement / batch-dialog / toggle-off state can
                # also change between click and fire.  Both re-checks reuse
                # existing predicates rather than inventing a parallel path.
                if not self._right_click_navigate_allowed(layer):
                    _crumb(
                        f"select_label skipped-gate label={label_id} "
                        f"token={token}"
                    )
                    return
                # Mirrors `select_labels`'s own `_rebuild_pending` drop and the
                # deferred-lasso patch-switch gate at controller.py:14092: a
                # coalesced rebuild is pending, so the table is about to be
                # torn down + repopulated and this selection is stale.
                if getattr(nav_now, "_rebuild_pending", False):
                    _crumb(
                        f"select_label skipped-rebuild-pending "
                        f"label={label_id} token={token}"
                    )
                    return
                # Durable breadcrumb bracketing the render-triggering call:
                # if the session log shows BEFORE but no AFTER/RAISED, the
                # crash is inside select_label's isolate/colormap/vispy-
                # render path — a C-level access violation that Python
                # try/except cannot catch.
                _crumb(
                    f"select_label BEFORE label={label_id} token={token} "
                    "fired "
                    + _liveness(annotator=self, navigator=nav_now)
                )
                try:
                    nav_now.select_label(label_id)
                except Exception as _exc:
                    print(
                        f"[RC-WHY]   nav.select_label raised: {_exc!r}",
                        flush=True,
                    )
                    _crumb(
                        f"select_label RAISED label={label_id} exc={_exc!r}"
                    )
                else:
                    _crumb(f"select_label AFTER-OK label={label_id}")

            _defer_to_event_loop(_fire_select)
        return callback

    def _rclick_select_alive(self, token: int, label_id: int) -> bool:
        """Liveness + staleness gate for the deferred right-click select.

        Returns True only when this deferred select is still the newest
        one (``token`` matches) AND both the annotator and its navigator
        are still live Qt objects.  Emits a breadcrumb for every rejected
        reason so a session log shows exactly why a right-click produced
        no selection.
        """
        try:
            from qtpy import sip as _sip
        except Exception:  # pragma: no cover — non-sip binding
            _sip = None

        def _dead(obj) -> bool:
            if obj is None:
                return True
            if _sip is None:
                return False
            try:
                return bool(_sip.isdeleted(obj))
            except (TypeError, RuntimeError):
                return False
            except Exception:  # pragma: no cover
                return False

        if token != getattr(self, "_rclick_select_token", token):
            _crumb(
                f"select_label skipped-stale label={label_id} token={token} "
                f"newest={getattr(self, '_rclick_select_token', None)}"
            )
            return False
        if getattr(self, "_is_closing", False) or _dead(self):
            _crumb(
                f"select_label skipped-dead label={label_id} token={token} "
                "reason=annotator"
            )
            return False
        nav = getattr(self, "_navigator_widget", None)
        if _dead(nav) or getattr(nav, "_destroyed", False):
            _crumb(
                f"select_label skipped-dead label={label_id} token={token} "
                "reason=navigator"
            )
            return False
        return True

    def _on_new_label_from_navigator(self) -> None:
        """Handle Navigator's "New Label" button.

        Bumps ``selected_label`` to the next unused ID across all
        viewers (main + 3 subviewers).  The label badge updates via
        the layer's ``events.selected_label`` subscription; the
        navigator row registration fires when the user actually
        paints a stroke (UPSERT semantics — no row is created for
        an unused label_id).

        Uses ``next_unused_label_id`` (always scans the volume across
        ALL Labels layers in the viewer) — NOT
        ``_next_label_id_after_accept`` which falls back to
        ``max(navigator items) + 1`` and can return an ID that
        already exists when the navigator shows a filtered view of
        the labels (bug observed 2026-05-16: New Label produced an
        in-use ID, painted stroke joined an existing cell).  See
        ``packages/epicycle_annotator/label_ids.py`` for the
        rationale and ``tests/annotator/test_label_ids.py::
        TestNewLabelButtonBugCase`` for the regression guard.

        Subviewers are already in PAINT mode by default per
        ``add_labels_to_subviewers`` — no mode change needed.
        """
        try:
            from napari.layers import Labels
            from packages.epicycle_annotator.label_ids import (
                next_unused_label_id,
            )
            label_layers = [
                l for l in self.viewer.layers if isinstance(l, Labels)
            ]
            next_label = next_unused_label_id(label_layers)
            self._set_selected_label_all(next_label)
        except Exception as exc:
            try:
                print(
                    f"[NEW-LABEL] _on_new_label_from_navigator failed: "
                    f"{exc!r}",
                    flush=True,
                )
            except Exception:
                pass

    def _on_delete_label_from_navigator(self, label_id) -> None:
        """Handle Navigator's "Delete Label" button (Task A, 2026-06-01).

        Destructively zero the given label in the shared volume,
        following the split-accept canonical pattern verbatim so undo
        flows through the existing curation-undo system (the controller
        receives the ROI + old_roi_data via ``_on_delete_label_cb``
        and pushes a ``CurationUndoEntry`` keyed on the standard
        action label).

        Gating: silent no-op if any refinement op is staged or if an
        uncommitted paint stroke is in flight or if the merge dialog
        is open.  The Delete Label button is also visually disabled
        in those states (see ``_label_action_buttons_active``) — the
        in-handler gate is a defensive belt-and-braces check.

        Steps:
          1. Defensive type-cast on ``label_id``.
          2. Op-active gate (silent return).
          3. Resolve ``shared_data`` from sync_manager.
          4. Snapshot the label's ROI via ``_snapshot_label_roi``.
          5. Mutate in place: zero the label_id within the ROI.
          6. ``_sync_all_views_after_operation`` refreshes main +
             subviewers + isolation proxy.
          7. Fire ``_on_delete_label_cb(label_id, roi_slices,
             old_roi_data)`` so the controller commits the action
             to its curation-undo stack and marks the QC row as
             ``label_status="deleted"``.
          8. Remove the row from the navigator (mirrors
             ``_handle_manually_deleted_label``).
        """
        # 1. Defensive type-cast.
        try:
            lid = int(label_id)
        except (TypeError, ValueError):
            return
        if lid <= 0:
            return

        # 2. Op-active gate — silent no-op.  Belt + braces vs the
        #    button-disable gate; covers a programmatic caller that
        #    bypasses the button.
        if not self._label_action_buttons_active():
            print(
                f"[DELETE-LABEL] _on_delete_label_from_navigator BLOCKED "
                f"lid={lid} "
                f"refine_op={self._refinement_op_active()} "
                f"draw_snapshot="
                f"{getattr(self, '_draw_snapshot', None) is not None} "
                f"merge_dialog="
                f"{getattr(self, '_merge_dialog', None) is not None}"
            )
            return

        # 3. Resolve shared volume.
        labels_name = self._active_labels_name
        if not labels_name:
            return
        shared_data = self.sync_manager.shared_labels.get(labels_name)
        if shared_data is None:
            return

        # Verify label exists before snapshot — otherwise the snapshot
        # returns the full volume (fallback path) which would mean
        # the undo entry holds a copy of the entire labels array.
        if not (shared_data == lid).any():
            print(
                f"[DELETE-LABEL] _on_delete_label_from_navigator: "
                f"lid={lid} not found in volume — no-op"
            )
            return

        # 4. Snapshot bbox of the label (split-accept canonical step).
        roi_slices, old_roi_data = self._snapshot_label_roi(
            shared_data, lid
        )

        # 5. Mutate shared volume in place — only the target label
        #    within the ROI.  The mask-on-view approach avoids a full-
        #    volume scan; the snapshot above guarantees lid is present
        #    inside roi_slices so this is sufficient.
        roi_view = shared_data[roi_slices]
        roi_view[roi_view == lid] = 0

        print(
            f"[DELETE-LABEL] _on_delete_label_from_navigator: "
            f"lid={lid} roi={roi_slices} snapshot_shape={old_roi_data.shape}"
        )

        # 6. Sync all views (main + subviewers + isolation proxy).
        self._sync_all_views_after_operation(roi=roi_slices, label_id=lid)

        # 7. Fire callback to controller → commits to undo stack.
        if self._on_delete_label_cb:
            try:
                self._on_delete_label_cb(
                    lid,
                    roi_slices=roi_slices,
                    old_roi_data=old_roi_data,
                )
            except Exception as exc:
                print(
                    f"[DELETE-LABEL] _on_delete_label_cb raised: {exc!r}"
                )

        # 8. Remove the row from the navigator UI (mirrors
        #    `_handle_manually_deleted_label`).
        nav = getattr(self, "_navigator_widget", None)
        if nav is not None:
            try:
                nav.remove_item(lid)
            except Exception:
                pass

    def _on_paint_isolation_toggled(self, enabled: bool) -> None:
        """Handle Navigator's "Isolate while painting" checkbox
        (companion, 2026-05-16).

        ON: fire isolation immediately on the current ``selected_label``
        if it's a real label (>0).  Future ``selected_label`` changes
        (via paint auto-increment, New Label button, navigator row
        clicks, key presses) flow through ``_on_paint_label_changed``
        below, which re-fires isolation on each change.

        OFF: clear paint-driven isolation if active.  Does NOT touch
        navigator-driven isolation (`_isolation_active` set by the
        regular Isolate checkbox) — that path manages its own state.

        Race-safety: ``_paint_isolation_label_id`` deduplicates same-
        label re-fires to avoid the ~245ms cost when napari emits
        ``events.selected_label`` for an identical value (which can
        happen as a side-effect of property assignments downstream
        of sync code).
        """
        new_state = bool(enabled)
        if new_state == self._paint_isolation_enabled:
            return  # no-op
        self._paint_isolation_enabled = new_state

        if new_state:
            # Toggle ON: isolate the brush's current label if real.
            layer = self._layer_by_name(
                self.viewer.layers, self._active_labels_name
            )
            if layer is None:
                return
            try:
                current = int(getattr(layer, "selected_label", 0) or 0)
            except (AttributeError, TypeError, ValueError):
                return
            if current <= 0:
                # Background — wait for the user to pick a real label
                # via New Label, navigator click, or a paint stroke.
                return
            try:
                self._on_isolation_changed({current})
                self._paint_isolation_active = True
                self._paint_isolation_label_id = current
            except Exception as exc:
                print(
                    f"[PAINT-ISO] toggle-ON isolate failed: {exc!r}",
                    flush=True,
                )
        else:
            # Toggle OFF: clear paint-driven isolation if it was active.
            if not self._paint_isolation_active:
                return
            self._paint_isolation_active = False
            self._paint_isolation_label_id = None
            try:
                # Clear isolation entirely.  If the navigator-driven
                # isolation is also active, the navigator can re-fire
                # at any time; we don't try to restore prior state.
                self._on_isolation_changed(set())
            except Exception as exc:
                print(
                    f"[PAINT-ISO] toggle-OFF clear failed: {exc!r}",
                    flush=True,
                )

    def _on_paint_label_changed(self, label_id: int) -> None:
        """Forwarded from the navigator's
        ``layer.events.selected_label`` subscription (set up by
        ``CurationLabelNavigator.set_label_layer`` for the label
        badge).  Drives paint-isolation when the toggle is on.

        No-ops fast (most calls — the toggle is off the majority of
        the time) so it's safe to forward every brush change.  Dedupe
        on ``_paint_isolation_label_id`` avoids re-firing isolation
        for the same label.

        Background (label_id <= 0) is ignored — the brush "resting"
        at zero doesn't trigger an empty isolation.
        """
        if not self._paint_isolation_enabled:
            return
        try:
            lid = int(label_id)
        except (TypeError, ValueError):
            return
        if lid <= 0:
            return
        if lid == self._paint_isolation_label_id:
            return  # dedup — same label already isolated
        try:
            self._on_isolation_changed({lid})
            self._paint_isolation_active = True
            self._paint_isolation_label_id = lid
        except Exception as exc:
            print(
                f"[PAINT-ISO] event isolate({lid}) failed: {exc!r}",
                flush=True,
            )

    def _on_mark_tier_from_navigator(self, label_id, tier) -> None:
        """Forward navigator tier-flag click to the controller callback.

        Replaces the previous ``_on_mark_good_from_navigator`` and
        the flag-portion of ``_on_delete_from_navigator``.  The
        destructive volume-zeroing path (previously bundled into the
        "Delete Label" button) is no longer reachable from the
        navigator — destructive deletion happens via the separate
        Discard All Garbage button (``_on_discard_garbage_from_navigator``).
        """
        if self._on_mark_tier_cb:
            try:
                self._on_mark_tier_cb(label_id, tier)
            except Exception:
                pass

    def _on_merge_from_navigator(self, keep_id, absorb_id) -> None:
        """Legacy single-label merge entry. Kept for backward
        compatibility with any caller still wiring through the old
        ``on_merge=(keep_id, absorb_id)`` callback path. New code
        opens the floating MergeDialog instead — see
        ``_open_merge_dialog``.
        """
        if self._split_running:
            return
        lbl_name = self.labels_combo.currentText()
        roi_slices = None
        old_roi_data = None
        if lbl_name and lbl_name in self.viewer.layers:
            layer = self.viewer.layers[lbl_name]
            # Capture ROI of the absorb label before relabeling.
            roi_slices, old_roi_data = self._snapshot_label_roi(
                layer.data, absorb_id
            )
            # Relabel only within the ROI (avoids second full-volume scan).
            roi_view = layer.data[roi_slices]
            roi_view[roi_view == absorb_id] = keep_id
            layer.refresh()
        try:
            self._refresh_label_data_in_subviewers()
        except Exception:
            pass
        if self._on_merge_cb:
            try:
                # Adapt to the new batch signature: pass a single-element
                # list so the controller's updated callback works for
                # both code paths.
                self._on_merge_cb(
                    keep_id, [absorb_id],
                    roi_slices=roi_slices,
                    old_roi_data=old_roi_data,
                )
            except Exception:
                pass

    # ---------------------------------------------------------------
    # Merge Dialog (Phase 1.5 of the merge tool — floating dialog
    # with multi-label selection, neighbor discovery, batch merge).
    # ---------------------------------------------------------------

    def _open_merge_dialog(self, keep_id: int) -> None:
        """Launch the floating MergeDialog for *keep_id*.

        Wired to the navigator's ``on_open_merge_dialog`` callback.
        Called when the user clicks "Merge Into..." with a label
        selected — the selected label is the keep target, dialog
        surfaces neighbors via graduated dilation.
        """
        # Cross-op gate: refresh the centralized state and refuse if
        # any refinement op is currently staged.
        self._sync_refine_state()
        if not self._refine_state.can_open_merge_dialog():
            return

        if self._split_running:
            return

        lbl_name = self.labels_combo.currentText()
        if not lbl_name or lbl_name not in self.viewer.layers:
            return
        layer = self.viewer.layers[lbl_name]
        volume = layer.data

        # Discover neighbors (graduated dilation, single-label, ROI-scoped).
        from napari_epicycle.fusion_reconstruction.neighbor_query import (
            get_neighbor_ids,
        )
        neighbors = get_neighbor_ids(volume, keep_id, max_radius=3)

        # Pull metadata from Navigator items so the dialog can display
        # area / Z / tier alongside each candidate.
        neighbor_ids = [int(n["label_id"]) for n in neighbors]
        label_metadata = self._get_label_metadata_for_ids(neighbor_ids)

        # Tear down any previous dialog before opening a new one.
        if getattr(self, "_merge_dialog", None) is not None:
            try:
                self._merge_dialog.close()
            except Exception:
                pass
            self._merge_dialog = None

        from napari_epicycle.annotator.merge_dialog import MergeDialog

        self._merge_dialog = MergeDialog(
            keep_id, neighbors, label_metadata,
            on_merge=self._execute_batch_merge,
            on_accept=self._finalize_merge_accept,
            on_isolation_changed=self._on_isolation_changed,
            on_cancel=self._cancel_merge_dialog,
            main_label_layer=layer,
            parent=self,
        )
        self._merge_dialog.show()
        self._merge_dialog.raise_()

        # New/Delete Label buttons must disable while the merge dialog
        # is open — creating or destroying a label mid-merge-pick could
        # invalidate the keep_id the dialog is operating on.
        self._refresh_merge_button_state()
        self._refresh_label_action_buttons_state()
        # Force-isolate the keep label as the dialog opens.
        try:
            self._on_isolation_changed({keep_id})
        except Exception:
            pass

    def _get_label_metadata_for_ids(self, label_ids: list) -> dict:
        """Build a {label_id: {area, tier, centroid_z}} dict for the dialog
        from the Navigator's current item list. Missing labels get empty
        metadata; the dialog renders "—" for those cells."""
        out: dict = {int(lid): {} for lid in label_ids}
        nav = getattr(self, "_navigator_widget", None)
        if nav is None or not getattr(nav, "_items", None):
            return out
        wanted = set(out.keys())
        for it in nav._items:
            lid = int(it.get("label_id", -1))
            if lid in wanted:
                centroid = it.get("centroid")
                meta = {
                    "area": it.get("area"),
                    "tier": it.get("tier"),
                    "centroid_z": (centroid[0] if centroid else None),
                }
                out[lid] = meta
        return out

    def _execute_batch_merge(self, keep_id: int, absorb_ids: list) -> tuple:
        """Relabel each absorb_id → keep_id within the union ROI.

        Called by the dialog when the user clicks [Merge Selected].
        Returns ``(roi_slices, old_roi_data)`` — the dialog stores
        this for rollback if the user later clicks [Cancel].
        """
        lbl_name = self.labels_combo.currentText()
        if not lbl_name or lbl_name not in self.viewer.layers:
            return (None, None)
        layer = self.viewer.layers[lbl_name]

        # Defensive shared_data alignment — see _ensure_shared_data_aligned.
        # Without this, the in-place mutation below modifies layer.data but
        # leaves sync_manager.shared_labels[name] pointing at the old array;
        # the next refinement op (split/smooth/interp) would then see the
        # pre-merge state. Aligning here guarantees the mutation propagates
        # because layer.data and shared_data become the same numpy array.
        self._ensure_shared_data_aligned(lbl_name)

        import numpy as np

        # Combined snapshot ROI = union bounding box of all absorbed labels.
        combined_mask = np.isin(layer.data, absorb_ids)
        # Reuse the existing _snapshot_label_roi helper via the
        # _precomputed_mask shortcut so we don't duplicate bbox math.
        roi_slices, old_roi_data = self._snapshot_label_roi(
            layer.data, absorb_ids[0] if absorb_ids else 0,
            _precomputed_mask=combined_mask,
        )

        # Relabel within the ROI. np.isin is faster than per-id loop on
        # large ROIs; per-id loop is fine for small ones — use isin.
        roi_view = layer.data[roi_slices]
        absorb_mask_roi = np.isin(roi_view, absorb_ids)
        roi_view[absorb_mask_roi] = int(keep_id)
        # [XCRASH-7] post-merge state — does cache get invalidated?
        print(
            f"[XCRASH-7] post-merge keep={keep_id} absorbed={absorb_ids} "
            f"iso_label_set_BEFORE_clear={set(self._isolation_label_set) if self._isolation_label_set else None} "
            f"proxy_active={self._isolation_proxy_active} "
            f"proxy_lid={self._isolation_proxy_label_id}",
            flush=True,
        )

        layer.refresh()
        try:
            self._refresh_label_data_in_subviewers()
        except Exception:
            pass

        # Post-mutation: the multi-label sub-proxies for absorb_ids
        # are stale, AND the keep sub-proxy is also stale — its bbox
        # was sized for the pre-merge keep voxels, but the relabel
        # moved absorbed voxels into keep_id, so keep_id's true
        # bbox now extends to cover the merge ROI as well.
        #
        # The naive `_apply_multi_label_proxy(..., {keep_id})` doesn't
        # fix this: its set-diff sees keep_id already present in
        # _multi_label_proxy_layers and doesn't rebuild it. Result:
        # absorb cells visually disappear (proxies torn down), keep
        # cell looks unchanged (proxy didn't grow) — appears as a
        # deletion in the 3D viewer.
        #
        # Fix: tear down absorb sub-proxies, then resize the keep
        # sub-proxy's data + translate to bbox-union(existing,
        # merge_ROI). Pure integer math + one small crop — no
        # full-volume scan, so this stays cheap. Falls back to the
        # naive _apply_multi_label_proxy on any error.
        #
        # No-op when no proxies are active (e.g. 2D mode — the
        # DirectLabelColormap on the main layer already renders the
        # post-mutation state correctly because the absorb_ids no
        # longer exist as separate voxels and keep_id's colormap
        # entry still colors what's now there).
        if (
            getattr(self, "_isolation_proxy_active", False)
            and roi_slices is not None
        ):
            try:
                keep_int = int(keep_id)
                absorb_set = {int(a) for a in absorb_ids}
                proxy_layers = getattr(
                    self, "_multi_label_proxy_layers", None
                ) or {}

                # 1. Remove sub-proxies for the absorbed labels.
                for absorb_id in absorb_set:
                    lyr = proxy_layers.pop(absorb_id, None)
                    if lyr is None:
                        continue
                    try:
                        lyr.visible = False
                        self.viewer.layers.remove(lyr)
                    except (ValueError, KeyError):
                        pass

                # 2. Resize the keep sub-proxy's bbox to include the
                #    merge ROI. New bbox = union(old proxy bbox, ROI).
                keep_proxy = proxy_layers.get(keep_int)
                if keep_proxy is not None:
                    old_t = [float(x) for x in keep_proxy.translate]
                    old_shape = keep_proxy.data.shape
                    shape = layer.data.shape
                    new_starts = (
                        min(int(old_t[0]), roi_slices[0].start),
                        min(int(old_t[1]), roi_slices[1].start),
                        min(int(old_t[2]), roi_slices[2].start),
                    )
                    new_stops = (
                        max(int(old_t[0]) + old_shape[0], roi_slices[0].stop),
                        max(int(old_t[1]) + old_shape[1], roi_slices[1].stop),
                        max(int(old_t[2]) + old_shape[2], roi_slices[2].stop),
                    )
                    new_slices = (
                        slice(max(0, new_starts[0]),
                              min(shape[0], new_stops[0])),
                        slice(max(0, new_starts[1]),
                              min(shape[1], new_stops[1])),
                        slice(max(0, new_starts[2]),
                              min(shape[2], new_stops[2])),
                    )
                    new_data = layer.data[new_slices].copy()
                    new_translate = [float(s.start) for s in new_slices]
                    keep_proxy.data = new_data
                    keep_proxy.translate = new_translate
                    # Re-assert show_selected_label so the renderer only
                    # shows keep_id within the (now larger) crop.
                    keep_proxy.selected_label = keep_int
                    keep_proxy.show_selected_label = True

                # 3. Update bookkeeping — only keep_id remains in the set.
                self._multi_label_proxy_set = frozenset({keep_int})

                # 4. Propagate the post-mutation colormap to subviewers
                #    so DirectLabelColormap entries for the now-dead
                #    absorb_ids are cleared.
                self._propagate_multi_label_colormap_to_subviewers(
                    {keep_int}
                )
            except Exception:
                # Fallback: full rebuild path. Slower (single-label
                # scan happens via _build_single_label_subproxy on
                # next isolation change), but guaranteed correct.
                try:
                    self._apply_multi_label_proxy(layer, {int(keep_id)})
                    self._propagate_multi_label_colormap_to_subviewers(
                        {int(keep_id)}
                    )
                except Exception:
                    pass

        return (roi_slices, old_roi_data)

    def _finalize_merge_accept(self, keep_id: int, absorb_ids: list) -> None:
        """Run the post-Accept commit chain: controller callback for
        DataFrame updates, Navigator row removal, optional auto-advance.

        Called by the dialog when the user clicks [Accept]."""
        # B2: refuse while load-locked (e.g. async Save in flight) — same as the
        # split/smooth/interp Accepts.  The dialog stays in its 'merged' stage, so
        # the user re-clicks Accept once the save unlocks.  (The dirty-epoch
        # backstop also covers this path; refusing keeps behaviour consistent with
        # the other Accepts.)
        if getattr(self, "_load_locked", False):
            print("[REFINE-ACCEPT] merge accept refused — controller busy (load-locked)")
            return
        snapshot = (None, None)
        if (
            getattr(self, "_merge_dialog", None) is not None
            and self._merge_dialog.state.snapshot is not None
        ):
            snapshot = self._merge_dialog.state.snapshot

        # Controller callback: updates DataFrame (label_status,
        # qc_tier, area/centroid, border-flag inheritance) and
        # commits to the undo stack.
        if self._on_merge_cb is not None:
            try:
                self._on_merge_cb(
                    keep_id, list(absorb_ids),
                    roi_slices=snapshot[0],
                    old_roi_data=snapshot[1],
                )
            except Exception:
                pass

        # Remove absorbed rows from the navigator (keep label row is
        # updated by the controller via re-emit; we just drop ghosts).
        nav = getattr(self, "_navigator_widget", None)
        if nav is not None:
            for absorb_id in absorb_ids:
                try:
                    nav.remove_item(int(absorb_id))
                except Exception:
                    pass

        # Keep the merged label isolated post-accept so the user can
        # immediately chain split/refine/smooth on it without re-clicking.
        # Single-label set ({keep_id}) routes through
        # _on_isolation_changed's fast path: single-label proxy +
        # show_selected_label=True — same as a normal navigator click
        # with isolate-on, so subsequent ops behave identically.
        # ANNOTATOR_ISOLATION ownership stays claimed across the
        # transition (idempotent claim).  Cancel still calls
        # _on_isolation_changed(set()) to fully release because Cancel
        # means "abandon the merge entirely".
        try:
            self._on_isolation_changed({int(keep_id)})
        except Exception:
            pass

        # Sync the navigator's Isolate checkbox to match the active
        # isolation state.  The merge dialog force-isolates on open
        # regardless of the navigator's prior cb state — so if it was
        # unchecked, the cb and the actual isolation are now out of
        # sync.  set_isolate_checked uses blockSignals so this does
        # not re-trigger _on_isolate_toggled (which would re-fire
        # isolation and double-dispatch).
        nav = getattr(self, "_navigator_widget", None)
        if nav is not None and hasattr(nav, "set_isolate_checked"):
            try:
                nav.set_isolate_checked(True)
            except Exception:
                pass

        # Clear the dialog reference.
        self._merge_dialog = None
        # Merge-dialog gate just lifted — re-enable New/Delete Label
        # buttons if no other gate is up.
        self._refresh_merge_button_state()
        self._refresh_label_action_buttons_state()
    def _cancel_merge_dialog(self) -> None:
        """Restore the volume from the dialog's snapshot if a merge
        was performed, then drop the dialog reference. No DataFrame
        change ever occurred (controller callback only fires on
        Accept)."""
        dlg = getattr(self, "_merge_dialog", None)
        if dlg is None:
            print("[MERGE-CLOSE] _cancel_merge_dialog: no dialog, no-op")
            return

        stage = getattr(dlg.state, "stage", "?")
        had_snapshot = dlg.state.snapshot is not None
        print(
            f"[MERGE-CLOSE] _cancel_merge_dialog ENTER "
            f"stage={stage} had_snapshot={had_snapshot} "
            f"is_closing={getattr(self, '_is_closing', False)}"
        )

        # If we never reached "merged" stage, nothing to restore.
        if dlg.state.stage == "merged" and dlg.state.snapshot is not None:
            roi_slices, old_roi_data = dlg.state.snapshot
            lbl_name = self.labels_combo.currentText()
            if lbl_name and lbl_name in self.viewer.layers and roi_slices is not None:
                layer = self.viewer.layers[lbl_name]
                # B3 belt-and-suspenders (2026-06-10): only restore if the
                # snapshot still matches the active layer's ROI shape.  A
                # mismatch means the active layer changed under the dialog
                # (e.g. a stale dialog survived a patch switch) — skip the
                # write so we never broadcast one patch's data into another.
                _target = layer.data[roi_slices]
                if getattr(_target, "shape", None) == getattr(old_roi_data, "shape", None):
                    layer.data[roi_slices] = old_roi_data
                    layer.refresh()
                    try:
                        self._refresh_label_data_in_subviewers()
                    except Exception:
                        pass
                else:
                    print(
                        "[MERGE-CLOSE] snapshot/layer ROI shape mismatch "
                        f"({getattr(old_roi_data, 'shape', None)} vs "
                        f"{getattr(_target, 'shape', None)}) — skipping restore"
                    )

        # Clear isolation back to no-isolation (or restore prior — for
        # now just clear; isolate state was a transient dialog feature).
        try:
            print(
                "[MERGE-CLOSE] _cancel_merge_dialog firing "
                "_on_isolation_changed(set()) — clears multi-label proxies"
            )
            self._on_isolation_changed(set())
        except Exception as e:
            print(f"[MERGE-CLOSE] isolation clear raised: {e!r}")

        self._merge_dialog = None
        # Merge-dialog gate just lifted — re-enable New/Delete Label
        # buttons if no other gate is up.
        self._refresh_merge_button_state()
        self._refresh_label_action_buttons_state()
        print("[MERGE-CLOSE] _cancel_merge_dialog DONE — dialog ref cleared")

    # ------------------------------------------------------------------
    # Batch Refine — floating dialog + threaded worker (Step 3)
    # ------------------------------------------------------------------
    #
    # Lifecycle mirrors the merge dialog: open (gated by
    # can_start_batch_refine) -> build recipe -> Run/Preview -> worker
    # computes on a QThread over a *copy* of the volume -> on finish the
    # union-ROI is swapped into the live shared volume (preview, staged)
    # -> Accept (fire controller callback + curation-undo entry) or
    # Reject (restore pre-batch snapshot).  Nothing hits disk until the
    # user clicks Save Curation.
    #
    # State (all getattr-guarded, lazily initialised):
    #   _batch_refine_dialog : the BatchRefineDialog (or None)
    #   _batch_running       : worker in flight
    #   _batch_staged        : preview applied, awaiting Accept/Reject
    #   _batch_snapshot      : pre-batch voxels (full vol while running;
    #                          cropped to the union ROI once staged)
    #   _batch_union_roi     : tuple of slices over all changed voxels
    #   _batch_touched_ids   : labels whose voxels actually changed
    #   _batch_thread/_batch_worker : QThread + BatchRefineWorker

    def _open_batch_refine_from_navigator(self) -> None:
        """The navigator's Batch Refine button: open, or SAY why not.

        The refusal used to be a console line only;
        the navigator door now shows it in a one-button themed notice.
        """
        why = self._open_batch_refine_dialog()
        from packages.epicycle_annotator.batch_refine import (
            BATCH_ALREADY_OPEN,
        )

        # The open dialog was raised: it IS the answer, no notice on top
        # of it (D-L4).
        if why and why != BATCH_ALREADY_OPEN:
            from packages.epicycle_annotator.batch_refine import (
                BATCH_NOTICE_TITLE,
            )
            from napari_epicycle.ui_shell.theme import themed_notice

            try:
                themed_notice(self, BATCH_NOTICE_TITLE, why)
            except Exception as exc:  # noqa: BLE001 -- never raise into a click
                print(f"[BATCH] notice failed: {exc!r}")

    def _open_batch_refine_dialog(self, *, explicit_targets=None,
                                  default_target=None):
        """Open the floating Batch Refine dialog; ``None``, or the sentence
        that refused it (gate order: a stopping
        worker, the controller lock, a staged edit, no volume).

        ``explicit_targets`` (label ids) overrides the "Selected" target source
        (used by the Add tool's "Refine added…" so it targets the added pieces
        directly, independent of the navigator selection/dock).  ``default_target
        == "selected"`` pre-checks the Selected radio so the obvious Run flow
        hits those pieces.  Both default to the navigator's own behavior.
        """
        from packages.epicycle_annotator.batch_refine import (
            BATCH_ALREADY_OPEN,
            BATCH_BUSY,
            BATCH_NO_VOLUME,
            BATCH_NOT_CLEAN,
            BATCH_STILL_STOPPING,
        )

        # (0) The dialog is open and busy -- running, or holding a staged
        # result -- so another door (the navigator, Add's "Refine all")
        # raises it and changes NOTHING: not its targets, not its state
        # (UX audit H1).  Before (1): its own worker is not "stopping".
        existing = _safe_attr(self, "_batch_refine_dialog")
        if existing is not None and (
                _safe_attr(self, "_batch_running", False)
                or _safe_attr(self, "_batch_staged", False)):
            _raise_batch_dialog(existing)
            print("[BATCH] open refused — the dialog is already open "
                  "(running or staged); raised it")
            return BATCH_ALREADY_OPEN
        # (1) A worker that has not finished yet -- a run closed mid-flight
        # and still stopping -- names itself FIRST, before the lock it also
        # holds would say "busy" (invariant 14).
        if _safe_attr(self, "_batch_worker") is not None:
            print("[BATCH] open refused — the last batch is still stopping")
            return BATCH_STILL_STOPPING
        # Controller-lock gate (2026-06-07) — refuse to START a new batch refine
        # while the controller holds the unified Step-3 lock (predict /
        # reconstruct / extract / commit / save / patch-load / training / another
        # batch refine in flight).  The navigator's Batch Refine button is NOT
        # disabled by set_load_locked (which only gates New/Delete/Merge), so
        # this is the defense-in-depth gate.  _load_locked is set by
        # set_load_locked; absent on a standalone annotator → not load-locked.
        if getattr(self, "_load_locked", False):
            print("[BATCH] open refused — controller busy (load-locked)")
            return BATCH_BUSY
        # An IDLE dialog already open: the plain door (the navigator)
        # raises it as it is, choices kept (D-L4 -- it used to say "Accept
        # or Discard the change on screen first", which named nothing); a
        # door with its own targets (Add's "Refine all") replaces it, so
        # those targets are the ones on screen.
        existing = _safe_attr(self, "_batch_refine_dialog")
        if existing is not None:
            if explicit_targets is None and default_target is None:
                _raise_batch_dialog(existing)
                print("[BATCH] dialog already open (idle) — raised it")
                return None
            self._teardown_batch_dialog()
        # Cross-op gate — refuse if any op is staged / mid-paint / merge open.
        self._sync_refine_state()
        if not self._refine_state.can_start_batch_refine():
            print("[BATCH] open refused — not a clean state")
            return BATCH_NOT_CLEAN
        labels_name = self._active_labels_name
        shared = (
            self.sync_manager.shared_labels.get(labels_name)
            if labels_name
            else None
        )
        if shared is None:
            print("[BATCH] open refused — no shared label volume")
            return BATCH_NO_VOLUME

        # Every gate passed: only now does this door's target set replace
        # the last one (UX audit H1 -- a refused door used to wipe the
        # targets of the batch it was refused for).  Reset on every open
        # (None for the navigator button → no leak of a prior Add session's
        # explicit target set).
        self._batch_explicit_targets = (
            [int(i) for i in explicit_targets] if explicit_targets else None
        )

        nav = getattr(self, "_navigator_widget", None)
        # Tiers present in the navigator (fall back to the standard set).
        tiers_present = []
        for it in (getattr(nav, "_items", None) or []):
            t = it.get("tier")
            if t and t not in tiers_present:
                tiers_present.append(t)
        if not tiers_present:
            tiers_present = ["good", "review", "bad", "garbage", "curated"]

        # One find_objects pass per open (was np.unique here AND on every
        # count refresh): the present ids + their boxes are cached for the
        # dialog's life and every count / resolve reads the cache.  The
        # worker takes its own authoritative pass at Run.
        from packages.epicycle_annotator.batch_refine import label_bboxes

        self._batch_bboxes = label_bboxes(shared)
        self._batch_present_ids = frozenset(self._batch_bboxes)
        all_count = len(self._batch_present_ids)
        selected_count = (
            len(self._batch_explicit_targets)
            if self._batch_explicit_targets
            else (len(nav.selected_label_ids()) if nav is not None else 0)
        )

        from napari_epicycle.annotator.batch_refine_dialog import (
            BatchRefineDialog,
        )

        self._batch_running = False
        self._batch_staged = False
        self._batch_touched_ids = []
        self._batch_union_roi = None
        self._batch_snapshot = None

        self._batch_refine_dialog = BatchRefineDialog(
            count_provider=lambda spec: len(self._resolve_batch_targets(spec)),
            all_count=all_count,
            selected_count=selected_count,
            available_tiers=tiers_present,
            on_preview_one=self._on_batch_preview_one,
            on_run=self._on_batch_run,
            on_cancel=self._on_batch_cancel,
            on_accept=self._on_batch_accept,
            on_reject=self._on_batch_reject,
            on_close=self._on_batch_close,
            on_save_recipe=self._save_batch_recipe,
            can_save_recipe=getattr(self, "_save_batch_recipe_cb", None) is not None,
            on_chain_changed=_safe_attr(self, "_refresh_batch_chain_block"),
            added_targets=bool(self._batch_explicit_targets) or None,
            parent=self,
        )
        # Populate Recent (session) + Saved (persisted) recipe dropdowns.
        self._refresh_batch_recipe_history(self._batch_refine_dialog)
        # Sticky recipe (2026-06-07): pre-populate the recipe builder with the
        # last-run recipe so the user can iterate across patches without
        # rebuilding.  STEPS only — the target (all/selected/by-tier) is left at
        # the dialog default so the user re-picks per patch (labels differ).
        # Guarded: a malformed last-recipe must NOT break dialog-open — log +
        # open with an empty builder on any failure.
        last_recipe = getattr(self, "_batch_last_recipe", None)
        if last_recipe:
            try:
                from packages.epicycle_annotator.batch_refine import (
                    recipe_from_json,
                )

                self._batch_refine_dialog.set_recipe(
                    recipe_from_json(last_recipe)
                )
            except Exception as exc:  # noqa: BLE001 — never block dialog-open
                print(f"[BATCH] pre-load last recipe failed: {exc!r}")
        # Every radio's count, the plot's included (read live).
        try:
            self._refresh_batch_target_counts(self._batch_refine_dialog)
        except Exception as exc:  # noqa: BLE001 -- never block dialog-open
            print(f"[BATCH] target counts at open failed: {exc!r}")
        # A plain open with points highlighted in the QC plot starts on them
        # (UX audit M7, owner decision O4 -- reversible): the lasso is the
        # obvious reason to open Batch Refine.
        if default_target is None and explicit_targets is None:
            try:
                if int(getattr(self._batch_refine_dialog, "_plot_count",
                               0) or 0) > 0:
                    self._batch_refine_dialog._radio_plot.setChecked(True)
            except Exception:
                pass
        # Pre-check the requested target radio (fires _on_target_changed → count
        # refresh).  "selected" + explicit_targets = the Add tool's added pieces.
        if default_target == "selected":
            try:
                self._batch_refine_dialog._radio_selected.setChecked(True)
            except Exception:
                pass
        # The Refine-tab recipe row: the Refine file, READ
        # only; a plain open (the navigator: no explicit targets, no preset)
        # re-selects the last chain run this session.  An explicit-target or
        # preset open never does -- Run would otherwise execute the chain
        # while the status says the preset is ready.
        try:
            self._populate_batch_chain_rows(
                self._batch_refine_dialog,
                reselect=(explicit_targets is None and default_target is None))
        except Exception as exc:  # noqa: BLE001 -- never block dialog-open
            print(f"[BATCH] recipe row at open failed: {exc!r}")
        # Open at the width the dialog NEEDS (UX audit L3): with the
        # reason line up its size hint asked for 988 px; everything wraps
        # at the minimum (~726).
        try:
            d = self._batch_refine_dialog
            d.resize(d.minimumSizeHint().width(), d.sizeHint().height())
        except Exception:  # noqa: BLE001 -- a size is never worth a refusal
            pass
        self._batch_refine_dialog.show()
        self._batch_refine_dialog.raise_()
        # Dialog now present → batch_refine_active True (mutual exclusion).
        self._sync_refine_state()
        self._refresh_merge_button_state()
        self._refresh_label_action_buttons_state()
        _poll = _safe_attr(self, "_start_batch_poll")
        if _poll is not None:
            _poll()
        print("[BATCH] dialog opened")
        return None

    # ---- the Refine-tab recipe row ----

    def _populate_batch_chain_rows(self, dlg, *, reselect=False) -> None:
        """Fill the recipe combo (store READ only), the channel combo and the
        preview object; re-select the last chain on a plain open."""
        if dlg is None or not hasattr(dlg, "set_chain_recipes"):
            return
        store, problem = self._recipe_load_store()
        dlg.set_chain_recipes(store or {}, problem=problem,
                              no_workspace=store is None)
        last = _safe_attr(self, "_batch_last_chain_name")
        if reselect and last:
            if dlg.select_chain(last):
                print(f"[BATCH] re-selected the Refine recipe '{last}'")
        self._refresh_batch_chain_channels(first=True)
        self._push_batch_preview_object(force=True)
        self._refresh_batch_chain_block()

    def _batch_image_names(self) -> list:
        """The viewer's image layers a recipe may read (the Refine combo's
        own filter: temporary layers skipped)."""
        names = []
        for layer in self.viewer.layers:
            if getattr(layer, "_epicycle_temporary", False):
                continue
            if isinstance(layer, Image):
                ndim = layer.data.ndim
                if ndim in {2, 3}:
                    names.append(layer.name)
                elif ndim == 4 and int(layer.data.shape[-1]) in {3, 4}:
                    names.append(layer.name)
        return names

    def _refresh_batch_chain_channels(self, first=False) -> None:
        """Refill the Batch dialog's channel combo (layer insert / remove
        while it is open).  Pre-selects the Refine page's channel on the
        first fill only; blank stays blank after that."""
        dlg = _safe_attr(self, "_batch_refine_dialog")
        if dlg is None or not hasattr(dlg, "set_chain_channels"):
            return
        preferred = None
        if first:
            adv = self._get_advanced_features_widget()
            try:
                preferred = adv.refine_channel_name() if adv is not None else None
            except Exception:
                preferred = None
        dlg.set_chain_channels(self._batch_image_names(), preferred=preferred)

    def _batch_preview_id(self, *, check_volume=False) -> int:
        """The object "Preview" would run on, or 0 (D-L5).

        ``_get_selected_label`` falls back to 1 when nothing is selected,
        so an id that is not in the volume is "nothing selected": the
        dialog then says "Select one object…" instead of offering "Preview
        on object 1".  The 500 ms poll reads the open-time id cache only;
        a click (``check_volume``) also asks the volume about an id painted
        after the open.
        """
        try:
            sel = int(self._get_selected_label() or 0)
        except Exception:
            sel = 0
        if sel <= 0:
            return 0
        present = _safe_attr(self, "_batch_present_ids")
        if present is None or sel in present:
            return sel
        if check_volume:
            shared = self.sync_manager.shared_labels.get(
                self._active_labels_name)
            if shared is not None:
                from packages.epicycle_annotator.batch_refine import (
                    label_bboxes,
                )

                if int(sel) in label_bboxes(shared, [sel]):
                    return sel
        return 0

    def _push_batch_preview_object(self, force=False) -> None:
        """Tell the dialog which object "Preview" would run on (the id
        ``_get_selected_label`` uses, when the volume holds it), when it
        changed."""
        dlg = _safe_attr(self, "_batch_refine_dialog")
        if dlg is None or not hasattr(dlg, "set_preview_object"):
            return
        sel = Curation3DAnnotator._batch_preview_id(self)
        if force or sel != _safe_attr(self, "_batch_preview_pushed"):
            self._batch_preview_pushed = sel
            dlg.set_preview_object(obj_id=sel)

    def _batch_chain_inputs(self, recipe, channel_name):
        """``(channel_source, prob_volume, availability, shapes)`` read on the
        MAIN thread -- the array-likes uncopied."""
        from packages.epicycle_annotator import refine_recipe as rr

        needs_ch = any(rr.step_needs_channel(s.method, s.params)
                       for s in recipe.steps)
        needs_pr = any(rr.step_needs_prob(s.method, s.params)
                       for s in recipe.steps)
        ch = self._refine_channel_source(channel_name) if (
            needs_ch and channel_name) else None
        pr = self._ensure_prob_volume() if needs_pr else None
        av = {"spacing_known": bool(self._refine_spacing_known())}
        try:
            av["prob"] = bool(self._has_prob_volume())
        except Exception:
            av["prob"] = True
        av["guided_grow"] = av["prob"]
        return ch, pr, av

    def _batch_blockers(self, dlg, recipe, targets, *, for_preview=None):
        """``(run_block, preview_block)`` from ``refine_recipe.batch_blocker``,
        evaluated for Run (n = count, every target's box) and for Preview
        (n = 1, the preview object's box only).  Reads only
        cached boxes and shapes: nothing is loaded or copied here."""
        from packages.epicycle_annotator import refine_recipe as rr

        shared = self.sync_manager.shared_labels.get(self._active_labels_name)
        if shared is None:
            return None, None
        name = dlg.chain_channel() if hasattr(dlg, "chain_channel") else ""
        needs_ch = any(rr.step_needs_channel(s.method, s.params)
                       for s in recipe.steps)
        src = self._refine_channel_source(name) if (needs_ch and name) else None
        channel = name if (not needs_ch or src is not None) else ""
        prob_shape = None
        if any(rr.step_needs_prob(s.method, s.params) for s in recipe.steps):
            # The map's SHAPE, read from its header when it is not loaded
            # yet (D-M1): the cached-only read let a mis-sized map through
            # until after the run had staged misaligned objects.
            prob_shape = Curation3DAnnotator._prob_volume_shape(self)
        av = {"spacing_known": bool(self._refine_spacing_known())}
        try:
            av["prob"] = bool(self._has_prob_volume())
        except Exception:
            av["prob"] = True
        av["guided_grow"] = av["prob"]
        boxes = _safe_attr(self, "_batch_bboxes") or {}
        common = dict(
            channel=channel, availability=av, volume_shape=shared.shape,
            spacing=self._refine_spacing(), prob_shape=prob_shape,
            labels_shape=shared.shape,
            # The whole shape: a 2-D layer (offered by the combo) is
            # refused here, not cropped into an exception per object (D-M2).
            channel_shape=(tuple(int(n) for n in src.shape)
                           if src is not None else None),
        )
        run_b = rr.batch_blocker(
            recipe, n_objects=len(targets),
            bboxes=[boxes.get(int(i)) for i in targets], **common)
        pid = for_preview if for_preview is not None else int(
            _safe_attr(self, "_batch_preview_pushed") or 0)
        prev_b = rr.batch_blocker(
            recipe, n_objects=1,
            bboxes=[boxes.get(int(pid))] if pid else [], **common)
        return run_b, prev_b

    def _refresh_batch_chain_block(self) -> None:
        """Re-evaluate the batch blocker for Run and for Preview and put the
        reason on the dialog (every target / count / combo change)."""
        dlg = _safe_attr(self, "_batch_refine_dialog")
        if dlg is None or not hasattr(dlg, "set_chain_block"):
            return
        recipe = dlg.chain_recipe()
        if recipe is None:
            dlg.set_chain_block(run=None, preview=None)
            return
        try:
            targets = self._resolve_batch_targets(dlg.target_spec())
        except Exception:
            targets = []
        _t0 = _time.perf_counter()
        try:
            run_b, prev_b = self._batch_blockers(dlg, recipe, targets)
        except Exception as exc:  # noqa: BLE001 -- never raise into a combo
            print(f"[BATCH] blocker check failed: {exc!r}")
            return
        _batch_perf("blockers", _t0)
        dlg.set_chain_block(run=run_b.text if run_b else None,
                            preview=prev_b.text if prev_b else None)

    # ---- orphans + the dialog-life poll ----

    def _live_batch_orphans(self) -> int:
        orphans = _safe_attr(self, "_batch_orphans")
        if not orphans:
            return 0
        prune = getattr(orphans, "prune", None)
        if prune is not None:
            # `OrphanList`: the prune holds the lock the worker appends
            # under, so no future is lost in between (D-L1).
            return int(prune())
        orphans[:] = [f for f in orphans if not f.done()]
        return len(orphans)

    def _start_batch_poll(self) -> None:
        """A 500 ms timer while the dialog is open (and while stopped-chain
        work still finishes): prunes the orphans, and keeps the preview
        object current."""
        try:
            t = _safe_attr(self, "_batch_poll_timer")
            if t is None:
                from qtpy.QtCore import QTimer

                t = QTimer(self)
                t.setInterval(500)
                t.timeout.connect(self._on_batch_poll)
                self._batch_poll_timer = t
            if not t.isActive():
                t.start()
            self._on_batch_poll()
        except Exception as exc:  # noqa: BLE001 -- never block an open
            print(f"[BATCH] poll not started: {exc!r}")

    def _on_batch_poll(self) -> None:
        n = self._live_batch_orphans()
        dlg = _safe_attr(self, "_batch_refine_dialog")
        if dlg is not None:
            if hasattr(dlg, "set_orphans_running") and n != _safe_attr(
                    self, "_batch_orphans_pushed"):
                self._batch_orphans_pushed = n
                dlg.set_orphans_running(n=n)
            before = _safe_attr(self, "_batch_preview_pushed")
            self._push_batch_preview_object()
            if _safe_attr(self, "_batch_preview_pushed") != before:
                self._refresh_batch_chain_block()
        elif n == 0:
            t = _safe_attr(self, "_batch_poll_timer")
            if t is not None:
                t.stop()

    def _resolve_batch_targets(self, spec, *, present=None) -> list:
        """Resolve a dialog target spec to a sorted list of label ids.

        ``spec`` = ``{"mode": "all"|"selected"|"plot"|"tier", "tiers": [...]}``.
        Reads the present-id set cached at dialog open
        (``_batch_present_ids``, one ``find_objects`` pass) through the pure
        ``resolve_target_ids`` (present-intersection + sort), so a count
        refresh never re-scans the volume; with no cache (never opened) the
        set is taken from the volume here.  ``present`` overrides both --
        Run passes a fresh set so an object painted while the dialog was
        open is not dropped.  Tier membership comes from the navigator's
        item list (the QC tiers live on the controller's df, but the
        navigator already mirrors each label's tier).
        """
        from packages.epicycle_annotator.batch_refine import (
            label_bboxes,
            resolve_target_ids,
        )

        labels_name = self._active_labels_name
        shared = (
            self.sync_manager.shared_labels.get(labels_name)
            if labels_name
            else None
        )
        if shared is None:
            return []
        if present is None:
            present = getattr(self, "_batch_present_ids", None)
        if present is None:
            present = frozenset(label_bboxes(shared))
        mode = (spec or {}).get("mode", "all")
        nav = getattr(self, "_navigator_widget", None)
        if mode == "selected":
            # An explicit target set (e.g. the Add tool's "Refine added…") wins
            # over the navigator selection — so it targets the right labels even
            # when the Label Navigator dock is closed (audit HIGH-1 / MED-1).
            explicit = getattr(self, "_batch_explicit_targets", None)
            if explicit:
                ids = set(int(i) for i in explicit)
            else:
                ids = nav.selected_label_ids() if nav is not None else set()
            out = resolve_target_ids(present, "selected", selected_ids=ids)
            Curation3DAnnotator._batch_resolve_print(
                self,
                f"[BATCH] resolve selected: explicit={bool(explicit)} "
                f"nav_present={nav is not None} n_selected={len(ids)} "
                f"-> {len(out)} labels"
            )
            return out
        if mode == "plot":
            # What the QC plot highlights RIGHT NOW, read live from the
            # controller at every count refresh and again at Run -- never
            # cached.
            ids = self._read_plot_selection() or []
            out = resolve_target_ids(present, "selected", selected_ids=ids)
            Curation3DAnnotator._batch_resolve_print(
                self,
                f"[BATCH] resolve plot: n_highlighted={len(ids)} "
                f"-> {len(out)} labels"
            )
            return out
        if mode == "tier":
            wanted = set((spec or {}).get("tiers", []))
            items = getattr(nav, "_items", None) or []
            ids = {
                int(it["label_id"])
                for it in items
                if it.get("tier") in wanted and it.get("label_id") is not None
            }
            out = resolve_target_ids(present, "selected", selected_ids=ids)
            present_tiers = sorted({str(it.get("tier")) for it in items})
            print(
                f"[BATCH] resolve tier: checked={sorted(wanted)} "
                f"navigator_tiers={present_tiers} n_items={len(items)} "
                f"-> {len(out)} labels"
            )
            return out
        out = resolve_target_ids(present, "all")
        print(f"[BATCH] resolve all -> {len(out)} labels")
        return out

    def _batch_resolve_print(self, line) -> None:
        """A ``[BATCH] resolve`` line -- once per debounced refresh."""
        seen = _safe_attr(self, "_batch_resolve_printed")
        if seen is not None:
            if line in seen:
                return
            seen.add(line)
        print(line)

    def _on_navigator_selection_changed(self) -> None:
        """Navigator table selection changed (manual multi-select OR a
        QC-plot lasso routed through ``select_labels``) — keep the open
        Batch Refine "Selected" count live.

        Debounced: ``select_labels`` fires one ``selectionChanged`` per row,
        so a lasso of N labels would otherwise re-resolve N times.  A single
        coalescing QTimer collapses the burst into one refresh.
        """
        self._schedule_batch_count_refresh()

    def notify_spacing_changed(self) -> None:
        """Step 3's QC spacing changed (the controller's QC Settings OK).
        Never raises into the controller."""
        try:
            self._add_spacing_changed()
        except Exception as exc:  # noqa: BLE001
            print("[ADD] spacing notify failed: {!r}".format(exc))

    def _add_spacing_changed(self) -> None:
        """The amber default-spacing note follows the new spacing at once
        (the route its own sentence sends the user down), and a live box
        re-measures: the dark threshold's blur is in microns, so its cache
        is dropped and the box info re-requested."""
        self._push_add_ui(default_spacing=self._add_default_spacing())
        if (getattr(self, "_add_running", False)
                and getattr(self, "_add_box", None) is not None
                and not getattr(self, "_add_review", False)):
            # defensive (the QC Settings button is locked while a box
            # holds the Step 3 lock) -- A-L3 of the A1-A4 audit
            if getattr(self, "_add_busy", False):
                # nothing moves while a preview runs
                print("[ADD] spacing change ignored - a preview is running")
                return
            try:
                now = tuple(float(v) for v in self._refine_spacing())
            except Exception:  # noqa: BLE001
                now = None
            seen = getattr(self, "_add_info_spacing", None)
            if now is not None and seen is not None and tuple(seen) == now:
                # an OK that changed nothing re-measures nothing
                self._add_refresh_ui()
                return
            if getattr(self, "_add_preview_result", None) is not None:
                # candidates made at the old spacing are stale
                self._invalidate_add_preview()
            self._add_dark_cache = {}
            self._request_add_box_info()
        self._add_refresh_ui()

    def notify_plot_selection_changed(self) -> None:
        """The QC plot's highlighted set changed (controller-driven).

        Called by the controller from ``_on_qc_plot_label_selected`` and
        ``_apply_plot_highlight_adapter``, so an open Batch Refine dialog's
        "Plot selection (N)" follows the plot -- navigator clicks that
        repaint the plot included.  The same debounced recount as a
        navigator selection change.
        """
        self._schedule_batch_count_refresh()

    def _schedule_batch_count_refresh(self) -> None:
        """Restart the 120 ms coalescing timer behind the target recount."""
        if getattr(self, "_batch_refine_dialog", None) is None:
            return
        t = getattr(self, "_batch_count_timer", None)
        if t is None:
            from qtpy.QtCore import QTimer

            t = QTimer(self)
            t.setSingleShot(True)
            t.timeout.connect(self._do_refresh_batch_count)
            self._batch_count_timer = t
        t.start(120)

    def _read_plot_selection(self):
        """The controller's live plot selection, or ``None`` for no plot.

        ``None`` when no getter is wired (standalone annotator), when the
        controller says there is no QC plot, or when the getter raises.
        """
        scope = _safe_attr(self, "_batch_plot_scope")
        if scope is not None:
            # Inside one debounced refresh: the plot was read ONCE (L6).
            return scope[0]
        self._batch_plot_state = "none"
        cb = getattr(self, "_get_plot_selection_cb", None)
        if cb is None:
            return None
        try:
            ids = cb()
        except Exception as exc:  # noqa: BLE001 -- never raise into a count
            print(f"[BATCH] plot selection getter failed: {exc!r}")
            return None
        if ids is None:
            return None
        from packages.epicycle_annotator.batch_refine import PLOT_OTHER_PATCH

        if isinstance(ids, str) and ids == PLOT_OTHER_PATCH:
            # The plot shows another patch (M6): nothing to target, and the
            # dialog says so rather than "lasso first".
            self._batch_plot_state = "other"
            return []
        self._batch_plot_state = "ok"
        return [int(i) for i in ids]

    def _refresh_batch_target_counts(self, dlg) -> None:
        """Push every target radio's current count into the dialog.

        The radio texts used to be frozen at construction ("Selected (N)"
        never moved); every debounced recount now rewrites them.
        """
        from packages.epicycle_annotator.batch_refine import (
            resolve_target_ids,
        )

        present = getattr(self, "_batch_present_ids", None) or frozenset()
        n_selected = len(self._resolve_batch_targets({"mode": "selected"}))
        plot_ids = self._read_plot_selection()
        n_plot = len(
            resolve_target_ids(present, "selected", selected_ids=plot_ids or [])
        )
        _t0 = _time.perf_counter()
        dlg.set_target_counts(
            all=len(present),
            selected=n_selected,
            plot=n_plot,
            plot_available=plot_ids is not None,
            plot_other_patch=_safe_attr(self, "_batch_plot_state") == "other",
        )
        _batch_perf("set_target_counts", _t0)
        _t0 = _time.perf_counter()
        self._push_batch_preview_object()
        _batch_perf("preview object", _t0)

    def _do_refresh_batch_count(self) -> None:
        dlg = getattr(self, "_batch_refine_dialog", None)
        if dlg is not None:
            # ONE plot read for the whole refresh, and each resolve printed
            # once (UX audit L6): the counts, the count line and the batch
            # blocker all resolve the same targets.
            self._batch_plot_scope = None
            self._batch_resolve_printed = set()
            _all = _time.perf_counter()
            _t0 = _all
            try:
                self._batch_plot_scope = (self._read_plot_selection(),)
            except Exception:
                self._batch_plot_scope = None
            _batch_perf("plot read", _t0)
            try:
                _t0 = _time.perf_counter()
                try:
                    self._refresh_batch_target_counts(dlg)
                except Exception as exc:  # noqa: BLE001
                    print(f"[BATCH] target recount failed: {exc!r}")
                _batch_perf("target counts", _t0)
                _t0 = _time.perf_counter()
                try:
                    dlg.refresh_count()
                except Exception:
                    pass
                _batch_perf("dialog refresh_count", _t0)
            finally:
                self._batch_plot_scope = None
                self._batch_resolve_printed = None
                _batch_perf("count refresh TOTAL", _all)

    def _on_batch_preview_one(self) -> None:
        """Preview the recipe on the single currently-selected label."""
        sel = Curation3DAnnotator._batch_preview_id(self, check_volume=True)
        dlg = getattr(self, "_batch_refine_dialog", None)
        if not sel or int(sel) <= 0:
            if dlg is not None:
                from packages.epicycle_annotator.batch_refine import (
                    BATCH_PREVIEW_NO_OBJECT,
                )

                # set_state("idle") clears the status line: state first (M4).
                dlg.set_state("idle")
                _batch_recount(dlg)
                dlg.set_status(BATCH_PREVIEW_NO_OBJECT)
            return
        self._run_batch_refine([int(sel)])

    def _on_batch_run(self) -> None:
        """Run the recipe across the full resolved target set."""
        dlg = getattr(self, "_batch_refine_dialog", None)
        if dlg is None:
            return
        # The open-time cache serves the COUNT; Run re-resolves against the
        # live volume (one find_objects pass) so an object painted -- or
        # brought back by an undo -- while the dialog was open is not
        # dropped (audit A2-M1).
        from packages.epicycle_annotator.batch_refine import label_bboxes

        labels_name = self._active_labels_name
        shared = (
            self.sync_manager.shared_labels.get(labels_name)
            if labels_name
            else None
        )
        fresh_boxes = label_bboxes(shared) if shared is not None else None
        fresh = frozenset(fresh_boxes) if fresh_boxes is not None else None
        # The fresh pass REPLACES the cache: the chain path reads its boxes
        # from it, so it no longer takes a second pass for ids painted
        # after the open (D-L6), and the count line agrees with a refusal
        # below (the open-time cache still counted objects that are gone).
        if fresh_boxes is not None:
            self._batch_bboxes = fresh_boxes
            self._batch_present_ids = fresh
        targets = self._resolve_batch_targets(dlg.target_spec(), present=fresh)
        if not targets:
            from packages.epicycle_annotator.batch_refine import (
                BATCH_NO_TARGETS,
            )

            # set_state("idle") clears the status line, so it goes first;
            # the recount puts the count line and Run right before the
            # sentence (wiring audit M1).
            dlg.set_state("idle")
            _batch_recount(dlg)
            dlg.set_status(BATCH_NO_TARGETS)
            return
        self._run_batch_refine(targets)

    # ---- recipe history (session-recent + persisted "Saved recipes") ----
    #
    # Reload-only: a recipe (steps + params) repopulates the dialog's recipe
    # builder; the TARGET (all/selected/by-tier) is NOT stored.  Two sources:
    #   * session-recent — in-memory ``_batch_recipe_history`` (most-recent
    #     first, capped, de-duped), appended on each applied (accepted) batch;
    #     survives patch switches + dialog re-opens within the session, reset
    #     on annotator teardown.
    #   * persisted "Saved recipes" — workspace JSON via the controller
    #     ``_get_saved_batch_recipes_cb`` / ``_save_batch_recipe_cb`` callbacks
    #     (None on a standalone annotator → no Saved population, Save disabled).

    _BATCH_RECIPE_HISTORY_CAP = 10

    def _append_batch_recipe_history(self, recipe) -> None:
        """Push a just-applied recipe onto the session-recent list.

        ``recipe`` is a ``list[RefineStep]``; it's serialized to the
        round-trippable list-of-dicts form, de-duped against existing
        entries (identical recipe → moved to front, not duplicated), and the
        list is capped at ``_BATCH_RECIPE_HISTORY_CAP`` (most-recent first).
        """
        if not recipe:
            return
        from packages.epicycle_annotator.batch_refine import recipe_to_json

        recipe_json = recipe_to_json(recipe)
        hist = getattr(self, "_batch_recipe_history", None)
        if hist is None:
            hist = []
            self._batch_recipe_history = hist
        # De-dupe: drop any identical existing entry, then prepend.
        hist[:] = [r for r in hist if r != recipe_json]
        hist.insert(0, recipe_json)
        del hist[self._BATCH_RECIPE_HISTORY_CAP:]

    def _save_batch_recipe(self, name: str) -> None:
        """Forward a dialog Save-recipe request to the controller callback.

        Persists the dialog's CURRENT recipe under ``name`` via the injected
        ``_save_batch_recipe_cb`` (None on a standalone annotator → no-op),
        then refreshes the dialog's Saved dropdown.  Guarded so a callback
        error can't propagate into the Qt button handler.
        """
        dlg = getattr(self, "_batch_refine_dialog", None)
        if dlg is None:
            return
        from packages.epicycle_annotator.batch_refine import (
            BATCH_SAVE_FAILED,
            BATCH_SAVE_INVALID,
            BATCH_SAVE_NO_WORKSPACE,
            BATCH_SAVED,
            recipe_to_json,
        )

        recipe = dlg.recipe()  # validated list[RefineStep] | None
        if not recipe:
            # Invalid / empty — dialog.recipe() already set a status.
            dlg.set_status(BATCH_SAVE_INVALID)
            return
        cb = getattr(self, "_save_batch_recipe_cb", None)
        if cb is None:
            dlg.set_status(BATCH_SAVE_NO_WORKSPACE)
            return
        try:
            cb(name, recipe_to_json(recipe))
        except Exception as exc:
            # A fixed sentence on screen; the raw reason to the log (M5).
            print(f"[BATCH] save-recipe callback raised: {exc!r}")
            dlg.set_status(BATCH_SAVE_FAILED)
            return
        dlg.set_status(BATCH_SAVED.format(name=name))
        self._refresh_batch_recipe_history(dlg)

    def _refresh_batch_recipe_history(self, dlg=None) -> None:
        """(Re)populate the dialog's Recent/Saved dropdowns.

        Pulls the in-memory session-recent list + the persisted saved store
        (via the controller callback, guarded) and hands both to the dialog's
        ``set_recipe_history``.  Called on dialog open + after a save.
        """
        dlg = dlg or getattr(self, "_batch_refine_dialog", None)
        if dlg is None or not hasattr(dlg, "set_recipe_history"):
            return
        recent = list(getattr(self, "_batch_recipe_history", []) or [])
        saved = {}
        cb = getattr(self, "_get_saved_batch_recipes_cb", None)
        if cb is not None:
            try:
                saved = cb() or {}
            except Exception as exc:
                print(f"[BATCH] get-saved-recipes callback raised: {exc!r}")
                saved = {}
        try:
            dlg.set_recipe_history(recent, saved)
        except Exception as exc:
            print(f"[BATCH] set_recipe_history failed: {exc!r}")

    def _fire_batch_refine_lock(self, active: bool) -> None:
        """Notify the controller that a Batch Refine started/resolved.

        ``active=True`` at START (worker launch); ``active=False`` on EVERY
        terminal path — accept / reject / worker error / cancel / dialog close /
        discard — so the controller's unified Step-3 lock covers BOTH the worker
        compute AND the staged-preview window.  Guarded with
        ``getattr(self, "_batch_refine_lock_cb", None)`` so a standalone
        annotator (no controller wiring) is a no-op; callback errors are
        swallowed so a buggy controller callback can't strand the batch flow.
        """
        cb = getattr(self, "_batch_refine_lock_cb", None)
        if cb is None:
            return
        try:
            cb(bool(active))
        except Exception as exc:
            print(f"[BATCH] lock callback raised: {exc!r}")

    # ======================================================================
    # De-novo "Add" mode (2026-06-30) — box → threshold → watershed → commit.
    # Mirrors the split lifecycle; box is slider-driven (napari Shapes are
    # only editable in ndisplay=2) and rendered as a read-only wireframe in
    # the main 3D view.  Editable subviewer-rect dragging + a box-scoped CLAHE
    # Image overlay are deferred Phase-1.1 polish.
    # ======================================================================

    def _fire_add_mode_lock(self, active: bool) -> None:
        """Notify the controller that Add mode entered (True) / resolved (False).

        Mirrors ``_fire_batch_refine_lock`` — fired True on Start and False on
        EVERY exit path (accept / reject / cancel / undo-all / patch-switch
        teardown).  Guarded so a standalone annotator is a no-op.
        """
        cb = getattr(self, "_add_mode_lock_cb", None)
        if cb is None:
            return
        try:
            cb(bool(active))
        except Exception as exc:
            print(f"[ADD] lock callback raised: {exc!r}")

    def _refresh_add_image_combo(self) -> None:
        """Populate the three Add channel combos from the viewer's image
        layers (blank stays blank), and the page's patch context."""
        image_names = self._add_image_channel_names()
        adv = self._get_advanced_features_widget()
        if adv is not None and hasattr(adv, "refresh_add_image_combo"):
            adv.refresh_add_image_combo(image_names)
        try:
            self._add_push_context()
        except Exception as exc:  # noqa: BLE001
            print(f"[ADD] context refresh skipped: {exc!r}")

    def _push_add_ui(self, **kwargs) -> None:
        """Proxy to the widget's update_add_ui_state (guarded)."""
        adv = self._get_advanced_features_widget()
        if adv is not None and hasattr(adv, "update_add_ui_state"):
            adv.update_add_ui_state(**kwargs)

    def _add_main_label_layer(self):
        return self._layer_by_name(self.viewer.layers, self._active_labels_name)

    def _add_shared_volume(self):
        sm = getattr(self, "sync_manager", None)
        if sm is None:
            return None
        return sm.shared_labels.get(self._active_labels_name)

    def _add_label_id_floor(self) -> int:
        """The controller's df id floor (ghost rows included), or 0."""
        cb = getattr(self, "_get_label_id_floor_cb", None)
        if cb is None:
            return 0
        try:
            return max(0, int(cb() or 0))
        except Exception as exc:
            print(f"[ADD] id floor callback raised: {exc!r}")
            return 0

    def _add_next_label(self, cached: bool = False) -> int:
        """``max(max(volume), floor) + 1`` -- new ids skip the ids of deleted
        objects.  Taken on the main
        thread at Preview launch (always fresh there).

        ``cached=True`` (a marker placement, B-L3 of the A5 audit) reuses
        the volume max of this session -- the Add lock means only Add's own
        Accept writes the volume while a box is up -- keyed on the array's
        identity, and dropped at Accept, at every reset and at a realign
        (``_add_forget_volume_max``).  The df floor is always read."""
        shared = self._add_shared_volume()
        floor = self._add_label_id_floor()
        if shared is None:
            return floor + 1
        try:
            memo = getattr(self, "_add_volume_max", None)
        except Exception:  # noqa: BLE001
            memo = None
        try:
            if cached and memo is not None and memo[0] == id(shared):
                vmax = memo[1]
            else:
                vmax = int(np.max(shared))
        except Exception:
            return floor + 1
        try:
            # the memo is a convenience: never let storing it cost the id
            self._add_volume_max = (id(shared), vmax)
        except Exception:  # noqa: BLE001
            pass
        return max(vmax, floor) + 1

    def _add_forget_volume_max(self) -> None:
        """Drop the session's cached volume max (B-L3): Accept wrote the
        volume, the session reset, or the shared array was realigned."""
        self._add_volume_max = None

    def _add_id_max(self):
        """The largest id the label volume's dtype can hold (the 3.2 dtype
        guard), or ``None`` for a non-integer / missing volume."""
        shared = self._add_shared_volume()
        try:
            return int(np.iinfo(np.asarray(shared).dtype).max)
        except Exception:
            return None

    def _add_spacing(self):
        """Per-axis voxel spacing for the anisotropy-aware watershed (or None)."""
        main = self._add_main_label_layer()
        if main is None:
            return None
        try:
            sc = tuple(float(s) for s in main.scale[-3:])
            return sc if len(sc) == 3 else None
        except Exception:
            return None

    def _add_channel_error_text(self) -> str:
        """Why the box has no signal to work with, naming the channel."""
        adv = self._get_advanced_features_widget()
        name = ""
        if adv is not None and hasattr(adv, "add_channel_combo"):
            try:
                name = adv.add_channel_combo.currentText()
            except Exception:
                name = ""
        if not name:
            return "Pick the image channel the cell is bright in."
        return (
            "Cannot read the image layer '{}'. Pick the channel "
            "again.".format(name)
        )

    def _add_refused_while_batch(self, door) -> bool:
        """True (and the sentence in the Add status) while Batch Refine runs
        or is staged.  Its finish writes its union and its Reject
        pastes the snapshot back, so a cell Add commits meanwhile would be
        erased either way."""
        if not (_safe_attr(self, "_batch_running", False)
                or _safe_attr(self, "_batch_staged", False)):
            return False
        from packages.epicycle_annotator.batch_refine import ADD_BATCH_BUSY

        print(f"[ADD] {door} refused — Batch Refine is running or staged")
        self._push_add_ui(status_text=ADD_BATCH_BUSY, status_kind="warn")
        return True

    def _start_add(self, reuse_box: bool = False) -> None:
        """Enter Add mode: drop a box at the crosshair (or volume centre) +
        tune.  ``reuse_box`` ("+ Add another") re-places the retained box
        and skips the crosshair."""
        if self._add_refused_while_batch("start"):
            return
        if getattr(self, "_add_running", False):
            return
        # re-point shared at the main layer first, as split / smooth /
        # merge do: a Commit Triage, Discard or undo may have reassigned
        # layer.data (final fix pass U1)
        self._ensure_shared_data_aligned(self._active_labels_name)
        main = self._add_main_label_layer()
        if main is None:
            from packages.epicycle_annotator.add_preview import ADD_S1_NO_PATCH

            self._push_add_ui(status_text=ADD_S1_NO_PATCH, status_kind="warn")
            return
        from packages.epicycle_annotator.roi_box import BoxRec

        # a new box starts with the outline tick down
        self._add_put_outline_down()
        shape = main.data.shape
        if not reuse_box or getattr(self, "_add_box", None) is None:
            # the preset used last in this workspace (Nuclei the first time)
            self._add_restore_preset()
        # Drop the box where the user is looking — the crosshair intersection —
        # falling back to the volume centre when no crosshair position is known.
        # Reuse the last box size (sticky) so repeated adds keep the size you set.
        kept = getattr(self, "_add_box", None) if reuse_box else None
        if kept is not None:
            box = kept.clamped(shape)
        else:
            box = BoxRec.centered(shape,
                                  size=getattr(self, "_add_last_size", None))
            center = self._add_crosshair_center(shape)
            if center is not None:
                box = box.with_center(*center).clamped(shape)
            # on Cells the start box is clamped to the budget before it is
            # drawn; Nuclei keeps today's box
            box = self._add_clamp_box(box, shape)
        self._add_box = box
        self._add_roi = self._add_box.bounds_slices(shape)
        self._add_running = True
        self._add_review = False
        self._add_threshold_value = None
        self._add_threshold_touched = False
        self._add_threshold_auto_written = None
        self._fire_add_mode_lock(True)
        # Any setup failure AFTER acquiring the lock must release it, or the
        # whole Step-3 surface strands locked (audit MED-5).
        try:
            # Lock drawing IN-WIDGET — do NOT rely on the controller round-trip
            # (it silently no-ops if the callback chain misses, which is why the
            # user could still paint).  Mirrors Split: main labels stay locked +
            # freeze the subviewer label layers so painting is dead while a box
            # is being placed.  Restored on accept/reject.
            self._lock_subviewers_for_preview()
            try:
                main.editable = False
                main.mode = "pan_zoom"
            except Exception:
                pass
            self._refresh_add_image_combo()
            self._render_add_box()
            self._install_add_side_callbacks()
            b = self._add_box
            self._push_add_ui(
                box_active=True,
                box_shape=tuple(int(s) for s in shape),
                box_center=(b.cz, b.cy, b.cx),
                box_size=(b.sz, b.sy, b.sx),
                review_active=False,
                start_enabled=False,
                preview_enabled=True,
                accept_enabled=False,
                accept_count=0,
                reject_enabled=True,
                footer_text="",
            )
            self._add_transient = None
            self._add_info = None
            self._add_brightness_channel_seen = self._add_brightness_channel()
            self._push_add_ui(threshold_pending=True)
            # Populate the histogram + Otsu line only — the watershed runs on the
            # Preview click, not live (keeps box placement snappy).
            self._compute_add_histogram(reset_threshold=True)
            self._request_add_box_info()
            self._add_refresh_ui(nuclei_auto=float("nan"), run_count=())
        except Exception as exc:
            print(f"[ADD] _start_add failed: {exc!r}")
            self._reject_add()

    def _add_crosshair_center(self, shape):
        """The crosshair intersection ``(z, y, x)`` clamped to *shape*, or None.

        The intersection is the current slice position of the 3 subviewers —
        Z from XY, Y from XZ, X from YZ — the same axis mapping the crosshair
        overlays use (``crosshairs.py::update_crosshairs``).
        """
        cm = getattr(self, "crosshair_manager", None)
        vm_xy = getattr(self, "viewer_model_xy", None)
        vm_xz = getattr(self, "viewer_model_xz", None)
        vm_yz = getattr(self, "viewer_model_yz", None)
        if cm is None or vm_xy is None or vm_xz is None or vm_yz is None:
            return None
        try:
            z = cm._safe_current_step0(vm_xy)
            y = cm._safe_current_step0(vm_xz)
            x = cm._safe_current_step0(vm_yz)
        except Exception:
            return None
        Z, Y, X = (int(s) for s in shape)
        return (
            int(np.clip(z, 0, max(0, Z - 1))),
            int(np.clip(y, 0, max(0, Y - 1))),
            int(np.clip(x, 0, max(0, X - 1))),
        )

    def _recenter_add_box_on_crosshair(self) -> None:
        """Snap the box centre to the crosshair intersection (current view).

        Deliberately on-demand (a button), not live — scrolling to inspect must
        not drag the box around.  Reuses the exact slider-move path so the ROI,
        wireframe, stale-preview reset, and histogram all update identically."""
        if not getattr(self, "_add_running", False) or self._add_box is None:
            return
        if self._add_refuse_busy("recenter"):
            return
        main = self._add_main_label_layer()
        if main is None:
            return
        center = self._add_crosshair_center(main.data.shape)
        if center is None:
            from packages.epicycle_annotator.add_preview import (
                ADD_RECENTER_NO_CROSSHAIR,
            )

            self._push_add_ui(status_text=ADD_RECENTER_NO_CROSSHAIR,
                              status_kind="warn")
            return
        cz, cy, cx = center
        b = self._add_box
        self._on_add_box_changed(cz, cy, cx, b.sz, b.sy, b.sx)
        nb = self._add_box
        self._push_add_ui(
            box_center=(nb.cz, nb.cy, nb.cx),
            box_size=(nb.sz, nb.sy, nb.sx),
            status_text="Box recentered on the crosshair.",
            status_kind="ready",
        )

    def _on_add_box_changed(self, cz, cy, cx, sz, sy, sx) -> None:
        """Slider edit → update box + wireframe + histogram.  Does NOT run the
        watershed — that only happens on an explicit Preview click (keeps box
        tuning snappy; the live watershed was the lag the user hit)."""
        if not getattr(self, "_add_running", False):
            return
        if self._add_refuse_busy("box change"):
            b = self._add_box
            if b is not None:     # put the Numbers... sliders back
                self._push_add_ui(box_center=(b.cz, b.cy, b.cx),
                                  box_size=(b.sz, b.sy, b.sx))
            return
        main = self._add_main_label_layer()
        if main is None:
            return
        from packages.epicycle_annotator.roi_box import BoxRec

        shape = main.data.shape
        self._add_box = BoxRec(int(cz), int(cy), int(cx),
                               int(sz), int(sy), int(sx))
        self._add_last_size = (int(sz), int(sy), int(sx))  # sticky size
        self._add_roi = self._add_box.bounds_slices(shape)
        self._render_add_box()       # cheap — just updates the Shapes geometry
        self._invalidate_add_preview()  # the box moved → any preview is stale
        self._add_drop_info()           # ... and so is the box info
        self._add_keep_clicks_in_box()  # after the invalidate: it says so
        self._schedule_add_histogram()  # debounced; intensity only, no watershed

    def _set_add_box_size_fraction(self, frac: float) -> None:
        """Preset: resize the box to *frac* of each patch axis (keeps centre).

        Relative so it works on any data scale — S/M/L ≈ 1/8 / 1/4 / 1/2.  Reuses
        the slider-move path + syncs the sliders; the sticky last-size follows."""
        if not getattr(self, "_add_running", False) or self._add_box is None:
            return
        if self._add_refuse_busy("box size"):
            return
        main = self._add_main_label_layer()
        if main is None:
            return
        Z, Y, X = (int(s) for s in main.data.shape)
        sz = max(1, int(round(Z * frac)))
        sy = max(1, int(round(Y * frac)))
        sx = max(1, int(round(X * frac)))
        b = self._add_box
        # on Cells only, scaled uniformly to the budget
        from packages.epicycle_annotator.roi_box import BoxRec

        c = self._add_clamp_box(BoxRec(b.cz, b.cy, b.cx, sz, sy, sx),
                                main.data.shape)
        sz, sy, sx = c.sz, c.sy, c.sx
        self._on_add_box_changed(b.cz, b.cy, b.cx, sz, sy, sx)
        nb = self._add_box
        self._push_add_ui(
            box_center=(nb.cz, nb.cy, nb.cx),
            box_size=(nb.sz, nb.sy, nb.sx),
        )

    def _schedule_add_histogram(self) -> None:
        """Debounce the (cheap) in-box histogram refresh."""
        from qtpy.QtCore import QTimer

        t = getattr(self, "_add_box_timer", None)
        if t is None:
            t = QTimer(self)
            t.setSingleShot(True)
            t.timeout.connect(self._do_add_histogram)
            self._add_box_timer = t
        t.start(150)

    def _do_add_histogram(self) -> None:
        if not getattr(self, "_add_running", False):
            return
        self._request_add_box_info()
        self._compute_add_histogram(
            reset_threshold=self._add_threshold_value is None
        )

    def _invalidate_add_preview(self) -> None:
        """Drop any stale preview when an input changes.

        The watershed only runs on an explicit Preview click, so whenever the
        box / threshold / parameters change we clear the old candidate + disable
        Accept and prompt for a re-Preview.  Cheap (candidate removal no-ops when
        none exists)."""
        if getattr(self, "_add_busy", False):
            # every change is refused while a preview runs;
            # reaching here busy is a missing refusal upstream
            print("[ADD] _invalidate_add_preview called while busy - ignored")
            return
        self._add_transient = None
        # the box info is NOT dropped here: it depends on the box, the
        # channels, the source and the Advanced knobs -- the callers that
        # change one of those call _add_drop_info() and re-ask.  Dropping
        # it for a click, a threshold, Auto, Discard or "Preview again" left
        # nothing to ask again: Cells' S1 sat at "Measuring the box..." and
        # "Preview again" was refused (live-shot fix F9)
        self._drop_add_walls(keep_intent=True)
        had = getattr(self, "_add_candidate_layer", None) is not None \
            or getattr(self, "_add_preview_result", None) is not None
        self._add_preview_result = None
        self._add_preview_res = None
        self._add_preview_roi = None
        if getattr(self, "_add_candidate_layer", None) is not None:
            self._remove_add_candidate()
        if had:
            self._push_add_ui(
                accept_enabled=False, accept_count=0,
                candidate_ids=[],  # clear the stale candidate checklist
                footer_text="",
            )
        self._add_refresh_ui()

    def _add_drop_info(self) -> None:
        """An INPUT of the box info changed (the box, a channel, the source,
        the preset, an Advanced knob): drop the answer and show it pending.
        Every caller also schedules the debounced request
        (``_schedule_add_histogram`` -> ``_request_add_box_info``)."""
        self._add_info = None
        # final fix pass B3: the Brightness bars blank and Auto greys NOW,
        # not after the 150 ms debounce -- Auto is never clickable-but-dead
        self._add_refresh_ui(nuclei_auto=float("nan"), run_count=(),
                             histogram_data={"counts": [], "edges": []},
                             brightness_auto=float("nan"))

    def _add_brightness_channel(self) -> str:
        try:
            return str(self._add_params().get("channel", "") or "")
        except Exception:
            return ""

    def _add_clear_threshold(self) -> None:
        """The Brightness threshold's value space (or its channel) changed:
        forget it, untouched, and show it pending until the next Auto lands
        (final fix pass B1 / B4 / B6)."""
        self._add_threshold_value = None
        self._add_threshold_touched = False
        self._add_threshold_auto_written = None
        self._push_add_ui(threshold_pending=True)

    def _compute_add_histogram(self, *, reset_threshold: bool = False) -> None:
        """Show the Brightness histogram and its Auto as PENDING; the box-info
        worker measures both (Add rebuild B4.2) and
        ``_on_add_info_finished`` draws them.  Nothing is read or computed
        here: CLAHE + Otsu of the crop cost 169-928 ms per tick on the main
        thread.  ``reset_threshold`` lets the next landing move the line
        (the untouched rule); a threshold the user set is kept otherwise."""
        if self._add_roi is None:
            return
        if reset_threshold:
            self._add_threshold_value = None
        if self._add_preset() == "nuclei" and not self._add_brightness_source_ok():
            # Both callers used to return in SILENCE when the channel could
            # not be read, so the histogram and the threshold simply never
            # updated and nothing said why (policy 3).
            self._push_add_ui(status_text=self._add_channel_error_text(),
                              status_kind="warn")
            return
        self._push_add_ui(histogram_data={"counts": [], "edges": []},
                          brightness_auto=float("nan"))

    def _add_brightness_source_ok(self) -> bool:
        """The Brightness channel resolves to a layer (metadata only -- no
        pixel is read)."""
        name = str(self._add_params().get("channel", "") or "")
        return bool(name) and self._refine_channel_source(name) is not None

    def _add_brightness_key(self):
        """What the Brightness histogram and Auto depend on besides the box
        (which the info stale guard already carries)."""
        p = self._add_params()
        return (self._add_preset(), str(p.get("channel", "") or ""),
                bool(p.get("clahe", False)),
                float(p.get("clahe_clip", 0.01)))

    def _on_add_clahe_toggled(self, checked: bool) -> None:
        # CLAHE changes the value space → re-histogram + reset the Otsu line.
        # No watershed (Preview only).
        if not getattr(self, "_add_running", False):
            return
        if self._add_refuse_busy("clahe_toggled"):
            return
        self._add_clear_threshold()
        self._invalidate_add_preview()
        self._add_drop_info()
        self._schedule_add_histogram()

    def _on_add_threshold_changed(self, value: float) -> None:
        # Just record the new cut; the histogram is unchanged and the watershed
        # waits for Preview.
        if not getattr(self, "_add_running", False):
            return
        if self._add_refuse_busy("threshold_changed"):
            return
        self._add_threshold_value = float(value)
        # the user set it (typed or dragged): landings keep it (B1)
        self._add_threshold_touched = True
        self._invalidate_add_preview()

    def _on_add_otsu_reset(self, _checked=False, *, _silent: bool = False) -> None:
        """Auto: put the threshold at the Otsu the box-info worker measured
        for the CURRENT box and Brightness inputs (B4.2).  Nothing is
        computed here; while the answer is pending the button is grey and
        this returns."""
        if self._add_roi is None:
            return
        if not _silent and self._add_refuse_busy("auto threshold"):
            return
        # Auto again: the threshold follows the next landings again (B1)
        self._add_threshold_touched = False
        info = getattr(self, "_add_info", None)
        auto = None
        if (info is not None and getattr(self, "_add_info_brightness_key",
                                         None) == self._add_brightness_key()):
            auto = getattr(info, "brightness_auto", None)
        if auto is None:
            if not self._add_brightness_source_ok():
                self._push_add_ui(status_text=self._add_channel_error_text(),
                              status_kind="warn")
            return
        self._add_threshold_value = float(auto)
        self._add_threshold_auto_written = float(auto)
        self._push_add_ui(threshold_value=float(auto))
        # Just moves the line; the user clicks Preview to segment (no watershed
        # here).
        if not _silent:
            self._invalidate_add_preview()

    def _on_add_advanced_changed(self) -> None:
        # Params changed (min size / separate / clahe clip / …) → stale preview;
        # re-histogram (clahe clip can shift it).  No watershed (Preview only).
        if not getattr(self, "_add_running", False):
            return
        if self._add_refuse_busy("advanced_changed"):
            return
        self._invalidate_add_preview()
        self._add_drop_info()           # dark factor / shrink feed the info
        self._schedule_add_histogram()

    # ======================================================================
    # Add rebuild B2.4 -- presets, sources, the run guide, the box limit
    #
    # ======================================================================

    _ADD_SETTINGS = ("EpiCycle", "Step3Add")

    def _add_settings_key(self, what):
        import hashlib

        ws = str(self._reshape_context().get("workspace") or "")
        if not ws:
            return None
        return "{}/{}".format(what, hashlib.sha1(ws.encode("utf-8"))
                              .hexdigest())

    def _add_load_setting(self, what, default):
        key = self._add_settings_key(what)
        mem = getattr(self, "_add_session_memory", None) or {}
        if key is None:
            return mem.get(what, default)
        try:
            from qtpy.QtCore import QSettings

            val = QSettings(*self._ADD_SETTINGS).value(key, None)
        except Exception:
            val = None
        return default if val is None else val

    def _add_save_setting(self, what, value):
        mem = getattr(self, "_add_session_memory", None)
        if mem is None:
            mem = {}
            self._add_session_memory = mem
        mem[what] = value
        key = self._add_settings_key(what)
        if key is None:
            return
        try:
            from qtpy.QtCore import QSettings

            QSettings(*self._ADD_SETTINGS).setValue(key, value)
        except Exception as exc:
            print(f"[ADD] could not remember {what}: {exc!r}")

    def _add_restore_preset(self) -> None:
        """The preset used last in this workspace (Nuclei the first time),
        and the dark tick beside it (remembered on its own)."""
        adv = self._get_advanced_features_widget()
        if adv is None or not hasattr(adv, "set_add_preset"):
            return
        preset = str(self._add_load_setting("last_preset", "nuclei"))
        if preset not in ("nuclei", "cells"):
            preset = "nuclei"
        dark = self._add_load_setting("dark_outside", True)
        dark = str(dark).lower() not in ("false", "0", "no")
        try:
            adv.set_add_preset(preset, silent=True)
            cb = getattr(adv, "add_dark_outside_cb", None)
            if cb is not None:
                cb.blockSignals(True)
                cb.setChecked(bool(dark))
                cb.blockSignals(False)
            adv._apply_add_visibility()
        except Exception as exc:
            print(f"[ADD] preset restore failed: {exc!r}")

    def _add_marker_source_entries(self):
        """The Markers source combo's items: the other runs
        (curated first, then by name), the nuclei item, the clicks item."""
        from packages.epicycle_annotator import add_preview as AP

        ctx = self._reshape_context()
        srcs = []
        try:
            from napari_epicycle.annotator.reshape_guide import (
                _DEFAULT_RUN_ID,
                list_guide_sources,
            )

            # ctx["run_id"] is None on the DEFAULT (flat) run, and
            # list_guide_sources excludes nothing for None -- so the patch's
            # own run was offered as "another run", an enabled automatic
            # source, and the membrane-only amber never showed on a
            # single-channel patch (live-shot fix F4)
            srcs = list(list_guide_sources(
                ctx.get("workspace"), ctx.get("patch_name"),
                exclude_run_id=ctx.get("run_id") or _DEFAULT_RUN_ID))
        except Exception as exc:
            print(f"[ADD] listing runs failed: {exc!r}")
        srcs = [s for s in srcs if getattr(s, "kind", "run") == "run"]
        srcs.sort(key=lambda s: (
            0 if getattr(s, "which_file", "") == "labels_curated.tif" else 1,
            str(getattr(s, "display_name", "")).lower()))
        self._add_run_sources = {s.run_id: s for s in srcs}
        entries = []
        for s in srcs:
            name = str(s.display_name or s.run_id)
            if len(name) > 28:
                name = name[:13] + "…" + name[-13:]
            text = AP.ADD_ITEM_RUN.format(name=name)
            if getattr(s, "which_file", "") != "labels_curated.tif":
                text += AP.ADD_ITEM_RAW_SUFFIX
            entries.append((text, True, ("labels", s.run_id)))
        if not srcs:
            entries.append((AP.ADD_ITEM_NO_RUN, False, ("labels", None)))
        two = len(self._add_image_channel_names()) >= 2
        entries.append((AP.ADD_ITEM_NUCLEI if two else AP.ADD_ITEM_NUCLEI_GREY,
                        two, ("nuclei_masks", None)))
        entries.append((AP.ADD_ITEM_CLICKS, True, ("paint", None)))
        return entries

    def _add_push_context(self, *, sources=True) -> None:
        """What the page needs to know about the patch: the voxel size, the
        channel count, the marker sources."""
        kw = dict(default_spacing=self._add_default_spacing(),
                  single_channel=len(self._add_image_channel_names()) < 2)
        if sources:
            kw["marker_sources"] = self._add_marker_source_entries()
        self._push_add_ui(**kw)

    def _add_budget(self) -> int:
        try:
            return int(self._add_params().get("box_budget", 4_000_000))
        except Exception:
            return 4_000_000

    def _add_clamp_box(self, box, shape):
        """On Cells, scale the box uniformly by ``(budget / n) ** (1/3)``
        (floored, >= 1) when it is over the budget."""
        if self._add_preset() != "cells" or box is None:
            return box
        budget = self._add_budget()
        n = int(box.sz) * int(box.sy) * int(box.sx)
        if n <= budget:
            return box
        from packages.epicycle_annotator.roi_box import BoxRec

        f = (budget / float(n)) ** (1.0 / 3.0)
        sz, sy, sx = (max(1, int(np.floor(v * f)))
                      for v in (box.sz, box.sy, box.sx))
        while sz * sy * sx > budget:
            sz, sy, sx = max(1, sz - 1), max(1, sy - 1), max(1, sx - 1)
        return BoxRec(box.cz, box.cy, box.cx, sz, sy, sx).clamped(shape)

    def _on_add_preset_changed(self, preset) -> None:
        """A preset switch: remembered per workspace; never
        resizes the box; unchecks Place markers; drops a preview.  Refused
        while busy (the panel disables the combo too -- B2 fix pass L-L1)."""
        if self._add_refuse_busy("preset change"):
            return
        self._add_save_setting("last_preset", str(preset))
        self._add_place_markers_off()
        self._add_update_ball_visibility()
        if getattr(self, "_add_running", False):
            self._add_run_guide_reset()
            self._invalidate_add_preview()
            self._add_drop_info()
            self._schedule_add_histogram()
        self._add_push_context(sources=False)
        self._add_refresh_ui()

    def _on_add_source_changed(self, key) -> None:
        """A marker-source change: the run guide is dropped, a picked run is
        read on a worker, Place markers goes off.  Refused while
        busy (B2 fix pass L-L1)."""
        if self._add_refuse_busy("source change"):
            return
        self._add_place_markers_off()
        self._add_run_guide_reset()
        kind, run_id = (tuple(key) + (None, None))[:2]
        if kind == "labels" and run_id:
            self._add_load_run_guide(run_id)
        if getattr(self, "_add_running", False):
            self._invalidate_add_preview()
            self._add_drop_info()
            self._schedule_add_histogram()
        self._add_refresh_ui()

    def _on_add_channels_changed(self) -> None:
        if not getattr(self, "_add_running", False):
            return
        if self._add_refuse_busy("channel change"):
            return
        chan = self._add_brightness_channel()
        if chan != getattr(self, "_add_brightness_channel_seen", None):
            # another Brightness channel: another value range (B6, like the
            # Local contrast toggle)
            self._add_brightness_channel_seen = chan
            self._add_clear_threshold()
        self._invalidate_add_preview()
        self._add_drop_info()
        self._schedule_add_histogram()
        self._add_refresh_ui()

    def _on_add_dark_outside_toggled(self, checked) -> None:
        self._add_save_setting("dark_outside", bool(checked))

    def _on_add_numbers(self):
        """Numbers... opens the six sliders; refused while busy."""
        if self._add_refuse_busy("numbers"):
            return False
        return True

    def _on_add_nuclei_auto(self, _checked=False) -> None:
        if self._add_refuse_busy("auto"):
            return
        info = getattr(self, "_add_info", None)
        auto = None if info is None else info.nuclei_auto
        if auto is None:
            return
        self._add_nuclei_threshold_value = float(auto)
        # Auto again: the value follows the box's next landings again
        self._add_nuclei_threshold_typed = False
        adv = self._get_advanced_features_widget()
        if adv is not None and hasattr(adv, "set_add_nuclei_threshold"):
            adv.set_add_nuclei_threshold(auto)
        self._invalidate_add_preview()

    def _on_add_nuclei_threshold_typed(self, value) -> None:
        if not getattr(self, "_add_running", False):
            return
        if self._add_refuse_busy("threshold"):
            return
        self._add_nuclei_threshold_value = float(value)
        self._add_nuclei_threshold_typed = True
        self._invalidate_add_preview()

    def _add_secondary(self, _checked=False) -> None:
        """The footer's secondary: Done in REVIEW, Discard with candidates
        up (drops them, keeps the box and markers --), Cancel
        otherwise (ends the session)."""
        if getattr(self, "_add_review", False):
            self._reject_add()
            return
        if (getattr(self, "_add_preview_result", None) is not None
                and not getattr(self, "_add_busy", False)):
            self._discard_add_candidates()
            return
        self._reject_add()

    def _discard_add_candidates(self) -> None:
        """Discard (candidates up): back to the box state; the side views
        stay locked for the session (grill E-L10)."""
        self._invalidate_add_preview()
        self._add_transient = None
        self._push_add_ui(candidate_ids=[], accept_enabled=False,
                          accept_count=0, footer_text="")
        self._add_refresh_ui()

    # --- the run guide (a picked run's labels, read on a worker) -----------
    def _add_run_guide_reset(self) -> None:
        self._add_run_guide = None
        self._add_run_state = None
        adv = self._get_advanced_features_widget()
        if adv is not None and hasattr(adv, "set_add_source_note"):
            adv.set_add_source_note("")       # a refusal note goes with it
        self._add_run_guide_token = int(
            getattr(self, "_add_run_guide_token", 0)) + 1
        self._add_guide_thread_cleanup()

    def _add_load_run_guide(self, run_id) -> None:
        from qtpy.QtCore import QThread

        from napari_epicycle.annotator.reshape_worker import GuideLoadWorker

        src = (getattr(self, "_add_run_sources", None) or {}).get(run_id)
        main = self._add_main_label_layer()
        if src is None or main is None:
            return
        ctx = self._reshape_context()
        self._add_run_state = "loading"
        token = int(getattr(self, "_add_run_guide_token", 0)) + 1
        self._add_run_guide_token = token
        try:
            th = QThread()
            w = GuideLoadWorker(src.path, tuple(main.data.shape),
                                downsample_factor=int(
                                    ctx.get("downsample_factor") or 1))
            w._epicycle_token = token
            self._add_guide_thread = th
            self._add_guide_worker = w
            w.moveToThread(th)
            th.started.connect(w.run)
            w.finished.connect(self._on_add_guide_loaded)
            w.refused.connect(self._on_add_guide_refused)
            # an error is not a refusal: raw text to the log, a fixed
            # sentence on the page (B2 fix pass U-M4)
            w.error.connect(self._on_add_guide_error)
            th.start()
            print(f"[ADD] reading run {run_id!r} for markers")
        except Exception as exc:  # noqa: BLE001
            print(f"[ADD] run read failed to start: {exc!r}")
            self._add_run_state = "refused"
            self._add_guide_thread = None
            self._add_guide_worker = None

    def _add_guide_is_stale(self) -> bool:
        w = self._signal_sender() or getattr(self, "_add_guide_worker", None)
        tok = getattr(w, "_epicycle_token", None)
        return tok is not None and tok != getattr(
            self, "_add_run_guide_token", 0)

    def _on_add_guide_loaded(self, array, _notes=None) -> None:
        stale = self._add_guide_is_stale()
        self._add_guide_thread_cleanup()
        if stale or not getattr(self, "_add_running", False):
            return
        self._add_run_guide = array
        self._add_run_state = None
        self._add_info = None
        self._request_add_box_info()
        self._add_refresh_ui(run_count=())

    def _on_add_guide_refused(self, message) -> None:
        stale = self._add_guide_is_stale()
        self._add_guide_thread_cleanup()
        if stale:
            return
        print(f"[ADD] run refused: {message}")
        from packages.epicycle_annotator.add_preview import plain_reason

        self._add_run_refused(plain_reason(message))

    def _on_add_guide_error(self, message) -> None:
        """The guide worker RAISED: the raw text (a traceback) to the log,
        ``ADD_RUN_READ_FAILED`` on the page (B2 fix pass U-M4)."""
        stale = self._add_guide_is_stale()
        self._add_guide_thread_cleanup()
        if stale:
            return
        print(f"[ADD] run read FAILED: {message}")
        from packages.epicycle_annotator.add_preview import (
            ADD_RUN_READ_FAILED,
        )

        self._add_run_refused(ADD_RUN_READ_FAILED)

    def _add_run_refused(self, note) -> None:
        """The run source cannot be used: S1 reads ADD_S1_RUN_UNUSABLE and
        the reason is KEPT in the Markers help slot (the panel prefers it
        over the ordinary help until it is cleared -- B2 fix pass U-H2)."""
        self._add_run_guide = None
        self._add_run_state = "refused"
        adv = self._get_advanced_features_widget()
        if adv is not None and hasattr(adv, "set_add_source_note"):
            adv.set_add_source_note(str(note))
        self._add_refresh_ui()

    def _add_guide_thread_cleanup(self) -> None:
        th = getattr(self, "_add_guide_thread", None)
        w = getattr(self, "_add_guide_worker", None)
        self._add_guide_thread = None
        self._add_guide_worker = None
        if th is None:
            return
        try:
            th.quit()
            if th.wait(3000):
                return
        except Exception:
            return
        parked = getattr(self, "_refine_parked_threads", None)
        if parked is None:
            parked = []
            self._refine_parked_threads = parked
        for sig in ("finished", "refused", "error"):
            try:
                getattr(w, sig).disconnect()
            except Exception:
                pass
        parked.append((th, w))

    # ======================================================================
    # Add rebuild B2.5 -- click markers, the ball layer, the side views'
    # active layer, the exits
    # ======================================================================

    _ADD_BALL_LAYER = "_epicycle_add_markers"

    # --- the walls picture -------------------------------
    def _on_add_show_walls(self, checked) -> None:
        """"Show what the tool sees": the result's elevation as a placed
        crop through ``_add_temp_layer``; the intent is ``_add_walls_wanted``,
        not the box."""
        self._add_walls_wanted = bool(checked)
        if checked:
            self._show_add_walls()
        else:
            self._drop_add_walls(keep_intent=True)

    def _show_add_walls(self) -> bool:
        from packages.epicycle_annotator.add_preview import (
            WALLS_TOGGLE_NO_WALLS,
        )

        res = getattr(self, "_add_preview_res", None)
        main = self._add_main_label_layer()
        if (res is None or res.elevation is None or main is None
                or self._add_preset() != "cells"):
            self._push_add_ui(walls_toggle_enabled=False,
                              walls_toggle_reason=WALLS_TOGGLE_NO_WALLS)
            return False
        crop = np.asarray(res.elevation, np.float32)
        if crop.nbytes > WALLS_DISPLAY_BUDGET_BYTES:
            # on screen, in the legend slot, as Reshape does (B2 fix pass
            # L-L7) -- Qt shows no tooltip on a disabled tick
            from napari_epicycle.annotator.advanced_features_widget import (
                WALLS_TOGGLE_TOO_LARGE,
            )
            from packages.epicycle_annotator.add_preview import (
                ADD_WALLS_TOO_LARGE,
            )

            print("[ADD] walls picture refused: {} bytes over the {} byte "
                  "display budget".format(crop.nbytes,
                                          WALLS_DISPLAY_BUDGET_BYTES))
            self._push_add_ui(
                walls_toggle_enabled=False,
                walls_toggle_reason=chr(10).join((WALLS_TOGGLE_TOO_LARGE,
                                                  ADD_WALLS_TOO_LARGE)))
            return False
        try:
            start = np.array([int(sl.start) for sl in res.roi], float)
            mscale = np.asarray(main.scale, float)[-3:]
            mtrans = np.asarray(main.translate, float)[-3:]
            translate = tuple(mtrans + mscale * start)
            scale = tuple(mscale)
        except Exception as exc:
            print(f"[ADD] walls picture cannot be placed: {exc!r}")
            return False
        layer = self._add_temp_layer("add_walls", crop, kind="image",
                                     name="guide_add_walls",
                                     translate=translate, scale=scale)
        return layer is not None

    def _drop_add_walls(self, keep_intent=False) -> None:
        try:
            self._remove_temp_layer("add_walls")
        except Exception:
            pass
        if not keep_intent:
            self._add_walls_wanted = False

    def _add_walls_toggle_state(self):
        """(enabled, reason) for the walls tick: grey on Nuclei with
        WALLS_TOGGLE_NO_WALLS, before a result with WALLS_TOGGLE_DISABLED."""
        from napari_epicycle.annotator.advanced_features_widget import (
            WALLS_TOGGLE_DISABLED,
        )
        from packages.epicycle_annotator.add_preview import (
            WALLS_TOGGLE_NO_WALLS,
        )

        if self._add_preset() != "cells":
            return False, WALLS_TOGGLE_NO_WALLS
        res = getattr(self, "_add_preview_res", None)
        if res is None or res.elevation is None:
            return False, WALLS_TOGGLE_DISABLED
        return True, ""

    def _close_add_numbers(self) -> None:
        """Numbers... is closed at every Add exit."""
        adv = self._get_advanced_features_widget()
        if adv is not None and hasattr(adv, "close_add_numbers"):
            try:
                adv.close_add_numbers()
            except Exception:
                pass

    def _add_toggle_targets(self) -> list:
        """The Add artefacts the labels key toggles: the candidate, the ball
        layer and their side-view copies."""
        out = []
        sub_mgr = getattr(self, "subviewer_manager", None)
        for layer in (getattr(self, "_add_candidate_layer", None),
                      getattr(self, "_add_marker_layer", None)):
            if layer is None:
                continue
            out.append(layer)
            if sub_mgr is not None:
                for orient in ("XY", "XZ", "YZ"):
                    try:
                        sub = sub_mgr.get_subviewer_layer(layer.name, orient)
                    except Exception:
                        sub = None
                    if sub is not None:
                        out.append(sub)
        return out

    def _add_tab_current(self) -> bool:
        """Is the Add tab the one on screen?  Every side-view action of Add
        returns untouched when it is not."""
        adv = self._get_advanced_features_widget()
        tabs = getattr(adv, "_tool_tabs", None)
        if tabs is None:
            return True                  # no tab strip (a test / standalone)
        try:
            return tabs.tabText(tabs.currentIndex()) == "Add"
        except Exception:
            return True

    def _on_add_place_markers_toggled(self, on) -> None:
        if on and self._add_refuse_busy("place markers"):
            self._push_add_ui(place_markers_on=False)
            return
        self._add_place_markers_on = bool(on)

    def _add_place_markers_off(self) -> None:
        self._add_place_markers_on = False
        self._push_add_ui(place_markers_on=False)

    def _add_ball_radius_vox(self):
        """The click ball radius per axis, in voxels (at the true spacing)."""
        params = self._add_params()
        r = float(params.get("click_radius_um", 2.0))
        sp = self._refine_spacing()
        return tuple(max(0.5, r / max(float(s), 1e-6)) for s in sp)

    # --- A5: marker records, reserved ids, ticks ---------------
    def _add_markers(self) -> list:
        """The click markers as records ``{"zyx", "id", "ticked"}`` in
        placement order.  Identity is the reserved id,
        never the list index.  A bare ``(z, y, x)`` left by an older caller
        is normalised here, reserving its id as a click would."""
        raw = list(getattr(self, "_add_click_markers", None) or [])
        if all(isinstance(m, dict) for m in raw):
            return self._add_click_markers if raw else raw
        out = []
        for m in raw:
            if isinstance(m, dict):
                out.append(m)
                continue
            zyx = tuple(int(round(float(v))) for v in m)
            mid, _rebased = self._add_reserve_marker_id(out)
            out.append({"zyx": zyx, "id": mid, "ticked": True})
        self._add_click_markers = out
        return out

    def _add_reserve_marker_id(self, markers):
        """``(id, rebased)`` for the next marker: ``max(next_label,
        max(reserved) + 1)`` with the same ``_add_next_label()`` Preview
        calls.  The existing ids are re-based first when
        next_label moved since they were reserved."""
        nl = int(self._add_next_label(cached=True))
        rebased = self._add_rebase_markers(nl, markers)
        top = max([int(m["id"]) for m in markers] or [0])
        self._add_markers_base = nl
        return max(nl, top + 1), rebased

    def _add_rebase_markers(self, next_label, markers=None) -> bool:
        """Shift EVERY reserved id by ``next_label - base`` (order and gaps
        kept) when next_label moved since the ids were reserved -- the df
        or the volume changed.  True when anything shifted."""
        markers = self._add_click_markers if markers is None else markers
        base = getattr(self, "_add_markers_base", None)
        nl = int(next_label)
        if not markers or base is None:
            self._add_markers_base = nl if markers else None
            return False
        d = nl - int(base)
        self._add_markers_base = nl
        if d == 0:
            return False
        for m in markers:
            m["id"] = int(m["id"]) + d
        absorbed = getattr(self, "_add_absorbed_ids", None)
        if absorbed:
            # the last Preview's absorbed markers keep their marks (B-M1)
            self._add_absorbed_ids = {int(i) + d for i in absorbed}
        self._add_bump_markers_epoch()
        # S1 says why every colour just changed (B-M2): taken as a
        # transient by the caller AFTER its own invalidation, which would
        # otherwise clear it (_add_take_rebase_note)
        self._add_rebase_note = (d, len(markers))
        print("[ADD] marker ids re-based by {:+d}: {}".format(
            d, [int(m["id"]) for m in markers]))
        return True

    def _add_take_rebase_note(self) -> None:
        """A pending re-base becomes S1's transient ``ADD_S1_REBASED``
        (B-M2 of the A5 audit)."""
        note = getattr(self, "_add_rebase_note", None)
        self._add_rebase_note = None
        if note is None:
            return
        from packages.epicycle_annotator.add_preview import rebased_line

        self._add_transient = rebased_line(*note)

    def _add_bump_markers_epoch(self) -> None:
        """Every place / remove / undo / clear / box-move drop / tick change
       : a finish launched on other markers is stale."""
        self._add_markers_epoch = int(getattr(self, "_add_markers_epoch", 0)) + 1

    def _add_marker_push(self) -> dict:
        """The marker list as the panel shows it (a VIEW,  M4),
        with the last Preview's absorbed markers marked (B-M1)."""
        markers = self._add_markers()
        ids = {int(m["id"]) for m in markers}
        absorbed = {int(i) for i in (getattr(self, "_add_absorbed_ids", None)
                                     or ())} & ids
        return dict(click_count=len(markers),
                    click_markers=[(int(m["id"]), tuple(m["zyx"]),
                                    bool(m["ticked"])) for m in markers],
                    click_absorbed_ids=sorted(absorbed))

    def _add_in_ball(self, p, m, rv=None) -> bool:
        rv = self._add_ball_radius_vox() if rv is None else rv
        return sum(((a - b) / r) ** 2
                   for a, b, r in zip(p, m["zyx"], rv)) <= 1.0

    def _add_marker_click(self, zyx, ctrl=False) -> bool:
        """One click in a side view (B3 calls this).  Returns True
        when a marker was placed or removed."""
        from packages.epicycle_annotator import add_preview as AP

        if (not getattr(self, "_add_running", False)
                or getattr(self, "_add_review", False)
                or getattr(self, "_add_box", None) is None):
            return False
        if not self._add_tab_current():
            return False
        if not getattr(self, "_add_place_markers_on", False):
            return False
        if self._add_refuse_busy("marker click"):
            return False
        p = tuple(int(round(float(v))) for v in zyx)
        if ctrl:
            return self._add_remove_marker_near(p)
        roi = self._add_roi
        inside = all(sl.start <= v < sl.stop for sl, v in zip(roi, p))
        if not inside:
            self._add_transient = AP.ADD_S1_CLICK_OUTSIDE
            self._add_refresh_ui()
            return False
        shared = self._add_shared_volume()
        kept = int(shared[p]) if shared is not None else 0
        if kept != 0:
            self._add_transient = AP.ADD_S1_CLICK_ON_KEPT.format(id=kept)
            self._add_refresh_ui()
            return False
        markers = self._add_markers()
        rv = self._add_ball_radius_vox()
        for m in markers:
            # ticked or not: rasterise_points lets the EARLIER ball keep a
            # shared voxel, so a click here would be shadowed and every
            # later id would shift
            if self._add_in_ball(p, m, rv):
                last = getattr(self, "_add_last_placed", None)
                if (last is not None and last[0] is m
                        and _time.monotonic() - last[1]
                        <= self._add_double_click_s()):
                    # the second press of a double click lands in the
                    # marker its first press just placed: refused quietly,
                    # not an amber sentence (C-L3, the A5 UX audit)
                    print("[ADD] double click's second press ignored - "
                          "marker {} was just placed".format(int(m["id"])))
                    return False
                tpl = (AP.ADD_S1_CLICK_SAME_SPOT if m["ticked"]
                       else AP.ADD_S1_CLICK_SAME_SPOT_UNTICKED)
                self._add_transient = tpl.format(id=int(m["id"]))
                self._add_refresh_ui()
                return False
        mid, rebased = self._add_reserve_marker_id(markers)
        rec = {"zyx": p, "id": int(mid), "ticked": True}
        markers.append(rec)
        self._add_click_markers = markers
        self._add_last_placed = (rec, _time.monotonic())
        print(f"[ADD] marker {rec['id']} at {p}")
        self._add_bump_markers_epoch()
        self._add_after_click_change(changed=None if rebased else [rec])
        return True

    def _add_double_click_s(self) -> float:
        """Qt's double-click interval, in seconds (C-L3); 0.4 s when no
        application is up."""
        try:
            from qtpy.QtWidgets import QApplication

            if QApplication.instance() is not None:
                return max(0.0, QApplication.doubleClickInterval() / 1000.0)
        except Exception:  # noqa: BLE001
            pass
        return 0.4

    def _add_remove_marker_near(self, p) -> bool:
        """Ctrl+click: the nearest DRAWN (ticked) marker whose ball contains
        the point (an unticked ball is not on screen)."""
        rv = self._add_ball_radius_vox()
        markers = self._add_markers()
        best, best_d = None, None
        for i, m in enumerate(markers):
            if not m["ticked"]:
                continue
            d = sum(((a - b) / r) ** 2 for a, b, r in zip(p, m["zyx"], rv))
            if d <= 1.0 and (best_d is None or d < best_d):
                best, best_d = i, d
        if best is None:
            return False
        gone = markers.pop(best)
        print(f"[ADD] marker {gone['id']} removed")
        self._add_after_marker_removal([gone])
        return True

    def _add_click_undo(self, _checked=False) -> None:
        if self._add_refuse_busy("undo marker"):
            return
        markers = self._add_markers()
        if markers:
            gone = markers.pop()
            self._add_after_marker_removal([gone])

    def _add_click_clear(self, _checked=False) -> None:
        if self._add_refuse_busy("clear markers"):
            return
        markers = self._add_markers()
        if markers:
            gone = list(markers)
            self._add_click_markers = []
            self._add_after_marker_removal(gone)

    def _add_after_marker_removal(self, gone) -> None:
        if not self._add_click_markers:
            self._add_markers_base = None      # Clear / the last one: reset
        self._add_bump_markers_epoch()
        self._add_after_click_change(changed=gone)

    def _on_add_click_tick(self, marker_id, ticked) -> None:
        """A row ticked or unticked in the marker list.  The
        tick lives HERE, keyed by the reserved id; the list is a view."""
        if not getattr(self, "_add_running", False):
            # the row the user ticked is put back as the orchestrator
            # holds it -- a view never keeps a tick nobody owns (B-L4)
            self._push_add_ui(**self._add_marker_push())
            return
        if self._add_refuse_busy("marker tick"):
            # nothing moves: the list is pushed back as it was
            self._add_refresh_ui(**self._add_marker_push())
            return
        rec = next((m for m in self._add_markers()
                    if int(m["id"]) == int(marker_id)), None)
        if rec is None or bool(rec["ticked"]) == bool(ticked):
            self._push_add_ui(**self._add_marker_push())
            return
        rec["ticked"] = bool(ticked)
        print("[ADD] marker {} {}".format(
            rec["id"], "ticked" if rec["ticked"] else "unticked"))
        self._add_bump_markers_epoch()
        # the list is hidden over candidates, so there is none to drop in
        # practice -- the call is the invariant, not the hope
        self._invalidate_add_preview()
        self._render_add_balls(changed=[rec])
        self._add_refresh_ui(**self._add_marker_push())

    def _add_after_click_change(self, changed=None) -> None:
        self._invalidate_add_preview()
        self._add_take_rebase_note()       # after the invalidation (B-M2)
        self._render_add_balls(changed=changed)
        self._add_refresh_ui(**self._add_marker_push())

    def _add_keep_clicks_in_box(self) -> None:
        """A box move keeps the clicks inside the new box and drops the
        others -- unticked ones too -- saying so."""
        from packages.epicycle_annotator import add_preview as AP

        roi = getattr(self, "_add_roi", None)
        markers = self._add_markers()
        if roi is None or not markers:
            return
        keep = [m for m in markers
                if all(sl.start <= v < sl.stop for sl, v in zip(roi, m["zyx"]))]
        gone = [m for m in markers if m not in keep]
        self._add_click_markers = keep
        n = len(gone)
        if n > 0:
            if not keep:
                self._add_markers_base = None
            self._add_bump_markers_epoch()
            # a pure box move never moves a ball: the ball layer is
            # touched only when a click was dropped (B3 fix pass --
            # Shift+arrow autorepeat re-rendered it per key), and then
            # only the dropped balls (invariant 23)
            self._render_add_balls(changed=gone)
        if n:
            self._add_transient = AP.count_phrase(
                n, AP.ADD_S1_CLICKS_DROPPED_ONE, AP.ADD_S1_CLICKS_DROPPED)
        self._add_refresh_ui(**self._add_marker_push())

    # --- the ball layer ----------------------------------------------------
    def _add_ball_layer(self):
        return getattr(self, "_add_marker_layer", None)

    def _add_ball_region(self, zyx, rv, shape):
        """``(slices, mask)``: a marker's ball inside its bounding box."""
        sl = tuple(slice(max(0, int(np.floor(c - r))),
                         min(n, int(np.ceil(c + r)) + 1))
                   for c, r, n in zip(zyx, rv, shape))
        grid = np.ogrid[sl]
        ball = sum(((g - c) / r) ** 2 for g, c, r in zip(grid, zyx, rv)) <= 1.0
        return sl, ball

    def _render_add_balls(self, changed=None) -> None:
        """The click balls on a full-volume uint8 Labels, built, FLAGGED,
        locked and only then appended; mirrored into the side views as
        views of the same array; the side views' active layer put back
        (grills E-L10 / E-M11).

        a ball's value is ``ball_value(id)`` -- the colour
        of the object it will become -- written only where the layer is 0
        (the earlier ball keeps a shared voxel, as the engine does), and
        an UNTICKED ball is not drawn.  ``changed`` (records whose ball was
        added, removed or re-ticked) rewrites only the balls it touches:
        those, and every ball whose box overlaps them (to a fixed point),
        erased and redrawn in placement order -- the same voxels a full
        redraw gives, without zeroing a 73 MB array per tick (invariant
        23).  ``None`` redraws everything (a new layer, a re-base)."""
        from packages.epicycle_annotator.add_preview import ball_value

        main = self._add_main_label_layer()
        if main is None:
            return
        markers = self._add_markers()
        layer = self._add_ball_layer()
        if layer is None:
            if not markers:
                return
            layer = self._build_add_ball_layer(main)
            if layer is None:
                return
            changed = None                     # a new layer: draw it all
        data = layer.data
        rv = self._add_ball_radius_vox()
        regions = {}

        def region(m):
            key = id(m)
            if key not in regions:
                regions[key] = self._add_ball_region(m["zyx"], rv, data.shape)
            return regions[key]

        if changed is None:
            data[...] = 0
            redraw = list(markers)
        else:
            def overlaps(a, b):
                return all(x.start < y.stop and y.start < x.stop
                           for x, y in zip(a, b))

            touched = list(changed)
            boxes = [region(m)[0] for m in touched]
            grow = True
            while grow:
                grow = False
                for m in markers:
                    if any(m is t for t in touched):
                        continue
                    if any(overlaps(region(m)[0], b) for b in boxes):
                        touched.append(m)
                        boxes.append(region(m)[0])
                        grow = True
            for m in touched:
                sl, ball = region(m)
                view = data[sl]
                view[ball] = 0
            redraw = [m for m in markers if any(m is t for t in touched)]
        for m in redraw:
            if not m.get("ticked", True):
                continue
            sl, ball = region(m)
            view = data[sl]
            view[ball & (view == 0)] = ball_value(m["id"])
        self._refresh_add_ball_views()
        self._add_update_ball_visibility()

    def _build_add_ball_layer(self, main):
        from packages.epicycle_annotator.sync import lock_brush_size_to_1

        try:
            layer = Labels(np.zeros(main.data.shape, np.uint8),
                           name=self._ADD_BALL_LAYER, opacity=0.9)
            layer._epicycle_temporary = True          # flag, THEN append
            layer.editable = False
            try:
                layer.scale = main.scale
                layer.translate = main.translate
            except Exception:
                pass
            lock_brush_size_to_1(layer)
            self.viewer.layers.append(layer)
            self._add_marker_layer = layer
        except Exception as exc:
            print(f"[ADD] ball layer failed: {exc!r}")
            return None
        try:
            self.subviewer_manager.add_labels_to_subviewers(
                layer, sync_manager=self.sync_manager,
                drawing_handler=self.drawing_handler)
            for orient in ("XY", "XZ", "YZ"):
                sub = self.subviewer_manager.get_subviewer_layer(
                    layer.name, orient)
                if sub is not None:
                    try:
                        sub.editable = False
                        sub.mode = "pan_zoom"
                    except Exception:
                        pass
                    if not np.shares_memory(sub.data, layer.data):
                        print(f"[ADD] ball copy {orient} is NOT a view of "
                              "the main array")
        except Exception as exc:
            print(f"[ADD] ball mirror failed: {exc!r}")
        self._restore_side_active_layers()
        try:
            self.viewer.layers.selection.active = main
        except Exception:
            pass
        return layer

    def _refresh_add_ball_views(self) -> None:
        layer = self._add_ball_layer()
        if layer is None:
            return
        targets = [layer]
        sub_mgr = getattr(self, "subviewer_manager", None)
        if sub_mgr is not None:
            for orient in ("XY", "XZ", "YZ"):
                try:
                    sub = sub_mgr.get_subviewer_layer(layer.name, orient)
                except Exception:
                    sub = None
                if sub is not None:
                    targets.append(sub)
        for t in targets:
            try:
                t.refresh()
            except Exception:
                pass

    def _add_update_ball_visibility(self) -> None:
        """Cells -> Nuclei HIDES the balls (the clicks are kept); Nuclei ->
        Cells shows them again."""
        layer = self._add_ball_layer()
        if layer is None:
            return
        show = self._add_preset() == "cells"
        targets = [layer]
        sub_mgr = getattr(self, "subviewer_manager", None)
        if sub_mgr is not None:
            for orient in ("XY", "XZ", "YZ"):
                try:
                    sub = sub_mgr.get_subviewer_layer(layer.name, orient)
                except Exception:
                    sub = None
                if sub is not None:
                    targets.append(sub)
        for t in targets:
            try:
                t.visible = show
            except Exception:
                pass

    def _remove_add_marker_layer(self) -> None:
        """A twin of ``_remove_add_candidate``: disconnect the sync, the side
        copies, unregister, hide, remove.  Idempotent."""
        layer = getattr(self, "_add_marker_layer", None)
        self._add_marker_layer = None
        if layer is None:
            return
        name = getattr(layer, "name", self._ADD_BALL_LAYER)
        sm = getattr(self, "sync_manager", None)
        if sm is not None:
            try:
                sm.disconnect_label_sync(layer)
            except Exception:
                pass
        try:
            self.subviewer_manager.remove_layer_from_subviewers(name)
        except Exception:
            pass
        if sm is not None:
            try:
                sm.unregister_layer(name)
            except Exception:
                pass
        try:
            layer.visible = False
            if layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
        except Exception:
            pass

    def _restore_side_active_layers(self) -> None:
        """napari makes an appended layer ACTIVE and runs mouse callbacks
        for the active layer only, so right-click select dies in a side
        view whose active layer is a mirrored Add copy (grill E-M11).  Put
        each side view's active layer back to the main labels copy."""
        sub_mgr = getattr(self, "subviewer_manager", None)
        name = getattr(self, "_active_labels_name", "")
        if sub_mgr is None or not name:
            return
        for orient, attr in (("XY", "viewer_model_xy"),
                             ("XZ", "viewer_model_xz"),
                             ("YZ", "viewer_model_yz")):
            vm = getattr(self, attr, None)
            if vm is None:
                continue
            try:
                sub = sub_mgr.get_subviewer_layer(name, orient)
                if sub is not None and sub in vm.layers:
                    vm.layers.selection.active = sub
            except Exception as exc:
                print(f"[ADD] active-layer restore {orient}: {exc!r}")

    # --- the side-view callbacks -------------------------------
    # One drag generator and one hover callback per side ``ViewerModel``,
    # VIEWER-level (napari runs them before the active layer's).  Button 1
    # only; a drag starts only on the box outline; a press elsewhere with
    # Place markers on is a click candidate; anything else returns with
    # ``handled`` untouched, so napari pans -- and right-click select
    # (button 2, on the side LAYERS' own callbacks) is never seen here.
    _ADD_SIDE_VIEWS = (("XY", "viewer_model_xy", "qt_viewer_xy"),
                       ("XZ", "viewer_model_xz", "qt_viewer_xz"),
                       ("YZ", "viewer_model_yz", "qt_viewer_yz"))

    def _install_add_side_callbacks(self) -> None:
        """At Start and "+ Add another".  Idempotent: a second call never
        installs anything twice."""
        self._remove_add_side_callbacks()
        cbs = {}
        extra = {}
        for orient, vm_attr, _qtv in self._ADD_SIDE_VIEWS:
            vm = getattr(self, vm_attr, None)
            if vm is None:
                continue

            def _drag(obj, event, _o=orient):
                return self._add_side_drag(_o, obj, event)

            def _hover(obj, event, _o=orient):
                self._add_side_hover(_o, obj, event)

            try:
                vm.mouse_drag_callbacks.append(_drag)
                vm.mouse_move_callbacks.append(_hover)
            except Exception as exc:
                print(f"[ADD] side callbacks {orient}: {exc!r}")
                continue
            cbs[orient] = (vm, _drag, _hover)
            extra[orient] = self._install_add_side_extras(orient, vm)
        self._add_side_cbs = cbs
        self._add_side_extra = extra
        self._add_side_callbacks_installed = True
        print(f"[ADD] side callbacks installed on {sorted(cbs)}")

    def _remove_add_side_callbacks(self) -> None:
        """Idempotent; decoupled from ``_remove_add_box_shapes`` (grill
        E-M9): REVIEW removes the callbacks and keeps the overlay.  Also
        ends a drag in flight (its token), puts the camera's pan back and
        the cursors."""
        self._add_drag_token = getattr(self, "_add_drag_token", 0) + 1
        self._add_end_drag_state()
        self._add_clear_side_cursors()
        for orient, (vm, drag, hover) in dict(
                getattr(self, "_add_side_cbs", None) or {}).items():
            for lst_name, cb in (("mouse_drag_callbacks", drag),
                                 ("mouse_move_callbacks", hover)):
                try:
                    lst = getattr(vm, lst_name)
                    if cb in lst:
                        lst.remove(cb)
                except Exception as exc:
                    print(f"[ADD] side callback remove {orient}: {exc!r}")
            self._add_close_drag_gen(vm, drag)
        for orient, ex in dict(
                getattr(self, "_add_side_extra", None) or {}).items():
            self._remove_add_side_extras(orient, ex)
        self._add_side_cbs = {}
        self._add_side_extra = {}
        self._add_side_callbacks_installed = False

    def _add_close_drag_gen(self, vm, drag) -> None:
        """A drag generator napari still holds (LOW-1, B3 fix pass): pop
        it AND its persisted event, then ``close()`` it so its ``finally``
        runs (the real box is redrawn)."""
        try:
            gen = getattr(vm, "_mouse_drag_gen", {}).pop(drag, None)
        except Exception:
            gen = None
        if gen is None:
            return
        try:
            getattr(vm, "_persisted_mouse_event", {}).pop(gen, None)
        except Exception:
            pass
        try:
            gen.close()
        except Exception as exc:       # e.g. closed from inside itself
            print(f"[ADD] drag generator close: {exc!r}")

    def _install_add_side_extras(self, orient, vm) -> dict:
        """The double-click route (LOW-5) and the zoom listener (LOW-6),
        installed and removed with the drag callbacks (cb).

        napari delivers a second fast click as ``mouse_double_click`` with
        no press, and its viewer-level ``double_click_to_zoom`` zooms 2x.
        The side view's ``double_click_to_zoom`` is SWAPPED for a wrapper
        that, while Place markers is on, places the marker instead and
        otherwise calls the original; removal puts the original back.
        napari's ``_on_cursor`` resets the canvas cursor on a zoom, so the
        hover flag is cleared there."""
        ex = {"vm": vm, "dbl": None, "orig_dbl": None, "zoom": None,
              "slice": None}
        try:
            lst = vm.mouse_double_click_callbacks
            orig = None
            for cb in list(lst):
                if getattr(cb, "__name__", "") == "double_click_to_zoom":
                    orig = cb
                    break

            def _dbl(obj, event, _o=orient, _orig=orig):
                if self._add_side_double_click(_o, obj, event):
                    return
                if _orig is not None:
                    _orig(obj, event)

            if orig is not None:
                lst[lst.index(orig)] = _dbl
            else:
                lst.append(_dbl)
            ex["dbl"], ex["orig_dbl"] = _dbl, orig
        except Exception as exc:
            print(f"[ADD] side double-click {orient}: {exc!r}")

        def _zoomed(_event=None, _o=orient):
            shown = getattr(self, "_add_side_cursor", None)
            if shown:
                shown[_o] = False

        try:
            vm.camera.events.zoom.connect(_zoomed)
            ex["zoom"] = _zoomed
        except Exception as exc:
            print(f"[ADD] side zoom listener {orient}: {exc!r}")

        def _sliced(_event=None, _o=orient):     # dim / undim (LOW-4)
            self._add_update_side_dim(_o)

        try:
            vm.dims.events.point.connect(_sliced)
            ex["slice"] = _sliced
        except Exception as exc:
            print(f"[ADD] side slice listener {orient}: {exc!r}")
        self._add_update_side_dim(orient)
        return ex

    def _remove_add_side_extras(self, orient, ex) -> None:
        vm = ex.get("vm")
        dbl, orig = ex.get("dbl"), ex.get("orig_dbl")
        if vm is None:
            return
        if dbl is not None:
            try:
                lst = vm.mouse_double_click_callbacks
                if dbl in lst:
                    if orig is not None:
                        lst[lst.index(dbl)] = orig
                    else:
                        lst.remove(dbl)
            except Exception as exc:
                print(f"[ADD] side double-click remove {orient}: {exc!r}")
        if ex.get("zoom") is not None:
            try:
                vm.camera.events.zoom.disconnect(ex["zoom"])
            except Exception:
                pass
        if ex.get("slice") is not None:
            try:
                vm.dims.events.point.disconnect(ex["slice"])
            except Exception:
                pass
        # no drag without the callbacks (REVIEW keeps the drawing): the
        # rectangle goes back to full colour
        overlay = getattr(self, "_add_box_overlay", None)
        if overlay is not None and hasattr(overlay, "set_in_range"):
            try:
                overlay.set_in_range(orient, True)
            except Exception:
                pass

    def _add_side_slice(self, orient):
        """The side view's current slice as a voxel index on its slicing
        axis (identity side-layer scale, as ``outline_tolerance``), or
        None."""
        for o, vm_attr, _q in self._ADD_SIDE_VIEWS:
            if o != orient:
                continue
            vm = getattr(self, vm_attr, None)
            try:
                axis = tuple(vm.dims.not_displayed)[-1]
                return float(vm.dims.point[axis])
            except Exception:
                return None
        return None

    def _add_side_in_range(self, box, shape) -> dict:
        """``{orient: slice inside the box's range}`` for the side views,
        only while the drag callbacks are live (REVIEW draws full
        colour)."""
        from packages.epicycle_annotator import box_drag as BD

        if box is None or not getattr(self, "_add_side_callbacks_installed",
                                      False):
            return {}
        try:
            b = box.bounds_dict(shape)
        except Exception:
            return {}
        out = {}
        for o, _vm, _q in self._ADD_SIDE_VIEWS:
            out[o] = BD.slice_in_box(b, o, self._add_side_slice(o))
        return out

    def _add_update_side_dim(self, orient) -> None:
        overlay = getattr(self, "_add_box_overlay", None)
        box = getattr(self, "_add_box", None)
        shape = self._add_volume_shape()
        if overlay is None or box is None or shape is None \
                or not hasattr(overlay, "set_in_range"):
            return
        from packages.epicycle_annotator import box_drag as BD

        try:
            overlay.set_in_range(orient, BD.slice_in_box(
                box.bounds_dict(shape), orient, self._add_side_slice(orient)))
        except Exception as exc:
            print(f"[ADD] side dim {orient}: {exc!r}")

    def _add_side_double_click(self, orient, vm, event) -> bool:
        """True when Add took the double-click (Place markers on): the
        second fast click is a marker click, never a 2x zoom."""
        if getattr(event, "button", 1) not in (None, 1):
            return False
        if not (self._add_side_ready()
                and getattr(self, "_add_place_markers_on", False)):
            return False
        zyx = self._add_event_zyx(orient, vm, event)
        if zyx is None:
            return False
        try:
            event.handled = True
        except Exception:
            pass
        print(f"[ADD] double-click in {orient} -> marker click")
        self._add_defer_marker_click(getattr(self, "_add_drag_token", 0),
                                     zyx, self._add_event_ctrl(event))
        return True

    def _add_side_ready(self) -> bool:
        """A box session on the Add tab, outside REVIEW (inv. 17)."""
        return bool(getattr(self, "_add_running", False)
                    and not getattr(self, "_add_review", False)
                    and getattr(self, "_add_box", None) is not None
                    and self._add_tab_current())

    def _add_volume_shape(self):
        main = self._add_main_label_layer()
        if main is None:
            return None
        try:
            return tuple(int(v) for v in main.data.shape[-3:])
        except Exception:
            return None

    def _add_event_zyx(self, orient, vm, event):
        """The event's world position in that side view, back in volume
        ZYX voxels (its last three entries; the side labels layer's own
        scale / translate undone -- identity in practice)."""
        from packages.epicycle_annotator.orientation import (
            subview_point_to_zyx,
        )

        pos = getattr(event, "position", None)
        try:
            if pos is None or len(tuple(pos)) < 3:
                return None
        except TypeError:
            return None
        scale = translate = None
        sub_mgr = getattr(self, "subviewer_manager", None)
        name = getattr(self, "_active_labels_name", "")
        if sub_mgr is not None and name:
            try:
                sub = sub_mgr.get_subviewer_layer(name, orient)
                if sub is not None:
                    scale = tuple(sub.scale)
                    translate = tuple(sub.translate)
            except Exception:
                scale = translate = None
        try:
            return subview_point_to_zyx(orient, pos, scale=scale,
                                        translate=translate)
        except Exception:
            return None

    def _add_event_px(self, event):
        pos = getattr(event, "pos", None)
        try:
            return (float(pos[0]), float(pos[1]))
        except Exception:
            return None

    def _add_defer(self, fn) -> None:
        """Out of vispy's mouse dispatch (the 2026-09-07 right-click
        lesson): a layer or overlay change runs on the next event-loop
        turn."""
        from qtpy.QtCore import QTimer

        QTimer.singleShot(0, fn)

    def _add_widget_alive(self) -> bool:
        try:
            from qtpy import sip as _sip

            return not bool(_sip.isdeleted(self))
        except Exception:
            return True

    def _add_side_drag(self, orient, vm, event):
        """The drag generator for one side view (napari's protocol: code up
        to the first ``yield`` runs on press, the loop on each move, the
        tail on release)."""
        from packages.epicycle_annotator import box_drag as BD

        if getattr(event, "button", None) != 1:
            return
        if not self._add_side_ready():
            return
        zyx = self._add_event_zyx(orient, vm, event)
        shape = self._add_volume_shape()
        if zyx is None or shape is None:
            return
        box0 = self._add_box
        bounds = box0.bounds_dict(shape)
        tol = self._add_outline_tol(vm, bounds, orient)
        if BD.outline_hit(bounds, orient, zyx, tol):
            if self._add_refuse_busy("box drag"):
                yield from self._add_refused_press(event)
                return
            yield from self._add_drag_box(orient, vm, event, box0, zyx,
                                          shape)
            return
        if getattr(self, "_add_place_markers_on", False):
            yield from self._add_click_candidate(orient, vm, event, zyx)
        # else: untouched -- napari pans

    def _add_outline_tol(self, vm, bounds, orient) -> float:
        """The grab band: ``6 / zoom`` voxels, capped at a quarter of the
        box's smaller displayed extent (B3 fix pass) so a zoomed-out view
        still leaves the box's middle to marker clicks."""
        from packages.epicycle_annotator import box_drag as BD

        return BD.outline_tolerance(
            getattr(getattr(vm, "camera", None), "zoom", None),
            max_vox=BD.outline_cap(bounds, orient))

    def _add_refused_press(self, event):
        """An outline press while busy (L3): refused, and every event of it
        is handled so napari does not pan either."""
        try:
            event.handled = True
        except Exception:
            pass
        yield
        while getattr(event, "type", None) == "mouse_move":
            event.handled = True
            yield
        event.handled = True

    def _add_event_ctrl(self, event) -> bool:
        try:
            return "Control" in (getattr(event, "modifiers", None) or ())
        except Exception:
            return False

    def _add_pan_truth(self, vm) -> bool:
        """What napari's own state says the camera's pan should be: the
        active layer's ``mouse_pan``, or True with no active layer (napari's
        ``_on_active_layer``).  Never a value captured at press (a lost
        release left it False for good)."""
        try:
            active = vm.layers.selection.active
        except Exception:
            active = None
        if active is None:
            return True
        return bool(getattr(active, "mouse_pan", True))

    def _add_drag_box(self, orient, vm, event, box0, zyx0, shape):
        from packages.epicycle_annotator import box_drag as BD

        st = getattr(self, "_add_drag_state", None)
        if st:
            # a drag whose release was lost (focus stolen): end it first --
            # its generator closed (napari is still holding it, with its
            # persisted event), pan back from napari's truth, the real box
            # redrawn
            print("[ADD] stale box drag ended before a new one")
            cb = (getattr(self, "_add_side_cbs", None) or {}).get(
                st.get("orient"))
            if cb is not None:
                self._add_close_drag_gen(st.get("vm"), cb[1])
            self._add_end_drag_state()
            try:
                self._render_add_box()
            except Exception:
                pass
        token = getattr(self, "_add_drag_token", 0) + 1
        self._add_drag_token = token
        cam = getattr(vm, "camera", None)
        event.handled = True
        px0 = self._add_event_px(event)
        ctrl = self._add_event_ctrl(event)
        try:          # the named fallback: no pan for this drag
            cam.mouse_pan = False
        except Exception:
            pass
        self._add_drag_state = {"orient": orient, "vm": vm,
                                "token": token}
        print(f"[ADD] box drag start in {orient} at "
              f"{tuple(round(float(v), 1) for v in zyx0)}")
        last = box0
        moved = False
        done = False
        try:
            yield
            while getattr(event, "type", None) == "mouse_move":
                if not self._add_drag_alive(token):
                    return
                event.handled = True
                z1 = self._add_event_zyx(orient, vm, event)
                if z1 is not None:
                    d = BD.drag_delta(orient, zyx0, z1)
                    # keep-size, two displayed axes only (B3 fix pass); a
                    # zero delta is no move at all, never a commit
                    nb = (BD.moved_box(box0, d, shape,
                                       axes=BD.view_axes(orient)[:2])
                          if any(d) else box0)
                    if nb != last:
                        last = nb
                        moved = True
                        self._add_drag_redraw(nb, shape)
                yield
            event.handled = True             # the release
            if getattr(event, "button", 1) not in (None, 1):
                # another button's release (LOW-2): napari advances every
                # generator on it and then drops ours -- the left drag is
                # over, and it commits nothing
                print("[ADD] box drag ended by another button - not moved")
                return
            if not self._add_drag_alive(token):
                return
            done = True
            if moved and last != box0:
                final = last
                self._add_defer(
                    lambda: self._add_finish_drag(token, final))
            elif (not moved
                  and getattr(self, "_add_place_markers_on", False)
                  and BD.is_click(px0, self._add_event_px(event))):
                # a click in the grab band is still a marker click
                self._add_defer_marker_click(token, zyx0, ctrl)
        finally:
            st = getattr(self, "_add_drag_state", None)
            if st is not None and st.get("token") == token:
                self._add_end_drag_state()
            if moved and not done:
                # an interrupted drag: the drawing goes back to the real box
                try:
                    if getattr(self, "_add_running", False):
                        self._render_add_box()
                except Exception:
                    pass

    def _add_click_candidate(self, orient, vm, event, zyx0):
        """A press with Place markers on: a click when the release is within
        ``ADD_CLICK_DRAG_PX`` of the press, else it was a pan (``handled``
        is never set here, so napari pans).  The click itself runs on the
        next event-loop turn, behind the drag token."""
        from packages.epicycle_annotator import box_drag as BD

        px0 = self._add_event_px(event)
        ctrl = self._add_event_ctrl(event)
        token = getattr(self, "_add_drag_token", 0)
        yield
        while getattr(event, "type", None) == "mouse_move":
            yield
        if not BD.is_click(px0, self._add_event_px(event)):
            return
        self._add_defer_marker_click(token, zyx0, ctrl)

    def _add_defer_marker_click(self, token, zyx, ctrl) -> None:
        """The marker click, one event-loop turn later, behind *token*."""
        def _fire():
            if (token != getattr(self, "_add_drag_token", 0)
                    or not self._add_widget_alive()
                    or not self._add_side_ready()):
                print("[ADD] marker click dropped (session moved on)")
                return
            self._add_marker_click(zyx, ctrl=ctrl)

        self._add_defer(_fire)

    def _add_drag_alive(self, token) -> bool:
        return bool(token == getattr(self, "_add_drag_token", None)
                    and not getattr(self, "_add_busy", False)
                    and self._add_side_ready())

    def _add_drag_redraw(self, box, shape) -> None:
        """The three side rectangles follow the drag; the 3D wireframe
        waits for the release ("Live redraw")."""
        overlay = getattr(self, "_add_box_overlay", None)
        if overlay is None:
            return
        try:
            overlay.update(box, shape, keys=("XY", "XZ", "YZ"),
                           in_range=self._add_side_in_range(box, shape))
        except Exception as exc:
            print(f"[ADD] drag redraw failed: {exc!r}")

    def _add_finish_drag(self, token, box) -> None:
        """The release, one event-loop turn later, behind the drag token and
        a liveness check -- then Recenter's path."""
        if (token != getattr(self, "_add_drag_token", None)
                or not self._add_widget_alive()
                or getattr(self, "_add_box_overlay", None) is None
                or not self._add_side_ready()):
            print("[ADD] box drag release dropped (session moved on)")
            if (getattr(self, "_add_running", False)
                    and getattr(self, "_add_box_overlay", None) is not None
                    and self._add_widget_alive()):
                try:              # the side rectangles back on the real box
                    self._render_add_box()
                except Exception:
                    pass
            return
        if self._add_refuse_busy("box drag"):
            self._render_add_box()
            return
        self._on_add_box_changed(box.cz, box.cy, box.cx,
                                 box.sz, box.sy, box.sx)
        nb = self._add_box
        self._push_add_ui(box_center=(nb.cz, nb.cy, nb.cx),
                          box_size=(nb.sz, nb.sy, nb.sx))
        print(f"[ADD] box dragged to centre {(nb.cz, nb.cy, nb.cx)}")

    def _add_end_drag_state(self) -> None:
        st = getattr(self, "_add_drag_state", None)
        self._add_drag_state = None
        if not st:
            return
        try:
            st["vm"].camera.mouse_pan = self._add_pan_truth(st["vm"])
        except Exception:
            pass

    # --- the hover cursor (owner: yes) -------------------------------------
    def _add_side_native(self, orient):
        for o, _vm, qtv_attr in self._ADD_SIDE_VIEWS:
            if o == orient:
                try:
                    return getattr(self, qtv_attr).canvas.native
                except Exception:
                    return None
        return None

    def _add_side_hover(self, orient, vm, event) -> None:
        from packages.epicycle_annotator import box_drag as BD

        if getattr(self, "_add_drag_state", None):
            return
        on = False
        if self._add_side_ready():
            zyx = self._add_event_zyx(orient, vm, event)
            shape = self._add_volume_shape()
            if zyx is not None and shape is not None:
                bounds = self._add_box.bounds_dict(shape)
                on = BD.outline_hit(bounds, orient, zyx,
                                    self._add_outline_tol(vm, bounds, orient))
        self._add_set_side_cursor(orient, vm, on)

    def _add_set_side_cursor(self, orient, vm, on) -> None:
        shown = getattr(self, "_add_side_cursor", None)
        if shown is None:
            shown = self._add_side_cursor = {}
        if bool(on) == bool(shown.get(orient)):
            return
        native = self._add_side_native(orient)
        if native is None:
            return
        try:
            if on:
                from qtpy.QtCore import Qt

                native.setCursor(Qt.SizeAllCursor)
            else:
                native.unsetCursor()
                try:           # napari puts its own cursor back
                    vm.cursor.events.style()
                except Exception:
                    pass
            shown[orient] = bool(on)
        except Exception as exc:
            print(f"[ADD] side cursor {orient}: {exc!r}")

    def _add_clear_side_cursors(self) -> None:
        shown = getattr(self, "_add_side_cursor", None) or {}
        for orient, vm_attr, _q in self._ADD_SIDE_VIEWS:
            if shown.get(orient):
                self._add_set_side_cursor(orient, getattr(self, vm_attr, None),
                                          False)

    # --- Shift+arrow ---------------------------------
    def _add_nudge_box(self, orient, key) -> bool:
        """One Shift+arrow press, from a side view's key filter.  True when
        Add took the key (moved the box, or refused it while busy); False
        falls through to the filter's own code, unchanged."""
        from packages.epicycle_annotator import box_drag as BD

        import time

        if not self._add_side_ready():
            return False
        d = BD.nudge_delta(orient, key)
        if d is None:
            return False
        # Add was ready: from here on the key is Add's, even if the nudge
        # fails (LOW-7) -- it must never ALSO navigate
        try:
            if self._add_refuse_busy("box nudge"):
                return True
            shape = self._add_volume_shape()
            if shape is None:
                return True
            t0 = time.perf_counter()
            nb = BD.moved_box(self._add_box, d, shape)  # the nudge axis only
            if nb != self._add_box:
                self._on_add_box_changed(nb.cz, nb.cy, nb.cx,
                                         nb.sz, nb.sy, nb.sx)
                b = self._add_box
                self._push_add_ui(box_center=(b.cz, b.cy, b.cx),
                                  box_size=(b.sz, b.sy, b.sx))
            print(f"[ADD] nudge {orient} {key}: "
                  f"{(time.perf_counter() - t0) * 1000:.1f} ms")
        except Exception as exc:
            print(f"[ADD] box nudge failed: {exc!r}")
        return True

    # ======================================================================
    # Add rebuild B2.3 -- the preview worker and the box-info worker
    #.  Every input is a box crop copied HERE, on the
    # main thread; the pixels are read on the worker.  Stop is a STATE, and
    # no second Add worker starts before the previous thread is gone.
    # ======================================================================

    def _add_preset(self) -> str:
        adv = self._get_advanced_features_widget()
        if adv is not None and hasattr(adv, "get_add_preset"):
            try:
                return str(adv.get_add_preset())
            except Exception:
                pass
        return "nuclei"

    def _add_marker_source(self):
        """``(kind, run_id)`` -- the engine's source name and, for a run, its
        id.  Clicks when the panel has no picker."""
        adv = self._get_advanced_features_widget()
        if adv is not None and hasattr(adv, "get_add_marker_source"):
            try:
                kind, run_id = adv.get_add_marker_source()
                return kind, run_id
            except Exception:
                pass
        return "paint", None

    def _add_wall_params(self) -> dict:
        adv = self._get_advanced_features_widget()
        if adv is not None and hasattr(adv, "get_add_wall_params"):
            try:
                return dict(adv.get_add_wall_params())
            except Exception:
                pass
        return {"channel": "", "chain": "zcorrect", "wall_kind": "membrane"}

    def _add_nuclei_channel(self) -> str:
        adv = self._get_advanced_features_widget()
        if adv is not None and hasattr(adv, "add_nuclei_channel_name"):
            try:
                return str(adv.add_nuclei_channel_name() or "")
            except Exception:
                pass
        return ""

    def _add_params(self) -> dict:
        adv = self._get_advanced_features_widget()
        if adv is not None and hasattr(adv, "get_add_params"):
            try:
                return dict(adv.get_add_params())
            except Exception:
                pass
        return {}

    def _add_image_channel_names(self) -> list:
        """Every image channel of the patch (the Add channel combo's rule):
        non-temporary Image layers of 3D data (or RGB-last 4D)."""
        names = []
        for layer in self.viewer.layers:
            if getattr(layer, "_epicycle_temporary", False):
                continue
            if isinstance(layer, Image):
                try:
                    ndim = layer.data.ndim
                except Exception:
                    continue
                if ndim in {2, 3}:
                    names.append(layer.name)
                elif ndim == 4 and int(layer.data.shape[-1]) in {3, 4}:
                    names.append(layer.name)
        return names

    def _add_channel_crop(self, name, roi, dtype=np.float64):
        """``name``'s box crop, COPIED (crop, then copy -- never the whole
        layer), or None."""
        data = self._refine_channel_source(name)
        if data is None or roi is None:
            return None
        try:
            return np.array(data[roi], dtype, copy=True)
        except Exception as exc:
            print(f"[ADD] crop of {name!r} failed: {exc!r}")
            return None

    def _add_dark_key(self):
        ctx = self._reshape_context()
        return (ctx.get("patch_name"), ctx.get("run_epoch"),
                tuple(sorted(self._add_image_channel_names())))

    def _add_cached_dark(self):
        cache = getattr(self, "_add_dark_cache", None) or {}
        return cache.get(self._add_dark_key())

    def _add_default_spacing(self) -> bool:
        """True when Step 3's spacing is the default (1, 1, 1) -- Cells runs
        on it as it is and the page says so in amber.  Reads the context itself, not ``_refine_spacing_known``, which
        is Refine's gate and stays Refine's."""
        from packages.epicycle_annotator.add_preview import spacing_is_default

        try:
            spacing = self._reshape_context().get("spacing")
        except Exception:
            spacing = None
        return spacing_is_default(spacing)

    def _add_s1_state(self) -> dict:
        """Everything :func:`add_preview.s1_sentence` reads."""
        shared = self._add_shared_volume()
        preset = self._add_preset()
        params = self._add_params()
        kind, _run_id = self._add_marker_source()
        box = getattr(self, "_add_box", None)
        shape = None
        if getattr(self, "_add_roi", None) is not None:
            shape = tuple(int(sl.stop - sl.start) for sl in self._add_roi)
        info = getattr(self, "_add_info", None)
        dark_on = bool(params.get("dark_outside", True))
        info_pending = False
        if preset == "cells":
            if info is None:
                info_pending = True
            elif dark_on and self._add_cached_dark() is None:
                info_pending = True
            elif kind == "labels" and info.run_count is None:
                info_pending = True
        state = {
            "has_patch": shared is not None,
            "transient": getattr(self, "_add_transient", None),
            "busy": bool(getattr(self, "_add_busy", False)),
            "has_box": box is not None and getattr(self, "_add_running", False),
            "preset": preset,
            "box_shape": shape,
            "box_budget": int(params.get("box_budget", 4_000_000)),
            "channel": str(params.get("channel", "") or ""),
            "wall_channel": str(self._add_wall_params().get("channel") or ""),
            "nuclei_channel": self._add_nuclei_channel(),
            "source": kind,
            "run_state": getattr(self, "_add_run_state", None),
            "n_clicks": len(self._add_markers()),
            # 4.9a row 12b and the "k of n ticked" tail
            "n_ticked": sum(1 for m in self._add_markers() if m["ticked"]),
            "info_pending": info_pending,
            "empty_share": None if info is None else info.empty_share,
            "user_emptied": bool(getattr(self, "_add_user_emptied", False)),
        }
        if getattr(self, "_add_review", False):
            sizes = {}
            for e in getattr(self, "_add_session", []) or []:
                sizes.update(e.get("sizes", {}))
            ids = list(getattr(self, "_add_piece_ids", []) or [])
            adv = self._get_advanced_features_widget()
            sel = []
            if adv is not None and hasattr(adv, "get_selected_add_pieces"):
                try:
                    sel = list(adv.get_selected_add_pieces())
                except Exception:
                    sel = []
            state["review"] = (len(ids),
                               sum(int(sizes.get(i, 0)) for i in ids),
                               len(sel))
        res = getattr(self, "_add_preview_res", None)
        if (getattr(self, "_add_preview_result", None) is not None
                and res is not None):
            adv = self._get_advanced_features_widget()
            ticked = list(res.ids)
            if adv is not None and hasattr(adv, "get_checked_add_pieces"):
                try:
                    ticked = list(adv.get_checked_add_pieces())
                except Exception:
                    pass
            edge, small = set(res.edge_ids), set(res.small_ids)
            # the LIVE unticked rows (B3.4): the reason is given only while
            # every unticked row is one that started unticked for it
            off = {int(i) for i in res.ids} - {int(i) for i in ticked}
            if off <= (edge | small):
                state["candidates"] = (len(res.ids), len(off & edge),
                                       len(off & small),
                                       len(off & edge & small), len(ticked))
            else:
                state["candidates"] = (len(res.ids), 0, 0, 0, len(ticked),
                                       len(off))
        return state

    def _add_refresh_ui(self, **extra) -> None:
        """Push S1 (the 4.9a table), Preview's enable and the strip."""
        from packages.epicycle_annotator.add_preview import (
            preview_block,
            s1_kind,
            s1_sentence,
        )

        state = self._add_s1_state()
        busy = state["busy"]
        review = bool(getattr(self, "_add_review", False))
        running = bool(getattr(self, "_add_running", False))
        preview_ok = (running and not busy and not review
                      and preview_block(state) is None)
        kw = dict(status_text=s1_sentence(state), status_kind=s1_kind(state),
                  default_spacing=self._add_default_spacing(),
                  preview_enabled=preview_ok,
                  busy=busy,
                  stop_enabled=busy and not getattr(self, "_add_stopping",
                                                    False))
        kw.update(extra)
        self._push_add_ui(**kw)

    def _add_refuse_busy(self, what) -> bool:
        """True (and ``ADD_S1_BUSY_REFUSED``) while a preview runs -- nothing
        moves and nothing is cancelled."""
        if not getattr(self, "_add_busy", False):
            return False
        from packages.epicycle_annotator.add_preview import ADD_S1_BUSY_REFUSED

        print(f"[ADD] {what} refused - a preview is running")
        self._add_transient = ADD_S1_BUSY_REFUSED
        self._add_refresh_ui()
        return True

    def _add_gather_preview(self):
        """``(kwargs, None)`` for :func:`run_add_preview`, or ``(None,
        sentence)``.  Main thread; every array a copied box crop."""
        from packages.epicycle_annotator import add_preview as AP

        shared = self._add_shared_volume()
        roi = getattr(self, "_add_roi", None)
        if shared is None or roi is None:
            return None, AP.ADD_S1_NO_PATCH
        state = self._add_s1_state()
        block = AP.preview_block(state)
        if block is not None:
            return None, block
        params = self._add_params()
        preset = state["preset"]
        existing_roi = np.array(shared[roi], copy=True)
        kw = dict(preset=preset, roi=tuple(roi), existing_roi=existing_roi,
                  next_label=self._add_next_label(), id_max=self._add_id_max())
        common = {k: params.get(k, d) for k, d in (
            ("min_size", 0), ("clear_border", True), ("edge_margin", 3),
            ("auto_smooth", True), ("smooth_sigma", 1.5))}
        if preset != "cells":
            image_roi = self._add_channel_crop(params.get("channel", ""), roi)
            if image_roi is None:
                return None, self._add_channel_error_text()
            thr = (self._add_threshold_value
                   if self._add_threshold_value is not None else "otsu")
            p = dict(common, clahe=bool(params.get("clahe", False)),
                     clahe_clip=float(params.get("clahe_clip", 0.01)),
                     threshold=thr,
                     separate=bool(params.get("separate", True)),
                     min_distance=int(params.get("min_distance", 3)))
            kw.update(params=p, image_roi=image_roi,
                      spacing=self._add_spacing())
            return kw, None
        kind, _run_id = self._add_marker_source()
        wall = self._add_wall_params()
        wall_roi = self._add_channel_crop(wall.get("channel"), roi, np.float32)
        if wall_roi is None:
            return None, AP.ADD_S1_PICK_WALL_CHANNEL
        dark_on = bool(params.get("dark_outside", True))
        dark_rois = None
        dark = None
        if dark_on:
            dark = self._add_cached_dark()
            if dark is None:
                return None, AP.ADD_S1_MEASURING
            dark_rois = {}
            for name in self._add_image_channel_names():
                crop = self._add_channel_crop(name, roi, np.float32)
                if crop is not None:
                    dark_rois[name] = crop
        nuclei_roi = None
        run_guide_roi = None
        if kind == "nuclei_masks":
            nuclei_roi = self._add_channel_crop(self._add_nuclei_channel(), roi)
            if nuclei_roi is None:
                return None, AP.ADD_S1_PICK_NUCLEI_CHANNEL
        elif kind == "labels":
            guide = getattr(self, "_add_run_guide", None)
            if guide is None:
                return None, AP.ADD_S1_RUN_LOADING
            run_guide_roi = np.array(guide[roi], copy=True)
        nthr = getattr(self, "_add_nuclei_threshold_value", None)
        # the TICKED markers, in placement order, with the
        # ids they reserved -- re-based first if next_label moved since
        if self._add_rebase_markers(kw["next_label"]):
            self._render_add_balls()
            self._add_take_rebase_note()      # S1 says why (B-M2)
            self._add_refresh_ui(**self._add_marker_push())
        ticked = [m for m in self._add_markers() if m["ticked"]]
        p = dict(common, source=kind,
                 wall_kind=wall.get("wall_kind", "membrane"),
                 wall_chain=wall.get("chain", "none"),
                 dark_outside=dark_on,
                 dark_otsu_factor=float(params.get("dark_otsu_factor", 0.5)),
                 dark_clearance_um=float(params.get("dark_clearance_um", 2.0)),
                 core_erode_um=float(params.get("core_erode_um", 1.0)),
                 click_radius_um=float(params.get("click_radius_um", 2.0)),
                 box_budget=int(params.get("box_budget", 4_000_000)),
                 nuclei_threshold=("otsu" if nthr is None else float(nthr)))
        kw.update(params=p, spacing=self._refine_spacing(), wall_roi=wall_roi,
                  dark_rois=dark_rois, dark_threshold=dark,
                  nuclei_roi=nuclei_roi, run_guide_roi=run_guide_roi,
                  click_points=[tuple(m["zyx"]) for m in ticked],
                  reserved_ids=[int(m["id"]) for m in ticked])
        return kw, None

    def _preview_add(self) -> None:
        """Preview: gather the box's inputs and start the worker (4.5)."""
        if not getattr(self, "_add_running", False) or self._add_roi is None:
            return
        # the fifth door: a batch's finish or Reject rewrites the
        # volume this preview would measure
        if self._add_refused_while_batch("preview"):
            return
        if self._add_refuse_busy("preview"):
            return
        if getattr(self, "_add_thread", None) is not None:
            # the previous thread is still quitting (4.6, invariant 11)
            print("[ADD] preview refused - the previous thread is still "
                  "quitting")
            return
        self._add_transient = None
        # the snapshot and the write read shared: realign it first (U1)
        self._ensure_shared_data_aligned(self._active_labels_name)
        self._add_forget_volume_max()      # B-L3: a realign may replace it
        if getattr(self, "_add_preview_result", None) is not None:
            self._invalidate_add_preview()
        kwargs, why = self._add_gather_preview()
        if kwargs is None:
            print(f"[ADD] Preview refused: {why}")
            self._add_transient = why
            self._add_refresh_ui()
            return
        self._launch_add_worker(kwargs)

    def _launch_add_worker(self, kwargs) -> None:
        """Start ``run_add_preview`` on a ``RefineWorker``.

        The worker's OWN cancel event goes into ``kwargs`` -- ``RefineWorker``
        never forwards it to ``fn`` (grill E-M6)."""
        import threading

        from qtpy.QtCore import QThread

        from napari_epicycle.annotator.reshape_worker import (
            RefineWorker,
            stage_text,
        )
        from packages.epicycle_annotator.add_preview import (
            ADD_S1_FAILED,
            run_add_preview,
        )

        ctx = self._reshape_context()
        self._add_token = int(getattr(self, "_add_token", 0)) + 1
        token = self._add_token
        self._add_launch_labels_name = self._active_labels_name
        self._add_launch_patch = ctx.get("patch_name")
        self._add_launch_run_epoch = ctx.get("run_epoch")
        self._add_launch_roi = tuple(self._add_roi)
        # a finish launched on other markers is stale
        self._add_launch_markers_epoch = int(
            getattr(self, "_add_markers_epoch", 0))
        ev = threading.Event()
        kwargs = dict(kwargs)
        kwargs["cancel_event"] = ev
        self._add_busy = True
        self._add_stopping = False
        self._add_refresh_ui(stage_text=stage_text("add_box"))
        try:
            self._add_thread = QThread()
            self._add_worker = RefineWorker(
                run_add_preview, kwargs, kind="add", cancel_event=ev,
                wants_progress=True)
            self._add_worker._epicycle_token = token
            self._add_worker.moveToThread(self._add_thread)
            self._add_thread.started.connect(self._add_worker.run)
            self._add_worker.progress.connect(self._on_add_progress)
            self._add_worker.finished.connect(self._on_add_finished)
            self._add_worker.error.connect(self._on_add_error)
            # Every exit of run() emits finished or error; each must also
            # QUIT the thread, or its event loop runs on, thread.finished
            # never comes and busy stays up until Accept / Cancel (B2 fix
            # pass L-H1).  NOT _add_thread_cleanup from the handlers: it
            # nulls _add_thread, and _on_add_thread_finished's sender check
            # would then never clear busy.
            self._add_worker.finished.connect(self._add_thread.quit)
            self._add_worker.error.connect(self._add_thread.quit)
            self._add_thread.finished.connect(self._on_add_thread_finished)
            self._add_thread.start()
            print("[ADD] preview worker started token={} preset={}".format(
                token, kwargs.get("preset")))
        except Exception as exc:  # noqa: BLE001 - never strand the surface
            print("[ADD] preview launch FAILED: {!r}".format(exc))
            self._add_thread = None
            self._add_worker = None
            self._add_busy = False
            self._add_transient = ADD_S1_FAILED
            self._add_refresh_ui()

    def _on_add_progress(self, stage, _i=0, _n=0) -> None:
        if getattr(self, "_add_stopping", False):
            return
        from napari_epicycle.annotator.reshape_worker import stage_text

        self._push_add_ui(stage_text=stage_text(stage))

    def _add_is_stale(self, result_token=None) -> bool:
        """The launch quintuple + "is there still a session"."""
        why = None
        if result_token is not None and result_token != getattr(
                self, "_add_token", 0):
            why = "token {} != {}".format(result_token, self._add_token)
        elif self._active_labels_name != getattr(
                self, "_add_launch_labels_name", None):
            why = "labels layer changed"
        else:
            ctx = self._reshape_context()
            if ctx.get("patch_name") != getattr(self, "_add_launch_patch",
                                                None):
                why = "patch changed"
            elif ctx.get("run_epoch") != getattr(
                    self, "_add_launch_run_epoch", None):
                why = "run changed"
            elif (self._add_roi is None or tuple(self._add_roi)
                  != getattr(self, "_add_launch_roi", None)):
                why = "the box moved"
            elif not getattr(self, "_add_running", False):
                why = "add session gone"
            elif int(getattr(self, "_add_markers_epoch", 0)) != int(getattr(
                    self, "_add_launch_markers_epoch",
                    getattr(self, "_add_markers_epoch", 0))):
                why = "markers changed"
        if why is not None:
            print(f"[ADD] STALE finish discarded - {why}")
            return True
        return False

    def _on_add_finished(self, outcome) -> None:
        """The preview worker's answer.  Stopping FIRST (a Stop inside an
        unpolled span comes back ``ok``), then stale, then the result."""
        if getattr(self, "_add_stopping", False):
            self._on_add_stopped()
            return
        if self._add_is_stale(self._result_token()):
            return
        res = getattr(outcome, "result", None)
        if res is None or getattr(res, "status", "") == "cancelled":
            self._on_add_stopped()
            return
        try:
            print(res.log_line())
        except Exception:
            pass
        if res.status != "ok":
            self._add_transient = res.refusal
            self._add_preview_result = None
            self._add_preview_res = None
            self._add_refresh_ui()
            return
        self._apply_add_result(res)

    def _on_add_error(self, message) -> None:
        if getattr(self, "_add_stopping", False):
            self._on_add_stopped()
            return
        if self._add_is_stale(self._result_token()):
            return
        from packages.epicycle_annotator.add_preview import ADD_S1_FAILED

        print("[ADD] preview FAILED: {}".format(message))
        self._add_transient = ADD_S1_FAILED
        self._add_refresh_ui()

    def _stop_add_preview(self) -> None:
        """Stop is a STATE: flag, cancel, say so, return -- no
        pump, no wait.  The finish handlers route to the stopped path."""
        if not getattr(self, "_add_busy", False):
            return
        worker = getattr(self, "_add_worker", None)
        if worker is None or getattr(self, "_add_stopping", False):
            return
        from napari_epicycle.annotator.reshape_worker import stage_text

        self._add_stopping = True
        try:
            worker.cancel()
        except Exception:
            pass
        print("[ADD] Stop pressed")
        self._add_refresh_ui(stage_text=stage_text("stopping"),
                             stop_enabled=False)

    def _on_add_stopped(self) -> None:
        """Drop the answer; say Stopped.  ``_add_busy`` stays until the
        thread is gone (``_on_add_thread_finished``)."""
        from packages.epicycle_annotator.add_preview import ADD_S1_STOPPED

        self._add_stopping = False
        self._add_transient = ADD_S1_STOPPED
        print("[ADD] preview stopped - nothing from this box was added")
        self._add_refresh_ui()

    def _on_add_thread_finished(self) -> None:
        """The QThread's own ``finished``: only now is Preview live again."""
        from packages.epicycle_annotator.add_preview import ADD_S1_BUSY_REFUSED

        sender = self._signal_sender()
        if sender is not None and sender is not getattr(self, "_add_thread",
                                                        None):
            return
        self._add_thread_cleanup("preview")
        was_stopping = getattr(self, "_add_stopping", False)
        self._add_busy = False
        if was_stopping:
            self._on_add_stopped()
            return
        if getattr(self, "_add_transient", None) == ADD_S1_BUSY_REFUSED:
            self._add_transient = None
        self._add_refresh_ui()

    def _add_thread_cleanup(self, which="preview") -> None:
        """quit -> wait(3000) -> else PARK (a twin of
        ``_refine_thread_cleanup``), for the preview or the info thread."""
        th_attr, w_attr = (("_add_thread", "_add_worker") if which == "preview"
                           else ("_add_info_thread", "_add_info_worker"))
        th = getattr(self, th_attr, None)
        worker = getattr(self, w_attr, None)
        setattr(self, th_attr, None)
        setattr(self, w_attr, None)
        if th is None:
            return
        try:
            th.quit()
            if th.wait(3000):
                return
        except Exception:
            return
        print(f"[ADD] {which} worker will not stop in time - parking the "
              "thread (dropping it here would abort the process)")
        parked = getattr(self, "_refine_parked_threads", None)
        if parked is None:
            parked = []
            self._refine_parked_threads = parked
        for sig in ("progress", "message", "finished", "error"):
            try:
                getattr(worker, sig).disconnect()
            except Exception:
                pass
        try:
            th.finished.disconnect()
        except Exception:
            pass
        parked.append((th, worker))

    def _add_abandon_workers(self) -> None:
        """Cancel + clean both Add workers (every exit of).
        Idempotent; never raises."""
        for w_attr in ("_add_worker", "_add_info_worker"):
            w = getattr(self, w_attr, None)
            if w is not None:
                try:
                    w.cancel()
                except Exception:
                    pass
        for which in ("preview", "info"):
            try:
                self._add_thread_cleanup(which)
            except Exception:
                pass
        self._add_busy = False
        self._add_stopping = False
        self._add_info_dirty = False

    def _apply_add_result(self, res) -> None:
        """A result on screen."""
        from packages.epicycle_annotator.add_preview import (
            ADD_FOOT_CANDIDATES,
            counts_note,
        )

        self._add_preview_result = res.result
        self._add_preview_roi = (tuple(res.roi) if res.roi is not None
                                 else None)
        self._add_preview_res = res
        self._add_transient = None
        # B-M1 (the A5 audit): markers whose ball sat inside a kept object
        # made nothing -- the counts note says so, and their rows are marked
        # for when the list comes back on Discard
        self._add_absorbed_ids = {int(i) for i in
                                  (getattr(res, "absorbed_ids", None) or ())}
        if self._add_absorbed_ids:
            self._push_add_ui(**self._add_marker_push())
        self._render_add_candidate(res.result, roi=self._add_preview_roi)
        edge, small = set(res.edge_ids), set(res.small_ids)
        order = sorted(res.ids, key=lambda i: (
            0 if i in edge else 1 if i in small else 2, i))
        note = counts_note(res.counts(), source=res.source)
        on, why = self._add_walls_toggle_state()
        self._push_add_ui(
            candidate_ids=order,
            candidate_sizes=dict(res.sizes),
            candidate_edge_ids=edge,
            candidate_small_ids=small,
            notes=([("counts", note, False)] if note else []),
            footer_text=ADD_FOOT_CANDIDATES,
            reject_enabled=True,
            walls_toggle_enabled=on,
            walls_toggle_reason=why,
        )
        if getattr(self, "_add_walls_wanted", False) and on:
            self._show_add_walls()
        # the fill above fired the check callback with the rows as they
        # STARTED; that is not the user unticking anything (V3)
        self._add_user_emptied = False
        self._add_refresh_ui()

    # --------------------------------------------------------- box info --
    def _request_add_box_info(self) -> None:
        """At most ONE info thread; a request while it runs sets dirty and
        the finish relaunches once."""
        if (not getattr(self, "_add_running", False)
                or getattr(self, "_add_box", None) is None
                or getattr(self, "_add_review", False)):
            return
        if getattr(self, "_add_info_thread", None) is not None:
            self._add_info_dirty = True
            return
        self._launch_add_info_worker()

    def _add_gather_info(self):
        from packages.epicycle_annotator.add_preview import dark_stride

        shared = self._add_shared_volume()
        roi = getattr(self, "_add_roi", None)
        if shared is None or roi is None:
            return None
        preset = self._add_preset()
        params = self._add_params()
        kind, _run_id = self._add_marker_source()
        dark_on = bool(params.get("dark_outside", True))
        cache = self._add_cached_dark() if preset == "cells" else None
        sources = None
        stride = 1
        if preset == "cells" and dark_on and cache is None:
            sources = {}
            for name in self._add_image_channel_names():
                data = self._refine_channel_source(name)
                if data is not None:
                    sources[name] = data
            if sources:
                stride = dark_stride(next(iter(sources.values())).shape)
        dark_rois = None
        run_roi = None
        nuc_roi = None
        if preset == "cells" and kind == "labels":
            guide = getattr(self, "_add_run_guide", None)
            if guide is not None:
                run_roi = np.array(guide[roi], copy=True)
                if dark_on:
                    dark_rois = {}
                    for n in self._add_image_channel_names():
                        c = self._add_channel_crop(n, roi, np.float32)
                        if c is not None:
                            dark_rois[n] = c
        if preset == "cells" and kind == "nuclei_masks":
            nuc_roi = self._add_channel_crop(self._add_nuclei_channel(), roi)
        image_roi = None
        if preset != "cells":
            # the Brightness histogram and its Auto (B4.2): the box crop,
            # copied here, measured on the worker
            image_roi = self._add_channel_crop(params.get("channel", ""), roi)
        p = dict(params, preset=preset)
        p.setdefault("dark_outside", dark_on)
        return dict(existing_roi=np.array(shared[roi], copy=True),
                    dark_sources=sources, dark_rois=dark_rois,
                    spacing=self._refine_spacing(), stride=stride,
                    dark_cache=cache, source=kind, run_guide_roi=run_roi,
                    nuclei_roi=nuc_roi, image_roi=image_roi, params=p,
                    roi=tuple(roi))

    def _launch_add_info_worker(self) -> None:
        import threading

        from qtpy.QtCore import QThread

        from napari_epicycle.annotator.reshape_worker import RefineWorker
        from packages.epicycle_annotator.add_preview import add_box_info

        kwargs = self._add_gather_info()
        if kwargs is None:
            return
        # the spacing this answer is measured at (A-L3: an unchanged QC
        # spacing re-measures nothing)
        try:
            self._add_info_spacing = tuple(
                float(v) for v in kwargs.get("spacing") or ())
        except Exception:  # noqa: BLE001
            self._add_info_spacing = None
        ctx = self._reshape_context()
        self._add_info_token = int(getattr(self, "_add_info_token", 0)) + 1
        ev = threading.Event()
        kwargs["cancel_event"] = ev
        self._add_info_dirty = False
        try:
            th = QThread()
            w = RefineWorker(add_box_info, kwargs, kind="add_info",
                             cancel_event=ev)
            w._epicycle_token = self._add_info_token
            w._epicycle_patch = ctx.get("patch_name")
            w._epicycle_epoch = ctx.get("run_epoch")
            w._epicycle_roi = tuple(self._add_roi)
            w._epicycle_dark_key = self._add_dark_key()
            w._epicycle_brightness_key = self._add_brightness_key()
            import time

            w._epicycle_t0 = time.perf_counter()
            self._add_info_thread = th
            self._add_info_worker = w
            w.moveToThread(th)
            th.started.connect(w.run)
            w.finished.connect(self._on_add_info_finished)
            w.error.connect(self._on_add_info_error)
            # the same quit as the preview's (B2 fix pass L-H2): without it
            # the info thread never finishes, the next request only sets
            # dirty, and Measuring the box... never ends after a box move
            w.finished.connect(th.quit)
            w.error.connect(th.quit)
            th.finished.connect(self._on_add_info_thread_finished)
            th.start()
        except Exception as exc:  # noqa: BLE001
            print("[ADD-INFO] launch FAILED: {!r}".format(exc))
            self._add_info_thread = None
            self._add_info_worker = None

    def _add_info_is_stale(self, worker) -> bool:
        why = None
        ctx = self._reshape_context()
        if worker is None:
            why = "no worker"
        elif getattr(worker, "_epicycle_token", None) != getattr(
                self, "_add_info_token", 0):
            why = "token"
        elif getattr(worker, "_epicycle_patch", None) != ctx.get("patch_name"):
            why = "patch changed"
        elif getattr(worker, "_epicycle_epoch", None) != ctx.get("run_epoch"):
            why = "run changed"
        elif (self._add_roi is None
              or getattr(worker, "_epicycle_roi", None)
              != tuple(self._add_roi)):
            why = "the box moved"
        elif not getattr(self, "_add_running", False):
            why = "add session gone"
        if why:
            print(f"[ADD-INFO] STALE - {why}")
            return True
        return False

    def _on_add_info_finished(self, outcome, worker=None) -> None:
        worker = worker or self._signal_sender() or getattr(
            self, "_add_info_worker", None)
        info = getattr(outcome, "result", None)
        dark = getattr(info, "dark_threshold", None)
        # the patch-level threshold is good whatever the box did since
        if dark is not None and worker is not None:
            key = getattr(worker, "_epicycle_dark_key", None)
            if key is not None and key == self._add_dark_key():
                if getattr(self, "_add_dark_cache", None) is None:
                    self._add_dark_cache = {}
                self._add_dark_cache[key] = dark
        if self._add_info_is_stale(worker):
            return
        if info is None or getattr(info, "status", "") != "ok":
            return
        bkey = getattr(worker, "_epicycle_brightness_key", None)
        brightness_current = (bkey is not None
                              and bkey == self._add_brightness_key())
        if not brightness_current:
            # measured with other Brightness inputs (Local contrast toggled,
            # another channel): its bars and Auto are in the wrong value
            # space -- the dirty relaunch brings the right ones
            info.histogram = None
            info.brightness_auto = None
        self._add_info = info
        self._add_info_brightness_key = bkey if brightness_current else None
        t0 = getattr(worker, "_epicycle_t0", None)
        if t0 is not None:
            import time

            # the 1-2 s "Measuring the box..." after a change is this
            # worker's real time, not a lost relaunch (live-shot F9)
            print("[ADD-INFO] landed in {:.0f} ms (auto={}, run_count={}, "
                  "empty={}, brightness_auto={})".format(
                      (time.perf_counter() - t0) * 1000.0,
                      info.nuclei_auto, info.run_count, info.empty_share,
                      info.brightness_auto))
        # the touched rule (live-shot fix F2): an Auto landing sets the
        # value Preview uses AND shows it in the Threshold box, unless the
        # user typed a value since -- a typed value is never overwritten
        if (info.nuclei_auto is not None
                and not getattr(self, "_add_nuclei_threshold_typed", False)):
            self._add_nuclei_threshold_value = float(info.nuclei_auto)
            adv = self._get_advanced_features_widget()
            if adv is not None and hasattr(adv, "set_add_nuclei_threshold"):
                try:
                    adv.set_add_nuclei_threshold(info.nuclei_auto)
                except Exception as exc:  # noqa: BLE001
                    print(f"[ADD-INFO] threshold box not updated: {exc!r}")
        if brightness_current and info.histogram is not None:
            # the Brightness card (B4.2): bars and Auto from the worker; an
            # untouched threshold follows Auto at full precision, a value
            # the user set (typed or dragged) is kept
            auto = info.brightness_auto
            self._push_add_ui(
                histogram_data=info.histogram,
                brightness_auto=(float("nan") if auto is None
                                 else float(auto)),
                # bars but no Auto: a flat box, nothing to split (B2) --
                # said, instead of "Measuring..." for ever
                brightness_flat=auto is None)
            # the untouched rule on EVERY landing (final fix pass B1): the
            # threshold follows the new Auto unless the user set it since
            cur = self._add_threshold_value
            untouched = (not getattr(self, "_add_threshold_touched", False)
                         and (cur is None or cur == getattr(
                             self, "_add_threshold_auto_written", None)))
            if auto is not None and untouched:
                self._add_threshold_value = float(auto)
                self._add_threshold_auto_written = float(auto)
                self._push_add_ui(threshold_value=float(auto))
        self._add_refresh_ui(run_count=info.run_count,
                             nuclei_auto=info.nuclei_auto)

    def _on_add_info_error(self, message) -> None:
        print("[ADD-INFO] failed: {}".format(message))

    def _on_add_info_thread_finished(self) -> None:
        sender = self._signal_sender()
        if sender is not None and sender is not getattr(
                self, "_add_info_thread", None):
            return
        self._add_thread_cleanup("info")
        if getattr(self, "_add_info_dirty", False):
            self._add_info_dirty = False
            self._request_add_box_info()

    def _add_existing_roi(self):
        shared = self._add_shared_volume()
        if shared is None or self._add_roi is None:
            return None
        try:
            return shared[self._add_roi]
        except Exception:
            return None

    def _add_edge_flags(self, labels, margin: int) -> set:
        """IDs of pieces reaching the box edge zone -- delegates to the pure
        ``add_segment.edge_flag_ids``."""
        from packages.epicycle_annotator.add_segment import edge_flag_ids

        return edge_flag_ids(labels, margin)

    def _apply_add_edge_cleanup(self, labels, margin: int):
        """Legacy DROP path — retained for the edge-touch tests + as a pure
        helper.  Delegates the per-piece predicate to :meth:`_add_edge_flags`,
        then deletes the flagged ids.  The interactive Add flow no longer drops
        — it FLAGS (see ``_preview_add`` / the candidate checklist)."""
        flagged = self._add_edge_flags(labels, margin)
        ids = [int(v) for v in np.unique(labels) if v != 0]
        if flagged and len(flagged) < len(ids):
            out = labels.copy()
            for v in flagged:
                out[labels == v] = 0
            return out, sorted(flagged)
        return labels, []

    def _sync_add_candidate_overlay(self) -> None:
        """Live: re-render the candidate overlay to the CHECKED pieces so the 3D
        view (+ subviewers) shows EXACTLY what Accept will commit.

        Fires when a candidate checkbox toggles (and once after Preview populates
        the list).  ``_add_preview_result`` keeps the FULL set — only the DISPLAY
        is narrowed; ``_accept_add`` re-derives the checked set from that same
        source, so the model stays the single source of truth.

        Cost note: this re-runs the full candidate render (incl. the 3 subviewer
        transposes) per toggle.  That's the heavy Labels-render path, but it's
        bounded to human checkbox-click cadence (not a continuous drag), so no
        debounce is needed; the GPU-hang risk was the SMOOTH iso-gradient, which
        stays off for this layer.
        """
        result = getattr(self, "_add_preview_result", None)
        roi = getattr(self, "_add_preview_roi", None) or self._add_roi
        if result is None or roi is None:
            return
        if getattr(self, "_add_candidate_layer", None) is None:
            return
        adv = self._get_advanced_features_widget()
        if adv is None or not hasattr(adv, "get_checked_add_pieces"):
            return
        checked = [int(v) for v in adv.get_checked_add_pieces()]
        # a toggle by the USER: zero ticked now means "the user unticked the
        # last row" -> TICK_ONE (live-shot fix V3).  _apply_add_result
        # resets it after the list is filled, so rows that STARTED
        # unticked never count.
        self._add_user_emptied = not checked
        masked = (
            np.where(np.isin(result, checked), result, 0)
            if checked else np.zeros_like(result)
        )
        self._render_add_candidate(masked, roi=roi)
        self._add_refresh_ui()

    def _add_main_qt_viewer(self):
        """The main viewer's ``QtViewer`` (private-API, guarded).

        Needed to parent the 3D box wireframe onto the main canvas scene — the
        annotator otherwise never touches the main canvas, so probe a couple of
        known attribute chains defensively across napari versions.
        """
        win = getattr(self.viewer, "window", None)
        if win is None:
            return None
        for chain in (("_qt_viewer",), ("qt_viewer",), ("_qt_window", "_qt_viewer")):
            obj = win
            try:
                for attr in chain:
                    obj = getattr(obj, attr)
                if obj is not None and hasattr(obj, "canvas"):
                    return obj
            except Exception:
                continue
        return None

    def _render_add_box(self) -> None:
        """Draw / update the read-only ROI box across all four views.

        A 3D wireframe in the main view + a 2D rectangle in each XY/XZ/YZ
        subviewer, rendered as vispy ``Line`` overlays (the crosshair idiom),
        NOT a napari Shapes layer.  A Shapes-in-3D layer's mesh VBO can be freed
        while a draw is still queued on teardown/patch-switch — the confirmed
        ``glDrawArrays`` access violation.  An overlay never enters
        ``viewer.layers``, so it cannot orphan into the next patch; the AV class
        is gone by construction.  See ``box_overlay.AddBoxOverlay``.
        """
        if not self._ADD_BOX_RENDER_ENABLED:
            return
        if self._add_box is None:
            return
        main = self._add_main_label_layer()
        if main is None:
            return
        try:
            overlay = getattr(self, "_add_box_overlay", None)
            if overlay is None:
                from packages.epicycle_annotator.box_overlay import AddBoxOverlay

                overlay = AddBoxOverlay(
                    self._add_main_qt_viewer(),
                    getattr(self, "qt_viewer_xy", None),
                    getattr(self, "qt_viewer_xz", None),
                    getattr(self, "qt_viewer_yz", None),
                )
                self._add_box_overlay = overlay
            scale = list(main.scale) if main.scale is not None else None
            translate = (
                list(main.translate)
                if getattr(main, "translate", None) is not None
                else None
            )
            overlay.update(
                self._add_box, main.data.shape, scale=scale,
                translate=translate,
                in_range=self._add_side_in_range(self._add_box,
                                                 main.data.shape),
            )
        except Exception as exc:
            print(f"[ADD] box overlay render failed: {exc!r}")

    def _render_add_candidate(self, result, roi=None) -> None:
        """Show the candidate pieces as a temporary full-volume Labels layer.

        ``roi`` is the ROI the RESULT was computed on, never the
        live box."""
        main = self._add_main_label_layer()
        if roi is None:
            roi = (getattr(self, "_add_preview_roi", None)
                   or getattr(self, "_add_roi", None))
        if main is None or roi is None:
            return
        name = "_epicycle_add_candidate"
        cand = self._layer_by_name(self.viewer.layers, name)
        if cand is not None and cand.data.shape == main.data.shape:
            # In-place clear + fill + refresh (avoids the data= 3D recalc).
            cand.data[...] = 0
            cand.data[roi] = result
            import warnings

            try:
                # the outline may be on: napari re-raises the 3D-contour
                # warning on every refresh, so the same ONE-message filter
                # as the write (A-M1, the A1-A4 audit)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message=_CONTOUR_3D_WARNING + ".*")
                    cand.refresh()
            except Exception:
                pass
            self._add_candidate_layer = cand
        else:
            if cand is not None:
                self._remove_add_candidate()
            full = np.zeros(main.data.shape, dtype=main.data.dtype)
            full[roi] = result
            # Import locally like every other call site (2359 / 2742 / 7316) —
            # this method previously called it unqualified → NameError, which
            # aborted candidate setup BEFORE the subviewer-mirror block below
            # (preview vanished from the subviewers).  The brush-1 lock is a
            # locked product invariant (brush size is 1 on every annotator
            # Labels layer), so it MUST run — keep the call.
            from packages.epicycle_annotator.sync import lock_brush_size_to_1
            try:
                # flag, THEN append: layers.events.inserted
                # fires inside the append, so a flag set afterwards is one
                # statement too late for the channel lists.
                cand = Labels(full, name=name, opacity=0.75)
                cand._epicycle_temporary = True
                cand.editable = False
                try:
                    cand.scale = main.scale
                    cand.translate = main.translate
                except Exception:
                    pass
                # Smooth iso-gradient for a nicer preview surface (user request,
                # 2026-07-03).  A prior build suspected this of a GPU hang on a
                # FULL-VOLUME candidate, but that was never confirmed (the real
                # crash was the Shapes-in-3D box, since fixed).  Guarded; if it
                # ever stalls on a very large patch, the fix is to render the
                # candidate BOUNDED to the ROI box (small) so smooth is cheap.
                try:
                    cand.rendering = "iso_categorical"
                    cand.iso_gradient_mode = "smooth"
                except Exception:
                    pass
                lock_brush_size_to_1(cand)
                self.viewer.layers.append(cand)
                self._add_candidate_layer = cand
            except Exception as exc:
                print(f"[ADD] candidate layer failed: {exc!r}")
                return
        # Mirror into the subviewers (best-effort).
        try:
            self.subviewer_manager.add_labels_to_subviewers(
                self._add_candidate_layer,
                sync_manager=self.sync_manager,
                drawing_handler=self.drawing_handler,
            )
            # The candidate is a READ-ONLY preview — add_labels_to_subviewers
            # forces editable=True/paint on every copy, which would let the user
            # paint onto it (silently discarded on Accept, audit M2).  Lock the
            # 3 subviewer copies to pan/zoom.
            for orient in ("XY", "XZ", "YZ"):
                sub = self.subviewer_manager.get_subviewer_layer(name, orient)
                if sub is not None:
                    try:
                        sub.editable = False
                        sub.mode = "pan_zoom"
                    except Exception:
                        pass
        except Exception:
            pass
        self._restore_side_active_layers()
        try:
            self.viewer.layers.selection.active = main
        except Exception:
            pass
        # the remembered outline, AFTER the side copies
        # exist -- nothing copies ``contour`` at creation; the copies get
        # it from the property-change propagation (the Reshape rule).
        if getattr(self, "_add_contour_on", False):
            self._add_set_contour(True)

    def _add_set_contour(self, on: bool) -> None:
        """Write ``contour`` on the candidate layer, quietly.

        ``sync_label_properties`` carries it to the candidate's three side
        copies (same base name) and to nothing else.  napari refuses
        contours in the 3D main view and warns; that ONE message is
        filtered, exactly as ``_set_refine_contour`` does -- every other
        warning still gets out."""
        import warnings

        cand = getattr(self, "_add_candidate_layer", None)
        if cand is None:
            _add_outline_filter(False)
            return
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=_CONTOUR_3D_WARNING + ".*")
            cand.contour = 1 if on else 0
        # every LATER refresh of the contoured layer (a position move, a
        # refill) warns again: the same one message, for the layer's life
        # with the outline on (A-M1)
        _add_outline_filter(bool(on))

    def _add_put_outline_down(self) -> None:
        """Forget the outline preference and put the panel's tick down,
        silently (Accept, Cancel, Done / Undo all, a new box,
        patch switch, close)."""
        self._add_contour_on = False
        _add_outline_filter(False)
        self._push_add_ui(contour_on=False)

    def _on_add_contour_toggled(self, checked) -> None:
        """"Show candidates as outlines": a property set and
        nothing else; the propagation to the side views is the existing
        ``connect_label_sync`` contour event.  Remembered for the next
        Preview of this box; reset at Accept and every full exit."""
        on = bool(checked)
        self._add_contour_on = on
        try:
            self._add_set_contour(on)
            print("[ADD] outline={} - the three side views; the 3D view "
                  "cannot show outlines".format(int(on)))
        except Exception as exc:  # noqa: BLE001
            print("[ADD] could not set the outline: {!r}".format(exc))

    def _remove_add_candidate(self) -> None:
        cand = getattr(self, "_add_candidate_layer", None)
        self._add_candidate_layer = None
        # the outline's warning filter lives as long as the layer (A-M1)
        _add_outline_filter(False)
        if cand is None:
            return
        name = getattr(cand, "name", "_epicycle_add_candidate")
        # De-register from the SyncManager FIRST — the candidate is registered
        # (shared-data + 8 event handlers) on every preview but is an unmanaged
        # temp layer, so no removed-handler cleans it up; without this it leaks
        # a full-volume array + keeps the removed layer alive via the handler
        # refs (audit H1).
        sm = getattr(self, "sync_manager", None)
        if sm is not None:
            try:
                sm.disconnect_label_sync(cand)
            except Exception:
                pass
        try:
            self.subviewer_manager.remove_layer_from_subviewers(name)
        except Exception:
            pass
        if sm is not None:
            try:
                sm.unregister_layer(name)
            except Exception:
                pass
        try:
            cand.visible = False
            if cand in self.viewer.layers:
                self.viewer.layers.remove(cand)
        except Exception:
            pass

    def _remove_add_box_shapes(self) -> None:
        """Tear down the ROI-box overlay (all four vispy Line visuals).

        Named for its many callers (accept / reject / switch-teardown / close /
        reset).  The box is now a vispy overlay, not a Shapes layer, so this
        just detaches the visuals (``parent = None``) — there is nothing in
        ``viewer.layers`` to remove and no orphan to leave behind.  Idempotent.
        """
        overlay = getattr(self, "_add_box_overlay", None)
        if overlay is not None:
            try:
                overlay.destroy()
            except Exception:
                pass
        self._add_box_overlay = None
        self._add_box_shapes = None

    def _smooth_add_result(self, result, sigma: float = 1.5):
        """Isotropic SDF smooth of every piece -- delegates to the pure
        ``add_segment.smooth_pieces`` (``run_batch_recipe``,
        ``empty="erase"``, byte-identical)."""
        from packages.epicycle_annotator.add_segment import smooth_pieces

        return smooth_pieces(result, sigma)

    def _accept_add(self) -> None:
        """Commit the candidate pieces (background-only safe-write) → REVIEW."""
        if self._add_refused_while_batch("accept"):
            return
        if not getattr(self, "_add_running", False):
            return
        # the re-entry belt (A-L2, the A1-A4 audit): both footer buttons go
        # grey before any work, so a second click queued behind this one
        # finds them disabled; every refusal below re-pushes its own state
        self._push_add_ui(accept_enabled=False)
        result = self._add_preview_result
        shared = self._add_shared_volume()
        if result is None or shared is None or self._add_roi is None:
            return
        preview_roi = getattr(self, "_add_preview_roi", None)
        if preview_roi is not None and tuple(preview_roi) != tuple(
                self._add_roi):
            # cannot happen while every box change drops the preview; the
            # refusal is the invariant, not the hope
            from packages.epicycle_annotator.add_preview import (
                ADD_S1_INPUTS_CHANGED,
            )

            self._add_transient = ADD_S1_INPUTS_CHANGED
            self._add_refresh_ui()
            return
        ids = [int(v) for v in np.unique(result) if v != 0]
        if not ids:
            from packages.epicycle_annotator.add_preview import (
                ADD_S1_NOTHING_YET,
            )

            self._push_add_ui(status_text=ADD_S1_NOTHING_YET, status_kind="warn")
            return
        # Cherry-pick: commit ONLY the pieces the user left CHECKED in the
        # candidate list.  Filtering `result` down to the checked ids means
        # every downstream step (write_mask / committed / sizes / undo) keys off
        # the kept set with no extra plumbing.
        adv = self._get_advanced_features_widget()
        checked = (
            adv.get_checked_add_pieces()
            if adv is not None and hasattr(adv, "get_checked_add_pieces")
            else list(ids)
        )
        checked = [int(v) for v in checked if int(v) in set(ids)]
        if not checked:
            from packages.epicycle_annotator.add_preview import ADD_S1_TICK_ONE

            self._add_transient = ADD_S1_TICK_ONE
            self._add_refresh_ui()
            return
        # Work on a LOCAL `result` only — do NOT narrow self._add_preview_result
        # here (audit LOW): if the "voxels already taken" early-return below fires,
        # narrowing the model attr would strand the unchecked pieces so re-checking
        # them (without moving the box) couldn't recover them.  The attr stays the
        # full preview set until the None-clear at commit end.
        if set(checked) != set(ids):
            result = np.where(np.isin(result, checked), result, 0)
        # NOTE: auto-smooth already ran at PREVIEW time (WYSIWYG) — _add_preview_result
        # is the smoothed result, so we commit it as-is; no re-smooth here.
        roi = tuple(preview_roi) if preview_roi is not None else self._add_roi
        target = shared[roi]
        write_mask = (result > 0) & (target == 0)
        # The ids that WOULD land — only background voxels are written; do NOT
        # fall back to the candidate ids.  If the box was nudged fully onto
        # existing labels nothing is written and those ids are phantoms
        # (audit LOW-1).  Checked BEFORE the write + BEFORE the lock-releasing
        # try, so this stays in TUNE with the lock held for more tuning.
        committed = [int(v) for v in np.unique(result[write_mask]) if v != 0]
        if not committed:
            from packages.epicycle_annotator.add_preview import ADD_S1_TAKEN

            self._add_transient = ADD_S1_TAKEN
            self._push_add_ui(accept_enabled=False, accept_count=0)
            self._add_refresh_ui()
            return
        # The commit MUST release the lock on any failure or a committed-but-
        # unreleased Add strands the whole surface (audit MED-5).
        try:
            snap = shared[roi].copy()
            target[write_mask] = result[write_mask]
            # Per-piece area + centroid, computed on the ROI-sized committed
            # mask in ONE bounded pass (no full-volume scan).  The sizes are
            # surfaced in the REVIEW list so fragments are obvious at a glance;
            # the centroids ride the commit callback so the controller can
            # register each new df row WITHOUT its own per-piece
            # `data == id` + `np.argwhere` over the whole volume (that loop was
            # ~2.5 s of an 8.3 s accept with 47 pieces).  Add ids are
            # `max(volume) + 1`, so a fresh piece exists nowhere else and these
            # numbers are bit-identical to the full-volume scan.
            from packages.epicycle_annotator.roi_stats import (
                piece_stats_from_write,
            )

            piece_stats = piece_stats_from_write(
                result, write_mask, tuple(int(sl.start) for sl in roi)
            )
            size_map = {
                int(v): int(piece_stats[int(v)][0])
                for v in committed
                if int(v) in piece_stats
            }
            # COMMIT FIRST (B2 fix pass L-M1):
            # the volume is already written, and the controller's callback
            # is the ONLY thing that sets the dirty flag and pushes the
            # add_denovo undo entry -- so it runs before any teardown step
            # that could raise and lose the edit.
            committed_ok = True
            cb = getattr(self, "_on_add_committed_cb", None)
            if cb is not None:
                try:
                    cb(committed, roi_slices=roi, old_roi_data=snap,
                       piece_stats=piece_stats)
                except Exception as exc:
                    committed_ok = False
                    print(f"[ADD] commit callback raised: {exc!r}")
            if committed_ok:
                # Record this box's accept (ids + sizes) only once the
                # controller holds its undo entry: "Undo all" reuses that
                # proven path, so ids whose commit failed are NOT counted
                # (B2 fix pass L-L3).
                self._add_session.append(
                    {"ids": list(committed), "sizes": size_map})
            else:
                from packages.epicycle_annotator.add_preview import (
                    ADD_S1_COMMIT_FAILED,
                )

                self._add_transient = ADD_S1_COMMIT_FAILED
            # Tear down the tuning layers; sync committed labels into all
            # views.  The box stays DRAWN in REVIEW, without
            # its side-view callbacks, so "+ Add another" can re-place it.
            self._remove_add_candidate()
            self._drop_add_walls()
            self._remove_add_side_callbacks()
            self._remove_add_marker_layer()
            self._add_click_markers = []
            self._add_markers_base = None
            self._add_absorbed_ids = set()
            self._add_forget_volume_max()      # Accept wrote it (B-L3)
            self._add_bump_markers_epoch()
            self._add_place_markers_off()
            self._close_add_numbers()
            self._sync_all_views_after_operation(roi=roi)
            # REVIEW shows EVERY piece added this session, not just this box.
            all_ids = [i for e in self._add_session for i in e["ids"]]
            all_sizes = {}
            for e in self._add_session:
                all_sizes.update(e.get("sizes", {}))
            self._add_piece_ids = all_ids
            self._add_preview_result = None
            self._add_review = True
            self._add_put_outline_down()      # REVIEW has no candidates
            self._push_add_ui(
                review_active=True,
                piece_ids=all_ids,
                piece_sizes=all_sizes,
                add_batch_refine_enabled=bool(all_ids),
                start_enabled=True,
                accept_count=0,
                click_count=0,
                click_markers=[],
                footer_text="",
            )
        finally:
            # Release the lock exactly once, even if a step above raised.
            self._add_running = False
            self._fire_add_mode_lock(False)
            # REVIEW skips _reset_add_state, so stop the debounce timer here so a
            # pending tick can't fire post-Accept (audit LOW; the _add_running
            # guard already no-ops it, this is belt).
            _t = getattr(self, "_add_box_timer", None)
            if _t is not None:
                try:
                    _t.stop()
                except Exception:
                    pass
            self._add_abandon_workers()
            self._add_preview_res = None
            self._add_preview_roi = None
            # Restore normal subviewer drawing — REVIEW lets the user select +
            # refine pieces (pairs with the in-widget lock in _start_add).
            try:
                self._restore_ready_to_draw_state()
            except Exception:
                pass
            self._add_refresh_ui()

    def _accept_add_and_refine(self, *_a):
        """"Accept and refine ▸": ``_accept_add`` exactly as
        Accept, then ``_batch_refine_add_pieces`` exactly as "Refine all ▸"
        -- only when that Accept really committed (REVIEW reached AND the
        session grew; a refused or failed Accept has said its own sentence).
        Refused first with ``ADD_BATCH_BUSY``, like every other door."""
        if self._add_refused_while_batch("accept and refine"):
            return None
        # the re-entry belt (A-L2): grey both footer buttons first
        self._push_add_ui(accept_enabled=False)
        n_before = len(getattr(self, "_add_session", []) or [])
        self._accept_add()
        n_after = len(getattr(self, "_add_session", []) or [])
        if not getattr(self, "_add_review", False) or n_after <= n_before:
            print("[ADD] accept and refine: the Accept did not commit - "
                  "nothing handed to Batch Refine")
            return None
        return self._batch_refine_add_pieces()

    def _reject_add(self) -> None:
        """Cancel Add mode before commit — tear down, restore, release lock."""
        was_running = getattr(self, "_add_running", False)
        self._remove_add_candidate()
        self._remove_add_box_shapes()
        # Nothing was written to the shared volume pre-accept, so no restore
        # needed; just refresh the current ROI defensively.
        if self._add_roi is not None:
            try:
                self._sync_all_views_after_operation(roi=self._add_roi)
            except Exception:
                pass
        self._reset_add_state()
        if was_running:
            self._fire_add_mode_lock(False)
        # Re-enable subviewer drawing (pairs with the _start_add lock).
        try:
            self._restore_ready_to_draw_state()
        except Exception:
            pass
        adv = self._get_advanced_features_widget()
        if adv is not None and hasattr(adv, "clear_add_touched"):
            adv.clear_add_touched()          # a full exit
        self._push_add_ui(
            box_active=False, review_active=False,
            start_enabled=True, accept_enabled=False, accept_count=0,
            reject_enabled=False, click_count=0, click_markers=[],
            footer_text="",
        )
        self._add_refresh_ui()

    def _on_add_piece_selected(self, piece_id) -> None:
        """A REVIEW selection change: select the one object in the views
        (``piece_id`` is ``None`` unless exactly one row is selected) and
        refresh S1, whose second sentence follows the selection (B2 fix
        pass U-M3)."""
        if piece_id is not None:
            try:
                self._set_selected_label_all(int(piece_id))
            except Exception:
                pass
        self._add_refresh_ui()

    def _refine_add_piece(self) -> None:
        """Hand the selected piece to the Refine tab (pieces→refine handoff)."""
        adv = self._get_advanced_features_widget()
        if adv is None:
            return
        sel = adv.get_selected_add_pieces() if hasattr(adv, "get_selected_add_pieces") else []
        if not sel:
            return
        pid = int(sel[0])
        try:
            self._set_selected_label_all(pid)
        except Exception:
            pass
        # Switch to the Refine tab for SDF-smooth / reshape.  BY NAME --
        # the old hard-coded index pinned a construction order the panel
        # rework relabels.
        if hasattr(adv, "focus_tab"):
            try:
                adv.focus_tab("Refine")
            except Exception:
                pass
        from packages.epicycle_annotator.add_preview import (
            ADD_REFINING_OBJECT,
        )

        self._push_add_ui(status_text=ADD_REFINING_OBJECT.format(id=pid),
                         status_kind="review")

    def _batch_refine_add_pieces(self):
        """REVIEW → Batch Refine on the cells added this session.

        Targets the added pieces EXPLICITLY (not via navigator selection, which
        would be empty when the Label Navigator dock is closed and wouldn't
        override the dialog's "All active" default), and pre-checks the Selected
        radio so the obvious Run flow hits those pieces.  Reuses the existing
        dialog's full snapshot / preview / accept / undo machinery + every op
        (SDF smooth, dilate-erode, fill holes).  No parallel code path.
        """
        ids = [int(i) for i in getattr(self, "_add_piece_ids", []) if i]
        if not ids:
            from packages.epicycle_annotator.add_preview import (
                ADD_NO_ADDED_TO_REFINE,
            )

            self._push_add_ui(status_text=ADD_NO_ADDED_TO_REFINE,
                             status_kind="warn")
            return
        why = self._open_batch_refine_dialog(
            explicit_targets=ids, default_target="selected"
        )
        # A refused open says WHICH gate refused -- the stopping
        # worker, the lock, a staged edit, no volume -- in the Add status.
        if why:
            self._push_add_ui(status_text=why, status_kind="warn")
        return why

    def _smooth_add_pieces(self) -> None:
        """One-click SDF surface smooth on the added pieces.

        Opens Batch Refine on the added cells (via the explicit-target handler)
        and pre-sets an SDF smooth recipe directly on the dialog, so the user
        just clicks Run → staged preview → Accept.  Reuses the dialog's proven
        snapshot / preview / accept / undo machinery — no separate worker path,
        and no clobber of the user's sticky recipe (set on the dialog, not the
        _batch_last_recipe slot).
        """
        ids = [int(i) for i in getattr(self, "_add_piece_ids", []) if i]
        if not ids:
            from packages.epicycle_annotator.add_preview import (
                ADD_NO_ADDED_TO_SMOOTH,
            )

            self._push_add_ui(status_text=ADD_NO_ADDED_TO_SMOOTH,
                             status_kind="warn")
            return
        why = self._batch_refine_add_pieces()
        dlg = getattr(self, "_batch_refine_dialog", None)
        if why or dlg is None:
            # open refused — _batch_refine_add_pieces already said why (a
            # dialog still open from a running batch must not be touched)
            return
        try:
            from packages.epicycle_annotator.batch_refine import RefineStep

            dlg.set_recipe(
                [RefineStep(method="sdf",
                            params={"sigma_z": 3.0, "sigma_xy": 1.5})]
            )
            from packages.epicycle_annotator.batch_refine import (
                BATCH_PRESET_READY,
            )

            dlg.set_status(BATCH_PRESET_READY)
        except Exception as exc:
            print(f"[ADD] preset SDF smooth recipe failed: {exc!r}")

    def _merge_add_pieces(self) -> None:
        """Merge two over-split committed pieces (REVIEW) — DEFERRED to Phase 1.1.

        Kept as a stub: a correct merge has to create its own curation-undo
        entry (volume + the dropped-id DataFrame row) and interleaves with the
        per-commit "Undo all" counting, which is out of scope for the v1 cut.
        The button is hidden in the widget; this is never wired.
        """
        self._push_add_ui(status_text="Merge is coming in a later version.",
                         status_kind="warn")

    def _add_another(self) -> None:
        """Leave REVIEW and start a fresh box for another cell."""
        # BEFORE `_add_review` is cleared: a refusal leaves REVIEW as it is.
        if self._add_refused_while_batch("add another"):
            return
        self._add_review = False
        self._start_add(reuse_box=True)

    def _undo_add_all(self) -> None:
        """Revert EVERY de-novo Add commit from this REVIEW session.

        Delegates to the controller, which pops/applies the top *n*
        ``add_denovo`` curation-undo entries (one per Accept).  That reuses the
        PROVEN undo path — each entry restores its own volume crop + DataFrame
        snapshot, LIFO = reverse commit order (correct for overlapping boxes) —
        instead of a parallel annotator-side restore that desynced the global
        undo stack (re-audit MED + HIGH-1/HIGH-3).
        """
        if self._add_refused_while_batch("undo all"):
            return
        from packages.epicycle_annotator import add_preview as AP

        n = len(self._add_session)
        groups = [list(e.get("ids", [])) for e in self._add_session]
        n_obj = sum(len(g) for g in groups)
        if n and not self._add_confirm(AP.count_phrase(
                n_obj, AP.ADD_CONFIRM_UNDO_ALL_ONE, AP.ADD_CONFIRM_UNDO_ALL)):
            return
        undone = None
        cb = getattr(self, "_on_add_undo_session_cb", None)
        if cb is not None and n:
            try:
                undone = cb(n)
            except Exception as exc:
                print(f"[ADD] undo-session callback raised: {exc!r}")
        if undone == 0 and n:
            # Nothing was undone (a later edit -- an accepted batch, a
            # Refine -- sits on top): REVIEW stays, so the advice can be
            # followed by another Undo all (D-L7).
            self._add_transient = AP.ADD_UNDO_NONE
            self._add_refresh_ui()
            return
        self._reset_add_state()
        # Report the ACTUAL count -- the controller stops at an interleaved
        # non-Add edit, so never overclaim a full removal.
        if undone is not None and 0 < undone < n:
            k = sum(len(g) for g in groups[n - undone:])
            msg = (AP.ADD_UNDO_PARTIAL_ONE.format(k=k) if n_obj == 1
                   else AP.ADD_UNDO_PARTIAL.format(k=k, n=n_obj))
        else:
            msg = AP.ADD_UNDO_DONE
        self._push_add_ui(
            review_active=False, box_active=False, start_enabled=True,
            accept_enabled=False, accept_count=0, reject_enabled=False,
            piece_ids=[], footer_text="",
        )
        self._add_transient = msg
        self._add_refresh_ui()
        self._add_transient = None

    def _add_confirm(self, text) -> bool:
        """The themed Undo-all confirm.  A seam for the tests."""
        try:
            from napari_epicycle.ui_shell.theme import themed_confirm

            return bool(themed_confirm(self, "Undo all", str(text),
                                       confirm_text="Take them back",
                                       cancel_text="Keep them"))
        except Exception as exc:
            print(f"[ADD] confirm failed: {exc!r}")
            return False

    def _reset_add_state(self) -> None:
        """Return Add mode to idle (mirrors _reset_split_state)."""
        # forget the cached Advanced Features dock (final fix pass P1); a
        # plain attribute write so a borrowed-method harness needs nothing
        self._afw_cache = None
        # Stop the debounce timer so a pending fire can't land post-teardown
        # (audit L1; the _add_running guard already no-ops it, this is belt).
        t = getattr(self, "_add_box_timer", None)
        if t is not None:
            try:
                t.stop()
            except Exception:
                pass
        self._add_abandon_workers()
        for _fn in (self._remove_add_side_callbacks,
                    self._remove_add_marker_layer, self._close_add_numbers,
                    self._add_guide_thread_cleanup, self._drop_add_walls):
            try:
                _fn()
            except Exception:
                pass
        self._add_place_markers_on = False
        self._add_running = False
        self._add_review = False
        self._add_box = None
        self._add_roi = None
        self._add_preview_result = None
        self._add_preview_res = None
        self._add_preview_roi = None
        self._add_info = None
        self._add_info_brightness_key = None
        self._add_threshold_touched = False
        self._add_threshold_auto_written = None
        self._add_brightness_channel_seen = None
        self._add_transient = None
        self._add_click_markers = []
        self._add_markers_base = None
        self._add_absorbed_ids = set()
        self._add_rebase_note = None
        self._add_forget_volume_max()
        self._add_bump_markers_epoch()
        self._add_run_guide = None
        self._add_run_state = None
        self._add_dark_cache = {}
        self._add_nuclei_threshold_value = None
        self._add_nuclei_threshold_typed = False
        self._add_piece_ids = []
        self._add_session = []
        self._add_threshold_value = None
        # the outline tick goes down at every full exit
        self._add_put_outline_down()
        # DESTROY (not just null) the candidate layer + box overlay first — a
        # future caller that reaches reset without removing them would otherwise
        # orphan the overlay onto the persistent main canvas (audit LOW-1).  Both
        # removers are idempotent + no-op when the ref is already None.
        try:
            self._remove_add_candidate()
        except Exception:
            pass
        try:
            self._remove_add_box_shapes()
        except Exception:
            pass
        self._add_candidate_layer = None
        self._add_box_shapes = None
        self._add_box_overlay = None

    def _teardown_add_for_switch(self) -> None:
        """Synchronously clear any Add session at the START of a patch switch.

        Called by the controller BEFORE the new patch's layers load (next to the
        merge-dialog teardown).  A committed REVIEW session keeps its labels in
        the OLD patch, but its `_add_session` / `_add_review` state and the
        still-clickable "Undo all" / "Add another" buttons MUST be cleared now —
        otherwise a click during the async load runs `_undo_last_curation`
        against the NEW patch's volume + df (race audit MED-1: cross-patch
        corruption).  Idempotent; safe to call when Add is idle.
        """
        if not (getattr(self, "_add_running", False)
                or getattr(self, "_add_review", False)):
            return
        if getattr(self, "_add_running", False):
            # TUNE is normally switch-blocked, but cancel defensively.
            try:
                self._reject_add()
            except Exception:
                pass
            return
        # REVIEW: drop the transient session + UI WITHOUT undoing the committed
        # work (that stays in the old patch; the global Undo can revert it).
        for fn in (self._remove_add_candidate, self._remove_add_box_shapes,
                   self._reset_add_state):
            try:
                fn()
            except Exception:
                pass
        adv = self._get_advanced_features_widget()
        if adv is not None and hasattr(adv, "clear_add_touched"):
            adv.clear_add_touched()          # a full exit
        try:
            self._push_add_ui(
                review_active=False, box_active=False, start_enabled=True,
                accept_enabled=False, accept_count=0, reject_enabled=False,
                piece_ids=[], footer_text="",
            )
            self._add_refresh_ui()
        except Exception:
            pass

    def _run_batch_refine(self, label_ids) -> None:
        """Snapshot, lock, and launch the batch worker for *label_ids*."""
        from packages.epicycle_annotator.batch_refine import (
            BATCH_ADD_STEP,
            BATCH_NO_TARGETS,
            BATCH_NO_VOLUME,
            BATCH_STAGE_PREPARING,
        )

        dlg = getattr(self, "_batch_refine_dialog", None)
        if dlg is None:
            return
        labels_name = self._active_labels_name
        shared = (
            self.sync_manager.shared_labels.get(labels_name)
            if labels_name
            else None
        )
        # set_state("idle") clears the status line, so each refusal sets the
        # state FIRST and then its sentence (the other order showed nothing).
        if shared is None:
            dlg.set_state("idle")
            _batch_recount(dlg)
            dlg.set_status(BATCH_NO_VOLUME)
            return
        label_ids = [int(i) for i in label_ids]
        if not label_ids:
            dlg.set_state("idle")
            _batch_recount(dlg)
            dlg.set_status(BATCH_NO_TARGETS)
            return
        # Abandoned pool work from a stopped chain run still computing:
        # never a second run over it (the buttons are already off,
        # this answers a keyboard activation).
        n_orphans = self._live_batch_orphans()
        if n_orphans:
            from packages.epicycle_annotator.batch_refine import (
                BATCH_ORPHANS_RUNNING,
                objects_word,
            )

            dlg.set_state("idle")
            dlg.set_status(BATCH_ORPHANS_RUNNING.format(
                n=objects_word(n_orphans)))
            return
        # A saved Refine recipe takes the chain path -- BEFORE the geometry
        # builder is read, its "Add at least one step" check and the sticky
        # `_batch_last_recipe`.
        chain = dlg.chain_recipe() if hasattr(dlg, "chain_recipe") else None
        if chain is not None:
            self._run_batch_chain(dlg, shared, labels_name, label_ids, chain)
            return
        recipe = dlg.recipe()
        if recipe is None:  # invalid params — dialog already showed status
            dlg.set_state("idle")
            return
        if not recipe:
            dlg.set_state("idle")
            _batch_recount(dlg)
            dlg.set_status(BATCH_ADD_STEP)
            return

        # Sticky recipe (2026-06-07): capture the recipe of EVERY run (Preview
        # or full Run) AS SOON AS it's validated + non-empty, so a later dialog
        # re-open pre-populates the builder with the user's most recent attempt —
        # whether that run was later accepted OR rejected.  Serialized to the
        # round-trippable list-of-dicts form; the TARGET is NOT captured (labels
        # differ per patch, so the user re-picks all/selected/by-tier on reopen).
        from packages.epicycle_annotator.batch_refine import recipe_to_json

        self._batch_last_recipe = recipe_to_json(recipe)
        # A geometry run clears the chain a plain reopen would re-select.
        self._batch_last_chain_name = None
        self._batch_chain_steps = None
        self._batch_chain_has_settle = False
        self._launch_batch_worker(dlg, shared, labels_name, label_ids, recipe)

    def _run_batch_chain(self, dlg, shared, labels_name, label_ids,
                         recipe) -> None:
        """A saved Refine recipe over ``label_ids``.  Every refusal
        comes before any copy, worker or lock (invariant 11).

        Order (D-M1): the blocker on what is known without loading (the
        probability map's shape from its header); then the running state
        and its marquee; then the inputs -- the full map load, which can
        take seconds, now happens under the marquee -- and the blocker once
        more on what was loaded.  Anything that raises on the way puts the
        dialog back to idle with ``BATCH_LAUNCH_FAILED`` (UX audit M1): Run
        and Preview were already disabled by the click.
        """
        from packages.epicycle_annotator.batch_chain import OrphanList
        from packages.epicycle_annotator.batch_refine import (
            BATCH_LAUNCH_FAILED,
            label_bboxes,
        )

        name = dlg.chain_name()
        channel_name = dlg.chain_channel()

        def _refuse(block, when):
            dlg.set_state("idle")
            _batch_recount(dlg)
            dlg.set_status(block.text)
            self._refresh_batch_chain_block()
            print(f"[BATCH] recipe '{name}' refused {when}: "
                  f"{block.key} (step {block.step})")

        try:
            boxes = dict(_safe_attr(self, "_batch_bboxes") or {})
            missing = [i for i in label_ids if i not in boxes]
            if missing:
                boxes.update(label_bboxes(shared, missing))
            self._batch_bboxes = boxes

            def _block():
                run_b, prev_b = self._batch_blockers(
                    dlg, recipe, label_ids, for_preview=label_ids[0])
                return prev_b if len(label_ids) == 1 else run_b

            block = _block()
            if block is not None:
                _refuse(block, "before any work")
                return
            self._show_batch_preparing(dlg, len(label_ids))
            ch, pr, _av = self._batch_chain_inputs(recipe, channel_name)
            # Once more, on what the load actually returned (D-M1).
            block = _block()
            if block is not None:
                _refuse(block, "after reading its inputs")
                return
            spacing = self._refine_spacing()
        except Exception as exc:  # noqa: BLE001 -- never strand the dialog
            print(f"[BATCH] recipe '{name}' could not start: {exc!r}")
            try:
                dlg.set_state("idle")
                _batch_recount(dlg)
                dlg.set_status(BATCH_LAUNCH_FAILED)
            except Exception:
                pass
            return
        self._batch_last_chain_name = name
        self._batch_chain_steps = tuple(recipe.steps)
        self._batch_chain_has_settle = any(
            s.method == "settle" for s in recipe.steps)
        orphans = _safe_attr(self, "_batch_orphans")
        if not isinstance(orphans, OrphanList):
            # One lock for the worker's appends and the poll's prune (D-L1).
            self._batch_orphans = OrphanList(list(orphans or []))
        print(f"[BATCH] recipe '{name}' on {len(label_ids)} objects "
              f"(channel={channel_name!r}, prob={pr is not None})")
        self._launch_batch_worker(
            dlg, shared, labels_name, label_ids, recipe,
            spacing=spacing, channel=ch, prob=pr,
            orphans=self._batch_orphans)

    def _show_batch_preparing(self, dlg, n) -> None:
        """The running state, the marquee and "Preparing…" -- up BEFORE
        the first stall (the snapshot copy; a chain's probability map)."""
        from packages.epicycle_annotator.batch_refine import (
            BATCH_STAGE_PREPARING,
            objects_word,
        )

        dlg.set_state("running")
        self._batch_n_targets = int(n)
        self._batch_eta_t0 = None
        dlg.set_busy()
        dlg.set_status(BATCH_STAGE_PREPARING.format(n=objects_word(int(n))))
        try:
            # Paints the dialog now without pumping the event loop.
            dlg.repaint()
        except Exception:
            pass

    def _launch_batch_worker(self, dlg, shared, labels_name, label_ids,
                             recipe, **worker_kw) -> None:
        """Snapshot, lock and start the worker -- shared by the geometry and
        the chain paths.  Every slot is connected BEFORE ``start()``."""
        # The running state and its marquee FIRST (UX audit L7): the copy
        # below is the first stall on a big patch, and it had no indicator.
        self._show_batch_preparing(dlg, len(label_ids))

        from qtpy.QtCore import QThread

        from napari_epicycle.annotator.batch_refine_worker import (
            BatchRefineWorker,
        )

        # Stale-token belt: every handler compares the FINISHING
        # worker's token with this one.
        self._batch_token = int(_safe_attr(self, "_batch_token") or 0) + 1
        # Launch-failure guard: EVERYTHING after the running state is inside
        # it -- the snapshot copy included (UX audit M2: a MemoryError there
        # left "running" for ever, Cancel a no-op).  On failure: release the
        # lock, reset the state, restore drawing if it was locked, and say
        # so on screen (mirrors _on_batch_worker_error).  MemoryError is an
        # Exception, so one clause covers it.
        drawing_locked = False
        try:
            # Pre-batch snapshot (full copy; cropped to the union once
            # staged).  The worker reads THIS copy (not the live `shared`)
            # so there is no concurrent main-thread/worker access to the
            # live volume.
            self._batch_snapshot = shared.copy()
            self._batch_labels_name = labels_name  # patch-switch guard
            self._batch_staged = False
            self._batch_running = True
            self._batch_touched_ids = []
            self._batch_union_roi = None

            # Lock the WHOLE Step-3 surface for the duration of the batch
            # op (worker compute + staged-preview window): the controller
            # ORs _batch_refine_running into its unified lock → predict /
            # reconstruct / extract / commit / save / step-nav /
            # patch-switch + the navigator action buttons (via
            # set_load_locked) all lock until accept/reject/error/cancel.
            self._fire_batch_refine_lock(True)
            # Lock drawing for the duration of the run.  (The ETA clock
            # starts when the worker reports "refining", .)
            self._lock_subviewers_for_preview()
            drawing_locked = True

            self._batch_thread = QThread()
            self._batch_worker = BatchRefineWorker(
                self._batch_snapshot, label_ids, recipe, **worker_kw
            )
            self._batch_worker._epicycle_token = self._batch_token
            self._batch_worker.moveToThread(self._batch_thread)
            self._batch_thread.started.connect(self._batch_worker.run)
            self._batch_worker.stage.connect(self._on_batch_worker_stage)
            self._batch_worker.progress.connect(self._on_batch_worker_progress)
            self._batch_worker.finished.connect(self._on_batch_worker_finished)
            self._batch_worker.error.connect(self._on_batch_worker_error)
            self._batch_thread.start()
            print(f"[BATCH] worker started on {len(label_ids)} labels")
        except Exception as exc:  # noqa: BLE001 — never strand the surface
            self._batch_running = False
            self._batch_snapshot = None
            self._batch_thread = None
            self._batch_worker = None
            self._fire_batch_refine_lock(False)
            if drawing_locked:
                self._restore_ready_to_draw_state()
            self._sync_refine_state()
            try:
                from packages.epicycle_annotator.batch_refine import (
                    BATCH_LAUNCH_FAILED,
                )

                # A fixed sentence on screen; the raw text to the log below.
                dlg.set_state("idle")
                _batch_recount(dlg)
                dlg.set_status(BATCH_LAUNCH_FAILED)
            except Exception:
                pass
            print(f"[BATCH] launch FAILED — controller lock released: {exc!r}")

    def _on_batch_worker_stage(self, stage, done, total) -> None:
        """Worker stage change: preparing and writing are
        a marquee; "refining" starts the ETA clock, here on the main
        thread, and makes the bar determinate."""
        from packages.epicycle_annotator.batch_refine import (
            BATCH_STAGE_PREPARING,
            BATCH_STAGE_REFINING,
            BATCH_STAGE_WRITING,
            objects_word,
        )

        if self._batch_is_stale("stage"):
            return
        dlg = getattr(self, "_batch_refine_dialog", None)
        n = int(getattr(self, "_batch_n_targets", 0) or total)
        print(f"[BATCH] stage {stage} {int(done)}/{int(total)}")
        if stage == "refining":
            self._batch_eta_t0 = _time.monotonic()
        if dlg is None or self._batch_cancel_requested():
            return
        if stage == "preparing":
            dlg.set_busy()
            dlg.set_status(BATCH_STAGE_PREPARING.format(n=objects_word(n)))
        elif stage == "refining":
            dlg.set_progress(int(done), int(total))
            dlg.set_status(
                BATCH_STAGE_REFINING.format(done=int(done), total=int(total))
            )
        elif stage == "writing":
            dlg.set_busy()
            dlg.set_status(BATCH_STAGE_WRITING.format(
                n=objects_word(int(total) or n)))

    def _on_batch_worker_progress(self, done, total) -> None:
        from packages.epicycle_annotator.batch_refine import (
            BATCH_ETA_SEP,
            BATCH_STAGE_REFINING,
            eta_text,
        )

        if self._batch_is_stale("progress"):
            return
        dlg = getattr(self, "_batch_refine_dialog", None)
        if dlg is not None:
            dlg.set_progress(int(done), int(total))
            if self._batch_cancel_requested():
                return
            text = BATCH_STAGE_REFINING.format(done=int(done), total=int(total))
            t0 = getattr(self, "_batch_eta_t0", None)
            eta = (
                eta_text(_time.monotonic() - t0, done, total)
                if t0 is not None
                else ""
            )
            dlg.set_status(text + (BATCH_ETA_SEP + eta if eta else ""))

    def _on_batch_worker_finished(self, result) -> None:
        """Worker done — swap the changed region into the live volume and
        move to the staged state (or discard on cancel / no-op)."""
        if self._batch_is_stale("finished"):
            return
        self._batch_running = False
        self._batch_thread_cleanup()
        # A stopped chain run leaves pool work behind: keep the poll alive
        # until it drains, so Run comes back by itself.
        if self._live_batch_orphans():
            self._start_batch_poll()
        dlg = getattr(self, "_batch_refine_dialog", None)

        # Dialog gone (closed mid-run) → discard; volume was never mutated.
        # Terminal (no staged state) → release the controller lock.
        if dlg is None:
            self._batch_snapshot = None
            self._fire_batch_refine_lock(False)
            self._restore_ready_to_draw_state()
            self._sync_refine_state()
            print("[BATCH] finished but dialog gone — discarded")
            return

        # Patch switched mid-run → results target the OLD patch; discard.
        # Terminal → release the controller lock.
        if self._active_labels_name != getattr(self, "_batch_labels_name", None):
            self._batch_snapshot = None
            self._fire_batch_refine_lock(False)
            from packages.epicycle_annotator.batch_refine import (
                BATCH_PATCH_CHANGED,
            )

            dlg.set_state("idle")
            dlg.set_status(BATCH_PATCH_CHANGED)
            self._restore_ready_to_draw_state()
            self._sync_refine_state()
            print("[BATCH] finished but patch changed — discarded")
            return

        from packages.epicycle_annotator.batch_refine import (
            BATCH_NONE_CHANGED,
            BATCH_NONE_CHANGED_LISTED,
            summary_lines,
        )

        emptied = list(getattr(result, "emptied_ids", []) or [])
        absent = list(getattr(result, "absent_ids", []) or [])
        # Chain path: {id: (step, method word)} for the summary.
        steps = _safe_attr(self, "_batch_chain_steps") or ()
        from packages.epicycle_annotator.refine_recipe import METHOD_WORD

        stopped = {}
        for _lid, (_k, _reason) in dict(
                getattr(result, "stopped", {}) or {}).items():
            _word = (METHOD_WORD.get(steps[_k - 1].method, "")
                     if 0 < _k <= len(steps) else "")
            # The reason picks the line's wording (step 1, an enclosed
            # Settle); it is never shown itself.
            stopped[int(_lid)] = (int(_k), _word, str(_reason or ""))
        extra = {"stopped": stopped} if stopped else {}

        # Cancelled / no-op → discard.  Terminal → release the controller lock.
        if result.cancelled or not result.touched_ids:
            self._batch_snapshot = None
            self._fire_batch_refine_lock(False)
            dlg.set_state("idle")
            if result.cancelled:
                from packages.epicycle_annotator.batch_refine import (
                    BATCH_CANCELLED,
                )

                dlg.set_status(BATCH_CANCELLED)
            else:
                # Name the objects a step would have emptied, and the ones
                # no longer in the volume.
                lines = summary_lines(emptied=emptied, absent=absent,
                                      **extra)
                dlg.set_status(
                    BATCH_NONE_CHANGED_LISTED.format(lines="\n".join(lines))
                    if lines
                    else BATCH_NONE_CHANGED
                )
            self._restore_ready_to_draw_state()
            self._sync_refine_state()
            print(
                f"[BATCH] finished cancelled={result.cancelled} "
                f"touched={len(result.touched_ids)} emptied={emptied} "
                f"absent={absent} — discarded"
            )
            return

        # Apply only the changed region into the live shared volume.
        labels_name = self._active_labels_name
        shared = self.sync_manager.shared_labels.get(labels_name)
        union = result.union_roi
        if shared is not None and result.working_volume.shape == shared.shape:
            if union is not None:
                shared[union] = result.working_volume[union]
            else:
                shared[:] = result.working_volume

        # Crop the snapshot to the union ROI — that's all Reject / undo need.
        if union is not None and getattr(self, "_batch_snapshot", None) is not None:
            self._batch_snapshot = self._batch_snapshot[union].copy()
        self._batch_union_roi = union
        self._batch_touched_ids = list(result.touched_ids)
        self._batch_staged = True

        self._sync_all_views_after_operation(roi=union)
        # The refined count is touched_ids, never n_processed (which counts
        # the emptied objects too).
        dlg.set_preview_summary(
            len(result.touched_ids),
            result.n_changed_total,
            emptied=emptied,
            absent=absent,
            **extra,
        )
        from packages.epicycle_annotator.batch_refine import BATCH_STAGED

        # The "Writing…" line must not stay next to Accept / Reject (M1).
        dlg.set_status(BATCH_STAGED)
        dlg.set_state("staged")
        self._sync_refine_state()
        # NOTE: do NOT release the controller lock here — the batch is now
        # STAGED (preview pending accept/reject) and the volume is not final.
        # The lock is cleared only on accept / reject / error / cancel / close.
        print(
            f"[BATCH] staged: {len(result.touched_ids)} labels, "
            f"{result.n_changed_total} voxels changed, emptied={emptied} "
            f"absent={absent}"
        )

    def _on_batch_worker_error(self, msg) -> None:
        if self._batch_is_stale("error"):
            return
        self._batch_running = False
        self._batch_thread_cleanup()
        self._batch_snapshot = None
        # Terminal → release the controller lock.
        self._fire_batch_refine_lock(False)
        dlg = getattr(self, "_batch_refine_dialog", None)
        if dlg is not None:
            from packages.epicycle_annotator.batch_refine import BATCH_FAILED

            # A fixed sentence on screen; the raw text goes to the log below.
            dlg.set_state("idle")
            dlg.set_status(BATCH_FAILED)
        self._restore_ready_to_draw_state()
        self._sync_refine_state()
        print(f"[BATCH] worker error: {msg}")

    def _batch_cancel_requested(self) -> bool:
        """True once Cancel reached the running worker: a stage or progress
        tick still queued behind it must not overwrite "Cancelling…"."""
        w = _safe_attr(self, "_batch_worker")
        ev = _safe_attr(w, "cancel_event") if w is not None else None
        try:
            return bool(ev is not None and ev.is_set())
        except Exception:
            return False

    def _batch_is_stale(self, what) -> bool:
        """True for a signal from a worker this session has moved past (its
        launch token is not the current one).  Printed; the caller returns
        WITHOUT touching ``_batch_thread``.  No sender (a direct call)
        abstains."""
        tok = self._result_token()
        if tok is None or tok == _safe_attr(self, "_batch_token"):
            return False
        print(f"[BATCH] stale {what} from token {tok} ignored "
              f"(current {_safe_attr(self, '_batch_token')})")
        return True

    def _batch_thread_cleanup(self) -> None:
        th = getattr(self, "_batch_thread", None)
        if th is not None:
            try:
                th.quit()
                # Worker polls cancel between labels; a single label's
                # compute is ROI-bounded, so 3s after cancel is generous.
                if not th.wait(3000):
                    print(
                        "[BATCH] WARNING: worker thread did not exit within 3s"
                    )
            except Exception:
                pass
        self._batch_thread = None
        self._batch_worker = None

    def _on_batch_accept(self) -> None:
        """Commit the staged batch: fire the controller callback (marks
        labels curated + pushes ONE curation-undo entry over the union
        ROI) and tear the dialog down."""
        if not getattr(self, "_batch_staged", False):
            return
        touched = list(getattr(self, "_batch_touched_ids", []))
        union_roi = getattr(self, "_batch_union_roi", None)
        # The cropped snapshot IS the pre-batch mask of the union ROI.
        old_union = getattr(self, "_batch_snapshot", None)
        # "Keep original QC tier" opt-out — read before the dialog is torn down.
        dlg = getattr(self, "_batch_refine_dialog", None)
        keep_tier = bool(dlg.keep_original_tier()) if dlg is not None else False

        # Record the just-applied recipe in the session-recent history (read
        # the dialog's builder BEFORE teardown).  Serialized + de-duped so a
        # later dialog re-open can offer it under "Recent recipes".
        # A chain run is never appended: the histories stay apart.
        is_chain = dlg is not None and hasattr(dlg, "chain_name") and \
            dlg.chain_name() is not None
        if dlg is not None and not is_chain:
            try:
                applied = dlg.recipe()  # validated list[RefineStep] | None
                if applied:
                    self._append_batch_recipe_history(applied)
            except Exception as exc:
                print(f"[BATCH] recipe-history capture failed: {exc!r}")

        cb = getattr(self, "_on_batch_refined_cb", None)
        if cb is not None:
            try:
                cb(
                    touched,
                    roi_slices=union_roi,
                    old_roi_data=old_union,
                    mark_curated=not keep_tier,
                )
            except Exception as exc:
                print(f"[BATCH] accept callback raised: {exc!r}")

        self._batch_snapshot = None
        self._batch_staged = False
        self._batch_touched_ids = []
        self._batch_union_roi = None
        # Terminal (commit) → release the controller lock AFTER the accept
        # callback finalized the volume + df.
        self._fire_batch_refine_lock(False)
        self._teardown_batch_dialog()
        self._restore_ready_to_draw_state()
        self._sync_refine_state()
        self._refresh_merge_button_state()
        self._refresh_label_action_buttons_state()
        print(f"[BATCH] accepted {len(touched)} labels")

    def _on_batch_reject(self) -> None:
        """Discard the staged batch: restore the pre-batch union ROI."""
        if not getattr(self, "_batch_staged", False):
            return
        labels_name = self._active_labels_name
        shared = self.sync_manager.shared_labels.get(labels_name)
        union_roi = getattr(self, "_batch_union_roi", None)
        snap = getattr(self, "_batch_snapshot", None)
        if shared is not None and snap is not None and union_roi is not None:
            shared[union_roi] = snap
            self._sync_all_views_after_operation(roi=union_roi)
        elif shared is not None and snap is not None:
            shared[:] = snap
            self._sync_all_views_after_operation()

        self._batch_snapshot = None
        self._batch_staged = False
        self._batch_touched_ids = []
        self._batch_union_roi = None
        # Terminal → release the controller lock.
        self._fire_batch_refine_lock(False)
        # Sticky recipe (2026-06-07): Reject discards the RESULT but keeps the
        # dialog OPEN in editable/idle so the user can tweak + re-run the same
        # recipe without rebuilding.  set_state("idle") re-enables the target +
        # recipe rows WITHOUT clearing them (the _RecipeRow widgets survive).
        # The _batch_refine_dialog ref STAYS set — do NOT null it.  Idle =
        # surface unlocked (the lock was just released above); a subsequent Run
        # re-locks.  Defensive fall-back to teardown if the dialog ref is
        # somehow gone (shouldn't happen on a staged reject).
        dlg = getattr(self, "_batch_refine_dialog", None)
        if dlg is not None:
            dlg.set_state("idle")
        else:
            self._teardown_batch_dialog()
        self._restore_ready_to_draw_state()
        self._sync_refine_state()
        self._refresh_merge_button_state()
        self._refresh_label_action_buttons_state()
        print("[BATCH] rejected — restored pre-batch state, dialog kept open (idle)")

    def _on_batch_cancel(self) -> None:
        """Request cancellation of the running worker."""
        worker = getattr(self, "_batch_worker", None)
        if worker is not None:
            worker.cancel()
        dlg = getattr(self, "_batch_refine_dialog", None)
        if dlg is not None:
            from packages.epicycle_annotator.batch_refine import (
                BATCH_CANCELLING,
                BATCH_CANCELLING_SETTLE,
            )

            # A Settle object in flight runs to its end (no cancel event).
            dlg.set_status(
                BATCH_CANCELLING_SETTLE
                if _safe_attr(self, "_batch_chain_has_settle")
                and _safe_attr(self, "_batch_worker") is not None
                else BATCH_CANCELLING)
        print("[BATCH] cancel requested")

    def _on_batch_close(self) -> None:
        """Fired by the dialog's closeEvent for a non-clean close
        (X button / parent destroy).  Treat as Reject (if staged) or
        Cancel (if running); the dialog is already closing."""
        # The present-id / bbox cache lives exactly as long as the dialog.
        self._batch_present_ids = None
        self._batch_bboxes = None
        # Staged → restore the pre-batch state.  Terminal → release the lock.
        if getattr(self, "_batch_staged", False):
            self._batch_refine_dialog = None
            labels_name = self._active_labels_name
            shared = self.sync_manager.shared_labels.get(labels_name)
            union_roi = getattr(self, "_batch_union_roi", None)
            snap = getattr(self, "_batch_snapshot", None)
            if shared is not None and snap is not None and union_roi is not None:
                shared[union_roi] = snap
                self._sync_all_views_after_operation(roi=union_roi)
            self._batch_snapshot = None
            self._batch_staged = False
            self._batch_touched_ids = []
            self._batch_union_roi = None
            self._fire_batch_refine_lock(False)
            self._restore_ready_to_draw_state()
            self._sync_refine_state()
            print("[BATCH] closed while staged — restored")
            return
        # Running → cancel; the finished handler discards (dialog is None) and
        # releases the lock there.  When there is no live worker, the finished
        # handler won't fire, so release here.  (The callback is idempotent, so
        # the worker-cancel path firing it in both places is harmless.)
        self._batch_refine_dialog = None
        worker = getattr(self, "_batch_worker", None)
        if worker is not None:
            worker.cancel()
        else:
            self._fire_batch_refine_lock(False)
            self._restore_ready_to_draw_state()
        self._sync_refine_state()
        print("[BATCH] closed (idle/running)")

    def _teardown_batch_dialog(self) -> None:
        """Actively close + drop the dialog (Accept/Reject button paths)."""
        dlg = getattr(self, "_batch_refine_dialog", None)
        self._batch_refine_dialog = None
        self._batch_token = int(_safe_attr(self, "_batch_token") or 0) + 1
        # The present-id / bbox cache lives exactly as long as the dialog.
        self._batch_present_ids = None
        self._batch_bboxes = None
        if dlg is not None:
            try:
                dlg.mark_clean_close()
                dlg.close()
                dlg.deleteLater()
            except Exception:
                pass

    def _discard_batch_refine(self) -> None:
        """OperationSlot.reject hook — discard any in-flight/staged batch.

        Called by ``_discard_uncommitted_work`` (e.g. on annotator close
        with the user choosing Discard).  For a running batch, cancel and
        block until the worker thread exits so teardown is clean.
        """
        self._batch_token = int(_safe_attr(self, "_batch_token") or 0) + 1
        if getattr(self, "_batch_running", False):
            worker = getattr(self, "_batch_worker", None)
            if worker is not None:
                worker.cancel()
            self._batch_thread_cleanup()
            self._batch_running = False
            self._batch_snapshot = None
            # Terminal (synchronous discard — the finished handler won't fire
            # since the thread was quit directly) → release the controller lock.
            self._fire_batch_refine_lock(False)
            self._teardown_batch_dialog()
            self._restore_ready_to_draw_state()
            self._sync_refine_state()
        elif getattr(self, "_batch_staged", False):
            # Delegates to _on_batch_reject which releases the lock.
            self._on_batch_reject()

    def _on_discard_garbage_from_navigator(self, garbage_ids) -> None:
        """Forward discard-garbage from navigator: zero labels, refresh, notify."""
        if not garbage_ids or self._split_running:
            return
        lbl_name = self.labels_combo.currentText()
        roi_slices = None
        old_roi_data = None
        if lbl_name and lbl_name in self.viewer.layers:
            layer = self.viewer.layers[lbl_name]
            # Capture combined ROI for all garbage labels before zeroing.
            import numpy as np
            combined_mask = np.isin(layer.data, garbage_ids)
            coords = np.argwhere(combined_mask)
            if coords.size > 0:
                margin = 2
                mins = np.maximum(coords.min(axis=0) - margin, 0)
                maxs = np.minimum(
                    coords.max(axis=0) + margin + 1,
                    np.array(layer.data.shape),
                )
                roi_slices = tuple(
                    slice(int(mn), int(mx)) for mn, mx in zip(mins, maxs)
                )
                old_roi_data = layer.data[roi_slices].copy()
            # Single-pass zeroing: reuse the combined_mask instead of
            # looping over each garbage ID (avoids N full-volume scans).
            layer.data[combined_mask] = 0
            layer.refresh()
        try:
            self._refresh_label_data_in_subviewers()
        except Exception:
            pass
        if self._on_discard_garbage_cb:
            try:
                self._on_discard_garbage_cb(
                    garbage_ids,
                    roi_slices=roi_slices,
                    old_roi_data=old_roi_data,
                )
            except Exception:
                pass

    # ---------------------------------------------------------------
    # set_focused_refinement override — also expands Navigation
    # ---------------------------------------------------------------

    def set_focused_refinement(self, flagged_ids, centroids):
        """Override parent to auto-open the Label Navigator when labels are pushed."""
        super().set_focused_refinement(flagged_ids, centroids)
        if flagged_ids and self._3d_curation_mode:
            self._open_label_navigator()

    def _current_flagged_label(self):
        """Return the label ID currently being navigated, or None.

        Checks the floating navigator first (table selection), then falls
        back to the parent's index-based navigation.
        """
        if self._navigator_widget is not None:
            try:
                lid = self._navigator_widget.current_label_id()
                if lid is not None:
                    return lid
            except Exception:
                pass
        # Fallback to parent index-based logic
        if not self._flagged_label_ids:
            return None
        idx = self._flagged_nav_index
        if idx < 0 or idx >= len(self._flagged_label_ids):
            return None
        return self._flagged_label_ids[idx]

    def _remove_from_flagged(self, label_id):
        """Remove a label from the parent's flagged list."""
        if label_id in self._flagged_label_ids:
            idx = self._flagged_label_ids.index(label_id)
            self._flagged_label_ids.remove(label_id)
            self._flagged_centroids.pop(label_id, None)
            if not self._flagged_label_ids:
                self._flagged_nav_index = -1
                return
            if idx >= len(self._flagged_label_ids):
                idx = 0
            self._flagged_nav_index = idx - 1
            self._navigate_next_flagged()
            self._focused_nav_label.setText(
                f"{len(self._flagged_label_ids)} flagged labels remaining"
            )

    # ---------------------------------------------------------------
    # Slice step size for arrow-key scrolling
    # ---------------------------------------------------------------

    def _on_step_size_changed(self, _value: int) -> None:
        """Store the new step size (event filters read it live)."""
        # Nothing else to do — the _SliceStepFilter reads _slice_step_sb
        # each time an arrow key is pressed.
        pass

    def _apply_smart_step_size(self) -> None:
        """Set a reasonable default scroll step based on subviewer dimension sizes.

        Uses max(1, min_dim // 50) so ~50 key presses traverses the smallest view.
        """
        sizes = []
        for vm in [self.viewer_model_xy, self.viewer_model_xz, self.viewer_model_yz]:
            try:
                dim_size = int(vm.dims.range[0][1])
                if dim_size > 0:
                    sizes.append(dim_size)
            except Exception:
                pass
        if sizes:
            smart_step = max(1, min(sizes) // 50)
            self._slice_step_sb.setValue(smart_step)

    def _remove_arrow_key_filters(self) -> None:
        """Take the filters the last install put on, off, and delete them
        (``removeEventFilter`` + ``deleteLater``).  Idempotent."""
        for filt, targets in list(getattr(self, "_arrow_key_filters",
                                          None) or []):
            for target in targets:
                try:
                    target.removeEventFilter(filt)
                except Exception:
                    pass            # the target is already gone
            try:
                filt.deleteLater()
            except Exception:
                pass
        self._arrow_key_filters = []

    def _install_arrow_key_filters(self) -> None:
        """Install event filters on subviewer widgets for arrow-key control.

        Four-layer defense for arrow key management:
        1. _SliceStepFilter on canvas.native — handles Left/Right stepping and
           routes Up/Down to Label Navigator prev/next.
        2. Qt.NoFocus on custom slice sliders — prevents QSlider from capturing
           keyboard focus (sliders still work via mouse drag/click/wheel).
        3. _UpDownBlocker on ALL child widgets of each qt_viewer — routes
           Up/Down to Label Navigator prev/next on any focusable widget
           (canvas overlays, hidden napari dims, etc.).
        4. bind_key on main viewer — blocks all arrow keys on the main napari
           viewer to prevent accidental main-viewer scrolling while annotating.
        """
        nav_cb = self._on_subviewer_navigate
        nudge_cb = self._add_nudge_box   # Shift+arrow, Add only
        # this runs on every patch switch: the previous set comes off first
        # (B3 fix pass, LOW-3 -- they used to pile up, 6 more per switch)
        self._remove_arrow_key_filters()
        installed = []

        # --- Layer 1: _SliceStepFilter on canvas.native (Left/Right + Up/Down)
        for orient, qt_viewer, vm in [
            ("XY", self.qt_viewer_xy, self.viewer_model_xy),
            ("XZ", self.qt_viewer_xz, self.viewer_model_xz),
            ("YZ", self.qt_viewer_yz, self.viewer_model_yz),
        ]:
            try:
                canvas = getattr(qt_viewer, "canvas", None)
                native = getattr(canvas, "native", None) if canvas else None
                target = native if native is not None else qt_viewer
                filt = _SliceStepFilter(
                    vm, self._slice_step_sb,
                    navigate_callback=nav_cb, parent=target,
                    nudge_callback=nudge_cb, orient=orient,
                )
                target.installEventFilter(filt)
                installed.append((filt, [target]))
            except Exception:
                pass

        # --- Layer 2: NoFocus on custom slice sliders
        for slider in [
            getattr(self, "xy_slice_slider", None),
            getattr(self, "xz_slice_slider", None),
            getattr(self, "yz_slice_slider", None),
        ]:
            if slider is not None:
                try:
                    slider.setFocusPolicy(Qt.NoFocus)
                except Exception:
                    pass

        # --- Layer 3: _UpDownBlocker on all subviewer child widgets
        # Routes Up/Down to Navigator prev/next on any focusable child
        # (canvas overlay, hidden dims widgets, etc.) that Layer 1 misses.
        # One blocker per side view, so its Shift branch knows the view.
        for orient, qt_viewer in [("XY", self.qt_viewer_xy),
                                  ("XZ", self.qt_viewer_xz),
                                  ("YZ", self.qt_viewer_yz)]:
            blocker = _UpDownBlocker(navigate_callback=nav_cb, parent=self,
                                     nudge_callback=nudge_cb, orient=orient)
            targets = []
            try:
                qt_viewer.installEventFilter(blocker)
                targets.append(qt_viewer)
                for child in qt_viewer.findChildren(QWidget):
                    child.installEventFilter(blocker)
                    targets.append(child)
            except Exception:
                pass
            installed.append((blocker, targets))
        self._arrow_key_filters = installed

        # --- Layer 4: Block all arrow keys on the main napari viewer
        # Prevents the main viewer from scrolling dims when the user presses
        # arrow keys.  Subviewer scrolling is handled by hover-based routing
        # (Left/Right from the Label Navigator) and _SliceStepFilter (Left/Right
        # when canvas has focus).
        self._block_main_viewer_arrow_keys()

    def _close_cleanup(self) -> None:
        """Perform all teardown logic for closeEvent.

        Extracted so tests can exercise the full teardown sequence without
        requiring Qt infrastructure (``super().closeEvent()``).

        **Re-entrancy guard:** ``_close_cleanup_done`` prevents double-execution.
        This can happen when ``_run_widget_cleanup`` calls ``_close_cleanup``
        explicitly, and then Qt's ``closeEvent`` (triggered by ``deleteLater``)
        calls it again.

        **Safe ordering:**

        1. Force-reject active operations (restore shared volume)
        2. Lightweight isolation state cleanup (if active) — but do NOT
           fire ``_on_isolation_cleared_cb`` here (moved to step 8)
        3. Stop ``_alive_check_timer`` (debounce timer)
        4. Clear ``_navigator_widget`` reference (dock cleaned up by launcher)
        5. Disconnect viewer-level event handlers
        6. Disconnect ``viewer.layers.events.inserted`` handler
        7. Disconnect layer-level handlers
        8. Fire ``_on_isolation_cleared_cb`` UNCONDITIONALLY — after all event
           handlers are disconnected so the controller's colormap update
           cannot cascade into half-destroyed subviewers
        9. Restore main viewer arrow key bindings
        """
        # Re-entrancy guard: prevent double-execution across close paths.
        if getattr(self, "_close_cleanup_done", False):
            return
        self._close_cleanup_done = True

        # 2026-09-02: a perf re-show (QTimer) scheduled in this same tick
        # must not fire on a torn-down widget / a layer already removed.
        # Inline token bump (== _cancel_main_perf_reshow) — lightweight
        # test doubles run _close_cleanup without the helper bound.
        self._perf_reshow_token = getattr(self, "_perf_reshow_token", 0) + 1

        # --- FM2 fix (2026-05-16): cancel any open merge dialog BEFORE
        # the `_is_closing` flag is set.  Once `_is_closing=True`, the
        # MergeDialog.closeEvent guard at merge_dialog.py:645-660 skips
        # firing the cancel callback (a defensive guard against
        # partially-destroyed parent state), which leaves the dialog's
        # multi-label sub-proxies orphaned in the viewer.  Routing
        # through `_cancel_merge_dialog` here cleans up isolation +
        # restores any post-merge snapshot + nulls the dialog ref while
        # everything is still fully alive.
        dlg = getattr(self, "_merge_dialog", None)
        if dlg is not None:
            stage = "?"
            try:
                stage = getattr(dlg.state, "stage", "?")
            except Exception:
                pass
            print(
                f"[MERGE-CLOSE] _close_cleanup detected open merge dialog — "
                f"routing through _cancel_merge_dialog before _is_closing=True "
                f"stage={stage}"
            )
            try:
                self._cancel_merge_dialog()
            except Exception as e:
                print(f"[MERGE-CLOSE] _cancel_merge_dialog raised: {e!r}")
            # Close the dialog widget itself so its Qt resources release.
            # _clean_close=True prevents the dialog's own closeEvent from
            # re-firing cancel (idempotent).
            try:
                dlg._clean_close = True
                dlg.close()
            except Exception:
                pass

        import time as _time

        _t0 = _time.perf_counter()
        _safe_curation_debug(self, "close cleanup start")
        self._is_closing = True
        try:
            self._invalidate_session_callbacks()
        except Exception:
            try:
                self._session_token = int(getattr(self, "_session_token", 0)) + 1
            except Exception:
                self._session_token = 1

        # --- Step 1: Force-reject active operations ---
        if self._smooth_running:
            self._reject_smooth()
        if self._split_running:
            self._reject_split()
        # Cancel an active Add session (tears down box/candidate layers +
        # releases the controller lock); otherwise sweep any stray temp layers.
        # Fully guarded so a minimal test fake without these methods is a no-op.
        try:
            if getattr(self, "_add_running", False):
                # Still running here means closeEvent's prompt was
                # never reached -- the launcher's _run_widget_cleanup path.
                # Said, not patched: the driver reports which path napari's
                # X takes (B2 fix pass L-M4).
                print("[ADD] close via cleanup path: no prompt")
                self._reject_add()
            else:
                # REVIEW / idle: reset fully (destroys candidate+overlay AND
                # clears _add_session/_add_review/_add_piece_ids) — was leaving
                # the session state set, asymmetric with the switch path (audit
                # LOW-4).  Harmless today (widget is destroyed) but tidy.
                self._reset_add_state()
        except Exception as exc:
            print(f"[ADD] _close_cleanup add teardown raised: {exc!r}")
        if self._interp_snapshot is not None:
            self._reject_interpolation()
        self._interp_refine_mode = False
        # Batch refine — the dock-close path bypasses closeEvent (and thus
        # _has_uncommitted_work -> _discard_batch_refine), so we MUST cancel
        # + join the worker QThread here or it keeps running into a
        # torn-down widget (orphaned thread / access violation).
        if getattr(self, "_batch_running", False) or getattr(
            self, "_batch_staged", False
        ):
            try:
                self._discard_batch_refine()
            except Exception as e:
                print(f"[BATCH] _close_cleanup discard raised: {e!r}")
        # Idle-but-open dialog (no staged/running work) — tear it down so it
        # doesn't linger (mirrors the merge-dialog teardown above).
        if getattr(self, "_batch_refine_dialog", None) is not None:
            try:
                self._teardown_batch_dialog()
            except Exception:
                pass

        # Reshape R1: a guided split open at close has the target label
        # PUNCHED OUT of the shared volume and may have a worker in
        # flight.  Tear it down here for the same reason batch refine is
        # torn down here: the dock-close path bypasses closeEvent, so an
        # orphaned worker would run on into a destroyed widget and the
        # user would lose the cell.  Idempotent and never raises.
        # Two try blocks, not one: the teardown is the step that PUTS THE
        # CELL BACK, so a raise while tidying the marker worker must not
        # skip it.
        try:
            self._abandon_seed_find_worker()
            self._remove_temp_layer("markers")
            # Brick U3.  `_teardown_reshape_for_switch` below sweeps
            # every temp layer anyway, so this is belt and braces on the
            # one path where a raise in the teardown would otherwise
            # leave an Image layer behind on a dying widget.
            self._drop_walls_picture()
        except Exception as e:
            print(f"[RESHAPE-FIND] _close_cleanup marker teardown raised: {e!r}")
        try:
            self._teardown_reshape_for_switch()
            self._teardown_refine_for_switch()
        except Exception as e:
            print(f"[RESHAPE] _close_cleanup teardown raised: {e!r}")

        # --- Step 2: Lightweight isolation state cleanup ---
        # Release isolation authority on close without touching subviewers
        # (they're being torn down).  Do NOT call _on_isolation_changed(set())
        # which tries to propagate colormap/show_selected_label to subviewers
        # that may already be partially destroyed.
        if self._isolation_active:
            self._isolation_active = False
            self._isolation_label_set = None
            self._pre_isolation_colormap = None
            self._pre_isolation_data = None
            auth = getattr(self, "_colormap_authority", None)
            if auth is not None:
                from napari_epicycle.fusion_reconstruction.colormap_authority import ColormapOwner
                auth.release(ColormapOwner.ANNOTATOR_ISOLATION)
        # Phase 2: Remove proxy display layer on close
        if getattr(self, "_isolation_proxy_active", False):
            self._remove_isolation_proxy()
        # Belt-and-suspenders: always clear suppress flag on close, even if
        # _remove_isolation_proxy was already called or proxy wasn't active.
        sync = getattr(self, "sync_manager", None)
        if sync is not None:
            sync._suppress_show_selected_sync = False
        # UNCONDITIONAL: Reset show_selected_label on the main labels layer.
        # Controller-initiated isolation (lasso) sets CONTROLLER_ISOLATION
        # without setting _isolation_active, so gating on _isolation_active
        # would miss that case and leave the viewer in single-label mode.
        try:
            main_layer = self.viewer.layers[self._active_labels_name]
            if hasattr(main_layer, "show_selected_label"):
                main_layer.show_selected_label = False
        except (KeyError, Exception):
            pass

        # --- Step 3: Stop debounce timer ---
        timer = getattr(self, "_alive_check_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
        init_timer = getattr(self, "_init_auto_load_timer", None)
        if init_timer is not None:
            try:
                init_timer.stop()
            except Exception:
                pass
        deferred_timer = getattr(self, "_deferred_auto_load_timer", None)
        if deferred_timer is not None:
            try:
                deferred_timer.stop()
            except Exception:
                pass

        # --- Step 4: Clear navigator widget callbacks + UI state + reference ---
        # The navigator dock may survive in Qt memory after close.  Clear
        # its callbacks so no stale callback fires into this (dead) annotator
        # between close and relaunch.  The launcher rebinds on reuse.
        nav = self._navigator_widget
        if nav is not None:
            _safe_curation_debug(self,
                f"close cleanup navigator found type={type(nav).__name__} id={id(nav)}"
            )
            # Reset UI state (selection + isolate checkbox) silently so the
            # dock doesn't carry stale visual state when reused on relaunch.
            if hasattr(nav, "reset_selection_ui"):
                try:
                    nav.reset_selection_ui()
                except Exception:
                    pass
            if hasattr(nav, "set_callbacks"):
                try:
                    nav.set_callbacks()
                except Exception:
                    pass
        else:
            _safe_curation_debug(self, "close cleanup navigator widget was None")
        # Destroy the navigator dock.  Use stored _navigator_dock ref
        # (set by launcher) for reliable lookup — the old parent-traversal
        # approach was fragile and often failed to find the QDockWidget.
        # Full destruction (not just hide/close) ensures the navigator
        # doesn't persist as a visible dock after the annotator closes.
        nav_dock = getattr(self, "_navigator_dock", None)
        if nav_dock is None and nav is not None:
            # Fallback: parent traversal + _resolve_existing_dock.
            try:
                from qtpy.QtWidgets import QDockWidget
                p = nav.parent()
                while p is not None and not isinstance(p, QDockWidget):
                    p = p.parent()
                nav_dock = p
            except Exception:
                nav_dock = None
        if nav_dock is None:
            try:
                from napari_epicycle.annotator.launcher import _resolve_existing_dock
                nav_dock = _resolve_existing_dock(
                    self.viewer, "EpiCycle Label Navigator"
                )
            except Exception:
                pass
        if nav_dock is not None:
            try:
                from napari_epicycle.annotator.launcher import _destroy_dock
                _destroy_dock(self.viewer, nav_dock)
                _safe_curation_debug(self,
                    f"navigator dock destroyed type={type(nav_dock).__name__}"
                )
            except Exception:
                _safe_curation_debug(self, "navigator dock destroy failed")
        else:
            _safe_curation_debug(self, "navigator dock teardown: dock not found")
        self._navigator_widget = None
        self._navigator_dock = None

        # --- Step 4b: Destroy remaining satellite docks ---
        # Advanced Features and Multiviewer Settings docks can survive
        # annotator close and accumulate across relaunch cycles.
        try:
            from napari_epicycle.annotator.launcher import _destroy_satellite_docks
            _destroy_satellite_docks(self.viewer)
        except Exception:
            pass

        # --- Step 5: Disconnect viewer-level event handlers ---
        self._disconnect_viewer_events()

        # --- Step 6: Disconnect inserted handler ---
        try:
            self.viewer.layers.events.inserted.disconnect(self._on_layer_inserted)
        except (ValueError, AttributeError):
            pass
        try:
            self.viewer.layers.events.removed.disconnect(self._on_layer_removed)
        except (ValueError, AttributeError):
            pass
        # ... and the tool-tab hook, which _open_advanced_features connects
        # on every click (the launcher reuses an open dock).
        try:
            _adv_tt = self._get_advanced_features_widget()
            if _adv_tt is not None and hasattr(_adv_tt, "_tool_tabs"):
                _adv_tt._tool_tabs.currentChanged.disconnect(
                    self._on_tool_tab_changed
                )
        except (TypeError, RuntimeError, AttributeError):
            pass

        # --- Step 7: Disconnect layer-level handlers ---
        self._disconnect_label_handlers()

        # --- Step 8: Fire isolation-cleared callback UNCONDITIONALLY ---
        # Must fire AFTER all event handlers are disconnected (steps 5-7)
        # so the controller's colormap update doesn't cascade into
        # half-destroyed subviewers via sync.py.
        #
        # Fires unconditionally because lasso selection on the QC plot sets
        # CONTROLLER_ISOLATION on the controller without ever setting
        # _isolation_active on the annotator — so the controller must
        # always be notified to clean up its own state.
        cb = self._on_isolation_cleared_cb
        if cb is not None:
            try:
                cb(reason="close")
            except TypeError:
                cb()
            except Exception:
                pass

        # --- Step 9: Restore arrow keys ---
        self._restore_main_viewer_arrow_keys()

        # --- Post-close: Null out callback refs and clear caches ---
        self._on_isolation_cleared_cb = None
        self._on_mark_tier_cb = None
        self._on_delete_label_cb = None
        self._on_discard_garbage_cb = None
        self._on_merge_cb = None
        self._on_navigator_label_selected_cb = None
        self._on_interpolation_accepted_cb = None
        self._on_label_split_cb = None
        self._on_label_smoothed_cb = None
        self._on_recalculate_cb = None
        self._on_new_label_painted_cb = None
        self._on_batch_refined_cb = None
        self._batch_refine_lock_cb = None
        self._add_mode_lock_cb = None
        self._on_add_committed_cb = None
        self._on_add_undo_session_cb = None
        self._save_batch_recipe_cb = None
        self._get_saved_batch_recipes_cb = None
        self._save_refine_recipe_cb = None
        self._get_saved_refine_recipes_cb = None
        self._delete_refine_recipe_cb = None
        self._get_plot_selection_cb = None
        self._get_label_id_floor_cb = None
        # The Refine-tab recipe a plain Batch open re-selects (session-only)
        # and the dialog-life poll timer.
        self._batch_last_chain_name = None
        _poll = getattr(self, "_batch_poll_timer", None)
        if _poll is not None:
            try:
                _poll.stop()
            except Exception:
                pass
        # Reset the per-session Batch Refine recipe history so a relaunch starts
        # clean (the in-memory "Recent recipes" list is session-only; matches the
        # clean-slate lifecycle even though a reused instance is not the norm).
        self._batch_recipe_history = []
        # Sticky last-run recipe (2026-06-07) is also session-only — clear it on
        # annotator teardown so a relaunch starts with an empty recipe builder.
        self._batch_last_recipe = None
        # Paint-isolation state (companion, 2026-05-16) — reset
        # full state on annotator close so a relaunch starts clean.
        self._paint_isolation_enabled = False
        self._paint_isolation_active = False
        self._paint_isolation_label_id = None

        _dt = _time.perf_counter() - _t0
        if _dt > 0.03:
            _safe_curation_debug(self, f"close cleanup done dt={_dt:.3f}s")

    # ------------------------------------------------------------------
    # Operation guard system — generic uncommitted-work detection
    # ------------------------------------------------------------------

    def _register_operation_slots(self) -> None:
        """Register all operations that can have uncommitted work.

        Adding a future operation = append one OperationSlot here.
        No guard method modifications needed.
        """
        self._operation_slots = [
            OperationSlot(
                name="smooth",
                is_active=lambda: self._smooth_running,
                description=lambda: (
                    f"Smooth preview active for label {self._smooth_label_id}."
                ),
                reject=self._reject_smooth,
                priority=30,
            ),
            OperationSlot(
                name="split",
                is_active=lambda: self._split_running,
                description=lambda: (
                    f"Split fragment refine active for label {self._split_label_id}."
                    if self._split_fragment_refine
                    else f"Split in progress for label {self._split_label_id}."
                ),
                reject=self._reject_split,
                priority=20,
            ),
            OperationSlot(
                name="interpolation",
                is_active=lambda: (
                    self._interp_snapshot is not None
                    or self._interp_refine_mode
                ),
                description=lambda: (
                    f"Interpolation preview active for label {self._interp_label_id}."
                ),
                reject=self._reject_interpolation,
                priority=10,
            ),
            OperationSlot(
                name="batch_refine",
                is_active=lambda: (
                    getattr(self, "_batch_running", False)
                    or getattr(self, "_batch_staged", False)
                ),
                description=lambda: (
                    "Batch refine preview staged "
                    f"({len(getattr(self, '_batch_touched_ids', []))} labels)."
                    if getattr(self, "_batch_staged", False)
                    else "Batch refine is running."
                ),
                reject=self._discard_batch_refine,
                priority=40,
            ),
            # Add: a preview running, candidates
            # on screen or click markers placed count as work.  REVIEW alone
            # is not (everything in it is already written).
            OperationSlot(
                name="add",
                is_active=self._add_has_work,
                description=self._add_close_description,
                reject=self._reject_add,
                priority=25,
            ),
        ]

    def _add_has_work(self) -> bool:
        return bool(
            getattr(self, "_add_running", False)
            and (getattr(self, "_add_busy", False)
                 or getattr(self, "_add_preview_result", None) is not None
                 or bool(getattr(self, "_add_click_markers", None))))

    def _add_close_description(self) -> str:
        """The close prompt's sentence, by state: busy, candidates, clicks."""
        from packages.epicycle_annotator import add_preview as AP

        if getattr(self, "_add_busy", False):
            return AP.ADD_CLOSE_BUSY
        res = getattr(self, "_add_preview_res", None)
        if getattr(self, "_add_preview_result", None) is not None:
            n = len(getattr(res, "ids", []) or []) if res is not None else 0
            return AP.count_phrase(n, AP.ADD_CLOSE_CANDIDATES_ONE,
                                   AP.ADD_CLOSE_CANDIDATES)
        n = len(getattr(self, "_add_click_markers", []) or [])
        return AP.count_phrase(n, AP.ADD_CLOSE_CLICKS_ONE, AP.ADD_CLOSE_CLICKS)

    def _has_uncommitted_work(self, *, include_edited_slices=True):
        """Check whether any operation has uncommitted work.

        Returns ``(has_work: bool, reason: str)``.  Iterates operation
        slots in priority order (highest first).
        """
        for slot in sorted(self._operation_slots, key=lambda s: -s.priority):
            if slot.is_active():
                return True, slot.description()
        if include_edited_slices:
            labels_name = self._active_labels_name
            if labels_name:
                edited = self.sync_manager.get_edited_slices(labels_name)
                label_id = self._get_selected_label()
                total_slices = 0
                for orient_dict in edited.values():
                    for lid, slices in orient_dict.items():
                        if int(lid) == label_id and slices:
                            total_slices += len(slices)
                if total_slices > 0:
                    return True, (
                        f"Unsaved edits on label {label_id} "
                        f"({total_slices} slices)."
                    )
        return False, ""

    def _discard_uncommitted_work(self) -> None:
        """Reject all active operations and clear edited slices."""
        for slot in self._operation_slots:
            if slot.is_active():
                slot.reject()
        labels_name = self._active_labels_name
        if labels_name:
            label_id = self._get_selected_label()
            self.sync_manager.clear_edited_slices_for_label(
                labels_name, label_id
            )

    def _confirm_discard_dialog(self, reason: str) -> bool:
        """Show a Discard / Go Back dialog. Returns True if user chose Discard."""
        try:
            box = QMessageBox(self)
            box.setWindowTitle("Uncommitted Work")
            box.setText(reason)
            box.setInformativeText("Discard changes and continue?")
            discard_btn = box.addButton("Discard", QMessageBox.DestructiveRole)
            box.addButton("Go Back", QMessageBox.RejectRole)
            box.setDefaultButton(discard_btn)
            box.exec_()
            return box.clickedButton() == discard_btn
        except Exception:
            return False

    def _next_label_id_after_accept(self) -> int:
        """Compute the next label ID after accepting an operation.

        Tries navigator items first (fast), falls back to volume scan.
        """
        # Try navigator items
        nav = getattr(self, "_navigator_widget", None)
        if nav is not None:
            try:
                items = getattr(nav, "_items", [])
                if items:
                    return max(it["label_id"] for it in items) + 1
            except (ValueError, KeyError, TypeError):
                pass
        # Fallback: scan shared volume
        labels_name = self._active_labels_name
        shared_data = self.sync_manager.shared_labels.get(labels_name)
        if shared_data is not None and shared_data.size > 0:
            return int(np.max(shared_data)) + 1
        return 1

    def _on_subviewer_pre_sync(self) -> None:
        """Called BEFORE sync_to_main_view on each stroke release.

        Snapshots shared_data for the current label on the FIRST stroke,
        so Clear Edits can restore the label to its pre-drawing state.
        Must run before sync modifies shared_data.
        """
        if self._split_running:
            return
        labels_name = self._active_labels_name
        if not labels_name:
            return
        shared_data = self.sync_manager.shared_labels.get(labels_name)
        if shared_data is None:
            return
        current_label = self._get_selected_label()
        if current_label <= 0:
            return
        # Only snapshot on first stroke for this label.
        if self._draw_label_id != current_label:
            self._draw_roi, self._draw_snapshot = self._snapshot_label_roi(
                shared_data, current_label
            )
            self._draw_label_id = current_label
            self._clear_edits_btn.setEnabled(True)
            self._interp_status_label.setText(
                f"Drawing on label {current_label}. "
                f"Preview to interpolate, or Clear Edits to undo."
            )
            # New/Delete Label buttons must disable while an
            # uncommitted paint stroke is in flight — see
            # `_label_action_buttons_active`.
            self._refresh_merge_button_state()
            self._refresh_label_action_buttons_state()
    def _on_subviewer_draw_complete(self) -> None:
        """Called AFTER sync_to_main_view on each stroke release.

        Updates the isolation proxy so the user sees drawn corrections
        in the main 3D viewer.
        """
        if self._split_running:
            return
        if not self._isolation_proxy_active:
            return
        proxy = self._isolation_proxy_layer
        label_id = self._isolation_proxy_label_id
        if proxy is None or label_id is None:
            return
        labels_name = self._active_labels_name
        main_layer = self._layer_by_name(self.viewer.layers, labels_name)
        if main_layer is not None:
            self._update_isolation_proxy(label_id, main_layer, force=True)

    def _clear_draw_edits(self) -> None:
        """Undo all drawn corrections and restore the label's original shape.

        Restores the label's ROI from the snapshot taken before the first
        draw stroke, clears edited_slices tracking, and refreshes views.
        """
        # Block during active operations — those have their own undo.
        if self._interp_snapshot is not None or self._interp_refine_mode:
            self._interp_status_label.setText(
                "Use Reject to undo interpolation preview."
            )
            return
        if self._split_running:
            self._interp_status_label.setText(
                "Use Discard to exit split mode."
            )
            return
        if self._smooth_running:
            self._interp_status_label.setText(
                "Use Discard to undo the refine preview."
            )
            return

        if self._draw_snapshot is None or self._draw_roi is None:
            self._interp_status_label.setText("No edits to clear.")
            return

        labels_name = self._active_labels_name
        shared_data = self.sync_manager.shared_labels.get(labels_name)
        if shared_data is None:
            return

        label_id = self._draw_label_id
        roi = self._draw_roi

        # 1. Zero out ALL voxels of this label first — catches strokes
        #    drawn outside the snapshot ROI (which only covers the label's
        #    original bounding box + margin).  Any voxels outside the ROI
        #    MUST be new draws, since the ROI fully enclosed the label's
        #    original extent.
        if label_id is not None:
            shared_data[shared_data == label_id] = 0

        # 2. Restore the ROI from snapshot — puts back the label's
        #    original geometry plus all other labels within the ROI.
        shared_data[roi] = self._draw_snapshot

        # Clear edited_slices tracking for this label.
        if labels_name and label_id is not None:
            self.sync_manager.clear_edited_slices_for_label(
                labels_name, label_id
            )

        # Refresh all views.  Pass roi=None when we zeroed outside the
        # ROI so the full volume is synced to the main layer.
        self._sync_all_views_after_operation(
            roi=None, label_id=label_id
        )

        # Clear draw snapshot state.
        self._draw_snapshot = None
        self._draw_roi = None
        self._draw_label_id = None
        self._clear_edits_btn.setEnabled(False)
        self._interp_status_label.setText(
            f"Edits cleared for label {label_id}. Original shape restored."
        )
        # Re-enable New/Delete Label buttons (snapshot gate just lifted).
        self._refresh_merge_button_state()
        self._refresh_label_action_buttons_state()
    def _clear_draw_snapshot(self) -> None:
        """Silently discard the draw snapshot without restoring.

        Called when the snapshot is no longer needed (e.g., after Preview
        takes its own snapshot, after Accept, or after navigating away).
        """
        self._draw_snapshot = None
        self._draw_roi = None
        self._draw_label_id = None
        self._clear_edits_btn.setEnabled(False)
        # Re-enable New/Delete Label buttons if no other gate is up.
        self._refresh_merge_button_state()
        self._refresh_label_action_buttons_state()
    def _restore_ready_to_draw_state(self) -> None:
        """Restore drawing readiness after any operation completes.

        Called after every accept/reject to ensure the user can immediately
        draw on subviewers.  Cheap — only sets properties, no data ops.

        1. Active layer → main labels (not EDT/Refine/proxy)
        2. Main layer → editable (mode untouched in 3D to avoid re-render)
        3. Subviewers → paint mode + editable
        4. Edit constraints → enforce target mode (all editable in 3D curation)
        """
        labels_name = self._active_labels_name
        main_layer = self._layer_by_name(self.viewer.layers, labels_name)
        if not isinstance(main_layer, Labels):
            return

        # 1. Restore active layer to main labels.
        try:
            self.viewer.layers.selection.active = main_layer
        except Exception:
            pass

        # 2. Lock main layer — drawing only happens on subviewers.
        #    editable=False blocks painting even if napari keybindings
        #    switch mode; pan_zoom prevents drag-painting entirely.
        main_layer.editable = False
        main_layer.mode = "pan_zoom"

        # 3. Set subviewer label layers to paint mode.
        mode = str(getattr(self, "_default_draw_mode", "paint")).strip().lower() or "paint"
        for layer in self._all_subviewer_label_layers():
            try:
                layer.mode = mode
                layer.editable = True
            except Exception:
                pass

        # 4. Apply edit constraints (may lock views based on target mode).
        self._apply_edit_constraints()

    def set_load_locked(self, locked: bool) -> None:
        """Lock/unlock the annotator while the controller loads a new patch.

        Called from ``FusionReconstructionController._set_curation_locked``
        across the async patch-load boundary (2026-06-05).  While ``locked``
        the user may still pan/zoom/rotate the 3D viewer + subviewers, but
        drawing + navigator label actions (New / Delete / Merge) are disabled
        so no edit can land on the about-to-be-swapped (or half-loaded) volume.

        On unlock (when no refinement op is staged) we re-enable the navigator
        action buttons AND restore the draw state via
        ``_restore_ready_to_draw_state``.  This restore is correct in BOTH
        contexts:

          * **Curate ops / predict / reconstruct** (2026-06-06 audit FIX-2):
            these lock the annotator via ``_refresh_step3_lock`` WITHOUT any
            subsequent reload, so nothing else would un-freeze drawing.  Here
            ``_active_labels_name`` is intact, so the call immediately restores
            drawability on the subviewers (otherwise: paint-freeze after the op).

          * **Patch load**: the controller's ``_post_switch_annotator_sync``
            sets ``_active_labels_name = ""`` BEFORE this unlock runs, so
            ``_restore_ready_to_draw_state`` early-returns (its first guard:
            empty name / missing main Labels layer → no-op).  The deferred
            ``_load_selected_layers`` then restores draw state after the swap.
            So the patch-load behavior is UNCHANGED.

        Re-enable / restore happen ONLY when no refinement op is staged (reuse
        the canonical ``_is_op_active_for_navigator`` predicate) so an in-flight
        op's gate isn't clobbered by the unlock.

        Everything is getattr/try-guarded so a closed annotator / missing
        navigator never raises during the async lifecycle.
        """
        # Track the load-locked state so other entry points (e.g. opening the
        # Batch Refine dialog) can refuse to start a NEW op while the controller
        # holds the unified Step-3 lock.  set_label_action_buttons_enabled only
        # gates New/Delete/Merge, not the Batch Refine button, so this flag is
        # the gate _open_batch_refine_dialog consults.
        self._load_locked = bool(locked)
        try:
            if locked:
                # Lock drawing: subviewers → editable=False + pan_zoom.
                try:
                    self._lock_subviewers_for_preview()
                except Exception:
                    pass
                nav = getattr(self, "_navigator_widget", None)
                if nav is not None:
                    try:
                        nav.set_label_action_buttons_enabled(False)
                    except Exception:
                        pass
                    try:
                        nav.set_merge_enabled(False)
                    except Exception:
                        pass
                print("[LOAD-LOCK] annotator locked")
            else:
                # Re-enable navigator action buttons — but only if no
                # refinement op is staged (reuse the canonical predicate so
                # we don't clobber an op's gate).
                op_active = False
                try:
                    op_active = bool(self._is_op_active_for_navigator())
                except Exception:
                    op_active = False
                nav = getattr(self, "_navigator_widget", None)
                if nav is not None and not op_active:
                    try:
                        nav.set_label_action_buttons_enabled(True)
                    except Exception:
                        pass
                    try:
                        nav.set_merge_enabled(True)
                    except Exception:
                        pass
                if not op_active:
                    # Restore draw state (2026-06-06 audit FIX-2).  Curate ops /
                    # predict / reconstruct lock via _refresh_step3_lock with NO
                    # reload, so without this the subviewers stay editable=False
                    # → drawing frozen.  During a PATCH LOAD this is a safe
                    # no-op: _post_switch_annotator_sync has already cleared
                    # _active_labels_name, so _restore_ready_to_draw_state
                    # early-returns and the deferred reload owns draw restore.
                    try:
                        self._restore_ready_to_draw_state()
                    except Exception:
                        pass
                print(
                    f"[LOAD-LOCK] annotator unlocked (op_active={op_active})"
                )
        except Exception as exc:
            print(f"[LOAD-LOCK] set_load_locked({locked}) raised: {exc}")

    def _lock_subviewers_for_preview(self) -> None:
        """Lock subviewer label layers during operation previews.

        Prevents drawing while a preview (smooth/interpolation) is active.
        Rejecting an operation restores its snapshot, which would silently
        undo any strokes drawn during the preview — confusing and data-lossy.

        Sets both ``mode='pan_zoom'`` AND ``editable=False`` — the latter
        is needed because napari keybindings (e.g. pressing '2') can switch
        mode back to paint, but ``editable=False`` blocks painting regardless.

        Subviewers are unlocked by ``_restore_ready_to_draw_state()`` on
        accept/reject, or explicitly by ``_enter_interp_refine_mode()``.
        """
        for layer in self._all_subviewer_label_layers():
            try:
                layer.editable = False
                layer.mode = "pan_zoom"
            except Exception:
                pass

    def _is_auto_advance_enabled(self) -> bool:
        """Check whether auto-advance is enabled.

        Uses the Navigator's auto-advance checkbox as the single source
        of truth.  Falls back to the Advanced Features checkbox if the
        Navigator is not open, for backwards compatibility.
        """
        nav = getattr(self, "_navigator_widget", None)
        if nav is not None and hasattr(nav, "_auto_advance_cb"):
            return nav._auto_advance_cb.isChecked()
        adv = self._get_advanced_features_widget()
        if adv is not None and hasattr(adv, "split_auto_advance_cb"):
            return adv.split_auto_advance_cb.isChecked()
        return True

    # ------------------------------------------------------------------
    # Unified view sync after operations
    # ------------------------------------------------------------------

    def _sync_all_views_after_operation(
        self, roi=None, label_id=None, *, skip_3d_refresh=False
    ) -> None:
        """Sync shared volume changes to main layer, subviewers, and proxy.

        Uses ``refresh()`` only — never the ``data = data`` property setter
        which triggers full ``iso_categorical`` recalculation in 3D mode.

        Parameters
        ----------
        roi : tuple of slices, optional
            If provided, copies only this region from shared_data to main_layer.
        label_id : int, optional
            If provided and isolation proxy is active, rebuilds the proxy.
        skip_3d_refresh : bool
            If True, skip ``main_layer.refresh()`` and proxy rebuild.  Used
            when the caller knows that the main layer is hidden (proxy active)
            AND auto-advance will immediately set up a new proxy — the
            refresh would be wasted work since the layer stays hidden.
        """
        labels_name = self._active_labels_name
        shared_data = self.sync_manager.shared_labels.get(labels_name)
        if shared_data is None:
            return
        main_layer = self._layer_by_name(self.viewer.layers, labels_name)
        if isinstance(main_layer, Labels):
            if main_layer.data is not shared_data:
                if roi is not None:
                    main_layer.data[roi] = shared_data[roi]
                else:
                    main_layer.data[:] = shared_data
            if not skip_3d_refresh:
                # Only refresh the main layer if it's visible — when the
                # proxy handles 3D rendering, refreshing the hidden main
                # layer is wasted work.
                if main_layer.visible:
                    main_layer.refresh()
                # Always rebuild the proxy if active — it IS visible and
                # must reflect data changes (smooth, interpolation, etc.).
                if self._isolation_proxy_active and label_id is not None:
                    roi_hint = getattr(self, "_fragment_refine_roi", None)
                    self._update_isolation_proxy(
                        label_id, main_layer, force=True, roi_hint=roi_hint
                    )
        self._refresh_label_data_in_subviewers()

        # Re-propagate isolation visual after data refresh.
        # The data setter on subviewer layers resets napari's rendering
        # pipeline, which can clear the isolation visual.  Re-assert it
        # so isolation is preserved after operations.
        if getattr(self, "_isolation_active", False) or getattr(
            self, "_isolation_proxy_active", False
        ):
            sel_label = label_id or self._get_selected_label()
            if sel_label and sel_label > 0:
                if self._pre_isolation_colormap is not None:
                    # Multi-label isolation uses DirectLabelColormap —
                    # re-propagate the colormap instead of
                    # show_selected_label which would hide all but one.
                    if isinstance(main_layer, Labels):
                        self._propagate_colormap_to_subviewers(main_layer.colormap)
                else:
                    self._propagate_show_selected_label(True, label_id=sel_label)

    def closeEvent(self, event) -> None:
        """Restore main viewer key bindings on close.

        If an operation has uncommitted work, show a discard dialog.
        """
        has_work, reason = self._has_uncommitted_work(include_edited_slices=False)
        if has_work:
            if not self._confirm_discard_dialog(reason):
                event.ignore()
                return
            self._discard_uncommitted_work()
        self._close_cleanup()
        # Close the 3 subviewer QtViewer instances to prevent orphaned
        # magicgui windows accumulating on each relaunch.
        self._close_subviewers()
        try:
            from napari_epicycle.annotator.launcher import (
                _schedule_close_orphan_sweep_retries,
            )

            _schedule_close_orphan_sweep_retries()
        except Exception:
            pass
        super().closeEvent(event)

    def _block_main_viewer_arrow_keys(self) -> None:
        """Override all arrow key bindings on the main napari viewer.

        Prevents accidental dims scrolling on the main viewer while the
        3D curation annotator is active.  Arrow key functionality is restored
        when the annotator is closed via ``_restore_main_viewer_arrow_keys``.
        """
        def _noop(viewer):
            pass

        try:
            for key in ("Up", "Down", "Left", "Right"):
                self.viewer.bind_key(key, _noop, overwrite=True)
        except Exception:
            pass

    def _restore_main_viewer_arrow_keys(self) -> None:
        """Unbind arrow key overrides on the main viewer.

        Called during teardown to restore napari's default arrow key behavior.
        """
        try:
            for key in ("Up", "Down", "Left", "Right"):
                self.viewer.bind_key(key, None, overwrite=True)
        except Exception:
            pass

    def _switch_patch_context_from_combo(self):
        """Block patch switching during an active split, smooth, or Add."""
        if self._smooth_running:
            self._reject_smooth()
        if self._split_running:
            self._update_split_ui(
                status_text="Complete or cancel split before switching patches."
            )
            return
        if getattr(self, "_add_running", False):
            # An in-flight Add (box/preview, uncommitted) blocks the switch —
            # the user must Accept or Cancel first (mirrors split).
            self._push_add_ui(
                status_text="Finish or cancel Add before switching patches.",
                status_kind="warn",
            )
            return
        super()._switch_patch_context_from_combo()

    def _load_selected_layers(self) -> None:
        """Override to set smart step size, install key filters, connect pinning."""
        # Diagnostic — FM5 fix (2026-05-16): surface every isolation
        # state value the patch-switch will reset, so a manual tester
        # can see exactly what carried over from the prior patch.
        prev_proxy_lid = getattr(self, "_isolation_proxy_label_id", None)
        prev_label_set = getattr(self, "_isolation_label_set", None)
        print(
            f"[PATCH-SWITCH-ISO] _load_selected_layers ENTER "
            f"iso_active={getattr(self, '_isolation_active', False)} "
            f"proxy_active={getattr(self, '_isolation_proxy_active', False)} "
            f"proxy_lid={prev_proxy_lid} "
            f"label_set={sorted(prev_label_set) if prev_label_set else None} "
            f"paint_iso_active={getattr(self, '_paint_isolation_active', False)} "
            f"paint_iso_label={getattr(self, '_paint_isolation_label_id', None)}"
        )

        # Force-reject any active smooth/split before tearing down old layers.
        if self._smooth_running:
            self._reject_smooth()
        # ...and abandon a RefineWorker that has not produced anything yet,
        # which `_reject_smooth` would not have seen.
        self._teardown_refine_for_switch()
        if self._split_running:
            self._reject_split()
        if getattr(self, "_add_running", False):
            self._reject_add()
        elif getattr(self, "_add_review", False):
            # A committed REVIEW must NOT carry its stale undo/session state
            # (OLD-patch ROI coords + crop) into the new patch — "Undo all"
            # would then write the old crop into the new volume (audit HIGH-2).
            # The committed labels themselves stay; only the REVIEW state is
            # cleared -- through the one switch teardown, whose S1 comes from
            # the 4.9a table (B2 fix pass L-L2: this branch used to push the
            # old page's hard-coded sentence and keep the touched set).
            self._teardown_add_for_switch()
        else:
            # Idle stray-layer sweep (a partial teardown could leave temp layers).
            self._remove_add_candidate()
            self._remove_add_box_shapes()
        self._interp_refine_mode = False
        self._refresh_merge_button_state()
        self._refresh_label_action_buttons_state()
        # WP-3 / Bug J: Reset colormap authority and isolation state on
        # patch switch so stale _pre_isolation_colormap doesn't corrupt
        # the new patch's colormap.
        if self._colormap_authority is not None:
            self._colormap_authority.reset()
        self._isolation_active = False
        self._pre_isolation_colormap = None
        self._isolation_label_set = None
        # FM5 fix (2026-05-16): clear _pre_isolation_data too — it
        # was already cleared in _close_cleanup but missing here,
        # which could leave stale ROI data referenced if a future
        # cleanup path checks both fields.
        self._pre_isolation_data = None
        # Paint-isolation: reset active/label_id state on patch switch
        # but PRESERVE _paint_isolation_enabled (the user's toggle
        # choice survives patch switches).  Next event on the new
        # patch's layer will re-fire isolation if a real label is
        # selected.
        self._paint_isolation_active = False
        self._paint_isolation_label_id = None
        # P0 #6 (2026-05-10 audit) — defensive clear of refinement
        # snapshots on patch switch.  Normal accept/reject paths clear
        # their own snapshots, but edge cases (interp preview taken
        # without accept, force-exit during fragment-refine, etc.) can
        # leave a stale snapshot pinned to the PREVIOUS patch's volume
        # coordinates.  Without this, an Accept/Reject on the new
        # patch would write the old patch's snapshot data into the
        # new patch's shared_data → silent corruption.  Costs nothing;
        # if values are already None, the assignments are no-ops.
        self._interp_snapshot = None
        self._interp_roi = None
        self._split_snapshot = None
        self._split_roi = None
        self._smooth_snapshot = None
        self._draw_snapshot = None
        # Phase 2: Remove proxy on patch switch
        if getattr(self, "_isolation_proxy_active", False):
            print(
                f"[PATCH-SWITCH-ISO] removing single-label proxy "
                f"(was isolating label {prev_proxy_lid})"
            )
            self._remove_isolation_proxy()
        # Belt-and-suspenders: clear suppress flag on patch switch
        sync = getattr(self, "sync_manager", None)
        if sync is not None:
            sync._suppress_show_selected_sync = False

        super()._load_selected_layers()
        self._apply_smart_step_size()
        self._install_arrow_key_filters()
        self._sync_show_selected_label()
        self._connect_label_pinning()

        print(
            "[PATCH-SWITCH-ISO] _load_selected_layers DONE — isolation "
            "state cleared; navigator isolate-cb intentionally "
            "preserved (user toggle survives patch switch)"
        )

    # ---------------------------------------------------------------
    # Show-selected-label sync to subviewers
    # ---------------------------------------------------------------

    def _sync_show_selected_label(self) -> None:
        """Sync the main viewer's show_selected_label to all subviewer label layers.

        Skipped when the 3D isolation proxy is active — in proxy mode the
        main layer deliberately has ``show_selected_label=False`` (to avoid
        expensive iso_categorical re-render) while subviewers should keep
        ``show_selected_label=True`` for 2D isolation display.
        """
        # Guard: don't overwrite subviewer state when proxy manages isolation.
        # Check both _isolation_proxy_active AND the suppress flag on sync
        # manager — the flag is set before the proxy is fully created, so
        # it catches the window between flag-set and proxy-active.
        if getattr(self, "_isolation_proxy_active", False):
            return
        sync = getattr(self, "sync_manager", None)
        if sync is not None and getattr(sync, "_suppress_show_selected_sync", False):
            return

        labels_name = self._active_labels_name
        if not labels_name:
            return
        main_layer = self._layer_by_name(self.viewer.layers, labels_name)
        if not isinstance(main_layer, Labels):
            return

        show_sel = bool(getattr(main_layer, "show_selected_label", False))

        # Connect event if not already connected.
        if not getattr(main_layer, "_epicycle_show_sel_connected", False):
            main_layer._epicycle_show_sel_connected = True
            # Store reference for clean disconnection in _disconnect_label_handlers.
            self._show_sel_cb = lambda _e: self._sync_show_selected_label()
            main_layer.events.show_selected_label.connect(self._show_sel_cb)

        for layer in self._all_subviewer_label_layers():
            try:
                layer.show_selected_label = show_sel
            except Exception:
                pass

    # ---------------------------------------------------------------
    # Label pinning — keep selected_label at max+1
    # ---------------------------------------------------------------

    def _connect_label_pinning(self) -> None:
        """Connect to main + subviewer label layers for auto-pin and fill-delete detection.

        Two separate events are used:

        * ``set_data`` — fires on colormap / show_selected_label changes
          (used for label-pinning only; does NOT fire on fill/paint).
        * ``labels_update`` — fires on actual fill/paint operations
          (used for fill-delete detection).
        """
        lbl_name = self._active_labels_name
        if not lbl_name:
            return
        layer = self._layer_by_name(self.viewer.layers, lbl_name)
        if not isinstance(layer, Labels):
            return
        if not getattr(layer, "_epicycle_pin_connected", False):
            layer._epicycle_pin_connected = True
            layer.events.set_data.connect(self._on_labels_data_changed)

        # labels_update fires on actual fill/paint — connect once per layer.
        if not getattr(layer, "_epicycle_labels_update_connected", False):
            layer._epicycle_labels_update_connected = True
            layer.events.labels_update.connect(self._on_labels_painted)

        # Also connect subviewer label layers so fill-with-0 in
        # subviewers triggers the debounced alive check.  The 300 ms
        # debounce ensures sync_to_main_view has completed before
        # the volume is checked.
        for sub_layer in self._all_subviewer_label_layers():
            if not getattr(sub_layer, "_epicycle_labels_update_connected", False):
                sub_layer._epicycle_labels_update_connected = True
                sub_layer.events.labels_update.connect(self._on_labels_painted)

    def _disconnect_label_handlers(self) -> None:
        """Disconnect layer-level event handlers bound to this widget.

        Called from closeEvent to prevent stale handler accumulation
        when the dock is destroyed and relaunched.

        Iterates ALL main viewer layers (not just ``_active_labels_name``)
        to catch renamed or dynamically added layers.  Guard flags are
        also reset by _disconnect_viewer_events in the base class as a
        safety belt.
        """
        from napari.layers import Labels as _Labels

        show_sel_cb = getattr(self, "_show_sel_cb", None)

        # Disconnect from ALL Labels layers in the main viewer —
        # not just the current _active_labels_name which can miss
        # layers from prior patches or renamed layers.
        try:
            for layer in list(self.viewer.layers):
                if not isinstance(layer, _Labels):
                    continue
                # show_selected_label lambda
                if show_sel_cb is not None:
                    try:
                        layer.events.show_selected_label.disconnect(show_sel_cb)
                    except (ValueError, AttributeError):
                        pass
                # Named method handlers
                for event_name, handler in [
                    ("set_data", self._on_labels_data_changed),
                    ("labels_update", self._on_labels_painted),
                ]:
                    try:
                        getattr(layer.events, event_name).disconnect(handler)
                    except (ValueError, AttributeError):
                        pass
                # Reset curation-specific guard flags
                for flag in (
                    "_epicycle_pin_connected",
                    "_epicycle_labels_update_connected",
                    "_epicycle_show_sel_connected",
                ):
                    if hasattr(layer, flag):
                        setattr(layer, flag, False)
        except Exception:
            pass

        # Disconnect subviewer label layer handlers
        try:
            for sub_layer in self._all_subviewer_label_layers():
                try:
                    sub_layer.events.labels_update.disconnect(self._on_labels_painted)
                except (ValueError, AttributeError):
                    pass
                if hasattr(sub_layer, "_epicycle_labels_update_connected"):
                    sub_layer._epicycle_labels_update_connected = False
        except Exception:
            pass

    def _on_labels_data_changed(self, _event=None) -> None:
        """After property changes (colormap, etc.), re-pin to max+1 if label is 0.

        NOTE: This fires on ``set_data`` which is triggered by property
        changes (colormap, show_selected_label) — NOT by fill/paint.
        Fill/paint detection uses ``_on_labels_painted`` instead.
        """
        try:
            current = self._get_selected_label()
            if current <= 0:
                shot = getattr(self, "_single_shot_in_session", None)
                if callable(shot):
                    shot(50, self._set_label_to_next_unused)
                else:
                    from qtpy.QtCore import QTimer
                    QTimer.singleShot(50, self._set_label_to_next_unused)
        except Exception:
            pass

    def _on_labels_painted(self, _event=None) -> None:
        """After a fill/paint stroke, schedule debounced fill-delete check.

        Connected to ``labels_update`` which fires on actual data
        modifications (fill, paint, erase) — unlike ``set_data`` which
        fires on property/colormap changes.

        Undo history is NOT parsed here because ``labels_update`` fires
        BEFORE napari populates ``_undo_history`` for fill operations.
        Instead, the source layer is stored and undo parsing is deferred
        to the 300 ms debounce callback (``_check_deleted_labels``).
        """
        source_layer = getattr(_event, "source", None)
        if source_layer is not None:
            if not hasattr(self, "_pending_undo_sources"):
                self._pending_undo_sources = set()
            self._pending_undo_sources.add(source_layer)

        self._schedule_label_alive_check()

    def _parse_undo_candidates(self) -> set:
        """Parse undo history from pending source layers (deferred).

        Called by ``_check_deleted_labels`` after the debounce timer,
        when napari has finished populating ``_undo_history``.

        Returns the set of label IDs that were overwritten with 0.
        """
        import numpy as np

        sources = getattr(self, "_pending_undo_sources", set())
        self._pending_undo_sources = set()

        candidates = set()

        # Build layer list: use stored sources, fall back to main+subviewers.
        layers_to_check = list(sources)
        if not layers_to_check:
            lbl_name = getattr(self, "_active_labels_name", None)
            if lbl_name:
                main_layer = self._layer_by_name(self.viewer.layers, lbl_name)
                if main_layer is not None:
                    layers_to_check.append(main_layer)
            for sub_layer in self._all_subviewer_label_layers():
                layers_to_check.append(sub_layer)

        for layer in layers_to_check:
            try:
                history = getattr(layer, "_undo_history", None)
                if not history or len(history) == 0:
                    continue
                last_entry = history[-1]
                for undo_item in last_entry:
                    try:
                        indices, old_vals, new_val = undo_item
                    except (ValueError, TypeError):
                        continue

                    # Handle new_val: int, np.integer, or 0-d numpy array
                    nv = None
                    if isinstance(new_val, (int, np.integer)):
                        nv = int(new_val)
                    elif isinstance(new_val, np.ndarray) and new_val.ndim == 0:
                        nv = int(new_val.item())

                    if nv is not None and nv == 0:
                        if isinstance(old_vals, np.ndarray):
                            if old_vals.ndim == 0:
                                v = int(old_vals.item())
                                if v != 0:
                                    candidates.add(v)
                            else:
                                for v in np.unique(old_vals):
                                    if int(v) != 0:
                                        candidates.add(int(v))
                        elif isinstance(old_vals, (int, np.integer)):
                            v = int(old_vals)
                            if v != 0:
                                candidates.add(v)
                if candidates:
                    break
            except Exception:
                pass

        return candidates

    def _schedule_label_alive_check(self) -> None:
        """Restart a 300 ms debounce timer for label-alive verification."""
        from qtpy.QtCore import QTimer

        if self._alive_check_timer is None:
            self._alive_check_timer = QTimer()
            self._alive_check_timer.setSingleShot(True)
            self._alive_check_timer.setInterval(300)
            self._alive_check_timer.timeout.connect(
                self._check_deleted_labels
            )
        # Restart — only the last event in a burst triggers the check.
        self._alive_check_timer.start()

    def _check_deleted_labels(self) -> None:
        """Check pending candidate labels and reconcile any that are gone.

        Undo history is parsed HERE (deferred from ``_on_labels_painted``)
        because ``labels_update`` fires before napari populates the
        history.  By the time this 300 ms debounce callback runs, the
        history is guaranteed to be populated.

        Intent-based deletion: when a label is partially erased (still
        has voxels) but it appeared in undo candidates, treat as
        intentional deletion — zero remaining voxels and reconcile.
        """
        undo_candidates = self._parse_undo_candidates()

        candidates = set(undo_candidates)
        # Also always check the navigated label as a safety net.
        if self._last_navigated_label_id is not None:
            candidates.add(self._last_navigated_label_id)

        # Read selected_label once — needed for both fallback and intent.
        try:
            selected = self._get_selected_label()
        except Exception:
            selected = -1

        # Fallback: if undo history parsing didn't produce specific
        # candidates AND the user is in delete mode (selected_label == 0),
        # check ALL Navigator labels.  This catches cases where the undo
        # format is unexpected or the event source didn't have history.
        # Only triggers in delete mode to avoid expensive scans on every
        # normal paint stroke.
        if not undo_candidates:
            if selected == 0:
                nav = getattr(self, "_navigator_widget", None)
                if nav is not None:
                    for item in getattr(nav, "_items", []):
                        lid = item.get("label_id")
                        if lid is not None:
                            candidates.add(int(lid))
        if not candidates:
            return

        for label_id in list(candidates):
            exists = self._check_label_exists_in_volume(label_id)
            if not exists:
                # Fully erased — reconcile immediately.
                self._handle_manually_deleted_label(label_id)
                if self._last_navigated_label_id == label_id:
                    self._last_navigated_label_id = None
            elif label_id in undo_candidates:
                # Intent-based: label partially erased and the undo
                # history confirmed it was filled/erased with 0 (that's
                # how the candidate was extracted).  The undo history is
                # sufficient proof of intent — the user explicitly
                # targeted this label.  No navigation requirement: the
                # user may have clicked directly in the 3D viewer, or
                # auto-advance moved past the label after deletion.
                self._zero_label_in_volume(label_id)
                self._handle_manually_deleted_label(label_id)
                self._last_navigated_label_id = None

    # ---------------------------------------------------------------
    # Interpolation — Preview / Accept / Reject
    # ---------------------------------------------------------------

    def _get_selected_label(self) -> int:
        """Return the currently selected label from subviewers or main viewer.

        If ANY layer (main or subviewer) has ``selected_label == 0``
        (delete mode), returns 0 immediately — this is an explicit user
        choice that takes priority over stale non-zero values on other
        layers.
        """
        first_nonzero = None
        for layer in self._all_subviewer_label_layers():
            val = int(getattr(layer, "selected_label", 0))
            if val == 0:
                return 0
            if val > 0 and first_nonzero is None:
                first_nonzero = val
        main_layer = self._layer_by_name(
            self.viewer.layers, self._active_labels_name
        )
        if isinstance(main_layer, Labels):
            main_val = int(main_layer.selected_label)
            if main_val == 0:
                return 0
            if main_val > 0 and first_nonzero is None:
                first_nonzero = main_val
        return first_nonzero if first_nonzero is not None else 1

    def _run_interpolation(self) -> None:
        """Run SDF interpolation for the currently selected label.

        Saves a snapshot of the shared volume before interpolation so
        Reject can restore it.
        """
        if self._interp_running:
            return
        if self._interp_refine_mode:
            self._interp_status_label.setText(
                "Finish refine first (Accept or Discard in the Refine tab)."
            )
            return
        if self._split_running and not self._split_fragment_refine:
            return
        if self._smooth_running:
            return
        if getattr(self, "_batch_staged", False) or getattr(
            self, "_batch_running", False
        ):
            self._interp_status_label.setText(
                "Finish or close Batch Refine first."
            )
            return

        from packages.epicycle_annotator.interpolation import (
            interpolate_label_all_orientations,
        )

        self._interp_running = True
        try:
            labels_name = self._active_labels_name
            if not labels_name:
                self._interp_status_label.setText("No labels loaded.")
                return

            # Defensive shared_data alignment — see _ensure_shared_data_aligned
            shared_data = self._ensure_shared_data_aligned(labels_name)
            if shared_data is None:
                self._interp_status_label.setText("No shared label volume found.")
                return

            selected_label = self._get_selected_label()
            edited = self.sync_manager.get_edited_slices(labels_name)

            # Check the selected label has >= 2 annotated slices in at
            # least one orientation.  The SDF interpolation needs two
            # anchor shapes to blend between.
            has_slices = False
            for orient_dict in edited.values():
                for lid, slices in orient_dict.items():
                    if int(lid) == selected_label and len(slices) >= 2:
                        has_slices = True
                        break
                if has_slices:
                    break

            if not has_slices:
                self._interp_status_label.setText(
                    f"Label {selected_label}: need \u2265 2 annotated slices "
                    f"in one orientation."
                )
                return

            # Snapshot for Reject — the label's full bounding box with generous
            # margin so postprocessing changes are fully captured.
            self._interp_roi, self._interp_snapshot = self._snapshot_label_roi(
                shared_data, selected_label, edited
            )
            self._refresh_merge_button_state()
            self._refresh_label_action_buttons_state()
            self._interp_label_id = selected_label
            # Discard draw snapshot — interp snapshot now owns the undo point.
            self._clear_draw_snapshot()

            total = interpolate_label_all_orientations(
                shared_data, selected_label, edited
            )

            self._interp_status_label.setText(
                f"Label {selected_label}: {total} slice(s) interpolated."
            )

            # Enable Accept/Reject/Refine Further.
            self._accept_interp_btn.setEnabled(True)
            self._reject_interp_btn.setEnabled(True)
            self._interp_refine_btn.setEnabled(True)

            # Refresh all views to show interpolated data.
            self._sync_all_views_after_operation(label_id=selected_label)

            # Lock subviewers — drawing during preview would be undone by
            # Reject (snapshot restores pre-interpolation state).
            self._lock_subviewers_for_preview()
        finally:
            self._interp_running = False

    def _accept_interpolation(self) -> None:
        """Accept the previewed interpolation and increment the label."""
        # B2: refuse while the surface is load-locked (e.g. an async Save is in
        # flight) — an Accept here would mutate data the in-flight save already
        # snapshotted, silently losing the edit.  Mirrors _open_batch_refine_dialog.
        if getattr(self, "_load_locked", False):
            print("[REFINE-ACCEPT] interp accept refused — controller busy (load-locked)")
            return
        # [XCRASH-8] interp accept entry — capture state BEFORE
        # cleanup so we can see exactly what gets cleared.
        print(
            f"[XCRASH-8] interp ACCEPT ENTER label={self._interp_label_id} "
            f"iso_label_set={set(self._isolation_label_set) if self._isolation_label_set else None} "
            f"proxy_active={self._isolation_proxy_active} "
            f"proxy_lid={self._isolation_proxy_label_id} "
            f"iso_active={self._isolation_active}",
            flush=True,
        )
        if self._interp_label_id is None:
            self._interp_status_label.setText("No active interpolation to accept.")
            return

        # Capture undo data before clearing state.
        undo_roi = self._interp_roi
        undo_snap = self._interp_snapshot

        self._interp_snapshot = None  # discard snapshot
        self._refresh_merge_button_state()
        self._refresh_label_action_buttons_state()
        self._interp_roi = None

        label_id = self._interp_label_id
        self._interp_label_id = None

        # Clear edited slices for the accepted label so they don't
        # re-trigger on next Preview.
        labels_name = self._active_labels_name
        if labels_name:
            self.sync_manager.clear_edited_slices_for_label(
                labels_name, label_id
            )

        # Set label to next unused ID (navigator-fast, volume-fallback).
        next_label = self._next_label_id_after_accept()
        self._set_selected_label_all(next_label)

        self._accept_interp_btn.setEnabled(False)
        self._reject_interp_btn.setEnabled(False)
        self._interp_refine_btn.setEnabled(False)
        self._interp_refine_mode = False
        self._refresh_merge_button_state()
        self._refresh_label_action_buttons_state()
        self._interp_status_label.setText(
            f"Accepted label {label_id}. Now on label {next_label}."
        )

        # Sync subviewers + isolation proxy with the freshly committed
        # interpolation result.  Without this the subviewers go blank
        # immediately after Accept (they still reference pre-Accept data
        # while shared_data has been updated) — matches the pattern used
        # in every other refinement op (split, smooth, merge, reject).
        # skip_3d_refresh mirrors split's optimization: when auto-advance
        # is enabled AND the isolation proxy is active, the main 3D layer
        # stays hidden behind the proxy, so refreshing it is wasted work.
        _skip_3d = (
            self._is_auto_advance_enabled()
            and getattr(self, "_isolation_proxy_active", False)
        )
        self._sync_all_views_after_operation(
            roi=undo_roi, label_id=label_id, skip_3d_refresh=_skip_3d,
        )

        # Notify controller so it can set tier → "curated" and recalculate QC
        cb = getattr(self, "_on_interpolation_accepted_cb", None)
        if cb is not None:
            cb(label_id, roi_slices=undo_roi, old_roi_data=undo_snap)

        self._clear_draw_snapshot()
        self._restore_ready_to_draw_state()

        # Auto-advance to next triage label (matches split/smooth behavior).
        if self._is_auto_advance_enabled():
            nav = getattr(self, "_navigator_widget", None)
            if nav is not None:
                try:
                    if not nav.select_next_triage():
                        self._on_isolation_changed(set())
                except Exception:
                    pass

    def _reject_interpolation(self) -> None:
        """Reject the previewed interpolation and restore the ROI snapshot."""
        # FM6 fix (2026-05-16): capture refine-mode state BEFORE resetting
        # it so we know whether `_enter_interp_refine_mode` had force-
        # isolated the label.  If yes, we must clear isolation on reject
        # so the user doesn't end up stuck with a hidden label after
        # backing out of refine-further mode.  Normal Reject (from
        # Preview, not refine-further) intentionally leaves isolation
        # alone (this is by design).
        was_in_refine_mode = bool(getattr(self, "_interp_refine_mode", False))
        print(
            f"[INTERP-REJECT-ISO] _reject_interpolation ENTER "
            f"was_in_refine_mode={was_in_refine_mode} "
            f"interp_label_id={getattr(self, '_interp_label_id', None)} "
            f"iso_active={getattr(self, '_isolation_active', False)} "
            f"proxy_active={getattr(self, '_isolation_proxy_active', False)}"
        )

        labels_name = self._active_labels_name
        shared_data = self.sync_manager.shared_labels.get(labels_name)

        if shared_data is None:
            self._interp_status_label.setText(
                "ERROR: Label volume not found. Cannot reject."
            )
            self._interp_snapshot = None
            self._refresh_merge_button_state()
            self._refresh_label_action_buttons_state()
            self._interp_roi = None
            self._interp_label_id = None
            self._accept_interp_btn.setEnabled(False)
            self._reject_interp_btn.setEnabled(False)
            return

        if self._interp_snapshot is not None and self._interp_roi is not None:
            shared_data[self._interp_roi] = self._interp_snapshot
            self._sync_all_views_after_operation(
                roi=self._interp_roi, label_id=self._interp_label_id
            )

        # Keep edited slices — the user's hand-drawn annotations are still
        # in the restored volume and should remain tracked so a subsequent
        # Preview can re-run interpolation.

        self._interp_snapshot = None
        self._refresh_merge_button_state()
        self._refresh_label_action_buttons_state()
        self._interp_roi = None
        self._interp_label_id = None
        self._interp_refine_mode = False
        self._refresh_merge_button_state()
        self._refresh_label_action_buttons_state()
        self._accept_interp_btn.setEnabled(False)
        self._reject_interp_btn.setEnabled(False)
        self._interp_refine_btn.setEnabled(False)
        self._interp_status_label.setText("Interpolation rejected. Volume restored.")
        self._clear_draw_snapshot()
        self._restore_ready_to_draw_state()

        # FM6 fix: clear isolation ONLY if we came from refine-further
        # mode (where isolation was force-applied on entry).  Normal
        # Preview→Reject paths intentionally leave isolation as it was.
        if was_in_refine_mode:
            print(
                "[INTERP-REJECT-ISO] firing _on_isolation_changed(set()) — "
                "exiting refine-further mode, must clear forced isolation"
            )
            try:
                self._on_isolation_changed(set())
            except Exception as e:
                print(f"[INTERP-REJECT-ISO] isolation clear raised: {e!r}")

    def _enter_interp_refine_mode(self) -> None:
        """Enter refine mode after interpolation preview.

        Interpolated data is already in shared_data. Keep the interp
        snapshot as undo point, disable interp UI, open Refine tab.
        """
        if self._interp_label_id is None:
            return

        self._interp_refine_mode = True
        self._refresh_merge_button_state()
        self._refresh_label_action_buttons_state()
        label_id = self._interp_label_id

        # Disable interp buttons (snapshot kept for full undo).
        self._interpolate_btn.setEnabled(False)
        self._accept_interp_btn.setEnabled(False)
        self._reject_interp_btn.setEnabled(False)
        self._interp_refine_btn.setEnabled(False)
        self._interp_status_label.setText(
            f"Refine mode: label {label_id}. Use the Refine tab \u2192 Accept "
            f"or Discard."
        )

        # Keep label selected + isolated.
        self._set_selected_label_all(label_id)
        self._on_isolation_changed({label_id})

        # Unlock subviewers — INTERP REFINE allows drawing (the user
        # intentionally entered "work on the result" mode).
        self._restore_ready_to_draw_state()

        # Open Advanced Features dock + focus the Refine tab BY NAME.
        self._open_advanced_features()
        adv = self._get_advanced_features_widget()
        if adv is not None and hasattr(adv, "focus_tab"):
            adv.focus_tab("Refine")

    # ---------------------------------------------------------------
    # Refine layer helpers
    # ---------------------------------------------------------------

    def _create_refine_layer(self, shared_data, selected_label,
                             *, _precomputed_mask=None):
        """Create a temporary Refine layer containing only the target label.

        The Refine layer is NOT in ``_managed_layer_names``, so the
        LayerLifecycleManager skips its ``on_layer_added`` auto-handling.
        We manually add to subviewers.  Removal is handled automatically
        by ``on_layer_removed`` (which has no filter).
        """
        from napari.layers import Labels as _Labels

        refine_data = np.zeros(shared_data.shape, dtype=shared_data.dtype)
        mask = _precomputed_mask if _precomputed_mask is not None else (shared_data == selected_label)
        refine_data[mask] = selected_label

        refine_layer = self.viewer.add_labels(
            refine_data, name="Refine", opacity=0.7
        )
        refine_layer._epicycle_temporary = True  # hide from labels combo
        # Exempt the Refine layer + its subviewers from draw-clipping
        # Split SEEDS are new IDs painted OVER the
        # target label, which preserve_labels / the enclosed-fill mask would
        # otherwise refuse.  add_labels_to_subviewers + the fill-mask both read
        # this flag, so the whole split-seeding surface stays unclipped.
        refine_layer._epicycle_clip_to_existing = False
        # brush_size locked at 1.  Refine seeding
        # has always wanted brush=1 (small precise seeds); the lock just
        # enforces it the same way every other layer is enforced.
        from packages.epicycle_annotator.sync import lock_brush_size_to_1
        lock_brush_size_to_1(refine_layer)

        # Manually propagate to subviewers (lifecycle manager skips).
        self.subviewer_manager.add_labels_to_subviewers(
            refine_layer,
            sync_manager=self.sync_manager,
            drawing_handler=self.drawing_handler,
        )

        # Do NOT copy DirectLabelColormap from main labels — it only has
        # colours for labels in the QC DataFrame.  Seed IDs (max+1) would
        # render as *transparent*, making the mask appear to vanish when
        # seeds are painted over it.  napari's default cycling colormap
        # assigns visible colours to every integer label.

        # Subviewer Refine layers also locked to brush=1 (no auto-increment).
        for orient in ("XY", "XZ", "YZ"):
            sub = self.subviewer_manager.get_subviewer_layer("Refine", orient)
            if sub is not None:
                lock_brush_size_to_1(sub)
                sub._epicycle_auto_increment_label = False
                # Defer main viewer 3D re-render during seed painting.
                # The Refine layer is visible in 3D and each stroke would
                # trigger an expensive iso_categorical re-render (~200-500ms).
                # Data still syncs in-memory; explicit refresh happens on
                # Next Seed / Preview / Accept / Reject.
                sub._epicycle_defer_main_refresh = True

        # Brick U2: the remembered outline preference,
        # applied at birth so the tick carries from object to object.
        #
        # AFTER the three mirrors exist, and that order is the opposite
        # of the one that looks right.  MEASURED on the live panel
        # (driver 11i, 2026-09-20): written before
        # `add_labels_to_subviewers` the next object came up outlined in
        # 3D and SOLID in all three side views -- `mirrors=[0, 0, 0]`.
        # Nothing copies `contour` when a mirror is created; the mirrors
        # get it from the property-change propagation
        # `sync_label_properties` runs, and a change made before those
        # layers existed has no listener to carry it.  Written here it
        # fires with the listeners attached and all three follow -- and
        # the side views are the whole point of the feature, so
        # "persists across objects" is only true if it persists there.
        try:
            # Same suppression as the toggle: napari warns "Contours are
            # not displayed during 3D rendering" on every write, because
            # the main viewer is 3D (audit H2).
            self._refine_layer = refine_layer
            self._set_refine_contour(bool(self._refine_contour_on))
            print("[OUTLINE] new object born with contour={} in the three "
                  "side views (the 3D view does not draw outlines)".format(
                      int(bool(self._refine_contour_on))))
        except Exception as exc:  # noqa: BLE001
            print("[OUTLINE] could not apply the remembered outline: "
                  "{!r}".format(exc))

        self._refine_layer = refine_layer
        return refine_layer

    def _set_refine_contour(self, on: bool) -> None:
        """Write ``contour`` on the Refine layer, quietly.

        The write is correct and reaches the three side views.  What it
        also does is make napari warn: `Labels._calculate_contour`
        returns None and raises "Contours are not displayed during 3D
        rendering" whenever the sliced labels have more than two
        dimensions, and the main curation viewer is 3D.  So every tick
        of the box raised a toast about a limitation the help line now
        states outright.

        The filter names THAT message only -- swallowing whatever else a
        layer might say would hide the next real problem -- and
        `catch_warnings` restores the global filter state on the way
        out, so nothing about warnings elsewhere changes.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=_CONTOUR_3D_WARNING + ".*")
            self._refine_layer.contour = 1 if on else 0

    def _on_refine_contour_toggled(self, checked) -> None:
        """"Show this object as an outline".

        A property set and nothing else.  `sync_label_properties`
        already propagates `contour` to every layer whose
        `_base_label_name` matches, i.e. `Refine_XY`, `Refine_XZ` and
        `Refine_YZ` -- which is the point, because the side views are
        where the markers are painted.  Its `main_layer` lookup resolves
        to the Refine layer itself and is skipped by the
        `is not active_layer` guard, so the main `curation_3d_*` layer
        is provably out of reach: it has a different base name.

        The 3D view is NOT outlined and cannot be: napari refuses
        contours whenever the sliced labels have more than two
        dimensions (audit H2, 2026-09-20).  The side views are the ones
        that matter here, so the feature stands -- but it is said
        plainly in the help line, the guide and the line below, rather
        than promised and then not delivered.

        No isolation, sync, rendering, mirror or proxy code is edited or
        called from here.
        """
        on = bool(checked)
        self._refine_contour_on = on
        if self._refine_layer is None:
            print("[OUTLINE] remembered for the next object: {}".format(on))
            return
        try:
            self._set_refine_contour(on)
            print("[OUTLINE] Refine.contour={} - the three side views, "
                  "where you paint; the 3D view is unchanged because "
                  "napari does not draw outlines in 3D".format(int(on)))
        except Exception as exc:  # noqa: BLE001
            print("[OUTLINE] could not set the outline: {!r}".format(exc))

    def _remove_refine_layer(self) -> None:
        """Remove the Refine layer.  ``on_layer_removed`` auto-cleans subviewers."""
        if self._refine_layer is not None:
            try:
                # Hide before removing — disconnects vispy visual from
                # scene graph so no stale texture references remain.
                self._refine_layer.visible = False
                if self._refine_layer in self.viewer.layers:
                    self.viewer.layers.remove(self._refine_layer)
            except Exception:
                pass
            self._refine_layer = None
        self._refine_baseline = None

    def _refresh_refine_in_subviewers(self) -> None:
        """Push updated Refine layer data to all subviewer copies.

        Uses in-place copy when shapes match to avoid GPU texture
        reallocation (same pattern as subviewers.py:_add_or_replace_labels).
        """
        if self._refine_layer is None:
            return
        from packages.epicycle_annotator.subviewers import transpose_data

        data = self._refine_layer.data
        for orient in ("XY", "XZ", "YZ"):
            sub = self.subviewer_manager.get_subviewer_layer("Refine", orient)
            if sub is not None:
                try:
                    transposed = transpose_data(data, orient)
                    if sub.data.shape == transposed.shape:
                        sub.data[:] = transposed  # in-place, no realloc
                    else:
                        sub.data = transposed
                    sub.refresh()
                except Exception:
                    pass

    # ---------------------------------------------------------------
    # Assisted Split orchestration
    # ---------------------------------------------------------------

    def _reset_split_state(self) -> None:
        """Reset all split-related state variables to idle defaults.

        Extracted to avoid duplicating this 12-variable reset block
        across accept/reject/error paths.

        Also invalidates ``_isolation_label_set`` (P0 #1 fix,
        2026-05-16) — split-accept destroys the original label_id;
        if it was in the multi-isolation cache, the cache is now
        stale and subsequent isolation calls would fast-skip
        (line ~2143 ``if frozen == prev_cache: return``) with the
        old set including a label that no longer exists in the
        volume.  Clearing the cache forces a fresh rebuild on the
        next ``_apply_multi_isolation_colormap`` call.
        """
        self._split_snapshot = None
        self._split_roi = None
        self._split_label_id = None
        self._split_running = False
        self._refresh_merge_button_state()
        self._refresh_label_action_buttons_state()
        self._split_previewed = False
        self._split_seed_ids = []
        self._split_painted_marker_ids = []
        self._split_isolate_was_on = False
        self._split_main_visible = True
        self._refine_pre_preview = None
        self._split_fragment_refine = False
        self._split_fragment_ids = []
        self._fragment_refine_roi = None
        # U4: the fragment commit's two captured values go
        # with the session.  Carrying `_reshape_fragment_ticked` into the
        # next one would route the NEXT object's fragment Accept through
        # a record built from ids that are not on screen any more.
        self._reshape_fragment_ticked = None
        self._reshape_fragment_written = 0
        self._detected_peaks = None
        self._auto_detect_active = False
        self._seed_id_base = 0
        # Reshape R1: the guided session's own state.  The GUIDE LAYER is
        # deliberately NOT dropped here -- it survives across successive
        # splits in one sitting and is torn down on patch switch / run
        # switch / annotator close.
        self._reshape_active = False
        # AUDIT 2026-09-09: this MUST be cleared here.  Every terminal
        # split path funnels through this reset, and a stranded True
        # makes the next Preview refuse to start (`_preview_reshape`
        # early-returns on it) with no way for the user to clear it.
        self._reshape_running = False
        self._seed_find_running = False
        self._split_busy_kind = None
        self._reshape_result = None
        self._reshape_final_ids = {}
        self._reshape_flags = {}
        self._reshape_override = False
        self._reshape_arbiter_merged = False
        # The last Find's seed crop, kept only so a painted addition can
        # be merged with it on the next Preview, and the
        # rows it filled the list with, kept so a painted addition does
        # not REPLACE them.
        self._reshape_found_seeds = None
        self._reshape_found_marker_ids = []
        self._reshape_found_marker_stats = {}
        # The ids the LAST Find wrote into the Refine layer, so the next
        # Find can take them out again (fix pass H2).  Session state --
        # it goes with the session, or the next object's first Find
        # would try to erase ids that belong to somebody else.
        self._materialised_ids = []
        # ...and so is the parked-marker stash.  It holds ROI
        # coordinates, and the ROI belongs to THIS object: carrying it
        # into the next session would let a re-tick paint one object's
        # marker onto another one.
        self._parked_marker_voxels = {}
        # Brick U3: the wall picture is about ONE cut on ONE object.
        # A stranded True would put the next object's first Preview up
        # with a picture the user never asked for.
        self._walls_wanted = False
        # P0 #1 (2026-05-10 audit): invalidate multi-isolation cache so
        # the next _apply_multi_isolation_colormap doesn't fast-skip
        # with stale label IDs.
        self._isolation_label_set = None

    def _start_split(self) -> None:
        """Begin the assisted split workflow for the currently selected label.

        Creates a temporary **Refine** layer containing only the target label,
        hides the main labels layer, and sets ``selected_label`` on the refine
        layer to ``max + 1`` for the first seed.  Painting happens on the
        refine layer; the main volume is restored/merged on accept/reject.
        """
        if self._split_running:
            self._update_split_ui(status_text="Split session already active.")
            return
        if self._smooth_running:
            self._update_split_ui(
                status_text="Complete or cancel smooth before starting split."
            )
            return
        if getattr(self, "_batch_staged", False) or getattr(
            self, "_batch_running", False
        ):
            self._update_split_ui(
                status_text="Finish or close Batch Refine before starting a split."
            )
            return

        labels_name = self._active_labels_name
        if not labels_name:
            self._update_split_ui(status_text="No labels loaded.")
            return

        # Defensive shared_data alignment — see _ensure_shared_data_aligned.
        # Was the explicit fix that resolved merge → split truncation
        # (commit cfe3c6b). Now delegated to the shared helper so smooth,
        # interp, and merge get the same guarantee.
        shared_data = self._ensure_shared_data_aligned(labels_name)
        if shared_data is None:
            self._update_split_ui(status_text="No shared label volume found.")
            return

        selected_label = self._get_selected_label()
        if selected_label <= 0:
            self._update_split_ui(status_text="Select an object to reshape.")
            return

        # Build label mask once — reused for existence check, ROI, and zeroing.
        # Avoids 3 separate full-volume scans of `shared_data == label`.
        label_mask = shared_data == selected_label
        if not label_mask.any():
            self._update_split_ui(
                status_text=f"Label {selected_label} not found in volume."
            )
            return

        # --- Reshape R1: is this a GUIDED split? -------------
        # Decided here, once, so every branch below reads one flag.  This
        # is the last thing before the destructive steps, and it is after
        # every validation early-return -- so the lock below cannot be
        # stranded by a return.
        _adv = self._get_advanced_features_widget()
        self._reshape_active = bool(
            _adv is not None
            and getattr(_adv, "is_guided_split", None) is not None
            and _adv.is_guided_split()
        )
        if self._reshape_active:
            # THE LOCK GOES ON BEFORE THE HOLE IS PUNCHED.
            # Between here and Accept the target label does
            # not exist in the volume; the manual split leaves Save and
            # Commit Triage enabled in that window and will happily
            # persist a volume with a hole in it (follow-up F2).  The
            # guided path does not.  This is a deliberate behaviour
            # change, and it is the point.
            self._fire_reshape_lock(True)
            self._refresh_reshape_inputs()
            print("[RESHAPE] guided split armed - lock ON before the punch")

        # 1. Snapshot for reject (full ROI from shared_data).
        #    The Reshape panel's "Search padding" is read HERE and nowhere
        #    else: the engine ignores its own margin= once a roi is passed,
        #    and this ROI is the one every later stage uses.
        _padding = self._reshape_search_padding()
        self._split_roi, self._split_snapshot = self._snapshot_label_roi(
            shared_data, selected_label, _precomputed_mask=label_mask,
            margin=_padding,
        )
        print("[RESHAPE] search padding at Start = {} voxels".format(_padding))

        self._split_label_id = selected_label
        self._split_running = True
        self._refresh_merge_button_state()
        self._refresh_label_action_buttons_state()
        self._clear_draw_snapshot()

        # 2. Create Refine layer with target label extracted.
        refine_layer = self._create_refine_layer(
            shared_data, selected_label, _precomputed_mask=label_mask
        )

        # 3. Save refine baseline (ROI, for seed detection in watershed).
        self._refine_baseline = refine_layer.data[self._split_roi].copy()

        # 4. Zero out target in shared volume (creates "hole").
        # Reuse the mask instead of recomputing shared_data == selected_label.
        shared_data[label_mask] = 0

        # 5. Hide main layer FIRST, then clear isolation state.
        #    Hiding first prevents the expensive full-volume 3D re-render
        #    that _on_isolation_changed(set()) would trigger when restoring
        #    all labels to visible.
        main_layer = self._layer_by_name(self.viewer.layers, labels_name)
        if isinstance(main_layer, Labels):
            self._split_isolate_was_on = self._isolation_active
            self._split_main_visible = bool(getattr(main_layer, "visible", True))
            main_layer.visible = False
            # Clear isolation state after hiding — no 3D render triggered.
            if self._isolation_active:
                self._remove_isolation_proxy(restore_visibility=False)
                self._isolation_active = False
                self._propagate_show_selected_label(False)

        # 6. Refresh main subviewers (propagate the zeroed-out hole).
        self._sync_all_views_after_operation(roi=self._split_roi)

        # 7. Lock main labels layer (prevent accidental edits during split).
        if isinstance(main_layer, Labels):
            main_layer.editable = False
            main_layer.mode = "pan_zoom"

        # 8. Set refine layer active, paint mode, selected_label = seed ID.
        # Use navigator items for max label ID (fast path) to avoid two
        # full-volume np.max() scans.  Falls back to volume scan if needed.
        next_label = self._next_label_id_after_accept()
        self._split_seed_ids = [next_label]
        self._split_painted_marker_ids = [next_label]
        refine_layer.selected_label = next_label  # name-scoped sync
        self.viewer.layers.selection.active = refine_layer
        # Set paint mode AFTER setting active — sync_active_layer fires on
        # the selection change and would propagate pan_zoom to subviewers.
        # Setting paint here triggers sync_label_modes → subviewers get paint.
        refine_layer.mode = "paint"

        # 9. Auto-enable EDT overlay (highlights target on refine layer).
        # Safe to auto-enable: the overlay is hidden in the main 3D viewer
        # (visible=False) to avoid expensive MIP rendering, but visible in
        # the 2D subviewers where it's useful for seed placement.
        self._auto_enable_edt_for_split()

        # Restore active layer to Refine — adding the EDT overlay via
        # viewer.add_image() auto-selects it, and sync_active_layer
        # propagates that to subviewers.  Without this, painting targets
        # the EDT overlay (Image) instead of the Refine layer (Labels).
        try:
            self.viewer.layers.selection.active = refine_layer
        except Exception:
            pass

        # Reset auto-detect state.
        self._detected_peaks = None
        self._auto_detect_active = False

        # NOTHING IS PAINTED YET, so nothing is listed (fix pass H1).
        # `seed_ids=` fills the TICKABLE list now, and a row there is a
        # marker the cut is made from: listing the id allocated for the
        # first brush stroke put a zero-voxel "Marker 1" in it, which
        # made `get_checked_marker_ids()` answer `[id]` instead of
        # `None`.  The guided Preview then handed that id over as
        # `keep_seed_ids` and `np.isin(seeds, keep)` filtered away every
        # marker the detector had just found -- "No markers found in
        # this object" on a perfectly good object, straight after Start.
        # The rule is unconditional, because routing is decided LIVE:
        # a session started on source 0 can be previewed on source 4.
        self._update_split_ui(
            seed_count=0,
            seed_ids=[],
            split_active=True,
            edt_enabled=True,
            start_enabled=False,
            next_seed_enabled=True,
            undo_seed_enabled=False,
            refine_seeds_enabled=False,
            preview_enabled=False,
            accept_enabled=False,
            reject_enabled=True,
            reject_text="Discard",
            review_active=False,
            detect_enabled=True,
            detect_visible=False,
            status_text=(
                f"Painting marker 1 (label {next_label}). "
                "Paint inside the object in the 2D side views, then Add "
                "marker.  Or press {} to place them for you.".format(
                    self._marker_finder_name())
            ),
        )

        # A guided session has nothing to paint, so Preview is live at
        # once.  Additive: the legacy path never reaches this branch.
        if self._reshape_active:
            self._update_split_ui(
                preview_enabled=True,
                status_text=self._reshape_ready_text(selected_label),
            )

    def _next_seed(self) -> None:
        """Advance to the next seed ID for painting.

        If called after Preview, restores the pre-preview seed painting state
        so the user can add more seeds at junction points and re-Preview.
        """
        if not self._split_running or self._refine_layer is None:
            return

        # Transition from auto-detect to manual seeding.
        self._auto_detect_active = False

        # If called after Preview, put the markers back on screen and
        # take the result down.  This is the LOOP: a
        # marker added after a preview must not sit beside a result the
        # new marker had no part in.
        self._leave_reshape_review("Marker added - press Preview to cut again.")

        # Robust next label: max of tracked markers AND current refine data.
        next_label = max(
            max(self._split_painted_marker_ids or [0]),
            max(self._split_seed_ids or [0]),
            int(np.max(self._refine_layer.data)),
        ) + 1
        self._split_painted_marker_ids.append(next_label)
        self._split_seed_ids = list(self._split_painted_marker_ids)
        self._refine_layer.selected_label = next_label  # name-scoped sync
        print("[MARKERS] Add marker -> id {} ({} painted markers)".format(
            next_label, len(self._split_painted_marker_ids)))

        _rows, _stats = self._marker_rows_for_panel()
        seed_count = len(_rows)
        self._update_split_ui(
            seed_count=seed_count,
            seed_ids=_rows, marker_stats=_stats,
            active_seed_id=next_label,
            start_enabled=False,
            next_seed_enabled=True,
            refine_seeds_enabled=False,
            preview_enabled=(seed_count >= 2),
            accept_enabled=False,
            reject_enabled=True,
            reject_text="Discard",
            review_active=False,
            status_text=(
                f"Painting marker {seed_count} (label {next_label})."
            ),
            undo_seed_enabled=(seed_count >= 2),
        )
        # H2, and this is where the audit's failure chain ended: this
        # repopulate preserves OFF ticks BY ID, so over a list that some
        # earlier path had emptied it brings a parked marker's row back
        # TICKED while its voxels are still stashed.  S1 would say "3
        # ticked" and the engine would be handed 2.
        self._sync_marker_parking()

    def _undo_last_seed(self) -> None:
        """Remove the last marker and go back to painting the previous one.

        If a preview is on screen it is put away first (the markers come
        back).  Cannot undo below 1 marker: the first
        one is the starting label.
        """
        if not self._split_running or self._refine_layer is None:
            return
        if len(self._split_painted_marker_ids) < 2:
            return

        # The preview goes away and the MARKERS come back before
        # anything is popped: after a preview the ROI holds pieces, and
        # zeroing "the last id" out of that would erase a piece.
        self._leave_reshape_review("Marker removed - press Preview to cut again.")

        # Pop the last marker and zero it out in the refine layer.
        removed_id = self._split_painted_marker_ids.pop()
        self._split_seed_ids = list(self._split_painted_marker_ids)
        roi_data = self._refine_layer.data[self._split_roi]
        roi_data[roi_data == removed_id] = 0
        # ...and if it was PARKED its voxels are not on the layer at
        # all, so zeroing found nothing: the stash is what has to be
        # emptied, or a later tick would paint an undone marker back.
        (getattr(self, "_parked_marker_voxels", None) or {}).pop(
            int(removed_id), None)
        print("[MARKERS] Undo -> removed id {} ({} painted markers)".format(
            removed_id, len(self._split_painted_marker_ids)))

        # Switch to the new last marker.
        current_seed = self._split_painted_marker_ids[-1]
        self._refine_layer.selected_label = current_seed

        # Restore paint mode + active layer.
        self.viewer.layers.selection.active = self._refine_layer
        self._refine_layer.mode = "paint"

        # Refresh refine layer in all subviewers.
        self._refine_layer.refresh()
        self._refresh_refine_in_subviewers()

        _rows, _stats = self._marker_rows_for_panel()
        seed_count = len(_rows)
        self._update_split_ui(
            seed_count=seed_count,
            seed_ids=_rows, marker_stats=_stats,
            active_seed_id=current_seed,
            start_enabled=False,
            next_seed_enabled=True,
            undo_seed_enabled=(seed_count >= 2),
            refine_seeds_enabled=False,
            preview_enabled=(seed_count >= 2),
            accept_enabled=False,
            reject_enabled=True,
            reject_text="Discard",
            review_active=False,
            status_text=(
                f"Undid marker {seed_count + 1}. "
                f"Painting marker {seed_count} (label {current_seed})."
            ),
        )

    def _refine_seeds(self) -> None:
        """"Edit markers": put the preview away and paint some more.

        Unlike ``_next_seed`` this does NOT add a marker -- it lets the
        user extend the ones already there (paint more of marker 2 near
        a junction) and then Preview again.
        """
        if not self._split_running or self._refine_layer is None:
            return

        # One predicate for "the preview goes away and the markers come
        # back", shared with Add marker, Undo, Find and every knob.
        self._leave_reshape_review("Markers changed - press Preview to cut again.")

        # Set selected_label to the last marker painted.
        if self._split_painted_marker_ids:
            self._refine_layer.selected_label = (
                self._split_painted_marker_ids[-1])

        # Restore paint mode and active layer (may have changed during
        # Preview interaction — e.g., clicking elsewhere sets pan_zoom).
        self.viewer.layers.selection.active = self._refine_layer
        self._refine_layer.mode = "paint"

        _rows, _stats = self._marker_rows_for_panel()
        seed_count = len(_rows)
        self._update_split_ui(
            seed_count=seed_count,
            seed_ids=_rows, marker_stats=_stats,
            active_seed_id=(
                self._split_painted_marker_ids[-1]
                if self._split_painted_marker_ids else None
            ),
            start_enabled=False,
            next_seed_enabled=True,
            undo_seed_enabled=(seed_count >= 2),
            refine_seeds_enabled=False,
            preview_enabled=(seed_count >= 2),
            accept_enabled=False,
            reject_enabled=True,
            reject_text="Discard",
            review_active=False,
            status_text=(
                "Markers are back. Add or extend them, then Preview."),
        )

    def _on_select_seed(self, label_id: int) -> None:
        """Switch painting to an existing marker (clicked in the list)."""
        if self._refine_layer is not None and self._split_running:
            self._refine_layer.selected_label = label_id

    # ------------------------------------------------------------------
    # Brick U1 -- THE LOOP
    #
    # One predicate, one exit.  Everything that changes the MARKERS --
    # a tick, a Find, Add marker, Undo, any marker knob -- and everything
    # that changes HOW the cut is made -- the wall kind, the wall
    # channel, the prepare ticks, the solver, anything under How to cut
    # -- puts the preview away and shows the markers again.
    #
    # It has to be a SNAP-BACK, not just "leave the cut painted".  The
    # preview writes PIECE ids over the whole Refine ROI, so while it is
    # on screen the layer holds no markers at all: the reconciler would
    # list pieces as markers, the recipe would hand the engine the whole
    # previous partition as seeds, and Undo would erase a piece.
    # ------------------------------------------------------------------
    def _review_is_on_screen(self) -> bool:
        """Is a PREVIEW currently painted into the Refine ROI?

        The Refine layer holds either the MARKERS or the PIECES, never
        both, and everything that reads that ROI has to
        know which.  `_split_previewed` is the legacy path's flag and
        `_reshape_result` the engine's; a preview on screen sets one or
        the other, so the honest predicate is the OR.

        It exists because `_refine_pre_preview` is the snapshot the
        snap-back restores, and re-taking it while a preview is up
        captures the PIECES -- after which the next marker tick "puts
        the markers back" by painting the previous cut onto the object.
        """
        return bool(getattr(self, "_split_previewed", False)
                    or getattr(self, "_reshape_result", None) is not None)

    def _leave_reshape_review(self, reason: str) -> None:
        """Put the preview away and show the markers again.  Idempotent.

        Does exactly these steps and nothing else.  In
        particular it does NOT clear the marker list -- a wall knob did
        not change the markers -- and it does not touch any layer except
        the Refine layer's own data.
        """
        if not getattr(self, "_split_running", False):
            return
        # U4 (2.7 B3).  There is no review to leave in fragment refine:
        # the pieces are already in the volume, the Refine layer this
        # method snaps back has been removed, and the only thing it
        # could still do is null `_reshape_result` -- i.e. destroy the
        # commit record on a marker tick.  The panel locks the marker
        # work in that mode; this is the second lock on the same door.
        if getattr(self, "_split_fragment_refine", False):
            return
        had_review = bool(getattr(self, "_split_previewed", False)
                          or getattr(self, "_reshape_result", None) is not None)
        if not had_review:
            return
        print("[RESHAPE-LOOP] leaving review: {}".format(reason))

        # 1. The result is gone.  Nothing downstream may read it again.
        self._reshape_result = None
        self._reshape_final_ids = {}
        self._reshape_flags = {}
        self._reshape_arbiter_merged = False
        self._reshape_override = False

        # 2. `_split_previewed` goes down BEFORE anything can run a
        #    finder: `_detect_seeds` early-returns on it, so clearing it
        #    late makes Detect a silent no-op (A-H2).
        self._split_previewed = False

        # 2b. The wall picture is part of the preview.
        #     The elevation it was built from belongs to the result that
        #     was just nulled, so leaving it up would show the user what
        #     the tool saw for a cut that is no longer on screen.
        self._drop_walls_picture()

        # 3. The Refine ROI goes back to the markers exactly as they
        #    were when Preview was pressed.
        if (self._refine_pre_preview is not None
                and self._refine_layer is not None
                and self._split_roi is not None):
            self._refine_layer.data[self._split_roi] = (
                self._refine_pre_preview.copy()
            )
            try:
                self._refine_layer.refresh()
            except Exception:
                pass
            self._refresh_refine_in_subviewers()
            print("[RESHAPE-LOOP] snap-back: the object shows its markers "
                  "again ({} painted)".format(
                      len(self._split_painted_marker_ids)))
        # 4. ...and so do the ids the rest of the code reads.
        self._split_seed_ids = list(self._split_painted_marker_ids)

        # 5. S5 goes away, Accept goes grey and says why.
        adv = self._get_advanced_features_widget()
        if adv is not None:
            try:
                adv.update_split_ui_state(review_active=False)
            except Exception:
                pass
            # The same sentence on S1, in the footer hint and on Accept
            # itself -- a grey button with no explanation is the bug
            # this round is about.  All THREE, not two: the footer hint
            # is keyed on the reject button's text, which a snap-back
            # never touches, so without this it went on reading "Accept
            # keeps the ticked pieces" right under a dead Accept.
            try:
                adv.set_accept_reason(reason)
                adv.set_footer_reason(reason)
            except Exception:
                pass
        self._update_split_ui(
            accept_enabled=False, preview_enabled=True, reject_enabled=True,
            reject_text="Discard", review_active=False,
            refine_further_enabled=False,
            # There is no preview left to go back from, so "Edit
            # markers" has nothing to do and goes with it.
            refine_seeds_enabled=False,
            status_text=reason,
        )

    def _on_reshape_cut_knob_changed(self) -> None:
        """A wall / solver / how-to-cut knob moved while a result was up."""
        self._leave_reshape_review(
            "Cut settings changed - press Preview to cut again.")

    def _on_reshape_marker_check_changed(self) -> None:
        """A marker was ticked or unticked.

        TWO things, in this order, and the order is load-bearing (the order of
        the two steps).  Leaving review restores `_refine_pre_preview`, which
        was
        taken off the live layer and therefore holds whatever was parked
        when Preview was pressed; parking BEFORE that would be undone by
        it, and a re-tick before it would be overwritten by it.  So:
        put the markers back on screen first, then apply the tick.

        It fires outside review too now -- there is nothing to leave
        then, and `_leave_reshape_review` returns at once, but the tick
        still has to lift a marker off the object or put it back.
        """
        self._leave_reshape_review(
            "Markers changed - press Preview to cut again.")
        self._sync_marker_parking()

    def _marker_rows_from_refine(self) -> list:
        """Reconcile the marker list with what is PAINTED on the layer.

        A RECONCILER, never the truth (A-M7).  The user paints with a
        napari brush the panel never hears about, so ids can appear on
        the Refine layer that `_split_painted_marker_ids` has never
        seen.  This adds those, and only those: it looks at ids ABOVE
        the ones already known, so it can never invent a marker out of
        the target's own voxels or out of a piece.

        It must never run while a preview is on screen -- the ROI holds
        pieces then, not markers -- which is why every caller either
        runs before a Preview or after `_leave_reshape_review`.

        That rule is doing TWO jobs, and the second is easy to miss
        (audit M6).  The obvious one is meaning: the ids in the ROI
        during review are pieces, and listing them as markers would be
        nonsense.  The other is arithmetic.  Piece ids are minted from
        `_next_label_id_after_accept()` and marker ids from
        `max(refine.max(), shared_max, max(painted)) + 1`; nothing
        guarantees the two sequences stay apart, so a piece id can land
        at or above this method's `floor`.  Run during review, the
        adds-only rule would then read a PIECE as a newly painted marker
        -- and it would look entirely plausible, because the arithmetic
        is right and only the moment is wrong.  The snap-back (section
        1.2) is what keeps that moment from arising at all: the pieces
        are gone from the ROI before any caller here runs.
        """
        known = list(self._split_painted_marker_ids or [])
        if (self._refine_layer is None or self._split_roi is None
                or getattr(self, "_split_previewed", False)):
            return known
        try:
            present = np.unique(self._refine_layer.data[self._split_roi])
        except Exception:
            return known
        floor = max(known) if known else 0
        target = self._split_label_id
        added = [
            int(v) for v in present
            if int(v) > floor and int(v) != int(target or 0)
        ]
        if added:
            known.extend(sorted(added))
            self._split_painted_marker_ids = known
            self._split_seed_ids = list(known)
            print("[MARKERS] reconcile found {} painted id(s) the panel had "
                  "not seen: {}".format(len(added), sorted(added)))
        return known

    def _painted_marker_ids_on_layer(self) -> list:
        """The marker ids that actually have voxels on the Refine layer.

        `_start_split` allocates the first id before anything is
        painted, so "the list is non-empty" is NOT the same question as
        "the user has painted something".  Every rule that changes what
        the engine is asked for -- the index-1 re-map and the
        painted-addition promotion -- reads THIS, so a session that has
        only been started still sends exactly what it always sent.
        """
        # A PARKED marker is not on the layer by definition --
        # that is what parking means -- and it is subtracted in BOTH
        # branches: during review the ROI holds pieces, so the list is
        # the only truth and the list still carries the parked id.
        #
        # The stash is read as an ATTRIBUTE here, not through
        # `_parked_marker_ids()`, for the same reason `_materialised_ids`
        # is read with `getattr` all over this file: this is a reader
        # that must answer on any half-built session, and it must not
        # acquire a dependency on a method being present.
        parked = set(getattr(self, "_parked_marker_voxels", None) or {})
        ids = [int(i) for i in (self._split_painted_marker_ids or [])
               if int(i) != int(self._split_label_id or 0)
               and int(i) not in parked]
        if not ids or self._refine_layer is None or self._split_roi is None:
            return []
        if getattr(self, "_split_previewed", False):
            # The ROI holds pieces right now; the list is the only truth.
            return ids
        try:
            present = set(
                int(v) for v in
                np.unique(self._refine_layer.data[self._split_roi])
            )
        except Exception:
            return ids
        return [i for i in ids if i in present]

    def _merged_marker_seed_layer(self):
        """The ticked FOUND markers plus the painted ones, in one array.

        The promotion of the marker list's last row.  Sources 2-4 keep their found
        markers in ``guide_markers``, not on the Refine layer, so a
        marker the user then paints by hand would be the ONLY thing a
        `paint` recipe carried -- and re-running the detector would
        throw the addition away.  This merges the two.

        Found ids are offset above every painted one, because the
        detectors number from 1 and the painted ids are minted from the
        volume's maximum; the solver is invariant to the numbering.
        Returns ``None`` when there is nothing found to merge.
        """
        found = getattr(self, "_reshape_found_seeds", None)
        if found is None or self._refine_layer is None or self._split_roi is None:
            return None
        refine_roi = self._refine_layer.data[self._split_roi]
        found = np.asarray(found)
        if found.shape != refine_roi.shape:
            print("[MARKERS] promotion refused - the found markers are a "
                  "different shape from the ROI")
            return None
        painted = [int(i) for i in (self._split_painted_marker_ids or [])
                   if int(i) != int(self._split_label_id or 0)]
        keep = self._checked_marker_ids()
        keep_set = None if keep is None else {int(k) for k in keep}
        out = np.zeros_like(refine_roi)
        offset = (max(painted) if painted else 0) + 1
        kept_found = 0
        for fid in (int(v) for v in np.unique(found) if int(v) > 0):
            if keep_set is not None and fid not in keep_set:
                continue
            out[found == fid] = offset + kept_found
            kept_found += 1
        kept_painted = [
            i for i in painted
            if keep_set is None or i in keep_set
        ]
        if kept_painted:
            mask = np.isin(refine_roi, kept_painted)
            out[mask] = refine_roi[mask]
        if not out.any():
            return None
        print("[MARKERS] promoted to a painted cut: {} found + {} painted"
              .format(kept_found, len(kept_painted)))
        return out.astype(refine_roi.dtype, copy=False)

    def _marker_rows_for_panel(self):
        """``(ids, stats)`` for the ONE list: found markers AND painted.

        `_next_seed` used to push the painted ids alone, which on a
        found source REPLACED the list -- press Add marker after a Find
        and the three markers you were about to cut with vanished from
        the panel, while the recipe still used them.  The list is what
        the cut uses, so it has to show both.

        Order is found-then-painted, so the row numbers a Find gave do
        not shuffle when a marker is added by hand.
        """
        found = [int(i) for i in (
            getattr(self, "_reshape_found_marker_ids", None) or [])]
        stats = dict(getattr(self, "_reshape_found_marker_stats", None) or {})
        painted = [int(i) for i in (self._split_painted_marker_ids or [])]
        seen = set(found)
        return found + [p for p in painted if p not in seen], stats

    def _checked_marker_ids(self):
        """The TICKED marker ids, or ``None`` when the list is empty.

        ``None`` and ``[]`` are different answers, exactly as they are
        in the panel: ``None`` means nothing has ever been listed, and
        ``[]`` means the user unticked every one.
        """
        adv = self._get_advanced_features_widget()
        if adv is None or not hasattr(adv, "get_checked_marker_ids"):
            return None
        try:
            return adv.get_checked_marker_ids()
        except Exception:
            return None

    def _unticked_marker_ids(self):
        """The UNTICKED marker ids, or ``None`` when the list is empty.

        The other half of `_checked_marker_ids`, and the one parking
        asks: "which rows exist AND are off".  The distinction matters,
        because *no list at all* is not the same as *every row ticked* --
        with no list there is nothing that could ever put a parked
        marker back, so U2b unparks instead of leaving voxels stranded.
        """
        adv = self._get_advanced_features_widget()
        if adv is None or not hasattr(adv, "get_unticked_marker_ids"):
            return None
        try:
            return adv.get_unticked_marker_ids()
        except Exception:
            return None

    # ---------------------------------------------------------------
    # Brick U2b -- an UNTICKED marker comes OFF the object
    #
    # The owner tested the shipped panel on real data and found the old
    # rule misleading: a marker you have just said is NOT an object went
    # on sitting on the object, in 3D and in all three side views,
    # looking exactly like the ones that were about to cut it.  So a
    # tick is now a LIFT: the voxels go into a stash keyed by id, and a
    # re-tick puts them back byte-identically.
    #
    # Three rules, each one a bug in the version without it:
    #
    # * never while a preview is on screen -- the ROI holds PIECES then,
    #   and parking would stash a piece.  The tick handler leaves review
    #   FIRST, which restores the markers, and only then parks.
    # * the restore is a SAFE WRITE: only where the ROI is 0 or the id
    #   itself, so a re-tick can never steal a voxel a neighbouring
    #   marker has been painted over in the meantime.
    # * the stash is session state and goes wherever the markers go: a
    #   re-Find, an Undo, and `_reset_split_state`.
    # ---------------------------------------------------------------
    def _parked_marker_ids(self) -> list:
        """The ids whose voxels are currently lifted off the object."""
        return sorted(
            int(i) for i in (getattr(self, "_parked_marker_voxels", None)
                             or {})
        )

    def _park_marker(self, marker_id: int) -> int:
        """Lift one marker off the Refine layer.  Returns the voxel count.

        Idempotent: a marker already in the stash is left alone, because
        re-parking would overwrite its real shape with an empty one.

        **What goes back under it is the BASELINE, not zero** (audit H1).
        A marker is painted ON the object, so writing 0 did not hide it
        -- it punched a background-coloured bite out of the very thing
        being reshaped, in the 3D view and in all three side views.  The
        owner asked for the marker to be hidden, not for the object to
        be damaged.  `_refine_baseline` is the ROI crop as it was before
        any marker was painted, so it answers voxel by voxel: the object
        where the marker covered the object, background where it
        overhung the edge.  With no baseline to consult we fall back to
        0, which is the pre-H1 behaviour and the honest answer when the
        only thing we know is "there is no object record".
        """
        mid = int(marker_id)
        stash = getattr(self, "_parked_marker_voxels", None)
        if stash is None:
            stash = self._parked_marker_voxels = {}
        if mid in stash:
            return 0
        if self._refine_layer is None or self._split_roi is None:
            return 0
        roi = self._refine_layer.data[self._split_roi]
        idx = np.nonzero(roi == mid)
        stash[mid] = tuple(a.copy() for a in idx)
        n = int(idx[0].size)
        if n:
            target = int(self._split_label_id or 0)
            base = getattr(self, "_refine_baseline", None)
            if base is not None and target and base.shape == roi.shape:
                roi[idx] = np.where(base[idx] == target, target, 0)
            else:
                roi[idx] = 0
        print("[MARKERS] parked id {} ({} voxels)".format(mid, n))
        return n

    def _unpark_marker(self, marker_id: int) -> int:
        """Put one marker back, exactly where it was.

        SAFE WRITE (the same guard interpolation uses): a voxel is
        rewritten only where the ROI is background, the TARGET, or
        already this id -- so a marker that was painted over while this
        one was parked keeps what it has.  The target is in that set
        because it is what parking leaves behind (H1); without it a
        re-tick would find its own footprint "occupied" by the object
        and restore nothing at all.
        """
        mid = int(marker_id)
        stash = getattr(self, "_parked_marker_voxels", None) or {}
        idx = stash.pop(mid, None)
        if idx is None:
            return 0
        if self._refine_layer is None or self._split_roi is None:
            return 0
        roi = self._refine_layer.data[self._split_roi]
        if not len(idx) or idx[0].size == 0:
            print("[MARKERS] restored id {}".format(mid))
            return 0
        target = int(self._split_label_id or 0)
        try:
            current = roi[idx]
            writable = (current == 0) | (current == mid)
            if target:
                writable = writable | (current == target)
            roi[idx] = np.where(writable, mid, current)
            n = int(writable.sum())
        except Exception as exc:  # noqa: BLE001
            # The entry is already popped, so a raise here would lose
            # the voxels silently.  Put it back and say so (M1).
            stash[mid] = idx
            print("[MARKERS] could not restore id {}: {!r}".format(mid, exc))
            return 0
        print("[MARKERS] restored id {} ({} voxels)".format(mid, n))
        return n

    def _unpark_all_markers(self, *, reason="") -> list:
        """Put every parked marker back and empty the stash.

        The escape hatch for "the panel can no longer show a row for
        it": with no row there is no tick that could ever restore it,
        so a marker left lifted off would simply be gone, silently.
        """
        ids = self._parked_marker_ids()
        if not ids:
            return []
        for mid in ids:
            self._unpark_marker(mid)
        try:
            self._refine_layer.refresh()
        except Exception:
            pass
        self._refresh_refine_in_subviewers()
        print("[MARKERS] {} parked marker(s) put back{}: {}".format(
            len(ids), " ({})".format(reason) if reason else "", ids))
        return ids

    def _sync_marker_parking(self) -> tuple:
        """Make the object agree with the ticks.  ONE pass, ONE refresh.

        Called on every marker tick -- including the All / None chips,
        which reach the panel handler once for the whole list, so the
        side views are pushed once rather than once per row.

        Returns ``(parked, unparked)``.
        """
        if not getattr(self, "_split_running", False):
            return [], []
        if self._refine_layer is None or self._split_roi is None:
            return [], []
        if getattr(self, "_split_previewed", False):
            # The ROI holds the cut, not the markers.  The tick handler
            # leaves review before it gets here; anything else that
            # reaches this during review must do nothing at all.
            print("[MARKERS] parking skipped - a preview is on screen")
            return [], []
        off = self._unticked_marker_ids()
        off_set = None if off is None else {int(i) for i in off}
        target = int(self._split_label_id or 0)
        candidates = [int(i) for i in (self._split_painted_marker_ids or [])
                      if int(i) != target]
        stash = getattr(self, "_parked_marker_voxels", None)
        if stash is None:
            stash = self._parked_marker_voxels = {}
        to_park = ([] if off_set is None
                   else [i for i in candidates
                         if i in off_set and i not in stash])
        live = set(candidates)
        # A stash entry whose id has LEFT the marker set is not a marker
        # any more, and writing it back would paint a marker the panel
        # cannot show and the recipe will not send (M1).  Every
        # legitimate remover already pops its own ids, so one here is a
        # leak: drop it, do not restore it.
        stale = [i for i in self._parked_marker_ids() if i not in live]
        for mid in stale:
            stash.pop(mid, None)
        if stale:
            print("[MARKERS] {} parked id(s) are no longer markers and "
                  "were dropped, not restored: {}".format(
                      len(stale), stale))
        to_unpark = [i for i in self._parked_marker_ids()
                     if i in live and (off_set is None or i not in off_set)]
        if not to_park and not to_unpark:
            return [], []
        # PARK FIRST, then unpark.  When one pass does both (All / None
        # after some hand-ticking) the marker that is LEAVING vacates
        # its voxels before the one that is arriving asks for them, so
        # the restore's safe write has somewhere to land.  The other
        # order would silently drop the overlap.
        for mid in to_park:
            self._park_marker(mid)
        for mid in to_unpark:
            self._unpark_marker(mid)
        try:
            self._refine_layer.refresh()
        except Exception:
            pass
        self._refresh_refine_in_subviewers()
        return to_park, to_unpark

    def _apply_marker_unticks(self, roi_array):
        """*roi_array* with every UNTICKED marker id zeroed out.

        Returns the array itself when nothing is unticked, so the
        legacy solver call stays byte-identical in the default case
        ("the legacy lock, restated").

        Since U2b an unticked marker is PARKED -- its voxels are already
        off the layer -- so in the ordinary flow there is nothing left
        for this to zero and the identity return holds for an unticked
        session too.  It stays as the belt-and-braces behind parking:
        if a park was refused for any reason, the solver still must not
        be handed a marker the user unticked.
        """
        keep = self._checked_marker_ids()
        if keep is None:
            return roi_array
        parked = set(getattr(self, "_parked_marker_voxels", None) or {})
        painted = [int(i) for i in (self._split_painted_marker_ids or [])
                   if int(i) not in parked]
        drop = [i for i in painted if i not in set(int(k) for k in keep)]
        if not drop:
            return roi_array
        out = np.asarray(roi_array).copy()
        out[np.isin(out, drop)] = 0
        print("[MARKERS] {} unticked marker(s) left out of the cut: {}"
              .format(len(drop), drop))
        return out

    # ---------------------------------------------------------------
    # Auto-detect seeds
    # ---------------------------------------------------------------

    def _detect_seeds(self) -> None:
        """Auto-detect seed peaks via EDT local maxima.

        Guard: only valid during split, before Preview, with refine layer.
        """
        if not self._split_running:
            return
        # THE LOOP: Detect after a preview puts the preview away first,
        # and `_split_previewed` is down before the guard below reads it
        # -- setting it late is what made Detect a silent no-op (A-H2).
        self._leave_reshape_review(
            "New markers - press Preview to cut again.")
        if self._split_previewed:
            return
        if self._refine_layer is None or self._refine_baseline is None:
            return

        from packages.epicycle_annotator.seed_detection import detect_seed_peaks

        # Read min_distance from the UI.
        adv = self._get_advanced_features_widget()
        # A mask seed source IGNORES whatever Detect finds -- `_reshape_recipe`
        # reads the guide instead -- and running it here would also hand
        # `_apply_seed_threshold` the chance to DISABLE the Preview button
        # this guided session just enabled.
        if (
            adv is not None
            and getattr(adv, "guided_needs_guide_mask", None) is not None
            and adv.guided_needs_guide_mask()
        ):
            self._update_split_ui(
                status_text=(
                    "The markers come from the mask you chose, so Detect "
                    "is not used. Press Preview."
                ),
            )
            return
        min_distance = 5
        if adv is not None and hasattr(adv, "split_detect_min_dist_sb"):
            min_distance = adv.split_detect_min_dist_sb.value()

        peaks, _markers = detect_seed_peaks(
            self._refine_baseline, self._split_label_id,
            min_distance=min_distance,
        )

        if not peaks:
            self._update_split_ui(
                status_text="No EDT peaks found in this label.",
                detect_seed_count=0,
            )
            return

        self._detected_peaks = peaks
        self._auto_detect_active = True

        # Cache max label ID once — reused by _apply_seed_threshold on every
        # drag step instead of scanning the full volume each time.
        labels_name = self._active_labels_name
        shared_data = self.sync_manager.shared_labels.get(labels_name)
        max_existing = 0
        if shared_data is not None:
            max_existing = max(max_existing, int(np.max(shared_data)))
        if self._refine_baseline is not None:
            max_existing = max(max_existing, int(np.max(self._refine_baseline)))
        self._seed_id_base = max_existing + 1

        # Default threshold: 25th percentile of EDT values (keeps most peaks).
        edt_vals = [p["edt_value"] for p in peaks]
        default_thresh = float(np.percentile(edt_vals, 25))

        # Build histogram data — read bin count from UI.
        n_bins = 15
        if adv is not None and hasattr(adv, "split_detect_bins_sb"):
            n_bins = adv.split_detect_bins_sb.value()
        hist_data = {"edt_values": edt_vals, "n_bins": n_bins}

        # Apply the threshold to generate seeds.
        self._apply_seed_threshold(default_thresh)

        # Update histogram UI.
        self._update_split_ui(
            detect_visible=True,
            detect_histogram_data=hist_data,
            detect_threshold=default_thresh,
        )

    def _apply_seed_threshold(self, edt_threshold: float) -> None:
        """Apply threshold to detected peaks and write seeds to refine layer.

        Resets the refine layer ROI to baseline, then paints seed spheres
        for all peaks above the threshold.
        """
        if self._detected_peaks is None or self._refine_layer is None:
            return
        if self._refine_baseline is None or self._split_roi is None:
            return

        from packages.epicycle_annotator.seed_detection import (
            peaks_to_seed_volume,
            threshold_peaks,
        )

        filtered = threshold_peaks(self._detected_peaks, edt_threshold)

        # Use cached base ID (computed once in _detect_seeds, not on every drag).
        base_id = self._seed_id_base

        # Remap seed_ids to avoid collision with existing labels.
        remapped = []
        for i, p in enumerate(filtered):
            remapped.append({
                "coord": p["coord"],
                "edt_value": p["edt_value"],
                "seed_id": base_id + i,
            })

        # Generate seed volume (in ROI coordinates).
        roi_shape = self._refine_baseline.shape
        seed_vol = peaks_to_seed_volume(roi_shape, remapped, sphere_radius=2)

        # Reset refine layer ROI to baseline.
        self._refine_layer.data[self._split_roi] = self._refine_baseline.copy()

        # Write seeds where (seed_vol > 0) & (baseline == label_id).
        roi_data = self._refine_layer.data[self._split_roi]
        mask = (seed_vol > 0) & (self._refine_baseline == self._split_label_id)
        roi_data[mask] = seed_vol[mask]

        # Update tracked marker IDs.  Detect REPLACES the painted set --
        # it has just reset the ROI to baseline, so nothing the user
        # painted survived it either (this is the one place that is
        # allowed to, and it is why source 1 never goes through here:
        # see `_materialise_found_markers`).
        self._split_painted_marker_ids = [p["seed_id"] for p in remapped]
        self._split_seed_ids = list(self._split_painted_marker_ids)
        # ...and the parked stash goes with them.  The ROI
        # was just reset to baseline, so every stashed coordinate now
        # describes a layer that no longer exists -- and `_seed_id_base`
        # is cached for the session, so a second drag MINTS THE SAME
        # IDS.  Left in place, a marker parked before the drag would be
        # repainted by this method, keep its OFF tick through
        # `preserve=True`, and end up drawn and unticked at once.
        # `_sync_marker_parking()` below re-lifts whatever is still
        # unticked, from the voxels this drag actually wrote.
        self._parked_marker_voxels = {}

        # Refresh subviewers only (2D slices where seeds are visible).
        # Skip self._refine_layer.refresh() — it triggers a full 3D
        # iso_categorical re-render on the full-volume Refine layer,
        # which is extremely expensive during interactive threshold drag.
        self._refresh_refine_in_subviewers()

        n_seeds = len(remapped)
        self._update_split_ui(
            seed_count=n_seeds,
            seed_ids=list(self._split_painted_marker_ids),
            active_seed_id=(
                self._split_painted_marker_ids[-1]
                if self._split_painted_marker_ids else None
            ),
            detect_seed_count=n_seeds,
            start_enabled=False,
            next_seed_enabled=True,
            undo_seed_enabled=False,
            refine_seeds_enabled=False,
            # A guided session needs no painted markers, so a thin detect
            # result must not take away the Preview button `_start_split`
            # enabled for it.
            preview_enabled=(
                n_seeds >= 2 or bool(getattr(self, "_reshape_active", False))
            ),
            accept_enabled=False,
            reject_enabled=True,
            reject_text="Discard",
            review_active=False,
            status_text=(
                f"Found {n_seeds} markers from the shape. "
                f"Drag the threshold to adjust, or Add marker by hand."
            ),
        )
        # The list has just been repopulated, and `preserve=True` keeps
        # an OFF tick by id -- so anything still unticked has to be
        # lifted off again, from the voxels THIS drag wrote.
        # Costs nothing on the ordinary drag, where every row is ticked
        # and the sync finds nothing to do.
        self._sync_marker_parking()

    def _on_seed_threshold_changed(self, value: float) -> None:
        """Handle user dragging the threshold line in the histogram.

        Throttled to max ~20 updates/sec (50ms) to keep the UI responsive
        during fast drags.
        """
        if not self._auto_detect_active or self._split_previewed:
            return
        import time as _time
        now = _time.perf_counter()
        if now - getattr(self, "_last_threshold_time", 0) < 0.05:
            return
        self._last_threshold_time = now
        self._apply_seed_threshold(value)

    def _preview_split(self) -> None:
        """Run seeded watershed on the Refine layer and show the result."""
        if not self._split_running or self._refine_baseline is None:
            return
        # Reshape R1: a guided session takes the threaded engine path.
        # While `_reshape_active` is False NOTHING below changes.
        #
        # Belt and braces (2026-09-18): the pickers arm this as they move,
        # but the callback that does it is wired through the panel, and a
        # panel rebuilt mid-session would not have it.  The routing
        # decision is the LIVE predicate, not a flag latched at Start.
        self._arm_guided_if_needed()
        if getattr(self, "_reshape_active", False) and self._is_guided_now():
            self._preview_reshape()
            return
        if self._refine_layer is None:
            self._update_split_ui(status_text="Refine layer missing.")
            return

        from packages.epicycle_annotator.split import (
            apply_split_to_volume,
            seeded_watershed_split,
            validate_split_seeds,
        )

        refine_data = self._refine_layer.data
        current_roi = refine_data[self._split_roi]

        # The manual path is SHAPE-ONLY.  It used to read
        # `split_method_combo` / `split_image_combo` and flood a Sobel
        # elevation instead; that branch became unreachable in Brick 2a,
        # when `is_guided_split()` began routing every wall index past
        # "shape only" into the engine, and is removed here rather than
        # left as a second, divergent membrane implementation.
        method = "mask"
        image_roi = None

        # Save the MARKERS before anything else.  This
        # is what `_leave_reshape_review` snaps back to, so it has to be
        # the full painted set -- including the markers that are about
        # to be unticked out of the solver's input below.
        #
        # ...but ONLY when there is no preview on screen.  "Preview
        # again" pressed straight out of review reaches here with the
        # ROI holding PIECES, and re-taking the snapshot then made the
        # snap-back paint the previous cut back onto the object as if it
        # were the markers -- and `_marker_rows_from_refine`, whose own
        # rule is that it must never run during review, would list those
        # pieces as freshly hand-painted markers.  The snapshot from the
        # FIRST Preview is still the right one: it holds the markers,
        # and nothing has been able to change them without going through
        # the snap-back, which clears this flag.
        if not self._review_is_on_screen():
            self._refine_pre_preview = current_roi.copy()
            # Anything painted with a brush the panel never saw joins
            # the list now, before the ticks are read off it.
            self._marker_rows_from_refine()
        else:
            print("[RESHAPE-LOOP] preview again - keeping the marker "
                  "snapshot (the ROI holds pieces right now)")

        # The ticks decide what the solver starts from (the
        # recipe sends the marker set the list shows).  With nothing
        # unticked -- the default -- `solve_roi` is bit-identical to
        # `current_roi` and the legacy call is byte-for-byte what it
        # always was.
        solve_roi = self._apply_marker_unticks(current_roi)

        valid, seed_ids, msg = validate_split_seeds(
            self._refine_baseline, solve_roi, self._split_label_id
        )
        if not valid:
            self._update_split_ui(
                start_enabled=False,
                next_seed_enabled=True,
                preview_enabled=True,
                accept_enabled=False,
                reject_enabled=True,
                status_text=msg,
            )
            return

        result, new_ids, msg = seeded_watershed_split(
            self._refine_baseline,
            solve_roi,
            self._split_label_id,
            image_roi=image_roi,
            method=method,
        )

        # Apply result to the Refine layer (NOT shared_data).
        n_modified = apply_split_to_volume(
            refine_data, self._split_roi, result, self._split_label_id
        )

        # Auto-smooth preview: if auto-smooth is enabled, smooth the split
        # fragments now so the preview shows the final result.
        smooth_msg = ""
        adv = self._get_advanced_features_widget()
        if (
            adv is not None
            and getattr(adv, "split_auto_smooth_cb", None) is not None
            and adv.split_auto_smooth_cb.isChecked()
        ):
            sigma_z = adv.smooth_sigma_z_sb.value() if hasattr(adv, "smooth_sigma_z_sb") else 3.0
            sigma_xy = adv.smooth_sigma_xy_sb.value() if hasattr(adv, "smooth_sigma_xy_sb") else 1.5
            from packages.epicycle_annotator.smooth import smooth_split_fragments
            n_smooth, _smsg = smooth_split_fragments(
                refine_data, self._split_roi, new_ids,
                sigma_z=sigma_z, sigma_xy=sigma_xy,
            )
            if n_smooth > 0:
                smooth_msg = f" Smoothed {n_smooth} voxels."

        from napari_epicycle.annotator.advanced_features_widget import (
            WALLS_TOGGLE_MANUAL as _WALLS_TOGGLE_MANUAL,
        )

        # Refresh refine layer + subviewers.
        self._refine_layer.refresh()
        self._refresh_refine_in_subviewers()
        self._split_previewed = True
        # Defensive (audit L3).  Accept routes on `_reshape_result`, and
        # from here on the Refine layer holds a WATERSHED partition -- so
        # a result left over from an earlier guided preview in the same
        # session would send this one down the engine's commit with an
        # array that does not describe what is on screen.  Nothing
        # reaches here with one today (`_leave_reshape_review` nulls it
        # on every S3 change, and the pickers route guided previews
        # away); the cost of the guarantee is one assignment.
        self._reshape_result = None
        # What the watershed ACTUALLY put in the volume is what Accept
        # has to report to the controller.  It used to be the same list
        # as the painted markers, because every painted marker got a
        # piece; unticking one breaks that, and Accept
        # would name a label that is not there.  `_split_seed_ids` holds
        # the PIECES from here until `_leave_reshape_review` puts the
        # markers back -- the same split of duties the guided path has.
        self._split_seed_ids = list(new_ids)

        self._update_split_ui(
            start_enabled=False,
            # THE LOOP: the marker work stays live and
            # Preview stays pressable in review.  Each of these puts the
            # preview away and brings the markers back before it acts.
            next_seed_enabled=True,
            undo_seed_enabled=(len(self._split_painted_marker_ids) >= 2),
            refine_seeds_enabled=True,
            preview_enabled=True,
            accept_enabled=True,
            reject_enabled=True,
            reject_text="Discard",
            review_active=True,
            refine_further_enabled=True,
            # Brick U3 (fix pass H4).  This path never builds an engine
            # result, so there is no elevation and the tick stays grey --
            # but it was reached by pressing Preview, so the default
            # "Preview first - there is nothing to show yet." is the one
            # sentence that cannot be true here.  Without a reason pushed
            # from HERE the grey tick keeps whatever the last state left
            # on it, which is exactly that.
            walls_toggle_enabled=False,
            walls_toggle_reason=_WALLS_TOGGLE_MANUAL,
            status_text=(
                f"Object {self._split_label_id} - {len(new_ids)} pieces "
                f"({n_modified} voxels).{smooth_msg} Scroll up to change "
                f"the markers and Preview again."
            ),
        )

    def _accept_split(self) -> None:
        """Accept the split: merge Refine data back to shared volume.

        The 3D iso_categorical re-render is deferred to after the controller
        callback and navigator update so that the isolation colormap is
        correctly applied before the first (and only) expensive render.
        """
        # AUDIT 2026-09-09: the guided lock comes off on EVERY exit,
        # including the five bare returns below (load-locked, no
        # label, not previewed, fragment-refine, no result).  Holding
        # it past a return leaves Step 3 locked with the cell either
        # written or restored and nothing left to press.  Idempotent,
        # so a legacy accept pays nothing.
        #
        # Bug A.  The marker ticks have to be cleared
        # after a commit -- `clear_split_guided_ui` was called from
        # Discard and from a patch switch and from NOWHERE ELSE, so the
        # next object's Preview was cut with the previous object's
        # marker ids and refused with "No markers found in this object".
        # It has to be GATED on a commit actually happening: the
        # `_load_locked` refusal below changes nothing and its whole
        # contract is "re-click after unlock works", so an unconditional
        # clear in the `finally` would destroy the user's ticks on a
        # refusal.  And it has to be in the `finally`, not at the top,
        # because `_write_reshape_result` reads the piece ticks.
        committed = False
        try:
            # B2: refuse while load-locked (e.g. async Save in flight).  Guard at the
            # very top so the internal _accept_smooth() chain (fragment-refine commit)
            # is also blocked.  Op state is preserved, so re-click after unlock works.
            if getattr(self, "_load_locked", False):
                print("[REFINE-ACCEPT] split accept refused — controller busy (load-locked)")
                return
            # The markers belong to the cut that is being committed; they
            # are not part of the result (one of the five).
            self._remove_temp_layer("markers")
            # ...and so does the wall picture.
            self._drop_walls_picture()
            # --- Fragment Refine Accept path ---
            if self._split_fragment_refine:
                # Fragments are already in shared_data. Accept any pending smooth.
                if self._smooth_running:
                    self._accept_smooth()

                original_id = self._split_label_id
                new_ids = list(self._split_fragment_ids)
                # The fragments are already in shared_data; this path IS
                # the commit (Bug A's gate).
                committed = True
                restore_isolate = self._split_isolate_was_on
                # Capture undo data before state reset.
                undo_roi = self._split_roi
                undo_snap = self._split_snapshot

                # U4 (2.7 B2).  A GUIDED cut taken through fragment
                # refine used to be reported as a MANUAL split, so it
                # lost the whole of the engine's bookkeeping: the
                # `reshape_flag` sidecar, the `superseded=False` an
                # unticked piece earns, the piece stats and the
                # guide/preset provenance.  The record is built HERE,
                # before `_reset_split_state` nulls the result it reads.
                # The volume is NOT re-written -- fragment entry wrote
                # it and the user has been editing those voxels since.
                _frag_commit = None
                _frag_ticked = getattr(self, "_reshape_fragment_ticked", None)
                if (_frag_ticked is not None
                        and getattr(self, "_reshape_result", None) is not None):
                    _frag_commit = self._reshape_commit_record(
                        original_id, ticked=_frag_ticked,
                        written=int(
                            getattr(self, "_reshape_fragment_written", 0)),
                        t0=_time.perf_counter(),
                    )
                if _frag_commit is not None:
                    # M1.  The record's `piece_stats` came out of the
                    # ENGINE, which measured the cut BEFORE the user
                    # shaped it -- and fragment refine exists precisely
                    # so they can shape it.  The `_accept_smooth()` at
                    # the top of this branch has already moved those
                    # voxels, so the df would get an area and a centroid
                    # for a shape that is not in the volume any more.
                    # The legacy split callback always re-measured from
                    # the live volume (`_on_annotator_label_split`); so
                    # does this one, in the same VOLUME coordinates the
                    # engine reports (ROI offset applied), because the
                    # controller scales them by the downsample factor
                    # and stores them as canonical full-res coords.
                    _frag_commit["piece_stats"] = self._live_piece_stats(
                        _frag_commit["new_ids"], undo_roi)
                    if sorted(_frag_commit["new_ids"]) != sorted(new_ids):
                        # They agree by construction -- the fragment list
                        # IS the ticked set read back off the write --
                        # so a disagreement is worth a line rather than a
                        # silent choice.
                        print("[FRAG] fragment list {} and commit record {} "
                              "disagree - trusting the record".format(
                                  sorted(new_ids),
                                  sorted(_frag_commit["new_ids"])))
                    new_ids = list(_frag_commit["new_ids"])

                # Clean up split state.
                self._reset_split_state()

                self._update_split_ui(
                    seed_count=0,
                    seed_ids=[],
                    split_active=False,
                    edt_enabled=False,
                    start_enabled=True,
                    next_seed_enabled=False,
                    undo_seed_enabled=False,
                    refine_seeds_enabled=False,
                    preview_enabled=False,
                    accept_enabled=False,
                    reject_enabled=False,
                    reject_text="Cancel",
                    review_active=False,
                    refine_further_enabled=False,
                    fragment_refine_active=False,
                    detect_enabled=False,
                    detect_visible=False,
                    status_text=(
                        f"Pieces accepted. Object {original_id} → {new_ids}."
                    ),
                )

                # Fire callback to controller -- the RESHAPE one when the
                # pieces came from the engine (U4, 2.7 B2), the split one
                # when they came from the hand-painted watershed.
                if _frag_commit is not None:
                    print("[FRAG] accept all - guided commit, {} "
                          "fragment(s)".format(len(new_ids)))
                    rcb = getattr(self, "_on_reshape_committed_cb", None)
                    if rcb is not None:
                        try:
                            rcb(
                                original_id, new_ids,
                                roi_slices=undo_roi, old_roi_data=undo_snap,
                                flags=_frag_commit["flags"],
                                superseded=_frag_commit["superseded"],
                                piece_stats=_frag_commit["piece_stats"],
                                guide_run=_frag_commit["guide_run"],
                                preset=_frag_commit["preset"],
                            )
                        except Exception as exc:  # noqa: BLE001
                            print("[RESHAPE] commit callback raised: "
                                  "{!r}".format(exc))
                else:
                    print("[FRAG] accept all - legacy split commit, {} "
                          "fragment(s)".format(len(new_ids)))
                    cb = getattr(self, "_on_label_split_cb", None)
                    if cb is not None:
                        cb(original_id, new_ids,
                           roi_slices=undo_roi, old_roi_data=undo_snap)

                self._clear_draw_snapshot()
                self._restore_ready_to_draw_state()

                # Auto-advance.
                nav = getattr(self, "_navigator_widget", None)
                if nav is not None:
                    try:
                        if restore_isolate:
                            nav._isolate_cb.blockSignals(True)
                            nav._isolate_cb.setChecked(True)
                            nav._isolate_cb.blockSignals(False)
                        if self._is_auto_advance_enabled():
                            if not nav.select_next_triage():
                                self._on_isolation_changed(set())
                    except Exception:
                        pass
                return

            # --- Normal split Accept path ---
            if self._split_label_id is None or self._refine_layer is None:
                return
            if not self._split_previewed:
                self._update_split_ui(status_text="Run Preview before Accept.")
                return

            original_id = self._split_label_id
            new_ids = list(self._split_seed_ids)
            restore_isolate = self._split_isolate_was_on
            restore_visible = self._split_main_visible
            labels_name = self._active_labels_name
            # Capture undo data before state reset clears them.
            undo_roi = self._split_roi
            undo_snap = self._split_snapshot

            # --- Instant UI feedback ---
            self._update_split_ui(
                status_text="Applying split...",
                accept_enabled=False,
                reject_enabled=False,
            )
            from qtpy.QtWidgets import QApplication
            QApplication.processEvents()

            # NOTE: Auto-smooth runs during Preview (not Accept) so the user
            # sees the smoothed result before committing.  No need to re-smooth here.
            adv = self._get_advanced_features_widget()

            # 1. Copy refine data back into the shared volume (safe-write
            #    into the hole we created — only where shared_data == 0).
            shared_data = self.sync_manager.shared_labels.get(labels_name)
            _reshape_commit = None
            # Route on what the LAST PREVIEW produced, not on a live flag
            # (2026-09-19).  `_preview_split` routes on `_reshape_active
            # AND _is_guided_now()`; this used to route on
            # `_reshape_active` alone, and the two disagree the moment the
            # flag is armed AFTER a legacy preview -- which
            # `_arm_guided_if_needed` does on any picker move, and which
            # the walls combo does to itself when shape-thickness markers
            # force it back to "shape only".  Accept then asked for an
            # engine result that was never computed, got None, restored
            # the cell to its pre-Start shape and said "there was nothing
            # to apply": a legitimate preview, accepted, silently
            # reverted.  `_reshape_result` IS what the last preview left,
            # so it is what decides.
            if (getattr(self, "_reshape_active", False)
                    and getattr(self, "_reshape_result", None) is not None):
                # Reshape R1: the SAME safe-write, expressed in the engine's
                # explicit `allowed_ids` form -- `(volume == 0) | (volume ==
                # target)` -- so it does not silently depend on the
                # hole-punch that precedes it.  Returns the
                # bookkeeping the controller callback needs.
                _reshape_commit = self._write_reshape_result(
                    shared_data, original_id
                )
                if _reshape_commit is None:
                    # Nothing to write and nothing to record.  Put the cell back
                    # AND end the session: this path leaves Accept and Reject
                    # disabled (they were just greyed for "Applying split..."),
                    # so returning with the lock still on would strand the whole
                    # of Step 3 with no button left to press (AUDIT 2026-09-09).
                    self._restore_reshape_target("accept found no result")
                    self._teardown_reshape_for_switch()
                    self._update_split_ui(
                        split_active=False, start_enabled=True,
                        accept_enabled=False, reject_enabled=False,
                        status_text=(
                            "There was nothing to apply - the object is back as it "
                            "was."
                        ),
                    )
                    return
                new_ids = list(_reshape_commit["new_ids"])
                committed = True
            elif shared_data is not None and self._split_roi is not None:
                refine_roi = self._refine_layer.data[self._split_roi]
                target_area = shared_data[self._split_roi]
                write_mask = (refine_roi > 0) & (target_area == 0)
                target_area[write_mask] = refine_roi[write_mask]
                committed = True

            # 2. Remove EDT overlay + Refine layer (fast layer cleanup).
            #    Process Qt events after removal so vispy completes texture
            #    (processEvents runs inside each removal method automatically.)
            self._remove_edt_overlay()
            self._remove_refine_layer()

            # 3. Prepare main layer (unlock) + sync views.
            main_layer = self._layer_by_name(self.viewer.layers, labels_name)
            if isinstance(main_layer, Labels):
                main_layer.editable = True

            # 4. Sync all views (main layer data copy + subviewer refresh).
            #    When auto-advance will immediately isolate the next label
            #    (creating a new proxy), skip the expensive main_layer.refresh()
            #    since the main layer stays hidden while the proxy renders.
            #
            #    FM4 fix (2026-05-16): the original check only covered
            #    single-label proxy mode (`_isolation_proxy_active`).  When
            #    the user is in MULTI-label colormap mode (e.g., lasso
            #    selection, merge dialog leaving residual multi-iso), the
            #    main layer is also kept hidden by the colormap, so a
            #    refresh causes the same isosurface flicker.  Include the
            #    multi-label cache in the skip predicate.
            _single_iso = bool(getattr(self, "_isolation_proxy_active", False))
            _multi_iso = getattr(self, "_isolation_label_set", None) is not None
            _skip_3d = (
                self._is_auto_advance_enabled()
                and restore_isolate
                and (_single_iso or _multi_iso)
            )
            print(
                f"[SPLIT-ACCEPT] skip_3d_refresh decision: "
                f"auto_advance={self._is_auto_advance_enabled()} "
                f"restore_isolate={restore_isolate} "
                f"single_iso={_single_iso} multi_iso={_multi_iso} "
                f"-> skip={_skip_3d}"
            )
            self._sync_all_views_after_operation(
                roi=self._split_roi, skip_3d_refresh=_skip_3d
            )

            # 5. Set selected_label to first split result.
            if new_ids:
                result_label = new_ids[0]
            elif shared_data is not None:
                result_label = int(np.max(shared_data)) + 1
            else:
                result_label = max(1, original_id + 1)
            self._set_selected_label_all(result_label)

            # 6. Reset state (before callbacks so navigator methods see clean state).
            self._reset_split_state()

            self._update_split_ui(
                seed_count=0,
                seed_ids=[],
                split_active=False,
                edt_enabled=False,
                start_enabled=True,
                next_seed_enabled=False,
                undo_seed_enabled=False,
                refine_seeds_enabled=False,
                preview_enabled=False,
                accept_enabled=False,
                reject_enabled=False,
                reject_text="Cancel",
                review_active=False,
                refine_further_enabled=False,
                fragment_refine_active=False,
                detect_enabled=False,
                detect_visible=False,
                status_text=f"Split accepted. Label {original_id} → {new_ids}.",
            )

            # 7. Fire callback to controller (rebuilds navigator table with
            #    new split labels, updates QC DataFrame).
            if _reshape_commit is not None:
                # Reshape R1: its OWN callback, because a guided commit can
                # leave the original id ALIVE (an unticked piece went back to
                # the cell, or the arbiter kept it whole) -- the split
                # callback always supersedes, which would be a lie.
                rcb = getattr(self, "_on_reshape_committed_cb", None)
                if rcb is not None:
                    try:
                        rcb(
                            original_id, new_ids,
                            roi_slices=undo_roi, old_roi_data=undo_snap,
                            flags=_reshape_commit["flags"],
                            superseded=_reshape_commit["superseded"],
                            piece_stats=_reshape_commit["piece_stats"],
                            guide_run=_reshape_commit["guide_run"],
                            preset=_reshape_commit["preset"],
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(
                            "[RESHAPE] commit callback raised: {!r}".format(exc)
                        )
            else:
                cb = getattr(self, "_on_label_split_cb", None)
                if cb is not None:
                    cb(original_id, new_ids,
                       roi_slices=undo_roi, old_roi_data=undo_snap)

            self._clear_draw_snapshot()
            self._restore_ready_to_draw_state()

            # 8. Post-split auto-advance: restore isolate checkbox state and
            #    advance to the next triage label (review/bad/garbage).
            nav = getattr(self, "_navigator_widget", None)
            if nav is not None:
                try:
                    if restore_isolate:
                        nav._isolate_cb.blockSignals(True)
                        nav._isolate_cb.setChecked(True)
                        nav._isolate_cb.blockSignals(False)
                    if self._is_auto_advance_enabled():
                        if not nav.select_next_triage():
                            # No triage labels remain — clear isolation.
                            self._on_isolation_changed(set())
                except Exception:
                    pass

            # 9. NOW make the main layer visible — deferred to this point so
            #    the isolation colormap from step 8 is already applied.  This
            #    is the single expensive 3D iso_categorical render (~2-3s).
            if isinstance(main_layer, Labels) and restore_visible:
                main_layer.visible = True
                _heal_render(self, main_layer, tag="split-accept")

            # 10. Reshape R1: release the Step-3 lock LAST, once the target is
            #     written and the surface is whole again.  Idempotent, so the
            #     legacy path's call is a no-op.
            self._fire_reshape_lock(False)
            if _reshape_commit is not None:
                print(
                    "[RESHAPE-PERF] accept wall={:.0f}ms written={} ids={}"
                    .format(
                        (_time.perf_counter() - _reshape_commit["t0"]) * 1000.0,
                        _reshape_commit["written"], new_ids,
                    )
                )

        # ====================================================================== #
        # Reshape R1 -- guided Split
        #
        # The whole mode rides the EXISTING split lifecycle: `_start_split`
        # punches the hole and builds the Refine layer, and the Refine layer IS
        # the mirrored preview (it is already pushed into all three subviewers by
        # `_refresh_refine_in_subviewers`, and it is already Priority 1 in
        # `_get_toggle_targets_for_name`).  So this brick adds NO new managed
        # layer, NO new toggle-hook branch and NO isolation change -- it changes
        # only WHERE THE SEEDS COME FROM and WHAT ELEVATION IS FLOODED, which is
        # exactly the R1 scope the design names.
        #
        # The one genuinely new layer is the GUIDE: unmanaged, read-only, outside
        # `_active_labels_names`, outside `shared_labels`, outside the mirror set,
        # swept by name prefix on patch switch and step exit.
        #
        # Diagnostics are `[RESHAPE]` / `[RESHAPE-PERF]`, ASCII only (cp1252
        # console).  STRIP BEFORE THE REAL-BASELINE MERGE, with the other Step 3
        # dev diagnostics.
        # ====================================================================== #

        finally:
            self._fire_reshape_lock(False)
            if committed:
                # Bug A.  The markers belonged to the object that
                # has just been written; the next one must start clean,
                # or its Preview is filtered by ids it has never had.
                adv = self._get_advanced_features_widget()
                if adv is not None:
                    for _meth in ("clear_split_guided_ui",
                                  "clear_expected_count"):
                        _fn = getattr(adv, _meth, None)
                        if _fn is None:
                            continue
                        try:
                            _fn()
                        except Exception:
                            pass
                self._split_painted_marker_ids = []
                self._reshape_found_seeds = None
                self._reshape_found_marker_ids = []
                self._reshape_found_marker_stats = {}
                self._refine_pre_preview = None
                print("[MARKERS] Accept committed - marker ticks, the "
                      "marker layer and the expected-count tick cleared")
            else:
                print("[MARKERS] Accept did not commit - the markers are "
                      "kept exactly as they were")

    def _fire_reshape_lock(self, active: bool) -> None:
        """Set / clear the controller's `_reshape_running`.  Idempotent.

        Idempotence matters: the release is called from the accept tail, the
        reject tail, the worker-launch failure path and the teardown, and any
        of those can run twice.  Firing False twice must not look like a
        second op resolving.
        """
        active = bool(active)
        if active == bool(getattr(self, "_reshape_lock_fired", False)):
            return
        self._reshape_lock_fired = active
        cb = getattr(self, "_reshape_lock_cb", None)
        if cb is None:
            return
        try:
            cb(active)
        except Exception as exc:  # noqa: BLE001 - never strand the surface
            print("[RESHAPE] lock callback raised: {!r}".format(exc))

    def _is_guided_now(self) -> bool:
        """What the pickers say RIGHT NOW.

        `_reshape_active` records that this session armed the engine (and
        took the Step-3 lock); it deliberately stays True if the user
        switches back to a legacy source, because the lock must not come
        off while the cell is punched out.  Which PATH a Preview takes is
        a different question, and this is it.
        """
        adv = self._get_advanced_features_widget()
        if adv is None or getattr(adv, "is_guided_split", None) is None:
            return False
        try:
            return bool(adv.is_guided_split())
        except Exception:
            return False

    def _reshape_context(self) -> dict:
        """Workspace / patch / run / spacing, from the controller.

        This widget owns none of it.  A standalone annotator (no controller)
        gets safe defaults and the guided modes that need a workspace simply
        find no runs to offer.
        """
        cb = getattr(self, "_reshape_context_cb", None)
        ctx = {}
        if cb is not None:
            try:
                ctx = dict(cb() or {})
            except Exception as exc:  # noqa: BLE001
                print("[RESHAPE] context callback raised: {!r}".format(exc))
                ctx = {}
        ctx.setdefault("workspace", "")
        ctx.setdefault("patch_name", "")
        ctx.setdefault("run_id", None)
        ctx.setdefault("run_epoch", 0)
        ctx.setdefault("downsample_factor", 1)
        ctx.setdefault("spacing", (1.0, 1.0, 1.0))
        return ctx

    def _reshape_ready_text(self, label_id) -> str:
        adv = self._get_advanced_features_widget()
        needs = bool(
            adv is not None
            and getattr(adv, "guided_needs_guide_mask", None) is not None
            and adv.guided_needs_guide_mask()
        )
        if needs and self._reshape_guide is None:
            return (
                "Object {} is ready. Load a mask to take the markers from, "
                "then press Preview.".format(label_id)
            )
        return (
            "Object {} is ready. Press Preview - the window stays alive while "
            "it thinks.".format(label_id)
        )

    # ------------------------------------------------------------- inputs --
    def _refresh_reshape_inputs(self) -> None:
        """Fill the wall-channel combo and the guide-source picker."""
        adv = self._get_advanced_features_widget()
        if adv is None:
            return
        if hasattr(adv, "set_wall_channels"):
            names = []
            for layer in self.viewer.layers:
                if getattr(layer, "_epicycle_temporary", False):
                    continue
                if isinstance(layer, Image):
                    names.append(layer.name)
            try:
                adv.set_wall_channels(names)
            except Exception:
                pass
        # The MARKER channel is the fifth combo and follows the same
        # policy; it is refreshed here (tool start / picker change) AND
        # from `_refresh_all_image_combos` (layer insert / remove / tab).
        try:
            self._refresh_seed_channels()
        except Exception:
            pass
        if hasattr(adv, "set_guide_sources"):
            try:
                from napari_epicycle.annotator import reshape_guide as RG

                ctx = self._reshape_context()
                sources = RG.list_guide_sources(
                    ctx["workspace"], ctx["patch_name"],
                    exclude_run_id=ctx.get("run_id"),
                ) if ctx["workspace"] and ctx["patch_name"] else []
                adv.set_guide_sources(sources)
            except Exception as exc:  # noqa: BLE001
                print("[RESHAPE] guide source scan failed: {!r}".format(exc))
                try:
                    adv.set_guide_sources([])
                except Exception:
                    pass

    def _arm_guided_if_needed(self) -> bool:
        """Turn a running LEGACY session into a guided one, if the user
        has just asked for something the legacy path cannot do.

        `_start_split` used to decide this ONCE and latch it, which got
        the real flow backwards: the user presses Start, THEN opens
        Markers and picks "Find them in an image channel", THEN presses
        Preview.  The flag was still False at that point, so Preview ran
        the painted-seed path and told them to "paint more seeds" about a
        session that was never going to have any (found by a real-button
        shots run, 2026-09-18).

        Arming late is safe in the direction that matters: the cell is
        already punched out of the volume, so putting the Step-3 lock ON
        can only ADD protection -- Save and Commit Triage stop being
        available over a volume with a hole in it.  Idempotent.
        """
        if not getattr(self, "_split_running", False):
            return False
        if getattr(self, "_reshape_active", False):
            return False
        adv = self._get_advanced_features_widget()
        if adv is None or getattr(adv, "is_guided_split", None) is None:
            return False
        try:
            if not adv.is_guided_split():
                return False
        except Exception:
            return False
        self._reshape_active = True
        self._fire_reshape_lock(True)
        try:
            self._refresh_reshape_inputs()
        except Exception:
            pass
        # ...and light PREVIEW up, exactly as `_start_split`'s guided
        # branch does.  A found source has nothing to paint, so it never
        # reaches the ">= 2 painted markers" rule that gates Preview on
        # the legacy path.  Arming late did everything else -- the flag,
        # the Step-3 lock, the inputs -- and not this, so in the order
        # the user actually works (Start, THEN pick "Find them in an
        # image channel") the button stayed GREY under a line telling
        # them to paint markers the source finds for itself.  Same class
        # as the routing bug this helper exists to fix: work that only
        # the Start-time branch did (live driver U1-01e, 2026-09-19).
        self._update_split_ui(
            preview_enabled=True,
            status_text=self._reshape_ready_text(self._split_label_id),
        )
        print("[RESHAPE] guided split armed after Start")
        return True

    def _on_reshape_source_changed(self, _index) -> None:
        """Either picker moved -- refresh what the new choice needs.

        And, if the new choice needs the ENGINE, arm the session for it
        now rather than leaving Preview to run the legacy path.
        """
        self._arm_guided_if_needed()
        self._refresh_reshape_inputs()
        adv = self._get_advanced_features_widget()
        if adv is not None and hasattr(adv, "update_split_guided_ui"):
            adv.update_split_guided_ui(guide_note="", engine_messages=[])

    # -------------------------------------------------------- guide layer --
    def _on_reshape_guide_browse(self) -> None:
        from qtpy.QtWidgets import QFileDialog

        path, _filter = QFileDialog.getOpenFileName(
            self, "Pick a mask to take the markers from", "",
            "Label masks (*.tif *.tiff);;All files (*)",
        )
        if not path:
            return
        self._reshape_guide_path = str(path)
        adv = self._get_advanced_features_widget()
        if adv is not None:
            adv.update_split_guided_ui(
                guide_note="Chosen: {}. Press Load mask.".format(
                    _Path(path).name)
            )

    def _on_reshape_guide_load(self) -> None:
        """Read the chosen mask on a WORKER, then add its layer here.

        A full-resolution guide is hundreds of MB; reading it (and
        majority-downsampling it, in a 2x session) on the main thread freezes
        napari for seconds, so it runs on a worker (the UI thread stays
        within roughly 100 ms).  Only the napari layer add happens here, because only
        that has to.
        """
        from napari_epicycle.annotator import reshape_guide as RG
        from napari_epicycle.annotator.advanced_features_widget import (
            SEEDS_FROM_EXTERNAL_MASK,
        )

        adv = self._get_advanced_features_widget()
        if adv is None:
            return
        if getattr(self, "_guide_thread", None) is not None:
            return                                    # a read is already going

        if adv.split_seeds_from_combo.currentIndex() == SEEDS_FROM_EXTERNAL_MASK:
            path = self._reshape_guide_path
            if not path:
                adv.update_split_guided_ui(
                    guide_note="Press Browse first, then Load mask.")
                return
            src = RG.guide_source_for_file(path)
        else:
            src = adv.selected_guide_source()
            if src is None:
                adv.update_split_guided_ui(
                    guide_note=(
                        "No other run of this patch has labels to trust. Use "
                        "Browse to pick a mask file instead."
                    )
                )
                return
            path = getattr(src, "path", "")
        run_id = getattr(src, "run_id", "") or ""
        display = getattr(src, "display_name", "") or run_id or _Path(path).stem

        ctx = self._reshape_context()
        main = self._layer_by_name(self.viewer.layers, self._active_labels_name)
        if main is None:
            adv.update_split_guided_ui(guide_note="No labels are loaded.")
            return

        self._guide_pending = {
            "display": display, "run_id": run_id, "path": path, "src": src,
            "patch": ctx["patch_name"] or "patch",
            "t0": _time.perf_counter(),
        }
        adv.update_split_guided_ui(busy=True, busy_progress=(0, 0),
                                   guide_note="Reading the mask...")

        from qtpy.QtCore import QThread

        from napari_epicycle.annotator.reshape_worker import GuideLoadWorker

        try:
            self._guide_thread = QThread()
            self._guide_worker = GuideLoadWorker(
                path, tuple(main.data.shape),
                downsample_factor=int(ctx["downsample_factor"]),
            )
            self._guide_worker.moveToThread(self._guide_thread)
            self._guide_thread.started.connect(self._guide_worker.run)
            self._guide_worker.finished.connect(self._on_guide_loaded)
            self._guide_worker.refused.connect(self._on_guide_refused)
            self._guide_worker.error.connect(self._on_guide_refused)
            self._guide_thread.start()
            print("[RESHAPE] guide read started: {}".format(path))
        except Exception as exc:  # noqa: BLE001 - never strand the busy bar
            self._guide_thread = None
            self._guide_worker = None
            self._guide_pending = None
            adv.update_split_guided_ui(
                busy=False,
                guide_note="That mask could not be read: {}".format(exc),
            )
            print("[RESHAPE] guide read FAILED to start: {!r}".format(exc))

    def _guide_thread_cleanup(self) -> None:
        """Join the guide reader -- or park it, for the same fatal reason.

        See `_reshape_thread_cleanup`: a QThread dropped while still running
        aborts the process.  A cold-cache read of a full-resolution mask can
        outlast any sensible wait.
        """
        th = getattr(self, "_guide_thread", None)
        worker = getattr(self, "_guide_worker", None)
        self._guide_thread = None
        self._guide_worker = None
        if th is None:
            return
        try:
            th.quit()
            if th.wait(5000):
                return
        except Exception:
            return
        print("[RESHAPE] guide reader will not stop in time - parking it")
        parked = getattr(self, "_reshape_parked_threads", None)
        if parked is None:
            parked = []
            self._reshape_parked_threads = parked
        for sig in ("finished", "refused", "error"):
            try:
                getattr(worker, sig).disconnect()
            except Exception:
                pass
        entry = (th, worker)
        parked.append(entry)

        def _release():
            try:
                parked.remove(entry)
            except Exception:
                pass

        try:
            th.finished.connect(_release)
        except Exception:
            pass

    def _on_guide_refused(self, message) -> None:
        """A refusal is an ANSWER -- show it verbatim, add nothing."""
        self._guide_thread_cleanup()
        pending = getattr(self, "_guide_pending", None)
        self._guide_pending = None
        adv = self._get_advanced_features_widget()
        if adv is not None:
            adv.update_split_guided_ui(
                busy=False, guide_loaded=False,
                guide_note=str(message).splitlines()[0],
            )
        print("[RESHAPE] guide refused: {}".format(
            str(message).splitlines()[0]))
        if pending is None:
            return

    def _on_guide_loaded(self, guide, notes) -> None:
        """The array arrived -- add its layer, on the main thread."""
        from napari_epicycle.annotator import reshape_guide as RG

        self._guide_thread_cleanup()
        pending = getattr(self, "_guide_pending", None)
        self._guide_pending = None
        adv = self._get_advanced_features_widget()
        if pending is None:
            return
        # The split session may have ended while the disk was busy.  Adding a
        # guide layer for a patch nobody is looking at is exactly the foreign-
        # layer hazard the sweep exists to prevent.
        ctx = self._reshape_context()
        if (ctx["patch_name"] or "patch") != pending["patch"]:
            print("[RESHAPE] guide read discarded - patch changed")
            if adv is not None:
                adv.update_split_guided_ui(busy=False)
            return

        self._remove_guide_layer()
        self._reshape_guide = guide
        self._reshape_guide_run = pending["run_id"] or _Path(
            pending["path"]).name
        name = RG.guide_layer_name(pending["patch"], pending["src"])
        self._add_guide_layer(guide, name)
        n = RG.seed_count(guide)
        note = " ".join(notes or [])
        if adv is not None:
            adv.update_split_guided_ui(
                busy=False, guide_loaded=True,
                guide_note=(
                    "{} - {:,} labels in the mask. {}".format(
                        pending["display"], n, note)
                ).strip(),
            )
        print(
            "[RESHAPE-PERF] guide load wall={:.0f}ms labels={} bytes={} "
            "notes={}".format(
                (_time.perf_counter() - pending["t0"]) * 1000.0, n,
                guide.nbytes, len(notes or []))
        )

    def _add_temp_layer(self, key, array, *, kind="labels", name=None,
                        translate=None, scale=None, units=None):
        """Add ONE unmanaged, read-only ``guide_``-prefixed layer under *key*.

        The registry that replaces the single ``_reshape_guide_layer``
        slot.  With one slot a markers layer would have
        silently evicted the loaded mask -- the two exist at the same
        time and have different lifetimes.

        Deliberately excluded from `_active_labels_names`, from
        `sync_manager.shared_labels`, from the subviewer mirror set and
        from `_managed_layer_names`: roughly 30 sites read the SCALAR
        `_active_labels_name` to reach the shared volume, and there is no
        controller, state, path or save support for a second Step-3
        labels volume.  It participates in nothing; the engine reads its
        array directly.
        """
        registry = getattr(self, "_temp_guide_layers", None)
        if registry is None:
            registry = self._temp_guide_layers = {}
        self._remove_temp_layer(key)
        name = name or "guide_{}".format(key)
        # PLACEMENT (Brick U3).  A temp layer does not have to be
        # volume-shaped to line up: a CROP with `translate` + `scale` is
        # in the same world coordinates as the layer it was cropped
        # from, and costs two orders of magnitude less memory.  The
        # kwargs are optional and dropped if this napari does not take
        # them, because a layer in the RIGHT PLACE with no units is fine
        # and a layer that failed to appear is not.
        placement = {}
        if translate is not None:
            placement["translate"] = tuple(translate)
        if scale is not None:
            placement["scale"] = tuple(scale)
        if units is not None:
            placement["units"] = units
        #
        # ...but only as far as UNITS (fix pass H3).  The old third
        # attempt was the bare call, which adds the array at the volume
        # ORIGIN.  An unplaced crop is not a degraded picture, it is a
        # WRONG one -- a greyscale block sitting off the object under a
        # caption that says "this is what the tool saw".  So when there
        # is a placement to honour, failing to honour it is a refusal;
        # the bare call stays the last resort only for a layer that
        # carried no placement in the first place (the guide mask).
        attempts = [placement]
        if "units" in placement:
            attempts.append(
                {k: v for k, v in placement.items() if k != "units"})
        if not ({"translate", "scale"} & set(placement)):
            attempts.append({})
        layer, last_exc = None, None
        for i, attempt in enumerate(attempts):
            if i and attempt == attempts[i - 1]:
                continue                       # nothing left to drop
            try:
                if kind == "image":
                    layer = Image(
                        array, name=name, opacity=0.5, colormap="magma",
                        **attempt)
                else:
                    layer = Labels(array, name=name, opacity=0.45, **attempt)
                break
            except Exception as exc:  # noqa: BLE001
                # ANY failure, not only a TypeError: a napari that takes
                # `units` but rejects the VALUE would otherwise kill the
                # layer outright.  The last attempt is the bare call, so
                # a real failure still reports.
                last_exc = exc
                if i < len(attempts) - 1:
                    print("[RESHAPE] temp layer {!r}: {} refused ({!r}) - "
                          "retrying with less".format(
                              key, sorted(attempt) or "the plain call", exc))
                    continue
                print("[RESHAPE] temp layer {!r} failed: {!r}".format(
                    key, exc))
                return None
        if layer is None:
            print("[RESHAPE] temp layer {!r} failed: {!r}".format(
                key, last_exc))
            return None
        # `_epicycle_temporary` is what keeps it out of the annotator's
        # labels combo (native_multiview_widget skips any Labels layer
        # carrying it).
        #
        # BEFORE THE INSERT (fix pass H2).  `layers.events.inserted` fires
        # SYNCHRONOUSLY, and both filtered list builders are connected to
        # it -- `_refresh_layer_options` (the annotator's image combo) and
        # `_refresh_all_image_combos` (the five panel channel combos).
        # Adding through `viewer.add_image` and flagging the layer it
        # RETURNS is one statement too late: every one of them has by then
        # walked a list holding an unflagged `guide_walls` and offered it
        # to the user as a channel, for the whole life of the picture --
        # and picking it feeds the wall elevation back into the engine as
        # if it were an image.  So the layer is built, flagged, and only
        # then appended.  `ViewerModel.add_layer` is itself exactly
        # `self.layers.append(layer)` (napari's own docstring forbids it
        # doing more), so nothing is lost by taking that door.
        layer._epicycle_temporary = True
        try:
            # An Image layer has no meaningful `editable`, and in some
            # napari versions setting it raises.
            layer.editable = False
        except Exception:
            pass
        try:
            layer.mode = "pan_zoom"
        except Exception:
            pass
        if kind != "image":
            # EVERY Labels layer in the annotator flow is
            # locked at brush size 1, including one never drawn on.
            from packages.epicycle_annotator.sync import lock_brush_size_to_1

            lock_brush_size_to_1(layer)
        try:
            self.viewer.layers.append(layer)
        except Exception as exc:  # noqa: BLE001
            print("[RESHAPE] temp layer {!r} could not be inserted: "
                  "{!r}".format(key, exc))
            return None
        registry[key] = layer
        if key == "mask":
            self._reshape_guide_layer = layer
        def _placed(attr):
            """The layer's own placement, for the log -- and never a
            reason for the add to fail.

            Read element by element: a real napari layer answers
            `translate` with an ndarray, and `ndarray or ()` raises on
            the truth test, which would have printed `translate=None`
            over a perfectly placed layer.  The log is the live driver's
            only evidence that the placement reached the viewer, so it
            has to be the layer's own numbers.
            """
            try:
                raw = getattr(layer, attr, None)
                if raw is None:
                    return ()
                return tuple(float(v) for v in raw)
            except Exception:
                return ()

        print(
            "[RESHAPE] temp layer added key={} name={} shape={} dtype={} "
            "translate={} scale={} unmanaged=True".format(
                key, name, getattr(array, "shape", None),
                getattr(array, "dtype", None),
                _placed("translate") or None, _placed("scale") or None)
        )
        return layer

    def _remove_temp_layer(self, key) -> None:
        """De-register, hide, then remove, so the render loop never draws a
        freed GPU buffer.

        The layers are deliberately unmanaged, but selecting one in the
        layer list still makes it the active layer, and
        `subviewer_mirror.connect_label_mode_sync` registers ANY Labels
        layer that becomes active with the SyncManager.  Nothing cleans
        that up for a temp layer, so without the de-register the handlers
        keep the removed layer (and its full-volume array) alive -- the
        same leak the de-novo Add candidate hit (audit H1).
        """
        registry = getattr(self, "_temp_guide_layers", None)
        if registry is None:
            registry = self._temp_guide_layers = {}
        layer = registry.pop(key, None)
        if key == "mask":
            layer = layer or getattr(self, "_reshape_guide_layer", None)
            self._reshape_guide_layer = None
        if layer is None:
            return
        sm = getattr(self, "sync_manager", None)
        if sm is not None:
            try:
                sm.disconnect_label_sync(layer)
            except Exception:
                pass
        try:
            layer.visible = False
            if layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
        except Exception:
            pass
        # If it was the active layer, hand focus back to the layer the
        # user is actually curating -- otherwise the next paint has no
        # target.
        try:
            main = self._layer_by_name(
                self.viewer.layers, self._active_labels_name
            )
            if main is not None and self.viewer.layers.selection.active is None:
                self.viewer.layers.selection.active = main
        except Exception:
            pass
        print("[RESHAPE] temp layer removed key={}".format(key))

    def _remove_all_temp_layers(self) -> None:
        """Every temporary layer, in one call.  Idempotent."""
        for key in list(getattr(self, "_temp_guide_layers", {}) or {}):
            self._remove_temp_layer(key)
        self._remove_temp_layer("mask")

    # ------------------------------------------------------------------ #
    # Brick U3 -- "Show what the tool sees"
    # ------------------------------------------------------------------ #
    def _walls_display_crop(self):
        """``(crop, translate, scale, units)``, or ``None``.

        FOUR values, not three: ``units`` rides along so the layer can be
        labelled the way the object's own layer is.  The docstring said
        three while every caller unpacked four.

        ``ReshapeResult.elevation`` IS the ROI crop (``walls.py``;
        ``raise_neighbour_walls`` keeps float32) and is already retained
        by ``_reshape_result``, so the layer is that array itself -- no
        copy, no volume-shaped allocation, nothing new retained.

        **The crop, not a full-size array (Brick U3 fix, measured).**
        The first build wrote the crop into
        ``np.zeros(main.data.shape, np.float32)``.  On the owner's own
        patch -- (317, 480, 480) -- that is 292 MB, and the byte budget
        refused it on every Preview of the live driver run: the feature
        was dead on the only data it exists for.  A
        full-size array only gives ALIGNMENT, and ``translate`` + ``scale``
        give exactly that: the same crop was 69x99x97 = 2.7 MB, a factor
        of 110.

        The placement is read off the MAIN LABELS LAYER rather than
        assumed, because napari composites in world coordinates:
        ``world = translate + scale x data``.  A crop whose own index
        ``d`` is volume index ``d + roi_start`` therefore needs
        ``scale = main.scale`` and
        ``translate = main.translate + main.scale x roi_start`` -- which
        is the identity placement when neither is set, and still lands on
        the object when they are.
        """
        # Brick B1: the SAME layer, the same budget, the
        # same placement arithmetic -- only the two sources are now
        # accessors that answer for whichever tool has a picture staged.
        elevation = self._active_elevation()
        roi = self._active_display_roi()
        if elevation is None or roi is None:
            return None
        crop = np.asarray(elevation, np.float32)

        try:
            slices = tuple(roi)
        except TypeError:
            slices = (roi,)
        if crop.ndim != len(slices):
            print("[WALLS] refused: the result is {}-D but the ROI has {} "
                  "axes".format(crop.ndim, len(slices)))
            return None

        # The ROI's own start, and its extent where the volume shape is
        # known -- a crop that does not match the ROI belongs to another
        # session, and a wall picture in the WRONG PLACE is worse than
        # none (it would read as "the tool saw THAT").
        volume = None
        try:
            volume = self.sync_manager.shared_labels.get(
                self._active_labels_name)
        except Exception:
            volume = None
        if volume is None and self._refine_layer is not None:
            volume = self._refine_layer.data
        starts, extents = [], []
        for axis, sl in enumerate(slices):
            if not isinstance(sl, slice):
                print("[WALLS] refused: the ROI is not a tuple of slices")
                return None
            if volume is not None:
                lo, hi, _step = sl.indices(int(volume.shape[axis]))
            else:
                lo = int(sl.start or 0)
                hi = int(sl.stop) if sl.stop is not None else lo + crop.shape[axis]
            starts.append(int(lo))
            extents.append(int(max(hi - lo, 0)))
        if volume is not None and tuple(extents) != tuple(crop.shape):
            print("[WALLS] refused: the result is {} but the ROI is "
                  "{}".format(tuple(crop.shape), tuple(extents)))
            return None

        main = None
        try:
            main = self._layer_by_name(
                self.viewer.layers, self._active_labels_name)
        except Exception:
            main = None
        if main is None:
            # REFUSE (fix pass H3).  The placement is read off the main
            # labels layer; with no layer to read, the old code fell
            # through to identity -- scale 1, translate 0 -- and placed
            # the crop at the bare ROI start.  That is right only if the
            # object's own layer happens to be unscaled and untranslated,
            # and there is no way to know: under a physical voxel size it
            # is quietly the wrong place.  A wall picture in the wrong
            # place is worse than no wall picture.
            print("[WALLS] refused: the object's layer {!r} is not in the "
                  "viewer, so there is nothing to line the picture up "
                  "with".format(self._active_labels_name))
            return None
        ndim = crop.ndim

        def _vec(attr, default):
            raw = getattr(main, attr, None) if main is not None else None
            try:
                out = [float(v) for v in tuple(raw)]
            except Exception:
                return [default] * ndim
            if len(out) < ndim:
                return [default] * ndim
            return out[-ndim:]          # napari pads leading axes

        scale = _vec("scale", 1.0)
        base = _vec("translate", 0.0)
        translate = [b + sc * st for b, sc, st in zip(base, scale, starts)]
        units = getattr(main, "units", None) if main is not None else None
        return crop, tuple(translate), tuple(scale), units

    def _show_walls_picture(self) -> bool:
        """Put ``guide_walls`` up.  Returns False if it could not be.

        Called by the tick and again by ``_apply_reshape_result`` when a
        re-Preview lands with the box still on -- otherwise the PREVIOUS
        cut's wall picture would sit on screen next to the new pieces
        (audit M2).
        """
        adv = self._get_advanced_features_widget()
        placed = self._walls_display_crop()
        if placed is None:
            print("[WALLS] nothing to show - there is no result yet")
            if adv is not None and hasattr(adv, "clear_walls_toggle"):
                adv.clear_walls_toggle()
            return False
        crop, translate, scale, units = placed
        n_bytes = int(crop.nbytes)
        if n_bytes > WALLS_DISPLAY_BUDGET_BYTES:
            print("[WALLS] refused: the crop is {} bytes, over the {} byte "
                  "budget".format(n_bytes, WALLS_DISPLAY_BUDGET_BYTES))
            if adv is not None:
                if hasattr(adv, "clear_walls_toggle"):
                    adv.clear_walls_toggle()
                if hasattr(adv, "clear_refine_walls_toggle"):
                    adv.clear_refine_walls_toggle()
                if hasattr(adv, "set_refine_walls_toggle_enabled"):
                    from napari_epicycle.annotator.advanced_features_widget \
                        import WALLS_TOGGLE_TOO_LARGE as _WTL

                    adv.set_refine_walls_toggle_enabled(False, _WTL)
                    adv.set_refine_walls_message(
                        "This object's region is too large to show the "
                        "wall picture.")
                if hasattr(adv, "set_walls_toggle_enabled"):
                    # The grey tick has to say something OTHER than
                    # "Preview first" -- a Preview is exactly what just
                    # happened (fix pass H5).  `clear_walls_toggle` put
                    # the default back, so this replaces it, and the
                    # message below then takes the legend's slot.
                    from napari_epicycle.annotator.advanced_features_widget \
                        import WALLS_TOGGLE_TOO_LARGE

                    adv.set_walls_toggle_enabled(
                        False, WALLS_TOGGLE_TOO_LARGE)
                if hasattr(adv, "set_walls_message"):
                    # "patch" was wrong once the layer became the CROP
                    # (fix pass M2): the budget is measured against the
                    # object's own region, and a user told the PATCH is
                    # too large would go looking for a smaller patch
                    # instead of a smaller object.
                    adv.set_walls_message(
                        "This object's region is too large to show the "
                        "wall picture.")
            return False
        layer = self._add_temp_layer(
            "walls", crop, kind="image", name="guide_walls",
            translate=translate, scale=scale, units=units)
        if layer is None:
            # The registry refuses rather than placing the crop at the
            # volume origin (fix pass H3), so this is now reachable by a
            # napari that will not take the placement -- and the user
            # gets a sentence instead of a tick that springs back up for
            # no stated reason.
            print("[WALLS] refused: the picture could not be placed on the "
                  "object, so it was not shown")
            if adv is not None:
                if hasattr(adv, "clear_walls_toggle"):
                    adv.clear_walls_toggle()
                if hasattr(adv, "set_walls_message"):
                    adv.set_walls_message(
                        "The wall picture could not be lined up with the "
                        "object, so it was not shown.")
            return False
        print("[WALLS] added guide_walls crop={} bytes={} translate={} "
              "scale={} roi={}".format(
                  tuple(crop.shape), n_bytes, tuple(translate), tuple(scale),
                  tuple((sl.start, sl.stop)
                        for sl in (self._active_display_roi() or ()))))
        return True

    def _drop_walls_picture(self) -> None:
        """Take the wall picture away and put the tick back down.

        Idempotent, never raises: it runs on the six exits, and one of
        them is the teardown that puts the object back in the volume.

        This is an EXIT, so it also forgets that the picture was wanted
        -- unlike the removal inside ``_preview_reshape``, which takes
        the stale picture off while a new solve runs and deliberately
        keeps the intent so the new result can rebuild it.
        """
        self._walls_wanted = False
        had = "walls" in (getattr(self, "_temp_guide_layers", {}) or {})
        try:
            self._remove_temp_layer("walls")
        except Exception as exc:  # noqa: BLE001
            print("[WALLS] remove raised: {!r}".format(exc))
        adv = self._get_advanced_features_widget()
        if adv is not None and hasattr(adv, "clear_walls_toggle"):
            try:
                adv.clear_walls_toggle()
            except Exception:
                pass
        if had:
            print("[WALLS] removed guide_walls")

    def _on_reshape_show_walls_toggled(self, checked: bool) -> None:
        """The user ticked or unticked "Show what the tool sees".

        ``_walls_wanted`` is the INTENT, and it is deliberately not the
        checkbox: pressing "Preview again" pushes ``review_active=False``
        through the panel, which puts every result-shaped control back to
        idle -- the tick with them.  Reading the box in
        ``_apply_reshape_result`` would therefore find it down on every
        re-Preview and the rebuild would be dead code.
        """
        self._walls_wanted = bool(checked)
        if checked:
            if not self._show_walls_picture():
                self._walls_wanted = False
            return
        try:
            self._remove_temp_layer("walls")
        except Exception as exc:  # noqa: BLE001
            print("[WALLS] remove raised: {!r}".format(exc))
        print("[WALLS] removed guide_walls - unticked")

    def _add_guide_layer(self, array, name: str) -> None:
        """Add the guide as an UNMANAGED, read-only Labels layer.

        Deliberately excluded from `_active_labels_names`, from
        `sync_manager.shared_labels`, from the subviewer mirror set and from
        `_managed_layer_names`.  Roughly 30 sites read the SCALAR
        `_active_labels_name` to reach the shared volume, and there is no
        controller, state, path or save support for a second Step-3 labels
        volume -- making the guide a second MANAGED volume would touch every
        off-limits surface in the design.  It participates in nothing; the
        engine reads its array directly.
        """
        layer = self._add_temp_layer("mask", array, kind="labels", name=name)
        if layer is None:
            return
        adv = self._get_advanced_features_widget()
        if adv is not None and hasattr(adv, "split_guide_show_cb"):
            layer.visible = bool(adv.split_guide_show_cb.isChecked())

    def _on_reshape_guide_show(self, visible) -> None:
        layer = getattr(self, "_reshape_guide_layer", None)
        if layer is None:
            return
        try:
            layer.visible = bool(visible)
        except Exception:
            pass

    def _remove_guide_layer(self) -> None:
        """De-register, hide, then remove, so the render loop never draws a
        freed GPU buffer.

        The guide is deliberately unmanaged, but selecting it in the layer
        list still makes it the active layer, and `subviewer_mirror
        .connect_label_mode_sync` registers ANY Labels layer that becomes
        active with the SyncManager.  Nothing cleans that up for a temp
        layer, so without the de-register the handlers keep the removed layer
        (and its full-volume array) alive -- the same leak the de-novo Add
        candidate hit (audit H1).  AUDIT 2026-09-09.
        """
        self._remove_temp_layer("mask")
        self._reshape_guide = None
        print("[RESHAPE] guide layer removed")

    # ---------------------------------------------------------- the recipe --
    def _reshape_recipe(self, for_find: bool = False):
        """Build the engine spec from the pickers.

        ``for_find=True`` builds the recipe a "Find markers" press
        needs: the source's OWN detector, with neither the index-1
        re-map nor the painted-addition promotion below, because both
        of those describe how the CUT reads the markers, not how they
        are found.  Without it, pressing Find twice on source 1 would
        send `paint` the second time and find nothing new.
        """
        from napari_epicycle.annotator.advanced_features_widget import (
            FIND_MARKER_SOURCES, SEED_DETECTORS, SEEDS_FROM_DETECTED,
            SEEDS_FROM_EXTERNAL_MASK, SEEDS_FROM_IMAGE, SEEDS_FROM_PAINTED,
            SEEDS_FROM_RUN_LABELS, WALL_KINDS,
        )
        from packages.epicycle_annotator.reshape import (
            GEOMETRY_ONLY_WALLS, IMAGE_DERIVED_SEEDS, IMAGE_DERIVED_WALLS,
            ReshapeSpec,
        )

        adv = self._get_advanced_features_widget()
        if adv is None:
            return None, "The tools panel is not open.", None
        p = adv.get_split_params()

        # BOTH mappings are TOTAL: an index outside the
        # table is an explicit refusal naming it, never a fallback.  A
        # combo one item ahead of this table would be a KeyError; one
        # behind would be a silently different algorithm.
        _legacy_seed_kinds = {
            SEEDS_FROM_PAINTED: "paint",
            # Today's "Detect Seeds" finds EDT peaks inside the label; the
            # engine's geometry-blind detector is `distance` (h-maxima of the
            # interior distance transform), which is the same idea.
            SEEDS_FROM_DETECTED: "distance",
            SEEDS_FROM_RUN_LABELS: "labels",
            SEEDS_FROM_EXTERNAL_MASK: "labels",
        }
        seeds_from = int(p["seeds_from"])
        # Bug B, cured.  Index 1 maps to the engine's
        # `distance` detector, which finds its OWN markers and never
        # looks at the Refine layer -- so with the painting rows on
        # screen everything the user painted was silently discarded.
        # Once its Find has materialised the peaks into the Refine
        # layer it is an ordinary PAINTING source, and painted
        # additions are honoured by construction.
        painted_now = self._painted_marker_ids_on_layer()
        # U1b widened that to EVERY found source: a Find on 1, 2,
        # 3 or 4 materialises its markers onto the Refine layer, so the
        # cut reads them back as paint -- one round trip for all five
        # sources, and the markers are mirrored into the side views
        # because the Refine layer is.
        #
        # THE RECEIPT IS THE VOXELS (audit H1, 2026-09-20).  It used to
        # be `_reshape_found_seeds is None`, which is None in three
        # different situations and means "materialised" in only one of
        # them: before any Find at all, after a marker knob moved (the
        # handler nulls it), and after a Find that really did paint.
        # The first two silently cut from something the user did not
        # ask for -- load a mask, paint one marker by hand, press
        # Preview, and the mask was dropped in favour of the dot.
        # `_materialised_ids` is [] at Start, [] when a materialise is
        # refused, and non-empty only when voxels actually landed, so it
        # answers the question that is being asked.
        _materialised = bool(getattr(self, "_materialised_ids", None))
        if (not for_find and painted_now and _materialised
                and seeds_from in FIND_MARKER_SOURCES):
            seeds_from = SEEDS_FROM_PAINTED
        # ...and a found set that is still a CROP (the materialise was
        # refused) is merged with any hand-painted addition and sent as
        # `paint` instead (the promotion rule), so the addition is
        # not dropped on the floor by a detector that re-finds its own.
        # Unreachable in the ordinary flow, because the line above has
        # already taken every materialised source.
        promote_found = (
            not for_find
            and seeds_from in (SEEDS_FROM_RUN_LABELS,
                               SEEDS_FROM_EXTERNAL_MASK, SEEDS_FROM_IMAGE)
            and bool(painted_now)
        )
        if seeds_from == SEEDS_FROM_IMAGE:
            det = int(p.get("seed_detector", 0))
            if not 0 <= det < len(SEED_DETECTORS):
                return (
                    None,
                    "That way of finding markers is not one this tool "
                    "knows (item {}).".format(det),
                    None,
                )
            seeds_kind = SEED_DETECTORS[det]
        elif seeds_from in _legacy_seed_kinds:
            seeds_kind = _legacy_seed_kinds[seeds_from]
        else:
            return (
                None,
                "That marker source is not one this tool knows "
                "(item {}).".format(seeds_from),
                None,
            )

        walls_from = int(p["walls_from"])
        if not 0 <= walls_from < len(WALL_KINDS):
            return (
                None,
                "That wall source is not one this tool knows "
                "(item {}).".format(walls_from),
                None,
            )
        walls_kind = WALL_KINDS[walls_from]

        # Two channels now, each named by its own combo and each an
        # explicit refusal when it is blank -- never the first image
        # layer.  The keys "membrane" / "nuclei" are keys
        # into the engine's `images` mapping, NOT layer names.
        images = {}
        if walls_kind not in GEOMETRY_ONLY_WALLS:
            name = p["wall_channel"]
            layer = self._layer_by_name(self.viewer.layers, name)
            if layer is None or not isinstance(layer, Image):
                return (
                    None,
                    "Pick the image channel the walls are in.",
                    None,
                )
            images["membrane"] = self._squeeze_image_channel(
                np.asarray(layer.data)
            )
        if seeds_kind in IMAGE_DERIVED_SEEDS:
            name = p.get("seed_channel", "")
            layer = self._layer_by_name(self.viewer.layers, name)
            if layer is None or not isinstance(layer, Image):
                return (
                    None,
                    "Pick the image channel the markers are in.",
                    None,
                )
            images["nuclei"] = self._squeeze_image_channel(
                np.asarray(layer.data)
            )

        seed_layer = None
        # Read back by `_preview_reshape`, which must NOT also send the
        # tick list once the ids have been renumbered.
        self._reshape_promoted = False
        if promote_found:
            # The one rule that makes "Preview cuts with what is ticked
            # now" true for every source: a found set the user has added
            # to by hand is sent as PAINT, so the addition is not
            # dropped on the floor by a detector that re-finds its own.
            merged = self._merged_marker_seed_layer()
            if merged is None:
                promote_found = False
            else:
                seeds_kind = "paint"
                seed_layer = merged
                self._reshape_promoted = True
        if promote_found:
            pass
        elif seeds_kind == "labels":
            if self._reshape_guide is None:
                return (
                    None,
                    "Load the mask you want the markers to come from first.",
                    None,
                )
            seed_layer = self._reshape_guide
        elif seeds_kind == "paint":
            # The painted marker ids only -- the target's own voxels on
            # the Refine layer are the region, not markers.  Source of
            # truth is `_split_painted_marker_ids`, NOT `_split_seed_ids`:
            # a preview overwrites the latter with PIECE ids, and the
            # next Preview would then hand the engine the whole previous
            # partition as seeds.
            roi = self._split_roi
            refine_roi = self._refine_layer.data[roi]
            self._marker_rows_from_refine()
            keep = self._checked_marker_ids()
            wanted = [
                i for i in (self._split_painted_marker_ids or [])
                if i != self._split_label_id
            ]
            if keep is not None:
                # The ticks decide.  An unticked marker stays
                # DRAWN on the layer; it is only left out of here.
                keep_set = {int(k) for k in keep}
                wanted = [i for i in wanted if int(i) in keep_set]
            seed_layer = np.where(
                np.isin(refine_roi, wanted), refine_roi, 0
            ).astype(refine_roi.dtype, copy=False)
            if not wanted or not seed_layer.any():
                if keep is not None and self._split_painted_marker_ids:
                    # They HAVE markers; they unticked them all.  The
                    # "paint one first" line would be a lie.
                    return (
                        None,
                        "You unticked every marker - tick at least one, "
                        "or Find again.",
                        None,
                    )
                return (
                    None,
                    "Paint a marker inside each object first, then press "
                    "Preview.",
                    None,
                )

        refine = []
        # Snap needs image evidence, exactly as the arbiter does (M1).
        # `refine_result_guided` falls back to the WALL elevation when
        # there is no image one, so a stale tick against shape walls
        # silently snapped every piece's edge onto the distance
        # transform -- a real change to the committed geometry with
        # nothing on screen to explain it.  The widget unchecks the box
        # when the walls stop being image-derived; this is the
        # belt-and-braces behind that, and it reads the same set.
        if p["refine_guided"] and walls_kind in IMAGE_DERIVED_WALLS:
            refine.append("guided")
        if p["refine_merge_small"]:
            refine.append("merge_small")
        if p["target_count"]:
            refine.append("merge_to_count")

        params = {
            "wall_channel": (
                "membrane" if walls_kind not in GEOMETRY_ONLY_WALLS else None
            ),
            "seed_channel": (
                "nuclei" if seeds_kind in IMAGE_DERIVED_SEEDS else None
            ),
            "seed_preproc": str(p.get("seed_preproc", "none")),
            "wall_preproc": str(p.get("wall_preproc", "none")),
            "clip": float(p["clip"]),
            "ridge": bool(p["ridge"]),
            "h": float(p["h"]),
            "min_distance_um": float(p["min_distance_um"]),
            "rw_beta": float(p["rw_beta"]),
            "compactness": float(p["compactness"]),
            "min_size": int(p["min_size"]),
            "target_count": p["target_count"],
            "arbiter_percentile": float(p["arbiter_percentile"]),
            # --- Brick 2b: the rest of the engine's params.
            # `ReshapeSpec` validates every key, so a name that drifted
            # here fails loudly rather than being silently ignored.
            "seed_smooth": float(p["seed_smooth"]),
            "max_sigma_um": float(p["max_sigma_um"]),
            "seed_thresh": (None if p["seed_thresh"] is None
                            else float(p["seed_thresh"])),
            "seed_clip": float(p["seed_clip"]),
            "seed_gamma": float(p["seed_gamma"]),
            "seed_bg": int(p["seed_bg"]),
            "seed_denoise": int(p["seed_denoise"]),
            "smooth": float(p["smooth"]),
            "refine_radius": int(p["refine_radius"]),
            "refine_eps": float(p["refine_eps"]),
            # The wall chain runs the SAME lift, haze and despeckle sizes
            # as the marker chain: one set of knobs, because the wall
            # chain either mirrors the marker one outright ("Same as
            # markers") or ticks the same steps, and two copies could
            # disagree with nothing on screen saying which won.
            "wall_gamma": float(p["seed_gamma"]),
            "wall_bg": int(p["seed_bg"]),
            "wall_denoise": int(p["seed_denoise"]),
        }
        # The arbiter needs an image-derived elevation as evidence; the engine
        # REFUSES it otherwise (arbitrating a shape-derived cut against its own
        # shape is circular).  The widget hides the checkbox in that case, so
        # this is belt-and-braces against a stale checked state.
        arbiter = bool(p["arbiter"]) and walls_kind in IMAGE_DERIVED_WALLS
        if self._reshape_override:
            arbiter = False   # "Split anyway": commit the unarbitrated cut
        # --- [COUNT-DIAG] line (A): what was ASKED for.
        # Printed BEFORE the spec is validated, so a combination the
        # engine refuses still says what it was.  Not on a `for_find`
        # build: the expected count plays no part in FINDING markers,
        # and a stray line there would break 6.3's first table row,
        # which reads "(A) absent and (B) absent" as the legacy path.
        if not for_find:
            try:
                self._reshape_count_diag = {
                    "target": p["target_count"],
                    "stage_on": "merge_to_count" in refine,
                }
                print(
                    "[COUNT-DIAG] recipe target_count={} refine={} "
                    "guided_now={} seeds={} walls={} min_size={} arbiter={}"
                    .format(
                        p["target_count"], list(refine),
                        self._is_guided_now(), seeds_kind, walls_kind,
                        p["min_size"], arbiter,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                # A diagnostic may not be the thing that stops a cut.
                print("[COUNT-DIAG] could not report the recipe: "
                      "{!r}".format(exc))
        try:
            spec = ReshapeSpec(
                target="label", seeds=seeds_kind, walls=walls_kind,
                solver=p["solver"], refine=tuple(refine), params=params,
                arbiter=arbiter,
            )
        except ValueError as exc:
            return None, "That combination is not possible: {}".format(exc), None
        return spec, images, seed_layer

    def _reshape_search_padding(self) -> int:
        """The "Search padding" knob, in voxels.

        Read ONCE, at Start.  ``reshape_region(margin=)`` is ignored
        whenever a ``roi`` is passed and the panel always passes one,
        so this is the only moment the number can act -- and the
        snapshot, the Refine layer and the write-back ROI then all
        move together or not at all.  Anything unreadable
        falls back to 15, which is what this was hard-coded to.
        """
        adv = self._get_advanced_features_widget()
        if adv is None or not hasattr(adv, "get_split_params"):
            return 15
        try:
            return max(0, int(adv.get_split_params().get(
                "search_padding", 15)))
        except Exception:
            return 15

    def _signal_sender(self):
        """The QObject whose signal we are handling, or None.

        A seam, because the stale-finish gates need the token of the
        FINISHING worker and a test drives those handlers directly.
        Qt sets ``sender()`` correctly for a queued connection, which
        is exactly the late-result case the token exists for.
        """
        try:
            return self.sender()
        except Exception:
            return None

    def _result_token(self):
        """The launch token carried by the worker that just finished.

        ``None`` when there is no sender (outside Qt); the gate then
        abstains on the token leg and the other three guards stand.
        """
        return getattr(self._signal_sender(), "_epicycle_token", None)

    def _reshape_labels_for_engine(self):
        """The FULL-VOLUME labels array both the Find and the Preview use.

        The engine wants a volume whose ROI holds the PRISTINE target.
        The shared volume has a hole in it right now (Start punched it)
        and the Refine layer may have painted seeds over the footprint,
        so neither is usable directly.  ``np.zeros`` is calloc-backed,
        so the untouched pages cost nothing -- only the ROI is written.

        Built in ONE place because a Find and a Preview that saw
        different arrays would find different markers with different
        ids, and the ticks would then select the wrong ones.
        """
        roi = self._split_roi
        main = self._layer_by_name(self.viewer.layers, self._active_labels_name)
        if main is None or roi is None or self._refine_baseline is None:
            return None
        labels = np.zeros(main.data.shape, dtype=main.data.dtype)
        labels[roi] = self._refine_baseline
        return labels

    # ---------------------------------------------------------- markers --
    def _find_reshape_markers(self) -> None:
        """The "Find markers" button: the seed stage only, on a worker.

        Everything a Preview does up to the markers, and nothing after
        it.  The ROI is the Preview's ROI (``self._split_roi``) and the
        labels array is built the same way, because the detectors
        normalise and threshold over the CROP and number the markers by
        enumeration order -- two different crops give different markers
        with different ids.
        """
        if getattr(self, "_reshape_running", False):
            print("[RESHAPE-FIND] refused - a preview is in flight")
            return
        if getattr(self, "_seed_find_running", False):
            print("[RESHAPE-FIND] refused - a find is already in flight")
            return
        adv = self._get_advanced_features_widget()
        if not getattr(self, "_split_running", False):
            self._update_split_ui(
                status_text="Start a reshape on an object first.")
            print("[RESHAPE-FIND] refused - no session")
            return

        # THE LOOP.  A Find while a preview is on
        # screen puts the preview away first -- and `_split_previewed`
        # has to be down before any finder runs, or `_detect_seeds`
        # becomes a silent no-op (A-H2).
        self._leave_reshape_review(
            "New markers - press Preview to cut again.")

        spec, images_or_reason, seed_layer = self._reshape_recipe(
            for_find=True)
        if spec is None:
            if adv is not None:
                adv.update_split_ui_state(
                    marker_busy=False, marker_ids=[],
                    marker_note=str(images_or_reason))
                # H2: the list is empty now, so no tick can put a parked
                # marker back.  Reconcile or the voxels are stranded.
                self._sync_marker_parking()
            print("[RESHAPE-FIND] recipe refused: {}".format(
                images_or_reason))
            return
        images = images_or_reason

        labels_for_engine = self._reshape_labels_for_engine()
        roi = self._split_roi
        if labels_for_engine is None or roi is None:
            self._update_split_ui(
                status_text=(
                    "The object is not held out of the volume any more - "
                    "press Start again."
                )
            )
            print("[RESHAPE-FIND] refused - no ROI snapshot")
            return

        ctx = self._reshape_context()
        self._seed_find_token = int(
            getattr(self, "_seed_find_token", 0)) + 1
        token = self._seed_find_token
        self._seed_find_launch_token = token
        self._seed_find_launch_labels_name = self._active_labels_name
        self._seed_find_launch_patch = ctx.get("patch_name")
        self._seed_find_launch_run_epoch = ctx.get("run_epoch")
        self._seed_find_running = True
        self._split_busy_kind = "find"
        self._seed_find_t0 = _time.perf_counter()
        # The BUSY state here is LOCAL: it must not collapse S3, which
        # would hide the button that started it and the list it is about
        # to fill.
        if adv is not None:
            adv.update_split_ui_state(marker_ids=[], marker_busy=True)
            # H2: emptied for the duration of the Find, so the markers
            # come back on the object for it -- a stash held across a
            # worker's lifetime has no row to release it.
            self._sync_marker_parking()
        self._update_split_ui(
            preview_enabled=False, accept_enabled=False,
            reject_enabled=True, reject_text="Discard",
            status_text="Looking for the markers - the window stays "
                        "alive; Stop waiting gives up on it.",
        )

        from qtpy.QtCore import QThread

        from napari_epicycle.annotator.reshape_worker import SeedFindWorker

        try:
            self._seed_find_thread = QThread()
            self._seed_find_worker = SeedFindWorker(
                labels_for_engine, images, spec,
                target_id=self._split_label_id, roi=roi,
                seed_layer=seed_layer, spacing=tuple(ctx["spacing"]),
            )
            # THE token has to travel with the worker.  Reading it back
            # off `self` at finish time compares the launch token with
            # itself -- a tautology that discards nothing.
            self._seed_find_worker._epicycle_token = token
            self._seed_find_worker.moveToThread(self._seed_find_thread)
            self._seed_find_thread.started.connect(
                self._seed_find_worker.run)
            self._seed_find_worker.finished.connect(
                self._on_seed_find_finished)
            self._seed_find_worker.error.connect(self._on_seed_find_error)
            self._seed_find_thread.start()
            print(
                "[RESHAPE-FIND] worker started token={} target={} "
                "seeds={} walls={}".format(
                    token, self._split_label_id, spec.seeds, spec.walls)
            )
        except Exception as exc:  # noqa: BLE001 - never strand the surface
            self._seed_find_running = False
            self._split_busy_kind = None
            self._seed_find_thread = None
            self._seed_find_worker = None
            if adv is not None:
                adv.update_split_ui_state(
                    marker_busy=False,
                    marker_note="Could not start: {}".format(exc))
            self._update_split_ui(preview_enabled=True, reject_enabled=True)
            print("[RESHAPE-FIND] launch FAILED: {!r}".format(exc))

    def _on_reshape_markers_invalidated(self) -> None:
        """A marker knob moved -- the drawn markers are no longer true.

        The panel empties its list; this takes the MARKERS away with it,
        both the bookkeeping and the voxels.

        The voxels are the half the first version missed (audit H1).
        Since U1b a Find PAINTS its markers onto the Refine layer, so
        nulling the crop here left the old generation drawn on the
        object and still in ``_split_painted_marker_ids`` -- while the
        panel's now-empty list made `get_checked_marker_ids()` answer
        None, which filters nothing.  The next Preview therefore cut
        from precisely the markers the knob had just invalidated, and
        the new value was never used.  The count line's promise --
        "Find again, or Preview to find them fresh" -- is true only
        because of the call below.
        """
        self._remove_temp_layer("markers")
        # THE LOOP: the result on screen was cut with markers
        # that no longer exist, so it goes with them.  It runs FIRST
        #: `_leave_reshape_review` writes `_refine_pre_preview`
        # over the ROI, and that snapshot still holds the generation
        # that is being invalidated -- so dropping the voxels before the
        # snap-back would simply paint them back, leaving markers drawn
        # that the panel has no row for and `_split_painted_marker_ids`
        # no entry for.
        self._leave_reshape_review(
            "Markers changed - press Preview to cut again.")
        self._drop_materialised_markers(reason="a marker knob moved")
        # The list is empty now, so no tick can ever restore a parked
        # marker; a hand-painted one that is still in the bookkeeping
        # goes back on the object rather than vanishing silently.
        self._unpark_all_markers(reason="the marker list was emptied")
        self._reshape_found_seeds = None
        self._reshape_found_marker_ids = []
        self._reshape_found_marker_stats = {}
        print("[RESHAPE-FIND] markers invalidated - layer removed")

    def _seed_find_is_stale(self, result_token=None) -> bool:
        """The same four guards the solve uses.

        A lock stops a SECOND find starting; it does nothing about a LATE
        one.  The token is the guard the other three cannot replace:
        Find, Discard, Find again on the same patch, run and layer.
        """
        if result_token is not None and result_token != self._seed_find_token:
            print(
                "[RESHAPE-FIND] STALE finish discarded - token {} != "
                "{}".format(result_token, self._seed_find_token)
            )
            return True
        if self._active_labels_name != self._seed_find_launch_labels_name:
            print("[RESHAPE-FIND] STALE finish discarded - labels layer changed")
            return True
        ctx = self._reshape_context()
        if ctx.get("patch_name") != self._seed_find_launch_patch:
            print("[RESHAPE-FIND] STALE finish discarded - patch changed")
            return True
        if ctx.get("run_epoch") != self._seed_find_launch_run_epoch:
            print("[RESHAPE-FIND] STALE finish discarded - run changed")
            return True
        if not getattr(self, "_split_running", False):
            print("[RESHAPE-FIND] STALE finish discarded - session gone")
            return True
        return False

    def _abandon_seed_find_worker(self) -> str:
        """Cancel the marker search and clear the slot.  Idempotent."""
        worker = getattr(self, "_seed_find_worker", None)
        had = worker is not None or getattr(self, "_seed_find_thread", None)
        if worker is not None:
            try:
                worker.cancel()
            except Exception:
                pass
        self._seed_find_running = False
        if getattr(self, "_split_busy_kind", None) == "find":
            self._split_busy_kind = None
        self._seed_find_thread_cleanup()
        if had:
            print("[RESHAPE-FIND] abandoned")
        return "abandoned" if had else "idle"

    def _seed_find_thread_cleanup(self) -> None:
        """Join the marker worker -- or PARK it if it will not stop.

        Same rule as the solve's: a QThread garbage-collected while
        running aborts the whole process (observed live 2026-09-08), so
        an unstoppable thread is held alive with its signals
        disconnected until it finishes on its own.
        """
        th = getattr(self, "_seed_find_thread", None)
        worker = getattr(self, "_seed_find_worker", None)
        self._seed_find_thread = None
        self._seed_find_worker = None
        if th is None:
            return
        try:
            th.quit()
            if th.wait(3000):
                return
        except Exception:
            return
        print(
            "[RESHAPE-FIND] worker will not stop in time - parking the "
            "thread (dropping it here would abort the process)"
        )
        parked = getattr(self, "_reshape_parked_threads", None)
        if parked is None:
            parked = []
            self._reshape_parked_threads = parked
        for sig in ("message", "finished", "error"):
            try:
                getattr(worker, sig).disconnect()
            except Exception:
                pass
        entry = (th, worker)
        parked.append(entry)

        def _release():
            try:
                parked.remove(entry)
            except Exception:
                pass
            print("[RESHAPE-FIND] parked worker thread finished and released")

        try:
            th.finished.connect(_release)
        except Exception:
            pass

    def _on_seed_find_error(self, message) -> None:
        """The marker search raised.  Nothing was changed either way.

        The stale gate runs BEFORE anything is touched: a late error
        from an abandoned find must not detach the worker the LIVE one
        is waiting on, and must not release a busy state the SOLVE owns.
        """
        if self._seed_find_is_stale(result_token=self._result_token()):
            return
        was_running = bool(getattr(self, "_seed_find_running", False))
        self._seed_find_running = False
        if getattr(self, "_split_busy_kind", None) == "find":
            self._split_busy_kind = None
        self._seed_find_thread_cleanup()
        if not was_running:
            # Cancelled: the user has already been told, and has already
            # moved on.  The finish handler drops a late ANSWER for the
            # same reason; a late CRASH is no different.
            print("[RESHAPE-FIND] late error discarded - cancelled")
            return
        adv = self._get_advanced_features_widget()
        if adv is not None:
            adv.update_split_ui_state(
                marker_busy=False, marker_ids=[],
                marker_note="That did not work. Nothing was changed.")
            self._sync_marker_parking()   # H2
        self._update_split_ui(preview_enabled=True, reject_enabled=True)
        print("[RESHAPE-FIND] worker error: {}".format(message))

    def _on_seed_find_finished(self, report) -> None:
        """The markers arrived -- list them and show them.

        NOTHING is touched before the stale gate.  The handler used to
        clean the thread slot and release the busy state first, so a
        late answer detached the worker the live session was waiting on
        and hid a running SOLVE's Stop button on its way out.
        """
        if self._seed_find_is_stale(result_token=self._result_token()):
            return
        was_running = bool(getattr(self, "_seed_find_running", False))
        self._seed_find_running = False
        if getattr(self, "_split_busy_kind", None) == "find":
            self._split_busy_kind = None
        self._seed_find_thread_cleanup()
        adv = self._get_advanced_features_widget()
        if not was_running:
            print("[RESHAPE-FIND] late result discarded - cancelled")
            return
        if adv is not None:
            adv.update_split_ui_state(marker_busy=False)

        wall = (_time.perf_counter()
                - getattr(self, "_seed_find_t0", 0.0)) * 1000.0
        status = getattr(report, "status", "ok")
        messages = list(getattr(report, "messages", []) or [])
        print(
            "[RESHAPE-FIND] finished wall={:.0f}ms status={} found={} "
            "in_region={}".format(
                wall, status, getattr(report, "n_found", 0),
                getattr(report, "n_in_region", 0))
        )

        if status != "ok":
            self._remove_temp_layer("markers")
            if adv is not None:
                adv.update_split_ui_state(
                    marker_ids=[],
                    marker_note=(
                        messages[-1] if messages
                        else "No markers found in this object."
                    ),
                )
                self._sync_marker_parking()   # H2
            self._update_split_ui(
                preview_enabled=True, accept_enabled=False,
                reject_enabled=True, reject_text="Discard",
                status_text="Object {} - no markers found".format(
                    self._split_label_id),
            )
            return

        markers = list(getattr(report, "markers", []) or [])
        marker_ids = [int(m["id"]) for m in markers]
        # Size AND place: the detectors return single-voxel peaks, so the
        # size alone is "1" for every marker and tells the user nothing.
        marker_stats = {
            int(m["id"]): {
                "voxels": int(m.get("voxels", 0)),
                "centroid_zyx": m.get("centroid_zyx"),
            }
            for m in markers
        }
        # Retained so a hand-painted ADDITION can be merged with them on
        # the next Preview (the staged-result promotion) rather than replacing
        # them.  ROI-shaped, so it is the crop, not the volume.
        self._reshape_found_seeds = getattr(report, "seeds", None)
        self._reshape_found_marker_ids = list(marker_ids)
        self._reshape_found_marker_stats = dict(marker_stats)
        # The ticks a Find may NOT touch: the markers the user painted
        # by hand.  A Find re-ticks what IT found, because the detectors
        # renumber from 1 on every run -- but a painted marker keeps its
        # id, so re-ticking it would undo a decision the user made on
        # purpose (fix pass H2).
        preserve_ids = None
        if self._marker_source_materialises():
            # EVERY found source writes its markers INTO the Refine
            # layer.  Bug B made index 1 do it; the
            # owner then found on real data that the other three were
            # invisible in the side views, because `guide_markers` is a
            # MAIN-viewer layer and the Refine layer is the mirrored
            # one.  One round trip for all of them: painted additions
            # are honoured by construction and no second colormap draws
            # the same markers twice.
            _was_materialised = set(
                int(i) for i in (getattr(self, "_materialised_ids", None)
                                 or []))
            preserve_ids = [
                int(i) for i in (self._split_painted_marker_ids or [])
                if int(i) not in _was_materialised
            ]
            marker_ids, marker_stats = self._materialise_found_markers(
                report, markers)
            if getattr(self, "_materialise_ok", False):
                # They are PAINTED ids now, so they come back through
                # `_split_painted_marker_ids` and must not be listed
                # twice.  Nulling the crop is also the RECEIPT the
                # recipe reads: no crop means the voxels are on the
                # layer, so the cut re-reads them as paint.
                self._reshape_found_seeds = None
                self._reshape_found_marker_ids = []
                self._reshape_found_marker_stats = dict(marker_stats)
            else:
                # Nothing landed (a crop that is not the ROI's shape).
                # The crop is KEPT, so the engine still finds its own
                # markers on the next Preview and the promotion rule
                # still merges a hand-painted addition with them --
                # today's behaviour, minus the picture.
                print("[MARKERS] nothing was materialised - the cut will "
                      "find its own markers again")
                self._reshape_found_marker_stats = dict(marker_stats)
            # Belt and braces: a `guide_markers` layer from a session
            # that pre-dates U1b must not outlive its list.
            self._remove_temp_layer("markers")
        if adv is not None:
            adv.update_split_ui_state(
                marker_ids=marker_ids, marker_stats=marker_stats,
                marker_preserve_ids=preserve_ids)
            # H2, and the reason it is here on the SUCCESS path too: the
            # rows have been replaced wholesale, and only the ids named
            # in `preserve_ids` keep an OFF tick.  Anything parked that
            # is not in the new list has lost the row that could have
            # released it; anything re-ticked by the repopulate has to
            # come back on the object to match its own tick.
            self._sync_marker_parking()
        self._update_split_ui(
            preview_enabled=True, accept_enabled=False, reject_enabled=True,
            reject_text="Discard",
            status_text="Object {} - {} marker{} found, {} ticked".format(
                self._split_label_id, len(marker_ids),
                "" if len(marker_ids) == 1 else "s", len(marker_ids)),
        )
        if adv is not None and hasattr(adv, "_focus_panel_stage"):
            try:
                adv._focus_panel_stage(
                    "Reshape", reveal=getattr(adv, "split_marker_list", None))
            except Exception:
                pass

    def _marker_finder_name(self) -> str:
        """The finder that is ACTUALLY on screen for the chosen source.

        One finder per card: "Find from shape" for source 0,
        "Find markers" for 1-4.  The Start line offered "Detect Seeds"
        to every source, and index 1 -- the legacy path whenever the
        walls are shape-derived, so it reaches that line -- lost that
        button in Brick U1.  The one instruction the user is given must
        not name a button that is not on the page (fix pass M5).
        """
        from napari_epicycle.annotator.advanced_features_widget import (
            DETECT_SEED_SOURCES,
        )

        adv = self._get_advanced_features_widget()
        idx = 0
        if adv is not None:
            try:
                idx = int(adv.get_split_params()["seeds_from"])
            except Exception:
                idx = 0
        return "Find from shape" if idx in DETECT_SEED_SOURCES else (
            "Find markers")

    def _marker_source_index(self) -> int:
        """The "Markers from" index the pickers show right now.

        Read LIVE, never latched -- the same rule the routing
        predicate follows.  Falls back to 0, which is the source that
        has no finder at all, so an unreadable panel can never be
        mistaken for a found one.
        """
        adv = self._get_advanced_features_widget()
        if adv is None:
            return 0
        try:
            return int(adv.get_split_params()["seeds_from"])
        except Exception:
            return 0

    def _drop_materialised_markers(self, *, reason="") -> list:
        """Take the LAST Find's markers back off the object.

        Only what a Find wrote: the ids in ``_materialised_ids``.  A
        marker the user painted by hand has never been in that list and
        is never touched.

        Two callers, and the second is the one the audit found missing.
        A re-Find calls it so the previous generation does not sit under
        the new one (two generations, one list, a doubled count).  A
        MARKER KNOB moving calls it because the panel empties its list
        at that moment: with the list empty `get_checked_marker_ids()`
        answers None, so nothing filters the cut, while the voxels of
        the invalidated generation were still painted on the object --
        so the next Preview cut from exactly the markers the user had
        just invalidated, and the knob they turned was never used.  The
        count line promises "Find again, or Preview to find them
        fresh", which is only true once they are really gone.

        Returns the ids it removed.
        """
        previous = [int(i) for i in (getattr(self, "_materialised_ids", None)
                                     or [])]
        self._materialised_ids = []
        if not previous:
            return []
        # A generation that is leaving takes its STASH with it: a parked marker from the previous Find has no row any
        # more, and its voxels belong to a set of markers that is no
        # longer on screen.  Restoring them later would paint a
        # superseded generation back onto the object.
        stash = getattr(self, "_parked_marker_voxels", None) or {}
        for mid in previous:
            stash.pop(int(mid), None)
        if self._refine_layer is None or self._split_roi is None:
            return previous
        try:
            roi_data = self._refine_layer.data[self._split_roi]
            roi_data[np.isin(roi_data, previous)] = 0
            drop = set(previous)
            self._split_painted_marker_ids = [
                int(i) for i in (self._split_painted_marker_ids or [])
                if int(i) not in drop
            ]
            self._split_seed_ids = list(self._split_painted_marker_ids)
            self._refine_layer.refresh()
        except Exception as exc:  # noqa: BLE001
            print("[MARKERS] could not take the previous markers off the "
                  "object: {!r}".format(exc))
            return previous
        self._refresh_refine_in_subviewers()
        print("[MARKERS] the previous Find's {} marker(s) removed {}: "
              "{}".format(len(previous), reason or "from the object",
                          previous))
        return previous

    def _marker_source_materialises(self) -> bool:
        """True when a Find must write its markers into the Refine layer.

        EVERY found source, since U1b.  The markers of sources
        2-4 used to go into `guide_markers`, which only the MAIN viewer
        holds -- so they were invisible in the XY / XZ / YZ side views,
        which is where they are painted and checked (owner,
        2026-09-20).  The Refine layer is mirrored into all three, and
        the cut reads its ids back as paint, so painted additions are
        honoured by construction.  Source 0 has nothing to materialise:
        its markers are already on that layer.
        """
        from napari_epicycle.annotator.advanced_features_widget import (
            MATERIALISED_MARKER_SOURCES,
        )

        adv = self._get_advanced_features_widget()
        if adv is None:
            return False
        try:
            return int(
                adv.get_split_params()["seeds_from"]
            ) in MATERIALISED_MARKER_SOURCES
        except Exception:
            return False

    def _materialise_found_markers(self, report, markers):
        """Write the report's marker voxels into the Refine layer.

        Three rules, each one a bug the obvious version would have had
        (audit A-H4):

        * **Mint with an offset.**  Ids are ``base + k`` where ``base``
          clears both the Refine layer's maximum and every live label,
          so a materialised marker can never collide with a painted one.
        * **Write the report's voxels verbatim.**  NOT spheres:
          ``_apply_seed_threshold`` inflates every peak with
          ``sphere_radius=2``, which would hand the engine a different
          seed array from the one the report described.
        * **Never go through ``_apply_seed_threshold``.**  It resets the
          ROI to ``_refine_baseline`` and overwrites the id list
          wholesale -- both of which erase the user's painted markers.

        Returns ``(marker_ids, marker_stats)`` in the NEW numbering.

        It is also REPLACING, not merely additive (fix pass H2).  It
        used to only append, so changing *Prominence* and pressing Find
        again left the previous run's voxels painted on the object under
        their old ids, still in `_split_painted_marker_ids` and still in
        the list -- the object was then cut from both generations at
        once and the count the user had just tuned was silently doubled.
        Only what THIS method wrote is taken out; a marker the user
        painted by hand is never touched.
        """
        seeds = getattr(report, "seeds", None)
        self._materialise_ok = False
        if (seeds is None or self._refine_layer is None
                or self._split_roi is None):
            return [int(m["id"]) for m in markers], {}
        roi_data = self._refine_layer.data[self._split_roi]
        # The shape check comes BEFORE anything is erased (audit M1).
        # It used to sit after, so a refused materialise had already
        # taken the previous generation off the object -- leaving the
        # panel listing markers that were no longer drawn anywhere.
        seeds = np.asarray(seeds)
        if seeds.shape != roi_data.shape:
            print("[MARKERS] materialise refused - the report is a "
                  "different shape from the ROI (the previous markers "
                  "are left alone)")
            return [int(m["id"]) for m in markers], {}
        self._drop_materialised_markers(reason="before this one")
        shared_max = 0
        try:
            shared = self.sync_manager.shared_labels.get(
                self._active_labels_name)
            if shared is not None:
                shared_max = int(np.max(shared))
        except Exception:
            shared_max = 0
        base = max(
            int(np.max(self._refine_layer.data)),
            shared_max,
            max(self._split_painted_marker_ids or [0]),
        ) + 1
        stats_by_old = {
            int(m["id"]): {
                "voxels": int(m.get("voxels", 0)),
                "centroid_zyx": m.get("centroid_zyx"),
            }
            for m in markers
        }
        new_ids, new_stats = [], {}
        for k, old in enumerate(
                int(m["id"]) for m in markers):
            mask = seeds == old
            if not mask.any():
                continue
            nid = base + k
            roi_data[mask] = nid
            new_ids.append(nid)
            if old in stats_by_old:
                new_stats[nid] = stats_by_old[old]
        # ADDITIVE: whatever was already painted stays painted and stays
        # in the list.  This is the difference from Detect Seeds, which
        # is allowed to replace the set because it resets the ROI first.
        for nid in new_ids:
            if nid not in self._split_painted_marker_ids:
                self._split_painted_marker_ids.append(nid)
        self._materialised_ids = list(new_ids)
        # ...but a marker with no VOXELS is not a marker.  `_start_split`
        # allocates the id of the first brush stroke before anything is
        # painted, and Start was taught not to list it; returning the
        # whole bookkeeping list here put it straight back, so a Find on
        # this source said "Found 2 markers" over an object the engine
        # had found one in, and the phantom was then counted in "3
        # ticked" for a cut that could only make two pieces (live driver
        # U1-08d).  Only ids that are really on the layer survive --
        # a HAND-PAINTED marker has voxels, so it is never touched.
        on_layer = self._painted_marker_ids_on_layer()
        if on_layer and all(int(n) in on_layer for n in new_ids):
            # A PARKED marker also has no voxels, and for a reason the
            # user chose -- so it is keepable, not phantom.
            # Without this the next Find would quietly delete every
            # marker the user had unticked.
            keepable = set(on_layer) | set(
                getattr(self, "_parked_marker_voxels", None) or {})
            dropped = [int(i) for i in self._split_painted_marker_ids
                       if int(i) not in keepable]
            if dropped:
                self._split_painted_marker_ids = [
                    int(i) for i in self._split_painted_marker_ids
                    if int(i) in keepable
                ]
                print("[MARKERS] {} allocated id(s) had no voxels and are "
                      "not markers: {}".format(len(dropped), dropped))
                try:
                    # the brush was aimed at one of them
                    if int(self._refine_layer.selected_label) in dropped:
                        self._refine_layer.selected_label = int(on_layer[-1])
                except Exception:
                    pass
        self._split_seed_ids = list(self._split_painted_marker_ids)
        try:
            self._refine_layer.refresh()
        except Exception:
            pass
        self._refresh_refine_in_subviewers()
        self._materialise_ok = bool(new_ids)
        print("[MARKERS] materialised {} marker(s) onto the object "
              "(source {}) as ids {}".format(
                  len(new_ids), self._marker_source_index(), new_ids))
        return list(self._split_painted_marker_ids), new_stats

    # -------------------------------------------------------------- preview --
    def _preview_reshape(self) -> None:
        """Launch the engine on a QThread.  The GUI stays alive."""
        if getattr(self, "_reshape_running", False):
            return
        adv = self._get_advanced_features_widget()
        if getattr(self, "_seed_find_running", False):
            # The mirror of the Find's own refusal.  Cutting while the
            # markers are still being looked for would cut with the
            # PREVIOUS set, or with none.
            self._update_split_ui(
                status_text=(
                    "Still looking for the markers - wait, or press "
                    "Stop waiting."
                )
            )
            print("[RESHAPE] preview refused - a find is in flight")
            return
        # On failure the middle slot carries a plain-language REASON rather
        # than the images mapping -- the recipe cannot be built, and the user
        # is told what to do about it.
        spec, images_or_reason, seed_layer = self._reshape_recipe()
        if spec is None:
            if adv is not None:
                # A refusal is an ANSWER (inv. 6), so it is shown as one:
                # red, in the result slot, not only on the status line.
                adv.update_split_guided_ui(
                    busy=False, engine_error=str(images_or_reason))
            self._update_split_ui(
                preview_enabled=True, accept_enabled=False,
                reject_enabled=True, status_text=str(images_or_reason),
            )
            print("[RESHAPE] recipe refused: {}".format(images_or_reason))
            return
        images = images_or_reason

        roi = self._split_roi
        labels_for_engine = self._reshape_labels_for_engine()
        if labels_for_engine is None or roi is None:
            self._update_split_ui(
                status_text=(
                    "The object is not held out of the volume any more - "
                    "press Start again."
                )
            )
            print("[RESHAPE] preview refused - no ROI snapshot")
            return

        # The cherry-pick: None means no Find has happened (the engine
        # finds them itself and keeps them all); [] means the user
        # unticked every one, which is a REFUSAL, not an unseeded cut.
        keep_seed_ids = None
        if adv is not None and hasattr(adv, "get_checked_marker_ids"):
            try:
                keep_seed_ids = adv.get_checked_marker_ids()
            except Exception:
                keep_seed_ids = None
        # A PROMOTED recipe has already applied the ticks, and it
        # RENUMBERS the found ids above the painted ones.  Passing the
        # tick list on as well would run `np.isin(seeds, keep)` against
        # ids that no longer exist and filter every promoted marker away
        #.  Narrow on purpose: an unpromoted `paint`
        # recipe is the legacy shape and keeps handing the list over,
        # where it is a no-op the call recorder pins.
        if getattr(self, "_reshape_promoted", False):
            print("[MARKERS] promoted cut - the ticks are already in the "
                  "seed array, so keep_seed_ids is not sent")
            keep_seed_ids = None
        if keep_seed_ids is not None and not keep_seed_ids:
            adv.update_split_ui_state(
                marker_note=(
                    "You unticked every marker - tick at least one, or "
                    "Find again."
                )
            )
            self._update_split_ui(
                preview_enabled=True, accept_enabled=False,
                reject_enabled=True,
                status_text="Tick at least one marker first.",
            )
            print("[RESHAPE] preview refused - every marker unticked")
            return

        # THE SNAP-BACK'S SOURCE.  The guided path had
        # no snapshot at all -- only the legacy one took it -- so there
        # was nothing to put back when the user changed a marker after a
        # preview.  Taken BEFORE the worker starts, and taken from the
        # layer as the user last left it, so it holds the MARKERS.
        #
        # NOT RE-TAKEN WHILE A PREVIEW IS UP.  Pressing "Preview again"
        # straight out of review arrives here with the ROI holding
        # PIECES; taking the snapshot then meant the next marker tick
        # snapped BACK TO THE PREVIOUS CUT -- pieces drawn on the object
        # with no rows behind them -- and `_marker_rows_from_refine`,
        # which must never run during review, read them as freshly
        # hand-painted markers.  The snapshot taken at the FIRST Preview
        # is still correct: it holds the markers, and every path that
        # can change them goes through the snap-back, which clears the
        # flag and lets the next Preview take a fresh one.
        try:
            if (self._refine_layer is not None and roi is not None
                    and not self._review_is_on_screen()):
                self._refine_pre_preview = (
                    self._refine_layer.data[roi].copy())
                # Anything painted with a brush the panel never heard
                # about joins the list before the cut replaces the ROI.
                self._marker_rows_from_refine()
            elif self._review_is_on_screen():
                print("[RESHAPE-LOOP] preview again - keeping the marker "
                      "snapshot (the ROI holds pieces right now)")
        except Exception as exc:  # noqa: BLE001
            print("[RESHAPE-LOOP] could not snapshot the markers: "
                  "{!r}".format(exc))

        ctx = self._reshape_context()
        self._reshape_token = int(getattr(self, "_reshape_token", 0)) + 1
        self._reshape_launch_labels_name = self._active_labels_name
        self._reshape_launch_patch = ctx.get("patch_name")
        self._reshape_launch_run_epoch = ctx.get("run_epoch")
        token = self._reshape_token
        self._reshape_running = True
        self._split_busy_kind = "solve"
        self._reshape_t0 = _time.perf_counter()

        if adv is not None:
            adv.update_split_guided_ui(
                busy=True, busy_progress=(0, 0),
                pieces=[], caption="", arbiter_message="",
                split_anyway_visible=False, engine_messages=[],
            )
        # Brick U3: the picture on screen belongs to the cut that is
        # being replaced, so it comes off while the solve runs.  The
        # LAYER only -- `_walls_wanted` survives, so a result that lands
        # with the box still wanted rebuilds it.
        if "walls" in (getattr(self, "_temp_guide_layers", {}) or {}):
            self._remove_temp_layer("walls")
            print("[WALLS] removed guide_walls - a new cut is being solved")
        self._update_split_ui(
            preview_enabled=False, accept_enabled=False, reject_enabled=True,
            reject_text="Discard", review_active=False,
            status_text="Working - the window stays alive; Stop waiting "
                        "gives up on it.",
        )
        self._lock_subviewers_for_preview()

        from qtpy.QtCore import QThread

        from napari_epicycle.annotator.reshape_worker import ReshapeWorker

        try:
            self._reshape_thread = QThread()
            self._reshape_launch_token = token
            self._reshape_worker = ReshapeWorker(
                labels_for_engine, images, spec,
                target_id=self._split_label_id, roi=roi,
                seed_layer=seed_layer, spacing=tuple(ctx["spacing"]),
                keep_seed_ids=keep_seed_ids,
            )
            # Same tautology, same cure (the solve's was pre-existing).
            self._reshape_worker._epicycle_token = token
            self._reshape_worker.moveToThread(self._reshape_thread)
            self._reshape_thread.started.connect(self._reshape_worker.run)
            self._reshape_worker.progress.connect(self._on_reshape_progress)
            self._reshape_worker.message.connect(self._on_reshape_message)
            self._reshape_worker.finished.connect(self._on_reshape_finished)
            self._reshape_worker.error.connect(self._on_reshape_error)
            self._reshape_thread.start()
            print(
                "[RESHAPE] worker started token={} target={} seeds={} "
                "walls={} solver={} arbiter={}".format(
                    token, self._split_label_id, spec.seeds, spec.walls,
                    spec.solver, spec.arbiter)
            )
        except Exception as exc:  # noqa: BLE001 - never strand a locked surface
            self._reshape_running = False
            self._reshape_thread = None
            self._reshape_worker = None
            if adv is not None:
                adv.update_split_guided_ui(busy=False)
            self._restore_ready_to_draw_state()
            self._update_split_ui(
                preview_enabled=True, reject_enabled=True,
                status_text="Could not start: {}".format(exc),
            )
            print("[RESHAPE] launch FAILED: {!r}".format(exc))

    def _on_reshape_progress(self, stage, i, n) -> None:
        adv = self._get_advanced_features_widget()
        if adv is None:
            return
        from napari_epicycle.annotator.reshape_worker import stage_text

        adv.update_split_guided_ui(
            busy=True, busy_progress=(int(i), int(n)),
            busy_text=stage_text(stage),
        )

    def _on_reshape_message(self, text) -> None:
        self._reshape_live_messages = list(
            getattr(self, "_reshape_live_messages", [])
        ) + [str(text)]
        adv = self._get_advanced_features_widget()
        if adv is not None:
            adv.update_split_guided_ui(
                engine_messages=self._reshape_live_messages)

    def _reshape_is_stale(self, result_token=None) -> bool:
        """A lock stops a SECOND op; it does nothing about a LATE one.

        FOUR independent guards, all captured at launch: the
        **dispatch token**, the labels layer, the patch and the run epoch.

        The token is the one that catches the case the other three cannot:
        Reject-then-Preview-again on the SAME patch, run and layer.  Session
        A's answer arrives while session B is staged and looks, to every
        other guard, exactly like session B's own.  (AUDIT 2026-09-09 -- the
        docstring promised this comparison before the code did.)
        """
        if result_token is not None and result_token != self._reshape_token:
            print(
                "[RESHAPE] STALE finish discarded - token {} != {}".format(
                    result_token, self._reshape_token)
            )
            return True
        if self._active_labels_name != self._reshape_launch_labels_name:
            print("[RESHAPE] STALE finish discarded - labels layer changed")
            return True
        ctx = self._reshape_context()
        if ctx.get("patch_name") != self._reshape_launch_patch:
            print("[RESHAPE] STALE finish discarded - patch changed")
            return True
        if ctx.get("run_epoch") != self._reshape_launch_run_epoch:
            print("[RESHAPE] STALE finish discarded - run changed")
            return True
        if not getattr(self, "_split_running", False):
            print("[RESHAPE] STALE finish discarded - split session gone")
            return True
        return False

    def _abandon_reshape_worker(self) -> str:
        """Cancel the solve and clear the active thread slot.

        Used by every path that leaves the guided session while a worker
        may still be running -- Reject, Cancel and the teardown.  It is
        NOT enough to clear `_reshape_running`: the worker keeps going,
        and its answer would land in the NEXT split session, which is
        indistinguishable from a correct one without the dispatch token.
        Idempotent; never raises.
        """
        worker = getattr(self, "_reshape_worker", None)
        had = worker is not None or getattr(self, "_reshape_thread", None)
        if worker is not None:
            try:
                worker.cancel()
            except Exception:
                pass
        self._reshape_running = False
        if getattr(self, "_split_busy_kind", None) == "solve":
            self._split_busy_kind = None
        self._reshape_thread_cleanup()
        return "abandoned" if had else "idle"

    def _reshape_thread_cleanup(self) -> None:
        """Join the worker -- or PARK it if it will not stop.

        Cancel is polled BETWEEN stages, and a stage already entered runs to
        completion (`random_walker`, `diffuse_labels3D`, `guidedfilter`, and
        in practice a multi-second watershed).  So a teardown can arrive while
        the thread is genuinely still running.

        Dropping the reference then is FATAL, not untidy: a QThread garbage-
        collected while running aborts the whole process with
        "QThread: Destroyed while thread is still running".  Observed live on
        2026-09-08 -- tearing a preview down mid-solve took napari with it.

        So an unstoppable thread is PARKED: its signals are disconnected (its
        answer is already being abandoned) and it is held alive until it
        finishes on its own, at which point it drops out of the list.
        """
        th = getattr(self, "_reshape_thread", None)
        worker = getattr(self, "_reshape_worker", None)
        self._reshape_thread = None
        self._reshape_worker = None
        if th is None:
            return
        try:
            th.quit()
            if th.wait(3000):
                return
        except Exception:
            return

        print(
            "[RESHAPE] worker will not stop in time - parking the thread "
            "(dropping it here would abort the process)"
        )
        parked = getattr(self, "_reshape_parked_threads", None)
        if parked is None:
            parked = []
            self._reshape_parked_threads = parked
        # Its answer is being abandoned, so stop it reaching a widget that may
        # be half torn down by the time it lands.
        for sig in ("progress", "message", "finished", "error"):
            try:
                getattr(worker, sig).disconnect()
            except Exception:
                pass
        entry = (th, worker)
        parked.append(entry)

        def _release():
            try:
                parked.remove(entry)
            except Exception:
                pass
            print("[RESHAPE] parked worker thread finished and released")

        try:
            th.finished.connect(_release)
        except Exception:
            pass

    def _on_reshape_cancel(self) -> None:
        """Abandon the result.  Nothing is changed either way.

        Cancel re-enables Preview, so the thread slot MUST be empty when this
        returns: the next Preview overwrites `_reshape_thread` /
        `_reshape_worker`, and if those were the only strong references to a
        still-running QThread, Qt aborts the process (AUDIT 2026-09-09).
        `_abandon_reshape_worker` parks a thread that will not stop.
        """
        adv = self._get_advanced_features_widget()
        # Brick U3: whichever job is being stopped, there is no
        # result left to explain, so the wall picture goes with it.
        self._drop_walls_picture()
        # ONE Stop button, two jobs: `_split_busy_kind` says which one it
        # is for.  Stopping a Find must not tear down a solve
        # that is not running, and vice versa.
        if getattr(self, "_split_busy_kind", None) == "find":
            self._abandon_seed_find_worker()
            self._remove_temp_layer("markers")
            if adv is not None:
                adv.update_split_ui_state(
                    marker_busy=False, marker_ids=[],
                    marker_note="Stopped. No markers were found.")
                self._sync_marker_parking()   # H2
            self._update_split_ui(
                preview_enabled=True, accept_enabled=False,
                reject_enabled=True, reject_text="Discard",
                status_text=(
                    "Stopped. The object is still held out of the volume "
                    "- Discard puts it back."
                ),
            )
            print("[RESHAPE-FIND] cancelled by the user")
            return
        self._abandon_reshape_worker()
        self._remove_temp_layer("markers")
        if adv is not None:
            # The block sits ABOVE the list and describes the pieces
            # in it, so it goes with them (M4).
            adv.update_split_guided_ui(busy=False, pieces=[], caption="",
                                       quality_notes=[])
        self._update_split_ui(
            preview_enabled=True, accept_enabled=False, reject_enabled=True,
            reject_text="Discard",
            status_text=(
                "Stopped. The object is still held out of the volume - "
                "Discard puts it back."
            ),
        )
        print("[RESHAPE] cancelled by the user - result will be discarded")

    def _on_reshape_error(self, message) -> None:
        if self._reshape_is_stale(result_token=self._result_token()):
            return
        was_running = bool(getattr(self, "_reshape_running", False))
        self._reshape_running = False
        if getattr(self, "_split_busy_kind", None) == "solve":
            self._split_busy_kind = None
        self._reshape_thread_cleanup()
        if not was_running:
            print("[RESHAPE] late error discarded - cancelled")
            return
        adv = self._get_advanced_features_widget()
        if adv is not None:
            adv.update_split_guided_ui(busy=False)
        self._restore_ready_to_draw_state()
        self._update_split_ui(
            preview_enabled=True, accept_enabled=False, reject_enabled=True,
            status_text=(
                "That did not work. Nothing was written, but the object "
                "is still held out of the volume - Discard puts it back."
            ),
        )
        print("[RESHAPE] worker error: {}".format(message))

    def _on_reshape_finished(self, result) -> None:
        # The token the finishing WORKER was launched with.  Reading it
        # back off `self` compared the launch token with itself, so the
        # one guard the other three cannot replace never fired; and
        # tearing the thread slot down before the gate detached the
        # worker the live session was waiting on.  Gate first, always.
        if self._reshape_is_stale(result_token=self._result_token()):
            return
        was_running = bool(getattr(self, "_reshape_running", False))
        self._reshape_running = False
        if getattr(self, "_split_busy_kind", None) == "solve":
            self._split_busy_kind = None
        self._reshape_thread_cleanup()
        adv = self._get_advanced_features_widget()
        if adv is not None:
            adv.update_split_guided_ui(busy=False)
        if not was_running:
            # Cancelled: the user already saw "nothing was changed".
            print("[RESHAPE] late result discarded - cancelled")
            return

        wall = (_time.perf_counter() - getattr(self, "_reshape_t0", 0.0)) * 1000.0
        status = getattr(result, "status", "ok")
        messages = list(getattr(result, "messages", []) or [])
        print(
            "[RESHAPE-PERF] preview wall={:.0f}ms status={} seeds={} "
            "pieces={}".format(
                wall, status, getattr(result, "n_seeds", 0),
                getattr(result, "n_pieces", 0))
        )

        if status == "cancelled":
            if adv is not None:
                adv.update_split_guided_ui(engine_messages=messages)
            self._update_split_ui(
                preview_enabled=True, reject_enabled=True,
                status_text=(
                    "Stopped. The object is still held out of the volume "
                    "- Discard puts it back."))
            return

        if status != "ok":
            if adv is not None:
                # The engine's last message IS the reason (zero markers,
                # over budget, the cell would vanish); the rest are its
                # remarks and keep their grey.
                adv.update_split_guided_ui(
                    pieces=[], caption="", quality_notes=[],
                    engine_messages=messages[:-1] if messages else [],
                    engine_error=(
                        messages[-1] if messages
                        else "Nothing to show - Discard puts the object back."
                    ),
                )
            self._update_split_ui(
                preview_enabled=True, accept_enabled=False,
                reject_enabled=True, reject_text="Discard",
                status_text=(
                    messages[-1] if messages
                    else "Nothing to show - Discard puts the object back."
                ),
            )
            return

        self._apply_reshape_result(result)

    def _print_count_diag_result(self, result) -> None:
        """[COUNT-DIAG] line (B): what the engine DID.

        Read beside line (A) it separates the six reasons an expected
        count can come out wrong -- the 6.3 decision table.  Three of
        them are otherwise invisible: a stage that was never sent, a
        stage that ran and merged nothing, and a count the ARBITER or
        absorb-slivers had already reduced before it got there.

        Best-effort by construction.  A diagnostic that raises would
        take the preview down with it, and every field here is read off
        an object that a refusal or an older engine may not have
        filled.
        """
        try:
            rec = getattr(self, "_reshape_count_diag", None) or {}
            arb = getattr(result, "arbiter", None)
            print(
                "[COUNT-DIAG] engine n_seeds={} n_before={} target={} "
                "stage_on={} merged={} arbiter_merged={} n_pieces={}"
                .format(
                    getattr(result, "n_seeds", 0),
                    # None, not 0: "the engine never got that far" is
                    # not the same answer as "it counted zero" (L5).
                    getattr(result, "n_pieces_before_merge", None),
                    rec.get("target"),
                    rec.get("stage_on", False),
                    getattr(result, "merged_to_count", False),
                    len(getattr(arb, "merged_pairs", []) or []),
                    getattr(result, "n_pieces", 0),
                )
            )
        except Exception as exc:  # noqa: BLE001
            print("[COUNT-DIAG] could not report the counts: {!r}".format(exc))

    def _apply_reshape_result(self, result) -> None:
        """Mint the final ids, paint the preview, fill the piece list."""
        from packages.epicycle_annotator.reshape import remap_flags

        self._reshape_result = result
        self._print_count_diag_result(result)
        piece_ids = [int(r["id"]) for r in (result.piece_stats or [])]
        _arb = getattr(result, "arbiter", None)
        _verdict = getattr(_arb, "verdict", "not-run") if _arb else "not-run"
        kept_whole = bool(_verdict == "dividing" or len(piece_ids) <= 1)
        if kept_whole and piece_ids:
            # AUDIT 2026-09-09 /  a "dividing" verdict is a NO-OP
            # WRITE.  The cell was NOT cut, so it keeps its own id and its row
            # stays ACTIVE -- minting a fresh id here would retire a label
            # nothing happened to, and would make the commit's
            # `superseded=False` / "dividing" branch unreachable.
            self._reshape_final_ids = {pid: int(self._split_label_id)
                                       for pid in piece_ids}
        else:
            base = self._next_label_id_after_accept()
            # Ids are minted through the SAME helper the manual split uses, so
            # a guided split's ids behave exactly like a manual one's -- the
            # original is superseded and every piece is new.
            # Swapping that helper for `next_unused_label_id` is a separate,
            # independently grilled change (follow-up F1).
            self._reshape_final_ids = {
                pid: base + i for i, pid in enumerate(piece_ids)
            }
        self._reshape_flags = remap_flags(
            getattr(result, "flags", {}) or {}, self._reshape_final_ids
        )
        sizes = {
            self._reshape_final_ids[int(r["id"])]: int(r["voxels"])
            for r in (result.piece_stats or [])
        }
        final_ids = [self._reshape_final_ids[p] for p in piece_ids]
        # NOT `self._split_seed_ids = final_ids` any more.  That is what made the loop impossible: after one
        # preview the "markers" the panel, the recipe and Undo all read
        # were the PIECE ids, so re-editing a marker meant discarding
        # the session.  `_reshape_final_ids` already carries the pieces;
        # the markers stay the markers.

        arb, verdict = _arb, _verdict
        self._reshape_arbiter_merged = bool(
            verdict == "dividing" or (arb and getattr(arb, "merged_pairs", []))
        )
        arb_msg = ""
        if verdict == "dividing":
            arb_msg = (
                "kept whole - no wall between the markers "
                "(possibly dividing)"
            )
        elif self._reshape_arbiter_merged:
            arb_msg = (
                "some proposed cuts had no wall behind them and were "
                "put back together"
            )

        self._render_reshape_preview()
        self._split_previewed = True

        notes = self._reshape_quality_notes(result)

        adv = self._get_advanced_features_widget()
        if adv is not None:
            adv.update_split_guided_ui(
                pieces=final_ids, piece_sizes=sizes,
                piece_flags=dict(self._reshape_flags),
                caption=(
                    "unticked pieces stay part of object {}".format(
                        self._split_label_id)
                ),
                arbiter_message=arb_msg,
                split_anyway_visible=bool(
                    self._reshape_arbiter_merged
                    and not self._reshape_override
                ),
                engine_messages=list(result.messages or []),
                # A refusal left up over a successful preview is worse
                # than one never shown.
                engine_error="",
                quality_notes=notes,
            )
        n = len(final_ids)
        n_markers = len(self._split_painted_marker_ids or [])
        self._update_split_ui(
            start_enabled=False,
            # THE LOOP: the marker work stays live in review.
            # Add marker, Undo and Edit markers each put the preview
            # away and bring the markers back before they act.
            next_seed_enabled=True,
            undo_seed_enabled=(n_markers >= 2),
            refine_seeds_enabled=True,
            preview_enabled=True, accept_enabled=n > 0, reject_enabled=True,
            reject_text="Discard",
            # U4 (2.7): "Refine Fragments..." used to be pushed FALSE
            # here, which is why the hand-off existed only for a
            # hand-painted, shape-only preview -- pick a marker source or
            # a wall kind and it vanished.
            #
            # `ticked=None` reads the LIVE ticks (audit L1).  "Every row
            # arrives ticked" is not true: `update_split_guided_ui` has
            # just run and the quality gate unticks the slivers it names,
            # so passing `final_ids` would light the button for a cut
            # whose only pieces are the ones the gate threw out.
            refine_further_enabled=(
                self._n_reshape_fragments(ticked=None) >= 1),
            review_active=True,
            # Brick U3: there IS an elevation now, so the tick is
            # offered.  Pushed here rather than from `review_active`,
            # because the manual path reaches review with no engine
            # result and must leave it grey.
            walls_toggle_enabled=(
                getattr(result, "elevation", None) is not None),
            status_text=self._reshape_review_status(result, n),
        )

        # ...and if the picture was already up, REPLACE it (audit M2).
        # Without this a re-Preview leaves the previous cut's wall
        # picture on screen next to the new pieces -- the one state in
        # which "what the tool sees" would be a lie.
        if getattr(self, "_walls_wanted", False):
            if self._show_walls_picture():
                if adv is not None and hasattr(adv, "set_walls_toggle_checked"):
                    adv.set_walls_toggle_checked(True)
            else:
                self._walls_wanted = False

    def _reshape_quality_notes(self, result) -> list:
        """The gate's sentences for this result.

        The notes name PIECE ids in the engine's numbering; the panel
        ticks the FINAL ones, so they are translated through the same
        `_reshape_final_ids` map the piece list was built from -- an
        untranslated id would untick nothing and read as "the rule does
        not work".
        """
        from packages.epicycle_annotator.reshape import (
            QualityNote, quality_warnings,
        )

        adv = self._get_advanced_features_widget()
        expected = None
        try:
            expected = adv.get_split_params().get("target_count")
        except Exception:
            expected = None
        # The ENGINE says whether its merge-to-count stage fired (M2).
        # This used to search the messages for "Merged ", which is
        # user-facing prose -- one rewording, or one other sentence with
        # that word in it, and the suppression silently flipped.
        merged = bool(getattr(result, "merged_to_count", False))
        try:
            notes = quality_warnings(
                result, target_id=int(self._split_label_id),
                expected_count=expected, merged_to_count=merged,
            )
        except Exception as exc:
            # The gate is commentary and must never take the review down
            # (M4/inv. 6) -- but a silent `except` that eats every bug in
            # it forever is how a broken quality block looks exactly like
            # a clean cut.  One line, on the console the user is told to
            # copy when something looks wrong.
            print("[RESHAPE-QUALITY] gate failed ({!r}) - no notes "
                  "shown".format(exc))
            return []
        out = []
        for note in notes:
            ids = tuple(self._reshape_final_ids.get(int(pid), int(pid))
                        for pid in note.piece_ids)
            out.append(QualityNote(note.kind, note.text, ids))
        return out

    def _reshape_review_status(self, result, n_pieces: int) -> str:
        """S1's line at review -- and the OTHER place the coverage number
        appears (one source, formatted twice).

        "94 % assigned" here and "6 % was not given to any piece" in the
        amber block are `coverage_percent` and its complement, so the two
        can never round apart.
        """
        from packages.epicycle_annotator.reshape import coverage_percent

        n_markers = len(self._split_painted_marker_ids or [])
        if n_pieces <= 1:
            # "0 piece" is not a sentence, and zero pieces IS reachable:
            # untick every piece and the count is zero.
            return "{} piece{} - boundary refined.".format(
                n_pieces, "" if n_pieces == 1 else "s")
        try:
            assigned = coverage_percent(getattr(result, "coverage", 0.0))
            share = " {} % assigned.".format(assigned)
        except Exception:
            share = ""
        # The line that tells the user the LOOP exists.
        # It used to stop at the percentage, so nothing on screen ever
        # said the markers were still editable.
        from_markers = (
            " from {} marker{}".format(
                n_markers, "" if n_markers == 1 else "s")
            if n_markers else ""
        )
        return (
            "Object {} - {} pieces{}.{} Scroll up to change the markers "
            "and Preview again.".format(
                self._split_label_id, n_pieces, from_markers, share)
        )

    def _render_reshape_preview(self) -> None:
        """Paint the partition into the Refine layer, ticks respected.

        An UNTICKED piece is drawn with the TARGET's id, because that is
        exactly where its voxels go on Accept: unticking means "not a
        separate cell", never "delete that tissue".
        No re-solve; this is a relabel of an array we already have.
        """
        result = getattr(self, "_reshape_result", None)
        if result is None or self._refine_layer is None:
            return
        adv = self._get_advanced_features_widget()
        ticked = set(
            adv.get_checked_split_pieces()
            if adv is not None and hasattr(adv, "get_checked_split_pieces")
            else self._reshape_final_ids.values()
        )
        partition = np.asarray(result.result)
        out = np.zeros_like(partition, dtype=self._refine_layer.data.dtype)
        for pid, final in self._reshape_final_ids.items():
            out[partition == pid] = (
                final if final in ticked else self._split_label_id
            )
        self._refine_layer.data[self._split_roi] = out
        self._refine_layer.refresh()
        self._refresh_refine_in_subviewers()

    def _n_reshape_fragments(self, ticked=None) -> int:
        """How many pieces would become a SEPARATE object (U4, 2.7.2).

        This is the set fragment refine is about to write, so it is the
        honest predicate for offering "Refine Fragments...": it is zero
        when the arbiter kept the object whole (every final id IS the
        target's, so there are no fragments, only the object) and zero
        when every piece has been unticked.

        It is deliberately NOT ``n_pieces >= 2``.  The legacy path's
        effective two comes from `validate_split_seeds` refusing a
        one-seed split, not from a rule about fragments -- with three
        pieces and two unticked there is exactly one fragment to refine,
        and refining it is the point.
        """
        final_map = getattr(self, "_reshape_final_ids", None) or {}
        if not final_map:
            return 0
        if ticked is None:
            adv = self._get_advanced_features_widget()
            ticked = (
                adv.get_checked_split_pieces()
                if adv is not None and hasattr(adv, "get_checked_split_pieces")
                else final_map.values()
            )
        ticked = {int(v) for v in ticked}
        target = int(self._split_label_id or 0)
        return len({int(v) for v in final_map.values()
                    if int(v) in ticked and int(v) != target})

    def _on_reshape_pieces_changed(self) -> None:
        """A tick changed -- re-draw only.  Never re-solve."""
        if not getattr(self, "_reshape_active", False):
            return
        if getattr(self, "_reshape_result", None) is None:
            return
        self._render_reshape_preview()
        adv = self._get_advanced_features_widget()
        if adv is not None:
            ticked = adv.get_checked_split_pieces()
            n = len(ticked)
            # U4: the hand-off follows the ticks too.  Untick everything
            # and there is nothing to refine, so the button that writes
            # "the pieces" into the volume must not stay lit.
            self._update_split_ui(
                accept_enabled=n > 0,
                refine_further_enabled=(
                    self._n_reshape_fragments(ticked=ticked) >= 1),
            )
            # ...and SAY so (audit H3, U4).  At zero ticks Accept greys
            # AND "Refine Fragments..." disappears -- two things leaving
            # the page at once, neither of them with a word about why.
            # The same sentence goes on Accept's tooltip and in the
            # footer hint, which is the rule the snap-back already
            # follows.
            _reason = ("Nothing ticked - tick a piece to Accept it or to "
                       "refine the fragments." if n == 0 else "")
            for _meth in ("set_accept_reason", "set_footer_reason"):
                _fn = getattr(adv, _meth, None)
                if _fn is None:
                    continue
                try:
                    _fn(_reason)
                except Exception:
                    pass

    def _on_reshape_split_anyway(self) -> None:
        """Commit the cut the arbiter declined (owner Q6)."""
        if not getattr(self, "_reshape_active", False):
            return
        self._reshape_override = True
        print("[RESHAPE] Split anyway - re-running unarbitrated")
        self._preview_reshape()

    # ---------------------------------------------------------- the commit --
    def _write_reshape_result(self, shared_data, original_id):
        """Safe-write the ticked pieces.  Returns the commit bookkeeping."""
        from packages.epicycle_annotator.reshape import apply_reshape_to_volume

        result = getattr(self, "_reshape_result", None)
        if result is None or shared_data is None or self._split_roi is None:
            return None
        t0 = _time.perf_counter()
        adv = self._get_advanced_features_widget()
        ticked = set(
            adv.get_checked_split_pieces()
            if adv is not None and hasattr(adv, "get_checked_split_pieces")
            else self._reshape_final_ids.values()
        )
        # An unticked piece maps back to the TARGET id, which is how those
        # voxels rejoin the cell instead of becoming background.
        id_map = {
            pid: (final if final in ticked else int(original_id))
            for pid, final in self._reshape_final_ids.items()
        }
        written, _flags = apply_reshape_to_volume(
            shared_data, self._split_roi, result.result,
            # Only overwrite the target label (`original_id`) or empty space (`0`),
            # so a neighbouring label is never overwritten.  `0` covers the
            # hole the split punched; `original_id` makes the write correct
            # even without one, so the pure layer never depends on it.
            allowed_ids={0, int(original_id)},
            id_map=id_map,
            flags=getattr(result, "flags", {}) or {},
            return_flags=True,
        )
        # Everything past the WRITE is bookkeeping, and fragment refine
        # needs the bookkeeping without the write (U4, 2.7 B2) -- it has
        # already put the pieces in the volume and the user has edited
        # them since, so re-writing the engine partition would re-fill
        # voxels a smooth had just given up.
        return self._reshape_commit_record(
            original_id, ticked=ticked, written=written, t0=t0)

    def _live_piece_stats(self, new_ids, roi):
        """Re-measure the given ids from the volume as it is NOW (M1).

        The engine's `piece_stats` describe the cut it computed; fragment
        refine then lets the user smooth, dilate and grow those pieces,
        so by the time Accept fires the engine's numbers describe a shape
        that is no longer in the volume.  This mirrors
        `_on_annotator_label_split`'s arithmetic: voxel count plus the
        mean of `np.argwhere`, offset by the ROI starts so the centroid
        comes out in VOLUME coordinates -- which is what
        `_piece_stats` promises and what the controller's
        downsample-factor multiply assumes.

        An id that has been edited out of existence is dropped rather
        than reported with zero voxels: a row with no voxels is not a
        cell, and `argwhere([]).mean()` is a nan.
        """
        labels_name = self._active_labels_name
        shared = self.sync_manager.shared_labels.get(labels_name)
        if shared is None or roi is None:
            return []
        crop = shared[roi]
        starts = [int(getattr(sl, "start", 0) or 0) for sl in roi]
        rows = []
        for raw in (new_ids or []):
            lid = int(raw)
            coords = np.argwhere(crop == lid)
            if coords.shape[0] == 0:
                print("[FRAG] {} has no voxels left - not reported as a "
                      "piece".format(lid))
                continue
            centroid = coords.mean(axis=0)
            rows.append({
                "id": lid,
                "voxels": int(coords.shape[0]),
                "centroid_zyx": tuple(
                    float(centroid[i]) + starts[i]
                    for i in range(len(starts))
                ),
            })
        return rows

    def _reshape_commit_record(self, original_id, *, ticked, written, t0):
        """The commit bookkeeping for the current result.  Writes nothing.

        Shared by the ordinary Accept (which has just written the volume
        itself) and by the fragment-refine Accept (whose write happened
        at fragment ENTRY).  Both must reach the
        controller with the SAME record, or a cut taken through fragment
        refine loses its `reshape_flag` sidecar, its `superseded` answer
        and its provenance.
        """
        from packages.epicycle_annotator.reshape import remap_flags

        result = getattr(self, "_reshape_result", None)
        if result is None:
            return None
        adv = self._get_advanced_features_widget()
        ticked = {int(v) for v in ticked}
        id_map = {
            pid: (final if int(final) in ticked else int(original_id))
            for pid, final in self._reshape_final_ids.items()
        }
        flags = remap_flags(getattr(result, "flags", {}) or {}, id_map)
        new_ids = sorted({v for v in id_map.values() if v != int(original_id)})
        superseded = int(original_id) not in set(id_map.values())
        stats = []
        for row in (result.piece_stats or []):
            final = id_map.get(int(row["id"]))
            if final is None or final == int(original_id):
                continue
            stats.append({
                "id": final,
                "voxels": int(row["voxels"]),
                "centroid_zyx": tuple(row["centroid_zyx"]),
            })
        if self._reshape_override:
            # The arbiter kept it whole and a human insisted.  Recorded so
            # that months later an algorithmic split and a human-insisted one
            # are still distinguishable.
            flags = {i: "split-override" for i in new_ids}
        elif not new_ids:
            # A "dividing" verdict: nothing changed in the volume, but the
            # verdict is the tool's most important answer and MUST still be
            # recorded.
            flags = dict(flags or {})
            flags.setdefault(int(original_id), "dividing")
        adv_params = adv.get_split_params() if adv is not None else {}
        print(
            "[RESHAPE] commit written={} new_ids={} superseded={} flags={}"
            .format(written, new_ids, superseded, flags)
        )
        return {
            "written": int(written), "new_ids": new_ids,
            "superseded": bool(superseded), "flags": dict(flags or {}),
            "piece_stats": stats,
            "guide_run": getattr(self, "_reshape_guide_run", ""),
            "preset": adv_params, "t0": t0,
        }

    def _restore_reshape_target(self, reason: str) -> None:
        """Put the cell back exactly as it was, and say so.

        Every exit from a guided session that is not a commit routes here or
        through `_reject_split`, because between Start and Accept the cell
        does not exist in the volume.
        """
        labels_name = self._active_labels_name
        shared = self.sync_manager.shared_labels.get(labels_name)
        if (
            shared is not None
            and self._split_snapshot is not None
            and self._split_roi is not None
        ):
            shared[self._split_roi] = self._split_snapshot
            self._sync_all_views_after_operation(
                roi=self._split_roi, label_id=self._split_label_id)
        print("[RESHAPE] target restored ({})".format(reason))

    def _teardown_refine_for_switch(self) -> None:
        """Leave any Refine worker and staged result behind cleanly.

        Called from the same two places `_teardown_reshape_for_switch` is:
        a patch switch and the annotator close.  Abandon-and-park first,
        because dropping a running QThread aborts the process.
        """
        # a run between two steps has no worker for the
        # abandon hook to catch -- end it here, first.
        if getattr(self, "_refine_recipe_run", None) is not None:
            try:
                self._recipe_end("reset", why="patch switch or close")
            except Exception as exc:  # noqa: BLE001
                print("[RECIPE] end on teardown raised: {!r}".format(exc))
        try:
            self._abandon_refine_worker()
        except Exception as exc:  # noqa: BLE001
            print("[REFINE] abandon on teardown raised: {!r}".format(exc))
        try:
            self._drop_refine_walls()
        except Exception as exc:  # noqa: BLE001
            print("[REFINE] walls drop on teardown raised: {!r}".format(exc))
        self._refine_staged_method = None
        self._refine_steps = []
        _refine_clear_records(self)
        self._fire_refine_lock(False)

    def _teardown_reshape_for_switch(self) -> None:
        """Patch switch / run switch / annotator close.

        Cancels the worker, detaches it, restores the cell, drops the guide
        layer and releases the lock.  Idempotent and never raises -- it runs
        on paths where a raise would leave the surface locked.
        """
        try:
            # UNCONDITIONAL (AUDIT 2026-09-09): a Cancel a moment ago cleared
            # `_reshape_running` but may have left a parked-or-running thread
            # behind, and gating the cleanup on that flag would drop it.
            self._abandon_reshape_worker()
            # Same rule for the marker worker.
            self._abandon_seed_find_worker()
            # Every temporary layer, not just the two named ones: a
            # patch switch must leave nothing of this session behind.
            self._remove_all_temp_layers()
            # ...and the wall picture's INTENT with it (fix pass M5).
            # `_remove_all_temp_layers` takes the layer; `_walls_wanted`
            # is an orchestrator flag it knows nothing about, and it
            # survived into the next patch -- where the first Preview
            # would have rebuilt a picture nobody asked for.  Called
            # UNCONDITIONALLY, beside the sweep, so this sixth exit
            # behaves like the other five instead of depending on the
            # `_split_running` gate below.
            self._drop_walls_picture()
            # The stash holds ROI coordinates, and the ROI belongs to a
            # session that is ending.  Cleared UNCONDITIONALLY (M1): the
            # reset below is gated, and a stash that outlived its gate
            # could paint one object's marker onto the next one.
            self._parked_marker_voxels = {}
            # ...and the gate is `_split_running` alone now, not
            # `_reshape_active and _split_running` (M1).  The predicate
            # that matters is "is the object punched out", and a LEGACY
            # (shape-only) split punches it out exactly the same way --
            # `_start_split` is one method.  With the narrower gate a
            # manual split open at a patch switch or at annotator close
            # left the object missing from the volume with its session
            # state stranded on a widget about to be destroyed.
            # `_restore_reshape_target` reads `_split_snapshot`, which
            # both paths set, so it is path-agnostic.
            if getattr(self, "_split_running", False):
                # The cell is punched out right now -- put it back before the
                # session is torn down, or the user loses it.
                self._restore_reshape_target("teardown")
                try:
                    self._remove_edt_overlay()
                    self._remove_refine_layer()
                except Exception:
                    pass
                self._reset_split_state()
            self._guide_pending = None
            self._guide_thread_cleanup()
            self._remove_guide_layer()
            adv = self._get_advanced_features_widget()
            if adv is not None and hasattr(adv, "clear_split_guided_ui"):
                try:
                    adv.clear_split_guided_ui()
                except Exception:
                    pass
        finally:
            self._fire_reshape_lock(False)
            # Brick B1: the Refine twin, on the same two
            # paths -- a patch switch and the annotator close.  Last, and
            # outside the try, because abandoning a QThread must happen
            # even if the Reshape teardown above raised.
            try:
                self._teardown_refine_for_switch()
            except Exception as exc:  # noqa: BLE001
                print("[REFINE] teardown raised: {!r}".format(exc))

    def _reject_split(self) -> None:
        """Reject the split: restore shared volume from snapshot."""
        # AUDIT 2026-09-09: the guided lock comes off on EVERY exit from
        # this method, including its early returns.  A bare return while
        # holding it leaves Step 3 locked with the session already gone.
        try:
            # The markers were about the object being put back,
            # and the wall picture about the cut being thrown away.
            self._abandon_seed_find_worker()
            self._remove_temp_layer("markers")
            self._drop_walls_picture()
            # --- Fragment Refine Reject path ---
            if self._split_fragment_refine:
                # Clear any pending smooth WITHOUT calling _reject_smooth()
                # (smooth snapshot is intermediate; split snapshot overrides).
                self._reset_smooth_state()
                self._remove_prob_overlay()

                labels_name = self._active_labels_name
                shared_data = self.sync_manager.shared_labels.get(labels_name)
                if shared_data is not None and self._split_snapshot is not None:
                    # U4 (2.7 B3): the WHOLE object comes back -- the
                    # fragments entry wrote and the remainder it kept
                    # with the parent id alike.  The snapshot predates
                    # the hole punch, so one assignment is the answer on
                    # both the guided and the legacy path.
                    roi = self._split_roi
                    shared_data[roi] = self._split_snapshot
                    print("[FRAG] discard - the object is back whole")
                    self._sync_all_views_after_operation(
                        roi=roi, label_id=self._split_label_id
                    )

                # Restore selected label.
                label_id = self._split_label_id
                restore_isolate = self._split_isolate_was_on
                if label_id is not None:
                    self._set_selected_label_all(label_id)
                if restore_isolate and label_id is not None:
                    self._on_isolation_changed({label_id})

                # Reset all state.
                self._reset_split_state()

                self._update_split_ui(
                    seed_count=0,
                    seed_ids=[],
                    split_active=False,
                    edt_enabled=False,
                    start_enabled=True,
                    next_seed_enabled=False,
                    undo_seed_enabled=False,
                    refine_seeds_enabled=False,
                    preview_enabled=False,
                    accept_enabled=False,
                    reject_enabled=False,
                    reject_text="Cancel",
                    review_active=False,
                    refine_further_enabled=False,
                    fragment_refine_active=False,
                    detect_enabled=False,
                    detect_visible=False,
                    status_text="Fragment refine rejected. Original label restored.",
                )
                self._clear_draw_snapshot()
                self._restore_ready_to_draw_state()
                return

            # --- Normal split Reject path ---
            # AUDIT 2026-09-09: Reject is enabled WHILE the solve runs, so it
            # is a live exit from the punched-out window.  Stop the worker
            # FIRST: restoring the ROI while the solve continues would let
            # session A's answer arrive during session B, and the volume it
            # was computed against no longer exists.  Idempotent, so a Reject
            # with no worker costs nothing.
            self._abandon_reshape_worker()
            labels_name = self._active_labels_name
            shared_data = self.sync_manager.shared_labels.get(labels_name)

            if shared_data is None:
                self._update_split_ui(
                    status_text="ERROR: Volume not found. Cannot reject.",
                    edt_enabled=False,
                    detect_enabled=False,
                    detect_visible=False,
                )
                self._remove_refine_layer()
                self._reset_split_state()
                return

            # 1. Restore shared volume from snapshot (fills the hole).
            if self._split_snapshot is not None and self._split_roi is not None:
                shared_data[self._split_roi] = self._split_snapshot

            # 2. Remove EDT overlay + Refine layer.
            #    (processEvents runs inside each removal method automatically.)
            self._remove_edt_overlay()
            self._remove_refine_layer()

            # 3. Unlock main layer + sync data to all views BEFORE setting
            #    visible.  This ensures the 3D render triggered by
            #    visible=True sees the restored volume.
            main_layer = self._layer_by_name(self.viewer.layers, labels_name)
            restore_visible = self._split_main_visible
            if isinstance(main_layer, Labels):
                main_layer.editable = True
            self._sync_all_views_after_operation(
                roi=self._split_roi, label_id=self._split_label_id
            )
            if isinstance(main_layer, Labels):
                main_layer.visible = restore_visible

            # 5. Restore selected_label to original + re-isolate if was active.
            label_id = self._split_label_id
            restore_isolate = self._split_isolate_was_on
            if label_id is not None:
                self._set_selected_label_all(label_id)
            if restore_isolate and label_id is not None:
                self._on_isolation_changed({label_id})

            # 6. Reset state.
            self._reset_split_state()

            self._update_split_ui(
                seed_count=0,
                seed_ids=[],
                split_active=False,
                edt_enabled=False,
                start_enabled=True,
                next_seed_enabled=False,
                undo_seed_enabled=False,
                refine_seeds_enabled=False,
                preview_enabled=False,
                accept_enabled=False,
                reject_enabled=False,
                reject_text="Cancel",
                review_active=False,
                refine_further_enabled=False,
                fragment_refine_active=False,
                detect_enabled=False,
                detect_visible=False,
                status_text="Split rejected. Volume restored.",
            )
            self._clear_draw_snapshot()
            self._restore_ready_to_draw_state()

        # ---------------------------------------------------------------
        # Fragment Refine Mode
        # ---------------------------------------------------------------
        finally:
            # Reshape R1: the guided session ends here, on EVERY exit.
            # This used to sit at the end of the normal path only, so the
            # fragment-refine branch and the "volume not found" bail left
            # the panel-local Find busy LATCHED -- the Find button and the
            # ten marker knobs greyed, the count line stuck on
            # "Finding..." -- with nothing left on screen able to clear
            # them.  Idempotent, so a legacy reject pays nothing.  The
            # GUIDE LAYER stays: the user may run several splits against
            # the same trusted mask in one sitting.
            _adv_r = self._get_advanced_features_widget()
            if _adv_r is not None and hasattr(_adv_r, "clear_split_guided_ui"):
                try:
                    _adv_r.clear_split_guided_ui()
                except Exception:
                    pass
            self._fire_reshape_lock(False)

    def _enter_fragment_refine_mode(self) -> None:
        """Transition from split preview to fragment refine mode.

        Writes split fragments from the refine layer into shared_data,
        removes the refine layer, and enables per-fragment refinement
        using existing refine + interpolation pipelines.
        """
        if not self._split_running or not self._split_previewed:
            return
        if getattr(self, "_split_fragment_refine", False):
            return
        # U4 (2.7 B3).  A solve or a Find in flight is reading and
        # writing the very Refine layer this method is about to consume
        # and then destroy.  Refuse rather than race it.
        if (getattr(self, "_reshape_running", False)
                or getattr(self, "_seed_find_running", False)):
            print("[FRAG] refused - a worker is still running")
            self._update_split_ui(
                status_text=("Wait for the current job to finish before "
                             "refining the pieces."))
            return

        labels_name = self._active_labels_name
        shared_data = self.sync_manager.shared_labels.get(labels_name)
        if shared_data is None:
            return

        roi = self._split_roi
        if self._refine_layer is None or roi is None:
            return

        # 1. Write fragments from refine layer → shared_data.
        #    Safe-write: only overwrite voxels where the original label was.
        #    Exclude the original label ID — residual voxels from watershed
        #    boundary artifacts or auto-smooth shrinkage are treated as gaps.
        refine_roi = self._refine_layer.data[roi]
        original_mask = self._split_snapshot == self._split_label_id
        orig_id = self._split_label_id
        fragment_mask = (refine_roi > 0) & (refine_roi != orig_id)
        write_mask = fragment_mask & original_mask
        shared_data[roi][write_mask] = refine_roi[write_mask]

        # U4 (2.7 B1).  On the GUIDED path an orig-id voxel in the ROI is
        # not residue: `_render_reshape_preview` draws an UNTICKED piece
        # with the target's own id, which is exactly where those voxels
        # go on a normal Accept (`id_map` maps them back to the target).
        # The leftover rule below treated them as gaps and ZEROED them,
        # so unticking a piece and pressing "Refine Fragments..." deleted
        # that tissue.  Unticking means "not a separate object", never
        # "delete it".
        #
        # LEGACY stays exactly as it was: there an orig-id voxel really
        # IS unassigned watershed residue, so `keep_mask` is None and the
        # two statements around it are untouched.
        _guided = getattr(self, "_reshape_result", None) is not None
        keep_mask = (original_mask & (refine_roi == orig_id)) if _guided else None
        n_kept = 0
        if keep_mask is not None and keep_mask.any():
            shared_data[roi][keep_mask] = orig_id
            n_kept = int(keep_mask.sum())

        # Zero out original-label voxels NOT covered by real fragments
        # (includes residual orig-ID voxels + unassigned gaps).
        leftover_mask = original_mask & ~fragment_mask
        if keep_mask is not None:
            leftover_mask = leftover_mask & ~keep_mask
        shared_data[roi][leftover_mask] = 0
        print(
            "[FRAG] entry {} - wrote {} fragment voxel(s), kept {} with the "
            "parent id, dropped {} unassigned".format(
                "guided" if _guided else "legacy",
                int(write_mask.sum()), n_kept, int(leftover_mask.sum()))
        )

        # 2. Collect fragment IDs (exclude original label — it's being split).
        fragment_ids = sorted(
            set(int(v) for v in np.unique(refine_roi) if v > 0 and v != orig_id)
        )
        self._split_fragment_ids = fragment_ids
        self._split_fragment_refine = True

        # U4 (2.7 B2).  The panel's piece list is cleared three steps
        # below, so the two things the commit record still needs are
        # captured HERE -- and `fragment_ids` is the honest source for
        # the ticks, because it is read back off what the write actually
        # put in the volume rather than off a widget.  `None` on the
        # legacy path is what routes its Accept to the split callback.
        if _guided:
            self._reshape_fragment_ticked = list(fragment_ids)
            self._reshape_fragment_written = int(write_mask.sum()) + n_kept
        else:
            self._reshape_fragment_ticked = None
            self._reshape_fragment_written = 0

        # 3. Remove EDT overlay + refine layer.
        #    (processEvents runs inside each removal method automatically.)
        self._remove_edt_overlay()
        self._remove_refine_layer()

        # 4. Unlock + sync views BEFORE setting visible (so 3D render
        #    triggered by visible=True sees correct data).
        #    If isolation will immediately take over, skip the expensive
        #    main_layer.refresh() and defer visibility to the isolation path.
        restore_visible = self._split_main_visible
        main_layer = self._layer_by_name(self.viewer.layers, labels_name)
        if isinstance(main_layer, Labels):
            main_layer.editable = True
        _will_isolate = self._split_isolate_was_on and bool(fragment_ids)
        self._sync_all_views_after_operation(
            roi=roi, skip_3d_refresh=_will_isolate
        )
        if isinstance(main_layer, Labels) and not _will_isolate:
            main_layer.visible = restore_visible

        # Cache the split ROI for fast proxy bounding-box scans during
        # fragment selection (avoids full-volume comparisons).
        self._fragment_refine_roi = roi

        # Restore isolation if it was on before split — isolate the
        # selected fragment so the user focuses on one at a time.
        # Uses standard single-label isolation in both 2D and 3D:
        #   3D → proxy (fast ROI isosurface)
        #   2D → show_selected_label (fast shader uniform)
        # Fragment switching via _on_fragment_selected updates both.
        if _will_isolate:
            self._on_isolation_changed({fragment_ids[0]})

        # 5. Select first fragment.
        if fragment_ids:
            self._set_selected_label_all(fragment_ids[0])

        # 5b. Clear the piece checklist FIRST: nothing used to clear it, so
        # S5 would show the pieces and the fragments at once (#10).
        _adv_fr = self._get_advanced_features_widget()
        if _adv_fr is not None and hasattr(_adv_fr, "update_split_guided_ui"):
            try:
                _adv_fr.update_split_guided_ui(
                    pieces=[], piece_sizes={}, piece_flags={},
                    caption="Refining {} fragments - Accept all commits "
                            "every one of them.".format(len(fragment_ids)),
                    arbiter_message="", split_anyway_visible=False,
                    # The warnings were about the PIECES, which this
                    # call has just replaced with fragments (M4).
                    quality_notes=[],
                )
            except Exception:
                pass

        # 6. Update UI — show fragment list, enable accept/reject.
        self._update_split_ui(
            start_enabled=False,
            next_seed_enabled=False,
            undo_seed_enabled=False,
            refine_seeds_enabled=False,
            preview_enabled=False,
            accept_enabled=True,
            reject_enabled=True,
            reject_text="Discard",
            accept_text="Accept all ({})".format(len(fragment_ids)),
            review_active=True,
            refine_further_enabled=False,
            fragment_refine_active=True,
            fragment_ids=fragment_ids,
            active_fragment_id=fragment_ids[0] if fragment_ids else None,
            # Plain words (audit M2): "fragment refine" is the method
            # name, not something the user chose.  What they pressed was
            # "Refine Fragments...", and what they need to know is that
            # the pieces are in the volume now and the markers are gone
            # until this ends.
            status_text=(
                "Refining the pieces: {} of them, now in the volume. Pick "
                "one and shape it; the markers are put away until you "
                "Accept all or Discard.".format(len(fragment_ids))
            ),
        )
        self._clear_draw_snapshot()
        self._restore_ready_to_draw_state()

    def _on_fragment_selected(self, fragment_id: int) -> None:
        """Handle user clicking a fragment in the list."""
        if not self._split_fragment_refine or fragment_id is None:
            return

        # If smooth is running for a different label, reject it first.
        if self._smooth_running and self._smooth_label_id != fragment_id:
            self._reject_smooth()

        self._set_selected_label_all(fragment_id)

        # Update isolation proxy to show the newly selected fragment in 3D.
        if getattr(self, "_isolation_proxy_active", False):
            labels_name = self._active_labels_name
            main_layer = self._layer_by_name(self.viewer.layers, labels_name)
            if isinstance(main_layer, Labels):
                roi_hint = getattr(self, "_fragment_refine_roi", None)
                self._update_isolation_proxy(
                    fragment_id, main_layer, force=True, roi_hint=roi_hint
                )

    # ---------------------------------------------------------------
    # Standalone Smooth Label
    # ---------------------------------------------------------------

    def _preview_smooth(self) -> None:
        """Preview one Refine step.

        Chained preview: if ``_smooth_running`` is already True the new step
        is applied on top of the previous result without overwriting the
        original snapshot.  Accept undoes all or nothing.

        Two of the seven methods run on a :class:`RefineWorker` -- the
        contour and Settle, measured at 2.4-7.0 s -- and the other five are
        measured under 100 ms and stay synchronous.  Which one it is decides
        where this method ends: at a launch, or at the shared tail
        :meth:`_finish_refine_preview`.
        """
        if self._split_running and not self._split_fragment_refine:
            self._update_smooth_ui(status_text="Reshape is in progress.")
            return
        if getattr(self, "_batch_staged", False) or getattr(
            self, "_batch_running", False
        ):
            self._update_smooth_ui(
                status_text="Finish or close Batch Refine first."
            )
            return
        # A second Preview while a worker is in flight is REFUSED, not
        # queued: the ROI is copied at launch, so a second
        # solve would be built on a volume the first one is about to
        # overwrite.  This covers the Stopping... window too, which is
        # why Preview cannot be pressed until the thread actually exits.
        if getattr(self, "_refine_running", False):
            self._update_smooth_ui(
                status_text="Still working - wait, or press Stop."
            )
            print("[REFINE] preview refused - a worker is in flight")
            return
        # ...and while a SYNCHRONOUS step is inside its processEvents pump
        # (B1b audit C-L3): `_refine_running` is False there, so a click
        # delivered by the pump would start a second step on a volume the
        # first one is about to write.  No push -- the running step owns
        # every widget and is about to write them all.
        if getattr(self, "_refine_sync_busy", False):
            print("[REFINE] preview refused - a synchronous step is running")
            return

        labels_name = self._active_labels_name
        if not labels_name:
            self._update_smooth_ui(status_text="No labels loaded.")
            return

        # Defensive shared_data alignment -- see _ensure_shared_data_aligned
        shared_data = self._ensure_shared_data_aligned(labels_name)
        if shared_data is None:
            self._update_smooth_ui(status_text="No shared label volume found.")
            return

        # Get selected label from the main layer.
        main_layer = self._layer_by_name(self.viewer.layers, labels_name)
        if not isinstance(main_layer, Labels):
            self._update_smooth_ui(status_text="Labels layer not found.")
            return
        label_id = int(main_layer.selected_label)
        if label_id <= 0:
            self._update_smooth_ui(status_text=REFINE_SELECT_OBJECT,
                                   status_kind="warn")
            return

        # Determine refine method from Advanced Features widget.  The dict
        # below is the wire format's fourth holder; it moves with
        # `REFINE_METHOD_KEYS` and `test_refine_tab_contract.py` re-derives
        # the pairs from that constant and walks this source for each one.
        adv = self._get_advanced_features_widget()
        method = "sdf"  # default
        if adv is not None and hasattr(adv, "refine_method_combo"):
            idx = adv.refine_method_combo.currentIndex()
            method = {0: "sdf", 1: "contour", 2: "dilate_erode", 3: "guided_grow", 4: "fill_holes", 5: "flood_fill", 6: "settle"}.get(idx, "sdf")

        # Any method chains onto any other, in any order, with any settings
        # (owner decisions 2026-09-23: the old state 4m refusal
        # and the Settle crossing are both gone).  A later step that needs
        # more room than the first step reserved GROWS the region, exactly
        # (`_grow_refine_region`), rather than being refused.

        params = {}
        if adv is not None and hasattr(adv, "get_refine_params"):
            try:
                params = dict(adv.get_refine_params())
            except Exception as exc:  # noqa: BLE001
                print("[REFINE] could not read the panel: {!r}".format(exc))
                params = {}
        # the five prepare ticks, read WITH the params --
        # the dict carries only the chain they produce.
        prep_ticks = None
        if adv is not None and hasattr(adv, "get_refine_prep_ticks"):
            try:
                prep_ticks = dict(adv.get_refine_prep_ticks())
            except Exception:
                prep_ticks = None

        # The channel refusal, belt-and-braces behind the disabled Preview
        #.  The button is already off with the reason in its
        # tooltip; this is what answers a keyboard activation.
        if params.get("needs_channel") and not params.get("channel"):
            if self._smooth_running:
                # A chain is staged: nothing has been bumped or pushed yet,
                # so the refusal is a sentence and nothing else.
                self._update_smooth_ui(
                    status_text="Pick a channel first - this method reads "
                                "one. {}".format(
                                    self._refine_still_staged_text()),
                    status_kind="warn")
                return
            self._reset_smooth_state(
                "Pick a channel first - this method reads one.")
            return

        if self._smooth_running:
            # --- CHAINED PREVIEW: re-entry while preview is active ---
            # Keep the ORIGINAL snapshot; work on current shared volume state.
            if label_id != self._smooth_label_id:
                self._update_smooth_ui(
                    status_text=("Chaining needs the same object. Accept or "
                                 "Discard first."),
                    status_kind="warn",
                )
                return
            # The chain reuses the FIRST step's ROI, cut with ITS margin.
            # A step that needs more (a later Settle with a wider band)
            # would be clipped by those walls without a word, so the
            # region -- and the snapshot with it -- GROWS first (B1c,
            # subsumes B1b audit C-L4).
            need, have = self._refine_room(method, params)
            if have is None or need > have:
                self._grow_refine_region(shared_data, label_id, need)
            roi = self._smooth_roi  # reuse (or the grown) ROI
            self._smooth_chain_count += 1
        else:
            # --- FIRST PREVIEW: take snapshot ---
            # Settle passes a SCALAR margin -- the largest padding any axis
            # needs for its band -- because `_snapshot_label_roi` takes one
            # number for all three axes and is shared with Reshape's Start
            #.  It is never too small, and at worst
            # a few voxels too generous on the fine axes.
            if method == "settle":
                from packages.epicycle_annotator.reshape.settle import (
                    roi_margin_voxels,
                )

                margin = self._settle_roi_margin(params)
                roi, snapshot = self._snapshot_label_roi(
                    shared_data, label_id, margin=margin)
                print("[SETTLE] roi margin={} band={}um".format(
                    margin, params.get("band_um")))
            else:
                margin = self._refine_step_margin(method, params)
                roi, snapshot = self._snapshot_label_roi(
                    shared_data, label_id, margin=margin)
            # Recorded for EVERY method: the next step compares its own
            # need against it, whichever method opened the chain.
            self._smooth_roi_margin = int(margin)
            self._smooth_snapshot = snapshot
            self._smooth_roi = roi
            self._smooth_label_id = label_id
            self._smooth_running = True
            self._refresh_merge_button_state()
            self._refresh_label_action_buttons_state()
            self._smooth_chain_count = 1
            self._refine_steps = []
            _refine_clear_records(self)

        method_label = {
            "sdf": "Smoothing", "contour": "Sliding the outline of",
            "dilate_erode": "Dilating/eroding",
            "guided_grow": "Guided growing", "fill_holes": "Filling holes",
            "flood_fill": "Flood filling",
            "settle": "Settling the outline of",
        }
        pre_solve = dict(
            preview_enabled=False,
            accept_enabled=False,
            reject_enabled=False,
            status_text=f"{method_label.get(method, 'Refining')} object {label_id}...",
        )
        if self._smooth_chain_count <= 1:
            # Only a FIRST step clears S5.  On a chain the card belongs to
            # the staged result and stays until the next step writes
            # (B1b audit C-M1): a refused step would otherwise
            # leave the page without it -- "Delete object 7" on the button
            # and no "This removes the whole object" under it.
            pre_solve.update(result_note="", result_warning="")
        self._update_smooth_ui(**pre_solve)
        from qtpy.QtWidgets import QApplication
        # C-L3: the pump can deliver a layer event or a click.  Neither
        # may act on this step: the gate stands down and Preview refuses.
        self._refine_sync_busy = True
        try:
            QApplication.processEvents()
        finally:
            self._refine_sync_busy = False

        #  -- captured AFTER the pump: `params` is what the
        # three params-driven branches run on, and the other four read their
        # widgets from here on, so a spin-box edit the pump delivered is
        # recorded exactly when it is used.  Committed only if the step
        # WRITES (`_finish_refine_preview`).
        self._refine_pending_record = _refine_capture_record(
            self, method, params, prep_ticks)

        roi_data = shared_data[roi]

        # ---------------------------------------------------------------
        # The two threaded methods.  Everything they need is
        # read HERE, on the main thread, and the ROI is COPIED, so the
        # worker touches no napari object at all.
        # ---------------------------------------------------------------
        if method in ("contour", "settle"):
            self._launch_refine_method(method, params, roi, roi_data, label_id)
            return

        if method == "sdf":
            # --- SDF Smooth path ---
            from packages.epicycle_annotator.smooth import smooth_label_sdf

            smooth_result, n_mod, msg = smooth_label_sdf(
                roi_data, label_id,
                sigma_z=float(params.get("sigma_z", 3.0)),
                sigma_xy=float(params.get("sigma_xy", 1.5)),
                preserve_topology=bool(params.get("preserve_topology", True)),
            )

        elif method == "dilate_erode":
            # --- Dilate/Erode path (morphological) ---
            from packages.epicycle_annotator.refine import dilate_erode_label

            radius = adv.refine_de_radius_sb.value() if adv is not None and hasattr(adv, "refine_de_radius_sb") else 2
            axis = adv.refine_de_axis_combo.currentText() if adv is not None and hasattr(adv, "refine_de_axis_combo") else "3D"

            smooth_result, n_mod, msg = dilate_erode_label(
                roi_data, label_id, radius=radius, axis=axis,
            )

        elif method == "guided_grow":
            # --- Guided Grow path (probability guided) ---
            from packages.epicycle_annotator.refine import guided_grow

            max_radius = adv.refine_gg_max_radius_sb.value() if adv is not None and hasattr(adv, "refine_gg_max_radius_sb") else 5
            prob_threshold = adv.refine_gg_prob_threshold_sb.value() if adv is not None and hasattr(adv, "refine_gg_prob_threshold_sb") else 0.5
            axis = adv.refine_gg_axis_combo.currentText() if adv is not None and hasattr(adv, "refine_gg_axis_combo") else "3D"

            # The IMAGE mode is retired: it banded on the object's
            # own p10-p90 interior intensity, which admitted 82 % of the
            # region and leaked straight through an open top.  The
            # probability mode measured Dice 0.104 -> 0.838 on the same case.
            gg_prob_roi = self._get_prob_roi(roi)
            if gg_prob_roi is None:
                # Settle chains like any other method now (B1c), so it is
                # offered mid-chain too.
                alternative = "Dilate/Erode or Settle work without one."
                self._refuse_refine_step(
                    "Guided Grow needs the reconstruction's probability map, "
                    "and this patch has none. {}".format(alternative))
                return

            smooth_result, n_mod, msg = guided_grow(
                roi_data, label_id,
                image_roi=None, prob_roi=gg_prob_roi,
                max_radius=max_radius, prob_threshold=prob_threshold,
                axis=axis,
            )

        elif method == "fill_holes":
            # --- Fill Holes path ---
            from packages.epicycle_annotator.refine import fill_holes_label

            axis = adv.refine_fh_axis_combo.currentText() if adv is not None and hasattr(adv, "refine_fh_axis_combo") else "3D"
            smooth_result, n_mod, msg = fill_holes_label(
                roi_data, label_id, axis=axis,
            )

        elif method == "flood_fill":
            # --- Flood Fill path ---
            from packages.epicycle_annotator.refine import flood_fill_label

            axis = adv.refine_ff_axis_combo.currentText() if adv is not None and hasattr(adv, "refine_ff_axis_combo") else "3D"
            max_fill = adv.refine_ff_max_size_sb.value() if adv is not None and hasattr(adv, "refine_ff_max_size_sb") else 50000

            # 3D cursor from subviewer dims (XY->Z, XZ->Y, YZ->X).
            z = int(self.viewer_model_xy.dims.current_step[0])
            y = int(self.viewer_model_xz.dims.current_step[0])
            x = int(self.viewer_model_yz.dims.current_step[0])

            # Convert global -> ROI-local coordinates.
            seed_local = (z - roi[0].start, y - roi[1].start, x - roi[2].start)

            # Which faces of the ROI are INTERIOR walls (the crop stopped
            # there) rather than the volume's own edge (B3 audit, HIGH): a
            # region that runs out through one is open background.
            vshape = shared_data.shape
            walls = []
            for ax_ in range(3):
                start = roi[ax_].start or 0
                stop = roi[ax_].stop if roi[ax_].stop is not None else vshape[ax_]
                walls.extend([start > 0, stop < vshape[ax_]])

            smooth_result, n_mod, msg = flood_fill_label(
                roi_data, label_id, seed_local,
                axis=axis, max_fill_size=max_fill, walls=tuple(walls),
            )
            from packages.epicycle_annotator.refine import (
                FLOOD_NO_CONTACT,
                FLOOD_NOT_ENCLOSED,
                FLOOD_SEED_OUTSIDE,
            )

            if n_mod == 0 and msg in (FLOOD_NO_CONTACT, FLOOD_NOT_ENCLOSED,
                                      FLOOD_SEED_OUTSIDE):
                # State 9, keyed on the reason (`no_contact` and friends).
                print("[REFINE] flood fill refused: {}".format(msg))
                self._finish_refine_preview(
                    method, smooth_result, 0, msg, label_id, roi,
                    nothing_moved=msg)
                return

        else:
            self._refuse_refine_step(f"Unknown refine method: {method}")
            return

        self._finish_refine_preview(
            method, smooth_result, n_mod, msg, label_id, roi)

    # ------------------------------------------------------------------ #
    # The shared tail: safety, the result card, the chain
    # ------------------------------------------------------------------ #

    #: state 9's sentence, keyed on the REASON rather than on nothing.  A
    #: "nothing moved" that does not say why is the failure the UX audit
    #: named.
    _REFINE_NOTHING_MOVED = {
        "sdf": ("Nothing moved: the blur was too small to change this "
                "object's shape at all."),
        "contour": ("Nothing moved: the outline is already where the image "
                    "puts it."),
        "dilate_erode": ("Nothing moved: this object is boxed in by its "
                         "neighbours on every side."),
        "guided_grow": ("Nothing moved: no neighbouring voxel is above the "
                        "probability threshold. Lower it, or raise "
                        "'Max radius'."),
        "fill_holes": "Nothing moved: this object has no enclosed cavity.",
        "flood_fill": ("Nothing moved: there is no empty space connected to "
                       "the crosshair, or it is bigger than 'Max size'."),
        "settle": ("Nothing moved: this object's outline already touches its "
                   "neighbours everywhere, and Settle only moves edges that "
                   "face background."),
    }

    #: The word each step contributes to the S5 breadcrumb.  Moved to
    #: `refine_recipe.STEP_WORD` -- the recipe summaries
    #: use the same words -- and re-exported here unchanged.
    from packages.epicycle_annotator.refine_recipe import (  # noqa: E402
        STEP_WORD as _REFINE_STEP_WORD,
    )

    #: ...and the words S5's result line leads with.
    _REFINE_METHOD_WORD = {
        "sdf": "Smooth", "contour": "Active contour",
        "dilate_erode": "Dilate/Erode", "guided_grow": "Guided Grow",
        "fill_holes": "Fill Holes", "flood_fill": "Flood Fill",
        "settle": "Settle",
    }

    def _refuse_refine_step(self, msg: str) -> None:
        """A refusal between the chain bump and the solve.

        With cross-method chaining allowed (decided 2026-09-23), a
        step can be refused AFTER `_preview_smooth` has bumped the chain
        count -- Guided Grow with no probability map, a wall picture that
        cannot be built, a channel that has left the viewer.  On a chain the
        earlier steps are already WRITTEN into `shared_data`, so
        `_reset_smooth_state` (which does not restore the snapshot) would
        leave them there with Accept and Discard gone and the lock off.
        This mirrors the chained "nothing moved" path instead: the count
        steps back, the staged result and its two buttons stay.
        """
        self._refine_pending_record = None     # recipes audit L5
        if self._smooth_running and self._smooth_chain_count > 1:
            self._smooth_chain_count -= 1
            # The staged window keeps its lock and its locked side views.
            self._fire_refine_lock(True)
            self._lock_subviewers_for_preview()
            self._push_still_staged("{} {}".format(
                msg, self._refine_still_staged_text()))
            print("[REFINE] step refused mid-chain - {} step(s) still "
                  "staged".format(self._smooth_chain_count))
            return
        self._reset_smooth_state(msg)

    def _refine_still_staged_text(self) -> str:
        """The tail every mid-chain refusal carries -- singular or plural
        by what is ACTUALLY still staged (B1b audit UX-L1)."""
        if int(getattr(self, "_smooth_chain_count", 0) or 0) > 1:
            return "The earlier steps are still staged."
        return "The staged result is still there."

    def _push_still_staged(self, status_text: str,
                           result_warning: str = "") -> None:
        """Put the staged result's page back after a step that did not
        write (B1b audit C-M1).

        `_finish_refine_preview` remembers what it pushed for S5 -- note,
        warning, breadcrumb and the primary's words -- and this pushes it
        again, so a refused step cannot leave "Delete object 7" on the
        button without the red block that explains it.  A step's own
        reason, when it has one, leads the warning.

        The side views are LOCKED again here (B1c audit MED-1): a worker's
        finish handler has already called `_restore_ready_to_draw_state`,
        and a staged result must not be painted over -- Discard restores
        the snapshot, and the region growth's exactness rests on every
        write since the first step landing inside the staged ROI.  Every
        chained no-op path (Settle refused, nothing moved, Settle's empty
        write refused, a refused step) comes through here.
        """
        self._lock_subviewers_for_preview()
        self._refine_pending_record = None     # recipes audit L5
        s5 = dict(getattr(self, "_refine_staged_s5", None) or {})
        kw = dict(
            preview_enabled=True, accept_enabled=True, reject_enabled=True,
            busy=False, stop_visible=False, stage_text="",
            rows_enabled=True, status_text=status_text,
        )
        for key in ("accept_text", "breadcrumb", "result_note",
                    "result_warning"):
            if key in s5:
                kw[key] = s5[key]
        if result_warning:
            staged = s5.get("result_warning", "")
            kw["result_warning"] = (
                "{}\n{}".format(result_warning, staged) if staged
                else str(result_warning))
        self._update_smooth_ui(**kw)

    def _refine_step_margin(self, method, params) -> int:
        """The ROI margin one step of ``method`` needs.

        Settle's grows with its band (never below the shared 15); every
        other method uses the shared margin.
        """
        if method == "settle":
            return self._settle_roi_margin(params)
        if method == "contour":
            return self._contour_roi_margin(params)
        return int(REFINE_SHARED_ROI_MARGIN)

    def _contour_roi_margin(self, params) -> int:
        """The contour's margin: its limit, in voxels of the finest axis,
        plus 2 -- never below the shared 15 (B2 audit C-M1).

        *Edge may move* goes to 30 um; at the shared 15 voxels a limit
        above ~13 um flattened the outline against the region's wall
        without a word.  Mirrors `_settle_roi_margin`.
        """
        import math

        limit = float((params or {}).get("contour_limit_um", 3.0))
        finest = max(float(min(self._refine_spacing())), 1e-6)
        return int(max(int(REFINE_SHARED_ROI_MARGIN),
                       math.ceil(limit / finest) + 2))

    def _refine_room(self, method, params):
        """``(need, have)`` for chaining ``method`` onto the staged result.

        ``have`` is the object's ACTUAL clearance (B3 audit, MED): the
        smallest distance, over the six sides, from the object's current
        bounding box to the staged ROI's wall -- infinite on a side where
        that wall is the volume's own edge, because there is nothing to grow
        into there.  It used to be the margin the ROI was CUT with, so two
        steps that each need 15 never grew, and Dilate +10 twice clipped
        silently at the wall 5 voxels out (39,280 voxels against 44,920).

        Measured INSIDE the staged ROI: every write the chain has made
        landed there.  ``None`` when there is no ROI to measure against --
        the region is then grown to ``need``, which is always safe.
        """
        need = int(self._refine_step_margin(method, params or {}))
        roi = getattr(self, "_smooth_roi", None)
        label_id = getattr(self, "_smooth_label_id", None)
        if roi is None or label_id is None:
            return need, None
        try:
            shared = self.sync_manager.shared_labels.get(
                self._active_labels_name)
        except Exception:
            shared = None
        if shared is None:
            return need, None
        mask = np.asarray(shared[roi]) == int(label_id)
        if not mask.any():
            # nothing left to pad around; `_grow_refine_region` would
            # decline anyway
            return need, float("inf")
        have = float("inf")
        for ax_ in range(3):
            other = tuple(a for a in range(3) if a != ax_)
            idx = np.where(np.any(mask, axis=other))[0]
            lo, hi = int(idx[0]), int(idx[-1])
            start = int(roi[ax_].start or 0)
            stop = int(roi[ax_].stop if roi[ax_].stop is not None
                       else shared.shape[ax_])
            if start > 0:                        # an interior wall
                have = min(have, lo)
            if stop < shared.shape[ax_]:
                have = min(have, (stop - start) - 1 - hi)
        return need, have

    def _grow_refine_region(self, shared_data, label_id, need) -> None:
        """Grow the staged ROI and its snapshot EXACTLY.

        The new ROI is the old one united with the object's CURRENT
        extent padded by ``need`` (clipped to the volume), so it contains
        the old ROI and clears the object by ``need``.  The new snapshot
        is the live volume over the new ROI with the old snapshot written
        back into the old ROI's place -- exact, because every write the
        chain has made so far landed inside the old ROI, so outside it
        the live volume IS the pre-chain volume.  Discard therefore puts
        back the whole larger region byte for byte, and Accept hands the
        controller the larger undo region.

        The wall picture is NOT touched: it carries its own ROI
        (`_refine_elevation_roi`), so it stays placed correctly, and the
        next wall step rebuilds it for the grown region.
        """
        old_roi = self._smooth_roi
        old_snap = self._smooth_snapshot
        if old_roi is None or old_snap is None:
            return
        cur_roi = self._label_bbox_roi(shared_data, label_id, int(need))
        if cur_roi is None:
            # The object is gone (an earlier step emptied it).  Nothing to
            # pad around -- and `_snapshot_label_roi`'s "not found" fallback
            # is the WHOLE volume, copied twice (B1c audit LOW-1).  Settle
            # refuses an empty object anyway.
            print("[REFINE] region not grown - object {} is empty".format(
                int(label_id)))
            return
        new_roi = tuple(
            slice(min(int(a.start), int(b.start)), max(int(a.stop), int(b.stop)))
            for a, b in zip(old_roi, cur_roi))
        if new_roi == tuple(old_roi):
            # Already as big as the volume allows; nothing to grow.
            self._smooth_roi_margin = int(need)
            return
        new_snap = np.array(shared_data[new_roi], copy=True)
        inner = tuple(
            slice(int(o.start) - int(n.start), int(o.stop) - int(n.start))
            for o, n in zip(old_roi, new_roi))
        new_snap[inner] = old_snap
        self._smooth_roi = new_roi
        self._smooth_snapshot = new_snap
        self._smooth_roi_margin = int(need)
        print("[REFINE] region grown for margin {}: {} -> {}".format(
            int(need),
            tuple((int(s.start), int(s.stop)) for s in old_roi),
            tuple((int(s.start), int(s.stop)) for s in new_roi)))

    @staticmethod
    def _label_bbox_roi(volume, label_id, margin):
        """The label's bounding box padded by ``margin``, clipped -- the
        ROI `_snapshot_label_roi` would cut, WITHOUT copying the volume.
        None when the label is absent."""
        mask = volume == int(label_id)
        z_any = np.any(mask, axis=(1, 2))
        if not z_any.any():
            return None
        y_any = np.any(mask, axis=(0, 2))
        x_any = np.any(mask, axis=(0, 1))
        margin = max(0, int(margin))
        out = []
        for ax, anyv in enumerate((z_any, y_any, x_any)):
            idx = np.where(anyv)[0]
            out.append(slice(max(0, int(idx[0]) - margin),
                             min(volume.shape[ax], int(idx[-1]) + margin + 1)))
        return tuple(out)

    def _settle_roi_margin(self, params) -> int:
        """The scalar ROI margin a Settle step needs for its band.

        ``max(15, ...)`` -- the other six methods' margin is the floor.
        """
        from packages.epicycle_annotator.reshape.settle import (
            roi_margin_voxels,
        )

        return int(max(15, roi_margin_voxels(
            float(params.get("band_um", 2.0)), self._refine_spacing())))

    def _refine_spacing(self):
        """``(z, y, x)`` in microns, or the (1, 1, 1) fallback.

        The same source Reshape's two launches use.  When the fallback is
        in play every micron number on the page would be a lie, so
        :meth:`_refine_spacing_known` gates the read-outs and S5 carries the
        amber note instead.
        """
        try:
            return tuple(float(s) for s in self._reshape_context()["spacing"])
        except Exception:
            return (1.0, 1.0, 1.0)

    def _refine_spacing_known(self) -> bool:
        ctx = {}
        try:
            ctx = self._reshape_context()
        except Exception:
            ctx = {}
        sp = tuple(ctx.get("spacing") or ())
        return bool(sp) and tuple(float(s) for s in sp) != (1.0, 1.0, 1.0)

    def _refine_breadcrumb_text(self) -> str:
        steps = list(getattr(self, "_refine_steps", []) or [])
        if len(steps) < 2:
            return ""
        return " -> ".join(steps)

    def _finish_refine_preview(self, method, result_roi, n_mod, msg,
                               label_id, roi, extra_note="",
                               extra_warning="", nothing_moved="") -> None:
        """Write the step, then say what it did -- honestly.

        The empty-result gate is here rather than in each branch because
        every method can empty an object and only one of them (Settle) is
        allowed to refuse: the others WRITE the result, so the user sees
        the truth on screen, and the primary button changes its words to
        "Delete object N".
        """
        from packages.epicycle_annotator.smooth import apply_smooth_to_volume

        labels_name = self._active_labels_name
        shared_data = self.sync_manager.shared_labels.get(labels_name)
        if shared_data is None:
            self._reset_smooth_state("No shared label volume found.")
            return

        # The contour's limit is applied INSIDE `refine.contour_refine` now,
        # in the page's own microns and on both sides
        # -- the interim clip that stood here (C-M1) is gone with the
        # voxel constant it read.

        if n_mod == 0:
            # On chained preview with zero change, don't reset -- keep the
            # previous result.  The chain count goes back with it.  A
            # caller that knows WHY nothing moved (the contour's reasons)
            # passes its own sentence; otherwise it is the method's.
            sentence = (str(nothing_moved) if nothing_moved
                        else self._REFINE_NOTHING_MOVED.get(method, str(msg)))
            if self._smooth_chain_count > 1:
                self._smooth_chain_count -= 1  # undo the increment
                self._push_still_staged("{} {}".format(
                    sentence, self._refine_still_staged_text()))
                return
            self._reset_smooth_state(sentence)
            return

        # ---- the write ------------------------------------------------
        if method == "settle":
            from packages.epicycle_annotator.reshape.settle import (
                apply_settle_to_volume,
            )

            written = apply_settle_to_volume(
                shared_data, roi, result_roi, label_id)
            if written < 0:
                # By design, Settle REFUSES to empty the object.
                print("[SETTLE] write refused - the result was empty")
                if self._smooth_chain_count > 1:
                    # earlier steps are still staged and still acceptable
                    # -- so "the object is unchanged" would be false (B1b
                    # audit UX-L4): it is unchanged by THIS step.
                    self._smooth_chain_count -= 1
                    self._push_still_staged(
                        "Nothing moved in that step. {}".format(
                            self._refine_still_staged_text()),
                        result_warning=(
                            "Settling would leave nothing of object {}, so "
                            "nothing was changed.".format(label_id)),
                    )
                    return
                # Nothing is staged, so the SESSION goes down with it --
                # leaving `_smooth_running` True here would strand the tab
                # with a disabled Discard and no way out but an Accept of
                # something that was never written (the shape the Reshape
                # audit named on 2026-09-09).
                self._reset_smooth_state(
                    "Settling would leave nothing of object {}, so nothing "
                    "was changed. Lower 'Edge may move'.".format(label_id))
                return
        else:
            # C-L5: how thick the object was going INTO this step -- the
            # live ROI, read before the write -- not the pre-chain snapshot.
            if method == "sdf":
                try:
                    self._refine_step_input_slices = int(np.count_nonzero(
                        np.any(np.asarray(shared_data[roi]) == label_id,
                               axis=(1, 2))))
                except Exception:
                    self._refine_step_input_slices = 0
            apply_smooth_to_volume(shared_data, roi, result_roi, label_id)

        # Sync all views.
        self._sync_all_views_after_operation(roi=roi, label_id=label_id)

        # C-H1: the staged window is open from HERE, whichever method
        # opened it -- the shape is in `shared_data` and has not been
        # accepted.  Idempotent, so a chained step does not re-fire it.
        self._fire_refine_lock(True)

        self._refine_staged_method = method
        # The wall picture follows the step that WROTE (B1c, audit LOW-2).
        # A worker step's picture waits in `_refine_pending_elevation`
        # until here, so a refused or nothing-moved step leaves the staged
        # step's picture alone.  A step with no picture of its own (Smooth
        # after a Settle, an inside-vs-outside contour) clears the old one:
        # showing it under this step's breadcrumb would be a lie.
        pending = getattr(self, "_refine_pending_elevation", None)
        self._refine_pending_elevation = None
        if method in ("contour", "settle") and pending is not None:
            self._refine_elevation, self._refine_elevation_roi = pending
            staged_kind = getattr(self, "_refine_launch_wall_kind", None)
        else:
            self._refine_elevation = None
            self._refine_elevation_roi = None
            staged_kind = None
        # The legend follows the STAGED picture's wall kind, not the combo
        # (B3 audit): turning the combo afterwards must not relabel it.
        _adv = self._get_advanced_features_widget()
        if _adv is not None and hasattr(_adv, "set_refine_walls_kind"):
            try:
                _adv.set_refine_walls_kind(staged_kind)
            except Exception:
                pass
        steps = list(getattr(self, "_refine_steps", []) or [])
        steps.append(self._REFINE_STEP_WORD.get(method, method))
        self._refine_steps = steps
        # a step is recorded iff it WROTE -- every write
        # passes through here, Settle's included.
        records = list(getattr(self, "_refine_step_records", []) or [])
        records.append(getattr(self, "_refine_pending_record", None))
        self._refine_step_records = records
        self._refine_pending_record = None

        n_left = int((result_roi == label_id).sum())
        word = self._REFINE_METHOD_WORD.get(method, method)
        note = "From {}: {}".format(word, msg)
        if extra_note:
            note = "{}\n{}".format(extra_note, note)
        warning = str(extra_warning or "")
        accept_text = "Accept"
        if self._smooth_chain_count > 1:
            accept_text = "Accept ({})".format(self._smooth_chain_count)

        if n_left == 0:
            # the result IS written, loudly, and Accept stays
            # enabled -- deletion is sometimes intended.  Smooth says which
            # knob did it and how thick the object was: a
            # 3-slice object under a 3.0-voxel blur through depth is the
            # measured case, and "lower the blur" is the one useful advice.
            advice = "press Discard"
            thick = ""
            if method == "sdf":
                advice = "lower 'Blur through depth', or press Discard"
                n_z = int(getattr(self, "_refine_step_input_slices", 0) or 0)
                if n_z > 0:
                    thick = " The object is {} slice{} thick.".format(
                        n_z, "" if n_z == 1 else "s")
            warning = (
                "This removes the whole object. Nothing would be left of "
                "object {}.{} Accept only if that is what you want - "
                "otherwise {}.\nObject {} is held until you "
                "Accept or Discard, so it has not left the list yet."
                .format(label_id, thick, advice, label_id)
            )
            accept_text = "Delete object {}".format(label_id)
            status = "Object {} - nothing is left of it".format(label_id)
            print("[REFINE] EMPTY RESULT - object {} would be deleted"
                  .format(label_id))
        elif self._smooth_chain_count > 1:
            status = "Object {} - {} steps staged".format(
                label_id, self._smooth_chain_count)
        else:
            status = "Object {} - {}".format(label_id, msg)

        if not self._refine_spacing_known():
            warning = (warning + "\n" if warning else "") + (
                "Voxel size unknown - these numbers are in voxels, not "
                "microns."
            )

        breadcrumb = self._refine_breadcrumb_text()
        # Remembered for the session: a later step that does NOT write puts
        # this page back (`_push_still_staged`, C-M1), and the gate puts
        # this status back rather than a generic one (UX-L2).
        self._refine_staged_s5 = dict(
            accept_text=accept_text, breadcrumb=breadcrumb,
            result_note=note, result_warning=warning, status_text=status,
        )
        self._update_smooth_ui(
            preview_enabled=True,
            accept_enabled=True,
            reject_enabled=True,
            accept_text=accept_text,
            preview_text="Preview again",
            preview_reason="",
            breadcrumb=breadcrumb,
            result_note=note,
            result_warning=warning,
            status_text=status,
        )
        self._refresh_refine_walls_toggle()

        # U-M5: staged is 3.26 screens, so the card that just appeared is
        # below the fold.  The page scrolls to it, the way Reshape's does.
        adv = self._get_advanced_features_widget()
        if adv is not None and hasattr(adv, "_focus_panel_stage"):
            try:
                adv._focus_panel_stage(
                    "Refine", reveal=getattr(
                        adv, "_refine_result_container", None))
            except Exception as exc:  # noqa: BLE001
                print("[REFINE] could not reveal the result: "
                      "{!r}".format(exc))

        # Lock subviewers -- drawing during preview would be undone by
        # Reject (snapshot restores pre-refine state).
        self._lock_subviewers_for_preview()

    # ------------------------------------------------------------------ #
    # The wall pipeline, the channel, and the worker
    # ------------------------------------------------------------------ #

    def _launch_refine_method(self, method, params, roi, roi_data,
                              label_id) -> None:
        """Read the inputs on the main thread, then hand them to a worker.

        Both worker methods hand over only CROPS and decisions: the channel
        crop, the probability crop, the wall kind, the prep chain (and for
        the contour the blur).  The channel is prepared and the wall picture
        built ON THE WORKER -- `refine.contour_with_walls` for the contour
        (B2 audit C-M3), `settle.settle_with_walls` for Settle (B3).  Both
        were ~1.5 s of frozen GUI on the owner's patch, before the busy
        strip had even appeared.
        """
        if method == "contour":
            self._launch_refine_contour(params, roi, roi_data, label_id)
            return
        self._launch_refine_settle(params, roi, roi_data, label_id)

    def _launch_refine_settle(self, params, roi, roi_data, label_id):
        """Settle's launch: crops and decisions only (B3, the C-M3 twin).

        The same two refusals the main-thread picture used to make, made
        here BEFORE the worker starts, so a missing input never reaches it.
        """
        kind = str(params.get("wall_kind", "membrane"))
        channel_roi = prob_roi = None
        if kind == "probability":
            prob_roi = self._get_prob_roi(roi)
            if prob_roi is None:
                self._refuse_refine_step(
                    "This wall type reads the reconstruction's probability "
                    "map, and this patch has none. Pick another wall type.")
                return
        elif kind not in ("distance", "skeleton"):
            channel_roi = self._get_refine_channel_roi(roi)
            if channel_roi is None:
                self._refuse_refine_step(
                    "The channel you chose is not in the viewer any more. "
                    "Pick one again.")
                return

        # The ROI is COPIED: the worker must not hold a view into the
        # shared volume, which the main thread may write while it runs.
        labels_copy = np.array(roi_data, copy=True)
        # The picture arrives with the ANSWER (`SettleResult.elevation`) and
        # is parked in `_refine_pending_elevation` by the finish handler --
        # NOT assigned to `_refine_elevation`, which is the STAGED step's
        # picture until this step writes (B1c audit LOW-2).
        self._refine_pending_elevation = None

        from packages.epicycle_annotator.reshape.settle import (
            settle_with_walls,
        )

        kwargs = dict(
            labels_roi=labels_copy, label_id=label_id,
            wall_kind=kind, channel_roi=channel_roi, prob_roi=prob_roi,
            prep_chain=str(params.get("prep_chain", "none")),
            band_um=float(params.get("band_um", 2.0)),
            spacing=self._refine_spacing(),
            solver=str(params.get("solver", "watershed")),
            compactness=float(params.get("compactness", 0.0)),
            n_iter=int(params.get("n_iter", 10)),
            clamp=float(params.get("clamp", 0.5)),
        )
        self._launch_refine_worker(settle_with_walls, kwargs, "settle",
                                   label_id, roi, wants_progress=True)

    def _launch_refine_contour(self, params, roi, roi_data, label_id):
        """The contour's launch: crops and decisions only (C-M3)."""
        mode = int(params.get("contour_mode", 0))
        kind = str(params.get("wall_kind", "membrane"))
        channel_roi = prob_roi = None
        needs_channel = mode == 1 or kind not in (
            "probability", "distance", "skeleton")
        if needs_channel:
            channel_roi = self._get_refine_channel_roi(roi)
            if channel_roi is None:
                self._refuse_refine_step(
                    "The channel you chose is not in the viewer any more. "
                    "Pick one again.")
                return
        if mode == 0 and kind == "probability":
            prob_roi = self._get_prob_roi(roi)
            if prob_roi is None:
                self._refuse_refine_step(
                    "This wall type reads the reconstruction's probability "
                    "map, and this patch has none. Pick another wall type.")
                return
        labels_copy = np.array(roi_data, copy=True)
        # The picture arrives with the ANSWER (`details["elevation"]`) and
        # is parked in `_refine_pending_elevation` by the finish handler.
        self._refine_pending_elevation = None
        # The contour is the one countable job on this tab: it
        # reports each plane, and Stop is honoured BETWEEN planes through
        # the same event the worker's cancel() sets.
        import threading

        cancel_event = threading.Event()
        fn, kwargs = self._refine_contour_call(
            params, labels_copy, label_id, channel_roi, prob_roi,
            cancel_event=cancel_event)
        self._launch_refine_worker(fn, kwargs, "contour", label_id, roi,
                                   cancel_event=cancel_event,
                                   wants_progress=True)

    def _refine_contour_call(self, params, labels_copy, label_id,
                             channel_roi=None, prob_roi=None,
                             cancel_event=None):
        """``(fn, kwargs)`` for the merged Active contour.

        Everything is read off ``params`` -- ``get_refine_params()``, by
        index through the panel's module constants -- so the worker is
        handed plain numbers and arrays and imports nothing from Qt-land.
        ``refine.contour_with_walls`` prepares the channel and builds the
        wall picture ON THE WORKER, then runs ``refine.contour_refine``:
        the gate passed explicitly, ACWE restricted to its band, the limit
        two-sided in microns, a plane orientation traced slice by slice.
        """
        from packages.epicycle_annotator.refine import contour_with_walls
        from packages.epicycle_annotator.reshape.spec import (
            DEFAULT_SIZE_BUDGET,
        )

        p = params or {}
        return contour_with_walls, dict(
            volume_roi=labels_copy, label_id=int(label_id),
            mode=int(p.get("contour_mode", 0)),
            wall_kind=str(p.get("wall_kind", "membrane")),
            channel_roi=channel_roi, prob_roi=prob_roi,
            prep_chain=str(p.get("prep_chain", "none")),
            blur_um=float(p.get("contour_blur_um", 2.0)),
            spacing=self._refine_spacing(),
            limit_um=float(p.get("contour_limit_um", 3.0)),
            iterations=int(p.get("contour_iterations", 100)),
            balloon=float(p.get("contour_balloon", 0.0)),
            gate=float(p.get("contour_gate", 0.30)),
            lambda1=float(p.get("contour_lambda1", 1.0)),
            lambda2=float(p.get("contour_lambda2", 1.0)),
            sharpness=float(p.get("contour_sharpness", 100.0)),
            orientation=str(p.get("contour_axis", "3D")),
            sampling=int(p.get("contour_sampling", 0)),
            share_pct=float(p.get("contour_share_pct", 25.0)),
            n_slices=int(p.get("contour_n_slices", 12)),
            rebuild_um=float(p.get("contour_rebuild_um", 1.5)),
            size_budget=int(DEFAULT_SIZE_BUDGET),
            cancel_event=cancel_event,
        )

    def _launch_refine_worker(self, fn, kwargs, method, label_id, roi,
                              cancel_event=None,
                              wants_progress=False) -> None:
        """Start the worker, and take the drawing lock BEFORE it starts.

        The labels ROI is copied at launch, so anything painted between
        Preview and the result would be silently overwritten when the
        result is applied -- the same class of loss `_lock_subviewers_for_
        preview` already exists to prevent, in a window that only a worker
        opens.  The lock is released on all four exits.
        """
        from qtpy.QtCore import QThread

        from napari_epicycle.annotator.reshape_worker import RefineWorker

        ctx = self._reshape_context()
        self._refine_token = int(getattr(self, "_refine_token", 0)) + 1
        token = self._refine_token
        self._refine_launch_labels_name = self._active_labels_name
        self._refine_launch_patch = ctx.get("patch_name")
        self._refine_launch_run_epoch = ctx.get("run_epoch")
        self._refine_launch_roi = tuple(roi)
        self._refine_launch_label = int(label_id)
        self._refine_launch_wall_kind = (kwargs or {}).get("wall_kind")
        self._refine_running = True
        self._refine_busy_kind = method
        self._refine_t0 = _time.perf_counter()
        self._refine_plane_t0 = None
        self._fire_refine_lock(True)

        from napari_epicycle.annotator.reshape_worker import stage_text

        # The stage line names the FIRST stage the worker will report, so
        # the strip says something true from the click on.  It no longer
        # carries the drawing-off sentence (owner decision on B2 audit
        # UX-M7): the one-line strip clipped it, and the side views are
        # visibly locked.
        _mode = (kwargs or {}).get("mode")
        _first = ("prepare" if method == "contour"
                  and _mode in (1, "brightness") else "elevation")
        self._update_smooth_ui(
            preview_enabled=False, accept_enabled=False, reject_enabled=False,
            busy=True, busy_progress=(0, 0),
            stop_visible=True, stop_enabled=True,
            stage_text=stage_text(_first),
            rows_enabled=False,
        )
        # The picture on screen belongs to the step that is being replaced.
        self._drop_refine_walls_layer()
        self._lock_subviewers_for_preview()

        try:
            self._refine_thread = QThread()
            self._refine_worker = RefineWorker(
                fn, kwargs, kind=method, cancel_event=cancel_event,
                wants_progress=bool(wants_progress))
            self._refine_worker._epicycle_token = token
            self._refine_worker.moveToThread(self._refine_thread)
            self._refine_thread.started.connect(self._refine_worker.run)
            self._refine_worker.progress.connect(self._on_refine_progress)
            self._refine_worker.message.connect(self._on_refine_message)
            self._refine_worker.finished.connect(self._on_refine_finished)
            self._refine_worker.error.connect(self._on_refine_error)
            self._refine_thread.finished.connect(
                self._on_refine_thread_finished)
            # the run's one judge for this step is
            # bound HERE, to (token, step), before the thread starts -- a
            # fast Settle could emit before a later connect and strand the
            # run.  Connected after the main handler, so Qt delivers it
            # after that handler has written (or refused).
            _run = getattr(self, "_refine_recipe_run", None)
            if _run is not None and _run.get("phase") == "previewing":
                import functools

                _slot = functools.partial(self._recipe_on_worker_done,
                                          _run["token"], _run["i"])
                self._refine_worker.finished.connect(_slot)
                self._refine_worker.error.connect(_slot)
            self._refine_thread.start()
            # Recipes audit L3: "waiting" only once start() has returned --
            # a start that raises must not leave the run waiting for a
            # worker that never ran.
            if _run is not None and _run.get("phase") == "previewing":
                _run["phase"] = "waiting"
            print("[REFINE] worker started token={} method={} object={}"
                  .format(token, method, label_id))
        except Exception as exc:  # noqa: BLE001 - never strand a locked surface
            self._refine_running = False
            self._refine_busy_kind = None
            self._refine_thread = None
            self._refine_worker = None
            print("[REFINE] launch FAILED: {!r}".format(exc))
            _run = getattr(self, "_refine_recipe_run", None)
            # C-M2: on a chain the earlier steps are WRITTEN, so the staged
            # window -- its lock, its locked side views, Accept and Discard
            # -- stays; `_refuse_refine_step` steps the count back.  On a
            # first step nothing was written and the session ends, which is
            # what releases the lock.
            if not (self._smooth_running and self._smooth_chain_count > 1):
                self._restore_ready_to_draw_state()
            self._refuse_refine_step("Could not start: {}".format(exc))
            if _run is not None and getattr(
                    self, "_refine_recipe_run", None) is _run:
                from packages.epicycle_annotator import refine_recipe as rr

                _k, _n = _run["i"] + 1, len(_run["steps"])
                _w = self._REFINE_METHOD_WORD.get(method, method)
                self._recipe_end(
                    "error", why="the worker could not start: {!r}".format(
                        exc),
                    stage=self._recipe_stopped_stage(_run, _w),
                    status=rr.S1_NOT_STARTED.format(k=_k, n=_n, word=_w))

    def _refine_is_stale(self, result_token=None) -> bool:
        """A lock stops a SECOND op; it does nothing about a LATE one.

        The Reshape quadruple, copied rather than re-invented:
        the dispatch token, the labels layer, the patch and the run epoch,
        plus "is there still a session".  The token is the one that catches
        Discard-then-Preview-again on the SAME patch, run and layer, which
        looks to the other three exactly like the current run's own answer.

        It is compared against the token STAMPED ON THE WORKER, never
        against ``self._refine_token`` on both sides -- that comparison is
        a tautology, because a second launch has already bumped it.
        """
        if result_token is not None and result_token != self._refine_token:
            print("[REFINE] STALE finish discarded - token {} != {}".format(
                result_token, self._refine_token))
            return True
        if self._active_labels_name != getattr(
                self, "_refine_launch_labels_name", None):
            print("[REFINE] STALE finish discarded - labels layer changed")
            return True
        ctx = self._reshape_context()
        if ctx.get("patch_name") != getattr(
                self, "_refine_launch_patch", None):
            print("[REFINE] STALE finish discarded - patch changed")
            return True
        if ctx.get("run_epoch") != getattr(
                self, "_refine_launch_run_epoch", None):
            print("[REFINE] STALE finish discarded - run changed")
            return True
        if not getattr(self, "_smooth_running", False):
            print("[REFINE] STALE finish discarded - refine session gone")
            return True
        return False

    def _abandon_refine_worker(self) -> str:
        """Cancel the method and clear the active thread slot.

        Used by every path that leaves the staged window while a worker may
        still be running.  It is NOT enough to clear ``_refine_running``:
        the worker keeps going, and its answer would land in the NEXT
        refine session, which is indistinguishable from a correct one
        without the dispatch token.  Idempotent; never raises.
        """
        worker = getattr(self, "_refine_worker", None)
        had = worker is not None or getattr(self, "_refine_thread", None)
        if worker is not None:
            try:
                worker.cancel()
            except Exception:
                pass
        self._refine_running = False
        self._refine_busy_kind = None
        self._refine_stopping = False
        self._refine_thread_cleanup()
        # a run WAITING on the worker being
        # abandoned would wait for ever -- its signals were just parked.
        # Exact: a step's own endings clean the thread up before they reach
        # here, so `had` is false for them and the run keeps its judge.
        _run = getattr(self, "_refine_recipe_run", None)
        if _run is not None and had and _run.get("phase") == "waiting":
            self._recipe_end("reset", why="the worker was abandoned")
        return "abandoned" if had else "idle"

    def _refine_thread_cleanup(self) -> None:
        """Join the worker -- or PARK it if it will not stop.

        A stage already entered runs to completion (``random_walker``,
        ``diffuse_labels3D``, and in practice a multi-second contour), so a
        teardown can arrive while the thread is genuinely still running.
        Dropping the reference then is FATAL, not untidy: a QThread garbage-
        collected while running aborts the whole process with "QThread:
        Destroyed while thread is still running" (observed live 2026-09-08).
        So an unstoppable thread is PARKED -- its signals disconnected,
        since its answer is already being abandoned -- and held alive until
        it finishes on its own.
        """
        th = getattr(self, "_refine_thread", None)
        worker = getattr(self, "_refine_worker", None)
        self._refine_thread = None
        self._refine_worker = None
        if th is None:
            return
        try:
            th.quit()
            if th.wait(3000):
                return
        except Exception:
            return

        print("[REFINE] worker will not stop in time - parking the thread "
              "(dropping it here would abort the process)")
        parked = getattr(self, "_refine_parked_threads", None)
        if parked is None:                      # a bare object in a test
            parked = []
            self._refine_parked_threads = parked
        for sig in ("progress", "message", "finished", "error"):
            try:
                getattr(worker, sig).disconnect()
            except Exception:
                pass
        parked.append((th, worker))

    def _fire_refine_lock(self, active: bool) -> None:
        """Set / clear the controller's ``_refine_running``.  Idempotent.

        Raised while a worker is in flight OR a result is staged, because a
        staged preview is ALREADY written into ``shared_data`` -- so Save
        Curation, Commit Triage and a run switch must be unavailable for
        the whole staged window.
        """
        active = bool(active)
        if active == bool(getattr(self, "_refine_lock_fired", False)):
            return
        self._refine_lock_fired = active
        print("[REFINE-LOCK] {}".format("on" if active else "off"))
        cb = getattr(self, "_refine_lock_cb", None)
        if cb is None:
            return
        try:
            cb(active)
        except Exception as exc:  # noqa: BLE001 - never strand the surface
            print("[REFINE] lock callback raised: {!r}".format(exc))

    def _cancel_refine_preview(self) -> None:
        """Stop is not instant, and the panel must not pretend it is.

        ``random_walker`` and ``diffuse_labels3D`` cannot be interrupted
        once entered, so Stop sets the cancel event and the strip enters
        state 3s: the Stop button disables itself, the bar keeps running and
        the stage line reads "Stopping..." until the thread actually exits.
        Preview stays disabled for that whole window, so a second solve
        cannot start while the first still holds the GIL and the ROI.
        """
        if not getattr(self, "_refine_running", False):
            # Stop stays live for a whole recipe, so
            # between two steps -- no worker -- it ends the run exactly as a
            # Stop does, putting back the whole session snapshot.
            _run = getattr(self, "_refine_recipe_run", None)
            if _run is not None:
                if getattr(self, "_refine_sync_busy", False):
                    # Recipes audit H1: this click was delivered by a
                    # synchronous step's processEvents pump.  That step is
                    # still on the stack and WRITES when the pump returns,
                    # so ending the session here would let it write with
                    # no snapshot behind it.  The Stop is REQUESTED and
                    # honoured by `_recipe_next` once the step returns.
                    _run["stop_requested"] = True
                    print("[RECIPE] Stop pressed inside a step - honoured "
                          "when the step returns")
                    self._update_smooth_ui(stop_enabled=False,
                                           stage_text="Stopping...")
                    return
                print("[RECIPE] Stop pressed between steps")
                self._on_refine_stopped()
            return
        if getattr(self, "_refine_stopping", False):
            return
        self._refine_stopping = True
        print("[REFINE] Stop pressed - the answer will be thrown away")
        worker = getattr(self, "_refine_worker", None)
        if worker is not None:
            try:
                worker.cancel()
            except Exception:
                pass
        # State 3s, and NOTHING ELSE: the button disables itself, the bar
        # keeps running and the stage line holds "Stopping..." until the
        # thread actually exits.  Preview stays disabled for that whole
        # window, so a second solve cannot start while the first still
        # holds the GIL and the ROI.
        self._update_smooth_ui(
            preview_enabled=False,
            accept_enabled=False,
            # The button disables ITSELF: pressing Stop twice
            # cannot make an uninterruptible call stop sooner, and a live
            # button that does nothing is a button that looks broken.
            stop_enabled=False, stop_visible=True,
            busy=True,
            stage_text="Stopping...",
            status_text="Stopping...",
        )

    def _on_refine_thread_finished(self) -> None:
        """The QThread's own ``finished``.

        The backstop for the one case no worker signal covers: a cancel that
        lands BEFORE ``run()`` starts makes the worker return without
        emitting anything at all, and the strip would hold "Stopping..."
        for ever.
        """
        if not getattr(self, "_refine_stopping", False):
            return
        self._refine_thread_cleanup()
        self._on_refine_stopped()

    def _on_refine_stopped(self) -> None:
        """Every piece of staged state goes down together.

        A Stop that leaves them set makes the next Preview chain onto a
        result that never existed.

        And the object goes back.  Stop's tooltip says "The object is not
        changed", so a chain staged before the stopped step is put back too
        -- the snapshot is the whole session's, and half a session restored
        would be a third thing for the user to reason about.
        """
        self._refine_stopping = False
        labels_name = self._active_labels_name
        shared_data = None
        try:
            shared_data = self.sync_manager.shared_labels.get(labels_name)
        except Exception:
            shared_data = None
        if (shared_data is not None and self._smooth_snapshot is not None
                and self._smooth_roi is not None):
            shared_data[self._smooth_roi] = self._smooth_snapshot
            try:
                self._sync_all_views_after_operation(
                    roi=self._smooth_roi, label_id=self._smooth_label_id)
            except Exception as exc:  # noqa: BLE001
                print("[REFINE] restore after Stop raised: {!r}".format(exc))
            print("[REFINE] Stop restored the object from the snapshot")
        self._refine_running = False
        self._refine_busy_kind = None
        self._smooth_running = False
        self._smooth_snapshot = None
        self._smooth_roi = None
        self._smooth_label_id = None
        self._smooth_chain_count = 0
        self._refine_steps = []
        _refine_clear_records(self)
        self._refine_staged_method = None
        self._refine_staged_s5 = None
        self._refine_elevation = None
        self._refine_pending_elevation = None     # C-L4
        self._fire_refine_lock(False)
        self._drop_refine_walls()
        self._refresh_merge_button_state()
        self._refresh_label_action_buttons_state()
        self._update_smooth_ui(
            preview_enabled=True, accept_enabled=False, reject_enabled=False,
            busy=False, stop_visible=False, stage_text="",
            rows_enabled=True, preview_text="Preview", accept_text="Accept",
            result_note="", result_warning="", breadcrumb="",
            status_text="Stopped. The object is unchanged.",
        )
        self._restore_ready_to_draw_state()
        # the last statement -- Stop in R4w, its thread
        # backstop, and Stop between steps all come through here.
        if getattr(self, "_refine_recipe_run", None) is not None:
            self._recipe_on_stopped()

    def _on_refine_progress(self, stage, i, n) -> None:
        # C-L1: a progress signal queued before Stop must not overwrite
        # "Stopping..." on the strip.
        if getattr(self, "_refine_stopping", False):
            return
        if self._refine_is_stale(result_token=self._result_token()):
            return
        from napari_epicycle.annotator.reshape_worker import stage_text

        text = stage_text(stage)
        if str(stage) == "contour" and int(n) > 0:
            # Determinate: "Plane 4 of 11 - about 1 s left",
            # the estimate being (n - done) x (elapsed / done) once one
            # plane is done, and blank before.
            now = _time.perf_counter()
            i, n = int(i), int(n)
            if i <= 1 or getattr(self, "_refine_plane_t0", None) is None:
                self._refine_plane_t0 = now
            done = i - 1
            left = ""
            if done >= 1:
                per = (now - self._refine_plane_t0) / float(done)
                left = " \u2014 about {} s left".format(
                    max(1, int(round((n - done) * per))))
            text = "Plane {} of {}{}".format(i, n, left)
        # The stage alone: the drawing-off sentence came off the strip
        # (owner decision on B2 audit UX-M7) -- it was clipped behind the
        # plane count, and the side views are visibly locked.
        self._update_smooth_ui(
            busy=True, busy_progress=(int(i), int(n)),
            stage_text=text,
        )

    def _on_refine_message(self, text) -> None:
        self._refine_live_messages = list(
            getattr(self, "_refine_live_messages", [])) + [str(text)]

    def _on_refine_error(self, text) -> None:
        if getattr(self, "_refine_stopping", False):
            print("[REFINE] error discarded - Stop was pressed")
            self._refine_thread_cleanup()
            self._on_refine_stopped()
            return
        if self._refine_is_stale(result_token=self._result_token()):
            # L4: join it, like the finish handler's stale branch does.
            # Leaving the QThread in the slot means the next launch
            # overwrites the reference and the parking guard never runs --
            # and a QThread dropped while running aborts the process.
            self._refine_thread_cleanup()
            return
        print("[REFINE] worker error: {}".format(text))
        self._refine_running = False
        self._refine_busy_kind = None
        self._refine_thread_cleanup()
        self._restore_ready_to_draw_state()
        reason = str(text).splitlines()[0] if text else ""

        if self._smooth_chain_count > 1:
            # A chain keeps what it already staged: those steps are written,
            # acceptable, and have nothing to do with this one failing.
            self._smooth_chain_count -= 1
            self._lock_subviewers_for_preview()
            self._push_still_staged(
                "That step could not be run. {}".format(
                    self._refine_still_staged_text()),
                result_warning=reason)
            return

        # C-H2: an UNCHAINED error ends the session.  `_smooth_chain_count`
        # is 1 here, so the old `> 1` did not decrement and the old `<= 0`
        # never fired: Accept and Discard stayed dead and `_refine_running`
        # stayed true on the controller for the rest of the session.
        self._update_smooth_ui(
            busy=False, stop_visible=False, stage_text="", rows_enabled=True,
            result_warning=reason,
        )
        self._reset_smooth_state("That step could not be run.")

    def _on_refine_finished(self, outcome) -> None:
        """The worker's answer, if it is still this session's."""
        token = self._result_token()
        if getattr(self, "_refine_stopping", False):
            # The user pressed Stop.  The answer arrived anyway -- the
            # solve cannot be interrupted once entered -- and it is
            # THROWN AWAY, not applied.
            print("[REFINE] answer discarded - Stop was pressed")
            self._refine_thread_cleanup()
            self._on_refine_stopped()
            return
        if self._refine_is_stale(result_token=token):
            self._refine_thread_cleanup()
            if not getattr(self, "_smooth_running", False):
                # Recipes audit H1: the session this answer belonged to is
                # gone.  Nothing else will ever clear the flag or the lock
                # the launch took, so a stale answer on a dead session
                # releases them (a live session belongs to a newer launch,
                # which keeps its own).
                self._refine_running = False
                self._refine_busy_kind = None
                self._fire_refine_lock(False)
            return
        self._refine_running = False
        method = getattr(outcome, "kind", "") or "settle"
        self._refine_busy_kind = None
        self._refine_thread_cleanup()
        self._update_smooth_ui(
            busy=False, stop_visible=False, stage_text="", rows_enabled=True)
        self._restore_ready_to_draw_state()

        roi = getattr(self, "_refine_launch_roi", None) or self._smooth_roi
        label_id = int(getattr(self, "_refine_launch_label", 0)
                       or self._smooth_label_id or 0)
        result = getattr(outcome, "result", None)
        if result is None:
            # C-L1: mid-chain the earlier steps are written and stay staged.
            self._refuse_refine_step("That step produced no answer.")
            return

        if method == "settle":
            # B3: the wall picture was built on the worker and rides the
            # answer, exactly as the contour's does (C-M3).
            elev = getattr(result, "elevation", None)
            self._refine_pending_elevation = (
                (elev, tuple(roi)) if elev is not None else None)
            self._finish_settle_result(result, label_id, roi)
            return

        try:
            result_roi, n_mod, msg = result
        except Exception:
            self._refuse_refine_step("That step produced no answer.")
            return
        details = dict(getattr(result, "details", None) or {})
        # C-M3: the wall picture was built on the worker and rides the
        # answer; it becomes the pending picture exactly where a Settle's
        # would have been set at launch, so the walls tick works the same.
        elev = details.pop("elevation", None)
        self._refine_pending_elevation = (
            elev, tuple(roi) if elev is not None else None)
        reason = details.get("reason")
        if reason == "too_large":
            # State 8, not 9: refused before it ran, amber with its reason.
            self._refine_pending_elevation = None
            chained = int(self._smooth_chain_count or 0) > 1
            self._refuse_refine_step(str(msg))
            if not chained:
                self._update_smooth_ui(
                    status_text=REFINE_NOTHING_TO_SHOW, status_kind="warn",
                    result_warning=str(msg))
            return
        nothing = ""
        if reason == "push_below_gate":
            # the SIGNED push: "-0.2" is not "0.2" (B2 audit UX-L)
            nothing = REFINE_CONTOUR_PUSH_BELOW_GATE.format(
                push=float(details.get("balloon", 0.0)),
                gate=float(details.get("gate", 0.30)))
        elif reason == "limit":
            nothing = REFINE_CONTOUR_AT_LIMIT.format(
                limit=float(details.get("limit_um", 3.0)))
        elif reason == "fence":
            nothing = REFINE_CONTOUR_FENCED
        elif reason == "missing":
            nothing = str(msg)
        warning = ""
        if details.get("hit_limit") and int(n_mod) > 0:
            warning = (
                "The outline reached the {:.1f} um limit on one side - raise "
                "'Edge may move' if the wall is further out.".format(
                    float(details.get("limit_um", 3.0))))
        traced = str(details.get("traced_text") or "")
        print("[REFINE] contour {} {} reason={} n_mod={} traced={}/{} "
              "planes={:.2f}s rebuild={:.2f}s".format(
                  details.get("mode"), details.get("orientation"), reason,
                  int(n_mod), details.get("traced"), details.get("total"),
                  float(sum(details.get("plane_seconds") or [0.0])),
                  float(details.get("rebuild_seconds") or 0.0)))
        self._finish_refine_preview(
            "contour", result_roi, int(n_mod), str(msg), label_id, roi,
            extra_note=traced, extra_warning=warning, nothing_moved=nothing)

    def _finish_settle_result(self, res, label_id, roi) -> None:
        """Settle's own numbers, in the order  puts them.

        The border fraction leads, because it is the number that decides
        whether the tool can do anything at all and no user can guess it.
        """
        if getattr(res, "status", "") == "refused":
            # Lifecycle LOW-2: a refused step's picture must not wait for a
            # later write to pick it up.
            self._refine_pending_elevation = None
            reason = "\n".join(res.messages) or "Nothing was changed."
            print("[SETTLE] refused: {}".format("; ".join(res.messages)))
            if self._smooth_chain_count > 1:
                # The earlier steps are written and still staged, so "the
                # object is unchanged" would be false -- only THIS step
                # moved nothing (B1b audit UX-L4).
                self._smooth_chain_count -= 1
                self._push_still_staged(
                    "Nothing moved in that step. {}".format(
                        self._refine_still_staged_text()),
                    result_warning=reason)
                return
            self._update_smooth_ui(
                preview_enabled=True,
                accept_enabled=False,
                reject_enabled=False,
                result_warning=reason,
                status_text="Nothing to show - the object is unchanged.",
            )
            self._smooth_running = False
            self._smooth_snapshot = None
            self._smooth_roi = None
            self._smooth_label_id = None
            self._smooth_chain_count = 0
            self._fire_refine_lock(False)
            self._refresh_merge_button_state()
            self._refresh_label_action_buttons_state()
            return

        pct = int(round(100.0 * res.border_background_fraction))
        lead = ("{} % of this object's border faces background; the rest "
                "touches neighbours and was left alone.".format(pct))
        if self._refine_spacing_known():
            moved = ("the outline moved on {:,} voxels - {:,} out, {:,} in, "
                     "on average {:.1f} um and at most {:.1f} um.".format(
                         res.n_modified, res.n_grown, res.n_shrunk,
                         res.mean_move_um, res.max_move_um))
        else:
            moved = ("the outline moved on {:,} voxels - {:,} out, {:,} in."
                     .format(res.n_modified, res.n_grown, res.n_shrunk))
        warning = ""
        if res.hit_band_limit:
            warning = (
                "The outline reached the {:.1f} um limit on one side - raise "
                "'Edge may move' if the wall is further out.".format(
                    float(self._refine_band_um())))
        elif res.n_modified:
            lead = lead + ("  The outline stopped on the wall signal, not at "
                           "the limit - 'Edge may move' is not what is "
                           "holding it.")
        # {:.6f}, not {:.4f} (L6): the B1 driver measures this number
        # independently and compares it, and four places left the
        # comparison inside the printing error rather than inside the
        # arithmetic.
        print("[SETTLE] border bg={} held={} fraction={:.6f} moved={} "
              "grown={} shrunk={} mean={:.3f}um max={:.3f}um".format(
                  res.n_border_background, res.n_border_held,
                  res.border_background_fraction, res.n_modified,
                  res.n_grown, res.n_shrunk, res.mean_move_um,
                  res.max_move_um))
        # B3 audit: when nothing moved, the ENGINE knows why -- "already
        # touches its neighbours everywhere" and "the wall signal already
        # runs along this object's outline" are different answers, and the
        # generic table entry is only the first.
        nothing = ""
        if not int(res.n_modified) and res.messages:
            nothing = str(res.messages[-1])
        self._finish_refine_preview(
            "settle", res.result_roi, int(res.n_modified), moved,
            label_id, roi, extra_note=lead, extra_warning=warning,
            nothing_moved=nothing)

    def _refine_band_um(self) -> float:
        adv = self._get_advanced_features_widget()
        try:
            return float(adv.refine_settle_band_sb.value())
        except Exception:
            return 2.0

    # ------------------------------------------------------------------ #
    # The walls overlay on Refine
    # ------------------------------------------------------------------ #

    def _active_elevation(self):
        """Whichever tool has an elevation staged right now."""
        elev = getattr(self, "_refine_elevation", None)
        if elev is not None:
            return elev
        result = getattr(self, "_reshape_result", None)
        return getattr(result, "elevation", None) if result is not None else None

    def _active_display_roi(self):
        """...and the ROI that elevation was built for."""
        if getattr(self, "_refine_elevation", None) is not None:
            return getattr(self, "_refine_elevation_roi", None)
        return getattr(self, "_split_roi", None)

    def _refresh_refine_walls_toggle(self) -> None:
        """Offer the tick only when there is a picture behind it."""
        adv = self._get_advanced_features_widget()
        if adv is None or not hasattr(adv, "set_refine_walls_toggle_enabled"):
            return
        from napari_epicycle.annotator.advanced_features_widget import (
            WALLS_TOGGLE_DISABLED,
            WALLS_TOGGLE_LAST_STEP,
            WALLS_TOGGLE_NO_WALLS,
        )

        staged = getattr(self, "_refine_staged_method", None)
        has = getattr(self, "_refine_elevation", None) is not None
        if staged is None:
            adv.set_refine_walls_toggle_enabled(False, WALLS_TOGGLE_DISABLED)
            return
        if not has:
            # No picture behind the staged step: a layer left from an
            # earlier (wall) step goes, the tick comes down SILENTLY and
            # the reason is written on screen.  The intent is kept, so a
            # later wall step in the same chain brings the picture back.
            self._drop_refine_walls_layer()
            if hasattr(adv, "set_refine_walls_toggle_checked"):
                adv.set_refine_walls_toggle_checked(False)
            chained = int(getattr(self, "_smooth_chain_count", 0) or 0) > 1
            adv.set_refine_walls_toggle_enabled(
                False,
                WALLS_TOGGLE_LAST_STEP if chained else WALLS_TOGGLE_NO_WALLS)
            return
        adv.set_refine_walls_toggle_enabled(True)
        if getattr(self, "_refine_walls_wanted", False):
            if self._show_walls_picture():
                adv.set_refine_walls_toggle_checked(True)
            else:
                self._refine_walls_wanted = False

    def _on_refine_show_walls_toggled(self, checked: bool) -> None:
        """The user ticked or unticked "Show what the tool sees"."""
        self._refine_walls_wanted = bool(checked)
        adv = self._get_advanced_features_widget()
        if checked:
            if not self._show_walls_picture():
                self._refine_walls_wanted = False
                return
            # The legend is what the user reads, and the tick THEY pressed
            # does not pass through `set_refine_walls_toggle_checked`, which
            # is where it was being written (driver 06e).
            if adv is not None and hasattr(adv, "set_refine_walls_toggle_checked"):
                try:
                    adv.set_refine_walls_toggle_checked(True)
                except Exception:
                    pass
            return
        if adv is not None and hasattr(adv, "set_refine_walls_toggle_checked"):
            try:
                adv.set_refine_walls_toggle_checked(False)
            except Exception:
                pass
        self._drop_refine_walls_layer()
        print("[WALLS] removed guide_walls - unticked (Refine)")

    def _drop_refine_walls_layer(self) -> None:
        """Take the layer away and KEEP the intent, so a new result can
        rebuild it (the Reshape distinction)."""
        try:
            self._remove_temp_layer("walls")
        except Exception as exc:  # noqa: BLE001
            print("[WALLS] remove raised: {!r}".format(exc))

    def _drop_refine_walls(self) -> None:
        """An EXIT: the layer, the tick and the intent all go."""
        self._refine_walls_wanted = False
        self._refine_elevation = None
        self._refine_elevation_roi = None
        self._drop_refine_walls_layer()
        adv = self._get_advanced_features_widget()
        if adv is not None and hasattr(adv, "clear_refine_walls_toggle"):
            try:
                adv.clear_refine_walls_toggle()
            except Exception:
                pass

    def _on_refine_method_combo_changed(self, _index: int) -> None:
        """A method change, staged result or not.

        Decided 2026-09-23: a method change under a staged result
        keeps Preview LIVE -- chaining across methods is the feature, and
        the result card and breadcrumb name each step.  The gate decides
        the button for the new method: off only for a missing channel,
        with its sentence (B1c: a step that needs more room grows the
        region instead of being refused).
        """
        # The PREVIOUS method may have needed a channel and left Preview
        # off (driver 2026-09-22, checks 10/11/13 all pressed a dead button
        # because a geometry method inherited Settle's refusal).
        # The gate also puts S1 back once its refusal no longer holds
        # (B1b audit C-L2): the staged status, or the idle line.
        self._update_smooth_ui(preview_reason="")
        self._refresh_refine_preview_gate()

    def _refresh_refine_inputs(self) -> None:
        """Fill the ONE Refine channel combo and tell the page its spacing.

        Called from `_refresh_all_image_combos` (layer insert / remove /
        tab change) and at launch, exactly as the Reshape combos are.
        """
        adv = self._get_advanced_features_widget()
        if adv is None:
            return
        if hasattr(adv, "set_refine_channels"):
            names = []
            for layer in self.viewer.layers:
                if getattr(layer, "_epicycle_temporary", False):
                    continue
                if isinstance(layer, Image):
                    ndim = layer.data.ndim
                    if ndim in {2, 3}:
                        names.append(layer.name)
                    elif ndim == 4 and int(layer.data.shape[-1]) in {3, 4}:
                        names.append(layer.name)
            try:
                adv.set_refine_channels(names)
            except Exception:
                pass
        if hasattr(adv, "set_refine_solver_available"):
            # Cached on the CLASS: `have_usegment3d()` does a real import,
            # and this runs on every layer insert / remove.
            avail = getattr(Curation3DAnnotator, "_usegment3d_available", None)
            if avail is None:
                try:
                    from packages.epicycle_annotator.reshape.optional_deps \
                        import have_usegment3d

                    avail = bool(have_usegment3d())
                except Exception:
                    avail = False
                Curation3DAnnotator._usegment3d_available = avail
                print("[REFINE] u-Segment3D available: {}".format(avail))
            try:
                from napari_epicycle.annotator.advanced_features_widget \
                    import REFINE_SOLVER_MISSING

                adv.set_refine_solver_available(avail, REFINE_SOLVER_MISSING)
            except Exception as exc:  # noqa: BLE001
                print("[REFINE] solver availability not applied: "
                      "{!r}".format(exc))
        if hasattr(adv, "set_refine_spacing"):
            try:
                adv.set_refine_spacing(
                    self._refine_spacing()
                    if self._refine_spacing_known() else None)
            except Exception:
                pass
        self._refresh_refine_guided_grow()
        # The gate has to follow the two combos that can change what the
        # method needs.  Wired ONCE per widget -- the flag lives on the
        # combo itself, so a relaunch that builds a fresh dock wires the
        # fresh one and a reused dock does not stack connections.
        self._hook_refine_selection()
        for _attr in ("refine_channel_combo", "refine_wall_kind_combo",
                      "refine_contour_mode_combo"):
            combo = getattr(adv, _attr, None)
            if combo is None or combo.property("_epicycle_refine_gate"):
                continue
            try:
                combo.currentIndexChanged.connect(
                    lambda _i: self._refresh_refine_preview_gate())
                combo.setProperty("_epicycle_refine_gate", True)
            except Exception as exc:  # noqa: BLE001
                print("[REFINE] could not wire {}: {!r}".format(_attr, exc))
        self._refresh_refine_preview_gate()

    def _hook_refine_selection(self) -> None:
        """Re-run the Preview gate when the selected object changes.

        B3 audit LOW: with Preview greyed for "no object selected", a
        selection that nothing re-reads would leave it greyed.  One
        ``events.selected_label`` subscription per main labels layer (the
        flag lives ON the layer, so a new layer after a patch switch is
        hooked afresh and an old one is never hooked twice).  The handler is
        deferred to the next event-loop turn -- never inside the selection
        cascade -- and does nothing unless the Refine tab is in front, so
        navigator browsing on the other tabs pays nothing.
        """
        try:
            layer = self._layer_by_name(self.viewer.layers,
                                        self._active_labels_name)
        except Exception:
            layer = None
        if layer is None or getattr(layer, "_epicycle_refine_sel_hooked",
                                    False):
            return
        events = getattr(layer, "events", None)
        sel = getattr(events, "selected_label", None)
        if sel is None:
            return
        try:
            sel.connect(self._on_refine_selection_changed)
            layer._epicycle_refine_sel_hooked = True
        except Exception as exc:  # noqa: BLE001
            print("[REFINE] selection hook not wired: {!r}".format(exc))

    def _on_refine_selection_changed(self, *_a) -> None:
        from qtpy.QtCore import QTimer

        def _later():
            try:
                adv = self._get_advanced_features_widget()
                tabs = getattr(adv, "_tool_tabs", None)
                if tabs is None or tabs.tabText(
                        tabs.currentIndex()) != "Refine":
                    return
                self._refresh_refine_preview_gate()
            except Exception as exc:  # noqa: BLE001
                print("[REFINE] selection gate skipped: {!r}".format(exc))

        QTimer.singleShot(0, _later)

    def _refresh_refine_object_row(self) -> None:
        """S2's read-only value row: which object, and how big it is.

        A value row rather than a dropdown, because the tool FOLLOWS the
        3D selection and a combo here would invite a click that does
        nothing.
        """
        adv = self._get_advanced_features_widget()
        if adv is None or not hasattr(adv, "refine_object_value"):
            return
        try:
            label_id = int(self._get_selected_label() or 0)
        except Exception:
            label_id = 0
        if label_id <= 0:
            self._update_smooth_ui(object_text="none selected")
            return
        n = None
        try:
            vol = self.sync_manager.shared_labels.get(self._active_labels_name)
            if vol is not None:
                n = int(np.count_nonzero(vol == label_id))
        except Exception:
            n = None
        self._update_smooth_ui(object_text=(
            "object {} - {:,} voxels".format(label_id, n) if n
            else "object {}".format(label_id)))

    def _refresh_refine_preview_gate(self):
        """Disable Preview BEFORE the click, with the reason in its tooltip.

        A refusal that fires after the click teaches nothing.  Never touched while a worker is in flight -- that state owns
        the button.  With a result staged, any method may chain onto any
        other in any order: a step that needs more room
        grows the region at the click, so the only refusal here is a
        missing channel.  Returns whether Preview was left enabled, or None
        when the gate did not decide.
        """
        adv = self._get_advanced_features_widget()
        if adv is None or not hasattr(adv, "get_refine_params"):
            return None
        if getattr(self, "_refine_running", False):
            return None
        if getattr(self, "_refine_sync_busy", False):
            # C-L3: a synchronous step is inside its pump and owns the page.
            return None
        staged = getattr(self, "_refine_staged_method", None)
        if not (staged is not None
                and getattr(self, "_smooth_running", False)):
            # The object row is held while a result is staged.
            self._refresh_refine_object_row()
        try:
            params = adv.get_refine_params()
        except Exception:
            return None
        if not (staged is not None and getattr(self, "_smooth_running", False)):
            # Preview is enabled only when an
            # object is selected.  Not while staged -- the object is HELD.
            try:
                selected = int(self._get_selected_label() or 0)
            except Exception:
                selected = None      # cannot tell: the gate does not decide
            if selected is not None and selected <= 0:
                self._update_smooth_ui(
                    preview_enabled=False,
                    preview_reason=REFINE_SELECT_OBJECT,
                    status_text=REFINE_SELECT_OBJECT,
                    status_kind="warn")
                return False
        if params.get("needs_channel") and not params.get("channel"):
            # BOTH places (U-H2).  Qt shows no tooltip on a disabled
            # widget, so a reason that lives only there is a reason nobody
            # can read -- the same mistake `WALLS_TOGGLE_*` exists to
            # avoid.
            self._update_smooth_ui(
                preview_enabled=False,
                preview_reason=REFINE_PICK_CHANNEL,
                status_text=REFINE_PICK_CHANNEL,
                status_kind="warn")
            return False
        self._update_smooth_ui(preview_enabled=True, preview_reason="")
        # C-L2: the gate's own refusal sentence must not outlive its
        # condition.  Staged: the staged status goes back (the one
        # `_finish_refine_preview` wrote, UX-L2).  Idle: S1 is replaced
        # only when it holds a sentence the gate put there.
        if staged is not None and getattr(self, "_smooth_running", False):
            self._update_smooth_ui(
                status_text=self._refine_staged_status(),
                status_kind="review")
        elif self._refine_s1_text(adv) in _REFINE_GATE_SENTENCES:
            self._update_smooth_ui(
                status_text=REFINE_IDLE_STATUS, status_kind="idle")
        return True

    def _refine_staged_status(self) -> str:
        """The status the staged result was written with, or a plain one."""
        s5 = getattr(self, "_refine_staged_s5", None) or {}
        text = s5.get("status_text")
        if text:
            return str(text)
        n = int(getattr(self, "_smooth_chain_count", 0) or 0)
        if n > 1:
            return "Object {} - {} steps staged".format(
                self._smooth_label_id, n)
        return "Object {} - a result is staged.".format(self._smooth_label_id)

    def _refine_s1_text(self, adv) -> str:
        """What S1 says right now -- the gate replaces only its own words."""
        label = getattr(adv, "smooth_status_label", None)
        try:
            return str(label.text()) if label is not None else ""
        except Exception:
            return ""

    def _get_refine_channel_roi(self, roi: tuple):
        """The ONE Refine channel, cropped to *roi* -- or None.

        There is no fallback.  `_get_refine_image_roi` used to fall through
        to the hidden `image_combo`, i.e. to the FIRST image layer, whenever
        its per-method combo was blank -- which defeated the blank-stays-
        blank policy and made every "that layer is gone" refusal
        unreachable.  `_get_image_roi`, whose only two callers
        were those two fallback lines, is deleted with them.
        """
        adv = self._get_advanced_features_widget()
        if adv is None or not hasattr(adv, "refine_channel_name"):
            return None
        name = adv.refine_channel_name()
        if not name:
            print("[REFINE] no channel chosen")
            return None
        # Called through the class so a bare test fake that binds only this
        # method keeps working (the lookup is a pure helper of `self`).
        data = Curation3DAnnotator._refine_channel_source(self, name)
        if data is None:
            print("[REFINE] the chosen channel {!r} is not in the viewer"
                  .format(name))
            return None
        # CROP, THEN copy (B2 audit C-M3): `np.asarray(layer.data)` on the
        # whole layer materialised the entire volume -- a lazy or big layer
        # paid for all of it on the main thread to use one object's box.
        # Lifecycle LOW-1: ALWAYS a copy.  `np.asarray(..., float64)` of a
        # float64 layer is a VIEW of the live layer, and the worker reads
        # this array while the user may edit that layer.
        return np.array(data[roi], dtype=np.float64, copy=True)

    def _refine_channel_source(self, name):
        """The named image layer's 3-D data, UNCOPIED -- the lookup and the
        multiscale / 4-D squeeze `_get_refine_channel_roi` uses, factored
        out so Batch Refine resolves the same array on the main thread and
        crops it per object on its worker.
        None when no such layer."""
        if not name:
            return None
        layer = self._layer_by_name(self.viewer.layers, name)
        if layer is None:
            return None
        data = layer.data
        if getattr(layer, "multiscale", False):
            data = data[0]
        if getattr(data, "ndim", 3) == 4:
            data = data[..., 0] if int(data.shape[-1]) in (3, 4) else data[0]
        return data

    def _accept_smooth(self) -> None:
        """Accept the refine result and update tier."""
        # B2: refuse while load-locked (e.g. async Save in flight) — see
        # _accept_split / _accept_interpolation.  Internal calls from
        # _accept_split (fragment commit) only occur when NOT locked.
        if getattr(self, "_load_locked", False):
            print("[REFINE-ACCEPT] smooth accept refused — controller busy (load-locked)")
            return
        if not self._smooth_running:
            return

        # C-H3: the teardown belongs BELOW both early returns.  It used to
        # run first, so a refused Accept -- the load-locked one above, which
        # is the case the guard exists for -- abandoned the worker, took the
        # wall picture down, forgot which method made the staged shape and
        # RELEASED THE CONTROLLER LOCK, while the unaccepted shape stayed in
        # `shared_data`.  The refusal is meant to change nothing.
        #
        # A worker may still be in flight behind a staged chain; its answer
        # must not land on a volume this accept has already committed.
        self._abandon_refine_worker()
        self._drop_refine_walls()
        self._refine_staged_method = None
        self._refine_steps = []
        # the debug trail is copied BEFORE the clear.
        prov_steps, prov_recipes = _refine_provenance(self)
        _refine_clear_records(self)
        self._refine_staged_s5 = None
        self._fire_refine_lock(False)

        # --- Interp refine branch: accept smooth = commit everything. ---
        if self._interp_refine_mode:
            label_id = self._smooth_label_id
            # Capture undo data before clearing.
            # Use explicit None checks — numpy arrays can't be used with `or`.
            undo_roi = self._smooth_roi if self._smooth_roi is not None else self._interp_roi
            undo_snap = self._smooth_snapshot if self._smooth_snapshot is not None else self._interp_snapshot

            # Clear smooth state.
            self._smooth_running = False
            self._refresh_merge_button_state()
            self._refresh_label_action_buttons_state()
            self._smooth_snapshot = None
            self._smooth_roi = None
            self._smooth_label_id = None
            self._smooth_chain_count = 0
            self._remove_prob_overlay()

            # Discard interp snapshot (changes committed).
            self._interp_snapshot = None
            self._refresh_merge_button_state()
            self._refresh_label_action_buttons_state()
            self._interp_roi = None
            interp_label = self._interp_label_id
            self._interp_label_id = None
            self._interp_refine_mode = False
            self._refresh_merge_button_state()
            self._refresh_label_action_buttons_state()
            # Clear edited slices for the label.
            labels_name = self._active_labels_name
            if labels_name and interp_label is not None:
                edited = self.sync_manager.get_edited_slices(labels_name)
                for orient_dict in edited.values():
                    orient_dict.pop(interp_label, None)
                    for key in list(orient_dict.keys()):
                        if int(key) == interp_label:
                            orient_dict.pop(key, None)

            # Re-enable interp UI.
            self._interpolate_btn.setEnabled(True)
            self._accept_interp_btn.setEnabled(False)
            self._reject_interp_btn.setEnabled(False)
            self._interp_refine_btn.setEnabled(False)

            self._update_smooth_ui(
                preview_enabled=True,
                accept_enabled=False,
                reject_enabled=False,
                accept_text="Accept",
                preview_reason="",
                result_note="", result_warning="", breadcrumb="",
                status_text=f"Object {label_id} - accepted.",
            )

            # Fire both callbacks.   the chain's steps
            # and the recipes that contributed ride along as a debug trail.
            cb = getattr(self, "_on_label_smoothed_cb", None)
            if cb is not None:
                cb(label_id, roi_slices=undo_roi, old_roi_data=undo_snap,
                   steps=prov_steps, recipes=prov_recipes)
            cb = getattr(self, "_on_interpolation_accepted_cb", None)
            if cb is not None:
                cb(label_id, roi_slices=undo_roi, old_roi_data=undo_snap)

            self._interp_status_label.setText(
                f"Accepted refined label {label_id}."
            )

            self._clear_draw_snapshot()
            self._restore_ready_to_draw_state()
            self._refresh_refine_preview_gate()      # UX-M1

            # Auto-advance to next triage label.
            if self._is_auto_advance_enabled():
                nav = getattr(self, "_navigator_widget", None)
                if nav is not None:
                    try:
                        if not nav.select_next_triage():
                            self._on_isolation_changed(set())
                    except Exception:
                        pass
            return

        label_id = self._smooth_label_id
        # Capture undo data before clearing.
        undo_roi = self._smooth_roi
        undo_snap = self._smooth_snapshot

        # Clear state.
        self._smooth_running = False
        self._refresh_merge_button_state()
        self._refresh_label_action_buttons_state()
        self._smooth_snapshot = None
        self._smooth_roi = None
        self._smooth_label_id = None
        self._smooth_chain_count = 0

        # Clean up prob overlay if active.
        self._remove_prob_overlay()

        self._update_smooth_ui(
            preview_enabled=True,
            accept_enabled=False,
            reject_enabled=False,
            accept_text="Accept",
            preview_reason="",
            result_note="", result_warning="", breadcrumb="",
            status_text=f"Object {label_id} - accepted.",
        )

        # Fire callback to controller (sets tier to 'curated').  Recipes
        # the steps and recipes ride along as a debug trail.
        cb = getattr(self, "_on_label_smoothed_cb", None)
        if cb is not None:
            cb(label_id, roi_slices=undo_roi, old_roi_data=undo_snap,
               steps=prov_steps, recipes=prov_recipes)

        self._clear_draw_snapshot()
        self._restore_ready_to_draw_state()
        # UX-M1: the gate decides the button for the method now on the
        # combo -- a Settle page with no channel must not come back live,
        # and a refusal must not stay in Preview's tooltip.
        self._refresh_refine_preview_gate()

        # Auto-advance if toggle is on.
        if self._is_auto_advance_enabled():
            if self._split_fragment_refine and self._split_fragment_ids:
                # During fragment refine, cycle within the fragment list.
                current = self._get_selected_label()
                ids = self._split_fragment_ids
                try:
                    idx = ids.index(current)
                    next_idx = (idx + 1) % len(ids)
                    next_fid = ids[next_idx]
                except ValueError:
                    next_fid = ids[0]
                self._set_selected_label_all(next_fid)
                # Update 3D proxy to show the newly selected fragment.
                if getattr(self, "_isolation_proxy_active", False):
                    labels_name = self._active_labels_name
                    main_layer = self._layer_by_name(
                        self.viewer.layers, labels_name
                    )
                    if isinstance(main_layer, Labels):
                        roi_hint = getattr(self, "_fragment_refine_roi", None)
                        self._update_isolation_proxy(
                            next_fid, main_layer, force=True,
                            roi_hint=roi_hint,
                        )
            else:
                nav = getattr(self, "_navigator_widget", None)
                if nav is not None:
                    try:
                        if not nav.select_next_triage():
                            self._on_isolation_changed(set())
                    except Exception:
                        pass

    def _reject_smooth(self) -> None:
        """Discard the refine result: restore the shared volume from snapshot."""
        self._abandon_refine_worker()
        self._drop_refine_walls()
        self._refine_staged_method = None
        self._refine_steps = []
        _refine_clear_records(self)
        self._refine_staged_s5 = None
        self._fire_refine_lock(False)
        if not self._smooth_running:
            return

        # --- Interp refine branch: reject smooth = full undo to pre-interp. ---
        if self._interp_refine_mode:
            label_id = self._smooth_label_id
            labels_name = self._active_labels_name
            shared_data = self.sync_manager.shared_labels.get(labels_name)

            # The refine chain's snapshot goes back FIRST (B1c audit
            # MED-2): a grown region reaches beyond `_interp_roi`, and the
            # interp snapshot alone would leave those writes in place.
            # Then the interp snapshot takes the interp ROI back to its
            # pre-interpolation state.
            if (shared_data is not None
                    and self._smooth_snapshot is not None
                    and self._smooth_roi is not None):
                shared_data[self._smooth_roi] = self._smooth_snapshot
                self._sync_all_views_after_operation(
                    roi=self._smooth_roi, label_id=label_id)

            # Clear smooth state first.
            self._smooth_running = False
            self._refresh_merge_button_state()
            self._refresh_label_action_buttons_state()
            self._smooth_snapshot = None
            self._smooth_roi = None
            self._smooth_label_id = None
            self._smooth_chain_count = 0
            self._remove_prob_overlay()

            # Restore from interp snapshot (pre-interpolation state).
            if (
                shared_data is not None
                and self._interp_snapshot is not None
                and self._interp_roi is not None
            ):
                shared_data[self._interp_roi] = self._interp_snapshot
                self._sync_all_views_after_operation(
                    roi=self._interp_roi, label_id=label_id
                )

            # Keep edited slices — user's annotations preserved for re-Preview.

            # Clear interp state.
            self._interp_snapshot = None
            self._refresh_merge_button_state()
            self._refresh_label_action_buttons_state()
            self._interp_roi = None
            self._interp_label_id = None
            self._interp_refine_mode = False
            self._refresh_merge_button_state()
            self._refresh_label_action_buttons_state()
            # Re-enable interp UI.
            self._interpolate_btn.setEnabled(True)
            self._accept_interp_btn.setEnabled(False)
            self._reject_interp_btn.setEnabled(False)
            self._interp_refine_btn.setEnabled(False)

            self._update_smooth_ui(
                preview_enabled=True,
                accept_enabled=False,
                reject_enabled=False,
                accept_text="Accept",
                preview_reason="",
                result_note="", result_warning="", breadcrumb="",
                status_text=f"Object {label_id} - put back as it was.",
            )
            self._interp_status_label.setText(
                "Refine rejected. Volume restored to pre-interpolation state."
            )
            self._clear_draw_snapshot()
            self._restore_ready_to_draw_state()
            self._refresh_refine_preview_gate()      # UX-M1
            return

        labels_name = self._active_labels_name
        shared_data = self.sync_manager.shared_labels.get(labels_name)

        # Restore from snapshot.
        if (
            shared_data is not None
            and self._smooth_snapshot is not None
            and self._smooth_roi is not None
        ):
            shared_data[self._smooth_roi] = self._smooth_snapshot
            self._sync_all_views_after_operation(
                roi=self._smooth_roi, label_id=self._smooth_label_id
            )

        label_id = self._smooth_label_id

        # Clear state.
        self._smooth_running = False
        self._refresh_merge_button_state()
        self._refresh_label_action_buttons_state()
        self._smooth_snapshot = None
        self._smooth_roi = None
        self._smooth_label_id = None
        self._smooth_chain_count = 0

        # Clean up prob overlay if active.
        self._remove_prob_overlay()

        self._update_smooth_ui(
            preview_enabled=True,
            accept_enabled=False,
            reject_enabled=False,
            accept_text="Accept",
            preview_reason="",
            result_note="", result_warning="", breadcrumb="",
            status_text=f"Object {label_id} - put back as it was.",
        )
        self._clear_draw_snapshot()
        self._restore_ready_to_draw_state()
        # UX-M1: the gate decides the button for the method now on the
        # combo -- a Settle page with no channel must not come back live.
        self._refresh_refine_preview_gate()

    def _update_smooth_ui(self, **kwargs) -> None:
        """Proxy to update smooth UI state in the AdvancedFeaturesWidget."""
        run = getattr(self, "_refine_recipe_run", None)
        if run is not None:
            #  while a recipe runs, what each step
            # ASKS for the three buttons is remembered (and pushed back when
            # the run ends), the stage line is composed "i/n · stage", and
            # the buttons themselves are held below.
            kwargs = _recipe_filter_ui(self, run, dict(kwargs))
        adv_widget = self._get_advanced_features_widget()
        if adv_widget and hasattr(adv_widget, "update_smooth_ui_state"):
            adv_widget.update_smooth_ui_state(**kwargs)
            if run is not None:
                _recipe_hold_buttons(self, adv_widget)
        # Also mirror status to the main interp label as a fallback.
        status = kwargs.get("status_text", "")
        if status and hasattr(self, "_interp_status_label"):
            self._interp_status_label.setText(status)

    # ---------------------------------------------------------------
    # Refine helpers — image/prob ROI + state reset
    # ---------------------------------------------------------------

    def set_prob_volume_path(self, path: str) -> None:
        """Set the path to the probability volume (loaded lazily on first use)."""
        self._prob_volume_path = path if path else None
        self._prob_volume = None  # invalidate cache
        # Guided Grow reads it, so its item follows it.
        self._refresh_refine_guided_grow()

    def _prob_volume_shape(self):
        """The probability map's shape WITHOUT loading it (D-M1): the
        loaded array's, else the ``.npy`` header's through
        ``np.load(mmap_mode="r")`` -- no voxel is read.  Cached per
        (path, mtime).  ``None`` when there is no map or it will not open
        (the blocker is re-run after the real load, so that case is caught
        there)."""
        pv = getattr(self, "_prob_volume", None)
        if pv is not None:
            return tuple(int(n) for n in pv.shape)
        path = getattr(self, "_prob_volume_path", None)
        if not path:
            return None
        try:
            key = (str(path), _Path(str(path)).stat().st_mtime_ns)
        except OSError:
            return None
        cached = _safe_attr(self, "_prob_shape_cache")
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            m = np.load(str(path), mmap_mode="r")
            shape = tuple(int(n) for n in m.shape)
            del m
        except Exception as exc:  # noqa: BLE001 -- the load re-checks
            print(f"[BATCH] probability map shape not readable: {exc!r}")
            return None
        self._prob_shape_cache = (key, shape)
        return shape

    def _has_prob_volume(self) -> bool:
        """Is there a probability map to read -- WITHOUT loading it."""
        if getattr(self, "_prob_volume", None) is not None:
            return True
        path = getattr(self, "_prob_volume_path", None)
        if not path:
            return False
        try:
            from pathlib import Path

            return Path(path).exists()
        except Exception:
            return False

    def _refresh_refine_guided_grow(self) -> None:
        """Grey Guided Grow's item BEFORE any click when there is no map."""
        adv = self._get_advanced_features_widget()
        if adv is None or not hasattr(adv, "set_refine_guided_grow_available"):
            return
        try:
            from napari_epicycle.annotator.advanced_features_widget import (
                REFINE_GUIDED_GROW_NO_PROB,
            )

            avail = self._has_prob_volume()
            adv.set_refine_guided_grow_available(
                avail, REFINE_GUIDED_GROW_NO_PROB)
            # Said once per CHANGE: this runs on every layer insert/remove.
            if avail != getattr(self, "_refine_gg_avail_said", None):
                self._refine_gg_avail_said = avail
                print("[REFINE] Guided Grow {}".format(
                    "available" if avail else
                    "greyed - this patch has no probability map"))
        except Exception as exc:  # noqa: BLE001
            print("[REFINE] Guided Grow availability not applied: "
                  "{!r}".format(exc))

    @staticmethod
    def _squeeze_image_channel(data: np.ndarray) -> np.ndarray:
        """Reduce a multichannel image array to a single 3D channel.

        ``(Z, Y, X, C)`` or ``(C, Z, Y, X)`` collapse to ``(Z, Y, X)`` using
        channel 0; 2D/3D arrays pass through unchanged. This MUST run before a
        3D ROI slice — slicing a 4D array with a 3-slice ROI silently keeps the
        channel axis and breaks any 3D consumer (e.g. ``sobel`` → watershed
        shape mismatch in intensity-mode split). Single source of truth for the
        channel-collapse pattern used by every image-ROI accessor below.
        """
        if data.ndim == 4:  # multichannel: (Z, Y, X, C) or (C, Z, Y, X)
            if data.shape[-1] in (3, 4):
                return data[..., 0]
            return data[0]
        return data

    def _get_prob_roi(self, roi: tuple) -> np.ndarray | None:
        """Load probability volume and crop to ROI. Returns None if unavailable.

        Prob volume is NOT a napari layer. Path is set via set_prob_volume_path().
        Lazy-loads on first access, caches in self._prob_volume.
        """
        if Curation3DAnnotator._ensure_prob_volume(self) is None:
            return None
        return self._prob_volume[roi].astype(np.float64)

    def _ensure_prob_volume(self):
        """The probability volume, lazily loaded and cached -- the load
        `_get_prob_roi` does, factored out for Batch Refine.
        None when there is none."""
        if self._prob_volume_path is None:
            return None
        if self._prob_volume is None:
            from pathlib import Path
            p = Path(self._prob_volume_path)
            if not p.exists():
                return None
            self._prob_volume = np.load(str(p))
            if self._prob_volume.dtype == np.float16:
                self._prob_volume = self._prob_volume.astype(np.float32)
        return self._prob_volume

    def _reset_smooth_state(self, msg: str = "") -> None:
        """Reset smooth/refine state and update UI on error or early return."""
        self._abandon_refine_worker()
        self._refine_pending_elevation = None     # C-L4
        self._drop_refine_walls()
        self._refine_staged_method = None
        self._refine_steps = []
        _refine_clear_records(self)
        self._refine_staged_s5 = None
        self._fire_refine_lock(False)
        self._smooth_running = False
        self._refresh_merge_button_state()
        self._refresh_label_action_buttons_state()
        self._smooth_snapshot = None
        self._smooth_roi = None
        self._smooth_label_id = None
        self._smooth_chain_count = 0
        self._update_smooth_ui(
            preview_enabled=True,
            accept_enabled=False,
            reject_enabled=False,
            preview_text="Preview",
            accept_text="Accept",
            preview_reason="",
            result_note="", result_warning="", breadcrumb="",
            busy=False, stop_visible=False, stage_text="",
            rows_enabled=True,
            status_text=msg,
        )

    # ---------------------------------------------------------------
    # Refine recipes -- the runner
    # ---------------------------------------------------------------
    #
    # A run is the Previews the user would have pressed, one after another.
    # It never re-implements a refusal, a write, a region growth or a Stop:
    # it sets the panel (`set_refine_step_params`), calls the real
    # `_preview_smooth`, and OBSERVES the ending -- a step wrote iff the
    # breadcrumb grew.  One judge per step, bound to (token, step) with a
    # latch; never a poll of `_refine_running`.

    def _recipe_active(self) -> bool:
        return getattr(self, "_refine_recipe_run", None) is not None

    def _recipe_selected_object(self) -> int:
        """The object Preview would act on: the main layer's selection."""
        try:
            layer = self._layer_by_name(self.viewer.layers,
                                        self._active_labels_name)
            return int(getattr(layer, "selected_label", 0) or 0)
        except Exception:
            return 0

    def _recipe_load_store(self):
        """``(store, problem)`` -- ``(None, None)`` with no workspace."""
        cb = getattr(self, "_get_saved_refine_recipes_cb", None)
        if cb is None:
            return None, None
        try:
            got = cb()
        except Exception as exc:  # noqa: BLE001
            print("[RECIPE] could not read the recipes: {!r}".format(exc))
            from packages.epicycle_annotator.refine_recipe import (
                FILE_UNREADABLE,
            )

            return {}, FILE_UNREADABLE
        if isinstance(got, tuple) and len(got) == 2:
            if got[0] is None:                 # no workspace (audit L1)
                return None, None
            return dict(got[0] or {}), got[1]
        return dict(got or {}), None

    def _recipe_availability(self) -> dict:
        adv = self._get_advanced_features_widget()
        av = {"spacing_known": bool(self._refine_spacing_known())}
        try:
            av["prob"] = bool(self._has_prob_volume())
        except Exception:
            av["prob"] = True
        for key, meth in (("guided_grow", "refine_guided_grow_available"),
                          ("diffusion", "refine_solver_available")):
            fn = getattr(adv, meth, None)
            try:
                av[key] = bool(fn()) if fn is not None else True
            except Exception:
                av[key] = True
        return av

    def _recipe_busy_sentence(self):
        """The sentence `_preview_smooth` would give, or None."""
        from packages.epicycle_annotator import refine_recipe as rr

        if self._recipe_active():
            return rr.S1_RUNNING_BLOCK
        if self._split_running and not self._split_fragment_refine:
            return "Reshape is in progress."
        if getattr(self, "_batch_staged", False) or getattr(
                self, "_batch_running", False):
            return "Finish or close Batch Refine first."
        if getattr(self, "_refine_running", False):
            return "Still working - wait, or press Stop."
        if getattr(self, "_refine_sync_busy", False):
            return rr.HEADER_BUSY
        return None

    def _recipe_mode(self) -> str:
        if getattr(self, "_split_fragment_refine", False):
            return "fragment"
        if getattr(self, "_interp_refine_mode", False):
            return "interp"
        obj = self._recipe_selected_object()
        if (getattr(self, "_smooth_running", False) and obj > 0
                and obj != self._smooth_label_id):
            return "other_object"
        return "ok"

    def _recipe_preflight(self, name, recipe, problem=None):
        """The sentence that refuses the WHOLE recipe, or None."""
        from packages.epicycle_annotator import refine_recipe as rr

        if recipe is None or problem:
            print("[RECIPE] '{}' cannot be read: {}".format(
                name, problem or "it is not in the file"))
            return rr.S1_UNREADABLE
        if getattr(self, "_split_fragment_refine", False):
            return rr.HEADER_FRAGMENT
        if getattr(self, "_interp_refine_mode", False):
            return rr.HEADER_INTERP
        busy = self._recipe_busy_sentence()
        if busy:
            return busy
        obj = self._recipe_selected_object()
        if obj <= 0:
            return REFINE_SELECT_OBJECT
        if self._smooth_running and obj != self._smooth_label_id:
            return rr.HEADER_OTHER_OBJECT.format(
                staged=self._smooth_label_id, obj=obj)
        adv = self._get_advanced_features_widget()
        channel = ""
        try:
            channel = adv.refine_channel_name() if adv is not None else ""
        except Exception:
            channel = ""
        b = rr.step_blocker(recipe, channel=channel,
                            availability=self._recipe_availability())
        if b is not None:
            return rr.blocker_sentence(name, recipe, b)
        return None

    def _recipe_start(self, name, recipe, problem=None) -> bool:
        """R1 -> R2: check everything, then run step 1.  Nothing runs half."""
        from packages.epicycle_annotator import refine_recipe as rr

        why = self._recipe_preflight(name, recipe, problem)
        if why:
            print("[RECIPE] refused before step 1: {}".format(why))
            self._update_smooth_ui(status_text=why, status_kind="warn")
            return False
        adv = self._get_advanced_features_widget()
        pre = int(self._smooth_chain_count or 0) if self._smooth_running else 0
        ctx = {}
        try:
            ctx = self._reshape_context()
        except Exception:
            ctx = {}
        shadow = {}
        stop_tip = ""
        for key, attr in (("preview_enabled", "smooth_preview_btn"),
                          ("accept_enabled", "smooth_accept_btn"),
                          ("reject_enabled", "smooth_reject_btn")):
            w = getattr(adv, attr, None)
            if w is not None:
                shadow[key] = bool(w.isEnabled())
        pb = getattr(adv, "smooth_preview_btn", None)
        if pb is not None:
            shadow["preview_reason"] = ""
        stop = getattr(adv, "refine_cancel_preview_btn", None)
        if stop is not None:
            stop_tip = stop.toolTip()
        self._refine_recipe_token = int(
            getattr(self, "_refine_recipe_token", 0)) + 1
        label = self._recipe_selected_object()
        self._refine_recipe_run = {
            "name": str(name), "steps": list(recipe.steps), "i": 0,
            "n_before": 0, "label": int(label),
            "token": self._refine_recipe_token,
            "patch": ctx.get("patch_name"), "run_epoch": ctx.get("run_epoch"),
            "labels_name": self._active_labels_name, "pre_staged": pre,
            "phase": "checking", "judge_pending": False, "retries": 0,
            "shadow": shadow, "stop_tip": stop_tip, "contributed": False,
            "judged": 0,
        }
        self._recipe_stage_left = False
        if stop is not None:
            stop.setToolTip(rr.stop_tooltip(pre))
        print("[RECIPE] start '{}' ({} steps) on object {} - {} staged "
              "before".format(name, len(recipe.steps), label, pre))
        self._update_smooth_ui(
            status_text=rr.S1_RUNNING.format(
                steps=rr.plural(len(recipe.steps), "step"), obj=label),
            status_kind="review", stop_visible=True, stop_enabled=True)
        self._recipe_next(self._refine_recipe_token)
        return True

    def _recipe_stale_reason(self, run):
        try:
            ctx = self._reshape_context()
        except Exception:
            ctx = {}
        if self._active_labels_name != run.get("labels_name"):
            return "labels layer changed"
        if ctx.get("patch_name") != run.get("patch"):
            return "patch changed"
        if ctx.get("run_epoch") != run.get("run_epoch"):
            return "run changed"
        return None

    def _recipe_session_holds(self, run) -> bool:
        return bool(self._smooth_running
                    and self._smooth_label_id == run["label"])

    def _recipe_next(self, token) -> None:
        """R2 -> R3: set the panel for step i, then press Preview."""
        from packages.epicycle_annotator import refine_recipe as rr

        run = getattr(self, "_refine_recipe_run", None)
        if run is None or run["token"] != token:
            return
        i, n = run["i"], len(run["steps"])
        step = run["steps"][i]
        stale = self._recipe_stale_reason(run)
        if stale:
            self._recipe_end("reset", why=stale)
            return
        if (i >= 1 or run["pre_staged"]) and not self._recipe_session_holds(
                run):
            self._recipe_end("reset", why="the staged session is gone")
            return
        run["phase"] = "setting"
        adv = self._get_advanced_features_widget()
        ok, why = (False, "the panel is gone")
        if adv is not None and hasattr(adv, "set_refine_step_params"):
            ok, why = adv.set_refine_step_params(step.method, step.params)
        # the setter's own signals may have ended the run (never today)
        if getattr(self, "_refine_recipe_run", None) is not run:
            return
        if not ok:
            # Fixed words on screen (UX M3); the raw reason goes to the
            # console for whoever reads the log.
            word = rr.METHOD_WORD.get(step.method, step.method)
            what = (rr.STAGE_PREP_DIFFERS if why == rr.STAGE_PREP_DIFFERS
                    else word)
            print("[RECIPE] step {}/{} ({}) could not be set: {}".format(
                i + 1, n, step.method, why))
            staged = bool(self._smooth_running)
            tmpl = rr.S1_NOT_SET if staged else rr.S1_NOT_SET_FIRST
            self._recipe_end(
                "refused", why=why,
                stage=rr.STAGE_STOPPED.format(name=run["name"], k=i + 1, n=n,
                                              what=what),
                status=tmpl.format(k=i + 1, n=n, word=word))
            return
        run["n_before"] = (len(self._refine_steps or [])
                           if self._smooth_running else 0)
        run["phase"] = "previewing"
        print("[RECIPE] step {}/{} {}".format(i + 1, n, step.method))
        self._update_smooth_ui(
            stage_text=rr.METHOD_WORD.get(step.method, step.method))
        self._preview_smooth()
        if getattr(self, "_refine_recipe_run", None) is not run:
            return                      # ended inside the step (abandon)
        if run.get("stop_requested"):
            self._recipe_honour_stop(run)
            return
        if run["phase"] == "waiting":
            return                      # the worker's slot schedules the judge
        if getattr(self, "_refine_running", False):
            # a worker is running with no recipe slot -- cannot happen, and
            # waiting on it would be for ever
            self._recipe_end("reset", why="a worker ran without the slot")
            return
        self._recipe_schedule_judge(run, i)

    def _recipe_schedule_judge(self, run, step_index) -> None:
        import functools

        from qtpy.QtCore import QTimer

        run["phase"] = "judging"
        run["judge_pending"] = True
        QTimer.singleShot(0, functools.partial(
            self._recipe_after_step, run["token"], step_index))

    def _recipe_on_worker_done(self, token, step_index, *_args) -> None:
        """The worker's finished/error, bound at connect time."""
        run = getattr(self, "_refine_recipe_run", None)
        if run is None or run["token"] != token or run["i"] != step_index:
            print("[RECIPE] late worker signal ignored (token {} step {})"
                  .format(token, step_index))
            return
        if run["judge_pending"] or run["phase"] != "waiting":
            return
        self._recipe_schedule_judge(run, step_index)

    def _recipe_after_step(self, token, step_index) -> None:
        """R4: THE judge of step ``step_index`` -- exactly one per step."""
        from packages.epicycle_annotator import refine_recipe as rr

        run = getattr(self, "_refine_recipe_run", None)
        if (run is None or run["token"] != token or run["i"] != step_index
                or not run["judge_pending"]):
            print("[RECIPE] late judge ignored (token {} step {})".format(
                token, step_index))
            return
        run["judge_pending"] = False
        if run.get("stop_requested"):
            self._recipe_honour_stop(run)
            return
        if getattr(self, "_refine_sync_busy", False):
            run["retries"] += 1
            if run["retries"] > 3:
                self._recipe_end("reset", why="a step stayed busy")
                return
            self._recipe_schedule_judge(run, step_index)
            return
        run["retries"] = 0
        run["judged"] += 1
        stale = self._recipe_stale_reason(run)
        if stale:
            self._recipe_end("reset", why=stale)
            return
        i, n = run["i"], len(run["steps"])
        step = run["steps"][i]
        wrote = bool(self._smooth_running) and len(
            self._refine_steps or []) > run["n_before"]
        if not wrote:
            if (i >= 1 or run["pre_staged"]) and not (
                    self._recipe_session_holds(run)):
                self._recipe_end("reset", why="the staged session is gone")
                return
            self._recipe_end(
                "refused", why="step {} did not write".format(i + 1),
                stage=rr.STAGE_STOPPED.format(
                    name=run["name"], k=i + 1, n=n,
                    what=self._REFINE_METHOD_WORD.get(step.method,
                                                      step.method)))
            return
        if not run["contributed"]:
            run["contributed"] = True
            names = list(getattr(self, "_refine_chain_recipe_names", [])
                         or [])
            names.append(run["name"])
            self._refine_chain_recipe_names = names
        shared = self.sync_manager.shared_labels.get(self._active_labels_name)
        roi = self._smooth_roi
        emptied = False
        if shared is not None and roi is not None:
            emptied = not bool(np.any(np.asarray(shared[roi])
                                      == run["label"]))
        if emptied:
            self._recipe_end(
                "emptied", why="step {} left nothing".format(i + 1),
                stage=rr.STAGE_STOPPED.format(name=run["name"], k=i + 1, n=n,
                                              what=rr.STAGE_EMPTY))
            return
        if i + 1 >= n:
            self._recipe_end(
                "done", status=rr.S1_DONE.format(
                    steps=rr.plural(n, "step"),
                    obj=run["label"]), status_kind="review")
            return
        run["i"] = i + 1
        run["phase"] = "advancing"
        self._recipe_next(token)

    def _recipe_honour_stop(self, run) -> None:
        """A Stop that arrived inside a synchronous step (audit H1), acted
        on once the step has returned: a worker it launched is stopped the
        normal way; otherwise the whole session goes back at once."""
        print("[RECIPE] honouring the Stop requested inside step {}".format(
            run["i"] + 1))
        if getattr(self, "_refine_running", False):
            self._cancel_refine_preview()
        else:
            self._on_refine_stopped()

    def _recipe_stopped_stage(self, run, what) -> str:
        from packages.epicycle_annotator import refine_recipe as rr

        return rr.STAGE_STOPPED.format(name=run["name"], k=run["i"] + 1,
                                       n=len(run["steps"]), what=what)

    def _recipe_on_stopped(self) -> None:
        from packages.epicycle_annotator import refine_recipe as rr

        run = getattr(self, "_refine_recipe_run", None)
        if run is None:
            return
        tmpl = (rr.S1_STOPPED_STAGED if run["pre_staged"]
                else rr.S1_STOPPED_FRESH)
        self._recipe_end("stopped", why="Stop", status=tmpl.format(
            k=run["i"] + 1, n=len(run["steps"])),
            status_kind="warn")

    def _recipe_end(self, kind, *, why="", stage=None, status=None,
                    status_kind="warn") -> None:
        """Every ending.  Idempotent; the token moves on."""
        run = getattr(self, "_refine_recipe_run", None)
        if run is None:
            return
        self._refine_recipe_run = None
        self._refine_recipe_token = int(
            getattr(self, "_refine_recipe_token", 0)) + 1
        running = bool(getattr(self, "_refine_running", False))
        kw = dict(run.get("shadow") or {})
        kw.update(stop_visible=running,
                  stop_enabled=running and not getattr(
                      self, "_refine_stopping", False),
                  stage_text=stage or "")
        if status:
            kw.update(status_text=status, status_kind=status_kind)
        elif kind in ("refused", "emptied", "stopped", "error"):
            # these endings are amber even when the S1 sentence
            # is the step's own, which the runner leaves untouched.
            kw.update(status_kind="warn")
        self._update_smooth_ui(**kw)
        self._recipe_stage_left = bool(stage)
        adv = self._get_advanced_features_widget()
        stop = getattr(adv, "refine_cancel_preview_btn", None)
        if stop is not None:
            try:
                stop.setToolTip(run.get("stop_tip") or stop.toolTip())
            except Exception:
                pass
        print("[RECIPE] ended: {} ({}) '{}' at step {}/{} - {} judged"
              .format(kind, why or "-", run["name"], run["i"] + 1,
                      len(run["steps"]), run.get("judged", 0)))

    def _recipe_stage_text(self, stage) -> str:
        """``"2/3 · Dilate/Erode"`` -- or ``"2/3"`` with no stage."""
        run = getattr(self, "_refine_recipe_run", None)
        stage = str(stage or "")
        if run is None:
            return stage
        head = "{}/{}".format(run["i"] + 1, len(run["steps"]))
        # Position first, name LAST, so right-elision cuts the name
        # (recipes UX audit M1 / M4).
        return "{}{} — '{}'".format(
            head, " · " + stage if stage else "", run["name"])

    def _recipe_clear_stage_line(self) -> None:
        """The stopped-at line stays until the next Preview, Accept, Discard
        or menu action."""
        if getattr(self, "_recipe_stage_left", False) and not getattr(
                self, "_refine_running", False):
            self._recipe_stage_left = False
            self._update_smooth_ui(stage_text="")

    # --- the three thin slots ------------------------------------

    def _recipe_refuse_click(self) -> bool:
        if not self._recipe_active():
            return False
        from packages.epicycle_annotator.refine_recipe import S1_RUNNING_BLOCK

        self._update_smooth_ui(status_text=S1_RUNNING_BLOCK,
                               status_kind="warn")
        return True

    def _on_refine_preview_clicked(self, *_):
        if self._recipe_refuse_click():
            return
        self._recipe_clear_stage_line()
        self._preview_smooth()

    def _on_refine_accept_clicked(self, *_):
        if self._recipe_refuse_click():
            return
        self._recipe_clear_stage_line()
        self._accept_smooth()

    def _on_refine_discard_clicked(self, *_):
        if self._recipe_refuse_click():
            return
        self._recipe_clear_stage_line()
        self._reject_smooth()

    # --- the menu -------------------------------------------

    def _on_recipes_menu_about_to_show(self, *_) -> None:
        """Rebuilt on every opening, from the live state."""
        from packages.epicycle_annotator import refine_recipe as rr

        adv = self._get_advanced_features_widget()
        if adv is None or not hasattr(adv, "populate_refine_recipes_menu"):
            return
        store, problem = self._recipe_load_store()
        mode = "no_workspace" if store is None else self._recipe_mode()
        channel = ""
        try:
            channel = adv.refine_channel_name()
        except Exception:
            channel = ""
        pre = int(self._smooth_chain_count or 0) if self._smooth_running else 0
        recs = (list(getattr(self, "_refine_step_records", []) or [])
                if self._smooth_running else [])
        entries = rr.menu_entries(
            store, problem, staged_records=recs,
            object_id=self._recipe_selected_object(), pre_staged=pre,
            availability=self._recipe_availability(), channel=channel,
            busy=self._recipe_busy_sentence() or False, mode=mode,
            staged_object=int(self._smooth_label_id or 0)
            if self._smooth_running else 0)
        adv.populate_refine_recipes_menu(entries)

    def _recipe_confirm(self, text, confirm_text="Continue",
                        cancel_text="Cancel") -> bool:
        """The themed confirm, its buttons named for the action (UX M5)."""
        from napari_epicycle.ui_shell.theme import themed_confirm

        parent = self._get_advanced_features_widget() or self
        return bool(themed_confirm(parent, "Recipes", str(text),
                                   confirm_text=confirm_text,
                                   cancel_text=cancel_text))

    def _recipe_ask_name(self, n, suggested):
        from qtpy.QtWidgets import QInputDialog

        parent = self._get_advanced_features_widget() or self
        prompt = ("Name for this step:" if int(n) == 1
                  else "Name for these {} steps:".format(int(n)))
        dlg = QInputDialog(parent)
        dlg.setWindowTitle("Save recipe")
        dlg.setLabelText(prompt)
        dlg.setTextValue(str(suggested))
        # 40 characters is the rule -- the box will not take
        # a 41st, rather than refuse it afterwards (UX audit LOW).
        from qtpy.QtWidgets import QLineEdit

        for edit in dlg.findChildren(QLineEdit):
            edit.setMaxLength(40)
        ok = bool(dlg.exec())
        text = dlg.textValue()
        dlg.deleteLater()
        return str(text or ""), ok

    def _on_recipe_run(self, name, *_) -> None:
        """A run item: the append confirm, then R1."""
        from packages.epicycle_annotator import refine_recipe as rr

        self._recipe_clear_stage_line()
        if self._recipe_refuse_click():
            return
        store, problem = self._recipe_load_store()
        if store is None:
            self._update_smooth_ui(status_text=rr.NO_WORKSPACE,
                                   status_kind="warn")
            return
        if problem:
            self._update_smooth_ui(status_text=problem, status_kind="warn")
            return
        value = store.get(name)
        recipe, rproblem = (rr.recipe_from_json(value) if value is not None
                            else (None, "it is no longer in the file"))
        pre = int(self._smooth_chain_count or 0) if self._smooth_running else 0
        if recipe is not None and pre > 0:
            text = rr.CONFIRM_APPEND.format(
                name=name, steps=rr.plural(len(recipe.steps), "step"),
                staged=rr.plural(pre, "step"), obj=self._smooth_label_id,
                total=pre + len(recipe.steps))
            if not self._recipe_confirm(text, confirm_text="Add"):
                print("[RECIPE] append declined")
                return
        self._recipe_start(name, recipe, rproblem)

    def _on_recipe_load(self, name, *_) -> None:
        """Step 1's settings onto the page; nothing runs."""
        from packages.epicycle_annotator import refine_recipe as rr

        self._recipe_clear_stage_line()
        busy = self._recipe_busy_sentence()
        if busy:
            self._update_smooth_ui(status_text=busy, status_kind="warn")
            return
        store, problem = self._recipe_load_store()
        value = (store or {}).get(name)
        recipe, rproblem = (rr.recipe_from_json(value) if value is not None
                            else (None, "it is no longer in the file"))
        if recipe is None or problem:
            print("[RECIPE] '{}' cannot be read: {}".format(
                name, problem or rproblem))
            self._update_smooth_ui(status_text=rr.S1_UNREADABLE,
                                   status_kind="warn")
            return
        step = recipe.steps[0]
        word = rr.METHOD_WORD.get(step.method, step.method)
        adv = self._get_advanced_features_widget()
        ok, why = adv.set_refine_step_params(step.method, step.params)
        if not ok:
            print("[RECIPE] could not load step 1 of '{}': {}".format(
                name, why))
            self._update_smooth_ui(status_text=rr.S1_LOAD_FAILED.format(
                word=word), status_kind="warn")
            return
        # The status dot follows the page's state, not the action (UX LOW).
        self._update_smooth_ui(status_text=rr.S1_LOADED.format(word=word),
                               status_kind=self._recipe_status_kind())

    def _on_recipe_save(self, *_) -> None:
        """Save the staged chain under a name."""
        import datetime

        from packages.epicycle_annotator import refine_recipe as rr

        self._recipe_clear_stage_line()
        if self._recipe_refuse_click():
            return
        cb = getattr(self, "_save_refine_recipe_cb", None)
        store, problem = self._recipe_load_store()
        if cb is None or store is None:
            self._update_smooth_ui(status_text=rr.NO_WORKSPACE,
                                   status_kind="warn")
            return
        if problem:
            self._update_smooth_ui(status_text=problem, status_kind="warn")
            return
        recs = (list(getattr(self, "_refine_step_records", []) or [])
                if self._smooth_running else [])
        why = rr.save_blocker(recs)
        if why:
            self._update_smooth_ui(status_text=why, status_kind="warn")
            return
        # The steps summary is the suggested name (UX audit LOW), cut to
        # the 40 characters a name may have.
        suggested = rr.elide(rr.steps_summary(recs, max_chars=rr.MAX_NAME),
                             rr.MAX_NAME)
        raw, ok = self._recipe_ask_name(len(recs), suggested)
        if not ok:
            return
        name = rr.normalise_name(raw)
        if name is None:
            self._update_smooth_ui(status_text=rr.S1_BAD_NAME,
                                   status_kind="warn")
            return
        key = name
        existing = rr.find_existing(store, name)
        if existing is not None:
            summary = rr.summary_of(store[existing]) or "cannot be read"
            if not self._recipe_confirm(rr.CONFIRM_REPLACE.format(
                    name=existing, summary=summary,
                    steps=rr.plural(len(recs), "step")),
                    confirm_text="Replace"):
                return
            key = existing                  # the existing spelling (M3)
        try:
            known = bool(self._refine_spacing_known())
            obj = rr.recipe_to_json(
                recs,
                created=datetime.datetime.now().isoformat(timespec="seconds"),
                spacing_known=known,
                spacing=self._refine_spacing() if known else ())
            err = cb(key, obj)
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
        if err:
            # Fixed words on screen (UX M3); the raw reason to the console
            # (the controller has already written it to the log).
            print("[RECIPE] save of '{}' failed: {}".format(key, err))
            self._update_smooth_ui(status_text=rr.S1_SAVE_FAILED,
                                   status_kind="warn")
            return
        print("[RECIPE] saved '{}' ({} steps)".format(key, len(recs)))
        self._update_smooth_ui(status_text=rr.S1_SAVED.format(
            steps=rr.plural(len(recs), "step")),
            status_kind=self._recipe_status_kind())

    def _on_recipe_delete(self, name, *_) -> None:
        from packages.epicycle_annotator import refine_recipe as rr

        self._recipe_clear_stage_line()
        if self._recipe_refuse_click():              # audit L4
            return
        cb = getattr(self, "_delete_refine_recipe_cb", None)
        if cb is None:
            self._update_smooth_ui(status_text=rr.NO_WORKSPACE,
                                   status_kind="warn")
            return
        if not self._recipe_confirm(rr.CONFIRM_DELETE.format(name=name),
                                    confirm_text="Delete",
                                    cancel_text="Keep it"):
            return
        try:
            err = cb(name)
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
        if err:
            print("[RECIPE] delete of '{}' failed: {}".format(name, err))
            self._update_smooth_ui(status_text=rr.S1_DELETE_FAILED,
                                   status_kind="warn")
            return
        print("[RECIPE] deleted '{}'".format(name))
        self._update_smooth_ui(status_text=rr.S1_DELETED,
                               status_kind=self._recipe_status_kind())

    def _recipe_status_kind(self) -> str:
        """The dot for a sentence about the menu: the PAGE's state."""
        return "review" if getattr(self, "_smooth_running", False) else "idle"

    # ---------------------------------------------------------------
    # EDT Overlay
    # ---------------------------------------------------------------

    def _toggle_edt_overlay(self, checked: bool) -> None:
        """Toggle the EDT distance transform overlay layer.

        When checked, computes the EDT of the currently selected label
        and renders it as a ``turbo`` colormap Image layer.

        Key fix (matching napari-multiview-annotator commit c42191a):
        ``contrast_limits`` MUST be pinned to ``[0, edt_max]`` from the
        ROI computation.  Without this, napari auto-computes clims from
        the current visible slice; if that slice is outside the label's
        bounding box (all zeros), clims=[0,0] → invisible / wrong overlay.
        """
        if not checked:
            self._remove_edt_overlay()
            return

        from packages.epicycle_annotator.split import compute_edt_overlay

        labels_name = self._active_labels_name

        # During an active split, the target label lives on the Refine layer
        # (the shared volume has a hole where the target was).
        if self._split_running and self._refine_layer is not None:
            source_data = self._refine_layer.data
            selected_label = self._split_label_id or 0
        else:
            source_data = self.sync_manager.shared_labels.get(labels_name)
            selected_label = self._get_selected_label()

        if source_data is None or selected_label <= 0:
            return

        edt_crop, roi, edt_max = compute_edt_overlay(source_data, selected_label)
        if edt_crop is None or edt_max <= 0:
            return

        # Compute translate from ROI slices so the small crop is positioned
        # correctly without embedding into a full-volume array (~380x smaller).
        z_start = roi[0].start
        y_start = roi[1].start
        x_start = roi[2].start
        translate_zyx = [z_start, y_start, x_start]

        # Pin contrast limits from the global EDT max — prevents napari
        # auto-computing from a zero-filled slice.
        clims = [0.0, edt_max]

        overlay_name = "EDT Overlay"
        existing = self._layer_by_name(self.viewer.layers, overlay_name)
        if existing is not None:
            if existing.data.shape == edt_crop.shape:
                existing.data = edt_crop
                existing.translate = translate_zyx
                existing.contrast_limits = clims
                existing.refresh()
            else:
                # Hide before removing, so the render loop never draws a freed GPU texture.
                existing.visible = False
                self.viewer.layers.remove(existing)
                self.viewer.add_image(
                    edt_crop,
                    name=overlay_name,
                    colormap="turbo",
                    opacity=0.5,
                    blending="additive",
                    contrast_limits=clims,
                    translate=translate_zyx,
                )
        else:
            self.viewer.add_image(
                edt_crop,
                name=overlay_name,
                colormap="turbo",
                opacity=0.5,
                blending="additive",
                contrast_limits=clims,
                translate=translate_zyx,
            )

        # Ensure the paintable labels layer stays on top so painting is not
        # blocked.  During split, that's the Refine layer; otherwise main.
        from napari.layers import Labels
        if self._split_running and self._refine_layer is not None:
            paint_layer = self._refine_layer
        else:
            paint_layer = self._layer_by_name(self.viewer.layers, labels_name)
        if isinstance(paint_layer, Labels):
            try:
                layer_idx = list(self.viewer.layers).index(paint_layer)
                top_idx = len(self.viewer.layers) - 1
                if layer_idx < top_idx:
                    self.viewer.layers.move(layer_idx, top_idx)
            except (ValueError, IndexError):
                pass

        # Propagate to subviewers manually with per-orientation transpose
        # and permuted translate (ROI-sized crop, not full volume).
        from packages.epicycle_annotator.orientation import transpose_data
        style = {
            "colormap": "turbo",
            "opacity": 0.5,
            "blending": "additive",
            "contrast_limits": clims,
            "rendering": "mip",
        }
        orient_configs = [
            ("XY", self.viewer_model_xy, edt_crop, translate_zyx),
            ("XZ", self.viewer_model_xz,
             transpose_data(edt_crop, "XZ"),
             [y_start, z_start, x_start]),
            ("YZ", self.viewer_model_yz,
             transpose_data(edt_crop, "YZ"),
             [x_start, y_start, z_start]),
        ]
        for orient, vm, data, trans in orient_configs:
            sub_name = f"{overlay_name}_{orient}"
            existing_sub = self._layer_by_name(vm.layers, sub_name)
            if existing_sub is not None:
                if existing_sub.data.shape == data.shape:
                    existing_sub.data = data
                    existing_sub.translate = trans
                    for k, v in style.items():
                        try:
                            setattr(existing_sub, k, v)
                        except Exception:
                            pass
                else:
                    # Hide before removing, so the render loop never draws a freed GPU texture.
                    existing_sub.visible = False
                    vm.layers.remove(existing_sub)
                    vm.add_image(data, name=sub_name, translate=trans, **style)
            else:
                vm.add_image(data, name=sub_name, translate=trans, **style)

        # During split, hide the EDT layer in the main 3D viewer.
        # The 3D rendering (MIP) is expensive and unnecessary — only the
        # 2D subviewer slices need the overlay.
        overlay_layer = self._layer_by_name(self.viewer.layers, overlay_name)
        if overlay_layer is not None and self._split_running:
            overlay_layer.visible = False

    # ---------------------------------------------------------------
    # Split helpers
    # ---------------------------------------------------------------

    def _update_split_ui(self, **kwargs) -> None:
        """Proxy to update split UI state in the AdvancedFeaturesWidget."""
        adv_widget = self._get_advanced_features_widget()
        if adv_widget and hasattr(adv_widget, "update_split_ui_state"):
            adv_widget.update_split_ui_state(**kwargs)
        # Also mirror status to the main interp label as a fallback.
        status = kwargs.get("status_text", "")
        if status and hasattr(self, "_interp_status_label"):
            self._interp_status_label.setText(status)

    def _invalidate_afw_cache(self) -> None:
        """Forget the cached Advanced Features (dock, widget) pair."""
        self._afw_cache = None

    def _get_advanced_features_widget(self):
        """Find the AdvancedFeaturesWidget dock if open.

        The search (``launcher._find_dock_by_title`` -> ``findChildren``) costs
        ~31 ms on a real session and this getter is called 14-20 times per user
        action, so the resolved (dock, widget) pair is cached on the annotator.
        A cached pair is used only while the dock is alive, still parented and
        still holds the same widget; otherwise the search runs again (final
        fix pass P1).  ``None`` is never cached.
        """
        cached = getattr(self, "_afw_cache", None)
        if cached is not None:
            if _afw_pair_alive(*cached):
                return cached[1]
            self._afw_cache = None
        try:
            from napari_epicycle.annotator.launcher import (
                _ADVANCED_DOCK_NAMES,
                _resolve_existing_dock,
            )

            dock_name = _ADVANCED_DOCK_NAMES.get(
                "3d_curation", "EpiCycle 3D Advanced Features"
            )
            dock = _resolve_existing_dock(self.viewer, dock_name)
            if dock is not None:
                widget = dock.widget() if hasattr(dock, "widget") else dock
                if widget is not None:
                    self._afw_cache = (dock, widget)
                return widget
        except Exception:
            pass
        return None

    def _remove_edt_overlay(self) -> None:
        """Remove the 'EDT Overlay' layer from main viewer and subviewers."""
        overlay_name = "EDT Overlay"
        existing = self._layer_by_name(self.viewer.layers, overlay_name)
        if existing is not None:
            try:
                # Hide before removing — disconnect the vispy visual from the
                # scene graph so the render loop cannot touch a freed GPU
                # texture.
                existing.visible = False
                self.viewer.layers.remove(existing)
            except Exception:
                pass
        # Remove from subviewers.
        for suffix in ("_XY", "_XZ", "_YZ"):
            sub_name = overlay_name + suffix
            for vm in (
                self.viewer_model_xy,
                self.viewer_model_xz,
                self.viewer_model_yz,
            ):
                layer = self._layer_by_name(vm.layers, sub_name)
                if layer is not None:
                    try:
                        # Hide before removing, so the render loop never draws a freed GPU texture.
                        layer.visible = False
                        vm.layers.remove(layer)
                    except Exception:
                        pass

    # ---------------------------------------------------------------
    # Probability Overlay
    # ---------------------------------------------------------------

    def _toggle_prob_overlay(self, checked: bool) -> None:
        """Toggle the probability volume overlay layer.

        When checked, crops the prob volume to the currently selected
        label's bounding box and renders it as an Image layer with the
        colormap selected in the Refine tab combo.  Follows the EDT
        overlay pattern: ROI-based crop, pinned contrast_limits,
        translate from ROI origin, propagated to subviewers with transpose.
        """
        adv = self._get_advanced_features_widget()

        def _say(text):
            # B3 audit: the tick used to stay ticked over NOTHING.  Take it
            # back off and say why, on screen.
            if adv is not None and hasattr(
                    adv, "set_refine_prob_overlay_message"):
                adv.set_refine_prob_overlay_message(text)
            if text:
                print("[REFINE] probability overlay: {}".format(text))

        if not checked:
            self._remove_prob_overlay()
            _say("")
            return

        from napari_epicycle.annotator.advanced_features_widget import (
            REFINE_PROB_OVERLAY_NO_MAP,
            REFINE_PROB_OVERLAY_NO_OBJECT,
        )

        labels_name = self._active_labels_name
        shared_data = self.sync_manager.shared_labels.get(labels_name)
        selected_label = self._get_selected_label()

        if shared_data is None or selected_label <= 0:
            _say(REFINE_PROB_OVERLAY_NO_OBJECT)
            return

        # Get the label's bounding box.
        roi, _ = self._snapshot_label_roi(shared_data, selected_label)

        # Get prob volume for that ROI.
        prob_crop = self._get_prob_roi(roi)
        if prob_crop is None:
            _say(REFINE_PROB_OVERLAY_NO_MAP)
            return
        _say("")

        prob_crop = prob_crop.astype(np.float32)

        max_val = float(prob_crop.max()) if prob_crop.size > 0 and np.any(prob_crop > 0) else 1.0
        clims = [0.0, max_val]

        z_start = roi[0].start
        y_start = roi[1].start
        x_start = roi[2].start
        translate_zyx = [z_start, y_start, x_start]

        # Read colormap from the Refine tab combo (default viridis).
        adv = self._get_advanced_features_widget()
        cmap = "viridis"
        if adv is not None and hasattr(adv, "refine_prob_cmap_combo"):
            cmap = adv.refine_prob_cmap_combo.currentText() or "viridis"

        overlay_name = "Prob Overlay"

        # --- Main viewer ---
        existing = self._layer_by_name(self.viewer.layers, overlay_name)
        if existing is not None:
            if existing.data.shape == prob_crop.shape:
                existing.data = prob_crop
                existing.translate = translate_zyx
                existing.contrast_limits = clims
                existing.colormap = cmap
                existing.refresh()
            else:
                # Hide before removing, so the render loop never draws a freed GPU texture.
                existing.visible = False
                self.viewer.layers.remove(existing)
                self.viewer.add_image(
                    prob_crop, name=overlay_name, colormap=cmap,
                    opacity=0.5, blending="additive",
                    contrast_limits=clims, translate=translate_zyx,
                )
        else:
            self.viewer.add_image(
                prob_crop, name=overlay_name, colormap=cmap,
                opacity=0.5, blending="additive",
                contrast_limits=clims, translate=translate_zyx,
            )

        # Ensure labels layer stays on top.
        main_layer = self._layer_by_name(self.viewer.layers, labels_name)
        if isinstance(main_layer, Labels):
            try:
                idx = list(self.viewer.layers).index(main_layer)
                top = len(self.viewer.layers) - 1
                if idx < top:
                    self.viewer.layers.move(idx, top)
            except (ValueError, IndexError):
                pass

        # --- Propagate to subviewers ---
        from packages.epicycle_annotator.orientation import transpose_data

        style = {
            "colormap": cmap,
            "opacity": 0.5,
            "blending": "additive",
            "contrast_limits": clims,
            "rendering": "mip",
        }
        orient_configs = [
            ("XY", self.viewer_model_xy, prob_crop, translate_zyx),
            ("XZ", self.viewer_model_xz,
             transpose_data(prob_crop, "XZ"),
             [y_start, z_start, x_start]),
            ("YZ", self.viewer_model_yz,
             transpose_data(prob_crop, "YZ"),
             [x_start, y_start, z_start]),
        ]
        for orient, vm, data, trans in orient_configs:
            sub_name = f"{overlay_name}_{orient}"
            existing_sub = self._layer_by_name(vm.layers, sub_name)
            if existing_sub is not None:
                if existing_sub.data.shape == data.shape:
                    existing_sub.data = data
                    existing_sub.translate = trans
                    for k, v in style.items():
                        try:
                            setattr(existing_sub, k, v)
                        except Exception:
                            pass
                else:
                    # Hide before removing, so the render loop never draws a freed GPU texture.
                    existing_sub.visible = False
                    vm.layers.remove(existing_sub)
                    vm.add_image(data, name=sub_name, translate=trans, **style)
            else:
                vm.add_image(data, name=sub_name, translate=trans, **style)

        # Hide in main 3D viewer (MIP rendering is expensive).
        overlay_layer = self._layer_by_name(self.viewer.layers, overlay_name)
        if overlay_layer is not None:
            overlay_layer.visible = False

    def _remove_prob_overlay(self) -> None:
        """Remove the 'Prob Overlay' layer from main viewer and subviewers."""
        overlay_name = "Prob Overlay"
        existing = self._layer_by_name(self.viewer.layers, overlay_name)
        if existing is not None:
            try:
                # Hide before removing — disconnect the vispy visual from the
                # scene graph so the render loop cannot touch a freed GPU
                # texture.
                existing.visible = False
                self.viewer.layers.remove(existing)
            except Exception:
                pass
        for suffix in ("_XY", "_XZ", "_YZ"):
            sub_name = overlay_name + suffix
            for vm in (
                self.viewer_model_xy,
                self.viewer_model_xz,
                self.viewer_model_yz,
            ):
                layer = self._layer_by_name(vm.layers, sub_name)
                if layer is not None:
                    try:
                        # Hide before removing, so the render loop never draws a freed GPU texture.
                        layer.visible = False
                        vm.layers.remove(layer)
                    except Exception:
                        pass

        # Uncheck the prob overlay checkbox if it exists.
        adv = self._get_advanced_features_widget()
        if adv is not None and hasattr(adv, "refine_prob_overlay_cb"):
            adv.refine_prob_overlay_cb.blockSignals(True)
            adv.refine_prob_overlay_cb.setChecked(False)
            adv.refine_prob_overlay_cb.blockSignals(False)

    @staticmethod
    def _snapshot_label_roi(
        volume: np.ndarray, label_id: int, edited_slices: dict = None,
        *, _precomputed_mask: np.ndarray = None, margin: int = 15,
    ) -> tuple:
        """Return (roi_slices, roi_copy) covering the label's full bounding box.

        The ROI is computed from the union of:
        1. The label's actual extent in the volume (via axis-reduced
           ``np.any`` — avoids materializing a dense coordinate array)
        2. The range of all edited slices for this label (interpolation
           fills between these, potentially far beyond current extent)

        A generous margin accounts for postprocessing growth
        (``binary_fill_holes`` + ``binary_closing``).
        Falls back to the full volume if the label is not found.

        Parameters
        ----------
        _precomputed_mask : np.ndarray, optional
            If provided, reuse this boolean mask (``volume == label_id``)
            instead of recomputing it.  Avoids a redundant full-volume scan
            when the caller already has the mask.
        margin : int, optional
            Voxels of padding around the label, default 15 (the value this
            was hard-coded to).  The Reshape panel reads its "Search
            padding" knob into this at Start -- ``reshape_region``'s own
            ``margin=`` is ignored whenever a ``roi`` is passed, and the
            panel always passes one, so the snapshot, the Refine layer and
            the write-back ROI all move together or not at all.
        """
        # Use axis-reduced np.any + np.where for bounding box —
        # much faster than np.argwhere which materialises a dense
        # (N_voxels, 3) coordinate array on large volumes.
        mask = _precomputed_mask if _precomputed_mask is not None else (volume == label_id)
        z_any = np.any(mask, axis=(1, 2))
        if not z_any.any():
            # Label not present — fall back to full volume.
            roi = (
                slice(0, volume.shape[0]),
                slice(0, volume.shape[1]),
                slice(0, volume.shape[2]),
            )
            return roi, volume[roi].copy()

        shape = volume.shape
        # Generous margin for postprocessing growth.  The Reshape panel's
        # "Search padding" knob overrides it at Start: the
        # engine's own `margin=` is IGNORED whenever a `roi` is passed,
        # and the panel always passes one, so this is the only place the
        # number can be read.
        margin = max(0, int(margin))

        y_any = np.any(mask, axis=(0, 2))
        x_any = np.any(mask, axis=(0, 1))
        z_idx = np.where(z_any)[0]
        y_idx = np.where(y_any)[0]
        x_idx = np.where(x_any)[0]
        mins = np.array([z_idx[0], y_idx[0], x_idx[0]])
        maxs = np.array([z_idx[-1], y_idx[-1], x_idx[-1]])

        # Extend bounds to cover edited slice ranges.
        # XY slices → Z axis, XZ slices → Y axis, YZ slices → X axis.
        if edited_slices is not None:
            axis_map = {"XY": 0, "XZ": 1, "YZ": 2}
            for orient, axis in axis_map.items():
                orient_dict = edited_slices.get(orient, {})
                for key, indices in orient_dict.items():
                    if int(key) == int(label_id) and indices:
                        int_indices = [int(s) for s in indices]
                        mins[axis] = min(int(mins[axis]), min(int_indices))
                        maxs[axis] = max(int(maxs[axis]), max(int_indices))

        z_lo = max(0, int(mins[0]) - margin)
        z_hi = min(shape[0], int(maxs[0]) + margin + 1)
        y_lo = max(0, int(mins[1]) - margin)
        y_hi = min(shape[1], int(maxs[1]) + margin + 1)
        x_lo = max(0, int(mins[2]) - margin)
        x_hi = min(shape[2], int(maxs[2]) + margin + 1)

        roi = (slice(z_lo, z_hi), slice(y_lo, y_hi), slice(x_lo, x_hi))
        return roi, volume[roi].copy()

    def _set_selected_label_all(self, label_id: int) -> None:
        """Set selected_label on main layer.

        Subviewers are synced automatically via ``sync_selected_label()``
        in the ``LabelSyncManager`` — no manual loop needed.  The sync
        manager has a reentrant guard to prevent N² event cascades.
        """
        labels_name = self._active_labels_name
        main_layer = self._layer_by_name(self.viewer.layers, labels_name)
        if isinstance(main_layer, Labels):
            # Phase 2: When 3D proxy is active, hide main layer BEFORE
            # changing selected_label to avoid the expensive iso_categorical
            # re-render.
            #
            # Phase 3: ALSO hide for multi-label DirectLabelColormap
            # isolation in 3D — assigning selected_label on a visible 3D
            # iso_categorical layer triggers a synchronous re-render even
            # though selected_label has no visual effect under
            # DirectLabelColormap.  Difference from proxy mode: multi-label
            # needs main visible in steady state, so restore visibility on
            # next event loop tick via QTimer (same pattern as the
            # multi-isolation colormap apply).
            _dims = getattr(self.viewer, "dims", None)
            _ndisplay = getattr(_dims, "ndisplay", 2) if _dims else 2
            _multi_iso_3d = (
                _ndisplay == 3
                and getattr(self, "_isolation_active", False)
                and not self._isolation_proxy_active
            )
            # Phase 4: Hide-set-show in plain 3D mode too, even with no
            # isolation active.  selected_label assignment on a visible 3D
            # Labels layer triggers a synchronous iso_categorical re-render
            # via napari's refresh() — that's the 240-370ms steady-state
            # lag during navigator browsing.  Same pattern as multi-isolation:
            # hide → assign → QTimer restore.  Trade-off: brief subviewer
            # flicker per click (cascade follows main visibility).
            _plain_3d_nav = (
                _ndisplay == 3
                and not self._isolation_proxy_active
                and not getattr(self, "_isolation_active", False)
            )
            _was_visible = main_layer.visible
            _hide_for_perf = (
                _ndisplay == 3
                and (self._isolation_proxy_active or _multi_iso_3d
                     or _plain_3d_nav)
            )
            if _hide_for_perf:
                if getattr(main_layer, "show_selected_label", False):
                    main_layer.show_selected_label = False
                if _was_visible:
                    # 2026-09-02 (lasso/isolation sync audit).  The hide
                    # MUST stay unblocked: napari's vispy node only learns
                    # about it through ``events.visible`` (verified live —
                    # a blocked hide leaves the full volume drawn under the
                    # proxy while the model says hidden).  The same event
                    # drives ``subviewers.py::sync_layer_visibility``, which
                    # switches the XY/XZ/YZ label layers OFF.  In the
                    # plain-3D / multi-iso cases the deferred re-show
                    # switches them back ON; with a PROXY active nothing
                    # re-shows main (the proxy renders 3D), so nothing ever
                    # switched the subviewers back on — "selected cells not
                    # rendered in the multiview annotator".  Snapshot the
                    # subviewer label layers that are visible now and put
                    # them back right after the cascade (3 cheap 2D writes,
                    # proxy case only).
                    _subs_to_restore = (
                        self._visible_subviewer_label_layers()
                        if self._isolation_proxy_active else ()
                    )
                    main_layer.visible = False
                    for _sub in _subs_to_restore:
                        try:
                            _sub.visible = True
                        except Exception:
                            pass
                    if _subs_to_restore:
                        _heal_render(self, None, tag="proxy-sub-restore")

            main_layer.selected_label = label_id

            # Restore visibility for multi-label 3D AND plain 3D nav —
            # proxy mode keeps main hidden (proxy layer handles rendering).
            _restore_after = (
                (_multi_iso_3d or _plain_3d_nav)
                and _was_visible
            )
            if _restore_after:
                self._schedule_main_perf_reshow(main_layer)

    # -- perf hide / re-show bookkeeping (2026-09-02) -----------------------
    #
    # ``_set_selected_label_all`` hides the main layer for one event-loop
    # tick around a ``selected_label`` assignment and re-shows it via a
    # ``QTimer.singleShot(0, ...)``.  That timer used to be an unconditional
    # ``setattr(layer, "visible", True)`` — if a proxy took over in the
    # same tick (navigator click with Isolate ON → ``_update_isolation_proxy``
    # hides main by design), the stale timer un-hid the main layer
    # underneath the proxy.  A monotonically increasing token makes every
    # scheduled re-show verifiable: proxy creation bumps the token so any
    # re-show scheduled before it becomes a no-op.

    def _heal_label_render(self, main_layer=None, *, tag: str = "") -> int:
        """Repair vispy colormap nodes poisoned while hidden (napari >= 0.7).

        Call right after a Labels layer (main and/or its XY/XZ/YZ mirrors)
        is re-shown.  Zero cost when healthy; rebuilds only a node whose
        shader class disagrees with the layer's colormap — the
        hide -> recolour -> navigate -> re-show sequence leaves exactly that
        behind (``packages/epicycle_annotator/labels_render.py``).  Returns
        the number of layers rebuilt.  Never raises.
        """
        try:
            from packages.epicycle_annotator.labels_render import heal_labels_render
        except Exception:
            return 0
        pairs = []
        try:
            if main_layer is None and self._active_labels_name:
                main_layer = self.viewer.layers[self._active_labels_name]
        except Exception:
            main_layer = None
        if main_layer is not None:
            try:
                pairs.append((main_layer, self.viewer.window._qt_viewer))
            except Exception:
                pass
        sm = getattr(self, "subviewer_manager", None)
        if sm is not None:
            qt_by_vm = {
                id(getattr(sm, "viewer_model_xy", None)): getattr(sm, "qt_viewer_xy", None),
                id(getattr(sm, "viewer_model_xz", None)): getattr(sm, "qt_viewer_xz", None),
                id(getattr(sm, "viewer_model_yz", None)): getattr(sm, "qt_viewer_yz", None),
            }
            try:
                for layer in self._all_subviewer_label_layers():
                    # find the ViewerModel that owns this layer
                    owner = None
                    for vm in (self.viewer_model_xy, self.viewer_model_xz, self.viewer_model_yz):
                        try:
                            if layer in vm.layers:
                                owner = vm
                                break
                        except Exception:
                            continue
                    pairs.append((layer, qt_by_vm.get(id(owner))))
            except Exception:
                pass
        try:
            return heal_labels_render(pairs, tag=tag)
        except Exception:
            return 0

    def _schedule_main_perf_reshow(self, main_layer) -> None:
        """Re-show *main_layer* on the next event-loop tick, unless a proxy
        has taken over (or a newer re-show superseded this one) by then."""
        from qtpy.QtCore import QTimer

        token = getattr(self, "_perf_reshow_token", 0) + 1
        self._perf_reshow_token = token
        _l = main_layer

        def _reshow(l=_l, tok=token):
            if getattr(self, "_perf_reshow_token", None) != tok:
                return  # superseded or cancelled
            if getattr(self, "_isolation_proxy_active", False):
                return  # a proxy now owns 3D rendering — main stays hidden
            l.visible = True  # unblocked: vispy node + subviewer cascade follow
            _heal_render(self, l, tag="perf-reshow")
            try:
                from napari_epicycle.fusion_reconstruction._render_diag import render_state as _rs
                _rs('ann-after-reshow', l, viewer=self.viewer)
            except Exception:
                pass

        QTimer.singleShot(0, _reshow)

    def _cancel_main_perf_reshow(self) -> None:
        """Invalidate any pending perf re-show (call before a proxy hides
        the main layer on purpose, and on close)."""
        self._perf_reshow_token = getattr(self, "_perf_reshow_token", 0) + 1

    def _visible_subviewer_label_layers(self) -> list:
        """The XY/XZ/YZ label layers mirroring the active labels layer that
        are currently visible (same lookup as ``_propagate_show_selected_label``).
        Empty when there is no sync manager (tests / not yet loaded)."""
        labels_name = getattr(self, "_active_labels_name", None)
        sync = getattr(self, "sync_manager", None)
        if not labels_name or sync is None:
            return []
        try:
            base = sync._base_label_name(labels_name)
        except Exception:
            return []
        out = []
        for vm_name in ("viewer_model_xy", "viewer_model_xz", "viewer_model_yz"):
            vm = getattr(sync, vm_name, None)
            for layer in (getattr(vm, "layers", None) or []):
                try:
                    if (
                        isinstance(layer, Labels)
                        and sync._base_label_name(layer.name) == base
                        and getattr(layer, "visible", False)
                    ):
                        out.append(layer)
                except Exception:
                    continue
        return out
