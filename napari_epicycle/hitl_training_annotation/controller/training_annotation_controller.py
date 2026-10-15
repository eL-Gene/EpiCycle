import numpy as np
import json
import os
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
import tifffile

from qtpy.QtCore import Qt, QThread, QTimer, Signal, QObject
from qtpy.QtGui import QBrush, QColor
from qtpy.QtWidgets import (
    QFileDialog,
    QMessageBox,
)


def _compact_label_ids(mask: np.ndarray) -> np.ndarray:
    """Renumber foreground label IDs in ``mask`` to a dense 1..N range.

    2026-05-28 (label-compaction-v2): Step 2's
    ``_insert_slice_mask_with_unique_labels`` applies a monotonic
    ``base_max`` offset on every prediction insert, and the persisted
    masks carry those offsets back to disk.  On reload the restore loop
    re-offsets the already-offset masks, and re-persist writes the
    re-inflated IDs — so across sessions ``max(label_id)`` compounds into
    the MILLIONS while any single slice still has ~100 cells.  Bloated IDs
    make napari's 3D label LUT (sized to max_id) crawl.

    Compacting the 2D mask right before it is written to disk breaks that
    compounding loop at the source: every persisted mask lands at minimal-
    width IDs, so reloads start clean and the volume max stays bounded to
    roughly the cell count.

    Behavior (all verified by tests):
      - Background (0) is preserved — always stays 0.
      - Foreground labels are renumbered to a dense 1..N by sorted-unique
        order.
      - **Identity preserved:** two pixels sharing an old label share the
        same new label (a bijection over distinct values).
      - Voxel count, foreground/background partition, and shape preserved.
      - dtype preserved.
      - Idempotent — compacting a compact mask returns an equivalent mask.

    Safe at persist time: no persisted artifact outside the TIFF itself
    references per-cell integer label IDs (``sample_records``,
    ``round_metadata.json``, ``final_label_id_map`` are all slice-label-
    STRING keyed; Cellpose training is ID-agnostic — it treats each
    distinct value as one cell regardless of the number).

    Aliasing note: the fast paths (all-background, already-compact 1..N)
    return the INPUT array, not a copy.  Callers that mutate the result
    in place must ``.copy()`` first — ``_finalize_labels`` does;
    ``_persist_curated_training_pair`` only reads it for the TIFF write.
    """
    arr = np.asarray(mask)
    uniq = np.unique(arr)
    # Fast path: already compact (0..N contiguous) — return as-is.
    fg = uniq[uniq != 0]
    if fg.size == 0:
        return arr  # all background
    if fg[0] == 1 and fg[-1] == fg.size:
        # Foreground is exactly 1..N already; nothing to do.
        return arr
    # Build a lookup: old value -> new dense id (bg stays 0).
    # Vectorized (2026-06-02): `np.searchsorted(fg, v)` gives each foreground
    # value its rank in the sorted-unique array; +1 maps that to a dense
    # 1..N while background stays 0.  Equivalent to the old per-label Python
    # loop (`for new_id, old_id in enumerate(fg): out[arr==old_id]=new_id`)
    # but O(S log F) in one pass instead of O(F·S) — the loop held the GIL
    # for ~hundreds of ms on busy slices, which would have defeated moving
    # the persist off the main thread.  fg is sorted-unique-nonzero, so every
    # foreground value is guaranteed present in fg (searchsorted is exact).
    out = np.zeros_like(arr)
    fg_mask = arr != 0
    out[fg_mask] = (np.searchsorted(fg, arr[fg_mask]) + 1).astype(out.dtype)
    return out


def _array_signature(arr: np.ndarray) -> int:
    """Cheap in-memory content signature for change detection (no disk I/O).

    2026-06-02 (fix/step2-persist-tiff-perf): used by
    ``_persist_curated_training_pair`` to decide whether a training-pair
    TIFF actually needs rewriting, replacing a per-edit ``tifffile.imread``
    of the on-disk mask (a ~hundreds-of-ms read on a slow/network drive).

    Signature combines shape + dtype + raw bytes so two arrays with the
    same content hash equal and different content (almost certainly) does
    not.  ``hash(bytes)`` is salted per-process (PYTHONHASHSEED) but the
    signatures are only ever compared WITHIN a single session against
    other signatures computed by this same function — never persisted —
    so per-process salting is irrelevant.  Cost is one in-memory
    ``tobytes()`` copy + hash (single-digit ms for a 1024^2 slice), versus
    a full disk read it replaces.
    """
    a = np.ascontiguousarray(arr)
    return hash((a.shape, str(a.dtype), a.tobytes()))


def _pair_write_plan(img_exists, mask_exists, img_sig, mask_sig, prior):
    """Decide which of (image, mask) TIFFs actually need (re)writing.

    A write is SKIPPED only when the on-disk file already exists AND its
    in-memory content signature is unchanged since the last persist this
    session (``prior`` is the per-slice ``{"img": sig, "mask": sig}`` dict,
    possibly empty/missing keys).  Missing file OR changed/absent signature
    forces the write.

    Pure function (no I/O, no ``self``) so the skip logic is unit-testable
    in isolation.  Returns ``(write_img, write_mask)``.
    """
    write_img = (not img_exists) or prior.get("img") != img_sig
    write_mask = (not mask_exists) or prior.get("mask") != mask_sig
    return write_img, write_mask


def _atomic_imwrite(path: str, data: np.ndarray) -> None:
    """Write a TIFF atomically: stage to a temp file in the SAME directory,
    then ``os.replace`` (atomic on a single filesystem).  A crash or kill
    mid-write leaves the PREVIOUS file intact instead of a torn TIFF — which
    matters now that writes run on a background thread (Stage 2).
    """
    import tempfile
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".tif", dir=str(p.parent))
    os.close(fd)
    try:
        tifffile.imwrite(tmp, data)
        os.replace(tmp, str(p))
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


class _AsyncPairWriter:
    """Single background FIFO thread that writes training-pair TIFFs off the
    Qt main thread (Stage 2 of the 2D-paint-lag fix, 2026-06-02).

    Why a single worker (not a pool): writes to the SAME path must stay
    ordered (last-wins).  One FIFO worker guarantees that for free and caps
    the disk to one concurrent writer.  ``tifffile.imwrite`` releases the GIL
    during I/O, so the main thread keeps painting/zooming while a write is in
    flight.

    Contract: arrays handed to :meth:`submit` MUST be snapshots the main
    thread will not mutate (callers pass freshly ``.astype()``-copied /
    compacted arrays).  :meth:`flush` is the correctness barrier — it blocks
    until the queue is fully drained, and is called ONLY at hard barriers
    (train / run-switch / browse / import / shutdown — every path that reads
    the pairs back from disk), NEVER on the per-stroke debounce path, so
    steady-state drawing never waits on disk.  (Patch-switch does NOT flush
    and does not need to: it never reads the just-written pair back.)
    """

    def __init__(self):
        self._q = queue.Queue()
        self._thread = None
        self._start_lock = threading.Lock()

    def _ensure_thread(self):
        with self._start_lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, name="epicycle-pair-writer", daemon=True
                )
                self._thread.start()

    def submit(self, path: str, data: np.ndarray) -> None:
        self._ensure_thread()
        self._q.put((str(path), data))

    def _run(self):
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                path, data = item
                try:
                    _atomic_imwrite(path, data)
                except Exception as exc:
                    # A failed write leaves no file (atomic), so the next
                    # persist's `not file_exists` check forces a rewrite —
                    # the in-memory sig optimism is self-healing.  Surfaced
                    # so a dropped background write of the user's edit is
                    # never fully silent.
                    print(f"[WARN] async training-pair write FAILED "
                          f"{path}: {exc!r}")
            finally:
                self._q.task_done()

    def flush(self, timeout: float | None = None) -> bool:
        """Block until all queued writes have completed (correctness barrier).

        Called only at hard barriers so Train / switch / close never read a
        not-yet-written pair.  No-op when the queue is already empty.

        Args:
            timeout: optional bound in seconds.  ``None`` (the default, used
                by every correctness barrier that then READS the pairs back)
                waits indefinitely.  The shutdown path passes a bound —
                release-hardening 1b#1: a stuck/slow TIFF write must not hold
                napari's close hostage, and the abort flag is already set by
                then so the training thread is on its way down regardless.

        Returns:
            True if the queue drained, False if the timeout expired first
            (in which case the backlog is reported on stdout).
        """
        # Ensure a live worker exists before joining.  If the worker ever died
        # with items still queued (no realistic path does this, but it's the
        # one way join() could block forever), this revives it so the backlog
        # drains instead of hanging the barrier.  Defensive; audit 2026-06-02.
        self._ensure_thread()
        if timeout is None:
            self._q.join()
            return True
        with self._q.all_tasks_done:
            drained = self._q.all_tasks_done.wait_for(
                lambda: self._q.unfinished_tasks == 0, timeout
            )
            pending = self._q.unfinished_tasks
        if not drained:
            print(f"[WARN] training-pair write queue NOT drained within "
                  f"{timeout}s at shutdown — {pending} write(s) still "
                  f"pending; the last edit(s) may not have reached disk.")
        return bool(drained)


# Release-hardening 1b#1 (2026-09-07): how long shutdown() waits for the
# background training-pair write queue before giving up and closing anyway.
_SHUTDOWN_WRITE_FLUSH_TIMEOUT_S = 10.0

# Brick D (2026-09-07): how long shutdown() waits for the patch-load thread.
# It holds a single TIFF read (10 ms on real lab patches), so this is a
# generous bound, not an expected wait.
_PATCH_LOAD_JOIN_TIMEOUT_MS = 5000

# Release-hardening 1b#2 (2026-09-07): the worker reports a user cancel
# through the SAME `error` signal a real crash uses.  This sentinel (and
# the abort flag) let `_on_training_error` keep the log line while
# suppressing the "Training failed" modal — a cancel is not a failure.
_TRAINING_CANCELLED_MSG = "Training cancelled by user."


class _AbortFlag:
    """Thread-safe abort flag checked by the training loop."""
    def __init__(self):
        self._aborted = False

    def set(self):
        self._aborted = True

    def is_set(self):
        return self._aborted


# TrainingAbortedError moved to packages/epicycle_training/exceptions.py in
# Bundle 2 (2026-05-27) — base class changed from RuntimeError to
# BaseException so the abort propagates through third-party except-Exception
# blocks (notably Python's logging chain and Cellpose's training loop).
# See exceptions.py module docstring (cancel via BaseException bypasses `except Exception`).
from packages.epicycle_training.exceptions import TrainingAbortedError
# Release-hardening #3 (2026-09-07): shared atomic-write / quarantine
# helpers (extracted from JSONStateStore.save_state).
from packages.epicycle_io.state_store import (
    atomic_write_json as _atomic_write_json,
    quarantine_corrupt_file as _quarantine_corrupt_file,
)


class _TrainingWorker(QObject):
    """Runs training in a background thread so the UI stays responsive."""
    progress = Signal(int, str)      # (percent, message)
    finished = Signal(dict)          # result dict
    error = Signal(str)              # error message
    # Bundle 2 audit fix #4 (2026-05-27): cross-thread warning signal.
    # Used by archive-fallback path in training_backend_adapter when
    # prior weights get discarded due to Windows file-lock etc.  Routed
    # to _append_ui_log via Qt's queued connection so the user sees
    # warnings from the worker thread without thread-unsafe UI calls.
    log_warn = Signal(str)           # warning message

    def __init__(self, train_fn, **kwargs):
        super().__init__()
        self._train_fn = train_fn
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._train_fn(
                progress_cb=self.progress.emit,
                log_warn_cb=self.log_warn.emit,
                **self._kwargs,
            )
            self.finished.emit(result)
        except TrainingAbortedError:
            self.error.emit(_TRAINING_CANCELLED_MSG)
        except Exception as exc:
            # F7.3: print full traceback to console — debugging Cellpose
            # failures from str(exc) alone is hard.
            import traceback as _tb
            print(f"[TRAIN-SAFETY] _TrainingWorker.run caught {type(exc).__name__}: {exc}")
            _tb.print_exc()
            self.error.emit(str(exc))

class _Step2PatchLoadWorker(QObject):
    """Reads one Step-2 patch TIFF off the Qt main thread (Brick D / #6).

    Named for Step 2 deliberately: Step 3 has its own, differently-shaped
    ``_PatchLoadWorker`` (`fusion_reconstruction/controller.py`) that loads a
    reconstruction + labels.  They are unrelated; do not merge them.

    Deliberately dumb: it runs ONE pure function (the ImageManager read) and
    hands the result back.  Every layer / widget mutation stays on the main
    thread in the controller's finished slot.
    """
    finished = Signal(object)     # payload dict
    error = Signal(str)

    def __init__(self, read_fn, file_path):
        super().__init__()
        self._read_fn = read_fn
        self._file_path = file_path

    def run(self):
        try:
            self.finished.emit(self._read_fn(self._file_path))
        except Exception as exc:
            import traceback as _tb
            print(f"[STEP2-LOAD] _Step2PatchLoadWorker caught "
                  f"{type(exc).__name__}: {exc}")
            _tb.print_exc()
            self.error.emit(f"{type(exc).__name__}: {exc}")


from .image_manager import ImageManager
from .training_manager import TrainingManager
from .training_backend_adapter import (
    build_epicycle_inference_backend,
    build_epicycle_training_backend,
)
from .segmentation_manager import SegmentationManager
from .interactive_segmentation_manager import InteractiveSegmentationManager
from napari_epicycle.annotator import launch_hitl_2d_annotator
from napari_epicycle.domain import InferenceArtifact, ModelArtifact
from napari_epicycle.io_registry import CurationSessionService, HITLPersistenceService


#: Keys that were WORKSPACE-level in step2_state.json before 2026-09-07 and
#: are now PER-RUN (see `_migrate_step2_layout.WORKSPACE_LEVEL_KEYS_V2`).
#: Reads fall back to the top-level value automatically (the v3 unwrap merges
#: top level first, per-run wins).  The SAVE path carries the old top-level
#: values forward so runs that have not been re-saved keep reading them.
_LEGACY_WORKSPACE_FALLBACK_KEYS = (
    "selected_view",
    "view_target",
    "axial_plane",
    "curation_model_source_mode",
    "imported_curation_model_path",
    "curation_pretrained_model_type",
)


class TrainingAnnotationController:
    def __init__(self, widget, viewer):
        self.ui = widget
        self.viewer = viewer

        # Managers
        self.image_manager = ImageManager()
        self.segmentation_manager = SegmentationManager()
        self.interactive_manager = InteractiveSegmentationManager()
        self.training_manager = TrainingManager(
            backend_trainer=build_epicycle_training_backend(),
            backend_inferencer=build_epicycle_inference_backend(),
        )
        self.persistence = HITLPersistenceService()
        self.curation_sessions = CurationSessionService(self.persistence.artifact_registry)

        # State tracking information — per-view round counters.
        # Lateral and axial train independently with separate round numbers.
        self.current_round_by_view = {"lateral": 0, "axial": 0}
        # Backward-compat alias (used by tests and legacy code paths).
        self.current_round = 0
        self.current_config = {}
        self.training_data_by_round = {}  # {round_idx: [slice_ids]}
        self.losses_by_round = {}         # {round_idx: [loss_values]}
        self.working_dir = None
        self.loaded_stack_path = None
        self.selected_view = "XY"  # default
        self.view_target = "lateral"
        self.axial_plane = "XZ"
        self.model_checkpoint_paths = {}  # {round_idx: path_to_model}
        self.round_backend_info = {}  # {round_idx: {"training":{}, "inference":{}, "failed_views":[]}}
        self.sample_records = {}  # {sample_id: metadata}
        self.active_curation_checkpoints = {"lateral": "", "axial": ""}
        self.training_mode = "cumulative"  # "cumulative" or "from_scratch"
        # Release-hardening 2026-09-07 (bucket-switch item #5): training
        # mode is per BUCKET in reality -- lateral and axial train
        # independently.  `training_mode` above stays the ACTIVE bucket's
        # value (legacy alias, still persisted for compatibility).
        self.training_mode_by_bucket = {"lateral": "cumulative",
                                        "axial": "cumulative"}
        self.curation_model_source_mode = "auto"
        self.imported_curation_model_path = ""
        self.curation_pretrained_model_type = "cyto3"

        # state guards
        self.in_refine_mode = False
        self.active_refine_slice = None
        self.loaded_patch_name = None
        self.loaded_patch_axis_view = "XY"
        self._llm_removal_cb_installed: bool = False

        # Initiate connextions
        self._connect_signals()
        self._set_axial_combo_enabled()
        self._ensure_segmentation_config_defaults()
        self._refresh_inference_channel_options()
        self._sync_inference_controls_from_config()
        self._refresh_model_source_controls()
        self._apply_model_source_selection(emit_log=False)
        self._refresh_round_status()
        self._refresh_sample_queue()

        # Graceful shutdown on viewer close.
        self._connect_shutdown_hooks()

    # method to connect ui elements with user inputs
    def _connect_signals(self):
        def _connect(attr_name, signal_name, callback):
            obj = getattr(self.ui, attr_name, None)
            if obj is None:
                return
            signal = getattr(obj, signal_name, None)
            if signal is None:
                return
            signal.connect(callback)

        # 2026-05-27 channel-pair sync.  Training section (chan1_combo /
        # chan2_combo) and Inference section (inference_channel1_combo /
        # inference_channel2_combo) both feed `cfg["channel1"]` and
        # `cfg["channel2"]`.  Without sync, changing one widget leaves
        # the other stale, and the save-time widget snapshot picks
        # whichever the save path happens to read last (inference wins),
        # silently discarding the user's edit on the other widget.
        # Wire bidirectional sync so the user's intent persists
        # regardless of which combo they use.
        self._channel_sync_in_progress = False
        _connect("chan1_combo", "currentTextChanged",
                 lambda t: self._sync_channel_combos("chan1_combo", t))
        _connect("chan2_combo", "currentTextChanged",
                 lambda t: self._sync_channel_combos("chan2_combo", t))
        _connect("inference_channel1_combo", "currentTextChanged",
                 lambda t: self._sync_channel_combos("inference_channel1_combo", t))
        _connect("inference_channel2_combo", "currentTextChanged",
                 lambda t: self._sync_channel_combos("inference_channel2_combo", t))
        # 2026-05-27 same dual-widget sync for normalize / rescale —
        # training has norm_cb / rescale_cb, inference has
        # inference_normalize_cb / inference_rescale_cb.  Both feed
        # cfg["normalize"] / cfg["rescale"].  Without sync, the user's
        # toggle in one section is silently overwritten by the other
        # section's default at save time.
        _connect("norm_cb", "toggled",
                 lambda c: self._sync_bool_checkboxes("norm_cb", c))
        _connect("rescale_cb", "toggled",
                 lambda c: self._sync_bool_checkboxes("rescale_cb", c))
        _connect("inference_normalize_cb", "toggled",
                 lambda c: self._sync_bool_checkboxes("inference_normalize_cb", c))
        _connect("inference_rescale_cb", "toggled",
                 lambda c: self._sync_bool_checkboxes("inference_rescale_cb", c))

        # 2026-05-27 init-transparency.  When user picks
        # a new Base Model or types a Checkpoint, auto-flip Mode to
        # 'From Scratch' so the picks actually apply (vs being silently
        # overridden by the priority chain).  Refresh the hint label
        # on every relevant change so user sees what'll happen on Train.
        self._init_transparency_loading = False
        _connect("model_selector", "currentTextChanged",
                 lambda t: self._on_init_widget_changed("model_selector", t))
        _connect("starting_checkpoint_path", "textChanged",
                 lambda t: self._on_init_widget_changed("starting_checkpoint_path", t))
        _connect("training_mode_combo", "currentTextChanged",
                 lambda _t: self._refresh_init_hint())

        # Multi-run UI (Phase 8a — Curation header dropdown + New Run button)
        _connect("run_picker_combo", "activated", self._on_run_picker_activated)
        _connect("new_run_button", "clicked", self._on_new_run_button_clicked)
        # Phase 8d — Import Pairs button
        _connect("import_pairs_button", "clicked", self._on_import_pairs_clicked)
        # Phase 8c — Manage Runs button
        _connect("manage_runs_button", "clicked", self._on_manage_runs_clicked)
        # Phase 8f — Import External Pairs button
        _connect("import_external_button", "clicked",
                 self._on_import_external_clicked)

        # Set WorkDir & Load Patch Data
        _connect("set_workdir_browse_button", "clicked", self._browse_workdir)
        _connect("train_patch_browse_button", "clicked", self._browse_training_stack)
        # 2026-05-26 audit-J MEDIUM fix: Refresh button wiring.
        _connect("train_patch_refresh_button", "clicked",
                 self._on_refresh_patch_list_clicked)
        _connect("train_patch_list", "itemClicked", self._on_patch_list_clicked)
        # Step 2 look (brick L): Up / Down in the patch list
        # load it after a 250 ms key burst through the SAME slot; the list
        # emits ``loadRequested`` itself, so there is NO currentItemChanged
        # connection (a mouse press would load twice).  A fake list without
        # the signal skips this (``_connect`` guards).
        _connect("train_patch_list", "loadRequested",
                 self._on_patch_list_clicked)

        # Training Configuration
        _connect("save_and_set_config_button", "clicked", self._save_and_set_training_config)
        _connect("load_config_button", "clicked", self._load_training_config)
        # 2026-05-27 training history dialog button.
        _connect("training_history_button", "clicked", self._open_training_history_dialog)

        # Segmentation
        _connect("launch_multiview_button", "clicked", self._launch_multiview_annotator)
        _connect("create_empty_mask_button", "clicked", self._create_empty_mask_seed)
        _connect("segment_current_slice_button", "clicked", self._run_segmentation)
        _connect("view_results_button", "clicked", self._view_selected_results)

        # Curation/SAM Curation
        _connect("load_sam_model_button", "clicked", self._load_sam_model)
        _connect("refine_results_button", "clicked", self._refine_segmentation)
        _connect("accept_annotation_button", "clicked", self._accept_annotation)
        _connect("reject_annotation_button", "clicked", self._reject_annotation)
        _connect("finalize_labels_button", "clicked", self._finalize_labels)
        _connect("model_source_combo", "currentTextChanged", self._on_model_source_changed)
        _connect("imported_model_browse_button", "clicked", self._browse_imported_model_checkpoint)
        _connect("apply_model_source_button", "clicked", self._apply_model_source_selection)
        _connect("curation_pretrained_combo", "currentTextChanged", self._on_model_source_changed)
        _connect("inference_preset_combo", "currentTextChanged", self._on_inference_preset_changed)
        _connect("apply_inference_params_button", "clicked", self._apply_inference_params_from_ui)

        # Starting checkpoint
        _connect("starting_checkpoint_browse", "clicked", self._browse_starting_checkpoint)
        _connect("starting_checkpoint_path", "textChanged", self._on_starting_checkpoint_changed)

        # Training and Data Update
        _connect("training_mode_combo", "currentTextChanged", self._on_training_mode_changed)
        _connect("update_and_train_button", "clicked", self._train_model)
        # Step 2 look (brick T, seam (a)): Cancel is its OWN button, connected
        # ONCE; ``cancel_training`` has the idle gate, so a permanent
        # connection is safe.  Train is never renamed or re-wired.
        _connect("train_cancel_btn", "clicked", self.cancel_training)
        # journey #15 (2026-09-07): NOT connected.  The Benchmarking panel is
        # behind ComingSoonOverlay, and `_run_benchmark` compares final
        # TRAINING losses (not held-out predictions, as the panel claims) into
        # a JSON the panel never displays.  Unwired rather than un-veiled.
        # _connect("run_cross_round_benchmark_button", "clicked", self._run_benchmark)

        # Queue/status controls
        _connect("queue_filter_combo", "currentTextChanged", self._refresh_sample_queue)
        _connect("stage_current_view_button", "clicked", self._stage_accepted_in_current_view)
        _connect("unstage_selected_button", "clicked", self._unstage_selected_queue_items)
        _connect("sample_queue_list", "itemSelectionChanged", self._jump_to_selected_queue_item)
        # Slice filter toggle buttons.
        for btn_attr in ("filter_all_btn", "filter_predicted_btn", "filter_curating_btn",
                         "filter_accepted_btn", "filter_staged_btn", "filter_committed_btn"):
            _connect(btn_attr, "toggled", self._on_slice_filter_changed)
        _connect("get_view_lateral", "toggled", self._on_view_changed)
        _connect("get_view_axial", "toggled", self._on_view_changed)
        _connect("axial_plane_combo", "currentTextChanged", self._on_view_changed)
        # Backward-compat fallbacks for legacy widgets.
        _connect("get_view_XY", "toggled", self._on_view_changed)
        _connect("get_view_XZ", "toggled", self._on_view_changed)
        _connect("get_view_YZ", "toggled", self._on_view_changed)

    def _safe_ui_set_text(self, attr_name, value):
        ui = getattr(self, "ui", None)
        if ui is None:
            return
        widget = getattr(ui, attr_name, None)
        if widget is not None and hasattr(widget, "setText"):
            widget.setText(str(value))

    def _append_ui_log(self, text):
        ui = getattr(self, "ui", None)
        if ui is None:
            return
        output = getattr(ui, "training_log_output", None)
        if output is not None and hasattr(output, "append"):
            output.append(text)

    def _ensure_runtime_defaults(self):
        if not hasattr(self, "selected_view"):
            self.selected_view = "XY"
        if not hasattr(self, "view_target"):
            self.view_target = "lateral" if str(self.selected_view).upper() == "XY" else "axial"
        if not hasattr(self, "axial_plane"):
            self.axial_plane = "XZ"
        if not hasattr(self, "active_curation_checkpoints"):
            self.active_curation_checkpoints = {"lateral": "", "axial": ""}
        if not hasattr(self, "curation_model_source_mode"):
            self.curation_model_source_mode = "auto"
        if not hasattr(self, "imported_curation_model_path"):
            self.imported_curation_model_path = ""
        if not hasattr(self, "curation_pretrained_model_type"):
            cfg = getattr(self, "current_config", {}) if isinstance(getattr(self, "current_config", {}), dict) else {}
            self.curation_pretrained_model_type = str(cfg.get("model_type", "cyto3"))
        if not hasattr(self, "sample_records"):
            self.sample_records = {}
        if not hasattr(self, "round_backend_info"):
            self.round_backend_info = {}
        if not hasattr(self, "current_round_by_view") or not isinstance(getattr(self, "current_round_by_view", None), dict):
            self.current_round_by_view = self._migrate_round_to_per_view()
        if not hasattr(self, "training_mode"):
            self.training_mode = "cumulative"
        if not isinstance(getattr(self, "training_mode_by_bucket", None), dict):
            self.training_mode_by_bucket = self._coerce_training_mode_by_bucket(
                None, fallback=getattr(self, "training_mode", "cumulative")
            )

    def _read_view_selection_from_ui(self):
        ui = getattr(self, "ui", None)
        target = str(getattr(self, "view_target", "lateral")).lower()
        plane = str(getattr(self, "selected_view", "XZ")).upper()
        if plane not in {"XZ", "YZ"}:
            plane = "XZ"

        if ui is None:
            return target, ("XY" if target == "lateral" else plane)

        lateral_btn = getattr(ui, "get_view_lateral", None)
        axial_btn = getattr(ui, "get_view_axial", None)

        if lateral_btn is not None and axial_btn is not None:
            if axial_btn.isChecked():
                target = "axial"
                # No explicit axial plane selector in Phase J UX.
                plane = str(getattr(self, "selected_view", "XZ")).upper()
                if plane not in {"XZ", "YZ"}:
                    plane = "XZ"
            else:
                target = "lateral"
                plane = "XY"
            return target, plane

        # Legacy fallback if old radios are still present.
        if getattr(ui, "get_view_XY", None) is not None and ui.get_view_XY.isChecked():
            return "lateral", "XY"
        if getattr(ui, "get_view_XZ", None) is not None and ui.get_view_XZ.isChecked():
            return "axial", "XZ"
        if getattr(ui, "get_view_YZ", None) is not None and ui.get_view_YZ.isChecked():
            return "axial", "YZ"
        return target, ("XY" if target == "lateral" else plane)

    def _format_target_label(self):
        target = str(getattr(self, "view_target", "lateral")).lower()
        if target == "axial":
            return "Axial (XZ/YZ)"
        return "Lateral (XY)"

    def _set_axial_combo_enabled(self):
        # Axial plane combo was removed from Session UX.
        return

    def _refresh_model_source_controls(self):
        ui = getattr(self, "ui", None)
        if ui is None:
            return
        mode = str(getattr(self, "curation_model_source_mode", "auto"))
        imported_enabled = mode in {"auto", "imported"}
        # In auto mode, pretrained combo is only relevant when no round/imported
        # checkpoint exists (i.e. we'd fall back to a base Cellpose model).
        if mode == "auto":
            self._ensure_curation_checkpoint_state()
            has_any_ckpt = any(
                str(v).strip() and not self._is_placeholder_checkpoint(str(v))
                for v in self.active_curation_checkpoints.values()
            ) or bool(str(getattr(self, "imported_curation_model_path", "")).strip())
            pretrained_enabled = not has_any_ckpt
        else:
            pretrained_enabled = mode == "pretrained"
        path_edit = getattr(ui, "imported_model_path", None)
        browse_btn = getattr(ui, "imported_model_browse_button", None)
        pre_combo = getattr(ui, "curation_pretrained_combo", None)
        if path_edit is not None and hasattr(path_edit, "setEnabled"):
            path_edit.setEnabled(imported_enabled)
        if browse_btn is not None and hasattr(browse_btn, "setEnabled"):
            browse_btn.setEnabled(imported_enabled)
        if pre_combo is not None and hasattr(pre_combo, "setEnabled"):
            pre_combo.setEnabled(pretrained_enabled)

    def _model_source_key(self, text_value):
        text = str(text_value or "").strip().lower()
        if text.startswith("auto"):
            return "auto"
        if text.startswith("round"):
            return "round"
        # 2026-09-07 (item #9): the combo now reads "External model file".
        # "imported" is still accepted so a legacy label / older persisted
        # UI text keeps resolving to the same mode.
        if text.startswith("imported") or text.startswith("external"):
            return "imported"
        if text.startswith("cellpose"):
            return "pretrained"
        if text.startswith("none"):
            return "none"
        return "auto"

    def _last_checkpoint_text(self):
        """Status-bar text for "Last checkpoint", scoped to the ACTIVE bucket.

        Release-hardening 2026-09-07 (bucket-switch item #3): the fallback
        used to be ``max(self.model_checkpoint_paths)`` over STRING keys like
        ``{"lateral_0", "lateral_1", "axial_0"}``.  "lateral_1" sorts last, so
        the axial view proudly displayed a LATERAL checkpoint.  Now the
        fallback only ever considers keys carrying the active bucket's
        prefix; with nothing for this bucket the honest answer is "None".
        """
        try:
            paths = getattr(self, "model_checkpoint_paths", None) or {}
            bucket = self._current_round_key()
            last_ckpt = paths.get(f"{bucket}_{self._current_round()}")
            if last_ckpt is None:
                # Highest round we DO have a checkpoint for, in this bucket.
                prefix = f"{bucket}_"
                candidates = []
                for key in paths:
                    if not isinstance(key, str) or not key.startswith(prefix):
                        continue
                    try:
                        candidates.append((int(key[len(prefix):]), key))
                    except (TypeError, ValueError):
                        continue
                if candidates:
                    last_ckpt = paths[max(candidates)[1]]
            if last_ckpt is None and not any(
                isinstance(k, str) and ("lateral_" in k or "axial_" in k)
                for k in paths
            ):
                # Pre-per-bucket workspace: bare int round keys, no bucket to
                # attribute them to.  Keep the legacy display rather than
                # regressing those workspaces to "None".
                last_ckpt = paths.get(self._current_round())
            if isinstance(last_ckpt, dict):
                parts = [f"{k}: {Path(v).name}" for k, v in last_ckpt.items() if v]
                return " | ".join(parts) if parts else "None"
            if last_ckpt:
                return Path(str(last_ckpt)).name
            return "None"
        except Exception:
            return "None"

    def _refresh_curation_model_status(self):
        self._ensure_curation_checkpoint_state()
        base_model = str(getattr(self, "current_config", {}).get("model_type", "")).strip()
        if not base_model:
            base_model = str(getattr(self, "curation_pretrained_model_type", "cyto3")).strip() or "cyto3"

        lateral_ckpt = str(self.active_curation_checkpoints.get("lateral", "")).strip()
        axial_ckpt = str(self.active_curation_checkpoints.get("axial", "")).strip()
        imported_ckpt = str(getattr(self, "imported_curation_model_path", "")).strip()
        mode = str(getattr(self, "curation_model_source_mode", "auto"))

        lateral_display = lateral_ckpt if lateral_ckpt else f"base:{base_model}"
        axial_display = axial_ckpt if axial_ckpt else f"base:{base_model}"

        source_label = {
            "auto": "auto",
            "round": "round model",
            "imported": "external model file",
            "pretrained": f"cellpose pretrained ({base_model})",
            "none": "none (manual/SAM only)",
        }.get(mode, mode)
        if mode == "imported" and imported_ckpt:
            source_label = f"external model file ({imported_ckpt})"

        self._safe_ui_set_text("active_model_source_value", source_label)
        self._safe_ui_set_text("active_lateral_model_value", lateral_display)
        self._safe_ui_set_text("active_axial_model_value", axial_display)
        # the neutral one-line "Active:" sentence
        # (the words live in ui/_step2_text.py, so the reword is not lost).
        from ..ui import _step2_text as _T2

        def _nm(v):
            # Cell_lateral_round_02.model -> "Cell · round_02" (D16 disk
            # numbering); "base:cyto3" -> "cyto3"; anything else -> its stem.
            v = str(v or "")
            if not v:
                return "-"
            if v.startswith("base:"):
                return v[5:]
            import re as _re
            stem = Path(v).stem
            m = _re.match(r"^(.*)_(?:lateral|axial)_round_(\d+)$", stem)
            if m:
                return _T2.CUR_MODEL_ACTIVE_NAME.format(
                    run=m.group(1), tag=f"round_{int(m.group(2)):02d}")
            return stem

        if mode == "imported" and imported_ckpt:
            active_text = _T2.CUR_MODEL_ACTIVE_FILE.format(
                name=Path(imported_ckpt).name)
        elif mode == "none":
            active_text = _T2.CUR_MODEL_ACTIVE_NONE
        else:
            # Name what Predict will actually use per bucket (a pure read of
            # the resolver), so every Use: mode is told truthfully.
            def _used(view):
                try:
                    r = self._resolve_model_source_for_view(view)
                except Exception:
                    return _nm(lateral_display if view == "XY" else axial_display)
                if not r.get("can_segment", False):
                    return "none"
                ck = str(r.get("checkpoint_path", "") or "").strip()
                if ck:
                    return _nm(ck)
                return str(r.get("model_type", "") or "").strip() or "none"

            active_text = _T2.CUR_MODEL_ACTIVE_LINE.format(
                lat=_used("XY"), ax=_used("XZ"))
        self._safe_ui_set_text("active_model_summary_value", active_text)
        _lbl = getattr(getattr(self, "ui", None), "active_model_summary_value", None)
        if _lbl is not None and hasattr(_lbl, "setToolTip"):
            # the raw detail (full paths) goes on hover
            _lbl.setToolTip(
                f"Source={source_label} | lateral={lateral_display} | "
                f"axial={axial_display}")

    def _on_model_source_changed(self, *_args):
        ui = getattr(self, "ui", None)
        if ui is not None:
            combo = getattr(ui, "model_source_combo", None)
            if combo is not None and hasattr(combo, "currentText"):
                self.curation_model_source_mode = self._model_source_key(combo.currentText())
            path_edit = getattr(ui, "imported_model_path", None)
            if path_edit is not None and hasattr(path_edit, "text"):
                self.imported_curation_model_path = str(path_edit.text() or "").strip()
            pre_combo = getattr(ui, "curation_pretrained_combo", None)
            if pre_combo is not None and hasattr(pre_combo, "currentText"):
                self.curation_pretrained_model_type = str(pre_combo.currentText() or "cyto3").strip()
        self._refresh_model_source_controls()
        self._apply_model_source_selection(emit_log=False)
        # Persist the selection immediately — otherwise it only survives exit
        # if a later accept/stage/train/segment happens to save first (the
        # "had to re-import on resume" bug).
        self._schedule_state_save()

    def _browse_imported_model_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Imported Model Checkpoint",
            self.working_dir if self.working_dir else "",
            "All Files (*)",
        )
        if not path:
            return
        ui = getattr(self, "ui", None)
        if ui is not None:
            edit = getattr(ui, "imported_model_path", None)
            if edit is not None and hasattr(edit, "setText"):
                edit.setText(path)
        self.imported_curation_model_path = str(path)
        self._apply_model_source_selection(emit_log=True)
        self._schedule_state_save()

    def _browse_starting_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Starting Checkpoint",
            self.working_dir if self.working_dir else "",
            "All Files (*)",
        )
        if not path:
            return
        ui = getattr(self, "ui", None)
        if ui is not None:
            edit = getattr(ui, "starting_checkpoint_path", None)
            if edit is not None and hasattr(edit, "setText"):
                edit.setText(path)

    def _on_starting_checkpoint_changed(self, text):
        ui = getattr(self, "ui", None)
        if ui is None:
            return
        path = str(text or "").strip()
        label = getattr(ui, "starting_checkpoint_status", None)
        if label is not None:
            if path and Path(path).is_file():
                label.setText(f"Will fine-tune from: {Path(path).name}")
                label.setStyleSheet("color: #4caf50; font-size: 11px;")
            elif path:
                label.setText("File not found — will use Base Model")
                label.setStyleSheet("color: #ff9800; font-size: 11px;")
            else:
                label.setText("No checkpoint — will use Base Model")
                label.setStyleSheet("color: #aaa; font-size: 11px;")
        # brick T: the "File not found" warning is mirrored into the hint
        try:
            self._refresh_init_hint()
        except Exception:
            pass
        # 2026-05-29: persist the Checkpoint path so it survives exit/resume
        # (same un-saved-mutator class as the model-source + channel fixes).
        # Skip during programmatic restore (_init_transparency_loading).
        if getattr(self, "_init_transparency_loading", False):
            return
        if isinstance(getattr(self, "current_config", None), dict):
            self.current_config["starting_checkpoint"] = path
            self._schedule_state_save()

    def _ensure_segmentation_config_defaults(self):
        if not isinstance(getattr(self, "current_config", None), dict):
            self.current_config = {}
        cfg = self.current_config
        cfg.setdefault("model_name", "epicycle_hitl")
        cfg.setdefault("model_type", str(getattr(self, "curation_pretrained_model_type", "cyto3")) or "cyto3")
        cfg.setdefault("channel1", "2: green")   # membrane (matches UI default)
        cfg.setdefault("channel2", "1: red")     # nuclei (matches UI default)
        cfg.setdefault("normalize", True)
        cfg.setdefault("flow_threshold", 0.4)
        cfg.setdefault("cellprob_threshold", 0.0)
        cfg.setdefault("diameter", 0)
        cfg.setdefault("rescale", True)
        cfg.setdefault("invert", False)
        cfg.setdefault("device", "GPU")

    def _adjust_channels_for_input(self, input_stack):
        """Auto-adjust Cellpose channel config based on actual input dimensions.

        If input has only 1 channel, force channels=[0, 0] (grayscale).
        Otherwise use the UI-configured channel settings.
        """
        if input_stack is None:
            return
        n_channels = input_stack.shape[0] if input_stack.ndim >= 3 else 1
        if n_channels == 1:
            self.current_config["channel1"] = "0: gray"
            self.current_config["channel2"] = "0: none"
            print(f"[CHANNEL-DIAG] single-channel input detected, using channels=[0, 0]")

    def _inference_presets(self):
        # Tuned for quick, practical curation defaults.
        return {
            "Balanced": {"flow_threshold": 0.4, "cellprob_threshold": 0.0, "normalize": True, "rescale": True, "diameter": 0},
            "Conservative": {"flow_threshold": 0.7, "cellprob_threshold": 0.2, "normalize": True, "rescale": True, "diameter": 0},
            "Aggressive": {"flow_threshold": 0.2, "cellprob_threshold": -0.2, "normalize": True, "rescale": True, "diameter": 0},
        }

    def _set_combo_text_if_present(self, combo, text):
        if combo is None or not hasattr(combo, "findText") or not hasattr(combo, "setCurrentText"):
            return
        if combo.findText(str(text)) >= 0:
            combo.setCurrentText(str(text))

    # ------------------------------------------------------------------
    # Init transparency (2026-05-27)
    # ------------------------------------------------------------------

    def _init_spec_for_active_bucket(self):
        """Read the active run's ``created_from`` init_spec from its
        run_manifest.json.  Returns ``None`` on any failure (no run,
        no manifest, no created_from field).

        Pure read.  Only used by ``_compute_init_hint`` for the round 0
        case to decide if widgets will be overridden by the run-creation
        init pick.
        """
        try:
            from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
            from packages.epicycle_io.step2_run_manifest import read_run_manifest
            ws = getattr(self, "working_dir", None)
            run_id = getattr(self, "active_run_id", None)
            if not ws or not run_id:
                return None
            rm_path = _step2_paths.run_manifest_path(ws, run_id)
            rm = read_run_manifest(rm_path)
            if not rm or not isinstance(rm, dict):
                return None
            spec = rm.get("created_from")
            if not isinstance(spec, dict):
                return None
            return spec
        except Exception:
            return None

    def _compute_init_hint(self):
        """Return ``(severity, text)`` describing what the next Train
        click will use as starting weights.  Brick T: the text is one of the
        six ``_step2_text.HINT_CASES`` sentences (``_init_hint_parts`` picks
        the case and its placeholders); the severity is unchanged.
        """
        from ..ui import _step2_text as _T2
        severity, case, kw = self._init_hint_parts()
        if case is None:
            return (severity, kw.get("text", ""))
        return (severity, _T2.hint_text(case, **kw))

    def _init_hint_parts(self):
        """``(severity, case, kwargs)`` -- the six branches of the hint.
        ``case`` is ``None`` (kwargs ``{"text": ...}``) when no run is loaded.

        ``severity`` ∈ {"apply", "override", "warn", "neutral"}:
        * "apply" — Base Model + Checkpoint widgets will actually affect
          the next training.  Render in amber.
        * "override" — The widgets are silently ignored; a higher
          priority (init_spec OR cumulative prior) wins.  Render dim.
        * "warn" — Edge case (e.g. cumulative configured but prior
          checkpoint is missing); widgets will apply as fallback.
        * "neutral" — No run loaded / unknown state.  Render dim.

        Pure read.  Does NOT mutate self.current_config, widgets, or any
        other state.  Idempotent.
        """
        ui = getattr(self, "ui", None)
        ws = getattr(self, "working_dir", None)
        run_id = getattr(self, "active_run_id", None)
        if ui is None or not ws or not run_id:
            return ("neutral", None, {"text": "(no run loaded)"})

        # Determine the active bucket's round_idx.  view_target is
        # always "lateral" or "axial" after _ensure_runtime_defaults.
        view_target = str(getattr(self, "view_target", "lateral") or "lateral").lower()
        bucket = "lateral" if view_target == "lateral" else "axial"
        by_view = getattr(self, "current_round_by_view", {}) or {}
        round_idx = int(by_view.get(bucket, 0))
        # Audit MEDIUM #5 fix: defensive fallback.  If bucket lookup
        # returns 0 BUT self.current_round > 0 (legacy or mid-migration
        # state), trust the legacy counter as a lower bound.  Prevents
        # false-routing to the round-0 init_spec branch when we're
        # genuinely past round 0.
        legacy_round = int(getattr(self, "current_round", 0) or 0)
        if round_idx == 0 and legacy_round > 0:
            round_idx = legacy_round

        mode_combo = getattr(ui, "training_mode_combo", None)
        mode_text = mode_combo.currentText() if mode_combo is not None else ""
        is_cumulative = "cumulative" in mode_text.lower()

        # ----- Round 0 — init_spec from run_manifest decides -----
        if round_idx == 0:
            spec = self._init_spec_for_active_bucket()
            spec_mode = (spec or {}).get("mode", "scratch")
            if spec_mode == "scratch":
                # scratch never resolves to a file → widgets apply.
                base_model = ui.model_selector.currentText() if hasattr(ui, "model_selector") else ""
                ckpt = ""
                ckpt_w = getattr(ui, "starting_checkpoint_path", None)
                if ckpt_w is not None and hasattr(ckpt_w, "text"):
                    ckpt = str(ckpt_w.text() or "").strip()
                if ckpt:
                    import os
                    return ("apply", "round0_scratch",
                            {"source": os.path.basename(ckpt), "bucket": bucket})
                return ("apply", "round0_scratch",
                        {"source": base_model, "bucket": bucket})
            # Non-scratch init_spec wins; widgets are ignored.
            spec_path = (spec or {}).get("path", "")
            display = spec_mode if not spec_path else f"{spec_mode}: {spec_path}"
            return ("override", "round0_locked", {"display": display})

        # ----- Round 1+ -----
        if is_cumulative:
            # Cumulative — prior round wins if it resolves.
            try:
                prior = self._resolve_prior_model_checkpoint(bucket=bucket) or {}
            except Exception:
                prior = {}
            # Resolver returns a dict keyed by view_type ("spatial"/"axial").
            has_prior = bool(prior) and any(prior.values())
            if has_prior:
                prev_round = max(0, round_idx - 1)
                return ("override", "cumulative_prior", {
                    "model": f"{run_id}_{bucket}_round_{prev_round:02d}",
                    "n": prev_round, "bucket": bucket})
            # Cumulative but prior is gone — fallback to widgets.
            base_model = ui.model_selector.currentText() if hasattr(ui, "model_selector") else ""
            return ("warn", "cumulative_missing", {"base": base_model})

        # From Scratch — Checkpoint widget wins over Base Model.
        ckpt = ""
        ckpt_w = getattr(ui, "starting_checkpoint_path", None)
        if ckpt_w is not None and hasattr(ckpt_w, "text"):
            ckpt = str(ckpt_w.text() or "").strip()
        if ckpt:
            import os
            return ("apply", "scratch_file", {"file": os.path.basename(ckpt)})
        base_model = ui.model_selector.currentText() if hasattr(ui, "model_selector") else "scratch"
        return ("apply", "scratch_base", {"base": base_model})

    def _set_channel_combo_by_index(self, combo, cfg_value):
        """Set a channel combo to the item whose INTEGER PREFIX matches
        ``cfg_value``'s integer prefix.

        Why this exists separately from ``_set_combo_text_if_present``:
        channel combos have items like ``"1: red"`` (generic, no patch
        loaded) or ``"1: cdh1"`` (viewer-aware, after patch load).
        Saved cfg values use whichever was current at save time.  Text
        match fails when the saved label doesn't match the current item
        list (e.g., reload before patch loads).

        But the INTEGER index is invariant — both "1: red" and "1: cdh1"
        have integer prefix 1 and map to the same Cellpose channel.
        Setting by integer index preserves the user's saved channel
        choice across the label-text change that happens on patch load.
        """
        if combo is None or not hasattr(combo, "count") or not hasattr(combo, "itemText"):
            return
        target_idx = self._channel_index_from_text(cfg_value)
        if target_idx is None:
            return
        for i in range(combo.count()):
            if self._channel_index_from_text(combo.itemText(i)) == target_idx:
                combo.setCurrentIndex(i)
                return

    def _channel_index_from_text(self, text):
        value = str(text or "").split(":", 1)[0].strip()
        try:
            return int(value)
        except Exception:
            return None

    def _refresh_inference_channel_options(self):
        """Re-populate both Inference Parameters AND Advanced (Training)
        Parameters channel combos with names from the currently loaded
        patch — or the default "0: gray, 1: red, 2: green, 3: blue"
        fallback when nothing is loaded.

        2026-05-26 generalized to also cover the training combos
        (chan1_combo / chan2_combo) — pre-fix only the inference
        combos had this dynamic-name sync, leaving Advanced Parameters
        stuck with the hardcoded red/green/blue labels even when the
        viewer showed DAPI/actin/etc.

        Wiring is shared: both pairs read/write the same
        ``config["channel1"]`` / ``config["channel2"]`` keys, and the
        training backend adapter (training_backend_adapter.py:138-139)
        does ``int(text.split(":")[0])`` — same integer-extraction
        pattern as ``_channel_index_from_text``.  So whether the item
        text says "1: red" or "1: DAPI", Cellpose gets the integer 1.
        """
        ui = getattr(self, "ui", None)
        if ui is None:
            return

        # Collect every (ch1_combo, ch2_combo) pair that exists on the
        # UI.  Inference pair is the original; Advanced (training) pair
        # added 2026-05-26.  Both share the same fallback config keys.
        combo_pairs = []
        inf_ch1 = getattr(ui, "inference_channel1_combo", None)
        inf_ch2 = getattr(ui, "inference_channel2_combo", None)
        if inf_ch1 is not None and inf_ch2 is not None:
            combo_pairs.append((inf_ch1, inf_ch2))
        train_ch1 = getattr(ui, "chan1_combo", None)
        train_ch2 = getattr(ui, "chan2_combo", None)
        if train_ch1 is not None and train_ch2 is not None:
            combo_pairs.append((train_ch1, train_ch2))
        if not combo_pairs:
            return

        # Build channel_entries ONCE — same source layers feed all combo
        # pairs.  When no patch is loaded, fall back to the red/green/
        # blue defaults (per user 2026-05-26: "by default when nothing
        # is loaded it should just be the default 0: gray, 1: red etc").
        source_layers = self._get_patch_view_image_layers(view_name=self.selected_view)
        if not source_layers:
            source_layers = self._get_any_patch_view_image_layers()

        if source_layers:
            channel_entries = []
            for i, layer in enumerate(source_layers[:3], start=1):
                layer_name = str(getattr(layer, "name", ""))
                if self.loaded_patch_name and layer_name.startswith(f"{self.loaded_patch_name}_"):
                    layer_name = layer_name[len(self.loaded_patch_name) + 1 :]
                layer_name = layer_name.replace("_view", "")
                # Strip axis tag (_XY, _XZ, _YZ) — channels are axis-independent.
                for axis_tag in ("_XY", "_XZ", "_YZ"):
                    layer_name = layer_name.replace(axis_tag, "")
                channel_entries.append((i, f"{i}: {layer_name}"))
        else:
            channel_entries = [(1, "1: red"), (2, "2: green"), (3, "3: blue")]

        ch1_options = ["0: gray"] + [label for _, label in channel_entries]
        ch2_options = ["0: none"] + [label for _, label in channel_entries]

        cfg = self.current_config if isinstance(getattr(self, "current_config", None), dict) else {}
        fallback_ch1 = str(cfg.get("channel1", "2: green"))
        fallback_ch2 = str(cfg.get("channel2", "1: red"))
        # Heuristic: when defaults are still gray/none and we have explicit channels,
        # default to first two channels for curation inference.
        if (
            source_layers
            and self._channel_index_from_text(fallback_ch1) == 0
            and self._channel_index_from_text(fallback_ch2) == 0
        ):
            if len(channel_entries) >= 2:
                fallback_ch1 = channel_entries[0][1]
                fallback_ch2 = channel_entries[1][1]
            elif len(channel_entries) == 1:
                fallback_ch1 = channel_entries[0][1]
                fallback_ch2 = "0: none"

        def _restore_selection(combo, previous_idx, fallback_text):
            if previous_idx is not None:
                for i in range(combo.count()):
                    item_text = combo.itemText(i)
                    if self._channel_index_from_text(item_text) == previous_idx:
                        combo.setCurrentIndex(i)
                        return
            self._set_combo_text_if_present(combo, fallback_text)
            if combo.currentIndex() < 0 and combo.count() > 0:
                combo.setCurrentIndex(0)

        # Apply the same items + selection logic to every combo pair.
        # Each pair keeps its own previous selection if it can be
        # preserved by integer index (so swapping "1: red" → "1: DAPI"
        # keeps the user's index-1 pick intact).
        for ch1_combo, ch2_combo in combo_pairs:
            ch1_selected = ch1_combo.currentText() if hasattr(ch1_combo, "currentText") else ""
            ch2_selected = ch2_combo.currentText() if hasattr(ch2_combo, "currentText") else ""
            ch1_idx = self._channel_index_from_text(ch1_selected)
            ch2_idx = self._channel_index_from_text(ch2_selected)

            ch1_combo.blockSignals(True)
            ch2_combo.blockSignals(True)
            try:
                ch1_combo.clear()
                ch2_combo.clear()
                ch1_combo.addItems(ch1_options)
                ch2_combo.addItems(ch2_options)
            finally:
                ch1_combo.blockSignals(False)
                ch2_combo.blockSignals(False)

            _restore_selection(ch1_combo, ch1_idx, fallback_ch1)
            _restore_selection(ch2_combo, ch2_idx, fallback_ch2)

        # The combos were refilled with signals blocked: when the restored
        # index equals the one set while blocked no currentTextChanged fires,
        # so the grey inference summary would keep the old channel names.
        # one guarded, presentation-only refresh.
        fn = getattr(getattr(self, "ui", None), "_refresh_inference_summary", None)
        if fn:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                print(f"[STEP2-UI] inference summary refresh failed: {exc}")

    def _sync_inference_controls_from_config(self):
        # 2026-05-29 audit fix: guard the programmatic widget sync with the
        # init-transparency flag, mirroring _sync_training_controls_from_config.
        # Without it, _set_channel_combo_by_index -> setCurrentIndex fires
        # currentTextChanged -> _sync_channel_combos, which (guard False) writes
        # current_config + schedules a SPURIOUS save on every workspace open.
        # Re-entrant prev-value pattern so nested guards are preserved.
        prev_guard = getattr(self, "_init_transparency_loading", False)
        self._init_transparency_loading = True
        try:
            self._sync_inference_controls_from_config_impl()
        finally:
            self._init_transparency_loading = prev_guard

    def _sync_inference_controls_from_config_impl(self):
        self._ensure_segmentation_config_defaults()
        ui = getattr(self, "ui", None)
        if ui is None:
            return
        self._refresh_inference_channel_options()
        cfg = self.current_config
        # 2026-05-27 fix: channel combos use INTEGER-INDEX set instead
        # of text-match.  Saved cfg may carry viewer-aware text like
        # '1: cdh1' from a session that had a patch loaded, but at
        # workspace-reload time before patch loads the combo items are
        # still generic ['0: gray', '1: red', '2: green', '3: blue'] —
        # text match fails, integer index survives.  When a patch later
        # loads and items refresh, the saved integer index resolves to
        # the correct channel.
        if hasattr(ui, "inference_channel1_combo"):
            self._set_channel_combo_by_index(ui.inference_channel1_combo, cfg.get("channel1", "2: green"))
        if hasattr(ui, "inference_channel2_combo"):
            self._set_channel_combo_by_index(ui.inference_channel2_combo, cfg.get("channel2", "1: red"))
        if hasattr(ui, "inference_flow_thresh"):
            ui.inference_flow_thresh.setValue(float(cfg.get("flow_threshold", 0.4)))
        if hasattr(ui, "inference_cellprob_thresh"):
            ui.inference_cellprob_thresh.setValue(float(cfg.get("cellprob_threshold", 0.0)))
        if hasattr(ui, "inference_diameter_sb"):
            ui.inference_diameter_sb.setValue(int(cfg.get("diameter", 0) or 0))
        if hasattr(ui, "inference_niter_dynamics"):
            ui.inference_niter_dynamics.setValue(int(cfg.get("n_iterations", 0) or 0))
        if hasattr(ui, "inference_normalize_cb"):
            ui.inference_normalize_cb.setChecked(bool(cfg.get("normalize", True)))
        if hasattr(ui, "inference_rescale_cb"):
            ui.inference_rescale_cb.setChecked(bool(cfg.get("rescale", True)))
        if hasattr(ui, "inference_invert_cb"):
            ui.inference_invert_cb.setChecked(bool(cfg.get("invert", False)))

        presets = self._inference_presets()
        matched = "Custom"
        for name, values in presets.items():
            if (
                abs(float(cfg.get("flow_threshold", 0.4)) - float(values["flow_threshold"])) < 1e-8
                and abs(float(cfg.get("cellprob_threshold", 0.0)) - float(values["cellprob_threshold"])) < 1e-8
                and bool(cfg.get("normalize", True)) == bool(values["normalize"])
                and bool(cfg.get("rescale", True)) == bool(values["rescale"])
                and int(cfg.get("diameter", 0) or 0) == int(values["diameter"])
            ):
                matched = name
                break
        self._set_combo_text_if_present(getattr(ui, "inference_preset_combo", None), matched)

    def _on_inference_preset_changed(self, *_args):
        ui = getattr(self, "ui", None)
        if ui is None:
            return
        combo = getattr(ui, "inference_preset_combo", None)
        if combo is None or not hasattr(combo, "currentText"):
            return
        preset_name = str(combo.currentText() or "").strip()
        if preset_name == "Custom":
            return
        values = self._inference_presets().get(preset_name)
        if not values:
            return
        if hasattr(ui, "inference_flow_thresh"):
            ui.inference_flow_thresh.setValue(float(values["flow_threshold"]))
        if hasattr(ui, "inference_cellprob_thresh"):
            ui.inference_cellprob_thresh.setValue(float(values["cellprob_threshold"]))
        if hasattr(ui, "inference_diameter_sb"):
            ui.inference_diameter_sb.setValue(int(values["diameter"]))
        if hasattr(ui, "inference_normalize_cb"):
            ui.inference_normalize_cb.setChecked(bool(values["normalize"]))
        if hasattr(ui, "inference_rescale_cb"):
            ui.inference_rescale_cb.setChecked(bool(values["rescale"]))

    def _apply_inference_params_from_ui(self, *, emit_log: bool = True):
        ui = getattr(self, "ui", None)
        if ui is None:
            return
        self._ensure_segmentation_config_defaults()
        cfg = self.current_config
        if hasattr(ui, "inference_channel1_combo"):
            cfg["channel1"] = str(ui.inference_channel1_combo.currentText() or "2: green")
        if hasattr(ui, "inference_channel2_combo"):
            cfg["channel2"] = str(ui.inference_channel2_combo.currentText() or "1: red")
        if hasattr(ui, "inference_flow_thresh"):
            cfg["flow_threshold"] = float(ui.inference_flow_thresh.value())
        if hasattr(ui, "inference_cellprob_thresh"):
            cfg["cellprob_threshold"] = float(ui.inference_cellprob_thresh.value())
        if hasattr(ui, "inference_diameter_sb"):
            cfg["diameter"] = int(ui.inference_diameter_sb.value())
        if hasattr(ui, "inference_niter_dynamics"):
            cfg["n_iterations"] = int(ui.inference_niter_dynamics.value())
        if hasattr(ui, "inference_normalize_cb"):
            cfg["normalize"] = bool(ui.inference_normalize_cb.isChecked())
        if hasattr(ui, "inference_rescale_cb"):
            cfg["rescale"] = bool(ui.inference_rescale_cb.isChecked())
        if hasattr(ui, "inference_invert_cb"):
            cfg["invert"] = bool(ui.inference_invert_cb.isChecked())
        if emit_log:
            self._append_ui_log(
                "[CURATION] Updated inference params "
                f"(ch1={cfg.get('channel1')}, ch2={cfg.get('channel2')}, flow={cfg.get('flow_threshold')}, "
                f"cellprob={cfg.get('cellprob_threshold')}, diameter={cfg.get('diameter')}, "
                f"normalize={cfg.get('normalize')}, rescale={cfg.get('rescale')}, invert={cfg.get('invert')})."
            )

    def _sync_channel_combos(self, source_attr: str, new_text: str) -> None:
        """Propagate a channel-pair selection from one widget to its peer.

        Training section's ``chan1_combo`` and Inference section's
        ``inference_channel1_combo`` share the same conceptual value
        (which channel feeds Cellpose).  Without this sync, changing
        one leaves the other stale, and the save-time widget snapshot
        picks whichever the save path reads last — silently discarding
        the user's edit on the other widget.

        ``_channel_sync_in_progress`` guard prevents infinite recursion
        when the peer's setCurrentText triggers its own
        currentTextChanged signal.
        """
        if getattr(self, "_channel_sync_in_progress", False):
            return
        peer_attr = {
            "chan1_combo": "inference_channel1_combo",
            "chan2_combo": "inference_channel2_combo",
            "inference_channel1_combo": "chan1_combo",
            "inference_channel2_combo": "chan2_combo",
        }.get(source_attr)
        if peer_attr is None:
            return
        ui = getattr(self, "ui", None)
        if ui is None:
            return
        peer = getattr(ui, peer_attr, None)
        if peer is None or not hasattr(peer, "setCurrentText"):
            return
        self._channel_sync_in_progress = True
        try:
            self._set_combo_text_if_present(peer, new_text)
        finally:
            self._channel_sync_in_progress = False
        # 2026-05-29 audit fix: persist the channel choice immediately so it
        # survives exit/resume.  Previously the combo updated the widget only;
        # current_config got the channel only at Apply/Segment/Train, so
        # "set channels -> exit without segmenting" silently lost them (the
        # channel analog of the model-source save-timing bug).  Skip during
        # restore (_init_transparency_loading) to avoid re-saving loaded state.
        if getattr(self, "_init_transparency_loading", False):
            return
        if source_attr in ("chan1_combo", "inference_channel1_combo"):
            cfg_key = "channel1"
        elif source_attr in ("chan2_combo", "inference_channel2_combo"):
            cfg_key = "channel2"
        else:
            cfg_key = None
        if cfg_key is not None and isinstance(getattr(self, "current_config", None), dict):
            self.current_config[cfg_key] = str(new_text)
            self._schedule_state_save()

    def _sync_bool_checkboxes(self, source_attr: str, new_checked: bool) -> None:
        """Propagate normalize/rescale toggle from one section to the
        other (same pattern as ``_sync_channel_combos``).

        cfg["normalize"] is the canonical state for both Cellpose
        training (train_seg normalize arg) and inference (model.eval
        normalize arg) — the dual widget is purely UI sugar.
        """
        if getattr(self, "_channel_sync_in_progress", False):
            return
        peer_attr = {
            "norm_cb": "inference_normalize_cb",
            "rescale_cb": "inference_rescale_cb",
            "inference_normalize_cb": "norm_cb",
            "inference_rescale_cb": "rescale_cb",
        }.get(source_attr)
        if peer_attr is None:
            return
        ui = getattr(self, "ui", None)
        if ui is None:
            return
        peer = getattr(ui, peer_attr, None)
        if peer is None or not hasattr(peer, "setChecked"):
            return
        self._channel_sync_in_progress = True
        try:
            peer.setChecked(bool(new_checked))
        finally:
            self._channel_sync_in_progress = False

    def _on_init_widget_changed(self, source_attr: str, new_value) -> None:
        """Auto-switch Training Mode to 'From Scratch' when the user
        explicitly picks a new Base Model or types a Checkpoint path.

        Intent: when the user selects an override widget, they clearly
        want that override to take effect.  But in Cumulative mode the
        override is silently ignored by the priority chain (prior round's
        checkpoint wins).  Auto-flipping Mode to 'From Scratch' makes
        their selection actually apply.

        Guard: ``self._init_transparency_loading`` flag is set during
        programmatic state-restore (``_sync_training_controls_from_config``)
        so that path doesn't accidentally trigger the auto-switch.

        Special case: CLEARING the Checkpoint widget does NOT trigger
        a flip — clearing isn't "selecting an override".  Only a
        non-empty new value counts.
        """
        # Guard 1: skip during programmatic state restore.
        if getattr(self, "_init_transparency_loading", False):
            return

        # Guard 2: clearing the checkpoint widget is a no-op for
        # auto-switch (user is removing an override, not picking one).
        if source_attr == "starting_checkpoint_path":
            if not str(new_value or "").strip():
                # Still refresh the hint so its text updates to reflect
                # the cleared state.
                try:
                    self._refresh_init_hint()
                except Exception:
                    pass
                return

        # Flip Mode to From Scratch if not already there.
        ui = getattr(self, "ui", None)
        if ui is None:
            return
        mode_combo = getattr(ui, "training_mode_combo", None)
        if mode_combo is None or not hasattr(mode_combo, "currentText"):
            return
        if "scratch" not in mode_combo.currentText().lower():
            # Find the "From Scratch" item by case-insensitive substring.
            if hasattr(mode_combo, "count") and hasattr(mode_combo, "itemText"):
                for i in range(mode_combo.count()):
                    if "scratch" in mode_combo.itemText(i).lower():
                        mode_combo.setCurrentIndex(i)
                        break
            else:
                # Fallback for combos without count/itemText: try setCurrentText.
                mode_combo.setCurrentText("From Scratch — ignore prior rounds")

        # Refresh the hint label regardless (text may have changed).
        try:
            self._refresh_init_hint()
        except Exception:
            pass

    def _refresh_init_hint(self) -> None:
        """Recompute the init-transparency hint and apply it.  Reads via
        ``_compute_init_hint`` (pure).  Brick T: the ONE hint under the Mode
        combo is the setup column's ``set_hint`` (the amber block only for
        ``warn``; Starting model + file grey iff the severity is
        ``override``, D19); ``starting_checkpoint_status``'s "File not found"
        warning is mirrored into it.  A UI without the column keeps the old
        ``_init_hint_label`` path.
        """
        ui = getattr(self, "ui", None)
        if ui is None:
            return
        try:
            severity, text = self._compute_init_hint()
        except Exception as exc:
            severity, text = ("neutral", f"(hint unavailable: {type(exc).__name__})")
        column = getattr(ui, "training_setup_column", None)
        if column is not None and hasattr(column, "set_hint"):
            try:
                severity, text = self._mirror_checkpoint_warning(severity, text)
                column.set_hint(severity, text)
            except Exception as exc:      # noqa: BLE001 - a sentence must not break a refresh
                print(f"[STEP2-HOOK] set_hint failed: {type(exc).__name__}: {exc}")
            return
        label = getattr(ui, "_init_hint_label", None)
        if label is None or not hasattr(label, "setText"):
            return
        # Severity -> color (legacy single-label path)
        color = {
            "apply":    "#d4a843",   # amber -- widgets will apply
            "warn":     "#d4a843",   # amber -- fallback path
            "override": "#888888",   # dim -- widgets ignored
            "neutral":  "#888888",   # dim -- no info
        }.get(severity, "#888888")
        style = (
            f"color: {color}; font-size: 10px; font-style: italic;"
            f" background: transparent; border: none;"
            f" padding-left: 2px;"
        )
        if hasattr(label, "setStyleSheet"):
            try:
                label.setStyleSheet(style)
            except Exception:
                pass
        label.setText(text)

    def _mirror_checkpoint_warning(self, severity, text):
        """A start-from-a-file path that is not a file (the status label's
        "File not found" case) joins the hint as an amber warning, so it is
        not lost when the status label is hidden."""
        ui = getattr(self, "ui", None)
        edit = getattr(ui, "starting_checkpoint_path", None)
        if edit is None or not hasattr(edit, "text"):
            return severity, text
        path = str(edit.text() or "").strip()
        if not path or severity == "override" or Path(path).is_file():
            return severity, text
        from ..ui import _step2_text as _T2
        return "warn", f"{text} {_T2.STARTING_FILE_NOT_FOUND}"

    def _refresh_current_config_from_widgets(self) -> None:
        """Pull current widget values into ``self.current_config``.

        Without this, training-side widget changes (epochs, LR, channels,
        normalize, etc.) never reach ``self.current_config`` until the
        user clicks Train (which calls ``_collect_training_config``).
        Same for inference widgets (need Apply Inference Params click).

        Called at the top of ``_save_internal_state`` so any persisted
        state JSON reflects the user's CURRENT widget picks, not the
        last-trained / last-applied snapshot.  Quiet — no UI log spam.
        """
        ui = getattr(self, "ui", None)
        if ui is None:
            return
        try:
            # Capture training-side widgets via the existing collector.
            # MERGE into self.current_config (not replace) so non-widget-
            # managed keys like `pretrained_model_path` (set during
            # training-complete promotion) survive the refresh.
            _widget_cfg = self._collect_training_config()
            if not isinstance(getattr(self, "current_config", None), dict):
                self.current_config = {}
            self.current_config.update(_widget_cfg)
            print(f"[STATE-SAVE] training widgets snapshot: "
                  f"epochs={self.current_config.get('epochs')} "
                  f"batch={self.current_config.get('batch_size')} "
                  f"lr={self.current_config.get('learning_rate')} "
                  f"wd={self.current_config.get('weight_decay')} "
                  f"minmask={self.current_config.get('min_masks_per_image')} "
                  f"ch1={self.current_config.get('channel1')!r} "
                  f"ch2={self.current_config.get('channel2')!r} "
                  f"norm={self.current_config.get('normalize')} "
                  f"rescale={self.current_config.get('rescale')} "
                  f"compute_flows={self.current_config.get('compute_flows')} "
                  f"device={self.current_config.get('device')} "
                  f"model_type={self.current_config.get('model_type')!r}")
        except Exception as e:
            print(f"[STATE-SAVE] WARN: training collect failed "
                  f"({type(e).__name__}: {e}); preserving prior current_config")
        try:
            # Capture inference-side widgets — overwrites the keys the
            # training collector also wrote (channels / normalize / rescale)
            # with the user's Inference Parameters picks.  Both sections
            # share these keys; inference values are the inference-time
            # truth.  Silent (emit_log=False) to avoid log spam every save.
            self._apply_inference_params_from_ui(emit_log=False)
            print(f"[STATE-SAVE] inference widgets snapshot: "
                  f"ch1={self.current_config.get('channel1')!r} "
                  f"ch2={self.current_config.get('channel2')!r} "
                  f"flow={self.current_config.get('flow_threshold')} "
                  f"cellprob={self.current_config.get('cellprob_threshold')} "
                  f"diameter={self.current_config.get('diameter')} "
                  f"niter={self.current_config.get('n_iterations')} "
                  f"norm={self.current_config.get('normalize')} "
                  f"rescale={self.current_config.get('rescale')} "
                  f"invert={self.current_config.get('invert')}")
        except Exception as e:
            print(f"[STATE-SAVE] WARN: inference apply failed "
                  f"({type(e).__name__}: {e}); preserving prior current_config")

    def _resolve_model_source_for_view(self, view_name=None):
        self._ensure_curation_checkpoint_state()
        active_view = str(view_name or self.selected_view or "XY").upper()
        bucket = self._bucket_for_view(active_view)
        mode = str(getattr(self, "curation_model_source_mode", "auto"))

        round_ckpt = str(self.active_curation_checkpoints.get(bucket, "")).strip()
        if round_ckpt and self._is_placeholder_checkpoint(round_ckpt):
            round_ckpt = ""
        imported = str(getattr(self, "imported_curation_model_path", "")).strip()
        pretrained = str(getattr(self, "curation_pretrained_model_type", "")).strip() or "cyto3"

        if mode == "round":
            return {"mode": mode, "checkpoint_path": round_ckpt, "model_type": pretrained, "can_segment": bool(round_ckpt)}
        if mode == "imported":
            return {"mode": mode, "checkpoint_path": imported, "model_type": pretrained, "can_segment": bool(imported)}
        if mode == "pretrained":
            return {"mode": mode, "checkpoint_path": "", "model_type": pretrained, "can_segment": True}
        if mode == "none":
            return {"mode": mode, "checkpoint_path": "", "model_type": "", "can_segment": False}

        # auto mode
        if round_ckpt:
            return {"mode": "auto", "checkpoint_path": round_ckpt, "model_type": pretrained, "can_segment": True}
        if imported:
            return {"mode": "auto", "checkpoint_path": imported, "model_type": pretrained, "can_segment": True}
        return {"mode": "auto", "checkpoint_path": "", "model_type": pretrained, "can_segment": True}

    def _apply_model_source_selection(self, emit_log=True):
        self._ensure_runtime_defaults()
        resolved = self._resolve_model_source_for_view()
        checkpoint = str(resolved.get("checkpoint_path", "")).strip()
        model_type = str(resolved.get("model_type", "")).strip()

        if model_type:
            self._ensure_segmentation_config_defaults()
            if str(self.current_config.get("model_type", "")).strip() != model_type:
                self.current_config["model_type"] = model_type
                seg_manager = getattr(self, "segmentation_manager", None)
                if seg_manager is not None and hasattr(seg_manager, "segmenter"):
                    seg_manager.segmenter = None

        self._set_segmentation_pretrained_override(checkpoint)
        self._refresh_curation_model_status()

        if emit_log:
            mode = str(resolved.get("mode", "auto"))
            if resolved.get("can_segment", False):
                if checkpoint:
                    self._append_ui_log(f"[CURATION] Model source '{mode}' active with checkpoint: {checkpoint}")
                else:
                    self._append_ui_log(
                        f"[CURATION] Model source '{mode}' active with Cellpose model: {self.current_config.get('model_type', '-')}"
                    )
            else:
                self._append_ui_log("[CURATION] Manual/SAM-only mode active. Use 'Create Empty Mask (Manual/SAM)'.")
        return resolved

    def _infer_view_from_label(self, slice_label):
        parts = str(slice_label).rsplit("_", 2)
        if len(parts) < 3:
            return self.selected_view
        axis = str(parts[1]).upper()
        return axis if axis in {"XY", "XZ", "YZ"} else self.selected_view

    def _infer_slice_index(self, slice_label):
        parts = str(slice_label).rsplit("_", 1)
        if len(parts) < 2:
            return -1
        # 2026-05-25 fix: handle both legacy plain-int format
        # (`{stack}_{VIEW}_{idx}`) AND the new z-prefixed format
        # (`{stack}_{VIEW}_z{idx:04d}`) used by parse_slice_label /
        # all Phase 8 import + curation writes.  Without this strip,
        # `_ensure_work_labels_layer` restore loop silently skips
        # every z-prefixed sample (returns -1, hits `if s_idx < 0:
        # continue` without incrementing skipped_no_mask), so no
        # [RESTORE-DIAG] line ever prints and the imported masks
        # never stamp into work_labels_3d.  Same root cause hides
        # in [DISCARD-DIAG] s_idx=-1 entries (cleanup skipped on
        # in-memory volume — disk delete still ran, so partial
        # consistency).
        last = parts[-1]
        if last.startswith("z") and len(last) > 1:
            last = last[1:]
        try:
            return int(last)
        except ValueError:
            return -1

    def _bucket_for_view(self, view_name):
        return "lateral" if str(view_name).upper() == "XY" else "axial"

    def _get_auto_update_model_checked(self):
        """Read the auto-update model checkbox state (default True)."""
        ui = getattr(self, "ui", None)
        cb = getattr(ui, "auto_update_model_cb", None) if ui is not None else None
        if cb is not None and hasattr(cb, "isChecked"):
            return cb.isChecked()
        return True

    def _current_round_key(self):
        """Return the current view's bucket key ('lateral' or 'axial')."""
        return self._bucket_for_view(self.selected_view)

    def _current_round(self):
        """Return the round number for the currently active view."""
        by_view = getattr(self, "current_round_by_view", None)
        if isinstance(by_view, dict):
            return by_view.get(self._current_round_key(), 0)
        return getattr(self, "current_round", 0)

    @staticmethod
    def _clean_patch_filename(item):
        """Get clean filename from a patch list item (strips badge display text).

        Prefers UserRole data (set at population time). Falls back to
        stripping known badge patterns from the display text.
        """
        from qtpy.QtCore import Qt
        # Prefer stored clean filename.
        data = None
        if hasattr(item, "data"):
            try:
                data = item.data(Qt.UserRole)
            except Exception:
                pass
        if data:
            return str(data).strip()
        # Fallback: strip everything after first double-space run.
        text = item.text() if hasattr(item, "text") else str(item)
        # The display format is: "patch_stem          ●●●   N stg"
        # Split on 2+ consecutive spaces to get the stem, then re-add extension.
        import re
        parts = re.split(r"\s{2,}", text.strip())
        return parts[0].strip() if parts else text.strip()

    def _migrate_round_to_per_view(self):
        """Convert a legacy scalar current_round to per-view dict.

        Called during state load when old format is detected.
        """
        old_round = getattr(self, "current_round", 0)
        if not isinstance(old_round, int):
            old_round = 0
        return {"lateral": old_round, "axial": old_round}

    def _training_bucket_for_view(self, view_name):
        return "spatial" if str(view_name).upper() == "XY" else "axial"

    def _curation_bucket_for_training_view(self, view_type):
        key = str(view_type).lower()
        if key == "spatial":
            return "lateral"
        if key == "axial":
            return "axial"
        return None

    def _is_placeholder_checkpoint(self, checkpoint_path):
        return str(checkpoint_path or "").strip().lower().endswith(".ckpt.json")

    def _ensure_curation_checkpoint_state(self):
        state = getattr(self, "active_curation_checkpoints", None)
        if not isinstance(state, dict):
            self.active_curation_checkpoints = {"lateral": "", "axial": ""}
            return
        self.active_curation_checkpoints = {
            "lateral": str(state.get("lateral", "")),
            "axial": str(state.get("axial", "")),
        }

    def _set_segmentation_pretrained_override(self, checkpoint_path):
        if not isinstance(getattr(self, "current_config", None), dict):
            self.current_config = {}
        checkpoint_path = str(checkpoint_path or "").strip()
        previous = str(self.current_config.get("pretrained_model_path", "")).strip()
        if checkpoint_path:
            if previous == checkpoint_path:
                return False
            self.current_config["pretrained_model_path"] = checkpoint_path
        else:
            if not previous:
                return False
            self.current_config.pop("pretrained_model_path", None)

        seg_manager = getattr(self, "segmentation_manager", None)
        if seg_manager is not None and hasattr(seg_manager, "segmenter"):
            seg_manager.segmenter = None
        return True

    def _apply_curation_checkpoint_for_view(self, view_name=None, emit_log=False):
        if view_name:
            self.selected_view = str(view_name).upper()
        resolved = self._apply_model_source_selection(emit_log=emit_log)
        return str(resolved.get("checkpoint_path", "")).strip()

    def _promote_round_checkpoints_to_curation(self, round_checkpoints):
        if not isinstance(round_checkpoints, dict):
            return
        self._ensure_curation_checkpoint_state()

        promoted = 0
        skipped_placeholder = 0
        promoted_rows = []
        for view_type, checkpoint_path in round_checkpoints.items():
            bucket = self._curation_bucket_for_training_view(view_type)
            if bucket is None:
                continue
            checkpoint_path = str(checkpoint_path or "").strip()
            if not checkpoint_path:
                continue
            if self._is_placeholder_checkpoint(checkpoint_path):
                skipped_placeholder += 1
                continue
            # Release-hardening #9 (2026-09-07): the lift can fail (Windows
            # file lock) and leave `checkpoint_path` pointing at the pre-lift
            # `models/` path that was then cleaned up.  Pre-fix that dead
            # path was promoted as the active model and the next predict
            # silently used it.  Only a real file may be promoted.
            if not Path(checkpoint_path).is_file():
                self._append_ui_log(
                    f"[CURATION] Skipped promoting {bucket} checkpoint — "
                    f"not a file on disk: {checkpoint_path}"
                )
                print(f"[STEP2-TRAIN] #9 promotion SKIP — not a file: "
                      f"{checkpoint_path}")
                continue
            if self.active_curation_checkpoints.get(bucket, "") != checkpoint_path:
                self.active_curation_checkpoints[bucket] = checkpoint_path
                promoted += 1
                promoted_rows.append((bucket, checkpoint_path))

        if promoted:
            for bucket, checkpoint_path in promoted_rows:
                self._append_ui_log(f"[CURATION] Active {bucket} model set from latest trained checkpoint: {checkpoint_path}")
            # 2026-05-29 UX fix: flip explicitly to "Round model" (not "Auto") so
            # the dropdown VISIBLY shows the just-trained model is now active.
            # Previously flipped to Auto, which resolved-to-round correctly but
            # gave the user no visible confirmation that their fresh training
            # took effect.  Persistence carries the choice through exit/resume.
            ui = getattr(self, "ui", None)
            auto_update_cb = getattr(ui, "auto_update_model_cb", None) if ui is not None else None
            should_auto_update = auto_update_cb is None or auto_update_cb.isChecked()
            prev_mode = str(getattr(self, "curation_model_source_mode", "auto"))
            if should_auto_update and prev_mode != "round":
                self.curation_model_source_mode = "round"
                self._append_ui_log(
                    f"[CURATION] Model source switched from '{prev_mode}' to 'round' "
                    "to use latest trained checkpoint."
                )
                combo = getattr(ui, "model_source_combo", None) if ui is not None else None
                if combo is not None and hasattr(combo, "setCurrentText"):
                    combo.setCurrentText("Round model")
            elif not should_auto_update and prev_mode not in ("round", "auto"):
                self._append_ui_log(
                    f"[CURATION] Model source kept as '{prev_mode}' (auto-update disabled)."
                )
            # Release-hardening 2026-09-07 (bucket-switch item #1): do NOT
            # write the fresh round checkpoint into the "External model file"
            # slot.  That slot is workspace/run-level and is read back by
            # `_on_model_source_changed` into
            # `imported_curation_model_path`, so the AXIAL round checkpoint
            # ended up being the "external" model for EVERY view and every
            # run -- and lateral prediction silently ran the axial model.
            # The just-trained checkpoint is already shown by the per-bucket
            # active-model labels; the External slot must only ever hold a
            # path the USER browsed to.
            self._apply_curation_checkpoint_for_view(emit_log=True)
        elif skipped_placeholder:
            self._append_ui_log(
                "[CURATION] Placeholder checkpoints detected; keeping current curation model selection."
            )

    def _touch_sample(self, slice_label, *, status=None, view=None, round_tag=None, origin=None):
        """Touch / create a sample_record.

        Parameters
        ----------
        slice_label
            Canonical slice identifier, e.g. "patch_XY_42".
        status
            Workflow state — predicted / curating / accepted / staged / committed.
        view
            "XY" / "XZ" / "YZ".  Inferred from slice_label if not provided.
        round_tag
            Round identifier this record was last touched in.
        origin
            Annotation origin / provenance.  Standard values:
              "cellpose"  — Cellpose-trained model prediction (set by the
                            segment path).
              "manual"    — user drew without any segmentation seed.
              "sam" / "stardist" / "omnipose" / "watershed" / "scribble" —
                            reserved for future annotation backends.
              None        — provenance not specified (legacy / unknown).
            Preserves the existing record's value when omitted — important
            for downstream provenance to survive status updates (e.g. an
            Accept call doesn't lose "manual" / "cellpose" provenance).

            origin reflects the SEED of the annotation, not its final-state
            authorship.  A predicted mask that has been 100% repainted still
            carries origin="cellpose" because that was the starting point.
            This is the right semantic for the classifier-training-data use
            case (training data for a future classifier).

            origin is ORTHOGONAL to ``inherited_from_pair`` / ``inherited_-
            from_external`` (lineage edges set by the cherry-pick + external
            import flows).  A cherry-picked pair still carries an ``origin``
            describing how the ORIGINAL annotation was created.
        """
        now = datetime.now(timezone.utc).isoformat()
        view_name = (view or self._infer_view_from_label(slice_label) or self.selected_view).upper()
        record = self.sample_records.get(slice_label, {})
        # Derive patch_id from slice_label (e.g., "patch_name_XY_42" → "patch_name").
        # Only fall back to self.loaded_patch_name for brand-new records.
        existing_patch = record.get("patch_id", "")
        if not existing_patch or existing_patch == "-":
            derived_patch = slice_label.rsplit("_", 2)[0] if slice_label.count("_") >= 2 else ""
            patch_id = derived_patch or str(self.loaded_patch_name or "-")
        else:
            patch_id = existing_patch
        record.update(
            {
                "sample_id": str(slice_label),
                "roi_id": patch_id,
                "patch_id": patch_id,
                "view": view_name,
                "slice_index": self._infer_slice_index(slice_label),
                "bucket": self._bucket_for_view(view_name),
                "status": status or record.get("status", "predicted"),
                "mask_ref": str(slice_label),
                "updated_at": now,
                "round_tag": round_tag if round_tag is not None else record.get("round_tag", "-"),
                "origin": origin if origin is not None else record.get("origin"),
            }
        )
        self.sample_records[slice_label] = record
        return record

    def _iter_list_widget_items(self, list_widget):
        if list_widget is None or not hasattr(list_widget, "count") or not hasattr(list_widget, "item"):
            return []
        items = []
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            if item is None:
                continue
            if hasattr(item, "text"):
                items.append(item.text())
            else:
                items.append(str(item))
        return items

    def _sync_data_to_add_list(self, force=False):
        """Populate the training data list with staged slices for the ACTIVE view only.

        Only includes slices matching the current bucket (lateral or axial).
        This ensures training uses only the correct view's data.
        """
        ui = getattr(self, "ui", None)
        if ui is None:
            return
        if not hasattr(self, "sample_records"):
            self.sample_records = {}
        # No "empty records -> keep the list" early return: a workspace / run
        # switch or New Run reloads with EMPTY records and the list must
        # follow (it used to keep the previous run's rows, and Train counted
        # them).  ``force`` is kept for callers; the sync is always honest now.

        data_list = getattr(ui, "data_to_add_list", None)
        if data_list is None or not hasattr(data_list, "clear") or not hasattr(data_list, "addItem"):
            return

        active_bucket = self._bucket_for_view(self.selected_view)
        data_list.clear()
        staged_ids = sorted(
            [sid for sid, rec in self.sample_records.items()
             if rec.get("status") == "staged" and rec.get("bucket") == active_bucket],
            key=lambda sid: self.sample_records[sid].get("updated_at", ""),
        )
        for sample_id in staged_ids:
            data_list.addItem(sample_id)
        # Step 2 look (brick A): paint the staged rows (``item.text()`` stays
        # the sample id -- ``_train_model``'s "No staged slices" gate reads it)
        self._notify_slice_rows(staged=True, ids=list(staged_ids),
                                bucket=active_bucket)

        # Update the label to show which view is being trained.
        self._staged_new_count = len(staged_ids)
        self._write_staged_count_text(active_bucket)
        self._refresh_training_stage_line()

    def _staged_count_for(self, bucket) -> int:
        """Finalized records of ``bucket`` in THIS run's records (audit M3):
        read from ``sample_records`` itself, not from a cache only a
        non-empty list refill writes -- a workspace or run with no records
        must not show the previous one's count."""
        return sum(
            1 for rec in (getattr(self, "sample_records", {}) or {}).values()
            if rec.get("status") == "staged" and rec.get("bucket") == bucket)

    def _write_staged_count_text(self, active_bucket=None) -> None:
        """The honest count under the staged list + the line about the bucket
        this Train does not use (the words live in ``_step2_text``)."""
        ui = getattr(self, "ui", None)
        staging_label = getattr(ui, "dataset_staging", None)
        if staging_label is None or not hasattr(staging_label, "setText"):
            return
        if active_bucket is None:
            active_bucket = self._bucket_for_view(self.selected_view)
        from ..ui import _step2_text as _T2
        n_new = self._staged_count_for(active_bucket)
        prior = self._staged_prior_for(active_bucket)
        by_view = getattr(self, "current_round_by_view", {}) or {}
        round_n = int(by_view.get(active_bucket, 0) or 0)
        other = "axial" if active_bucket == "lateral" else "lateral"
        n_other = sum(
            1 for rec in (getattr(self, "sample_records", {}) or {}).values()
            if rec.get("status") == "staged" and rec.get("bucket") == other)
        staging_label.setText(
            _T2.staged_count_text(n_new, prior + n_new, round_n, active_bucket)
            + "\n" + _T2.staged_other_text(other, n_other))
        desc = getattr(getattr(ui, "training_staged_column", None), "set_desc", None)
        if callable(desc):
            desc(active_bucket)

    def _refresh_round_status(self):
        self._ensure_runtime_defaults()
        ui = getattr(self, "ui", None)
        if ui is None:
            return
        segmented_count = len(self._iter_list_widget_items(getattr(ui, "segment_slice_data_list", None)))
        accepted_count = sum(1 for rec in self.sample_records.values() if rec.get("status") == "accepted")
        staged_lateral = sum(
            1
            for rec in self.sample_records.values()
            if rec.get("status") == "staged" and rec.get("bucket") == "lateral"
        )
        staged_axial = sum(
            1
            for rec in self.sample_records.values()
            if rec.get("status") == "staged" and rec.get("bucket") == "axial"
        )

        self._safe_ui_set_text("status_workspace_value", self.working_dir or "-")
        # Run indicator (Phase 6 sub-D).  Reads self.active_run_id which is
        # set from the v3 step2_state.json peek in _load_internal_state.
        # Pre-Phase-8 always "default"; future multi-run UI updates this.
        try:
            run_id = getattr(self, "active_run_id", None) or "default"
            # Indicate if workspace is on legacy layout (no multi-run yet).
            from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
            if self.working_dir and not _step2_paths.is_new_layout_state(self.working_dir):
                run_id_label = f"{run_id} (legacy layout)"
            else:
                run_id_label = run_id
        except Exception:
            run_id_label = "default"
        self._safe_ui_set_text("status_run_value", run_id_label)
        by_view = getattr(self, "current_round_by_view", {})
        lat_r = by_view.get("lateral", 0)
        axi_r = by_view.get("axial", 0)
        self._safe_ui_set_text("status_round_value", f"Lat {lat_r} | Axi {axi_r}")
        self._safe_ui_set_text("status_view_value", self._format_target_label())
        self._safe_ui_set_text("status_segmented_value", segmented_count)
        self._safe_ui_set_text("status_accepted_value", accepted_count)
        self._safe_ui_set_text("status_staged_lateral_value", staged_lateral)
        self._safe_ui_set_text("status_staged_axial_value", staged_axial)
        self._safe_ui_set_text("status_staged_value", staged_lateral + staged_axial)

        # --- Total accumulated training pairs & last checkpoint ---
        try:
            prior_pairs = self._collect_prior_round_pairs()
            total = len(prior_pairs)
            lateral_count = sum(1 for p in prior_pairs if p.get("view_group") == "spatial")
            axial_count = sum(1 for p in prior_pairs if p.get("view_group") == "axial")
            _bk = self._current_round_key()
            self._staged_prior_pairs = (
                _bk, self._prior_pair_total_excluding_staged(_bk, prior_pairs))
            self._safe_ui_set_text("status_total_pairs_value", total)
            self._safe_ui_set_text("status_pairs_lateral_value", lateral_count)
            self._safe_ui_set_text("status_pairs_axial_value", axial_count)
        except Exception:
            self._safe_ui_set_text("status_total_pairs_value", "?")

        # Last model checkpoint (for the ACTIVE bucket only -- see
        # _last_checkpoint_text).
        self._safe_ui_set_text(
            "status_last_checkpoint_value", self._last_checkpoint_text()
        )

        self._refresh_curation_model_status()
        self._refresh_model_name_display(
            staged_lateral=staged_lateral,
            staged_axial=staged_axial,
        )
        # brick T: the prior-pair count just changed -- the count line and the
        # idle stage line follow (a pure text write; the scan above is shared)
        try:
            self._write_staged_count_text()
            self._refresh_training_stage_line()
        except Exception:
            pass
        # 2026-05-27 init-transparency: round counter changes affect the
        # hint (round 0 vs 1+ logic differs).  Refresh here.
        try:
            self._refresh_init_hint()
        except Exception:
            pass

    def _refresh_model_name_display(self, *, staged_lateral=0, staged_axial=0):
        """Update the read-only model-name labels in Training Setup.

        Shows the on-disk filename Cellpose will write per bucket
        (formula matches controller.py:5618).  Highlight rule (simple):
        the bucket matching the current view is amber; the other is dim.
        XY tab → lateral lit, XZ/YZ tab → axial lit.  Exactly one is lit
        at any time (the active view).

        ``staged_lateral`` / ``staged_axial`` kwargs are accepted for
        back-compat with the caller signature but unused — the simpler
        view-driven rule is the only signal.

        Replaces the vestigial free-text model_name_line QLineEdit which
        had been silently ignored since Phase B.2 (2026-05-25) locked
        the on-disk filename pattern.
        """
        ui = getattr(self, "ui", None)
        if ui is None:
            return
        lat_label = getattr(ui, "model_name_lateral_label", None)
        ax_label = getattr(ui, "model_name_axial_label", None)
        if lat_label is None or ax_label is None:
            return

        run_id = str(getattr(self, "active_run_id", "") or "")
        by_view = getattr(self, "current_round_by_view", {}) or {}
        lat_round = int(by_view.get("lateral", 0))
        ax_round = int(by_view.get("axial", 0))
        from ..ui import _step2_text as _T2

        if not run_id:
            lat_text = "lateral:  (no run loaded)"
            ax_text = "axial:    (no run loaded)"
            lat_active = False
            ax_active = False
        else:
            lat_name = f"{run_id}_lateral_round_{lat_round:02d}"
            ax_name = f"{run_id}_axial_round_{ax_round:02d}"
            view_target = str(getattr(self, "view_target", "lateral") or "lateral").lower()
            lat_active = (view_target == "lateral")
            ax_active = (view_target == "axial")
            # brick T: the ACTIVE bucket says the real file name and
            # its round; the other says it is not in this round
            lat_text = (_T2.writes_text(lat_name, lat_round) if lat_active
                        else _T2.WRITES_OTHER_IDLE.format(other="lateral"))
            ax_text = (_T2.writes_text(ax_name, ax_round) if ax_active
                       else _T2.WRITES_OTHER_IDLE.format(other="axial"))

        def _apply_color(label, active):
            # look rule 1: the accent is never the colour of a sentence --
            # active = the neutral row colour, bold; inactive grey
            color = "#bbbbbb" if active else "#666666"
            weight = "bold" if active else "normal"
            label.setStyleSheet(
                f"color: {color}; font-size: 11px; font-weight: {weight};"
                " font-family: 'Consolas', 'Cascadia Mono', 'Courier New', monospace;"
                " background: transparent; border: none;"
                " padding-left: 2px;"
            )

        _apply_color(lat_label, lat_active)
        _apply_color(ax_label, ax_active)
        lat_label.setText(lat_text)
        ax_label.setText(ax_text)
        # the idle bold loss line follows the bucket the next Train writes
        if run_id:
            fn = getattr(getattr(ui, "loss_plot", None),
                         "set_idle_family", None)
            if fn is not None:
                try:
                    fn("lateral" if lat_active else "axial")
                except Exception:      # noqa: BLE001 -- cosmetic
                    pass

    def _queue_filter_passes(self, record, filter_name):
        filt = str(filter_name or "All")
        status = record.get("status", "")
        bucket = record.get("bucket", "")
        if filt == "All":
            return True
        if filt == "Lateral":
            return bucket == "lateral"
        if filt == "Axial":
            return bucket == "axial"
        if filt == "Accepted":
            return status == "accepted"
        if filt == "Staged":
            return status == "staged"
        if filt == "Rejected":
            return status == "rejected"
        return True

    # Status color palette — simplified workflow: Predicted → Curating → Accepted → Finalized.
    _STATUS_COLORS = {
        "predicted": QColor("#d4a843"),   # amber — raw prediction (BUILD)
        "curating": QColor("#a78bca"),    # purple — user editing
        "accepted": QColor("#6abf69"),    # green — user approved
        "staged": QColor("#80deea"),      # ice blue — finalized (ready for training)
        "committed": QColor("#80deea"),   # ice blue — finalized (already trained)
        "rejected": QColor("#ef5350"),    # red — rejected
    }

    # Status icons (qtawesome).
    _STATUS_ICONS = {
        "predicted": "mdi6.layers-outline",
        "curating": "mdi6.pencil-outline",
        "accepted": "mdi6.check-circle-outline",
        "staged": "mdi6.diamond-stone",
        "committed": "mdi6.diamond-stone",
        "rejected": "mdi6.close-circle-outline",
    }
    # Unicode symbols matching the qtawesome icon concepts.
    # Used in Curation Progress HTML and slice list fallback text.
    _STATUS_DOTS = {
        "predicted": "\u25a7",    # ▧ (layers/stacked)
        "curating": "\u270f",     # ✏ (pencil)
        "accepted": "\u2714",     # ✔ (check)
        "staged": "\u2b25",       # ⬥ (diamond)
        "committed": "\u2b25",    # ⬥ (diamond)
        "rejected": "\u2715",     # ✕ (cross)
    }

    # Display names — maps internal status strings to user-facing labels.
    _STATUS_DISPLAY = {
        "predicted": "Predicted",
        "curating": "Curating",
        "accepted": "Accepted",
        "staged": "Finalized",
        "committed": "Finalized",
        "rejected": "Rejected",
    }

    def _format_queue_row(self, record):
        """Format a review queue item — clean monospace layout.

        Phase 8e (2026-05-24): prepends ``↳`` glyph for pairs imported
        from another run (Phase 8d cherry-pick) or external source
        (future Phase 8f).  Visually signals "this slice's mask wasn't
        curated here — it came in via import".
        """
        view = str(record.get("view", "-")).upper()
        idx = record.get("slice_index", "-")
        patch = str(record.get("patch_id") or record.get("roi_id") or "").strip()
        if not patch or patch == "-":
            sid = str(record.get("sample_id", ""))
            parts = sid.rsplit("_", 2)
            patch = parts[0] if len(parts) >= 3 else sid
        status = record.get("status", "predicted")
        dots = self._STATUS_DOTS.get(status, "")
        round_tag = str(record.get("round_tag", "")).strip()
        if round_tag and round_tag != "-":
            # round_tag may already be "round_01" or just "01" or "1".
            round_str = f"  {round_tag}" if round_tag.startswith("round") else f"  r{round_tag}"
        else:
            round_str = ""
        # Provenance glyph prefix (Phase 8e badge) — single character so
        # the row's monospace alignment is preserved.
        if record.get("inherited_from_pair") or record.get("inherited_from_external"):
            prefix = "↳ "
        else:
            prefix = "  "  # blank padding so non-imported rows align
        return f"{prefix}{view:<3s} slice {str(idx):<5s}  {patch:<16s} {dots}{round_str}"

    def _build_provenance_tooltip(self, record):
        """Build a multi-line tooltip describing a slice's provenance.

        Returns empty string for records with no provenance (no tooltip).
        Phase 8e helper for ``_add_colored_item``.

        Priority (audit-documented 2026-05-24): ``inherited_from_pair``
        wins over ``inherited_from_external`` if both happen to be set
        on a single record (shouldn't happen in normal flow — a slice
        is imported via ONE mechanism, not both — but defensive).
        Cross-run pair provenance is more informative (carries source
        run + bucket + round) than external provenance (just a filename),
        so it's preferred for the tooltip.
        """
        prov = record.get("inherited_from_pair")
        if isinstance(prov, dict):
            src_run = prov.get("source_run_id", "?")
            src_bucket = prov.get("source_bucket", "?")
            src_round = prov.get("source_round_id", "?")
            src_label = prov.get("source_label", "?")
            imported_at = record.get("imported_at", "")
            lines = [
                "↳ Imported from another run",
                f"   run:    {src_run}",
                f"   bucket: {src_bucket}",
                f"   round:  {src_round}",
                f"   label:  {src_label}",
            ]
            if imported_at:
                lines.append(f"   when:   {imported_at[:19]}")  # strip ms+TZ
            return "\n".join(lines)
        # Phase 8f forward-compat — external import provenance
        ext = record.get("inherited_from_external")
        if isinstance(ext, dict):
            src_path = ext.get("source_path", "?")
            src_file = ext.get("source_filename", "?")
            lines = [
                "↳ Imported from external file",
                f"   file:   {src_file}",
                f"   folder: {src_path}",
            ]
            # Audit-fix 2026-05-24 (E): surface source_channel_axis so the
            # provenance audit trail is complete in the tooltip
            ch_axis = ext.get("source_channel_axis")
            if ch_axis:
                lines.append(f"   axis:   {ch_axis}")
            imported_at = record.get("imported_at", "")
            if imported_at:
                lines.append(f"   when:   {imported_at[:19]}")
            return "\n".join(lines)
        return ""

    def _add_colored_item(self, list_widget, text, status, sample_id=None,
                          record=None):
        """Add a list item with status icon and color.

        If *sample_id* is provided it is stored via ``Qt.UserRole`` so
        downstream code can retrieve the canonical key without parsing
        the display text.

        Phase 8e: if *record* is provided and has provenance, sets a
        tooltip describing the source.  Backward-compatible — older
        callers without ``record`` get no tooltip.
        """
        list_widget.addItem(text)
        item = list_widget.item(list_widget.count() - 1)
        if item is None:
            return
        if sample_id is not None:
            from qtpy.QtCore import Qt
            item.setData(Qt.UserRole, sample_id)
        color = self._STATUS_COLORS.get(status)
        if color is not None and hasattr(item, "setForeground"):
            item.setForeground(QBrush(color))
        # Add qtawesome icon.
        icon_name = self._STATUS_ICONS.get(status)
        if icon_name and color is not None:
            try:
                import qtawesome as qta
                item.setIcon(qta.icon(icon_name, color=color))
            except Exception:
                pass
        # Phase 8e provenance tooltip (only set if record carries import data)
        if record is not None and hasattr(item, "setToolTip"):
            tooltip = self._build_provenance_tooltip(record)
            if tooltip:
                item.setToolTip(tooltip)

    def _get_active_status_filters(self):
        """Return the set of statuses to show based on filter toggle buttons."""
        ui = getattr(self, "ui", None)
        if ui is None:
            return {"predicted", "accepted", "staged", "committed"}

        # If "All" is checked, show everything.
        all_btn = getattr(ui, "filter_all_btn", None)
        if all_btn is not None and hasattr(all_btn, "isChecked") and all_btn.isChecked():
            return {"predicted", "curating", "accepted", "staged", "committed"}

        # Otherwise, collect checked individual filters.
        # "Finalized" button (filter_staged_btn) matches both "staged" and "committed".
        active = set()
        for attr, statuses in (
            ("filter_predicted_btn", ("predicted",)),
            ("filter_curating_btn", ("curating",)),
            ("filter_accepted_btn", ("accepted",)),
            ("filter_staged_btn", ("staged", "committed")),
        ):
            btn = getattr(ui, attr, None)
            if btn is not None and hasattr(btn, "isChecked") and btn.isChecked():
                active.update(statuses)
        # If nothing checked, show all (safety fallback).
        return active if active else {"predicted", "curating", "accepted", "staged", "committed"}

    def _on_slice_filter_changed(self, *_args):
        """Handle filter toggle button change — uncheck All when individual selected."""
        ui = getattr(self, "ui", None)
        if ui is None:
            return
        all_btn = getattr(ui, "filter_all_btn", None)
        individual_btns = [
            getattr(ui, attr, None) for attr in
            ("filter_predicted_btn", "filter_curating_btn", "filter_accepted_btn",
             "filter_staged_btn", "filter_committed_btn")
        ]
        # If All was just checked, uncheck individuals.
        if all_btn is not None and all_btn.isChecked():
            for btn in individual_btns:
                if btn is not None and btn.isChecked():
                    btn.blockSignals(True)
                    btn.setChecked(False)
                    btn.blockSignals(False)
        else:
            # If any individual is checked, uncheck All.
            any_individual = any(
                btn is not None and btn.isChecked() for btn in individual_btns
            )
            if any_individual and all_btn is not None and all_btn.isChecked():
                all_btn.blockSignals(True)
                all_btn.setChecked(False)
                all_btn.blockSignals(False)
        self._refresh_sample_queue()

    def _refresh_sample_queue(self, *_args):
        """Populate the unified slice list with filtered slices.

        Filters by: selected patch, active view bucket, and status toggle buttons.
        Sorted by status rank (finalized first, predicted last).
        """
        ui = getattr(self, "ui", None)
        if ui is None:
            return

        active_bucket = self._bucket_for_view(self.selected_view)
        active_statuses = self._get_active_status_filters()
        active_patch = str(getattr(self, "loaded_patch_name", "") or "").strip()

        # Populate unified slice list (filtered by patch + bucket + status).
        slice_list = getattr(ui, "segment_slice_data_list", None)
        if slice_list is not None and hasattr(slice_list, "clear"):
            # 2026-05-25 fix (Audit 5): snapshot selection BEFORE clear()
            # so we can restore it after repopulate.  Without this, every
            # status-changing handler (Accept / Finalize / discard /
            # mark-good) drops the user's selection silently, breaking
            # itemSelectionChanged → focus_sample → apply_active_view_to_viewer
            # (no auto-pan after Finalize).  The user reported the
            # symptom for Finalize but Accept has the same gap.
            prev_selected_ids = set()
            try:
                for it in (slice_list.selectedItems() or []):
                    sid = it.data(Qt.UserRole) if hasattr(it, "data") else None
                    if sid:
                        prev_selected_ids.add(str(sid))
            except Exception:
                prev_selected_ids = set()
            prev_current_id = None
            try:
                cur = slice_list.currentItem() if hasattr(slice_list, "currentItem") else None
                if cur is not None and hasattr(cur, "data"):
                    cid = cur.data(Qt.UserRole)
                    if cid:
                        prev_current_id = str(cid)
            except Exception:
                pass

            slice_list.clear()
            # Step 2 look (brick A, D24): the one Slices order, ``curating``
            # ranked (``_step2_lists.slice_sort_key`` is the old rank table)
            from ..ui._step2_lists import slice_sort_key as _slice_sort_key
            matching = []
            for sid, rec in self.sample_records.items():
                if rec.get("bucket") != active_bucket:
                    continue
                if rec.get("status", "predicted") not in active_statuses:
                    continue
                # Filter by selected patch.
                rec_patch = str(rec.get("patch_id") or rec.get("roi_id") or "").strip()
                if not rec_patch or rec_patch == "-":
                    parts = sid.rsplit("_", 2)
                    rec_patch = parts[0] if len(parts) >= 3 else sid
                if active_patch and rec_patch != active_patch:
                    continue
                matching.append((sid, rec))
            matching.sort(key=lambda kv: _slice_sort_key(kv[1]))
            for sample_id, record in matching:
                status = record.get("status", "predicted")
                self._add_colored_item(
                    slice_list, self._format_queue_row(record), status,
                    sample_id=sample_id, record=record,
                )

            # Restore selection if any of the previously-selected rows
            # still exist after repopulate.  Block signals during restore
            # so we don't fire itemSelectionChanged N times.
            if prev_selected_ids:
                try:
                    slice_list.blockSignals(True)
                    restored = 0
                    new_current = None
                    for row in range(slice_list.count()):
                        item = slice_list.item(row)
                        sid = item.data(Qt.UserRole) if (item is not None and hasattr(item, "data")) else None
                        if sid is None:
                            continue
                        sid_str = str(sid)
                        if sid_str in prev_selected_ids:
                            item.setSelected(True)
                            restored += 1
                            if new_current is None or sid_str == prev_current_id:
                                new_current = item
                    if new_current is not None and hasattr(slice_list, "setCurrentItem"):
                        slice_list.setCurrentItem(new_current)
                except Exception as exc:
                    print(f"[WARN] Slice-list selection restore failed: {exc}")
                    restored = 0
                finally:
                    try:
                        slice_list.blockSignals(False)
                    except Exception:
                        pass
                if restored:
                    pass

        # Step 2 look (brick A, 4.1 (b)): the painted Slices rows, the chip
        # counts and the pictures -- read-only state, one guarded hook.
        if slice_list is not None and hasattr(slice_list, "count"):
            self._notify_slice_rows(
                staged=False, ids=self._listed_ids(slice_list),
                bucket=active_bucket, patch=active_patch)

        # Curation progress: the 5 x 2 Lateral |
        # Axial table is built once in the UI; this only hands it the counts
        # (stored status keys: staged = Finalized, committed = Trained).
        progress_table = getattr(ui, "curation_progress_table", None)
        if progress_table is not None:
            counts = {"lateral": {}, "axial": {}}
            for r in self.sample_records.values():
                b = r.get("bucket", "")
                s = r.get("status", "")
                if b in counts and s in (
                        "predicted", "curating", "accepted", "staged",
                        "committed"):
                    counts[b][s] = counts[b].get(s, 0) + 1
            try:
                from ..ui._step2_curation import compose_progress_note
                by_view = getattr(self, "current_round_by_view", None) or {}
                lat_n = int(by_view.get("lateral", 0) or 0)
                axi_n = int(by_view.get("axial", 0) or 0)
                # current_round_by_view = the 0-based round a Train would
                # write NEXT (disk numbering, D16): rounds 0..n-1 are trained.
                # the span is the ACTIVE bucket's (an axial-only run names
                # its axial rounds, not "no round yet")
                act_n = axi_n if active_bucket == "axial" else lat_n
                note = compose_progress_note(
                    (0, act_n - 1) if act_n > 0 else None, lat_n, axi_n,
                    bucket="axial" if active_bucket == "axial" else "lateral")
                progress_table.refresh(counts, active_bucket, note)
            except Exception as exc:  # noqa: BLE001 -- presentation only
                print(f"[STEP2-UI] progress table refresh failed: {exc}")

        # Refresh patch list badges with per-patch counts.
        self._refresh_patch_list_badges()
        self._refresh_curation_stage_line()

    @staticmethod
    def _listed_ids(list_widget):
        """The ``UserRole`` sample ids of a list's rows, in row order."""
        from qtpy.QtCore import Qt as _Qt
        out = []
        try:
            for i in range(list_widget.count()):
                it = list_widget.item(i)
                sid = it.data(_Qt.UserRole) if it is not None else None
                if sid is None and it is not None:
                    sid = it.text()
                if sid:
                    out.append(str(sid))
        except Exception:      # noqa: BLE001 -- a list mid-rebuild
            pass
        return out

    def _slice_stack_paths(self):
        """``{patch stem: patch file}`` from the patch list's rows (names
        only; no disk access) -- the fallback source of a slice picture
        when a record has no ``img_uri``."""
        out = {}
        try:
            stack_dir = self._effective_stack_path()
            plist = getattr(self.ui, "train_patch_list", None)
            if not stack_dir or plist is None:
                return out
            for name in self._listed_ids(plist):
                out[os.path.splitext(name)[0]] = os.path.join(stack_dir, name)
        except Exception:      # noqa: BLE001 -- pictures only
            pass
        return out

    def _notify_slice_rows(self, *, staged, ids, bucket, patch=""):
        """Hook ``on_slice_rows``: hand the
        UI the records the rows are painted from, which sample ids are
        listed, and where the pictures come from.  Read-only; a UI without
        the hook (a fake) simply has plain rows; a failing hook never breaks
        the refresh."""
        fn = getattr(getattr(self, "ui", None), "on_slice_rows", None)
        if fn is None:
            return
        try:
            fn({
                "records": self.sample_records,
                "ids": ids,
                "staged": bool(staged),
                "bucket": bucket,
                "patch": patch,
                "loaded_patch": getattr(self, "loaded_patch_name", ""),
                "view": self.selected_view,
                "workspace": self.working_dir or "",
                "stack_paths": self._slice_stack_paths(),
            })
        except Exception as exc:      # noqa: BLE001 -- a hook must never break the refresh
            print(f"[STEP2-UI] on_slice_rows failed: {exc}")

    def _refresh_patch_list_badges(self):
        """Refresh the patch lists' rows.

        Step 2 look (brick L): the
        rows are painted by the shared delegate from a record each item
        carries; this method no longer builds ``setItemWidget`` rows or
        blanks ``item.text()`` -- ``item.text()`` and ``UserRole`` stay the
        patch file name (``_clean_patch_filename`` reads them).  It hands the
        UI the read-only state the rows are made from (``on_session_patches``,
        one tally for Session's list and Curation's); a UI without the hook
        (a fake) simply has no painted rows.
        """
        ui = getattr(self, "ui", None)
        patch_list = getattr(ui, "train_patch_list", None) if ui is not None else None
        if patch_list is None or not hasattr(patch_list, "count"):
            return
        fn = getattr(ui, "on_session_patches", None)
        if fn is None:
            return
        try:
            staged_by_bucket = {"lateral": 0, "axial": 0}
            for rec in self.sample_records.values():
                if rec.get("status") == "staged" and rec.get("bucket") in staged_by_bucket:
                    staged_by_bucket[rec["bucket"]] += 1
            try:
                stack_dir = self._effective_stack_path()
            except Exception:      # noqa: BLE001 -- a hint for the pictures only
                stack_dir = ""
            run_pairs = run_desc = None
            for _s in (getattr(self, "_run_summaries_cache", None) or []):
                if _s.get("is_active"):
                    run_pairs = _s.get("accepted_pair_count")
                    run_desc = _s.get("description") or None
                    break
            fn({
                "run_pairs": run_pairs,
                "run_desc": run_desc,
                "records": self.sample_records,
                "shapes": getattr(self, "_patch_shapes", {}),
                "round_by_view": getattr(self, "current_round_by_view", {}),
                "loaded": getattr(self, "loaded_patch_name", ""),
                "view": self.selected_view,
                "staged_by_bucket": staged_by_bucket,
                "workspace": self.working_dir or "",
                "stack_dir": stack_dir,
                "run_id": getattr(self, "active_run_id", None) or "default",
            })
        except Exception as exc:      # noqa: BLE001 -- a hook must never break the refresh
            print(f"[STEP2-UI] on_session_patches failed: {exc}")

    def _stage_accepted_in_current_view(self):
        current_bucket = self._bucket_for_view(self.selected_view)
        changed = 0
        for sample_id, record in self.sample_records.items():
            if record.get("bucket") == current_bucket and record.get("status") == "accepted":
                self._touch_sample(sample_id, status="staged", view=record.get("view", self.selected_view))
                self._persist_curated_training_pair(sample_id)
                changed += 1
        self._sync_data_to_add_list()
        self._refresh_sample_queue()
        self._refresh_round_status()
        # Edit-states: staged slices flip to the ice soft-lock visual.
        self._refresh_annotator_edit_state_visuals()
        if changed:
            self._append_ui_log(f"[QUEUE] Staged {changed} accepted sample(s) in {current_bucket}.")
            self._schedule_state_save()
            # 2026-05-25 Group D: bump dropdown pair count after stage —
            # _persist_curated_training_pair has just written N new
            # img.tif/mask.tif files under accepted/, so the counter
            # changes.
            try:
                self._refresh_run_picker()
            except Exception as e:
                pass
        else:
            self._append_ui_log(f"[QUEUE] No accepted samples to stage in {current_bucket}.")

    def _persist_curated_training_pair(self, slice_label, async_write=False):
        """Persist the (image, mask) training pair for ``slice_label`` to disk.

        ``async_write`` (Stage 2, 2026-06-02): when True, the TIFF write is
        handed to the background ``_pair_writer`` (FIFO, off the Qt main
        thread) instead of being written inline — used ONLY by the per-stroke
        debounced curating-persist, so steady-state drawing never blocks on the
        ~300ms disk write.  All deliberate callers (accept / predict / stage)
        keep the default synchronous write so the file is guaranteed on disk
        before the method returns (their contract is unchanged).  Either way
        the returned uris + record fields are set immediately; hard barriers
        (train / switch / shutdown) flush the async writer.
        """
        if not self.working_dir:
            return {}
        input_stack = self.segmentation_manager.get_input_stack(slice_label)
        mask = self.segmentation_manager.get_masks(slice_label)
        if input_stack is None or mask is None:
            return {}

        try:
            arr_img = np.asarray(input_stack)
            arr_mask = np.asarray(mask)
        except Exception:
            return {}
        if arr_img.ndim != 3 or arr_mask.ndim != 2:
            return {}

        # In-memory change detection (no disk read).  Replaces the prior
        # disk-read-based compare (it read the old mask back via tifffile
        # just to diff), which cost a full mask read on every edit of an
        # actively-curated slice (~hundreds of ms to a network/spinning
        # drive) and was largely defeated anyway by the pre-write
        # _compact_label_ids relabel (the on-disk mask is compacted, so
        # np.array_equal vs the raw in-memory mask almost never matched).
        # We hash the raw in-memory arrays; each write is skipped only when
        # its on-disk file exists AND its signature is unchanged since the
        # last persist THIS session.  The image signature additionally lets
        # us skip the large float32 IMAGE rewrite on every mask-only edit —
        # the image for a given (stack, view, slice) never changes.
        if not hasattr(self, "_persisted_pair_sigs"):
            self._persisted_pair_sigs = {}
        img_sig = _array_signature(arr_img)
        mask_sig = _array_signature(arr_mask)

        parts = str(slice_label).rsplit("_", 2)
        if len(parts) != 3:
            return {}
        stack_id, view_name, slice_index_txt = parts
        view = str(view_name).upper()
        if view not in {"XY", "XZ", "YZ"}:
            return {}
        try:
            slice_index = int(slice_index_txt)
        except Exception:
            return {}

        # Phase C.5 fix (2026-05-25, audit 1 HIGH-1): use the slice's
        # OWN bucket counter, not `_current_round()` which reads
        # `selected_view` and returns the SELECTED bucket's counter.
        # Pre-fix in Mode=Both with lateral=5 + axial=2 + selected_view=XY,
        # accepting an XZ slice would write to `axial/round_05/accepted/`
        # (using lateral's counter).  Same DRIFT class as Phase B.1's
        # WRITE-side fix but on the accept path.  Pre-B.1 the counters
        # were shared so this was masked; B.1's per-bucket divergence
        # exposed the bug.
        from napari_epicycle.hitl_training_annotation import _paths as _step2_paths_for_round
        _bucket_for_round = _step2_paths_for_round.bucket_for_view(view)
        _by_view = getattr(self, "current_round_by_view", None) or {}
        if isinstance(_by_view, dict) and _bucket_for_round in _by_view:
            round_idx = int(_by_view[_bucket_for_round])
        else:
            round_idx = int(self._current_round())

        # Step 2 layout restructure (Phase 3 sub-C): if workspace is on
        # the new layout, write to runs/{active_run_id}/{bucket}/rounds/
        # round_NN/accepted/ via _paths helpers.  Otherwise legacy path.
        # This keeps the on-disk layout consistent post-migration —
        # without it, new pairs would land in legacy training_data/
        # creating MIXED layout drift.
        # 2026-09-07 (console-encoding hazard): NO print() inside this
        # try.  A print can RAISE (UnicodeEncodeError on a non-UTF-8
        # Windows console is a ValueError subclass), and this except
        # reassigns out_dir to the LEGACY training_data/ folder — which
        # is exactly how two accepted pairs were written to the wrong
        # place on a v3 workspace.  Diagnostics are collected here and
        # emitted AFTER the try/except, where a raise is harmless.
        _layout_diag = []
        try:
            from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
            # User-bug fix 2026-05-24: if state JSON is missing but the
            # workspace is otherwise eligible for new layout (UNKNOWN
            # or already has runs/ dirs), AUTO-BOOTSTRAP the default
            # run instead of silently falling back to legacy
            # training_data/.  Without this, the first action on a
            # fresh workspace (e.g., Accept) could create training_data/
            # before the workspace-load bootstrap had a chance to run.
            if not _step2_paths.is_new_layout_state(self.working_dir):
                layout_v = _step2_paths.detect_layout_version(self.working_dir)
                if layout_v == _step2_paths.LAYOUT_VERSION_UNKNOWN:
                    _layout_diag.append(
                        "[STEP2-WRITE] no state JSON + UNKNOWN layout - "
                        "bootstrapping default run before write"
                    )
                    try:
                        self._auto_bootstrap_default_run_if_fresh(self.working_dir)
                        # Sync runtime active_run_id since the bootstrap
                        # wrote it to disk but didn't update controller field
                        if not getattr(self, "active_run_id", None):
                            self.active_run_id = _step2_paths.DEFAULT_RUN_ID
                        # Audit-fix 2026-05-24 (G): refresh the run picker
                        # so the dropdown reflects the newly-created
                        # default run.  Without this, the dropdown keeps
                        # showing "(legacy layout)" synthetic entry until
                        # user does something else that triggers a refresh.
                        try:
                            self._refresh_run_picker()
                        except Exception as _refresh_err:
                            _layout_diag.append(
                                f"[STEP2-WRITE] post-bootstrap picker "
                                f"refresh failed: {_refresh_err}"
                            )
                    except Exception as _boot_err:
                        _layout_diag.append(
                            f"[STEP2-WRITE] bootstrap failed: {_boot_err}"
                        )
                # Only LEGACY/MIXED stays on legacy path — those workspaces
                # have pre-existing training_data/ and the migration prompt
                # will handle the conversion on next session.
            if _step2_paths.is_new_layout_state(self.working_dir):
                active_run_id = getattr(
                    self, "active_run_id", _step2_paths.DEFAULT_RUN_ID
                )
                if not active_run_id:
                    active_run_id = _step2_paths.DEFAULT_RUN_ID
                    self.active_run_id = active_run_id
                bucket = _step2_paths.bucket_for_view(view)
                out_dir = _step2_paths.round_accepted_dir(
                    self.working_dir, active_run_id, bucket, round_idx
                )
                _layout_diag.append(
                    f"[STEP2-WRITE] _persist_curated_training_pair -> NEW "
                    f"layout: {view} -> {bucket} bucket / "
                    f"run='{active_run_id}' / round_{round_idx:02d}"
                )
            else:
                # LEGACY/MIXED workspace — preserve legacy write path
                round_id = f"round_{round_idx:02d}"
                target_dir = "spatial" if view == "XY" else "axial"
                out_dir = Path(self.working_dir) / "training_data" / round_id / "accepted_pairs" / target_dir
                _layout_diag.append(
                    "[STEP2-WRITE] _persist_curated_training_pair -> LEGACY "
                    "layout (workspace has pre-existing training_data/; "
                    "migrate via Workspace > Browse to use new layout)"
                )
        except Exception as _e:
            # Defensive: any path-resolution issue falls back to legacy
            _layout_diag.append(
                f"[STEP2-WRITE] new-layout path resolution failed "
                f"({type(_e).__name__}: {_e}), using legacy path"
            )
            round_id = f"round_{round_idx:02d}"
            target_dir = "spatial" if view == "XY" else "axial"
            out_dir = Path(self.working_dir) / "training_data" / round_id / "accepted_pairs" / target_dir

        # Emit the collected diagnostics OUTSIDE the guarded region: a
        # raise here cannot redirect the write any more.
        for _line in _layout_diag:
            print(_line)

        out_dir.mkdir(parents=True, exist_ok=True)

        fname_base = f"{stack_id}_{view}_z{slice_index:04d}"
        img_path = out_dir / f"{fname_base}_img.tif"
        mask_path = out_dir / f"{fname_base}_mask.tif"

        # Only (re)write the files that actually changed + the missing ones.
        # In the curating hot-loop the mask changes every edit (writes) but
        # the image is invariant (skipped after its first write) — that alone
        # halves the per-edit disk I/O on a 2-channel float32 slice.
        #
        # Cache key = the RESOLVED on-disk target (out_dir bakes in
        # active_run_id + round_idx; fname_base bakes in view + slice), NOT
        # slice_label.  A training-round advance or a run switch redirects the
        # same slice_label to a DIFFERENT file; keying on the path guarantees
        # the skip decision is always made against the signature of THIS exact
        # file, so a signature cached for another round/run can never wrongly
        # skip a write.  (Independent audit, 2026-06-02 — fixes the cross-
        # context staleness the slice_label key would have allowed.)
        prior_sigs = self._persisted_pair_sigs.setdefault(
            str(out_dir / fname_base), {}
        )
        write_img, write_mask = _pair_write_plan(
            img_exists=img_path.is_file(),
            mask_exists=mask_path.is_file(),
            img_sig=img_sig,
            mask_sig=mask_sig,
            prior=prior_sigs,
        )
        # Stage 2 (2026-06-02): compaction is now cheap (vectorized) and stays
        # on the main thread; for the per-stroke debounce path (async_write),
        # only the GIL-releasing TIFF write is offloaded to the background FIFO
        # writer so steady-state drawing/zoom/pan never blocks on the ~300ms
        # disk write.  Deliberate callers (accept/predict/stage) use the
        # synchronous path so the file is on disk before they return.  Arrays
        # handed to either path are fresh `.astype()` copies (snapshots) so a
        # later main-thread edit can never race an in-flight write.  Hard
        # barriers (train / switch / shutdown) flush the async writer.  The
        # in-memory sig is updated optimistically; a failed write leaves no
        # file (atomic), so the next persist's `not file_exists` check forces a
        # rewrite (self-healing).
        if async_write and getattr(self, "_pair_writer", None) is None:
            self._pair_writer = _AsyncPairWriter()

        def _emit(path, data):
            if async_write:
                self._pair_writer.submit(str(path), data)
            else:
                _atomic_imwrite(str(path), data)

        if write_img:
            _emit(img_path, arr_img.astype(np.float32))
            prior_sigs["img"] = img_sig
        if write_mask:
            # 2026-05-28 (label-compaction-v2): compact the 2D mask's label IDs
            # to a dense 1..N range BEFORE writing to disk.  This breaks the
            # cross-session compounding that inflates IDs into the millions
            # (the in-memory volume's monotonic base_max offset gets persisted,
            # reloaded, re-offset, re-persisted...).  DISK-ONLY: the in-memory
            # volume + displayed layers are NOT touched here, so the subviewers
            # keep their cross-slice-unique offset IDs during the session.  The
            # restore loop re-applies cross-slice offsets on reload, so the
            # rebuilt volume stays collision-free.  Per-slice disk masks are
            # independent training pairs; Cellpose is ID-agnostic.
            #
            # .astype(np.int32) forces a fresh copy (snapshot) for the writer —
            # _compact_label_ids may return its input array on the fast paths.
            compact_mask = _compact_label_ids(arr_mask.astype(np.int32)).astype(np.int32)
            _emit(mask_path, compact_mask)
            prior_sigs["mask"] = mask_sig

        record = self.sample_records.get(slice_label, {})
        record["img_uri"] = str(img_path)
        record["mask_uri"] = str(mask_path)
        if write_img or write_mask:
            record["pair_saved_at"] = datetime.now(timezone.utc).isoformat()
        self.sample_records[slice_label] = record
        return {"img_uri": str(img_path), "mask_uri": str(mask_path)}

    # Statuses whose on-disk pair is genuine training truth.  Everything else
    # in a round's accepted/ dir is prediction scratch or work in progress —
    # the dir doubles as both today (journey #1; the honest-storage split is
    # a deferred follow-up).
    _TRAINABLE_PRIOR_STATUSES = ("accepted", "staged", "committed")

    def _prior_pair_status_index(self):
        """Map ``(stack, VIEW, idx)`` -> status for every sample record.

        Records are keyed by slice_label in EITHER format
        (``{stack}_{VIEW}_z{idx:04d}`` or the pre-2026-05 ``{stack}_{VIEW}_
        {idx}``), so the key is decomposed with the existing tolerant
        helpers rather than `parse_slice_label`, which only accepts the
        z-padded form.
        """
        index = {}
        for label, record in (self.sample_records or {}).items():
            try:
                text = str(label)
                if text.count("_") < 2:
                    continue
                stack = text.rsplit("_", 2)[0]
                view = str(self._infer_view_from_label(text)).upper()
                idx = int(self._infer_slice_index(text))
            except Exception:
                continue
            if idx < 0:
                continue
            index[(stack, view, idx)] = str(
                (record or {}).get("status", "") or ""
            )
        return index

    def _slice_key_for_pair_filename(self, filename):
        """``{stack}_{VIEW}_z{idx:04d}_img.tif`` -> ``(stack, VIEW, idx)``.

        Falls back to the tolerant label helpers for the VERY-LEGACY
        ``training_data/round_NN/accepted_pairs/`` tier, whose pairs are named
        ``{stack}_{VIEW}_{idx}_img.tif`` (no ``z`` prefix, no zero padding) and
        so do not match ``parse_slice_label``'s regex.  Without the fallback a
        legacy workspace would see every prior pair dropped as "no record".
        """
        stem = str(filename)
        for suffix in ("_img.tif", "_img.tiff", "_mask.tif", "_mask.tiff"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        try:
            from napari_epicycle.hitl_training_annotation._paths import (
                parse_slice_label,
            )
            return parse_slice_label(stem)
        except Exception:
            pass
        try:
            if stem.count("_") < 2:
                return None
            stack = stem.rsplit("_", 2)[0]
            view = str(self._infer_view_from_label(stem)).upper()
            idx = int(self._infer_slice_index(stem))
        except Exception:
            return None
        if idx < 0 or view not in {"XY", "XZ", "YZ"}:
            return None
        return (stack, view, idx)

    def _collect_prior_round_pairs(self, bucket=None, log_skips=False):
        """Scan disk for training pairs from prior rounds.

        Only collects pairs matching the TARGET training bucket
        (lateral → spatial, axial → axial). Does NOT mix views.

        Args:
            bucket: explicit "lateral" or "axial".  When None, falls
                back to `self._current_round_key()` (the selected-view's
                bucket).  Phase C.5 fix (2026-05-25, audit 3 HIGH-2):
                callers in the per-view training loop must pass an
                explicit bucket so Mode=Both's axial iteration finds
                axial's own prior pairs instead of falling through to
                the selected-view's bucket (silent scratch training).
            log_skips: emit the B-lite "left out N slice(s)" summary.  ONLY
                the training call site passes True — `_refresh_round_status`
                calls this method on every accept / predict / load / status
                tick, and an unconditional log would flood the panel.

        Returns list of dicts: [{"img_uri", "mask_uri", "view_group"}].

        The B-lite record filter runs BEFORE the latest-round-wins dedupe, so
        a slice is all-or-nothing: an unaccepted copy in a later round does
        not shadow (or resurrect) an accepted copy in an earlier one — the
        status index is keyed on (stack, view, idx), not on the round.
        """
        if not self.working_dir:
            return []

        pairs = []

        # Resolve target bucket — explicit arg overrides selected-view fallback.
        if bucket is not None and bucket in {"lateral", "axial"}:
            active_bucket = bucket
        else:
            active_bucket = self._current_round_key()
        target_view_group_legacy = "spatial" if active_bucket == "lateral" else "axial"
        target_bucket_new = active_bucket
        # Per-bucket round counter — pre-fix used `self._current_round()`
        # which reads `selected_view`.  In Mode=Both with divergent
        # counters, axial iteration of the per-view loop would scan
        # rounds 0..(lateral_counter) for axial — too many AND wrong
        # files (axial round dirs may not exist that high).  Post-fix:
        # use the target bucket's OWN counter.
        _by_view = getattr(self, "current_round_by_view", None) or {}
        if isinstance(_by_view, dict) and active_bucket in _by_view:
            target_round_count = int(_by_view[active_bucket])
        else:
            target_round_count = int(self._current_round())

        # Map basename -> pair dict, tracking which round each came from.
        # 2026-05-27 (edit-states branch): LATEST-round-wins.
        #
        # Pre-fix this used a `seen_uris` set with first-occurrence-wins
        # AND an ascending round scan (range(target_round_count)).  That
        # meant the EARLIEST round's copy of a slice won.  When a user
        # reverts an accepted slice, re-edits it, and re-accepts it in a
        # later round, the old-round TIFF still exists on disk (revert
        # doesn't delete prior-round files) — so the stale original
        # silently shadowed the correction and training used the OLD mask.
        #
        # Post-fix: keep a (basename -> (round_idx, pair)) map and let the
        # HIGHEST round_idx win.  A re-edited slice's newest version now
        # correctly overwrites its prior-round copy in cumulative training.
        #
        # WITHIN a single round, NEW-layout dir is still scanned before
        # legacy dirs (see round_search_dirs ordering below), and within
        # the same round we keep first-seen (new beats legacy) — so the
        # mid-migration "new is authoritative" invariant from the
        # 2026-05-23 audit is preserved at the per-round level.
        latest_by_basename = {}

        # B-lite (2026-09-07, journey #1 + #7): the round dir is not status-
        # honest, so filter on the RECORD.  A pair whose slice was only
        # predicted, is still being edited, or has no record at all (the
        # discard-then-resurrect path) must never train as ground truth.
        # READER-side only — nothing on disk moves or is deleted.
        status_index = self._prior_pair_status_index()

        for round_idx in range(target_round_count):
            round_search_dirs = []

            # Phase B.2 (2026-05-25): triple-path scan with priority
            # NEW FLAT > LEGACY SPLIT > VERY-LEGACY training_data/.
            # `round_accepted_dir` now returns the FLAT-layout path
            # `bucket/round_NN/accepted/`.  Pre-Phase-B.2 workspaces
            # have accepted/ at `bucket/rounds/round_NN/accepted/`
            # (legacy split) — reach that via `legacy_split_round_dir`.
            # Pre-Phase-6c workspaces (very legacy) have it at
            # `training_data/round_NN/accepted_pairs/{spatial,axial}/`.
            try:
                from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
                active_run_id = getattr(
                    self, "active_run_id", _step2_paths.DEFAULT_RUN_ID
                )
                # NEW FLAT layout (Phase B.2)
                new_accepted = _step2_paths.round_accepted_dir(
                    self.working_dir, active_run_id, target_bucket_new, round_idx
                )
                if new_accepted.is_dir():
                    round_search_dirs.append(new_accepted)
                # LEGACY split layout (post-Phase-6c, pre-Phase-B.2)
                legacy_split_accepted = (
                    _step2_paths.legacy_split_round_dir(
                        self.working_dir, active_run_id, target_bucket_new, round_idx
                    ) / "accepted"
                )
                if legacy_split_accepted.is_dir():
                    round_search_dirs.append(legacy_split_accepted)
            except Exception:
                pass  # defensive — very-legacy fallback still runs

            # VERY-LEGACY location: training_data/round_NN/accepted_pairs/{spatial,axial}/
            # (pre-Phase-6c).  Last fallback for un-migrated workspaces.
            legacy_base = Path(self.working_dir) / "training_data" / f"round_{round_idx:02d}"
            legacy_vg = legacy_base / "accepted_pairs" / target_view_group_legacy
            if legacy_vg.is_dir():
                round_search_dirs.append(legacy_vg)

            # Track basenames seen WITHIN this round so new-layout beats
            # legacy at the per-round level (mid-migration invariant).
            seen_this_round = set()
            skipped_unaccepted = 0
            skipped_no_record = 0
            for vg_dir in round_search_dirs:
                img_files = sorted(vg_dir.glob("*_img.tif"))
                for img_path in img_files:
                    mask_path = img_path.parent / img_path.name.replace("_img.tif", "_mask.tif")
                    if not mask_path.is_file():
                        self._append_ui_log(f"[WARN] Prior pair missing mask: {mask_path}")
                        continue
                    key = img_path.name
                    if key in seen_this_round:
                        # Already found in a higher-priority dir THIS round
                        # (new layout scanned before legacy).
                        continue
                    seen_this_round.add(key)
                    # B-lite status filter — see status_index above.
                    slice_key = self._slice_key_for_pair_filename(img_path.name)
                    if slice_key is None:
                        skipped_no_record += 1
                        continue
                    status = status_index.get(slice_key)
                    if status is None:
                        skipped_no_record += 1
                        continue
                    if status not in self._TRAINABLE_PRIOR_STATUSES:
                        skipped_unaccepted += 1
                        continue
                    # LATEST-round-wins: a higher round_idx overwrites the
                    # entry from any earlier round.  range() is ascending,
                    # so simply overwriting on each occurrence yields the
                    # highest round's copy.
                    latest_by_basename[key] = {
                        "img_uri": str(img_path),
                        "mask_uri": str(mask_path),
                        "view_group": target_view_group_legacy,
                    }

            if log_skips and skipped_unaccepted:
                self._append_ui_log(
                    f"[TRAIN] round_{round_idx:02d} ({active_bucket}): left out "
                    f"{skipped_unaccepted} slice(s) you had not accepted yet "
                    "(raw predictions or unfinished edits)."
                )
            if log_skips and skipped_no_record:
                self._append_ui_log(
                    f"[TRAIN] round_{round_idx:02d} ({active_bucket}): left out "
                    f"{skipped_no_record} slice(s) EpiCycle has no curation "
                    "record for — if this workspace predates slice tracking, "
                    "those slices will not train."
                )

        # Deterministic order (by basename) for reproducible training sets.
        for key in sorted(latest_by_basename.keys()):
            pairs.append(latest_by_basename[key])

        return pairs

    def _resolve_prior_checkpoints_for_buckets(self, buckets):
        """Prior-round checkpoints for every bucket that is about to train,
        each honouring THAT bucket's own training mode.

        Returns a dict keyed by Cellpose view_type ("spatial" / "axial") --
        the shape `_run_training_work` consumes.  A bucket set to
        "From Scratch" contributes NO entry, so the worker starts it from
        the Base Model / Checkpoint instead of the prior round.

        Release-hardening 2026-09-07 fix pass.  Pre-fix the pipeline
        read the flat `training_mode` once and applied it to every bucket,
        so in a mixed XY+XZ round the per-bucket setting never reached the
        trainer.
        """
        resolved = {}
        for bucket in sorted(buckets or ()):
            mode = self._training_mode_for_bucket(bucket)
            if mode == "cumulative":
                # Keyed by view_type ("spatial" / "axial") -- in a mixed
                # round each bucket contributes its OWN entry.
                resolved.update(self._resolve_prior_model_checkpoint(bucket=bucket))
            else:
                self._append_ui_log(
                    f"[TRAIN] {bucket}: from-scratch mode -- training a fresh "
                    f"model (no prior checkpoint)."
                )
        return resolved

    def _resolve_prior_model_checkpoint(self, bucket=None):
        """Find the most recent model checkpoint for the target bucket's
        previous round.

        Args:
            bucket: explicit "lateral" or "axial".  When None, falls
                back to `self._current_round_key()` (the selected-view's
                bucket).  Phase C.5 fix (2026-05-25, audit 3 HIGH-2):
                callers in the per-view training loop must pass an
                explicit bucket so Mode=Both's axial iteration inherits
                from axial's own prior checkpoint instead of falling
                through to the selected-view's bucket (silent scratch
                training start).
        """
        # Resolve target bucket — explicit arg overrides selected-view fallback.
        if bucket is not None and bucket in {"lateral", "axial"}:
            target_bucket = bucket
        else:
            target_bucket = self._current_round_key()
        # Per-bucket round counter — see analogous fix in
        # `_collect_prior_round_pairs`.
        _by_view = getattr(self, "current_round_by_view", None) or {}
        if isinstance(_by_view, dict) and target_bucket in _by_view:
            target_round = int(_by_view[target_bucket])
        else:
            target_round = int(self._current_round())
        if target_round == 0:
            return {}
        prev = target_round - 1
        # Try composite key first (new format), fall back to bare int (legacy).
        round_key = f"{target_bucket}_{prev}"
        checkpoints = self.model_checkpoint_paths.get(round_key)
        if checkpoints is None:
            checkpoints = self.model_checkpoint_paths.get(prev, {})
        # When called per-bucket (explicit `bucket` arg), return just the
        # bucket's view_type entry — pre-fix wrapped a string as both
        # `spatial` AND `axial` keys, which would have caused the OTHER
        # bucket's iteration to inherit from this bucket's checkpoint
        # in Mode=Both.
        view_type_key = "spatial" if target_bucket == "lateral" else "axial"
        if isinstance(checkpoints, str):
            if bucket is not None:
                return {view_type_key: checkpoints}
            # Legacy no-arg call: preserve pre-fix dual-key wrap
            return {"spatial": checkpoints, "axial": checkpoints}
        if isinstance(checkpoints, dict):
            return dict(checkpoints)
        return {}

    def _parse_all_selected_sample_ids(self):
        """Parse sample_ids from ALL selected slice list items.

        Preferred path: read the canonical ``sample_id`` stored via
        ``Qt.UserRole`` on each item (set by ``_add_colored_item``).
        Falls back to text-based parsing for legacy items.
        """
        ui = getattr(self, "ui", None)
        if ui is None:
            return []
        queue = getattr(ui, "sample_queue_list", None)
        if queue is None or not hasattr(queue, "selectedItems"):
            return []
        from qtpy.QtCore import Qt
        import re
        results = []
        for item in queue.selectedItems():
            # Preferred: read sample_id from item data.
            stored = item.data(Qt.UserRole) if hasattr(item, "data") else None
            if stored and str(stored) in self.sample_records:
                results.append(str(stored))
                continue
            # Fallback: text-based parsing.
            text = item.text() if hasattr(item, "text") else str(item)
            stripped = text.strip()
            if stripped in self.sample_records:
                results.append(stripped)
                continue
            if " | " in text:
                candidate = text.split(" | ", 1)[0].strip()
                if candidate in self.sample_records:
                    results.append(candidate)
                continue
            parts = re.split(r"\s{2,}", stripped)
            if len(parts) >= 3:
                view = parts[0].strip().upper()
                idx_match = re.search(r"(\d+)", parts[1].strip())
                if idx_match:
                    patch = parts[2].strip()
                    candidate = f"{patch}_{view}_{idx_match.group(1)}"
                    if candidate in self.sample_records:
                        results.append(candidate)
        return results

    def _unstage_selected_queue_items(self):
        """Revert the selected slice(s) to the editable `curating` state.

        2026-05-27 (edit-states branch): generalized from "unstage only"
        (staged -> accepted) to "revert to editable" (any finalized
        state -> curating), per the locked design.  Rationale: a "Revert"
        button should do ONE predictable thing — make the slice editable
        again — rather than a state-dependent step-back the user can't
        predict.  This is the deliberate main-widget gate for re-editing
        finalized data (the annotator can't revert by design, so accidental
        edits there are soft-blocked).

        Semantics:
          * predicted / curating  -> no-op (already editable)
          * accepted / staged     -> curating (no confirm)
          * committed             -> curating, but ONLY after a themed
            confirm, because a committed slice already shaped a trained
            model — reverting it has provenance weight.

        `_touch_sample` preserves origin + inherited_from_* fields (it only
        rewrites the listed keys), so imported/cherry-picked provenance
        survives the revert.
        """
        ui = getattr(self, "ui", None)
        if ui is None:
            return
        # Prefer staged_queue_list (new 3-tab layout), fallback to sample_queue_list.
        queue = getattr(ui, "staged_queue_list", None)
        if queue is None or not hasattr(queue, "selectedItems"):
            queue = getattr(ui, "sample_queue_list", None)
        if queue is None or not hasattr(queue, "selectedItems"):
            return
        selected_items = queue.selectedItems()
        if not selected_items:
            self._append_ui_log("[QUEUE] No sample selected to revert.")
            return

        from qtpy.QtCore import Qt

        # Resolve selected sample_ids first (so the committed-confirm can
        # report exactly what it's about to revert).
        sample_ids = []
        for item in selected_items:
            sample_id = ""
            stored = item.data(Qt.UserRole) if hasattr(item, "data") else None
            if stored and str(stored) in self.sample_records:
                sample_id = str(stored)
            else:
                # Fallback: text-based parsing.
                import re
                text = item.text() if hasattr(item, "text") else str(item)
                if " | " in text:
                    sample_id = text.split(" | ", 1)[0].strip()
                else:
                    parts = re.split(r"\s{2,}", text.strip())
                    if len(parts) >= 3:
                        view = parts[0].strip().upper()
                        idx_match = re.search(r"(\d+)", parts[1].strip())
                        if idx_match:
                            patch = parts[2].strip()
                            sample_id = f"{patch}_{view}_{idx_match.group(1)}"
            if sample_id and sample_id in self.sample_records:
                sample_ids.append(sample_id)

        if not sample_ids:
            self._append_ui_log("[QUEUE] No matching samples to revert.")
            return

        # Partition by status.
        committed_ids = [
            sid for sid in sample_ids
            if self.sample_records.get(sid, {}).get("status") == "committed"
        ]
        revertable_ids = [
            sid for sid in sample_ids
            if self.sample_records.get(sid, {}).get("status") in ("accepted", "staged")
        ]
        already_editable = [
            sid for sid in sample_ids
            if self.sample_records.get(sid, {}).get("status") in ("predicted", "curating")
        ]

        # Committed slices: confirm before reverting (provenance weight).
        if committed_ids:
            if not self._confirm_revert_committed(committed_ids):
                self._append_ui_log(
                    f"[QUEUE] Revert cancelled for {len(committed_ids)} "
                    "committed slice(s)."
                )
                committed_ids = []  # user declined — skip these

        changed = 0
        for sid in revertable_ids + committed_ids:
            record = self.sample_records.get(sid)
            if not record:
                continue
            prev = record.get("status", "")
            self._touch_sample(sid, status="curating", view=record.get("view"))
            print(f"[REVERT] {prev} -> curating: {sid!r}")
            changed += 1

        self._sync_data_to_add_list()
        self._refresh_sample_queue()
        self._refresh_round_status()
        # Keep the annotator's soft-lock visuals in sync with the new statuses.
        self._refresh_annotator_edit_state_visuals()
        if changed:
            self._append_ui_log(
                f"[QUEUE] Reverted {changed} slice(s) to editable (curating)."
            )
        elif already_editable:
            self._append_ui_log(
                f"[QUEUE] {len(already_editable)} slice(s) already editable — nothing to revert."
            )
        else:
            self._append_ui_log("[QUEUE] No slices were reverted.")

    def _confirm_revert_committed(self, committed_ids):
        """Themed confirm dialog for reverting committed (already-trained)
        slices.  Returns True if the user confirms, False otherwise.

        Falls back to True (proceed) in headless/test contexts where Qt
        dialogs aren't available — the sync-training path used by tests
        sets `_use_sync_training`, and a blocking modal would hang."""
        if getattr(self, "_use_sync_training", False):
            # Headless/test — don't block on a modal.
            return True
        try:
            from qtpy.QtWidgets import QMessageBox
            n = len(committed_ids)
            box = QMessageBox(getattr(self, "ui", None))
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Revert committed slice(s)?")
            box.setText(
                f"{n} selected slice(s) are committed — they were already "
                "used to train a model round."
            )
            box.setInformativeText(
                "Reverting reopens them for editing. The model already "
                "trained on the prior version; your edits will only affect "
                "future rounds. Continue?"
            )
            box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            box.setDefaultButton(QMessageBox.No)
            return box.exec_() == QMessageBox.Yes
        except Exception as exc:
            # If the dialog can't be shown, err on the SAFE side: do NOT
            # revert committed slices silently.
            print(f"[REVERT] committed-confirm dialog failed "
                  f"({type(exc).__name__}: {exc}) — declining to revert")
            return False

    @staticmethod
    def _push_signals_blocked(widget):
        """Block *widget*'s signals and return the PRIOR state.

        Returns None when the widget can't be blocked at all.  Pair with
        `_pop_signals_blocked` in a `finally`.  Release-hardening
        2026-09-07 fix pass: the display-only refreshes used to
        hard-unblock (`blockSignals(False)`), which silently cleared an
        OUTER guard's block and let the rest of that guard's setters fire
        their change handlers.
        """
        if widget is None or not hasattr(widget, "blockSignals"):
            return None
        prev = None
        if hasattr(widget, "signalsBlocked"):
            try:
                prev = bool(widget.signalsBlocked())
            except Exception:
                prev = None
        try:
            was = widget.blockSignals(True)
        except Exception:
            return None
        if prev is None:
            # Qt's blockSignals returns the previous state; a fake that
            # returns None falls back to "was not blocked".
            prev = bool(was) if isinstance(was, bool) else False
        return prev

    @staticmethod
    def _pop_signals_blocked(widget, prev):
        """Restore what `_push_signals_blocked` reported."""
        if widget is None or prev is None:
            return
        try:
            widget.blockSignals(bool(prev))
        except Exception:
            pass

    @staticmethod
    def _coerce_training_mode_by_bucket(value, fallback="cumulative"):
        """Normalize a persisted ``training_mode_by_bucket`` blob.

        Corrupt-tolerant: a missing key, a non-dict, or a junk value falls
        back to ``fallback`` (which is the legacy single-valued
        ``training_mode``, so old state files restore unchanged).
        """
        fallback = "from_scratch" if str(fallback) == "from_scratch" else "cumulative"
        src = value if isinstance(value, dict) else {}
        out = {}
        for bucket in ("lateral", "axial"):
            raw = src.get(bucket)
            out[bucket] = "from_scratch" if raw == "from_scratch" else (
                "cumulative" if raw == "cumulative" else fallback
            )
        return out

    def _training_mode_for_bucket(self, bucket=None):
        """The stored training mode for ``bucket`` (default: the active one)."""
        if bucket is None:
            bucket = self._current_round_key()
        by_bucket = getattr(self, "training_mode_by_bucket", None)
        if not isinstance(by_bucket, dict):
            by_bucket = self._coerce_training_mode_by_bucket(
                None, fallback=getattr(self, "training_mode", "cumulative")
            )
            self.training_mode_by_bucket = by_bucket
        return by_bucket.get(bucket, getattr(self, "training_mode", "cumulative"))

    def _apply_training_mode_for_active_bucket(self):
        """Push the ACTIVE bucket's stored training mode into the combo.

        Release-hardening 2026-09-07 (bucket-switch item #5): training mode
        is per bucket, so switching XZ<->XY must show that bucket's own
        choice.  Signals are blocked while the combo is repopulated so this
        display-only refresh can't re-enter ``_on_training_mode_changed``
        and overwrite the OTHER bucket's stored value (the restore-clobber
        guard pattern used elsewhere in this controller).
        """
        mode = self._training_mode_for_bucket()
        self.training_mode = mode
        ui = getattr(self, "ui", None)
        combo = getattr(ui, "training_mode_combo", None) if ui is not None else None
        if combo is None or not hasattr(combo, "count"):
            return
        want_scratch = mode == "from_scratch"
        target_index = None
        for i in range(combo.count()):
            is_scratch = "scratch" in str(combo.itemText(i)).lower()
            if is_scratch == want_scratch:
                target_index = i
                break
        if target_index is None:
            return
        # Restore the PRIOR block state rather than force-unblocking: an
        # outer guard (a restore, another display-only refresh) may already
        # hold this widget blocked, and clearing it mid-guard would let the
        # guard's own remaining setters fire their handlers.  Same
        # prev-value pattern as `_init_transparency_loading`.
        _prev = self._push_signals_blocked(combo)
        try:
            combo.setCurrentIndex(target_index)
        finally:
            self._pop_signals_blocked(combo, _prev)

    def _on_training_mode_changed(self, text):
        mode_text = str(text or "").lower()
        if "scratch" in mode_text:
            self.training_mode = "from_scratch"
        else:
            self.training_mode = "cumulative"
        # Record it against the ACTIVE bucket -- lateral and axial keep
        # their own choice (release-hardening 2026-09-07, item #5).
        if not isinstance(getattr(self, "training_mode_by_bucket", None), dict):
            self.training_mode_by_bucket = self._coerce_training_mode_by_bucket(
                None, fallback=self.training_mode
            )
        bucket = self._current_round_key()
        self.training_mode_by_bucket[bucket] = self.training_mode
        self._append_ui_log(
            f"[TRAIN] Training mode for {bucket} set to: {self.training_mode}"
        )
        try:
            self._refresh_init_hint()
        except Exception:
            pass

    def _on_view_changed(self, *_args):
        # The two view radios are both wired to `toggled`, which fires on the
        # one going OFF as well as the one coming ON -- so one click ran this
        # whole handler TWICE (measured 2026-09-07: ~100 ms of it, doubled).
        # A radio going off is never a view change: the partner's `toggled`
        # carries it.  Combo signals pass a str, so only a bool False is
        # treated as "unchecked".
        if len(_args) == 1 and isinstance(_args[0], bool) and not _args[0]:
            return
        self._ensure_runtime_defaults()
        ui = getattr(self, "ui", None)
        if ui is None:
            return
        target, view = self._read_view_selection_from_ui()
        view_changed = view != self.selected_view
        target_changed = target != self.view_target
        self.view_target = target
        self.selected_view = view
        self._set_axial_combo_enabled()

        if target_changed or view_changed:
            self._append_ui_log(f"[VIEW] Active target set to {self._format_target_label()}.")
        self._apply_curation_checkpoint_for_view(view_name=view, emit_log=True)
        self.loaded_patch_axis_view = view
        self._refresh_inference_channel_options()
        self.apply_active_view_to_viewer()
        self._refresh_round_status()

        if target_changed or view_changed:
            # Update work_labels visibility for the new view target.
            if self.loaded_patch_name:
                self._set_work_labels_visibility_for_context(
                    patch_name=self.loaded_patch_name,
                    target_view=self.view_target,
                )
            # Refresh slice list and staged data list for the new view.
            self._refresh_sample_queue()
            self._sync_data_to_add_list()
            # Update checkpoint and model name for the new view's round.
            self._update_training_ui_for_view()
            # If annotator is open, update its context for the new view.
            try:
                from napari_epicycle.annotator.launcher import _find_dock_by_title
                dock = _find_dock_by_title(self.viewer, "EpiCycle 2D Training Annotator")
                if dock is not None and dock.isVisible():
                    widget = dock.widget() if hasattr(dock, "widget") else dock
                    image_name = self._resolve_native_image_layer_name()
                    labels_name = self._work_labels_layer_name_for_view(view_name=view)
                    if hasattr(widget, "set_context"):
                        widget.set_context(
                            image_layer_name=image_name,
                            labels_layer_name=labels_name,
                            auto_load=True,
                            target_view=self.view_target,
                            selected_view=view,
                        )
            except Exception:
                pass

    _MODEL_SOURCE_LABELS = {
        "auto": "Auto (Recommended)",
        "round": "Round model",
        "imported": "External model file",
        "pretrained": "Cellpose pretrained",
        "none": "None (Manual/SAM only)",
    }

    def _coerce_legacy_model_state(self):
        """Map persisted values whose combo items were removed.

        ``"none"`` (Manual/SAM only) and ``"scratch"`` (Cellpose pretrained
        combo) are no longer offered; ``setCurrentText`` of a missing item is
        a silent no-op, so the combo would disagree with the state.
        """
        if str(getattr(self, "curation_model_source_mode", "")) == "none":
            print("[STEP2-MODEL] legacy source mode 'none' -> 'auto'")
            self.curation_model_source_mode = "auto"
        if str(getattr(self, "curation_pretrained_model_type", "")) == "scratch":
            print("[STEP2-MODEL] legacy pretrained type 'scratch' -> 'cyto3'")
            self.curation_pretrained_model_type = "cyto3"

    def _should_heal_imported_slot(self):
        """True when the External-model-file slot should be cleared on load.

        Two conditions.  (a) the stored path IS one of this run's own active
        round checkpoints (`_is_leaked_round_checkpoint`) -- only the
        pre-2026-09-07 promote-leak put it there.  (b) it is not a
        deliberate external pick: release-hardening 2026-09-07 fix pass:
        if the user explicitly chose "External model file" AND browsed
        to a file OUTSIDE this workspace, that is their choice even when the
        tail segments happen to match, so leave it alone.
        """
        path = str(getattr(self, "imported_curation_model_path", "") or "").strip()
        if not self._is_leaked_round_checkpoint(
            path, getattr(self, "active_curation_checkpoints", None)
        ):
            return False
        mode = str(getattr(self, "curation_model_source_mode", "") or "").lower()
        if mode not in ("imported", "external"):
            # The leak wrote the path while the mode was auto/round -- the
            # user never picked External, so there is nothing deliberate to
            # protect.
            return True
        ws = str(getattr(self, "working_dir", "") or "")
        if not ws:
            return True
        try:
            inside = Path(path).resolve().is_relative_to(Path(ws).resolve())
        except (OSError, ValueError):
            return True
        return bool(inside)

    @staticmethod
    def _is_leaked_round_checkpoint(imported_path, active_checkpoints):
        """True when the External-model-file slot holds one of THIS run's own
        active round checkpoints.

        Only the pre-2026-09-07 promote-leak (bucket-switch item #1) could
        have put it there -- the user browses to a file OUTSIDE the run when
        they pick an external model.  Anything else, including another run's
        model, is treated as a genuine pick and left alone.
        """
        path = str(imported_path or "").strip()
        if not path or not isinstance(active_checkpoints, dict):
            return False

        def _tail(value):
            """Last 6 path segments, case-folded, "/"-joined.

            `active_curation_checkpoints` is stored RELATIVE to the
            workspace while the leaked external path was stored ABSOLUTE,
            so a plain string compare misses as soon as the workspace is
            copied or moved.  6 segments is exactly
            artifacts/step2/runs/<run>/<bucket>/round_NN/model/<file>
            minus the leading "artifacts/step2" -- specific enough that
            nothing outside a Step 2 round model can collide.
            """
            parts = [
                seg for seg in str(value or "").replace("\\", "/").split("/")
                if seg not in ("", ".")
            ]
            if len(parts) < 6:
                return None
            return "/".join(seg.lower() for seg in parts[-6:])

        mine = _tail(path)
        if mine is None:
            return False
        return any(_tail(v) == mine for v in active_checkpoints.values() if v)

    def _refresh_model_widgets_from_state(self):
        """Push the stored model-source / External-model-file / Pretrained
        values back into their widgets.  Display only, except that a legacy
        ``none`` / ``scratch`` value is first coerced in memory by
        ``_coerce_legacy_model_state`` (those items no longer exist).

        Every setter is wrapped in ``blockSignals`` so showing a value can
        never re-enter ``_on_model_source_changed`` (which would read the
        widgets back into state and schedule a save of what it just drew --
        the 2026-05-29 restore-clobber class).
        """
        ui = getattr(self, "ui", None)
        if ui is None:
            return

        def _quiet(widget, setter):
            if widget is None:
                return
            # Restore the PRIOR block state -- see
            # `_push_signals_blocked`.
            prev = self._push_signals_blocked(widget)
            try:
                setter(widget)
            except Exception:
                pass
            finally:
                self._pop_signals_blocked(widget, prev)

        self._coerce_legacy_model_state()
        mode = str(getattr(self, "curation_model_source_mode", "auto") or "auto")
        label = self._MODEL_SOURCE_LABELS.get(mode, "Auto (Recommended)")
        combo = getattr(ui, "model_source_combo", None)
        if combo is not None and hasattr(combo, "setCurrentText"):
            _quiet(combo, lambda w: w.setCurrentText(label))

        path_edit = getattr(ui, "imported_model_path", None)
        if path_edit is not None and hasattr(path_edit, "setText"):
            _quiet(
                path_edit,
                lambda w: w.setText(
                    str(getattr(self, "imported_curation_model_path", "") or "")
                ),
            )

        pre_combo = getattr(ui, "curation_pretrained_combo", None)
        if pre_combo is not None and hasattr(pre_combo, "setCurrentText"):
            _quiet(
                pre_combo,
                lambda w: w.setCurrentText(
                    str(getattr(self, "curation_pretrained_model_type", "") or "cyto3")
                ),
            )

    def _auto_reload_saved_patch(self):
        """Re-load the patch recorded in state (``loaded_stack_path`` +
        ``loaded_patch_name``), if its file is still there.

        Shared by ``_switch_active_run`` and ``_browse_workdir`` — a run
        switch and a workspace open are both "restore this snapshot", so
        they must restore the patch the same way (release-hardening
        2026-09-07, bucket-switch item #4; opening a workspace used to
        leave the viewer empty).

        No-op when nothing is saved or the file is gone; logs why.
        """
        new_stack = self._effective_stack_path()
        new_patch = str(getattr(self, "loaded_patch_name", "") or "")
        if not new_stack and new_patch:
            # The recorded folder is gone and Step 1 has no patches to fall
            # back to, so there is nothing to reload from -- say so rather
            # than failing silently (the "not found at" line below covers the
            # case where the FOLDER is fine but the patch file is not).
            recorded = str(getattr(self, "loaded_stack_path", "") or "")
            if recorded:
                msg = (
                    f"Saved patch '{new_patch}' not found at "
                    f"'{recorded}' — the patch folder is no longer there, "
                    f"and Project Setup has no patches to fall back to.  "
                    f"Use Browse... to point at one."
                )
                try:
                    self._append_ui_log(f"[STEP2-RUN] {msg}")
                except Exception:      # noqa: BLE001
                    pass
        if new_stack and new_patch:
            try:
                from pathlib import Path as _Path
                # Audit-fix 2026-05-24: loaded_stack_path is the PARENT
                # DIRECTORY of the patches (set at line ~2405:
                # `self.loaded_stack_path = stack_dir`).  Previously
                # we checked `is_file()` which always returned False
                # for a directory → "stack file missing" → no
                # auto-reload.  Now: derive the actual TIFF path from
                # parent dir + patch name + ".tiff" and verify THAT
                # exists.  Falls back to dir-existence check if the
                # specific TIFF can't be resolved (legacy state JSON
                # may have stack_path pointing at the TIFF directly).
                stack_path = _Path(new_stack)
                patch_tiff = stack_path / f"{new_patch}.tiff"
                patch_tif = stack_path / f"{new_patch}.tif"
                if stack_path.is_file():
                    # Legacy: stack_path WAS the patch file directly
                    patch_file_ok = True
                    target_label = stack_path.name
                elif patch_tiff.is_file():
                    patch_file_ok = True
                    target_label = patch_tiff.name
                elif patch_tif.is_file():
                    patch_file_ok = True
                    target_label = patch_tif.name
                elif stack_path.is_dir():
                    # Dir exists but neither .tiff nor .tif found —
                    # the specific patch was deleted but dir is OK.
                    # Try the load anyway; _load_patch will report
                    # the missing file properly.
                    patch_file_ok = True
                    target_label = f"<from dir {stack_path.name}>"
                else:
                    patch_file_ok = False
                    target_label = None
                if patch_file_ok:
                    # Audit-fix 2026-05-25 (J CRITICAL): _load_patch
                    # reads from `train_patch_list.selectedItems()`,
                    # NOT from `self.loaded_patch_name`.  If no row
                    # is selected, it silently no-ops + prints
                    # "Please select a patch to load."  So we must
                    # programmatically select the row first.
                    selected_ok = self._select_patch_item(new_patch)
                    if not selected_ok:
                        msg = (
                            f"Auto-reload: saved patch '{new_patch}' "
                            f"not in the Patches list — likely the "
                            f"list needs Browse... to populate.  Pick "
                            f"a patch manually to continue."
                        )
                        try:
                            self._append_ui_log(f"[STEP2-RUN] {msg}")
                        except Exception:
                            pass
                    else:
                        self._load_patch()
                else:
                    msg = (
                        f"Saved patch '{new_patch}' not found at "
                        f"'{new_stack}' — file may have been moved or "
                        f"deleted.  Pick a patch manually to continue."
                    )
                    try:
                        self._append_ui_log(f"[STEP2-RUN] {msg}")
                    except Exception:
                        pass
            except Exception as e:
                pass
        else:
            pass

    def _update_training_ui_for_view(self):
        """Update Training Setup UI fields when switching lateral/axial view.

        Shows the prior checkpoint and model name for the NEW view's round
        context. Read-only — does not modify any training state or data.
        """
        ui = getattr(self, "ui", None)
        if ui is None:
            return
        # Checkpoint: show prior round's checkpoint for this view (if any).
        # 2026-05-29 UX fix: guard the setText so the auto-populate (system,
        # not user) doesn't fire _on_init_widget_changed -> auto-flip Training
        # Mode to From Scratch.  The docstring above says this method is
        # "read-only — does not modify any training state or data," which the
        # signal cascade was silently violating.
        prior_ckpts = self._resolve_prior_model_checkpoint()
        ckpt_edit = getattr(ui, "starting_checkpoint_path", None)
        if ckpt_edit is not None and hasattr(ckpt_edit, "setText"):
            _prev_guard = getattr(self, "_init_transparency_loading", False)
            self._init_transparency_loading = True
            try:
                if prior_ckpts:
                    latest = next(iter(prior_ckpts.values()), "")
                    ckpt_edit.setText(str(latest))
                else:
                    ckpt_edit.setText("")  # no prior checkpoint for this view
            finally:
                self._init_transparency_loading = _prev_guard
            # _on_init_widget_changed was suppressed above — refresh the
            # init-hint label so it reflects the per-view widget value.
            try:
                self._refresh_init_hint()
            except Exception:
                pass
        # Release-hardening 2026-09-07 (bucket-switch item #6): the view
        # radio used to update ONLY the Checkpoint field + loss filter, so
        # the Model-source picker, the External model file, the Pretrained
        # combo, the Training-mode combo and the status "Last checkpoint"
        # kept showing whatever the OTHER view had left behind -- the
        # user-reported "some UI fields don't change".
        #
        # Everything below is DISPLAY-ONLY.  Signals are blocked around each
        # setter so re-showing a value can never write it back to state or
        # schedule a save (the restore-clobber guard pattern established
        # 2026-05-29).
        self._refresh_model_widgets_from_state()
        try:
            self._apply_training_mode_for_active_bucket()
        except Exception:
            pass
        try:
            self._safe_ui_set_text(
                "status_last_checkpoint_value", self._last_checkpoint_text()
            )
        except Exception:
            pass
        try:
            self._refresh_init_hint()
        except Exception:
            pass

        # Filter loss plot to show only this view's curves.
        loss_plot = getattr(ui, "loss_plot", None)
        if loss_plot is not None and hasattr(loss_plot, "set_view_filter"):
            loss_plot.set_view_filter(self._current_round_key())

    def _set_view_radio(self, view):
        ui = getattr(self, "ui", None)
        if ui is None:
            return
        view_name = str(view).upper()
        lateral_btn = getattr(ui, "get_view_lateral", None)
        axial_btn = getattr(ui, "get_view_axial", None)
        if lateral_btn is not None and axial_btn is not None:
            if view_name == "XY":
                lateral_btn.setChecked(True)
                self.view_target = "lateral"
                self.selected_view = "XY"
            else:
                axial_btn.setChecked(True)
                self.view_target = "axial"
                self.selected_view = "YZ" if view_name == "YZ" else "XZ"
            self._set_axial_combo_enabled()
            return
        if view_name == "XY" and getattr(ui, "get_view_XY", None) is not None:
            ui.get_view_XY.setChecked(True)
        elif view_name == "XZ" and getattr(ui, "get_view_XZ", None) is not None:
            ui.get_view_XZ.setChecked(True)
        elif view_name == "YZ" and getattr(ui, "get_view_YZ", None) is not None:
            ui.get_view_YZ.setChecked(True)

    def _select_patch_item(self, patch_id):
        ui = getattr(self, "ui", None)
        if ui is None:
            return False
        patch_list = getattr(ui, "train_patch_list", None)
        if patch_list is None or not hasattr(patch_list, "count") or not hasattr(patch_list, "item"):
            return False

        target_index = -1
        patch_key = str(patch_id).strip()
        for i in range(patch_list.count()):
            item = patch_list.item(i)
            if item is None or not hasattr(item, "text"):
                continue
            fname = self._clean_patch_filename(item)
            stem, _ = os.path.splitext(fname)
            if stem == patch_key or fname == patch_key or patch_key in stem:
                target_index = i
                break

        if target_index < 0:
            return False
        if hasattr(patch_list, "setCurrentRow"):
            patch_list.setCurrentRow(target_index)
        elif hasattr(patch_list, "setCurrentItem"):
            patch_list.setCurrentItem(patch_list.item(target_index))
        return True

    def _parse_queue_selected_sample_id(self):
        """Extract sample_id from the selected slice list item.

        Preferred path: read the canonical ``sample_id`` stored via
        ``Qt.UserRole`` on each item.  Falls back to text parsing.
        """
        ui = getattr(self, "ui", None)
        if ui is None:
            return ""
        queue = getattr(ui, "sample_queue_list", None)
        if queue is None or not hasattr(queue, "selectedItems"):
            return ""
        selected_items = queue.selectedItems()
        if not selected_items:
            return ""
        item = selected_items[0]

        # Preferred: read sample_id from item data.
        from qtpy.QtCore import Qt
        stored = item.data(Qt.UserRole) if hasattr(item, "data") else None
        if stored and str(stored) in self.sample_records:
            return str(stored)

        # Fallback: text-based parsing.
        text = item.text() if hasattr(item, "text") else str(item)
        stripped = text.strip()
        if stripped in self.sample_records:
            return stripped
        if " | " in text:
            return text.split(" | ", 1)[0].strip()

        import re
        parts = re.split(r"\s{2,}", text.strip())
        if len(parts) >= 3:
            view = parts[0].strip().upper()
            idx_match = re.search(r"(\d+)", parts[1].strip())
            if idx_match:
                patch = parts[2].strip()
                candidate = f"{patch}_{view}_{idx_match.group(1)}"
                if candidate in self.sample_records:
                    return candidate
        return ""

    def _refresh_curation_stage_line(self):
        """Hook: the Curation foot's stage line.
        Presentation only; a failing hook never breaks the controller."""
        fn = getattr(getattr(self, "ui", None), "refresh_stage_line", None)
        if fn:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                print(f"[STEP2-UI] refresh_stage_line failed: {exc}")

    def _jump_to_selected_queue_item(self):
        sample_id = self._parse_queue_selected_sample_id()
        if not sample_id:
            self._refresh_curation_stage_line()
            return
        self.focus_sample(sample_id)
        self._refresh_curation_stage_line()

    def apply_active_view_to_viewer(self, slice_index=None, force_2d: bool = False):
        if not self.viewer:
            return
        dims = getattr(self.viewer, "dims", None)
        if dims is None:
            return
        if getattr(dims, "ndim", 0) < 3:
            return

        order_map = {
            "XY": (0, 1, 2),
            "XZ": (1, 0, 2),
            "YZ": (2, 0, 1),
        }
        slicing_axis_map = {"XY": 0, "XZ": 1, "YZ": 2}
        view_name = str(self.selected_view or "XY").upper()
        order = order_map.get(view_name, order_map["XY"])
        slicing_axis = slicing_axis_map.get(view_name, 0)

        if bool(force_2d):
            try:
                self.viewer.dims.ndisplay = 2
            except Exception:
                pass

        try:
            self.viewer.dims.order = order
        except Exception:
            pass

        if slice_index is None:
            return

        try:
            nsteps = int(self.viewer.dims.nsteps[slicing_axis])
            clamped = max(0, min(int(slice_index), nsteps - 1))
            self.viewer.dims.set_current_step(slicing_axis, clamped)
        except Exception:
            pass

    def focus_sample(self, sample_id):
        sample_key = str(sample_id).strip()
        if not sample_key:
            return

        record = self.sample_records.get(sample_key, {})
        view_name = str(record.get("view") or self._infer_view_from_label(sample_key) or "XY").upper()
        slice_index = record.get("slice_index", self._infer_slice_index(sample_key))
        patch_id = sample_key.rsplit("_", 2)[0] if sample_key.count("_") >= 2 else sample_key

        # 2026-05-25 audit-6 diagnostic: entry state.  in_seg_mgr=False on
        # an imported pair confirms the disk/memory gap.
        seg_mgr = getattr(self, "segmentation_manager", None)
        in_seg = bool(seg_mgr and sample_key in getattr(seg_mgr, "data_masks", {}))
        print(f"[MASK-LOAD] focus_sample ENTER sample_id={sample_key!r} "
              f"status={record.get('status')!r} view={view_name!r} "
              f"slice_index={slice_index!r} loaded_patch={self.loaded_patch_name!r} "
              f"target_patch={patch_id!r} in_seg_mgr={in_seg}")

        self._set_view_radio(view_name)
        self.selected_view = view_name
        self.loaded_patch_axis_view = view_name

        # Ensure the correct patch is selected/loaded when available.
        if self.loaded_patch_name != patch_id:
            selected = self._select_patch_item(patch_id)
            if selected and self._effective_stack_path():
                self._load_patch()
                print(f"[MASK-LOAD] _load_patch returned; "
                      f"_ensure_work_labels_layer NOT called here "
                      f"(handled by patch-select path).  loaded_patch={self.loaded_patch_name!r}")

        self.apply_active_view_to_viewer(slice_index=slice_index)
        print(f"[MASK-LOAD] focus_sample EXIT view={view_name!r} slice={slice_index!r}")
        self._append_ui_log(f"[QUEUE] Focused sample {sample_key} ({view_name}, slice={slice_index}).")

    # Each method calls into a manager and updates UI -- methods for pressable buttons
    def _browse_workdir(self):
        """
            Prompt the user to set a working directory, should set the path
            in QLineEdit field.
        """
        # select the directory
        dir_path = QFileDialog.getExistingDirectory(None, "Select Workspace Folder")

        if dir_path:
            # A workspace switch: drop everything the PREVIOUS workspace owns
            # before its successor is read (timers, annotator, layers,
            # segmenter).  See `_teardown_previous_workspace`.
            self._teardown_previous_workspace()

            self._set_workspace_path_quiet(dir_path)

            # Step 2 layout migration check -- prompts user if workspace is on
            # the legacy layout (pre-restructure).  Must happen BEFORE
            # _load_internal_state because migration renames hitl_state.json
            # -> step2_state.json.
            #
            # Browse-only: `adopt_workspace`
            # must never throw a migration modal at a user who merely walked
            # into the step, so these two calls sit between the two halves of
            # the quiet path rather than inside either of them.
            self._maybe_prompt_step2_migration(dir_path)

            # Phase C.3 (2026-05-25): split->flat layout migration check.
            # Runs AFTER the v2->v3 migration (above) so a workspace
            # needing both migrations chains correctly: legacy training_data/
            # -> v3 split layout -> v4 flat layout.
            self._maybe_prompt_split_to_flat_migration(dir_path)

            self._load_workspace_quiet()

            # Release-hardening 2026-09-07 (bucket-switch item #4): restore
            # the saved patch, exactly as a run switch does.  Pre-fix,
            # opening a workspace left the viewer empty and "Open Multiview
            # Workspace" failed with "missing image/labels context" until
            # the user re-picked the patch by hand -- while switching runs
            # inside the same workspace restored it.  Same snapshot
            # semantics, one shared helper.
            try:
                self._auto_reload_saved_patch()
            except Exception as e:
                print(f"[WORKSPACE-SWITCH] saved-patch auto-reload failed "
                      f"({type(e).__name__}: {e}); pick a patch manually")

            # User-feedback 2026-05-24: if no runs exist after load, prompt
            # the user to create their first run instead of silently
            # auto-creating "default" (the previous behavior).  User wanted
            # control over the first run's name + an explicit intro.
            #
            # The persist-path safety net (`_persist_curated_training_pair`
            # auto-bootstraps on UNKNOWN layout) remains as a fallback in
            # case the user dismisses the prompt + then triggers an action.
            self._maybe_prompt_first_run(dir_path)
            # An explicit Browse is the ONE thing that publishes a workspace
            # to the shell.
            self._publish_workspace_to_shell(dir_path)

    # ------------------------------------------------------------------
    # Shared workspace -- brick B2
    # ------------------------------------------------------------------

    def _teardown_previous_workspace(self):
        """Drop everything the workspace we are LEAVING still owns.

        Browse-only in spirit -- it is what makes a *switch* safe -- so a
        virgin adopt never calls it.  A re-adopt does (the shared workspace
        moved while this step held project A), because a re-adopt IS a
        switch.

        Three things, in this order:

        1. **Flush pending timers** (audit-found CRITICAL 2026-05-23).  A
           queued state-save could otherwise fire AFTER `working_dir` changes
           and write the OLD workspace's state into the NEW workspace's path.
           Same root cause as the switch-run timer race.
        2. **Close the annotator + clear its patch layers**
           (`fix/step2-workspace-switch-channels`, 3-agent trace 2026-06-03).
           Switching with the 2D annotator OPEN kept the old annotator
           mounted, with its stale layer wiring + channel context, still
           driving the newly-loaded workspace: a segment in the new workspace
           produced a mask that NEVER displayed.  `[FAST-RELOAD] FAIL: layer
           count mismatch` is that bug's fingerprint.  These helpers identify
           layers via `loaded_patch_name`, so they MUST run while the OLD
           context is intact -- i.e. before `_load_workspace_quiet` blanks it.
        3. **Drop the live segmenter** (CRITICAL 2026-06-03).  The
           segmenter-rebuild guard compares the resolved checkpoint +
           model_type against `current_config` -- but `_load_internal_state`
           REPLACES `current_config` wholesale with the new workspace's
           values, so by segment time that compare sees "no change" and never
           rebuilds, while the live CellposeSegmenter was built from the OLD
           workspace's config.  Net: the new workspace silently segments with
           the wrong model.  Nulling it forces `_run_segmentation`'s
           `if segmenter is None` to rebuild from the restored config.

           Only the segmenter, NOT the prediction caches
           (data_masks/data_inputs/...): those are keyed by the OLD patch's
           slice-labels, never collide with the new workspace's, and are
           repopulated from disk on the next patch load -- clearing them adds
           risk for no benefit.
        """
        try:
            if getattr(self, "_state_save_timer", None) is not None:
                self._flush_pending_state_timers()
        except Exception as e:
            print(f"[WARN] workspace-switch timer-flush failed "
                  f"(prior workspace state may not be saved): {e}")

        print(f"[WORKSPACE-SWITCH] tearing down previous workspace before "
              f"load: prev_patch={getattr(self, 'loaded_patch_name', None)!r}")
        try:
            self._close_step2_annotator_dock_safely()
            self._clear_patch_viewer_layers_safely()
            print("[WORKSPACE-SWITCH] annotator dock + stale patch layers "
                  "cleared")
        except Exception as e:
            print(f"[WARN] workspace-switch annotator/layer teardown "
                  f"failed (stale layers may linger): {e}")

        try:
            sm = getattr(self, "segmentation_manager", None)
            if sm is not None:
                sm.segmenter = None
                print("[WORKSPACE-SWITCH] segmenter reset (next segment "
                      "rebuilds from new workspace config)")
        except Exception as e:
            print(f"[WARN] workspace-switch segmenter reset failed: {e}")

    def _set_workspace_path_quiet(self, path):
        """Point this step at `path` -- the first half of the quiet load.

        Publish is suppressed for the duration: a path handed DOWN by the
        shell must never be pushed back UP as if the user had chosen it
        (2.5).
        """
        self.working_dir = path
        self._suppress_publish = True
        try:
            if hasattr(self.ui, "set_workdir_path") and hasattr(
                    self.ui.set_workdir_path, "setText"):
                self.ui.set_workdir_path.setText(path)
        finally:
            self._suppress_publish = False
        self._append_ui_log(f"[WORKSPACE] Using workspace: {path}")

    def _load_workspace_quiet(self):
        """Read the workspace and repaint the lists -- the second half.

        Shared verbatim by Browse and by `adopt_workspace`; the only thing
        between the two halves is Browse's migration prompts, which is why
        this is two methods and not one.
        """
        # Audit-fix 2026-05-24 (workspace-switch field leak): clear
        # in-memory fields BEFORE _load_internal_state.  Mirror of
        # the _switch_active_run pattern that was added 2026-05-23.
        # Without this, if the new workspace's state JSON is missing
        # any of these fields, the previous workspace's values would
        # leak into the new context.
        self.loaded_stack_path = ""
        self.loaded_patch_name = ""
        self.sample_records = {}
        self.training_data_by_round = {}
        self.losses_by_round = {}
        self.model_checkpoint_paths = {}
        self.round_backend_info = {}
        self.active_run_id = ""  # let load_internal_state's peek set this
        try:
            lbl = getattr(self.ui, "loaded_patch_label", None)
            if lbl is not None and hasattr(lbl, "setText"):
                lbl.setText("")
        except Exception:
            pass

        self._load_internal_state()  # should restore any saved HITL state
        self._refresh_round_status()
        self._refresh_sample_queue()
        self._refresh_patch_list_badges()
        # Step 2 look (brick A "Sweeps"): the UI sweeps the slice
        # pictures of samples no run has any more
        _o = getattr(getattr(self, "ui", None), "on_workspace_opened", None)
        if _o is not None:
            try:
                _o(self.working_dir or "", list(self.sample_records))
            except Exception as exc:      # noqa: BLE001 -- never break the load
                print(f"[STEP2-UI] on_workspace_opened failed: {exc}")
        try:
            self._refresh_run_picker()
        except Exception:
            pass

    def _adopt_workspace_quiet(self, path):
        """The two halves of the quiet load, with nothing in between."""
        self._set_workspace_path_quiet(path)
        self._load_workspace_quiet()

    @staticmethod
    def _same_workspace(a, b):
        """Path comparison for the shared-workspace handshake (case + slash)."""
        if not a or not b:
            return False
        try:
            return (os.path.normcase(os.path.normpath(a))
                    == os.path.normcase(os.path.normpath(b)))
        except Exception:      # noqa: BLE001
            return a == b

    def adopt_workspace(self, path, previous=""):
        """Quietly take the shell's shared workspace (2.3, 2.6).

        Called when the user enters Step 2 with no workspace of its own, or
        when the shared workspace moves because they browsed in ANOTHER step.
        It does the quiet half of Browse and nothing else: no migration
        modal, no first-run modal, no saved-patch auto-load, no publish.  The
        two things Browse would have SAID (old folder layout / no run yet)
        are said by the banner's second line instead (2b).

        `previous` is the workspace this step INHERITED before.  The path is
        replaced only when the box is empty or still holds exactly that
        folder; a folder the user browsed to HERE is never overwritten.  A replacement IS a workspace switch, so it runs the
        same teardown Browse does -- otherwise the old project's annotator
        keeps driving the new project's data.

        Runs under the patch busy bar: `_load_internal_state` re-reads every
        persisted training pair from disk, which is not free.
        """
        if not path:
            return
        current = (getattr(self, "working_dir", "") or "").strip()
        replacing = False
        if current:
            if not previous or not self._same_workspace(current, previous):
                return
            if self._same_workspace(current, path):
                return
            replacing = True

        if replacing and self._training_in_flight():
            # A re-adopt is a workspace switch the SHELL drives, without
            # asking anyone -- and it tears down the annotator and loads
            # another workspace.  `_switch_run_guard.check_switch_safety`
            # HARD-BLOCKS a run switch while training runs, for the same
            # reason: a half-trained model file.  Decline, and let the step
            # stay where it is (the shell then shows it as diverged).
            self._append_ui_log(
                "[WORKSPACE] Training is running, so this step stayed on its "
                "current workspace.  Switch it with Browse once training "
                "has finished.")
            return

        self._begin_busy(f"[WORKSPACE] Opening workspace: {path}")
        try:
            if replacing:
                self._teardown_previous_workspace()
            self._adopt_workspace_quiet(path)
            if replacing:
                # Journey (second B2 audit): the previous project's patch and
                # annotator have just been closed.  That is worth one plain
                # line where the user is already reading -- not a dialog.
                self._append_ui_log(
                    f"[WORKSPACE] Switched to workspace "
                    f"'{os.path.basename(os.path.normpath(path))}'; the "
                    f"previous patch and annotator were closed.")
        finally:
            self._end_busy()

    def _training_in_flight(self) -> bool:
        """True while a training run owns this workspace.

        Mirrors `_switch_run_guard.check_switch_safety`'s first blocker,
        including its "indeterminate means blocked" stance, and also covers
        the synchronous path (`_sync_training_in_flight`), which has no
        thread object to ask.
        """
        if getattr(self, "_sync_training_in_flight", False):
            return True
        thread = getattr(self, "_training_thread", None)
        if thread is None:
            return False
        try:
            return bool(thread.isRunning())
        except Exception:      # noqa: BLE001
            return True        # cannot prove it is safe

    def _publish_workspace_to_shell(self, path: str) -> None:
        """Tell the shell about a workspace the USER chose here.

        Called from the Browse handler only — never from a `textChanged`
        chain, so a future adopt cannot masquerade as a user choice.
        """
        if getattr(self, "_suppress_publish", False):
            return
        shell = getattr(self, "_workflow_shell", None)
        setter = getattr(shell, "set_workspace", None)
        if not callable(setter):
            return
        try:
            setter(path, source_step="roi_hitl_2d")
        except Exception as exc:      # noqa: BLE001
            print(f"[WORKSPACE] publish from Step 2 failed: {exc}")

    def _maybe_prompt_first_run(self, workspace_path):
        """Prompt user to create their first run if none exist (Phase 8 UX
        polish, user-reported 2026-05-24).

        Fires only when:
        * Workspace is loaded
        * Layout is UNKNOWN or NEW (not LEGACY/MIXED — those need migration)
        * Zero runs exist in `runs/` dir

        User can cancel — they can still click [+ New Run] later.  If they
        try to take an action that requires an active run (e.g., Accept),
        the persist-path safety net in `_persist_curated_training_pair`
        auto-bootstraps a 'default' run as a fallback.
        """
        try:
            from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
            from packages.epicycle_io.step2_run_manifest import list_run_ids
            ws = workspace_path
            layout_v = _step2_paths.detect_layout_version(ws)
            if layout_v not in (
                _step2_paths.LAYOUT_VERSION_UNKNOWN,
                _step2_paths.LAYOUT_VERSION_NEW,
            ):
                return  # LEGACY/MIXED → migration prompt handled it
            runs_root = _step2_paths.step2_runs_root(ws)
            existing = list_run_ids(runs_root) if runs_root.is_dir() else []
            if existing:
                return  # Some runs already exist

            # Open the New Run dialog with a friendly preamble.  Re-use the
            # existing _on_new_run_button_clicked code path (it builds the
            # dialog + handles the create + activate + refresh).
            # Show a brief intro dialog explaining why we're asking, then
            # open New Run.
            self._show_first_run_intro_dialog(ws)
            # Open the New Run dialog.  If user cancels, persist-path
            # safety net catches them later.
            self._on_new_run_button_clicked()
        except Exception as e:
            pass

    def _show_first_run_intro_dialog(self, workspace_path):
        """Brief themed dialog explaining why the New Run dialog is opening.

        OK-only — just gives the user context before the actual New Run
        dialog appears.  Don't bother with a Cancel here; the New Run
        dialog itself has Cancel.
        """
        try:
            from qtpy.QtWidgets import (
                QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
            )
            from napari_epicycle.ui_shell.theme import (
                _DIALOG_STYLE, _DIALOG_BTN_CONFIRM,
            )
            from pathlib import Path as _Path
            dlg = QDialog(getattr(self, "ui", None))
            dlg.setWindowTitle("Welcome — Create Your First Run")
            dlg.setStyleSheet(_DIALOG_STYLE)
            dlg.setMinimumWidth(460)
            layout = QVBoxLayout(dlg)
            layout.setContentsMargins(20, 16, 20, 16)
            layout.setSpacing(10)
            title = QLabel("Welcome — Create Your First Run", parent=dlg)
            title.setStyleSheet(
                "font-weight: bold; font-size: 14px; color: #d4a843;"
            )
            layout.addWidget(title)
            ws_name = _Path(str(workspace_path)).name or str(workspace_path)
            body = QLabel(
                f"This workspace ('{ws_name}') doesn't have any training "
                f"runs yet.\n\n"
                f"In Step 2, a 'run' is a training campaign — its own data, "
                f"channels, and model.  Most users start with one general "
                f"run (e.g., handle 'default' or 'pilot') and add more "
                f"later (e.g., for nuclei or membrane).\n\n"
                f"Click OK to open the New Run dialog and pick your first "
                f"run's name + init source.  You can cancel and use the "
                f"[+ New Run] button anytime.",
                parent=dlg,
            )
            body.setWordWrap(True)
            body.setStyleSheet("color: #cccccc; font-size: 11px;")
            layout.addWidget(body)
            btn_row = QHBoxLayout()
            btn_row.addStretch(1)
            ok_btn = QPushButton("OK — open New Run dialog", parent=dlg)
            ok_btn.setStyleSheet(_DIALOG_BTN_CONFIRM)
            ok_btn.setDefault(True)
            ok_btn.clicked.connect(dlg.accept)
            btn_row.addWidget(ok_btn)
            layout.addLayout(btn_row)
            dlg.exec_()
        except Exception as e:
            pass

    def _auto_bootstrap_default_run_if_fresh(self, workspace_path):
        """Auto-create a 'default' run on fresh workspaces.

        User-reported 2026-05-24: a brand-new workspace (no migration,
        no existing state) had New Run grayed out + no way to start
        Step 2 work.  Mirror the migration's "create default run"
        behavior for fresh workspaces so every workspace always has
        at least one active run.

        No-op if:
        * State JSON already exists (existing v3 workspace)
        * Legacy data exists (migration will handle it)
        * Any run dirs already exist on disk
        """
        try:
            from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
            from napari_epicycle.hitl_training_annotation._run_management import (
                create_run_on_disk, add_run_to_state_json,
            )
            ws = workspace_path
            # Skip if already has any Step 2 state
            if _step2_paths.is_new_layout_state(ws):
                return
            layout_v = _step2_paths.detect_layout_version(ws)
            if layout_v != _step2_paths.LAYOUT_VERSION_UNKNOWN:
                # LEGACY / MIXED → migration handles it; NEW → we already
                # have state.  Only UNKNOWN means truly fresh.
                return
            # Check if any run dirs already exist (defensive)
            existing = _step2_paths.list_run_ids if hasattr(
                _step2_paths, "list_run_ids"
            ) else None
            from packages.epicycle_io.step2_run_manifest import list_run_ids
            runs_root = _step2_paths.step2_runs_root(ws)
            if runs_root.is_dir() and list_run_ids(runs_root):
                return  # Some run dirs already exist; don't clobber

            # Create the default run + activate it
            create_run_on_disk(
                ws, _step2_paths.DEFAULT_RUN_ID,
                display_name="Default",
                target="",
                description=(
                    "Auto-created on fresh workspace load.  Rename via "
                    "the Manage Runs panel, or create additional runs "
                    "via the [+ New Run] button."
                ),
            )
            add_run_to_state_json(
                ws, _step2_paths.DEFAULT_RUN_ID, activate=True,
            )
        except Exception as e:
            pass
            # Non-fatal — user can still manually click [+ New Run]
            # (it's enabled on fresh workspaces by the picker refresh)

    def _maybe_prompt_step2_migration(self, workspace_path):
        """Show the Step 2 directory restructure migration prompt if the
        workspace is on the legacy layout.  No-op if already migrated,
        empty, or if user has marked the workspace as skip-forever.

        See ``napari_epicycle.hitl_training_annotation._migrate_step2_dialog
        .prompt_and_migrate_if_needed`` for the full decision tree.
        """
        from pathlib import Path as _Path
        print(f"[STEP2-MIGRATE] checking workspace at {workspace_path}")
        try:
            from napari_epicycle.hitl_training_annotation._migrate_step2_dialog import (
                prompt_and_migrate_if_needed,
                UserChoice,
            )
        except ImportError as e:
            print(f"[STEP2-MIGRATE] migration module not importable: {e} (skipping)")
            return

        try:
            parent_widget = getattr(self.ui, "parent", lambda: None)()
            choice, result = prompt_and_migrate_if_needed(parent_widget, _Path(workspace_path))
            print(f"[STEP2-MIGRATE] choice={choice.value if hasattr(choice, 'value') else choice} "
                  f"result={'success' if result and result.success else 'none' if result is None else 'failed'}")
            if result is not None:
                if result.success:
                    print(f"[STEP2-MIGRATE] applied {len(result.completed_steps)} operations; "
                          f"backup at {result.backup_dir}")
                    for step in result.completed_steps[:5]:
                        print(f"[STEP2-MIGRATE]   - {step}")
                    if len(result.completed_steps) > 5:
                        print(f"[STEP2-MIGRATE]   ... and {len(result.completed_steps) - 5} more")
                else:
                    print(f"[STEP2-MIGRATE] FAILED: {result.error}")
                    print(f"[STEP2-MIGRATE]   failed_step={result.failed_step} "
                          f"auto_rolled_back={result.auto_rolled_back}")
        except Exception as e:
            print(f"[STEP2-MIGRATE] prompt crashed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    def _maybe_prompt_split_to_flat_migration(self, workspace_path):
        """Phase C.3 (2026-05-25): show the split→flat migration prompt
        if the workspace has legacy SPLIT layout artifacts.  No-op if
        already on flat layout, empty, or user marked skip-forever.

        See ``_migrate_split_to_flat_dialog.prompt_and_migrate_if_needed``
        for the full decision tree.

        Defensive: any error in the prompt or migration falls back to
        a console log + continues workspace load (the dual-path readers
        introduced in Phase B.2 mean the workspace still functions on
        the legacy split layout — migration is a hygiene cleanup, not
        a functional dependency).
        """
        from pathlib import Path as _Path
        print(f"[STEP2-SPLIT-TO-FLAT] checking workspace at {workspace_path}")
        try:
            from napari_epicycle.hitl_training_annotation._migrate_split_to_flat_dialog import (
                prompt_and_migrate_if_needed,
                UserChoice,
            )
        except ImportError as e:
            print(f"[STEP2-SPLIT-TO-FLAT] dialog module not importable: "
                  f"{e} (skipping)")
            return

        try:
            parent_widget = getattr(self.ui, "parent", lambda: None)()
            # Brick D: the migration copies the whole legacy round tree
            # synchronously.  Bar only (no wait cursor) — the call opens a
            # modal prompt first and a wait cursor over a prompt reads as a
            # hang.
            self._begin_busy(cursor=False)
            try:
                choice, result = prompt_and_migrate_if_needed(
                    parent_widget, _Path(workspace_path)
                )
            finally:
                self._end_busy()
            print(f"[STEP2-SPLIT-TO-FLAT] choice="
                  f"{choice.value if hasattr(choice, 'value') else choice} "
                  f"result="
                  f"{'success' if result and result.success else 'none' if result is None else 'failed'}")
            if result is not None:
                if result.success:
                    print(f"[STEP2-SPLIT-TO-FLAT] applied "
                          f"{len(result.completed_steps)} operations; "
                          f"backup at {result.backup_dir}")
                    for step in result.completed_steps[:5]:
                        print(f"[STEP2-SPLIT-TO-FLAT]   - {step}")
                    if len(result.completed_steps) > 5:
                        print(f"[STEP2-SPLIT-TO-FLAT]   ... and "
                              f"{len(result.completed_steps) - 5} more")
                else:
                    print(f"[STEP2-SPLIT-TO-FLAT] FAILED: {result.error}")
                    print(f"[STEP2-SPLIT-TO-FLAT]   failed_step="
                          f"{result.failed_step} "
                          f"auto_rolled_back={result.auto_rolled_back}")
        except Exception as e:
            print(f"[STEP2-SPLIT-TO-FLAT] prompt crashed: "
                  f"{type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    # Browse and set the directory containing the image stacks to be used
    def _browse_training_stack(self):
        """
            Prompt the user to select a directory to load
            the stacks to be used for the training. Training stack
            names should be shown in the QWidgetList.
        """
        stack_dir = QFileDialog.getExistingDirectory(None, "Select Stack Directory")
        if not stack_dir:
            return
        self.loaded_stack_path = stack_dir
        self._populate_patch_list_from_disk(notify=True)

    def _step1_default_patch_dir(self) -> str:
        """``{workspace}/artifacts/step1/patches`` when it holds >=1 TIFF.

        The Step 1 patch default (B3).  Returns
        "" when there is no workspace, the folder is missing, or Step 1 has
        not extracted anything yet -- there is deliberately NO
        workspace-root scan.  Never raises.
        """
        ws = (getattr(self, "working_dir", "") or "").strip()
        if not ws:
            return ""
        try:
            from napari_epicycle.data_preparation._paths import step1_patches_dir
            patches = step1_patches_dir(ws)
            if not patches.is_dir():
                return ""
            for entry in patches.iterdir():
                if entry.is_file() and entry.suffix.lower() in (".tif", ".tiff"):
                    return str(patches)
        except Exception:      # noqa: BLE001
            return ""
        return ""

    def _effective_stack_path(self) -> str:
        """The user's chosen patch folder, else the Step 1 default.

        ``self.loaded_stack_path`` keeps meaning "the folder the user
        picked", and stays the only thing the state file writes -- setting
        the attribute to the default would silently persist a fallback as a
        user choice, and a persisted fallback can never re-resolve.  Every
        read of the patch folder (the list scan, the load join, the saved-
        patch reload, the "is a folder set?" guards) goes through here.
        """
        chosen = str(getattr(self, "loaded_stack_path", "") or "").strip()
        # A choice that is no longer THERE must not win.  The absolute path
        # travels in the state file, so a workspace copied to another machine
        # -- or a folder since deleted -- would otherwise hand back a dead
        # path forever.  Step 3 re-resolves its folder on every load; this
        # makes the two steps behave the same way.  The choice itself is left
        # alone: re-attaching the drive brings the user's folder straight
        # back.
        if chosen:
            try:
                if os.path.isdir(chosen):
                    return chosen
            except Exception:      # noqa: BLE001 -- a bad path reads as gone
                pass
        return self._step1_default_patch_dir()

    def _set_patch_dir_qualifier(self, defaulted: bool) -> None:
        """Show/hide the faint "Using the patches from Project Setup" line.

        The box itself shows the effective folder either way; this only says
        where it came from.  Browse clears it (and wins, and persists).
        """
        label = getattr(getattr(self, "ui", None), "patch_dir_qualifier", None)
        if label is None or not hasattr(label, "setVisible"):
            return
        try:
            label.setVisible(bool(defaulted))
        except Exception:      # noqa: BLE001
            pass

    def _populate_patch_list_from_disk(self, *, notify: bool = False):
        """Re-scan ``self.loaded_stack_path`` and rebuild the patch list.

        2026-05-26 audit-J MEDIUM fix: factored out of _browse_training_stack
        so it can also be called from:
        - workspace load (so re-extracted patches show up without user
          clicking Browse again)
        - explicit Refresh button (Group J)
        - run switch (loaded_stack_path is workspace-level so the list
          is the same; the call is idempotent + cheap)

        Args:
            notify: if True, append a [STACKS] log line about the scan.
                Default False so silent refresh-on-load doesn't spam.

        Idempotent — safe to call multiple times.  Preserves
        ``_patch_shapes`` cache across calls (new entries added, existing
        kept).
        """
        stack_dir = self._effective_stack_path()
        if not stack_dir:
            self._set_patch_dir_qualifier(False)
            return
        if not os.path.isdir(stack_dir):
            self._set_patch_dir_qualifier(False)
            return

        # Mirror the path into the UI textbox (helpful when called from
        # workspace load — the user can see where their patches are).  When
        # it is the Step 1 default rather than a folder the user picked, a
        # faint line under the box says so (B3).
        defaulted = stack_dir != str(
            getattr(self, "loaded_stack_path", "") or "").strip()
        if hasattr(self.ui, "load_training_path_patch") and hasattr(
                self.ui.load_training_path_patch, "setText"):
            self.ui.load_training_path_patch.setText(stack_dir)
        self._set_patch_dir_qualifier(defaulted)
        if defaulted and not getattr(self, "_patch_default_logged", "") == stack_dir:
            self._patch_default_logged = stack_dir
            print(f"[WORKSPACE] Step 2 patch folder defaulted to {stack_dir}")

        # Snapshot the previously-selected patch so we can re-select it
        # after the rebuild (otherwise user's selection vanishes on
        # every refresh — bad UX).
        prev_selected = ""
        try:
            patch_list = getattr(self.ui, "train_patch_list", None)
            if patch_list is not None and hasattr(patch_list, "currentItem"):
                cur = patch_list.currentItem()
                if cur is not None and hasattr(cur, "data"):
                    from qtpy.QtCore import Qt as _Qt
                    val = cur.data(_Qt.UserRole)
                    if val:
                        prev_selected = str(val)
        except Exception:
            prev_selected = ""

        if hasattr(self.ui, "train_patch_list") and hasattr(
                self.ui.train_patch_list, "clear"):
            self.ui.train_patch_list.clear()

        if notify:
            print("Retrieving image stack file names")
        # Store clean filename in UserRole data so badge text doesn't pollute paths.
        # Also read TIFF shape metadata for progress bar denominators.
        if not hasattr(self, "_patch_shapes"):
            self._patch_shapes = {}
        n_added = 0
        try:
            entries = sorted(os.listdir(stack_dir))
        except OSError as e:
            print(f"[WARN] Patch list scan failed for {stack_dir!r}: {e}")
            return
        for fname in entries:
            if not fname.endswith((".tif", ".tiff", ".npy")):
                continue
            if hasattr(self.ui, "train_patch_list") and hasattr(
                    self.ui.train_patch_list, "addItem"):
                from qtpy.QtWidgets import QListWidgetItem
                from qtpy.QtCore import Qt
                item = QListWidgetItem(fname)
                item.setData(Qt.UserRole, fname)
                self.ui.train_patch_list.addItem(item)
                n_added += 1
            # Cache volume shape from TIFF metadata (no full load).
            patch_stem = os.path.splitext(fname)[0]
            if patch_stem not in self._patch_shapes:
                try:
                    import tifffile
                    fpath = os.path.join(stack_dir, fname)
                    with tifffile.TiffFile(fpath) as tif:
                        shape = tif.series[0].shape
                        # Shape may be (C, Z, Y, X) or (Z, Y, X) — take last 3.
                        if len(shape) >= 3:
                            self._patch_shapes[patch_stem] = shape[-3:]
                except Exception:
                    pass

        # Restore prior selection if the same patch is still present.
        if prev_selected:
            try:
                patch_list = getattr(self.ui, "train_patch_list", None)
                if patch_list is not None and hasattr(patch_list, "count"):
                    for row in range(patch_list.count()):
                        item = patch_list.item(row)
                        from qtpy.QtCore import Qt as _Qt
                        val = item.data(_Qt.UserRole) if item is not None else None
                        if str(val or "") == prev_selected:
                            patch_list.setCurrentItem(item)
                            break
            except Exception:
                pass

        if notify:
            self._append_ui_log(f"[STACKS] Loaded stack directory: {stack_dir}")
        # Refresh badges + progress bars after populating list.
        self._refresh_patch_list_badges()
        self._refresh_sample_queue()

    def _on_refresh_patch_list_clicked(self):
        """Manual refresh button — re-scans loaded_stack_path.

        2026-05-26 audit-J MEDIUM fix: lets the user pick up
        newly-extracted patches without re-Browsing.
        """
        if not self._effective_stack_path():
            return
        self._populate_patch_list_from_disk(notify=True)

    def _on_patch_list_clicked(self, item):
        """Handle click on a patch in the patch list.

        If the annotator is open, delegates to the canonical patch
        switch method (with suppress flags + deferred auto-load).
        If closed, loads image + work labels directly.
        """
        if not self._effective_stack_path():
            self._append_ui_log("[WARN] Set the stack directory first.")
            return
        if item is None:
            return

        # Resolve patch name from clicked item (use clean filename, not badge text).
        patch_text = self._clean_patch_filename(item)
        patch_name = os.path.splitext(str(patch_text).strip())[0]

        # Sweep orphan annotator widgets to prevent stale handlers.
        try:
            from napari_epicycle.annotator.launcher import (
                _sweep_orphan_annotator_widgets,
                _find_dock_by_title,
            )
            _sweep_orphan_annotator_widgets()
        except Exception as exc:
            _find_dock_by_title = None

        # If the annotator is open, use the canonical switch method
        # (with suppress flags to prevent subviewer thrashing).
        if _find_dock_by_title is not None:
            try:
                dock = _find_dock_by_title(self.viewer, "EpiCycle 2D Training Annotator")
                if dock is not None and dock.isVisible():
                    self._switch_patch_context_from_native(patch_name)
                    return
            except Exception as exc:
                pass

        # No annotator open — load directly.
        # Count viewer handlers to detect stale callbacks from previous annotator
        try:
            from napari_epicycle.annotator.curation_2d_widget import _count_viewer_handlers
        except Exception:
            pass
        # Brick D / journey #6: this is the one user-initiated load, so it is
        # the one that gets the worker + scan bar.  Everything that used to
        # follow `_load_patch()` here now runs in the finished slot, on the
        # main thread, only for a load that was not superseded.
        def _after_load():
            if not self.loaded_patch_name:
                return
            # Always create work labels for ALL views — show/hide based on mode.
            # This unifies lateral and axial handling (previously lateral only
            # created XY, axial created XZ+YZ but not XY — asymmetric).
            # Lazy: `_set_work_labels_visibility_for_context` creates exactly
            # the view(s) about to be shown; the other views are built on the
            # first switch to them.
            self._set_work_labels_visibility_for_context(
                patch_name=self.loaded_patch_name,
                target_view=self.view_target,
            )
            # Refresh slice list for the newly selected patch.
            self._refresh_sample_queue()

        self._load_patch_async(on_done=_after_load)

    # Load and view selected image stack
    def _load_patch(self, force_axis=None):
        """Load a patch from disk into the viewer (SYNCHRONOUS).

        Unchanged contract: on return, ``loaded_patch_name`` is the new patch
        and its layers are in the viewer.  The five in-annotator callers
        (patch-context switch, focus_sample, view-results, run-switch restore)
        continue straight into `_ensure_work_labels_layer` /
        `apply_active_view_to_viewer`, so they need it to stay synchronous.

        The user-initiated patch click goes through
        :meth:`_load_patch_async` instead.
        """
        ctx = self._load_patch_prelude(force_axis=force_axis)
        if ctx is None:
            return
        try:
            payload = self._read_patch_payload(ctx["file_path"])
        except Exception as e:
            self._restore_patch_name_after_abandoned_load(ctx)
            print(f"Failed to load image: {e}")
            self._append_ui_log(
                f"[PATCH FAILED] {ctx['patch_hint']}: "
                f"{self._training_error_headline(e)}"
            )
            self._show_modal_error(
                "Could not load patch", e,
                f"Patch: {ctx['file_path']}",
            )
            return
        self._apply_loaded_patch(ctx, payload)

    def _load_patch_prelude(self, force_axis=None):
        """Main-thread work that MUST happen before the read starts.

        Drains the curating debounce while the OLD patch's layers are still
        loaded, exits refine mode, tears the old layers down, resolves the
        target axis + file path, and claims a load token.  Returns the load
        context, or None when there is nothing to load.

        Removes all old HITL layers (image channels, work labels, RGB
        composites) before loading. No caching — always loads fresh
        from disk to avoid stale data.
        """
        # Brick A / journey #4 (2026-09-07): drain the 5 s curating debounce
        # FIRST, while the OLD patch's work-labels layers are still in the
        # viewer.  `_curating_persist_pending` carries (slice_label, view,
        # idx) with no patch identity and the flush resolves the layer via
        # `loaded_patch_name`, so a flush after the swap would write the NEW
        # patch's pixels into the OLD slice's training pair.
        try:
            timer = getattr(self, "_curating_persist_timer", None)
            if timer is not None and hasattr(timer, "isActive") and timer.isActive():
                timer.stop()
            self._flush_curating_persist_pending()
        except Exception as exc:
            print(f"[STEP2-WRITE] pre-switch curating flush failed: "
                  f"{type(exc).__name__}: {exc}")
            self._append_ui_log(
                "[STEP2] Warning: could not save your last strokes on the "
                "previous patch before switching.  Go back and check that "
                "slice."
            )
        # The flush leaves an entry in place when its layer cannot be resolved
        # (annotator closed -> the work-labels layers are gone).  Such an entry
        # would drain later against the NEW patch's layers — exactly the bug
        # this flush exists to prevent — so drop whatever is left.
        try:
            leftover = getattr(self, "_curating_persist_pending", None)
            if isinstance(leftover, dict) and leftover:
                print(f"[STEP2-WRITE] dropping {len(leftover)} unsaveable "
                      f"pending edit(s) on patch switch: "
                      f"{sorted(leftover.keys())}")
                self._append_ui_log(
                    f"[STEP2] Warning: {len(leftover)} slice edit(s) could not "
                    "be saved before switching patch — the layer they were "
                    "drawn on is gone.  Go back and check those slices."
                )
                leftover.clear()
        except Exception:
            pass

        # Exit SAM refine mode if active.
        self._exit_refine_mode()

        # Unconditional SAM cleanup — layers can persist even outside
        # refine mode (e.g., user left refine but layers remain).
        try:
            self.interactive_manager.cleanup_prompt_layers(self.viewer)
        except Exception:
            pass

        # Forget old patch layers from lifecycle manager BEFORE manual removal.
        # Uses forget (not clear) to avoid visibility cascades that bypass
        # the annotator's suppress flags.  Matches Step 3 pattern.
        llm = getattr(self, "layer_lifecycle_manager", None)
        if llm is not None:
            llm.forget_context("step2/patch")

        # Hide subviewer label layers BEFORE removing main viewer layers.
        # This prevents vispy access violations when transposed data views
        # become stale during main layer removal.
        try:
            from napari_epicycle.annotator.launcher import _find_dock_by_title
            dock = _find_dock_by_title(self.viewer, "EpiCycle 2D Training Annotator")
            if dock is not None:
                widget = dock.widget() if hasattr(dock, "widget") else dock
                if hasattr(widget, "_all_subviewer_label_layers"):
                    for sub_layer in widget._all_subviewer_label_layers():
                        try:
                            sub_layer.visible = False
                        except Exception:
                            pass
        except Exception:
            pass

        # Brick D (2026-09-07): every load claims a token.  A second patch
        # click SUPERSEDES the first -- the older worker's finish is discarded
        # in `_on_patch_load_finished`.  A synchronous `_load_patch` bumps it
        # too, so it also invalidates an async load still in flight.
        #
        # Lifecycle-audit fix (2026-09-07, patch-bar brick): the bump was
        # BELOW the removal loop.  `_pump_busy_ui()` inside that loop hands Qt
        # the main loop, which can deliver a PREVIOUS async load's queued
        # `finished` slot; with the old token still standing that result was
        # not stale, so it installed its layers into a viewer that was
        # half-torn-down -- and the rest of this method then cleared
        # `loaded_patch_name` out from under them.  Claiming the token FIRST
        # makes any such delivery "superseded" and discarded.  The early
        # `return None` below (nothing selected) then also invalidates an
        # in-flight load, which is correct: its layers are gone by then.
        self._patch_load_token = int(getattr(self, "_patch_load_token", 0)) + 1

        # Remove ALL old HITL layers (images + labels + RGB stacks).
        # Vispy-safe: hide before remove.
        to_remove = []
        for layer in list(self.viewer.layers):
            name = str(getattr(layer, "name", ""))
            if name.endswith("_view"):
                to_remove.append(layer)
            elif "_work_labels_" in name:
                to_remove.append(layer)
            elif name.endswith("_native_rgb_stack"):
                to_remove.append(layer)
            elif name == "final_labels":
                to_remove.append(layer)
        for layer in to_remove:
            try:
                # Save colormap from work labels so it can be restored
                # on the replacement layer (prevents random color changes).
                name = str(getattr(layer, "name", ""))
                if "_work_labels_" in name:
                    self._saved_work_labels_colormap = getattr(
                        layer, "colormap", None
                    )
                layer.visible = False
                self.viewer.layers.remove(layer)
            except Exception as exc:
                pass
            # Measured 2026-09-07: one napari layer removal is 215-280 ms and
            # a load removes 4 (annotator closed) to 11 (open) of them, so the
            # loop alone is the single longest UI freeze in Step 2.  Pump
            # between removals so the busy bar keeps sweeping; user input is
            # excluded, and the load token / stale guards are unchanged.
            self._pump_busy_ui()

        # Brick D fix (2026-09-07): the layers for the old patch are gone, so
        # the name must go with them.  During the async window nothing —
        # Launch Multiview, Accept, a status refresh — may act on a patch name
        # whose layers are no longer in the viewer.  Every path that succeeds
        # re-sets it in `_apply_loaded_patch`.
        #
        # Re-audit fix (2026-09-07): the clear must be REVERSIBLE.  The flush
        # at the top of this method ends in `_schedule_state_save()` (500 ms)
        # and `_save_internal_state` writes `loaded_patch_name`
        # unconditionally, so a load that fails - or is discarded, or simply
        # takes longer than the debounce - used to persist "" and the next
        # session forgot which patch was open (Step 1's downstream-used guard
        # then reads the patch as unused).
        prev_patch_name = str(getattr(self, "loaded_patch_name", "") or "")
        self.loaded_patch_name = ""

        # Determine view axis.
        target, axis = self._read_view_selection_from_ui()
        if force_axis:
            axis = str(force_axis).upper()
            if axis not in {"XY", "XZ", "YZ"}:
                axis = "XY"
            target = "lateral" if axis == "XY" else "axial"
        self.view_target = target
        self._set_axial_combo_enabled()

        # Get selected patch file.
        selected_items = self.ui.train_patch_list.selectedItems()
        if not selected_items:
            print("Please select a patch to load.")
            return None

        filename = self._clean_patch_filename(selected_items[0])
        # B3: the effective folder — the user's choice, else the Step 1 default.
        file_path = os.path.join(self._effective_stack_path(), filename)

        print(f"Loading patch: {file_path} with view: {axis.upper()}")

        return {
            "token": self._patch_load_token,
            "file_path": file_path,
            "patch_hint": os.path.splitext(filename)[0],
            "axis": axis,
            "target": target,
            "working_dir": str(getattr(self, "working_dir", "") or ""),
            "run_id": str(getattr(self, "active_run_id", "") or ""),
            "prev_patch_name": prev_patch_name,
        }

    def _load_patch_async(self, force_axis=None, on_done=None):
        """Patch load with the disk read on a QThread (Brick D / journey #6).

        The prelude (curating-debounce drain, refine exit, old-layer teardown,
        token claim) runs SYNCHRONOUSLY here, before the worker starts, exactly
        as it does in :meth:`_load_patch` — a flush that ran after the swap
        would write the new patch's pixels into the old slice's pair.
        ``loaded_patch_name`` changes only in the finished slot.

        Args:
            force_axis: as `_load_patch`.
            on_done: optional callable run on the main thread after the
                layers are in place (skipped for a superseded / failed load).

        Returns:
            True if a load was started (or completed, in the sync fallback).
        """
        ctx = self._load_patch_prelude(force_axis=force_axis)
        if ctx is None:
            return False
        ctx["on_done"] = on_done
        self._begin_busy(
            "[PATCH] Loading '" + str(ctx["patch_hint"]) + "' ...",
            cursor=False,
        )

        # Synchronous fallback (tests / headless / no Qt event loop), mirroring
        # the `_use_sync_training` escape hatch on the training path.
        if getattr(self, "_use_sync_training", False) or not self._can_show_modal():
            try:
                payload = self._read_patch_payload(ctx["file_path"])
            except Exception as exc:
                self._on_patch_load_error(f"{type(exc).__name__}: {exc}", ctx)
                return True
            self._on_patch_load_finished(payload, ctx)
            return True

        worker = _Step2PatchLoadWorker(self._read_patch_payload, ctx["file_path"])
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(
            lambda payload, c=ctx: self._on_patch_load_finished(payload, c)
        )
        worker.error.connect(
            lambda msg, c=ctx: self._on_patch_load_error(msg, c)
        )
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(
            lambda t=thread: self._cleanup_patch_load_thread(t)
        )
        self._patch_load_thread = thread
        self._patch_load_worker = worker
        thread.start()
        return True

    def _cleanup_patch_load_thread(self, thread=None):
        """Retire the thread that just finished — NOT whatever is current.

        Brick D fix (2026-09-07): this used to read `self._patch_load_thread`,
        which a second patch click has already overwritten.  The late finish of
        load #1 then called `deleteLater()` on the RUNNING thread of load #2
        and dropped its last Python reference.  The finished thread is passed
        in explicitly; the controller's handles are cleared only if they still
        point at it.
        """
        if thread is None:
            thread = getattr(self, "_patch_load_thread", None)
        if thread is not None:
            try:
                thread.deleteLater()
            except Exception:
                pass
        if getattr(self, "_patch_load_thread", None) is thread:
            self._patch_load_thread = None
            self._patch_load_worker = None

    def _patch_load_is_stale(self, ctx):
        """Non-empty reason when this load's result must be thrown away.

        Four ways to go stale: a newer patch click bumped the token, the user
        left Step 2, the workspace changed under the read, or the active run
        changed.  The run check matters because a switch to a run with no saved
        patch leaves the OLD run's patch installed, and `loaded_patch_name` is
        then saved into the NEW run's slot.
        """
        if int(ctx.get("token", -1)) != int(getattr(self, "_patch_load_token", 0)):
            return "superseded by a newer patch selection"
        if getattr(self, "_step2_left", False):
            return "Step 2 was exited"
        if str(getattr(self, "working_dir", "") or "") != str(ctx.get("working_dir", "")):
            return "the workspace changed"
        if str(getattr(self, "active_run_id", "") or "") != str(ctx.get("run_id", "")):
            return "the active run changed"
        return ""

    def _restore_patch_name_after_abandoned_load(self, ctx):
        """Put back the name the prelude cleared, when nothing else claimed it.

        Skipped when a NEWER load owns the name (it will set its own), and
        when the workspace or run has moved on - writing the old context's
        patch name into a new run's slot is exactly the bug the run check in
        `_patch_load_is_stale` exists to prevent.
        """
        if str(getattr(self, "loaded_patch_name", "") or ""):
            return
        prev = str(ctx.get("prev_patch_name", "") or "")
        if not prev:
            return
        if str(getattr(self, "working_dir", "") or "") != str(ctx.get("working_dir", "")):
            return
        if str(getattr(self, "active_run_id", "") or "") != str(ctx.get("run_id", "")):
            return
        self.loaded_patch_name = prev
        try:
            self._schedule_state_save()
        except Exception:
            pass

    def _on_patch_load_finished(self, payload, ctx):
        """Main thread: install the loaded patch, or discard a stale read."""
        stale = self._patch_load_is_stale(ctx)
        if stale:
            print("[STEP2-LOAD] discarding load of "
                  f"{ctx.get('patch_hint')!r}: {stale}")
            if not stale.startswith("superseded"):
                self._restore_patch_name_after_abandoned_load(ctx)
            self._end_busy()
            return
        # `on_done` is INSIDE the busy scope (2026-09-07): it builds the
        # active view's work-labels volume and refreshes the slice list --
        # measured ~500 ms, which used to run with the bar already hidden and
        # with `_pump_busy_ui` inert, i.e. as a silent freeze after the
        # "finished" indicator had gone.
        try:
            self._apply_loaded_patch(ctx, payload)
            on_done = ctx.get("on_done")
            if callable(on_done):
                on_done()
        finally:
            self._end_busy()

    def _on_patch_load_error(self, message, ctx):
        """Main thread: a failed read is loud and never leaves the bar spinning."""
        self._end_busy()
        stale = self._patch_load_is_stale(ctx)
        # The load failed, so the patch that WAS open is still the honest
        # answer - and must not be persisted away as "".
        self._restore_patch_name_after_abandoned_load(ctx)
        headline = self._training_error_headline(message)
        print(f"Failed to load image: {message}")
        self._append_ui_log(f"[PATCH FAILED] {ctx.get('patch_hint')}: {headline}")
        if stale:
            return
        self._show_modal_error(
            "Could not load patch", headline,
            f"Patch: {ctx.get('file_path')}",
        )

    def _read_patch_payload(self, file_path):
        """Pure disk read — the ONLY part of a patch load safe off-thread.

        Measured (2026-09-07, real workspace): 10.4 ms for a 21 MB 2x81x256x256
        patch, i.e. ~0.5 GB/s, so today's lab patches are nowhere near the
        100 ms budget — but the cost is linear in file size and this is the
        one blocking call in the whole load.
        """
        channels_dict, metadata, patch_name = (
            self.image_manager.load_patch_with_metadata(file_path)
        )
        return {
            "channels_dict": channels_dict,
            "metadata": metadata,
            "patch_name": patch_name,
        }

    def _apply_loaded_patch(self, ctx, payload):
        """Main-thread half of a patch load: layers, state, UI refresh."""
        axis = ctx["axis"]
        channels_dict = payload["channels_dict"]
        patch_name = payload["patch_name"]

        # Patch name
        self.loaded_patch_name = patch_name
        self.loaded_patch_axis_view = axis

        # Cache volume shape for progress bar denominator.
        if not hasattr(self, "_patch_shapes"):
            self._patch_shapes = {}
        first_vol = next(iter(channels_dict.values()), None)
        if first_vol is not None:
            self._patch_shapes[patch_name] = tuple(first_vol.shape)

        # Save selected view to state
        self.selected_view = axis
        self.view_target = "lateral" if axis == "XY" else "axial"
        self._refresh_round_status()

        # Load channels onto the viewer with default colormaps.
        # Single-channel grayscale → inferno (better dynamic range visibility
        # than gray for biological signals). Multi-channel → magenta/green
        # so two channels are visually distinguishable when blended additively.
        # User can change any layer's colormap via napari's layer controls.
        if len(channels_dict) == 1:
            channel_colormaps = ["inferno"]
        else:
            channel_colormaps = ["magenta", "green"]

        for idx, (channel_name, volume) in enumerate(channels_dict.items()):
            layer_name = f"{patch_name}_{channel_name}_{axis}_view"
            colormap = channel_colormaps[idx % len(channel_colormaps)]  # fallback if more channels

            display_volume = volume

            existing = self._layer_by_name(layer_name)
            if existing is not None:
                existing.data = display_volume
                existing.visible = True
                try:
                    existing.colormap = colormap
                    existing.blending = "additive"
                except Exception:
                    pass
            else:
                self.viewer.add_image(
                    display_volume,
                    name=layer_name,
                    blending="additive",
                    colormap=colormap
                )
            print(f"Loaded channel '{channel_name}' as '{layer_name}' with colormap '{colormap}'")
            llm = getattr(self, "layer_lifecycle_manager", None)
            if llm is not None:
                llm.register_existing(layer_name, "step2/patch")
            # Measured 2026-09-07: ~85-150 ms per add.  Same reasoning as the
            # removal loop in `_load_patch_prelude`.
            self._pump_busy_ui()

        self.apply_active_view_to_viewer()
        self._pump_busy_ui()
        self._set_work_labels_visibility_for_context(patch_name=self.loaded_patch_name, target_view=self.view_target)
        self._refresh_inference_channel_options()
        print("Finished loading patch.")
        self._append_ui_log(f"[PATCH] Loaded '{patch_name}' in {axis} view.")
        # Re-audit fix: the prelude's debounced save may already have written
        # the empty name; re-schedule so the new one wins.
        try:
            self._schedule_state_save()
        except Exception:
            pass

    def _resolve_native_image_layer_name(self):
        """Resolve an existing patch-view layer for native annotator context."""
        source_layers = self._get_patch_view_image_layers()
        if not source_layers:
            any_plane = self._get_any_patch_view_image_layers()
            xy = [layer for layer in any_plane if str(getattr(layer, "name", "")).endswith("_XY_view")]
            source_layers = xy if xy else any_plane
            if not source_layers:
                return ""
        for layer in source_layers:
            if bool(getattr(layer, "visible", True)):
                return str(getattr(layer, "name", ""))
        return str(getattr(source_layers[0], "name", ""))

    def _foreign_run_checkpoint(self, path):
        """Return the OTHER run's slug when *path* points inside a different
        run's folder of THIS workspace, else None.

        Release-hardening 2026-09-07 fix pass: "Load Config..." used to
        apply a saved config verbatim, so loading run B's config while run A
        was active silently pointed A's training at B's checkpoint.  A path
        outside `artifacts/step2/runs/` (a genuine external model) is not
        foreign and is left alone.
        """
        raw = str(path or "").strip()
        ws = str(getattr(self, "working_dir", "") or "")
        if not raw or not ws:
            return None
        try:
            from napari_epicycle.hitl_training_annotation import _paths as _p
            runs_root = _p.step2_runs_root(ws).resolve()
            candidate = Path(raw).resolve()
        except Exception:
            return None
        try:
            rel = candidate.relative_to(runs_root)
        except ValueError:
            return None  # not under this workspace's runs/ at all
        if not rel.parts:
            return None
        slug = rel.parts[0]
        active = str(getattr(self, "active_run_id", "") or "")
        if slug == active:
            return None
        return slug

    def _sanitize_loaded_training_config(self, config):
        """Drop checkpoint paths that belong to a DIFFERENT run of this
        workspace from a config the user just loaded, and say so in the log.

        Everything else in the config is applied untouched.  Release-
        hardening 2026-09-07 fix pass.
        """
        if not isinstance(config, dict):
            return config
        cleaned = dict(config)
        for key in ("pretrained_model_path", "starting_checkpoint"):
            value = cleaned.get(key, "")
            foreign = self._foreign_run_checkpoint(value)
            if foreign is None:
                continue
            cleaned[key] = ""
            self._append_ui_log(
                f"[TRAIN] Ignored '{key}' from the loaded config: it points "
                f"at run '{foreign}', not the active run "
                f"'{getattr(self, 'active_run_id', '')}'.  Everything else "
                f"in the config was applied; pick a Checkpoint yourself if "
                f"you want one."
            )
        return cleaned

    def _training_config_dialog_dir(self):
        """Default directory for the manual "Save Config..." / "Load
        Config..." dialogs: the ACTIVE run's folder.

        Release-hardening 2026-09-07 (bucket-switch item #7).  These
        dialogs used to default to the workspace ROOT, next to the
        automatic `training_config.json` that Step 2 no longer writes.
        A config belongs to a run, so that is where the picker opens.
        Falls back to the workspace root when there is no run yet (or its
        folder does not exist), and to "" with no workspace at all.
        """
        ws = getattr(self, "working_dir", None)
        if not ws:
            return ""
        run_id = str(getattr(self, "active_run_id", "") or "")
        if run_id:
            try:
                from napari_epicycle.hitl_training_annotation import _paths as _p
                run_dir = _p.run_root(ws, run_id)
                if run_dir.is_dir():
                    return str(run_dir)
            except Exception:
                pass
        return str(ws)

    # Save and set the training configuration to be used
    def _save_and_set_training_config(self):
        """
            Should save the current training parameter configuration as json file,
            to the current working directory and sets current specified parameters
            to be used for the HITL training
        """
        if not self.working_dir:
            print("Please set a working directory first.")
            return
        # set the config specified the users accordingly
        config = self._collect_training_config()
        self.current_config = config  # Save to internal state
        self._sync_inference_controls_from_config()

        # MIGHT BE GOOD IDEA TO INITIATE THE SEGMENTATION MMODEL HERE
        # USING SEGMENTATION MANAGER


        # 2026-05-26: model_name_line removed.  Model name is now
        # auto-derived from {run_id}_{bucket}_round_NN at training launch
        # (controller.py:5618).  No user input required, no validation
        # needed.

        # Prompt where to save
        save_path, _ = QFileDialog.getSaveFileName(
            None, "Save Training Config",
            os.path.join(self._training_config_dialog_dir(),
                         "training_config.json"),
            "JSON Files (*.json)"
        )
        if save_path:
            # Release-hardening 1b#4 (2026-09-07): atomic, like the
            # auto-save path — the manual "Save Config…" writes to the same
            # file the next session reads back.
            _atomic_write_json(save_path, config)
            print(f"Training config saved to {save_path}")
            self._apply_model_source_selection(emit_log=False)
            self._save_internal_state()  # Optional: persist updated current_config

    # Load and set previous training configuations
    def _sync_training_controls_from_config(self):
        """Push values from ``self.current_config`` into the Training Setup
        + Advanced Parameters widgets.

        Mirror of ``_sync_inference_controls_from_config`` but for the
        training-side widgets.  Called:
        * From ``_load_training_config`` after parsing a saved JSON file
          (user-triggered Load Config button).
        * From ``_load_internal_state`` after restoring the workspace's
          saved state (automatic workspace re-open).

        Defensively guards every widget access via ``getattr(ui, name, None)``
        + ``hasattr(...)`` because test fakes and partial UIs may not carry
        every attribute.  Defaults match the widget-initial values in
        ``main_widget.py`` so a fresh workspace (empty current_config)
        produces the same UI as a clean napari launch.

        Does NOT mutate ``self.current_config`` — read-only sync from
        memory → widgets.
        """
        ui = getattr(self, "ui", None)
        if ui is None:
            return
        # Set the init-transparency guard so programmatic widget changes
        # below don't trigger _on_init_widget_changed → auto-flip Mode.
        # See _on_init_widget_changed docstring.
        # Audit MEDIUM fix 2026-05-27: save the prior guard value instead
        # of unconditionally resetting to False.  This makes the guard
        # re-entrant-safe — callers like _apply_history_entry_to_widgets
        # which also set the guard can rely on their context being
        # preserved across the nested call.
        prev_guard = getattr(self, "_init_transparency_loading", False)
        self._init_transparency_loading = True
        try:
            self._sync_training_controls_from_config_impl()
        finally:
            self._init_transparency_loading = prev_guard
        # Refresh the hint to reflect the just-loaded state.
        try:
            self._refresh_init_hint()
        except Exception:
            pass

    def _sync_training_controls_from_config_impl(self):
        """Actual widget-sync implementation, separated so the parent
        method can wrap it with the init-transparency loading guard.
        """
        ui = getattr(self, "ui", None)
        if ui is None:
            return
        cfg = self.current_config if isinstance(getattr(self, "current_config", None), dict) else {}

        def _set_value(attr, key, default):
            w = getattr(ui, attr, None)
            if w is not None and hasattr(w, "setValue"):
                try:
                    w.setValue(cfg.get(key, default))
                except (TypeError, ValueError):
                    pass

        def _set_text(attr, key, default):
            w = getattr(ui, attr, None)
            if w is not None and hasattr(w, "setText"):
                w.setText(str(cfg.get(key, default) or ""))

        def _set_combo(attr, key, default):
            # Use _set_combo_text_if_present (findText-first) — safer
            # than bare setCurrentText for legacy configs that may carry
            # a value no longer in the combo's item list.  Mirror of
            # _sync_inference_controls_from_config's pattern.
            w = getattr(ui, attr, None)
            self._set_combo_text_if_present(w, cfg.get(key, default))

        def _set_checked(attr, key, default):
            w = getattr(ui, attr, None)
            if w is not None and hasattr(w, "setChecked"):
                w.setChecked(bool(cfg.get(key, default)))


        # ----- Training Setup (Basic) -----
        _set_value("epochs_sb", "epochs", 500)
        _set_value("batch_sb", "batch_size", 8)
        _set_text("starting_checkpoint_path", "starting_checkpoint", "")

        # Device: GPU vs CPU radio pair.  Defensive both-or-neither.
        device = str(cfg.get("device", "GPU")).upper()
        gpu_btn = getattr(ui, "device_gpu", None)
        cpu_btn = getattr(ui, "device_cpu", None)
        if gpu_btn is not None and hasattr(gpu_btn, "setChecked"):
            gpu_btn.setChecked(device == "GPU")
        if cpu_btn is not None and hasattr(cpu_btn, "setChecked"):
            cpu_btn.setChecked(device != "GPU")

        # Base Model selector — default "scratch" matches the widget's
        # initial state (first item, index 0).  Audit-found: previously
        # defaulted to "cyto3" which would silently flip a fresh
        # workspace's combo from "scratch" to "cyto3" on the first load.
        _set_combo("model_selector", "model_type", "scratch")

        # ----- Advanced Parameters -----
        _set_value("lr_db", "learning_rate", 0.01)
        _set_value("wd_db", "weight_decay", 0.0001)
        _set_value("minmask_sb", "min_masks_per_image", 5)
        # 2026-05-27 channel combos use integer-index set — see
        # _set_channel_combo_by_index docstring.  Generic _set_combo
        # would silently fail when cfg carries viewer-aware labels
        # like '1: cdh1' that don't match the generic combo items.
        train_ch1 = getattr(ui, "chan1_combo", None)
        train_ch2 = getattr(ui, "chan2_combo", None)
        self._set_channel_combo_by_index(train_ch1, cfg.get("channel1", "2: green"))
        self._set_channel_combo_by_index(train_ch2, cfg.get("channel2", "1: red"))
        _set_checked("norm_cb", "normalize", True)
        _set_checked("rescale_cb", "rescale", True)
        _set_checked("flow_cb", "compute_flows", True)

        # Verify widgets actually received the values (diagnostic).
        ep_w = getattr(ui, "epochs_sb", None)
        lr_w = getattr(ui, "lr_db", None)
        ch1_w = getattr(ui, "chan1_combo", None)
        norm_w = getattr(ui, "norm_cb", None)

    def _apply_history_entry_to_widgets(self, entry) -> None:
        """Push a ``RoundHistoryEntry``'s training params into the
        Training Setup widgets.

        Wired as the ``use_config_callback`` when constructing the
        history dialog (a no-op param at first commit; now functional).
        Click a round in the timeline → that round's params populate
        the Training Setup form, so user can click Train and reproduce
        the same config.

        Strategy: merge the entry's fields into ``self.current_config``,
        then call the existing ``_sync_training_controls_from_config``
        (which already handles channel-by-integer-index, the
        init-transparency loading guard, etc.).

        Also flips Mode dropdown based on entry's ``init_mode``:
        * ``"inherited"`` → Cumulative (the round was trained that way)
        * Anything else (scratch, pretrained, browse, etc.) → From Scratch

        Defensive: never raises, logs failure to console.
        """
        if entry is None:
            return
        try:
            # 1. Merge entry's fields into current_config — NOT replace,
            # so non-widget keys (pretrained_model_path etc.) survive.
            if not isinstance(getattr(self, "current_config", None), dict):
                self.current_config = {}
            cfg = self.current_config
            cfg["epochs"] = int(getattr(entry, "n_epochs", 0) or 0)
            cfg["n_train_pairs"] = int(getattr(entry, "n_train_pairs", 0) or 0)
            # channels_used is a list of NAME strings (e.g. ["membrane", "nuclei"])
            # from the manifest writer.  The training/inference combos
            # carry "{index}: {name}" labels — we can't perfectly recover
            # the integer index from just the name, so we DON'T overwrite
            # cfg["channel1"]/cfg["channel2"] here.  User can re-pick if
            # needed.  Future improvement: store integer indices in the
            # manifest alongside channel names.

            # Audit FAIL fix 2026-05-27: also set cfg["model_type"] from
            # the entry's init_path when init_mode is pretrained / browse /
            # scratch.  Without this, Base Model combo stays at its
            # current value (could be stale).  For inherited mode, leave
            # model_type alone — the prior round's checkpoint wins via
            # cumulative anyway, and model_type isn't the meaningful
            # control there.
            init_mode = str(getattr(entry, "init_mode", "scratch") or "scratch")
            init_path = getattr(entry, "init_path", "") or ""
            if init_mode == "scratch":
                cfg["model_type"] = "scratch"
            elif init_mode == "pretrained":
                # init_path for "pretrained" is the Cellpose builtin name
                # (e.g. "cyto3").  Use as-is.
                if init_path:
                    cfg["model_type"] = init_path
            elif init_mode == "browse":
                # init_path is an arbitrary on-disk weights file.  Push to
                # the Checkpoint widget rather than the Base Model combo.
                cfg["starting_checkpoint"] = init_path
            # For "inherited", "step3-refined", "run-model": leave
            # model_type/starting_checkpoint as-is — these modes don't
            # map cleanly to the simple Base Model + Checkpoint surface.

            # 2. Set Mode based on init lineage.  The Mode combo lives
            # in self.ui — flip via the existing _on_init_widget_changed
            # path (which respects guards).  But that's only the
            # auto-switch direction (to From Scratch).  For Cumulative
            # we set directly.
            ui = getattr(self, "ui", None)
            mode_combo = getattr(ui, "training_mode_combo", None) if ui else None
            if mode_combo is not None and hasattr(mode_combo, "count"):
                init_mode = str(getattr(entry, "init_mode", "scratch") or "scratch")
                want_scratch = init_mode != "inherited"
                # Set _init_transparency_loading so our setCurrentIndex
                # doesn't trigger the auto-switch wiring.
                prev_guard = getattr(self, "_init_transparency_loading", False)
                self._init_transparency_loading = True
                try:
                    for i in range(mode_combo.count()):
                        item_text = mode_combo.itemText(i).lower()
                        if want_scratch and "scratch" in item_text:
                            mode_combo.setCurrentIndex(i)
                            break
                        if not want_scratch and "cumulative" in item_text:
                            mode_combo.setCurrentIndex(i)
                            break
                finally:
                    self._init_transparency_loading = prev_guard

            # 3. Push widgets from the updated config.
            self._sync_training_controls_from_config()

            self._append_ui_log(
                f"[HISTORY] applied config from round_{entry.round_idx:02d} "
                f"({entry.init_mode}, ep={entry.n_epochs}, "
                f"{entry.n_train_pairs} pairs)."
            )
        except Exception as exc:
            print(f"[WARN] 'Use this config' from history failed: "
                  f"{type(exc).__name__}: {exc}")

    def _open_training_history_dialog(self):
        """Open (or raise) the non-modal Training History dialog for the
        active run.

        Single-instance pattern: reuses the existing dialog if it's
        already open + visible (just raises it).  Otherwise constructs
        a fresh one.  Non-modal so the user can interact with the main
        window while it's open.

        Safe to call when no run is loaded — the dialog gracefully
        renders empty buckets.
        """
        if not getattr(self, "working_dir", None):
            self._append_ui_log("[HISTORY] no workspace loaded — open one first.")
            return
        run_id = getattr(self, "active_run_id", None) or "default"
        try:
            from napari_epicycle.hitl_training_annotation._training_history_dialog import (
                TrainingHistoryDialog,
            )
        except Exception as exc:
            print(f"[WARN] Training History dialog import failed: "
                  f"{type(exc).__name__}: {exc}")
            return

        # Single-instance: reuse the dialog if it already exists + is
        # still attached to a live Qt object.
        existing = getattr(self, "_training_history_dialog", None)
        if existing is not None:
            try:
                # Cheap liveness check — if the underlying Qt object has
                # been destroyed, isVisible() raises RuntimeError.
                existing.isVisible()
                # Reuse — bump to active run + refresh, then raise.
                existing._workspace = self.working_dir
                existing._run_id = run_id
                existing._display_name = self._display_name_for_run(run_id)
                existing.setWindowTitle(
                    f"Training History — {existing._display_name}"
                )
                existing.refresh()
                existing.show()
                existing.raise_()
                existing.activateWindow()
                return
            except RuntimeError:
                # Underlying Qt widget got destroyed (e.g., user closed
                # the dialog and Qt cleaned up).  Fall through to create
                # a fresh one.
                self._training_history_dialog = None

        try:
            dialog = TrainingHistoryDialog(
                workspace=self.working_dir,
                run_id=run_id,
                display_name=self._display_name_for_run(run_id),
                use_config_callback=self._apply_history_entry_to_widgets,
                # User feedback 2026-05-27: clicks should only fire
                # "use config" for the bucket matching the current view.
                # Callback returns "lateral" or "axial" based on the
                # controller's view_target (re-resolved on each click).
                active_bucket_callback=lambda: (
                    "lateral"
                    if str(getattr(self, "view_target", "lateral") or "lateral").lower() == "lateral"
                    else "axial"
                ),
                parent=getattr(self, "ui", None),
            )
        except Exception as exc:
            print(f"[WARN] Training History dialog construction failed: "
                  f"{type(exc).__name__}: {exc}")
            return
        self._training_history_dialog = dialog
        dialog.show()  # non-modal — does NOT block

    def _display_name_for_run(self, run_id: str) -> str:
        """Return human-readable display name for a run, falling back to
        the handle.  Defensive: never crashes."""
        try:
            from packages.epicycle_io.step2_run_manifest import read_run_manifest
            from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
            rm_path = _step2_paths.run_manifest_path(self.working_dir, run_id)
            rm = read_run_manifest(rm_path)
            if rm and isinstance(rm, dict):
                dn = rm.get("display_name", "").strip()
                if dn:
                    return dn
        except Exception:
            pass
        return run_id

    def _load_training_config(self):
        """User-triggered: read training config JSON file and push values
        into UI widgets.

        For automatic workspace-load restoration, see
        ``_sync_training_controls_from_config`` which is called from
        ``_load_internal_state`` after the state JSON populates
        ``self.current_config``.
        """
        config_path, _ = QFileDialog.getOpenFileName(
            None, "Load Training Config",
            self._training_config_dialog_dir(),
            "JSON Files (*.json)"
        )
        if not config_path:
            return

        with open(config_path, "r") as f:
            config = json.load(f)

        # Release-hardening 2026-09-07 fix pass: a config saved from a
        # DIFFERENT run of this workspace carries that run's checkpoint
        # paths.  Applying them verbatim would silently point this run's
        # training at another run's model.  Drop just those two keys (with a
        # log line saying so) and apply everything else.
        config = self._sanitize_loaded_training_config(config)

        self.current_config = config

        # Push to widgets via the shared sync helper (covers both Basic
        # Training Setup + Advanced Parameters).
        self._sync_training_controls_from_config()

        # curation_pretrained_combo + the inference combo block are
        # outside the training-sync helper because they're driven from a
        # different concern (curation model picker, not training config).
        ui = self.ui
        pre_combo = getattr(ui, "curation_pretrained_combo", None)
        if pre_combo is not None and hasattr(pre_combo, "setCurrentText"):
            pre_combo.setCurrentText(config.get("model_type", "cyto3"))
            self.curation_pretrained_model_type = str(pre_combo.currentText() or "cyto3")
        self._sync_inference_controls_from_config()
        self._apply_model_source_selection(emit_log=False)
        # 2026-05-29 audit fix: persist the loaded config + pretrained type so
        # it survives exit/resume (same un-saved-mutator class as the model-
        # source handlers).
        self._schedule_state_save()

        print(f"Loaded training configuration from {config_path}")
 
    def _collect_current_slice_input(self):
        if not self.viewer:
            print("Viewer is not initialized.")
            return None, None
        if not self.loaded_patch_name:
            print("Please load a patch before segmenting.")
            return None, None

        dims = self.viewer.dims
        visible_axes = dims.order[-2:]
        all_axes = list(range(len(dims.current_step)))
        slicing_axis_candidates = [ax for ax in all_axes if ax not in visible_axes]
        if not slicing_axis_candidates:
            print("Could not determine slicing axis.")
            return None, None

        slicing_axis = slicing_axis_candidates[0]
        slice_index = dims.current_step[slicing_axis]
        patch_name = self.loaded_patch_name
        axis_name = self.selected_view
        slice_label = f"{patch_name}_{axis_name}_{slice_index}"

        # Critical: only collect channels from the active patch context.
        # Prefer currently visible layers for this patch, with a patch-scoped fallback.
        source_layers = []
        prefix = f"{patch_name}_"
        for layer in self.viewer.layers:
            if layer.__class__.__name__ != "Image":
                continue
            name = str(getattr(layer, "name", ""))
            data = getattr(layer, "data", None)
            if not name.startswith(prefix) or not name.endswith("_view"):
                continue
            if data is None or getattr(data, "ndim", 0) != 3:
                continue
            if bool(getattr(layer, "visible", False)):
                source_layers.append(layer)
        if not source_layers:
            for layer in self.viewer.layers:
                if layer.__class__.__name__ != "Image":
                    continue
                name = str(getattr(layer, "name", ""))
                data = getattr(layer, "data", None)
                if not name.startswith(prefix) or not name.endswith("_view"):
                    continue
                if data is None or getattr(data, "ndim", 0) != 3:
                    continue
                source_layers.append(layer)
        # Preserve layer insertion order (matches source channel order from metadata).
        # Do NOT sort alphabetically — that destroys the original channel ordering.

        collected_channels = []
        base_shape = None
        for layer in source_layers:
            try:
                if slicing_axis == 0:
                    img2d = layer.data[slice_index, :, :]
                elif slicing_axis == 1:
                    img2d = layer.data[:, slice_index, :]
                elif slicing_axis == 2:
                    img2d = layer.data[:, :, slice_index]
                else:
                    raise ValueError("Unsupported slicing axis.")
            except Exception:
                continue

            if img2d.ndim == 3 and img2d.shape[-1] in [3, 4]:
                img2d = img2d[..., :3].mean(axis=-1)
            if base_shape is None:
                base_shape = img2d.shape
            if img2d.shape != base_shape:
                continue
            collected_channels.append(img2d)

        if not collected_channels:
            print(
                f"[ERROR] No valid 2D slices were collected for active patch '{patch_name}' "
                f"(view={axis_name})."
            )
            return None, None
        input_stack = np.stack(collected_channels, axis=0)
        return slice_label, input_stack

    def _layer_by_name(self, name):
        if not self.viewer or not name:
            return None
        try:
            if name in self.viewer.layers:
                return self.viewer.layers[name]
        except Exception:
            pass
        for layer in self.viewer.layers:
            if getattr(layer, "name", "") == name:
                return layer
        return None

    def _current_slice_result_label(self):
        ui = getattr(self, "ui", None)
        if ui is None:
            return ""
        list_widget = getattr(ui, "segment_slice_data_list", None)
        if list_widget is None:
            return ""
        try:
            selected = list_widget.selectedItems()
            if selected:
                return str(selected[0].text())
            current = list_widget.currentItem()
            if current is not None:
                return str(current.text())
        except Exception:
            return ""
        return ""

    def _get_patch_view_image_layers(self, view_name=None):
        if not self.viewer or not self.loaded_patch_name:
            return []
        view = str(view_name or self.selected_view or "XY").upper()
        prefix = f"{self.loaded_patch_name}_"
        suffix = f"_{view}_view"
        layers = []
        for layer in self.viewer.layers:
            if layer.__class__.__name__ != "Image":
                continue
            name = str(getattr(layer, "name", ""))
            if not name.startswith(prefix) or not name.endswith(suffix):
                continue
            data = getattr(layer, "data", None)
            if data is None or getattr(data, "ndim", 0) != 3:
                continue
            layers.append(layer)
        # Preserve layer order (matches source channel order from metadata).
        return layers

    def _get_any_patch_view_image_layers(self):
        if not self.viewer or not self.loaded_patch_name:
            return []
        prefix = f"{self.loaded_patch_name}_"
        layers = []
        for layer in self.viewer.layers:
            if layer.__class__.__name__ != "Image":
                continue
            name = str(getattr(layer, "name", ""))
            if not name.startswith(prefix) or not name.endswith("_view"):
                continue
            data = getattr(layer, "data", None)
            if data is None or getattr(data, "ndim", 0) != 3:
                continue
            layers.append(layer)
        # Preserve layer order (matches source channel order from metadata).
        return layers

    def _work_labels_layer_name_for_view(self, patch_name=None, view_name=None):
        patch = str(patch_name or self.loaded_patch_name or "").strip()
        view = str(view_name or self.selected_view or "XY").upper()
        if not patch:
            return ""
        if view == "XY":
            return f"{patch}_work_labels_lateral_3d"
        if view == "XZ":
            return f"{patch}_work_labels_axial_xz_3d"
        if view == "YZ":
            return f"{patch}_work_labels_axial_yz_3d"
        return f"{patch}_work_labels_3d"

    def _iter_work_label_layers(self):
        if not self.viewer:
            return
        for layer in self.viewer.layers:
            if layer.__class__.__name__ != "Labels":
                continue
            name = str(getattr(layer, "name", ""))
            if "_work_labels_" in name:
                yield layer

    def _set_active_main_labels_layer(self, labels_name):
        if not self.viewer:
            return
        layer = self._layer_by_name(str(labels_name or "").strip())
        if layer is None:
            return
        try:
            self.viewer.layers.selection.active = layer
        except Exception:
            pass

    def _set_work_labels_visibility_for_context(self, patch_name=None, target_view=None):
        if not self.viewer:
            return
        patch = str(patch_name or self.loaded_patch_name or "").strip()
        target = str(target_view or self.view_target or "lateral").lower()
        show_names = set()
        needed_views = []
        if patch:
            if target == "axial":
                # Axial context exposes both orthoview work layers.
                show_names.add(f"{patch}_work_labels_axial_xz_3d")
                show_names.add(f"{patch}_work_labels_axial_yz_3d")
                needed_views = ["XZ", "YZ"]
            else:
                show_names.add(f"{patch}_work_labels_lateral_3d")
                needed_views = ["XY"]
            show_names.add(f"{patch}_work_labels_3d")

        # Lazy work-labels (2026-09-07 patch-bar brick).  A patch load used to
        # build all THREE view volumes (XY + XZ + YZ) even though only the
        # active view is ever shown -- measured 650-780 ms per load for two
        # layers nobody looks at.  The layers are now created HERE, the single
        # choke point every path goes through before a work-labels layer
        # becomes visible, so a view switch pays for its own volume once.
        # `_ensure_work_labels_layer` is idempotent, so a caller that already
        # created the layer costs nothing extra.
        #
        # CONTRACT (lifecycle audit 2026-09-07): the set built here is
        # deliberately the SAME set the 2D annotator mirrors into its
        # subviewers -- XY for a lateral context, XZ + YZ for an axial one.
        # The annotator does NOT mirror a layer inserted while it is already
        # loaded, so ANY NEW lazy-creation site (or any widening of this set)
        # MUST be followed by an annotator re-sync -- `set_context(...,
        # auto_load=True)` or `_schedule_deferred_auto_load(...)` -- or the
        # freshly built layer exists in the main viewer and is invisible in
        # the side views.  The existing call sites are all already followed by
        # one (patch load -> deferred auto-load; view switch -> set_context).
        if patch and self.viewer is not None:
            if not getattr(self, "_ensuring_work_labels_visibility", False):
                missing = []
                for view in needed_views:
                    name = self._work_labels_layer_name_for_view(
                        patch_name=patch, view_name=view,
                    )
                    if name and self._layer_by_name(name) is None:
                        missing.append(view)
                if missing:
                    self._ensuring_work_labels_visibility = True
                    # Building one view's volume is ~230 ms (a napari
                    # add_labels plus the per-slice mask restore), so the
                    # FIRST switch to a view gets the same sweeping bar as a
                    # patch load.  Nested inside a load's busy scope this is
                    # just a depth bump.
                    self._begin_busy(
                        f"[PATCH] Preparing {'/'.join(missing)} labels ...",
                        cursor=False,
                    )
                    try:
                        for view in missing:
                            # Per-view try: a bad XZ volume (shape drift on a
                            # re-extracted patch) must not cost the user YZ.
                            try:
                                self._ensure_work_labels_layer(view_name=view)
                            except Exception as exc:
                                print(f"[WORK-LABELS] lazy create failed for "
                                      f"{patch!r} view={view}: "
                                      f"{type(exc).__name__}: {exc}")
                            self._pump_busy_ui()
                    finally:
                        self._ensuring_work_labels_visibility = False
                        self._end_busy()

        for layer in self._iter_work_label_layers():
            name = str(getattr(layer, "name", ""))
            try:
                visible = name in show_names
                layer.visible = visible
                # Main-canvas read-only (2026-05-28): the main napari canvas
                # is for VISUALIZATION only in Step 2 — all drawing happens
                # on the annotator subviewers (which sync back to these
                # layers via array-copy, NOT via painting, so read-only does
                # not break sync).  Forcing editable=False here means a
                # finalized (or any) slice can never be painted on the main
                # canvas, closing the surface the subviewer hard-lock didn't
                # cover.  Was `editable = bool(visible)`.
                layer.editable = False
            except Exception:
                continue

    def _ensure_native_rgb_stack_layer(self):
        if not self.viewer or not self.loaded_patch_name:
            return ""
        view = str(self.selected_view or "XY").upper()
        source_layers = self._get_patch_view_image_layers()
        if not source_layers:
            return ""

        channel_volumes = [np.asarray(layer.data) for layer in source_layers]
        if len(channel_volumes) == 1:
            stack_zyx = channel_volumes[0][:, np.newaxis, :, :]
            channel_indices = [0]
        else:
            stack_zyx = np.stack(channel_volumes[:2], axis=1)  # (Z, C, Y, X)
            channel_indices = [0, 1]

        rgb_stack = self.image_manager.convert_to_rgb(
            image=stack_zyx,
            channel_indices=channel_indices,
            normalize=True,
            is_stack=True,
        )
        layer_name = f"{self.loaded_patch_name}_{view}_native_rgb_stack"
        existing = self._layer_by_name(layer_name)
        if existing is not None:
            existing.data = rgb_stack
            existing.rgb = True
            existing.visible = True
        else:
            self.viewer.add_image(
                rgb_stack,
                name=layer_name,
                rgb=True,
                blending="additive",
            )
        llm = getattr(self, "layer_lifecycle_manager", None)
        if llm is not None:
            llm.register_existing(layer_name, "step2/patch")
        return layer_name

    def _apply_slice_mask_to_volume(self, volume, mask2d, view_name, slice_index):
        view = str(view_name or "XY").upper()
        idx = int(slice_index)
        if view == "XY":
            if mask2d.shape != tuple(volume.shape[1:3]):
                raise ValueError(f"XY mask shape {mask2d.shape} != {(volume.shape[1], volume.shape[2])}")
            volume[idx, :, :] = mask2d
            return
        if view == "XZ":
            if mask2d.shape != (volume.shape[0], volume.shape[2]):
                raise ValueError(f"XZ mask shape {mask2d.shape} != {(volume.shape[0], volume.shape[2])}")
            volume[:, idx, :] = mask2d
            return
        if view == "YZ":
            if mask2d.shape != (volume.shape[0], volume.shape[1]):
                raise ValueError(f"YZ mask shape {mask2d.shape} != {(volume.shape[0], volume.shape[1])}")
            volume[:, :, idx] = mask2d
            return
        raise ValueError(f"Unsupported view: {view_name}")

    def _clear_slice_in_volume(self, volume, view_name, slice_index):
        view = str(view_name or "XY").upper()
        idx = int(slice_index)
        if view == "XY":
            volume[idx, :, :] = 0
            return
        if view == "XZ":
            volume[:, idx, :] = 0
            return
        if view == "YZ":
            volume[:, :, idx] = 0
            return
        raise ValueError(f"Unsupported view: {view_name}")

    def _sync_selected_label_to_volume(self, labels_layer):
        """Set selected_label to max(ALL volumes)+1 so the next paint stroke gets a unique ID."""
        if labels_layer is None:
            return
        try:
            global_max = int(np.asarray(labels_layer.data).max())
            # Check ALL work_labels volumes for cross-view uniqueness.
            for wl_layer in self._iter_work_label_layers():
                try:
                    other_max = int(np.max(np.asarray(wl_layer.data)))
                    global_max = max(global_max, other_max)
                except Exception:
                    pass
            labels_layer.selected_label = max(1, global_max + 1)
        except Exception:
            pass

    def _connect_label_safety_hooks(self, labels_layer):
        """Connect event handlers that keep selected_label safe on work labels layers."""
        if labels_layer is None:
            return
        try:
            # Guard: don't double-connect.
            if getattr(labels_layer, "_epicycle_safety_connected", False):
                return
            labels_layer.events.mode.connect(self._on_labels_mode_changed)
            labels_layer.events.paint.connect(self._on_work_labels_painted)
            labels_layer._epicycle_safety_connected = True
        except Exception:
            pass

    def _on_labels_mode_changed(self, event):
        """Auto-bump selected_label when entering paint/fill with a conflicting label.

        If the user has selected_label > 0 and that label already exists
        somewhere in the volume, bump to max(volume)+1 so the new paint
        stroke creates a genuinely unique object.  Label 0 (erase) is
        never touched.
        """
        layer = getattr(event, "source", None)
        if layer is None:
            return
        mode = str(getattr(layer, "mode", "")).lower()
        if mode not in ("paint", "fill"):
            return
        current = getattr(layer, "selected_label", 0)
        if current <= 0:
            return  # erase mode — don't interfere
        try:
            volume = np.asarray(layer.data)
            if current <= int(volume.max()):
                # Current label already exists in the volume — unsafe.
                next_label = max(1, int(volume.max()) + 1)
                layer.selected_label = next_label
        except Exception:
            pass

    def _view_from_labels_layer_name(self, layer_name):
        """Infer the view orientation from a work_labels layer name."""
        name = str(layer_name or "")
        if "_lateral_" in name:
            return "XY"
        if "_axial_xz_" in name:
            return "XZ"
        if "_axial_yz_" in name:
            return "YZ"
        return "XY"

    def _current_slice_label_for_labels_layer(self, labels_layer):
        """Resolve the slice label (e.g. 'patch1_XY_5') for the current viewer position on this labels layer."""
        patch = str(self.loaded_patch_name or "").strip()
        if not patch:
            return None
        view = self._view_from_labels_layer_name(getattr(labels_layer, "name", ""))
        try:
            dims_point = list(self.viewer.dims.current_step)
            idx = int(dims_point[0]) if dims_point else 0
        except Exception:
            return None
        return f"{patch}_{view}_{idx}"

    def _on_work_labels_painted(self, event):
        """Demote accepted/staged/committed slices back to 'predicted' when user paints on them.

        This prevents the silent data-staleness bug where a user edits a curated
        slice in the viewer but the old persisted mask is used for training.
        """
        layer = getattr(event, "source", None)
        if layer is None:
            return
        slice_label = self._current_slice_label_for_labels_layer(layer)
        if not slice_label:
            return
        record = self.sample_records.get(slice_label)
        if record is None:
            return
        status = record.get("status", "")
        if status not in ("accepted", "staged", "committed"):
            return
        # Demote to predicted — user must re-accept after editing.
        record["status"] = "predicted"
        self._append_ui_log(
            f"[CURATION] Edited {status} slice {slice_label} — status reset to 'predicted'. "
            f"Re-accept when done editing."
        )
        self._refresh_sample_queue()

    def _insert_slice_mask_with_unique_labels(self, volume, mask2d, view_name, slice_index):
        """
        Write a 2D predicted mask into the target slice while preserving existing
        predictions in other slices and avoiding label-id collisions.
        """
        view = str(view_name or "XY").upper()
        raw = np.asarray(mask2d, dtype=np.int32)
        if raw.ndim != 2:
            raise ValueError("Expected 2D mask for slice insertion.")

        # Replace only the target slice but preserve all other slices.
        self._clear_slice_in_volume(volume, view, slice_index)
        # Compute max across ALL work_labels volumes (not just the current one)
        # to prevent duplicate label IDs across XZ/YZ axial volumes.
        base_max = int(np.asarray(volume).max())
        for wl_layer in self._iter_work_label_layers():
            try:
                other_max = int(np.max(np.asarray(wl_layer.data)))
                base_max = max(base_max, other_max)
            except Exception:
                pass

        if int(raw.max()) <= 0:
            self._apply_slice_mask_to_volume(volume, raw, view, slice_index)
            # fix/step2-autoinc-perf: labels just introduced — invalidate the
            # annotator's enclosed-fill next-label cache so it re-seeds and
            # can't hand out a colliding ID.
            self._invalidate_annotator_fill_label_cache()
            return max(1, base_max + 1)

        relabeled = raw.astype(np.int64, copy=True)
        fg = relabeled > 0
        relabeled[fg] = relabeled[fg] + base_max
        relabeled = relabeled.astype(np.int32, copy=False)
        self._apply_slice_mask_to_volume(volume, relabeled, view, slice_index)
        # fix/step2-autoinc-perf: see note above — invalidate after insert.
        self._invalidate_annotator_fill_label_cache()
        return max(1, int(np.asarray(volume).max()) + 1)

    def _ensure_work_labels_layer(self, seed_slice_label="", view_name=None, restore_existing=True):
        # 2026-05-25 audit-6 diagnostic: confirm reachability + caller.
        # Absence of this line in console after a slice click is itself
        # the diagnostic — proves restore loop never ran.
        try:
            import inspect as _inspect
            _caller = _inspect.stack()[1].function if len(_inspect.stack()) > 1 else "?"
        except Exception:
            _caller = "?"
        print(f"[PATCH-SWITCH-MASK] _ensure_work_labels_layer ENTER "
              f"caller={_caller} view_name={view_name!r} seed={seed_slice_label!r} "
              f"loaded_patch={self.loaded_patch_name!r} "
              f"sample_records_n={len(getattr(self, 'sample_records', {}))}")
        if not self.viewer or not self.loaded_patch_name:
            return ""

        view = str(view_name or self.selected_view or "XY").upper()
        ref_shape = None
        ref_sources = self._get_patch_view_image_layers(view_name=view)
        if not ref_sources:
            ref_sources = self._get_any_patch_view_image_layers()
        if ref_sources:
            ref_shape = tuple(np.asarray(ref_sources[0].data).shape)
        if ref_shape is None:
            native_name = f"{self.loaded_patch_name}_{view}_native_rgb_stack"
            native_layer = self._layer_by_name(native_name)
            if native_layer is not None:
                data = np.asarray(native_layer.data)
                if data.ndim >= 3:
                    ref_shape = tuple(data.shape[:3])
        if ref_shape is None:
            return ""

        labels_name_primary = self._work_labels_layer_name_for_view(view_name=view_name)
        legacy_name = f"{self.loaded_patch_name}_work_labels_3d"
        labels_layer = self._layer_by_name(labels_name_primary) or self._layer_by_name(legacy_name)
        labels_name = labels_name_primary if labels_layer is None else str(getattr(labels_layer, "name", labels_name_primary))
        layer_created = False
        if labels_layer is None or getattr(labels_layer, "data", None) is None:
            labels_layer = self.viewer.add_labels(
                np.zeros(ref_shape, dtype=np.int32),
                name=labels_name,
                opacity=0.8,
            )
            self._connect_label_safety_hooks(labels_layer)
            layer_created = True
        else:
            existing_shape = tuple(np.asarray(labels_layer.data).shape)
            if existing_shape != ref_shape:
                labels_layer.data = np.zeros(ref_shape, dtype=np.int32)
                layer_created = True

        # Always register with lifecycle manager (idempotent — dedup guard
        # in _register() prevents duplicates).  Covers both new and reused paths.
        llm = getattr(self, "layer_lifecycle_manager", None)
        if llm is not None:
            llm.register_existing(labels_name, "step2/patch")

        # Restore colormap from previous work_labels layer (if saved during
        # _load_patch removal) so label colors stay consistent across patches.
        saved_cmap = getattr(self, "_saved_work_labels_colormap", None)
        if saved_cmap is not None:
            try:
                labels_layer.colormap = saved_cmap
            except Exception:
                pass

        # 2026-05-28 (label-compaction-v2 perf+climb fix): steady-state reuse
        # fast path.  When re-segmenting (restore_existing=False) and we are
        # REUSING an already-populated layer (not freshly created), the new
        # mask is already in the volume (the segment path inserted it via
        # _run_segmentation) and every other slice is already present.  The
        # full seed-insert + restore below would re-insert EVERY slice onto the
        # non-zeroed volume, which (a) re-offsets each slice by the ever-growing
        # whole-volume max -> the label-ID CLIMB the user reported, and (b)
        # costs an np.max-per-slice rebuild on EVERY segment (scales with slice
        # count -> the lag).  Skipping it is both correct and faster.
        if not restore_existing and not layer_created:
            self._sync_selected_label_to_volume(labels_layer)
            return labels_name

        # Rebuild path (fresh layer, or an explicit restore).  Zero the volume
        # first so base_max resets to 0 and the rebuilt IDs stay bounded to
        # ~total cell count instead of compounding across successive rebuilds.
        labels_layer.data = np.zeros(ref_shape, dtype=np.int32)

        if seed_slice_label:
            mask2d = self.segmentation_manager.get_masks(seed_slice_label)
            if mask2d is not None and getattr(mask2d, "ndim", 0) == 2:
                view = self._infer_view_from_label(seed_slice_label)
                slice_idx = self._infer_slice_index(seed_slice_label)
                if slice_idx >= 0:
                    try:
                        data = np.asarray(labels_layer.data)
                        next_label = self._insert_slice_mask_with_unique_labels(
                            data,
                            np.asarray(mask2d, dtype=np.int32),
                            view,
                            slice_idx,
                        )
                        labels_layer.data = data
                        try:
                            labels_layer.selected_label = int(next_label)
                        except Exception:
                            pass
                        labels_layer.refresh()
                    except Exception as exc:
                        self._append_ui_log(f"[WARN] Could not seed work labels from {seed_slice_label}: {exc}")

        # Restore ALL masks for this patch+view (including predicted/curating
        # so they survive napari restart and can be accepted directly).
        patch = str(self.loaded_patch_name or "")
        if patch and hasattr(self, "sample_records"):
            restored = 0
            skipped_no_mask = 0
            # 2026-05-26 audit-J HIGH fix: track masks that failed to
            # insert due to shape/bounds mismatch (typically caused by
            # patch re-extraction with different dimensions).  See the
            # [RESTORE-WARN] block below.
            failed_shape_mismatch = 0
            for sl, rec in self.sample_records.items():
                status = rec.get("status", "")
                if status not in ("predicted", "curating", "accepted", "staged", "committed"):
                    continue
                if self._infer_view_from_label(sl) != view:
                    continue
                label_patch = sl.rsplit("_", 2)[0] if sl.count("_") >= 2 else ""
                if label_patch != patch:
                    continue
                if sl == seed_slice_label:
                    continue
                mask2d = self.segmentation_manager.get_masks(sl)
                if mask2d is None or getattr(mask2d, "ndim", 0) != 2:
                    skipped_no_mask += 1
                    # 2026-05-25 audit-6: per-label skip cause (first 5).
                    if skipped_no_mask <= 5:
                        mask_uri = rec.get("mask_uri", "")
                        try:
                            on_disk = bool(mask_uri) and Path(mask_uri).is_file()
                        except Exception:
                            on_disk = False
                        print(f"[PATCH-SWITCH-MASK] SKIP label={sl!r} "
                              f"status={status!r} mask_uri_on_disk={on_disk} "
                              f"mask_uri={mask_uri!r}")
                    continue
                s_idx = self._infer_slice_index(sl)
                if s_idx < 0:
                    continue
                try:
                    data = np.asarray(labels_layer.data)
                    self._insert_slice_mask_with_unique_labels(
                        data, np.asarray(mask2d, dtype=np.int32), view, s_idx
                    )
                    labels_layer.data = data
                    restored += 1
                except (ValueError, IndexError) as shape_exc:
                    # 2026-05-26 audit-J HIGH fix: shape mismatch or
                    # out-of-bounds slice on re-extraction.  Previously
                    # bare `except: pass` SILENTLY DROPPED the mask —
                    # user had no clue why their imported pair wasn't
                    # rendering.  Now: log explicitly + count toward a
                    # new `failed_shape_mismatch` bucket so the user
                    # sees the count in the [RESTORE-DIAG] line.
                    failed_shape_mismatch += 1
                    try:
                        mask_shape = tuple(np.asarray(mask2d).shape)
                    except Exception:
                        mask_shape = "?"
                    try:
                        vol_shape = tuple(data.shape) if 'data' in locals() else "?"
                    except Exception:
                        vol_shape = "?"
                    print(f"[RESTORE-WARN] shape/bounds mismatch — "
                          f"label={sl!r} view={view!r} slice={s_idx} "
                          f"mask_shape={mask_shape} volume_shape={vol_shape} "
                          f"err={type(shape_exc).__name__}: {shape_exc}.  "
                          f"This usually means the patch was re-extracted "
                          f"with different dimensions after the pair was "
                          f"saved — the imported mask is now unreachable "
                          f"for this volume.  Mask data is preserved on "
                          f"disk; re-importing or matching the original "
                          f"extraction dimensions is the recovery path.")
                except Exception as other_exc:
                    # Catch-all for any other unexpected error — log it
                    # too so we don't silently drop masks for any reason.
                    failed_shape_mismatch += 1
                    print(f"[RESTORE-WARN] unexpected error inserting "
                          f"mask for {sl!r}: {type(other_exc).__name__}: "
                          f"{other_exc}")
            if restored or skipped_no_mask or failed_shape_mismatch:
                print(f"[RESTORE-DIAG] patch={patch} view={view} | "
                      f"restored={restored} skipped_no_mask={skipped_no_mask} "
                      f"failed_shape_mismatch={failed_shape_mismatch}")
            # 2026-05-26 audit-J followup: bubble shape-mismatch failures
            # into the UI log panel too — most users don't watch the
            # console, so a silent console-only warning was easy to miss.
            if failed_shape_mismatch:
                self._append_ui_log(
                    f"[WARN] {failed_shape_mismatch} mask(s) for patch "
                    f"{patch!r} view={view} failed to render due to "
                    f"shape/bounds mismatch — usually means patch was "
                    f"re-extracted with different dimensions.  See "
                    f"console for [RESTORE-WARN] details per label."
                )
            if restored:
                labels_layer.refresh()

        # Always sync selected_label to the next unique value.
        self._sync_selected_label_to_volume(labels_layer)

        return labels_name

    def _available_patch_ids(self):
        ui = getattr(self, "ui", None)
        patch_list = getattr(ui, "train_patch_list", None) if ui is not None else None
        if patch_list is None or not hasattr(patch_list, "count") or not hasattr(patch_list, "item"):
            return []
        out = []
        for i in range(patch_list.count()):
            item = patch_list.item(i)
            if item is None:
                continue
            text = self._clean_patch_filename(item)
            stem, _ = os.path.splitext(str(text))
            if stem:
                out.append(stem)
        # preserve order while deduping
        seen = set()
        unique = []
        for x in out:
            if x in seen:
                continue
            seen.add(x)
            unique.append(x)
        return unique

    # Reentrancy guard for patch switching (instance variable).
    _patch_switch_loading = False

    def _switch_patch_context_from_native(self, patch_id):
        """Switch patch from annotator dropdown.

        Follows Step 3's _switch_to_curation_patch pattern:
        same-patch guard, reentrancy guard, suppress annotator
        handlers during layer swap, deferred auto-load after.
        """
        patch = str(patch_id or "").strip()
        if not patch:
            return {}

        # Same-patch guard.
        if patch == str(self.loaded_patch_name or "").strip():
            return {}

        # Reentrancy guard — MUST be checked and set BEFORE _select_patch_item
        # which changes the boxlist selection and can trigger _on_patch_list_clicked
        # → re-enter this method via the "annotator is open" branch.
        if self._patch_switch_loading:
            return {}
        self._patch_switch_loading = True

        # 2026-09-07 (patch-bar brick): this is the annotator-OPEN patch load
        # and it is fully SYNCHRONOUS -- measured ~6.1 s on the main thread.
        # It gets the same sweeping bar as the async (annotator-closed) path.
        # The `processEvents` immediately after `_begin_busy` paints the bar
        # BEFORE the block starts; `_pump_busy_ui` calls inside the layer
        # add/remove loops keep it sweeping through it.
        self._patch_switch_epoch = int(
            getattr(self, "_patch_switch_epoch", 0)
        ) + 1
        _switch_epoch = self._patch_switch_epoch
        try:
            self._begin_busy(f"[PATCH] Loading '{patch}' ...", cursor=False)
            # Paint the bar before the block starts.  `_pump_busy_ui` (not a
            # bare `processEvents`) so user input is EXCLUDED -- a second
            # patch click here would re-enter through the list handler.
            self._pump_busy_ui()
            selected = self._select_patch_item(patch)
            if not selected:
                self._append_ui_log(f"[WARN] Patch '{patch}' is not available in current patch list.")
                return {}

            # Find the open annotator widget (if any).
            annotator_widget = None
            try:
                from napari_epicycle.annotator.launcher import _find_dock_by_title
                dock = _find_dock_by_title(self.viewer, "EpiCycle 2D Training Annotator")
                if dock is not None:
                    annotator_widget = dock.widget() if hasattr(dock, "widget") else dock
            except Exception:
                pass

            # Suppress annotator handlers during layer swap (Step 3 pattern).
            if annotator_widget is not None:
                annotator_widget._suppress_layer_removal_cleanup = True
                if hasattr(annotator_widget, "layer_manager"):
                    annotator_widget.layer_manager._suppress_events = True

            try:
                saved_target = str(self.view_target or "lateral")
                saved_view = str(self.selected_view or "XY")
                self._load_patch(force_axis="XY")
                # Restore curation target/view semantics after XY context load.
                self.view_target = saved_target
                self.selected_view = saved_view
                self.apply_active_view_to_viewer()

                # Lazy: the visibility call below creates exactly the view(s)
                # about to be shown (see `_set_work_labels_visibility_for_context`).
                self._set_work_labels_visibility_for_context(
                    patch_name=self.loaded_patch_name, target_view=self.view_target,
                )
            except Exception as exc:
                import traceback; traceback.print_exc()
            finally:
                # Post-switch annotator sync (Step 3 _post_switch_annotator_sync pattern).
                if annotator_widget is not None:
                    annotator_widget._suppress_layer_removal_cleanup = False
                    if hasattr(annotator_widget, "layer_manager"):
                        annotator_widget.layer_manager._suppress_events = False
                    # Reset active names so deferred auto-load finds new layers.
                    annotator_widget._active_labels_name = ""
                    annotator_widget._active_image_name = ""
                    if hasattr(annotator_widget, "_schedule_deferred_auto_load"):
                        annotator_widget._schedule_deferred_auto_load(50)

        finally:
            self._patch_switch_loading = False
            # The annotator rebuilds its three subviewers on a 50 ms timer
            # scheduled just above -- measured ~1.5 s of further main-thread
            # layer work that is part of the same patch switch as far as the
            # user is concerned (and lives in the SHARED annotator package, so
            # it cannot be paced from here).  Releasing the bar now would hide
            # the indicator and THEN freeze the window.  A queued single-shot
            # is delivered after that handler returns, so the bar stays up for
            # the whole switch.  `_end_busy` still runs exactly once either
            # way, so the depth stays paired.
            self._end_busy_after_annotator_rebuild(_switch_epoch)
            # CRITICAL fix 2026-05-24 (3-audit-confirmed bug): when patch
            # switch happens via the annotator-open path, the Slices list
            # was never refreshed → stale display of previous patch's
            # slices.  Mirror the dock-CLOSED branch in _on_patch_list_clicked
            # which calls _refresh_sample_queue at line ~2345.  Also
            # refresh patch list badges + round status for full UI sync.
            try:
                self._refresh_sample_queue()
                self._refresh_patch_list_badges()
                self._refresh_round_status()
            except Exception as _refresh_err:
                pass

        return {}

    # Statuses that must NOT be silently overwritten by a re-prediction.
    # "curating" joined the tuple 2026-09-07 (Brick A, journey #3): the
    # predict path is a whole-plane overwrite with no 2D undo, so hand-drawn
    # edits were destroyed without a word.
    _RESEGMENT_GUARDED_STATUSES = ("accepted", "staged", "committed", "curating")

    def _confirm_resegment(self, slice_label):
        """Ask before a re-prediction overwrites curated or hand-edited work.

        Returns True when segmentation may proceed (the slice is unguarded, or
        the user confirmed).  On confirmation the record is demoted to
        ``predicted`` — the behaviour both predict paths had inline before this
        helper existed.  Shared by :meth:`_segment_current_slice_from_native`
        (annotator panel) and :meth:`_run_segmentation` (main panel) so the two
        guards can never drift apart again.
        """
        record = self.sample_records.get(slice_label, {}) or {}
        status = str(record.get("status", "") or "")
        if status not in self._RESEGMENT_GUARDED_STATUSES:
            return True

        if status == "curating":
            title = "Replace the drawing on this slice?"
            body = (
                f"You have drawn on {slice_label}.\n\n"
                "Running the model again will replace everything you drew "
                "here, and this cannot be undone.\n\n"
                "Until you accept this slice again, it also stops being used "
                "to train from earlier rounds.\n\n"
                "If you want a fresh prediction, cancel and discard the slice "
                "first.\n\nReplace your drawing?"
            )
            cancel_msg = (
                f"[STEP2] Left {slice_label} alone — it still has your "
                "drawing.  Discard the slice first if you want a fresh "
                "prediction."
            )
        else:
            title = "Replace a slice you already finished?"
            body = (
                f"You have already marked {slice_label} as {status}.\n\n"
                "Running the model again will replace the mask you "
                "approved.\n\nReplace it?"
            )
            cancel_msg = (
                f"[STEP2] Left {slice_label} alone — you had already marked "
                f"it {status}."
            )

        reply = QMessageBox.question(
            self.ui,
            title,
            body,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            self._append_ui_log(cancel_msg)
            return False
        # The debounce may still hold an unsaved stroke for this slice; it
        # would write the OLD drawing over the fresh prediction a few seconds
        # from now.  The user just chose to replace that drawing, so drop it.
        try:
            pending = getattr(self, "_curating_persist_pending", None)
            if isinstance(pending, dict):
                pending.pop(slice_label, None)
        except Exception:
            pass
        # Audit fix (2026-09-07): remember what the slice WAS.  The demotion
        # below happens before cellpose runs, so a failing checkpoint would
        # otherwise knock an accepted / staged / hand-drawn slice out of the
        # training set while leaving its mask untouched — invisible damage.
        # `_run_segmentation` restores this on failure.
        self._resegment_prior_status = (slice_label, status)
        # Reset status to predicted since we're overwriting.
        self._touch_sample(slice_label, status="predicted", origin="cellpose")
        return True

    def _restore_status_after_failed_segment(self, slice_label):
        """Undo `_confirm_resegment`'s demotion when the predict failed."""
        stash = getattr(self, "_resegment_prior_status", None)
        self._resegment_prior_status = None
        if not stash or stash[0] != slice_label or not stash[1]:
            return
        prior = stash[1]
        try:
            self._touch_sample(slice_label, status=prior)
        except Exception as exc:
            print(f"[SEGMENT] could not restore {slice_label} to {prior}: "
                  f"{type(exc).__name__}: {exc}")
            return
        self._append_ui_log(
            f"[SEGMENT FAILED] {slice_label} is still marked {prior} — "
            "its mask was not changed."
        )
        # The restore only touched `sample_records`; without these the Slices
        # list and step2_state.json keep saying `predicted`.
        for fn_name in ("_refresh_sample_queue", "_schedule_state_save"):
            fn = getattr(self, fn_name, None)
            if callable(fn):
                try:
                    fn()
                except Exception as exc:
                    print(f"[SEGMENT] {fn_name} after status restore failed: "
                          f"{type(exc).__name__}: {exc}")

    def _segment_current_slice_from_native(self, orientation, slice_index=None):
        plane = str(orientation or "").upper()
        if plane not in {"XY", "XZ", "YZ"}:
            return {"ok": False, "message": f"Unsupported orientation: {orientation}"}

        self._ensure_runtime_defaults()
        self._set_view_radio(plane)
        self.selected_view = plane
        self.view_target = "lateral" if plane == "XY" else "axial"
        self._set_axial_combo_enabled()
        self._apply_curation_checkpoint_for_view(view_name=plane, emit_log=False)
        # Drop the main viewer to 2D before segmenting (force-2D path).  In 3D
        # the mask insert + visibility/refresh each force a full label-VOLUME
        # GPU re-render on the main thread (4–11s freeze, confirmed via
        # [SEGMENT-TIMING] main_ndisplay=3); in 2D only one slice redraws.
        # The user annotates in the subviewers and flips the main viewer back
        # to 3D manually to inspect.  Reuses the existing force_2d path
        # (apply_active_view_to_viewer -> dims.ndisplay = 2); independent of
        # the subviewer ViewerModels, so it can't disturb subviewer sync.
        _was_3d = getattr(getattr(self.viewer, "dims", None), "ndisplay", 2) == 3
        if slice_index is None:
            self.apply_active_view_to_viewer(force_2d=True)
        else:
            self.apply_active_view_to_viewer(slice_index=int(slice_index), force_2d=True)
        if _was_3d:
            self._append_ui_log(
                "Main viewer switched to 2D for segmenting — toggle it back to "
                "3D to inspect the volume."
            )

        slice_label, _ = self._collect_current_slice_input()
        if not slice_label:
            return {"ok": False, "message": f"No active {plane} slice context to segment."}

        # Guard: warn if the slice is already curated OR carries hand edits.
        if not self._confirm_resegment(slice_label):
            return {"ok": False, "message": f"Re-segmentation of {slice_label} cancelled by user."}

        pre_mask = self.segmentation_manager.get_masks(slice_label)
        self._run_segmentation()
        post_mask = self.segmentation_manager.get_masks(slice_label)
        ok = post_mask is not None and (
            pre_mask is None or id(pre_mask) != id(post_mask) or slice_label in getattr(self, "segmented_slice_list", set())
        )

        image_layer_name = self._resolve_native_image_layer_name()
        # restore_existing=False: on re-segment the layer already holds every
        # slice and _run_segmentation already inserted the new mask, so skip
        # the redundant full restore (faster + avoids the label-ID climb).
        # A first-segment / fresh layer still rebuilds (layer_created path).
        labels_layer_name = self._ensure_work_labels_layer(
            seed_slice_label=slice_label, view_name=plane, restore_existing=False,
        )
        self._set_work_labels_visibility_for_context(
            patch_name=self.loaded_patch_name,
            target_view=self.view_target,
        )
        self._set_active_main_labels_layer(labels_layer_name)
        return {
            "ok": bool(ok),
            "message": (
                f"Segmented {slice_label}."
                if ok
                else f"No segmentation output for {slice_label}. Check active model source."
            ),
            "slice_label": slice_label,
            "slice_index": int(self._infer_slice_index(slice_label)),
            "image_layer_name": image_layer_name,
            "labels_layer_name": labels_layer_name,
            "active_patch_id": str(self.loaded_patch_name or ""),
            "selected_view": plane,
            "target_view": str(self.view_target or "lateral"),
        }

    def _fetch_slice_overlay_from_native(self, orientation, slice_index=None):
        plane = str(orientation or "").upper()
        if plane not in {"XY", "XZ", "YZ"}:
            return {"ok": False, "message": f"Unsupported orientation: {orientation}"}
        patch = str(getattr(self, "loaded_patch_name", "") or "").strip()
        if not patch:
            return {"ok": False, "message": "No active patch."}

        idx = slice_index
        if idx is None:
            if plane == "XY":
                idx = int(getattr(self.viewer.dims, "current_step", (0,))[0])
            elif plane == "XZ":
                idx = int(getattr(self.viewer.dims, "current_step", (0, 0))[1])
            else:
                idx = int(getattr(self.viewer.dims, "current_step", (0, 0, 0))[2])
        try:
            idx = int(idx)
        except Exception:
            return {"ok": False, "message": f"Invalid slice index: {slice_index}"}

        slice_label = f"{patch}_{plane}_{idx}"
        mask = self.segmentation_manager.get_masks(slice_label)
        flows = self.segmentation_manager.get_flows(slice_label)
        cell_probs = self.segmentation_manager.get_cell_probs(slice_label)

        if mask is None and flows is None and cell_probs is None:
            return {
                "ok": False,
                "message": f"No cached prediction for {slice_label}.",
                "slice_label": slice_label,
                "slice_index": idx,
                "orientation": plane,
            }
        return {
            "ok": True,
            "slice_label": slice_label,
            "slice_index": idx,
            "orientation": plane,
            "mask": np.asarray(mask) if mask is not None else None,
            "flow": np.asarray(flows) if flows is not None else None,
            "cellprob": np.asarray(cell_probs) if cell_probs is not None else None,
        }

    def _launch_multiview_annotator(self):
        self._ensure_runtime_defaults()
        self._ensure_llm_removal_callback()
        ui = getattr(self, "ui", None)
        # Open acts on the LOADED patch
        # (``loaded_patch_name``), never on the selected row.  The row's
        # text() used to be blank (the old row widgets), so the "selected
        # row differs from the loaded patch -> load it" branch was dormant;
        # now text() is the file name and it would run a synchronous
        # ``_load_patch`` from the Open button.  Loading is the list's
        # click / Up / Down job.

        seed_slice_label = self._current_slice_result_label()
        image_layer_name = self._resolve_native_image_layer_name()
        # Lazy: create only the view(s) this context will actually show.  The
        # `_set_work_labels_visibility_for_context` call further down does it;
        # do it here too so `labels_layer_name` below names a live layer.
        self._set_work_labels_visibility_for_context(
            patch_name=self.loaded_patch_name, target_view=self.view_target,
        )
        # Seed the active view's work labels with the current prediction (if any).
        if seed_slice_label:
            self._ensure_work_labels_layer(seed_slice_label=seed_slice_label, view_name=self.selected_view)
        # Determine which labels layer name to pass to the annotator.
        if str(self.view_target or "lateral").lower() == "axial":
            labels_layer_name = self._work_labels_layer_name_for_view(
                view_name="YZ" if str(self.selected_view or "").upper() == "YZ" else "XZ"
            )
        else:
            labels_layer_name = self._work_labels_layer_name_for_view(view_name="XY")
        self._set_work_labels_visibility_for_context(patch_name=self.loaded_patch_name, target_view=self.view_target)
        self._set_active_main_labels_layer(labels_layer_name)

        if not image_layer_name or not labels_layer_name:
            # User-feedback 2026-05-24: this fires when user clicks "Open
            # Multiview" on a fresh new run (or after switching to a run
            # whose saved patch's stack file no longer exists).  The old
            # behavior was a silent console print, which users miss.
            # Now: themed dialog with actionable guidance + diagnostic
            # detail.  Also detects the most common cause (no patch
            # loaded) and tailors the message.
            no_patch_loaded = not str(getattr(self, "loaded_patch_name", "") or "")
            patches_available = len(self._available_patch_ids() or [])
            run_id = getattr(self, "active_run_id", "default")

            if no_patch_loaded:
                if patches_available:
                    title = "Pick a patch first"
                    body = (
                        f"Run '{run_id}' is active but no patch is loaded into "
                        f"the viewer yet.\n\n"
                        f"Pick a patch from the Patches list (left column) — "
                        f"the patch will load + you can then click Open "
                        f"2D annotator again to start annotating."
                    )
                else:
                    title = "No patches in this workspace"
                    body = (
                        f"Run '{run_id}' has no patches available to load.\n\n"
                        f"Use 'Browse...' next to the Patches list to add a "
                        f"training-data folder to this workspace.  Once "
                        f"patches appear in the list, select one and click "
                        f"Open 2D annotator."
                    )
            else:
                title = "Cannot open annotator"
                body = (
                    f"Patch '{getattr(self, 'loaded_patch_name', '')}' is "
                    f"recorded as loaded but the viewer has no matching "
                    f"image/labels layers.  This usually means the stack "
                    f"file moved or was deleted.\n\n"
                    f"Pick a patch from the Patches list to reload."
                )
            console_msg = (
                "Cannot open native multiview annotator: missing image/labels "
                "context. Load a patch first and ensure a working labels layer "
                "exists."
            )
            self._append_ui_log(f"[WARN] {console_msg}")
            print(console_msg)
            # Show themed info dialog so user actually sees the guidance
            try:
                from napari_epicycle.ui_shell.theme import themed_confirm
                # themed_confirm with only OK button: pass cancel_text="" to
                # hide cancel; but it doesn't support that natively.  Use a
                # custom small dialog instead.
                from qtpy.QtWidgets import (
                    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
                )
                from napari_epicycle.ui_shell.theme import (
                    _DIALOG_STYLE, _DIALOG_BTN_CANCEL,
                )
                dlg = QDialog(getattr(self, "ui", None))
                dlg.setWindowTitle(title)
                dlg.setStyleSheet(_DIALOG_STYLE)
                dlg.setMinimumWidth(440)
                layout = QVBoxLayout(dlg)
                layout.setContentsMargins(20, 16, 20, 16)
                layout.setSpacing(10)
                title_lbl = QLabel(title, parent=dlg)
                title_lbl.setStyleSheet(
                    "font-weight: bold; font-size: 13px; color: #d4a843;"
                )
                layout.addWidget(title_lbl)
                body_lbl = QLabel(body, parent=dlg)
                body_lbl.setWordWrap(True)
                body_lbl.setStyleSheet("color: #cccccc; font-size: 11px;")
                layout.addWidget(body_lbl)
                btn_row = QHBoxLayout()
                btn_row.addStretch(1)
                ok_btn = QPushButton("OK", parent=dlg)
                ok_btn.setStyleSheet(_DIALOG_BTN_CANCEL)
                ok_btn.setDefault(True)
                ok_btn.clicked.connect(dlg.accept)
                btn_row.addWidget(ok_btn)
                layout.addLayout(btn_row)
                dlg.exec_()
            except Exception as _dlg_err:
                print(f"[WARN] could not show open-annotator info dialog: {_dlg_err}")
            return

        ok, message = launch_hitl_2d_annotator(
            self.viewer,
            image_layer_name=image_layer_name,
            labels_layer_name=labels_layer_name,
            patch_ids=self._available_patch_ids(),
            active_patch_id=str(self.loaded_patch_name or ""),
            target_view=str(self.view_target or "lateral"),
            selected_view=str(self.selected_view or "XY"),
            switch_patch_callback=self._switch_patch_context_from_native,
            segment_slice_callback=self._segment_current_slice_from_native,
            overlay_slice_callback=self._fetch_slice_overlay_from_native,
            accept_slice_callback=self._accept_native_annotation,
        )

        if ok:
            self._append_ui_log(f"[CURATION] {message}")
            # Wire the draw callback for curating status detection.
            try:
                from napari_epicycle.annotator.launcher import _find_dock_by_title
                dock = _find_dock_by_title(self.viewer, "EpiCycle 2D Training Annotator")
                if dock is not None:
                    widget = dock.widget() if hasattr(dock, "widget") else dock
                    if hasattr(widget, "_on_slice_edited_callback"):
                        widget._on_slice_edited_callback = self._on_annotator_draw_complete
                    # Edit-states (2026-05-27): inject the status callback so
                    # the annotator's dims-scroll soft-lock visuals know each
                    # slice's status.  Then prime the visuals once.
                    if hasattr(widget, "_slice_status_callback"):
                        widget._slice_status_callback = self._status_for_view_slice
                    if hasattr(widget, "refresh_edit_state_visuals"):
                        try:
                            widget.refresh_edit_state_visuals()
                        except Exception:
                            pass
            except Exception:
                pass
        else:
            self._append_ui_log(f"[WARN] {message}")
        print(message)

    def _create_empty_mask_seed(self):
        self._ensure_runtime_defaults()
        self._exit_refine_mode()
        slice_label, input_stack = self._collect_current_slice_input()
        if slice_label is None or input_stack is None:
            return

        mask_shape = input_stack.shape[1:]
        empty_mask = np.zeros(mask_shape, dtype=np.int32)
        empty_flow = np.zeros((*mask_shape, 3), dtype=np.float32)
        empty_prob = np.zeros(mask_shape, dtype=np.float32)

        self.segmentation_manager.data_names[slice_label] = True
        self.segmentation_manager.data_inputs[slice_label] = input_stack
        self.segmentation_manager.data_masks[slice_label] = empty_mask
        self.segmentation_manager.data_flows[slice_label] = empty_flow
        self.segmentation_manager.data_cellprobs[slice_label] = empty_prob
        # Use [0] for grayscale to avoid IndexError; [0, 1] for multi-channel.
        n_ch = input_stack.shape[0] if input_stack.ndim >= 3 else 1
        rgb_indices = [0] if n_ch == 1 else [0, 1]
        self.segmentation_manager.data_inputs_rgb[slice_label] = self.image_manager.convert_to_rgb(
            image=input_stack,
            channel_indices=rgb_indices,
            normalize=True,
            is_stack=False,
        )

        if not hasattr(self, "segmented_slice_list"):
            self.segmented_slice_list = set()
        if slice_label not in self.segmented_slice_list:
            self.ui.segment_slice_data_list.addItem(slice_label)
            self.segmented_slice_list.add(slice_label)

        self._touch_sample(slice_label, status="predicted", view=self.selected_view, origin="cellpose")
        self._refresh_sample_queue()
        self._refresh_round_status()
        # Persist predicted record so it survives napari restart.
        self._schedule_state_save()
        self._append_ui_log(f"[SEGMENT] Created empty manual seed for {slice_label}.")

    # Run segmentation
    def _run_segmentation(self):
        self._ensure_runtime_defaults()
        """
        Segments the currently viewed 2D slice from all active image layers in the viewer.
        A single slice label is tracked in the QWidget list, e.g., 'patch2_XY_Z100'.
        Under the hood, the stacked channels from active layers are prepared for segmentation.
        """
        # exit refine mode if an image is going to be segmented
        self._exit_refine_mode()

        resolved = self._apply_model_source_selection(emit_log=False)
        if not resolved.get("can_segment", False):
            print("No segmentation model is active. Use Create Empty Mask (Manual/SAM) or switch model source.")
            return

        self._ensure_segmentation_config_defaults()
        self._apply_inference_params_from_ui()

        slice_label, input_stack = self._collect_current_slice_input()
        if slice_label is None or input_stack is None:
            return

        # Auto-adjust channels for single-channel input.
        self._adjust_channels_for_input(input_stack)

        # Guard: warn if the slice is already curated OR carries hand edits
        # (main panel path).
        if not self._confirm_resegment(slice_label):
            return

        # Brick D / journey #6 (2026-09-07): a predict is SYNCHRONOUS and
        # measurably slow (real workspace: ~1.6 s cold, ~270 ms warm for one
        # 256x256 slice), so it gets a busy indicator; and it used to fail
        # into `print` alone, so a bad checkpoint looked like "nothing
        # happened".  Both fixed here.
        _cfg = getattr(self, "current_config", {}) or {}
        _ckpt = str(resolved.get("checkpoint_path", "") or "").strip()
        _model_desc = _ckpt or str(_cfg.get("model_type", "") or "(unset)")
        _chan_desc = (f"{_cfg.get('channel1', '?')} / "
                      f"{_cfg.get('channel2', '?')}")
        self._begin_busy(f"[SEGMENT] Predicting {slice_label} ...")
        # Run the segmentation here
        try:
            # Set model preference if not initialized
            if self.segmentation_manager.segmenter is None:
                self.segmentation_manager.set_model_preference(self.current_config, working_dir=self.working_dir)

            # Run segmentation and store in manager -- should save all results
            masks, flows, cell_probs = self.segmentation_manager.segment_image(
                input_stack,
                self.current_config,
                slice_label
            )

            # save the input image as rgb and store inside the manager too
            # Use [0] for grayscale to avoid IndexError; [0, 1] for multi-channel.
            n_ch = input_stack.shape[0] if input_stack.ndim >= 3 else 1
            rgb_indices = [0] if n_ch == 1 else [0, 1]
            self.segmentation_manager.data_inputs_rgb[slice_label] = self.image_manager.convert_to_rgb(
                image=input_stack,
                channel_indices=rgb_indices,
                normalize=True,
                is_stack=False
            )

        except Exception as e:
            # Release the busy state BEFORE anything blocking: a modal shown
            # under a wait cursor reads as a hang.
            self._end_busy()
            self._restore_status_after_failed_segment(slice_label)
            headline = self._training_error_headline(e)
            print(f"Segmentation failed: {type(e).__name__}: {e}")
            self._append_ui_log(
                f"[SEGMENT FAILED] {slice_label}: {headline}"
            )
            self._append_ui_log(
                f"[SEGMENT FAILED] model: {_model_desc}"
            )
            self._append_ui_log(
                f"[SEGMENT FAILED] channels: {_chan_desc}"
            )
            self._show_modal_error(
                "Segmentation failed",
                headline,
                f"Model: {_model_desc}  |  Channels: {_chan_desc}.  "
                f"See the training log for details.",
            )
            return
        # Success: the demotion to `predicted` was correct, so drop the stash.
        self._resegment_prior_status = None
        self._end_busy()

        # Track segmented slice name for use in view results list -- if not attribute called 
        # segmented_slice_list -- create one
        if not hasattr(self, "segmented_slice_list"):
            self.segmented_slice_list = set()

        # Store the results slice_label and results -- add results label name to the segmented_slice_list
        if slice_label not in self.segmented_slice_list:
            self.ui.segment_slice_data_list.addItem(slice_label)
            self.segmented_slice_list.add(slice_label)
        self._touch_sample(slice_label, status="predicted", view=self.selected_view, origin="cellpose")

        # Immediately insert predicted mask into work_labels volume so the
        # user sees it in real time (no separate "View" step needed).
        try:
            mask2d = self.segmentation_manager.get_masks(slice_label)
            if mask2d is not None:
                view_name = self._infer_view_from_label(slice_label)
                s_idx = self._infer_slice_index(slice_label)
                if s_idx >= 0:
                    labels_layer_name = self._work_labels_layer_name_for_view(view_name=view_name)
                    labels_layer = self._layer_by_name(labels_layer_name)
                    if labels_layer is not None:
                        volume = np.asarray(labels_layer.data)
                        self._insert_slice_mask_with_unique_labels(
                            volume, np.asarray(mask2d, dtype=np.int32), view_name, s_idx,
                        )
                        labels_layer.data = volume
                        labels_layer.refresh()
        except Exception as exc:
            self._append_ui_log(f"[WARN] Could not insert prediction into volume: {exc}")

        # Store input stack in segmentation_manager so _persist_curated_training_pair
        # can write both img + mask TIFFs to disk.  segment_image() stores data_masks
        # but NOT data_inputs — only data_inputs_rgb is set (line above).
        if input_stack is not None:
            self.segmentation_manager.data_inputs[slice_label] = input_stack

        # Persist predicted mask to disk so it survives napari restart.
        # This allows direct Accept on predicted slices after reload.
        _persisted_predicted = False
        try:
            persisted = self._persist_curated_training_pair(slice_label)
            if persisted:
                _persisted_predicted = True
        except Exception as exc:
            print(f"[ERROR] predicted pair persist failed for {slice_label!r}: "
                  f"{type(exc).__name__}: {exc} "
                  f"(prediction will be lost on session close)")

        self._refresh_sample_queue()
        self._refresh_round_status()
        # 2026-05-25 audit-2 H1 fix: predicted-pair persist creates a new
        # *_img.tif in accepted/ — pair count goes up.  Without this
        # refresh, the Run dropdown stays stale until the next picker
        # refresh trigger (stage/discard/switch/etc.).
        if _persisted_predicted:
            try:
                self._refresh_run_picker()
            except Exception as e:
                pass
        # Persist predicted record so it survives napari restart.
        # Without this, predicted sample_records live only in memory until
        # the next save trigger (Accept, Stage, Train); closing napari
        # before any of those drops them.
        self._schedule_state_save()
        self._append_ui_log(f"[SEGMENT] Prediction for {slice_label} applied to volume.")

    # View selected segmentation for a selected slice
    def _view_selected_results(self):
        """Navigate the native multiview annotator to the selected prediction slice.

        Writes the predicted mask into the work_labels_3d volume so it is
        visible as a label overlay in the multiviewer, then opens/raises the
        native annotator dock and scrolls to the correct slice index.
        """
        slice_label = self._parse_queue_selected_sample_id()
        if not slice_label:
            self._append_ui_log("[WARN] No slice selected — select a prediction first.")
            return
        view_name = self._infer_view_from_label(slice_label)
        slice_index = self._infer_slice_index(slice_label)

        if slice_index < 0:
            self._append_ui_log(f"[WARN] Cannot parse slice index from '{slice_label}'.")
            return

        # Extract patch id from slice_label (format: {patch}_{view}_{index}).
        parts = str(slice_label).rsplit("_", 2)
        target_patch = parts[0] if len(parts) >= 3 else str(self.loaded_patch_name or "")

        # If the prediction belongs to a different patch, switch to it first.
        if target_patch and target_patch != str(self.loaded_patch_name or ""):
            selected = self._select_patch_item(target_patch)
            if selected:
                self._load_patch(force_axis=view_name)
            else:
                self._append_ui_log(f"[WARN] Cannot switch to patch '{target_patch}' for viewing.")

        # Switch view target to match the slice orientation.
        target = "lateral" if view_name == "XY" else "axial"
        if str(self.view_target or "").lower() != target:
            self.view_target = target
            self.selected_view = view_name
            self._set_view_radio(view_name)

        # Write predicted mask into work_labels_3d so it shows in the multiviewer.
        mask = self.segmentation_manager.get_masks(slice_label)
        if mask is not None:
            labels_layer_name = self._work_labels_layer_name_for_view(
                patch_name=self.loaded_patch_name, view_name=view_name,
            )
            labels_layer = self._layer_by_name(labels_layer_name)
            if labels_layer is not None:
                try:
                    volume = np.asarray(labels_layer.data)
                    self._insert_slice_mask_with_unique_labels(
                        volume, np.asarray(mask, dtype=np.int32), view_name, slice_index,
                    )
                    labels_layer.data = volume
                    self._sync_selected_label_to_volume(labels_layer)
                    labels_layer.refresh()
                except Exception as exc:
                    self._append_ui_log(f"[WARN] Could not write mask to volume: {exc}")

        # Ensure native annotator is open and navigate to the slice.
        self._launch_multiview_annotator()
        self._navigate_native_annotator_to_slice(view_name, slice_index)

        self._append_ui_log(f"[REVIEW] Viewing {slice_label} in native multiview annotator.")

    def _navigate_native_annotator_to_slice(self, orientation, slice_index):
        """Find the native annotator dock and navigate it to a specific slice."""
        from napari_epicycle.annotator.launcher import _resolve_existing_dock

        dock_name = "EpiCycle 2D Training Annotator"
        dock = _resolve_existing_dock(self.viewer, dock_name)
        if dock is None:
            return
        widget = dock.widget() if hasattr(dock, "widget") else dock
        if hasattr(widget, "navigate_to_slice"):
            widget.navigate_to_slice(orientation, slice_index)


    def _load_sam_model(self): 
        """
            Should load the necessary SA model selected by the user model should only be loaded once
            or when updated different model to use.
        """
        model_type = self.ui.select_sam_model_combo.currentText()
        device = "cuda" if self.ui.sam_device_gpu.isChecked() else "cpu"

        # Should load the model
        self.interactive_manager.initialize_segmenter(model_type=model_type, device=device)


    def _refine_segmentation(self):
        if self.in_refine_mode:
            current_target = self.ui.segment_slice_data_list.currentItem().text()
            if self.active_refine_slice == current_target:
                print("[INFO] Already in rfine mode for this slice")
                return
            else:
                print("[INFO] Switching refine target. Cleaning up previous refinement session")
                self._exit_refine_mode()

        # Reset the logits for the segmenter
        self.interactive_manager.reset()

        # Check if user wants to save raw predicted masks for comparisons

        # Step 1: Get selected slice
        selected = self.ui.segment_slice_data_list.selectedItems()

        if not selected:
            print("[WARN] No slice selected.")
            return
        slice_label = selected[0].text()
        self.curation_sessions.ensure_session(
            source_artifact=f"hitl_slice:{slice_label}",
            workflow_name="sam_refine",
        )


        try:
            # get the corresponding rgb image for the slice_label to be sent to SAM
            rgb_image = self.segmentation_manager.get_rgb_input_stack(slice_label)
            if rgb_image is None:
                print(f"[WARN] No input image found for {slice_label}.")
                return
        except Exception as e:
            print(f"[ERROR] Failed to retrieve RGB image for {slice_label} because of the following exception {e}")
            return

        # set the rgb image to be fed in to SAM for image embeddings
        self.interactive_manager.set_image(rgb_image)

        # Step 4: Create prompt layers if needed
        if "sam_points" not in self.viewer.layers:
            self.viewer.add_points(name="sam_points", ndim=2, size=8, face_color="white", symbol="ring")
        if "sam_boxes" not in self.viewer.layers:
            self.viewer.add_shapes(name="sam_boxes", shape_type="rectangle", edge_color="green", face_color='transparent')
        if "sam_pred_masks" not in self.viewer.layers:
            shape = rgb_image.shape[:2] # make this layer have the same spatial dimension as teh input image
            self.viewer.add_labels(np.zeros(shape, dtype=np.int32), name="sam_pred_masks", opacity=0.5)
        llm = getattr(self, "layer_lifecycle_manager", None)
        if llm is not None:
            for sam_name in ("sam_points", "sam_boxes", "sam_pred_masks"):
                llm.register_existing(sam_name, "step2/annotator")

        # set to refinement mode after setting up the layers for sam refinement
        self.in_refine_mode = True
        self.active_refine_slice = slice_label

        # Step 5: Prompt update handler
        def on_prompt_change(event=None):
            # SAM expects XY currenlyt everything is in YZ cause napari -- make sure to reshape
            points = np.array(self.viewer.layers["sam_points"].data)[:,::-1]
            boxes = self.viewer.layers["sam_boxes"].data

            if len(points) == 0 and len(boxes) == 0:
                return

            # Determine label from color
            input_labels = np.ones(len(points), dtype=np.int64)
            colors = np.array(self.viewer.layers["sam_points"].face_color)
            if len(colors) > 0:
                is_white = np.all(np.isclose(colors, [1, 1, 1, 1], atol=1e-2), axis=1)
                input_labels = np.where(is_white, 1, 0)

            input_boxes = None
            if len(boxes) > 0:
                box = boxes[-1]
                # respect coordinate for SAM
                y0, x0 = box[0]
                y1, x1 = box[2]
                input_boxes = np.array([[x0, y0, x1, y1]])

            seg = self.interactive_manager.segmenter
            if seg.last_mask_logits is not None:
                masks, _ = seg.predict_with_logits(
                    input_points=points,
                    input_labels=input_labels,
                    input_boxes=input_boxes,
                    multimask_output=False
                )
            else:
                masks, _, _ = seg.predict(
                    input_points=points,
                    input_labels=input_labels,
                    input_boxes=input_boxes,
                    multimask_output=False
                )

            mask = (masks[0] > 0.5).astype(np.int32) # threshold the masks
            self.viewer.layers["sam_pred_masks"].data = mask

        # Step 6: Connect event listener (only once)
        points_layer = self.viewer.layers["sam_points"]
        if not hasattr(points_layer, "connected"):
            points_layer.events.data.connect(on_prompt_change)
            points_layer.connected = True
        boxes_layer = self.viewer.layers["sam_boxes"]
        if not hasattr(boxes_layer, "connected"):
            boxes_layer.events.data.connect(on_prompt_change)
            boxes_layer.connected = True


    def _accept_annotation(self):
        """Batch accept all selected slices from the slice list.

        Reads the 2D mask from the work_labels volume for each selected
        slice, stores in segmentation_manager, persists to disk, and
        marks as accepted.
        """
        selected = self._parse_all_selected_sample_ids()
        if not selected:
            self._append_ui_log("[WARN] No slice selected to accept.")
            return

        accepted_count = 0
        for slice_label in selected:
            view_name = self._infer_view_from_label(slice_label)
            s_idx = self._infer_slice_index(slice_label)
            if s_idx < 0:
                continue

            # Read mask from work_labels volume (includes user edits).
            labels_layer_name = self._work_labels_layer_name_for_view(view_name=view_name)
            labels_layer = self._layer_by_name(labels_layer_name)
            if labels_layer is None:
                continue
            volume = np.asarray(labels_layer.data)
            if volume.ndim != 3:
                continue

            if view_name == "XY" and s_idx < volume.shape[0]:
                mask2d = volume[s_idx, :, :].copy()
            elif view_name == "XZ" and s_idx < volume.shape[1]:
                mask2d = volume[:, s_idx, :].copy()
            elif view_name == "YZ" and s_idx < volume.shape[2]:
                mask2d = volume[:, :, s_idx].copy()
            else:
                continue

            if np.max(mask2d) == 0:
                self._append_ui_log(f"[WARN] Slice {slice_label} has empty mask — skipped.")
                continue

            # Store in segmentation_manager.
            self.segmentation_manager.data_masks[slice_label] = mask2d.astype(np.int32)

            # Extract input image for training pair.
            source_layers = self._get_patch_view_image_layers(view_name=view_name)
            if not source_layers:
                source_layers = self._get_any_patch_view_image_layers()
            if source_layers:
                channels = []
                for src in source_layers:
                    src_vol = np.asarray(src.data)
                    if view_name == "XY" and s_idx < src_vol.shape[0]:
                        channels.append(src_vol[s_idx])
                    elif view_name == "XZ" and s_idx < src_vol.shape[1]:
                        channels.append(src_vol[:, s_idx, :])
                    elif view_name == "YZ" and s_idx < src_vol.shape[2]:
                        channels.append(src_vol[:, :, s_idx])
                if channels:
                    self.segmentation_manager.data_inputs[slice_label] = np.stack(channels)

            accepted_voxels = int(np.sum(mask2d > 0))
            next_label = int(mask2d.max())
            self.curation_sessions.record_accept(
                slice_label=slice_label,
                label_id=int(next_label),
                voxel_count=accepted_voxels,
            )

            self._touch_sample(slice_label, status="accepted", view=view_name)
            self._persist_curated_training_pair(slice_label)
            accepted_count += 1
        self._sync_data_to_add_list()
        self._refresh_sample_queue()
        self._refresh_round_status()
        # Edit-states: accepted slices flip to the emerald soft-lock visual.
        self._refresh_annotator_edit_state_visuals()
        if accepted_count:
            self._append_ui_log(f"[CURATION] Accepted {accepted_count} slice(s). Finalize to add to training.")
            # 2026-05-25 audit-2 H1 fix: SAM-assist accept persists N
            # new pairs to disk via the loop above.  Bump the dropdown
            # pair counter once post-loop (not N times).
            try:
                self._refresh_run_picker()
            except Exception as e:
                pass
        self._schedule_state_save()

        # Clean up temporary prediction layers (if they exist).
        if "sam_pred_masks" in self.viewer.layers:
            self.viewer.layers["sam_pred_masks"].data[:] = 0
        if "sam_points" in self.viewer.layers:
            self.viewer.layers["sam_points"].data = []
        if "sam_boxes" in self.viewer.layers:
            self.viewer.layers["sam_boxes"].data = []

        # Reset logits so future prompts start clean
        if self.interactive_manager.segmenter:
            self.interactive_manager.segmenter.reset_logits()

    def _accept_native_annotation(self, orientation, slice_index=None):
        """Accept the current native annotator slice edit into the curation pipeline.

        Reads the 2D mask from the work_labels_3d volume and the corresponding
        2D input image, registers both in segmentation_manager, and marks the
        sample as 'accepted'.
        """
        plane = str(orientation or "").upper()
        if plane not in {"XY", "XZ", "YZ"}:
            return {"ok": False, "message": f"Unsupported orientation: {orientation}"}

        self._ensure_runtime_defaults()
        patch = str(self.loaded_patch_name or "").strip()
        if not patch:
            return {"ok": False, "message": "No active patch context."}

        if slice_index is None:
            return {"ok": False, "message": "No slice index provided."}
        try:
            idx = int(slice_index)
        except (TypeError, ValueError):
            return {"ok": False, "message": f"Invalid slice index: {slice_index}"}

        slice_label = f"{patch}_{plane}_{idx}"

        # --- Extract 2D mask from work_labels volume ---
        labels_layer_name = self._work_labels_layer_name_for_view(patch_name=patch, view_name=plane)
        labels_layer = self._layer_by_name(labels_layer_name)
        if labels_layer is None:
            return {"ok": False, "message": f"Labels layer not found: {labels_layer_name}"}
        volume = np.asarray(labels_layer.data)
        if volume.ndim != 3:
            return {"ok": False, "message": f"Labels volume is not 3D: shape={volume.shape}"}

        if plane == "XY":
            if idx < 0 or idx >= volume.shape[0]:
                return {"ok": False, "message": f"Slice index {idx} out of range for XY (0..{volume.shape[0]-1})."}
            mask2d = volume[idx, :, :].copy()
        elif plane == "XZ":
            if idx < 0 or idx >= volume.shape[1]:
                return {"ok": False, "message": f"Slice index {idx} out of range for XZ (0..{volume.shape[1]-1})."}
            mask2d = volume[:, idx, :].copy()
        elif plane == "YZ":
            if idx < 0 or idx >= volume.shape[2]:
                return {"ok": False, "message": f"Slice index {idx} out of range for YZ (0..{volume.shape[2]-1})."}
            mask2d = volume[:, :, idx].copy()
        else:
            return {"ok": False, "message": f"Unsupported plane: {plane}"}

        if int(np.max(mask2d)) == 0:
            return {"ok": False, "message": f"Empty mask at {slice_label}. Draw annotations first."}

        # --- Check for duplicates: skip if already accepted/staged and mask unchanged ---
        existing_record = self.sample_records.get(slice_label, {})
        existing_status = existing_record.get("status", "")
        existing_mask = self.segmentation_manager.get_masks(slice_label)
        mask_changed = True
        if existing_mask is not None:
            try:
                mask_changed = not np.array_equal(np.asarray(existing_mask), mask2d)
            except Exception:
                mask_changed = True

        if not mask_changed and existing_status in ("accepted", "staged"):
            return {
                "ok": True,
                "message": f"{slice_label} unchanged, already {existing_status}. Skipped.",
                "slice_label": slice_label,
                "skipped": True,
            }

        # --- Extract 2D input image from image layers ---
        source_layers = self._get_patch_view_image_layers(view_name=plane)
        if not source_layers:
            source_layers = self._get_any_patch_view_image_layers()

        collected_channels = []
        for layer in source_layers:
            data = np.asarray(layer.data)
            if data.ndim != 3:
                continue
            if plane == "XY":
                ch = data[idx, :, :]
            elif plane == "XZ":
                ch = data[:, idx, :]
            else:
                ch = data[:, :, idx]
            if ch.ndim == 3 and ch.shape[-1] in (3, 4):
                ch = ch[..., :3].mean(axis=-1)
            collected_channels.append(ch)

        if not collected_channels:
            return {"ok": False, "message": f"No image data for {slice_label}."}
        input_stack = np.stack(collected_channels, axis=0)

        # --- Register in segmentation manager (always update with latest data) ---
        self.segmentation_manager.data_names[slice_label] = True
        self.segmentation_manager.data_inputs[slice_label] = input_stack
        self.segmentation_manager.data_masks[slice_label] = mask2d.astype(np.int32)

        # Add to predicted list if not already there.
        if not hasattr(self, "segmented_slice_list"):
            self.segmented_slice_list = set()
        if slice_label not in self.segmented_slice_list:
            ui = getattr(self, "ui", None)
            if ui is not None and hasattr(ui, "segment_slice_data_list"):
                ui.segment_slice_data_list.addItem(slice_label)
            self.segmented_slice_list.add(slice_label)

        # --- Status logic: don't demote staged slices ---
        # If already staged and mask changed, update the data but keep staged status.
        # If already accepted, re-accept with updated data.
        # If new or predicted, promote to accepted.
        if existing_status == "staged":
            # Data updated in-place above; re-persist but preserve staged status.
            view_name = self._infer_view_from_label(slice_label)
            persisted = self._persist_curated_training_pair(slice_label)
        else:
            view_name = self._infer_view_from_label(slice_label)
            self._touch_sample(slice_label, status="accepted", view=view_name)
            persisted = self._persist_curated_training_pair(slice_label)
        self._sync_data_to_add_list()
        self._refresh_sample_queue()
        self._refresh_round_status()
        # Edit-states: the just-accepted slice flips to emerald soft-lock.
        self._refresh_annotator_edit_state_visuals()

        msg = f"Accepted {slice_label} -> {self._bucket_for_view(view_name)}."
        self._append_ui_log(f"[CURATION] {msg}")
        if persisted:
            self._append_ui_log(
                f"[DATA] Saved curated pair: {persisted.get('img_uri', '-')}, {persisted.get('mask_uri', '-')}"
            )
            # 2026-05-25 audit-2 H1 fix: 2D-annotator accept persisted a
            # new *_img.tif under accepted/.  This is the MAIN curation
            # path — without this refresh, the Run dropdown pair counter
            # stays stale during normal HITL workflow.
            try:
                self._refresh_run_picker()
            except Exception as e:
                pass
        self._schedule_state_save()
        return {"ok": True, "message": msg, "slice_label": slice_label}

    def _reject_annotation(self):
        """Batch discard all selected slices — clears masks from volume and records."""
        selected = self._parse_all_selected_sample_ids()
        if not selected:
            self._append_ui_log("[WARN] No slice selected to discard.")
            return

        # Guard: check if any finalized slices are in the selection.
        has_finalized = any(
            self.sample_records.get(sl, {}).get("status") in ("staged", "committed")
            for sl in selected
        )
        if has_finalized:
            from qtpy.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self.ui,
                "Discard finalized slices?",
                f"{len(selected)} slice(s) selected, some are finalized.\n"
                "Discarding will remove them from the training set.\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        discarded = 0
        for slice_label in selected:
            # Clear the mask from the work_labels volume.
            view_name = self._infer_view_from_label(slice_label)
            s_idx = self._infer_slice_index(slice_label)
            if s_idx >= 0:
                labels_layer_name = self._work_labels_layer_name_for_view(view_name=view_name)
                labels_layer = self._layer_by_name(labels_layer_name)
                if labels_layer is not None:
                    volume = np.asarray(labels_layer.data)
                    if view_name == "XY" and s_idx < volume.shape[0]:
                        volume[s_idx, :, :] = 0
                    elif view_name == "XZ" and s_idx < volume.shape[1]:
                        volume[:, s_idx, :] = 0
                    elif view_name == "YZ" and s_idx < volume.shape[2]:
                        volume[:, :, s_idx] = 0
                    labels_layer.data = volume
                    labels_layer.refresh()
            else:
                pass

            # Clear from segmentation_manager.
            if hasattr(self.segmentation_manager, "data_masks"):
                self.segmentation_manager.data_masks.pop(slice_label, None)
            if hasattr(self.segmentation_manager, "data_inputs"):
                self.segmentation_manager.data_inputs.pop(slice_label, None)

            # Delete persisted TIFF files from disk so they don't get
            # resurrected by _reload_persisted_pairs_from_disk on restart.
            record = self.sample_records.get(slice_label, {})
            for uri_key in ("img_uri", "mask_uri"):
                fpath = str(record.get(uri_key, "")).strip()
                if fpath:
                    try:
                        p = Path(fpath)
                        if p.is_file():
                            p.unlink()
                    except Exception as exc:
                        print(f"[WARN] Discard: failed to delete "
                              f"{fpath!r}: {type(exc).__name__}: {exc}")

            # Remove from sample_records.
            was_in_records = slice_label in self.sample_records
            self.sample_records.pop(slice_label, None)
            discarded += 1

        self._refresh_sample_queue()
        self._refresh_round_status()
        if discarded:
            self._append_ui_log(f"[CURATION] Discarded {discarded} slice(s).")
            self._refresh_2d_annotator_subviewers()
            # Edit-states: a discarded slice loses its record -> its soft-lock
            # tint must clear (the buttons would otherwise stay emerald/ice
            # on a now-deleted record until the next scroll).
            self._refresh_annotator_edit_state_visuals()
            # Persist state so discarded slices don't reappear on restart.
            self._schedule_state_save()
            # 2026-05-25 Group D: bump dropdown pair count (deleted files
            # mean the active run lost N accepted pairs).
            try:
                self._refresh_run_picker()
            except Exception as e:
                pass

    def _refresh_2d_annotator_subviewers(self):
        """Ask the open 2D annotator dock (if any) to refresh its subviewers."""
        try:
            from napari_epicycle.annotator.launcher import _find_dock_by_title
            dock = _find_dock_by_title(self.viewer, "EpiCycle 2D Training Annotator")
            if dock is None:
                return
            widget = dock.widget() if hasattr(dock, "widget") else dock
            if hasattr(widget, "_refresh_label_data_in_subviewers"):
                widget._refresh_label_data_in_subviewers()
        except Exception:
            pass

    def _finalize_labels(self):
        print("[INFO] Finalizing all accepted annotations...")
        labels = self.ui.segment_slice_data_list
        num_slices = labels.count()
        if num_slices == 0:
            print("[WARN] No annotated slices found.")
            return

        final_stack = []
        n_skipped = 0

        # 2026-05-28 (label-compaction-v2): running cross-slice offset so
        # distinct cells across slices get distinct IDs in final_labels.
        # CRITICAL for the 3D view: pre-change this relied on data_masks
        # holding pre-offset (cross-slice-unique) masks.  After disk
        # compaction, a reload repopulates data_masks with COMPACT masks
        # (each starting at 1), so stacking them raw would make
        # slice-1-cell-1 and slice-5-cell-1 both label 1 → merged into one
        # object in the 3D final_labels view.  We normalize each slice to
        # 1..M (via _compact_label_ids — robust whether the input is
        # compact OR offset) then add a running offset so every slice's
        # cells occupy a unique ID range.  Result: cross-slice-unique +
        # bounded (≈ total cell count), regardless of input ID ranges.
        running_offset = 0

        for i in range(num_slices):
            item = labels.item(i)
            # 2026-05-25 fix: read sample_id from Qt.UserRole, not from
            # .text() — _add_colored_item sets text to the formatted
            # display string ("↳ XY  slice 36 ..."), not the sample_id.
            # Calling get_masks(display_string) returns None for every
            # slice, which caused the post-Finalize hang the user
            # reported (7× [WARN] cascade + empty final_stack path).
            sample_id = item.data(Qt.UserRole) if (item is not None and hasattr(item, "data")) else None
            if not sample_id:
                # Fall back to text() for any non-colored-item legacy.
                sample_id = item.text() if item is not None else ""
            slice_label = str(sample_id)

            mask = self.segmentation_manager.get_masks(slice_label)
            if mask is None:
                # 2026-05-25: this branch now should not fire for imported
                # pairs after the [IMPORT-RELOAD] populates data_masks.
                # If it fires, it's a sign reload didn't run.
                print(f"[WARN] No mask for {slice_label}. Skipping.")
                n_skipped += 1
                continue

            # Normalize to 1..M then apply the running cross-slice offset.
            mask = _compact_label_ids(mask.astype(np.int32))
            fg = mask > 0
            if np.any(fg):
                mask = mask.copy()
                mask[fg] = mask[fg] + running_offset
                running_offset = int(mask.max())
            final_stack.append(mask.astype(np.int32))

        if len(final_stack) == 0:
            print("[ERROR] No valid masks were found.")
            return

        # Stack into final 3D volume
        label_volume = np.stack(final_stack, axis=0)  # shape: (Z, H, W)

        # Add to viewer (vispy-safe: hide before remove).
        if "final_labels" in self.viewer.layers:
            self.viewer.layers["final_labels"].visible = False
            self.viewer.layers.remove("final_labels")

        self.viewer.add_labels(label_volume, name="final_labels", opacity=0.7)
        llm = getattr(self, "layer_lifecycle_manager", None)
        if llm is not None:
            llm.register_existing("final_labels", "step2/patch")
        print(f"[INFO] Final label volume added to viewer: shape={label_volume.shape}, dtype={label_volume.dtype}")
        try:
            unique_labels = [int(x) for x in np.unique(label_volume) if int(x) > 0]
            final_map = {str(lbl): int(lbl) for lbl in unique_labels}
            self.curation_sessions.finalize(final_label_id_map=final_map)
            if self.working_dir:
                persisted_path = self.curation_sessions.persist_active(self.working_dir)
                if persisted_path:
                    self.ui.training_log_output.append(f"[CURATION] Session saved: {persisted_path}")
        except Exception as e:
            self.ui.training_log_output.append(f"[WARN] Failed to persist curation session: {e}")

    def _exit_refine_mode(self):
        if self.in_refine_mode:
            print(f"[INFO] Exiting refine mode for slice {self.active_refine_slice}")
            self.interactive_manager.cleanup_prompt_layers(self.viewer) # call this to remove the sam prompt layers
            self.in_refine_mode = False
            self.active_refine_slice = None


    def _collect_training_config(self):
        """Read training config from UI widgets, falling back to self.current_config."""
        ui = self.ui
        fallback = self.current_config if isinstance(getattr(self, "current_config", None), dict) else {}

        def _text(attr, key, default=""):
            w = getattr(ui, attr, None)
            return w.text() if w is not None and hasattr(w, "text") else fallback.get(key, default)

        def _value(attr, key, default=0):
            w = getattr(ui, attr, None)
            return w.value() if w is not None and hasattr(w, "value") else fallback.get(key, default)

        def _combo(attr, key, default=""):
            w = getattr(ui, attr, None)
            return w.currentText() if w is not None and hasattr(w, "currentText") else fallback.get(key, default)

        def _checked(attr, key, default=True):
            w = getattr(ui, attr, None)
            return w.isChecked() if w is not None and hasattr(w, "isChecked") else fallback.get(key, default)

        # 2026-05-26: model_name auto-derived from active_run_id (was
        # read from model_name_line which is now removed).  Falls back
        # to whatever was previously in config (legacy saved-config
        # compat) if no active run is loaded.  The actual on-disk
        # filename is `{run_id}_{bucket}_round_NN` (controller:5618);
        # this config field is now purely a display hint for legacy
        # JSON save/load.
        _legacy_mname = str(fallback.get("model_name", "") or "")
        _derived_mname = (
            str(getattr(self, "active_run_id", "") or "") or _legacy_mname
        )
        config = {
            "model_name": _derived_mname,
            "epochs": _value("epochs_sb", "epochs", 200),
            "batch_size": _value("batch_sb", "batch_size", 8),
            "learning_rate": _value("lr_db", "learning_rate", 0.005),
            "weight_decay": _value("wd_db", "weight_decay", 1e-5),
            "min_masks_per_image": _value("minmask_sb", "min_masks_per_image", 0),
            "channel1": _combo("chan1_combo", "channel1", "0"),
            "channel2": _combo("chan2_combo", "channel2", "0"),
            # 2026-05-27 cleanup: flow_threshold / cellprob_threshold /
            # n_iterations now driven by Inference Parameters widgets only
            # (inference_flow_thresh / inference_cellprob_thresh /
            # inference_niter_dynamics).  Read from inference widgets to
            # keep the post-training-inference call site behavior
            # identical for callers that pass through this config dict.
            "flow_threshold": _value("inference_flow_thresh", "flow_threshold", 0.4),
            "cellprob_threshold": _value("inference_cellprob_thresh", "cellprob_threshold", 0.0),
            "n_iterations": _value("inference_niter_dynamics", "n_iterations", 0),
            "normalize": _checked("norm_cb", "normalize", True),
            "rescale": _checked("rescale_cb", "rescale", True),
            "compute_flows": _checked("flow_cb", "compute_flows", True),
            "device": "GPU" if _checked("device_gpu", "device", False) else "CPU",
            "model_type": _combo("model_selector", "model_type", "cyto3"),
            "starting_checkpoint": _text("starting_checkpoint_path", "starting_checkpoint", ""),
        }
        return config

    # Widgets disabled during training.  Audited 2026-05-27 (Bundle 1
    # follow-up: 3-reviewer audit caught 6 typos in this list which
    # silently fell through getattr(None) guards — same class of bug
    # F1.4 was meant to close).  Every name here is verified against
    # main_widget.py.  See test_set_training_ui_locked_widgets_exist_-
    # on_real_ui for the regression guard.
    _TRAIN_LOCK_WIDGET_ATTRS = (
        # Workspace + run controls.
        "set_workdir_browse_button",
        # Q8: the typed paths are editable too -- a path typed under the lock
        # would desync the workspace from the running round.
        "set_workdir_path",
        "load_training_path_patch",
        # Training config inputs.
        "epochs_sb",                    # was epochs_input (didn't exist)
        "starting_checkpoint_path",
        "starting_checkpoint_browse",   # was browse_checkpoint_button (didn't exist)
        "model_source_combo",
        "imported_model_path",
        "imported_model_browse_button", # was browse_imported_model_button (didn't exist)
        "curation_pretrained_combo",
        "training_mode_combo",
        # View radio (bucket choice mid-train is undefined).
        "get_view_lateral",
        "get_view_axial",
        # Bucket-state-mutation buttons + lists.
        "accept_annotation_button",
        "reject_annotation_button",
        "finalize_labels_button",
        "stage_current_view_button",
        "unstage_selected_button",
        "train_patch_list",
        "launch_multiview_button",
        "refine_results_button",
        "apply_model_source_button",
        "apply_inference_params_button",
        # NOTE: removed:
        # - model_name_input (vestigial post Phase B.2 — model name is
        #   a display-only QLabel now; nothing to lock)
        # - segment_current_slice_button (ghost — never declared on the
        #   UI; pre-existing dead `_connect_signals` ref)
        # - create_empty_mask_button (ghost — same)
    )

    def _set_training_ui_locked(self, locked: bool):
        """Disable/enable UI controls that should not change during training.

        See ``_TRAIN_LOCK_WIDGET_ATTRS`` for the canonical list.  A widget
        attr that doesn't exist on the current UI (e.g. a future rename)
        is silently skipped — defensive against UI churn.

        Step 2 training-lock extensions (2026-05-27):
        - Workflow-shell side-nav: lock_step_nav() disables the pipeline
          step buttons so the user can't navigate away from Step 2 while
          training is in flight.
        - 2D annotator drawing surfaces: if the annotator is open,
          set_training_lock() forces editable=False + pan_zoom on all
          Labels layers in the main viewer + 3 subviewers, disables the
          6 per-orient Segment/Accept buttons, and surfaces a visible
          "LOCKED (TRAINING)" mode indicator.
        Both extensions are scoped to Step 2 — Step 3 annotator is
        unaffected; workflow_shell.lock_step_nav is only called from
        this Step 2 controller.
        """
        enabled = not locked
        ui = getattr(self, "ui", None)
        if ui is None:
            return
        print(f"[TRAIN-SAFETY] _set_training_ui_locked locked={locked}")
        for attr_name in self._TRAIN_LOCK_WIDGET_ATTRS:
            widget = getattr(ui, attr_name, None)
            if widget is not None and hasattr(widget, "setEnabled"):
                widget.setEnabled(enabled)

        # Step 2 training-lock extension #1: side-nav.
        shell = getattr(self, "_workflow_shell", None)
        if shell is not None and hasattr(shell, "lock_step_nav"):
            try:
                shell.lock_step_nav(locked)
            except Exception as exc:
                print(f"[STEP2-LOCK] workflow_shell.lock_step_nav "
                      f"failed: {type(exc).__name__}: {exc}")

        # Step 2 training-lock extension #2: annotator drawing surfaces.
        annotator = self._find_open_2d_annotator()
        if annotator is not None and hasattr(annotator, "set_training_lock"):
            try:
                annotator.set_training_lock(locked)
            except Exception as exc:
                print(f"[STEP2-LOCK] Curation2DAnnotator.set_training_lock "
                      f"failed: {type(exc).__name__}: {exc}")

        # Step 2 look hook 1 (brick T): the lock lines of Session, Curation
        # and Training, the dimmed lists, the setup column greyed per widget.
        # ``round_name`` is the name built for the Writes label.
        self._train_ui_locked = bool(locked)
        try:
            fn = getattr(self.ui, "on_training_locked", None)
            if fn:
                bucket = self._current_round_key()
                fn(locked, self._training_round_name(bucket, self._current_round()),
                   bucket)
        except Exception as exc:      # noqa: BLE001 - a hook must not break the lock
            print(f"[STEP2-HOOK] on_training_locked failed: "
                  f"{type(exc).__name__}: {exc}")
        self._refresh_training_stage_line()

    def _training_round_name(self, bucket, round_num) -> str:
        """``{run}_{bucket}_round_{NN}`` -- the file name a Train of
        ``bucket`` writes (the Writes label's, ``_refresh_model_name_display``)."""
        run_id = str(getattr(self, "active_run_id", "") or "")
        return f"{run_id}_{bucket}_round_{int(round_num or 0):02d}"

    def _notify_train_finished(self) -> None:
        """Hook 6: the round's curve is in the loss plot (or was dropped)."""
        try:
            fn = getattr(self.ui, "on_train_finished", None)
            if fn:
                fn()
        except Exception as exc:      # noqa: BLE001
            print(f"[STEP2-HOOK] on_train_finished failed: "
                  f"{type(exc).__name__}: {exc}")

    def _refresh_training_stage_line(self) -> None:
        """The Training foot's idle stage line: what the next Train
        writes and from how many pairs.  Skipped while a round trains (the
        epoch hook owns the line then)."""
        foot = getattr(getattr(self, "ui", None), "training_foot", None)
        if foot is None:
            return
        try:
            # the Loss header's idle caption follows the plot (a load / switch
            # / finished round changes it; a live caption wins inside)
            foot.loss.refresh_caption()
        except Exception:      # noqa: BLE001
            pass
        if getattr(self, "_train_ui_locked", False):
            return
        try:
            from ..ui import _step2_text as _T2
            bucket = self._current_round_key()
            run_id = str(getattr(self, "active_run_id", "") or "")
            n_new = self._staged_count_for(bucket)
            prior = self._staged_prior_for(bucket)
            name = (self._training_round_name(bucket, self._current_round())
                    if (run_id and n_new > 0) else "")
            foot.set_stage_line(_T2.compose_training_stage_line(
                run=run_id, bucket=bucket, file_name=name,
                n_pairs=prior + n_new))
        except Exception as exc:      # noqa: BLE001 - a sentence must not break a refresh
            print(f"[STEP2-HOOK] training stage line failed: "
                  f"{type(exc).__name__}: {exc}")

    def _staged_slice_keys(self, bucket) -> set:
        """``(stack, VIEW, idx)`` of every slice CURRENTLY staged in ``bucket``."""
        staged_keys = set()
        for label, rec in (getattr(self, "sample_records", {}) or {}).items():
            if rec.get("status") == "staged" and rec.get("bucket") == bucket:
                key = self._slice_key_for_pair_filename(str(label))
                if key is not None:
                    staged_keys.add(key)
        return staged_keys

    def _drop_pairs_of_staged_slices(self, bucket, prior_pairs) -> list:
        """Q7 -- the ONE rule: a prior-round pair whose slice is staged now is
        stale (the slice was trained, reverted, re-edited, re-finalized), so
        the newest staged labels win and the old pair is left out.  Nothing on
        disk is touched."""
        staged_keys = self._staged_slice_keys(bucket)
        if not staged_keys:
            return list(prior_pairs)
        kept = []
        for pair in prior_pairs:
            key = self._slice_key_for_pair_filename(
                Path(str(pair.get("img_uri", ""))).name)
            if key is None or key not in staged_keys:
                kept.append(pair)
        return kept

    def _prior_pairs_for_training(self, buckets, log_skips=False) -> list:
        """The prior-round pairs handed to training, for each bucket in
        ``buckets``: the disk scan minus the stale pairs of staged slices."""
        out = []
        for _b in buckets:
            out.extend(self._drop_pairs_of_staged_slices(
                _b, self._collect_prior_round_pairs(bucket=_b, log_skips=log_skips)))
        return out

    def _prior_pair_total_excluding_staged(self, bucket, prior_pairs=None) -> int:
        """Prior-round pairs of ``bucket`` that train: the same rule as
        ``_prior_pairs_for_training`` (the "pairs in all" line and the trained
        set cannot disagree).  ``_collect_prior_round_pairs`` stays unfiltered."""
        if prior_pairs is None:
            prior_pairs = self._collect_prior_round_pairs(bucket)
        return len(self._drop_pairs_of_staged_slices(bucket, prior_pairs))

    def _staged_prior_for(self, bucket) -> int:
        """Pairs of earlier rounds for ``bucket`` (cached by the last
        ``_refresh_round_status`` -- no disk scan here)."""
        cached = getattr(self, "_staged_prior_pairs", None)
        if isinstance(cached, tuple) and cached[0] == bucket:
            return int(cached[1])
        return 0

    def _find_open_2d_annotator(self):
        """Return the Curation2DAnnotator widget if it's currently open,
        else None.

        Used by `_set_training_ui_locked` to drive the annotator's
        training-lock state.  Walks the napari window's docks looking
        for the "EpiCycle 2D Training Annotator" dock title (mirrors
        the discovery pattern in `_launch_multiview_annotator` at the
        post-launch wire-up step).
        """
        try:
            from napari_epicycle.annotator.launcher import _find_dock_by_title
        except Exception:
            return None
        try:
            dock = _find_dock_by_title(self.viewer, "EpiCycle 2D Training Annotator")
        except Exception:
            return None
        if dock is None:
            return None
        try:
            widget = dock.widget() if hasattr(dock, "widget") else dock
        except Exception:
            return None
        return widget

    def _invalidate_annotator_fill_label_cache(self):
        """fix/step2-autoinc-perf (2026-05-28): invalidate the open 2D
        annotator's enclosed-fill next-label cache.

        Called from `_insert_slice_mask_with_unique_labels` (the single
        chokepoint for label-introducing inserts — predictions, restore,
        segment).  After those add labels, the cached fill counter would be
        stale-LOW; invalidating forces a fresh global-max re-seed on the
        next enclosed-fill, so it can't hand out a colliding ID (which
        would merge two cells).  No-op if the annotator isn't open or the
        sync manager doesn't expose the method (defensive)."""
        ann = self._find_open_2d_annotator()
        if ann is None:
            return
        sm = getattr(ann, "sync_manager", None)
        if sm is not None and hasattr(sm, "invalidate_next_fill_label_cache"):
            try:
                sm.invalidate_next_fill_label_cache()
            except Exception:
                pass

    def _status_for_view_slice(self, view, idx):
        """Edit-states bridge (2026-05-27): return the sample_record status
        for the slice at (current patch, view, idx), or "" if no record.

        Injected into the open 2D annotator as `_slice_status_callback`
        so its dims-scroll soft-lock visuals can tell whether the
        currently-displayed slice is editable / accepted / finalized.
        Reads live from self.sample_records so run-switches + imports are
        reflected automatically (the dict is reloaded per run)."""
        patch = str(getattr(self, "loaded_patch_name", "") or "").strip()
        if not patch:
            return ""
        view = str(view or "").upper()
        try:
            slice_label = f"{patch}_{view}_{int(idx)}"
        except Exception:
            return ""
        rec = self.sample_records.get(slice_label)
        if not rec:
            return ""
        return str(rec.get("status", ""))

    def _refresh_annotator_edit_state_visuals(self):
        """Edit-states bridge (2026-05-27): ask the open 2D annotator to
        re-evaluate its per-orient soft-lock button visuals.

        Called after any status transition the user can't see directly
        (demote-on-draw, Revert, accept, stage) so the emerald/ice button
        tint + enable state stays in sync with sample_records.  No-op if
        the annotator isn't open or doesn't have the method (defensive
        against version skew)."""
        annotator = self._find_open_2d_annotator()
        if annotator is None:
            return
        fn = getattr(annotator, "refresh_edit_state_visuals", None)
        if callable(fn):
            try:
                fn()
            except Exception as exc:
                print(f"[EDIT-STATE] refresh_edit_state_visuals failed: "
                      f"{type(exc).__name__}: {exc}")

    def _train_model(self):
        """Validate inputs, collect config, and launch training in a background thread.

        2026-05-27 safety bundle:
        - F2.1: Flush pending curating-persist timer + state-save timer
          BEFORE reading staged-slice list, so a recent paint stroke
          (≤2s before clicking Train) doesn't silently bypass training.
        - F4.3: Pre-flight check that working_dir actually exists on
          disk — external deletion mid-session would otherwise cause
          silent path-rot.
        - F5.2: All validation (incl. the no-staged-slices gate) MUST
          run BEFORE any UI mutation.  Previous code mutated the button
          + locked widgets BEFORE the staged-slices check, leaving the
          user stuck in a fake "training in progress" UI when the
          check failed.
        """
        if not hasattr(self, "sample_records"):
            self.sample_records = {}
        if not hasattr(self, "round_backend_info"):
            self.round_backend_info = {}
        if not self.working_dir:
            self._append_ui_log("[TRAIN] Please set a working directory first.")
            return
        # F4.3: pre-flight check — working_dir must exist on disk.
        try:
            from pathlib import Path as _PathCheck
            if not _PathCheck(self.working_dir).is_dir():
                self._append_ui_log(
                    f"[TRAIN] Working directory does not exist on disk: {self.working_dir}. "
                    "Check that the folder hasn't been moved or deleted."
                )
                print(f"[TRAIN-SAFETY] F4.3 abort — working_dir missing: {self.working_dir!r}")
                return
        except Exception as exc:
            self._append_ui_log(f"[TRAIN] Working directory pre-flight check failed: {exc}")
            return
        # Auto-collect config from UI before every training run.
        self.current_config = self._collect_training_config()
        if not self.current_config:
            self._append_ui_log("[TRAIN] Please set or load a training config before training.")
            return
        model_name = self.current_config.get("model_name", "").strip()
        if not model_name:
            self._append_ui_log("[TRAIN] Please set a non-empty model name before training.")
            return

        # Prevent double-launch.
        if getattr(self, "_training_thread", None) is not None and self._training_thread.isRunning():
            self._append_ui_log("[TRAIN] Training is already running.")
            return

        # F2.1: flush pending debounced state-save + curating-persist
        # timers BEFORE reading the staged-slice list.  Otherwise a
        # paint stroke <2s before clicking Train silently lands on disk
        # AFTER staged_slice_labels is collected — the trainer doesn't
        # see the paint, but the file appears in accepted/ anyway.
        # Mirrors _switch_active_run + _browse_workdir: flush the pending
        # debounced state-save and curating-persist timers before reading the
        # staged-slice list, otherwise a late save lands after the list was
        # collected.
        #
        # Bundle 1 follow-up (R3 audit): _flush_pending_state_timers
        # ONLY stops timers — it doesn't actually save.  Plus the
        # nested _flush_curating_persist_pending re-schedules a NEW
        # state-save timer at the end.  Both gaps fixed below:
        #   1. Explicit _save_internal_state() after the flush — mirrors
        #      shutdown()'s pattern.  Persists the last-500ms in-memory
        #      sample_records updates before the trainer reads them.
        #   2. Stop the state-save timer AGAIN (after the curating
        #      flush re-armed it) so it doesn't fire mid-training.
        try:
            self._flush_pending_state_timers()
            try:
                self._save_internal_state()
                print("[TRAIN-SAFETY] F2.1 explicit state save after flush")
            except Exception as exc:
                print(f"[TRAIN-SAFETY] F2.1 explicit save FAILED (non-fatal): {exc}")
            # The curating-persist flush calls _schedule_state_save at
            # the end → re-arms the timer.  Stop it again.
            _state_timer = getattr(self, "_state_save_timer", None)
            if (
                _state_timer is not None
                and hasattr(_state_timer, "isActive")
                and _state_timer.isActive()
            ):
                _state_timer.stop()
                print("[TRAIN-SAFETY] F2.1 re-stopped state-save timer (re-armed by curating flush)")
            print("[TRAIN-SAFETY] F2.1 flushed pending state/persist timers + saved + drained")
        except Exception as exc:
            print(f"[TRAIN-SAFETY] F2.1 flush timers FAILED (non-fatal): {exc}")

        # F5.2: gather staged slices BEFORE any UI mutation.  If the
        # check fails we want to early-return without the user seeing
        # a fake "training in progress" state.
        self._sync_data_to_add_list()
        staged_slice_labels = [
            self.ui.data_to_add_list.item(i).text()
            for i in range(self.ui.data_to_add_list.count())
        ]
        if not staged_slice_labels:
            self._append_ui_log(
                "[TRAIN] No staged slices. Accept slices, then click Finalize "
                "to stage them, before training."
            )
            print(f"[TRAIN-SAFETY] F5.2 no-staged-slices early return — UI not mutated")
            return

        # ─── All validation passed — UI mutation begins here ──────────
        self.ui.training_progress_bar.setRange(0, 100)
        self.ui.training_progress_bar.setValue(5)
        # Step 2 look (brick T, seam (a)): Train greys and Cancel lights --
        # two buttons, nothing renamed or re-connected (a disabled Train plus
        # the ``isRunning`` check above stops a double launch).
        # ``train_cancel_btn`` is deliberately NOT in _TRAIN_LOCK_WIDGET_ATTRS
        # (it must stay live under the lock; its state is driven here).
        btn = self.ui.update_and_train_button
        if hasattr(btn, "setEnabled"):
            btn.setEnabled(False)
        cancel_btn = getattr(self.ui, "train_cancel_btn", None)
        if cancel_btn is not None:
            cancel_btn.setEnabled(True)
        self._set_training_ui_locked(True)

        bucket = self._current_round_key()
        round_num = self._current_round()
        round_id = f"round_{int(round_num):02d}"
        round_key = f"{bucket}_{round_num}"

        # Show which view is being trained.
        self._append_ui_log(f"[TRAIN] Starting {bucket} {round_id} (key={round_key})...")

        training_mode = getattr(self, "training_mode", "cumulative")

        # Phase C.5 fix (2026-05-25, audit 3 HIGH-2): collect prior
        # pairs + checkpoints PER-BUCKET that will train.  Pre-fix
        # called both helpers without args → they read `selected_view`
        # and returned data for ONE bucket only.  In Mode=Both, the
        # axial iteration of the per-view loop got empty prior_pairs
        # + empty prior_checkpoints → silent scratch training even
        # when axial_N-1 weights existed on disk.  Pre-existing bug;
        # Phase B.1's per-bucket counter divergence amplified the
        # visible impact (more cumulative weights now exist that get
        # silently ignored).
        buckets_to_train: set = set()
        for sl in staged_slice_labels:
            rec = self.sample_records.get(sl, {})
            v = str(rec.get("view", "")).upper()
            if v == "XY":
                buckets_to_train.add("lateral")
            elif v in ("XZ", "YZ"):
                buckets_to_train.add("axial")
        if not buckets_to_train:
            # Fall back to selected_view's bucket if we can't determine
            # from staged labels (defensive).
            buckets_to_train.add(self._current_round_key())

        # Always collect prior training pairs — both modes use all
        # accumulated data.  The difference is only the starting
        # checkpoint: cumulative fine-tunes the prior model, from_scratch
        # trains a fresh model on the same data.
        # Q7 (2026-10-13): stale pairs of re-staged slices are dropped here, so
        # the trainer, its round record and the display total share one rule.
        prior_pairs = self._prior_pairs_for_training(
            buckets_to_train, log_skips=True)
        if prior_pairs:
            self._append_ui_log(
                f"[TRAIN] {len(prior_pairs)} prior pair(s) from disk "
                f"(buckets={sorted(buckets_to_train)})."
            )

        # Release-hardening 2026-09-07 fix pass: resolve the prior
        # checkpoint PER BUCKET, using that bucket's OWN training mode.
        # Pre-fix this read the flat `training_mode` once and applied it to
        # every bucket, so a mixed XY+XZ round trained axial with lateral's
        # mode (and vice versa) -- the per-bucket setting never reached the
        # trainer at all.  `training_mode` below stays the ACTIVE bucket's
        # value and is used for the log line + the snapshot only.
        prior_checkpoints = self._resolve_prior_checkpoints_for_buckets(
            buckets_to_train
        )

        persisted_pairs = {}
        for sl in staged_slice_labels:
            rec = self.sample_records.get(sl, {})
            img_uri = rec.get("img_uri", "")
            mask_uri = rec.get("mask_uri", "")
            if img_uri and mask_uri:
                persisted_pairs[sl] = {"img_uri": img_uri, "mask_uri": mask_uri}

        # Log dataset composition.
        target_view = "spatial" if bucket == "lateral" else "axial"
        total_expected = len(prior_pairs) + len(staged_slice_labels)
        _mode_summary = ", ".join(
            f"{_b}={self._training_mode_for_bucket(_b)}"
            for _b in sorted(buckets_to_train)
        )
        self._append_ui_log(
            f"[TRAIN] Dataset: staged={len(staged_slice_labels)}, "
            f"prior={len(prior_pairs)}, total={total_expected}, "
            f"views={target_view}, mode={training_mode} "
            f"(per bucket: {_mode_summary})"
        )

        # --- Initialize live loss plot ---
        loss_plot = getattr(self.ui, "loss_plot", None)
        if loss_plot is not None:
            loss_plot.begin_live_round(round_id)
        # Hook 3 (brick T): the Loss collapsible opens on the live round and
        # the Training stage line says "starting".
        try:
            fn = getattr(self.ui, "on_train_started", None)
            if fn:
                fn(bucket, self._training_round_name(bucket, round_num))
        except Exception as exc:      # noqa: BLE001 - a hook must not break training
            print(f"[STEP2-HOOK] on_train_started failed: {type(exc).__name__}: {exc}")

        # Create abort flag for this training run.
        self._training_abort_flag = _AbortFlag()

        # Snapshot mutable state so background thread doesn't race with UI changes.
        config_snapshot = dict(self.current_config)
        working_dir_snapshot = str(self.working_dir)
        # Commit 4 audit M-B fix (2026-05-26): also snapshot active_run_id
        # at training launch.  If user switches active run mid-train
        # (the `_switch_run_guard` HARD-BLOCK should prevent this, but
        # defense-in-depth pairs with the C1 working_dir snapshot), the
        # worker uses the snapshot run rather than the live attribute.
        active_run_snapshot = str(
            getattr(self, "active_run_id", "default") or "default"
        )
        # Commit 5 audit Audit-2/5 MEDIUM #1 fix: same risk class as
        # M-B but for the loaded patch name.  Read at `_run_training_work
        # :5141` as `dataset_id` (used in artifact filenames/manifests).
        # Patch-click handler (`_on_patch_list_clicked`) is NOT in
        # `_set_training_ui_locked`'s widget list, so user clicking a
        # different patch mid-train would silently mis-tag the dataset.
        loaded_patch_snapshot = str(
            getattr(self, "loaded_patch_name", "") or ""
        )
        # Phase B.1 DRIFT-1 fix (2026-05-25): snapshot selected_view +
        # current_round_by_view at launch.  Previously the worker read
        # these LIVE via `self._current_round_key()` / `self._current_round()`
        # to set up `round_data_root_override` for the fallback write
        # path.  In Mode=Both this picked ONE bucket only.  In production
        # the fallback path is dead (persisted_pairs URIs always resolve)
        # but the live worker-thread reads are still race violations.
        selected_view_snapshot = str(
            getattr(self, "selected_view", "XY") or "XY"
        )
        current_round_by_view_snapshot = dict(
            getattr(self, "current_round_by_view", {}) or {}
        )

        # Common kwargs for the training work function.
        work_kwargs = dict(
            round_id=round_id,
            staged_slice_labels=staged_slice_labels,
            training_mode=training_mode,
            prior_pairs=prior_pairs,
            prior_checkpoints=prior_checkpoints,
            persisted_pairs=persisted_pairs,
            config_snapshot=config_snapshot,
            working_dir_snapshot=working_dir_snapshot,
            active_run_snapshot=active_run_snapshot,
            selected_view_snapshot=selected_view_snapshot,
            current_round_by_view_snapshot=current_round_by_view_snapshot,
            loaded_patch_snapshot=loaded_patch_snapshot,
            abort_flag=self._training_abort_flag,
        )

        # Synchronous fallback (for tests or headless environments).
        if getattr(self, "_use_sync_training", False):
            # Bundle 2 audit fix #1 (2026-05-27): set the sync-in-flight
            # flag so cancel_training() knows training is running.
            # Cleared in a finally so a crash mid-flight doesn't leave it
            # stuck True.
            self._sync_training_in_flight = True
            try:
                def _noop_progress(pct, msg):
                    self._append_ui_log(f"[TRAIN] {msg}")
                # Bundle 2 audit fix #4: surface archive-fallback
                # warnings to UI log in sync mode too.
                def _sync_log_warn(msg):
                    self._append_ui_log(f"[TRAIN-WARN] {msg}")
                result = self._run_training_work(
                    progress_cb=_noop_progress,
                    log_warn_cb=_sync_log_warn,
                    **work_kwargs,
                )
                self._on_training_finished(result, staged_slice_labels, round_id)
            # Bundle 2 audit fix #2 (2026-05-27): TrainingAbortedError now
            # derives from BaseException, so `except Exception` does NOT
            # catch it.  Without this explicit clause, mid-train cancel in
            # the sync fallback path would escape unhandled and leave the
            # button stuck at "Cancelling..." disabled.  Mirror the
            # _TrainingWorker.run pattern: explicit TrainingAbortedError
            # first, then Exception for everything else.
            except TrainingAbortedError:
                self._on_training_error("Training cancelled by user.")
            except Exception as exc:
                self._on_training_error(str(exc))
            finally:
                self._sync_training_in_flight = False
            return

        # --- Launch heavy work in background thread ---
        worker = _TrainingWorker(self._run_training_work, **work_kwargs)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_training_progress)
        # Bundle 2 audit fix #4 (2026-05-27): connect log_warn cross-thread
        # signal to UI log.  Used by archive-fallback path when prior
        # weights get discarded due to a Windows file-lock etc.
        worker.log_warn.connect(
            lambda msg: self._append_ui_log(f"[TRAIN-WARN] {msg}")
        )
        worker.finished.connect(lambda result: self._on_training_finished(result, staged_slice_labels, round_id))
        worker.error.connect(self._on_training_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        # Clean up references only after the thread's event loop has fully stopped.
        thread.finished.connect(self._cleanup_training_thread)
        # Keep references so they don't get garbage-collected.
        self._training_thread = thread
        self._training_worker = worker

        # F1.5 (2026-05-27 audit): write a .training_in_progress marker
        # file just before thread.start().  Cleared by _on_training_-
        # finished (clean exit) and _on_training_error (clean abort).
        # If napari crashes or is force-killed mid-train, the marker
        # survives to the next session — detected at controller init
        # and surfaced as a warning so the user knows partial state
        # may be on disk.
        try:
            self._write_training_in_progress_marker(round_id=round_id)
        except Exception as exc:
            print(f"[TRAIN-SAFETY] F1.5 marker write FAILED (non-fatal): {exc}")

        thread.start()

    def _training_in_progress_marker_path(self):
        """Path to the .training_in_progress marker file (or None if
        working_dir isn't set)."""
        if not self.working_dir:
            return None
        from pathlib import Path as _P
        return _P(self.working_dir) / ".training_in_progress"

    def _write_training_in_progress_marker(self, *, round_id):
        """Write the marker file with launch context (run, bucket,
        round_id, timestamp).  Cleared on clean completion / clean
        cancel.  If still present at next session start, it means
        training was force-killed mid-write — partial state may exist
        on disk."""
        marker = self._training_in_progress_marker_path()
        if marker is None:
            return
        import json as _json
        from datetime import datetime as _dt, timezone as _tz
        payload = {
            "run_id": str(getattr(self, "active_run_id", "") or ""),
            "bucket": self._current_round_key(),
            "round_id": str(round_id),
            "started_at": _dt.now(_tz.utc).isoformat(),
            "pid": __import__("os").getpid(),
        }
        marker.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[TRAIN-SAFETY] F1.5 marker written: {marker}")

    def _clear_training_in_progress_marker(self):
        """Delete the marker file (best-effort)."""
        marker = self._training_in_progress_marker_path()
        if marker is None:
            return
        try:
            if marker.is_file():
                marker.unlink()
                print(f"[TRAIN-SAFETY] F1.5 marker cleared: {marker}")
        except Exception as exc:
            print(f"[TRAIN-SAFETY] F1.5 marker clear FAILED (non-fatal): {exc}")

    def _detect_stale_training_marker(self):
        """Called from _load_internal_state.  If a marker file exists,
        surface a warning so the user knows the previous session was
        force-killed mid-train and partial state may be on disk.

        Bundle 2 audit fix #3 (2026-05-27): ALSO scans for orphan
        ``.pre_train_archive/`` directories.  If a crash interrupted
        training BEFORE the except-BaseException handler in the
        training backend could restore the prior weights, the archive
        sits on disk forever and the canonical models/ is empty or
        partial.  This scan auto-restores when canonical is missing
        (clear win), and surfaces the rest as warnings so the user
        can manually recover.
        """
        marker = self._training_in_progress_marker_path()
        marker_present = marker is not None and marker.is_file()

        if marker_present:
            try:
                import json as _json
                payload = _json.loads(marker.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            msg = (
                "[TRAIN] WARNING: a previous training session may have been "
                f"interrupted mid-run (bucket={payload.get('bucket', '?')!r}, "
                f"round={payload.get('round_id', '?')!r}, "
                f"started={payload.get('started_at', '?')!r}).  "
                "Partial / corrupt weights may exist on disk.  Re-run "
                "training to overwrite, or inspect the round directory."
            )
            print(f"[TRAIN-SAFETY] F1.5 STALE MARKER detected: {msg}")
            try:
                self._append_ui_log(msg)
            except Exception:
                pass
            # Clear it now so it doesn't keep re-warning every session.
            self._clear_training_in_progress_marker()

        # Bundle 2 audit fix #3: scan for orphan archives.  This runs
        # whether or not the marker was present — an orphan can exist
        # without a marker (e.g. user manually killed napari between
        # marker-clear and process exit).
        if not self.working_dir:
            return
        try:
            from packages.epicycle_training.training_backend_adapter import (
                scan_orphan_archives,
                restore_orphan_archive,
            )
            from pathlib import Path as _P
            orphans = scan_orphan_archives(_P(self.working_dir))
        except Exception as exc:
            print(f"[TRAIN-SAFETY] orphan-archive scan FAILED: {exc}")
            return
        if not orphans:
            return

        auto_restored = 0
        manual_recover = []
        for orphan in orphans:
            if not orphan.get("canonical_weights_present", False):
                # Clear win — canonical is empty, restoring is strictly
                # better than leaving the archive there.
                ok = restore_orphan_archive(orphan)
                if ok:
                    auto_restored += 1
                    print(
                        f"[TRAIN-SAFETY] F3.1 auto-restored orphan "
                        f"archive: {orphan.get('archive_path')}"
                    )
                else:
                    manual_recover.append(orphan)
            else:
                # Canonical has content — don't auto-touch.  This
                # archive might be from a prior successful retrain
                # whose _delete_archive failed.  Surface for manual
                # decision.
                manual_recover.append(orphan)

        if auto_restored:
            try:
                self._append_ui_log(
                    f"[TRAIN] F3.1 RECOVERY: auto-restored "
                    f"{auto_restored} prior-weights archive(s) from a "
                    "crashed training session.  Your previous round's "
                    "weights are back at the canonical location."
                )
            except Exception:
                pass
        if manual_recover:
            paths_str = "\n  ".join(
                str(o.get("archive_path")) for o in manual_recover
            )
            try:
                self._append_ui_log(
                    f"[TRAIN] F3.1 ORPHANS: found "
                    f"{len(manual_recover)} orphan training-state "
                    "archive(s) where the canonical weights are still "
                    "present.  Likely a prior successful training "
                    "whose cleanup failed.  Safe to delete manually:\n"
                    f"  {paths_str}"
                )
            except Exception:
                pass
            print(
                f"[TRAIN-SAFETY] F3.1 ORPHANS (manual recover): "
                f"{[str(o.get('archive_path')) for o in manual_recover]}"
            )

    def _run_training_work(
        self,
        *,
        progress_cb,
        round_id,
        staged_slice_labels,
        training_mode,
        prior_pairs,
        prior_checkpoints,
        persisted_pairs,
        config_snapshot,
        working_dir_snapshot,
        active_run_snapshot=None,
        loaded_patch_snapshot=None,
        selected_view_snapshot=None,
        current_round_by_view_snapshot=None,
        abort_flag=None,
        log_warn_cb=None,
    ):
        """Heavy training work — runs in background thread.

        Uses *config_snapshot*, *working_dir_snapshot*,
        *active_run_snapshot*, *loaded_patch_snapshot*,
        *selected_view_snapshot*, and *current_round_by_view_snapshot*
        instead of reading from ``self`` to avoid races with UI-thread
        mutations.
        """
        def _check_abort():
            if abort_flag is not None and abort_flag.is_set():
                raise TrainingAbortedError("Training cancelled by user.")

        _check_abort()
        progress_cb(10, "Preparing dataset...")

        # Step 2 restructure (Phase 3 sub-C): if workspace is on the new
        # layout, route working-copy writes to runs/{active_run_id}/{bucket}/
        # rounds/round_NN/ so they don't accumulate in legacy training_data/
        # post-migration.
        round_data_root_override = None
        try:
            from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
            if _step2_paths.is_new_layout_state(working_dir_snapshot):
                # M-B: use launch-time snapshot.  Commit 6 audit hardening:
                # the previous `if snap else live` fallback was unreachable
                # in production (launcher always provides non-empty) but
                # silently weakened M-B if a future test caller forgot
                # the kwarg.  Now: require snapshot, fail fast otherwise.
                if not active_run_snapshot:
                    raise RuntimeError(
                        "_run_training_work: active_run_snapshot is required "
                        "for new-layout workspaces (M-B contract)"
                    )
                active_run_id = active_run_snapshot
                # Phase B.1 DRIFT-1 fix (2026-05-25): use snapshots
                # instead of live `self._current_round_key()` /
                # `self._current_round()`.  Note: the override here
                # only feeds `prepare_round_dataset`'s rare fallback
                # write path (dead in production — `persisted_pairs`
                # URIs from `_persist_curated_training_pair` already
                # point at correct per-bucket `accepted/`).  The
                # per-view loop below (line ~5380) does per-bucket
                # save_path routing for ACTUAL training output — that
                # path is per-bucket correct via H-1 fix.
                _sv = str(selected_view_snapshot or "XY").upper()
                bucket = ("lateral"
                          if _sv in ("XY", "SPATIAL", "LATERAL")
                          else "axial")
                _by_view = current_round_by_view_snapshot or {}
                round_idx = int(_by_view.get(bucket, 0))
                round_data_root_override = _step2_paths.round_dir(
                    working_dir_snapshot, active_run_id, bucket, round_idx
                )
                print(f"[STEP2-TRAIN] prepare_round_dataset routing working "
                      f"copies to NEW layout: {round_data_root_override} "
                      f"(bucket={bucket} round_idx={round_idx} from snapshots)")
        except Exception as _e:
            print(f"[STEP2-TRAIN] new-layout path resolution failed "
                  f"({type(_e).__name__}: {_e}), using legacy training_data/")

        # Commit 5 audit Audit-2/5 MEDIUM #1 fix: use snapshot (captured
        # at launch) instead of live `self.loaded_patch_name` so a mid-
        # train patch click doesn't mis-tag dataset_id.
        _dataset_id = (
            loaded_patch_snapshot
            if loaded_patch_snapshot
            else (
                self.loaded_patch_name
                if hasattr(self, "loaded_patch_name") and self.loaded_patch_name
                else "default_dataset"
            )
        )
        # Release-hardening #5/#6 (2026-09-07): route the preparer's drops
        # to the UI log, and capture WHICH staged labels actually became
        # training pairs so `_on_training_finished` commits only those.
        used_slice_labels: list = []
        try:
            samples = self.training_manager.prepare_round_dataset(
                segmentation_manager=self.segmentation_manager,
                slice_labels=staged_slice_labels,
                working_dir=working_dir_snapshot,
                round_id=round_id,
                dataset_id=_dataset_id,
                prior_pairs=prior_pairs,
                persisted_pairs=persisted_pairs,
                round_data_root=round_data_root_override,
                log_warn_cb=log_warn_cb,
                used_slice_labels_out=used_slice_labels,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to prepare round dataset: {e}") from e

        if not samples:
            raise RuntimeError("No valid samples found for training.")

        progress_cb(20, "Splitting by view...")
        grouped_samples = self.training_manager.split_samples_by_view_strategy(samples)
        if not grouped_samples:
            raise RuntimeError("No valid view groups available for training.")

        round_losses = {}
        round_checkpoints = {}
        round_inference = {}
        round_backend = {"training": {}, "inference": {}, "failed_views": []}

        starting_ckpt = config_snapshot.get("starting_checkpoint", "").strip()
        n_views = len(grouped_samples)

        # Progress allocation: 20-80% for training+inference, divided per view.
        # Each view gets a contiguous range: [train_start..train_end..infer_end].
        train_range = 60  # total % for training+inference (20-80)
        per_view = train_range / max(n_views, 1)

        for idx, (view_type, view_samples) in enumerate(grouped_samples.items()):
            _check_abort()
            view_start = 20 + int(per_view * idx)
            view_train_end = view_start + int(per_view * 0.75)
            view_end = view_start + int(per_view)
            progress_cb(view_start, f"Training {view_type}...")

            # Release-hardening 2026-09-07 fix pass: NO scalar-mode
            # gate here.  `prior_checkpoints` was built per bucket by
            # `_resolve_prior_checkpoints_for_buckets`, so a from-scratch
            # bucket simply has no entry.  Re-gating on the flat
            # `training_mode` would undo that per-bucket decision for the
            # other bucket in a mixed round.
            pretrained = prior_checkpoints.get(view_type)
            if not pretrained and starting_ckpt and Path(starting_ckpt).is_file():
                pretrained = starting_ckpt
            # Commit 2 audit H1 fix (2026-05-26): on round 0, if no
            # prior checkpoint AND no manual `starting_ckpt`, resolve
            # the active run's `created_from` (init_spec from the New
            # Run dialog) into a concrete checkpoint path.  Before this
            # fix, init_spec was written to run_manifest.json but never
            # read at training time — round 0 of an "inherited"/"fork"/
            # "step3_refined"/"browse" run silently trained from scratch.
            # `scratch` and `pretrained` modes correctly return None
            # (the latter falls through to config["model_type"]).
            if not pretrained:
                try:
                    from napari_epicycle.hitl_training_annotation import (
                        _paths as _step2_paths,
                    )
                    from napari_epicycle.hitl_training_annotation._run_management import (
                        resolve_init_spec_to_checkpoint,
                    )
                    from packages.epicycle_io.step2_run_manifest import (
                        read_run_manifest,
                    )
                    if _step2_paths.is_new_layout_state(working_dir_snapshot):
                        # M-B: require launch-time snapshot (Commit 6
                        # audit hardening — see launcher-site comment).
                        if not active_run_snapshot:
                            raise RuntimeError(
                                "_run_training_work: active_run_snapshot "
                                "required for new-layout (M-B contract)"
                            )
                        _active_run = active_run_snapshot
                        _bucket = (
                            "lateral"
                            if str(view_type).lower() in ("xy", "spatial", "lateral")
                            else "axial"
                        )
                        # Only auto-resolve on round 0 of this bucket
                        # (subsequent rounds already inherit from
                        # `prior_checkpoints` via cumulative training).
                        # Phase B.1.1 fix (audit 3 HIGH unfixed DRIFT-1):
                        # use snapshot instead of live `self.current_round_by_view`
                        # to keep the worker thread race-free.
                        by_view = current_round_by_view_snapshot or {}
                        _bucket_round = (
                            int(by_view[_bucket])
                            if isinstance(by_view, dict) and _bucket in by_view
                            else 0
                        )
                        if _bucket_round == 0:
                            rm_path = _step2_paths.run_manifest_path(
                                working_dir_snapshot, _active_run
                            )
                            rm = read_run_manifest(rm_path)
                            init_spec = (
                                rm.get("created_from")
                                if rm and isinstance(rm.get("created_from"), dict)
                                else None
                            )
                            resolved = resolve_init_spec_to_checkpoint(
                                working_dir_snapshot, init_spec,
                                active_run_id=_active_run,
                                target_bucket=_bucket,
                            )
                            if resolved:
                                pretrained = resolved
                                print(f"[STEP2-TRAIN] H1 round-0 init: "
                                      f"resolved {init_spec.get('mode')!r} -> "
                                      f"{resolved}")
                            elif init_spec and init_spec.get("mode") not in (
                                None, "scratch", "pretrained"
                            ):
                                print(f"[STEP2-TRAIN] H1 round-0 init: "
                                      f"mode={init_spec.get('mode')!r} but "
                                      f"resolve returned None — source "
                                      f"checkpoint may have been deleted; "
                                      f"falling back to config.model_type or "
                                      f"scratch")
                except Exception as _e:
                    print(f"[STEP2-TRAIN] H1 round-0 init: resolve raised "
                          f"({type(_e).__name__}: {_e}); falling through")

            # Log exact initialization source for this view.
            if pretrained and prior_checkpoints.get(view_type):
                init_src = f"prior round checkpoint: {pretrained}"
            elif pretrained and pretrained == starting_ckpt:
                init_src = f"manual starting checkpoint: {pretrained}"
            elif pretrained:
                init_src = f"checkpoint: {pretrained}"
            else:
                base_model = config_snapshot.get("model_type", "scratch")
                init_src = f"base Cellpose model: {base_model}" if base_model != "scratch" else "scratch (random weights)"
            progress_cb(view_start, f"Training {view_type} — init from {init_src}")

            # Per-view training progress: view_start → view_train_end
            view_train_span = view_train_end - view_start

            def _epoch_cb(epoch, n_epochs, train_loss, test_loss, _vs=view_start, _vts=view_train_span, _vt=view_type):
                _check_abort()
                pct = _vs + int(_vts * min(epoch / max(n_epochs, 1), 1.0))
                loss_str = f"loss: {train_loss:.4f}"
                if test_loss is not None:
                    loss_str += f" | val: {test_loss:.4f}"
                progress_cb(pct, f"Training {_vt} — Epoch {epoch}/{n_epochs} | {loss_str}")

            # Phase 6c (2026-05-26): resolve per-bucket-round save_path
            # in the new multi-run layout.  Adapter receives this as
            # save_path_override and uses caller-supplied
            # weights_filename for the Cellpose model_name.
            #
            # weights_filename pattern: f"{run_id}_{bucket}" so the file
            # carries identity when it's reused in Step 3/4 (instead of
            # a generic 'weights' blob).  E.g. 'nucleiv1_lateral',
            # 'cell_axial'.  Round identity comes from the directory
            # path (`round_NN/`).  The user, 2026-05-26: "if we just
            # like reuse the run name? for the model name? and like maybe
            # add lateral or axial somewhere along the name."
            #
            # view_type comes in as legacy "spatial" / "axial"; translate
            # "spatial" -> "lateral" for the path helpers + filename.
            save_path_override = None
            predictions_dir_override = None
            weights_filename = None
            _bucket_for_view = None
            _active_run = None
            _round_idx = None
            try:
                from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
                if _step2_paths.is_new_layout_state(working_dir_snapshot):
                    _bucket_for_view = "lateral" if str(view_type).lower() in ("xy", "spatial", "lateral") else "axial"
                    # M-B: require launch-time snapshot (Commit 6
                    # audit hardening — see launcher-site comment).
                    if not active_run_snapshot:
                        raise RuntimeError(
                            "_run_training_work: active_run_snapshot "
                            "required for new-layout (M-B contract)"
                        )
                    _active_run = active_run_snapshot
                    # 2026-05-26 audit H-1 fix: use PER-BUCKET round
                    # counter instead of `self._current_round()` which
                    # reads `self.selected_view`.  In Mode=Both both
                    # iterations of this loop (lateral + axial) had the
                    # same `_round_idx` because the UI's selected view
                    # determined it — so both write to the same
                    # `round_NN/` dir even when the two buckets are at
                    # different round numbers.
                    # Phase B.1.1 fix (audit 3 HIGH unfixed DRIFT-1):
                    # use snapshot instead of live `self.current_round_by_view`
                    # to keep the worker thread race-free.
                    by_view = current_round_by_view_snapshot or {}
                    if isinstance(by_view, dict) and _bucket_for_view in by_view:
                        _round_idx = int(by_view[_bucket_for_view])
                    else:
                        _round_idx = 0
                    save_path_override = _step2_paths.model_dir(
                        working_dir_snapshot, _active_run,
                        _bucket_for_view, _round_idx,
                    )
                    predictions_dir_override = _step2_paths.round_predictions_dir(
                        working_dir_snapshot, _active_run,
                        _bucket_for_view, _round_idx,
                    )
                    # Phase B.2 (2026-05-25): filename now carries round
                    # identity.  Pre-B.2 was `{run}_{bucket}`; post-B.2 is
                    # `{run}_{bucket}_round_NN`.  When this file is
                    # exported or reused in Step 3/4, the name alone tells
                    # you the round.  Companion files (loss, training_config)
                    # inherit this prefix automatically via Cellpose's
                    # `{model_name}_{suffix}` convention.
                    weights_filename = f"{_active_run}_{_bucket_for_view}_round_{_round_idx:02d}"
                    print(f"[STEP2-TRAIN] Phase 6c routing view={view_type!r} "
                          f"-> bucket={_bucket_for_view} | "
                          f"round_idx={_round_idx} (per-bucket) | "
                          f"save_path={save_path_override} | "
                          f"weights_filename={weights_filename!r} | "
                          f"predictions_dir={predictions_dir_override}")
            except Exception as _e:
                print(f"[STEP2-TRAIN] Phase 6c path resolution failed "
                      f"({type(_e).__name__}: {_e}); falling back to legacy")

            try:
                result = self.training_manager.run_training(
                    round_id=round_id,
                    config=config_snapshot,
                    working_dir=working_dir_snapshot,
                    training_samples=view_samples,
                    view_type=view_type,
                    pretrained_model=pretrained,
                    epoch_callback=_epoch_cb,
                    save_path_override=str(save_path_override) if save_path_override else None,
                    weights_filename=weights_filename,
                    log_warn_cb=log_warn_cb,
                )
            except Exception as e:
                round_backend["failed_views"].append({"view_type": view_type, "stage": "training", "error": str(e)})
                continue

            checkpoint_path = result.get("checkpoint_path")
            losses = result.get("losses", [])
            # 2026-05-26 audit H-4 fix: capture val-loss series (was
            # silently dropped — manifest's final_loss.val stayed None).
            val_losses = result.get("val_losses", [])
            round_losses[view_type] = losses
            round_backend["training"][view_type] = {
                "mode": str(result.get("mode", self.training_manager.training_backend_info()["mode"])),
                "repo_path": str(result.get("repo_path", self.training_manager.training_backend_info()["repo_path"])),
            }
            if checkpoint_path:
                round_checkpoints[view_type] = checkpoint_path

                _check_abort()
                progress_cb(view_train_end, f"Inference {view_type} using checkpoint: {Path(checkpoint_path).name}")
                _inference_succeeded = False
                try:
                    inference_result = self.training_manager.run_inference(
                        round_id=round_id,
                        view_type=view_type,
                        model_checkpoint_path=checkpoint_path,
                        config=config_snapshot,
                        working_dir=working_dir_snapshot,
                        inference_samples=view_samples,
                        predictions_dir_override=str(predictions_dir_override) if predictions_dir_override else None,
                    )
                    _inference_succeeded = True
                    round_inference[view_type] = inference_result.get("prediction_uri", "")
                    round_backend["inference"][view_type] = {
                        "mode": str(inference_result.get("mode", self.training_manager.inference_backend_info()["mode"])),
                        "repo_path": str(
                            inference_result.get("repo_path", self.training_manager.inference_backend_info()["repo_path"])
                        ),
                    }
                except Exception as e:
                    round_backend["failed_views"].append({"view_type": view_type, "stage": "inference", "error": str(e)})

                # Phase 6c (2026-05-26): write per-bucket-round
                # manifest.json (schema 2.0) at the canonical location.
                # H-3 fix (Audit 3): moved AFTER inference completes
                # so an inference failure / mid-train abort doesn't
                # leave a manifest pointing at orphan state that
                # downstream readers (Run-model picker, Step 3 import)
                # would surface as a valid model.
                # C-3 gate: only write if adapter confirms the lift
                # actually moved the weights to canonical root (manifest
                # path fields would point at non-existent files
                # otherwise).
                # Release-hardening #11 (2026-09-07): dropped
                # `_inference_succeeded` from the gate.  The weights ARE
                # at the canonical path once the lift succeeded — skipping
                # the manifest left a usable model with no provenance
                # (Step 3 finds the weights file by name regardless).  The
                # manifest now records `inference_failed: true` instead.
                _manifest_gate = (
                    save_path_override is not None
                    and result.get("new_layout")
                    and result.get("weights_lifted")
                )
                if not _manifest_gate:
                    # Commit 2 audit M4 fix (2026-05-26): make the
                    # silent dropout visible.  Placeholder/legacy
                    # backends + lift-failures + inference-failures all
                    # skip manifest writes; without this print the gap
                    # is invisible in logs.
                    print(f"[STEP2-TRAIN] schema 2.0 manifest SKIPPED "
                          f"(save_path_override={'set' if save_path_override else 'None'}, "
                          f"new_layout={result.get('new_layout')}, "
                          f"weights_lifted={result.get('weights_lifted')}, "
                          f"inference_succeeded={_inference_succeeded})")
                if _manifest_gate:
                    try:
                        self._write_step2_model_manifest_for_round(
                            run_id=_active_run,
                            bucket=_bucket_for_view,
                            round_idx=_round_idx,
                            view_type=view_type,
                            losses=losses,
                            val_losses=val_losses,
                            config=config_snapshot,
                            n_train_pairs=len(view_samples),
                            pretrained_model=pretrained,
                            weights_filename=weights_filename,
                            working_dir=working_dir_snapshot,
                            # Phase B.3 (2026-05-25) — schema 2.1
                            # reproducibility fields from the backend
                            # adapter result.  Each is Optional; the
                            # manifest builder defaults missing/None to
                            # empty/None.
                            train_pair_labels=result.get("train_pair_labels"),
                            train_pair_uris=result.get("train_pair_uris"),
                            timing_started_at=result.get("timing_started_at"),
                            timing_finished_at=result.get("timing_finished_at"),
                            timing_duration_seconds=result.get("timing_duration_seconds"),
                            env_cellpose_version=result.get("env_cellpose_version"),
                            env_torch_version=result.get("env_torch_version"),
                            env_python_version=result.get("env_python_version"),
                            env_device=result.get("env_device"),
                            env_platform=result.get("env_platform"),
                            inference_failed=not _inference_succeeded,
                        )
                    except Exception as _e:
                        print(f"[STEP2-TRAIN] WARN: schema 2.0 manifest "
                              f"write failed ({type(_e).__name__}: {_e}); "
                              f"weights file is still at canonical path "
                              f"so this is hygiene-only.")

        if not round_losses:
            raise RuntimeError(f"No successful training view for {round_id}.")

        progress_cb(90, "Saving artifacts...")
        # M-1 fix (2026-05-26): snapshot is_new_layout in the round
        # result so `_on_training_finished` doesn't re-derive it from
        # `self.working_dir` (which may have changed mid-train if user
        # switched workspace).  The training pipeline already routed
        # outputs based on the start-of-train workspace; the legacy
        # JSON-write gate must use the SAME decision.
        try:
            from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
            _is_new_layout_snapshot = bool(
                _step2_paths.is_new_layout_state(working_dir_snapshot)
            )
        except Exception:
            _is_new_layout_snapshot = False
        return {
            "samples": samples,
            "round_losses": round_losses,
            "round_checkpoints": round_checkpoints,
            "round_inference": round_inference,
            "round_backend": round_backend,
            "is_new_layout": _is_new_layout_snapshot,
            # Release-hardening #6: staged labels that actually produced a
            # training pair (see `_on_training_finished`'s commit gate).
            "used_slice_labels": list(used_slice_labels),
        }

    _EPOCH_MSG_RE = None  # lazily compiled

    def _on_training_progress(self, percent, message):
        """Update UI from worker progress signal (runs on UI thread)."""
        self.ui.training_progress_bar.setValue(percent)
        self._append_ui_log(f"[TRAIN] {message}")

        # Feed epoch data to the live loss plot.
        loss_plot = getattr(self.ui, "loss_plot", None)
        if loss_plot is not None and "Epoch" in message:
            if self._EPOCH_MSG_RE is None:
                import re
                self.__class__._EPOCH_MSG_RE = re.compile(
                    r"Epoch\s+(\d+)/(\d+)\s*\|\s*loss:\s*([\d.]+)"
                )
            m = self._EPOCH_MSG_RE.search(message)
            if m:
                epoch = int(m.group(1))
                loss = float(m.group(3))
                loss_plot.append_live_point(epoch, loss)
                # seam (d): the running stage line + the Loss header caption
                try:
                    fn = getattr(self.ui, "on_training_epoch", None)
                    if fn:
                        fn(epoch, int(m.group(2)), loss)
                except Exception as exc:      # noqa: BLE001
                    print(f"[STEP2-HOOK] on_training_epoch failed: "
                          f"{type(exc).__name__}: {exc}")

    def _restore_train_button(self):
        """Restore the train button after training completes or is cancelled."""
        # Step 2 look (brick T, seam (a)): no connection swap and no rename --
        # Cancel goes back to its idle word and greys, Train lights.
        btn = self.ui.update_and_train_button
        cancel_btn = getattr(self.ui, "train_cancel_btn", None)
        if cancel_btn is not None:
            if hasattr(cancel_btn, "setText"):
                from ..ui._step2_text import CANCEL_BTN_TEXT
                cancel_btn.setText(CANCEL_BTN_TEXT)
            cancel_btn.setEnabled(False)
        btn.setEnabled(True)

    def _on_training_error(self, error_msg):
        """Handle worker error (runs on UI thread)."""
        self.ui.training_progress_bar.setValue(0)
        self._restore_train_button()
        self._set_training_ui_locked(False)
        self._append_ui_log(f"[TRAIN ERROR] {error_msg}")
        # Release-hardening #4 (2026-09-07): pre-fix a failed round only
        # left a line in the scrollable log while the button quietly went
        # back to "Train" — indistinguishable from "nothing happened".
        #
        # 1b#2: EXCEPT when the user asked for this.  The worker routes a
        # cancel through the same `error` signal, so a modal saying
        # "Training failed" would accuse the user of a crash they caused
        # deliberately.  Suppressed when the message is the cancel sentinel
        # or the abort flag is set (a late error raised while unwinding).
        if not self._is_user_cancellation(error_msg):
            try:
                self._show_training_error_dialog(str(error_msg))
            except Exception:
                pass
        # Discard incomplete live curve on error.
        loss_plot = getattr(self.ui, "loss_plot", None)
        if loss_plot is not None:
            loss_plot.cancel_live_round()
            self._notify_train_finished()
        # F1.5: clear marker on clean error/abort.
        try:
            self._clear_training_in_progress_marker()
        except Exception:
            pass

    def _can_show_modal(self) -> bool:
        """True only when a blocking Qt modal is actually showable (1b#3).

        `_use_sync_training` is a TEST-MODE flag, not a display check — a
        headless run (CI, `pytest` without a QApplication, a --print session)
        can have it False and still have no event loop, where constructing a
        QMessageBox aborts the process.  Ask Qt itself.
        """
        if getattr(self, "_use_sync_training", False):
            return False
        try:
            from qtpy.QtWidgets import QApplication
            return QApplication.instance() is not None
        except Exception:
            return False

    def _is_user_cancellation(self, error_msg) -> bool:
        """True when this "error" is really the user's own Cancel (1b#2)."""
        if str(error_msg or "").strip() == _TRAINING_CANCELLED_MSG:
            return True
        flag = getattr(self, "_training_abort_flag", None)
        try:
            return bool(flag is not None and flag.is_set())
        except Exception:
            return False

    @staticmethod
    def _training_error_headline(error_msg):
        """First non-empty line of an error message (release-hardening #4)."""
        for line in str(error_msg or "").strip().splitlines():
            if line.strip():
                return line.strip()
        return "Unknown error"

    def _show_training_error_dialog(self, error_msg):
        """Modal for a failed/cancelled training round (release-hardening #4).

        Shows only the FIRST line of the error (cellpose tracebacks are long)
        and points at the log for the rest.  Skipped in headless / sync-
        training contexts where a modal would hang.
        """
        if not self._can_show_modal():
            return
        first_line = self._training_error_headline(error_msg)
        try:
            from qtpy.QtWidgets import QMessageBox
            box = QMessageBox(getattr(self, "ui", None))
            box.setIcon(QMessageBox.Critical)
            box.setWindowTitle("Training failed")
            box.setText(first_line)
            box.setInformativeText("See the training log for details.")
            box.setStandardButtons(QMessageBox.Ok)
            box.exec_() if hasattr(box, "exec_") else box.exec()
        except Exception as _e:
            print(f"[STEP2-TRAIN] training-error modal unavailable "
                  f"({type(_e).__name__}: {_e}); log line already written")

    # ------------------------------------------------------------------
    # Brick D (2026-09-07) — shared error/busy surfaces
    # ------------------------------------------------------------------

    def _show_modal_error(self, title, headline, detail=""):
        """Critical modal for a total failure (journey #6 / #13).

        Shows only the FIRST line of the error (cellpose / OS tracebacks are
        long) and points at the log for the rest.  Skipped in headless /
        sync-training contexts where a modal would hang — the caller has
        always written the log line first, so nothing is lost.
        """
        if not self._can_show_modal():
            return
        try:
            from qtpy.QtWidgets import QMessageBox
            box = QMessageBox(getattr(self, "ui", None))
            box.setIcon(QMessageBox.Critical)
            box.setWindowTitle(str(title))
            box.setText(self._training_error_headline(headline))
            box.setInformativeText(
                str(detail) or "See the training log for details."
            )
            box.setStandardButtons(QMessageBox.Ok)
            box.exec_() if hasattr(box, "exec_") else box.exec()
        except Exception as _e:
            print(f"[STEP2] error modal unavailable "
                  f"({type(_e).__name__}: {_e}); log line already written")

    def _report_import_failure(self, what, error):
        """A copy that moved NOTHING must not be a console-only line."""
        headline = self._training_error_headline(error)
        print(f"[ERROR] {what} failed: {type(error).__name__}: {error}")
        self._append_ui_log(f"[{what.upper()} FAILED] {headline}")
        self._show_modal_error(
            f"{what} failed", headline,
            "Nothing was copied.  See the training log for details.",
        )

    @staticmethod
    def _run_post_import_step(label, fn, failures):
        """Run one post-import step, collecting (not swallowing) its failure.

        These steps used to be `except Exception: pass`, so a workspace whose
        state save or pair reload failed looked like a clean import and then
        behaved as if the pairs were not there.
        """
        try:
            fn()
            return True
        except Exception as exc:
            failures.append((label, f"{type(exc).__name__}: {exc}"))
            print(f"[WARN] post-import step {label!r} failed: "
                  f"{type(exc).__name__}: {exc}")
            return False

    def _report_post_import_failures(self, what, failures):
        """One modal for all post-import step failures (never one per step)."""
        if not failures:
            return
        for label, msg in failures:
            self._append_ui_log(f"[{what.upper()}] post-import step "
                                f"'{label}' failed: {msg}")
        first = failures[0]
        self._show_modal_error(
            "Copied, but not loaded",
            f"{first[0]}: {first[1]}",
            "The image/mask pairs were copied successfully, but EpiCycle "
            "could not finish updating this session.  Close and reopen the "
            "workspace to see them.",
        )

    def _begin_busy(self, message="", cursor=True):
        """Sweep the patch busy bar + wait cursor for a blocking op.

        Measured (2026-09-07, real workspace): a patch load is ~2.4 s with the
        annotator closed and ~6.1 s with it open, and a cold cellpose predict
        ~1.6 s -- all far past the 100 ms budget, so the user needs to see that
        something is running.  Paired with :meth:`_end_busy` in a ``finally``.

        2026-09-07 (patch-bar brick): the indicator is the DEDICATED
        ``patch_busy_bar`` at the BOTTOM of the curation box, spanning both
        columns -- exactly where Step 3 puts ``curation_busy_bar``.  It is
        PERMANENT and is never shown or hidden by these helpers; they only
        flip it between the idle static strip (``setRange(0, 100)`` +
        ``setValue(0)``) and busy scan mode (``setRange(0, 0)``, marquee).
        ``training_progress_bar`` is now TRAINING-ONLY, so the old "training
        owns the bar" special-casing (`_busy_owns_bar`) is gone: the two
        indicators can never collide.
        """
        if getattr(self, "_busy_depth", 0) > 0:
            self._busy_depth += 1
            return
        self._busy_depth = 1
        if message:
            self._append_ui_log(message)
        fn = getattr(getattr(self, "ui", None), "on_busy", None)   # hook 5
        if fn:
            try:
                fn(str(message) if message else None)
            except Exception as exc:  # noqa: BLE001
                print(f"[STEP2-UI] on_busy failed: {exc}")
        bar = getattr(getattr(self, "ui", None), "patch_busy_bar", None)
        if bar is not None:
            try:
                bar.setRange(0, 0)          # scan mode (marquee)
            except Exception:
                pass
        try:
            from qtpy.QtWidgets import QApplication
            from qtpy.QtCore import Qt
            if cursor and QApplication.instance() is not None:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                self._busy_cursor_set = True
                QApplication.processEvents()
        except Exception:
            pass

    def _end_busy(self):
        """Undo :meth:`_begin_busy`.  Safe to call when no busy is active."""
        depth = getattr(self, "_busy_depth", 0)
        if depth > 1:
            self._busy_depth = depth - 1
            return
        self._busy_depth = 0
        fn = getattr(getattr(self, "ui", None), "on_busy", None)   # hook 5
        if fn:
            try:
                fn(None)
            except Exception as exc:  # noqa: BLE001
                print(f"[STEP2-UI] on_busy failed: {exc}")
        bar = getattr(getattr(self, "ui", None), "patch_busy_bar", None)
        if bar is not None:
            try:
                bar.setRange(0, 100)        # idle static strip (no marquee)
                bar.setValue(0)
            except Exception:
                pass
        if getattr(self, "_busy_cursor_set", False):
            self._busy_cursor_set = False
            try:
                from qtpy.QtWidgets import QApplication
                QApplication.restoreOverrideCursor()
            except Exception:
                pass

    #: Delay for the queued busy release.  COUPLED to the annotator's own
    #: deferred auto-load, which `_switch_patch_context_from_native` schedules
    #: with `_schedule_deferred_auto_load(50)` just before returning.  Any
    #: value strictly greater than that 50 ms works and the exact number does
    #: not matter, because Qt is single-threaded here: this timer can only be
    #: delivered once the auto-load handler (the ~1.5 s subviewer rebuild) has
    #: RETURNED.  If that 50 ms is ever changed, raise this above it.
    _ANNOTATOR_REBUILD_RELEASE_MS = 120

    def _end_busy_after_annotator_rebuild(self, epoch=None):
        """`_end_busy`, but queued behind the annotator's deferred rebuild.

        The release is stamped with the current patch-switch epoch and tracked
        in `_pending_rebuild_busy`, so a duplicate delivery (or one that
        arrives after its own scope was already closed some other way) cannot
        pop a scope belonging to a LATER load.  Falls back to an immediate
        `_end_busy` when there is no Qt event loop (tests / headless), so the
        busy depth is always released.
        """
        if epoch is None:
            epoch = int(getattr(self, "_patch_switch_epoch", 0))
        pending = getattr(self, "_pending_rebuild_busy", None)
        if not isinstance(pending, set):
            pending = set()
            self._pending_rebuild_busy = pending
        try:
            from qtpy.QtWidgets import QApplication
            from qtpy.QtCore import QTimer
            if QApplication.instance() is None:
                self._end_busy()
                return
            pending.add(epoch)
            QTimer.singleShot(
                self._ANNOTATOR_REBUILD_RELEASE_MS,
                lambda e=epoch: self._release_annotator_rebuild_busy(e),
            )
        except Exception:
            pending.discard(epoch)
            self._end_busy()

    def _release_annotator_rebuild_busy(self, epoch):
        """Close the busy scope queued for *epoch*, exactly once."""
        pending = getattr(self, "_pending_rebuild_busy", None)
        if not isinstance(pending, set) or epoch not in pending:
            print(f"[BUSY] stale rebuild release for epoch {epoch} ignored")
            return
        pending.discard(epoch)
        self._end_busy()

    def _pump_busy_ui(self):
        """Give Qt one slice of the main loop so the busy bar keeps sweeping.

        Only pumps while a busy scope is open, so it is inert in tests, in
        headless runs, and outside a load.  User input is EXCLUDED so a second
        patch click cannot re-enter the load half-way through -- only paints,
        timers and the marquee animation get through.

        Measured (2026-09-07): without this, a patch load blocks the UI thread
        for ~1.2 s (annotator closed) / ~3.9 s (open) in a single stretch,
        because ~90% of the load is napari layer add/remove at ~85-280 ms each.
        Calling it BETWEEN layer operations drops the longest freeze to roughly
        one layer op.
        """
        if getattr(self, "_busy_depth", 0) <= 0:
            return
        try:
            from qtpy.QtWidgets import QApplication
            from qtpy.QtCore import QEventLoop
            app = QApplication.instance()
            if app is None:
                return
            app.processEvents(QEventLoop.ExcludeUserInputEvents)
        except Exception:
            pass

    def _cleanup_training_thread(self):
        """Called when the QThread event loop has fully stopped (thread.finished signal)."""
        thread = getattr(self, "_training_thread", None)
        if thread is not None:
            thread.deleteLater()
        self._training_thread = None
        self._training_worker = None

    def _connect_shutdown_hooks(self):
        """Wire every available close signal to :meth:`shutdown`.

        Release-hardening #1 (2026-09-07): on napari 0.8 the viewer no
        longer emits ``events.close``, so the ``events.close`` connection
        alone left ``shutdown()`` as dead code — closing napari mid-session
        dropped the debounced state save (up to 2 s of strokes) and never
        aborted a running training thread.  Steps 1 and 3 already carry a
        ``widget.destroyed`` fallback (``data_preparation/controller.py``
        ``_connect_shutdown_hooks``); Step 2 now mirrors it.  Both hooks are
        attempted; :meth:`shutdown` is idempotent so a viewer that fires
        both only shuts down once.
        """
        try:
            if hasattr(self.viewer, "events") and hasattr(
                    self.viewer.events, "close"):
                self.viewer.events.close.connect(self._on_viewer_closing)
        except Exception:
            pass
        try:
            ui = getattr(self, "ui", None)
            if ui is not None and hasattr(ui, "destroyed"):
                ui.destroyed.connect(lambda *_: self.shutdown())
        except Exception:
            pass

    def _on_viewer_closing(self, _event=None):
        """Handle napari viewer close — save state and stop training thread."""
        self.shutdown()

    def cancel_training(self):
        """Cancel a running training session. Safe to call from UI thread.

        Bundle 2 (2026-05-27): with TrainingAbortedError now derived from
        BaseException, the cancel signal propagates through Cellpose's
        logging chain mid-epoch.  The abort is raised from the epoch
        callback, and cellpose only logs at epoch 5 and every 10th epoch,
        so the real bound is the next epoch checkpoint — up to ~10 epochs
        (release-hardening #8).  UI feedback below clearly communicates that
        cancellation is in progress + prevents the user from re-clicking
        and spamming the abort flag.

        Bundle 2 audit fix #1 (2026-05-27): gate ALL state mutations on
        training actually being in flight.  Pre-fix this method
        unconditionally set button text to "Cancelling..." + disabled it.
        If invoked stale (no training running) or in a race after
        _on_training_finished already restored the button, the button
        ended up stuck at "Cancelling..." disabled with no recovery path
        until widget reload.
        """
        # Gate: is training actually running?  Sync mode AND background
        # thread modes both set _training_abort_flag at launch (line
        # ~6112), and _train_model creates a fresh flag every time.  But
        # the flag PERSISTS across runs (it's not reset to None on
        # completion).  Use the QThread state as the authoritative signal
        # for background mode, and `_sync_training_in_flight` for sync
        # mode (set by `_train_model` around the sync block).
        sync_in_flight = bool(getattr(self, "_sync_training_in_flight", False))
        thread = getattr(self, "_training_thread", None)
        thread_running = False
        if thread is not None:
            try:
                thread_running = bool(thread.isRunning())
            except (RuntimeError, AttributeError):
                # QThread deleted by Qt already — treat as not running.
                thread_running = False
        if not (sync_in_flight or thread_running):
            # Stale invocation (keyboard shortcut, late button click after
            # _on_training_finished, programmatic call when idle).  No-op
            # silently — do NOT mutate the button.
            print(
                "[TRAIN-SAFETY] cancel_training invoked while no training "
                "is in flight (sync_in_flight=%s thread_running=%s) — "
                "ignoring." % (sync_in_flight, thread_running)
            )
            return

        flag = getattr(self, "_training_abort_flag", None)
        if flag is not None:
            flag.set()

        # Bundle 2 — UI cancel state.  Change the button label to make
        # it visually clear cancellation is in progress, AND disable it
        # so the user can't spam the abort flag.  Button gets restored
        # to "Train" when _on_training_error fires (via _restore_train_button).
        try:
            # brick T: the Cancel button says it (Train is never renamed);
            # a UI without ``train_cancel_btn`` keeps the old one-button path
            btn = (getattr(self.ui, "train_cancel_btn", None)
                   or self.ui.update_and_train_button)
            if hasattr(btn, "setText"):
                btn.setText("Cancelling...")
            if hasattr(btn, "setEnabled"):
                btn.setEnabled(False)
        except Exception:
            pass

        self._append_ui_log(
            "[TRAIN] Cancellation requested. Training stops at the next "
            "epoch checkpoint — up to ~10 epochs, since cellpose only "
            "reports every 10th epoch (longer on slow hardware)."
        )

    # ------------------------------------------------------------------
    # External layer removal handling (lifecycle manager callback)
    # ------------------------------------------------------------------

    def _on_annotator_draw_complete(self, orientation):
        """Called after each drawing stroke in the annotator subviewers.

        Reads the current slice from subviewer dims (NOT main viewer's
        dims — subviewers have independent ViewerModels), then dispatches
        to :meth:`_process_draw_complete` which holds the testable
        decision logic.
        """
        patch = str(getattr(self, "loaded_patch_name", "") or "").strip()
        if not patch:
            return
        view = str(orientation or self.selected_view or "XY").upper()

        # Determine which slice index is currently viewed.
        # Read from the SUBVIEWER's dims (axis 0), not the main viewer,
        # because subviewers have independent ViewerModels with their own
        # dims that track the actual scroll position.
        try:
            from napari_epicycle.annotator.launcher import _find_dock_by_title
            dock = _find_dock_by_title(self.viewer, "EpiCycle 2D Training Annotator")
            if dock is None:
                return
            widget = dock.widget() if hasattr(dock, "widget") else dock
            vm_map = {"XY": "viewer_model_xy", "XZ": "viewer_model_xz", "YZ": "viewer_model_yz"}
            vm = getattr(widget, vm_map.get(view, "viewer_model_xy"), None)
            if vm is None:
                return
            idx = int(vm.dims.current_step[0])  # subviewers always slice along axis 0
        except Exception:
            return

        slice_label = f"{patch}_{view}_{idx}"
        self._process_draw_complete(slice_label, view, idx)

    def _process_draw_complete(self, slice_label, view, idx):
        """Decision logic for a completed draw stroke.

        Branches:
          1. **No existing record** — fresh manual draw without prior
             segmentation.  Creates a new record with status=curating,
             origin=manual.  ALSO registers the input_stack in
             ``segmentation_manager.data_inputs`` (mirrors the segment
             path's bookkeeping) so the debounced persist can write a
             real training pair to disk — without this, the persist
             silently no-ops because input_stack is missing.
          2. **status=predicted** — segment-then-edit flow.  Transitions
             to curating + refreshes queue.  Origin from segment path
             (typically "cellpose") is preserved.
          3. **status=curating** — already being actively edited.  Just
             debounces another persist.
          4. **status in {accepted, staged, committed}** — no-op.  User
             must explicitly un-accept to re-edit.

        Empty-stroke filter: if a fresh slice has no painted pixels at
        all (e.g. user activated erase on an empty layer and dragged),
        skip record creation.  Avoids phantom records that would clutter
        the queue with empty masks.

        Console emits `[MANUAL-DRAW]` lines for each branch so live
        testing can confirm the correct path fired.
        """
        record = self.sample_records.get(slice_label)
        if not record:
            # C3: skip empty-stroke fresh slices — no labels = nothing
            # meaningful to track.  Avoids phantom records.
            if not self._fresh_slice_has_content(view, idx):
                print(f"[MANUAL-DRAW] fresh slice has no painted content — skipping record creation: {slice_label!r}")
                return

            # C1: register input_stack in segmentation_manager BEFORE
            # creating the record.  Without this, the 2-second debounced
            # persist silently no-ops because _persist_curated_training_pair
            # reads from data_inputs which would be empty.
            input_stack = self._extract_input_stack_for_view(view, idx)
            if input_stack is not None:
                self.segmentation_manager.data_names[slice_label] = True
                self.segmentation_manager.data_inputs[slice_label] = input_stack
                print(f"[MANUAL-DRAW] registered input_stack shape={input_stack.shape} dtype={input_stack.dtype}: {slice_label!r}")
            else:
                print(f"[MANUAL-DRAW] WARN no input_stack available (no image layers found) — persist will rely on Accept-time re-extraction: {slice_label!r}")

            print(f"[MANUAL-DRAW] fresh slice — creating curating record (origin=manual): {slice_label!r}")
            self._touch_sample(slice_label, status="curating", view=view, origin="manual")
            self._refresh_sample_queue()
            self._schedule_curating_persist(slice_label, view, idx)
            return
        status = record.get("status", "")
        if status == "predicted":
            origin = record.get("origin", "?")
            print(f"[MANUAL-DRAW] predicted -> curating (origin={origin!r}): {slice_label!r}")
            self._touch_sample(slice_label, status="curating", view=view)
            self._refresh_sample_queue()
        elif status == "curating":
            origin = record.get("origin", "?")
            print(f"[MANUAL-DRAW] already curating — scheduling persist (origin={origin!r}): {slice_label!r}")
        elif status in ("accepted", "staged"):
            # 2026-05-27 (edit-states branch): demote accepted/staged back
            # to curating on edit.  Pre-fix this was a silent no-op — the
            # subviewer layer mutated visually but the record stayed
            # accepted/staged and NOTHING persisted, so training used the
            # OLD mask while the user thought they'd edited it (the
            # confirmed divergence).  Now: edit a soft-locked slice ->
            # demote to curating + persist.  `_touch_sample` preserves
            # origin + inherited_from_* provenance (it only updates the
            # listed keys), so imported/cherry-picked pairs keep their
            # lineage across the demote.  The annotator's soft visual lock
            # discourages accidental edits; a deliberate paint lands here.
            origin = record.get("origin", "?")
            print(f"[MANUAL-DRAW] {status} -> curating (edit-states demote; origin={origin!r}): {slice_label!r}")
            self._touch_sample(slice_label, status="curating", view=view)
            self._append_ui_log(
                f"[CURATION] Edited {status} slice {slice_label} — reopened "
                "for editing (status: curating). Re-Accept when done."
            )
            self._refresh_sample_queue()
            # Refresh the annotator's soft-lock visuals so the buttons
            # re-enable + the slice's tint flips back to editable.
            self._refresh_annotator_edit_state_visuals()
        else:
            # committed — PROTECTED.  A committed slice already shaped a
            # trained model; reverting it has provenance implications, so
            # editing requires the explicit main-widget Revert (which
            # shows a confirm).  Drawing here is a no-op.
            print(f"[MANUAL-DRAW] status={status!r} — PROTECTED (committed); use main-widget Revert to edit: {slice_label!r}")
            self._append_ui_log(
                f"[CURATION] {slice_label} is committed (trained). Use "
                "'Revert' in the panel to reopen it for editing."
            )
            return

        # Schedule debounced re-persist of the edited mask from the volume.
        # 2-second debounce; fires once per slice per quiet window.
        self._schedule_curating_persist(slice_label, view, idx)

    def _fresh_slice_has_content(self, view, idx) -> bool:
        """Return True if the work_labels volume at (view, idx) has any
        non-zero pixels — i.e. the just-completed stroke actually drew
        something.  Used to skip phantom records from empty/erase-only
        strokes on fresh slices.  Returns False on any error so the
        caller errs on the safe side (creates the record anyway).
        """
        try:
            labels_layer_name = self._work_labels_layer_name_for_view(view_name=view)
            labels_layer = self._layer_by_name(labels_layer_name)
            if labels_layer is None:
                return True  # can't check — assume content exists
            volume = np.asarray(labels_layer.data)
            if view == "XY" and idx < volume.shape[0]:
                slice_2d = volume[idx, :, :]
            elif view == "XZ" and idx < volume.shape[1]:
                slice_2d = volume[:, idx, :]
            elif view == "YZ" and idx < volume.shape[2]:
                slice_2d = volume[:, :, idx]
            else:
                return True
            return bool(int(np.max(slice_2d)) > 0)
        except Exception:
            return True  # err on the side of NOT silently dropping

    def _extract_input_stack_for_view(self, view, idx):
        """Build (C, H, W) input_stack from the patch view image layers
        for the given (view, idx).  Mirrors the segment-path extraction
        at line ~5391.  Returns None if no image layers are available.
        """
        try:
            source_layers = self._get_patch_view_image_layers(view_name=view)
            if not source_layers:
                source_layers = self._get_any_patch_view_image_layers()
            if not source_layers:
                return None
            channels = []
            for src in source_layers:
                src_vol = np.asarray(src.data)
                if view == "XY" and idx < src_vol.shape[0]:
                    channels.append(src_vol[idx])
                elif view == "XZ" and idx < src_vol.shape[1]:
                    channels.append(src_vol[:, idx, :])
                elif view == "YZ" and idx < src_vol.shape[2]:
                    channels.append(src_vol[:, :, idx])
            if not channels:
                return None
            return np.stack(channels)
        except Exception:
            return None

    def _schedule_curating_persist(self, slice_label, view, s_idx):
        """Debounced re-persist of curating mask from the work_labels volume.

        Extracts the current 2D mask from the volume (which includes
        user edits) and writes it to disk, replacing the original
        predicted mask TIFF.

        C3 (2026-05-22): pending edits are stored in a dict keyed by
        slice_label so multiple in-flight slice edits (paint A, then
        switch to B and paint, all within the 2-second debounce) are
        ALL preserved.  Before this fix, the pending state was a
        single tuple that got overwritten on each call, causing silent
        data loss on A when the user moved to B before the timer
        fired.  Per-slice latest-wins semantics are unchanged (repeat
        calls for the same slice still overwrite that slice's entry).
        """
        timer = getattr(self, "_curating_persist_timer", None)
        if timer is None:
            from qtpy.QtCore import QTimer
            timer = QTimer()
            timer.setSingleShot(True)
            self._curating_persist_timer = timer

        # C3: dict keyed by slice_label; one entry per pending edit.
        pending = getattr(self, "_curating_persist_pending", None)
        if not isinstance(pending, dict):
            pending = {}
            self._curating_persist_pending = pending
        pending[slice_label] = (slice_label, view, s_idx)

        try:
            timer.timeout.disconnect()
        except Exception:
            pass
        timer.timeout.connect(self._flush_curating_persist_pending)
        # 2026-06-02: bumped 2000 -> 5000ms.  Post Stage-2 the write is async,
        # but each debounce FIRE still costs ~80ms on the main thread (enqueue
        # + _refresh_run_picker + _save_internal_state), which is the periodic
        # "at times" stutter.  A longer quiet-window makes that hitch fire less
        # often during continuous editing.  Hard barriers (train / switch /
        # close) flush immediately regardless, so correctness is unaffected;
        # only the hard-crash loss window widens slightly.  Tunable.
        timer.start(5000)  # 5 second debounce (was 2s)

    def _flush_curating_persist_pending(self):
        """Drain ALL pending curating-persist entries in one pass.

        C3 + C4 (2026-05-22): called by the debounce timer on fire AND
        by :meth:`shutdown` so closing napari within the 2-second
        debounce window doesn't drop pending edits.

        Iterates a snapshot of the pending dict so a re-entrant
        :meth:`_schedule_curating_persist` call from within the persist
        path (e.g. via the state-save callback) cannot mutate the dict
        mid-iteration.  Clears each entry after successful persist;
        failures leave the entry in place so the next timer fire
        retries.  ``_schedule_state_save`` is called once at the end
        rather than once per slice to avoid timer thrash.
        """
        pending = getattr(self, "_curating_persist_pending", None)
        if not isinstance(pending, dict) or not pending:
            return
        items = list(pending.items())
        any_saved = False
        for slice_label, (sl, vw, idx) in items:
            try:
                labels_layer_name = self._work_labels_layer_name_for_view(view_name=vw)
                labels_layer = self._layer_by_name(labels_layer_name)
                if labels_layer is None:
                    continue
                volume = np.asarray(labels_layer.data)
                if vw == "XY" and idx < volume.shape[0]:
                    mask2d = volume[idx, :, :].copy()
                elif vw == "XZ" and idx < volume.shape[1]:
                    mask2d = volume[:, idx, :].copy()
                elif vw == "YZ" and idx < volume.shape[2]:
                    mask2d = volume[:, :, idx].copy()
                else:
                    continue
                self.segmentation_manager.data_masks[sl] = mask2d.astype(np.int32)
                # async_write=True: this is the per-stroke debounce hot-loop —
                # offload the TIFF write to the background writer so the UI
                # never blocks.  Hard barriers flush it before train/switch/close.
                result = self._persist_curated_training_pair(sl, async_write=True)
                if result:
                    any_saved = True
                    # Step 2 look (brick A, D24): the real write -- the
                    # picture re-draws 1.5 s after the LAST one
                    _w = getattr(getattr(self, "ui", None),
                                 "on_slice_written", None)
                    if _w is not None:
                        try:
                            _w(sl)
                        except Exception:      # noqa: BLE001 -- never break the persist
                            pass
                pending.pop(slice_label, None)
            except Exception as exc:
                continue
        if any_saved:
            try:
                self._schedule_state_save()
            except Exception:
                pass
            # 2026-05-25 audit-2 H1 fix: debounced curating-persist
            # creates new *_img.tif files when a curating slice's first
            # edit settles.  Subsequent edits of the same slice are
            # cache-hit no-ops in _persist_curated_training_pair, so
            # this fires at most once per debounce window per slice.
            try:
                self._refresh_run_picker()
            except Exception as e:
                pass

    def _ensure_llm_removal_callback(self):
        """One-time setup: register callback for trash-can layer deletion.

        Mirrors Step 3's ``_ensure_llm_removal_callback`` pattern.
        """
        if getattr(self, "_llm_removal_cb_installed", False):
            return
        llm = getattr(self, "layer_lifecycle_manager", None)
        if llm is None:
            return
        llm.on_external_removal(self._on_layer_externally_removed)
        self._llm_removal_cb_installed = True

    def _on_layer_externally_removed(self, layer_name: str, context: str):
        """Handle a managed layer being deleted via napari trash can.

        If a step2 layer is deleted while the annotator is open,
        close the annotator dock gracefully to prevent silent corruption.
        """
        if not context.startswith("step2/"):
            return
        try:
            from napari_epicycle.annotator.launcher import _find_dock_by_title
            dock = _find_dock_by_title(
                self.viewer, "EpiCycle 2D Training Annotator"
            )
            if dock is not None:
                dock.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Step lifecycle hooks (called by workflow shell on step switch)
    # ------------------------------------------------------------------

    def on_step_exit(self):
        """Called when user navigates away from Step 2.

        Follows Step 3's on_step_exit pattern: close annotator dock,
        clean up layers using targeted methods first, lifecycle manager
        as safety net.
        """
        # 1. Close the HITL 2D annotator dock if open.
        try:
            from napari_epicycle.annotator.launcher import _find_dock_by_title
            dock = _find_dock_by_title(self.viewer, "EpiCycle 2D Training Annotator")
            if dock is not None:
                dock.close()
        except Exception as exc:
            pass

        # 2. Clean up SAM prompt layers.
        try:
            self.interactive_manager.cleanup_prompt_layers(self.viewer)
        except Exception:
            pass

        # 2b. Forget ALL step2 contexts BEFORE layer removal.
        # Without this, each viewer.layers.remove() fires the LLM's
        # _on_layer_removed → _on_layer_externally_removed → dock.close()
        # on the already-closing dock (same bug as F1 in _load_patch).
        llm = getattr(self, "layer_lifecycle_manager", None)
        if llm is not None:
            llm.forget_context("step2/patch")
            llm.forget_context("step2/annotator")

        # 3. Remove all HITL layers (images, work labels, RGB stacks).
        to_remove = []
        for layer in list(self.viewer.layers):
            name = str(getattr(layer, "name", ""))
            if (
                name.endswith("_view")
                or "_work_labels_" in name
                or name.endswith("_native_rgb_stack")
                or name == "final_labels"
            ):
                to_remove.append(name)
                try:
                    layer.visible = False
                    self.viewer.layers.remove(layer)
                except Exception:
                    pass

        # 4. Lifecycle manager safety net.
        llm = getattr(self, "layer_lifecycle_manager", None)
        if llm is not None:
            llm.clear_step("step2")

        # 5. Brick D: an async patch load still in flight must not install its
        # layers into a step the user has left.
        self._step2_left = True

    def on_step_enter(self):
        """Called when user navigates to Step 2.

        Also does the cheap list refresh: a
        patch extracted or trimmed in Step 1 used to stay invisible here
        until the user pressed Browse, because entry only flipped a flag.
        Disk read + list rebuild only -- no state load, no patch load, no
        modal -- and never on a step with no workspace.
        """
        self._step2_left = False
        if not (getattr(self, "working_dir", "") or "").strip():
            return
        try:
            self._populate_patch_list_from_disk(notify=False)
            self._refresh_patch_list_badges()
        except Exception as exc:      # noqa: BLE001
            # A step switch must never die on a filesystem hiccup.
            print(f"[WORKSPACE] Step 2 re-entry refresh failed: {exc}")

    def shutdown(self, *_args):
        """Graceful shutdown: save state and wait for any running training thread.

        C4 (2026-05-22): also flushes the curating-persist timer.  Prior
        to this fix, closing napari within the 2-second debounce
        window after the last paint stroke would drop any pending
        curated-edit persists, silently losing the user's edits to the
        on-disk training pair masks.  See :meth:`_flush_curating_persist_pending`.

        Release-hardening #1 (2026-09-07): idempotent — both the viewer
        close event and the ``ui.destroyed`` fallback route here, and on
        some platforms both fire.
        """
        if getattr(self, "_shutdown_done", False):
            return
        self._shutdown_done = True
        # Release-hardening 1b#1 (2026-09-07): tell the training thread to
        # stop FIRST.  Everything below can block — the curating flush, the
        # pair-writer queue join, the state save — and a stuck write used to
        # mean the abort flag was never set at all.  Setting it here is
        # cheap, thread-safe and has no side effects when nothing is running.
        _abort = getattr(self, "_training_abort_flag", None)
        if _abort is not None:
            try:
                _abort.set()
            except Exception:
                pass
        # Step 2 look (brick L, hook 2; AFTER the abort flag so its bounded wait never delays the abort): stop the thumbnail batch (a bounded
        # wait inside the UI).  ``ui.destroyed`` routes here too.
        try:
            _hook = getattr(self.ui, "on_shutdown", None)
            if _hook:
                _hook()
        except Exception:      # noqa: BLE001 -- shutdown must go on
            pass
        # Flush any pending debounced state save.  Both debounce timers are
        # parentless QTimers: at napari close Qt may have deleted their C++
        # side already, and isActive() on such a wrapper raises RuntimeError
        # -- which skipped every step below (the curating flush, the writer
        # flush, the state save, the training join).  Same guard as
        # _flush_pending_state_timers.
        timer = getattr(self, "_state_save_timer", None)
        try:
            if timer is not None and timer.isActive():
                timer.stop()
        except RuntimeError:
            pass
        # C4: flush curating-persist timer + drain any pending edits.
        curating_timer = getattr(self, "_curating_persist_timer", None)
        try:
            if curating_timer is not None and curating_timer.isActive():
                curating_timer.stop()
        except RuntimeError:
            pass
        try:
            self._flush_curating_persist_pending()
        except Exception:
            pass
        # Stage 2: block until all background TIFF writes complete before we
        # save state / exit, so closing napari can't drop an in-flight async
        # write of the user's last edit.
        writer = getattr(self, "_pair_writer", None)
        if writer is not None:
            try:
                # Bounded on the shutdown path (1b#1) — see _AsyncPairWriter.flush.
                writer.flush(timeout=_SHUTDOWN_WRITE_FLUSH_TIMEOUT_S)
            except TypeError:
                # Older/stub writers without the timeout kwarg.
                try:
                    writer.flush()
                except Exception:
                    pass
            except Exception:
                pass
        try:
            self._save_internal_state()
        except Exception:
            pass
        # Brick D: join the patch-load thread.  It only holds a TIFF read
        # (measured 10 ms on real lab patches, linear in file size), so a
        # short bounded wait is enough; abandoning it would let a worker
        # emit into a half-torn-down controller.
        # Disarm FIRST (Brick D fix 2026-09-07): if the join below times out,
        # the worker can still emit into a controller that is half torn down.
        # `_step2_left` + a token bump make every in-flight finish stale, so
        # the late signal can only reach the discard branch.
        self._step2_left = True
        self._patch_load_token = int(getattr(self, "_patch_load_token", 0)) + 1
        patch_thread = getattr(self, "_patch_load_thread", None)
        if patch_thread is not None:
            try:
                if patch_thread.isRunning():
                    patch_thread.quit()
                    if not patch_thread.wait(_PATCH_LOAD_JOIN_TIMEOUT_MS):
                        print("[STEP2-LOAD] patch-load thread did not join in "
                              f"{_PATCH_LOAD_JOIN_TIMEOUT_MS} ms; abandoning")
            except Exception:
                pass

        # The abort flag was already set at the top of shutdown() (1b#1);
        # re-setting is harmless but unnecessary.  Stop the training thread.
        thread = getattr(self, "_training_thread", None)
        if thread is not None and thread.isRunning():
            thread.quit()
            # F1.5 (2026-05-27 audit): bumped from 5s → 30s.  Real
            # Cellpose epochs commonly take longer than 5s on small GPUs;
            # the old timeout would silently abandon the thread mid-
            # write, truncating the weights file.  30s isn't a hard
            # guarantee but covers the vast majority of single-epoch
            # tails.  True fix needs upstream cooperation (Cellpose
            # respecting abort mid-epoch).  The .training_in_progress
            # marker file (written at train start, deleted on clean
            # finish) lets the next session detect partial state.
            print("[TRAIN-SAFETY] F1.5 shutdown wait up to 30s for training thread...")
            finished = thread.wait(30000)
            print(f"[TRAIN-SAFETY] F1.5 shutdown wait finished={finished}")
            # Bundle 1 follow-up (R1 audit C1): if wait succeeded, the
            # thread completed cleanly — but the worker.finished slot
            # (which clears the marker via _on_training_finished) may
            # never run because the UI thread was parked in wait().
            # Clear marker synchronously here so next session doesn't
            # see a false "stale marker" warning.  If wait TIMED OUT,
            # leave the marker — training was killed without finishing,
            # marker correctly signals partial state.
            if finished:
                try:
                    self._clear_training_in_progress_marker()
                    print("[TRAIN-SAFETY] F1.5/C1 shutdown — marker cleared (thread finished cleanly)")
                except Exception:
                    pass

    def _write_step2_model_manifest_for_round(
        self, *, run_id, bucket, round_idx, view_type, losses, val_losses,
        config, n_train_pairs, pretrained_model, weights_filename="weights",
        working_dir=None,
        # Phase B.3 (2026-05-25) — schema 2.1 reproducibility kwargs.
        # All optional; build_step2_model_manifest tolerates None.
        train_pair_labels=None, train_pair_uris=None,
        timing_started_at=None, timing_finished_at=None,
        timing_duration_seconds=None,
        env_cellpose_version=None, env_torch_version=None,
        env_python_version=None, env_device=None, env_platform=None,
        # Release-hardening #11 (2026-09-07) — additive provenance flag.
        inference_failed=False,
    ):
        """Phase 6c (2026-05-26): write per-bucket-round schema 2.0 manifest.

        Lands at ``runs/{run_id}/{bucket}/models/round_NN/manifest.json``
        alongside the lifted Cellpose weights.  Uses the data-layer
        helpers from ``packages/epicycle_training/step2_model_manifest.py``
        (already shipped, just no production caller before now).

        Best-effort — if anything fails, the weights file is still at
        the canonical path so functional training works; only manifest
        provenance is degraded.

        Args:
            run_id: active run slug
            bucket: "lateral" or "axial"
            round_idx: 0-indexed round number
            view_type: legacy view tag from training pipeline
                ("spatial"/"axial"/etc.) — recorded as-is for ref
            losses: list of train-loss floats from training result
            val_losses: list of val-loss floats (may be shorter than
                losses; Cellpose validates every N epochs)
            config: training config snapshot
            n_train_pairs: number of samples used in this view
            pretrained_model: path to init checkpoint, or None
            weights_filename: filename Cellpose was told to use as
                model_name (e.g. f"{run_id}_{bucket}")
            working_dir: workspace path snapshot (Commit 3 audit C1
                fix).  When None, falls back to `self.working_dir`
                — but the worker thread should always pass the
                snapshot to avoid mid-train workspace-switch races.
        """
        from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
        from packages.epicycle_training.step2_model_manifest import (
            build_step2_model_manifest,
            write_step2_model_manifest,
        )

        # Commit 2 audit CRITICAL C1 fix (2026-05-26): use the caller-
        # supplied `working_dir` snapshot (captured at train-start),
        # NOT `self.working_dir`.  This helper runs from the worker
        # thread; if the user switches workspace mid-train, every
        # `self.working_dir` read here would land in the NEW workspace
        # while the trained weights live in the OLD workspace.  The
        # M-1 fix protected `_on_training_finished` but not this helper.
        ws = working_dir if working_dir is not None else self.working_dir

        # H-5 fix (2026-05-26): on the FIRST round of this bucket, read
        # the run_manifest's `created_from` (init_spec) to map into
        # schema 2.0's init.mode + inherited_from_* fields.  Subsequent
        # rounds inherit from the prior round within the same run, so
        # init_mode="inherited" via the pretrained_model checkpoint path
        # (existing logic below).  This is the only way the schema 2.0
        # manifest can record the user's choice in the New Run dialog
        # (scratch / pretrained cyto3 / step3_refined / browse / fork /
        # cross_bucket) — it was being lost entirely pre-fix.
        first_round_init_spec = None
        if int(round_idx) == 0:
            try:
                from packages.epicycle_io.step2_run_manifest import read_run_manifest
                rm_path = _step2_paths.run_manifest_path(ws, run_id)
                rm = read_run_manifest(rm_path)
                if rm and isinstance(rm.get("created_from"), dict):
                    first_round_init_spec = dict(rm["created_from"])
                    print(f"[STEP2-TRAIN] H-5 round 0: using run_manifest "
                          f"created_from={first_round_init_spec.get('mode')!r}")
            except Exception as _e:
                print(f"[STEP2-TRAIN] H-5 round 0 init_spec read failed "
                      f"({type(_e).__name__}: {_e}); falling through to "
                      f"checkpoint-path heuristic")

        # Default fields
        inherited_from_run = None
        inherited_from_run_bucket = None
        inherited_from_run_round_id = None
        inherited_from_bucket = None
        inherited_from_step3_bundle = None
        inherited_from_step3_bucket = None
        inherited_from_external_path = None
        inherited_from_external_label = None

        if first_round_init_spec is not None:
            _spec_mode = str(first_round_init_spec.get("mode", "scratch"))
            if _spec_mode == "scratch":
                init_mode = "scratch"
                init_path = None
                init_combo_text = "From scratch (random weights)"
            elif _spec_mode == "pretrained":
                init_mode = "pretrained"
                init_path = first_round_init_spec.get("pretrained_name")
                init_combo_text = f"Cellpose pretrained: {init_path}"
            elif _spec_mode == "inherited":
                init_mode = "inherited"
                init_path = pretrained_model and str(pretrained_model) or None
                inherited_from_run = first_round_init_spec.get("source_run_id")
                # H2 fix (2026-05-26): preserve source bucket + round
                # as recoverable provenance (was previously only in
                # human-readable init_combo_text).
                inherited_from_run_bucket = first_round_init_spec.get(
                    "source_bucket"
                )
                inherited_from_run_round_id = first_round_init_spec.get(
                    "source_round_id"
                )
                init_combo_text = (
                    f"From run {inherited_from_run!r} "
                    f"{inherited_from_run_bucket} "
                    f"{inherited_from_run_round_id}"
                )
            elif _spec_mode == "cross_bucket":
                init_mode = "cross_bucket"
                init_path = pretrained_model and str(pretrained_model) or None
                inherited_from_bucket = first_round_init_spec.get("source_bucket")
                init_combo_text = (
                    f"Cross-bucket from {inherited_from_bucket} "
                    f"{first_round_init_spec.get('source_round_id')}"
                )
            elif _spec_mode == "step3_refined":
                init_mode = "step3_refined"
                init_path = pretrained_model and str(pretrained_model) or None
                inherited_from_step3_bundle = first_round_init_spec.get("source_step3_bundle")
                inherited_from_step3_bucket = first_round_init_spec.get("source_step3_bucket")
                init_combo_text = (
                    f"Step 3 refined: "
                    f"{inherited_from_step3_bundle} / "
                    f"{inherited_from_step3_bucket}"
                )
            elif _spec_mode == "browse":
                init_mode = "browse"
                init_path = first_round_init_spec.get("external_path")
                inherited_from_external_path = init_path
                inherited_from_external_label = first_round_init_spec.get(
                    "external_label", ""
                )
                init_combo_text = f"External file: {inherited_from_external_label or init_path}"
            elif _spec_mode == "fork":
                # Fork = pairs copied from another run; init weights
                # come from that run's terminal round if any, else
                # scratch.  Schema 2.0 doesn't have "fork" — record as
                # "inherited" with source_run_id.  H2 fix: fork spec
                # doesn't carry bucket/round (forks point at a run, not
                # a specific round), so leave those fields None — that's
                # honest provenance: "this inherited from run X (we
                # don't know which exact round)".
                init_mode = "inherited"
                init_path = pretrained_model and str(pretrained_model) or None
                inherited_from_run = first_round_init_spec.get("source_run_id")
                init_combo_text = f"Fork of {inherited_from_run!r}"
            else:
                # Unknown mode — fall through to heuristic
                first_round_init_spec = None

        if first_round_init_spec is None:
            # Heuristic: prior-round checkpoint path → inherited.
            # config.model_type != scratch → pretrained.  Else scratch.
            if pretrained_model and Path(pretrained_model).is_file():
                init_mode = "inherited"
                init_path = str(pretrained_model)
                init_combo_text = f"Prior checkpoint: {Path(pretrained_model).name}"
            else:
                base_model = str(config.get("model_type", "scratch"))
                if base_model and base_model != "scratch":
                    init_mode = "pretrained"
                    init_path = base_model
                    init_combo_text = f"Cellpose pretrained: {base_model}"
                else:
                    init_mode = "scratch"
                    init_path = None
                    init_combo_text = "From scratch (random weights)"

        # Final losses
        final_loss_train = float(losses[-1]) if losses else None
        final_loss_val = float(val_losses[-1]) if val_losses else None

        # H-6 fix (2026-05-26): extract channels from config.  channel1
        # is image channel (e.g. "0: gray"), channel2 is optional 2nd
        # (e.g. "0: none").
        try:
            ch1 = str(config.get("channel1", "0: gray"))
            ch2 = str(config.get("channel2", "0: none"))
            channels_used = [ch1, ch2]
        except Exception:
            channels_used = []
        # Phase B.3 (2026-05-25) — channels_snapshot bug fix.
        # Commit 2 audit M1 noted that `self.channel_names_snapshot`
        # was never set anywhere → manifest always wrote [].  The
        # CORRECT source is `run_manifest.json::channels_snapshot`,
        # which the New Run dialog records at run creation from
        # `_snapshot_viewer_channels()`.  Reading from there gives
        # the user's committed channel order, immune to viewer
        # reordering mid-session.  Three-tier fallback:
        #   1. run_manifest.json::channels_snapshot (frozen at create)
        #   2. self._snapshot_viewer_channels() (live viewer)
        #   3. [] (genuine empty — no provenance available)
        channels_snapshot: List[str] = []
        try:
            from packages.epicycle_io.step2_run_manifest import read_run_manifest
            rm_path = _step2_paths.run_manifest_path(ws, run_id)
            rm = read_run_manifest(rm_path)
            if rm and isinstance(rm.get("channels_snapshot"), list) and rm["channels_snapshot"]:
                channels_snapshot = list(rm["channels_snapshot"])
        except Exception as _csne:
            print(f"[STEP2-TRAIN] B.3 channels_snapshot: run_manifest read "
                  f"failed ({type(_csne).__name__}: {_csne}); trying viewer")
        if not channels_snapshot:
            try:
                live = self._snapshot_viewer_channels()
                if isinstance(live, list) and live:
                    channels_snapshot = list(live)
            except Exception as _csve:
                print(f"[STEP2-TRAIN] B.3 channels_snapshot: viewer "
                      f"snapshot failed ({type(_csve).__name__}: {_csve}); "
                      f"writing [] (no provenance available)")
        # Defensive legacy compat — honor `self.channel_names_snapshot`
        # if anyone ever sets it (pre-fix code path; never hit today).
        if not channels_snapshot:
            _snap_attr = getattr(self, "channel_names_snapshot", None)
            if isinstance(_snap_attr, (list, tuple)) and _snap_attr:
                channels_snapshot = list(_snap_attr)

        manifest = build_step2_model_manifest(
            run_id=run_id,
            bucket=bucket,
            round_number=round_idx,
            source_workspace=str(ws or ""),
            weights_filename=weights_filename,
            init_mode=init_mode,
            init_path=init_path,
            init_combo_text=init_combo_text,
            final_loss_train=final_loss_train,
            final_loss_val=final_loss_val,
            n_train_pairs=int(n_train_pairs),
            n_epochs=int(config.get("epochs", 0)),
            batch_size=int(config.get("batch_size", 0)),
            channels_used=channels_used,
            channels_snapshot=channels_snapshot,
            # Phase B.3 (2026-05-25) — schema 2.1 reproducibility fields.
            train_pair_labels=train_pair_labels,
            train_pair_uris=train_pair_uris,
            random_seed=None,  # Cellpose doesn't expose its seed today
            timing_started_at=timing_started_at,
            timing_finished_at=timing_finished_at,
            timing_duration_seconds=timing_duration_seconds,
            env_cellpose_version=env_cellpose_version,
            env_torch_version=env_torch_version,
            env_python_version=env_python_version,
            env_device=env_device,
            env_platform=env_platform,
            inherited_from_run=inherited_from_run,
            inherited_from_run_bucket=inherited_from_run_bucket,
            inherited_from_run_round_id=inherited_from_run_round_id,
            inherited_from_bucket=inherited_from_bucket,
            inherited_from_step3_bundle=inherited_from_step3_bundle,
            inherited_from_step3_bucket=inherited_from_step3_bucket,
            inherited_from_external_path=inherited_from_external_path,
            inherited_from_external_label=inherited_from_external_label,
            backend="cellpose",
        )
        # Release-hardening #11 (2026-09-07): additive flag — the weights
        # are real and usable, only the post-train inference pass failed.
        # Pre-fix the manifest was skipped entirely, so the model had NO
        # provenance while Step 3 still found the weights file by name.
        if inference_failed:
            manifest["inference_failed"] = True
        manifest_path = _step2_paths.model_manifest_path(
            ws, run_id, bucket, round_idx,
        )

        # C-5 fix (2026-05-26): archive prior manifest before overwrite.
        # If an existing manifest is present at this path (re-train of
        # the same round), copy it into manifest_history/{ts}_{uuid}.json
        # so provenance is preserved.  Best-effort; archive failure
        # does NOT block the new manifest write.
        #
        # Commit 2 audit H3 fix (2026-05-26): include uuid4 hex suffix in
        # the archive filename.  Microsecond timestamp collisions on
        # rapid re-train (cancel + retrain within same microsecond) would
        # otherwise cause `shutil.copy2` to silently overwrite the prior
        # archive — invisible provenance loss.
        try:
            if manifest_path.exists():
                import shutil
                import uuid as _uuid
                from datetime import datetime as _dt, timezone as _tz
                archive_dir = manifest_path.parent / "manifest_history"
                archive_dir.mkdir(parents=True, exist_ok=True)
                ts = _dt.now(_tz.utc).strftime("%Y%m%dT%H%M%S%fZ")
                archive_path = archive_dir / f"{ts}_{_uuid.uuid4().hex[:8]}.json"
                shutil.copy2(manifest_path, archive_path)
                print(f"[STEP2-TRAIN] C-5 archived prior manifest to "
                      f"{archive_path}")
        except Exception as _e:
            print(f"[STEP2-TRAIN] C-5 manifest archive failed (continuing): "
                  f"{type(_e).__name__}: {_e}")

        write_step2_model_manifest(manifest_path, manifest)
        print(f"[STEP2-TRAIN] wrote schema 2.0 manifest: {manifest_path} "
              f"(init_mode={init_mode!r}, n_train={n_train_pairs}, "
              f"final_train={final_loss_train}, final_val={final_loss_val}, "
              f"channels={channels_used})")

    def _on_training_finished(self, result, staged_slice_labels, round_id):
        """Finalize round state after background training completes (runs on UI thread)."""
        self._ensure_runtime_defaults()
        samples = result["samples"]
        round_losses = result["round_losses"]
        round_checkpoints = result["round_checkpoints"]
        round_inference = result["round_inference"]
        round_backend = result["round_backend"]

        # Phase 6c (2026-05-26): gate the legacy artifact JSON writes
        # for new-layout workspaces.  The schema 2.0 manifest written
        # inside the model's bundle dir (model_dir/manifest.json) is
        # the canonical record post-Phase-6c.  On new-layout workspaces
        # we SKIP these legacy writes to avoid two stale records.  On
        # legacy workspaces we keep the original behavior.
        #
        # M-1 fix (2026-05-26): use the snapshot captured at training
        # start, NOT a fresh `is_new_layout_state(self.working_dir)`
        # read.  If the user switched workspace between train-start +
        # finished-signal, the two callsites must agree.  Fall back to
        # the fresh read for back-compat with placeholder/legacy paths
        # that don't populate the snapshot key.
        if "is_new_layout" in result:
            _on_new_layout = bool(result.get("is_new_layout"))
        else:
            try:
                from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
                _on_new_layout = _step2_paths.is_new_layout_state(self.working_dir)
            except Exception:
                _on_new_layout = False
        if _on_new_layout:
            print(f"[STEP2-TRAIN] new layout: skipping legacy "
                  f"artifacts/models + artifacts/inference JSON writes "
                  f"(schema 2.0 manifest in model_dir is canonical)")
        else:
            # Save artifacts (legacy layout — lightweight, fine on UI thread).
            for view_type, losses in round_losses.items():
                checkpoint_path = round_checkpoints.get(view_type)
                if checkpoint_path:
                    model_artifact = ModelArtifact(
                        schema_version="1.0",
                        model_id=f"{self.current_config.get('model_name')}_{view_type}_{round_id}",
                        view_type=view_type,
                        checkpoint_uri=checkpoint_path,
                        training_data_ref=round_id,
                        metrics={"final_loss": float(losses[-1]) if losses else 0.0},
                    )
                    self.persistence.artifact_registry.save_model_artifact(self.working_dir, model_artifact)

                prediction_uri = round_inference.get(view_type, "")
                if prediction_uri and checkpoint_path:
                    inference_artifact = InferenceArtifact(
                        schema_version="1.0",
                        model_id=f"{self.current_config.get('model_name')}_{view_type}_{round_id}",
                        dataset_id=(self.loaded_patch_name if hasattr(self, "loaded_patch_name") else "default_dataset"),
                        view_type=view_type,
                        prediction_uri=prediction_uri,
                    )
                    self.persistence.artifact_registry.save_inference_artifact(self.working_dir, inference_artifact)

        # Phase B.1 (2026-05-25): Mode=Both DRIFT fix.
        # Pre-fix, this block read `self._current_round_key()` ONCE and
        # wrote ALL of training_data_by_round / losses / checkpoints under
        # a SINGLE composite key.  In Mode=Both that meant:
        #   - lateral+axial samples lumped into `training_data_by_round["lateral_5"]`
        #   - axial's entry never written (silent data loss from state JSON)
        #   - second Mode=Both pass overwrote round_NN of the
        #     non-selected bucket (counter only incremented ONE bucket).
        #
        # Post-fix: iterate the buckets that ACTUALLY trained (derived
        # from `round_losses.keys()` which `_run_training_round` populated
        # per-view).  Split `samples` per bucket using sample["view"] tag.
        # Per-bucket: write composite-key state, write metadata at
        # bucket's own round_idx (read from per-bucket counter PRE-
        # increment), then increment that bucket's counter.
        #
        # Map view_type strings ("spatial"/"axial") used by training
        # pipeline back to bucket names ("lateral"/"axial") for state-
        # JSON composite keys.  The "spatial" → "lateral" remapping is
        # the same one used elsewhere.
        def _bucket_for_view_type(vt: str) -> str:
            return "lateral" if str(vt).lower() in ("xy", "spatial", "lateral") else "axial"

        # Group samples by bucket via their `view` tag.
        samples_by_bucket: dict = {}
        for s in samples:
            sv = str(s.get("view", "")).upper()
            bkt = "lateral" if sv == "XY" else ("axial" if sv in ("XZ", "YZ") else None)
            if bkt is None:
                continue
            samples_by_bucket.setdefault(bkt, []).append(s)

        # Determine PRE-increment round_idx for each bucket trained.
        # Per-bucket counter was already at its training round; this
        # block is the increment site.
        trained_buckets_round_idx: dict = {}
        for view_type in round_losses.keys():
            bkt = _bucket_for_view_type(view_type)
            if bkt not in trained_buckets_round_idx:
                trained_buckets_round_idx[bkt] = int(
                    self.current_round_by_view.get(bkt, 0)
                )

        # F4.1 (2026-05-27 audit follow-up): derive successful_buckets
        # HERE — needed by state-write gate, commit-gate, AND counter
        # increment.  A bucket is "successful" only if it produced a
        # checkpoint; merely emitting a loss curve isn't enough.
        successful_buckets: set = set()
        for _vt in round_checkpoints.keys():
            _b = _bucket_for_view_type(_vt)
            if _b:
                successful_buckets.add(_b)

        # Per-bucket state writes — F4.1: only for successful buckets.
        # Pre-fix: failed buckets still got training_data / losses /
        # backend_info written → orphan entries in state JSON.
        for bucket_name, bucket_round_idx in trained_buckets_round_idx.items():
            if bucket_name not in successful_buckets:
                print(f"[STEP2-TRAIN] F4.1 SKIP per-bucket state write for "
                      f"bucket={bucket_name!r} — no checkpoint produced "
                      f"(prevents orphan state entries)")
                continue
            round_key = f"{bucket_name}_{bucket_round_idx}"
            # training_data: this bucket's samples only
            self.training_data_by_round[round_key] = (
                samples_by_bucket.get(bucket_name, [])
            )
            # losses: find the view_type that maps to this bucket
            bucket_losses = {
                vt: losses for vt, losses in round_losses.items()
                if _bucket_for_view_type(vt) == bucket_name
            }
            if len(bucket_losses) == 1:
                self.losses_by_round[round_key] = next(iter(bucket_losses.values()))
            elif bucket_losses:
                self.losses_by_round[round_key] = bucket_losses
            # checkpoints: same per-bucket filter
            bucket_checkpoints = {
                vt: ckpt for vt, ckpt in round_checkpoints.items()
                if _bucket_for_view_type(vt) == bucket_name
            }
            if len(bucket_checkpoints) == 1:
                self.model_checkpoint_paths[round_key] = next(
                    iter(bucket_checkpoints.values())
                )
            elif bucket_checkpoints:
                self.model_checkpoint_paths[round_key] = bucket_checkpoints
            self.round_backend_info[round_key] = round_backend
            print(f"[STEP2-TRAIN] Phase B.1 per-bucket state write: "
                  f"key={round_key} "
                  f"n_samples={len(samples_by_bucket.get(bucket_name, []))} "
                  f"has_loss={bool(bucket_losses)} "
                  f"has_ckpt={bool(bucket_checkpoints)}")

        self._promote_round_checkpoints_to_curation(round_checkpoints)

        # Finalize loss plot — promote live curve and add completed round.
        # Prefix labels with bucket so the plot can filter by active view.
        # Phase B.1.1 fix (audit 1+2 HIGH-1): pre-fix label used
        # `round_key` from the per-bucket for-loop above — Python loop
        # variables leak, so `round_key` here was the LAST bucket's key.
        # In Mode=Both both curves got labeled with axial's key.  Post-
        # fix: derive label PER-VIEW from view_type → bucket → its own
        # round_idx (looked up in trained_buckets_round_idx).
        loss_plot = getattr(self.ui, "loss_plot", None)
        if loss_plot is not None:
            loss_plot.finish_live_round()
            for view_type, losses in round_losses.items():
                _label_bucket = _bucket_for_view_type(view_type)
                _label_round = trained_buckets_round_idx.get(_label_bucket, 0)
                _label_key = f"{_label_bucket}_{_label_round}"
                label = f"{_label_key}" if len(round_losses) == 1 else f"{_label_key}/{view_type}"
                loss_plot.add_round_curve(label, losses)
            self._notify_train_finished()

        # Log details.
        self.ui.training_log_output.append(
            f"[{datetime.now(timezone.utc).isoformat()}] Trained {round_id} with {len(samples)} sample(s)."
        )
        for view_type, checkpoint_path in round_checkpoints.items():
            self.ui.training_log_output.append(f"{view_type} checkpoint: {checkpoint_path}")
        for view_type, prediction_uri in round_inference.items():
            self.ui.training_log_output.append(f"{view_type} inference manifest: {prediction_uri}")

        # Mark staged samples as committed — F4.1: only for samples whose
        # bucket is in successful_buckets.  Pre-fix: a Mode=Single fail
        # still committed every staged sample, forcing the user to
        # manually un-stage to retry.  Post-fix: failed-bucket samples
        # stay staged.
        # Release-hardening #6 (2026-09-07): a staged slice with no
        # image/mask is silently skipped by `prepare_round_dataset`; it
        # must NOT be marked committed (= "a model trained on this").
        # `used_slice_labels` is absent for placeholder/legacy result dicts
        # -> fall back to the pre-fix behaviour of trusting every label.
        _used = result.get("used_slice_labels")
        _used_set = set(_used) if isinstance(_used, (list, tuple, set)) else None
        for sample_id in staged_slice_labels:
            rec = self.sample_records.get(sample_id)
            if rec is None:
                continue
            if _used_set is not None and sample_id not in _used_set:
                msg = (f"[TRAIN] {sample_id} was NOT used for training "
                       f"(no image/mask found) — it stays staged.")
                print(f"[STEP2-TRAIN] #6 SKIP commit for {sample_id!r} — "
                      f"not in used_slice_labels")
                self._append_ui_log(msg)
                continue
            sample_bucket = self._bucket_for_view(rec.get("view", ""))
            if sample_bucket not in successful_buckets:
                print(f"[STEP2-TRAIN] F4.1 SKIP commit for {sample_id!r} — "
                      f"bucket={sample_bucket!r} had no checkpoint; "
                      f"sample stays staged for retry")
                continue
            self._touch_sample(sample_id, status="committed", round_tag=round_id)

        # Phase B.1 (2026-05-25): Mode=Both DRIFT fix + F4.1.
        # successful_buckets was derived earlier (above the per-bucket
        # state-write loop) and is reused here.
        for bucket_name in trained_buckets_round_idx.keys():
            if bucket_name not in successful_buckets:
                print(f"[STEP2-TRAIN] F4.1 SKIP counter increment for "
                      f"bucket={bucket_name!r} — no checkpoint produced "
                      f"(training failed mid-round, see failed_views)")
                continue
            self.current_round_by_view[bucket_name] = (
                self.current_round_by_view.get(bucket_name, 0) + 1
            )
            print(f"[STEP2-TRAIN] Phase B.1 counter increment: "
                  f"{bucket_name}={self.current_round_by_view[bucket_name]}")
        # Keep legacy alias in sync for backward-compat code paths.
        self.current_round = self._current_round()

        # Persist state immediately after round increment so it's consistent on crash.
        try:
            self._save_internal_state()
        except Exception as exc:
            self._append_ui_log(f"[WARN] State save after training failed: {exc}")

        # Release-hardening 2026-09-07 (bucket-switch item #7): NO automatic
        # `training_config.json` at the workspace root any more.  It was a
        # workspace-global snapshot of one bucket's widgets, written after
        # every training round, that nothing reads back -- the real state
        # lives per-run in step2_state.json.  "Save Config..." / "Load
        # Config..." remain, and now default into the active run's folder.
        # Existing root files are left alone.

        # Auto-populate starting checkpoint for next round.
        # 2026-05-29 UX fix: guard the setText with _init_transparency_loading
        # so _on_init_widget_changed does NOT auto-flip Training Mode to From
        # Scratch.  This is a SYSTEM populate (not user picking) — the user
        # was in Cumulative; auto-populating the prior round's checkpoint into
        # the display field should not change their training mode.
        #
        # Side-effect note (audit-flagged, intentional): the guard also skips
        # `_on_starting_checkpoint_changed`'s write to
        # `current_config["starting_checkpoint"]` + save.  The user's last
        # manually-set Checkpoint value remains the persisted "what training
        # will start from" preference; the widget displays the prior-round
        # path for in-session context.  `_collect_training_config` reads the
        # live widget at train time, so training behavior is unaffected.
        if round_checkpoints:
            latest_ckpt = next(iter(round_checkpoints.values()))
            ckpt_edit = getattr(self.ui, "starting_checkpoint_path", None)
            if ckpt_edit is not None and hasattr(ckpt_edit, "setText"):
                _prev_guard = getattr(self, "_init_transparency_loading", False)
                self._init_transparency_loading = True
                try:
                    ckpt_edit.setText(str(latest_ckpt))
                finally:
                    self._init_transparency_loading = _prev_guard
                # _on_init_widget_changed was suppressed above — refresh the
                # init-hint label so it reflects the new widget value
                # ("Cumulative will inherit from round_NN's checkpoint" etc.).
                try:
                    self._refresh_init_hint()
                except Exception:
                    pass

        # Refresh UI.
        self._sync_data_to_add_list(force=True)
        self._refresh_sample_queue()
        self._refresh_round_status()
        # 2026-05-25 audit fix (Group C): bump the Run dropdown so the
        # new round_NN/ subdir is reflected in `lat:N ax:M` counters.
        # Without this, the dropdown stays stale until the user does
        # something else (workspace reload, switch run, create, import).
        try:
            self._refresh_run_picker()
        except Exception as e:
            pass

        self.ui.training_progress_bar.setValue(100)
        self._restore_train_button()
        self._set_training_ui_locked(False)
        self._append_ui_log(f"[TRAIN] Complete for {round_id}. Advanced to round {self.current_round}.")
        # Release-hardening #2 (2026-09-07): a per-view training or
        # inference failure was collected into round_backend["failed_views"]
        # and then never shown — the round reported "complete", the loss
        # plot was drawn, and only one bucket's model actually existed.
        self._report_failed_views(round_backend, successful_buckets=successful_buckets)
        # F1.5: clear marker on clean completion.
        try:
            self._clear_training_in_progress_marker()
        except Exception:
            pass

    def _quarantine_corrupt_state_file(self, path, error=None):
        """Move a corrupt JSON state file aside (release-hardening #3).

        Returns the quarantine path or None.  Best-effort: a failure here
        must never block the save that is about to rewrite the file.
        """
        try:
            return _quarantine_corrupt_file(path)
        except Exception as _e:
            print(f"[STATE-SAVE] quarantine of {path} failed "
                  f"({type(_e).__name__}: {_e})")
            return None

    def _report_failed_views(self, round_backend, successful_buckets=None):
        """Surface per-view training/inference failures (defect #2).

        `_run_training_round` catches a per-view exception, records it in
        ``round_backend["failed_views"]`` and continues so the OTHER bucket
        still trains.  Pre-fix that record never reached the user: the round
        announced "Complete", so a half-trained round looked identical to a
        good one.  Now every failure gets a log line and (outside headless /
        sync-training test contexts) one modal naming the affected bucket(s).
        """
        try:
            failed = list((round_backend or {}).get("failed_views") or [])
        except Exception:
            failed = []
        if not failed:
            return

        buckets = []
        for entry in failed:
            view_type = str((entry or {}).get("view_type", "?"))
            stage = str((entry or {}).get("stage", "?"))
            error = str((entry or {}).get("error", ""))
            bucket = "lateral" if view_type.lower() in ("xy", "spatial", "lateral") else "axial"
            if bucket not in buckets:
                buckets.append(bucket)
            self._append_ui_log(
                f"[TRAIN FAILED] {bucket} ({view_type}) {stage}: {error}"
            )
        bucket_text = " and ".join(buckets)
        # 1b#3: "only the other view's model was trained" is a lie when BOTH
        # buckets failed.  Say what actually happened.
        survivors = sorted(
            b for b in (successful_buckets or set()) if b not in buckets
        )
        if survivors:
            outcome = (
                f"Only the {' and '.join(survivors)} model was trained this "
                f"round."
            )
        else:
            outcome = "No model was trained this round."
        self._append_ui_log(
            f"[TRAIN FAILED] No model was produced for {bucket_text}; "
            f"those slices stay staged so you can retry.  {outcome}"
        )

        if not self._can_show_modal():
            # Headless / no event loop — the log lines above are the record.
            return
        try:
            from qtpy.QtWidgets import QMessageBox
            box = QMessageBox(getattr(self, "ui", None))
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Training incomplete")
            box.setText(
                f"Training failed for the {bucket_text} view(s)."
            )
            box.setInformativeText(
                f"{outcome} The failed view's slices remain staged — see the "
                "training log for the error."
            )
            box.setStandardButtons(QMessageBox.Ok)
            box.exec_() if hasattr(box, "exec_") else box.exec()
        except Exception as _e:
            print(f"[STEP2-TRAIN] failed-views modal unavailable "
                  f"({type(_e).__name__}: {_e}); log line already written")

    def _run_benchmark(self):
        """Cross-round training-loss summary.  NOT WIRED to any button.

        journey #15 (2026-09-07): kept for a future Benchmarking panel, but
        deliberately unreachable from the UI — it compares each round's final
        TRAINING loss and writes
        ``artifacts/benchmarks/round_benchmark_summary.json``; it does not
        predict on held-out slices or measure mask quality, which is what the
        panel's copy promises.
        """
        if not hasattr(self, "round_backend_info"):
            self.round_backend_info = {}
        if not self.working_dir:
            print("Please set a working directory first.")
            return
        if not self.losses_by_round:
            print("No round losses available yet. Train at least one round first.")
            return

        def _final_loss(loss_entry):
            if isinstance(loss_entry, dict):
                finals = []
                for _view, series in loss_entry.items():
                    if isinstance(series, list) and series:
                        finals.append(float(series[-1]))
                return float(np.mean(finals)) if finals else float("inf")
            if isinstance(loss_entry, list) and loss_entry:
                return float(loss_entry[-1])
            return float("inf")

        def _per_view_losses(loss_entry):
            if isinstance(loss_entry, dict):
                out = {}
                for view, series in loss_entry.items():
                    out[view] = float(series[-1]) if isinstance(series, list) and series else float("inf")
                return out
            if isinstance(loss_entry, list):
                return {"single": float(loss_entry[-1]) if loss_entry else float("inf")}
            return {}

        summary_rows = []
        # Collect all round keys (both composite "lateral_0" and legacy int/str).
        all_keys = sorted(self.losses_by_round.keys(), key=str)
        for round_key in all_keys:
            loss_entry = self.losses_by_round.get(round_key, [])
            samples_entry = self.training_data_by_round.get(round_key, [])
            checkpoint_entry = self.model_checkpoint_paths.get(round_key, "")
            backend_entry = self.round_backend_info.get(round_key, {})
            # Derive round_idx for display.
            try:
                round_idx = int(str(round_key).rsplit("_", 1)[-1])
            except ValueError:
                round_idx = 0
            summary_rows.append(
                {
                    "round_idx": round_idx,
                    "round_id": str(round_key),
                    "num_samples": len(samples_entry) if isinstance(samples_entry, list) else 0,
                    "final_loss": _final_loss(loss_entry),
                    "per_view_final_loss": _per_view_losses(loss_entry),
                    "checkpoints": checkpoint_entry,
                    "training_backend": backend_entry.get("training", {}),
                    "inference_backend": backend_entry.get("inference", {}),
                    "failed_views": backend_entry.get("failed_views", []),
                }
            )

        if not summary_rows:
            print("No benchmarkable rounds found.")
            return

        best = min(summary_rows, key=lambda row: row["final_loss"])
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "best_round": best["round_id"],
            "best_final_loss": best["final_loss"],
            "rows": summary_rows,
        }

        out_dir = Path(self.working_dir) / "artifacts" / "benchmarks"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "round_benchmark_summary.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        self.ui.training_log_output.append(
            f"[BENCHMARK] Best round: {best['round_id']} (final_loss={best['final_loss']:.6f})"
        )
        self.ui.training_log_output.append(f"[BENCHMARK] Summary written: {out_path}")
        print(f"[INFO] Benchmark summary written to {out_path}")

    def _reload_persisted_pairs_from_disk(self):
        """After state reload, re-populate segmentation_manager from persisted pairs.

        Scans sample_records for entries with img_uri/mask_uri and loads
        them back into data_inputs/data_masks so training can use them.
        """
        if not hasattr(self, "sample_records") or not self.sample_records:
            return
        reloaded = 0
        for slice_label, record in self.sample_records.items():
            img_uri = record.get("img_uri", "")
            mask_uri = record.get("mask_uri", "")
            if not img_uri or not mask_uri:
                continue
            img_path = Path(img_uri)
            mask_path = Path(mask_uri)
            if not img_path.is_file() or not mask_path.is_file():
                print(f"[WARN] Persisted pair missing on disk for {slice_label}: {img_uri}")
                continue
            # Only reload if not already in memory.
            if slice_label in self.segmentation_manager.data_inputs:
                continue
            try:
                img_data = tifffile.imread(str(img_path))
                mask_data = tifffile.imread(str(mask_path))
                self.segmentation_manager.data_names[slice_label] = True
                self.segmentation_manager.data_inputs[slice_label] = np.asarray(img_data)
                self.segmentation_manager.data_masks[slice_label] = np.asarray(mask_data, dtype=np.int32)
                reloaded += 1
            except Exception as exc:
                print(f"[WARN] Failed to reload pair for {slice_label}: {exc}")
        if reloaded:
            self._append_ui_log(f"[STATE] Reloaded {reloaded} persisted pair(s) from disk.")

    # --- Training state reset ---

    # --- Path portability helpers ---

    def _to_relative_path(self, abs_path: str) -> str:
        """Convert absolute path to relative (to working_dir) if possible."""
        if not abs_path or not self.working_dir:
            return abs_path
        try:
            return str(Path(abs_path).relative_to(self.working_dir))
        except ValueError:
            return abs_path  # outside working_dir, keep absolute

    def _to_absolute_path(self, rel_path: str) -> str:
        """Convert relative path back to absolute using working_dir."""
        if not rel_path or not self.working_dir:
            return rel_path
        p = Path(rel_path)
        if p.is_absolute():
            return rel_path
        return str(Path(self.working_dir) / p)

    def _relativize_state(self, state: dict) -> dict:
        """Convert absolute paths in state dict to relative for portability."""
        state = dict(state)

        # sample_records: img_uri, mask_uri
        if "sample_records" in state:
            sr = {}
            for k, rec in state["sample_records"].items():
                rec = dict(rec)
                for key in ("img_uri", "mask_uri"):
                    if key in rec and rec[key]:
                        rec[key] = self._to_relative_path(rec[key])
                sr[k] = rec
            state["sample_records"] = sr

        # model_checkpoint_paths: {round: path_or_dict}
        if "model_checkpoint_paths" in state:
            mcp = {}
            for rnd, val in state["model_checkpoint_paths"].items():
                if isinstance(val, dict):
                    mcp[rnd] = {vt: self._to_relative_path(p) for vt, p in val.items()}
                elif isinstance(val, str):
                    mcp[rnd] = self._to_relative_path(val)
                else:
                    mcp[rnd] = val
            state["model_checkpoint_paths"] = mcp

        # training_data_by_round: {round: [records with img_uri, mask_uri]}
        if "training_data_by_round" in state:
            tdr = {}
            for rnd, records in state["training_data_by_round"].items():
                tdr[rnd] = []
                for rec in records:
                    rec = dict(rec)
                    for key in ("img_uri", "mask_uri"):
                        if key in rec and rec[key]:
                            rec[key] = self._to_relative_path(rec[key])
                    tdr[rnd].append(rec)
            state["training_data_by_round"] = tdr

        # active_curation_checkpoints: {lateral: path, axial: path}
        if "active_curation_checkpoints" in state:
            acc = {}
            for k, v in state["active_curation_checkpoints"].items():
                acc[k] = self._to_relative_path(v) if v else v
            state["active_curation_checkpoints"] = acc

        return state

    def _absolutize_state(self, state: dict) -> dict:
        """Convert relative paths in state dict back to absolute."""
        state = dict(state)

        if "sample_records" in state:
            sr = {}
            for k, rec in state["sample_records"].items():
                rec = dict(rec)
                for key in ("img_uri", "mask_uri"):
                    if key in rec and rec[key]:
                        rec[key] = self._to_absolute_path(rec[key])
                sr[k] = rec
            state["sample_records"] = sr

        if "model_checkpoint_paths" in state:
            mcp = {}
            for rnd, val in state["model_checkpoint_paths"].items():
                if isinstance(val, dict):
                    mcp[rnd] = {vt: self._to_absolute_path(p) for vt, p in val.items()}
                elif isinstance(val, str):
                    mcp[rnd] = self._to_absolute_path(val)
                else:
                    mcp[rnd] = val
            state["model_checkpoint_paths"] = mcp

        if "training_data_by_round" in state:
            tdr = {}
            for rnd, records in state["training_data_by_round"].items():
                tdr[rnd] = []
                for rec in records:
                    rec = dict(rec)
                    for key in ("img_uri", "mask_uri"):
                        if key in rec and rec[key]:
                            rec[key] = self._to_absolute_path(rec[key])
                    tdr[rnd].append(rec)
            state["training_data_by_round"] = tdr

        if "active_curation_checkpoints" in state:
            acc = {}
            for k, v in state["active_curation_checkpoints"].items():
                acc[k] = self._to_absolute_path(v) if v else v
            state["active_curation_checkpoints"] = acc

        return state

    # State related methods
    _state_save_timer: QTimer | None = None

    def _schedule_state_save(self, delay_ms: int = 500):
        """Debounced state save — coalesces rapid calls within *delay_ms*."""
        if self._state_save_timer is None:
            self._state_save_timer = QTimer()
            self._state_save_timer.setSingleShot(True)
            self._state_save_timer.timeout.connect(self._debounced_state_save)
        self._state_save_timer.start(delay_ms)

    def _debounced_state_save(self):
        try:
            self._save_internal_state()
            n_records = len(getattr(self, "sample_records", {}))
            # Log the actual destination filename so debug logs are
            # accurate (was misleadingly hardcoded "hitl_state.json"
            # even when actually writing to step2_state.json post-
            # migration).  User-reported 2026-05-23.
            try:
                from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
                _state_name = (
                    _step2_paths.STEP2_STATE_FILENAME
                    if _step2_paths.is_new_layout_state(self.working_dir)
                    else _step2_paths.LEGACY_STATE_FILENAME
                )
            except Exception:
                _state_name = "hitl_state.json"
            print(f"[STATE-SAVE] saved {_state_name} | {n_records} sample records | working_dir={self.working_dir}")
        except Exception as exc:
            print(f"[WARN] Debounced state save failed: {exc}")

    def _save_internal_state(self):
        self._ensure_runtime_defaults()
        if not self.working_dir:
            return
        # 2026-05-27 widget→config refresh — pull current widget values
        # into self.current_config BEFORE writing the state JSON.
        # Pre-fix: training widgets only flowed to cfg when user clicked
        # Train; inference widgets only flowed when user clicked Apply.
        # Side-effect: closing napari without those clicks discarded
        # every UI change.  Now the save path snapshots widgets every
        # time, so persisted state always reflects what the user sees.
        self._refresh_current_config_from_widgets()
        # Audit-fix 2026-05-24: guard against saving phantom state for a
        # run that doesn't actually exist on disk.  Happens when
        # workspace is fresh + `active_run_id` defaulted to "default"
        # via the peek fallback BUT no `runs/default/` dir exists.  In
        # that case the previous save behavior would create a phantom
        # `runs.default` entry in step2_state.json, polluting the
        # multi-run list.
        try:
            from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
            active = getattr(self, "active_run_id", _step2_paths.DEFAULT_RUN_ID)
            runs_root = _step2_paths.step2_runs_root(self.working_dir)
            if runs_root.is_dir():
                run_dir = runs_root / (active or _step2_paths.DEFAULT_RUN_ID)
                if not run_dir.is_dir():
                    print(f"[STATE-SAVE] skipping — active_run_id "
                          f"{active!r} has no run dir on disk (likely "
                          f"a fresh workspace fallback default)")
                    return
        except Exception:
            pass  # any failure → fall through + attempt the save normally
        self._ensure_curation_checkpoint_state()
        state = {
            "schema_version": 2,
            "current_round": self.current_round,
            "current_round_by_view": getattr(self, "current_round_by_view", {"lateral": 0, "axial": 0}),
            "training_data_by_round": self.training_data_by_round,
            "losses_by_round": self.losses_by_round,
            "model_checkpoint_paths": self.model_checkpoint_paths,
            "round_backend_info": self.round_backend_info,
            "current_config": self.current_config,
            "selected_view": self.selected_view,
            "view_target": self.view_target,
            "axial_plane": self.axial_plane,
            "sample_records": self.sample_records,
            "active_curation_checkpoints": self.active_curation_checkpoints,
            "curation_model_source_mode": self.curation_model_source_mode,
            "imported_curation_model_path": self.imported_curation_model_path,
            "curation_pretrained_model_type": self.curation_pretrained_model_type,
            "training_mode": getattr(self, "training_mode", "cumulative"),
            # ADDED 2026-09-07 (bucket-switch item #5).  The legacy
            # single-valued "training_mode" above keeps being written (it is
            # the ACTIVE bucket's value) so an older build still reads this
            # file.
            "training_mode_by_bucket": self._coerce_training_mode_by_bucket(
                getattr(self, "training_mode_by_bucket", None),
                fallback=getattr(self, "training_mode", "cumulative"),
            ),
            "auto_update_model_after_training": self._get_auto_update_model_checked(),
            "loaded_stack_path": str(getattr(self, "loaded_stack_path", "") or ""),
            "loaded_patch_name": str(getattr(self, "loaded_patch_name", "") or ""),
        }
        state = self._relativize_state(state)

        # Step 2 directory restructure (Phase 3): if workspace has been
        # migrated to v3, write to step2_state.json in v3-wrapped shape
        # alongside (or instead of) the legacy save.  Otherwise legacy
        # service writes hitl_state.json.
        try:
            from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
            from napari_epicycle.hitl_training_annotation._migrate_step2_layout import (
                wrap_state_schema_v2_to_v3,
            )
            if _step2_paths.is_new_layout_state(self.working_dir):
                # Wrap the flat v2-shaped state into v3 + write to step2_state.json
                #
                # CRITICAL (audit-found 2026-05-23): pass active_run_id so the
                # wrap writes to runs[<current_run>] not always runs["default"].
                # Without this kwarg, saves on a non-default run would CLOBBER
                # the default run's state and LOSE the current run's edits.
                import json as _json
                current_active_run = getattr(
                    self, "active_run_id", _step2_paths.DEFAULT_RUN_ID
                )
                v3 = wrap_state_schema_v2_to_v3(
                    {**state, "schema_version": 2},
                    migration_timestamp=None,  # will use now() — fine for save
                    active_run_id=current_active_run,
                )
                # Preserve original run-manifest fields (display_name, target,
                # backend, created_at, created_from) from the existing v3 file
                # rather than overwriting with defaults — user may have edited
                # display_name post-migration.
                v3_path = _step2_paths.state_json_path(self.working_dir)
                if v3_path.is_file():
                    try:
                        with open(v3_path, "r", encoding="utf-8") as _f:
                            existing = _json.load(_f)
                        # Merge ALL existing run entries forward — the wrap above
                        # only knows about runs[current_active_run].  Without
                        # this merge, saving while active on run X would drop
                        # every other run's entry from the file.  Audit fix
                        # 2026-05-23 (Phase 8a).
                        existing_runs = existing.get("runs") or {}
                        if isinstance(existing_runs, dict):
                            for _other_run_id, _other_run_state in existing_runs.items():
                                if _other_run_id != current_active_run and isinstance(_other_run_state, dict):
                                    v3["runs"][_other_run_id] = _other_run_state
                        existing_current = existing_runs.get(current_active_run, {})
                        if not isinstance(existing_current, dict):
                            existing_current = {}
                        for preserve_key in ("display_name", "target", "backend",
                                             "created_at", "created_from"):
                            if preserve_key in existing_current:
                                v3["runs"][current_active_run][preserve_key] = (
                                    existing_current[preserve_key]
                                )
                        # Release-hardening 2026-09-07 (bucket-switch #2):
                        # the picker / view keys moved from workspace-level
                        # to PER-RUN.  An older file still carries them at
                        # top level, and every run that has NOT been
                        # re-saved since still reads them from there (the
                        # unwrap merges top-level first, per-run wins).
                        # Carry those legacy top-level values forward so
                        # this save doesn't silently blank the other runs'
                        # external model file / view.
                        for _legacy_key in _LEGACY_WORKSPACE_FALLBACK_KEYS:
                            if _legacy_key in existing and _legacy_key not in v3:
                                v3[_legacy_key] = existing[_legacy_key]
                        # Preserve active_run_id from existing file ONLY if
                        # the controller hasn't explicitly changed it.  If
                        # the user just switched runs, current_active_run
                        # IS the new active and should win.
                        v3["active_run_id"] = current_active_run
                    except (OSError, _json.JSONDecodeError, ValueError) as _corrupt_err:
                        # Corrupt existing v3 file: WARN LOUDLY (audit fix
                        # 2026-05-23).  User edits to display_name / target
                        # will be lost.  Don't fail the save — partial state
                        # is worse than overwrite — but make sure the user
                        # sees this.
                        #
                        # Release-hardening #3 (2026-09-07): QUARANTINE the
                        # bad file as `<name>.corrupt-<ts>` first.  Pre-fix
                        # the fresh-defaults write destroyed the only copy
                        # of state that could still be hand-recovered (it
                        # carries every OTHER run's entry).
                        _quarantined = self._quarantine_corrupt_state_file(
                            v3_path, _corrupt_err
                        )
                        # 1b#5: say what is ACTUALLY lost.  The file we are
                        # about to write carries only the ACTIVE run — every
                        # OTHER run's entry (and any edited display_name /
                        # target) came from the file we could not read.
                        _msg = (
                            f"[STATE-SAVE] WARNING: existing step2_state.json "
                            f"is corrupt ({type(_corrupt_err).__name__}: "
                            f"{_corrupt_err}); writing fresh defaults for the "
                            f"active run {current_active_run!r} only.  Every "
                            f"OTHER run's entry is NOT in the new file."
                        )
                        if _quarantined is not None:
                            _msg += (f"  Recover them from the quarantined "
                                     f"copy at {_quarantined}.")
                        else:
                            _msg += ("  The corrupt file could NOT be "
                                     "quarantined — it has been overwritten.")
                        print(_msg)
                        try:
                            self._append_ui_log(_msg)
                        except Exception:
                            pass
                _atomic_write_json(v3_path, v3)

                # Also write per-bucket round_metadata.json + selected_indices.json
                # to the new layout (audit fix 2026-05-23).  Without this, the
                # v3 save bypasses the legacy persistence service which is the
                # only writer of these files — downstream readers would see
                # stale snapshots from the previous legacy save.
                try:
                    self._write_per_bucket_metadata_for_active_run(
                        active_run_id=v3.get("active_run_id", _step2_paths.DEFAULT_RUN_ID),
                    )
                except Exception as _meta_err:
                    print(f"[STATE-SAVE] WARNING: failed to write per-bucket "
                          f"round_metadata / selected_indices to new layout "
                          f"({type(_meta_err).__name__}: {_meta_err}); the "
                          f"state JSON itself was saved correctly but "
                          f"per-bucket metadata files may be stale.")

                # Don't also call the legacy persistence service — would
                # write a stale hitl_state.json alongside step2_state.json.
                return
        except Exception as _e:
            print(f"[STATE-SAVE] v3 save attempt failed ({type(_e).__name__}: {_e}), "
                  f"falling back to legacy persistence service")

        self.persistence.save_hitl_state(
            working_dir=self.working_dir,
            state_dict=state,
            current_round=self.current_round,
            current_config=self.current_config,
            training_data_by_round=self.training_data_by_round,
        )


    def _write_per_bucket_metadata_for_active_run(self, active_run_id: str) -> None:
        """Write round_metadata.json + selected_indices.json to per-bucket
        locations in the new layout.

        Called from _save_internal_state when workspace is on new layout,
        because the v3 save path bypasses the legacy persistence service
        (which is the only writer of these files in the legacy code path).

        Splits the joint per-round content by VIEW code:
          * XY records / indices → lateral bucket
          * XZ + YZ records / indices → axial bucket

        Idempotent: overwrites existing files.  Safe to call repeatedly.
        Added 2026-05-23 per Phase 3-4 audit gate finding.

        Phase B.1 fix (2026-05-25): pre-fix used a single
        `round_idx = int(self._current_round())` reading
        `self.selected_view`'s round counter.  In Mode=Both with
        divergent per-bucket counters (lateral=5, axial=0), this wrote
        BOTH buckets' metadata at the SELECTED bucket's round_idx —
        e.g. axial metadata landed at `axial/rounds/round_05/` instead
        of `axial/rounds/round_00/`.  Post-fix: iterate buckets,
        compute round_idx PER-BUCKET from `current_round_by_view[bucket]
        - 1` (the round that was just written, since counters were
        already incremented in `_on_training_finished`).
        """
        from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
        import json as _json

        # Per-bucket round_idx: counter - 1 since increment already
        # happened in _on_training_finished.  But for the initial
        # post-creation state-save before any training, counter is 0
        # and there's nothing to write — guard at the write site.
        by_view = getattr(self, "current_round_by_view", None) or {}
        lateral_round_idx = max(0, int(by_view.get("lateral", 0)) - 1)
        axial_round_idx = max(0, int(by_view.get("axial", 0)) - 1)
        # Phase B.1.1 fix (audit 3 MEDIUM): pre-write derivation
        # diagnostic.  Surfaces the source-of-truth values BEFORE
        # any write — if a future B.x change miswires counters,
        # this log line will show the bad derivation rather than
        # making us reverse-engineer from a misplaced JSON file.
        print(f"[STATE-SAVE] Phase B.1 derivation: "
              f"lateral_round_idx={lateral_round_idx} "
              f"(from counter={by_view.get('lateral', 0)}-1), "
              f"axial_round_idx={axial_round_idx} "
              f"(from counter={by_view.get('axial', 0)}-1), "
              f"active_run='{active_run_id}'")

        # Build lateral + axial record lists from training_data_by_round.
        # Each bucket reads its OWN composite key now.
        def _records_for(bucket: str, round_idx: int) -> list:
            out = []
            key = f"{bucket}_{round_idx}"
            rec_list = self.training_data_by_round.get(key)
            if isinstance(rec_list, list):
                out.extend(rec_list)
            return out

        lateral_records_from_state = _records_for("lateral", lateral_round_idx)
        axial_records_from_state = _records_for("axial", axial_round_idx)

        # Also include legacy bare-key fallback (round_idx without bucket
        # prefix) for pre-Phase-B.1 state JSONs that wrote under e.g.
        # "5" instead of "lateral_5".  Apply ONLY to whichever bucket
        # matches the legacy `_current_round()` value to avoid double-
        # counting.
        legacy_round_idx = int(self._current_round())
        legacy_keys_to_check = [legacy_round_idx, str(legacy_round_idx)]
        for legacy_key in legacy_keys_to_check:
            rec_list = self.training_data_by_round.get(legacy_key)
            if not isinstance(rec_list, list):
                continue
            # Route legacy entries to the bucket whose round_idx matches
            # (or to BOTH buckets when they're equal — pre-divergence).
            if legacy_round_idx == lateral_round_idx:
                lateral_records_from_state.extend(rec_list)
            if legacy_round_idx == axial_round_idx and lateral_round_idx != axial_round_idx:
                axial_records_from_state.extend(rec_list)

        # The combined list below preserves the previous behavior of
        # filter-by-view for the records-to-write step.  In the
        # Phase-B.1 happy path, each bucket's records already match
        # its bucket's views (XY for lateral, XZ/YZ for axial), so the
        # view filter is a sanity guard, not a corrector.
        records_for_round = lateral_records_from_state + axial_records_from_state

        # Split records by view → bucket
        lateral_records = [r for r in records_for_round
                           if isinstance(r, dict) and r.get("view") == "XY"]
        axial_records = [r for r in records_for_round
                         if isinstance(r, dict) and r.get("view") in ("XZ", "YZ")]

        # Selected indices: dict keyed by view → stack_id → [int]
        # Phase B.1 fix (2026-05-25): match sample_records against
        # PER-BUCKET round_idx (lateral_round_idx for XY entries,
        # axial_round_idx for XZ/YZ).  Pre-fix used single round_idx
        # from `_current_round()` which dropped one bucket's entries
        # in Mode=Both with divergent counters.
        lateral_sel: dict = {"XY": {}}
        axial_sel: dict = {"XZ": {}, "YZ": {}}
        for slice_label, rec in self.sample_records.items():
            if not isinstance(rec, dict):
                continue
            rec_round = rec.get("round_idx")
            if rec_round is None:
                continue  # be conservative — don't include cross-round
            try:
                rec_round_int = int(rec_round)
            except (TypeError, ValueError):
                continue
            view = str(rec.get("view", "")).upper()
            stack_id = str(rec.get("stack_id", "")).strip()
            try:
                slice_idx = int(rec.get("slice_index", -1))
            except (ValueError, TypeError):
                continue
            if not stack_id or slice_idx < 0 or view not in ("XY", "XZ", "YZ"):
                continue
            # Per-bucket round match
            if view == "XY":
                if rec_round_int != lateral_round_idx:
                    continue
                lateral_sel.setdefault(view, {}).setdefault(stack_id, []).append(slice_idx)
            else:
                if rec_round_int != axial_round_idx:
                    continue
                axial_sel.setdefault(view, {}).setdefault(stack_id, []).append(slice_idx)

        # Phase B.1: Write per-bucket files with PER-BUCKET round_idx.
        for bucket, records, sel, bucket_round_idx in (
            ("lateral", lateral_records, lateral_sel, lateral_round_idx),
            ("axial", axial_records, axial_sel, axial_round_idx),
        ):
            if not records and not any(sel.values()):
                continue

            meta_path = _step2_paths.round_metadata_path(
                self.working_dir, active_run_id, bucket, bucket_round_idx
            )
            sel_path = _step2_paths.round_selected_indices_path(
                self.working_dir, active_run_id, bucket, bucket_round_idx
            )

            # Release-hardening #3 (2026-09-07): atomic — a crash here
            # used to leave a truncated metadata file behind.
            if records:
                _atomic_write_json(meta_path, records)

            # Always write selected_indices if any entries — drop empty inner dicts.
            sel_filtered = {k: v for k, v in sel.items() if v}
            if sel_filtered:
                _atomic_write_json(sel_path, sel_filtered)

            print(f"[STATE-SAVE] wrote per-bucket round_metadata + "
                  f"selected_indices for {bucket}/round_{bucket_round_idx:02d} "
                  f"(run='{active_run_id}')")

    # ------------------------------------------------------------------
    # Phase 8a — Multi-run management (create / switch / list)
    # ------------------------------------------------------------------
    # Thin wrappers around the pure-function helpers in
    # napari_epicycle.hitl_training_annotation._run_management.  The UI
    # (Curation header dropdown + New Run dialog) calls these methods;
    # they handle controller-side save-before-switch + UI refresh.
    #
    # All emit [STEP2-RUN] console diagnostics for traceability.
    # ------------------------------------------------------------------

    def _layout_allows_new_run(self) -> bool:
        """Can a new run be created in this workspace right now?

        User-reported 2026-05-24: a FRESH workspace (no state JSON, no legacy
        `training_data/`) had New Run greyed out, so nobody could bootstrap
        the first run.  Enable on UNKNOWN (fresh) or NEW (already migrated);
        disable only on LEGACY / MIXED, which must migrate first (and are
        auto-prompted to on Browse).
        """
        if not (getattr(self, "working_dir", "") or "").strip():
            return False
        try:
            from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
            return _step2_paths.detect_layout_version(self.working_dir) in (
                _step2_paths.LAYOUT_VERSION_UNKNOWN,
                _step2_paths.LAYOUT_VERSION_NEW,
            )
        except Exception:      # noqa: BLE001
            return False

    def _refresh_run_picker(self):
        """Populate the Curation header run dropdown from disk + state.

        Idempotent.  Called after workspace load, after run creation,
        and after run switch.  Blocks signal emission during repopulate
        so the dropdown doesn't fire an unwanted switch event.

        UI surface: ``self.ui.run_picker_combo``.  Each entry's user
        data is the run slug; the displayed text is "display_name (lat:N
        ax:M)".  Archived runs grouped at the bottom with prefix.
        """
        ui = getattr(self, "ui", None)
        if ui is None:
            return
        combo = getattr(ui, "run_picker_combo", None)
        if combo is None:
            return

        # Block signals during repopulate — setCurrentIndex below would
        # otherwise fire ``activated`` and re-enter the switch handler.
        combo.blockSignals(True)
        try:
            combo.clear()
            summaries = self._list_runs_for_ui()
            self._run_summaries_cache = summaries   # read by the Run summary
            if not summaries:
                # adopt skips the first-run modal,
                # so the picker is where a fresh project says what is missing.
                # "(no workspace)" was a lie whenever a workspace WAS open and
                # simply had no runs in it yet -- exactly the state adopt
                # leaves a fresh project in.  The wording is
                # `_workspace_facts.MSG_STEP2_NO_RUN` verbatim, so the picker
                # and the banner cannot tell the user two different stories
                # (a test pins the two strings together).
                if (getattr(self, "working_dir", "") or "").strip():
                    combo.addItem(
                        "No run yet \u2014 click New Run to start training.",
                        "")
                else:
                    combo.addItem("(no workspace)", "")
                combo.setEnabled(False)   # nothing to pick from, yet
                # New Run is the ONE thing this line asks for, so it must be
                # pressable -- gated exactly as the populated branch below
                # gates it (a LEGACY / MIXED workspace still migrates first).
                ready_for_new_run = self._layout_allows_new_run()
                for btn_name in ("new_run_button", "import_pairs_button",
                                 "import_external_button", "manage_runs_button"):
                    btn = getattr(ui, btn_name, None)
                    if btn is not None:
                        btn.setEnabled(ready_for_new_run)
                return

            # Dropdown + New Run / Import / Manage enable logic.
            # User-reported 2026-05-24: a FRESH workspace (no state JSON,
            # no legacy training_data) had the New Run button grayed out
            # — so users couldn't bootstrap the first run.
            #
            # Fix: enable on UNKNOWN (fresh, no markers) or NEW (already
            # migrated).  Disable only on LEGACY/MIXED — those require
            # migration first (which is auto-prompted on workspace load).
            ready_for_new_run = self._layout_allows_new_run()
            combo.setEnabled(ready_for_new_run)
            new_btn = getattr(ui, "new_run_button", None)
            if new_btn is not None:
                new_btn.setEnabled(ready_for_new_run)
            # Same gate for Import Pairs / Import External / Manage
            for btn_name in ("import_pairs_button", "import_external_button",
                             "manage_runs_button"):
                btn = getattr(ui, btn_name, None)
                if btn is not None:
                    btn.setEnabled(ready_for_new_run)

            # Add active + non-archived runs first (sorted: active first)
            active_idx = -1
            non_archived = [s for s in summaries if not s.get("archived")]
            non_archived.sort(key=lambda s: (not s.get("is_active"), s["run_id"]))
            for s in non_archived:
                label = self._format_run_picker_label(s)
                combo.addItem(label, s["run_id"])
                if s.get("is_active"):
                    active_idx = combo.count() - 1

            # Then archived runs (separator + prefix)
            archived = [s for s in summaries if s.get("archived")]
            if archived:
                combo.insertSeparator(combo.count())
                for s in sorted(archived, key=lambda x: x["run_id"]):
                    label = "[archived] " + self._format_run_picker_label(s)
                    combo.addItem(label, s["run_id"])

            if active_idx >= 0:
                combo.setCurrentIndex(active_idx)
        finally:
            combo.blockSignals(False)
            # audit M2: the Run summary's pairs / description read
            # ``_run_summaries_cache``, which this method just rewrote, and
            # every caller refreshes the badges BEFORE the picker -- so
            # re-send the payload (a pure re-notify; the call guards itself)
            try:
                self._refresh_patch_list_badges()
            except Exception:      # noqa: BLE001 -- cosmetic
                pass

    @staticmethod
    def _format_run_picker_label(summary):
        """Format one summary as a dropdown label.

        e.g. "Default Run (lat 2 · ax 1 · 18 pairs)"  or
             "Nuclei v1 (nuclei · lat 0 · ax — · 5 pairs)"

        2026-05-25 Group D: accepted-pair count appended so the user
        can see at-a-glance how much curated data each run has.
        Audit 4 finding C1 — without this, the dropdown gave no
        progress feedback as the user accepted samples.
        """
        display_name = summary.get("display_name") or summary.get("run_id", "?")
        # Step 2 look polish: the words live in ``_step2_text`` (string only;
        # the numbers are the summary's own)
        from ..ui import _step2_text as _T2
        return _T2.run_picker_label(
            display_name, summary.get("target", "") or "",
            summary.get("lateral_round_count", 0),
            summary.get("axial_round_count", 0),
            summary.get("accepted_pair_count", 0))

    def _on_run_picker_activated(self, index):
        """User picked an entry in the Curation header dropdown."""
        ui = getattr(self, "ui", None)
        if ui is None:
            return
        combo = getattr(ui, "run_picker_combo", None)
        if combo is None:
            return
        slug = combo.itemData(index)
        if not slug:
            return  # placeholder entry
        try:
            self._switch_active_run(slug)
        except Exception as e:
            # Restore dropdown to actual active run
            self._refresh_run_picker()

    def _on_new_run_button_clicked(self):
        """User clicked [+ New Run] in the Curation header.

        Opens the New Run dialog (Phase 8a-4).  On accept, creates the
        run + activates it + refreshes the dropdown.
        """
        if not self.working_dir:
            return
        try:
            from napari_epicycle.hitl_training_annotation._new_run_dialog import (
                NewRunDialog,
            )
        except ImportError as e:
            return

        # Gather existing run IDs for collision check inside the dialog
        existing_slugs = {s["run_id"] for s in self._list_runs_for_ui()}
        # Optional channel snapshot from napari viewer (if available)
        channels_snapshot = self._snapshot_viewer_channels()
        # Phase 8b — scan for init sources (run models + Step 3 refined)
        try:
            from napari_epicycle.hitl_training_annotation._run_management import (
                list_run_model_sources, list_step3_refined_sources,
            )
            run_model_sources = list_run_model_sources(self.working_dir)
            step3_refined_sources = list_step3_refined_sources(self.working_dir)
        except Exception as e:
            run_model_sources = []
            step3_refined_sources = []

        dialog = NewRunDialog(
            existing_slugs=existing_slugs,
            channels_snapshot=channels_snapshot,
            run_model_sources=run_model_sources,
            step3_refined_sources=step3_refined_sources,
            parent=getattr(self, "ui", None),
        )
        if not dialog.exec_():
            return

        result = dialog.get_result()
        if result is None:
            return
        try:
            self._create_new_run(
                slug=result["slug"],
                display_name=result["display_name"],
                target=result.get("target", ""),
                description=result.get("description", ""),
                backend=result.get("backend", "cellpose"),
                init_spec=result.get("init_spec"),
                channels_snapshot=result.get("channels_snapshot"),
                notes=result.get("notes", ""),
                activate=result.get("activate", True),
            )
            self._refresh_run_picker()
        except Exception as e:
            pass

    def _on_import_external_clicked(self):
        """Phase 8f — open External Pair Import dialog + execute on accept."""
        if not self.working_dir:
            return
        from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
        if not _step2_paths.is_new_layout_state(self.working_dir):
            return

        # Same training-block guard as Phase 8d Import Pairs
        train_thread = getattr(self, "_training_thread", None)
        if train_thread is not None:
            try:
                if hasattr(train_thread, "isRunning") and train_thread.isRunning():
                    try:
                        from napari_epicycle.hitl_training_annotation._switch_run_dialog import (
                            show_switch_blocked,
                        )
                        _active_id = getattr(self, "active_run_id", "default")
                        _active_dn = self._get_run_display_name(_active_id)
                        show_switch_blocked(
                            parent=getattr(self, "ui", None),
                            reasons=[
                                "Training is currently running.  Importing "
                                "external pairs while training would race "
                                "with the trainer's reads of the round "
                                "directory.  Wait for training to complete "
                                "and try again."
                            ],
                            old_run_id=_active_id,
                            new_run_id=_active_id,
                            old_display_name=_active_dn,
                            new_display_name=_active_dn,
                        )
                    except Exception:
                        pass
                    return
            except Exception:
                pass

        try:
            from napari_epicycle.hitl_training_annotation._import_external_pairs_dialog import (
                ImportExternalPairsDialog,
            )
            from napari_epicycle.hitl_training_annotation._import_external_pairs import (
                copy_external_pairs_to_target_round,
            )
        except ImportError as e:
            return

        active = getattr(self, "active_run_id", _step2_paths.DEFAULT_RUN_ID)
        # 2026-05-25 terminology sweep: thread display_name through to
        # the dialog's subtitle.
        active_display_name = self._get_run_display_name(active)

        def _suggest_target_round(bucket: str) -> int:
            try:
                by_view = getattr(self, "current_round_by_view", {}) or {}
                return int(by_view.get(bucket, 0))
            except Exception:
                return 0

        dialog = ImportExternalPairsDialog(
            active_run_id=active,
            suggest_target_round_callback=_suggest_target_round,
            active_display_name=active_display_name,
            parent=getattr(self, "ui", None),
        )
        if not dialog.exec_():
            return
        result = dialog.get_result()
        if result is None:
            return

        # Re-read active_run_id (race protection — same as Phase 8d)
        current_active = getattr(self, "active_run_id", _step2_paths.DEFAULT_RUN_ID)
        if current_active != active:
            return


        self._flush_pending_state_timers()

        self._begin_busy("[IMPORT] Copying external pairs ...")
        try:
            outcome = copy_external_pairs_to_target_round(
                self.working_dir,
                target_run_id=active,
                target_bucket=result["target_bucket"],
                target_round_idx=result["target_round_idx"],
                pair_records=result["pair_records"],
                relabel_basename=result["relabel_basename"],
                overwrite_collisions=result["overwrite_collisions"],
                source_channel_axis=result.get("source_channel_axis", "first"),
            )
        except Exception as e:
            self._report_import_failure("External import", e)
            return
        finally:
            self._end_busy()

        _post_failures = []
        if outcome["errors"]:
            for label, msg in outcome["errors"][:5]:
                print(f"[WARN] external import per-pair error: "
                      f"{label}: {msg}")
            # 2026-05-25 Group E: themed summary dialog so user sees the
            # errors instead of only the console.  Previous behavior:
            # silent skip if user didn't have a console open.
            try:
                self._show_external_import_error_summary(
                    n_copied=len(outcome["copied"]),
                    n_skipped=len(outcome["skipped"]),
                    errors=outcome["errors"],
                )
            except Exception as e:
                pass

        # Build sample_records for newly-imported pairs (mirror of Phase
        # 8d _add_sample_records_for_imported_pairs but for external).
        if outcome["copied"]:
            self._add_sample_records_for_external_pairs(
                provenance_records=outcome["provenance"],
                target_bucket=result["target_bucket"],
                target_round_idx=result["target_round_idx"],
            )
            self._run_post_import_step(
                "saving your workspace", self._save_internal_state,
                _post_failures,
            )

            # 2026-05-25 fix: mirror of the Phase 8d cherry-pick path.
            # Without this, external imports never reach
            # segmentation_manager so they can't render / finalize / train.
            def _reload_pairs():
                self._reload_persisted_pairs_from_disk()
                if self.loaded_patch_name:
                    # Lifecycle-audit fix (2026-09-07, patch-bar brick): this
                    # used to ensure all THREE views.  Under lazy creation
                    # that is a CREATION, and napari adds a layer with
                    # `visible=True`, so an import while in lateral mode left
                    # stray visible XZ/YZ label layers over the image.  The
                    # visibility choke point builds exactly the view(s) this
                    # context shows and hides the rest -- same call the three
                    # load paths make.
                    self._set_work_labels_visibility_for_context(
                        patch_name=self.loaded_patch_name,
                        target_view=self.view_target,
                    )

            self._run_post_import_step(
                "loading the copied pairs", _reload_pairs, _post_failures,
            )

        def _refresh_ui():
            self._refresh_run_picker()
            self._refresh_round_status()
            self._refresh_sample_queue()
            self._refresh_patch_list_badges()
            # Edit-states: imported pairs land as accepted/finalized — if the
            # import targets the open patch's view, refresh the annotator
            # soft-lock tint so they show emerald/ice immediately.
            self._refresh_annotator_edit_state_visuals()

        self._run_post_import_step(
            "refreshing the panel", _refresh_ui, _post_failures,
        )
        self._report_post_import_failures("External import", _post_failures)

    def _show_external_import_error_summary(self, *, n_copied, n_skipped, errors):
        """2026-05-25 Group E: themed dialog summarizing per-pair copy errors.

        Replaces silent console-only logging.  Always shows total
        copied + skipped + errored counts; lists first N errored
        labels with their reason.

        Args:
            n_copied: int, number of pairs successfully copied.
            n_skipped: int, number of pairs skipped (collisions, etc.).
            errors: list of (label, reason) tuples from the backend.
        """
        from qtpy.QtCore import Qt as _Qt
        from qtpy.QtWidgets import (
            QDialog as _QDlg, QVBoxLayout as _QVB, QHBoxLayout as _QHB,
            QLabel as _QL, QPushButton as _QB, QTextEdit as _QTE,
        )
        from napari_epicycle.ui_shell.theme import (
            _DIALOG_STYLE, _DIALOG_BTN_CANCEL,
        )

        parent = getattr(self, "ui", None)
        dlg = _QDlg(parent)
        dlg.setWindowTitle("External Import — partial completion")
        dlg.setStyleSheet(_DIALOG_STYLE)
        dlg.setMinimumWidth(560)

        layout = _QVB(dlg)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        title = _QL("Some pairs were not imported", parent=dlg)
        title.setStyleSheet(
            "font-weight: bold; font-size: 14px; color: #d4a843;"
        )
        layout.addWidget(title)

        summary = _QL(
            f"<b>{n_copied}</b> pair(s) copied, "
            f"<b>{n_skipped}</b> skipped, "
            f"<b>{len(errors)}</b> errored.",
            parent=dlg,
        )
        summary.setTextFormat(_Qt.RichText)
        summary.setStyleSheet("color: #dddddd; font-size: 11px;")
        layout.addWidget(summary)

        if errors:
            err_label = _QL("Errors (first 20):", parent=dlg)
            err_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
            layout.addWidget(err_label)

            err_view = _QTE(parent=dlg)
            err_view.setReadOnly(True)
            err_view.setMaximumHeight(200)
            err_view.setStyleSheet(
                "QTextEdit { background: #1e1e1e; color: #cccccc; "
                "font-family: 'Courier New', monospace; font-size: 10px; "
                "border: 1px solid #333; }"
            )
            err_text_lines = []
            for label, reason in errors[:20]:
                err_text_lines.append(f"{label}\n    → {reason}")
            if len(errors) > 20:
                err_text_lines.append(f"\n... and {len(errors) - 20} more")
            err_view.setPlainText("\n\n".join(err_text_lines))
            layout.addWidget(err_view)

        info = _QL(
            "Pairs that copied successfully are already in the run.  "
            "You can fix the source files and re-import the failing "
            "pairs without affecting the ones that already landed.",
            parent=dlg,
        )
        info.setWordWrap(True)
        info.setStyleSheet(
            "color: #888888; font-size: 10px; font-style: italic; "
            "padding-top: 6px;"
        )
        layout.addWidget(info)

        btn_row = _QHB()
        btn_row.addStretch(1)
        ok_btn = _QB("OK", parent=dlg)
        ok_btn.setStyleSheet(_DIALOG_BTN_CANCEL)
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

        dlg.exec_()

    def _add_sample_records_for_external_pairs(
        self, *, provenance_records, target_bucket, target_round_idx,
    ):
        """Add sample_records entries for external-imported pairs.

        Mirror of Phase 8d's _add_sample_records_for_imported_pairs but
        for external sources — writes ``inherited_from_external`` field
        instead of ``inherited_from_pair``.
        """
        from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
        from datetime import datetime, timezone
        target_run_id = getattr(self, "active_run_id", _step2_paths.DEFAULT_RUN_ID)
        target_round_id = f"round_{target_round_idx:02d}"
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            target_dir = _step2_paths.round_accepted_dir(
                self.working_dir, target_run_id, target_bucket, target_round_idx,
            )
        except (ValueError, TypeError) as e:
            return

        added = 0
        skipped = 0
        for prov in provenance_records:
            label = prov["target_label"]
            try:
                stack_id, view, slice_index = _step2_paths.parse_slice_label(label)
            except ValueError as e:
                skipped += 1
                continue
            existing = self.sample_records.get(label, {})
            existing_status = existing.get("status", "")
            record = dict(existing)
            record.setdefault("sample_id", label)
            record["bucket"] = target_bucket
            record["view"] = view
            record["slice_index"] = slice_index
            record["patch_id"] = stack_id
            record["roi_id"] = stack_id
            # 2026-05-25 audit-B HIGH fix: preserve finalized status on
            # re-import overwrite.  Mirror of cherry-pick path — see
            # _add_sample_records_for_imported_pairs for rationale.
            if existing_status in ("staged", "committed"):
                pass
            else:
                record["status"] = "accepted"
            record["round_tag"] = target_round_id
            record["img_uri"] = str(target_dir / f"{label}_img.tif")
            record["mask_uri"] = str(target_dir / f"{label}_mask.tif")
            record["pair_saved_at"] = now_iso
            record["updated_at"] = now_iso
            record["inherited_from_external"] = {
                "source_path": str(prov.get("source_img_path", "")),
                "source_filename": str(prov.get("source_basename", "") + "_img"),
                "source_basename": prov["source_basename"],
            }
            record["imported_at"] = now_iso
            self.sample_records[label] = record
            added += 1

    def _on_manage_runs_clicked(self):
        """Phase 8c — open Manage Runs panel."""
        if not self.working_dir:
            return
        from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
        if not _step2_paths.is_new_layout_state(self.working_dir):
            return

        try:
            from napari_epicycle.hitl_training_annotation._manage_runs_panel import (
                ManageRunsPanel,
            )
            from napari_epicycle.hitl_training_annotation._run_management import (
                archive_run, edit_run_metadata, find_runs_inheriting_from,
                delete_run_from_disk, remove_run_from_state_json, unarchive_run,
            )
        except ImportError as e:
            return

        active = getattr(self, "active_run_id", _step2_paths.DEFAULT_RUN_ID)
        workspace = self.working_dir
        run_summaries = self._list_runs_for_ui()

        # Action callbacks — each returns bool (True = action succeeded so
        # panel should reload its data).
        def _do_switch(slug: str) -> None:
            # Switch closes the panel (caller's dialog.accept() runs).
            self._switch_active_run(slug)

        def _do_edit(slug: str, updates: dict) -> bool:
            try:
                edit_run_metadata(workspace, slug, **updates)
                # Refresh picker to reflect new display_name
                self._refresh_run_picker()
                return True
            except Exception as e:
                raise

        def _do_archive(slug: str) -> bool:
            try:
                archive_run(workspace, slug)
                self._refresh_run_picker()
                return True
            except Exception as e:
                raise

        def _do_unarchive(slug: str) -> bool:
            try:
                unarchive_run(workspace, slug)
                self._refresh_run_picker()
                return True
            except Exception as e:
                raise

        def _do_delete(slug: str) -> bool:
            # Flush state timers before destructive op + double-check
            # the active run can't be deleted (panel blocks but be
            # defensive — race-condition guard if user-edits state JSON).
            current_active = getattr(self, "active_run_id", _step2_paths.DEFAULT_RUN_ID)
            if slug == current_active:
                raise RuntimeError(
                    f"refusing to delete currently-active run '{slug}'; "
                    f"switch to a different run first"
                )
            # Defense-in-depth: re-check provenance dependencies inside
            # _do_delete (panel already checks but TOCTOU could
            # technically race; this re-check costs ~1ms — audit-fix
            # 2026-05-24).
            dependents = find_runs_inheriting_from(workspace, slug)
            if dependents:
                raise RuntimeError(
                    f"refusing to delete '{slug}': {len(dependents)} other "
                    f"run(s) inherit from it ({', '.join(dependents[:5])})"
                )
            self._flush_pending_state_timers()
            # Order: state JSON FIRST, then rmtree.  Audit-found 2026-05-24:
            # original order (rmtree → JSON) could leave an orphan JSON
            # entry pointing at a missing dir if the JSON write failed
            # after rmtree succeeded.  This order means a failure mid-delete
            # leaves the dir on disk (recoverable — user can retry; or it
            # just sits as an orphan dir that list_runs_with_manifests
            # would surface).
            try:
                remove_run_from_state_json(workspace, slug)
            except Exception as e:
                raise
            try:
                delete_run_from_disk(workspace, slug, confirm=True)
            except Exception as e:
                # State JSON entry already gone but dir still on disk —
                # surface the half-state to the user explicitly.
                # Re-raise with richer context so the panel's error dialog
                # tells the user what happened
                raise RuntimeError(
                    f"State JSON entry was removed but the run directory "
                    f"could not be deleted (likely a file-lock from another "
                    f"process).  The directory remains on disk as an orphan "
                    f"and will reappear in the dropdown after next refresh.  "
                    f"Close any program using files in '{slug}' + retry.  "
                    f"Original error: {e}"
                ) from e
            self._refresh_run_picker()
            return True

        def _find_dependents(slug: str) -> list:
            return find_runs_inheriting_from(workspace, slug)

        panel = ManageRunsPanel(
            run_summaries=run_summaries,
            active_run_id=active,
            on_switch=_do_switch,
            on_edit=_do_edit,
            on_archive=_do_archive,
            on_unarchive=_do_unarchive,
            on_delete=_do_delete,
            find_dependents_callback=_find_dependents,
            parent=getattr(self, "ui", None),
        )
        # Wire a fresh-data provider so the panel can reload after each action
        panel.set_summaries_provider(self._list_runs_for_ui)
        panel.exec_()
        # Final refresh after panel closes (catches edits made via the
        # panel that weren't already picker-refreshed)
        try:
            self._refresh_run_picker()
        except Exception:
            pass

    def _flush_pending_state_timers(self):
        """Flush + stop the debounced state-save and curating-persist
        timers.  Called before any operation that mutates active_run_id
        or working_dir so a queued timer can't fire AFTER the context
        change and write OLD state into the NEW context's slot.

        Audit-found 2026-05-23 (state-machine completeness): timer
        firing between active_run_id assignment and _load_internal_state
        could write empty-state into NEW run's slot or fire saves against
        the wrong workspace.  Same root cause as Phase 5 audit findings.
        """
        try:
            timer = getattr(self, "_state_save_timer", None)
            if timer is not None and hasattr(timer, "isActive") and timer.isActive():
                timer.stop()
        except Exception as e:
            pass
        try:
            curating_timer = getattr(self, "_curating_persist_timer", None)
            if (curating_timer is not None and hasattr(curating_timer, "isActive")
                    and curating_timer.isActive()):
                curating_timer.stop()
            # Drain any pending curating edits (they belong to the OLD run)
            flush_method = getattr(self, "_flush_curating_persist_pending", None)
            if flush_method is not None:
                try:
                    flush_method()
                except Exception as e:
                    pass
        except Exception as e:
            pass
        # Stage 2 barrier: the curating-persist drain above ENQUEUES async TIFF
        # writes; block until they land on disk before this hard barrier
        # returns.  This method is the chokepoint for every path that reads the
        # pairs back from disk (train / run-switch / browse / import / delete-
        # run), so flushing here covers them all.  The per-stroke debounce path
        # (_flush_curating_persist_pending called directly off the timer) does
        # NOT flush — that's what keeps steady-state drawing async.
        writer = getattr(self, "_pair_writer", None)
        if writer is not None:
            try:
                writer.flush()
            except Exception:
                pass

    def _on_import_pairs_clicked(self):
        """Phase 8d — open Import Pairs dialog + execute import on accept."""
        if not self.working_dir:
            return
        from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
        if not _step2_paths.is_new_layout_state(self.working_dir):
            return

        # Audit-found SHIP-BLOCKING 2026-05-23: block import if training is
        # in flight.  Trainer's QThread reads round_metadata.json + the
        # pairs dir; concurrent writes from import_pairs would race +
        # potentially feed half-copied TIFFs into training.
        train_thread = getattr(self, "_training_thread", None)
        if train_thread is not None:
            try:
                if hasattr(train_thread, "isRunning") and train_thread.isRunning():
                    try:
                        from napari_epicycle.hitl_training_annotation._switch_run_dialog import (
                            show_switch_blocked,
                        )
                        _active_id = getattr(self, "active_run_id", "default")
                        _active_dn = self._get_run_display_name(_active_id)
                        show_switch_blocked(
                            parent=getattr(self, "ui", None),
                            reasons=[
                                "Training is currently running.  Importing pairs "
                                "while training would race with the trainer's "
                                "reads of the round directory, potentially "
                                "feeding half-copied files into training.  "
                                "Wait for training to complete and try again."
                            ],
                            old_run_id=_active_id,
                            new_run_id=_active_id,
                            old_display_name=_active_dn,
                            new_display_name=_active_dn,
                        )
                    except Exception:
                        pass
                    return
            except Exception:
                pass

        try:
            from napari_epicycle.hitl_training_annotation._import_pairs_dialog import (
                ImportPairsDialog,
            )
            from napari_epicycle.hitl_training_annotation._import_pairs import (
                copy_pairs_to_target_round,
                list_source_pairs,
                list_source_run_rounds,
            )
        except ImportError as e:
            return

        run_summaries = self._list_runs_for_ui()
        active = getattr(self, "active_run_id", _step2_paths.DEFAULT_RUN_ID)
        # 2026-05-25 terminology sweep: pass display_name for the
        # subtitle + target header in the dialog.
        active_display_name = self._get_run_display_name(active)

        def _suggest_target_round(bucket: str) -> int:
            """Suggest next round = current active round for that bucket."""
            try:
                by_view = getattr(self, "current_round_by_view", {}) or {}
                return int(by_view.get(bucket, 0))
            except Exception:
                return 0

        dialog = ImportPairsDialog(
            active_run_id=active,
            run_summaries=run_summaries,
            list_run_rounds_callback=lambda rid: list_source_run_rounds(
                self.working_dir, rid
            ),
            list_source_pairs_callback=lambda rid, bk, idx: list_source_pairs(
                self.working_dir, rid, bk, idx
            ),
            suggest_target_round_callback=_suggest_target_round,
            active_display_name=active_display_name,
            parent=getattr(self, "ui", None),
        )
        if not dialog.exec_():
            return
        result = dialog.get_result()
        if result is None:
            return

        # Audit-found RACE 2026-05-23 (state-machine completeness):
        # re-read active_run_id AFTER dialog exec.  If user switched runs
        # via the dropdown while the dialog was open, the captured `active`
        # is stale and pairs would land in the wrong run.
        current_active = getattr(self, "active_run_id", _step2_paths.DEFAULT_RUN_ID)
        if current_active != active:
            return


        # Flush pending state-save before mutating the round's accepted dir,
        # so a pending save doesn't fire mid-copy and observe partial state.
        self._flush_pending_state_timers()

        # Brick D: the copy is synchronous file I/O over N pairs — show
        # the user that something is happening.
        self._begin_busy("[IMPORT] Copying pairs ...")
        try:
            outcome = copy_pairs_to_target_round(
                self.working_dir,
                source_run_id=result["source_run_id"],
                source_bucket=result["source_bucket"],
                source_round_idx=result["source_round_idx"],
                target_run_id=active,
                target_bucket=result["target_bucket"],
                target_round_idx=result["target_round_idx"],
                selected_labels=result["selected_labels"],
                overwrite_collisions=result["overwrite_collisions"],
            )
        except Exception as e:
            self._report_import_failure("Cherry-pick", e)
            return
        finally:
            self._end_busy()

        _post_failures = []
        if outcome["errors"]:
            for label, msg in outcome["errors"][:5]:
                print(f"[WARN] cherry-pick per-pair error: "
                      f"{label}: {msg}")

        # Build sample_records entries for newly-copied pairs (Phase 8e
        # 2026-05-24).  Phase 8d shipped without this — imported pairs
        # ended up on disk + in round_metadata.json but NOT in the
        # controller's in-memory sample_records, so they didn't show in
        # the Slices list at all.  This loop creates the records (with
        # inherited_from_pair provenance for the badge), then state save
        # persists them.
        if outcome["copied"]:
            self._add_sample_records_for_imported_pairs(
                copied_labels=outcome["copied"],
                source_run_id=result["source_run_id"],
                source_bucket=result["source_bucket"],
                source_round_idx=result["source_round_idx"],
                target_bucket=result["target_bucket"],
                target_round_idx=result["target_round_idx"],
            )
            # Persist so reload finds them.
            self._run_post_import_step(
                "saving your workspace", self._save_internal_state,
                _post_failures,
            )

            # 2026-05-25 fix: pull imported masks into segmentation_manager
            # + stamp into work_labels_3d for current patch.  Without this,
            # imported pairs sit on disk but never reach in-memory caches
            # so (a) clicking a slice in the box list renders empty
            # work_labels, (b) Finalize Accepted skips them via the
            # get_masks() None check at line ~4457, (c) training silently
            # excludes them.  See [IMPORT-RELOAD] / [PATCH-SWITCH-MASK]
            # diagnostics + 6-audit consolidation 2026-05-25.
            def _reload_pairs():
                self._reload_persisted_pairs_from_disk()
                if self.loaded_patch_name:
                    # Lifecycle-audit fix (2026-09-07, patch-bar brick): this
                    # used to ensure all THREE views.  Under lazy creation
                    # that is a CREATION, and napari adds a layer with
                    # `visible=True`, so an import while in lateral mode left
                    # stray visible XZ/YZ label layers over the image.  The
                    # visibility choke point builds exactly the view(s) this
                    # context shows and hides the rest -- same call the three
                    # load paths make.
                    self._set_work_labels_visibility_for_context(
                        patch_name=self.loaded_patch_name,
                        target_view=self.view_target,
                    )

            self._run_post_import_step(
                "loading the copied pairs", _reload_pairs, _post_failures,
            )

        # Refresh UI to reflect new pairs in the active run
        def _refresh_ui():
            self._refresh_run_picker()
            self._refresh_round_status()
            self._refresh_sample_queue()
            self._refresh_patch_list_badges()
            # Edit-states: cherry-picked pairs land as accepted/finalized —
            # refresh the annotator soft-lock tint so they show emerald/ice
            # immediately if the import targets the open patch's view.
            self._refresh_annotator_edit_state_visuals()

        self._run_post_import_step(
            "refreshing the panel", _refresh_ui, _post_failures,
        )
        self._report_post_import_failures("Cherry-pick", _post_failures)

    def _add_sample_records_for_imported_pairs(
        self, *, copied_labels, source_run_id, source_bucket,
        source_round_idx, target_bucket, target_round_idx,
    ):
        """Add sample_records entries for cherry-pick-imported pairs.

        Without this, imported pairs are invisible in the Slices list
        (filter iterates self.sample_records).  Each new record carries
        the provenance dict that drives the Phase 8e ↳ badge in
        _format_queue_row + tooltip in _add_colored_item.

        Args:
            copied_labels: list of pair label strings (from
                copy_pairs_to_target_round outcome["copied"]).
            source_*, target_*: import-flow context for provenance.
        """
        from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
        from datetime import datetime, timezone
        target_run_id = getattr(self, "active_run_id", _step2_paths.DEFAULT_RUN_ID)
        source_round_id = f"round_{source_round_idx:02d}"
        target_round_id = f"round_{target_round_idx:02d}"
        now_iso = datetime.now(timezone.utc).isoformat()

        # Pre-compute target accepted dir for img_uri / mask_uri
        try:
            target_dir = _step2_paths.round_accepted_dir(
                self.working_dir, target_run_id, target_bucket, target_round_idx,
            )
        except (ValueError, TypeError) as e:
            return

        added = 0
        skipped = 0
        for label in copied_labels:
            try:
                stack_id, view, slice_index = _step2_paths.parse_slice_label(label)
            except ValueError as e:
                skipped += 1
                continue

            # Don't clobber an existing record (e.g., user has already
            # native-curated this slice).  Merge provenance into existing.
            existing = self.sample_records.get(label, {})
            existing_status = existing.get("status", "")
            record = dict(existing)
            record.setdefault("sample_id", label)
            record["bucket"] = target_bucket
            record["view"] = view
            record["slice_index"] = slice_index
            record["patch_id"] = stack_id
            record["roi_id"] = stack_id
            # 2026-05-25 audit-B HIGH fix: preserve finalized status on
            # re-import (overwrite case).  Previously this line was
            # `record["status"] = "accepted"` unconditionally — which
            # silently DEMOTED staged/committed pairs back to accepted
            # when the user re-imported the same label.  Now: only set
            # to "accepted" if the existing status was empty or below
            # accepted in the workflow.  Finalized work is preserved.
            if existing_status in ("staged", "committed"):
                pass
                # Keep the finalization — disk + provenance update but
                # status stays.
            else:
                record["status"] = "accepted"
            record["round_tag"] = target_round_id
            record["img_uri"] = str(target_dir / f"{label}_img.tif")
            record["mask_uri"] = str(target_dir / f"{label}_mask.tif")
            record["pair_saved_at"] = now_iso
            record["updated_at"] = now_iso
            record["inherited_from_pair"] = {
                "source_run_id": source_run_id,
                "source_bucket": source_bucket,
                "source_round_id": source_round_id,
                "source_label": label,
            }
            record["imported_at"] = now_iso
            self.sample_records[label] = record
            added += 1

    def _snapshot_viewer_channels(self):
        """Read channel-layer names from the napari viewer, if available.

        Returns:
            List of channel-name strings.  Empty list if viewer missing
            or no image layers.
        """
        viewer = getattr(self, "viewer", None)
        if viewer is None:
            return []
        try:
            return [
                str(layer.name) for layer in viewer.layers
                if getattr(layer, "_type_string", "") == "image"
            ]
        except Exception:
            return []

    def _get_run_display_name(self, run_id):
        """Look up display_name for a given run handle.

        2026-05-25 terminology sweep helper.  Falls back to the handle
        itself if (a) no workspace, (b) handle not found, (c) display_name
        is missing or matches the handle.  Result is suitable for
        user-facing strings (Switch dialog, log lines, etc.).

        Args:
            run_id: the run handle (a.k.a. slug, internal ``run_id``).

        Returns:
            display_name string if available, else the handle.
        """
        if not run_id:
            return ""
        try:
            for s in self._list_runs_for_ui():
                if s.get("run_id") == run_id:
                    dn = (s.get("display_name") or "").strip()
                    return dn if dn else run_id
        except Exception:
            pass
        return run_id

    def _list_runs_for_ui(self):
        """Return enriched run summaries for the Curation dropdown / Manage panel.

        Returns:
            List of dicts (see ``_run_management.list_run_summaries``).
            Empty list if no workspace loaded or workspace pre-restructure.
        """
        if not self.working_dir:
            return []
        try:
            from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
            from napari_epicycle.hitl_training_annotation._run_management import (
                list_run_summaries,
            )
            if not _step2_paths.is_new_layout_state(self.working_dir):
                if (_step2_paths.detect_layout_version(self.working_dir)
                        == _step2_paths.LAYOUT_VERSION_UNKNOWN):
                    # A folder with no Step 2 markers at all is not a "legacy
                    # workspace" -- it is an empty project.  Reporting the
                    # synthetic "Default (legacy layout)" row here made the
                    # picker and the banner tell two different stories about
                    # the same folder (second B2 audit).  Empty list -> the
                    # picker says "No run yet", exactly as the banner does.
                    return []
                # Legacy workspace — no multi-run yet; report just the synthetic
                # "default" so the dropdown isn't empty.  After migration this
                # branch goes away.
                return [{
                    "run_id": "default",
                    "display_name": "Default (legacy layout)",
                    "target": "",
                    "backend": "cellpose",
                    "archived": False,
                    "manifest_ok": False,
                    "is_active": True,
                    "lateral_round_count": 0,
                    "axial_round_count": 0,
                    "last_activity_iso": None,
                }]
            return list_run_summaries(self.working_dir)
        except Exception as e:
            return []

    def _create_new_run(self, slug, display_name, *, target="",
                        description="", backend="cellpose",
                        init_spec=None, channels_snapshot=None,
                        notes="", activate=True):
        """Create a new run + add to state JSON; optionally switch to it.

        Args:
            slug: run ID (validated against ``_paths.validate_run_id``).
            display_name: human name.
            target, description, backend, init_spec, channels_snapshot, notes:
                forwarded to ``create_run_on_disk``.
            activate: if True (default), switches active run to the new one
                after creation.  If False, run is created but the user stays
                on the current run.

        Returns:
            dict — the manifest that was written (with canonical created_at).

        Raises:
            RuntimeError: if no workspace loaded or workspace pre-restructure.
            InvalidRunIdError, RunAlreadyExistsError, ValueError, OSError:
                from the underlying helpers.
        """
        if not self.working_dir:
            raise RuntimeError(
                "Cannot create run: no workspace loaded"
            )
        from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
        # Audit-fix 2026-05-24 (user bug: "Run still set to legacy layout, no
        # directory created" after first-run prompt).  Previously this
        # required `is_new_layout_state` (state JSON to exist) — but on a
        # FRESH workspace the state JSON doesn't exist until the first
        # run is created.  Chicken-and-egg.  Now: allow create on UNKNOWN
        # (fresh) + NEW (already has runs); block only on LEGACY/MIXED
        # which require migration first.
        layout_v = _step2_paths.detect_layout_version(self.working_dir)
        if layout_v in (_step2_paths.LAYOUT_VERSION_LEGACY,
                        _step2_paths.LAYOUT_VERSION_MIXED):
            raise RuntimeError(
                "Cannot create run: workspace is on legacy layout.  "
                "Migrate via Workspace > Browse first (the migration "
                "prompt should fire on load)."
            )

        from napari_epicycle.hitl_training_annotation._run_management import (
            add_run_to_state_json,
            create_run_on_disk,
        )

        manifest = create_run_on_disk(
            self.working_dir,
            slug,
            display_name=display_name,
            target=target,
            description=description,
            backend=backend,
            init_spec=init_spec,
            channels_snapshot=channels_snapshot,
            notes=notes,
        )
        # Bug D fix (2026-05-29): pass the init_spec through so the per-run
        # state's Model section is pre-populated from the dialog choice.
        # ``target`` is the run's bucket (lateral/axial) — needed for fork
        # mode's checkpoint resolution.
        add_run_to_state_json(
            self.working_dir,
            slug,
            activate=False,
            init_spec=init_spec,
            target_bucket=target or None,
        )

        if activate:
            self._switch_active_run(slug)
        return manifest

    def _switch_active_run(self, slug):
        """Save current run's state, switch active_run_id, reload state.

        Pre-switch: calls ``_save_internal_state`` so the current run's
        in-memory edits are flushed to ``runs[<old_id>]`` first.

        Post-switch: calls ``_load_internal_state`` which peeks the
        updated ``active_run_id`` from disk and loads ``runs[<new_id>]``
        into the controller fields.  Refreshes the UI panels that
        depend on the loaded state.

        Args:
            slug: target run ID; must already exist in state JSON.

        Raises:
            RuntimeError: if no workspace loaded.
            RunNotFoundError: from ``set_active_run_in_state_json``.
        """
        if not self.working_dir:
            raise RuntimeError("Cannot switch run: no workspace loaded")
        from napari_epicycle.hitl_training_annotation._run_management import (
            set_active_run_in_state_json,
        )
        old_run_id = getattr(self, "active_run_id", "default")
        if slug == old_run_id:
            return

        # Re-entry guard: ignore a second run switch while one is in flight.
        # Rapid double-clicks during the save+load below could corrupt state
        # otherwise.
        if getattr(self, "_run_switch_in_flight", False):
            return

        # Safety check (Phase 8a polish, user bug-report 2026-05-23):
        # * Training in flight → HARD BLOCK
        # * 2D annotator open → WARN + CONFIRM (close on proceed)
        # * Stale patch layers in viewer → CLEAR on proceed
        from napari_epicycle.hitl_training_annotation._switch_run_guard import (
            check_switch_safety, collect_patch_layer_names, SwitchSafety,
        )
        safety, reasons = check_switch_safety(self, getattr(self, "viewer", None))
        # 2026-05-25 terminology sweep: resolve display_names for user-
        # facing strings.  Falls back to handle if no manifest / display.
        _old_dn = self._get_run_display_name(old_run_id)
        _new_dn = self._get_run_display_name(slug)
        if safety == SwitchSafety.BLOCKED:
            try:
                from napari_epicycle.hitl_training_annotation._switch_run_dialog import (
                    show_switch_blocked,
                )
                show_switch_blocked(
                    parent=getattr(self, "ui", None),
                    reasons=reasons,
                    old_run_id=old_run_id,
                    new_run_id=slug,
                    old_display_name=_old_dn,
                    new_display_name=_new_dn,
                )
            except Exception as e:
                pass
            # Restore dropdown to actual active so the UI doesn't lie
            try:
                self._refresh_run_picker()
            except Exception:
                pass
            return
        elif safety == SwitchSafety.NEEDS_CONFIRMATION:
            try:
                from napari_epicycle.hitl_training_annotation._switch_run_dialog import (
                    show_switch_confirm,
                )
                proceed = show_switch_confirm(
                    parent=getattr(self, "ui", None),
                    warnings=reasons,
                    old_run_id=old_run_id,
                    new_run_id=slug,
                    old_display_name=_old_dn,
                    new_display_name=_new_dn,
                )
            except Exception as e:
                proceed = False
            if not proceed:
                try:
                    self._refresh_run_picker()
                except Exception:
                    pass
                return

        self._run_switch_in_flight = True
        # Brick D: the switch is synchronous — flush, save, close the
        # annotator, rewrite state, reload, and (usually) reload a patch.
        self._begin_busy(f"[STEP2-RUN] Switching to '{_new_dn}' ...")
        try:
            # Audit-found CRITICAL 2026-05-23 (state-machine): flush pending
            # state-save + curating-persist timers BEFORE any context change.
            # A queued timer firing AFTER active_run_id mutation would write
            # stale state into NEW run's slot.  Mirror of shutdown() pattern.
            self._flush_pending_state_timers()
            # Brick D fix (2026-09-07): invalidate any patch load still in
            # flight, so the OLD run's patch cannot land after the switch.
            self._patch_load_token = int(getattr(self, "_patch_load_token", 0)) + 1
            # Remember if the 2D annotator was open BEFORE the switch so we
            # can auto-reopen it on the new run if the new run has a saved
            # patch context (user-feedback 2026-05-23: "is it even worth to
            # close the subviewer at all during switch").  Close + auto-
            # reopen gives clean teardown hygiene without the user noticing.
            from napari_epicycle.hitl_training_annotation._switch_run_guard import (
                _annotator_dock_open as _was_annotator_open,
            )
            annotator_was_open = _was_annotator_open(getattr(self, "viewer", None))
            # 1. Flush current run's state to runs[old_id]
            self._save_internal_state()
            # 1a. Close the 2D annotator dock if open (we got confirmation
            #     above).  Done BEFORE state mutation so the dock's teardown
            #     hooks see the still-valid old run context.
            self._close_step2_annotator_dock_safely()
            # 1b. Clear stale patch layers from the napari viewer (image +
            #     work_labels_*).  Switch reloads cleanly to the new run.
            self._clear_patch_viewer_layers_safely()
            # 2. Update active_run_id in step2_state.json
            set_active_run_in_state_json(self.working_dir, slug)
            # 3. Update controller's runtime field BEFORE _load_internal_state
            #    so the peek branch finds the right value.
            self.active_run_id = slug
            # 3a. Clear fields that `_load_internal_state` only overwrites
            #     if non-empty.  Originally I cleared both `loaded_stack_path`
            #     AND `loaded_patch_name` (audit-fix 2026-05-23) but the user
            #     reported 2026-05-25 that switching to a fresh new run left
            #     them unable to load any patch from the Patches list.
            #
            #     Root cause: `loaded_stack_path` is the WORKSPACE-LEVEL
            #     extractions/ directory (one per workspace, shared across
            #     all runs).  Clearing it broke patch-click on the new
            #     run because `_load_patch` does
            #     `os.path.join(loaded_stack_path, filename)` — empty
            #     stack_path = filename becomes a bare relative name +
            #     doesn't resolve.  The Patches list (Qt widget) stays
            #     populated visually but clicks fail.
            #
            #     FIX: only clear `loaded_patch_name` (truly per-run).
            #     Leave `loaded_stack_path` alone so the workspace-level
            #     directory persists across run switches.  If the new
            #     run has a different saved stack path,
            #     `_load_internal_state` will overwrite it via the
            #     `if saved_stack:` guard below.
            self.loaded_patch_name = ""
            # Also reset patch-label UI if present
            try:
                lbl = getattr(self.ui, "loaded_patch_label", None)
                if lbl is not None and hasattr(lbl, "setText"):
                    lbl.setText("")
            except Exception:
                pass
            # 4a. Loss plot: clear OLD run's points BEFORE the reload so
            #     `_load_internal_state`'s call to load_round_history can
            #     populate the plot with the new run's curves on a clean
            #     slate.  Bug 2026-05-27 (user-reported): previously this
            #     block ran AFTER _load_internal_state, immediately
            #     wiping the curves load_round_history had just rendered.
            #     Hence "switch back to a run → no curves visible".
            try:
                loss_plot = getattr(self.ui, "loss_plot", None)
                if loss_plot is not None:
                    if hasattr(loss_plot, "cancel_live_round"):
                        try:
                            loss_plot.cancel_live_round()
                        except Exception:
                            pass
                    if hasattr(loss_plot, "clear_all"):
                        loss_plot.clear_all()
            except Exception as e:
                pass
            # 4b. Reload — _load_internal_state peeks active_run_id and unwraps
            #    runs[slug] into the flat controller fields.  Loss plot
            #    gets populated via load_round_history inside this call
            #    (only when self.losses_by_round is non-empty).
            self._load_internal_state()
            # 5. Refresh UI surfaces that mirror state — same set the workspace
            #    load flow refreshes after _load_internal_state.
            try:
                self._refresh_round_status()
            except Exception as e:
                pass
            try:
                self._refresh_sample_queue()
            except Exception as e:
                pass
            try:
                self._refresh_patch_list_badges()
            except Exception as e:
                pass
            # 5b. Now that the new run's losses are loaded into the plot
            #     (via load_round_history at the end of _load_internal_state),
            #     restore the view filter for the active bucket so only
            #     relevant curves show.
            try:
                loss_plot = getattr(self.ui, "loss_plot", None)
                if loss_plot is not None and hasattr(loss_plot, "set_view_filter"):
                    loss_plot.set_view_filter(self._current_round_key())
            except Exception as e:
                pass

            # User-bug-report 2026-05-23: after switch, clicking "Open
            # Multiview Workspace" raised "Cannot open: missing image/
            # labels context" because the viewer was cleared but the new
            # run's saved patch was not auto-loaded.  Symmetric design:
            # switching is "restore the run's snapshot", so the patch the
            # user was last working on should be restored too.
            #
            # Release-hardening 2026-09-07 (bucket-switch item #4): this is
            # now the SHARED helper, so opening a workspace restores the
            # saved patch exactly the same way a run switch does.
            new_stack = self._effective_stack_path()
            new_patch = str(getattr(self, "loaded_patch_name", "") or "")
            self._auto_reload_saved_patch()

            # If annotator was open before switch, auto-reopen it on the
            # new run (requires the patch was auto-loaded above — checked
            # by _launch_multiview_annotator's own preconditions).  This
            # makes the close-on-switch transparent to the user.
            if annotator_was_open and new_stack and new_patch:
                try:
                    self._launch_multiview_annotator()
                except Exception as e:
                    pass
            elif annotator_was_open:
                pass


            # 2026-05-27 audit follow-up — if the Training History
            # dialog is open, refresh it to the new run.  Otherwise it
            # stays showing the prior run's data until user clicks
            # Refresh manually.
            try:
                dlg = getattr(self, "_training_history_dialog", None)
                if dlg is not None:
                    # Cheap liveness probe — destroyed Qt objects raise.
                    dlg.isVisible()
                    dlg._workspace = self.working_dir
                    dlg._run_id = slug
                    dlg._display_name = self._display_name_for_run(slug)
                    dlg.setWindowTitle(
                        f"Training History — {dlg._display_name}"
                    )
                    dlg.refresh()
            except RuntimeError:
                # Underlying Qt object destroyed; clear stale ref.
                self._training_history_dialog = None
            except Exception as exc:
                pass
        finally:
            self._run_switch_in_flight = False
            self._end_busy()

    def _close_step2_annotator_dock_safely(self):
        """Close the EpiCycle 2D Training Annotator dock if it's mounted.

        Called from ``_switch_active_run`` after the user confirms a switch
        with an open annotator.  Defensive — any failure logs + returns.
        """
        try:
            viewer = getattr(self, "viewer", None)
            if viewer is None:
                return
            from napari_epicycle.annotator.launcher import _find_dock_by_title
            dock = _find_dock_by_title(viewer, "EpiCycle 2D Training Annotator")
            if dock is None:
                return
            # Set parent to None to detach + let Qt garbage-collect; this is
            # the cleanest teardown that respects the annotator's own close
            # hooks (vs. directly calling viewer.window.remove_dock_widget
            # which can bypass them).
            try:
                viewer.window.remove_dock_widget(dock)
            except Exception:
                # Fallback: just hide it
                try:
                    dock.setVisible(False)
                except Exception:
                    pass
        except Exception as e:
            pass

    def _clear_patch_viewer_layers_safely(self):
        """Remove napari viewer layers tied to the current patch.

        Called from ``_switch_active_run`` so the new run starts with a
        clean viewer.  Saved data is on disk; this only clears the live
        UI representation.
        """
        try:
            viewer = getattr(self, "viewer", None)
            if viewer is None:
                return
            from napari_epicycle.hitl_training_annotation._switch_run_guard import (
                collect_patch_layer_names,
            )
            names = collect_patch_layer_names(viewer, self)
            if not names:
                return
            for name in names:
                try:
                    if name in viewer.layers:
                        # Hide before remove, so the render loop never draws a freed GPU buffer
                        try:
                            viewer.layers[name].visible = False
                        except Exception:
                            pass
                        viewer.layers.remove(name)
                except Exception as e:
                    pass
        except Exception as e:
            pass

    def _load_internal_state(self):
        self._ensure_runtime_defaults()
        # F1.5: detect a stale training-in-progress marker BEFORE
        # loading state.  If present, surface a warning so the user
        # knows the prior session was force-killed mid-train and
        # partial state may exist on disk.
        try:
            self._detect_stale_training_marker()
        except Exception:
            pass
        # pass in the current working directory and check for state files
        #
        # Step 2 directory restructure (Phase 3): try the unified v3-first /
        # v2-fallback loader.  If a v3 ``step2_state.json`` exists (workspace
        # has been migrated), unwrap ``runs.default`` into the flat shape the
        # controller has always consumed.  Else fall back to the legacy
        # persistence service which reads ``hitl_state.json`` (v2).
        state = None
        round_metadata = None
        # Pre-load: peek at the v3 state's active_run_id so the rest of the
        # method (and every subsequent path-helper call) uses the correct
        # run.  Without this peek, self.active_run_id stays at the previous
        # workspace's value when switching workspaces.  Audit fix 2026-05-23.
        try:
            from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
            import json as _json
            v3_state_path = _step2_paths.state_json_path(self.working_dir)
            if v3_state_path.is_file():
                try:
                    with open(v3_state_path, "r", encoding="utf-8") as _f:
                        _v3_peek = _json.load(_f)
                    if isinstance(_v3_peek, dict):
                        _peeked = _v3_peek.get("active_run_id")
                        if _peeked:
                            self.active_run_id = str(_peeked)
                except (OSError, _json.JSONDecodeError, ValueError):
                    pass  # corrupted v3 will be re-attempted below
            # If no active_run_id set (legacy workspace, or v3 missing the
            # field), default to "default".
            if not getattr(self, "active_run_id", None):
                self.active_run_id = _step2_paths.DEFAULT_RUN_ID
        except Exception:
            # Defensive: if _paths import fails, fall through and use
            # whatever the controller had.  Save path also defaults gracefully.
            pass

        try:
            from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
            v3_state = _step2_paths.load_step2_state_flat(
                self.working_dir,
                active_run_id=getattr(self, "active_run_id", _step2_paths.DEFAULT_RUN_ID),
            )
            if v3_state is not None and v3_state.get("schema_version", 0) >= 3:
                _loaded_run = getattr(self, "active_run_id", _step2_paths.DEFAULT_RUN_ID)
                print(f"[STATE-LOAD] using v3 step2_state.json "
                      f"(unwrapped from runs[{_loaded_run!r}])")
                state = v3_state
                # round_metadata is loaded separately by the legacy service;
                # post-migration it lives at runs/default/{bucket}/rounds/
                # round_NN/round_metadata.json (per-bucket).  The legacy
                # service can't find those — but the controller mostly uses
                # state['training_data_by_round'] not round_metadata, so this
                # is OK for now.  Phase 4 will route round_metadata loads
                # through _paths.round_metadata_path properly.
                round_metadata = None
        except Exception as e:
            print(f"[STATE-LOAD] v3 load attempt failed ({type(e).__name__}: {e}), "
                  f"falling back to legacy service")
            state = None

        if state is None:
            # Legacy path — v2 hitl_state.json via the persistence service
            state, round_metadata = self.persistence.load_hitl_state(self.working_dir)
        print(f"[STATE-LOAD] working_dir={self.working_dir} | state_found={state is not None} | "
              f"records={len(state.get('sample_records', {})) if state else 0}")
        if state:
            state = self._absolutize_state(state)
            self.current_round = state.get("current_round", 0)
            # Per-view rounds: load from state or migrate from legacy scalar.
            saved_by_view = state.get("current_round_by_view")
            if isinstance(saved_by_view, dict):
                self.current_round_by_view = saved_by_view
            else:
                self.current_round_by_view = self._migrate_round_to_per_view()
            # Normalize round-keyed dicts. New format uses composite keys like
            # "lateral_0", "axial_1". Legacy format uses bare int/str keys.
            # Preserve both formats for backward compatibility.
            def _normalize_round_keys(d):
                out = {}
                for k, v in d.items():
                    try:
                        out[int(k)] = v  # Legacy: bare int
                    except (ValueError, TypeError):
                        out[str(k)] = v  # New: composite key
                return out
            self.training_data_by_round = _normalize_round_keys(state.get("training_data_by_round", {}))
            self.losses_by_round = _normalize_round_keys(state.get("losses_by_round", {}))
            self.model_checkpoint_paths = _normalize_round_keys(state.get("model_checkpoint_paths", {}))
            self.round_backend_info = _normalize_round_keys(state.get("round_backend_info", {}))
            self.current_config = state.get("current_config", {})
            _cc = self.current_config if isinstance(self.current_config, dict) else {}
            print(f"[STATE-LOAD] current_config restored from disk: "
                  f"{len(_cc)} keys; "
                  f"epochs={_cc.get('epochs', '<unset>')!r} "
                  f"lr={_cc.get('learning_rate', '<unset>')!r} "
                  f"ch1={_cc.get('channel1', '<unset>')!r} "
                  f"ch2={_cc.get('channel2', '<unset>')!r} "
                  f"diameter={_cc.get('diameter', '<unset>')!r} "
                  f"niter={_cc.get('n_iterations', '<unset>')!r}")
            self.selected_view = state.get("selected_view", "XY")
            self.view_target = state.get("view_target", "lateral" if self.selected_view == "XY" else "axial")
            self.axial_plane = state.get("axial_plane", "XZ")
            self.sample_records = state.get("sample_records", {})
            # Repair: derive patch_id from sample_id for records that were
            # incorrectly stamped with the loaded patch name during training.
            for sl, rec in self.sample_records.items():
                derived = sl.rsplit("_", 2)[0] if sl.count("_") >= 2 else ""
                if derived and rec.get("patch_id") != derived:
                    rec["patch_id"] = derived
                    rec["roi_id"] = derived
            self.active_curation_checkpoints = state.get("active_curation_checkpoints", {"lateral": "", "axial": ""})
            self.curation_model_source_mode = state.get("curation_model_source_mode", "auto")
            self.imported_curation_model_path = state.get("imported_curation_model_path", "")
            self.curation_pretrained_model_type = state.get(
                "curation_pretrained_model_type",
                str(self.current_config.get("model_type", "cyto3")),
            )
            self._coerce_legacy_model_state()
            # One-time self-heal for workspaces poisoned by the pre-2026-09-07
            # leak (bucket-switch item #1): training used to write the fresh
            # ROUND checkpoint into the External-model-file slot.  If the
            # stored "external" path is byte-identical to one of this run's
            # own active round checkpoints, it was written by that bug, not
            # browsed to by the user -- clear it so the panel stops claiming
            # an external model that is really a round model.  A genuinely
            # browsed path (anything else, including another RUN's model)
            # is left alone.
            _imported = str(self.imported_curation_model_path or "")
            if self._should_heal_imported_slot():
                self.imported_curation_model_path = ""
                print(f"[STATE-LOAD] cleared External-model-file slot: it held "
                      f"this run's own round checkpoint ({_imported.strip()}), "
                      f"which only the pre-2026-09-07 promote-leak could have "
                      f"put there")
            self.training_mode = state.get("training_mode", "cumulative")
            # Per-bucket modes (ADDED 2026-09-07).  Absent in older state
            # files -> both buckets inherit the legacy single value.
            self.training_mode_by_bucket = self._coerce_training_mode_by_bucket(
                state.get("training_mode_by_bucket"),
                fallback=self.training_mode,
            )
            # Restore auto-update model checkbox.
            auto_update = state.get("auto_update_model_after_training", True)
            _ui = getattr(self, "ui", None)
            auto_cb = getattr(_ui, "auto_update_model_cb", None) if _ui is not None else None
            if auto_cb is not None and hasattr(auto_cb, "setChecked"):
                auto_cb.setChecked(bool(auto_update))
            # Restore patch context (don't auto-load data, just set the fields).
            saved_stack = state.get("loaded_stack_path", "")
            saved_patch = state.get("loaded_patch_name", "")
            if saved_stack:
                self.loaded_stack_path = saved_stack
            # 2026-05-26 audit-J MEDIUM fix: re-scan extractions/ on
            # workspace load so newly-extracted patches show up without the
            # user clicking Browse manually.  Previously the patch list was
            # empty until the user re-Browsed.
            # B3: this runs even with nothing persisted, so the Step 1
            # default populates the list on a plain workspace open.
            try:
                self._populate_patch_list_from_disk(notify=False)
            except Exception as e:
                pass
            if saved_patch:
                self.loaded_patch_name = saved_patch
                # Update UI label if available.
                _ui = getattr(self, "ui", None)
                patch_label = getattr(_ui, "loaded_patch_label", None) if _ui is not None else None
                if patch_label is not None and hasattr(patch_label, "setText"):
                    patch_label.setText(f"Patch: {saved_patch}")
            # Reload persisted pairs from disk into segmentation_manager.
            self._reload_persisted_pairs_from_disk()
        ui = getattr(self, "ui", None)
        if ui is not None:
            combo = getattr(ui, "model_source_combo", None)
            if combo is not None and hasattr(combo, "setCurrentText"):
                labels = {
                    "auto": "Auto (Recommended)",
                    "round": "Round model",
                    "imported": "External model file",
                    "pretrained": "Cellpose pretrained",
                    "none": "None (Manual/SAM only)",
                }
                # 2026-05-29 audit fix (the real "imported model lost on
                # resume" root cause): setCurrentText fires currentTextChanged
                # -> _on_model_source_changed, which would re-read the imported-
                # path widget BEFORE it's repopulated (next line) and CLOBBER
                # the just-restored imported_curation_model_path to "".  Block
                # signals so restore doesn't re-enter the change handler.
                _blocked = hasattr(combo, "blockSignals")
                if _blocked:
                    combo.blockSignals(True)
                try:
                    combo.setCurrentText(labels.get(self.curation_model_source_mode, "Auto (Recommended)"))
                finally:
                    if _blocked:
                        combo.blockSignals(False)
            path_edit = getattr(ui, "imported_model_path", None)
            if path_edit is not None and hasattr(path_edit, "setText"):
                path_edit.setText(str(self.imported_curation_model_path or ""))
            pre_combo = getattr(ui, "curation_pretrained_combo", None)
            if pre_combo is not None and hasattr(pre_combo, "setCurrentText"):
                _pblocked = hasattr(pre_combo, "blockSignals")
                if _pblocked:
                    pre_combo.blockSignals(True)
                try:
                    pre_combo.setCurrentText(str(self.curation_pretrained_model_type or "cyto3"))
                finally:
                    if _pblocked:
                        pre_combo.blockSignals(False)
            # Training mode is per bucket (2026-09-07, item #5) -- show the
            # ACTIVE bucket's stored choice.  The helper blocks the combo's
            # signals so the restore can't re-enter the change handler.
            try:
                self._apply_training_mode_for_active_bucket()
            except Exception:
                pass
            # 2026-05-29 audit fix: guard the view-radio restore — setChecked
            # fires _on_view_changed -> _update_training_ui_for_view, which
            # setText's the Checkpoint widget and would otherwise schedule a
            # spurious save (and persist a prior-round checkpoint) during restore.
            _prev_view_guard = getattr(self, "_init_transparency_loading", False)
            self._init_transparency_loading = True
            try:
                self._set_view_radio(self.selected_view)
            finally:
                self._init_transparency_loading = _prev_view_guard
            self._set_axial_combo_enabled()
            self._refresh_model_source_controls()
        self._ensure_curation_checkpoint_state()
        if round_metadata is not None and not self.current_config:
            self.current_config = dict(round_metadata.config_snapshot)
        # 2026-05-27 workspace-load widget restore: push the restored
        # current_config into Training Setup, Advanced, and Inference
        # widgets so the user doesn't have to re-enter parameters
        # every session.  Runs ONLY here (inside _load_internal_state)
        # which is only entered after a workspace is picked — the
        # controller-init call to _sync_inference_controls_from_config
        # is harmless (current_config still {} then) and doesn't need
        # changing.  Both calls placed AFTER the round_metadata
        # fallback above so any config_snapshot recovered from prior
        # round metadata also reaches the widgets.
        self._sync_training_controls_from_config()
        self._sync_inference_controls_from_config()
        self._apply_model_source_selection(emit_log=False)
        self._sync_data_to_add_list()
        self._refresh_sample_queue()
        self._refresh_round_status()

        # Restore checkpoint path in Training Setup UI.
        # 2026-05-26: model_name_line removed — model name is now an
        # auto-derived display label refreshed by
        # _refresh_model_name_display on every state change.
        if ui is not None:
            # Auto-populate starting checkpoint from prior round.
            prior_ckpts = self._resolve_prior_model_checkpoint()
            if prior_ckpts:
                latest = next(iter(prior_ckpts.values()), "")
                ckpt_edit = getattr(ui, "starting_checkpoint_path", None)
                if ckpt_edit is not None and hasattr(ckpt_edit, "setText") and latest:
                    # 2026-05-29 audit fix: guard the auto-populate so the
                    # checkpoint handler doesn't persist this prior-round path
                    # over the restored manual checkpoint + schedule a save.
                    _prev_ckpt_guard = getattr(self, "_init_transparency_loading", False)
                    self._init_transparency_loading = True
                    try:
                        ckpt_edit.setText(str(latest))
                    finally:
                        self._init_transparency_loading = _prev_ckpt_guard

        # Restore prior round loss curves into the plot widget.
        loss_plot = getattr(self.ui, "loss_plot", None) if ui is not None else None
        if loss_plot is not None and self.losses_by_round:
            loss_plot.load_round_history(self.losses_by_round)



