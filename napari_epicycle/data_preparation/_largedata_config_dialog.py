"""Themed Qt dialog for the Step-1 large-data "Prepare settings".

Lets the user pick a resource-adaptive PRESET (Safe / Balanced / Fast) or drop
into CUSTOM to hand-set every convert knob, with a live "Estimated peak RAM"
readout.  The adaptive math + validation are the PURE helpers in
``packages/epicycle_largedata/config.py`` (unit-tested headless); this module is
only the Qt shell that wires them, mirroring the other Step-1 dialogs
(``_migrate_step1_dialog.py``) for theming.

Contract:

* ``LargeDataConfigDialog(parent, cpu_count=, total_ram_bytes=,
  avail_ram_bytes=, initial_config=, dtype_itemsize=, source_shape=,
  source_regime=)`` — parent-owned (no taskbar orphan).  ``source_shape` /
  ``source_regime`` are optional and only sharpen the peak-memory readout (see
  ``config.estimate_peak_ram_bytes``); without them it shows the chunk-only
  bound.
* Choosing Safe / Balanced / Fast fills the advanced fields from
  :func:`recommend_convert_settings` (adaptive to *this* machine) and makes them
  read-only; Custom enables editing.
* :meth:`get_config` returns ``{preset, num_workers, chunk, chunk_zyx,
  compressor, downsample, min_size}``; the module helper
  :func:`open_largedata_config_dialog` returns that dict on OK or ``None`` on
  Cancel.  ``chunk_zyx`` is the shape the writer uses; ``chunk`` is a legacy
  cubic mirror of its Z axis, kept only so an OLDER checkout reading the same
  persisted config degrades to a small safe cube.
"""

from __future__ import annotations

from typing import Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from packages.epicycle_largedata.config import (
    ALLOWED_CHUNKS,
    ALLOWED_CHUNK_ZYX,
    ALLOWED_COMPRESSORS,
    DEFAULT_COMPRESSOR,
    DEFAULT_CHUNK_ZYX,
    normalize_chunk_zyx,
    DEFAULT_PRESET,
    PRESET_NAMES,
    estimate_peak_ram_bytes,
    normalize_convert_config,
    recommend_convert_settings,
)

_GB = 1024 ** 3
_MB = 1024 ** 2

# Combo label <-> internal preset name.  "Custom" maps to the non-preset
# "custom" sentinel the config layer tolerates (raw user knobs, no recommender).
_PRESET_LABELS = (("Safe", "safe"), ("Balanced", "balanced"),
                  ("Fast", "fast"), ("Custom", "custom"))
_LABEL_FOR = {name: label for label, name in _PRESET_LABELS}
_NAME_FOR = {label: name for label, name in _PRESET_LABELS}


def _dialog_styles():
    """Pull dialog stylesheets from theme.py at call time (avoid import cost /
    circulars); return ``("", "", "")`` if theme is unavailable (headless)."""
    try:
        from napari_epicycle.ui_shell.theme import (
            DIALOG_FORM_STYLE, _DIALOG_BTN_CANCEL, _DIALOG_BTN_CONFIRM,
        )
        return DIALOG_FORM_STYLE, _DIALOG_BTN_CANCEL, _DIALOG_BTN_CONFIRM
    except Exception:
        return "", "", ""


class LargeDataConfigDialog(QDialog):
    """Preset + power-user-override settings for the large-data convert."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        cpu_count: int,
        total_ram_bytes: int,
        avail_ram_bytes: int,
        initial_config: Optional[dict] = None,
        dtype_itemsize: int = 2,
        source_shape=None,
        source_regime=None,
    ):
        super().__init__(parent)
        self._cpu = max(1, int(cpu_count))
        self._total_ram = int(total_ram_bytes)
        self._avail_ram = int(avail_ram_bytes)
        self._itemsize = max(1, int(dtype_itemsize))
        # The SOURCE volume shape, when the caller knows it.  It selects the
        # honest compressed-path peak-RAM model (a chunk_z x Y x X slab per
        # worker) over the chunk-only memmap bound -- see
        # `config.estimate_peak_ram_bytes`.  None -> the chunk-only readout.
        self._source_shape = source_shape
        # "memmap" / "compressed" / None -- WHICH peak-RAM model applies.  An
        # uncompressed source really does read one chunk at a time, so charging
        # it the compressed path's Z-slab would clamp a big machine to 1 worker
        # for no reason.  See config.resolve_source_regime.
        self._source_regime = source_regime
        # Set by _on_preset_changed when the RAM clamp reduced the core count,
        # so the readout can explain the reduction in plain words.
        self._clamped_from = None
        # The store codec is not a user knob in the dialog (it is a property of
        # the writer, not of the machine); it is carried through so a persisted
        # config keeps whatever it had and a new one gets the default.
        self._compressor = DEFAULT_COMPRESSOR
        if isinstance(initial_config, dict):
            _c = initial_config.get("compressor")
            if isinstance(_c, str) and _c.lower() in ALLOWED_COMPRESSORS:
                self._compressor = _c.lower()

        self.setWindowTitle("Large-data preparation settings")
        self.setMinimumWidth(460)
        self.setModal(True)
        form_style, self._btn_cancel_style, self._btn_confirm_style = _dialog_styles()
        self.setStyleSheet(form_style)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        # --- header: detected resources ------------------------------------
        total_gb = self._total_ram / _GB
        avail_gb = self._avail_ram / _GB
        if self._total_ram > 0:
            detected = (f"Detected: {self._cpu} cores · {total_gb:.0f} GB "
                        f"RAM ({avail_gb:.0f} GB free)")
        else:
            # psutil-less host: RAM unknown, report cores only.
            detected = f"Detected: {self._cpu} cores · RAM unknown"
        header = QLabel(detected)
        header.setStyleSheet("font-weight: bold; color: #d4a843;")
        header.setWordWrap(True)
        root.addWidget(header)

        intro = QLabel(
            "Choose how much of your machine to use when converting a big stack "
            "for smooth viewing. A faster preset uses more cores and memory."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #9aa0a6; font-size: 10px;")
        root.addWidget(intro)

        # --- preset dropdown -----------------------------------------------
        preset_row = QFormLayout()
        preset_row.setLabelAlignment(Qt.AlignRight)
        self.preset_combo = QComboBox()
        for label, _name in _PRESET_LABELS:
            self.preset_combo.addItem(label)
        self.preset_combo.setToolTip(
            "Safe = 1 worker, most portable. Balanced = a few cores. "
            "Fast = all cores. They all write the file the same way — they "
            "differ only in how many cores are used while converting. "
            "Custom = set every value yourself.")
        preset_row.addRow("Preset", self.preset_combo)
        root.addLayout(preset_row)

        # --- advanced knobs -------------------------------------------------
        adv = QFormLayout()
        adv.setLabelAlignment(Qt.AlignRight)
        adv.setSpacing(8)

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, self._cpu)
        self.workers_spin.setToolTip(
            "Parallel workers — more = faster, uses more memory and CPU. "
            "Capped at your core count.")
        adv.addRow("Parallel workers", self.workers_spin)

        # Brick 4b: the chunk is a per-axis Z x Y x X SHAPE, not a cube edge.
        # Thin in Z + big in-plane is what makes 2D scrolling cheap (little
        # unread depth decompressed) AND the write fast (few, large chunks).
        # The cubic edges stay available under "Advanced" for reproducing an
        # older store's layout; they are never the default.
        self.chunk_combo = QComboBox()
        for zyx in ALLOWED_CHUNK_ZYX:
            self.chunk_combo.addItem(
                "{} deep x {} x {}".format(*zyx), tuple(zyx))
        for c in ALLOWED_CHUNKS:
            self.chunk_combo.addItem(
                f"Older layout: {c} in every direction", (int(c),) * 3)
        self.chunk_combo.setToolTip(
            "How big a piece of the image is read or written at a time, as "
            "depth x height x width. A piece that is shallow but wide keeps "
            "scrolling through slices fast — looking at one slice only has to "
            "unpack a few slices' worth of depth instead of a deep block. "
            "32 x 128 x 128 is the default and suits most images.")
        adv.addRow("Block size", self.chunk_combo)

        self.downsample_spin = QSpinBox()
        self.downsample_spin.setRange(2, 4)
        self.downsample_spin.setToolTip(
            "Coarsening factor per level — how much each overview level "
            "shrinks. 2 = halve each step (most levels); higher = fewer, "
            "coarser levels.")
        adv.addRow("Coarsening factor", self.downsample_spin)

        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(64, 1024)
        self.min_size_spin.setSingleStep(64)
        self.min_size_spin.setToolTip(
            "Smallest overview — stop making coarser levels once an axis "
            "would fall below this many voxels. Smaller = more overview levels.")
        adv.addRow("Smallest overview (min size)", self.min_size_spin)

        root.addLayout(adv)

        # --- live peak-RAM readout -----------------------------------------
        self.estimate_label = QLabel("")
        self.estimate_label.setStyleSheet("color: #9aa0a6; font-size: 10px;")
        self.estimate_label.setWordWrap(True)
        root.addWidget(self.estimate_label)

        # --- OK / Cancel ----------------------------------------------------
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet(self._btn_cancel_style)
        self.cancel_button.clicked.connect(self.reject)
        btn_row.addWidget(self.cancel_button)
        self.ok_button = QPushButton("OK")
        self.ok_button.setStyleSheet(self._btn_confirm_style)
        self.ok_button.clicked.connect(self.accept)
        btn_row.addWidget(self.ok_button)
        root.addLayout(btn_row)

        # --- wire signals + seed -------------------------------------------
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.workers_spin.valueChanged.connect(self._update_estimate)
        self.chunk_combo.currentIndexChanged.connect(self._update_estimate)
        self.downsample_spin.valueChanged.connect(self._update_estimate)
        self.min_size_spin.valueChanged.connect(self._update_estimate)

        self._seed(initial_config)

    # -- seeding ------------------------------------------------------------
    def _seed(self, initial_config: Optional[dict]) -> None:
        """Populate the dialog from *initial_config* (or the default preset).

        A NAMED preset re-selects the preset combo, so fields refill from the
        recommender for THIS machine (adaptive even if the machine changed since
        the config was saved).  A ``custom`` config selects Custom and loads its
        raw saved knobs verbatim.
        """
        cfg = normalize_convert_config(initial_config)
        preset = (cfg or {}).get("preset", DEFAULT_PRESET)
        if not isinstance(preset, str) or preset not in _LABEL_FOR:
            # Unknown / missing preset name but valid knobs -> treat as custom;
            # otherwise fall back to the default preset.
            preset = "custom" if cfg is not None else DEFAULT_PRESET

        if preset in PRESET_NAMES:
            # Selecting the combo fires _on_preset_changed -> fills + locks
            # fields from the recommender.  Force the handler even if the index
            # is already 0 (default) by calling it explicitly after set.
            self.preset_combo.setCurrentText(_LABEL_FOR[preset])
            self._on_preset_changed()
        else:  # custom
            self.preset_combo.setCurrentText(_LABEL_FOR["custom"])
            self._on_preset_changed()          # enables the fields
            if cfg is not None:
                self._set_fields(cfg)          # then load the saved raw knobs

    # -- preset handling ----------------------------------------------------
    def _current_preset_name(self) -> str:
        return _NAME_FOR.get(self.preset_combo.currentText(), "custom")

    def _on_preset_changed(self, *args) -> None:
        preset = self._current_preset_name()
        editable = preset == "custom"
        if not editable:
            rec = recommend_convert_settings(
                preset,
                total_ram_bytes=self._total_ram,
                avail_ram_bytes=self._avail_ram,
                cpu_count=self._cpu,
                dtype_itemsize=self._itemsize,
                shape=self._source_shape,
                regime=self._source_regime,
            )
            self._clamped_from = (
                self._cpu if int(rec["num_workers"]) < self._cpu else None)
            self._set_fields(rec)
        # Named presets are read-only; Custom is editable.
        for w in (self.workers_spin, self.chunk_combo,
                  self.downsample_spin, self.min_size_spin):
            w.setEnabled(editable)
        self._update_estimate()

    def _set_fields(self, cfg: dict) -> None:
        """Set the four advanced widgets from *cfg* with signals blocked (one
        estimate refresh at the end)."""
        widgets = (self.workers_spin, self.chunk_combo,
                   self.downsample_spin, self.min_size_spin)
        for w in widgets:
            w.blockSignals(True)
        try:
            self.workers_spin.setValue(int(cfg["num_workers"]))
            self._select_chunk(normalize_chunk_zyx(
                cfg.get("chunk_zyx", cfg.get("chunk")), DEFAULT_CHUNK_ZYX))
            self.downsample_spin.setValue(int(cfg["downsample"]))
            self.min_size_spin.setValue(int(cfg["min_size"]))
        finally:
            for w in widgets:
                w.blockSignals(False)
        self._update_estimate()

    # -- chunk shape helpers -------------------------------------------------
    def _current_chunk_zyx(self) -> tuple:
        """The selected ``(cz, cy, cx)``; the default if nothing is selected."""
        return normalize_chunk_zyx(
            self.chunk_combo.currentData(), DEFAULT_CHUNK_ZYX)

    def _select_chunk(self, zyx) -> None:
        """Select the entry whose data equals *zyx*; else add + select it (so a
        persisted custom shape is never silently rewritten to the default)."""
        want = tuple(int(v) for v in zyx)
        for i in range(self.chunk_combo.count()):
            if tuple(self.chunk_combo.itemData(i) or ()) == want:
                self.chunk_combo.setCurrentIndex(i)
                return
        self.chunk_combo.addItem("{} deep x {} x {}".format(*want), want)
        self.chunk_combo.setCurrentIndex(self.chunk_combo.count() - 1)

    # -- estimate -----------------------------------------------------------
    def _update_estimate(self, *args) -> None:
        zyx = self._current_chunk_zyx()
        workers = self.workers_spin.value()
        peak = estimate_peak_ram_bytes(
            workers, zyx, self._itemsize,
            shape=self._source_shape, regime=self._source_regime)
        mb = peak / _MB
        # Read as GB once the number stops being legible in MB.
        if mb >= 1024:
            amount = f"~{mb / 1024:.1f} GB"
        else:
            amount = f"~{mb:.0f} MB"
        if self._source_shape is not None:
            detail = (f"({workers} worker(s), each holding a {zyx[0]}-slice "
                      f"block covering the whole image width and height)")
        else:
            detail = (f"({workers} worker(s) x a {zyx[0]} x {zyx[1]} x {zyx[2]} "
                      f"block)")
        lines = [f"Estimated peak memory: {amount} {detail}"]
        # If the recommender had to hold workers back for memory, SAY SO -- a
        # silently reduced core count reads as the app being slow for no reason.
        if self._clamped_from is not None and self._clamped_from > workers:
            lines.append(
                f"Using {workers} of your {self._clamped_from} cores — more "
                f"would need more memory than you have free.")
        self.estimate_label.setText("\n".join(lines))

    # -- result -------------------------------------------------------------
    def get_config(self) -> dict:
        """Return ``{preset, num_workers, chunk, chunk_zyx, compressor,
        downsample, min_size}`` for the current widget state."""
        zyx = self._current_chunk_zyx()
        return {
            "preset": self._current_preset_name(),
            "num_workers": int(self.workers_spin.value()),
            # Legacy cubic mirror = the Z axis, NOT the in-plane edge: an older
            # checkout reads this int and prepares a CUBIC store with it, and a
            # cubic 128 is the worst case for 2D scrolling (the thing Brick 3c
            # reverted).  32 degrades safely.  See config.recommend_convert_settings.
            "chunk": int(zyx[0]),
            "chunk_zyx": tuple(int(v) for v in zyx),
            "compressor": self._compressor,
            "downsample": int(self.downsample_spin.value()),
            "min_size": int(self.min_size_spin.value()),
        }


def open_largedata_config_dialog(
    parent: Optional[QWidget],
    *,
    cpu_count: int,
    total_ram_bytes: int,
    avail_ram_bytes: int,
    initial_config: Optional[dict] = None,
    dtype_itemsize: int = 2,
) -> Optional[dict]:
    """Construct + exec the dialog; return the config dict on OK, else ``None``."""
    dlg = LargeDataConfigDialog(
        parent,
        cpu_count=cpu_count,
        total_ram_bytes=total_ram_bytes,
        avail_ram_bytes=avail_ram_bytes,
        initial_config=initial_config,
        dtype_itemsize=dtype_itemsize,
    )
    if dlg.exec_():
        return dlg.get_config()
    return None
