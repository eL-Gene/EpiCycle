"""Refined-model registry dialog.

Modal popup that lists every refined model in the workspace's
``artifacts/step3/refinement/models/`` tree (post-2026-05-20
restructure layout), scanned from each bundle's ``manifest.json``
(schema 2.0).  Three per-row actions:

  - Use for re-prediction  — copies the row's lateral/axial weight
    paths into Predict + Reconstruct's model fields.
  - Open folder            — reveals the bundle dir in the OS file
    browser (refined bundles only).
  - Set as init source     — flips Fine-Tuning's Init combo to
    "From prior refinement" and populates the lateral + axial
    source combos with the row's weight paths (refined bundles only).

2026-05-21 Stage 2: Cellpose pretrained models now appear as a
second section in the same picker.  Selecting a pretrained row +
"Use for re-prediction" sets both lateral + axial paths to the
pretrained name (mirroring `_on_pretrained_changed`'s behavior).
Open Folder and Set as Init are disabled for pretrained rows.

Future Stage 3 will add an external-Browse footer (lat/ax file
pickers) and remove the 3-mode source dropdown from Predict +
Reconstruct.

Usage from the controller::

    rows = self._scan_refinement_model_registry()
    RefinementRegistryDialog.show_dialog(
        parent=self.ui,
        rows=rows,
        on_use=self._registry_action_use,
        on_open=self._registry_action_open_folder,
        on_init=self._registry_action_set_init,
    )
"""

from __future__ import annotations

from typing import Callable, List, Optional

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


# Module-level constant — modular pretrained model catalog.  Add new
# entries here when Cellpose ships new general-purpose models worth
# exposing.  Each entry is (name, description).  Description renders in
# the dialog so users know what each is good for.
#
# 2026-05-21: deliberately minimal.  cyto3 is the recommended default
# for epithelial / cytoplasm.  nuclei is the only other broadly-useful
# option for this codebase's tissue users.  cyto2_cp3 dropped (legacy,
# superseded by cyto3).  Expand only when there's a real use case.
PRETRAINED_MODELS = (
    ("cyto3", "General epithelial / cytoplasm — recommended default"),
    ("nuclei", "Nuclear staining (DAPI, Hoechst, etc.)"),
)


# Style tokens — copied verbatim from extraction_config_dialog.py so
# the two dialogs render identically (dark bg, subtle borders, darker
# spinbox fills, amber primary action).
_BG = "#2b2b2b"
_BG_CONTENT = "#1e1e2e"
_TEXT = "#e0e0e0"
_TEXT_DIM = "#999999"
_ACCENT = "#d4a843"
_ACCENT_BG = "rgba(212,168,67,0.20)"
_ACCENT_BORDER = "rgba(212,168,67,0.40)"

_DIALOG_STYLESHEET = (
    f"QDialog {{ background: {_BG}; border: 1px solid "
    f"rgba(255,255,255,0.12); border-radius: 6px; }}"
    f"QLabel {{ color: {_TEXT}; font-size: 11px; }}"
    f"QTableWidget {{ background: {_BG_CONTENT}; color: {_TEXT}; "
    f"border: 1px solid rgba(255,255,255,0.10); border-radius: 4px; "
    f"gridline-color: rgba(255,255,255,0.06); }}"
    f"QHeaderView::section {{ background: rgba(255,255,255,0.04); "
    f"color: {_TEXT_DIM}; font-size: 10px; font-weight: bold; "
    f"padding: 4px 6px; border: none; "
    f"border-bottom: 1px solid rgba(255,255,255,0.10); }}"
    f"QTableWidget::item:selected {{ background: {_ACCENT_BG}; "
    f"color: {_TEXT}; }}"
)

_BTN_PRIMARY_STYLE = (
    f"QPushButton {{ background: {_ACCENT_BG}; color: {_TEXT}; "
    f"border: 1px solid {_ACCENT_BORDER}; border-radius: 3px; "
    f"padding: 4px 14px; font-weight: bold; }}"
    f"QPushButton:hover {{ background: {_ACCENT}; color: black; }}"
    f"QPushButton:disabled {{ background: rgba(212,168,67,0.06); "
    f"color: {_TEXT_DIM}; border: 1px solid rgba(255,255,255,0.10); }}"
)

_BTN_SECONDARY_STYLE = (
    f"QPushButton {{ background: rgba(255,255,255,0.05); "
    f"color: {_TEXT_DIM}; border: 1px solid rgba(255,255,255,0.10); "
    f"border-radius: 3px; padding: 4px 14px; }}"
    f"QPushButton:hover {{ background: rgba(255,255,255,0.10); "
    f"color: {_TEXT}; }}"
    f"QPushButton:disabled {{ background: rgba(255,255,255,0.02); "
    f"color: rgba(255,255,255,0.20); }}"
)

_BTN_CANCEL_STYLE = _BTN_SECONDARY_STYLE


class RefinementRegistryDialog(QDialog):
    """Modal dialog listing every refined model in the workspace."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        rows: Optional[List[dict]] = None,
        on_use: Optional[Callable[[dict], None]] = None,
        on_open: Optional[Callable[[dict], None]] = None,
        on_init: Optional[Callable[[dict], None]] = None,
        pretrained_models: Optional[List[tuple]] = None,
        step2_models: Optional[List[dict]] = None,
        show_init: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select model")
        self.setMinimumWidth(820)
        self.setMinimumHeight(440)
        self.setSizeGripEnabled(True)
        self.setStyleSheet(_DIALOG_STYLESHEET)

        self._rows: List[dict] = list(rows or [])
        self._pretrained_models: List[tuple] = (
            list(pretrained_models)
            if pretrained_models is not None
            else list(PRETRAINED_MODELS)
        )
        # 2026-06-08 reader-unify: Step 2 run models ("From HITL model"
        # init sources) get their own section.  Each entry is a dict
        # from the controller's _list_all_model_sources scanner with
        # source_type ∈ {"step2_run", "step2_legacy"} (weights_path,
        # bucket, run_id, round_id, human_label).
        self._step2_models: List[dict] = list(step2_models or [])
        self._on_use = on_use
        self._on_open = on_open
        self._on_init = on_init

        # Per-table-row payload cache, built during _populate_table.
        # Each entry is (kind, payload_dict).  kind ∈ {"header",
        # "refined", "pretrained"}.  Lookup table is what
        # _selected_row reads to dispatch action handlers correctly.
        self._row_payloads: List[tuple] = []

        # Resolve apply_button_icon once for all sections (Stages 1+3
        # both use it: dialog action buttons + external Apply button).
        # Wrapped because non-Qt test contexts may fail to import.
        try:
            from ..ui_shell.theme import apply_button_icon as _abi
            self._apply_button_icon = _abi
        except Exception:
            self._apply_button_icon = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(10)

        # ── Title ────────────────────────────────────────────────────
        title = QLabel("Select model")
        title.setStyleSheet(
            "color: #eeeeee; font-size: 14px; font-weight: bold;"
        )
        outer.addWidget(title)

        # ── Description ──────────────────────────────────────────────
        desc = QLabel(
            "Refined models + Step 2 run models trained in this "
            "workspace + Cellpose pretrained models.  Select a row + "
            "click an action.  <i>Open folder</i> and <i>Set as init "
            "source</i> apply to refined bundles + Step 2 run models "
            "(not pretrained)."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 10px;")
        outer.addWidget(desc)

        # ── Table ────────────────────────────────────────────────────
        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels([
            "Name", "Mode", "Lateral", "Axial", "Created", "Init",
        ])
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        h = self._table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Stretch)
        h.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(5, QHeaderView.Stretch)
        outer.addWidget(self._table, 1)

        self._populate_table()
        # Update action-button enable state when the selection changes.
        self._table.itemSelectionChanged.connect(
            self._refresh_button_enabled_state,
        )

        # ── Empty-state hint ─────────────────────────────────────────
        # Stage 2: only show "no refined models" hint if there are no
        # refined bundles — pretrained section is always populated.
        if not self._rows:
            empty = QLabel(
                "No refined models yet — train one from the "
                "Fine-Tuning section.  Until then, Cellpose pretrained "
                "models below are your starting point."
            )
            empty.setWordWrap(True)
            empty.setStyleSheet(
                f"color: {_TEXT_DIM}; font-style: italic; "
                "padding: 8px 12px;"
            )
            outer.addWidget(empty)

        # ── Browse external (Stage 3) ───────────────────────────────
        # Lets the user point at external model files on disk (covers
        # the rare case of importing a model trained elsewhere, OR
        # mixing lateral + axial from different sources).  Two file
        # pickers + an "Apply external" button.  Separate from the
        # table so it doesn't conflict with row selection.
        ext_box = QFrame()
        ext_box.setFrameShape(QFrame.NoFrame)
        ext_box.setStyleSheet(
            f"background: rgba(255,255,255,0.03); "
            "border: 1px solid rgba(255,255,255,0.07); "
            "border-radius: 4px;"
        )
        ext_lay = QVBoxLayout(ext_box)
        ext_lay.setContentsMargins(10, 8, 10, 8)
        ext_lay.setSpacing(4)

        ext_title = QLabel("Browse external paths")
        ext_title.setStyleSheet(
            "color: #cccccc; font-size: 11px; font-weight: bold;"
        )
        ext_lay.addWidget(ext_title)

        ext_desc = QLabel(
            "Point at model files on disk — e.g. a model trained "
            "elsewhere, or to mix lateral + axial from different "
            "sources.  Both fields are optional; an empty field "
            "leaves the existing path unchanged."
        )
        ext_desc.setWordWrap(True)
        ext_desc.setStyleSheet(
            f"color: {_TEXT_DIM}; font-size: 10px;"
        )
        ext_lay.addWidget(ext_desc)

        # Lateral row
        self._ext_lat_path = QLineEdit()
        self._ext_lat_path.setPlaceholderText(
            "Lateral (XY) model file…"
        )
        self._ext_lat_browse = QPushButton("Browse…")
        self._ext_lat_browse.setStyleSheet(_BTN_SECONDARY_STYLE)
        self._ext_lat_browse.clicked.connect(
            lambda: self._handle_ext_browse_clicked("lateral"),
        )
        lat_ext_row = QHBoxLayout()
        lat_ext_row.addWidget(QLabel("Lateral:"))
        lat_ext_row.addWidget(self._ext_lat_path, 1)
        lat_ext_row.addWidget(self._ext_lat_browse)
        ext_lay.addLayout(lat_ext_row)

        # Axial row
        self._ext_ax_path = QLineEdit()
        self._ext_ax_path.setPlaceholderText(
            "Axial (XZ/YZ) model file…"
        )
        self._ext_ax_browse = QPushButton("Browse…")
        self._ext_ax_browse.setStyleSheet(_BTN_SECONDARY_STYLE)
        self._ext_ax_browse.clicked.connect(
            lambda: self._handle_ext_browse_clicked("axial"),
        )
        ax_ext_row = QHBoxLayout()
        ax_ext_row.addWidget(QLabel("Axial:"))
        ax_ext_row.addWidget(self._ext_ax_path, 1)
        ax_ext_row.addWidget(self._ext_ax_browse)
        ext_lay.addLayout(ax_ext_row)

        # Apply button
        ext_apply_row = QHBoxLayout()
        ext_apply_row.addStretch(1)
        self._ext_apply_btn = QPushButton("Apply external paths")
        self._ext_apply_btn.setStyleSheet(_BTN_SECONDARY_STYLE)
        self._ext_apply_btn.setToolTip(
            "Set the paths above as the active prediction model.  "
            "Empty fields are skipped (existing path preserved)."
        )
        if self._apply_button_icon:
            self._apply_button_icon(self._ext_apply_btn, "mdi6.check")
        self._ext_apply_btn.clicked.connect(self._handle_ext_apply_clicked)
        ext_apply_row.addWidget(self._ext_apply_btn)
        ext_lay.addLayout(ext_apply_row)

        outer.addWidget(ext_box)

        # ── Action button row ────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        # Use the unified `_apply_button_icon` resolved earlier (Stage 3).
        apply_button_icon = self._apply_button_icon

        self._use_btn = QPushButton("Use for re-prediction")
        self._use_btn.setStyleSheet(_BTN_PRIMARY_STYLE)
        self._use_btn.setToolTip(
            "Set the selected model as the active prediction source "
            "(copies lateral + axial paths into Predict + Reconstruct)."
        )
        if apply_button_icon:
            apply_button_icon(self._use_btn, "mdi6.reload")
        self._use_btn.clicked.connect(self._handle_use_clicked)
        btn_row.addWidget(self._use_btn)

        self._open_btn = QPushButton("Open folder")
        self._open_btn.setStyleSheet(_BTN_SECONDARY_STYLE)
        self._open_btn.setToolTip(
            "Reveal the selected model's directory in the OS file "
            "browser."
        )
        if apply_button_icon:
            apply_button_icon(self._open_btn, "mdi6.folder-open-outline")
        self._open_btn.clicked.connect(self._handle_open_clicked)
        btn_row.addWidget(self._open_btn)

        self._init_btn = QPushButton("Set as init source")
        self._init_btn.setStyleSheet(_BTN_SECONDARY_STYLE)
        self._init_btn.setToolTip(
            "Populate Fine-Tuning's Init = 'From prior refinement' "
            "with the selected row's paths so the next training run "
            "uses it as a starting point."
        )
        if apply_button_icon:
            apply_button_icon(self._init_btn, "mdi6.source-branch")
        self._init_btn.clicked.connect(self._handle_init_clicked)
        btn_row.addWidget(self._init_btn)
        # Step 4 has no "use as init": it passes
        # show_init=False and the button is hidden.  Step 3 is unchanged.
        self._init_btn.setVisible(bool(show_init))

        btn_row.addStretch(1)

        self._close_btn = QPushButton("Close")
        self._close_btn.setStyleSheet(_BTN_CANCEL_STYLE)
        self._close_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._close_btn)

        outer.addLayout(btn_row)

        # Initialize button-enabled state (no row selected yet).
        self._refresh_button_enabled_state()

    # ----------------------------------------------------------------
    # Table population
    # ----------------------------------------------------------------

    def _populate_table(self) -> None:
        """Fill table rows from refined bundles + pretrained catalog.

        Renders TWO sections separated by non-selectable header rows:

            [Refined Bundles]          ← non-selectable header
            v1                  Both    246    54    2026-05-21  ...
            v2                  Lateral 120    —     2026-05-20  ...
            [Cellpose Pretrained]      ← non-selectable header
            cyto3               Pretrained                       General epithelial...
            nuclei              Pretrained                       Nuclear staining...

        Each row's payload (kind + dict) is cached in self._row_payloads
        for action-handler dispatch.  Header rows have kind="header"
        and are skipped by _selected_row.
        """
        self._row_payloads = []
        all_rows = []

        # Section 1: Refined bundles
        all_rows.append(("header", {"label": "Refined Bundles"}))
        for row in self._rows:
            all_rows.append(("refined", row))

        # Section 2: Step 2 run models (HITL-trained init sources).
        # 2026-06-08 reader-unify: surfaced here so the user can pick a
        # Step 2 round model for either Use (predict) or Set-as-init
        # (refine) without leaving the picker.  Only rendered when the
        # workspace actually has Step 2 models.
        if self._step2_models:
            all_rows.append(("header", {"label": "Step 2 Run Models"}))
            for entry in self._step2_models:
                all_rows.append(("step2_run", entry))

        # Section 3: Cellpose pretrained
        all_rows.append(("header", {"label": "Cellpose Pretrained"}))
        for name, description in self._pretrained_models:
            all_rows.append((
                "pretrained",
                {"name": name, "description": description},
            ))

        self._row_payloads = all_rows
        self._table.setRowCount(len(all_rows))

        header_bg = QColor("#3a3a3a")
        header_fg = QColor("#cccccc")

        for table_idx, (kind, payload) in enumerate(all_rows):
            if kind == "header":
                # Single bold cell spanning all columns.
                label = payload.get("label", "")
                self._table.setSpan(table_idx, 0, 1, 6)
                item = QTableWidgetItem(label)
                item.setBackground(header_bg)
                item.setForeground(header_fg)
                # Non-selectable + non-editable.
                item.setFlags(Qt.ItemIsEnabled)
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                self._table.setItem(table_idx, 0, item)
                continue

            # Per-cell tooltips, keyed by column index.  Used for the
            # step2_run "Init" cell so the full absolute path is
            # discoverable on hover without crowding the visible cell.
            tooltips: dict = {}

            if kind == "refined":
                cells = [
                    payload.get("name", "?"),
                    payload.get("mode", "?"),
                    (
                        str(payload["lateral_pairs"])
                        if payload.get("lateral_pairs") is not None
                        else "—"
                    ),
                    (
                        str(payload["axial_pairs"])
                        if payload.get("axial_pairs") is not None
                        else "—"
                    ),
                    payload.get("created", "") or "—",
                    payload.get("init_summary", "—") or "—",
                ]
            elif kind == "step2_run":
                # Header columns: Name | Mode | Lateral | Axial | Created | Init
                # Step 2 run models are single-bucket HITL round models —
                # they have no per-bucket pair counts (those are a
                # refined-bundle concept), so Lateral/Axial render "—".
                # 2026-06-24 fix: previously round_id was mis-placed under
                # "Lateral", the source label under "Created", and the raw
                # absolute path under "Init".  Map each value under its
                # correct header + show the FRIENDLY run name (not the raw
                # slug) in Name.
                bucket = (payload.get("bucket", "?") or "?")
                # Friendly run name: display name from the manifest, with a
                # fallback to the first segment of human_label, then slug.
                run_name = (
                    payload.get("run_display_name")
                    or (payload.get("human_label") or "").split(" · ")[0]
                    or payload.get("run_id")
                    or "?"
                )
                round_label = payload.get("round_id", "") or ""
                # Name carries the round suffix for provenance (there is no
                # dedicated "Round" column), e.g. "my run (round_03)".
                name_cell = (
                    f"{run_name} ({round_label})" if round_label else run_name
                )
                src_type = payload.get("source_type", "step2_run")
                init_label = (
                    "Step 2 (legacy)"
                    if src_type == "step2_legacy"
                    else "Step 2 run"
                )
                weights_path = payload.get("weights_path", "") or ""
                # "Created" — the scanner gives an epoch float (mtime);
                # render it as a YYYY-MM-DD date when present.
                created_cell = payload.get("created", "") or ""
                if not created_cell:
                    mtime = payload.get("mtime")
                    if mtime:
                        try:
                            from datetime import datetime as _dt
                            created_cell = _dt.fromtimestamp(
                                float(mtime)
                            ).strftime("%Y-%m-%d")
                        except (ValueError, OSError, OverflowError):
                            created_cell = ""
                cells = [
                    name_cell,            # Name
                    bucket.capitalize(),  # Mode
                    "—",                  # Lateral (no pair count)
                    "—",                  # Axial (no pair count)
                    created_cell or "—",  # Created
                    init_label,           # Init (short label; path in tooltip)
                ]
                if weights_path:
                    tooltips[5] = weights_path
            else:  # pretrained
                cells = [
                    payload.get("name", "?"),
                    "Pretrained",
                    "—", "—", "—",
                    payload.get("description", ""),
                ]

            for j, txt in enumerate(cells):
                item = QTableWidgetItem(str(txt))
                if j == 0:
                    # Stash the table-row index so the selection
                    # lookup can recover (kind, payload) from
                    # self._row_payloads.
                    item.setData(Qt.UserRole, table_idx)
                tip = tooltips.get(j)
                if tip:
                    item.setToolTip(str(tip))
                self._table.setItem(table_idx, j, item)

        n_refined = len(self._rows)
        n_step2 = len(self._step2_models)
        n_pretrained = len(self._pretrained_models)
        print(
            f"[REGISTRY-DLG] populated {n_refined} refined + "
            f"{n_step2} step2-run + {n_pretrained} pretrained row(s)"
        )

    # ----------------------------------------------------------------
    # Selection
    # ----------------------------------------------------------------

    def _selected_row(self) -> Optional[tuple]:
        """Return ``(kind, payload)`` for the selected row, or None.

        Header rows are non-selectable so this never returns a header.
        """
        sel = self._table.selectedItems()
        if not sel:
            return None
        idx_item = self._table.item(sel[0].row(), 0)
        if idx_item is None:
            return None
        idx = idx_item.data(Qt.UserRole)
        if idx is None or not (0 <= idx < len(self._row_payloads)):
            return None
        kind, payload = self._row_payloads[idx]
        if kind == "header":
            return None
        return (kind, payload)

    def _refresh_button_enabled_state(self) -> None:
        """Update which action buttons are enabled.

        Use applies to every selectable row (refined / step2_run /
        pretrained).  Open Folder + Set as Init apply to refined
        bundles AND Step 2 run models (both have an on-disk model dir
        to reveal + a weights path to seed the init combo).  Pretrained
        rows are name-only — no folder, no init.
        """
        sel = self._selected_row()
        if sel is None:
            self._use_btn.setEnabled(False)
            self._open_btn.setEnabled(False)
            self._init_btn.setEnabled(False)
            return
        kind, _payload = sel
        self._use_btn.setEnabled(True)
        has_disk_model = kind in ("refined", "step2_run")
        self._open_btn.setEnabled(has_disk_model)
        self._init_btn.setEnabled(has_disk_model)

    # ----------------------------------------------------------------
    # Action button handlers
    # ----------------------------------------------------------------

    def _handle_use_clicked(self) -> None:
        sel = self._selected_row()
        if sel is None or self._on_use is None:
            return
        kind, payload = sel
        # Tag the payload with its kind so the controller dispatch
        # can branch on it (refined → copy bundle paths; pretrained →
        # set both buckets to the pretrained name).
        payload_tagged = dict(payload)
        payload_tagged["_kind"] = kind
        print(
            f"[REGISTRY-DLG] use clicked: kind={kind} "
            f"name={payload.get('name')}"
        )
        try:
            self._on_use(payload_tagged)
        except Exception as exc:
            print(f"[REGISTRY-DLG] on_use raised: {exc}")

    def _handle_open_clicked(self) -> None:
        sel = self._selected_row()
        # Only rows with an on-disk model dir enable this button —
        # defensive guard (refined bundles + Step 2 run models).
        if sel is None or self._on_open is None:
            return
        kind, payload = sel
        if kind not in ("refined", "step2_run"):
            return
        # Tag the kind so the controller's open handler can resolve the
        # right dir (refined → bundle_dir; step2_run → weights' parent).
        payload_tagged = dict(payload)
        payload_tagged["_kind"] = kind
        print(
            f"[REGISTRY-DLG] open clicked: kind={kind} "
            f"name={payload.get('name') or payload.get('run_id')}"
        )
        try:
            self._on_open(payload_tagged)
        except Exception as exc:
            print(f"[REGISTRY-DLG] on_open raised: {exc}")

    def _handle_init_clicked(self) -> None:
        sel = self._selected_row()
        if sel is None or self._on_init is None:
            return
        kind, payload = sel
        if kind not in ("refined", "step2_run"):
            return
        # Tag the kind so the controller's init handler can route a
        # refined bundle vs a single-bucket Step 2 run model.
        payload_tagged = dict(payload)
        payload_tagged["_kind"] = kind
        print(
            f"[REGISTRY-DLG] init clicked: kind={kind} "
            f"name={payload.get('name') or payload.get('run_id')}"
        )
        try:
            self._on_init(payload_tagged)
        except Exception as exc:
            print(f"[REGISTRY-DLG] on_init raised: {exc}")

    # ----------------------------------------------------------------
    # External-path handlers (Stage 3)
    # ----------------------------------------------------------------

    def _handle_ext_browse_clicked(self, bucket: str) -> None:
        """Open file picker for an external model path.  Bucket
        is "lateral" or "axial".  Selected path lands in the
        corresponding LineEdit (user still needs to click Apply)."""
        # 2026-05-21 audit fix: Cellpose models are saved WITHOUT a
        # standard extension (e.g. `lateral`, `axial`, custom names) —
        # use "All files" filter so users aren't blocked by an
        # extension-based filter that excludes the actual model files.
        path, _filter = QFileDialog.getOpenFileName(
            self,
            f"Select external {bucket} model file",
            "",
            "All files (*)",
        )
        if not path:
            return
        if bucket == "lateral":
            self._ext_lat_path.setText(path)
        elif bucket == "axial":
            self._ext_ax_path.setText(path)
        print(f"[REGISTRY-DLG] external {bucket} browsed: {path}")

    def _handle_ext_apply_clicked(self) -> None:
        """Apply the external paths as the active model.

        Emits an external-flavored payload to the on_use callback
        with `_kind="external"` so the controller can dispatch
        appropriately (set paths as-is, persist mode=import).

        Either or both paths may be empty — the controller's handler
        treats empty as "leave unchanged".
        """
        lat = self._ext_lat_path.text().strip()
        ax = self._ext_ax_path.text().strip()
        if not lat and not ax:
            print(
                "[REGISTRY-DLG] external apply: both paths empty, skipping"
            )
            return
        if self._on_use is None:
            return
        payload = {
            "_kind": "external",
            "name": "external paths",
            "lateral_weights": lat or None,
            "axial_weights": ax or None,
        }
        print(
            f"[REGISTRY-DLG] external apply: lat={lat!r} ax={ax!r}"
        )
        try:
            self._on_use(payload)
        except Exception as exc:
            print(f"[REGISTRY-DLG] external on_use raised: {exc}")

    # ----------------------------------------------------------------
    # Convenience launcher
    # ----------------------------------------------------------------

    @classmethod
    def show_dialog(
        cls,
        parent: Optional[QWidget] = None,
        *,
        rows: Optional[List[dict]] = None,
        on_use: Optional[Callable[[dict], None]] = None,
        on_open: Optional[Callable[[dict], None]] = None,
        on_init: Optional[Callable[[dict], None]] = None,
        pretrained_models: Optional[List[tuple]] = None,
        step2_models: Optional[List[dict]] = None,
        show_init: bool = True,
    ) -> None:
        """Open the dialog modally.  No return value — the dialog's
        actions trigger the callbacks while it's open.

        ``pretrained_models`` defaults to module-level
        ``PRETRAINED_MODELS``.  Override for tests or to inject a
        custom catalog.

        ``step2_models`` is the list of Step 2 run-model init sources
        (from the controller's ``_list_all_model_sources`` scanner,
        filtered to ``source_type ∈ {"step2_run", "step2_legacy"}``).
        When non-empty it renders a "Step 2 Run Models" section.
        """
        dlg = cls(
            parent=parent,
            rows=rows,
            on_use=on_use,
            on_open=on_open,
            on_init=on_init,
            pretrained_models=pretrained_models,
            step2_models=step2_models,
            show_init=show_init,
        )
        dlg.exec_()
        print("[REGISTRY-DLG] closed")
