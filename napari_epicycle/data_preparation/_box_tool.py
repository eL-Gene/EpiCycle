"""Step 1 ③ box tool — EpiCycle-Viz-style live 3D box (v2, crash-safe render).

This module owns the ③ ROI boxes: their creation (Add / click-to-place),
selection, live geometry editing, list ordering, and 3D rendering.

**Render (v2):** the boxes are drawn by a crash-safe :class:`Step1BoxOverlay` —
raw vispy ``Line`` visuals parented to the shared main canvas — NOT a napari
Shapes layer.  A Shapes-in-3D layer's mesh VBO can be freed while a draw is still
queued on step-exit / teardown, faulting the 3D canvas with a ``glDrawArrays``
access violation; a vispy ``Line`` owns its own lifecycle and never enters
``viewer.layers``, so that access-violation class is gone by construction (this
is the reason for v2 — see ``_box_overlay.py`` and the de-novo Add tool's
identical idiom).

Coordinate model (decoupled two-resolution — Viz's decoupling, now PHYSICAL µm):

* Boxes live in **full-res voxel coords** (a :class:`roi_box.BoxRec`).
* The **display** world is PHYSICAL µm: ``world == full_res_voxel × voxel_size``.
  Every display layer sits at ``scale = factor × voxel_size`` — routine data
  reuses the loaded full-res channel layers at ``factor = 1`` (so ``scale ==
  voxel_size``); a genuinely large source gets a **downsampled** overview added
  at ``scale = downsample-factor × voxel_size``.  ``voxel_size`` is the
  anisotropic ``(vz, vy, vx)``; ``(1, 1, 1)`` when unknown (→ world == voxel
  index, byte-identical to the pre-physical display).
* The overlay multiplies its full-res-voxel coords by ``voxel_size`` (via
  :meth:`Step1BoxOverlay.set_voxel_size`) — never additionally by ``factor`` —
  then applies the ZYX→XYZ reversal, so it stays glued in every view.
* A click's ``cursor.position`` (and the 2D ``camera.center``) is now PHYSICAL µm,
  so placement DIVIDES the Y/X by ``voxel_size`` to reach the voxel indices the
  box stores.  The voxel is read LIVE from the controller (``_preview_spacing``),
  so a voxel-spin edit is picked up with no cache to re-seed.

Extraction is untouched: it reads full-res voxel coords (never world) and stays
byte-identical — a unit voxel keeps the whole pipeline pixel-for-pixel as before.

**Z is editable:** a box DEFAULTS to full Z depth (``cz = Z // 2``,
``sz = Z``), so an untouched box extracts byte-identically to before; the now-
editable Center-Z / Size-D spins narrow it to a sub-slab, which extraction crops
(``z_ranges``).  ④ then refines the Z-truncate WITHIN the box's slab.

**Placement is a 2D operation (v2, feedback #2/#3):** Add drops the box at the
current 2D view centre (``viewer.camera.center`` while ``ndisplay == 2`` tracks
pan + zoom); when Add is pressed in 3D it uses the LAST 2D centre.  A faint ghost
box previews the next landing spot live as the view moves.

**Size (v2, feedback #5/#6):** the next box's size is sticky — either an S/M/L
field-fraction (``set_size_fraction``) or an absolute voxel cube
(``set_size_voxels``, from the 128³/256³/512³ + custom presets); the last one
chosen wins.  Size changes never crop an existing box's Z.

Teardown destroys the overlay on stop (toggle-off / step-exit) AND reset
(source-unload), plus a ``QApplication.aboutToQuit`` hook, so no visual is ever
left dangling on the shared canvas.

The pure geometry (:func:`full_shape_from_source`, :func:`place_box`,
:func:`rect_for_box`, :func:`redraw_paths`) is Qt/napari-free and unit-tested;
the :class:`Step1BoxTool` class wires it to the viewer + the ③ geometry widgets.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from packages.epicycle_annotator.roi_box import (
    BoxRec,
    box_corners,
    box_edges,
)

OVERVIEW_LAYER_PREFIX = "Step 1 overview"

# Box outline colours (RGBA 0..1).  Active box = amber (Step 1's `#d4a843`),
# idle = teal (`_STEP1_ACCENT` = `#5ba4cf`).
_IDLE_RGBA = (0.357, 0.643, 0.812, 0.9)
_ACTIVE_RGBA = (0.831, 0.659, 0.263, 1.0)
_IDLE_WIDTH = 2.0
_ACTIVE_WIDTH = 4.0

# ④'s "colour" combo choices -> the RGB the box overlay draws (None = the
# default status colours: idle teal, active amber).  Kept here (not in ui.py) so
# the pure box tool owns the name->colour mapping and it is unit-testable.
_BOX_COLOR_CHOICES = {
    "status colour": None,
    "teal": (0.357, 0.643, 0.812),
    "amber": (0.831, 0.659, 0.263),
    "green": (0.361, 0.780, 0.420),
    "magenta": (0.850, 0.318, 0.850),
    "white": (0.950, 0.950, 0.970),
}

# Downsampled-overview target: cap the largest XY dim of the display copy so a
# genuinely large source renders in 3D without blowing VRAM (Viz keeps Z at ×1 —
# light-sheet/confocal Z is sparse and striding it wrecks reconstruction).
_OVERVIEW_MAX_XY = 512


# ---------------------------------------------------------------------------
# Pure geometry (Qt/napari-free — unit-tested)
# ---------------------------------------------------------------------------

def full_shape_from_source(shape, order: Optional[str]):
    """The full-res ``(Z, Y, X)`` spatial shape of a source.

    *shape* is the on-disk array shape (3D ``ZYX`` or 4D ``CZYX`` / ``ZCYX``);
    *order* is the resolved axis order (``_source_axes_resolved``).  A 3D shape
    is ``ZYX``; a 4D shape drops the channel axis per *order* (``ZCYX`` → axis
    0 is Z; ``CZYX`` / anything else → axis 0 is C).  Returns ``None`` for an
    unusable shape.
    """
    if shape is None:
        return None
    shape = tuple(int(s) for s in shape)
    if len(shape) == 3:
        return shape
    if len(shape) == 4:
        if order == "ZCYX":
            return (shape[0], shape[2], shape[3])
        return (shape[1], shape[2], shape[3])  # CZYX (channel-first) default
    return None


def place_box(full_shape, cy, cx, sy, sx, cz=None, sz=None) -> BoxRec:
    """A box centred at ``(cy, cx)`` with XY size ``(sy, sx)``.

    Z DEFAULTS to the full depth (``cz = Z // 2``, ``sz = Z``) so an untouched
    box spans the whole column and extracts byte-identically to before.  Passing
    ``cz`` / ``sz`` (from the now-editable Center-Z / Size-D spins) narrows it to
    a Z sub-slab.
    """
    Z = int(full_shape[0])
    if cz is None:
        cz = Z // 2
    if sz is None:
        sz = Z
    return BoxRec(
        cz=int(cz),
        cy=int(cy),
        cx=int(cx),
        sz=max(1, int(sz)),
        sy=max(1, int(sy)),
        sx=max(1, int(sx)),
    )


def centered_box(full_shape) -> BoxRec:
    """A full-Z box at the volume centre, XY size ≈ ¼ of the field (Viz's
    "so the first Add box is visible, not huge")."""
    Z, Y, X = (int(s) for s in full_shape)
    return place_box(full_shape, Y // 2, X // 2, max(1, Y // 4), max(1, X // 4))


def rect_for_box(box: BoxRec, full_shape) -> np.ndarray:
    """The box's clamped XY bounds as a ``(4, 2)`` ``(Y, X)`` rectangle.

    This is the exact format ``get_drawn_shapes`` returns and
    ``extract_patches_from_shapes`` consumes, so a box feeds the unchanged
    extract path (full-Z is supplied by extraction, which crops all Z).
    """
    b = box.bounds_dict(full_shape)
    return np.array(
        [[b["y0"], b["x0"]], [b["y0"], b["x1"]],
         [b["y1"], b["x1"]], [b["y1"], b["x0"]]],
        dtype=float,
    )


def redraw_paths(boxes, active_index, full_shape, style: str = "corners"):
    """Per-shape ``(data, edge_colors, widths)`` for a Shapes ``path`` layer.

    Rebuilds the outline of EVERY box (corner brackets or full wireframe) as
    ``(2, 3)`` ZYX segments — the active box brighter + thicker.  Mirrors Viz's
    ``style_arrays`` but outline-only (no fill in Phase 1).  Empty inputs give
    three empty lists, which :meth:`Step1BoxTool.redraw` renders as a cleared
    layer.
    """
    data: list = []
    ecolors: list = []
    widths: list = []
    if full_shape is None:
        return data, ecolors, widths
    for i, box in enumerate(boxes):
        b = box.bounds_dict(full_shape)
        segs = box_corners(b) if style == "corners" else box_edges(b)
        active = i == active_index
        rgba = _ACTIVE_RGBA if active else _IDLE_RGBA
        width = _ACTIVE_WIDTH if active else _IDLE_WIDTH
        for seg in segs:
            data.append(seg)
            ecolors.append(rgba)
            widths.append(width)
    return data, ecolors, widths


def _is_unit_voxel(vox) -> bool:
    """True for the ``(1, 1, 1)`` isotropic placeholder (== "unknown", not a real
    physical spacing).  Guarded — any bad input reads as unit (safe: no µm claim)."""
    try:
        return all(abs(float(v) - 1.0) < 1e-9 for v in vox)
    except (TypeError, ValueError):
        return True


def _units_for_voxel(vox):
    """The napari ``layer.units`` for a resolved voxel: ``("um",)*3`` (micrometer)
    when the spacing is a real physical value, else ``("pixel",)*3`` (unknown
    ``(1, 1, 1)``).

    A deliberate ``(1, 1, 1)`` is always typed "unknown" upstream (never "user"),
    so this value test is equivalent to keying on ``voxel_size_source`` — and it
    needs no source lookup.

    Use the token napari itself defaults to for an un-set layer — **``"pixel"``,
    NOT ``"px"``**.  pint parses ``"px"`` to ``css_pixel`` (a physical 1/96-inch
    unit), which MISMATCHES napari's plain ``pixel`` on any un-set layer and trips
    the ``"Inconsistent units across layers: units will not be used for
    rendering"`` warning → the scale bar silently falls back to pixels.  ``"um"``
    parses to micrometer (pint's ``u`` = micro), which the scale bar renders as
    ``µm``."""
    return ("pixel", "pixel", "pixel") if _is_unit_voxel(vox) else ("um", "um", "um")


def _apply_layer_scale(layers, scale) -> None:
    """Set ``layer.scale`` on one layer or a list AFTER add (best-effort + guarded).

    Setting scale post-add (rather than as an ``add_image`` kwarg) is uniform
    across every native call site and tolerant of a fake/older viewer whose
    ``add_image`` signature doesn't take ``scale`` — the value lands on the real
    layer either way and a fake that rejects the attribute simply no-ops.  It
    also avoids coupling a ``scale``-kwarg TypeError-retry to the other display
    kwargs (contrast/gamma/visible), which must be preserved.  Never raises."""
    seq = layers if isinstance(layers, (list, tuple)) else [layers]
    sc = tuple(scale)
    for layer in seq:
        if layer is None:
            continue
        try:
            layer.scale = sc
        except Exception:
            pass


def _apply_layer_units(layers, units) -> None:
    """Set ``layer.units`` on one layer or a list (best-effort + guarded).

    napari 0.8 accepts ``units`` as an ``add_image`` kwarg, but setting it AFTER
    add is uniform across every call site and tolerant of a fake/older viewer
    (a mismatched ndim or a viewer without ``units`` simply no-ops).  Never
    raises — a units label is cosmetic, never worth a crash out of a Qt slot."""
    seq = layers if isinstance(layers, (list, tuple)) else [layers]
    for layer in seq:
        if layer is None:
            continue
        try:
            layer.units = tuple(units)
        except Exception:
            pass


def overview_factors(full_shape, max_xy: int = _OVERVIEW_MAX_XY):
    """Per-axis ``(fz, fy, fx)`` downsample for the large-source 3D overview.

    Z is kept at ×1 (Viz default); Y/X are strided so the largest of the two is
    ≤ *max_xy*.  ``(1, 1, 1)`` when the source already fits.
    """
    _, Y, X = (int(s) for s in full_shape)
    biggest = max(Y, X)
    if biggest <= max_xy:
        return (1, 1, 1)
    f = int(np.ceil(biggest / max_xy))
    return (1, f, f)


# ---------------------------------------------------------------------------
# The viewer-wired box tool
# ---------------------------------------------------------------------------

class Step1BoxTool:
    """Owns the live-3D box: the Shapes layer, ndisplay, click-to-place, Add,
    the box list, and the live geometry-slider editing.

    Attached to the ③ controller (``controller.box_tool``); reads the viewer +
    the L0 lazy source off it and drives the ③ geometry widgets on the widget.
    Everything here is best-effort + guarded — a torn-down widget or a missing
    viewer must never raise out of a Qt slot.
    """

    def __init__(self, controller):
        self._ctl = controller
        self.viewer = getattr(controller, "viewer", None)
        self.widget = getattr(controller, "widget", None)

        self.boxes: list[BoxRec] = []
        # Per-box include/exclude tick, parallel to `boxes`.  The
        # box tool OWNS ③'s list, so this — not a per-rebuild re-tick — is the
        # source of truth, which is what preserves an untick across rebuilds.
        self.box_checked: list[bool] = []
        # Per-box lock, parallel to `boxes` (2026-07-27).  A locked box has its
        # six geometry spins disabled and `on_props_change` no-ops, so it cannot
        # be moved/resized until unlocked.  A new box is born unlocked; the flag
        # rides its box through reorder (permuted, not index-keyed).
        self.box_locked: list[bool] = []
        self.active_index: Optional[int] = None
        self._style = "corners"
        # ④'s Render/style controls drive these LIVE-overlay appearance params
        # (thickness / opacity / colour).  Stored on the tool so a chosen look
        # survives box-mode toggles (overlay recreation) and is re-pushed in
        # `_ensure_overlay`.  Defaults MATCH ④'s widget defaults: thickness 2,
        # opacity 100 %, "status colour" (idle teal / active amber).
        self._style_thickness = 2
        self._style_opacity = 1.0
        self._style_color = None
        # Sticky size the next Add / click-to-place reuses.  Two models, last one
        # chosen wins (`_last_size_mode`): a field FRACTION (S/M/L — feedback #5,
        # default 1/4 = the pre-feature size) or an ABSOLUTE voxel cube (the
        # 128³/256³/512³ + custom presets — feedback #6).
        self._last_size_frac = 0.25
        self._last_abs_size: Optional[int] = None
        self._last_size_mode = "frac"
        # At-edge behavior for LIVE geometry edits (③ box-editor toggle).
        #   "shrink" (default): collapse an edge-overflowing box onto its
        #       rendered (truncated) bounds — extraction byte-identical, the
        #       geometry handle never lags the drawn box (`shrunk_to_bounds`).
        #   "keep": preserve the requested size and slide the box inward off the
        #       edge — deliberately changes the edge extraction (`slid_inward`).
        # Per-session (no schema change); the toggle calls `set_edge_mode`.
        self._edge_mode = "shrink"

        # v2 crash-safe render: a vispy `Line` overlay on the shared canvas, NOT
        # a napari Shapes layer (the freed-VBO access-violation fix).
        self._overlay = None
        # The last 2D viewport centre (voxel), so an Add pressed in 3D still
        # places where the user last had the 2D view — placement stays a 2D op.
        self._last_2d_center: Optional[tuple] = None
        self._overview_layers: list = []
        self._mouse_cb = None
        self._prev_ndisplay: Optional[int] = None
        self._full_shape: Optional[tuple] = None

        self._active = False            # box mode on/off
        self._loading = False           # guard programmatic spin writes (Viz `_load`)
        self._geom_connected = False
        self._list_connected = False
        self._list_rebuilding = False
        self._camera_connected = False  # camera events drive the ghost preview
        self._close_hook_connected = False  # aboutToQuit -> stop() (once)

    # -- tiny helpers --------------------------------------------------------

    def _log(self, msg: str) -> None:
        log = getattr(self._ctl, "_log", None)
        if callable(log):
            try:
                log("STEP1-BOX", msg)
            except Exception:
                pass

    def is_active(self) -> bool:
        return self._active

    def has_boxes(self) -> bool:
        return bool(self.boxes)

    # -- shape / coordinate resolution --------------------------------------

    def _resolve_full_shape(self) -> Optional[tuple]:
        lazy = getattr(self._ctl, "_source_lazy", None)
        if lazy is not None and getattr(lazy, "ndim", None) in (3, 4):
            fs = full_shape_from_source(
                lazy.shape, getattr(self._ctl, "_source_axes_resolved", None))
            if fs is not None:
                return fs
        # Fall back to a loaded 3D display layer (per-channel ZYX).
        for name in getattr(self._ctl, "_loaded_source_layers", None) or []:
            layer = self._layer(name)
            data = getattr(layer, "data", None)
            if getattr(data, "ndim", None) == 3:
                return tuple(int(s) for s in data.shape)
        return None

    def _layer(self, name):
        viewer = self.viewer
        if viewer is None:
            return None
        try:
            layers = viewer.layers
            return layers[name] if name in layers else None
        except Exception:
            return None

    def _display_layers_present(self) -> bool:
        # Raw source layers on the canvas → the source is displayed.
        names = getattr(self._ctl, "_loaded_source_layers", None) or []
        if any(self._layer(n) is not None for n in names):
            return True
        # …or the PROCESSED view is the current display: the clean swap removed
        # the raw layers, but the ⟨processed⟩ layers ARE the source display, so no
        # downsampled overview is needed.  Without this, entering box mode while
        # viewing Processed stacks a redundant overview on top → the "4 layers"
        # bug (2 processed + 2 overview).
        is_proc = getattr(self._ctl, "_is_processed_view_showing", None)
        if callable(is_proc):
            try:
                return bool(is_proc())
            except Exception:
                return False
        return False

    # -- box mode lifecycle --------------------------------------------------

    def start(self) -> bool:
        """Enter box mode.  Returns True on success.  Idempotent — a second
        call while already active is a no-op (prevents double overview / double
        connects).  Does NOT force ndisplay — the box renders in the current
        view (2D or 3D), like the ② Preview box."""
        if self.viewer is None:
            return False
        if self._active:
            return True
        fs = self._resolve_full_shape()
        if fs is None:
            self._log("no source loaded — load a source in Load & Adjust first")
            return False
        self._full_shape = fs
        # Lift the geometry spin/slider ceilings off the true source shape so a
        # source larger than the 4096 build-time default is not silently cropped
        # on a geometry edit.  MUST precede `_connect_geometry` below so no
        # spurious `on_props_change` fires while the ranges settle.
        self._apply_geometry_spin_maxima()

        # Snapshot ndisplay ONCE so re-entry can't lose the user's mode.
        if self._prev_ndisplay is None:
            try:
                self._prev_ndisplay = int(self.viewer.dims.ndisplay)
            except Exception:
                self._prev_ndisplay = 2

        # Large source with no in-RAM display -> build a downsampled overview.
        if not self._display_layers_present():
            self._build_overview(fs)

        self._ensure_overlay()
        # NOTE: box mode no longer FORCES 3D — it renders in whatever view is
        # current (like the ② Preview box), so placement stays in the stable,
        # world-fixed 2D view instead of engaging the progressive-loading engine
        # (which re-derives the image's world placement per detail level on
        # zoom-out, making the box appear to drift).  The `_prev_ndisplay`
        # snapshot above + the `stop()` restore are now genuine no-ops (kept for
        # minimal, reversible diff).  "2D ⇄ 3D" (`toggle_ndisplay`) is the opt-in
        # 3D inspect; "⌖ Click-to-place" forces 2D via `on_place_toggled`.
        self._install_mouse_cb()
        self._connect_geometry()
        self._connect_list()
        self._own_list()
        self._connect_camera()
        self._connect_close_hook()
        self._active = True
        self._sync_mode_button(True)
        self.refresh_list()
        # Re-entry with boxes already present: re-hydrate the geometry editor
        # for the active box (else the sliders stay disabled until a row click).
        if self.active_index is not None and 0 <= self.active_index < len(self.boxes):
            self._load_geometry_into_spins(self.boxes[self.active_index])
            self._set_geometry_enabled(True)
        self.redraw()
        self._log(f"box mode ON  full_shape(Z,Y,X)={fs}")
        return True

    def stop(self, restore_ndisplay: bool = True,
             keep_visible: bool = False) -> None:
        """Leave box mode: tear down the EDITING surface (ghost + mouse / geometry
        / list / camera), keep the boxes (they are the extract source of truth).
        Idempotent + guarded.

        ``keep_visible`` (the ③ "📦 Box mode" toggle-OFF): KEEP the placed boxes
        on the canvas AND still selectable / movable — the navigator row-select +
        the geometry sliders stay connected (they don't gate on `_active`), so a
        box can be highlighted + nudged with box mode off.  Only the ghost
        placement-guide (+ its camera driver) and click-to-place-NEW go away.  The
        overlay stays alive (it follows the view on its own via the shared scene
        transform + its dims subscription), so the boxes track pan / zoom /
        rotate, and box-mode re-entry reuses it.  The source-lifecycle callers
        (step-exit, unload / reset, app-quit) leave it FALSE → the overlay,
        overview, and geometry / list connections are DESTROYED so nothing strands
        on another step's shared canvas.

        Each teardown step is itself guarded, but the ``ndisplay`` restore runs
        in a ``finally`` so a surprise raise anywhere above can never strand the
        canvas in 3D.
        """
        keep_editable = bool(
            keep_visible and self._overlay is not None and self.boxes)
        try:
            self._disconnect_camera()      # stop the ghost driver (both paths)
            self._remove_mouse_cb()        # stop click-to-place-NEW (both paths)
            if keep_editable:
                # Box mode OFF but keep the boxes on the canvas AND still
                # selectable / movable: the geometry sliders + navigator
                # row-select stay CONNECTED (they don't gate on `_active`), so a
                # box can be highlighted + nudged with box mode off.  Render with
                # the active-box highlight preserved; drop only the ghost.  The
                # overlay + any overview stay; re-entry reuses the overlay and
                # `_build_overview` self-guards against a double.
                try:
                    self._apply_overlay_voxel()   # physical-µm scale (live read)
                    self._overlay.update(
                        self.boxes, self.active_index, self._full_shape)
                    self._overlay.clear_ghost()
                except Exception as exc:
                    self._log(f"keep-visible redraw failed: {exc}")
                self._log(
                    f"box mode OFF (kept {len(self.boxes)} box(es) editable)")
            else:
                self._disconnect_geometry()
                self._disconnect_list()
                self._remove_overlay()
                self._remove_overview_layers()
        finally:
            if restore_ndisplay and self._prev_ndisplay is not None:
                self._set_ndisplay(self._prev_ndisplay)
            self._prev_ndisplay = None
            self._active = False
            if keep_editable and self.active_index is not None \
                    and 0 <= self.active_index < len(self.boxes):
                # Keep the geometry editor live for the current selection so the
                # sliders move the selected box with box mode off.
                locked = self._is_locked(self.active_index)
                self._set_geometry_enabled(not locked)
                self._sync_lock_checkbox(locked, has_selection=True)
            else:
                # No selection (or full teardown) -> no box to lock; reset the
                # checkbox so it never lingers enabled while the editor is dead.
                self._set_geometry_enabled(False)
                self._sync_lock_checkbox(False, has_selection=False)
            self._sync_mode_button(False)

    def reset(self) -> None:
        """Full teardown for source-unload: drop the boxes too."""
        self.stop()
        self.boxes = []
        self.box_checked = []
        self.box_locked = []
        self.active_index = None
        self._full_shape = None
        self._sync_lock_checkbox(False, has_selection=False)
        try:
            self.refresh_list()
        except Exception:
            pass

    def clear_all(self) -> None:
        """Remove every box but STAY in box mode (the ③ "Clear" action).

        Clears through :meth:`redraw` (``layer.data = []``), not a bare setter
        on live box data, so no degenerate polygons are left behind.
        """
        self.boxes = []
        self.box_checked = []
        self.box_locked = []
        self.active_index = None
        self._set_geometry_enabled(False)
        self._sync_lock_checkbox(False, has_selection=False)
        try:
            self.refresh_list()
            self.redraw()
        except Exception:
            pass

    def remove_boxes(self, indices) -> int:
        """Remove the boxes at ``indices`` (e.g. the ones just extracted) while
        STAYING in box mode; un-listed boxes (un-ticked / kept for later) are
        preserved.  Pops in DESCENDING order so surviving indices don't shift
        mid-loop, then refreshes the list + overlay once.  Returns the count
        removed.  Mirrors :meth:`delete_active`'s post-mutation resync."""
        idx = sorted({int(i) for i in indices
                      if 0 <= int(i) < len(self.boxes)}, reverse=True)
        if not idx:
            return 0
        for i in idx:
            self.boxes.pop(i)
            if i < len(self.box_checked):
                self.box_checked.pop(i)
            if i < len(self.box_locked):
                self.box_locked.pop(i)
        self.active_index = (len(self.boxes) - 1) if self.boxes else None
        if self.active_index is None:
            self._set_geometry_enabled(False)
            self._sync_lock_checkbox(False, has_selection=False)
        else:
            self._load_geometry_into_spins(self.boxes[self.active_index])
            locked = self._is_locked(self.active_index)
            self._set_geometry_enabled(not locked)
            self._sync_lock_checkbox(locked)
        try:
            self.refresh_list(select=self.active_index)
            self.redraw()
        except Exception:
            pass
        return len(idx)

    def load_committed_boxes(self, boxrecs) -> int:
        """Seed the tool with ALREADY-EXPORTED boxes as COMMITTED rows (F3).

        Called by the controller's ``_restore_bounding_boxes`` to restore a
        source's previously-exported patches into the CRASH-SAFE vispy overlay
        instead of the legacy napari Shapes layer.  A resident Shapes-in-3D
        layer's mesh VBO can be freed while a draw is queued during a
        Raw⇄Processed view swap → a native ``glDrawArrays`` access-violation
        crash; the vispy ``Line`` overlay never enters ``viewer.layers`` and so
        removes that crash class by construction.

        Each restored box is marked COMMITTED via ``box_locked=True`` — there is
        no dedicated exported flag, so the lock flag doubles as the "exported /
        committed" visual (a committed box is frozen from geometry edits) — and
        ``box_checked=True`` so it reads as included (extraction still dedups it
        via ``_committed_box_sigs``).

        The controller builds each ``BoxRec`` from the sidecar so the restored
        box's Z is the TRIMMED export Z (from ``selected_z_range``); the
        restored ROI is byte-identical to what was extracted, so a re-extract of
        it is deduped.

        **Idempotent:** a re-restore of the same source REPLACES the previously
        loaded committed (locked) rows rather than accumulating duplicates.  In
        the real flow the tool is already reset before restore
        (``load_source`` → ``unload_source`` → ``reset``), so the drop only
        guards a stray double call; any user-drawn (unlocked) box is preserved.

        Returns the number of committed boxes now loaded.
        """
        boxrecs = [b for b in (boxrecs or []) if b is not None]
        # Resolve the full (Z, Y, X) so `refresh_list`'s per-row dim text and a
        # later `redraw` / `rectangles` / `z_ranges` all read the same clamped
        # bounds.  A missing shape (no source handle yet) leaves rendering to the
        # next `start()` — the rows still populate the list.
        if self._full_shape is None:
            self._full_shape = self._resolve_full_shape()
        # Idempotent replace: drop the previously-restored committed rows first.
        self._drop_committed_boxes()
        for box in boxrecs:
            self.boxes.append(box)
            self.box_checked.append(True)
            self.box_locked.append(True)   # committed → frozen + exported visual
        self._sync_box_checked_length()
        # A restore does not force-select a box; keep any existing selection sane.
        if self.active_index is not None and not (
                0 <= self.active_index < len(self.boxes)):
            self.active_index = None
        # Take over ③'s Box List so the restored rows render as the box tool's
        # single native checkbox per row (matches box mode), NOT the panel's
        # row-card overlay — which assumes ③ is box-tool-owned (`refresh`'s
        # `_suppress_row_widgets` short-circuit).  Idempotent + re-run by
        # `start()`.
        self._own_list()
        try:
            self.refresh_list(select=self.active_index)
        except Exception:
            pass
        # Render the restored boxes on the canvas NOW — box mode OFF but VISIBLE
        # (the persist-boxes off-state), matching the old Shapes-layer behaviour
        # of showing exported boxes the moment the source loads.  Build the
        # CRASH-SAFE vispy overlay (never a napari layer, so it can't reintroduce
        # the freed-VBO Raw⇄Processed crash) + wire the row-select/geometry
        # editors (idempotent — `start()` reuses them; no ghost / no click-to-
        # place, which are box-mode-ON only).  Guarded: needs a resolved shape +
        # a live viewer; otherwise the next `start()` renders them.
        rendered = False
        _ov_state = "skip"
        if boxrecs and self._full_shape is not None and self.viewer is not None:
            try:
                ov = self._ensure_overlay()
                _ov_state = "live" if ov is not None else "None"
                if ov is not None:
                    self._connect_geometry()
                    self._connect_list()
                    self.redraw()
                    rendered = True
            except Exception as exc:
                _ov_state = f"err:{exc}"
                self._log(f"restore render failed: {exc}")
        # TEMP RESTORE-DIAG (strip before baseline): pins why restored boxes may
        # not paint — no boxrecs / unresolved shape / no viewer / overlay build
        # failed / rendered-but-invisible (coords).
        self._log(
            f"[RESTORE-DIAG] n_boxrecs={len(boxrecs)} "
            f"full_shape={self._full_shape} viewer={self.viewer is not None} "
            f"overlay={_ov_state} rendered={rendered}")
        if not rendered:
            # No overlay yet -> a no-op; `start()` re-renders `self.boxes` on entry.
            self.redraw()
        self._log(f"restored {len(boxrecs)} committed box(es) into the box tool")
        return len(boxrecs)

    def _drop_committed_boxes(self) -> None:
        """Remove the boxes previously loaded as COMMITTED (locked) — the
        idempotent-restore path.  Keeps any user-drawn (unlocked) box and
        re-maps the active index onto the survivors."""
        self._sync_box_checked_length()
        keep = [i for i in range(len(self.boxes)) if not self._is_locked(i)]
        if len(keep) == len(self.boxes):
            return  # nothing committed to drop
        active = self.active_index
        self.boxes = [self.boxes[i] for i in keep]
        self.box_checked = [self.box_checked[i] for i in keep]
        self.box_locked = [self.box_locked[i] for i in keep]
        if active is None:
            self.active_index = None
        else:
            self.active_index = keep.index(active) if active in keep else None

    def toggle_ndisplay(self) -> None:
        """The 2D⇄3D affordance: flip between 2D (click-to-place) and 3D
        (view) WITHOUT tearing down the box layer/overview."""
        if not self._active or self.viewer is None:
            return
        try:
            cur = int(self.viewer.dims.ndisplay)
        except Exception:
            return
        self._set_ndisplay(2 if cur == 3 else 3)

    def _set_ndisplay(self, nd: int) -> None:
        if self.viewer is None:
            return
        try:
            self.viewer.dims.ndisplay = int(nd)
        except Exception as exc:
            self._log(f"could not set ndisplay={nd}: {exc}")

    # -- overview (large-source only) ---------------------------------------

    def _build_overview(self, full_shape) -> None:
        """Add a downsampled 3D overview from the lazy source (large-data path).

        Only reached when there are NO in-RAM display layers to reuse, so this
        never touches the user's existing layers.  Uses L0's label-safe strided
        downsample; the overview world is PHYSICAL µm — ``scale = factor ×
        voxel_size`` per axis (``factor`` the strided downsample, ``voxel_size``
        the live raw spacing), so it stays glued to the physical-scaled box
        overlay + native layers.
        """
        lazy = getattr(self._ctl, "_source_lazy", None)
        if lazy is None or self.viewer is None:
            return
        # Never stack a second overview on top of a prior one (defensive: `start`
        # is idempotent, but a stale overview must not leak either way).
        self._remove_overview_layers()
        try:
            from ._largedata import downsample_for_display

            order = getattr(self._ctl, "_source_axes_resolved", None)
            arr = lazy
            ndim = getattr(arr, "ndim", None)
            if ndim == 4 and order == "ZCYX":
                # Reorder to CZYX (lazy view) so the channel axis leads.
                mv = np.moveaxis
                if hasattr(arr, "compute"):
                    import dask.array as da
                    mv = da.moveaxis
                arr = mv(arr, 1, 0)
            factors = overview_factors(full_shape)
            disp, ds = downsample_for_display(arr, factors)
            # Physical-µm scale: factor × voxel per (Z, Y, X) axis.  A unit voxel
            # → scale == factor (byte-identical to the pre-physical overview).
            vox = self._ctl_voxel()
            scale = tuple(float(f) * float(v) for f, v in zip(ds, vox))
            units = _units_for_voxel(vox)
            if disp.ndim == 4:
                names = [f"{OVERVIEW_LAYER_PREFIX} ch{c}"
                         for c in range(disp.shape[0])]
                layer = self.viewer.add_image(
                    disp, channel_axis=0, name=names, scale=scale,
                    blending="additive")
                added = layer if isinstance(layer, list) else [layer]
            else:
                added = [self.viewer.add_image(
                    disp, name=f"{OVERVIEW_LAYER_PREFIX} ch0", scale=scale,
                    blending="additive")]
            self._overview_layers = [l for l in added if l is not None]
            _apply_layer_units(self._overview_layers, units)
            self._log(f"built downsampled overview {tuple(disp.shape)} "
                      f"(factor={ds})")
        except Exception as exc:
            self._log(f"overview build failed (rendering full-res): {exc}")

    def _remove_overview_layers(self) -> None:
        # Remove by NAME so a stale object-ref can never strand an overview
        # layer on the shared canvas (a leaked overview pollutes other steps).
        viewer = self.viewer
        if viewer is not None:
            try:
                for layer in list(viewer.layers):
                    name = str(getattr(layer, "name", ""))
                    if name.startswith(OVERVIEW_LAYER_PREFIX):
                        self._hide_before_remove(layer)
            except Exception:
                pass
        # Belt-and-suspenders: drop any tracked ref still present.
        for layer in list(self._overview_layers):
            self._hide_before_remove(layer)
        self._overview_layers = []

    # -- the vispy Line overlay (v2 crash-safe box render) ------------------

    def _main_qt_viewer(self):
        """The shared viewer's ``QtViewer`` (private-API, guarded).

        Needed to parent the box wireframe onto the main canvas scene.  Probes a
        couple of known attribute chains defensively across napari versions —
        copied verbatim from ``curation_3d_widget._add_main_qt_viewer`` (the
        proven crash-safe path).
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

    def _ensure_overlay(self):
        if self._overlay is not None:
            return self._overlay
        try:
            from ._box_overlay import Step1BoxOverlay

            self._overlay = Step1BoxOverlay(
                self._main_qt_viewer(), style=self._style)
            # Re-apply the user's chosen appearance so it persists across
            # box-mode toggles (each toggle destroys + recreates the overlay).
            try:
                self._overlay.set_style(
                    thickness=self._style_thickness,
                    opacity=self._style_opacity,
                    color=self._style_color)
            except Exception:
                pass
        except Exception as exc:
            self._log(f"could not create box overlay: {exc}")
            self._overlay = None
        return self._overlay

    # -- live appearance (④ Render/style -> overlay) ------------------------

    def set_box_style(self, *, thickness=None, opacity=None,
                      color_name="__keep__") -> None:
        """Push ④'s Render/style choices onto the live box overlay.

        ``thickness`` (1..10) -> the outline ``Line.width`` (idle + a touch
        thicker for the active box); ``opacity`` (0..1) -> the outline colour
        alpha; ``color_name`` -> a global box colour (``"status colour"`` keeps
        the idle-teal / active-amber default).  Best-effort + guarded; a no-op
        (params still remembered) when box mode is off, so the look is applied
        the next time the overlay is built.  Controls live in ④ but reach here
        via the controller — the overlay persists across the ③/④ sections."""
        if thickness is not None:
            try:
                self._style_thickness = int(thickness)
            except (TypeError, ValueError):
                pass
        if opacity is not None:
            try:
                self._style_opacity = float(opacity)
            except (TypeError, ValueError):
                pass
        if color_name != "__keep__":
            self._style_color = _BOX_COLOR_CHOICES.get(color_name, None)
        self._apply_style_to_overlay()

    def _apply_style_to_overlay(self) -> None:
        overlay = self._overlay
        if overlay is None:
            return
        try:
            overlay.set_style(
                thickness=self._style_thickness,
                opacity=self._style_opacity,
                color=self._style_color)
        except Exception as exc:
            self._log(f"box style apply failed: {exc}")
        # Re-emit the box geometry so the new width/colour reach the visuals.
        self.redraw()

    def redraw(self) -> None:
        """Rebuild the overlay from ALL current boxes (idle + active) and refresh
        the ghost preview.  ``set_data`` is cheap, so this rides the per-slider-
        tick cadence unchanged."""
        overlay = self._overlay
        if overlay is None or self._full_shape is None:
            return
        self._apply_overlay_voxel()   # physical-µm scale, live-read per redraw
        try:
            overlay.update(self.boxes, self.active_index, self._full_shape)
        except Exception as exc:
            self._log(f"redraw failed: {exc}")
        self._update_ghost()

    def _remove_overlay(self) -> None:
        overlay = self._overlay
        if overlay is not None:
            try:
                overlay.destroy()
            except Exception as exc:
                self._log(f"overlay destroy failed: {exc}")
        self._overlay = None

    # -- ghost placement preview (feedback #3) ------------------------------

    def _update_ghost(self) -> None:
        """Refresh the faint ghost box showing where the NEXT Add / click lands
        (the current placement centre + sticky size).  Best-effort + guarded."""
        overlay = self._overlay
        if overlay is None or self._full_shape is None or not self._active:
            return
        try:
            self._apply_overlay_voxel()   # physical-µm scale (live read)
            cy, cx = self._camera_center_yx()
            sy, sx = self._next_size_yx()
            ghost = place_box(self._full_shape, cy, cx, sy, sx)
            overlay.update_ghost(ghost, self._full_shape)
        except Exception:
            pass

    def _connect_camera(self) -> None:
        """Drive the ghost preview off camera pan/zoom so it tracks the view.

        Each event is connected independently so a napari build missing one
        (e.g. ``zoom``) still wires the other rather than leaking a half
        connection."""
        if self._camera_connected or self.viewer is None:
            return
        cam = getattr(self.viewer, "camera", None)
        events = getattr(cam, "events", None)
        if events is None:
            return
        for name in ("center", "zoom"):
            ev = getattr(events, name, None)
            if ev is None:
                continue
            try:
                ev.connect(self._on_camera_moved)
                self._camera_connected = True
            except Exception:
                pass

    def _disconnect_camera(self) -> None:
        # Symmetry with `_connect_camera`'s guard: nothing was wired, so there is
        # nothing to unwire (the per-event disconnects are already no-ops here).
        if not self._camera_connected:
            return
        viewer = self.viewer
        cam = getattr(viewer, "camera", None)
        events = getattr(cam, "events", None)
        if events is not None:
            for name in ("center", "zoom"):
                ev = getattr(events, name, None)
                if ev is None:
                    continue
                try:
                    ev.disconnect(self._on_camera_moved)
                except (TypeError, RuntimeError, AttributeError, Exception):
                    pass
        self._camera_connected = False

    def _on_camera_moved(self, *_):
        self._update_ghost()

    # -- app-close safety net --------------------------------

    def _connect_close_hook(self) -> None:
        """Connect ``QApplication.aboutToQuit`` -> :meth:`stop` ONCE, so the
        overlay is always destroyed before the app tears down (no vispy visual
        left dangling during Qt teardown — the access-violation window)."""
        if self._close_hook_connected:
            return
        try:
            from qtpy.QtWidgets import QApplication

            app = QApplication.instance()
            if app is not None:
                app.aboutToQuit.connect(self._on_app_quit)
                self._close_hook_connected = True
        except Exception:
            pass

    def _on_app_quit(self) -> None:
        try:
            self.stop()
        except Exception:
            pass

    def _hide_before_remove(self, layer) -> None:
        if layer is None or self.viewer is None:
            return
        try:
            layer.visible = False  # disconnect the vispy visual before removal
        except Exception:
            pass
        try:
            if layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
        except Exception as exc:
            self._log(f"layer removal failed: {exc}")

    # -- create / edit / delete ---------------------------------------------

    def add_box(self, center_yx=None, size_yx=None) -> Optional[int]:
        """Add a box.

        Add-button (``center_yx=None``, feedback #2): drop at the current 2D view
        centre — the viewport middle while ``ndisplay == 2`` (or the LAST 2D
        centre when Add is pressed in 3D), so placement stays a 2D operation.
        Click-to-place passes the cursor voxel as ``center_yx``.  Both reuse the
        sticky size (S/M/L fraction OR absolute voxel preset) unless ``size_yx``
        is given.
        """
        if self._full_shape is None:
            self._full_shape = self._resolve_full_shape()
        if self._full_shape is None:
            self._log("cannot add box — no source loaded")
            return None
        if center_yx is None:
            cy, cx = self._camera_center_yx()
        else:
            cy, cx = center_yx
        if size_yx is None:
            sy, sx = self._next_size_yx()
        else:
            sy, sx = size_yx
        box = place_box(self._full_shape, cy, cx, sy, sx)
        self.boxes.append(box)
        self.box_checked.append(True)
        self.box_locked.append(False)                 # a fresh box is unlocked
        self.active_index = len(self.boxes) - 1
        self.refresh_list(select=self.active_index)
        self._load_geometry_into_spins(box)
        self._set_geometry_enabled(True)
        self._sync_lock_checkbox(False)               # selected + editable
        self.redraw()
        return self.active_index

    def _ctl_voxel(self):
        """The current anisotropic voxel ``(vz, vy, vx)`` from the controller
        (``_preview_spacing`` — the SAME value an extract bakes), read LIVE so a
        voxel-spin edit or a ②-preview-before-③-start is picked up with no cache
        to re-seed.  Falls back to ``(1, 1, 1)`` when the controller has no
        spacing hook (headless fakes) or it raises.  Guarded — never raises."""
        fn = getattr(self._ctl, "_preview_spacing", None)
        if callable(fn):
            try:
                v = tuple(float(x) for x in fn())
                if len(v) == 3:
                    return v
            except Exception:
                pass
        return (1.0, 1.0, 1.0)

    def _apply_overlay_voxel(self):
        """Push the live voxel onto the ③ box overlay so it renders scale-aware
        (physical µm).  Best-effort + guarded (a no-op when no overlay)."""
        ov = self._overlay
        if ov is None:
            return
        try:
            ov.set_voxel_size(self._ctl_voxel())
        except Exception:
            pass

    def _camera_center_yx(self):
        """The ``(cy, cx)`` voxel an Added box drops at — placement is a 2D op
        (feedback #2).

        In **2D** (``ndisplay == 2``) the camera centre IS the viewport middle
        (it tracks pan + zoom), so read it live and remember it as the last 2D
        centre.  In **3D** the camera centre is the orbit target, which is not a
        meaningful placement point — reuse the LAST 2D centre instead.  Falls
        back to the volume centre when there is no viewer / no camera / a
        non-finite value.  Clamped into the volume."""
        import math
        _, Y, X = self._full_shape
        fallback = (Y // 2, X // 2)                    # volume centre
        viewer = self.viewer
        if viewer is None:
            return fallback

        def _clamp(wy, wx):
            return (int(np.clip(round(wy), 0, max(0, Y - 1))),
                    int(np.clip(round(wx), 0, max(0, X - 1))))

        # Best-effort ndisplay probe (a bare fake viewer has no `dims`).
        nd = None
        try:
            nd = int(viewer.dims.ndisplay)
        except Exception:
            nd = None
        # In 3D the camera centre is the orbit target, NOT a placement point:
        # never read it — reuse the last 2D centre, else the volume centre.
        if nd == 3:
            return (_clamp(*self._last_2d_center)
                    if self._last_2d_center is not None else fallback)
        # 2D (or a bare fake viewer with no `dims`): the camera centre IS the
        # viewport middle (tracks pan + zoom).
        try:
            center = tuple(float(c) for c in viewer.camera.center)[-3:]
        except Exception:
            return fallback
        if len(center) < 3:
            return fallback
        _cz, wy, wx = center
        if not (math.isfinite(wy) and math.isfinite(wx)):
            return fallback
        # `center` is PHYSICAL µm (the display now sits in physical world).
        # Divide Y/X by the voxel to reach voxel indices (the box lives in raw-
        # full-res voxel space).  `_cz` is unused downstream, so no Z division.
        # Guard a zero voxel (treat as 1) so a degenerate spacing can't blow up.
        _vz, vy, vx = self._ctl_voxel()
        if vy:
            wy = wy / vy
        if vx:
            wx = wx / vx
        cy, cx = _clamp(wy, wx)      # _clamp rounds → `_last_2d_center` is VOXEL
        if nd == 2:
            self._last_2d_center = (cy, cx)
        return cy, cx

    def _next_size_yx(self):
        """The ``(sy, sx)`` voxel size the next box gets — the sticky size, in
        whichever model was last chosen (absolute voxel cube or S/M/L field
        fraction)."""
        _, Y, X = self._full_shape
        if self._last_size_mode == "abs" and self._last_abs_size:
            s = max(1, int(self._last_abs_size))
            return s, s
        frac = self._last_size_frac
        return max(1, int(Y * frac)), max(1, int(X * frac))

    def set_size_fraction(self, frac: float) -> None:
        """S/M/L preset (feedback #5): remember the field-fraction the NEXT Add /
        click-to-place uses.

        **Bug fix 2026-07-27 — the preset is sticky for the NEXT box ONLY; it no
        longer retroactively resizes any already-placed box.**  Size is a
        per-box property: resize an existing box through the Geometry spins on
        the SELECTED box (selecting a box loads its own size into the spins, so
        the selected box is always the edit target — no explicit lock needed).
        The old retroactive resize made "size" read as a shared/global property —
        pick a size and the selected box changed AND every new box was born at
        it — which the user experienced as "editing the size changes ALL the
        boxes".  The ghost preview refreshes so the next box's size is visible
        immediately."""
        try:
            frac = float(frac)
        except (TypeError, ValueError):
            return
        if not (0.0 < frac <= 1.0):
            return
        self._last_size_frac = frac
        self._last_size_mode = "frac"
        self._update_ghost()

    def set_size_voxels(self, size) -> None:
        """Absolute-size preset (feedback #6 — 128³/256³/512³ + custom): remember
        the sticky voxel XY size the NEXT Add / click-to-place uses.

        **Bug fix 2026-07-27 — sticky for the NEXT box ONLY; it does NOT
        retroactively resize any placed box** (see :meth:`set_size_fraction`).
        Resize an existing box via the Geometry spins on the selected box.  The
        ghost preview refreshes so the next box's size is visible immediately."""
        try:
            size = int(size)
        except (TypeError, ValueError):
            return
        if size < 1:
            return
        self._last_abs_size = size
        self._last_size_mode = "abs"
        self._update_ghost()

    def set_edge_mode(self, mode: str) -> None:
        """③ at-edge toggle: how a LIVE geometry edit resolves a box that would
        overflow the volume edge.

        ``"shrink"`` (default) collapses the box onto its rendered (truncated)
        bounds — extraction byte-identical, the geometry handle never lags the
        drawn box.  ``"keep"`` preserves the requested size and slides the box
        inward off the edge (deliberately changes the edge extraction).  An
        invalid value is ignored.  Redraws the active box so a toggle is applied
        without waiting for the next edit."""
        if mode not in ("shrink", "keep"):
            return
        self._edge_mode = mode
        if self._active and self.active_index is not None:
            self.redraw()

    def delete_active(self) -> None:
        if self.active_index is None or not self.boxes:
            return
        try:
            self.boxes.pop(self.active_index)
            if 0 <= self.active_index < len(self.box_checked):
                self.box_checked.pop(self.active_index)
            if 0 <= self.active_index < len(self.box_locked):
                self.box_locked.pop(self.active_index)
        except IndexError:
            return
        self.active_index = (len(self.boxes) - 1) if self.boxes else None
        if self.active_index is None:
            self._set_geometry_enabled(False)
            self._sync_lock_checkbox(False, has_selection=False)
        else:
            self._load_geometry_into_spins(self.boxes[self.active_index])
            locked = self._is_locked(self.active_index)
            self._set_geometry_enabled(not locked)
            self._sync_lock_checkbox(locked)
        self.refresh_list(select=self.active_index)
        self.redraw()

    def _sync_box_checked_length(self) -> None:
        """Keep ``box_checked`` + ``box_locked`` the same length as ``boxes``
        (defensive: any path that mutated ``boxes`` without its tick/lock
        defaults to included + unlocked)."""
        n = len(self.boxes)
        cur = self.box_checked
        if len(cur) < n:
            cur.extend([True] * (n - len(cur)))
        elif len(cur) > n:
            del cur[n:]
        lock = self.box_locked
        if len(lock) < n:
            lock.extend([False] * (n - len(lock)))
        elif len(lock) > n:
            del lock[n:]

    def _reorder_boxes(self, perm, checked=None) -> bool:
        """Reorder ``boxes`` + ``box_checked`` to *perm* (old indices in the new
        row order) so LIST ORDER == EXTRACTION ORDER.

        *checked* (optional) supplies the tick states in the NEW order — the
        drag handler passes the list items' own states, so the reorder never
        depends on a half-updated ``box_checked``.  No-op + ``False`` on an
        invalid permutation.
        """
        n = len(self.boxes)
        try:
            perm = [int(p) for p in perm]
        except (TypeError, ValueError):
            return False
        if sorted(perm) != list(range(n)):
            return False
        self._sync_box_checked_length()
        old_active = self.active_index
        self.boxes = [self.boxes[p] for p in perm]
        if checked is not None and len(checked) == n:
            self.box_checked = [bool(c) for c in checked]
        else:
            self.box_checked = [self.box_checked[p] for p in perm]
        # The lock rides its box through the reorder (permuted, not a per-row
        # override like `checked` — a lock is a property of the box, not the row).
        self.box_locked = [self.box_locked[p] for p in perm]
        if old_active is not None and 0 <= old_active < n:
            self.active_index = perm.index(old_active)
        return True

    def select(self, index: Optional[int]) -> None:
        if index is None or self._list_rebuilding:
            return
        if not (0 <= index < len(self.boxes)):
            return
        self.active_index = index
        self._load_geometry_into_spins(self.boxes[index])
        # A locked box loads read-only: spins disabled + the checkbox reflects
        # its lock state, so the selected box is always the edit target and a
        # locked one cannot be nudged (2026-07-27).
        locked = self._is_locked(index)
        self._set_geometry_enabled(not locked)
        self._sync_lock_checkbox(locked)
        self.redraw()

    def on_props_change(self, *_):
        """A geometry slider/spin changed → mutate the active box → re-seed the
        widgets to the RENDERED box → redraw (live 3D update).

        Guarded by ``_loading`` against feedback loops, and by the per-box lock
        (a locked box's spins are disabled, but a programmatic ``setValue`` still
        emits ``valueChanged`` — the lock gate is what makes the freeze
        airtight).

        **Handle-match fix (2026-08-17):** the render is already clamped to the
        volume by ``BoxRec.bounds_dict``, but the raw spin center could sit out
        of bounds → the slider handle lagged the drawn box near an edge.  After
        building the box from the spins we re-seed it to what ``bounds_dict``
        actually produces (per ``self._edge_mode``) and push that back into the
        spins, so stored == rendered == extracted in one pass.  Both modes are
        idempotent (no creep) — see ``BoxRec.shrunk_to_bounds`` /
        ``BoxRec.slid_inward``.  ``clamped()`` is deliberately NOT used (it
        re-grows an edge box + changes the extracted region)."""
        if self._full_shape is None:      # defense (matches ② set_preview_center)
            return
        if self._loading or self.active_index is None:
            return
        if not (0 <= self.active_index < len(self.boxes)):
            return
        if self._is_locked(self.active_index):
            return
        w = self.widget
        try:
            cy = int(w.box_center_y_spin.value())
            cx = int(w.box_center_x_spin.value())
            sy = int(w.box_size_h_spin.value())   # H = Y
            sx = int(w.box_size_w_spin.value())    # W = X
            cz = int(w.box_center_z_spin.value())  # editable-Z
            sz = int(w.box_size_d_spin.value())    # D = Z
        except Exception:
            return
        box = place_box(self._full_shape, cy, cx, sy, sx, cz=cz, sz=sz)
        # Re-seed to the rendered (clamped) box so the handle can't lag it.
        if self._edge_mode == "keep":
            box = box.slid_inward(self._full_shape)     # keep size, slide inward
        else:
            box = box.shrunk_to_bounds(self._full_shape)  # shrink-to-fit (default)
        self.boxes[self.active_index] = box
        # Push the re-seeded geometry back into the spins (idempotent, and
        # `_loading`-guarded so it cannot re-enter this handler).
        self._load_geometry_into_spins(box)
        self._update_row_dim(self.active_index)   # C1: live Z×Y×X on the box row
        self.redraw()

    # -- per-box lock (③ "lock" checkbox) -----------------------------------

    def _is_locked(self, index) -> bool:
        """True when box *index* is locked (frozen from geometry edits)."""
        return (index is not None and 0 <= index < len(self.box_locked)
                and bool(self.box_locked[index]))

    def set_active_locked(self, locked: bool) -> None:
        """Freeze / unfreeze the SELECTED box's geometry (the ③ lock checkbox).

        A locked box's six geometry spins are disabled and :meth:`on_props_change`
        no-ops, so it cannot be moved or resized until unlocked.  Per-box: only
        the active box is affected, the flag rides that box through reorder, and
        a new box is born unlocked.  Unlocking RE-SYNCS the spins to the box
        first, so a value that drifted while disabled can never be applied on
        the first post-unlock edit.  A no-op with no active box."""
        i = self.active_index
        if i is None or not (0 <= i < len(self.box_locked)):
            return
        locked = bool(locked)
        self.box_locked[i] = locked
        if not locked and 0 <= i < len(self.boxes):
            self._load_geometry_into_spins(self.boxes[i])
        self._set_geometry_enabled(not locked)
        self._sync_lock_checkbox(locked)

    def _sync_lock_checkbox(self, locked: bool, *, has_selection: bool = True) -> None:
        """Mirror the lock state onto the ③ checkbox WITHOUT re-firing its
        ``toggled`` signal (which would re-enter :meth:`set_active_locked`).
        Disabled when nothing is selected — there is no box to lock."""
        w = self.widget
        chk = getattr(w, "box_lock_size_check", None) if w is not None else None
        if chk is None:
            return
        try:
            was = chk.blockSignals(True)
            chk.setChecked(bool(locked))
            chk.setEnabled(bool(has_selection))
            chk.blockSignals(was)
        except Exception:
            pass

    # -- geometry widgets ----------------------------------------------------

    def _geometry_spins(self):
        w = self.widget
        # All six axes are editable now that Z is: X/Y/Z + W/H/D.
        names = ("box_center_x_spin", "box_center_y_spin", "box_center_z_spin",
                 "box_size_w_spin", "box_size_h_spin", "box_size_d_spin")
        return [getattr(w, n, None) for n in names]

    def _apply_geometry_spin_maxima(self) -> None:
        """Ask the widget to lift the geometry spin/slider ceilings off the
        resolved source shape.  A no-op when the widget predates the setter
        (headless / older builds) or no shape has resolved."""
        setter = getattr(self.widget, "set_box_geometry_maxima", None)
        if setter is None or self._full_shape is None:
            return
        try:
            setter(self._full_shape)
        except Exception:
            pass

    def _connect_geometry(self) -> None:
        if self._geom_connected:
            return
        for spin in self._geometry_spins():
            if spin is not None:
                try:
                    spin.valueChanged.connect(self.on_props_change)
                except Exception:
                    pass
        self._geom_connected = True

    def _disconnect_geometry(self) -> None:
        if not self._geom_connected:
            return
        for spin in self._geometry_spins():
            if spin is not None:
                try:
                    spin.valueChanged.disconnect(self.on_props_change)
                except (TypeError, RuntimeError):
                    pass
        self._geom_connected = False

    def _set_geometry_enabled(self, enabled: bool) -> None:
        """Enable / disable ALL six geometry spins (X/Y/Z + W/H/D) — Z is now
        editable.  An untouched box still defaults to full depth,
        so extraction is byte-identical until the user narrows Z."""
        w = self.widget
        if w is None:
            return
        for spin in self._geometry_spins():
            if spin is not None:
                try:
                    spin.setEnabled(enabled)
                except Exception:
                    pass

    def _load_geometry_into_spins(self, box: BoxRec) -> None:
        w = self.widget
        if w is None or self._full_shape is None:
            return
        self._loading = True
        try:
            self._set_spin(w, "box_center_x_spin", box.cx)
            self._set_spin(w, "box_center_y_spin", box.cy)
            self._set_spin(w, "box_size_w_spin", box.sx)
            self._set_spin(w, "box_size_h_spin", box.sy)
            # The box's REAL Z now (editable), full-depth by default.
            self._set_spin(w, "box_center_z_spin", box.cz)
            self._set_spin(w, "box_size_d_spin", box.sz)
        finally:
            self._loading = False

    @staticmethod
    def _set_spin(widget, name, value) -> None:
        spin = getattr(widget, name, None)
        if spin is None:
            return
        try:
            value = int(value)
            if value > spin.maximum():
                value = spin.maximum()
            if value < spin.minimum():
                value = spin.minimum()
            spin.setValue(value)
        except Exception:
            pass

    # -- mouse click-to-place -----------------------------------------------

    def _install_mouse_cb(self) -> None:
        if self.viewer is None or self._mouse_cb is not None:
            return
        cb = self._make_place_callback()
        try:
            self.viewer.mouse_drag_callbacks.append(cb)
            self._mouse_cb = cb
        except Exception as exc:
            self._log(f"could not install click-to-place: {exc}")

    def _remove_mouse_cb(self) -> None:
        cb = self._mouse_cb
        if cb is None or self.viewer is None:
            self._mouse_cb = None
            return
        try:
            self.viewer.mouse_drag_callbacks.remove(cb)
        except (ValueError, Exception):
            pass
        self._mouse_cb = None

    def _place_enabled(self) -> bool:
        w = self.widget
        check = getattr(w, "box_place_check", None)
        try:
            return bool(check is not None and check.isChecked())
        except Exception:
            return False

    def _make_place_callback(self):
        def callback(viewer, event):
            # 2D only, a plain click (not a drag), placement toggled on.
            if not self._active or not self._place_enabled():
                return
            try:
                if int(viewer.dims.ndisplay) != 2:
                    return
            except Exception:
                return
            if self._full_shape is None:
                return
            dragged = False
            yield
            while event.type == "mouse_move":
                dragged = True
                yield
            if dragged:
                return
            try:
                # LAST 3 world coords (Z,Y,X) so a leading channel axis can't
                # scramble them.  World is now PHYSICAL µm, so divide Y/X by the
                # voxel to reach the raw-full-res voxel indices the box stores.
                pos = tuple(viewer.cursor.position)[-3:]
            except Exception:
                return
            if len(pos) < 3:
                return
            _vz, vy, vx = self._ctl_voxel()
            cy = int(round(pos[1] / vy)) if vy else int(round(pos[1]))
            cx = int(round(pos[2] / vx)) if vx else int(round(pos[2]))
            idx = self.add_box(center_yx=(cy, cx))
            if idx is not None:
                self._log(f"placed box #{idx + 1} at voxel y={cy} x={cx}")
        return callback

    # -- the (minimal Phase-1) box list -------------------------------------

    def _list(self):
        w = self.widget
        panel = getattr(w, "draw_box_list", None)
        return getattr(panel, "list", None)

    def _connect_list(self) -> None:
        if self._list_connected:
            return
        lst = self._list()
        if lst is None:
            return
        try:
            lst.currentRowChanged.connect(self._on_row_changed)
            # A native-checkbox toggle → mirror onto `box_checked` so the untick
            # survives the next rebuild (fixes Phase-1 F1).
            lst.itemChanged.connect(self._on_item_changed)
            # A drag-drop reorder settling → mirror row order onto `boxes`.
            model = lst.model()
            model.rowsMoved.connect(self._on_list_structure_changed)
            model.rowsInserted.connect(self._on_list_structure_changed)
            self._list_connected = True
        except Exception:
            pass

    def _disconnect_list(self) -> None:
        if not self._list_connected:
            return
        lst = self._list()
        if lst is not None:
            for sig, slot in (
                ("currentRowChanged", self._on_row_changed),
                ("itemChanged", self._on_item_changed),
            ):
                try:
                    getattr(lst, sig).disconnect(slot)
                except (TypeError, RuntimeError, AttributeError):
                    pass
            try:
                model = lst.model()
                model.rowsMoved.disconnect(self._on_list_structure_changed)
                model.rowsInserted.disconnect(self._on_list_structure_changed)
            except (TypeError, RuntimeError, AttributeError):
                pass
        self._list_connected = False

    def _own_list(self) -> None:
        """Take over ③'s Box List: ONE native checkbox per row and
        drag-to-reorder.  Tells the ``BoxListPanel`` to stop overlaying its row-
        card widget (which added a SECOND checkbox and broke InternalMove drag)
        and turns on InternalMove on the inner list."""
        panel = getattr(self.widget, "draw_box_list", None)
        if panel is not None:
            try:
                panel._suppress_row_widgets = True
            except Exception:
                pass
        lst = self._list()
        if lst is None:
            return
        try:
            from qtpy.QtCore import Qt
            from qtpy.QtWidgets import QAbstractItemView

            lst.setSelectionMode(QAbstractItemView.SingleSelection)
            lst.setDragDropMode(QAbstractItemView.InternalMove)
            lst.setDefaultDropAction(Qt.MoveAction)
        except Exception as exc:
            self._log(f"could not enable list reorder: {exc}")

    def _on_row_changed(self, row: int) -> None:
        if self._list_rebuilding:
            return
        self.select(row)

    def _on_item_changed(self, item) -> None:
        """A row's native checkbox toggled → mirror onto ``box_checked``."""
        if self._list_rebuilding or item is None:
            return
        lst = self._list()
        if lst is None:
            return
        try:
            from qtpy.QtCore import Qt

            row = lst.row(item)
            if 0 <= row < len(self.box_checked):
                self.box_checked[row] = (item.checkState() == Qt.Checked)
        except Exception:
            pass

    def _on_list_structure_changed(self, *_):
        """A drag-drop reorder settled → mirror the new row order onto ``boxes``
        (list order == extraction order).  Reads each row's stable
        build-index + its live tick, so the reorder never depends on a half-
        updated ``box_checked``; deferred so the model isn't mutated mid-signal.
        """
        if self._list_rebuilding:
            return
        lst = self._list()
        if lst is None or lst.count() != len(self.boxes):
            return  # mid-move intermediate state — wait for the settle
        try:
            from qtpy.QtCore import Qt, QTimer
        except Exception:
            return
        perm, checked = [], []
        for r in range(lst.count()):
            item = lst.item(r)
            val = item.data(Qt.UserRole + 5) if item is not None else None
            if val is None:
                return  # no build-index to map by — skip
            perm.append(int(val))
            checked.append(item.checkState() == Qt.Checked)
        if perm == list(range(len(perm))):
            return  # identity — nothing moved

        def _do():
            if self._reorder_boxes(perm, checked=checked):
                self.refresh_list(select=self.active_index)
                self.redraw()

        try:
            QTimer.singleShot(0, _do)
        except Exception:
            _do()

    def _provisional_names(self) -> list:
        """Row labels = the provisional patch names extraction will assign
        (alias prefix + the cumulative next index), so the row you edit is
        labelled with the name the patch will actually get."""
        w = self.widget
        alias = ""
        try:
            alias = w.display_name_edit.text().strip()
        except Exception:
            pass
        prefix = f"{alias}_patch" if alias else "patch"
        start = 1
        nexter = getattr(w, "_next_patch_index", None)
        if callable(nexter):
            try:
                start = int(nexter(prefix))
            except Exception:
                start = 1
        return [f"{prefix}_{start + i}" for i in range(len(self.boxes))]

    def _box_dim_text(self, index: int) -> str:
        """The ``Z×Y×X`` voxel size of box *index* AS EXTRACTED — read from
        ``bounds_dict`` (the same clamped source ``rectangles()`` / ``z_ranges()``
        use), so the shown size matches the extracted extent even at a volume
        edge.  Empty string when unavailable (C1: per-row dimension)."""
        if self._full_shape is None or not (0 <= index < len(self.boxes)):
            return ""
        try:
            b = self.boxes[index].bounds_dict(self._full_shape)
            z = b["z1"] - b["z0"]
            y = b["y1"] - b["y0"]
            x = b["x1"] - b["x0"]
            return f"{z}×{y}×{x}"
        except Exception:
            return ""

    def _update_row_dim(self, index: int) -> None:
        """Live-update ONE box row's ``Z×Y×X`` size role as the geometry sliders
        move it (C1) — a single ``setData``, NOT a full ``refresh_list`` rebuild
        (no flicker / selection churn).  Guarded against the ``itemChanged``
        re-entrancy (``_on_item_changed`` short-circuits on ``_list_rebuilding``)."""
        lst = self._list()
        if lst is None or index is None or not (0 <= index < lst.count()):
            return
        try:
            from qtpy.QtCore import Qt
        except Exception:
            return
        prev = self._list_rebuilding
        self._list_rebuilding = True
        try:
            item = lst.item(index)
            if item is not None:
                item.setData(Qt.UserRole + 8, self._box_dim_text(index))
        except Exception:
            pass
        finally:
            self._list_rebuilding = prev

    def refresh_list(self, select: Optional[int] = None) -> None:
        """Rebuild ③'s Box List from the box tool's boxes (ONE
        native checkbox per row, drag-reorderable, unticks preserved).

        The per-box ``box_checked`` — NOT a per-rebuild re-tick — is the source
        of truth for each row's tick, so an untick survives a rebuild.  Each row
        carries its build-index (UserRole+5) so a drag-drop reorder can be mapped
        back onto ``boxes``.
        """
        lst = self._list()
        if lst is None:
            return
        self._sync_box_checked_length()
        if select is None:
            select = self.active_index
        names = self._provisional_names()
        self._list_rebuilding = True
        try:
            from qtpy.QtCore import Qt
            from qtpy.QtWidgets import QListWidgetItem

            lst.blockSignals(True)
            lst.clear()
            for i, name in enumerate(names):
                item = QListWidgetItem(name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                checked = self.box_checked[i] if i < len(self.box_checked) else True
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                item.setData(Qt.UserRole + 7, True)  # BoxListPanel: don't re-tick
                item.setData(Qt.UserRole + 5, i)     # stable build-index (reorder)
                item.setData(Qt.UserRole + 8, self._box_dim_text(i))  # C1: Z×Y×X
                lst.addItem(item)
            lst.blockSignals(False)
            if select is not None and 0 <= select < lst.count():
                lst.setCurrentRow(select)
        except Exception as exc:
            self._log(f"list refresh failed: {exc}")
            try:
                lst.blockSignals(False)
            except Exception:
                pass
        finally:
            self._list_rebuilding = False
        panel = getattr(self.widget, "draw_box_list", None)
        refresh = getattr(panel, "refresh", None)
        if callable(refresh):
            try:
                refresh()
            except Exception:
                pass

    # -- extract bridge ------------------------------------------------------

    def rectangles(self) -> list:
        """The current boxes as ``(4, 2)`` ``(Y, X)`` rectangles for the
        unchanged extract path (``get_drawn_shapes`` bridges to this)."""
        if self._full_shape is None:
            self._full_shape = self._resolve_full_shape()
        if self._full_shape is None:
            return []
        return [rect_for_box(box, self._full_shape) for box in self.boxes]

    def z_ranges(self) -> list:
        """Per-box ``[z0, z1]`` half-open Z crops, aligned to
        :meth:`rectangles`.  A full-Z box yields ``[0, Z]`` → the extract stays
        byte-
        identical; a narrowed box yields its sub-slab.  The controller zips these
        with the boxes so each patch's Z is cropped after the XY crop."""
        if self._full_shape is None:
            self._full_shape = self._resolve_full_shape()
        if self._full_shape is None:
            return []
        out = []
        for box in self.boxes:
            b = box.bounds_dict(self._full_shape)
            out.append([int(b["z0"]), int(b["z1"])])
        return out

    # -- mode-button sync ----------------------------------------------------

    def _sync_mode_button(self, checked: bool) -> None:
        btn = getattr(self.widget, "box3d_mode_button", None)
        if btn is None:
            return
        try:
            was = btn.blockSignals(True)
            btn.setChecked(checked)
            btn.blockSignals(was)
        except Exception:
            pass
