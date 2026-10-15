"""EpiCycle's progressive-3D shim + seam for STOCK napari 0.8.0.

The vendored napari-#9067 engine (``epicycle_largedata._vendor_progressive``)
was written against a *fork* of napari whose ``ScalarFieldBase`` carried a
296-line graft: 6 extra ``__init__`` attributes, a rewritten
``_update_level_and_corners`` with 3D sub-volume tiling, three new helpers
(``_view_direction_data``, ``_corners_for_locked_level``,
``_locked_tile_moved``), a ``locked_data_level`` setter override, and a
matching change to the *slice request* so a locked/auto 3D level is cropped to
``corner_pixels`` (stock napari only crops in 2D).

Stock 0.8.0 has none of that.  Rather than monkeypatch napari's **shared**
``ScalarFieldBase`` (which would change behaviour for every image/labels layer
in the process), this module confines the graft to **our own** layer
subclasses:

* :class:`_ProgressiveScalarFieldMixin` carries the ported graft methods and
  attributes.
* :class:`ProgressiveImage` / :class:`ProgressiveLabels` are
  ``(mixin, napari.layers.Image/Labels)`` subclasses.
* :class:`_ProgressiveScalarFieldSlicingState` +
  :class:`_ProgressiveScalarFieldSliceRequest` confine the slice-path graft
  (the ``crop_to_corners_3d`` behaviour) to those layers too, wired in via the
  mixin's ``_get_layer_slicing_state`` override.

THE SEAM
--------
:func:`add_progressive_image` / :func:`add_progressive_labels` install the
process-wide GLIR meter, swap the layer classes the engine builds to OURS,
call the vendored ``add_progressive_loading_*``, wire meter teardown, and fail
**closed** (return ``None`` so the caller can fall back to the manual-slider
path).  The failure path is *atomic*: the engine mutates the viewer + global
slicing settings INSIDE the swap window before its own later failure points,
so the seam snapshots that state up front and rolls it ALL back on any error.

All progressive layers MUST be created via this seam (never the vendored
engine directly) — the seam's registry (:data:`_live_refs`) is the sole
GLIR-uninstall authority, and only the seam wires the ``removed`` /
window-destroyed teardown that uninstalls the process-wide meter when the last
progressive layer goes away.

Layer-class swap note
---------------------
The vendored engine builds its layer with a *local* ``from napari.layers
import Image`` inside ``add_progressive_loading_image`` (likewise ``Labels``).
A module-level ``_progressive_loading.Image = ...`` is therefore **not**
picked up (the engine module has no module-level ``Image``/``Labels`` to swap).
The load-bearing — and only — swap target is the attribute on the
``napari.layers`` package (which the local import reads at call time); we swap
that inside a scoped, main-thread-only, non-reentrant context manager that
always restores on exit.

Reconstructed slice crop
-------------------------
The fork's ``_slice.py`` was **not** provided.
:meth:`_ProgressiveScalarFieldSliceRequest._call_multi_scale` is therefore
*reconstructed* from stock 0.8.0's ``_call_multi_scale`` by extending its 2D
``corner_pixels`` crop (+ tile2data translate) to the 3D case when
``crop_to_corners_3d`` is set.  It is a no-op when tile caps are unset
(``_max_tile_extent_3d is None``), i.e. for any bare layer, so it is identical
to stock until the engine attaches.  It MUST be validated in the GUI de-risk
gate (it cannot be exercised headlessly — 3D rendering needs a canvas).

This module imports ``napari.layers`` at module level (needed to subclass
Image/Labels) — that is Qt-free and headless-safe.  The vendored engine and
its GLIR meter are imported **lazily** inside the functions so importing this
module stays cheap.
"""

from __future__ import annotations

import dataclasses
import logging
import numbers
import threading
import weakref
from contextlib import contextmanager, nullcontext

import numpy as np

# Qt-free, headless-safe napari internals (verified present on 0.8.0).
from napari.layers import Image as _NapariImage, Labels as _NapariLabels
from napari.layers._scalar_field._slice import (
    _ScalarFieldSliceResponse,
    _ScalarFieldView,
)

# Image and Labels each have their OWN slice-request + slicing-state classes
# (they override _project_slice etc.); the graft must subclass THESE, not the
# shared base, or _project_slice would be the abstract base and every slice
# would raise NotImplementedError.
from napari.layers.image._slice import _ImageSliceRequest
from napari.layers.image.image import _ImageSlicingState
from napari.layers.labels._slice import _LabelsSliceRequest
from napari.layers.labels.labels import _LabelsSlicingState
from napari.utils.transforms import Affine

__all__ = (
    'ProgressiveImage',
    'ProgressiveLabels',
    'progressive_available',
    'add_progressive_image',
    'add_progressive_labels',
    'remove_progressive',
    'install_glir_if_needed',
    'uninstall_glir_if_idle',
    'ensure_async_slicing',
    'ensure_sync_slicing',
)

LOGGER = logging.getLogger('epicycle_largedata.progressive')

# --- constants ported verbatim from the fork's scalar_field.py -------------

#: Lower bound for the per-axis extent of 3D sub-volume tiles.
_MIN_TILE_EXTENT_3D = 32

#: 3D sub-volume tile extents are rounded up to a multiple of this so
#: consecutive camera poses produce tiles with repeating shapes (texture
#: reuse; avoids a GPU reallocation per pose on slow drivers).
_TILE_EXTENT_QUANTUM_3D = 32

_SENTINEL = object()


# =========================================================================
# Slice-path graft (confined to our layers via _get_layer_slicing_state).
#
# Shared bodies live as module functions so the Image and Labels variants
# (which must subclass their respective NATIVE request/state classes) don't
# duplicate them.
# =========================================================================
def _progressive_call_multi_scale(req):
    """RECONSTRUCTED ``_call_multi_scale`` (the fork's ``_slice.py`` was not
    provided).

    Stock 0.8.0's ``_ScalarFieldSliceRequest._call_multi_scale`` body with its
    2D ``corner_pixels`` crop (+ tile2data translate) extended to the 3D case
    when ``crop_to_corners_3d`` is set.  When the flag is False (the default;
    ``layer._max_tile_extent_3d is None``) this is byte-identical to stock, so
    a bare progressive layer behaves exactly like a normal one.  Without the
    3D crop, a locked/auto FINER level would materialize its full extent
    (VirtualData -> ``np.full(full_shape)``), stalling the UI and producing a
    texture the GL driver refuses to load (black canvas).
    """
    scale = np.ones(req.slice_input.ndim)
    for d in req.slice_input.displayed:
        scale[d] = req.downsample_factors[req.data_level][d]

    data = req.data_at_data_level

    translate = np.zeros(req.slice_input.ndim)
    disp_slice = [slice(None) for _ in data.shape]
    crop_to_corners = req.slice_input.ndisplay == 2 or (
        req.slice_input.ndisplay == 3 and req.crop_to_corners_3d
    )
    if crop_to_corners:
        for d in req.slice_input.displayed:
            disp_slice[d] = slice(
                req.corner_pixels[0, d],
                req.corner_pixels[1, d] + 1,
                1,
            )
        translate = req.corner_pixels[0] * scale

    tile_to_data = Affine(
        name='tile2data',
        scale=scale,
        translate=translate,
        ndim=req.slice_input.ndim,
    )

    data = data[tuple(disp_slice)]

    data_slice = req._thick_slice_at_level(req.data_level)
    data = req._project_thick_slice(data, data_slice)

    order = req._get_order()
    data = np.transpose(data, order)
    image = _ScalarFieldView.from_view(data)

    thumbnail_data_slice = req._thick_slice_at_level(req.thumbnail_level)
    thumbnail_data = req._project_thick_slice(
        req.data_at_thumbnail_level, thumbnail_data_slice
    )
    thumbnail_data = np.transpose(thumbnail_data, order)
    thumbnail = _ScalarFieldView.from_view(thumbnail_data)

    return _ScalarFieldSliceResponse(
        image=image,
        thumbnail=thumbnail,
        tile_to_data=tile_to_data,
        slice_input=req.slice_input,
        request_id=req.id,
    )


def _progressive_set_view_slice(state):
    """Ported fork hunk: reset corners for the coarsest 3D level through the
    tiling helper when switching to 3D without a viewer."""
    layer = state.layer
    if (
        layer.multiscale
        and state._slice_input.ndisplay == 3
        and layer._locked_data_level is None
    ):
        displayed = list(state._slice_input.displayed)
        level = len(layer.level_shapes) - 1
        data_bbox = None
        last_bbox = getattr(layer, '_last_data_bbox', None)
        if last_bbox is not None and last_bbox[0] == tuple(displayed):
            data_bbox = last_bbox[1]
        view_dir = layer._view_direction_data(displayed)
        layer._data_level = level
        layer.corner_pixels = layer._corners_for_locked_level(
            level, displayed, data_bbox, view_dir
        )
    request = state._make_slice_request_internal(
        slice_input=state._slice_input,
        data_slice=state.data_slice,
        dask_indexer=nullcontext,
    )
    response = request()
    state._update_slice_response(response)


def _progressive_make_slice_request_internal(
    state, slice_input, data_slice, dask_indexer
):
    """Stock ``_make_slice_request_internal`` body plus the
    ``crop_to_corners_3d`` kwarg (fork hunk)."""
    layer = state.layer
    data = layer.data
    if layer.multiscale:
        locked = layer._locked_data_level
        if locked is not None:
            data_level = locked
        elif slice_input.ndisplay == 3:
            data_level = len(data) - 1
        else:
            data_level = layer.data_level
    else:
        data_level = 0

    thumbnail_level = layer._thumbnail_level
    if layer._level_materializer:
        data_at_thumbnail_level = layer._level_materializer(thumbnail_level)
    elif layer.multiscale:
        data_at_thumbnail_level = data[thumbnail_level]
    else:
        data_at_thumbnail_level = data

    if not layer.multiscale:
        data_at_data_level = data
    elif data_level == thumbnail_level:
        data_at_data_level = data_at_thumbnail_level
    else:
        data_at_data_level = data[data_level]

    return state._slice_request_class(
        slice_input=slice_input,
        data_at_data_level=data_at_data_level,
        data_at_thumbnail_level=data_at_thumbnail_level,
        dtype=layer.dtype,
        dask_indexer=dask_indexer,
        data_slice=data_slice,
        projection_mode=layer.projection_mode,
        multiscale=layer.multiscale,
        corner_pixels=layer.corner_pixels,
        rgb=len(layer.data.shape) != state.ndim,
        data_level=data_level,
        thumbnail_level=thumbnail_level,
        level_shapes=layer.level_shapes,
        downsample_factors=layer.downsample_factors,
        crop_to_corners_3d=layer._max_tile_extent_3d is not None,
    )


@dataclasses.dataclass(frozen=True)
class _ProgressiveImageSliceRequest(_ImageSliceRequest):
    # Appended after every inherited field (parent's last field ``id`` has a
    # default_factory), so dataclass field ordering stays valid.
    crop_to_corners_3d: bool = False

    def _call_multi_scale(self):
        return _progressive_call_multi_scale(self)


@dataclasses.dataclass(frozen=True)
class _ProgressiveLabelsSliceRequest(_LabelsSliceRequest):
    crop_to_corners_3d: bool = False

    def _call_multi_scale(self):
        return _progressive_call_multi_scale(self)


class _ProgressiveImageSlicingState(_ImageSlicingState):
    _slice_request_class = _ProgressiveImageSliceRequest

    def _set_view_slice(self):
        _progressive_set_view_slice(self)

    def _make_slice_request_internal(
        self, *, slice_input, data_slice, dask_indexer
    ):
        return _progressive_make_slice_request_internal(
            self, slice_input, data_slice, dask_indexer
        )


class _ProgressiveLabelsSlicingState(_LabelsSlicingState):
    _slice_request_class = _ProgressiveLabelsSliceRequest

    def _set_view_slice(self):
        _progressive_set_view_slice(self)

    def _make_slice_request_internal(
        self, *, slice_input, data_slice, dask_indexer
    ):
        return _progressive_make_slice_request_internal(
            self, slice_input, data_slice, dask_indexer
        )


# =========================================================================
# The layer graft (ScalarFieldBase methods, confined to our subclasses).
# =========================================================================
class _ProgressiveScalarFieldMixin:
    """Ports the fork's ``ScalarFieldBase`` graft onto our Image/Labels
    subclasses.

    Mixin MUST come before ``napari.layers.Image``/``Labels`` in the MRO so
    these overrides win.  The 6 fork attributes are declared as class-level
    defaults (so the graft never hits ``AttributeError`` even if a slice runs
    mid-construction) and are also set explicitly per-instance in ``__init__``.
    """

    # Class-level defaults (bulletproofing) — instance values set in __init__.
    _last_data_bbox = None
    _max_tile_extent_3d = None
    _tile_max_bytes_3d = None
    _interval_max_bytes_3d = None
    _camera_view_direction = None
    _render_margin_2d = 1.0
    # "Zoom depth": how the 3D sub-volume tile's depth (Z) axis is sized when a
    # level is zoomed in.  None => Off (byte-identical to the coarse camera-
    # centered band); a float in {0.25,0.5,0.75,1.0} => pin depth to that
    # FRACTION of full Z and shrink only the lateral axes; "cube" => isotropic
    # box.  Read per-draw in `_corners_for_locked_level` so the controller can
    # flip it LIVE.
    _zoom_depth = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Viewport bbox from the most recent draw, as
        # (displayed_axes, data_bbox_int); see _update_level_and_corners.
        self._last_data_bbox = None
        # Max per-axis extent (data px) of the 3D region sliced for a locked
        # level; None keeps stock behaviour (slice the full locked level).
        self._max_tile_extent_3d = None
        self._tile_max_bytes_3d = None
        # Full-interval memory budget; a level within it renders whole.
        self._interval_max_bytes_3d = None
        # Camera view direction in world coords.  In the fork this was set
        # each frame by a `Layer._update_draw` graft on the *shared* base
        # class, which this shim deliberately does not touch — and the
        # vendored engine never sets it either.  So it stays None under this
        # shim and `_view_direction_data` returns None: the sub-volume tile
        # centers on the view bbox instead of biasing toward the camera-facing
        # near face.  Tiling still works; only the "closest-data-first after a
        # rotation" optimization is lost.  TODO (GUI de-risk): repopulate this
        # from the vispy canvas draw event if the bias proves worthwhile.
        self._camera_view_direction = None
        # 2D render margin (>1 renders beyond the viewport).
        self._render_margin_2d = 1.0
        # "Zoom depth" (LIVE, read per-draw in `_corners_for_locked_level`).
        # None => Off (coarse camera-centered depth band); a float => pin depth
        # to that fraction of full Z (lateral shrinks); "cube" => isotropic box.
        self._zoom_depth = None

    # -- confine the slice-path graft to our layers -----------------------
    def _get_layer_slicing_state(self, data, cache):
        # Dispatch on the concrete layer kind so we subclass the correct
        # native slicing state (Image vs Labels differ in _project_slice).
        if isinstance(self, _NapariLabels):
            return _ProgressiveLabelsSlicingState(
                layer=self, data=data, cache=cache
            )
        return _ProgressiveImageSlicingState(
            layer=self, data=data, cache=cache
        )

    # -- locked_data_level setter override --------------------------------
    @property
    def locked_data_level(self):
        """int or None: Locked multiscale resolution level."""
        return self._locked_data_level

    @locked_data_level.setter
    def locked_data_level(self, level):
        if level is not None:
            n_levels = len(self.level_shapes)
            if level < 0 or level >= n_levels:
                return
        old_level = self._data_level
        self._locked_data_level = level
        if level is not None:
            displayed_axes = self._slice_input.displayed
            # Size any 3D sub-volume tile to the viewport instead of the full
            # memory budget. Use the cached last-draw bbox when available,
            # else project the previous level's corner pixels to level-0.
            data_bbox_int = None
            if self._last_data_bbox is not None and self._last_data_bbox[
                0
            ] == tuple(displayed_axes):
                data_bbox_int = self._last_data_bbox[1]
            else:
                data_bbox_int = (
                    self.corner_pixels[:, displayed_axes]
                    * np.take(
                        np.asarray(self.downsample_factors[old_level]),
                        displayed_axes,
                    )
                ).astype(int)
            view_dir = self._view_direction_data(displayed_axes)
            self.corner_pixels = self._corners_for_locked_level(
                level, displayed_axes, data_bbox_int, view_dir
            )
            self._data_level = level
        else:
            self._reset_data_level()
        self.refresh(extent=False)
        self.events.locked_data_level()

    # -- live "zoom depth" re-apply ---------------------------------------
    def apply_zoom_depth_now(self):
        """Recompute + apply this layer's 3D tile corners for the CURRENT level
        using the LIVE ``_zoom_depth`` mode, then reslice.

        The controller sets ``_zoom_depth`` and calls this so a depth change
        takes effect WITHOUT a camera move.  This mirrors the recompute block of
        the ``locked_data_level`` setter (the reason the 3D-detail slider applies
        live), but keyed on the CURRENT ``_data_level`` so it works in BOTH the
        auto-3D and locked modes.  Setting ``corner_pixels`` then
        ``refresh(extent=False)`` reslices with the NEW corners — a bare refresh
        without recomputing corners would NOT (that was the inert-camera bug).
        Best-effort: never raises into the toggle."""
        try:
            if not self.multiscale or self._slice_input.ndisplay != 3:
                return
            old_level = self._data_level
            displayed_axes = self._slice_input.displayed
            # Size the sub-volume tile to the viewport: cached last-draw bbox
            # when it matches the displayed axes, else project the current
            # level's corner pixels to level-0 (setter's exact fallback).
            data_bbox_int = None
            if self._last_data_bbox is not None and self._last_data_bbox[
                0
            ] == tuple(displayed_axes):
                data_bbox_int = self._last_data_bbox[1]
            else:
                data_bbox_int = (
                    self.corner_pixels[:, displayed_axes]
                    * np.take(
                        np.asarray(self.downsample_factors[old_level]),
                        displayed_axes,
                    )
                ).astype(int)
            view_dir = self._view_direction_data(displayed_axes)
            self.corner_pixels = self._corners_for_locked_level(
                self._data_level, displayed_axes, data_bbox_int, view_dir
            )
            self.refresh(extent=False)
        except Exception:  # noqa: BLE001 — never raise into the live toggle
            LOGGER.debug('apply_zoom_depth_now failed', exc_info=True)

    # -- level / corner selection ----------------------------------------
    def _update_level_and_corners(
        self, data_bbox_int, shape_threshold, displayed_axes
    ):
        """Select data level + corner pixels for the current viewport.

        Ported verbatim from the fork.  For non-multiscale data delegates to
        the stock base implementation.
        """
        if not self.multiscale:
            super()._update_level_and_corners(
                data_bbox_int, shape_threshold, displayed_axes
            )
            return

        # remember the viewport bbox (level-0 data coords) so out-of-draw
        # corner computations (e.g. the locked_data_level setter) can size
        # 3D sub-volume tiles to the view
        self._last_data_bbox = (tuple(displayed_axes), data_bbox_int)

        if self._locked_data_level is not None:
            locked = self._locked_data_level
            old_level = self._data_level
            self._data_level = locked
            view_dir = self._view_direction_data(displayed_axes)
            corners = self._corners_for_locked_level(
                locked, displayed_axes, data_bbox_int, view_dir
            )
            level_changed = old_level != locked
            if level_changed or self._locked_tile_moved(
                corners, displayed_axes
            ):
                self.corner_pixels = corners
                self.refresh(extent=False, thumbnail=False)
        elif self._slice_input.ndisplay == 2:
            from napari.layers.utils.layer_utils import (
                compute_multiscale_level_and_corners,
            )

            level, scaled_corners = compute_multiscale_level_and_corners(
                data_bbox_int,
                shape_threshold,
                self.downsample_factors[:, displayed_axes],
            )
            margin = self._render_margin_2d
            if margin > 1.0:
                pad = (
                    (scaled_corners[1] - scaled_corners[0])
                    * (margin - 1.0)
                    / 2.0
                )
                scaled_corners = np.stack(
                    [scaled_corners[0] - pad, scaled_corners[1] + pad],
                )
            corners = np.zeros((2, self.ndim), dtype=int)
            max_coords = np.take(self.data[level].shape, displayed_axes) - 1
            corners[:, displayed_axes] = np.clip(scaled_corners, 0, max_coords)
            display_shape = tuple(
                corners[1, displayed_axes] - corners[0, displayed_axes]
            )
            if any(s == 0 for s in display_shape):
                return
            if (
                self.data_level != level
                or np.any(
                    corners[0, displayed_axes]
                    < self.corner_pixels[0, displayed_axes]
                )
                or np.any(
                    corners[1, displayed_axes]
                    > self.corner_pixels[1, displayed_axes]
                )
            ):
                self._data_level = level
                self.corner_pixels = corners
                self.refresh(extent=False, thumbnail=False)
        else:
            # 3D: coarsest level, routed through the same tiling as locked.
            new_level = len(self.level_shapes) - 1
            level_changed = self._data_level != new_level
            self._data_level = new_level
            view_dir = self._view_direction_data(displayed_axes)
            corners = self._corners_for_locked_level(
                new_level, displayed_axes, data_bbox_int, view_dir
            )
            if level_changed or self._locked_tile_moved(
                corners, displayed_axes
            ):
                self.corner_pixels = corners
                self.refresh(extent=False, thumbnail=False)

    def _view_direction_data(self, displayed_axes):
        """View direction in level-0 data coords for displayed axes, or None.

        Uses ``_camera_view_direction`` (set by the engine each frame) rather
        than reaching up to ``current_viewer()``.
        """
        if self._slice_input.ndisplay != 3:
            return None
        world_dir = self._camera_view_direction
        if world_dir is None:
            return None
        world_dir = np.asarray(world_dir, dtype=float)
        if not np.all(np.isfinite(world_dir)):
            return None
        full_dir = np.zeros(self.ndim, dtype=float)
        full_dir[list(displayed_axes)] = world_dir[-len(displayed_axes):]
        try:
            data_dir = self._world_to_data_ray(full_dir)
        except Exception:  # noqa: BLE001
            return None
        disp_dir = data_dir[list(displayed_axes)]
        norm = np.linalg.norm(disp_dir)
        if norm < 1e-12:
            return None
        return disp_dir / norm

    def _corners_for_locked_level(
        self,
        level,
        displayed_axes,
        data_bbox_int=None,
        view_direction=None,
    ):
        """Corner pixels to render for a locked multiscale level.

        Full extent normally; in 3D with ``_max_tile_extent_3d`` set and a
        level larger than the cap, a view-centered sub-volume tile.  Ported
        verbatim from the fork.
        """
        shape_at_level = np.take(
            np.asarray(self.level_shapes[level]), displayed_axes
        )
        corners = np.zeros((2, self.ndim), dtype=int)
        corners[1, displayed_axes] = shape_at_level - 1

        extent_cap = self._max_tile_extent_3d
        if self._slice_input.ndisplay != 3 or extent_cap is None:
            return corners

        try:
            from napari._vispy.utils.gl import get_max_texture_sizes

            _, gl_max_3d = get_max_texture_sizes()
        except Exception:  # noqa: BLE001
            gl_max_3d = None

        # Whole level fits the interval budget -> render it in full.
        interval_budget = self._interval_max_bytes_3d
        if interval_budget is not None and (
            gl_max_3d is None or np.all(shape_at_level <= gl_max_3d)
        ):
            tex_itemsize = 1 if self.dtype == np.uint8 else 4
            # dtype=np.int64: np.prod defaults to the platform int (int32 on
            # Windows), which OVERFLOWS for a >2^31-voxel finest level (a
            # Windows-overflow hardening over the faithful fork port).
            level_bytes = int(np.prod(shape_at_level, dtype=np.int64)) * tex_itemsize
            if level_bytes <= interval_budget:
                return corners

        downsample = np.take(
            np.asarray(self.downsample_factors[level]), displayed_axes
        )
        # EPICYCLE "Zoom depth": parameterize how the depth (Z) axis is sized.
        #   * `_zoom_depth is None` (Off)  -> depth_pos None, no pin, no shrink-
        #     exclusion -> BYTE-IDENTICAL to the coarse camera-centered band.
        #   * `_zoom_depth` a float (0.25..1.0) -> pin depth to that fraction of
        #     full Z (GL-clamped, MIN-floored) and EXCLUDE it from the lateral
        #     shrink.  1.0 reproduces the old full-depth behaviour exactly.
        #   * `_zoom_depth == "cube"` -> cap EVERY displayed axis to one cube edge
        #     and exclude nothing (isotropic; shrinks equally if a coarse level's
        #     shape < edge).
        # data-axis 0 is Z in the ZYX prep-pipeline layout; `shape_at_level` is
        # indexed POSITIONALLY by `displayed_axes`, so resolve Z to its position.
        zd = getattr(self, "_zoom_depth", None)
        depth_axis = 0
        # Accept any real number as a fraction (defensive: an int / np.float32 /
        # np.float64 slipped in by a future change still pins, instead of silently
        # reverting to Off) — but EXCLUDE bool, which registers as a Real, so
        # True/False fall through to "no pin".  `numbers.Real` is used (not the
        # literal `(int, float)`) precisely because numpy scalars are NOT Python
        # float/int subclasses, so `(int, float)` would miss np.float32/np.int32 —
        # the very case being hardened.  "cube" (str) and None also fall through
        # here (handled below / left as Off).
        depth_pos = (
            list(displayed_axes).index(depth_axis)
            if isinstance(zd, numbers.Real) and not isinstance(zd, bool)
            and depth_axis in tuple(displayed_axes)
            else None
        )
        # Anisotropic tile extent that fits the byte budget.
        tile_bytes = self._tile_max_bytes_3d
        if tile_bytes is not None:
            itemsize = max(int(self.dtype.itemsize), 1)
            max_elements = max(tile_bytes // itemsize, 1)
            gl_max = gl_max_3d if gl_max_3d is not None else extent_cap
            tile_extent = np.minimum(shape_at_level, gl_max).astype(np.int64)
            if depth_pos is not None:                                  # float mode: pin depth to frac*full Z (GL-clamped, MIN-floored)
                tile_extent[depth_pos] = max(
                    min(int(shape_at_level[depth_pos] * float(zd)),
                        int(gl_max)),
                    _MIN_TILE_EXTENT_3D,
                )
            elif zd == "cube":                                         # cube mode: cap every axis to one isotropic edge
                edge = max(
                    int(max_elements ** (1.0 / len(tile_extent))),
                    _MIN_TILE_EXTENT_3D,
                )
                tile_extent = np.minimum(tile_extent, edge)
            for _ in range(len(tile_extent)):
                vol = int(np.prod(tile_extent))
                if vol <= max_elements:
                    break
                over = np.where(tile_extent > _MIN_TILE_EXTENT_3D)[0]
                if depth_pos is not None:                              # float mode: only lateral axes shrink (depth stays pinned)
                    over = over[over != depth_pos]
                if len(over) == 0:
                    break
                ratio = (max_elements / vol) ** (1.0 / len(over))
                for ax in over:
                    tile_extent[ax] = max(
                        int(tile_extent[ax] * ratio), _MIN_TILE_EXTENT_3D
                    )
        else:
            tile_extent = np.minimum(shape_at_level, extent_cap)
        if data_bbox_int is not None and np.all(np.isfinite(data_bbox_int)):
            bbox = np.asarray(data_bbox_int, dtype=float) / downsample
            center = bbox.mean(axis=0)
            # Bias toward the near face (camera-facing side loads first).
            if view_direction is not None:
                vd = np.asarray(view_direction, dtype=float)
                vd_norm = np.linalg.norm(vd)
                if vd_norm > 1e-12:
                    vd = vd / vd_norm
                    shift = -vd * tile_extent * 0.35
                    center = center + shift
        else:
            center = shape_at_level / 2
        quantum = _TILE_EXTENT_QUANTUM_3D
        tile_extent = np.asarray(tile_extent, dtype=np.int64)
        tile_extent = np.minimum(
            -(-tile_extent // quantum) * quantum,
            shape_at_level,
        )
        center = np.clip(center, 0, shape_at_level - 1)
        if np.all(tile_extent >= shape_at_level):
            return corners
        low = np.clip(
            (center - tile_extent / 2).astype(int),
            0,
            shape_at_level - tile_extent,
        )
        high = low + tile_extent
        corners[0, displayed_axes] = low
        corners[1, displayed_axes] = high - 1
        return corners

    def _locked_tile_moved(self, corners, displayed_axes):
        """Whether new locked-level corners warrant a re-slice (quarter-tile
        hysteresis).  Ported verbatim from the fork."""
        current = self.corner_pixels
        if current.shape != corners.shape:
            return True
        extent = (corners[1] - corners[0])[displayed_axes] + 1
        current_extent = (current[1] - current[0])[displayed_axes] + 1
        if not np.array_equal(extent, current_extent):
            return True
        new_center = corners.mean(axis=0)[displayed_axes]
        current_center = current.mean(axis=0)[displayed_axes]
        return bool(np.any(np.abs(new_center - current_center) > extent / 4))


class ProgressiveImage(_ProgressiveScalarFieldMixin, _NapariImage):
    """``napari.layers.Image`` carrying the confined progressive-3D graft."""


class ProgressiveLabels(_ProgressiveScalarFieldMixin, _NapariLabels):
    """``napari.layers.Labels`` carrying the confined progressive-3D graft."""


# =========================================================================
# Lazy loaders for the vendored engine (keeps import of this module cheap).
# =========================================================================
def _load_engine():
    from ._vendor_progressive import _progressive_loading

    return _progressive_loading


def _load_metering():
    from ._vendor_progressive import _glir_metering

    # EpiCycle owns this policy, NOT the vendored file.  The meter's
    # deferred-DELETE reorder is the root of the 3D<->2D
    # `ibuf.activate() on None` / "Could not find texture with id" crashes (see
    # the `_METER_ENABLED` note below), so DELETEs must run INLINE in queue
    # order.  The flag used to be a hand edit INSIDE the DO-NOT-EDIT vendored
    # module, which a re-vendor would silently revert -- so assert it here, at the single door our code uses to reach
    # the module (`install_glir_if_needed` + `_meter_install_suppressed`).
    _glir_metering._DEFER_DELETES = False
    return _glir_metering


def _get_settings():
    """Lazy accessor for napari's global settings singleton.

    Factored out (rather than imported inline in the seam) so the atomic
    rollback's snapshot/restore of ``experimental.async_`` is monkeypatchable
    in headless tests without touching the real napari settings.
    """
    from napari.settings import get_settings

    return get_settings()


# =========================================================================
# Capability check.
# =========================================================================
def progressive_available() -> bool:
    """True iff the vendored engine imports AND the vispy/napari internals its
    GLIR-meter and texture-swap patches reach for are present.

    Fails CLOSED (returns False) on any ImportError/AttributeError.
    """
    try:
        # Vendored engine modules import cleanly.
        _load_engine()
        _load_metering()
        from ._vendor_progressive import _texture_swap  # noqa: F401

        # vispy internal the GLIR meter monkeypatches.
        from vispy.gloo import glir

        if not hasattr(glir, '_GlirQueueShare') or not hasattr(
            glir._GlirQueueShare, 'flush'
        ):
            return False

        # vispy internal the double-buffered texture swap needs.
        from vispy.visuals._scalable_textures import (  # noqa: F401
            GPUScaledTextureMixin,
        )

        # napari GL helpers the graft + engine use.
        from napari._vispy.utils.gl import (  # noqa: F401
            fix_data_dtype,
            get_max_texture_sizes,
        )

        # Stock ScalarFieldBase must expose the members the graft reads.
        from napari.layers._scalar_field.scalar_field import ScalarFieldBase

        required = (
            'level_shapes',
            'downsample_factors',
            '_update_level_and_corners',
            '_reset_data_level',
            '_world_to_data_ray',
            '_get_layer_slicing_state',
        )
        if not all(hasattr(ScalarFieldBase, m) for m in required):
            return False

        return True
    except (ImportError, AttributeError):
        return False


# =========================================================================
# Process-wide GLIR meter lifecycle (only active while a progressive layer
# exists).
# =========================================================================
# Weakrefs to live progressive layers; pruned lazily.  A list of weakrefs
# avoids any assumption about layer hashability.
_live_refs: list = []


def _prune() -> None:
    _live_refs[:] = [r for r in _live_refs if r() is not None]


def _register(layer) -> None:
    _prune()
    _live_refs.append(weakref.ref(layer))


def _live_count() -> int:
    _prune()
    return len(_live_refs)


# ---------------------------------------------------------------------------
# GLIR draw guard — skip (+ log) a DRAW whose index buffer is transiently
# missing, instead of crashing.
# ---------------------------------------------------------------------------
# The vendored metered flush (_glir_metering._metered_flush) reorders/defers
# GLIR CREATE/DATA/DELETE across flushes.  Its carry-net catches a
# `RuntimeError('does not exist')` for a command referencing a not-yet-created /
# already-deleted object — but a DRAW whose index buffer is missing crashes
# EARLIER, inside `GlirProgram.draw` at `ibuf = parser.get_object(id); ibuf.
# activate()` (vispy glir.py) -> `AttributeError: 'NoneType' ... activate`, which
# bypasses that net and aborts the whole paint (observed on 3D<->2D switches of a
# live-streaming progressive Volume: the metered reorder leaves the Volume's
# bounding-box index buffer absent for a frame).  This guard pre-checks the index
# buffer and SKIPS just that node's draw for the frame (it renders normally on the
# next, once the carried CREATE drains), turning a hard crash into an
# imperceptible one-frame skip.  Installed/uninstalled with the meter so it is
# scoped to progressive sessions; monkeypatches VISPY (not the vendored engine).
_draw_guard_installed = False
_ORIG_GLIR_DRAW = None
_draw_skip_count = 0


def _install_draw_guard() -> None:
    global _draw_guard_installed, _ORIG_GLIR_DRAW
    if _draw_guard_installed:
        return
    try:
        from vispy.gloo.glir import GlirProgram
    except Exception:  # noqa: BLE001
        return
    orig = GlirProgram.draw
    _ORIG_GLIR_DRAW = orig

    def _guarded_draw(self, mode, selection, instances=1):
        global _draw_skip_count
        # Indexed draw: selection == (index_buffer_id, gtype, count).
        if isinstance(selection, (tuple, list)) and len(selection) == 3:
            idx_id = selection[0]
            try:
                missing = self._parser.get_object(idx_id) is None
            except Exception:  # noqa: BLE001
                missing = False
            if missing:
                _draw_skip_count += 1
                if _draw_skip_count <= 5 or _draw_skip_count % 50 == 0:
                    print(
                        f"[PROG-GLGUARD] skipped a DRAW with a missing index "
                        f"buffer (id={idx_id}) from the metered-flush reorder "
                        f"-> prevented the ibuf.activate() crash "
                        f"(skip #{_draw_skip_count})"
                    )
                return  # self-corrects next frame when the CREATE drains
        return orig(self, mode, selection, instances)

    GlirProgram.draw = _guarded_draw
    _draw_guard_installed = True


def _uninstall_draw_guard() -> None:
    global _draw_guard_installed, _ORIG_GLIR_DRAW, _draw_skip_count
    if not _draw_guard_installed:
        return
    try:
        from vispy.gloo.glir import GlirProgram

        if _ORIG_GLIR_DRAW is not None:
            GlirProgram.draw = _ORIG_GLIR_DRAW
    except Exception:  # noqa: BLE001
        pass
    _ORIG_GLIR_DRAW = None
    _draw_guard_installed = False
    _draw_skip_count = 0   # fresh verbose window next session


# ---------------------------------------------------------------------------
# napari layer-list loading-spinner guard (napari 0.8 `_paint_loading` bug).
# ---------------------------------------------------------------------------
# Stock napari 0.8's `LayerDelegate._paint_loading` (`_layer_delegate.py:153`)
# does `index.data(SizeHintRole).height()` with NO None-check.  Our progressive
# layers correctly report `loaded=False` while streaming (napari's real async-
# slice state — the engine enables async slicing; we set it nowhere), so the
# loading spinner keeps repainting the layer list.  During the SYNCHRONOUS
# 3D<->2D swap those repaints — dispatched by the reentrant `processEvents`
# inside `viewer.add_image` / napari's progress bar as it removes each
# progressive layer and re-adds a native one — land on rows that are momentarily
# OUT OF RANGE, so the model's `getItem` raises `IndexError` (swallowed by the
# Qt binding) and `index.data(...)` returns None for BOTH LoadedRole and
# SizeHintRole -> `None.height()` -> AttributeError (crashes ~6x across the ~4s
# window).  For a VALID row SizeHintRole is a real `QSize(200,34)`, so the
# spinner still draws normally — only the transient stale-index frame is skipped
# (it repaints correctly on the next tick).  Reversible + idempotent, mirroring
# the GLIR draw guard above.  It is INSTALLED alongside the draw guard, but —
# unlike the draw guard — it is NOT torn down on the per-swap `_live_count()==0`
# idle path: that path fires the instant the last progressive layer is removed,
# whereas THIS guard's crash window is the native RE-ADD (`viewer.add_image`)
# that runs right after.  It is superset-safe (only ever skips a would-crash
# stale-index paint — never changes a valid paint), so it persists across swaps
# and is torn down on window-destroy (`_wire_teardown`).  This is a genuine
# stock-napari bug and is worth reporting upstream.
_paint_guard_installed = False
_ORIG_PAINT_LOADING = None
_paint_skip_count = 0


def _install_paint_loading_guard() -> None:
    global _paint_guard_installed, _ORIG_PAINT_LOADING
    if _paint_guard_installed:
        return
    try:
        from qtpy.QtCore import Qt
        from napari._qt.containers._layer_delegate import LayerDelegate
    except Exception:  # noqa: BLE001  (headless / no Qt -> nothing to guard)
        return
    orig = LayerDelegate._paint_loading
    _ORIG_PAINT_LOADING = orig

    def _guarded_paint_loading(self, painter, option, index):
        global _paint_skip_count
        # Skip the spinner for the one stale frame that would crash: an
        # out-of-range row returns None for SizeHintRole -> None.height().
        try:
            stale = (
                not index.isValid()
                or index.data(Qt.ItemDataRole.SizeHintRole) is None
            )
        except Exception:  # noqa: BLE001
            # If probing the index itself raised, we can't safely size the
            # spinner either -> skip this frame (harmless) rather than fall
            # through and let the original hit the same None.height().
            stale = True
        if stale:
            _paint_skip_count += 1
            if _paint_skip_count <= 5 or _paint_skip_count % 50 == 0:
                print(
                    f"[PROG-PAINTGUARD] skipped a layer-list loading-spinner "
                    f"paint on a stale index (None SizeHintRole) during the "
                    f"3D<->2D swap -> prevented napari _paint_loading's "
                    f"None.height() crash (skip #{_paint_skip_count})"
                )
            return
        return orig(self, painter, option, index)

    LayerDelegate._paint_loading = _guarded_paint_loading
    _paint_guard_installed = True


def _uninstall_paint_loading_guard() -> None:
    global _paint_guard_installed, _ORIG_PAINT_LOADING, _paint_skip_count
    if not _paint_guard_installed:
        return
    try:
        from napari._qt.containers._layer_delegate import LayerDelegate

        if _ORIG_PAINT_LOADING is not None:
            LayerDelegate._paint_loading = _ORIG_PAINT_LOADING
    except Exception:  # noqa: BLE001
        pass
    _ORIG_PAINT_LOADING = None
    _paint_guard_installed = False
    _paint_skip_count = 0   # fresh verbose window next session


# ---------------------------------------------------------------------------
# GLIR meter enablement.
# ---------------------------------------------------------------------------
# The meter (_glir_metering) spreads big GPU uploads across frames AND drives
# the progressive double-buffer's present cadence — notably `_on_uploads_drained`
# (the drain-notify present), the reliable path that swaps an async re-slice's
# freshly-uploaded BACK texture to the visible FRONT.  DISABLING the meter
# entirely produced a BLACK 3D (empirically confirmed 2026-08): content staged
# into the back texture but was never presented.  (Other present sites exist —
# `_on_fetch_finished`, `_update_node`, repair — but did NOT cover the
# async-staged / already-resident view, so meter-off = black.)  So the meter is
# LOAD-BEARING for display and must stay ON.
#
# The 3D<->2D crashes (`ibuf.activate() on None` / `Could not find texture with
# id` + the permanent-skip flood) were NOT the metering itself — they were the
# meter's DEFERRED-DELETE reorder specifically: a deferred DELETE draining in a
# later quiet flush, keyed on the PRIMARY object id, AHEAD of a still-live
# draw/set-texture referencing that id as a SECONDARY arg (root-caused by
# multiple independent audits).  That is fixed AT THE SOURCE by
# `_glir_metering._DEFER_DELETES = False`, which runs DELETEs inline in queue
# order so one can never precede a live reference — while the present/upload
# path stays intact.  Hence: meter ON, delete-deferral OFF.
_METER_ENABLED = True


@contextmanager
def _meter_install_suppressed():
    """No-op the vendored engine's DIRECT ``_glir_metering.install()`` call
    (``_progressive_loading.py`` ~3205, run inside every progressive add) while
    the meter is DISABLED — so neither the seam NOR the engine installs a metered
    flush.  The engine resolves ``_glir_metering.install`` at call time, so
    rebinding the module attribute wins; restored unconditionally.  A plain
    passthrough when the meter is ENABLED."""
    if _METER_ENABLED:
        yield
        return
    metering = _load_metering()
    orig_install = metering.install
    metering.install = lambda *a, **k: False   # engine's install() -> no-op
    try:
        yield
    finally:
        metering.install = orig_install


def install_glir_if_needed() -> bool:
    """Install the GLIR draw guard, and the process-wide meter IFF enabled.

    Returns True iff THIS call installed something (guard and/or meter) that a
    later failure in the caller should tear down.

    The draw guard is installed UNCONDITIONALLY (idempotent) — belt-and-
    suspenders against any residual missing-buffer draw (e.g. a not-fully-
    detached node) even with the meter off.  The METER is gated on
    ``_METER_ENABLED`` (OFF by default; see the enablement note above): its
    deferred-DELETE reorder is the root of the 3D<->2D dangling-reference
    crashes, and vispy's atomic flush avoids them entirely."""
    did = False
    if not _draw_guard_installed:
        _install_draw_guard()
        did = _draw_guard_installed   # True unless vispy import failed
    if not _paint_guard_installed:
        _install_paint_loading_guard()
        did = did or _paint_guard_installed   # True unless napari/Qt import failed
    if _METER_ENABLED:
        metering = _load_metering()
        if not metering.is_installed():
            metering.install()
            did = True
    return did


def uninstall_glir_if_idle() -> None:
    """Uninstall the process-wide GLIR meter iff no progressive layer is live."""
    if _live_count() > 0:
        return
    metering = _load_metering()
    if metering.is_installed():
        metering.uninstall()
    _uninstall_draw_guard()
    # NOTE: the paint-loading guard is DELIBERATELY *not* torn down here.  This
    # idle path fires on EVERY 3D->2D swap (the last progressive layer's
    # `removed` -> `_live_count()==0`), but the guard's crash window is the
    # SYNCHRONOUS native RE-ADD (`viewer.add_image`, with its reentrant
    # processEvents) that runs immediately AFTER that removal — so tearing it
    # down here would leave the final channel's re-add unguarded (the exact
    # crash).  It is superset-safe (only ever skips a would-crash stale-index
    # paint), so it persists across swaps and is torn down on window-destroy
    # (see `_wire_teardown`).


def remove_progressive(layer) -> None:
    """Deregister a progressive layer; uninstall the GLIR meter when the last
    one goes away."""
    _live_refs[:] = [
        r for r in _live_refs if r() is not None and r() is not layer
    ]
    uninstall_glir_if_idle()


# Strong refs to live teardown handlers.  napari's own ``EventEmitter`` (NOT
# psygnal) backs ``layers.events.removed`` and holds the connected callable
# object STRONGLY, so this list is defensive belt-and-suspenders that keeps each
# handler alive for the viewer's lifetime regardless.  Each handler removes
# itself here once it has fired.
_teardown_handlers: list = []


class _ProgressiveTeardown:
    """Self-disconnecting ``layers.events.removed`` handler for ONE layer.

    When its layer leaves the viewer it calls :func:`remove_progressive`
    (which prunes the registry and uninstalls the meter iff no progressive
    layer remains), then disconnects itself so it never fires — or
    double-frees — again.  A ``removed`` event for a *different* layer is
    ignored, so the 2-live-layers invariant (removing one must NOT uninstall)
    is respected via :func:`_live_count`.
    """

    def __init__(self, viewer, layer):
        self._viewer_ref = weakref.ref(viewer)
        self._layer_ref = weakref.ref(layer)
        self._done = False

    def __call__(self, event=None):
        if self._done:
            return
        layer = self._layer_ref()
        if layer is None:
            # our layer was GC'd without a removed event reaching us; tear
            # down anyway so the meter can be reclaimed.
            self._teardown(None)
            return
        removed = getattr(event, 'value', None)
        if removed is not layer:
            return  # a DIFFERENT layer was removed — not ours.
        self._teardown(layer)

    def _teardown(self, layer) -> None:
        self._done = True
        try:
            if layer is not None:
                remove_progressive(layer)
            else:
                uninstall_glir_if_idle()
        finally:
            self._disconnect()
            try:
                _teardown_handlers.remove(self)
            except ValueError:
                pass

    def _disconnect(self) -> None:
        viewer = self._viewer_ref()
        if viewer is None:
            return
        try:
            viewer.layers.events.removed.disconnect(self)
        except (ValueError, AttributeError, RuntimeError, TypeError):
            pass


def _wire_teardown(viewer, layer) -> None:
    """Wire GLIR-meter teardown for a successfully added progressive layer.

    Mirrors how the engine self-wires its own loader ``close()``: connect a
    guarded ``removed`` handler that uninstalls the meter when the LAST
    progressive layer goes away, plus a best-effort window-destroyed handler.
    Best-effort throughout — a wiring failure must not fail the add (the layer
    is already live)."""
    try:
        handler = _ProgressiveTeardown(viewer, layer)
        viewer.layers.events.removed.connect(handler)
        _teardown_handlers.append(handler)
    except Exception:  # noqa: BLE001
        LOGGER.debug(
            'progressive removed-teardown wiring failed', exc_info=True
        )
    # If the whole window goes away, uninstall the meter when idle. The
    # engine already connects loader.close() here; this is a belt-and-
    # suspenders meter reclaim. Guarded/best-effort (viewer may be headless).
    try:
        def _on_window_destroyed(*_a):
            try:
                uninstall_glir_if_idle()
            except Exception:  # noqa: BLE001  (never let it skip the restore below)
                pass
            # The paint-loading guard survives the per-swap idle path (its crash
            # window is the native re-add that follows; see
            # uninstall_glir_if_idle).  Window-destroy is its genuinely-final
            # teardown — no swap can follow — so restore napari's method here.
            # (Self-guarded internally, so this always runs even on the above.)
            _uninstall_paint_loading_guard()

        viewer.window._qt_window.destroyed.connect(_on_window_destroyed)
    except Exception:  # noqa: BLE001
        pass


# =========================================================================
# Layer-class swap (so the engine's local `from napari.layers import Image`
# builds OUR subclass) — scoped, always restored.
# =========================================================================
def _restore_attr(mod, name, saved) -> None:
    if saved is _SENTINEL:
        if hasattr(mod, name):
            try:
                delattr(mod, name)
            except AttributeError:
                pass
    else:
        setattr(mod, name, saved)


class _swap_layer_classes:
    """Context manager: temporarily rebind ``Image``/``Labels`` on
    ``napari.layers`` (load-bearing: the engine's *local* import reads it at
    call time).  Always restores on exit, including on exception.

    SINGLE-PATH: only ``napari.layers`` is swapped — the engine module has no
    module-level ``Image``/``Labels`` (it uses a local import), so no other
    swap target is load-bearing.

    Guarded so it cannot corrupt the saved originals: it MUST run on the main
    thread (napari layer construction is not thread-safe) and MUST NOT be
    re-entered (a nested swap would save ``ProgressiveImage`` as the
    "original" and never restore the real class).  Both checks raise BEFORE
    any attribute is saved or mutated.
    """

    _active = False  # class-level non-reentrancy guard

    def __enter__(self):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError(
                'progressive layer-class swap must run on the main thread'
            )
        if _swap_layer_classes._active:
            raise RuntimeError(
                'progressive layer-class swap is already active '
                '(non-reentrant)'
            )
        import napari.layers as _nl

        self._nl = _nl
        self._saved_image = getattr(_nl, 'Image', _SENTINEL)
        self._saved_labels = getattr(_nl, 'Labels', _SENTINEL)
        _nl.Image = ProgressiveImage
        _nl.Labels = ProgressiveLabels
        _swap_layer_classes._active = True
        return self

    def __exit__(self, *exc):
        _restore_attr(self._nl, 'Image', self._saved_image)
        _restore_attr(self._nl, 'Labels', self._saved_labels)
        _swap_layer_classes._active = False
        return False


# =========================================================================
# THE SEAM.
# =========================================================================
def add_progressive_image(viewer, levels, **kwargs):
    """Add a progressively-loading multiscale IMAGE, or None on failure.

    Installs the GLIR meter, swaps the engine's layer classes to
    :class:`ProgressiveImage`, and calls the vendored
    ``add_progressive_loading_image``.  Fails CLOSED: on ANY exception it
    logs, uninstalls the meter if this call installed it, and returns None so
    the caller can fall back to the manual-slider path.
    """
    return _add_progressive(viewer, levels, is_labels=False, **kwargs)


def add_progressive_labels(viewer, levels, **kwargs):
    """Add a progressively-loading multiscale LABELS layer, or None on failure.
    See :func:`add_progressive_image`."""
    return _add_progressive(viewer, levels, is_labels=True, **kwargs)


def _rollback_engine_side_effects(
    viewer, settings, pre_layer_ids, pre_async, pre_force_sync
) -> None:
    """Undo the engine's in-swap-window mutations after a failed add.

    Inside its swap window the engine appends the layer
    (``_progressive_loading.py`` ~3176) and flips the global async slicing
    settings (~3197-3198) BEFORE its own later failure points (loader ctor /
    debug overlay).  On failure this removes any layer the engine added — any
    layer now present whose ``id`` was NOT in the pre-call snapshot — with a
    safe hide-then-remove, and restores the async settings to their pre-call
    values.  Each step is independently guarded so one failure does not abort
    the rest (fail-closed).  The GLIR meter is uninstalled by the caller's
    outer handler, which owns the ``installed_here`` decision.
    """
    # 1) remove the engine's orphan layer(s).
    try:
        orphans = [
            layer
            for layer in list(viewer.layers)
            if id(layer) not in pre_layer_ids
        ]
    except Exception:  # noqa: BLE001
        orphans = []
    for orphan in orphans:
        try:
            # hide before remove: disconnects the vispy visual from the scene
            # graph so a freed GPU texture cannot cause an access violation.
            orphan.visible = False
        except Exception:  # noqa: BLE001
            pass
        try:
            viewer.layers.remove(orphan)
        except Exception:  # noqa: BLE001
            pass
        # Defensive: the orphan is never _register()ed (that happens only on
        # success), but drop it from the registry if somehow present WITHOUT
        # touching the meter (the outer handler owns that).
        try:
            _live_refs[:] = [
                r
                for r in _live_refs
                if r() is not None and r() is not orphan
            ]
        except Exception:  # noqa: BLE001
            pass

    # 2) restore the global async slicing settings the engine flipped.
    try:
        settings.experimental.async_ = pre_async
    except Exception:  # noqa: BLE001
        pass
    try:
        viewer._layer_slicer._force_sync = pre_force_sync
    except Exception:  # noqa: BLE001
        pass


def _add_progressive(viewer, levels, *, is_labels, **kwargs):
    kind = 'labels' if is_labels else 'image'

    # Finding 2: fail CLOSED before touching anything if the engine or the GL
    # internals it needs are unavailable.
    if not progressive_available():
        LOGGER.warning(
            'progressive loading unavailable; %s falls back to manual path',
            kind,
        )
        return None

    # Finding 4: REQUIRE a real viewer.  With viewer=None the engine would
    # construct a napari.Viewer() INSIDE the swap window — an implicit viewer
    # that escapes this seam's teardown wiring and leaks the GLIR patch.
    if viewer is None:
        LOGGER.error('add_progressive_%s requires a viewer; got None', kind)
        return None

    installed_here = False
    try:
        # Finding 2: engine import + meter install INSIDE the try so an
        # import/install failure returns None (fail closed), never raises.
        pl = _load_engine()
        installed_here = install_glir_if_needed()

        # Finding 1: snapshot the pre-call state so the engine's in-swap-window
        # side effects (layer append + async/force_sync flip, done BEFORE its
        # own later failure points) roll back atomically on ANY error.
        settings = _get_settings()
        pre_layer_ids = {id(layer) for layer in viewer.layers}
        pre_async = settings.experimental.async_
        pre_force_sync = viewer._layer_slicer._force_sync

        try:
            with _swap_layer_classes(), _meter_install_suppressed():
                if is_labels:
                    layer = pl.add_progressive_loading_labels(
                        levels, viewer=viewer, **kwargs
                    )
                else:
                    layer = pl.add_progressive_loading_image(
                        levels, viewer=viewer, **kwargs
                    )
        except Exception:
            # Roll the engine's partial mutations back so the caller's manual
            # fallback does not inherit an orphan layer or a stuck async
            # setting, then re-raise into the outer handler (meter owner).
            _rollback_engine_side_effects(
                viewer,
                settings,
                pre_layer_ids,
                pre_async,
                pre_force_sync,
            )
            raise

        _register(layer)
        _wire_teardown(viewer, layer)
        return layer
    except Exception:  # noqa: BLE001 — fail closed, caller falls back
        LOGGER.exception(
            'progressive %s add failed; falling back to manual path', kind
        )
        if installed_here:
            uninstall_glir_if_idle()
        return None


# ---------------------------------------------------------------------------
# Per-step slicing policy (2026-09-04).
# ---------------------------------------------------------------------------
# The vendored engine flips napari's GLOBAL ``experimental.async_`` setting ON
# for progressive 3D and napari persists that setting to disk on change.  A
# session that does not tear the loaders down cleanly (crash, force-quit)
# therefore leaves async slicing ON for every later session — and the Step 2/3
# annotators are built around synchronous hide -> recolour -> re-show cycles
# that misrender under async slicing (hidden layers keep an EMPTY placeholder
# slice until a deferred re-slice lands).  Annotator launch calls this helper:
# it turns async slicing OFF unless a progressive loader is live on the viewer
# (progressive 3D genuinely needs it), so the policy follows the step in use.


def _live_progressive_loaders(viewer) -> list:
    try:
        from ._vendor_progressive._progressive_loading import _progressive_loaders
    except Exception:  # noqa: BLE001
        return []
    try:
        return list(_progressive_loaders(viewer))
    except Exception:  # noqa: BLE001
        return []


def _set_async_slicing(viewer, on: bool, reason: str) -> bool:
    """Drive napari's persisted ``experimental.async_`` to *on*.

    The one place that reads/writes the setting and prints the ``[SLICING]``
    line, so the two public helpers cannot drift in how they report themselves.
    Returns True iff the setting was actually flipped; never raises.

    The progressive-loader guard applies ONLY when turning async OFF: a live
    Step-1 progressive loader needs async slicing, so switching it off under one
    would break it (hidden-layer recolour paths misrender).  Turning async ON
    needs no such guard — ON is what a loader wants — which is why the ON helper
    never consults the probe.
    """
    on = bool(on)
    try:
        settings = _get_settings()
        current = bool(settings.experimental.async_)
    except Exception:  # noqa: BLE001
        return False
    if current == on:
        return False
    if not on:
        live = _live_progressive_loaders(viewer) if viewer is not None else []
        if live:
            print(
                f"[SLICING] async slicing left ON ({reason or 'annotator'}): "
                f"{len(live)} progressive loader(s) live on this viewer — "
                "hidden-layer recolour paths may misrender until they are removed"
            )
            return False
    try:
        settings.experimental.async_ = on
    except Exception:  # noqa: BLE001
        return False
    if on:
        print(f"[SLICING] async slicing turned ON ({reason or 'step 1'})")
    else:
        print(f"[SLICING] async slicing turned OFF ({reason or 'annotator'}): no progressive loader live")
    return True


def ensure_async_slicing(viewer, *, reason: str = "") -> bool:
    """Turn napari's ``experimental.async_`` ON for *viewer*.  Returns True iff
    the setting was flipped.

    Step 1 browses 2D slices, and async slicing takes that work off the UI
    thread.  Measured live in napari on the wound store, UI-thread cost per 2D Z
    step: **36 -> 18 ms** on the new ``(32, 128, 128)`` store (45 -> 37 ms on an
    old cubic-64 one).

    Unlike :func:`ensure_sync_slicing` this has NO progressive-loader guard and
    ignores *viewer* for the decision: ON is the state a progressive loader
    wants, so there is nothing to decline for.  *viewer* is still accepted so the
    two helpers are call-compatible.

    Prints one ``[SLICING]`` line when it acts, silent when async is already ON.
    Never raises — a settings backend that cannot be read or written just
    returns False.
    """
    return _set_async_slicing(viewer, True, reason)


def ensure_sync_slicing(viewer, *, reason: str = "") -> bool:
    """Turn napari's ``experimental.async_`` OFF for *viewer* unless a
    progressive loader is live.  Returns True iff the setting was flipped.

    Prints one ``[SLICING]`` line whenever it acts or declines, so the
    console shows which policy a session runs under.
    """
    return _set_async_slicing(viewer, False, reason)
