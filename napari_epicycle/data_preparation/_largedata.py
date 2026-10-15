"""Compatibility shim — the Step-1 large-data helpers now live in the shared,
napari-free package ``packages/epicycle_largedata``.

Kept at this import path so the controller / box tool / extract that do
``from ._largedata import ...`` keep working unchanged.  NEW code should import
from ``packages.epicycle_largedata`` directly.

Why the move: this is now the SHARED large-data substrate (Step 1 today, Step 5 /
notebooks / HPC later), and it converges the axis resolver that used to be
duplicated between here and the scaled-seg engine..
"""
from packages.epicycle_largedata.axes import (  # noqa: F401
    AxisResolution,
    CHANNEL_AXIS_MAX,
    canonicalize_to_czyx,
    canonicalize_to_zcyx,
    resolve_axes,
)
from packages.epicycle_largedata.lazy import (  # noqa: F401
    content_bbox,
    crop_czyx,
    crop_zyx,
    downsample_for_display,
    materialize,
    open_lazy,
)
