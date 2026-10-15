"""Vendored napari progressive-loading engine (PR #9067).

This subpackage is a **temporary in-tree copy** of the load-bearing engine
modules from napari PR #9067 (progressive / chunk-wise loading for multiscale
images), taken from the PR head ``kephale/napari@07e4f3fc`` (branch
``progressive-loading-rebase``, rebased onto post-0.8.0 napari main).

EpiCycle runs this engine on STOCK napari 0.8.0 behind its own seam. Only the
core streaming modules are vendored; the demo/test scaffolding that requires
zarr v3 (``_generative_zarr``, ``_progressive_loading_datasets``) and the
upstream consumer shim (``_auto_progressive``) are intentionally NOT vendored.

Vendored modules (import them explicitly; nothing is eagerly imported here so
that the package can be imported in headless / no-Qt environments):

- ``_progressive_loading`` -- core ``ProgressiveLoader`` + view-driven fetch
- ``_virtual_data``        -- ``VirtualData`` / ``MultiScaleVirtualData``
- ``_texture_swap``        -- double-buffered vispy texture streaming
- ``_glir_metering``       -- per-frame vispy GLIR upload metering / patches
- ``_debug_overlay``       -- optional chunk-wireframe debug overlay

Intra-package imports have been rewritten to relative form; every
``from napari.*`` / ``from vispy.*`` import is left untouched so it resolves
to the installed packages.

Provenance, license, and the "delete when upstream releases" note live in
``NOTICE`` and ``README.md`` alongside this file.
"""
