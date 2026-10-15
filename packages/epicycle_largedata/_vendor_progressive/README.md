# `_vendor_progressive` — vendored napari progressive-loading engine

**Temporary in-tree copy. Delete when napari ships progressive loading in a release.**

This directory vendors the load-bearing engine modules from **napari PR #9067**
("Progressive loading" — chunk-wise streaming for multiscale images) so that
EpiCycle can ship progressive 3D loading on **stock napari 0.8.0** behind its
own seam, without depending on an unmerged napari branch at runtime.

## Provenance

| | |
|---|---|
| Project | napari (github.com/napari/napari) |
| PR | [#9067](https://github.com/napari/napari/pull/9067) |
| Head repo | `kephale/napari` |
| Head SHA | `07e4f3fcfb602878a864aa0de4d9b08bf5a9a2f7` |
| Head branch | `progressive-loading-rebase` (rebased onto post-0.8.0 napari `main`) |
| Fetched | 2026-08-02 |
| License | BSD 3-Clause (Copyright napari contributors) — see `NOTICE` |

We vendor from the **current PR head** (rebased onto post-0.8.0 main), not the
stale on-disk fork copy (`napari-pl-fork`, base 0.7.1.dev0).

## What is vendored (the core streaming engine)

| Module | Role |
|---|---|
| `_progressive_loading.py` | Core `ProgressiveLoader` + view-driven fetch pass |
| `_virtual_data.py` | `VirtualData` / `MultiScaleVirtualData` array wrappers |
| `_texture_swap.py` | Double-buffered vispy texture streaming |
| `_glir_metering.py` | Per-frame vispy GLIR texture-upload metering / patches |
| `_debug_overlay.py` | Optional chunk-wireframe + level-HUD debug overlay |

This is the exact transitive closure of `_progressive_loading.py`'s
intra-package imports (`_texture_swap`, `_virtual_data` at module level;
`_debug_overlay`, `_glir_metering` imported lazily inside methods).
`_debug_overlay` in turn imports `_virtual_data` (and `_progressive_loading`
under `TYPE_CHECKING` only).

## What is intentionally EXCLUDED and why

| Module | Why excluded |
|---|---|
| `_generative_zarr.py` | Demo/test data generator; requires **zarr v3** (this env has zarr 2). Not imported by any core module. |
| `_progressive_loading_datasets.py` | Demo/test dataset scaffolding; requires **zarr v3**. Imports `_generative_zarr`. Not imported by any core module. |
| `_auto_progressive.py` | Upstream consumer shim. **Replaced by EpiCycle's own seam.** Not imported by any core module. |
| `__init__.py` (upstream) | The upstream `experimental/__init__.py` re-exports unrelated `link_layers` helpers; irrelevant here. This package ships its own `__init__.py`. |

Because none of the five vendored modules import the excluded zarr-demo modules,
**no zarr-v3 import guards were needed** on the core streaming path.

## Modifications applied to the vendored files

Kept minimal and mechanical:

1. **Attribution header** prepended as a comment block at the top of each `.py`
   (above the module docstring, so `__doc__` is preserved).
2. **Intra-package imports rewritten to relative form**, e.g.
   `from napari.experimental._virtual_data import X` → `from ._virtual_data import X`
   (also `_texture_swap`, `_glir_metering`, `_debug_overlay`, `_progressive_loading`).

Everything else is **unchanged**. In particular, all `from napari.*` and
`from vispy.*` imports are left as-is so they resolve to the installed
packages, and logger-name strings / docstring cross-references that mention
`napari.experimental.*` are left untouched (they are not imports).

## Runtime note

Importing these modules only needs napari + vispy + numpy installed; Qt imports
in `_progressive_loading.py` are deferred so the pure-data helpers import in
headless environments. **Full runtime** (actually driving a `ProgressiveLoader`)
additionally needs EpiCycle's seam/shims, added separately.

## When to delete

Delete this entire directory once napari-core ships progressive loading in a
released version, and switch EpiCycle's seam to depend on the upstream public
API.
