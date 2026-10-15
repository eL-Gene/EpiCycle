"""epicycle_preprocessing — the pure, napari-free Step-1 preprocessing engine.

napari-free BY DESIGN (mirrors the other ``packages/*``): no Qt, no napari, no
torch, no imports from ``napari_epicycle``.  This is what lets Step 1 (GUI),
notebooks, and a future headless engine all consume the same code.

Brick 1a (this pass) builds only the metadata + data-model foundation:

* :mod:`registry` — the key-based op registry (single source of truth for the
  ``+ add step`` menu AND the executor).  Five ops carry ``implemented=True``;
  the rest of the catalogue is declared ``implemented=False`` so the menu can
  render them disabled ("— soon") instead of hiding them.
* :mod:`recipe` — the per-source recipe data-model: per-channel ordered
  intensity lists + one shared, linked geometry step at a boundary index, plus
  the (de)serializers and the corrupt-tolerant :func:`recipe.normalize_recipe`.

There is deliberately **no op math** here yet — the pure ``fn`` implementations
and the in-RAM executor land in Brick 2.
"""
